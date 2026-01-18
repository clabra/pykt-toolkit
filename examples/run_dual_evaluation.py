#!/usr/bin/env python3
"""
Dual Evaluation Script for Interpretability Validation

This script runs parallel evaluation across multiple experiments with both
prediction modes (supervised and reference) to validate interpretability.

Follows the same evaluation protocol as run_benchmarks_paper.py:
- Question-level, late fusion (mean average) evaluation
- GPU distribution and CPU throttling
- Background execution support via launch_dual_eval.sh
- Runs from examples/ directory like run_benchmarks_paper.py

Usage:
    # Via shell wrapper (recommended)
    ./launch_dual_eval.sh "0,1,2" 3
    
    # Direct invocation
    python3 run_dual_evaluation.py --gpu_ids 0,1,2 --max_workers 3
"""

import os
import sys
import json
import subprocess
import pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Must match run_benchmarks_paper.py pattern
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Key experiments for dual evaluation
EXPERIMENTS = {
    "090230": {
        "path": "saved_model/gtransformer_assist2009_qid_20260115_090230",
        "description": "Grounded only (no probing)",
        "config": "Grounded=✅, Probing=❌, Personalization=❌"
    },
    "334772": {
        "path": "saved_model/gtransformer_assist2009_qid_20260116_101107_gtransformer_assist2009_0_seed_42_d_model_64_n_heads_8_n_blocks_2_learning_rate_0.001_334772",
        "description": "Aligned Grounding (with probing)",
        "config": "Grounded=✅, Probing=✅, Personalization=❌"
    },
    "948799": {
        "path": "saved_model/gtransformer_assist2009_qid_20260116_120815_gtransformer_assist2009_0_seed_42_d_model_64_n_heads_8_n_blocks_2_learning_rate_0.001_948799",
        "description": "Full (with personalization)",
        "config": "Grounded=✅, Probing=✅, Personalization=✅"
    },
    "133835": {
        "path": "saved_model/gtransformer_assist2009_qid_20260115_133835",
        "description": "Ablated baseline (2/8, no grounding)",
        "config": "Grounded=❌, Probing=❌, Personalization=❌"
    }
}

def run_evaluation(exp_id, exp_info, prediction_type="supervised", gpu_id=0):
    """Run evaluation for a single experiment with specified prediction type."""
    save_dir = exp_info["path"]
    
    # Check if experiment directory exists
    if not os.path.exists(save_dir):
        print(f"⚠️  Experiment {exp_id} not found at {save_dir}")
        return None
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] {'='*80}")
    print(f"[{timestamp}] Evaluating Exp {exp_id}: {exp_info['description']}")
    print(f"[{timestamp}] Config: {exp_info['config']}")
    print(f"[{timestamp}] Prediction Type: {prediction_type}")
    print(f"[{timestamp}] GPU: {gpu_id}")
    print(f"[{timestamp}] {'='*80}\n")
    
    # Set environment variables (matches run_benchmarks_paper.py pattern)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "5"  # Match run_benchmarks_paper.py CPU throttling
    env["MKL_NUM_THREADS"] = "5"
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}"
    
    # Follow exact pattern from run_benchmarks_paper.py eval_explicit command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "wandb_gtransformer_predict.py",
        "--save_dir", save_dir,
        "--bz", "64",  # Match run_benchmarks_paper.py batch size
        "--use_wandb", "0",
        "--fusion_type", "late_fusion",  # Question-level late fusion (mean average)
        "--prediction_type", prediction_type
    ]
    
    # Execute evaluation from examples/ directory (matches run_benchmarks_paper.py)
    cwd = PROJECT_ROOT / "examples"
    log_path = Path(save_dir) / f"eval_dual_{prediction_type}.log"
    
    try:
        # Write log to experiment directory like run_benchmarks_paper.py does
        with open(log_path, "w") as f:
            result = subprocess.run(
                cmd,
                env=env,
                cwd=cwd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600
            )
        
        if result.returncode != 0:
            print(f"❌ Evaluation failed for {exp_id} ({prediction_type})")
            print(f"   See log: {log_path}")
            return None
        
        # Load results (parse from eval_results.json like run_benchmarks_paper.py)
        results_path = Path(save_dir) / "eval_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ✅ Evaluation completed successfully")
            print(f"[{timestamp}]    AUC: {results.get('testauc', 'N/A'):.4f}")
            print(f"[{timestamp}]    ACC: {results.get('testacc', 'N/A'):.4f}")
            print(f"[{timestamp}]    Log: {log_path}")
            
            return results
        else:
            print(f"⚠️  Results file not found: {results_path}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Evaluation timed out for {exp_id} ({prediction_type})")
        return None
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None

def eval_task(args):
    """Wrapper for parallel execution."""
    exp_id, exp_info, prediction_type, gpu_id = args
    return (exp_id, prediction_type, run_evaluation(exp_id, exp_info, prediction_type, gpu_id))

def main():
    parser = argparse.ArgumentParser(description="Dual evaluation for interpretability validation")
    parser.add_argument("--gpu_ids", type=str, default="0", 
                        help="Comma-separated GPU IDs (e.g., '0,1,2')")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Max parallel workers (default: number of GPUs)")
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    max_workers = args.max_workers or len(gpu_ids)
    
    print("\n" + "="*80)
    print("DUAL EVALUATION: Validating Interpretability via p_sup vs. p_ref Comparison")
    print("="*80)
    print(f"GPUs: {gpu_ids}")
    print(f"Max Workers: {max_workers}")
    print(f"Evaluation Protocol: Question-level, Late Fusion (Mean Average)")
    print("="*80)
    
    # Change to examples directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Build task list
    tasks = []
    gpu_counter = 0
    
    for exp_id in ["133835", "090230", "334772", "948799"]:  # Ordered by progression
        exp_info = EXPERIMENTS[exp_id]
        
        # Task for supervised predictions (p_sup)
        tasks.append((exp_id, exp_info, "supervised", gpu_ids[gpu_counter % len(gpu_ids)]))
        gpu_counter += 1
        
        # Task for reference predictions (p_ref) - only for grounded models
        if exp_id != "133835":  # Skip ablated baseline (no BKT logic)
            tasks.append((exp_id, exp_info, "reference", gpu_ids[gpu_counter % len(gpu_ids)]))
            gpu_counter += 1
    
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Running with {max_workers} parallel workers\n")
    
    # Execute tasks in parallel
    results_map = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_task, task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                exp_id, prediction_type, result = future.result()
                if result:
                    key = f"{exp_id}_{prediction_type}"
                    results_map[key] = result
            except Exception as e:
                print(f"❌ Task failed: {task}, Error: {e}")
    
    # Compile results
    all_results = []
    for exp_id in ["133835", "090230", "334772", "948799"]:
        exp_info = EXPERIMENTS[exp_id]
        
        sup_key = f"{exp_id}_supervised"
        ref_key = f"{exp_id}_reference"
        
        sup_results = results_map.get(sup_key)
        ref_results = results_map.get(ref_key)
        
        if sup_results:
            row = {
                "Exp ID": exp_id,
                "Description": exp_info["description"],
                "Config": exp_info["config"],
                "AUC (p_sup)": sup_results.get("testauc", None),
                "ACC (p_sup)": sup_results.get("testacc", None),
                "AUC (p_ref)": ref_results.get("testauc", None) if ref_results else None,
                "ACC (p_ref)": ref_results.get("testacc", None) if ref_results else None,
            }
            
            # Calculate interpretability metrics
            if ref_results and sup_results:
                row["Δ_AUC (sup-ref)"] = sup_results.get("testauc", 0) - ref_results.get("testauc", 0)
                # Try to extract interpretability score from probe MSE
                if "l0_probe_mse" in sup_results and "t_probe_mse" in sup_results:
                    row["L0 Probe MSE"] = sup_results.get("l0_probe_mse")
                    row["T Probe MSE"] = sup_results.get("t_probe_mse")
            
            all_results.append(row)
    
    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY: Dual Evaluation Results")
    print("="*80 + "\n")
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        
        # Save to CSV
        output_path = "dual_evaluation_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
        # Calculate key findings
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        # Find grounded vs. probing improvement in p_ref
        grounded_ref = df[df["Exp ID"] == "090230"]["AUC (p_ref)"].values
        probing_ref = df[df["Exp ID"] == "334772"]["AUC (p_ref)"].values
        
        if len(grounded_ref) > 0 and len(probing_ref) > 0:
            improvement = probing_ref[0] - grounded_ref[0]
            print(f"1. Probing improves p_ref AUC by: {improvement:.4f} ({improvement*100:.2f}%)")
        
        # Cost of interpretability (ablated vs. full)
        ablated_sup = df[df["Exp ID"] == "133835"]["AUC (p_sup)"].values
        full_sup = df[df["Exp ID"] == "948799"]["AUC (p_sup)"].values
        
        if len(ablated_sup) > 0 and len(full_sup) > 0:
            cost = ablated_sup[0] - full_sup[0]
            print(f"2. Interpretability cost (p_sup): {cost:.4f} ({cost*100:.2f}%)")
        
        # BKT improvement (assuming classical BKT AUC = 0.610)
        classical_bkt = 0.610
        full_ref = df[df["Exp ID"] == "948799"]["AUC (p_ref)"].values
        
        if len(full_ref) > 0:
            improvement_bkt = full_ref[0] - classical_bkt
            print(f"3. p_ref improvement over classical BKT: {improvement_bkt:.4f} (+{improvement_bkt*100:.2f}%)")
        
        print("\n" + "="*80)
    else:
        print("⚠️  No results to display")
    
    print("\n✨ Dual evaluation complete!\n")

if __name__ == "__main__":
    main()
