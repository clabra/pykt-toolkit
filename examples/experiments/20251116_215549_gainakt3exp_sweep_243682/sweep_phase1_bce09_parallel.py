#!/usr/bin/env python3
"""
Phase 1 Learning Curve Parameter Sweep with BCE Loss Weight = 0.9 (PARALLEL VERSION)

This sweep replicates the Phase 1 methodology but keeps bce_loss_weight fixed at 0.9
to find optimal learning curve parameters for high performance-focused training.

Key improvements:
- Runs 5 experiments in parallel (one per GPU)
- Stores all experiments inside the sweep folder
- Proper GPU assignment via CUDA_VISIBLE_DEVICES

Rationale:
- Experiment 714616 (dual-loss) achieved Test AUC=0.7183 with bce_loss_weight=0.9
- Experiment 889799 (bce-0.9) achieved Test AUC=0.6858 with same BCE but Phase 1+2 params
- Hypothesis: Phase 1+2 params (Beta=2.5, M_sat=0.7, etc.) were optimized FOR bce=0.3
- Goal: Find optimal learning curve params specifically FOR bce=0.9

Grid Search:
- beta_skill_init: [1.5, 2.0, 2.5] - Learning curve steepness
- m_sat_init: [0.7, 0.8, 0.9] - Mastery saturation ceiling
- gamma_student_init: [0.9, 1.0, 1.1] - Student learning rate
- sigmoid_offset: [1.5, 2.0, 2.5] - Mastery emergence timing
- Total: 81 configurations (3×3×3×3)

Fixed Parameters:
- bce_loss_weight: 0.9 (90% to Encoder 1 performance, 10% to Encoder 2 interpretability)
- dataset: assist2015
- fold: 0
- epochs: 6 (same as original Phase 1)

Expected Duration: ~1.5 hours on 5 GPUs in parallel (vs ~6 hours sequential)
"""

import os
import json
import subprocess
import time
import csv
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# Sweep configuration
SWEEP_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SWEEP_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# CSV output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = SWEEP_DIR / f"sweep_results_{TIMESTAMP}.csv"

# Parameter grid (same as original Phase 1)
BETA_SKILL_VALUES = [1.5, 2.0, 2.5]
M_SAT_VALUES = [0.7, 0.8, 0.9]
GAMMA_STUDENT_VALUES = [0.9, 1.0, 1.1]
SIGMOID_OFFSET_VALUES = [1.5, 2.0, 2.5]

# Fixed parameters
BCE_LOSS_WEIGHT = 0.9
DATASET = "assist2015"
FOLD = 0
EPOCHS = 6

# GPU configuration - use GPUs 0-4 (5 GPUs)
NUM_GPUS = 5
GPU_IDS = list(range(NUM_GPUS))

# Lock for CSV writing
csv_lock = threading.Lock()

def run_experiment_on_gpu(params):
    """Run a single experiment configuration on a specific GPU"""
    
    beta, m_sat, gamma, offset, exp_num, total_exps, gpu_id = params
    
    short_title = f"p1bce09_b{beta}_m{m_sat}_g{gamma}_o{offset}"
    
    print(f"[GPU {gpu_id}] Experiment {exp_num}/{total_exps}: {short_title}")
    print(f"[GPU {gpu_id}] Parameters: Beta={beta}, M_sat={m_sat}, Gamma={gamma}, Offset={offset}")
    
    # Set environment variable to use specific GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build command
    cmd = [
        "python", "examples/run_repro_experiment.py",
        "--short_title", short_title,
        "--dataset", DATASET,
        "--fold", str(FOLD),
        "--epochs", str(EPOCHS),
        "--bce_loss_weight", str(BCE_LOSS_WEIGHT),
        "--beta_skill_init", str(beta),
        "--m_sat_init", str(m_sat),
        "--gamma_student_init", str(gamma),
        "--sigmoid_offset", str(offset),
    ]
    
    start_time = time.time()
    
    try:
        # Run experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/workspaces/pykt-toolkit",
            env=env,
            timeout=900  # 15 minute timeout per experiment
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Try to read eval_results.json
            test_auc = None
            valid_auc = None
            encoder1_auc = None
            encoder2_auc = None
            exp_dir_actual = None
            
            # Find the experiment directory in examples/experiments/
            # The directory name will include timestamp and experiment_id
            import glob
            exp_search = f"/workspaces/pykt-toolkit/examples/experiments/*{short_title}*"
            matching_dirs = glob.glob(exp_search)
            
            if matching_dirs:
                # Get the most recent one
                exp_dir_original = Path(max(matching_dirs, key=os.path.getmtime))
                
                # Move to sweep folder
                exp_dir_actual = EXPERIMENTS_DIR / exp_dir_original.name
                import shutil
                if exp_dir_original.exists() and not exp_dir_actual.exists():
                    shutil.move(str(exp_dir_original), str(exp_dir_actual))
                
                # Read eval results
                eval_file = exp_dir_actual / "eval_results.json"
                if eval_file.exists():
                    with open(eval_file) as f:
                        eval_data = json.load(f)
                        test_auc = eval_data.get('test_auc')
                        valid_auc = eval_data.get('valid_auc')
                        encoder1_auc = eval_data.get('encoder1_test_auc')
                        encoder2_auc = eval_data.get('encoder2_test_auc')
                
                print(f"[GPU {gpu_id}] ✓ SUCCESS ({duration:.1f}s) - Test AUC: {test_auc:.4f}" if test_auc else f"[GPU {gpu_id}] ✓ SUCCESS ({duration:.1f}s)")
                print(f"[GPU {gpu_id}] Moved to: {exp_dir_actual}")
                
                return {
                    'short_title': short_title,
                    'beta_skill_init': beta,
                    'm_sat_init': m_sat,
                    'gamma_student_init': gamma,
                    'sigmoid_offset': offset,
                    'bce_loss_weight': BCE_LOSS_WEIGHT,
                    'test_auc': test_auc,
                    'valid_auc': valid_auc,
                    'encoder1_test_auc': encoder1_auc,
                    'encoder2_test_auc': encoder2_auc,
                    'duration': duration,
                    'status': 'success',
                    'exp_dir': str(exp_dir_actual),
                    'gpu_id': gpu_id
                }
        
        print(f"[GPU {gpu_id}] ✗ FAILED ({duration:.1f}s)")
        return {
            'short_title': short_title,
            'beta_skill_init': beta,
            'm_sat_init': m_sat,
            'gamma_student_init': gamma,
            'sigmoid_offset': offset,
            'bce_loss_weight': BCE_LOSS_WEIGHT,
            'test_auc': None,
            'valid_auc': None,
            'encoder1_test_auc': None,
            'encoder2_test_auc': None,
            'duration': duration,
            'status': 'failed',
            'error': result.stderr[:200] if result.stderr else 'Unknown error',
            'gpu_id': gpu_id
        }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[GPU {gpu_id}] ✗ TIMEOUT ({duration:.1f}s)")
        return {
            'short_title': short_title,
            'beta_skill_init': beta,
            'm_sat_init': m_sat,
            'gamma_student_init': gamma,
            'sigmoid_offset': offset,
            'bce_loss_weight': BCE_LOSS_WEIGHT,
            'test_auc': None,
            'valid_auc': None,
            'encoder1_test_auc': None,
            'encoder2_test_auc': None,
            'duration': duration,
            'status': 'timeout',
            'error': 'Experiment exceeded 15 minute timeout',
            'gpu_id': gpu_id
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"[GPU {gpu_id}] ✗ EXCEPTION ({duration:.1f}s): {str(e)}")
        return {
            'short_title': short_title,
            'beta_skill_init': beta,
            'm_sat_init': m_sat,
            'gamma_student_init': gamma,
            'sigmoid_offset': offset,
            'bce_loss_weight': BCE_LOSS_WEIGHT,
            'test_auc': None,
            'valid_auc': None,
            'encoder1_test_auc': None,
            'encoder2_test_auc': None,
            'duration': duration,
            'status': 'exception',
            'error': str(e)[:200],
            'gpu_id': gpu_id
        }

def main():
    """Run complete sweep in parallel"""
    
    print("\n" + "=" * 80)
    print("PHASE 1 LEARNING CURVE PARAMETER SWEEP - BCE LOSS WEIGHT = 0.9 (PARALLEL)")
    print("=" * 80)
    print()
    print(f"Sweep Directory:       {SWEEP_DIR}")
    print(f"Experiments Directory: {EXPERIMENTS_DIR}")
    print(f"Results CSV:           {CSV_FILE}")
    print()
    print("Parameter Grid:")
    print(f"  beta_skill_init:      {BETA_SKILL_VALUES}")
    print(f"  m_sat_init:           {M_SAT_VALUES}")
    print(f"  gamma_student_init:   {GAMMA_STUDENT_VALUES}")
    print(f"  sigmoid_offset:       {SIGMOID_OFFSET_VALUES}")
    print(f"  bce_loss_weight:      {BCE_LOSS_WEIGHT} (fixed)")
    print()
    print(f"Parallel Execution:")
    print(f"  GPUs used:            {NUM_GPUS} (IDs: {GPU_IDS})")
    print(f"  Max parallel jobs:    {NUM_GPUS}")
    print()
    
    # Calculate total experiments
    total_exps = (len(BETA_SKILL_VALUES) * len(M_SAT_VALUES) * 
                  len(GAMMA_STUDENT_VALUES) * len(SIGMOID_OFFSET_VALUES))
    print(f"Total Experiments: {total_exps}")
    print(f"Estimated Duration: ~{total_exps * 4.5 / 60 / NUM_GPUS:.1f} hours (parallel on {NUM_GPUS} GPUs)")
    print("=" * 80)
    print()
    
    # Initialize CSV
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'short_title', 'beta_skill_init', 'm_sat_init', 'gamma_student_init',
            'sigmoid_offset', 'bce_loss_weight', 'test_auc', 'valid_auc',
            'encoder1_test_auc', 'encoder2_test_auc', 'duration', 'status', 
            'exp_dir', 'gpu_id'
        ])
        writer.writeheader()
    
    # Prepare all experiment configurations
    experiments = []
    exp_num = 0
    for beta in BETA_SKILL_VALUES:
        for m_sat in M_SAT_VALUES:
            for gamma in GAMMA_STUDENT_VALUES:
                for offset in SIGMOID_OFFSET_VALUES:
                    exp_num += 1
                    # Assign GPU in round-robin fashion
                    gpu_id = (exp_num - 1) % NUM_GPUS
                    experiments.append((beta, m_sat, gamma, offset, exp_num, total_exps, gpu_id))
    
    # Run experiments in parallel
    results = []
    start_time = time.time()
    completed = 0
    
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        # Submit all jobs
        future_to_exp = {executor.submit(run_experiment_on_gpu, exp): exp for exp in experiments}
        
        # Process results as they complete
        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Append to CSV immediately (thread-safe)
                with csv_lock:
                    with open(CSV_FILE, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            'short_title', 'beta_skill_init', 'm_sat_init', 
                            'gamma_student_init', 'sigmoid_offset', 'bce_loss_weight',
                            'test_auc', 'valid_auc', 'encoder1_test_auc', 
                            'encoder2_test_auc', 'duration', 'status', 'exp_dir', 'gpu_id'
                        ])
                        writer.writerow(result)
                
                print(f"\n[PROGRESS] Completed {completed}/{total_exps} experiments ({completed/total_exps*100:.1f}%)\n")
                
            except Exception as e:
                print(f"Exception processing result: {e}")
    
    total_duration = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("SWEEP COMPLETED")
    print("=" * 80)
    print(f"Total Duration: {total_duration / 3600:.2f} hours")
    print(f"Results saved to: {CSV_FILE}")
    print(f"Experiments saved to: {EXPERIMENTS_DIR}")
    print()
    
    # Calculate statistics
    successful = [r for r in results if r['status'] == 'success' and r['test_auc'] is not None]
    
    if successful:
        successful_sorted = sorted(successful, key=lambda x: x['test_auc'], reverse=True)
        
        print(f"Successful Experiments: {len(successful)}/{total_exps}")
        print()
        print("Top 5 Configurations:")
        print("-" * 80)
        for i, result in enumerate(successful_sorted[:5], 1):
            print(f"{i}. {result['short_title']}")
            print(f"   Test AUC: {result['test_auc']:.4f} (GPU {result['gpu_id']})")
            print(f"   Beta={result['beta_skill_init']}, M_sat={result['m_sat_init']}, "
                  f"Gamma={result['gamma_student_init']}, Offset={result['sigmoid_offset']}")
        
        print()
        print(f"Best Test AUC:     {successful_sorted[0]['test_auc']:.4f}")
        print(f"Worst Test AUC:    {successful_sorted[-1]['test_auc']:.4f}")
        print(f"Mean Test AUC:     {sum(r['test_auc'] for r in successful) / len(successful):.4f}")
        print()
        print("Comparison with Experiment 714616 (dual-loss):")
        print(f"  Exp 714616 Test AUC:  0.7183")
        print(f"  Best Sweep AUC:       {successful_sorted[0]['test_auc']:.4f}")
        print(f"  Difference:           {successful_sorted[0]['test_auc'] - 0.7183:+.4f} ({(successful_sorted[0]['test_auc'] - 0.7183) / 0.7183 * 100:+.2f}%)")
    
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Analyze results: {CSV_FILE}")
    print("2. Update report.md with findings")
    print("3. Compare best config with Experiment 714616")
    print("4. Decide whether to update defaults for bce_loss_weight=0.9 use case")
    print()

if __name__ == "__main__":
    main()
