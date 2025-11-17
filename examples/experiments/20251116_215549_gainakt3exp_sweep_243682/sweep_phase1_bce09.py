#!/usr/bin/env python3
"""
Phase 1 Learning Curve Parameter Sweep with BCE Loss Weight = 0.9

This sweep replicates the Phase 1 methodology but keeps bce_loss_weight fixed at 0.9
to find optimal learning curve parameters for high performance-focused training.

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
- Other params: From configs/parameter_default.json (MD5: 92ab1f9df195f45b15f4e9859ae5c402)

Expected Duration: ~6 hours on 5 GPUs
Expected Outcome: Identify if Exp 714616's params are optimal or if better config exists
"""

import os
import sys
import subprocess
import time
import csv
from datetime import datetime
from pathlib import Path

# Sweep configuration
SWEEP_DIR = Path(__file__).parent
RESULTS_DIR = SWEEP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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

# Base command template
BASE_CMD = [
    "python", "examples/run_repro_experiment.py",
    "--dataset", DATASET,
    "--fold", str(FOLD),
    "--epochs", str(EPOCHS),
    "--bce_loss_weight", str(BCE_LOSS_WEIGHT),
]

def run_experiment(beta, m_sat, gamma, offset, exp_num, total_exps):
    """Run a single experiment configuration"""
    
    short_title = f"p1bce09_b{beta}_m{m_sat}_g{gamma}_o{offset}"
    
    print("=" * 80)
    print(f"EXPERIMENT {exp_num}/{total_exps}: {short_title}")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  beta_skill_init:      {beta}")
    print(f"  m_sat_init:           {m_sat}")
    print(f"  gamma_student_init:   {gamma}")
    print(f"  sigmoid_offset:       {offset}")
    print(f"  bce_loss_weight:      {BCE_LOSS_WEIGHT} (fixed)")
    print(f"  epochs:               {EPOCHS}")
    print("-" * 80)
    
    # Build command
    cmd = BASE_CMD + [
        "--short_title", short_title,
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
            cwd="/workspaces/pykt-toolkit"
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Parse results from output
            output = result.stdout
            
            # Extract experiment directory from output
            exp_dir = None
            for line in output.split('\n'):
                if 'Experiment directory:' in line or 'experiment directory:' in line:
                    exp_dir = line.split(':')[-1].strip()
                    break
            
            # Try to read eval_results.json
            test_auc = None
            valid_auc = None
            encoder1_auc = None
            encoder2_auc = None
            
            if exp_dir and os.path.exists(exp_dir):
                eval_file = os.path.join(exp_dir, "eval_results.json")
                if os.path.exists(eval_file):
                    import json
                    with open(eval_file) as f:
                        eval_data = json.load(f)
                        test_auc = eval_data.get('test_auc')
                        valid_auc = eval_data.get('valid_auc')
                        encoder1_auc = eval_data.get('encoder1_test_auc')
                        encoder2_auc = eval_data.get('encoder2_test_auc')
            
            print(f"✓ SUCCESS (duration: {duration:.1f}s)")
            print(f"  Test AUC:     {test_auc:.4f}" if test_auc else "  Test AUC: N/A")
            print(f"  Valid AUC:    {valid_auc:.4f}" if valid_auc else "  Valid AUC: N/A")
            print(f"  Encoder1 AUC: {encoder1_auc:.4f}" if encoder1_auc else "  Encoder1 AUC: N/A")
            print(f"  Encoder2 AUC: {encoder2_auc:.4f}" if encoder2_auc else "  Encoder2 AUC: N/A")
            
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
                'exp_dir': exp_dir
            }
        else:
            print(f"✗ FAILED (duration: {duration:.1f}s)")
            print(f"Error: {result.stderr[:500]}")
            
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
                'error': result.stderr[:200]
            }
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ EXCEPTION (duration: {duration:.1f}s)")
        print(f"Exception: {str(e)}")
        
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
            'error': str(e)[:200]
        }

def main():
    """Run complete sweep"""
    
    print("\n" + "=" * 80)
    print("PHASE 1 LEARNING CURVE PARAMETER SWEEP - BCE LOSS WEIGHT = 0.9")
    print("=" * 80)
    print()
    print(f"Sweep Directory: {SWEEP_DIR}")
    print(f"Results CSV:     {CSV_FILE}")
    print()
    print("Parameter Grid:")
    print(f"  beta_skill_init:      {BETA_SKILL_VALUES}")
    print(f"  m_sat_init:           {M_SAT_VALUES}")
    print(f"  gamma_student_init:   {GAMMA_STUDENT_VALUES}")
    print(f"  sigmoid_offset:       {SIGMOID_OFFSET_VALUES}")
    print(f"  bce_loss_weight:      {BCE_LOSS_WEIGHT} (fixed)")
    print()
    
    # Calculate total experiments
    total_exps = (len(BETA_SKILL_VALUES) * len(M_SAT_VALUES) * 
                  len(GAMMA_STUDENT_VALUES) * len(SIGMOID_OFFSET_VALUES))
    print(f"Total Experiments: {total_exps}")
    print(f"Estimated Duration: ~{total_exps * 4.5 / 60:.1f} hours (assuming ~4.5 min/exp)")
    print("=" * 80)
    print()
    
    # Initialize CSV
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'short_title', 'beta_skill_init', 'm_sat_init', 'gamma_student_init',
            'sigmoid_offset', 'bce_loss_weight', 'test_auc', 'valid_auc',
            'encoder1_test_auc', 'encoder2_test_auc', 'duration', 'status', 'exp_dir'
        ])
        writer.writeheader()
    
    # Run sweep
    results = []
    exp_num = 0
    start_time = time.time()
    
    for beta in BETA_SKILL_VALUES:
        for m_sat in M_SAT_VALUES:
            for gamma in GAMMA_STUDENT_VALUES:
                for offset in SIGMOID_OFFSET_VALUES:
                    exp_num += 1
                    
                    result = run_experiment(beta, m_sat, gamma, offset, exp_num, total_exps)
                    results.append(result)
                    
                    # Append to CSV immediately
                    with open(CSV_FILE, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            'short_title', 'beta_skill_init', 'm_sat_init', 
                            'gamma_student_init', 'sigmoid_offset', 'bce_loss_weight',
                            'test_auc', 'valid_auc', 'encoder1_test_auc', 
                            'encoder2_test_auc', 'duration', 'status', 'exp_dir'
                        ])
                        writer.writerow(result)
                    
                    print()
    
    total_duration = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("SWEEP COMPLETED")
    print("=" * 80)
    print(f"Total Duration: {total_duration / 3600:.2f} hours")
    print(f"Results saved to: {CSV_FILE}")
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
            print(f"   Test AUC: {result['test_auc']:.4f}")
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
    print("1. Analyze results in CSV file")
    print("2. Update report.md with findings")
    print("3. Compare best config with Experiment 714616")
    print("4. Decide whether to update defaults for bce_loss_weight=0.9 use case")
    print()

if __name__ == "__main__":
    main()
