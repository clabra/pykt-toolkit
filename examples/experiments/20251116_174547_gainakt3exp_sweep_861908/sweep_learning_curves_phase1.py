#!/usr/bin/env python3
"""
Phase 1: Learning Curve Parameter Sweep
========================================

Objective: Find optimal learning curve parameters by giving Encoder 2 strong training signal.

Strategy:
- Fixed: bce_loss_weight = 0.2 (80% signal to Encoder 2 via IM loss)
- Sweep: beta_skill_init, m_sat_init, gamma_student_init, sigmoid_offset
- Metric: Encoder 2 test AUC (interpretability quality)
- Epochs: 6 (balance convergence vs speed)

Rationale:
Learning curve parameters are trainable (torch.nn.Parameter) and learn via Encoder 2 
gradients. Strong IM loss weight (0.8) ensures sufficient gradient signal for parameter 
optimization. Default weight (0.8 BCE, 0.2 IM) provides only 20% signal to Encoder 2, 
insufficient for distinguishing parameter quality.

Copyright (c) 2025 Concha Labra. All Rights Reserved.
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from itertools import product
import argparse
import csv


def run_experiment(params, gpu_id, epochs, base_dataset, base_fold):
    """Run a single experiment with given parameters."""
    
    # Generate short title
    short_title = (
        f"lc-sweep_beta{params['beta_skill_init']:.1f}_"
        f"msat{params['m_sat_init']:.1f}_"
        f"gamma{params['gamma_student_init']:.1f}_"
        f"offset{params['sigmoid_offset']:.1f}"
    )
    
    # Build command
    python_path = sys.executable
    cmd = [
        python_path,
        'examples/run_repro_experiment.py',
        '--short_title', short_title,
        '--epochs', str(epochs),
        '--dataset', base_dataset,
        '--fold', str(base_fold),
        '--bce_loss_weight', str(params['bce_loss_weight']),
        '--beta_skill_init', str(params['beta_skill_init']),
        '--m_sat_init', str(params['m_sat_init']),
        '--gamma_student_init', str(params['gamma_student_init']),
        '--sigmoid_offset', str(params['sigmoid_offset']),
    ]
    
    # Set environment for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    start_time = time.time()
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 Starting: {short_title}")
        print(f"   GPU: {gpu_id} | Epochs: {epochs}")
        print(f"   beta={params['beta_skill_init']}, m_sat={params['m_sat_init']}, "
              f"gamma={params['gamma_student_init']}, offset={params['sigmoid_offset']}")
        print(f"{'='*80}\n")
        
        result = subprocess.run(
            cmd,
            cwd="/workspaces/pykt-toolkit",
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env=env
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed: {short_title} ({duration/60:.1f}min)")
            
            # Extract metrics from output
            metrics = extract_metrics_from_output(result.stdout + "\n" + result.stderr)
            
            return {
                'short_title': short_title,
                'gpu_id': gpu_id,
                'parameters': params,
                'metrics': metrics,
                'status': 'success',
                'duration_minutes': duration / 60
            }
        else:
            print(f"❌ Failed: {short_title} (code: {result.returncode})")
            print(f"Error: {result.stderr[-500:]}")
            return {
                'short_title': short_title,
                'gpu_id': gpu_id,
                'parameters': params,
                'status': 'failed',
                'error': result.stderr[-500:],
                'duration_minutes': duration / 60
            }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ Timeout: {short_title} after {duration/60:.1f}min")
        return {
            'short_title': short_title,
            'gpu_id': gpu_id,
            'parameters': params,
            'status': 'timeout',
            'duration_minutes': duration / 60
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 Exception: {short_title} - {str(e)}")
        return {
            'short_title': short_title,
            'gpu_id': gpu_id,
            'parameters': params,
            'status': 'error',
            'error': str(e),
            'duration_minutes': duration / 60
        }


def extract_metrics_from_output(output_text):
    """Extract key metrics from training output."""
    metrics = {}
    
    lines = output_text.split('\n')
    
    # Look for metrics in output
    for line in lines:
        # Test metrics from evaluation
        if 'Test metrics:' in line or 'test_auc' in line.lower():
            try:
                if 'encoder1_auc' in line.lower():
                    metrics['encoder1_test_auc'] = float(line.split('=')[-1].strip())
                elif 'encoder2_auc' in line.lower():
                    metrics['encoder2_test_auc'] = float(line.split('=')[-1].strip())
                elif 'test_auc' in line.lower() and 'encoder' not in line.lower():
                    metrics['test_auc'] = float(line.split('=')[-1].strip())
            except Exception:
                pass
        
        # Extract from metrics_epoch_eval.csv if mentioned
        if 'metrics_epoch_eval.csv' in line or 'Evaluation complete' in line:
            # Try to find experiment directory
            for search_line in lines:
                if 'Experiment directory:' in search_line or 'experiments/' in search_line:
                    try:
                        # Extract experiment directory
                        parts = search_line.split('experiments/')
                        if len(parts) > 1:
                            exp_dir = parts[1].split()[0].strip()
                            metrics_file = f'examples/experiments/{exp_dir}/metrics_epoch_eval.csv'
                            if os.path.exists(metrics_file):
                                # Read last row of CSV
                                with open(metrics_file, 'r') as f:
                                    reader = csv.DictReader(f)
                                    rows = list(reader)
                                    if rows:
                                        last_row = rows[-1]
                                        metrics['encoder1_test_auc'] = float(last_row.get('test_encoder1_auc', 0))
                                        metrics['encoder2_test_auc'] = float(last_row.get('test_encoder2_auc', 0))
                                        metrics['test_auc'] = float(last_row.get('test_auc', 0))
                                        break
                    except Exception as e:
                        print(f"Warning: Could not extract metrics from CSV: {e}")
    
    return metrics


def generate_parameter_grid():
    """Generate parameter grid for sweep."""
    
    # Phase 1: Learning curve parameter sweep with fixed high IM weight
    grid = {
        'bce_loss_weight': [0.2],  # Fixed: 80% to Encoder 2
        'beta_skill_init': [1.5, 2.0, 2.5, 3.0],  # Learning rate amplification
        'm_sat_init': [0.7, 0.8, 0.9],  # Maximum mastery saturation
        'gamma_student_init': [0.8, 1.0, 1.2],  # Student learning velocity
        'sigmoid_offset': [1.5, 2.0, 2.5, 3.0],  # Sigmoid inflection point
    }
    
    # Generate all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(product(*values))
    
    param_list = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        param_list.append(params)
    
    return param_list


def save_results(results, output_file):
    """Save results to CSV and JSON."""
    
    # Save detailed JSON
    json_file = output_file.replace('.csv', '.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Detailed results saved to: {json_file}")
    
    # Save CSV summary
    if results:
        fieldnames = [
            'short_title', 'status', 'gpu_id', 'duration_minutes',
            'beta_skill_init', 'm_sat_init', 'gamma_student_init', 'sigmoid_offset',
            'bce_loss_weight', 'encoder2_test_auc', 'encoder1_test_auc', 'test_auc'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'short_title': result.get('short_title', ''),
                    'status': result.get('status', ''),
                    'gpu_id': result.get('gpu_id', ''),
                    'duration_minutes': f"{result.get('duration_minutes', 0):.2f}",
                    'beta_skill_init': result['parameters'].get('beta_skill_init', ''),
                    'm_sat_init': result['parameters'].get('m_sat_init', ''),
                    'gamma_student_init': result['parameters'].get('gamma_student_init', ''),
                    'sigmoid_offset': result['parameters'].get('sigmoid_offset', ''),
                    'bce_loss_weight': result['parameters'].get('bce_loss_weight', ''),
                    'encoder2_test_auc': f"{result.get('metrics', {}).get('encoder2_test_auc', 0):.4f}",
                    'encoder1_test_auc': f"{result.get('metrics', {}).get('encoder1_test_auc', 0):.4f}",
                    'test_auc': f"{result.get('metrics', {}).get('test_auc', 0):.4f}",
                }
                writer.writerow(row)
        
        print(f"📊 Summary CSV saved to: {output_file}")
        
        # Print top 5 by Encoder 2 AUC
        sorted_results = sorted(
            [r for r in results if r.get('status') == 'success'],
            key=lambda x: x.get('metrics', {}).get('encoder2_test_auc', 0),
            reverse=True
        )
        
        print(f"\n{'='*80}")
        print("🏆 TOP 5 CONFIGURATIONS BY ENCODER 2 TEST AUC:")
        print(f"{'='*80}")
        for i, result in enumerate(sorted_results[:5], 1):
            params = result['parameters']
            metrics = result.get('metrics', {})
            print(f"\n{i}. {result['short_title']}")
            print(f"   Beta: {params['beta_skill_init']}, M_sat: {params['m_sat_init']}, "
                  f"Gamma: {params['gamma_student_init']}, Offset: {params['sigmoid_offset']}")
            print(f"   Encoder2 AUC: {metrics.get('encoder2_test_auc', 0):.4f} | "
                  f"Encoder1 AUC: {metrics.get('encoder1_test_auc', 0):.4f} | "
                  f"Overall AUC: {metrics.get('test_auc', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Learning Curve Parameter Sweep')
    parser.add_argument('--epochs', type=int, default=6,
                        help='Number of epochs per experiment (default: 6)')
    parser.add_argument('--max_parallel', type=int, default=5,
                        help='Maximum parallel jobs (default: 5 GPUs)')
    parser.add_argument('--dataset', type=str, default='assist2009',
                        help='Dataset to use (default: assist2009)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number (default: 0)')
    parser.add_argument('--output_dir', type=str, default='examples/sweep_results',
                        help='Output directory for results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print parameter grid without running')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate parameter grid
    param_list = generate_parameter_grid()
    
    print(f"\n{'='*80}")
    print("PHASE 1: LEARNING CURVE PARAMETER SWEEP")
    print(f"{'='*80}")
    print("Strategy: High IM Loss Weight (bce_loss_weight=0.2 → 80% to Encoder 2)")
    print(f"Total experiments: {len(param_list)}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Max parallel jobs: {args.max_parallel}")
    print(f"Dataset: {args.dataset}, Fold: {args.fold}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Print parameter grid
    print("Parameter Grid:")
    print(f"  beta_skill_init: {sorted(set(p['beta_skill_init'] for p in param_list))}")
    print(f"  m_sat_init: {sorted(set(p['m_sat_init'] for p in param_list))}")
    print(f"  gamma_student_init: {sorted(set(p['gamma_student_init'] for p in param_list))}")
    print(f"  sigmoid_offset: {sorted(set(p['sigmoid_offset'] for p in param_list))}")
    print(f"  bce_loss_weight: {sorted(set(p['bce_loss_weight'] for p in param_list))}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No experiments will be executed")
        return
    
    # Confirm before starting
    print(f"\n⚠️  This will run {len(param_list)} experiments (~{len(param_list) * args.epochs * 3 / 60:.1f} hours estimated)")
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments sequentially (can be parallelized with ProcessPoolExecutor if needed)
    results = []
    start_time = time.time()
    
    for idx, params in enumerate(param_list, 1):
        gpu_id = (idx - 1) % args.max_parallel  # Round-robin GPU assignment
        
        print(f"\n[{idx}/{len(param_list)}] Running experiment...")
        result = run_experiment(params, gpu_id, args.epochs, args.dataset, args.fold)
        results.append(result)
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{args.output_dir}/phase1_sweep_{timestamp}.csv"
        save_results(results, output_file)
    
    total_duration = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.get('status') == 'failed')}")
    print(f"Timeout: {sum(1 for r in results if r.get('status') == 'timeout')}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
