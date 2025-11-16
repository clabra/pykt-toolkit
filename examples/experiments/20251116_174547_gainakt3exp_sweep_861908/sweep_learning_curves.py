#!/usr/bin/env python3
"""
Phase 1: Learning Curve Parameter Sweep - REDUCED GRID
=======================================================

Quick validation sweep with reduced parameter space.

Strategy:
- Fixed: bce_loss_weight = 0.2 (80% signal to Encoder 2 via IM loss)
- Sweep: Reduced grid for faster validation
- Metric: Encoder 2 test AUC (interpretability quality)
- Epochs: 6 (balance convergence vs speed)
- Parallel: 5 GPUs simultaneously

Reduced Grid:
- beta_skill_init: [1.5, 2.0, 2.5] (3 values)
- m_sat_init: [0.7, 0.8, 0.9] (3 values)
- gamma_student_init: [0.9, 1.0, 1.1] (3 values)
- sigmoid_offset: match beta_skill_init [1.5, 2.0, 2.5] (3 values)
Total: 27 experiments (~3 hours with 5 parallel GPUs)

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue


def run_experiment(params, gpu_id, epochs, base_dataset, base_fold):
    """Run a single experiment with given parameters."""
    
    # Generate short title
    short_title = (
        f"lc_b{params['beta_skill_init']:.1f}_"
        f"m{params['m_sat_init']:.1f}_"
        f"g{params['gamma_student_init']:.1f}_"
        f"o{params['sigmoid_offset']:.1f}"
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
        print(f"🚀 GPU {gpu_id} | Starting: {short_title}")
        print(f"   Epochs: {epochs} | beta={params['beta_skill_init']}, m_sat={params['m_sat_init']}, "
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
            print(f"✅ GPU {gpu_id} | Completed: {short_title} ({duration/60:.1f}min)")
            
            # Extract experiment directory from output
            exp_dir = None
            for line in (result.stdout + "\n" + result.stderr).split('\n'):
                if 'Experiment directory:' in line or 'experiments/2025' in line:
                    parts = line.split('experiments/')
                    if len(parts) > 1:
                        exp_dir = 'examples/experiments/' + parts[1].split()[0].strip()
                        break
            
            # Extract metrics
            metrics = {}
            if exp_dir:
                metrics_file = os.path.join(exp_dir, 'metrics_epoch_eval.csv')
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            # Find the test row (last row should be test split)
                            test_row = None
                            for row in rows:
                                if row.get('split', '').lower() == 'test':
                                    test_row = row
                                    break
                            if test_row:
                                metrics['encoder1_test_auc'] = float(test_row.get('encoder1_auc', 0))
                                metrics['encoder2_test_auc'] = float(test_row.get('encoder2_auc', 0))
                                metrics['test_auc'] = float(test_row.get('auc', 0))
                                metrics['test_acc'] = float(test_row.get('accuracy', 0))
                    except Exception as e:
                        print(f"⚠️  Could not read metrics file: {e}")
            
            return {
                'short_title': short_title,
                'gpu_id': gpu_id,
                'parameters': params,
                'metrics': metrics,
                'exp_dir': exp_dir,
                'status': 'success',
                'duration_minutes': duration / 60
            }
        else:
            print(f"❌ GPU {gpu_id} | Failed: {short_title} (code: {result.returncode})")
            error_msg = result.stderr[-500:] if result.stderr else "No error message"
            print(f"Error: {error_msg}")
            return {
                'short_title': short_title,
                'gpu_id': gpu_id,
                'parameters': params,
                'status': 'failed',
                'error': error_msg,
                'duration_minutes': duration / 60
            }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ GPU {gpu_id} | Timeout: {short_title} after {duration/60:.1f}min")
        return {
            'short_title': short_title,
            'gpu_id': gpu_id,
            'parameters': params,
            'status': 'timeout',
            'duration_minutes': duration / 60
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 GPU {gpu_id} | Exception: {short_title} - {str(e)}")
        return {
            'short_title': short_title,
            'gpu_id': gpu_id,
            'parameters': params,
            'status': 'error',
            'error': str(e),
            'duration_minutes': duration / 60
        }


def generate_parameter_grid(reduced=True):
    """Generate parameter grid for sweep."""
    
    if reduced:
        # Reduced grid for faster validation (27 experiments)
        grid = {
            'bce_loss_weight': [0.2],  # Fixed: 80% to Encoder 2
            'beta_skill_init': [1.5, 2.0, 2.5],  # Reduced from 4 to 3 values
            'm_sat_init': [0.7, 0.8, 0.9],  # Keep 3 values
            'gamma_student_init': [0.9, 1.0, 1.1],  # Narrow range around 1.0
            'sigmoid_offset': [1.5, 2.0, 2.5],  # Match beta_skill_init
        }
    else:
        # Full grid (144 experiments)
        grid = {
            'bce_loss_weight': [0.2],  # Fixed: 80% to Encoder 2
            'beta_skill_init': [1.5, 2.0, 2.5, 3.0],
            'm_sat_init': [0.7, 0.8, 0.9],
            'gamma_student_init': [0.8, 1.0, 1.2],
            'sigmoid_offset': [1.5, 2.0, 2.5, 3.0],
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
            'bce_loss_weight', 'encoder2_test_auc', 'encoder1_test_auc', 'test_auc', 'test_acc',
            'exp_dir'
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
                    'test_acc': f"{result.get('metrics', {}).get('test_acc', 0):.4f}",
                    'exp_dir': result.get('exp_dir', ''),
                }
                writer.writerow(row)
        
        print(f"📊 Summary CSV saved to: {output_file}")
        
        # Print top 5 by Encoder 2 AUC
        sorted_results = sorted(
            [r for r in results if r.get('status') == 'success' and r.get('metrics', {}).get('encoder2_test_auc', 0) > 0],
            key=lambda x: x.get('metrics', {}).get('encoder2_test_auc', 0),
            reverse=True
        )
        
        if sorted_results:
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
                print(f"   Experiment: {result.get('exp_dir', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Learning Curve Parameter Sweep')
    parser.add_argument('--epochs', type=int, default=6,
                        help='Number of epochs per experiment (default: 6)')
    parser.add_argument('--max_parallel', type=int, default=7,
                        help='Maximum parallel jobs (default: 7 GPUs, leaving 1 for system)')
    parser.add_argument('--dataset', type=str, default='assist2015',
                        help='Dataset to use (default: assist2015)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number (default: 0)')
    parser.add_argument('--output_dir', type=str, default='examples/sweep_results',
                        help='Output directory for results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print parameter grid without running')
    parser.add_argument('--full_grid', action='store_true',
                        help='Use full parameter grid (144 experiments) instead of reduced (27)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate parameter grid
    param_list = generate_parameter_grid(reduced=not args.full_grid)
    
    grid_type = "FULL" if args.full_grid else "REDUCED"
    print(f"\n{'='*80}")
    print(f"PHASE 1: LEARNING CURVE PARAMETER SWEEP ({grid_type} GRID)")
    print(f"{'='*80}")
    print("Strategy: High IM Loss Weight (bce_loss_weight=0.2 → 80% to Encoder 2)")
    print(f"Total experiments: {len(param_list)}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Max parallel jobs: {args.max_parallel}")
    print(f"Dataset: {args.dataset}, Fold: {args.fold}")
    print(f"Estimated time: ~{len(param_list) * args.epochs * 3 / (60 * args.max_parallel):.1f} hours")
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
    print(f"\n⚠️  This will run {len(param_list)} experiments")
    print(f"   Estimated time: ~{len(param_list) * args.epochs * 3 / (60 * args.max_parallel):.1f} hours")
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments in parallel using ProcessPoolExecutor
    results = []
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/phase1_sweep_{timestamp}.csv"
    
    # Create GPU queue for round-robin assignment
    gpu_queue = queue.Queue()
    for gpu_id in range(args.max_parallel):
        gpu_queue.put(gpu_id)
    
    print(f"\n{'='*80}")
    print(f"STARTING PARALLEL SWEEP WITH {args.max_parallel} GPUs")
    print(f"{'='*80}\n")
    
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        # Submit all jobs
        future_to_params = {}
        for idx, params in enumerate(param_list, 1):
            gpu_id = (idx - 1) % args.max_parallel
            future = executor.submit(run_experiment, params, gpu_id, args.epochs, args.dataset, args.fold)
            future_to_params[future] = (idx, params)
        
        # Collect results as they complete
        for future in as_completed(future_to_params):
            idx, params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
                
                # Save intermediate results after each completion
                save_results(results, output_file)
                
                print(f"\n📊 Progress: {len(results)}/{len(param_list)} experiments complete")
                
            except Exception as e:
                print(f"💥 Exception processing experiment {idx}: {str(e)}")
                results.append({
                    'short_title': f'exp_{idx}',
                    'parameters': params,
                    'status': 'exception',
                    'error': str(e)
                })
    
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
    
    # Save final results
    save_results(results, output_file)


if __name__ == '__main__':
    main()
