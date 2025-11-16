#!/usr/bin/env python3
"""
Phase 2: Loss Weight Balancing Sweep
=====================================

Sweep bce_loss_weight to find optimal balance between Encoder 1 (performance) 
and Encoder 2 (interpretability) using optimal learning curve parameters from Phase 1.

Strategy:
- Fixed: Optimal learning curve parameters from Phase 1
  - beta_skill_init = 2.5
  - m_sat_init = 0.7
  - gamma_student_init = 1.1
  - sigmoid_offset = 1.5
- Sweep: bce_loss_weight in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- Metric: Combined score (E1_AUC + alpha * E2_AUC, where alpha balances objectives)
- Epochs: 12 (production quality convergence)
- Parallel: 6 GPUs simultaneously

Total: 6 experiments (~2-3 hours with 6 parallel GPUs)

Expected Outcome:
- Identify optimal bce_loss_weight balancing performance and interpretability
- Likely optimal range: 0.5-0.7 (more balanced than Phase 1's 0.2)

Copyright (c) 2025 Concha Labra. All Rights Reserved.
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import random


def run_experiment(params, gpu_id, epochs, base_dataset, base_fold):
    """Run a single experiment with given parameters."""
    
    # Generate short title
    bce_weight = params['bce_loss_weight']
    short_title = f"phase2_bce{bce_weight:.1f}"
    
    # Build command
    python_path = sys.executable
    cmd = [
        python_path,
        'examples/run_repro_experiment.py',
        '--short_title', short_title,
        '--epochs', str(epochs),
        '--dataset', base_dataset,
        '--fold', str(base_fold),
        '--bce_loss_weight', str(bce_weight),
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
        print(f"   BCE weight: {bce_weight:.1f} | Epochs: {epochs}")
        print(f"   Fixed params: beta=2.5, m_sat=0.7, gamma=1.1, offset=1.5")
        print(f"{'='*80}\n")
        
        result = subprocess.run(
            cmd,
            cwd="/workspaces/pykt-toolkit",
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout (longer epochs)
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
            
            # Extract metrics from metrics_epoch_eval.csv
            metrics = extract_metrics_from_dir(exp_dir) if exp_dir else {}
            
            return {
                'success': True,
                'short_title': short_title,
                'bce_loss_weight': bce_weight,
                'exp_dir': exp_dir,
                'duration_min': duration / 60,
                **metrics
            }
        else:
            print(f"❌ GPU {gpu_id} | Failed: {short_title}")
            print(f"   Error: {result.stderr[:500]}")
            return {
                'success': False,
                'short_title': short_title,
                'bce_loss_weight': bce_weight,
                'error': result.stderr[:500]
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  GPU {gpu_id} | Timeout: {short_title}")
        return {
            'success': False,
            'short_title': short_title,
            'bce_loss_weight': bce_weight,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"💥 GPU {gpu_id} | Exception: {short_title}")
        print(f"   Error: {str(e)}")
        return {
            'success': False,
            'short_title': short_title,
            'bce_loss_weight': bce_weight,
            'error': str(e)
        }


def extract_metrics_from_dir(exp_dir):
    """Extract test metrics from experiment directory."""
    if not exp_dir or not os.path.exists(exp_dir):
        return {
            'encoder1_test_auc': 0.0,
            'encoder2_test_auc': 0.0,
            'test_auc': 0.0,
            'test_acc': 0.0
        }
    
    metrics_file = os.path.join(exp_dir, 'metrics_epoch_eval.csv')
    
    if not os.path.exists(metrics_file):
        print(f"⚠️  Metrics file not found: {metrics_file}")
        return {
            'encoder1_test_auc': 0.0,
            'encoder2_test_auc': 0.0,
            'test_auc': 0.0,
            'test_acc': 0.0
        }
    
    try:
        # Read CSV and find test row
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('split', '').strip() == 'test':
                    return {
                        'encoder1_test_auc': float(row.get('encoder1_auc', 0.0)),
                        'encoder2_test_auc': float(row.get('encoder2_auc', 0.0)),
                        'test_auc': float(row.get('auc', 0.0)),
                        'test_acc': float(row.get('acc', 0.0))
                    }
        
        print(f"⚠️  No test row found in {metrics_file}")
        return {
            'encoder1_test_auc': 0.0,
            'encoder2_test_auc': 0.0,
            'test_auc': 0.0,
            'test_acc': 0.0
        }
    except Exception as e:
        print(f"⚠️  Error reading metrics from {metrics_file}: {e}")
        return {
            'encoder1_test_auc': 0.0,
            'encoder2_test_auc': 0.0,
            'test_auc': 0.0,
            'test_acc': 0.0
        }


def generate_parameter_grid(bce_weights):
    """Generate parameter combinations for Phase 2 sweep."""
    
    # Fixed optimal parameters from Phase 1
    fixed_params = {
        'beta_skill_init': 2.5,
        'm_sat_init': 0.7,
        'gamma_student_init': 1.1,
        'sigmoid_offset': 1.5
    }
    
    # Generate combinations
    param_grid = []
    for bce_weight in bce_weights:
        params = fixed_params.copy()
        params['bce_loss_weight'] = bce_weight
        param_grid.append(params)
    
    return param_grid


def save_results(results, output_file):
    """Save results to CSV."""
    
    if not results:
        print("⚠️  No results to save")
        return
    
    # Sort by encoder2_test_auc descending
    results_sorted = sorted(
        [r for r in results if r.get('success', False)],
        key=lambda x: x.get('encoder2_test_auc', 0.0),
        reverse=True
    )
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'short_title', 'bce_loss_weight',
            'encoder1_test_auc', 'encoder2_test_auc', 'test_auc', 'test_acc',
            'combined_score_0.5', 'combined_score_1.0',
            'run_dir', 'duration_min'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results_sorted:
            # Compute combined scores with different alpha values
            e1_auc = r.get('encoder1_test_auc', 0.0)
            e2_auc = r.get('encoder2_test_auc', 0.0)
            
            row = {
                'short_title': r['short_title'],
                'bce_loss_weight': r['bce_loss_weight'],
                'encoder1_test_auc': f"{e1_auc:.4f}",
                'encoder2_test_auc': f"{e2_auc:.4f}",
                'test_auc': f"{r.get('test_auc', 0.0):.4f}",
                'test_acc': f"{r.get('test_acc', 0.0):.4f}",
                'combined_score_0.5': f"{e1_auc + 0.5 * e2_auc:.4f}",
                'combined_score_1.0': f"{e1_auc + 1.0 * e2_auc:.4f}",
                'run_dir': r.get('exp_dir', ''),
                'duration_min': f"{r.get('duration_min', 0.0):.1f}"
            }
            writer.writerow(row)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print top 3
    print(f"\n{'='*80}")
    print("TOP 3 CONFIGURATIONS BY ENCODER2 AUC:")
    print(f"{'='*80}")
    for i, r in enumerate(results_sorted[:3], 1):
        e1_auc = r.get('encoder1_test_auc', 0.0)
        e2_auc = r.get('encoder2_test_auc', 0.0)
        combined = e1_auc + 0.5 * e2_auc
        print(f"\n{i}. {r['short_title']}")
        print(f"   BCE weight: {r['bce_loss_weight']:.1f}")
        print(f"   Encoder1 AUC: {e1_auc:.4f}")
        print(f"   Encoder2 AUC: {e2_auc:.4f}")
        print(f"   Combined (α=0.5): {combined:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Loss Weight Balancing Sweep')
    parser.add_argument('--bce_loss_weights', type=str, default='0.3,0.4,0.5,0.6,0.7,0.8',
                        help='Comma-separated bce_loss_weight values to sweep')
    parser.add_argument('--epochs', type=int, default=12,
                        help='Epochs per experiment (default: 12)')
    parser.add_argument('--dataset', type=str, default='assist2015',
                        help='Dataset to use (default: assist2015)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Data fold (default: 0)')
    parser.add_argument('--max_parallel', type=int, default=6,
                        help='Maximum parallel experiments (default: 6)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configuration and exit without running')
    
    args = parser.parse_args()
    
    # Parse bce_loss_weights
    bce_weights = [float(x.strip()) for x in args.bce_loss_weights.split(',')]
    
    # Generate parameter grid
    param_grid = generate_parameter_grid(bce_weights)
    
    print("\n" + "="*80)
    print("PHASE 2: LOSS WEIGHT BALANCING SWEEP")
    print("="*80)
    print(f"\nFixed Parameters (from Phase 1 optimal):")
    print(f"  beta_skill_init: 2.5")
    print(f"  m_sat_init: 0.7")
    print(f"  gamma_student_init: 1.1")
    print(f"  sigmoid_offset: 1.5")
    print(f"\nSwept Parameters:")
    print(f"  bce_loss_weight: {bce_weights}")
    print(f"\nConfiguration:")
    print(f"  Total experiments: {len(param_grid)}")
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"  Dataset: {args.dataset}, fold {args.fold}")
    print(f"  Max parallel GPUs: {args.max_parallel}")
    print(f"  Estimated time: {len(param_grid) * 40 / args.max_parallel:.0f}-{len(param_grid) * 60 / args.max_parallel:.0f} minutes")
    print("="*80 + "\n")
    
    if args.dry_run:
        print("DRY RUN - Exiting without execution")
        return
    
    # Confirm
    response = input("Start Phase 2 sweep? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_id = random.randint(100000, 999999)
    output_dir = "examples/sweep_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = f"{output_dir}/phase2_sweep_{timestamp}.csv"
    output_json = f"{output_dir}/phase2_sweep_{timestamp}.json"
    
    # Run experiments in parallel
    results = []
    start_time = time.time()
    
    print(f"\n🚀 Starting {len(param_grid)} experiments on {args.max_parallel} GPUs...\n")
    
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        # Submit all jobs
        future_to_params = {}
        for i, params in enumerate(param_grid):
            gpu_id = i % args.max_parallel
            future = executor.submit(
                run_experiment,
                params, gpu_id, args.epochs, args.dataset, args.fold
            )
            future_to_params[future] = params
        
        # Collect results as they complete
        for future in as_completed(future_to_params):
            result = future.result()
            results.append(result)
            
            # Progress update
            completed = len(results)
            total = len(param_grid)
            print(f"\n📊 Progress: {completed}/{total} experiments completed")
    
    total_time = time.time() - start_time
    
    # Save results
    save_results(results, output_csv)
    
    # Save detailed JSON
    with open(output_json, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'sweep_id': sweep_id,
            'config': {
                'epochs': args.epochs,
                'dataset': args.dataset,
                'fold': args.fold,
                'bce_loss_weights': bce_weights,
                'fixed_params': {
                    'beta_skill_init': 2.5,
                    'm_sat_init': 0.7,
                    'gamma_student_init': 1.1,
                    'sigmoid_offset': 1.5
                }
            },
            'results': results,
            'summary': {
                'total_experiments': len(param_grid),
                'successful': sum(1 for r in results if r.get('success', False)),
                'failed': sum(1 for r in results if not r.get('success', False)),
                'total_time_hours': total_time / 3600
            }
        }, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {output_json}")
    
    # Final summary
    successful = [r for r in results if r.get('success', False)]
    print(f"\n{'='*80}")
    print("PHASE 2 SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(param_grid)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"\nResults: {output_csv}")
    print(f"Details: {output_json}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
