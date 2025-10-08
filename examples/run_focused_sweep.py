#!/usr/bin/env python3
"""
All-in-one focused parameter sweep for GainAKT2Exp
Runs multiple experiments with parameter combinations around current defaults to find AUC >= 0.7259
"""

import os
import sys
import subprocess
from datetime import datetime
import random

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

def generate_parameter_combinations(count=20):
    """Generate parameter combinations around current defaults"""
    
    combinations = []
    
    for i in range(count):
        # Generate parameters with variations around defaults
        combo = {
            'learning_rate': random.uniform(0.0001, 0.0008),
            'weight_decay': random.uniform(0.00002, 0.0002),
            'batch_size': random.choice([64, 96, 128, 160, 192]),
            'num_epochs': random.choice([15, 18, 20, 22, 25, 30]),
            'enhanced_constraints': random.choice([True, False]),
            'patience': random.choice([15, 20, 25]),
            'dataset_name': 'assist2015',
            'fold': 0,
            'use_wandb': 0,  # Disable wandb for individual runs
            'experiment_suffix': f'focused_sweep_run_{i+1}'
        }
        combinations.append(combo)
    
    return combinations

def run_single_experiment(params, run_id):
    """Run a single experiment with given parameters"""
    
    print(f"\\n🔬 Run {run_id}: lr={params['learning_rate']:.6f}, wd={params['weight_decay']:.6f}")
    print(f"     bs={params['batch_size']}, epochs={params['num_epochs']}, enhanced={params['enhanced_constraints']}")
    
    # Build command
    cmd = [
        'python', 'wandb_gainakt2exp_train.py',
        '--learning_rate', str(params['learning_rate']),
        '--weight_decay', str(params['weight_decay']),
        '--batch_size', str(params['batch_size']),
        '--num_epochs', str(params['num_epochs']),
        '--enhanced_constraints', str(params['enhanced_constraints']),
        '--patience', str(params['patience']),
        '--use_wandb', '0',
        '--experiment_suffix', params['experiment_suffix']
    ]
    
    try:
        # Run experiment with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            # Parse AUC from output
            best_val_auc = None
            for line in result.stdout.split('\\n'):
                if 'FINAL_RESULTS: best_val_auc:' in line:
                    best_val_auc = float(line.split(':')[-1].strip())
                    break
            
            if best_val_auc is not None:
                status = "✅ SUCCESS" if best_val_auc >= 0.7259 else "📊 COMPLETED"
                print(f"     {status} - AUC: {best_val_auc:.4f}")
                return {
                    'run_id': run_id,
                    'params': params,
                    'auc': best_val_auc,
                    'status': 'success'
                }
            else:
                print(f"     ❌ FAILED - Could not parse AUC")
                return {'run_id': run_id, 'params': params, 'auc': 0.0, 'status': 'failed'}
        else:
            print(f"     ❌ FAILED - Return code: {result.returncode}")
            return {'run_id': run_id, 'params': params, 'auc': 0.0, 'status': 'failed'}
            
    except subprocess.TimeoutExpired:
        print(f"     ⏰ TIMEOUT - Experiment took >1 hour")
        return {'run_id': run_id, 'params': params, 'auc': 0.0, 'status': 'timeout'}
    except Exception as e:
        print(f"     ❌ ERROR - {e}")
        return {'run_id': run_id, 'params': params, 'auc': 0.0, 'status': 'error'}

def main():
    """Run focused parameter sweep"""
    
    print("🎯 GainAKT2Exp Focused Parameter Sweep")
    print("="*60)
    print("Goal: Find parameter combinations achieving AUC >= 0.7259")
    print("Current defaults: lr=0.0003, wd=0.000059, bs=128, epochs=20")
    
    # Generate parameter combinations
    num_experiments = 20
    print(f"\\n📋 Generating {num_experiments} parameter combinations...")
    combinations = generate_parameter_combinations(num_experiments)
    
    print("\\n🚀 Starting experiments...")
    results = []
    
    # Run experiments
    for i, params in enumerate(combinations, 1):
        result = run_single_experiment(params, i)
        results.append(result)
        
        # Show progress
        completed = len([r for r in results if r['status'] == 'success'])
        best_so_far = max([r['auc'] for r in results], default=0.0)
        print(f"     Progress: {i}/{num_experiments}, Best AUC so far: {best_so_far:.4f}")
    
    # Analyze results
    print(f"\\n📊 SWEEP RESULTS")
    print("="*60)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if successful_results:
        # Sort by AUC
        successful_results.sort(key=lambda x: x['auc'], reverse=True)
        
        print(f"✅ Successful runs: {len(successful_results)}/{num_experiments}")
        print(f"🏆 Best AUC: {successful_results[0]['auc']:.4f}")
        
        # Show top 5 results
        print("\\n🥇 TOP 5 RESULTS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'AUC':<8} {'LR':<10} {'WD':<10} {'BS':<4} {'Epochs':<6} {'Enhanced'}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results[:5], 1):
            params = result['params']
            print(f"{i:<4} {result['auc']:<8.4f} {params['learning_rate']:<10.6f} "
                  f"{params['weight_decay']:<10.6f} {params['batch_size']:<4} "
                  f"{params['num_epochs']:<6} {params['enhanced_constraints']}")
        
        # Check if target achieved
        target_runs = [r for r in successful_results if r['auc'] >= 0.7259]
        if target_runs:
            print(f"\\n🎉 TARGET ACHIEVED! {len(target_runs)} runs reached AUC >= 0.7259")
            best = target_runs[0]
            print(f"🥇 Best configuration:")
            for key, value in best['params'].items():
                if key not in ['dataset_name', 'fold', 'use_wandb', 'experiment_suffix']:
                    print(f"   {key}: {value}")
        else:
            print(f"\\n📈 Target not reached. Best AUC: {successful_results[0]['auc']:.4f}")
            print("Consider expanding search ranges or running more experiments.")
    
    else:
        print("❌ No successful experiments completed")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'focused_sweep_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("GainAKT2Exp Focused Sweep Results\\n")
        f.write("="*50 + "\\n")
        f.write(f"Timestamp: {datetime.now()}\\n")
        f.write(f"Total experiments: {num_experiments}\\n")
        f.write(f"Successful: {len(successful_results)}\\n")
        
        if successful_results:
            f.write(f"Best AUC: {successful_results[0]['auc']:.4f}\\n\\n")
            
            for i, result in enumerate(successful_results, 1):
                f.write(f"Run {result['run_id']}: AUC={result['auc']:.4f}\\n")
                for key, value in result['params'].items():
                    f.write(f"  {key}: {value}\\n")
                f.write("\\n")
    
    print(f"\\n💾 Results saved to: {results_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())