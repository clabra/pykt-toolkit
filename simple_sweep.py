#!/usr/bin/env python3
"""
Simple Comprehensive Sweep - 20 Parameter Combinations (4 per GPU)
Parallel execution with proper metrics extraction
"""

import subprocess
import os
import yaml
import random
import json
import time
from datetime import datetime
import re
import math


def sample_parameters(param_config):
    """Sample parameters based on wandb config format."""
    if 'values' in param_config:
        return random.choice(param_config['values'])
    elif 'distribution' in param_config:
        if param_config['distribution'] == 'uniform':
            return random.uniform(param_config['min'], param_config['max'])
        elif param_config['distribution'] == 'log_uniform_values':
            log_min = math.log(param_config['min'])
            log_max = math.log(param_config['max'])
            return math.exp(random.uniform(log_min, log_max))
    return param_config.get('value', 1.0)


def extract_final_results(output):
    """Extract FINAL_RESULTS from training output."""
    results = {}
    for line in output.split('\n'):
        if 'FINAL_RESULTS:' in line:
            try:
                parts = line.split(': ', 1)
                if len(parts) >= 2:
                    metric_name = parts[0].replace('FINAL_RESULTS: ', '').strip()
                    metric_value = float(parts[1].strip())
                    results[metric_name] = metric_value
            except Exception:
                pass
    return results





def main():
    print("🚀 SIMPLE COMPREHENSIVE SWEEP - 20 COMBINATIONS (4 per GPU)")
    print("=" * 70)
    
    # Load configuration
    config_file = 'sweep_config_comprehensive.yaml'
    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    parameters = sweep_config.get('parameters', {})
    num_runs = 20
    available_gpus = [0, 1, 2, 3, 4]
    runs_per_gpu = 4
    
    print(f"📊 Total parameter combinations: {num_runs}")
    print(f"🖥️  GPUs: {available_gpus}")
    print(f"🔄 Runs per GPU: {runs_per_gpu}")
    print("=" * 70)
    
    # Generate parameter combinations
    param_combinations = []
    for run_idx in range(num_runs):
        params = {}
        
        # Sample each parameter
        for param_name, param_config in parameters.items():
            if param_name in ['epochs', 'batch_size', 'lr', 'weight_decay', 'patience', 'enhanced_constraints']:
                params[param_name] = sample_parameters(param_config)
        
        param_combinations.append(params)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_sweep_results_{timestamp}.json"
    log_file = f"simple_sweep_{timestamp}.log"
    
    print(f"\n💾 Results: {results_file}")
    print(f"📜 Log: {log_file}")
    print("=" * 70)
    
    # Initialize log
    with open(log_file, 'w') as f:
        f.write(f"Simple Comprehensive Sweep Started: {datetime.now()}\n")
        f.write(f"Total runs: {num_runs}\n")
        f.write(f"GPUs: {available_gpus}\n")
        f.write(f"Runs per GPU: {runs_per_gpu}\n\n")
    
    # Print parameter assignments
    print("� PARAMETER ASSIGNMENTS:")
    for i, params in enumerate(param_combinations):
        run_id = i + 1
        gpu_id = available_gpus[(i // runs_per_gpu) % len(available_gpus)]
        print(f"   Run {run_id:2d} → GPU {gpu_id}: batch_size={params['batch_size']:3d}, lr={params['lr']:.4f}, enhanced_constraints={params['enhanced_constraints']}")
    
    print("=" * 70)
    print("🚀 Starting parallel execution...")
    
    # Run processes directly with simpler approach
    gpu_processes = []
    
    for i, params in enumerate(param_combinations):
        run_id = i + 1
        gpu_id = available_gpus[(i // runs_per_gpu) % len(available_gpus)]
        
        # Build command
        cmd = [
            'python', 'train_cumulative_mastery_full.py',
            '--epochs', str(params['epochs']),
            '--batch_size', str(params['batch_size']),
            '--lr', str(params['lr']),
            '--weight_decay', str(params['weight_decay']),
            '--patience', str(params['patience']),
            '--enhanced_constraints', str(params['enhanced_constraints']),
            '--use_wandb', 'True',
            '--experiment_suffix', f'simple_sweep_run_{run_id}'
        ]
        
        # Set GPU environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Start process
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd='/workspaces/pykt-toolkit',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        gpu_processes.append({
            'run_id': run_id,
            'gpu_id': gpu_id,
            'parameters': params,
            'process': process,
            'start_time': time.time()
        })
        
        print(f"   🚀 Started Run {run_id} on GPU {gpu_id}")
    
    # Wait for processes and collect results
    print(f"\n⏳ Waiting for {len(gpu_processes)} processes to complete...")
    all_results = []
    
    for proc_info in gpu_processes:
        process = proc_info['process']
        run_id = proc_info['run_id']
        gpu_id = proc_info['gpu_id']
        params = proc_info['parameters']
        start_time = proc_info['start_time']
        
        print(f"   Waiting for Run {run_id} (GPU {gpu_id})...")
        
        try:
            # Wait for process with timeout
            stdout, _ = process.communicate(timeout=3600)  # 1 hour timeout
            duration = time.time() - start_time
            
            # Extract results from output
            final_results = extract_final_results(stdout)
            
            # Log detailed output
            with open(log_file, 'a') as f:
                f.write(f"\n=== RUN {run_id} COMPLETED ===\n")
                f.write("Status: success\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Duration: {duration/60:.1f} min\n")
                f.write(f"Parameters: {params}\n")
                f.write(f"Results: {final_results}\n")
                f.write(f"\nTraining Output:\n{stdout}\n")
                f.write("-" * 50 + "\n")
            
            if process.returncode == 0:
                print(f"   ✅ Run {run_id} completed: AUC {final_results.get('best_val_auc', 0):.4f} in {duration/60:.1f}min")
                result = {
                    'run_id': run_id,
                    'gpu_id': gpu_id,
                    'parameters': params,
                    'results': final_results,
                    'status': 'success',
                    'duration_minutes': duration / 60
                }
            else:
                print(f"   ❌ Run {run_id} failed (code: {process.returncode})")
                result = {
                    'run_id': run_id,
                    'gpu_id': gpu_id,
                    'parameters': params,
                    'status': 'failed',
                    'error': f'Exit code: {process.returncode}',
                    'duration_minutes': duration / 60
                }
        
        except subprocess.TimeoutExpired:
            process.kill()
            duration = time.time() - start_time
            print(f"   ⏱️  Run {run_id} timed out after {duration/60:.1f}min")
            
            result = {
                'run_id': run_id,
                'gpu_id': gpu_id,
                'parameters': params,
                'status': 'timeout',
                'duration_minutes': duration / 60
            }
        
        all_results.append(result)
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Final analysis
    successful_runs = [r for r in all_results if r['status'] == 'success']
    
    print("\n" + "=" * 70)
    print("📋 FINAL SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(all_results)}")
    print(f"✅ Successful: {len(successful_runs)}")
    print(f"❌ Failed: {len(all_results) - len(successful_runs)}")
    
    if successful_runs:
        # Top 3 performers
        top_runs = sorted(successful_runs, key=lambda x: x['results'].get('best_val_auc', 0), reverse=True)[:3]
        
        print("\n🏆 TOP 3 PERFORMERS:")
        for i, run in enumerate(top_runs, 1):
            results = run['results']
            params = run['parameters']
            auc = results.get('best_val_auc', 0)
            
            print(f"\n   #{i}. RUN {run['run_id']} - AUC: {auc:.4f}")
            print(f"      🎯 GPU: {run['gpu_id']}, Duration: {run.get('duration_minutes', 0):.1f}min")
            print(f"      📋 batch_size={params.get('batch_size')}, lr={params.get('lr', 0):.4f}, enhanced_constraints={params.get('enhanced_constraints')}")
        
        # Log final summary
        with open(log_file, 'a') as f:
            f.write("\n\n=== FINAL SUMMARY ===\n")
            f.write(f"Total runs: {len(all_results)}\n")
            f.write(f"Successful: {len(successful_runs)}\n")
            f.write(f"Best AUC: {top_runs[0]['results'].get('best_val_auc', 0):.4f}\n")
            f.write(f"Completed: {datetime.now()}\n")
    
    print("\n✅ Simple sweep completed!")
    print(f"📊 Results: {results_file}")
    print(f"📜 Log: {log_file}")


if __name__ == "__main__":
    main()