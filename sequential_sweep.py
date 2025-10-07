#!/usr/bin/env python3
"""
Sequential Simple Sweep - 20 Parameter Combinations
Runs sequentially to avoid process multiplication issues
"""

import subprocess
import os
import yaml
import random
import json
import time
from datetime import datetime
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
                # Format: "FINAL_RESULTS: metric_name: value"
                parts = line.split('FINAL_RESULTS:', 1)[1].strip()
                if ':' in parts:
                    metric_name, metric_value = parts.split(':', 1)
                    metric_name = metric_name.strip()
                    metric_value = float(metric_value.strip())
                    results[metric_name] = metric_value
            except Exception as e:
                print(f"Warning: Could not parse FINAL_RESULTS line: {line} - {e}")
    return results


def run_single_training(run_id, params, gpu_id, log_file, results_file):
    """Run a single training job and capture results."""
    
    print(f"\n🏃 Starting RUN {run_id} on GPU {gpu_id}")
    print(f"   📋 batch_size={params['batch_size']}, lr={params['lr']:.4f}, weight_decay={params['weight_decay']:.5f}")
    print(f"   🎯 enhanced_constraints={params['enhanced_constraints']}, patience={params['patience']}")
    
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
        '--experiment_suffix', f'sequential_sweep_run_{run_id}'
    ]
    
    # Set GPU environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    start_time = time.time()
    
    try:
        # Run with timeout and capture output
        print(f"   ⚡ Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env=env,
            cwd='/workspaces/pykt-toolkit'
        )
        
        duration = time.time() - start_time
        
        # Extract results from stdout and stderr
        all_output = result.stdout + result.stderr
        final_results = extract_final_results(all_output)
        
        # Log detailed output
        with open(log_file, 'a') as f:
            f.write(f"\n=== RUN {run_id} COMPLETED ===\n")
            f.write(f"Status: {'success' if result.returncode == 0 else 'failed'}\n")
            f.write(f"GPU: {gpu_id}\n")
            f.write(f"Duration: {duration/60:.1f} min\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Parameters: {params}\n")
            f.write(f"Extracted results: {final_results}\n")
            f.write(f"\nTraining Output (last 50 lines):\n")
            output_lines = all_output.split('\n')
            f.write('\n'.join(output_lines[-50:]))
            f.write("\n" + "-" * 50 + "\n")
        
        if result.returncode == 0:
            best_auc = final_results.get('best_val_auc', 0.0)
            print(f"✅ RUN {run_id} SUCCESS: AUC {best_auc:.4f} ({duration/60:.1f}min)")
            
            run_result = {
                'run_id': run_id,
                'gpu_id': gpu_id,
                'parameters': params,
                'results': final_results,
                'status': 'success',
                'duration_minutes': duration / 60
            }
        else:
            print(f"❌ RUN {run_id} FAILED (code: {result.returncode}) ({duration/60:.1f}min)")
            
            run_result = {
                'run_id': run_id,
                'gpu_id': gpu_id,
                'parameters': params,
                'results': final_results,  # Include partial results if any
                'status': 'failed',
                'error': result.stderr[:500] if result.stderr else "Unknown error",
                'duration_minutes': duration / 60
            }
    
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏱️  RUN {run_id} TIMEOUT after {duration/60:.1f}min")
        
        with open(log_file, 'a') as f:
            f.write(f"\n=== RUN {run_id} TIMEOUT ===\n")
            f.write(f"Duration: {duration/60:.1f} min\n")
            f.write(f"Parameters: {params}\n")
            f.write("-" * 50 + "\n")
        
        run_result = {
            'run_id': run_id,
            'gpu_id': gpu_id,
            'parameters': params,
            'results': {},
            'status': 'timeout',
            'duration_minutes': duration / 60
        }
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 RUN {run_id} ERROR: {str(e)} ({duration/60:.1f}min)")
        
        run_result = {
            'run_id': run_id,
            'gpu_id': gpu_id,
            'parameters': params,
            'results': {},
            'status': 'error',
            'error': str(e),
            'duration_minutes': duration / 60
        }
    
    # Save intermediate results after each run
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        all_results.append(run_result)
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"💾 Results saved to {results_file}")
            
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    return run_result


def main():
    print("🚀 SEQUENTIAL SIMPLE SWEEP - 20 COMBINATIONS")
    print("=" * 70)
    
    # Load configuration
    config_file = 'sweep_config_comprehensive.yaml'
    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    parameters = sweep_config.get('parameters', {})
    num_runs = 20
    available_gpus = [0, 1, 2, 3, 4]
    
    print(f"📊 Total parameter combinations: {num_runs}")
    print(f"🖥️  Available GPUs: {available_gpus}")
    print(f"🔄 Execution: Sequential (one at a time)")
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
    results_file = f"sequential_sweep_results_{timestamp}.json"
    log_file = f"sequential_sweep_{timestamp}.log"
    
    print(f"\n💾 Results: {results_file}")
    print(f"📜 Log: {log_file}")
    
    # Initialize log
    with open(log_file, 'w') as f:
        f.write(f"Sequential Simple Sweep Started: {datetime.now()}\n")
        f.write(f"Total runs: {num_runs}\n")
        f.write(f"GPUs: {available_gpus}\n")
        f.write(f"Sequential execution to avoid process multiplication\n\n")
    
    print("=" * 70)
    
    # Run jobs sequentially
    all_results = []
    successful_count = 0
    
    for i, params in enumerate(param_combinations):
        run_id = i + 1
        gpu_id = available_gpus[i % len(available_gpus)]  # Cycle through GPUs
        
        print(f"\n📊 PROGRESS: {i+1}/{num_runs} | GPU {gpu_id}")
        
        result = run_single_training(run_id, params, gpu_id, log_file, results_file)
        all_results.append(result)
        
        if result['status'] == 'success':
            successful_count += 1
        
        print(f"🔄 Completed: {i+1}/{num_runs} | Success: {successful_count}/{i+1}")
    
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
            print(f"      📋 batch_size={params.get('batch_size')}, lr={params.get('lr', 0):.4f}")
            print(f"      🔧 enhanced_constraints={params.get('enhanced_constraints')}, patience={params.get('patience')}")
        
        # Log final summary
        with open(log_file, 'a') as f:
            f.write("\n\n=== FINAL SUMMARY ===\n")
            f.write(f"Total runs: {len(all_results)}\n")
            f.write(f"Successful: {len(successful_runs)}\n")
            if successful_runs:
                f.write(f"Best AUC: {top_runs[0]['results'].get('best_val_auc', 0):.4f}\n")
            f.write(f"Completed: {datetime.now()}\n")
    
    total_time = sum(r.get('duration_minutes', 0) for r in all_results)
    
    print(f"\n⏱️  Total execution time: {total_time:.1f} minutes")
    print(f"✅ Sequential sweep completed!")
    print(f"📊 Results: {results_file}")
    print(f"📜 Log: {log_file}")


if __name__ == "__main__":
    main()