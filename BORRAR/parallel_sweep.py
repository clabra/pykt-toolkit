#!/usr/bin/env python3
"""
Parallel Offline Wandb Sweep - Multi-GPU Version
Runs multiple training processes simultaneously across GPUs 0-4
"""

import wandb
import subprocess
import sys
import os
import argparse
import yaml
import random
import itertools
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def sample_parameters(param_config):
    """Sample parameters based on wandb config format."""
    if 'values' in param_config:
        return random.choice(param_config['values'])
    elif 'distribution' in param_config:
        if param_config['distribution'] == 'uniform':
            return random.uniform(param_config['min'], param_config['max'])
        elif param_config['distribution'] == 'log_uniform_values':
            import math
            log_min = math.log(param_config['min'])
            log_max = math.log(param_config['max'])
            return math.exp(random.uniform(log_min, log_max))
    else:
        # Fallback to first value if available
        if 'values' in param_config:
            return param_config['values'][0]
        return param_config.get('value', 1.0)


def run_single_training(args_tuple):
    """Run a single training job on assigned GPU."""
    run_idx, run_params, cmd_args, gpu_id, program = args_tuple
    
    print(f"🔥 Starting RUN {run_idx} on GPU {gpu_id}")
    
    # Build command
    python_path = "/home/vscode/.pykt-env/bin/python"
    cmd = [python_path, program] + cmd_args
    
    # Set environment for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/workspaces/pykt-toolkit",
            capture_output=True,
            text=True,
            timeout=2400,  # 40 minutes for 25 epochs
            env=env
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ RUN {run_idx} completed successfully on GPU {gpu_id} ({duration/60:.1f}min)")
            # Extract metrics from both stdout and stderr output
            all_output = result.stdout + "\n" + result.stderr
            output_lines = all_output.split('\n')
            metrics = extract_metrics_from_output(output_lines)
            return {
                'run_id': run_idx,
                'gpu_id': gpu_id,
                'parameters': run_params,
                'metrics': metrics,
                'status': 'success',
                'duration_minutes': duration / 60
            }
        else:
            print(f"❌ RUN {run_idx} failed on GPU {gpu_id} (code: {result.returncode})")
            return {
                'run_id': run_idx,
                'gpu_id': gpu_id,
                'parameters': run_params,
                'status': 'failed',
                'error': result.stderr[:500],
                'duration_minutes': duration / 60
            }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ RUN {run_idx} timed out on GPU {gpu_id} after {duration/60:.1f}min")
        return {
            'run_id': run_idx,
            'gpu_id': gpu_id,
            'parameters': run_params,
            'status': 'timeout',
            'duration_minutes': duration / 60
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 RUN {run_idx} error on GPU {gpu_id}: {e}")
        return {
            'run_id': run_idx,
            'gpu_id': gpu_id,
            'parameters': run_params,
            'status': 'error',
            'error': str(e),
            'duration_minutes': duration / 60
        }


def extract_metrics_from_output(output_lines):
    """Extract key metrics from training output."""
    metrics = {}
    
    for line in output_lines:
        line = line.strip()
        # Look for FINAL_RESULTS output format
        if 'FINAL_RESULTS:' in line:
            try:
                parts = line.split(': ')
                if len(parts) >= 2:
                    metric_name = parts[0].replace('FINAL_RESULTS: ', '')
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
            except Exception:
                pass
        # Fallback to other formats
        elif 'best_val_auc:' in line or 'Best Validation AUC:' in line:
            try:
                auc = float(line.split(':')[1].strip())
                metrics['best_val_auc'] = auc
            except Exception:
                pass
        elif 'Best validation AUC:' in line:
            try:
                auc = float(line.split(':')[1].strip())
                metrics['best_val_auc'] = auc
            except Exception:
                pass
    
    return metrics


def run_parallel_sweep(config_file, num_runs=15, max_parallel=5):
    """Run hyperparameter optimization with parallel GPU execution."""
    
    print("🚀 PARALLEL OFFLINE WANDB SWEEP")
    print("=" * 60)
    
    # Load configuration
    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    parameters = sweep_config.get('parameters', {})
    program = sweep_config.get('program', 'train_cumulative_mastery_full.py')
    
    # GPU assignment (0-4)
    available_gpus = [0, 1, 2, 3, 4]
    
    print(f"📊 Will run {num_runs} parameter combinations")
    print(f"🎯 Program: {program}")
    print(f"🖥️  Using GPUs: {available_gpus}")
    print(f"🔄 Max parallel jobs: {min(max_parallel, len(available_gpus))}")
    
    # Generate all parameter combinations
    job_args = []
    for run_idx in range(num_runs):
        # Sample parameters
        run_params = {}
        cmd_args = []
        
        # Assign GPU (round-robin) - controlled via CUDA_VISIBLE_DEVICES
        gpu_id = available_gpus[run_idx % len(available_gpus)]
        run_params['gpu_id'] = gpu_id
        # GPU controlled by environment variable, not command line arg
        
        # Map sweep parameters to supported script arguments (ACTUALLY SUPPORTED ONLY)
        supported_params = {
            'epochs': '--epochs',
            'batch_size': '--batch_size', 
            'lr': '--lr',
            'weight_decay': '--weight_decay',
            'patience': '--patience',
            'enhanced_constraints': '--enhanced_constraints'
        }
        
        for param_name, param_config in parameters.items():
            if param_name in supported_params:  # Only use supported parameters
                value = sample_parameters(param_config)
                run_params[param_name] = value
                cmd_args.extend([supported_params[param_name], str(value)])
        
        # Add required arguments
        cmd_args.extend(["--use_wandb", "False"])
        cmd_args.extend(["--experiment_suffix", f"sweep_run_{run_idx}"])
        
        job_args.append((run_idx + 1, run_params, cmd_args, gpu_id, program))
    
    # Save job configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"parallel_sweep_results_{timestamp}.json"
    
    print(f"\n🏁 Starting parallel execution...")
    print(f"💾 Results will be saved to: {results_file}")
    
    results = []
    completed_count = 0
    
    # Use ProcessPoolExecutor for parallel execution
    max_workers = min(max_parallel, len(available_gpus), num_runs)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(run_single_training, args): args for args in job_args}
        
        # Collect results as they complete
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            completed_count += 1
            
            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📊 Progress: {completed_count}/{num_runs} completed")
            
            # Show quick summary of this result
            if result['status'] == 'success' and 'metrics' in result:
                auc = result['metrics'].get('best_val_auc', 0)
                duration = result.get('duration_minutes', 0)
                print(f"   RUN {result['run_id']}: ✅ AUC {auc:.4f} ({duration:.1f}min, GPU {result['gpu_id']})")
            else:
                duration = result.get('duration_minutes', 0)
                print(f"   RUN {result['run_id']}: ❌ {result['status']} ({duration:.1f}min, GPU {result['gpu_id']})")
    
    print(f"\n🎉 PARALLEL SWEEP COMPLETED!")
    print(f"📋 Total runs: {len(results)}")
    print(f"✅ Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"❌ Failed: {sum(1 for r in results if r['status'] in ['failed', 'error', 'timeout'])}")
    
    # Analyze results
    analyze_parallel_results(results)
    
    return results_file


def analyze_parallel_results(results):
    """Analyze and display best results from parallel sweep."""
    
    successful_runs = [r for r in results if r['status'] == 'success' and 'metrics' in r]
    
    if not successful_runs:
        print("❌ No successful runs to analyze")
        return
    
    print(f"\n📈 ANALYZING {len(successful_runs)} SUCCESSFUL RUNS")
    print("=" * 60)
    
    # Find best AUC
    best_auc_run = max(successful_runs, key=lambda x: x['metrics'].get('best_val_auc', 0))
    best_auc = best_auc_run['metrics'].get('best_val_auc', 0)
    
    print(f"🏆 BEST VALIDATION AUC: {best_auc:.4f}")
    print(f"   Run ID: {best_auc_run['run_id']} (GPU {best_auc_run['gpu_id']})")
    print(f"   Duration: {best_auc_run.get('duration_minutes', 0):.1f} minutes")
    print("   Key Parameters:")
    
    key_params = ['epochs', 'num_encoder_blocks', 'd_ff', 'mastery_performance_loss_weight', 'gain_performance_loss_weight']
    for param in key_params:
        if param in best_auc_run['parameters']:
            value = best_auc_run['parameters'][param]
            if isinstance(value, float):
                print(f"     {param}: {value:.4f}")
            else:
                print(f"     {param}: {value}")
    
    # GPU utilization stats
    gpu_stats = {}
    for run in results:
        gpu_id = run['gpu_id']
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = {'total': 0, 'successful': 0, 'avg_duration': 0}
        
        gpu_stats[gpu_id]['total'] += 1
        if run['status'] == 'success':
            gpu_stats[gpu_id]['successful'] += 1
        
        duration = run.get('duration_minutes', 0)
        gpu_stats[gpu_id]['avg_duration'] += duration
    
    print(f"\n🖥️  GPU UTILIZATION:")
    for gpu_id, stats in sorted(gpu_stats.items()):
        avg_duration = stats['avg_duration'] / stats['total'] if stats['total'] > 0 else 0
        success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"   GPU {gpu_id}: {stats['successful']}/{stats['total']} successful ({success_rate:.1f}%), avg {avg_duration:.1f}min")
    
    # Performance distribution
    aucs = [r['metrics'].get('best_val_auc', 0) for r in successful_runs]
    if aucs:
        print(f"\n📊 PERFORMANCE DISTRIBUTION:")
        print(f"   Best AUC: {max(aucs):.4f}")
        print(f"   Mean AUC: {sum(aucs)/len(aucs):.4f}")
        print(f"   Std AUC:  {(sum((x-sum(aucs)/len(aucs))**2 for x in aucs)/len(aucs))**0.5:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel offline wandb sweep')
    parser.add_argument('--count', type=int, default=15,
                       help='Number of sweep runs (default: 15)')
    parser.add_argument('--max_parallel', type=int, default=5,
                       help='Maximum parallel jobs (default: 5)')
    parser.add_argument('--config', type=str, default='sweep_config_cumulative_mastery.yaml',
                       help='Sweep configuration file')
    
    args = parser.parse_args()
    
    # Check GPU availability
    print("🔍 Checking GPU availability...")
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split('\n'))
            print(f"✅ Found {gpu_count} GPUs")
        else:
            print("⚠️  Could not detect GPUs, proceeding anyway...")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found, proceeding anyway...")
    
    results_file = run_parallel_sweep(args.config, args.count, args.max_parallel)
    print(f"\n✅ Parallel sweep completed! Results saved to: {results_file}")