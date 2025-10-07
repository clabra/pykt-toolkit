#!/usr/bin/env python3
"""
Comprehensive Offline Wandb Sweep - 20 Parameter Combinations
Shows detailed metrics per epoch across GPUs 0-4
"""

import subprocess
import sys
import os
import yaml
import random
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import re


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
    return param_config.get('value', 1.0)


def extract_detailed_metrics(output_lines):
    """Extract detailed metrics including per-epoch data."""
    metrics = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [], 
        'train_aucs': [],
        'val_aucs': [],
        'best_val_auc': 0.0,
        'consistency_scores': [],
        'correlations': {}
    }
    
    for line in output_lines:
        line = line.strip()
        
        # Extract FINAL_RESULTS
        if 'FINAL_RESULTS:' in line:
            try:
                parts = line.split(': ')
                if len(parts) >= 2:
                    metric_name = parts[0].replace('FINAL_RESULTS: ', '')
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
            except Exception:
                pass
                
        # Extract per-epoch metrics from training logs
        elif 'Train - Loss:' in line and 'AUC:' in line:
            try:
                # Parse: "Train - Loss: 0.6274 (Main: 0.5537, Constraint: 0.0737), AUC: 0.6608, Acc: 0.7355"
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                auc_match = re.search(r'AUC: ([\d.]+)', line)
                if loss_match and auc_match:
                    metrics['train_losses'].append(float(loss_match.group(1)))
                    metrics['train_aucs'].append(float(auc_match.group(1)))
            except Exception:
                pass
                
        elif 'Valid - Loss:' in line and 'AUC:' in line:
            try:
                # Parse: "Valid - Loss: 0.5232, AUC: 0.7060, Acc: 0.7488"
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                auc_match = re.search(r'AUC: ([\d.]+)', line)
                if loss_match and auc_match:
                    metrics['val_losses'].append(float(loss_match.group(1)))
                    val_auc = float(auc_match.group(1))
                    metrics['val_aucs'].append(val_auc)
                    # Track best validation AUC
                    if val_auc > metrics['best_val_auc']:
                        metrics['best_val_auc'] = val_auc
            except Exception:
                pass
                
        elif 'Best validation AUC:' in line:
            try:
                auc = float(line.split(':')[1].strip())
                metrics['best_val_auc'] = auc
            except Exception:
                pass
                
        elif 'Correlations - Mastery:' in line:
            try:
                # Parse: "Correlations - Mastery: 0.088, Gains: 0.062"
                parts = line.split('Mastery: ')[1]
                mastery_corr = float(parts.split(',')[0])
                gains_corr = float(parts.split('Gains: ')[1])
                metrics['correlations']['mastery'] = mastery_corr
                metrics['correlations']['gains'] = gains_corr
            except Exception:
                pass
    
    # Calculate summary statistics
    if metrics['val_aucs']:
        metrics['epochs'] = list(range(1, len(metrics['val_aucs']) + 1))
        metrics['final_epoch'] = len(metrics['val_aucs'])
        metrics['auc_improvement'] = metrics['val_aucs'][-1] - metrics['val_aucs'][0] if len(metrics['val_aucs']) > 1 else 0.0
    
    return metrics


def run_single_training(args_tuple):
    """Run a single training job with detailed logging."""
    run_idx, run_params, cmd_args, gpu_id, program = args_tuple
    
    print(f"🔥 Starting RUN {run_idx} on GPU {gpu_id}")
    print(f"   Parameters: epochs={run_params.get('epochs', '?')}, batch_size={run_params.get('batch_size', '?')}, lr={run_params.get('lr', 0):.4f}")
    
    # Build command  
    python_path = "/home/vscode/.pykt-env/bin/python"
    cmd = [python_path, program] + cmd_args
    
    # Set environment for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['WANDB_MODE'] = 'offline'  # Force offline
    env['WANDB_DISABLED'] = 'false'
    
    start_time = time.time()
    
    try:
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
            print(f"✅ RUN {run_idx} completed successfully on GPU {gpu_id} ({duration/60:.1f}min)")
            
            # Extract detailed metrics from both stdout and stderr
            all_output = result.stdout + "\n" + result.stderr
            output_lines = all_output.split('\n')
            metrics = extract_detailed_metrics(output_lines)
            
            # Display key results
            best_auc = metrics.get('best_val_auc', 0)
            epochs_completed = metrics.get('final_epoch', 0)
            print(f"   📊 Best Val AUC: {best_auc:.4f} after {epochs_completed} epochs")
            
            return {
                'run_id': run_idx,
                'gpu_id': gpu_id,
                'parameters': run_params,
                'metrics': metrics,
                'status': 'success',
                'duration_minutes': duration / 60,
                'epochs_completed': epochs_completed
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


def run_comprehensive_sweep(config_file, num_runs=20, max_parallel=5):
    """Run comprehensive sweep with detailed metrics logging."""
    
    print("🚀 COMPREHENSIVE OFFLINE WANDB SWEEP")
    print("=" * 80)
    
    # Load configuration
    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    parameters = sweep_config.get('parameters', {})
    program = sweep_config.get('program', 'train_cumulative_mastery_full.py')
    
    # GPU assignment (0-4)
    available_gpus = [0, 1, 2, 3, 4]
    
    print(f"📊 Parameter combinations: {num_runs}")
    print(f"🎯 Program: {program}")
    print(f"🖥️  GPUs: {available_gpus}")
    print(f"🔄 Max parallel jobs: {min(max_parallel, len(available_gpus))}")
    print("📈 Tracking: Per-epoch AUC, Loss, Correlations, Consistency")
    print("=" * 80)
    
    # Generate all parameter combinations
    job_args = []
    for run_idx in range(num_runs):
        # Sample parameters
        run_params = {}
        cmd_args = []
        
        # Assign GPU (round-robin)
        gpu_id = available_gpus[run_idx % len(available_gpus)]
        run_params['gpu_id'] = gpu_id
        
        # Map to supported parameters only
        supported_params = {
            'epochs': '--epochs',
            'batch_size': '--batch_size',
            'lr': '--lr', 
            'weight_decay': '--weight_decay',
            'patience': '--patience',
            'enhanced_constraints': '--enhanced_constraints'
        }
        
        for param_name, param_config in parameters.items():
            if param_name in supported_params:
                value = sample_parameters(param_config)
                run_params[param_name] = value
                cmd_args.extend([supported_params[param_name], str(value)])
        
        # Add required arguments
        cmd_args.extend(["--use_wandb", "True"])  # Enable wandb for detailed logging
        cmd_args.extend(["--experiment_suffix", f"comprehensive_sweep_run_{run_idx+1}"])
        
        job_args.append((run_idx + 1, run_params, cmd_args, gpu_id, program))
    
    # Save job configuration and start execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_sweep_results_{timestamp}.json"
    log_file = f"comprehensive_sweep_{timestamp}.log"
    
    print(f"\n🏁 Starting comprehensive execution...")
    print(f"💾 Results: {results_file}")
    print(f"📜 Detailed log: {log_file}")
    
    results = []
    completed_count = 0
    
    # Log detailed progress
    with open(log_file, 'w') as log_f:
        log_f.write(f"Comprehensive Sweep Started: {datetime.now()}\\n")
        log_f.write(f"Total runs: {num_runs}\\n")
        log_f.write(f"GPUs: {available_gpus}\\n\\n")
    
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
            
            # Log detailed progress
            with open(log_file, 'a') as log_f:
                log_f.write(f"\\n=== RUN {result['run_id']} COMPLETED ===\\n")
                log_f.write(f"Status: {result['status']}\\n")
                log_f.write(f"GPU: {result['gpu_id']}\\n")
                log_f.write(f"Duration: {result.get('duration_minutes', 0):.1f} min\\n")
                if result['status'] == 'success':
                    metrics = result.get('metrics', {})
                    log_f.write(f"Best Val AUC: {metrics.get('best_val_auc', 0):.4f}\\n")
                    log_f.write(f"Epochs: {metrics.get('final_epoch', 0)}\\n")
                    log_f.write(f"Parameters: {result['parameters']}\\n")
                else:
                    log_f.write(f"Error: {result.get('error', 'Unknown')}\\n")
            
            print(f"\\n📊 Progress: {completed_count}/{num_runs} completed")
            
            # Show detailed result summary
            if result['status'] == 'success' and 'metrics' in result:
                metrics = result['metrics']
                best_auc = metrics.get('best_val_auc', 0)
                epochs = metrics.get('final_epoch', 0)
                duration = result.get('duration_minutes', 0)
                gpu = result['gpu_id']
                
                print(f"   ✅ RUN {result['run_id']}: AUC {best_auc:.4f} in {epochs} epochs ({duration:.1f}min, GPU {gpu})")
                
                # Show epoch-by-epoch progression
                val_aucs = metrics.get('val_aucs', [])
                if len(val_aucs) >= 3:
                    print(f"      📈 AUC progression: {val_aucs[0]:.3f} → {val_aucs[len(val_aucs)//2]:.3f} → {val_aucs[-1]:.3f}")
            else:
                duration = result.get('duration_minutes', 0)
                gpu = result['gpu_id']
                print(f"   ❌ RUN {result['run_id']}: {result['status']} ({duration:.1f}min, GPU {gpu})")
    
    print(f"\\n🎉 COMPREHENSIVE SWEEP COMPLETED!")
    analyze_comprehensive_results(results, log_file)
    
    return results_file, log_file


def analyze_comprehensive_results(results, log_file):
    """Analyze and display comprehensive results."""
    
    successful_runs = [r for r in results if r['status'] == 'success' and 'metrics' in r]
    
    print(f"\\n📈 COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    print(f"📋 Total runs: {len(results)}")
    print(f"✅ Successful: {len(successful_runs)}")
    print(f"❌ Failed: {len(results) - len(successful_runs)}")
    
    if not successful_runs:
        print("❌ No successful runs to analyze")
        return
    
    # Find top 3 performers
    top_runs = sorted(successful_runs, key=lambda x: x['metrics'].get('best_val_auc', 0), reverse=True)[:3]
    
    print(f"\\n🏆 TOP 3 PERFORMERS:")
    for i, run in enumerate(top_runs, 1):
        metrics = run['metrics']
        params = run['parameters']
        auc = metrics.get('best_val_auc', 0)
        epochs = metrics.get('final_epoch', 0)
        
        print(f"\\n   #{i}. RUN {run['run_id']} - AUC: {auc:.4f}")
        print(f"      🎯 Epochs: {epochs}, GPU: {run['gpu_id']}, Duration: {run.get('duration_minutes', 0):.1f}min")
        print(f"      📋 Key params: epochs={params.get('epochs')}, batch_size={params.get('batch_size')}, lr={params.get('lr', 0):.4f}")
        
        # Show progression
        val_aucs = metrics.get('val_aucs', [])
        if len(val_aucs) >= 2:
            improvement = val_aucs[-1] - val_aucs[0]
            print(f"      📈 AUC improvement: +{improvement:.4f}")
    
    # Parameter analysis
    print(f"\\n📊 PARAMETER IMPACT ANALYSIS:")
    
    # Group by key parameters
    param_analysis = {}
    for param in ['epochs', 'batch_size', 'enhanced_constraints']:
        param_analysis[param] = {}
        
        for run in successful_runs:
            param_val = run['parameters'].get(param)
            if param_val not in param_analysis[param]:
                param_analysis[param][param_val] = []
            param_analysis[param][param_val].append(run['metrics'].get('best_val_auc', 0))
    
    for param, values in param_analysis.items():
        print(f"\\n   {param.upper()}:")
        for val, aucs in values.items():
            avg_auc = sum(aucs) / len(aucs) if aucs else 0
            print(f"      {val}: {avg_auc:.4f} avg AUC ({len(aucs)} runs)")
    
    # Log summary to file
    with open(log_file, 'a') as f:
        f.write(f"\\n\\n=== FINAL ANALYSIS ===\\n")
        f.write(f"Total runs: {len(results)}\\n")
        f.write(f"Successful: {len(successful_runs)}\\n")
        f.write(f"Best AUC: {top_runs[0]['metrics'].get('best_val_auc', 0):.4f}\\n")
        f.write(f"Completed: {datetime.now()}\\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive offline wandb sweep')
    parser.add_argument('--count', type=int, default=20,
                       help='Number of parameter combinations (default: 20)')
    parser.add_argument('--max_parallel', type=int, default=5,
                       help='Maximum parallel jobs (default: 5)')
    parser.add_argument('--config', type=str, default='sweep_config_comprehensive.yaml',
                       help='Sweep configuration file')
    
    args = parser.parse_args()
    
    # Check GPU availability
    print("🔍 Checking GPU availability...")
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_count = len([line for line in result.stdout.strip().split('\\n') if line])
            print(f"✅ Found {gpu_count} GPUs")
        else:
            print("⚠️  Could not detect GPUs, proceeding anyway...")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found, proceeding anyway...")
    
    results_file, log_file = run_comprehensive_sweep(args.config, args.count, args.max_parallel)
    print(f"\\n✅ Comprehensive sweep completed!")
    print(f"📊 Results: {results_file}")
    print(f"📜 Detailed log: {log_file}")