#!/bin/bash
"""
Wandb Sweep Launcher for Cumulative Mastery Optimization
Following PyKT project guidelines for hyperparameter tuning
"""

import wandb
import subprocess
import sys
import os
import argparse
import yaml
import random
import itertools
from datetime import datetime


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


def run_single_experiment(run_idx, gpu_id, parameters, program, timestamp):
    """Run a single training experiment on specified GPU."""
    import subprocess
    import os
    import time
    
    print(f"🏃 Starting RUN {run_idx + 1} on GPU {gpu_id}")
    
    # Sample parameters for this run
    run_params = {'gpu_id': gpu_id}
    cmd_args = []
    
    # Map sweep parameters to supported script arguments
    supported_params = {
        'epochs': '--epochs',
        'batch_size': '--batch_size', 
        'lr': '--lr',
        'weight_decay': '--weight_decay',
        'patience': '--patience',
        'enhanced_constraints': '--enhanced_constraints',
        'monitor_freq': '--monitor_freq',
        'dataset': '--dataset',
        'fold': '--fold'
    }
    
    for param_name, param_config in parameters.items():
        if param_name in supported_params:
            value = sample_parameters(param_config)
            run_params[param_name] = value
            cmd_args.extend([supported_params[param_name], str(value)])
    
    # Add required arguments
    cmd_args.extend(["--use_wandb", "True"])  # Enable wandb in offline mode
    cmd_args.extend(["--experiment_suffix", f"sweep_run_{run_idx}"])
    
    # Build command
    python_path = "/home/vscode/.pykt-env/bin/python"
    cmd = [python_path, program] + cmd_args
    
    # Set environment for GPU and offline wandb
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['WANDB_MODE'] = 'offline'  # Force offline mode
    
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
            # Extract metrics from output (combine stdout and stderr)
            all_output = result.stdout + "\n" + result.stderr
            output_lines = all_output.split('\n')
            metrics = extract_metrics_from_output(output_lines)
            
            return {
                'run_id': run_idx + 1,
                'parameters': run_params,
                'metrics': metrics,
                'status': 'success',
                'duration_minutes': duration / 60,
                'gpu_id': gpu_id
            }
        else:
            return {
                'run_id': run_idx + 1,
                'parameters': run_params,
                'status': 'failed',
                'error': result.stderr[:500],
                'duration_minutes': duration / 60,
                'gpu_id': gpu_id
            }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            'run_id': run_idx + 1,
            'parameters': run_params,
            'status': 'timeout',
            'duration_minutes': duration / 60,
            'gpu_id': gpu_id
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'run_id': run_idx + 1,
            'parameters': run_params,
            'status': 'error',
            'error': str(e),
            'duration_minutes': duration / 60,
            'gpu_id': gpu_id
        }


def run_offline_sweep(config_file, num_runs=15):
    """Run hyperparameter optimization offline without wandb."""
    
    print("� OFFLINE WANDB SWEEP - ENHANCED VERSION")
    print("=" * 60)
    print("�🔄 Loading sweep configuration...")
    
    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    parameters = sweep_config.get('parameters', {})
    program = sweep_config.get('program', 'train_cumulative_mastery_full.py')
    
    # Check GPU availability
    import subprocess
    try:
        gpu_result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if gpu_result.returncode == 0:
            gpu_lines = [line for line in gpu_result.stdout.strip().split('\n') if line.strip()]
            available_gpus = list(range(min(len(gpu_lines), 5)))  # Use up to 5 GPUs
        else:
            available_gpus = [0]  # Fallback to single GPU
    except:
        available_gpus = [0]  # Fallback to single GPU
    
    print(f"📊 Parameter combinations: {num_runs}")
    print(f"🎯 Training program: {program}")
    print(f"🖥️  Available GPUs: {available_gpus} ({len(available_gpus)} GPUs)")
    print(f"⚡ Parallel execution: {len(available_gpus)} jobs at a time")
    print("=" * 60)
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"offline_sweep_results_{timestamp}.json"
    
    # Create batches for parallel execution
    batch_size = len(available_gpus)
    batches = [list(range(i, min(i + batch_size, num_runs))) for i in range(0, num_runs, batch_size)]
    
    print(f"📦 Created {len(batches)} batches for parallel execution")
    print(f"💾 Results will be saved to: {results_file}")
    print("=" * 60)
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    
    total_start_time = time.time()
    
    for batch_idx, batch_runs in enumerate(batches):
        print(f"\n🚀 BATCH {batch_idx + 1}/{len(batches)} - Runs {batch_runs[0] + 1}-{batch_runs[-1] + 1}")
        print("-" * 40)
        
        batch_start_time = time.time()
        batch_results = []
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid process multiplication
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
            # Submit all runs in this batch
            future_to_run = {}
            for i, run_idx in enumerate(batch_runs):
                gpu_id = available_gpus[i % len(available_gpus)]
                future = executor.submit(run_single_experiment, run_idx, gpu_id, parameters, program, timestamp)
                future_to_run[future] = run_idx
                print(f"  ▶️  RUN {run_idx + 1} submitted to GPU {gpu_id}")
            
            # Collect results as they complete
            for future in as_completed(future_to_run):
                run_idx = future_to_run[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                    status_emoji = "✅" if result['status'] == 'success' else "❌"
                    
                    if result['status'] == 'success' and 'metrics' in result:
                        auc = result['metrics'].get('best_val_auc', 0)
                        print(f"  {status_emoji} RUN {run_idx + 1} completed - AUC: {auc:.4f}")
                    else:
                        print(f"  {status_emoji} RUN {run_idx + 1} {result['status']}")
                        
                except Exception as e:
                    print(f"  💥 RUN {run_idx + 1} failed with exception: {e}")
                    batch_results.append({
                        'run_id': run_idx + 1,
                        'status': 'error',
                        'error': str(e)
                    })
        
        results.extend(batch_results)
        
        # Save intermediate results after each batch
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        batch_duration = time.time() - batch_start_time
        successful_in_batch = sum(1 for r in batch_results if r['status'] == 'success')
        
        print(f"  📊 Batch {batch_idx + 1} completed in {batch_duration/60:.1f} minutes")
        print(f"  ✅ {successful_in_batch}/{len(batch_runs)} successful runs")
        
        # Progress summary
        total_completed = len(results)
        total_successful = sum(1 for r in results if r['status'] == 'success')
        progress_pct = (total_completed / num_runs) * 100
        elapsed_time = time.time() - total_start_time
        
        print(f"\n📈 OVERALL PROGRESS: {total_completed}/{num_runs} ({progress_pct:.1f}%)")
        print(f"✅ Successful runs: {total_successful}")
        print(f"⏱️  Elapsed time: {elapsed_time/60:.1f} minutes")
        
        if total_completed < num_runs:
            estimated_remaining = (elapsed_time / total_completed) * (num_runs - total_completed)
            print(f"🔮 Estimated remaining: {estimated_remaining/60:.1f} minutes")
    
    total_duration = time.time() - total_start_time
    
    print(f"\n🎉 OFFLINE SWEEP COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Total duration: {total_duration/60:.1f} minutes")
    print(f"📋 Total runs: {len(results)}")
    
    successful_runs = [r for r in results if r['status'] == 'success']
    failed_runs = [r for r in results if r['status'] in ['failed', 'error', 'timeout']]
    
    print(f"✅ Successful: {len(successful_runs)}")
    print(f"❌ Failed: {len(failed_runs)}")
    print(f"📊 Success rate: {len(successful_runs)/len(results)*100:.1f}%")
    
    if successful_runs:
        # Show GPU utilization summary
        gpu_usage = {}
        for result in results:
            gpu = result.get('gpu_id', 'unknown')
            if gpu not in gpu_usage:
                gpu_usage[gpu] = {'total': 0, 'successful': 0}
            gpu_usage[gpu]['total'] += 1
            if result['status'] == 'success':
                gpu_usage[gpu]['successful'] += 1
        
        print(f"\n🖥️  GPU UTILIZATION SUMMARY:")
        for gpu, stats in sorted(gpu_usage.items()):
            success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"   GPU {gpu}: {stats['successful']}/{stats['total']} successful ({success_rate:.1f}%)")
    
    # Analyze best results
    analyze_sweep_results(results)
    
    # Final results summary
    print(f"\n💾 Final results saved to: {results_file}")
    
    return results_file


def extract_metrics_from_output(output_lines):
    """Extract key metrics from training output."""
    metrics = {}
    
    print(f"🔍 Extracting metrics from {len(output_lines)} output lines...")
    
    for line in output_lines:
        line = line.strip()
        
        # Look for FINAL_RESULTS: pattern (correct format from training script)
        if 'FINAL_RESULTS:' in line:
            try:
                # Format: "FINAL_RESULTS: metric_name: value"
                if ':' in line:
                    parts = line.split(':', 2)  # Split into max 3 parts
                    if len(parts) >= 3:
                        metric_name = parts[1].strip()
                        metric_value = float(parts[2].strip())
                        metrics[metric_name] = metric_value
                        print(f"  ✓ Extracted {metric_name}: {metric_value}")
                    elif len(parts) == 2:
                        # Handle case: "FINAL_RESULTS: best_val_auc: 0.1234"
                        remaining = parts[1].strip()
                        if ':' in remaining:
                            metric_parts = remaining.split(':', 1)
                            metric_name = metric_parts[0].strip()
                            metric_value = float(metric_parts[1].strip())
                            metrics[metric_name] = metric_value
                            print(f"  ✓ Extracted {metric_name}: {metric_value}")
            except (ValueError, IndexError) as e:
                print(f"  ⚠️ Failed to parse FINAL_RESULTS line: {line} - {e}")
                continue
        
        # Also look for alternative patterns (backup)
        elif 'Best validation AUC:' in line or 'best_val_auc:' in line:
            try:
                auc = float(line.split(':')[-1].strip())
                metrics['best_val_auc'] = auc
                print(f"  ✓ Extracted best_val_auc (backup): {auc}")
            except (ValueError, IndexError):
                continue
                
        elif 'Training completed!' in line and 'Best Val AUC:' in line:
            try:
                # Pattern: "Training completed! Best Val AUC: 0.1234"
                auc_part = line.split('Best Val AUC:')[1].strip()
                auc = float(auc_part)
                metrics['best_val_auc'] = auc
                print(f"  ✓ Extracted best_val_auc (completion): {auc}")
            except (ValueError, IndexError):
                continue
    
    print(f"📊 Total metrics extracted: {len(metrics)}")
    if metrics:
        for key, value in metrics.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️ No metrics found! Check output format.")
        # Debug: Show last 10 lines for analysis
        print("\n🔍 Last 10 lines of output for debugging:")
        for line in output_lines[-10:]:
            print(f"   {repr(line)}")
    
    return metrics


def analyze_sweep_results(results):
    """Analyze and display best results from sweep."""
    
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
    print("   Parameters:")
    for key, value in best_auc_run['parameters'].items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")
        else:
            print(f"     {key}: {value}")
    
    # Find best consistency (if available)
    runs_with_consistency = [r for r in successful_runs if 'consistency_score' in r['metrics']]
    if runs_with_consistency:
        best_consistency_run = max(runs_with_consistency, key=lambda x: x['metrics']['consistency_score'])
        best_consistency = best_consistency_run['metrics']['consistency_score']
        print(f"\n✅ BEST CONSISTENCY: {best_consistency:.4f}")
    
    # Show parameter ranges for successful runs
    print(f"\n📊 SUCCESSFUL PARAMETER RANGES:")
    param_names = set()
    for run in successful_runs:
        param_names.update(run['parameters'].keys())
    
    for param in sorted(param_names):
        values = [run['parameters'][param] for run in successful_runs if param in run['parameters']]
        if values:
            if isinstance(values[0], (int, float)):
                print(f"   {param}: {min(values):.4f} - {max(values):.4f}")
            else:
                unique_vals = list(set(values))
                print(f"   {param}: {unique_vals}")


def create_and_launch_sweep(offline=True):
    """Create and launch wandb sweep for cumulative mastery optimization."""
    
    print("🚀 CREATING WANDB SWEEP FOR CUMULATIVE MASTERY OPTIMIZATION")
    if offline:
        print("📴 OFFLINE MODE - Local hyperparameter optimization")
    print("=" * 70)
    print("Objectives:")
    print("1. 🎯 Maximize Validation AUC (Primary)")
    print("2. ✅ Maintain Perfect Educational Consistency")  
    print("3. 📊 Strengthen Mastery & Gain Correlations")
    print("=" * 70)
    
    # Check if sweep config exists
    config_file = "sweep_config_cumulative_mastery.yaml"
    if not os.path.exists(config_file):
        print(f"❌ Sweep configuration file not found: {config_file}")
        return None
    
    if offline:
        print("📴 Running offline hyperparameter optimization...")
        return run_offline_sweep(config_file)
    else:
        # Initialize wandb (requires API key setup)
        print("📡 Initializing wandb...")
        
        try:
            # Read and parse the config file
            import yaml
            with open(config_file, 'r') as f:
                sweep_config = yaml.safe_load(f)
            
            # Create the sweep
            print(f"📋 Creating sweep from config: {config_file}")
            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project="pykt-cumulative-mastery"
            )
            
            print("✅ Sweep created successfully!")
            print(f"🆔 Sweep ID: {sweep_id}")
            print(f"🌐 View sweep: https://wandb.ai/project/pykt-cumulative-mastery/sweeps/{sweep_id}")
            
            return sweep_id
            
        except Exception as e:
            print(f"❌ Failed to create sweep: {e}")
            print("💡 Make sure you have:")
            print("   1. Wandb installed: pip install wandb")
            print("   2. Wandb logged in: wandb login")
            print("   3. Valid wandb API key")
            return None


def start_sweep_agent(sweep_id, count=10):
    """Start wandb agent to run sweep experiments."""
    
    if not sweep_id:
        print("❌ No valid sweep ID provided")
        return
    
    print(f"\\n🤖 STARTING WANDB AGENT")
    print("=" * 50) 
    print(f"Sweep ID: {sweep_id}")
    print(f"Max runs: {count}")
    print("=" * 50)
    
    try:
        # Start the agent
        wandb.agent(
            sweep_id,
            count=count,
            project="pykt-cumulative-mastery"
        )
        
    except KeyboardInterrupt:
        print("\\n⏹️  Sweep interrupted by user")
    except Exception as e:
        print(f"❌ Sweep agent failed: {e}")


def monitor_sweep_progress(sweep_id):
    """Monitor and display sweep progress."""
    
    print(f"\\n📊 SWEEP MONITORING")
    print("=" * 40)
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: pykt-cumulative-mastery")
    print("\\n🔗 Links:")
    print(f"   Dashboard: https://wandb.ai/project/pykt-cumulative-mastery/sweeps/{sweep_id}")
    print("\\n📋 Key Metrics to Monitor:")
    print("   • best_val_auc (Primary objective)")
    print("   • consistency_score (Perfect = 1.0)")
    print("   • correlation_score (Higher = Better)")
    print("   • combined_objective (Balanced metric)")
    print("\\n⏱️  Expected Runtime: 2-4 hours for 10 runs")
    print("\\n💡 Stopping Criteria:")
    print("   • No improvement in last 100 rounds")
    print("   • >200 combinations tried per fold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch wandb sweep for cumulative mastery')
    parser.add_argument('--count', type=int, default=15, 
                       help='Number of sweep runs (default: 15)')
    parser.add_argument('--create_only', action='store_true',
                       help='Only create sweep, dont start agent')
    parser.add_argument('--sweep_id', type=str, 
                       help='Existing sweep ID to continue')
    parser.add_argument('--offline', action='store_true', default=True,
                       help='Run offline hyperparameter optimization (default: True)')
    parser.add_argument('--online', action='store_true', 
                       help='Force online wandb mode')
    
    args = parser.parse_args()
    
    # Determine offline mode
    offline_mode = args.offline and not args.online
    
    if offline_mode:
        print("🔌 Running in OFFLINE mode")
        results_file = run_offline_sweep("sweep_config_cumulative_mastery.yaml", args.count)
        print(f"\\n✅ Offline sweep completed! Results saved to: {results_file}")
        sys.exit(0)
    
    # Online wandb mode
    if args.sweep_id:
        # Continue existing sweep
        sweep_id = args.sweep_id
        print(f"🔄 Continuing existing sweep: {sweep_id}")
    else:
        # Create new sweep
        sweep_id = create_and_launch_sweep(offline=False)
    
    if not sweep_id:
        sys.exit(1)
    
    # Show monitoring info
    monitor_sweep_progress(sweep_id)
    
    if not args.create_only:
        # Start the agent
        start_sweep_agent(sweep_id, args.count)
    else:
        print("\\n✋ Sweep created but agent not started (--create_only)")
        print("\\nTo start agent manually:")
        print(f"   python launch_wandb_sweep.py --sweep_id {sweep_id} --count {args.count}")
        print("\\nOr using wandb CLI:")
        print(f"   wandb agent {sweep_id}")