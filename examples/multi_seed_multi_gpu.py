#!/usr/bin/env python3
"""
Multi-Seed Multi-GPU Runner for GainAKT2Exp
Runs the same configuration across multiple seeds and GPUs in parallel
"""

import os
import subprocess
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import yaml
from datetime import datetime

def load_config_args(config_path):
    """Load config file and convert to command line arguments."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    args = []
    
    # Essential parameters from config
    if 'batch_size' in config:
        args.extend(['--batch_size', str(config['batch_size'])])
    if 'learning_rate' in config:
        args.extend(['--learning_rate', str(config['learning_rate'])])
    if 'weight_decay' in config:
        args.extend(['--weight_decay', str(config['weight_decay'])])
    if 'epochs' in config:
        # Allow overriding default epochs passed via top-level --epochs
        args.extend(['--epochs', str(config['epochs'])])
    
    # Enable enhanced constraints by default
    args.append('--enhanced_constraints')

    # Individual constraint weights (if present)
    constraint_keys = [
        'non_negative_loss_weight',
        'monotonicity_loss_weight',
        'mastery_performance_loss_weight',
        'gain_performance_loss_weight',
        'sparsity_loss_weight',
        'consistency_loss_weight'
    ]
    for k in constraint_keys:
        if k in config:
            args.extend([f'--{k}', str(config[k])])

    # Alignment + lag objectives (if present)
    if 'alignment_weight' in config:
        args.extend(['--alignment_weight', str(config['alignment_weight'])])
        # enable alignment explicitly
        args.append('--enable_alignment_loss')
    if 'alignment_warmup_epochs' in config:
        args.extend(['--alignment_warmup_epochs', str(config['alignment_warmup_epochs'])])
    if 'alignment_global_validation_sample' in config:
        # Map to training script arg name if different
        args.extend(['--alignment_global_students', str(config['alignment_global_validation_sample'])])
    # Alignment share cap & decay factor
    if 'alignment_share_cap' in config:
        args.extend(['--alignment_share_cap', str(config['alignment_share_cap'])])
    if 'alignment_share_decay_factor' in config:
        args.extend(['--alignment_share_decay_factor', str(config['alignment_share_decay_factor'])])
    if 'lag_max_lag' in config:
        args.extend(['--lag_max_lag', str(config['lag_max_lag'])])
    if 'lag_gain_weight' in config:
        args.extend(['--lag_gain_weight', str(config['lag_gain_weight'])])
        args.append('--enable_lag_gain_loss')
    # Lag schedule parameters
    if config.get('lag_gain_schedule_enable', False):
        args.append('--lag_gain_schedule_enable')
    if 'lag_gain_schedule_min_gain_corr' in config:
        args.extend(['--lag_gain_schedule_min_gain_corr', str(config['lag_gain_schedule_min_gain_corr'])])
    if 'lag_gain_schedule_increase' in config:
        args.extend(['--lag_gain_schedule_increase', str(config['lag_gain_schedule_increase'])])
    if 'lag_gain_schedule_cap' in config:
        args.extend(['--lag_gain_schedule_cap', str(config['lag_gain_schedule_cap'])])

    # Adaptive alignment decay parameters
    if 'adaptive_patience' in config:
        args.extend(['--adaptive_patience', str(config['adaptive_patience'])])
    if 'adaptive_decay_factor' in config:
        args.extend(['--adaptive_decay_factor', str(config['adaptive_decay_factor'])])
    if 'adaptive_gain_corr_min' in config:
        args.extend(['--adaptive_gain_corr_min', str(config['adaptive_gain_corr_min'])])
    if 'adaptive_mastery_corr_min' in config:
        args.extend(['--adaptive_mastery_corr_min', str(config['adaptive_mastery_corr_min'])])
    if 'plateau_delta_auc' in config:
        args.extend(['--plateau_delta_auc', str(config['plateau_delta_auc'])])

    # Weight decay probe
    if config.get('weight_decay_probe_enable', False):
        args.append('--weight_decay_probe_enable')
    if 'weight_decay_probe_value' in config:
        args.extend(['--weight_decay_probe_value', str(config['weight_decay_probe_value'])])

    # Retention (only if active)
    if config.get('retention_logging_only', False) and config.get('retention_weight', 0.0) > 0.0:
        args.append('--enable_retention_loss')

    # Structural constraints are architectural; no CLI flags needed (training script lacks these arguments)
    
    # Use AMP if specified
    if config.get('amp', False):
        args.append('--use_amp')
    
    return args

def run_single_seed(seed, gpu_id, base_args, per_process_threads=None, launch_delay=0.0, base_suffix=""):
    """Run training for a single seed on specified GPU."""
    print(f"🔥 Starting seed {seed} on GPU {gpu_id}")
    
    # Build command
    python_path = "/home/vscode/.pykt-env/bin/python"
    script_path = "examples/train_gainakt2exp.py"
    
    composed_suffix = f"{base_suffix}_seed{seed}_gpu{gpu_id}" if base_suffix else f"seed{seed}_gpu{gpu_id}"
    # Remove any pre-existing --experiment_suffix from base_args to avoid duplication
    filtered_args = []
    skip_next = False
    for a in base_args:
        if skip_next:
            skip_next = False
            continue
        if a == '--experiment_suffix':
            skip_next = True
            continue
        filtered_args.append(a)
    cmd = [python_path, script_path] + filtered_args + [
        '--seed', str(seed),
        '--experiment_suffix', composed_suffix
    ]
    
    # Optional stagger to reduce simultaneous IO / CPU spikes
    if launch_delay > 0:
        time.sleep(launch_delay)

    # Set GPU environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Refined CPU thread limiting: distribute ~50% of total cores across parallel processes
    if per_process_threads is not None:
        thread_env_vars = [
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTORCH_NUM_THREADS'
        ]
        for var in thread_env_vars:
            env[var] = str(per_process_threads)
    # Avoid tokenizer parallelism overhead if any HF tokenizers are used indirectly
    env['TOKENIZERS_PARALLELISM'] = 'false'
    
    try:
        print(f"📟 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print(f"✅ Seed {seed} on GPU {gpu_id} completed successfully")
        return {
            'seed': seed,
            'gpu_id': gpu_id,
            'status': 'success',
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        print(f"❌ Seed {seed} on GPU {gpu_id} failed: {e}")
        return {
            'seed': seed,
            'gpu_id': gpu_id,
            'status': 'failed',
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }

def main():
    parser = argparse.ArgumentParser(description='Multi-Seed Multi-GPU GainAKT2Exp Runner')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--seeds', type=str, required=True, help='Comma-separated list of seeds')
    parser.add_argument('--devices', type=str, required=True, help='Comma-separated list of GPU IDs')
    parser.add_argument('--parallel_seeds', type=str, choices=['true', 'false'], default='true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--alignment_warmup', type=int, default=8)
    parser.add_argument('--global_validation_sample', type=int, default=600)
    parser.add_argument('--lag_max_lag', type=int, default=3)
    parser.add_argument('--alignment_weight', type=float, default=0.25)
    parser.add_argument('--retention_logging_only', type=str, choices=['true', 'false'], default='true')
    parser.add_argument('--experiment_suffix', type=str, default='', help='Optional base experiment suffix; seed/gpu appended automatically.')
    
    args = parser.parse_args()
    
    # Parse seeds and devices
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    devices = [int(d.strip()) for d in args.devices.split(',')]
    
    print("🚀 Multi-Seed Multi-GPU Runner")
    print(f"📊 Seeds: {seeds}")
    print(f"🖥️  Devices: {devices}")
    print(f"⚡ Parallel: {args.parallel_seeds}")
    
    # Load configuration from YAML file
    config_args = load_config_args(args.config)
    
    # Build complete arguments list
    # If epochs not in config, fall back to CLI provided --epochs
    base_args = list(config_args)
    if 'epochs' not in [a.lstrip('-') for a in base_args]:
        base_args.extend(['--epochs', str(args.epochs)])
    # Only add alignment / lag overrides if not specified in config
    arg_names = set()
    for i in range(0, len(base_args), 2):
        if base_args[i].startswith('--'):
            arg_names.add(base_args[i].lstrip('-'))
    if 'alignment_weight' not in arg_names and args.alignment_weight != 0.25:
        base_args.extend(['--alignment_weight', str(args.alignment_weight)])
    if 'alignment_warmup_epochs' not in arg_names and args.alignment_warmup != 8:
        base_args.extend(['--alignment_warmup_epochs', str(args.alignment_warmup)])
    if 'alignment_global_students' not in arg_names and args.global_validation_sample != 600:
        base_args.extend(['--alignment_global_students', str(args.global_validation_sample)])
    if 'lag_max_lag' not in arg_names and args.lag_max_lag != 3:
        base_args.extend(['--lag_max_lag', str(args.lag_max_lag)])
    # Enable losses only if not already enabled by config flags (we set flags explicitly in config loader)
    if not any(a == '--enable_alignment_loss' for a in base_args) and 'alignment_weight' in arg_names:
        base_args.append('--enable_alignment_loss')
    if not any(a == '--enable_lag_gain_loss' for a in base_args) and 'lag_gain_weight' in arg_names:
        base_args.append('--enable_lag_gain_loss')
    if args.retention_logging_only == 'true' and not any(a == '--enable_retention_loss' for a in base_args) and 'retention_weight' in arg_names:
        base_args.append('--enable_retention_loss')
    
    results = []
    
    if args.parallel_seeds == 'true':
        # Determine per-process thread allocation
        total_cores = os.cpu_count() or 1
        max_workers = min(len(seeds), len(devices))
        # Target: use at most ~50% of total cores cumulatively
        target_total = max(1, total_cores // 2)
        per_process_threads = max(1, target_total // max_workers)
        print(f"🧠 CPU cores: {total_cores} | target_total<=50%: {target_total} | per_process_threads: {per_process_threads}")
        stagger_seconds = 2.0  # small stagger to smooth IO

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, seed in enumerate(seeds):
                gpu_id = devices[i % len(devices)]  # Round-robin GPU assignment
                launch_delay = i * stagger_seconds
                future = executor.submit(run_single_seed, seed, gpu_id, base_args, per_process_threads, launch_delay, args.experiment_suffix)
                futures.append(future)
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        # Run seeds sequentially
        for i, seed in enumerate(seeds):
            gpu_id = devices[i % len(devices)]
            result = run_single_seed(seed, gpu_id, base_args, base_suffix=args.experiment_suffix)
            results.append(result)
    
    # Save results summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multi_seed_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 Results saved to: {results_file}")
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print("\n📈 SUMMARY:")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ❌ Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\n❌ Failed runs:")
        for result in results:
            if result['status'] == 'failed':
                print(f"   Seed {result['seed']} on GPU {result['gpu_id']}: {result['error']}")

if __name__ == '__main__':
    main()