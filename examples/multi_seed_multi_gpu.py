#!/usr/bin/env python3
"""
Multi-Seed Multi-GPU Runner for GainAKT2Exp
Runs the same configuration across multiple seeds and GPUs in parallel
"""

import os
import subprocess
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
    
    # Enable enhanced constraints by default
    args.append('--enhanced_constraints')
    
    # Use AMP if specified
    if config.get('amp', False):
        args.append('--use_amp')
    
    return args

def run_single_seed(seed, gpu_id, base_args):
    """Run training for a single seed on specified GPU."""
    print(f"🔥 Starting seed {seed} on GPU {gpu_id}")
    
    # Build command
    python_path = "/home/vscode/.pykt-env/bin/python"
    script_path = "examples/train_gainakt2exp.py"
    
    cmd = [python_path, script_path] + base_args + [
        '--seed', str(seed),
        '--experiment_suffix', f'seed{seed}_gpu{gpu_id}'
    ]
    
    # Set GPU environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
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
    base_args = config_args + [
        '--epochs', str(args.epochs),
        '--alignment_warmup_epochs', str(args.alignment_warmup),
        '--alignment_global_students', str(args.global_validation_sample),
        '--lag_max_lag', str(args.lag_max_lag),
        '--alignment_weight', str(args.alignment_weight),
        '--enable_alignment_loss',
        '--enable_lag_gain_loss'
    ]
    
    if args.retention_logging_only == 'true':
        base_args.append('--enable_retention_loss')
    
    results = []
    
    if args.parallel_seeds == 'true':
        # Run seeds in parallel across available GPUs
        max_workers = min(len(seeds), len(devices))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            futures = []
            for i, seed in enumerate(seeds):
                gpu_id = devices[i % len(devices)]  # Round-robin GPU assignment
                future = executor.submit(run_single_seed, seed, gpu_id, base_args)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        # Run seeds sequentially
        for i, seed in enumerate(seeds):
            gpu_id = devices[i % len(devices)]
            result = run_single_seed(seed, gpu_id, base_args)
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