#!/usr/bin/env python3
"""
Debug Single Run - Isolate process multiplication issue
"""

import subprocess
import os
import time

def run_single_training_debug():
    """Run a single training job for debugging."""
    
    print("🚀 Starting single debug run...")
    
    params = {
        'epochs': 1,
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'patience': 10,
        'enhanced_constraints': True
    }
    
    cmd = [
        'python', 'train_cumulative_mastery_full.py',
        '--epochs', str(params['epochs']),
        '--batch_size', str(params['batch_size']),
        '--lr', str(params['lr']),
        '--weight_decay', str(params['weight_decay']),
        '--patience', str(params['patience']),
        '--enhanced_constraints', str(params['enhanced_constraints']),
        '--use_wandb', 'False',  # Disable wandb for this test
        '--experiment_suffix', 'debug_single_run'
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, env=env)
    
    print(f"Process started with PID: {process.pid}")
    
    # Monitor for a short period
    time.sleep(20)
    
    print("\n🔍 Checking for process multiplication...")
    
    # Check how many training processes are running
    ps_cmd = 'ps aux | grep "train_cumulative_mastery_full.py" | grep -v grep'
    result = subprocess.run(ps_cmd, shell=True, capture_output=True, text=True)
    
    running_processes = result.stdout.strip().split('\n')
    num_processes = len(running_processes) if running_processes[0] else 0
    
    print(f"Found {num_processes} training processes.")
    print("-" * 20)
    for line in running_processes:
        print(line)
    print("-" * 20)

    if num_processes > 1:
        print("\n❌ Process multiplication detected!")
    else:
        print("\n✅ Single process running as expected.")
        
    print("\nTerminating the process...")
    process.terminate()
    process.wait()
    print("Process terminated.")

if __name__ == "__main__":
    run_single_training_debug()
