#!/usr/bin/env python3
"""
Monitor offline wandb sweep progress
Check running processes and display current results
"""

import os
import json
import glob
import time
import subprocess
from datetime import datetime


def find_latest_results_file():
    """Find the most recent offline sweep results file."""
    pattern = "offline_sweep_results_*.json"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def display_sweep_progress():
    """Display current sweep progress and results."""
    
    print("🔍 OFFLINE SWEEP MONITOR")
    print("=" * 50)
    print(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check if sweep process is running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "launch_wandb_sweep.py"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print("✅ Sweep process is RUNNING")
            pids = result.stdout.strip().split('\n')
            print(f"   PIDs: {', '.join(pids)}")
        else:
            print("⏸️  No sweep process detected")
    except:
        print("❓ Could not check process status")
    
    # Check for training processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_cumulative_mastery_full.py"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print("🏃 Training process is ACTIVE")
            pids = result.stdout.strip().split('\n')
            print(f"   PIDs: {', '.join(pids)}")
        else:
            print("💤 No training process running")
    except:
        print("❓ Could not check training status")
    
    # Find and display results
    results_file = find_latest_results_file()
    if not results_file:
        print("\\n📄 No results files found yet")
        return
    
    print(f"\\n📊 RESULTS FROM: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        total_runs = len(results)
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') in ['failed', 'error', 'timeout'])
        
        print(f"📋 Progress: {total_runs} runs completed")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        # Show best results so far
        successful_runs = [r for r in results if r.get('status') == 'success' and 'metrics' in r]
        
        if successful_runs:
            print(f"\\n🏆 BEST RESULTS SO FAR:")
            
            # Best AUC
            best_auc_run = max(successful_runs, key=lambda x: x['metrics'].get('best_val_auc', 0))
            best_auc = best_auc_run['metrics'].get('best_val_auc', 0)
            
            print(f"   🎯 Best AUC: {best_auc:.4f} (Run {best_auc_run['run_id']})")
            
            # Show top parameters
            params = best_auc_run['parameters']
            key_params = ['num_encoder_blocks', 'd_ff', 'mastery_performance_loss_weight', 'gain_performance_loss_weight']
            for param in key_params:
                if param in params:
                    value = params[param]
                    if isinstance(value, float):
                        print(f"      {param}: {value:.4f}")
                    else:
                        print(f"      {param}: {value}")
            
            # Show recent runs
            print(f"\\n📈 RECENT RUNS:")
            for run in results[-3:]:
                status = run['status']
                run_id = run['run_id']
                if status == 'success' and 'metrics' in run:
                    auc = run['metrics'].get('best_val_auc', 0)
                    print(f"   Run {run_id}: ✅ AUC {auc:.4f}")
                else:
                    print(f"   Run {run_id}: ❌ {status}")
        
        else:
            print("\\n⏳ No successful runs yet...")
    
    except Exception as e:
        print(f"❌ Error reading results: {e}")


def monitor_continuously(interval=60):
    """Monitor sweep continuously with updates every interval seconds."""
    
    print(f"🔄 Starting continuous monitoring (updates every {interval}s)")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            os.system('clear')  # Clear screen
            display_sweep_progress()
            print(f"\\n⏳ Next update in {interval} seconds...")
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Monitoring stopped by user")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor offline wandb sweep')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Monitor continuously')
    parser.add_argument('--interval', '-i', type=int, default=60,
                       help='Update interval in seconds (default: 60)')
    
    args = parser.parse_args()
    
    if args.continuous:
        monitor_continuously(args.interval)
    else:
        display_sweep_progress()