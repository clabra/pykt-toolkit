#!/usr/bin/env python3
"""
Monitor persistent sweep progress
Check background processes and display current results
"""

import os
import json
import glob
import time
import subprocess
from datetime import datetime


def check_sweep_processes():
    """Check if sweep processes are still running."""
    
    print("🔍 CHECKING BACKGROUND SWEEP PROCESSES")
    print("=" * 50)
    
    # Check for main sweep process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "parallel_sweep.py"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print("✅ Main sweep process is RUNNING")
            pids = result.stdout.strip().split('\n')
            print(f"   PIDs: {', '.join(pids)}")
            
            # Check how long it's been running
            for pid in pids:
                try:
                    ps_result = subprocess.run(
                        ["ps", "-o", "pid,etime,cmd", "-p", pid],
                        capture_output=True,
                        text=True
                    )
                    if ps_result.returncode == 0:
                        lines = ps_result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            print(f"   Runtime: {lines[1].split()[1]}")
                except:
                    pass
        else:
            print("⏸️  No main sweep process detected")
    except Exception as e:
        print(f"❓ Could not check main process: {e}")
    
    # Check for training processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_cumulative_mastery_full.py"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            training_pids = result.stdout.strip().split('\n')
            print(f"🏃 {len(training_pids)} Training process(es) ACTIVE")
            print(f"   PIDs: {', '.join(training_pids)}")
            
            # Check GPU usage
            try:
                gpu_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                if gpu_result.returncode == 0:
                    print("   GPU Usage:")
                    for line in gpu_result.stdout.strip().split('\n')[:5]:  # GPUs 0-4
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            gpu_id, util, mem = parts[0], parts[1], parts[2]
                            print(f"     GPU {gpu_id}: {util}% utilization, {mem}MB memory")
            except:
                pass
        else:
            print("💤 No training processes running")
    except Exception as e:
        print(f"❓ Could not check training processes: {e}")


def find_latest_results_file():
    """Find the most recent parallel sweep results file."""
    pattern = "parallel_sweep_results_*.json"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def display_sweep_results():
    """Display current sweep results."""
    
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
            
            # Find best AUC
            best_auc_run = max(successful_runs, key=lambda x: x['metrics'].get('best_val_auc', 0))
            best_auc = best_auc_run['metrics'].get('best_val_auc', 0)
            
            print(f"   🎯 Best AUC: {best_auc:.4f} (Run {best_auc_run['run_id']})")
            print(f"   🖥️  GPU: {best_auc_run['gpu_id']}")
            print(f"   ⏱️  Duration: {best_auc_run.get('duration_minutes', 0):.1f} min")
            
            # Show key parameters
            params = best_auc_run['parameters']
            key_params = ['epochs', 'num_encoder_blocks', 'd_ff', 'mastery_performance_loss_weight', 'gain_performance_loss_weight']
            print("   📋 Key Parameters:")
            for param in key_params:
                if param in params:
                    value = params[param]
                    if isinstance(value, float):
                        print(f"      {param}: {value:.4f}")
                    else:
                        print(f"      {param}: {value}")
            
            # Show recent successful runs
            recent_successful = [r for r in successful_runs if r['run_id'] > max(0, total_runs - 5)]
            if recent_successful:
                print(f"\\n📈 RECENT SUCCESSFUL RUNS:")
                for run in sorted(recent_successful, key=lambda x: x['run_id']):
                    auc = run['metrics'].get('best_val_auc', 0)
                    duration = run.get('duration_minutes', 0)
                    gpu = run['gpu_id']
                    print(f"   Run {run['run_id']}: AUC {auc:.4f} ({duration:.1f}min, GPU {gpu})")
        
        else:
            print("\\n⏳ No successful runs yet...")
    
    except Exception as e:
        print(f"❌ Error reading results: {e}")


def check_log_file():
    """Check the nohup log file for recent output."""
    
    log_file = "sweep_output.log"
    if os.path.exists(log_file):
        print(f"\\n📜 RECENT LOG OUTPUT (last 10 lines from {log_file}):")
        print("-" * 50)
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"   {line.rstrip()}")
        except Exception as e:
            print(f"   Error reading log: {e}")
    else:
        print(f"\\n📜 No log file found ({log_file})")


def show_control_commands():
    """Show commands for controlling the background sweep."""
    
    print(f"\\n🎛️  CONTROL COMMANDS:")
    print("=" * 40)
    print("📊 Monitor status:")
    print("   python monitor_persistent_sweep.py")
    print()
    print("📜 View full log:")
    print("   tail -f sweep_output.log")
    print()
    print("⏹️  Stop sweep:")
    print("   pkill -f parallel_sweep.py")
    print("   pkill -f train_cumulative_mastery_full.py")
    print()
    print("🔄 Restart sweep (if stopped):")
    print("   nohup python parallel_sweep.py --count 15 --max_parallel 5 > sweep_output.log 2>&1 &")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor persistent wandb sweep')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Monitor continuously with updates every 30 seconds')
    parser.add_argument('--log', '-l', action='store_true',
                       help='Show recent log output')
    parser.add_argument('--commands', action='store_true',
                       help='Show control commands')
    
    args = parser.parse_args()
    
    if args.commands:
        show_control_commands()
        exit(0)
    
    if args.continuous:
        print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
        print("Updates every 30 seconds")
        print("=" * 50)
        
        try:
            while True:
                os.system('clear')  # Clear screen
                print(f"⏰ Status at {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 50)
                
                check_sweep_processes()
                display_sweep_results()
                
                if args.log:
                    check_log_file()
                
                print("\\n⏳ Next update in 30 seconds... (Ctrl+C to stop)")
                time.sleep(30)
        
        except KeyboardInterrupt:
            print("\\n\\n⏹️  Monitoring stopped")
    
    else:
        # Single status check
        check_sweep_processes()
        display_sweep_results()
        
        if args.log:
            check_log_file()
        
        show_control_commands()