#!/usr/bin/env python3
"""
Monitor Comprehensive Sweep Progress
Shows real-time metrics and per-epoch progression
"""

import os
import json
import time
from datetime import datetime
import subprocess


def check_comprehensive_status():
    """Check comprehensive sweep status with detailed metrics."""
    
    print(f"🔍 COMPREHENSIVE SWEEP MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # Check processes
    try:
        result = subprocess.run(["pgrep", "-f", "comprehensive_sweep.py"], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Comprehensive sweep process RUNNING")
        else:
            print("⏸️  No comprehensive sweep process detected")
    except:
        print("❓ Could not check process status")
    
    # Check training processes
    try:
        result = subprocess.run(["pgrep", "-f", "train_cumulative_mastery_full.py"], capture_output=True, text=True)
        if result.stdout.strip():
            training_pids = result.stdout.strip().split('\n')
            print(f"🏃 {len(training_pids)} Training processes ACTIVE")
        else:
            print("💤 No training processes running")
    except:
        pass
    
    # GPU status
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("🖥️  GPU Status:")
            for line in result.stdout.strip().split('\n')[:5]:
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id, util, mem = parts[0], parts[1], parts[2]
                    print(f"   GPU {gpu_id}: {util} utilization, {mem} memory")
    except:
        pass
    
    # Check results file
    pattern = "comprehensive_sweep_results_*.json"
    import glob
    files = glob.glob(pattern)
    if files:
        latest_file = max(files, key=os.path.getctime)
        print(f"\\n📊 RESULTS FROM: {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            total = len(results)
            successful = sum(1 for r in results if r.get('status') == 'success')
            failed = sum(1 for r in results if r.get('status') != 'success')
            
            print(f"📋 Progress: {total}/20 runs completed")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            
            if successful > 0:
                print(f"\\n🏆 BEST RESULTS SO FAR:")
                successful_runs = [r for r in results if r.get('status') == 'success']
                best_run = max(successful_runs, key=lambda x: x.get('metrics', {}).get('best_val_auc', 0))
                
                metrics = best_run.get('metrics', {})
                best_auc = metrics.get('best_val_auc', 0)
                epochs = metrics.get('final_epoch', 0)
                
                print(f"   🎯 Best AUC: {best_auc:.4f} (Run {best_run['run_id']})")
                print(f"   📊 Epochs completed: {epochs}")
                print(f"   🖥️  GPU: {best_run['gpu_id']}")
                print(f"   ⏱️  Duration: {best_run.get('duration_minutes', 0):.1f} min")
                
                # Show parameters of best run
                params = best_run.get('parameters', {})
                print(f"   📋 Parameters:")
                for key in ['batch_size', 'lr', 'weight_decay', 'enhanced_constraints']:
                    if key in params:
                        val = params[key]
                        if isinstance(val, float) and key == 'lr':
                            print(f"      {key}: {val:.4f}")
                        elif isinstance(val, float):
                            print(f"      {key}: {val:.6f}")
                        else:
                            print(f"      {key}: {val}")
                
                # Show per-epoch progression for best run
                val_aucs = metrics.get('val_aucs', [])
                if len(val_aucs) >= 3:
                    print(f"   📈 AUC Progression: {val_aucs[0]:.3f} → {val_aucs[len(val_aucs)//2]:.3f} → {val_aucs[-1]:.3f}")
            
            # Show recent completions
            if total > 0:
                print(f"\\n📈 RECENT COMPLETIONS:")
                for run in results[-3:]:
                    status = run['status']
                    run_id = run['run_id']
                    duration = run.get('duration_minutes', 0)
                    gpu = run['gpu_id']
                    
                    if status == 'success':
                        auc = run.get('metrics', {}).get('best_val_auc', 0)
                        epochs = run.get('metrics', {}).get('final_epoch', 0)
                        print(f"   Run {run_id}: ✅ AUC {auc:.4f} in {epochs} epochs ({duration:.1f}min, GPU {gpu})")
                    else:
                        print(f"   Run {run_id}: ❌ {status} ({duration:.1f}min, GPU {gpu})")
        
        except Exception as e:
            print(f"❌ Error reading results: {e}")
    else:
        print("\\n📄 No results files found yet")
    
    # Check log file
    log_pattern = "comprehensive_sweep_*.log"  
    log_files = glob.glob(log_pattern)
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        if os.path.exists(latest_log):
            print(f"\\n📜 LATEST LOG ENTRIES:")
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-5:]:
                        print(f"   {line.rstrip()}")
            except:
                pass


def monitor_continuously():
    """Continuous monitoring with updates."""
    
    print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
    print("Updates every 30 seconds")
    
    try:
        while True:
            os.system('clear')
            check_comprehensive_status()
            print("\\n⏳ Next update in 30 seconds...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Monitoring stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor comprehensive sweep')
    parser.add_argument('--continuous', '-c', action='store_true', help='Continuous monitoring')
    
    args = parser.parse_args()
    
    if args.continuous:
        monitor_continuously()
    else:
        check_comprehensive_status()