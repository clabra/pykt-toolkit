#!/usr/bin/env python3

import os
import time
import glob
from datetime import datetime

def monitor_experiments():
    """Monitor background experiments in real-time"""
    print("🔍 Background Experiment Monitor")
    print("=" * 50)
    print("📊 Real-time monitoring of 20 experiments across 5 GPUs")
    print("⏱️  Updates every 30 seconds")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Get current timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Find all log files
            log_files = glob.glob('bg_sweep_gpu*.log')
            log_files.sort()
            
            print(f"\n🕐 Status at {timestamp}")
            print("-" * 50)
            
            completed_experiments = 0
            active_experiments = 0
            total_size = 0
            
            # Quick analysis of each log
            for log_file in log_files:
                try:
                    # Extract GPU and run ID from filename
                    parts = log_file.split('_')
                    gpu_id = parts[2].replace('gpu', '')
                    run_id = parts[3].replace('run', '')
                    
                    # Get file size
                    size = os.path.getsize(log_file)
                    total_size += size
                    
                    # Check content
                    with open(log_file, 'r') as f:
                        content = f.read()
                        
                        # Check for completion indicators
                        if 'Valid - Loss:' in content and 'AUC:' in content:
                            # Find the latest AUC
                            auc_lines = [line for line in content.split('\n') if 'Valid - Loss:' in line and 'AUC:' in line]
                            if auc_lines:
                                latest_auc = auc_lines[-1]
                                try:
                                    auc_value = float(latest_auc.split('AUC:')[1].split(',')[0].strip())
                                    status = "🎉 TARGET!" if auc_value >= 0.7259 else "✅ TRAINING"
                                    print(f"GPU{gpu_id} Run{run_id}: {status} - AUC: {auc_value:.4f} ({size//1024}KB)")
                                    active_experiments += 1
                                except:
                                    print(f"GPU{gpu_id} Run{run_id}: 🔄 TRAINING... ({size//1024}KB)")
                                    active_experiments += 1
                            else:
                                print(f"GPU{gpu_id} Run{run_id}: 🔄 TRAINING... ({size//1024}KB)")
                                active_experiments += 1
                                
                        elif 'fold	modelname' in content:  # Final results table
                            completed_experiments += 1
                            # Try to extract final AUC from table
                            lines = content.split('\n')
                            for line in lines:
                                if 'gainakt2exp' in line.lower() and line.strip().startswith('0'):
                                    parts = line.split()
                                    if len(parts) >= 8:
                                        try:
                                            final_auc = float(parts[7])
                                            status = "🎉 COMPLETE!" if final_auc >= 0.7259 else "✅ COMPLETE"
                                            print(f"GPU{gpu_id} Run{run_id}: {status} - Final AUC: {final_auc:.4f} ({size//1024}KB)")
                                            break
                                        except:
                                            print(f"GPU{gpu_id} Run{run_id}: ✅ COMPLETE ({size//1024}KB)")
                                            break
                            else:
                                print(f"GPU{gpu_id} Run{run_id}: ✅ COMPLETE ({size//1024}KB)")
                                
                        elif 'Epoch 1/20' in content:
                            print(f"GPU{gpu_id} Run{run_id}: 🚀 STARTING... ({size//1024}KB)")
                            active_experiments += 1
                        else:
                            print(f"GPU{gpu_id} Run{run_id}: 🔧 SETUP... ({size//1024}KB)")
                            active_experiments += 1
                            
                except Exception as e:
                    print(f"GPU{gpu_id} Run{run_id}: ❌ ERROR - {str(e)[:30]}")
            
            # Summary
            print(f"\n📈 SUMMARY:")
            print(f"   📊 Active: {active_experiments}, Complete: {completed_experiments}, Total: {len(log_files)}")
            print(f"   💾 Total log size: {total_size//1024//1024:.1f}MB")
            
            if completed_experiments == len(log_files):
                print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
                break
            
            print(f"\n⏳ Next update in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Monitoring stopped. Experiments continue running in background.")
        print(f"📝 Check individual logs: tail -f bg_sweep_gpu*_run*.log")

if __name__ == "__main__":
    monitor_experiments()