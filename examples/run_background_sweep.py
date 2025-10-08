#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/workspaces/pykt-toolkit')

import json
import time
import subprocess
import signal
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_improved_multi_gpu_sweep import ImprovedMultiGPUSweep
import threading

class BackgroundSweepManager:
    def __init__(self):
        self.active_processes = {}
        self.results = []
        self.sweep = ImprovedMultiGPUSweep()
        
    def run_background_experiment(self, gpu_id, run_id, params):
        """Run a single experiment in the background with nohup"""
        
        # Create unique log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"bg_sweep_gpu{gpu_id}_run{run_id}_{timestamp}.log"
        
        # Build nohup command with correct parameter names
        cmd = f"""nohup python wandb_gainakt2exp_train.py \
            --learning_rate {params['lr']} \
            --weight_decay {params['wd']} \
            --batch_size {params['bs']} \
            --epochs {params['epochs']} \
            --enhanced_constraints {str(params['enhanced']).lower()} \
            > {log_file} 2>&1 &"""
        
        # Set GPU environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"🚀 GPU{gpu_id} Run {run_id}: LAUNCHING BACKGROUND EXPERIMENT")
        print(f"   ⚡ PARAMS: lr={params['lr']:.6f}, wd={params['wd']:.6f}, bs={params['bs']}, epochs={params['epochs']}, enhanced={params['enhanced']}")
        print(f"   📝 Log: {log_file}")
        
        try:
            # Start background process
            process = subprocess.Popen(
                cmd,
                shell=True,
                env=env,
                cwd='/workspaces/pykt-toolkit/examples',
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Store process info
            self.active_processes[run_id] = {
                'process': process,
                'gpu_id': gpu_id,
                'params': params,
                'log_file': log_file,
                'start_time': time.time()
            }
            
            return True
            
        except Exception as e:
            print(f"❌ GPU{gpu_id} Run {run_id}: Failed to start - {e}")
            return False
    
    def monitor_experiments(self, timeout_minutes=15):
        """Monitor running experiments and collect results"""
        print(f"\\n👀 MONITORING {len(self.active_processes)} background experiments...")
        print(f"⏱️  Timeout: {timeout_minutes} minutes per experiment")
        
        while self.active_processes:
            completed_runs = []
            
            for run_id, proc_info in self.active_processes.items():
                process = proc_info['process']
                log_file = proc_info['log_file']
                start_time = proc_info['start_time']
                gpu_id = proc_info['gpu_id']
                params = proc_info['params']
                
                # Check if process is still running
                if process.poll() is not None:
                    # Process completed
                    duration = time.time() - start_time
                    completed_runs.append(run_id)
                    
                    # Parse results from log file
                    result = self.parse_log_file(log_file, gpu_id, run_id, params, duration)
                    if result:
                        self.results.append(result)
                    
                elif time.time() - start_time > timeout_minutes * 60:
                    # Process timed out
                    print(f"⏱️  GPU{gpu_id} Run {run_id}: TIMEOUT after {timeout_minutes} minutes")
                    try:
                        # Kill process group
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        process.wait(timeout=5)
                    except:
                        pass
                    completed_runs.append(run_id)
            
            # Remove completed processes
            for run_id in completed_runs:
                del self.active_processes[run_id]
            
            if self.active_processes:
                # Print status update
                print(f"\\r⏳ Still running: {len(self.active_processes)} experiments", end="", flush=True)
                time.sleep(10)  # Check every 10 seconds
            
        print(f"\\n✅ All experiments completed!")
    
    def parse_log_file(self, log_file, gpu_id, run_id, params, duration):
        """Parse AUC from log file"""
        log_path = f'/workspaces/pykt-toolkit/examples/{log_file}'
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Use same parsing logic as ImprovedMultiGPUSweep
            best_val_auc = None
            epoch_info = []
            
            for line in content.split('\\n'):
                line_lower = line.lower().strip()
                
                if '- INFO -   Valid - Loss:' in line and 'AUC:' in line:
                    try:
                        auc_part = line.split('AUC:')[1].split(',')[0].strip()
                        candidate_auc = float(auc_part)
                        if candidate_auc > (best_val_auc or 0):
                            best_val_auc = candidate_auc
                    except (ValueError, IndexError):
                        continue
                
                elif 'New best model saved (Val AUC:' in line:
                    try:
                        auc_match = line.split('Val AUC: ')[1].split(')')[0]
                        epoch_info.append(f"AUC:{auc_match}")
                        potential_auc = float(auc_match)
                        if potential_auc > (best_val_auc or 0):
                            best_val_auc = potential_auc
                    except (ValueError, IndexError):
                        continue
            
            if best_val_auc is not None:
                status = "🎉 TARGET!" if best_val_auc >= 0.7259 else "✅ SUCCESS"
                improvement = " ⭐ HITS TARGET!" if best_val_auc >= 0.7259 else ""
                speed_rating = "⚡ FAST" if duration < 120 else "🐌 SLOW" if duration > 300 else "⏱️ NORMAL"
                
                print(f"\\n🏁 GPU{gpu_id} Run {run_id}: {status} - AUC: {best_val_auc:.4f} ({duration/60:.1f}min) {speed_rating}{improvement}")
                if epoch_info:
                    print(f"   📈 Progress: {' → '.join(epoch_info[-3:])}")
                
                return {
                    'run_id': run_id,
                    'gpu_id': gpu_id,
                    'params': params,
                    'auc': best_val_auc,
                    'duration_minutes': duration / 60,
                    'status': 'success',
                    'log_file': log_file
                }
            else:
                print(f"\\n❌ GPU{gpu_id} Run {run_id}: FAILED - Could not parse AUC from {log_file}")
                return None
                
        except Exception as e:
            print(f"\\n💥 GPU{gpu_id} Run {run_id}: ERROR reading {log_file} - {e}")
            return None

def main():
    print("🔄 Background Multi-GPU GainAKT2Exp Parameter Sweep")
    print("=" * 70)
    print("🔧 This will run experiments that persist even if terminal is closed")
    print("📝 Each experiment logs to its own file for monitoring")
    print("")
    
    response = input("Run 20 background experiments across 5 GPUs? [y/N]: ")
    if response.lower() != 'y':
        return 0
    
    # Create manager
    manager = BackgroundSweepManager()
    
    # Generate parameters (same as original sweep)
    base_lr = 0.000348
    base_wd = 5.857e-05
    
    param_combinations = []
    for i in range(20):
        # Generate variations around base values
        lr_mult = [0.5, 0.75, 1.0, 1.3, 1.5][i % 5]
        wd_mult = [0.3, 0.5, 1.0, 1.7, 2.0][i % 5]
        
        param_combinations.append({
            'lr': base_lr * lr_mult,
            'wd': base_wd * wd_mult,
            'bs': [64, 96, 128][i % 3],
            'epochs': 10,  # Fast mode
            'enhanced': [True, False][i % 2]
        })
    
    print(f"\\n🚀 LAUNCHING {len(param_combinations)} BACKGROUND EXPERIMENTS...")
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    # Launch all experiments
    launched_count = 0
    for i, params in enumerate(param_combinations):
        gpu_id = i % 5  # Distribute across 5 GPUs
        run_id = i + 1
        
        if manager.run_background_experiment(gpu_id, run_id, params):
            launched_count += 1
        
        time.sleep(1)  # Small delay between launches
    
    print(f"\\n✅ LAUNCHED {launched_count}/{len(param_combinations)} experiments successfully!")
    print("\\n" + "="*70)
    print("🔄 EXPERIMENTS ARE NOW RUNNING IN BACKGROUND")
    print("💡 You can safely close this terminal - experiments will continue!")
    print("📊 Monitor progress by checking log files: bg_sweep_gpu*_run*.log")
    print("🔍 Or run this script again to monitor existing experiments")
    print("="*70)
    
    # Ask if user wants to monitor now
    response = input("\\nWould you like to monitor the experiments now? [y/N]: ")
    if response.lower() == 'y':
        try:
            manager.monitor_experiments(timeout_minutes=15)
            
            # Save results
            if manager.results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = f"cumulative_mastery_results_background_sweep_{timestamp}.json"
                with open(results_file, 'w') as f:
                    json.dump(manager.results, f, indent=2)
                
                print(f"\\n💾 Results saved to: {results_file}")
                
                # Print summary
                successful = len(manager.results)
                target_hits = sum(1 for r in manager.results if r['auc'] >= 0.7259)
                
                print(f"\\n📊 FINAL SUMMARY:")
                print(f"   Total experiments: {len(param_combinations)}")
                print(f"   Successful: {successful}")
                print(f"   Target AUC hits: {target_hits}")
                if successful > 0:
                    avg_auc = sum(r['auc'] for r in manager.results) / len(manager.results)
                    best_auc = max(r['auc'] for r in manager.results)
                    print(f"   Average AUC: {avg_auc:.4f}")
                    print(f"   Best AUC: {best_auc:.4f}")
        
        except KeyboardInterrupt:
            print("\\n\\n⚠️  MONITORING STOPPED")
            print("🔄 Experiments continue running in background!")
            print("📝 Check log files for progress: bg_sweep_gpu*_run*.log")
    else:
        print("\\n👋 Experiments launched! Check back later for results.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())