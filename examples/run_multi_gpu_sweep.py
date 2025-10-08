#!/usr/bin/env python3
"""
Multi-GPU Focused Parameter Sweep for GainAKT2Exp
Runs experiments in parallel across 5 GPUs for faster execution
Target: AUC >= 0.7259
"""

import sys
import subprocess
import random
import threading
import time
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

class MultiGPUSweep:
    def __init__(self, num_gpus=5, experiments_per_gpu=4):
        self.num_gpus = num_gpus
        self.experiments_per_gpu = experiments_per_gpu
        self.total_experiments = num_gpus * experiments_per_gpu
        self.results = []
        self.results_lock = threading.Lock()
        
    def generate_parameter_combinations(self):
        """Generate parameter combinations around current defaults"""
        
        combinations = []
        
        for i in range(self.total_experiments):
            combo = {
                'learning_rate': random.uniform(0.0001, 0.0008),
                'weight_decay': random.uniform(0.00002, 0.0002),
                'batch_size': random.choice([64, 96, 128, 160, 192]),
                'num_epochs': random.choice([15, 18, 20, 22, 25, 30]),
                'enhanced_constraints': random.choice([True, False]),
                'patience': random.choice([15, 20, 25]),
                'dataset_name': 'assist2015',
                'fold': 0,
                'use_wandb': 0,
                'experiment_suffix': f'mgpu_sweep_run_{i+1}'
            }
            combinations.append(combo)
        
        return combinations
    
    def run_single_experiment(self, params, run_id, gpu_id):
        """Run a single experiment on specified GPU"""
        
        print(f"🔬 GPU{gpu_id} Run {run_id}: lr={params['learning_rate']:.6f}, wd={params['weight_decay']:.6f}")
        print(f"   GPU{gpu_id} Run {run_id}: bs={params['batch_size']}, epochs={params['num_epochs']}, enhanced={params['enhanced_constraints']}")
        
        # Set CUDA_VISIBLE_DEVICES to use specific GPU
        env = {'CUDA_VISIBLE_DEVICES': str(gpu_id)}
        
        # Build command
        cmd = [
            'python', 'wandb_gainakt2exp_train.py',
            '--learning_rate', str(params['learning_rate']),
            '--weight_decay', str(params['weight_decay']),
            '--batch_size', str(params['batch_size']),
            '--num_epochs', str(params['num_epochs']),
            '--enhanced_constraints', str(params['enhanced_constraints']),
            '--patience', str(params['patience']),
            '--use_wandb', '0',
            '--experiment_suffix', f"{params['experiment_suffix']}_gpu{gpu_id}"
        ]
        
        start_time = time.time()
        
        try:
            # Run experiment with GPU assignment and timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600,  # 1 hour timeout
                env={**subprocess.os.environ, **env},
                cwd='/workspaces/pykt-toolkit/examples'
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Parse AUC from output
                best_val_auc = None
                for line in result.stdout.split('\\n'):
                    if 'FINAL_RESULTS: best_val_auc:' in line:
                        best_val_auc = float(line.split(':')[-1].strip())
                        break
                
                if best_val_auc is not None:
                    status = "🎉 TARGET!" if best_val_auc >= 0.7259 else "✅ SUCCESS"
                    print(f"   GPU{gpu_id} Run {run_id}: {status} - AUC: {best_val_auc:.4f} ({duration/60:.1f}min)")
                    
                    result_data = {
                        'run_id': run_id,
                        'gpu_id': gpu_id,
                        'params': params,
                        'auc': best_val_auc,
                        'duration_minutes': duration / 60,
                        'status': 'success'
                    }
                    
                    # Thread-safe result storage
                    with self.results_lock:
                        self.results.append(result_data)
                    
                    return result_data
                else:
                    print(f"   GPU{gpu_id} Run {run_id}: ❌ FAILED - Could not parse AUC")
            else:
                print(f"   GPU{gpu_id} Run {run_id}: ❌ FAILED - Return code: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"   GPU{gpu_id} Run {run_id}: ⏰ TIMEOUT - Experiment took >{duration/60:.1f}min")
        except Exception as e:
            duration = time.time() - start_time
            print(f"   GPU{gpu_id} Run {run_id}: ❌ ERROR - {e}")
        
        # Failed result
        failed_result = {
            'run_id': run_id,
            'gpu_id': gpu_id, 
            'params': params,
            'auc': 0.0,
            'duration_minutes': duration / 60 if 'duration' in locals() else 0,
            'status': 'failed'
        }
        
        with self.results_lock:
            self.results.append(failed_result)
        
        return failed_result
    
    def run_gpu_worker(self, gpu_id, experiment_queue):
        """Worker function for a single GPU"""
        print(f"🚀 GPU {gpu_id} worker started")
        
        while True:
            try:
                # Get next experiment (blocks if queue is empty)
                params, run_id = experiment_queue.get(timeout=1)
                
                # Run experiment on this GPU
                self.run_single_experiment(params, run_id, gpu_id)
                
                # Mark task as done
                experiment_queue.task_done()
                
            except queue.Empty:
                # No more experiments
                break
            except Exception as e:
                print(f"❌ GPU {gpu_id} worker error: {e}")
                experiment_queue.task_done()
        
        print(f"✅ GPU {gpu_id} worker finished")
    
    def run_sweep(self):
        """Run the multi-GPU parameter sweep"""
        
        print("🎯 Multi-GPU GainAKT2Exp Focused Parameter Sweep")
        print("=" * 70)
        print(f"🔧 Configuration:")
        print(f"   GPUs: {self.num_gpus}")
        print(f"   Experiments per GPU: {self.experiments_per_gpu}")
        print(f"   Total experiments: {self.total_experiments}")
        print(f"   Target AUC: >= 0.7259")
        
        # Generate parameter combinations
        print(f"\\n📋 Generating {self.total_experiments} parameter combinations...")
        combinations = self.generate_parameter_combinations()
        
        # Create experiment queue
        experiment_queue = queue.Queue()
        for i, params in enumerate(combinations, 1):
            experiment_queue.put((params, i))
        
        print(f"\\n🚀 Starting {self.num_gpus} GPU workers...")
        start_time = time.time()
        
        # Start GPU workers using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit GPU worker tasks
            futures = [
                executor.submit(self.run_gpu_worker, gpu_id, experiment_queue) 
                for gpu_id in range(self.num_gpus)
            ]
            
            # Monitor progress
            last_completed = 0
            while not experiment_queue.empty() or any(not f.done() for f in futures):
                time.sleep(10)  # Check every 10 seconds
                
                with self.results_lock:
                    completed = len(self.results)
                    if completed > last_completed:
                        best_auc = max([r['auc'] for r in self.results], default=0.0)
                        successful = len([r for r in self.results if r['status'] == 'success'])
                        target_achieved = len([r for r in self.results if r['auc'] >= 0.7259])
                        
                        print(f"📊 Progress: {completed}/{self.total_experiments} completed, "
                              f"Best AUC: {best_auc:.4f}, Target achieved: {target_achieved}")
                        last_completed = completed
            
            # Wait for all workers to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Worker error: {e}")
        
        total_time = time.time() - start_time
        
        # Analyze results
        self.analyze_results(total_time)
    
    def analyze_results(self, total_time_seconds):
        """Analyze and display sweep results"""
        
        print(f"\\n📊 MULTI-GPU SWEEP RESULTS")
        print("=" * 70)
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if successful_results:
            # Sort by AUC
            successful_results.sort(key=lambda x: x['auc'], reverse=True)
            
            print(f"⏱️  Total sweep time: {total_time_seconds/60:.1f} minutes")
            print(f"✅ Successful runs: {len(successful_results)}/{self.total_experiments}")
            print(f"🏆 Best AUC: {successful_results[0]['auc']:.4f}")
            
            # GPU utilization stats
            gpu_stats = {}
            for result in successful_results:
                gpu_id = result['gpu_id']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = []
                gpu_stats[gpu_id].append(result['auc'])
            
            print(f"\\n📈 GPU Performance:")
            for gpu_id in sorted(gpu_stats.keys()):
                aucs = gpu_stats[gpu_id]
                avg_auc = sum(aucs) / len(aucs)
                best_auc = max(aucs)
                print(f"   GPU {gpu_id}: {len(aucs)} runs, avg AUC: {avg_auc:.4f}, best: {best_auc:.4f}")
            
            # Show top results
            print("\\n🥇 TOP 10 RESULTS:")
            print("-" * 100)
            print(f"{'Rank':<4} {'AUC':<8} {'GPU':<4} {'LR':<10} {'WD':<10} {'BS':<4} {'Epochs':<6} {'Enhanced':<8} {'Time(min)'}")
            print("-" * 100)
            
            for i, result in enumerate(successful_results[:10], 1):
                params = result['params']
                print(f"{i:<4} {result['auc']:<8.4f} {result['gpu_id']:<4} "
                      f"{params['learning_rate']:<10.6f} {params['weight_decay']:<10.6f} "
                      f"{params['batch_size']:<4} {params['num_epochs']:<6} "
                      f"{params['enhanced_constraints']:<8} {result['duration_minutes']:<8.1f}")
            
            # Check target achievement
            target_runs = [r for r in successful_results if r['auc'] >= 0.7259]
            if target_runs:
                print(f"\\n🎉 TARGET ACHIEVED! {len(target_runs)} runs reached AUC >= 0.7259")
                best = target_runs[0]
                print(f"\\n🥇 Best configuration (AUC: {best['auc']:.4f}):")
                print(f"   GPU: {best['gpu_id']}")
                for key, value in best['params'].items():
                    if key not in ['dataset_name', 'fold', 'use_wandb', 'experiment_suffix']:
                        print(f"   {key}: {value}")
            else:
                print(f"\\n📈 Target not reached. Best AUC: {successful_results[0]['auc']:.4f}")
                print("Consider running more experiments or expanding search ranges.")
            
            # Save results
            self.save_results(successful_results, total_time_seconds)
            
        else:
            print("❌ No successful experiments completed")
            print(f"⏱️  Total time: {total_time_seconds/60:.1f} minutes")
    
    def save_results(self, successful_results, total_time):
        """Save detailed results to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'multi_gpu_sweep_results_{timestamp}.txt'
        
        with open(results_file, 'w') as f:
            f.write("Multi-GPU GainAKT2Exp Focused Sweep Results\\n")
            f.write("=" * 60 + "\\n")
            f.write(f"Timestamp: {datetime.now()}\\n")
            f.write(f"GPUs used: {self.num_gpus}\\n")
            f.write(f"Total experiments: {self.total_experiments}\\n")
            f.write(f"Successful: {len(successful_results)}\\n")
            f.write(f"Total time: {total_time/60:.1f} minutes\\n")
            
            if successful_results:
                f.write(f"Best AUC: {successful_results[0]['auc']:.4f}\\n\\n")
                
                f.write("All Results (sorted by AUC):\\n")
                f.write("-" * 60 + "\\n")
                
                for i, result in enumerate(successful_results, 1):
                    f.write(f"\\nRank {i}: AUC={result['auc']:.4f}, GPU={result['gpu_id']}, "
                           f"Time={result['duration_minutes']:.1f}min\\n")
                    for key, value in result['params'].items():
                        f.write(f"  {key}: {value}\\n")
        
        print(f"\\n💾 Detailed results saved to: {results_file}")

def main():
    """Main function"""
    
    # Configuration
    num_gpus = 5
    experiments_per_gpu = 4  # 20 total experiments
    
    # Check for auto-confirm flag
    auto_confirm = len(sys.argv) > 1 and sys.argv[1] in ['--yes', '-y', 'auto']
    
    if not auto_confirm:
        print("🎯 Multi-GPU GainAKT2Exp Parameter Sweep")
        print("This will run experiments in parallel across 5 GPUs")
        print()
        
        response = input(f"Run {num_gpus * experiments_per_gpu} experiments across {num_gpus} GPUs? [y/N]: ")
        if response.lower() != 'y':
            print("Sweep cancelled.")
            return 0
    else:
        print("🚀 Auto-starting Multi-GPU GainAKT2Exp Parameter Sweep")
        print(f"Running {num_gpus * experiments_per_gpu} experiments across {num_gpus} GPUs...")
        print()
    
    # Run sweep
    sweep = MultiGPUSweep(num_gpus=num_gpus, experiments_per_gpu=experiments_per_gpu)
    sweep.run_sweep()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())