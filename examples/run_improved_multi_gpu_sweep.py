#!/usr/bin/env python3
"""
Improved Multi-GPU Parameter Sweep for GainAKT2Exp
- Enhanced real-time monitoring and feedback  
- 5 GPUs simultaneous execution (no staggered launches)
- Better error handling and timeouts
- Comprehensive progress tracking
"""

import sys
import subprocess
import random
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

class ImprovedMultiGPUSweep:
    def __init__(self, num_gpus=5, experiments_per_gpu=4):
        self.num_gpus = num_gpus
        self.experiments_per_gpu = experiments_per_gpu
        self.total_experiments = num_gpus * experiments_per_gpu
        
    def generate_parameter_combinations(self):
        """Generate parameter combinations around current defaults with focused ranges"""
        
        # Base parameters from best run (Run 13)
        base_lr = 0.0003479574665378502
        base_wd = 5.857348235335391e-05
        
        combinations = []
        
        # Focused ranges around the best parameters - FAST MODE
        lr_multipliers = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]  # Tighter range
        wd_multipliers = [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]    # More conservative
        batch_sizes = [64, 96, 128]                               # Smaller batch sizes for memory
        epochs_options = [10]                                     # FAST: Only 10 epochs
        enhanced_options = [True, False]
        
        for i in range(self.total_experiments):
            # More focused parameter selection
            lr_mult = random.choice(lr_multipliers)
            wd_mult = random.choice(wd_multipliers) 
            
            combination = {
                'learning_rate': base_lr * lr_mult,
                'weight_decay': base_wd * wd_mult,
                'batch_size': random.choice(batch_sizes),
                'num_epochs': random.choice(epochs_options),
                'enhanced_constraints': random.choice(enhanced_options),
                'dataset_name': 'assist2015',
                'fold': 0,
                'use_wandb': 0,
                'experiment_suffix': f'improved_sweep_run_{i+1}'
            }
            combinations.append(combination)
        
        return combinations

def run_single_gpu_experiment(args):
    """Run a single experiment on specified GPU - runs in separate process"""
    params, run_id, gpu_id = args
    
    print(f"� GPU{gpu_id} Run {run_id}: LAUNCHING FAST EXPERIMENT")
    print(f"   ⚡ SPEED PARAMS: lr={params['learning_rate']:.6f}, wd={params['weight_decay']:.6f}, bs={params['batch_size']}, epochs={params['num_epochs']}, enhanced={params['enhanced_constraints']}")
    
    start_time = time.time()
    print(f"   ⏰ GPU{gpu_id} Run {run_id}: Started at {time.strftime('%H:%M:%S')}")
    
    try:
        # Set GPU environment for this process
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Build command for background execution with nohup
        log_file = f"sweep_gpu{gpu_id}_run{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd_str = f"nohup python wandb_gainakt2exp_train.py --lr {params['lr']} --wd {params['wd']} --bs {params['bs']} --epochs {params['epochs']} --enhanced {str(params['enhanced']).lower()} > {log_file} 2>&1"
        
        print(f"   ⚡ SPEED PARAMS: lr={params['lr']:.6f}, wd={params['wd']:.6f}, bs={params['bs']}, epochs={params['epochs']}, enhanced={params['enhanced']}")
        print(f"   📝 Log file: {log_file}")
        
        # Start process with nohup for persistence
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            cwd='/workspaces/pykt-toolkit/examples',
            timeout=timeout_seconds
        )
        
        # Read the log file for complete output
        try:
            with open(f'/workspaces/pykt-toolkit/examples/{log_file}', 'r') as f:
                log_content = f.read()
                if log_content.strip():
                    result.stdout = log_content
        except FileNotFoundError:
            # If log file doesn't exist, use captured output
            pass
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Parse AUC and other metrics from output - try multiple formats
            best_val_auc = None
            epoch_info = []
            
            for line in result.stdout.split('\\n'):
                line_lower = line.lower().strip()
                # Try multiple AUC parsing patterns
                if 'FINAL_RESULTS: best_val_auc:' in line:
                    best_val_auc = float(line.split(':')[-1].strip())
                    break
                elif '- INFO -   Valid - Loss:' in line and 'AUC:' in line:  # Main format we saw
                    # Format: "2025-10-08 00:25:48,859 - INFO -   Valid - Loss: 0.5155, AUC: 0.7180, Acc: 0.7511"
                    try:
                        auc_part = line.split('AUC:')[1].split(',')[0].strip()
                        best_val_auc = float(auc_part)
                        break
                    except (ValueError, IndexError):
                        continue
                elif 'validauc' in line_lower and 'testauc' in line_lower:  # Table header
                    continue  # Skip header, look for data row
                elif line.strip().startswith('0') and 'gainakt2exp' in line_lower:  # Table row format
                    # Format: fold modelname embtype testauc testacc window_testauc window_testacc validauc validacc best_epoch
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            # validauc is usually at index 7
                            val_auc = float(parts[7])
                            if 0 <= val_auc <= 1:  # Reasonable AUC range
                                best_val_auc = val_auc
                                break
                        except (ValueError, IndexError):
                            continue
                # Track training progress
                elif 'New best model saved (Val AUC:' in line:
                    try:
                        auc_match = line.split('Val AUC: ')[1].split(')')[0]
                        epoch_info.append(f"AUC:{auc_match}")
                        # Also extract this as potential final AUC
                        potential_auc = float(auc_match)
                        if potential_auc > (best_val_auc or 0):
                            best_val_auc = potential_auc
                    except (ValueError, IndexError):
                        continue
                        
            # Fallback: try to find any reasonable AUC value in the output
            if best_val_auc is None:
                import re
                # Enhanced patterns for AUC matching
                auc_patterns = [
                    r'AUC:\s*([0-9]*\.?[0-9]+)',  # "AUC: 0.7180"
                    r'Val AUC:\s*([0-9]*\.?[0-9]+)',  # "Val AUC: 0.7180"
                    r'validauc\s+([0-9]*\.?[0-9]+)',  # Table format
                ]
                
                all_matches = []
                for line in result.stdout.split('\\n'):
                    for pattern in auc_patterns:
                        matches = re.findall(pattern, line, re.IGNORECASE)
                        for match in matches:
                            try:
                                candidate_auc = float(match)
                                if 0.5 <= candidate_auc <= 1.0:  # Reasonable AUC range
                                    all_matches.append(candidate_auc)
                            except ValueError:
                                continue
                
                # Take the highest AUC found (likely the best validation result)
                if all_matches:
                    best_val_auc = max(all_matches)
            
            if best_val_auc is not None:
                status = "🎉 TARGET!" if best_val_auc >= 0.7259 else "✅ SUCCESS"
                improvement = " ⭐ HITS TARGET!" if best_val_auc >= 0.7259 else ""
                speed_rating = "⚡ FAST" if duration < 120 else "🐌 SLOW" if duration > 300 else "⏱️ NORMAL"
                
                print(f"   🏁 GPU{gpu_id} Run {run_id}: {status} - AUC: {best_val_auc:.4f} ({duration/60:.1f}min) {speed_rating}{improvement}")
                if epoch_info:
                    print(f"   📈 Progress: {' → '.join(epoch_info[-3:])}")  # Show last 3 improvements
                
                return {
                    'run_id': run_id,
                    'gpu_id': gpu_id,
                    'params': params,
                    'auc': best_val_auc,
                    'duration_minutes': duration / 60,
                    'status': 'success',
                    'epoch_info': epoch_info
                }
            else:
                print(f"   GPU{gpu_id} Run {run_id}: ❌ FAILED - Could not parse AUC from output")
                # Print detailed debug output
                output_lines = result.stdout.split('\\n')
                
                # Look for any lines with numbers that could be AUC
                potential_auc_lines = []
                for line in output_lines[-30:]:
                    if line.strip() and ('auc' in line.lower() or 'final' in line.lower() or 'best' in line.lower() or 'validauc' in line.lower() or 'gainakt2exp' in line.lower()):
                        potential_auc_lines.append(line.strip())
                
                if potential_auc_lines:
                    print(f"   🔍 Debug - Potential AUC lines found:")
                    for i, line in enumerate(potential_auc_lines[-5:]):  # Show last 5 potential lines
                        print(f"      [{i+1}] {line}")
                else:
                    # Show the last few lines of output
                    last_lines = [line.strip() for line in output_lines[-10:] if line.strip()]
                    print(f"   🔍 Debug - Last output lines:")
                    for i, line in enumerate(last_lines[-3:]):
                        print(f"      [{i+1}] {line}")
        else:
            if result.returncode == 124:  # timeout exit code
                print(f"   ⏰ GPU{gpu_id} Run {run_id}: TIMEOUT - Exceeded 15min limit (speed optimization)")
            else:
                print(f"   ❌ GPU{gpu_id} Run {run_id}: FAILED - Return code: {result.returncode} ({duration/60:.1f}min)")
                if result.stderr:
                    error_preview = result.stderr.replace('\\n', ' ')[:100]
                    print(f"      💥 Error: {error_preview}...")
                
    except Exception as e:
        duration = time.time() - start_time
        print(f"   GPU{gpu_id} Run {run_id}: ❌ ERROR - {e}")
    
    # Return failed result
    return {
        'run_id': run_id,
        'gpu_id': gpu_id,
        'params': params,
        'auc': 0.0,
        'duration_minutes': duration / 60 if 'duration' in locals() else 0,
        'status': 'failed'
    }

def main():
    """Main function"""
    
    # Configuration - FAST MODE with 5 GPUs and 20 experiments
    num_gpus = 5
    experiments_per_gpu = 4  # 20 total experiments
    
    auto_confirm = len(sys.argv) > 1 and sys.argv[1] in ['--yes', '-y', 'auto']
    
    if not auto_confirm:
        print("🎯 Improved Multi-GPU GainAKT2Exp Parameter Sweep")
        print("This will run experiments simultaneously across 5 GPUs")
        print()
        
        response = input(f"Run {num_gpus * experiments_per_gpu} experiments across {num_gpus} GPUs? [y/N]: ")
        if response.lower() != 'y':
            print("Sweep cancelled.")
            return 0
    else:
        print("⚡ FAST MODE - Multi-GPU GainAKT2Exp Parameter Sweep")
        print(f"Running {num_gpus * experiments_per_gpu} experiments across {num_gpus} GPUs simultaneously...")
        print("🚀 SPEED OPTIMIZED: 10 epochs per experiment for rapid results!")
        print()
    
    # Initialize sweep
    sweep = ImprovedMultiGPUSweep(num_gpus=num_gpus, experiments_per_gpu=experiments_per_gpu)
    
    print("⚡ FAST MODE - Multi-GPU GainAKT2Exp Parameter Sweep")
    print("=" * 70)
    print("🔧 SPEED Configuration:")
    print(f"   GPUs: {sweep.num_gpus} (simultaneous execution)")
    print(f"   Experiments per GPU: {sweep.experiments_per_gpu}")
    print(f"   Total experiments: {sweep.total_experiments}")
    print("   Epochs per experiment: 10 (FAST MODE)")
    print("   Target AUC: >= 0.7259")
    print("   Timeout: 15 minutes per experiment (reduced for speed)")
    
    # Generate parameter combinations
    print(f"\\n📋 Generating {sweep.total_experiments} parameter combinations...")
    combinations = sweep.generate_parameter_combinations()
    
    print("\\n⚡ LAUNCHING ALL EXPERIMENTS SIMULTANEOUSLY FOR MAXIMUM SPEED!")
    start_time = time.time()
    print(f"🕐 Sweep started at: {time.strftime('%H:%M:%S')}")
    
    # Prepare arguments for each experiment 
    experiment_args = []
    gpu_assignment = 0
    
    for i, params in enumerate(combinations, 1):
        gpu_id = gpu_assignment % num_gpus
        experiment_args.append((params, i, gpu_id))
        gpu_assignment += 1
    
    # Run experiments with ProcessPoolExecutor - all at once
    results = []
    
    print("\\n🎬 EXPERIMENT TIMELINE:")
    print("=" * 80)
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        
        # Submit all jobs simultaneously (no staggering)
        futures = {}
        launch_time = time.time()
        
        print("🚀 Launching all experiments simultaneously...")
        for i, args in enumerate(experiment_args):
            params, run_id, gpu_id = args
            
            future = executor.submit(run_single_gpu_experiment, args)
            futures[future] = {
                'run_id': run_id,
                'gpu_id': gpu_id, 
                'params': params,
                'start_time': time.time()
            }
        
        print(f"✅ All {len(futures)} experiments launched! Waiting for results...")
        print("=" * 80)
        
        # Collect results as they complete with enhanced monitoring
        completed_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                
                results.append(result)
                completed_count += 1
                
                # Enhanced progress reporting
                elapsed_total = (time.time() - launch_time) / 60
                successful_results = [r for r in results if r['status'] == 'success']
                
                if successful_results:
                    best_so_far = max([r['auc'] for r in successful_results])
                    target_count = len([r for r in results if r['auc'] >= 0.7259])
                    avg_auc = sum([r['auc'] for r in successful_results]) / len(successful_results)
                    
                    progress_pct = (completed_count / sweep.total_experiments) * 100
                    success_rate = (len(successful_results) / completed_count) * 100
                    
                    print(f"\\n⚡ [{elapsed_total:.1f}min] SPEED PROGRESS UPDATE:")
                    print(f"   ✅ Completed: {completed_count}/{sweep.total_experiments} ({progress_pct:.1f}%)")
                    print(f"   🎯 Success Rate: {success_rate:.1f}% ({len(successful_results)} successful)")
                    print(f"   🏆 Best AUC: {best_so_far:.4f} | 📊 Avg AUC: {avg_auc:.4f} | 🎉 Target (≥0.7259): {target_count}")
                    
                    # Speed analysis
                    avg_duration = sum([r['duration_minutes'] for r in successful_results]) / len(successful_results)
                    print(f"   ⏱️ Avg Duration: {avg_duration:.1f}min | ⚡ ETA: {(sweep.total_experiments - completed_count) * avg_duration / num_gpus:.1f}min")
                    
                    if target_count > 0:
                        print(f"   🎉 TARGET ACHIEVED! {target_count} experiments reached ≥0.7259")
                else:
                    print(f"\\n📊 [{elapsed_total:.1f}min] {completed_count}/{sweep.total_experiments} completed (no successful results yet)")
                
                print("-" * 80)
                
            except Exception as e:
                print(f"❌ Future error: {e}")
                completed_count += 1
    
    total_time = time.time() - start_time
    
    # Analyze results
    analyze_sweep_results(results, total_time, sweep.total_experiments)
    
    return 0

def analyze_sweep_results(results, total_time_seconds, total_experiments):
    """Analyze and display improved sweep results"""
    
    print("\\n📊 IMPROVED MULTI-GPU SWEEP RESULTS")
    print("=" * 70)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if successful_results:
        # Sort by AUC
        successful_results.sort(key=lambda x: x['auc'], reverse=True)
        
        print(f"⏱️  Total sweep time: {total_time_seconds/60:.1f} minutes")
        print(f"✅ Successful runs: {len(successful_results)}/{total_experiments}")
        print(f"🏆 Best AUC: {successful_results[0]['auc']:.4f}")
        
        # Show top results
        print("\\n🥇 TOP 5 RESULTS:")
        print("-" * 85)
        print(f"{'Rank':<4} {'AUC':<8} {'GPU':<4} {'LR':<12} {'WD':<12} {'BS':<4} {'Epochs':<6} {'Enhanced':<8}")
        print("-" * 85)
        
        for i, result in enumerate(successful_results[:5], 1):
            params = result['params']
            print(f"{i:<4} {result['auc']:<8.4f} {result['gpu_id']:<4} "
                  f"{params['learning_rate']:<12.8f} {params['weight_decay']:<12.8f} "
                  f"{params['batch_size']:<4} {params['num_epochs']:<6} "
                  f"{params['enhanced_constraints']:<8}")
        
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'improved_multi_gpu_sweep_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': str(datetime.now()),
                'total_experiments': total_experiments,
                'successful_experiments': len(successful_results),
                'total_time_minutes': total_time_seconds / 60,
                'best_auc': successful_results[0]['auc'] if successful_results else 0,
                'target_achieved_count': len(target_runs) if 'target_runs' in locals() else 0,
                'results': results
            }, f, indent=2)
        
        print(f"\\n💾 Results saved to: {results_file}")
        
    else:
        print("❌ No successful experiments completed")
        print(f"⏱️  Total time: {total_time_seconds/60:.1f} minutes")

if __name__ == "__main__":
    sys.exit(main())