#!/usr/bin/env python3
"""
GPU Multi-Model Knowledge Tracing Benchmark Script

This script runs all available knowledge tracing models from the pykt-toolkit
on 4 GPUs simultaneously, tests them on ASSIST2015 dataset for 5 epochs,
and generates a performance ranking table based on validation AUC scores.

Usage:
    cd /workspaces/pykt-toolkit/assistant
    source /home/vscode/.pykt-env/bin/activate
    python run_all_gpus.py

Features:
- Multi-GPU parallel execution (4 GPUs)
- Automatic model discovery from examples/wandb_*_train.py
- Performance monitoring and logging
- AUC-based ranking table generation
- Comprehensive error handling and timeouts
- Real-time progress tracking
"""

import os
import sys
import subprocess
import glob
import time
import threading
import queue
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

class ModelBenchmark:
    def __init__(self):
        self.results = {}
        self.failed_models = {}
        self.start_time = datetime.now()
        self.log_file = f"benchmark_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        self.results_file = f"model_rankings_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create assistant directory if it doesn't exist
        os.makedirs('/workspaces/pykt-toolkit/assistant', exist_ok=True)
        
        # Set up logging
        self.log_path = f"/workspaces/pykt-toolkit/assistant/{self.log_file}"
        
    def log_message(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_path, 'a') as f:
            f.write(log_entry + '\n')
    
    def discover_models(self):
        """Discover all wandb training scripts and extract model names"""
        examples_dir = "/workspaces/pykt-toolkit/examples"
        train_scripts = glob.glob(f"{examples_dir}/wandb_*_train.py")
        
        models = []
        for script in train_scripts:
            # Extract model name from filename (wandb_MODEL_train.py -> MODEL)
            script_name = os.path.basename(script)
            if script_name.startswith('wandb_') and script_name.endswith('_train.py'):
                model_name = script_name[6:-9]  # Remove 'wandb_' prefix and '_train.py' suffix
                models.append({
                    'name': model_name,
                    'script': script_name,
                    'path': script
                })
        
        # Sort models alphabetically
        models.sort(key=lambda x: x['name'])
        
        self.log_message(f"Discovered {len(models)} models:")
        for model in models:
            self.log_message(f"  - {model['name']} ({model['script']})")
        
        return models
    
    def extract_metrics_from_output(self, output):
        """Extract validation AUC, accuracy, and best epoch from training output"""
        metrics = {
            'validauc': None,
            'validacc': None,
            'best_epoch': None,
            'testauc': None,
            'testacc': None
        }
        
        # Look for final results line
        lines = output.split('\n')
        for line in lines:
            # Pattern: "0	modelname	embtype	testauc	testacc	window_testauc	window_testacc	validauc	validacc	best_epoch"
            if '\t' in line and 'validauc' not in line:  # Skip header line
                parts = line.strip().split('\t')
                if len(parts) >= 10:
                    try:
                        metrics['testauc'] = float(parts[3]) if parts[3] != '-1' else None
                        metrics['testacc'] = float(parts[4]) if parts[4] != '-1' else None
                        metrics['validauc'] = float(parts[7])
                        metrics['validacc'] = float(parts[8])
                        metrics['best_epoch'] = int(parts[9])
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Fallback: look for individual metric lines
        if metrics['validauc'] is None:
            # Look for validation AUC patterns
            auc_patterns = [
                r'validauc[:\s]+([0-9.]+)',
                r'validation.*auc[:\s]+([0-9.]+)',
                r'valid.*auc[:\s]+([0-9.]+)',
                r'Validation AUC[:\s]+([0-9.]+)'
            ]
            
            for pattern in auc_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    try:
                        metrics['validauc'] = float(matches[-1])  # Take the last (best) value
                        break
                    except ValueError:
                        continue
        
        return metrics
    
    def run_single_model(self, model, gpu_id, timeout=1800):
        """Run a single model on specified GPU with timeout (30 minutes default)"""
        model_name = model['name']
        script_name = model['script']
        
        self.log_message(f"🚀 Starting {model_name} on GPU {gpu_id}")
        
        # Create a temporary config modification for shorter training
        config_backup_cmd = "cp /workspaces/pykt-toolkit/configs/kt_config.json /tmp/kt_config_backup.json"
        config_restore_cmd = "cp /tmp/kt_config_backup.json /workspaces/pykt-toolkit/configs/kt_config.json"
        
        # Modify config to use 5 epochs for quick testing
        modify_config_cmd = '''python -c "
import json
with open('/workspaces/pykt-toolkit/configs/kt_config.json', 'r') as f:
    config = json.load(f)
config['train_config']['num_epochs'] = 5
with open('/workspaces/pykt-toolkit/configs/kt_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"'''
        
        # Prepare command with config modification
        cmd = [
            'bash', '-c',
            f"cd /workspaces/pykt-toolkit/examples && "
            f"source /home/vscode/.pykt-env/bin/activate && "
            f"{config_backup_cmd} && "
            f"{modify_config_cmd} && "
            f"CUDA_VISIBLE_DEVICES={gpu_id} python {script_name} "
            f"--dataset_name=assist2015 "
            f"--use_wandb=0 "
            f"--seed=42 && "
            f"{config_restore_cmd}"
        ]
        
        start_time = time.time()
        
        try:
            # Run the command with timeout
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()
            )
            
            duration = time.time() - start_time
            
            # Always restore config file even if training failed
            try:
                subprocess.run(["bash", "-c", config_restore_cmd], capture_output=True)
            except:
                pass
            
            if process.returncode == 0:
                # Extract metrics from output
                metrics = self.extract_metrics_from_output(process.stdout)
                
                if metrics['validauc'] is not None:
                    result = {
                        'model': model_name,
                        'script': script_name,
                        'gpu_id': gpu_id,
                        'duration_minutes': round(duration / 60, 2),
                        'status': 'success',
                        'metrics': metrics,
                        'stdout_lines': len(process.stdout.split('\n')),
                        'stderr_lines': len(process.stderr.split('\n')) if process.stderr else 0
                    }
                    
                    self.log_message(f"✅ {model_name} completed successfully on GPU {gpu_id}")
                    self.log_message(f"   📊 Validation AUC: {metrics['validauc']:.4f}")
                    self.log_message(f"   ⏱️  Duration: {result['duration_minutes']} minutes")
                    
                    return result
                else:
                    self.log_message(f"❌ {model_name} failed - could not extract validation AUC")
                    return {
                        'model': model_name,
                        'script': script_name,
                        'gpu_id': gpu_id,
                        'duration_minutes': round(duration / 60, 2),
                        'status': 'failed',
                        'error': 'Could not extract validation AUC from output',
                        'stdout_preview': process.stdout[:500] if process.stdout else '',
                        'stderr_preview': process.stderr[:500] if process.stderr else ''
                    }
            else:
                self.log_message(f"❌ {model_name} failed with return code {process.returncode}")
                return {
                    'model': model_name,
                    'script': script_name,
                    'gpu_id': gpu_id,
                    'duration_minutes': round(duration / 60, 2),
                    'status': 'failed',
                    'error': f'Process failed with return code {process.returncode}',
                    'stdout_preview': process.stdout[:500] if process.stdout else '',
                    'stderr_preview': process.stderr[:500] if process.stderr else ''
                }
                
        except subprocess.TimeoutExpired:
            # Restore config file on timeout
            try:
                subprocess.run(["bash", "-c", config_restore_cmd], capture_output=True)
            except:
                pass
                
            duration = time.time() - start_time
            self.log_message(f"⏰ {model_name} timed out after {timeout/60:.1f} minutes on GPU {gpu_id}")
            return {
                'model': model_name,
                'script': script_name,
                'gpu_id': gpu_id,
                'duration_minutes': round(duration / 60, 2),
                'status': 'timeout',
                'error': f'Training timed out after {timeout} seconds'
            }
            
        except Exception as e:
            # Restore config file on error
            try:
                subprocess.run(["bash", "-c", config_restore_cmd], capture_output=True)
            except:
                pass
                
            duration = time.time() - start_time
            self.log_message(f"💥 {model_name} crashed with exception: {str(e)}")
            return {
                'model': model_name,
                'script': script_name,
                'gpu_id': gpu_id,
                'duration_minutes': round(duration / 60, 2),
                'status': 'error',
                'error': str(e)
            }
    
    def run_benchmark(self):
        """Run benchmark on all models using 4 GPUs"""
        models = self.discover_models()
        
        if not models:
            self.log_message("❌ No models found to benchmark!")
            return
        
        self.log_message(f"🎯 Starting benchmark of {len(models)} models on 4 GPUs")
        self.log_message(f"📋 Configuration: ASSIST2015 dataset, 5 epochs, seed=42")
        self.log_message(f"⚡ GPUs: Tesla V100-SXM2-32GB (4 GPUs)")
        
        # Use ThreadPoolExecutor to manage GPU assignments
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            # Submit all models with round-robin GPU assignment
            for i, model in enumerate(models):
                gpu_id = i % 4  # Round-robin GPU assignment (0, 1, 2, 3)
                future = executor.submit(self.run_single_model, model, gpu_id)
                futures[future] = model
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(futures):
                model = futures[future]
                result = future.result()
                
                completed_count += 1
                
                if result['status'] == 'success':
                    self.results[model['name']] = result
                else:
                    self.failed_models[model['name']] = result
                
                # Progress update
                progress = (completed_count / len(models)) * 100
                self.log_message(f"📈 Progress: {completed_count}/{len(models)} ({progress:.1f}%)")
        
        self.log_message(f"🏁 Benchmark completed!")
        self.log_message(f"✅ Successful models: {len(self.results)}")
        self.log_message(f"❌ Failed models: {len(self.failed_models)}")
    
    def generate_ranking_table(self):
        """Generate and display ranking table based on validation AUC"""
        if not self.results:
            self.log_message("❌ No successful results to rank!")
            return
        
        # Sort results by validation AUC (descending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['metrics']['validauc'],
            reverse=True
        )
        
        # Generate table
        self.log_message("\n" + "="*100)
        self.log_message("🏆 KNOWLEDGE TRACING MODELS RANKING - ASSIST2015 DATASET")
        self.log_message("="*100)
        
        # Table header
        header = f"{'Rank':<4} {'Model':<20} {'Valid AUC':<10} {'Valid ACC':<10} {'Best Epoch':<10} {'Duration (min)':<12} {'GPU':<4}"
        self.log_message(header)
        self.log_message("-" * 100)
        
        # Table rows
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            metrics = result['metrics']
            row = (f"{rank:<4} "
                   f"{model_name:<20} "
                   f"{metrics['validauc']:.4f}{'':>4} "
                   f"{metrics['validacc']:.4f}{'':>4} "
                   f"{metrics['best_epoch']:<10} "
                   f"{result['duration_minutes']:<12} "
                   f"{result['gpu_id']:<4}")
            self.log_message(row)
        
        self.log_message("-" * 100)
        
        # Statistics
        aucs = [result['metrics']['validauc'] for result in self.results.values()]
        self.log_message(f"📊 Statistics:")
        self.log_message(f"   🥇 Best AUC: {max(aucs):.4f} ({sorted_results[0][0]})")
        self.log_message(f"   🥉 Worst AUC: {min(aucs):.4f} ({sorted_results[-1][0]})")
        self.log_message(f"   📈 Average AUC: {sum(aucs)/len(aucs):.4f}")
        self.log_message(f"   📊 AUC Range: {max(aucs) - min(aucs):.4f}")
        
        # Failed models summary
        if self.failed_models:
            self.log_message(f"\n❌ Failed Models ({len(self.failed_models)}):")
            for model_name, result in self.failed_models.items():
                self.log_message(f"   - {model_name}: {result['error'][:60]}...")
        
        # Total time
        total_duration = time.time() - self.start_time.timestamp()
        self.log_message(f"\n⏱️  Total Benchmark Time: {total_duration/60:.1f} minutes")
        self.log_message("="*100)
    
    def save_results(self):
        """Save detailed results to JSON file"""
        results_data = {
            'benchmark_info': {
                'start_time': self.start_time.isoformat(),
                'dataset': 'assist2015',
                'epochs': 5,
                'gpus_used': 4,
                'total_models_tested': len(self.results) + len(self.failed_models),
                'successful_models': len(self.results),
                'failed_models': len(self.failed_models)
            },
            'successful_results': self.results,
            'failed_results': self.failed_models
        }
        
        results_path = f"/workspaces/pykt-toolkit/assistant/{self.results_file}"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.log_message(f"💾 Detailed results saved to: {results_path}")
        self.log_message(f"📝 Log file saved to: {self.log_path}")

def main():
    """Main execution function"""
    print("🚀 GPU Multi-Model Knowledge Tracing Benchmark")
    print("=" * 60)
    
    # Check if we're in the right environment
    if not os.path.exists("/home/vscode/.pykt-env/bin/activate"):
        print("❌ PyKT environment not found at /home/vscode/.pykt-env/")
        print("   Please ensure the .pykt-env environment is properly set up.")
        sys.exit(1)
    
    # Check if we're in the right directory structure
    if not os.path.exists("/workspaces/pykt-toolkit/examples"):
        print("❌ PyKT examples directory not found!")
        print("   Please run this script from the pykt-toolkit workspace.")
        sys.exit(1)
    
    # Initialize and run benchmark
    benchmark = ModelBenchmark()
    
    try:
        benchmark.log_message("🎯 Initializing GPU Multi-Model Benchmark...")
        benchmark.run_benchmark()
        benchmark.generate_ranking_table()
        benchmark.save_results()
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📊 Check {benchmark.log_path} for detailed logs")
        print(f"💾 Check {benchmark.results_file} for JSON results")
        
    except KeyboardInterrupt:
        benchmark.log_message("\n⚠️  Benchmark interrupted by user")
        print("Benchmark stopped.")
    except Exception as e:
        benchmark.log_message(f"💥 Benchmark failed with error: {str(e)}")
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()