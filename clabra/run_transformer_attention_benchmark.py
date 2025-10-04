#!/usr/bin/env python3
"""
Transformer & Attention-Based Knowledge Tracing Models Benchmark - Sequential 6-GPU Training

This script benchmarks transformer/attention models by training each model sequentially 
using all 6 Tesla V100 GPUs, with detailed epoch-by-epoch progress tracking and metrics.

Usage:
    cd /workspaces/pykt-toolkit/assistant
    source /home/vscode/.pykt-env/bin/activate
    python run_transformer_attention_benchmark.py
"""

import os
import sys
import json
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path

class SequentialMultiGPUBenchmark:
    def __init__(self, num_gpus=6):
        self.num_gpus = num_gpus
        self.models = self.get_transformer_attention_models()
        self.results = {}
        self.start_time = None
        self.log_file = None
        self.json_file = None
        self.current_model_idx = 0
        
    def get_transformer_attention_models(self):
        """Get only transformer and attention-based models from taxonomy"""
        transformer_attention_models = [
            # Foundation Era (2017-2021) - Direct attention adaptations
            {'name': 'sakt', 'category': 'Foundation', 'year': 2019, 'mechanism': 'Self-Attention'},
            {'name': 'akt', 'category': 'Foundation', 'year': 2020, 'mechanism': 'Context-Aware Attention + IRT'},
            {'name': 'saint', 'category': 'Foundation', 'year': 2021, 'mechanism': 'Encoder-Decoder Transformer'},
            {'name': 'atkt', 'category': 'Foundation', 'year': 2021, 'mechanism': 'Adversarial Attention + LSTM'},
            {'name': 'dkvmn', 'category': 'Foundation', 'year': 2017, 'mechanism': 'Key-Value Memory Attention'},
            {'name': 'skvmn', 'category': 'Foundation', 'year': 2019, 'mechanism': 'Sequential Memory Attention'},
            
            # Specialized Era (2022-2023) - Engineering solutions
            {'name': 'simplekt', 'category': 'Specialized', 'year': 2023, 'mechanism': 'Simplified Attention + Rasch'},
            {'name': 'sparsekt', 'category': 'Specialized', 'year': 2023, 'mechanism': 'K-Sparse Attention Selection'},
            {'name': 'dtransformer', 'category': 'Specialized', 'year': 2023, 'mechanism': 'Temporal & Cumulative Attention'},
            {'name': 'qikt', 'category': 'Specialized', 'year': 2023, 'mechanism': 'Question-Centric Attention'},
            {'name': 'folibikt', 'category': 'Specialized', 'year': 2023, 'mechanism': 'Forgetting-Aware Attention'},
            
            # Theory-Driven Era (2024-2025) - Theoretical paradigms
            {'name': 'stablekt', 'category': 'Theory-Driven', 'year': 2024, 'mechanism': 'Length Generalization Attention'},
            {'name': 'fluckt', 'category': 'Theory-Driven', 'year': 2025, 'mechanism': 'Cognitive Fluctuation Attention'},
            {'name': 'lefokt_akt', 'category': 'Theory-Driven', 'year': 2025, 'mechanism': 'Relative Forgetting Attention'},
            {'name': 'cskt', 'category': 'Theory-Driven', 'year': 2025, 'mechanism': 'Cold-Start Cone Attention'},
        ]
        
        # Filter to only include models that exist in the examples directory
        examples_dir = Path("/workspaces/pykt-toolkit/examples")
        available_models = []
        
        for model in transformer_attention_models:
            script_name = f"wandb_{model['name']}_train.py"
            script_path = examples_dir / script_name
            if script_path.exists():
                model['script'] = script_name
                model['path'] = str(script_path)
                available_models.append(model)
            else:
                print(f"⚠️  Script not found: {script_name}")
        
        return available_models
    
    def log_message(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')
                f.flush()
    
    def setup_logging(self):
        """Setup logging files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f"sequential_6gpu_transformer_benchmark_{timestamp}.log"
        self.json_file = f"sequential_6gpu_transformer_rankings_{timestamp}.json"
        
        self.log_message("🎯 Initializing Sequential 6-GPU Transformer & Attention Benchmark...")
        self.log_message(f"🖥️  Using all 6 Tesla V100 GPUs sequentially for each model")
        self.log_message(f"📊 Found {len(self.models)} transformer/attention models:")
        
        # Group by category for better organization
        categories = {}
        for model in self.models:
            category = model['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(model)
        
        for category, models in categories.items():
            self.log_message(f"\n📊 {category} Era ({len(models)} models):")
            for model in models:
                self.log_message(f"   - {model['name']} ({model['year']}) - {model['mechanism']}")
    
    def display_progress_header(self, model, model_num, total_models):
        """Display progress header for current model"""
        progress_pct = (model_num / total_models) * 100
        
        self.log_message("\n" + "="*100)
        self.log_message(f"🚀 TRAINING MODEL {model_num}/{total_models} ({progress_pct:.1f}%)")
        self.log_message(f"📋 Model: {model['name']} ({model['year']}) - {model['mechanism']}")
        self.log_message(f"🖥️  Using 6 Tesla V100 GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5")
        self.log_message(f"📂 Script: {model['script']}")
        self.log_message("="*100)
    
    def run_model_with_6gpus(self, model, model_num, total_models):
        """Run a single model using all 6 GPUs with detailed epoch progress tracking"""
        self.display_progress_header(model, model_num, total_models)
        
        start_time = time.time()
        
        # Temporarily modify the config file to use 10 epochs
        config_path = "/workspaces/pykt-toolkit/configs/kt_config.json"
        original_config = None
        
        try:
            # Backup and modify config file
            with open(config_path, 'r') as f:
                original_config = json.load(f)
            
            modified_config = original_config.copy()
            modified_config["train_config"]["num_epochs"] = 10
            
            with open(config_path, 'w') as f:
                json.dump(modified_config, f, indent=2)
            
            self.log_message(f"🔧 Temporarily set num_epochs=10 in kt_config.json")
            
            # Set environment to use all 6 GPUs
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
            
            # Run the training script with detailed monitoring
            cmd = [
                'python', model['script'],
                '--dataset_name=assist2015',
                '--use_wandb=0',
                '--seed=42',
                '--fold=0'
            ]
            
            self.log_message(f"🏃 Starting training: {' '.join(cmd)}")
            
            # Use Popen for real-time output monitoring
            process = subprocess.Popen(
                cmd,
                cwd='/workspaces/pykt-toolkit/examples',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,  # Line buffering for real-time output
                universal_newlines=True
            )
            
            # Monitor output for detailed epoch progress
            epoch_metrics = []
            full_output = ""
            current_epoch = 0
            
            self.log_message("📈 Training Progress (Real-time Epoch Updates):")
            self.log_message("-" * 80)
            
            # Real-time output processing
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    full_output += output
                    line = output.strip()
                    
                    # Parse detailed epoch information
                    if 'Epoch:' in line and ('validauc:' in line or 'testauc:' in line):
                        try:
                            epoch_data = self.parse_epoch_line(line)
                            if epoch_data:
                                current_epoch = epoch_data.get('epoch', current_epoch + 1)
                                epoch_metrics.append(epoch_data)
                                
                                # Display beautiful epoch progress
                                self.display_epoch_progress(model['name'], epoch_data, epoch_metrics)
                                
                        except Exception as e:
                            # Continue if parsing fails
                            pass
                    
                    # Also look for simple validation lines
                    elif 'validauc:' in line.lower() and 'Epoch' not in line:
                        try:
                            # Extract AUC from simpler formats
                            parts = line.split(',')
                            for part in parts:
                                if 'validauc:' in part.lower():
                                    auc_value = float(part.split(':')[1].strip())
                                    current_epoch += 1
                                    simple_data = {'epoch': current_epoch, 'validauc': auc_value}
                                    epoch_metrics.append(simple_data)
                                    
                                    self.log_message(f"   📊 Epoch {current_epoch:2d}: Valid AUC = {auc_value:.4f}")
                                    break
                        except (ValueError, IndexError):
                            continue
            
            # Wait for process completion
            rc = process.poll()
            stderr_output = process.stderr.read()
            
            duration = time.time() - start_time
            
            self.log_message("-" * 80)
            
            if rc == 0:
                # Extract final metrics from complete output
                final_metrics = self.extract_metrics_from_output(full_output)
                
                # Combine epoch progression with final results
                final_auc = final_metrics.get('validauc', 0.0)
                if epoch_metrics and final_auc == 0.0:
                    final_auc = epoch_metrics[-1].get('validauc', 0.0)
                
                self.results[model['name']] = {
                    'model': model['name'],
                    'category': model['category'],
                    'year': model['year'],
                    'mechanism': model['mechanism'],
                    'script': model['script'],
                    'duration_minutes': round(duration / 60, 2),
                    'status': 'success',
                    'validauc': final_auc,
                    'validacc': final_metrics.get('validacc', 0.0),
                    'best_epoch': final_metrics.get('best_epoch', len(epoch_metrics)),
                    'epoch_metrics': epoch_metrics,  # Detailed epoch-by-epoch data
                    'epochs_completed': len(epoch_metrics),
                    'testauc': final_metrics.get('testauc', -1),
                    'testacc': final_metrics.get('testacc', -1),
                    'final_train_loss': epoch_metrics[-1].get('train_loss', 'N/A') if epoch_metrics else 'N/A',
                    'stdout_preview': full_output[-500:] if full_output else "",
                    'stderr_preview': stderr_output[-500:] if stderr_output else ""
                }
                
                # Display comprehensive training summary
                self.display_training_summary(model['name'], epoch_metrics, duration, final_auc)
                
            else:
                self.results[model['name']] = {
                    'model': model['name'],
                    'category': model['category'],
                    'year': model['year'],
                    'mechanism': model['mechanism'],
                    'script': model['script'],
                    'duration_minutes': round(duration / 60, 2),
                    'status': 'failed',
                    'error': f"Process failed with return code {rc}",
                    'stdout_preview': full_output[-500:] if full_output else "",
                    'stderr_preview': stderr_output[-500:] if stderr_output else ""
                }
                
                self.log_message(f"❌ {model['name']} FAILED with return code {rc}")
                if stderr_output:
                    self.log_message(f"   💥 Error: {stderr_output[-300:]}")
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.results[model['name']] = {
                'model': model['name'],
                'category': model['category'], 
                'year': model['year'],
                'mechanism': model['mechanism'],
                'script': model['script'],
                'duration_minutes': round(duration / 60, 2),
                'status': 'timeout',
                'error': f"Training timed out after {duration:.0f} seconds"
            }
            self.log_message(f"⏰ {model['name']} TIMED OUT after {duration/60:.1f} minutes")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results[model['name']] = {
                'model': model['name'],
                'category': model['category'],
                'year': model['year'],
                'mechanism': model['mechanism'],
                'script': model['script'],
                'duration_minutes': round(duration / 60, 2),
                'status': 'error',
                'error': str(e)
            }
            self.log_message(f"💥 {model['name']} ERROR: {str(e)}")
        
        finally:
            # Restore original config file
            if original_config is not None:
                try:
                    with open(config_path, 'w') as f:
                        json.dump(original_config, f, indent=2)
                    self.log_message(f"🔧 Restored original kt_config.json")
                except Exception as e:
                    self.log_message(f"⚠️  Warning: Could not restore config file: {str(e)}")
    
    def display_epoch_progress(self, model_name, epoch_data, all_epochs):
        """Display detailed progress for a single epoch"""
        epoch_num = epoch_data.get('epoch', len(all_epochs))
        valid_auc = epoch_data.get('validauc', 0)
        valid_acc = epoch_data.get('validacc', 0)
        train_loss = epoch_data.get('train_loss', 0)
        
        # Calculate improvement from previous epoch
        improvement = ""
        if len(all_epochs) > 1:
            prev_auc = all_epochs[-2].get('validauc', 0)
            delta = valid_auc - prev_auc
            improvement = f" (Δ{delta:+.4f})"
        
        # Format epoch line
        self.log_message(
            f"   📊 Epoch {epoch_num:2d}: "
            f"Valid AUC = {valid_auc:.4f}{improvement} | "
            f"Valid ACC = {valid_acc:.4f} | "
            f"Train Loss = {train_loss:.4f}"
        )
        
        # Show test results if available
        if epoch_data.get('testauc', -1) != -1:
            test_auc = epoch_data.get('testauc')
            test_acc = epoch_data.get('testacc', 0)
            self.log_message(f"        🧪 Test AUC = {test_auc:.4f} | Test ACC = {test_acc:.4f}")
    
    def display_training_summary(self, model_name, epoch_metrics, duration, final_auc):
        """Display comprehensive training summary"""
        if not epoch_metrics:
            self.log_message(f"✅ {model_name} completed - Final AUC: {final_auc:.4f}")
            return
            
        start_auc = epoch_metrics[0].get('validauc', 0)
        end_auc = epoch_metrics[-1].get('validauc', 0)
        total_improvement = end_auc - start_auc
        best_epoch_data = max(epoch_metrics, key=lambda x: x.get('validauc', 0))
        best_auc = best_epoch_data.get('validauc', 0)
        best_epoch_num = best_epoch_data.get('epoch', 0)
        
        # Create learning curve visualization
        auc_progression = [f"{ep.get('validauc', 0):.3f}" for ep in epoch_metrics[:10]]  # Show first 10 epochs
        progression_str = " → ".join(auc_progression)
        if len(epoch_metrics) > 10:
            progression_str += f" → ... → {end_auc:.3f}"
        
        self.log_message("✅ TRAINING COMPLETED SUCCESSFULLY!")
        self.log_message("📊 Training Summary:")
        self.log_message(f"   🎯 Final AUC:        {end_auc:.4f}")
        self.log_message(f"   📈 Total Improvement: {total_improvement:+.4f}")
        self.log_message(f"   🥇 Best Epoch:       {best_epoch_num} (AUC: {best_auc:.4f})")
        self.log_message(f"   📏 Total Epochs:     {len(epoch_metrics)}")
        self.log_message(f"   ⏱️  Duration:         {duration/60:.2f} minutes")
        self.log_message(f"   🔄 Learning Curve:   {progression_str}")
        
        # Performance classification
        if end_auc >= 0.75:
            performance = "🏆 EXCELLENT"
        elif end_auc >= 0.70:
            performance = "🥇 GOOD"
        elif end_auc >= 0.65:
            performance = "🥈 AVERAGE"
        else:
            performance = "🥉 NEEDS IMPROVEMENT"
            
        self.log_message(f"   🎖️  Performance:     {performance}")

    def parse_epoch_line(self, line):
        """Parse detailed epoch information from training output"""
        epoch_data = {}
        
        try:
            # Handle different output formats
            if 'Epoch:' in line:
                parts = line.split(',')
                
                for part in parts:
                    part = part.strip()
                    
                    if part.startswith('Epoch:'):
                        epoch_data['epoch'] = int(part.split(':')[1].strip())
                    elif 'validauc:' in part:
                        epoch_data['validauc'] = float(part.split(':')[1].strip())
                    elif 'validacc:' in part:
                        epoch_data['validacc'] = float(part.split(':')[1].strip())
                    elif 'testauc:' in part:
                        epoch_data['testauc'] = float(part.split(':')[1].strip())
                    elif 'testacc:' in part:
                        epoch_data['testacc'] = float(part.split(':')[1].strip())
                    elif 'train loss:' in part:
                        epoch_data['train_loss'] = float(part.split(':')[1].strip())
                    elif 'best epoch:' in part:
                        epoch_data['best_epoch'] = int(part.split(':')[1].strip())
                    elif 'best auc:' in part:
                        epoch_data['best_auc'] = float(part.split(':')[1].strip())
                
        except (ValueError, IndexError, AttributeError):
            return None
            
        return epoch_data if epoch_data else None

    def extract_metrics_from_output(self, output):
        """Extract validation AUC, accuracy, and best epoch from training output"""
        metrics = {
            'validauc': None,
            'validacc': None,
            'best_epoch': None,
            'testauc': None,
            'testacc': None
        }
        
        lines = output.split('\n')
        for line in lines:
            if '\t' in line and len(line.split('\t')) >= 9:
                parts = line.split('\t')
                try:
                    if len(parts) >= 10:
                        metrics['testauc'] = float(parts[3]) if parts[3] != '-1' else -1
                        metrics['testacc'] = float(parts[4]) if parts[4] != '-1' else -1
                        metrics['validauc'] = float(parts[7])
                        metrics['validacc'] = float(parts[8])
                        metrics['best_epoch'] = int(parts[9])
                        break
                except (ValueError, IndexError):
                    continue
        
        return metrics
    
    def run_benchmark(self):
        """Run the complete sequential 6-GPU benchmark"""
        self.setup_logging()
        self.start_time = time.time()
        
        self.log_message(f"🎯 Starting Sequential 6-GPU Benchmark")
        self.log_message(f"📋 Configuration: ASSIST2015 dataset, 6 GPUs per model, seed=42")
        self.log_message(f"🔢 Total Models: {len(self.models)}")
        
        # Run models sequentially
        successful_models = 0
        failed_models = 0
        
        for i, model in enumerate(self.models, 1):
            self.run_model_with_6gpus(model, i, len(self.models))
            
            if model['name'] in self.results:
                if self.results[model['name']]['status'] == 'success':
                    successful_models += 1
                else:
                    failed_models += 1
        
        # Generate final report
        self.generate_final_report(successful_models, failed_models)
        
    def generate_final_report(self, successful_models, failed_models):
        """Generate comprehensive final benchmark report"""
        total_time = time.time() - self.start_time
        
        self.log_message(f"\n🏁 Sequential 6-GPU Transformer Benchmark COMPLETED!")
        self.log_message(f"✅ Successful models: {successful_models}")
        self.log_message(f"❌ Failed models: {failed_models}")
        
        # Sort successful models by validation AUC
        successful_results = [
            result for result in self.results.values() 
            if result['status'] == 'success' and 'validauc' in result
        ]
        successful_results.sort(key=lambda x: x.get('validauc', 0), reverse=True)
        
        if successful_results:
            self.log_message("\n" + "="*120)
            self.log_message("🏆 TRANSFORMER & ATTENTION MODELS FINAL RANKINGS - ASSIST2015 DATASET")
            self.log_message("="*120)
            self.log_message(f"{'Rank':<4} {'Model':<15} {'Category':<12} {'Year':<6} {'Final AUC':<10} {'Improvement':<12} {'Epochs':<7} {'Duration':<10}")
            self.log_message("-"*120)
            
            for i, result in enumerate(successful_results, 1):
                epoch_metrics = result.get('epoch_metrics', [])
                improvement = "N/A"
                if len(epoch_metrics) > 1:
                    start_auc = epoch_metrics[0].get('validauc', 0)
                    end_auc = epoch_metrics[-1].get('validauc', 0)
                    improvement = f"+{end_auc - start_auc:.4f}"
                
                self.log_message(
                    f"{i:<4} {result['model']:<15} {result['category']:<12} "
                    f"{result['year']:<6} {result.get('validauc', 0):<10.4f} {improvement:<12} "
                    f"{len(epoch_metrics):<7} {result.get('duration_minutes', 0):<10.2f}min"
                )
            
            self.log_message("-"*120)
            
            # Performance statistics
            aucs = [r.get('validauc', 0) for r in successful_results if r.get('validauc', 0) > 0]
            if aucs:
                self.log_message("📊 Performance Statistics:")
                self.log_message(f"   🥇 Best AUC:       {max(aucs):.4f} ({successful_results[0]['model']})")
                self.log_message(f"   🥉 Worst AUC:      {min(aucs):.4f}")
                self.log_message(f"   📈 Average AUC:    {sum(aucs)/len(aucs):.4f}")
                self.log_message(f"   📊 AUC Range:      {max(aucs) - min(aucs):.4f}")
                self.log_message(f"   🎯 Median AUC:     {sorted(aucs)[len(aucs)//2]:.4f}")
            
            # Category analysis
            self.log_message("\n📊 Performance by Era:")
            categories = {}
            for result in successful_results:
                cat = result['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result.get('validauc', 0))
            
            for category, aucs in categories.items():
                if aucs and all(auc > 0 for auc in aucs):
                    avg_auc = sum(aucs) / len(aucs)
                    best_auc = max(aucs)
                    self.log_message(f"   {category:<15}: Avg {avg_auc:.4f} | Best {best_auc:.4f} | Models: {len(aucs)}")
        
        # Report failed models
        failed_results = [r for r in self.results.values() if r['status'] != 'success']
        if failed_results:
            self.log_message(f"\n❌ Failed Models ({len(failed_results)}):")
            for result in failed_results:
                error_msg = result.get('error', 'Unknown error')[:60] + "..."
                self.log_message(f"   - {result['model']:<15}: {error_msg}")
        
        self.log_message(f"\n⏱️  Total Benchmark Time: {total_time/60:.1f} minutes")
        self.log_message(f"🖥️  GPU Configuration: 6 Tesla V100s per model (sequential training)")
        self.log_message("="*120)
        
        # Save detailed results to JSON
        with open(self.json_file, 'w') as f:
            json.dump({
                'benchmark_info': {
                    'type': 'sequential_6gpu_transformer_attention_models',
                    'dataset': 'assist2015',
                    'gpus_per_model': 6,
                    'training_mode': 'sequential',
                    'total_models': len(self.models),
                    'successful_models': successful_models,
                    'failed_models': failed_models,
                    'total_time_minutes': round(total_time / 60, 2),
                    'timestamp': datetime.now().isoformat()
                },
                'results': self.results
            }, f, indent=2)
        
        self.log_message(f"💾 Detailed results saved to: {os.path.abspath(self.json_file)}")
        self.log_message(f"📝 Log file saved to: {os.path.abspath(self.log_file)}")

def main():
    """Main function to run the sequential 6-GPU transformer benchmark"""
    benchmark = SequentialMultiGPUBenchmark(num_gpus=6)
    
    try:
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Benchmark failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)