#!/usr/bin/env python3
"""
GainAKT2 Parameter Optimization - Offline Mode
Advanced parameter sweep with local result tracking and analysis
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import itertools

class GainAKT2Optimizer:
    def __init__(self):
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_parameter_combinations(self):
        """Generate optimized parameter combinations based on benchmark insights"""
        
        # Focused parameter ranges from benchmark analysis
        param_grid = {
            'd_model': [256, 384],  # Larger models performed better
            'learning_rate': [1e-4, 2e-4, 3e-4, 5e-4],  # Around optimal 2e-4
            'dropout': [0.15, 0.2, 0.25],  # Fine-tune around 0.2
            'num_encoder_blocks': [3, 4, 5],  # Deeper architectures
            'd_ff': [512, 768, 1024],  # Balanced dimensions
            'n_heads': [8, 12],  # Multi-head options
        }
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_grid.values()):
            combo = dict(zip(param_grid.keys(), values))
            combinations.append(combo)
        
        # Limit to manageable size and prioritize promising combinations
        prioritized = self.prioritize_combinations(combinations)
        return prioritized[:20]  # Top 20 combinations
    
    def prioritize_combinations(self, combinations):
        """Prioritize combinations based on benchmark insights"""
        def score_combination(combo):
            score = 0
            
            # Prefer larger models (performed better in benchmark)
            if combo['d_model'] >= 256:
                score += 3
                
            # Prefer learning rates around 2e-4 (optimal in benchmark)
            if 1.5e-4 <= combo['learning_rate'] <= 3e-4:
                score += 4
            elif combo['learning_rate'] == 2e-4:
                score += 2
                
            # Prefer dropout around 0.2
            if combo['dropout'] == 0.2:
                score += 3
            elif 0.15 <= combo['dropout'] <= 0.25:
                score += 1
                
            # Prefer 4 encoder blocks (optimal in benchmark)
            if combo['num_encoder_blocks'] == 4:
                score += 3
            elif combo['num_encoder_blocks'] >= 3:
                score += 1
                
            # Prefer balanced d_ff (512 was optimal)
            if combo['d_ff'] == 512:
                score += 2
            elif combo['d_ff'] <= 768:
                score += 1
                
            return score
        
        # Sort by score descending
        return sorted(combinations, key=score_combination, reverse=True)
    
    def run_training(self, params, combination_id):
        """Run training with given parameters"""
        print(f"🔥 Running Combination {combination_id}/20")
        print(f"   Parameters: {params}")
        
        # Prepare command
        cmd = [
            "python", "wandb_gainakt2_train.py",
            "--dataset_name=assist2015",
            "--use_wandb=0",
            f"--d_model={params['d_model']}",
            f"--learning_rate={params['learning_rate']}",
            f"--dropout={params['dropout']}",
            f"--num_encoder_blocks={params['num_encoder_blocks']}",
            f"--d_ff={params['d_ff']}",
            f"--n_heads={params['n_heads']}",
            "--num_epochs=5",  # Quick evaluation
            "--seed=42",
            "--fold=0"
        ]
        
        # Run with GPU allocation
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env,
                timeout=1800  # 30 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Extract metrics
            output = result.stdout
            auc = self.extract_metric(output, "validauc: ")
            acc = self.extract_metric(output, "validacc: ")
            loss = self.extract_metric(output, "train loss: ")
            
            return {
                'combination_id': combination_id,
                'parameters': params,
                'metrics': {
                    'validation_auc': auc,
                    'validation_acc': acc,
                    'train_loss': loss
                },
                'duration_seconds': duration,
                'success': result.returncode == 0,
                'output': output if result.returncode != 0 else ""
            }
            
        except subprocess.TimeoutExpired:
            return {
                'combination_id': combination_id,
                'parameters': params,
                'metrics': {'validation_auc': 0.0, 'validation_acc': 0.0, 'train_loss': 999.0},
                'duration_seconds': 1800,
                'success': False,
                'output': "TIMEOUT"
            }
        except Exception as e:
            return {
                'combination_id': combination_id,
                'parameters': params,
                'metrics': {'validation_auc': 0.0, 'validation_acc': 0.0, 'train_loss': 999.0},
                'duration_seconds': 0,
                'success': False,
                'output': f"ERROR: {str(e)}"
            }
    
    def extract_metric(self, output, pattern):
        """Extract metric value from training output"""
        try:
            lines = output.split('\n')
            for line in reversed(lines):  # Check from end
                if pattern in line:
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        value_part = parts[1].split(',')[0].split(' ')[0]
                        return float(value_part)
            return 0.0
        except:
            return 0.0
    
    def save_results(self, all_results):
        """Save optimization results"""
        results_file = self.results_dir / f"optimization_results_{self.timestamp}.json"
        
        # Sort by AUC descending
        sorted_results = sorted(
            [r for r in all_results if r['success']], 
            key=lambda x: x['metrics']['validation_auc'], 
            reverse=True
        )
        
        # Create summary
        summary = {
            'timestamp': self.timestamp,
            'total_combinations': len(all_results),
            'successful_runs': len(sorted_results),
            'best_result': sorted_results[0] if sorted_results else None,
            'top_5_results': sorted_results[:5],
            'all_results': all_results
        }
        
        # Save to file
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return results_file, summary
    
    def print_summary(self, summary):
        """Print optimization summary"""
        print("\n" + "="*60)
        print("🎯 GAINAKT2 PARAMETER OPTIMIZATION COMPLETE!")
        print("="*60)
        
        if summary['best_result']:
            best = summary['best_result']
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   Validation AUC: {best['metrics']['validation_auc']:.4f}")
            print(f"   Validation ACC: {best['metrics']['validation_acc']:.4f}")
            print(f"   Train Loss: {best['metrics']['train_loss']:.4f}")
            print(f"   Parameters:")
            for key, value in best['parameters'].items():
                print(f"     {key}: {value}")
            
            print(f"\n📊 OPTIMAL COMMAND:")
            params = best['parameters']
            print(f"CUDA_VISIBLE_DEVICES=0,1,2,3 python wandb_gainakt2_train.py \\")
            print(f"    --dataset_name=assist2015 \\")
            print(f"    --use_wandb=0 \\")
            print(f"    --d_model={params['d_model']} \\")
            print(f"    --learning_rate={params['learning_rate']} \\")
            print(f"    --dropout={params['dropout']} \\")
            print(f"    --num_encoder_blocks={params['num_encoder_blocks']} \\")
            print(f"    --d_ff={params['d_ff']} \\")
            print(f"    --n_heads={params['n_heads']} \\")
            print(f"    --num_epochs=200")
        
        print(f"\n📈 TOP 5 RESULTS:")
        for i, result in enumerate(summary['top_5_results'][:5], 1):
            print(f"   {i}. AUC: {result['metrics']['validation_auc']:.4f} | " +
                  f"d_model: {result['parameters']['d_model']} | " +
                  f"lr: {result['parameters']['learning_rate']} | " +
                  f"dropout: {result['parameters']['dropout']}")
        
        print(f"\n📋 Summary:")
        print(f"   Total combinations tested: {summary['total_combinations']}")
        print(f"   Successful runs: {summary['successful_runs']}")
        print(f"   Results saved to: optimization_results/optimization_results_{summary['timestamp']}.json")
    
    def run_optimization(self):
        """Run the complete parameter optimization"""
        print("🚀 Starting GainAKT2 Advanced Parameter Optimization")
        print("="*55)
        
        # Generate combinations
        combinations = self.generate_parameter_combinations()
        print(f"📊 Generated {len(combinations)} prioritized parameter combinations")
        
        # Run optimization
        all_results = []
        
        for i, params in enumerate(combinations, 1):
            result = self.run_training(params, i)
            all_results.append(result)
            
            if result['success']:
                auc = result['metrics']['validation_auc']
                acc = result['metrics']['validation_acc']
                print(f"   ✅ AUC: {auc:.4f}, ACC: {acc:.4f}")
            else:
                print(f"   ❌ Failed: {result['output'][:50]}...")
            
            print()
        
        # Save and summarize results
        results_file, summary = self.save_results(all_results)
        self.print_summary(summary)
        
        return results_file, summary

def main():
    optimizer = GainAKT2Optimizer()
    results_file, summary = optimizer.run_optimization()
    
    print(f"\n🎉 Optimization complete! Check {results_file} for detailed results.")

if __name__ == "__main__":
    main()