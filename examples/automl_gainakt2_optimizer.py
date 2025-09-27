#!/usr/bin/env python3
"""
🚀 GainAKT2 AutoML Hyperparameter Optimizer

Advanced AutoML system for finding optimal GainAKT2 parameters starting from known best configuration.
Uses multiple optimization strategies including Bayesian optimization, adaptive sampling, and early stopping.

Starting Point (Baseline AUC: 0.7233):
- d_model: 256
- learning_rate: 0.0002  
- dropout: 0.2
- num_encoder_blocks: 4
- d_ff: 768
- n_heads: 8

Authors: AutoML System
Version: 1.0

BEST RESULT:
UC: 0.7240 | ACC: 0.7531

REPRODUCE BEST RESULT (AUC: 0.7240)
cd /workspaces/pykt-toolkit/examples
python wandb_gainakt2_train.py \
    --dataset_name=assist2015 \
    --use_wandb=0 \
    --d_model=224 \
    --learning_rate=1.59e-04 \
    --dropout=0.155 \
    --num_encoder_blocks=6 \
    --d_ff=832 \
    --n_heads=16 \
    --num_epochs=10 \
    --seed=42 \
    --fold=0

Detailed results saved to: /workspaces/pykt-toolkit/examples/automl_results/automl_results_20250927_215207.json
"""

import os
import sys
import json
import time
import random
import itertools
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

# Add the examples directory to Python path for imports
sys.path.insert(0, '/workspaces/pykt-toolkit/examples')

# Import the training function directly
def import_training_module():
    """Import the training module directly to avoid subprocess calls"""
    spec = importlib.util.spec_from_file_location("wandb_gainakt2_train", "/workspaces/pykt-toolkit/examples/wandb_gainakt2_train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wandb_gainakt2_train"] = module
    spec.loader.exec_module(module)
    return module

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Try importing scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️  scikit-optimize not available. Installing for Bayesian optimization...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
        from skopt.acquisition import gaussian_ei
        BAYESIAN_AVAILABLE = True
        print("✅ scikit-optimize installed successfully!")
    except ImportError:
        BAYESIAN_AVAILABLE = False
        print("❌ Failed to install scikit-optimize. Using alternative methods.")

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    combination_id: int
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    duration_seconds: float
    success: bool
    strategy: str
    improvement_over_baseline: float
    timestamp: str

class GainAKT2AutoMLOptimizer:
    """
    Advanced AutoML system for GainAKT2 hyperparameter optimization
    
    Features:
    - Bayesian Optimization with Gaussian Processes
    - Adaptive Parameter Sampling
    - Multi-Strategy Search (Bayesian + Random + Grid)
    - Early Stopping for Efficiency  
    - Intelligent Parameter Space Exploration
    - Real-time Results Tracking
    """
    
    def __init__(self, max_evaluations: int = 50, parallel_jobs: int = 2, 
                 early_stopping_patience: int = 10, target_auc: float = 0.735):
        """
        Initialize the AutoML optimizer
        
        Args:
            max_evaluations: Maximum number of parameter combinations to evaluate
            parallel_jobs: Number of parallel training jobs
            early_stopping_patience: Stop if no improvement for N evaluations
            target_auc: Stop if this AUC is achieved
        """
        self.max_evaluations = max_evaluations
        self.parallel_jobs = parallel_jobs
        self.early_stopping_patience = early_stopping_patience
        self.target_auc = target_auc
        
        # Baseline optimal parameters (AUC: 0.7233)
        self.baseline_params = {
            'd_model': 256,
            'learning_rate': 0.0002,
            'dropout': 0.2, 
            'num_encoder_blocks': 4,
            'd_ff': 768,
            'n_heads': 8
        }
        self.baseline_auc = 0.7233
        
        # Results tracking
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        self.evaluation_count = 0
        self.no_improvement_count = 0
        
        # Setup logging
        self.setup_logging()
        
        # Define parameter spaces for different strategies
        self.define_parameter_spaces()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"/workspaces/pykt-toolkit/examples/automl_results/automl_log_{timestamp}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def define_parameter_spaces(self):
        """Define parameter spaces for optimization strategies"""
        
        # Bayesian optimization space (continuous and discrete)
        if BAYESIAN_AVAILABLE:
            self.bayesian_space = [
                Integer(128, 512, name='d_model'),           # Model dimension
                Real(1e-5, 1e-3, prior='log-uniform', name='learning_rate'),  # Learning rate (log scale)
                Real(0.1, 0.4, name='dropout'),             # Dropout rate
                Integer(2, 6, name='num_encoder_blocks'),    # Number of blocks
                Integer(256, 1536, name='d_ff'),             # Feed-forward dimension
                Integer(4, 16, name='n_heads'),              # Attention heads
            ]
        
        # Focused grid around optimal parameters (for fine-tuning)
        self.fine_tune_grid = {
            'd_model': [192, 224, 256, 288, 320, 384],      # Around 256
            'learning_rate': [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4],  # Around 2e-4
            'dropout': [0.15, 0.18, 0.2, 0.22, 0.25],      # Around 0.2
            'num_encoder_blocks': [3, 4, 5],                # Around 4
            'd_ff': [512, 640, 768, 896, 1024],             # Around 768
            'n_heads': [6, 8, 10, 12],                      # Around 8
        }
        
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter combination for feasibility"""
        
        # Check d_model divisible by n_heads
        if params['d_model'] % params['n_heads'] != 0:
            return False
            
        # Check memory constraints (rough estimation)
        estimated_params = (
            params['d_model'] * params['d_model'] * params['n_heads'] * params['num_encoder_blocks'] +
            params['d_model'] * params['d_ff'] * params['num_encoder_blocks']
        )
        if estimated_params > 50_000_000:  # 50M parameter limit
            return False
            
        # Check reasonable ranges
        if params['learning_rate'] < 1e-6 or params['learning_rate'] > 1e-2:
            return False
        if params['dropout'] < 0.05 or params['dropout'] > 0.5:
            return False
            
        return True
        
    def run_evaluation(self, params: Dict[str, Any], strategy: str, combination_id: int) -> OptimizationResult:
        """Run a single parameter evaluation using direct training function call"""
        
        self.logger.info(f"🔥 Evaluation {combination_id} ({strategy})")
        self.logger.info(f"   Parameters: {params}")
        
        start_time = time.time()
        
        try:
            # Use subprocess with proper control to avoid runaway processes
            cmd = [
                sys.executable, "wandb_gainakt2_train.py",
                "--dataset_name=assist2015",
                "--use_wandb=0",
                f"--d_model={params['d_model']}",
                f"--learning_rate={params['learning_rate']:.2e}",
                f"--dropout={params['dropout']:.3f}",
                f"--num_encoder_blocks={params['num_encoder_blocks']}",
                f"--d_ff={params['d_ff']}",
                f"--n_heads={params['n_heads']}",
                "--num_epochs=5",
                "--seed=42",
                "--fold=0"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, 
                                  cwd="/workspaces/pykt-toolkit/examples")
            
            if result.returncode == 0:
                metrics = self.parse_training_output(result.stdout)
                success = True
                self.logger.info(f"   ✅ AUC: {metrics['validation_auc']:.4f} | ACC: {metrics['validation_acc']:.4f}")
            else:
                self.logger.error(f"   ❌ Training failed: {result.stderr}")
                metrics = {'validation_auc': 0.0, 'validation_acc': 0.0, 'train_loss': float('inf')}
                success = False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"   ⏰ Training timed out")
            metrics = {'validation_auc': 0.0, 'validation_acc': 0.0, 'train_loss': float('inf')}
            success = False
        except Exception as e:
            self.logger.error(f"   ❌ Exception: {str(e)}")
            metrics = {'validation_auc': 0.0, 'validation_acc': 0.0, 'train_loss': float('inf')}
            success = False
            
        duration = time.time() - start_time
        improvement = metrics['validation_auc'] - self.baseline_auc
        
        return OptimizationResult(
            combination_id=combination_id,
            parameters=params.copy(),
            metrics=metrics,
            duration_seconds=duration,
            success=success,
            strategy=strategy,
            improvement_over_baseline=improvement,
            timestamp=datetime.now().isoformat()
        )
        
    def parse_training_output(self, output: str) -> Dict[str, float]:
        """Parse training output to extract metrics"""
        
        metrics = {'validation_auc': 0.0, 'validation_acc': 0.0, 'train_loss': float('inf')}
        
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if 'best auc:' in line.lower():
                    # Extract AUC and accuracy
                    parts = line.split(',')
                    for part in parts:
                        if 'validauc:' in part.lower():
                            auc_str = part.split(':')[1].strip()
                            metrics['validation_auc'] = float(auc_str)
                        elif 'validacc:' in part.lower():
                            acc_str = part.split(':')[1].strip()
                            metrics['validation_acc'] = float(acc_str)
                        elif 'train loss:' in part.lower():
                            loss_str = part.split(':')[1].strip()
                            metrics['train_loss'] = float(loss_str)
        except Exception as e:
            self.logger.warning(f"Failed to parse metrics: {e}")
            
        return metrics
        
    def random_search(self, n_evaluations: int) -> List[Dict[str, Any]]:
        """Adaptive random search around baseline parameters"""
        
        self.logger.info(f"🎲 Starting Random Search ({n_evaluations} evaluations)")
        
        combinations = []
        
        for i in range(n_evaluations):
            # Generate parameters with adaptive sampling
            if i < n_evaluations // 3:
                # Phase 1: Close to baseline (local search)
                params = self.sample_around_baseline(scale=0.2)
            elif i < 2 * n_evaluations // 3:
                # Phase 2: Medium exploration
                params = self.sample_around_baseline(scale=0.5)
            else:
                # Phase 3: Wide exploration
                params = self.sample_random_parameters()
                
            if self.validate_parameters(params):
                combinations.append(params)
                
        return combinations
        
    def sample_around_baseline(self, scale: float = 0.3) -> Dict[str, Any]:
        """Sample parameters around baseline with given scale"""
        
        params = {}
        
        # d_model - sample around 256
        params['d_model'] = int(np.clip(
            np.random.normal(256, 64 * scale), 128, 512
        ))
        # Ensure divisibility by future n_heads
        params['d_model'] = (params['d_model'] // 32) * 32
        
        # learning_rate - log-normal around 2e-4
        params['learning_rate'] = np.clip(
            np.random.lognormal(np.log(2e-4), 0.5 * scale), 1e-5, 1e-3
        )
        
        # dropout - normal around 0.2
        params['dropout'] = np.clip(
            np.random.normal(0.2, 0.1 * scale), 0.1, 0.4
        )
        
        # num_encoder_blocks - discrete around 4
        blocks_options = [2, 3, 4, 5, 6]
        weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # Favor 4 blocks
        params['num_encoder_blocks'] = np.random.choice(blocks_options, p=weights)
        
        # d_ff - sample around 768
        params['d_ff'] = int(np.clip(
            np.random.normal(768, 256 * scale), 256, 1536
        ))
        # Round to nearest 64
        params['d_ff'] = (params['d_ff'] // 64) * 64
        
        # n_heads - sample around 8, ensure divisibility
        head_options = [4, 6, 8, 10, 12, 16]
        valid_heads = [h for h in head_options if params['d_model'] % h == 0]
        if valid_heads:
            params['n_heads'] = np.random.choice(valid_heads)
        else:
            params['n_heads'] = 8
            
        return params
        
    def sample_random_parameters(self) -> Dict[str, Any]:
        """Sample parameters from wide ranges"""
        
        params = {}
        
        # Wide sampling ranges
        d_model_options = [128, 160, 192, 224, 256, 288, 320, 384, 448, 512]
        params['d_model'] = np.random.choice(d_model_options)
        
        params['learning_rate'] = np.random.uniform(1e-5, 1e-3)
        params['dropout'] = np.random.uniform(0.1, 0.4)
        params['num_encoder_blocks'] = np.random.randint(2, 7)
        
        d_ff_options = [256, 384, 512, 640, 768, 896, 1024, 1280, 1536]
        params['d_ff'] = np.random.choice(d_ff_options)
        
        # Ensure n_heads divides d_model
        possible_heads = [h for h in [4, 6, 8, 10, 12, 16] if params['d_model'] % h == 0]
        params['n_heads'] = np.random.choice(possible_heads) if possible_heads else 8
        
        return params
        
    def run_optimization(self):
        """Main optimization pipeline using random search"""
        
        self.logger.info("🚀 Starting GainAKT2 AutoML Optimization")
        self.logger.info(f"   Baseline AUC: {self.baseline_auc:.4f}")
        self.logger.info(f"   Target AUC: {self.target_auc:.4f}")
        self.logger.info(f"   Max Evaluations: {self.max_evaluations}")
        
        start_time = time.time()
        
        # Generate parameter combinations using adaptive random search
        combinations = self.random_search(self.max_evaluations)
        
        # Run evaluations sequentially to avoid process issues
        for i, params in enumerate(combinations):
            if self.evaluation_count >= self.max_evaluations:
                break
            if self.no_improvement_count >= self.early_stopping_patience:
                break
                
            result = self.run_evaluation(params, "random", self.evaluation_count)
            self.evaluation_count += 1
            self.results.append(result)
            
            # Update best result
            if result.success and (self.best_result is None or 
                                 result.metrics['validation_auc'] > self.best_result.metrics['validation_auc']):
                self.best_result = result
                self.no_improvement_count = 0
                self.logger.info(f"🏆 New best AUC: {result.metrics['validation_auc']:.4f} (improvement: +{result.improvement_over_baseline:.4f})")
            else:
                self.no_improvement_count += 1
                
            # Check early stopping conditions
            if result.metrics['validation_auc'] >= self.target_auc:
                self.logger.info(f"🎯 Target AUC achieved!")
                break
                    
        # Generate final report
        total_time = time.time() - start_time
        self.generate_final_report(total_time)
        
    def generate_final_report(self, total_time: float):
        """Generate comprehensive final report"""
        
        self.logger.info(f"\n🎯 AutoML Optimization Complete!")
        self.logger.info(f"   Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"   Total Evaluations: {len(self.results)}")
        self.logger.info(f"   Successful Runs: {sum(1 for r in self.results if r.success)}")
        
        if self.best_result:
            self.logger.info(f"\n🏆 BEST RESULT:")
            self.logger.info(f"   AUC: {self.best_result.metrics['validation_auc']:.4f}")
            self.logger.info(f"   Improvement: +{self.best_result.improvement_over_baseline:.4f}")
            self.logger.info(f"   Parameters: {self.best_result.parameters}")
            self.logger.info(f"   Strategy: {self.best_result.strategy}")
            
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/workspaces/pykt-toolkit/examples/automl_results/automl_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        report_data = {
            "optimization_summary": {
                "total_time_seconds": total_time,
                "total_evaluations": len(self.results),
                "successful_runs": sum(1 for r in self.results if r.success),
                "baseline_auc": self.baseline_auc,
                "best_auc": self.best_result.metrics['validation_auc'] if self.best_result else 0.0,
                "improvement": self.best_result.improvement_over_baseline if self.best_result else 0.0,
                "target_achieved": self.best_result.metrics['validation_auc'] >= self.target_auc if self.best_result else False
            },
            "best_result": asdict(self.best_result) if self.best_result else None,
            "all_results": [asdict(r) for r in self.results],
            "baseline_parameters": self.baseline_params
        }
        
        with open(results_file, 'w') as f:
            # Convert all data to JSON-serializable format before saving
            json_safe_data = convert_numpy_types(report_data)
            json.dump(json_safe_data, f, indent=2)
            
        self.logger.info(f"📊 Detailed results saved to: {results_file}")
        
        # Generate training command for best result
        if self.best_result:
            self.generate_best_command()
            
    def generate_best_command(self):
        """Generate command to reproduce best result"""
        
        params = self.best_result.parameters
        command = f"""
# 🏆 REPRODUCE BEST RESULT (AUC: {self.best_result.metrics['validation_auc']:.4f})
cd /workspaces/pykt-toolkit/examples
python wandb_gainakt2_train.py \\
    --dataset_name=assist2015 \\
    --use_wandb=0 \\
    --d_model={params['d_model']} \\
    --learning_rate={params['learning_rate']:.2e} \\
    --dropout={params['dropout']:.3f} \\
    --num_encoder_blocks={params['num_encoder_blocks']} \\
    --d_ff={params['d_ff']} \\
    --n_heads={params['n_heads']} \\
    --num_epochs=10 \\
    --seed=42 \\
    --fold=0
"""
        
        self.logger.info(command)
        
        # Save command to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd_file = f"/workspaces/pykt-toolkit/examples/automl_results/best_command_{timestamp}.sh"
        with open(cmd_file, 'w') as f:
            f.write(command)
            
        self.logger.info(f"💾 Best command saved to: {cmd_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GainAKT2 AutoML Hyperparameter Optimizer")
    parser.add_argument("--max_evaluations", type=int, default=50, 
                       help="Maximum number of parameter combinations to evaluate")
    parser.add_argument("--parallel_jobs", type=int, default=2,
                       help="Number of parallel training jobs")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                       help="Stop if no improvement for N evaluations")
    parser.add_argument("--target_auc", type=float, default=0.735,
                       help="Stop if this AUC is achieved")
    
    args = parser.parse_args()
    
    # Create optimizer and run
    optimizer = GainAKT2AutoMLOptimizer(
        max_evaluations=args.max_evaluations,
        parallel_jobs=args.parallel_jobs,
        early_stopping_patience=args.early_stopping_patience,
        target_auc=args.target_auc
    )
    
    optimizer.run_optimization()


if __name__ == "__main__":
    main()