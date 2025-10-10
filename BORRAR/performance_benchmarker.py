#!/usr/bin/env python3
"# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')ormance Benchmarking Suite for GainAKT2Monitored with Cumulative Mastery.
Compares educational consistency and predictive performance against baseline models.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
import json
import time
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
sys.path.insert(0, '/    print("\n📊 Performance Impact:")
    print(f"  AUC Change: {perf_analysis['auc_percentage_change']:+.2f}%")
    print(f"  Verdict: {perf_analysis['performance_verdict']}")
    
    cons_analysis = report['consistency_analysis']
    print("\n🎓 Educational Validity:")aces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model


class PerformanceBenchmarker:
    """Comprehensive benchmarking system for educational consistency vs performance trade-offs."""
    
    def __init__(self, output_dir: str = "performance_benchmarks"):
        """Initialize the benchmarker."""
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize results storage
        self.benchmark_results = {}
        
    def load_data(self, dataset_name: str = "assist2015", fold: int = 0, batch_size: int = 64):
        """Load dataset for benchmarking."""
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        data_config = {
            "assist2015": {
                "dpath": "/workspaces/pykt-toolkit/data/assist2015",
                "num_q": 0,
                "num_c": 100,
                "input_type": ["concepts"],
                "max_concepts": 1,
                "min_seq_len": 3,
                "maxlen": 200,
                "emb_path": "",
                "train_valid_original_file": "train_valid.csv",
                "train_valid_file": "train_valid_sequences.csv",
                "folds": [0, 1, 2, 3, 4],
                "test_original_file": "test.csv",
                "test_file": "test_sequences.csv",
                "test_window_file": "test_window_sequences.csv"
            }
        }
        
        self.data_config = data_config
        self.dataset_name = dataset_name
        
        self.train_loader, self.valid_loader = init_dataset4train(
            dataset_name, "gainakt2", data_config, fold, batch_size
        )
        
        self.logger.info("✓ Dataset loaded successfully")
    
    def evaluate_model_performance(self, model, data_loader, model_name: str) -> Dict:
        """Comprehensive model evaluation including consistency metrics."""
        self.logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        start_time = time.time()
        
        # Performance metrics
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        # Consistency metrics
        consistency_violations = {
            'monotonicity': 0,
            'negative_gains': 0,
            'bounds': 0,
            'total_students': 0,
            'total_timesteps': 0
        }
        
        correlations = {
            'mastery_performance': [],
            'gain_performance': []
        }
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'Evaluating {model_name}')):
                questions = batch['cseqs'].to(self.device)
                responses = batch['rseqs'].to(self.device)
                questions_shifted = batch['shft_cseqs'].to(self.device)
                responses_shifted = batch['shft_rseqs'].to(self.device)
                mask = batch['masks'].to(self.device)
                
                # Get model outputs
                try:
                    # Try with interpretability states first
                    outputs = model.forward_with_states(
                        q=questions, r=responses, qry=questions_shifted
                    )
                    predictions = outputs['predictions']
                    
                    # Extract consistency information if available
                    if 'projected_mastery' in outputs and 'projected_gains' in outputs:
                        skill_mastery = outputs['projected_mastery']
                        skill_gains = outputs['projected_gains']
                        
                        # Analyze consistency for this batch
                        batch_consistency = self._analyze_batch_consistency(
                            skill_mastery, skill_gains, mask, responses_shifted
                        )
                        
                        # Update consistency metrics
                        for key in consistency_violations:
                            if key in batch_consistency:
                                consistency_violations[key] += batch_consistency[key]
                        
                        # Update correlations
                        correlations['mastery_performance'].extend(batch_consistency.get('mastery_correlations', []))
                        correlations['gain_performance'].extend(batch_consistency.get('gain_correlations', []))
                
                except (AttributeError, KeyError):
                    # Fallback to regular forward for models without interpretability
                    outputs = model(q=questions, r=responses, qry=questions_shifted)
                    predictions = outputs['predictions'] if isinstance(outputs, dict) else outputs
                
                # Calculate performance metrics
                valid_mask = mask.bool()
                valid_predictions = predictions[valid_mask]
                valid_targets = responses_shifted[valid_mask].float()
                
                # Loss calculation
                batch_loss = criterion(valid_predictions, valid_targets)
                total_loss += batch_loss.item()
                
                # Store predictions and targets
                all_predictions.extend(valid_predictions.cpu().numpy())
                all_targets.extend(valid_targets.cpu().numpy())
                
                # Memory cleanup for large evaluations
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate performance metrics
        evaluation_time = time.time() - start_time
        
        auc_score = roc_auc_score(all_targets, all_predictions)
        accuracy = accuracy_score(all_targets, np.round(all_predictions))
        
        # Precision-recall AUC
        precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
        pr_auc = auc(recall, precision)
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate consistency rates
        consistency_rates = {}
        if consistency_violations['total_students'] > 0:
            consistency_rates = {
                'monotonicity_violation_rate': consistency_violations['monotonicity'] / max(consistency_violations['total_timesteps'], 1),
                'negative_gain_rate': consistency_violations['negative_gains'] / max(consistency_violations['total_timesteps'], 1),
                'bounds_violation_rate': consistency_violations['bounds'] / max(consistency_violations['total_timesteps'], 1),
                'perfect_consistency_rate': 1.0 - (consistency_violations['monotonicity'] + consistency_violations['negative_gains'] + consistency_violations['bounds']) / max(consistency_violations['total_timesteps'] * 3, 1)
            }
        else:
            # No consistency information available (baseline model)
            consistency_rates = {
                'monotonicity_violation_rate': None,
                'negative_gain_rate': None,
                'bounds_violation_rate': None,
                'perfect_consistency_rate': None
            }
        
        # Calculate correlation statistics
        correlation_stats = {}
        if correlations['mastery_performance']:
            mastery_corrs = [c for c in correlations['mastery_performance'] if not np.isnan(c)]
            correlation_stats['mastery_performance_mean'] = np.mean(mastery_corrs) if mastery_corrs else None
            correlation_stats['mastery_performance_std'] = np.std(mastery_corrs) if mastery_corrs else None
        else:
            correlation_stats['mastery_performance_mean'] = None
            correlation_stats['mastery_performance_std'] = None
            
        if correlations['gain_performance']:
            gain_corrs = [c for c in correlations['gain_performance'] if not np.isnan(c)]
            correlation_stats['gain_performance_mean'] = np.mean(gain_corrs) if gain_corrs else None
            correlation_stats['gain_performance_std'] = np.std(gain_corrs) if gain_corrs else None
        else:
            correlation_stats['gain_performance_mean'] = None
            correlation_stats['gain_performance_std'] = None
        
        return {
            'model_name': model_name,
            'performance_metrics': {
                'auc': auc_score,
                'accuracy': accuracy,
                'pr_auc': pr_auc,
                'loss': avg_loss,
                'evaluation_time': evaluation_time,
                'total_predictions': len(all_predictions)
            },
            'consistency_metrics': consistency_rates,
            'correlation_metrics': correlation_stats,
            'detailed_predictions': {
                'predictions': all_predictions[:1000],  # Sample for analysis
                'targets': all_targets[:1000]
            }
        }
    
    def _analyze_batch_consistency(self, skill_mastery, skill_gains, mask, responses) -> Dict:
        """Analyze consistency violations for a batch of students."""
        batch_size = skill_mastery.size(0)
        violations = {
            'monotonicity': 0,
            'negative_gains': 0,
            'bounds': 0,
            'total_students': batch_size,
            'total_timesteps': 0,
            'mastery_correlations': [],
            'gain_correlations': []
        }
        
        for i in range(batch_size):
            student_mask = mask[i].bool()
            student_mastery = skill_mastery[i][student_mask]
            student_gains = skill_gains[i][student_mask]
            student_performance = responses[i][student_mask].float()
            
            seq_len = student_mastery.size(0)
            if seq_len < 2:
                continue
            
            violations['total_timesteps'] += seq_len
            
            # Convert to numpy
            mastery_np = student_mastery.cpu().numpy()
            gains_np = student_gains.cpu().numpy()
            performance_np = student_performance.cpu().numpy()
            
            # Aggregate across concepts
            mean_mastery = np.mean(mastery_np, axis=1)
            mean_gains = np.mean(gains_np, axis=1)
            
            # Check monotonicity
            for t in range(1, seq_len):
                if mean_mastery[t] < mean_mastery[t-1] - 1e-6:
                    violations['monotonicity'] += 1
            
            # Check negative gains
            if np.any(gains_np < -1e-6):
                violations['negative_gains'] += seq_len  # Count all timesteps for this student
            
            # Check bounds
            if np.any((mastery_np < -1e-6) | (mastery_np > 1 + 1e-6)):
                violations['bounds'] += seq_len  # Count all timesteps for this student
            
            # Calculate correlations
            if seq_len >= 3:
                try:
                    mastery_corr = np.corrcoef(mean_mastery, performance_np)[0, 1]
                    if not np.isnan(mastery_corr):
                        violations['mastery_correlations'].append(mastery_corr)
                    
                    gain_corr = np.corrcoef(mean_gains, performance_np)[0, 1]
                    if not np.isnan(gain_corr):
                        violations['gain_correlations'].append(gain_corr)
                except (ValueError, np.linalg.LinAlgError):
                    pass
        
        return violations
    
    def benchmark_cumulative_mastery_model(self, model_path: str) -> Dict:
        """Benchmark the cumulative mastery model."""
        self.logger.info(f"Loading cumulative mastery model from: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        model = create_monitored_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Evaluate on validation set
        results = self.evaluate_model_performance(model, self.valid_loader, "CumulativeMastery")
        return results
    
    def benchmark_baseline_model(self, model_config: Dict, model_name: str = "Baseline") -> Dict:
        """Train and benchmark a baseline model without cumulative mastery constraints."""
        self.logger.info(f"Training baseline model: {model_name}")
        
        # Create baseline model (same architecture, different constraints)
        baseline_config = model_config.copy()
        baseline_config.update({
            'non_negative_loss_weight': 0.0,  # No constraints
            'monotonicity_loss_weight': 0.0,
            'mastery_performance_loss_weight': 0.1,  # Light regularization only
            'gain_performance_loss_weight': 0.1,
            'sparsity_loss_weight': 0.0,
            'consistency_loss_weight': 0.0
        })
        
        model = create_monitored_model(baseline_config)
        model = model.to(self.device)
        
        # Quick training (fewer epochs for comparison)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        
        num_epochs = 10  # Quick training for comparison
        
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Training {model_name} Epoch {epoch+1}')):
                questions = batch['cseqs'].to(self.device)
                responses = batch['rseqs'].to(self.device)
                questions_shifted = batch['shft_cseqs'].to(self.device)
                responses_shifted = batch['shft_rseqs'].to(self.device)
                mask = batch['masks'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model.forward_with_states(q=questions, r=responses, qry=questions_shifted)
                predictions = outputs['predictions']
                interpretability_loss = outputs.get('interpretability_loss', 0)
                
                # Main loss
                valid_mask = mask.bool()
                valid_predictions = predictions[valid_mask]
                valid_targets = responses_shifted[valid_mask].float()
                
                main_loss = criterion(valid_predictions, valid_targets)
                total_loss = main_loss + interpretability_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                # Quick training - only use first 50 batches per epoch
                if batch_idx >= 50:
                    break
            
            self.logger.info(f"  Epoch {epoch+1} Loss: {epoch_loss/(batch_idx+1):.4f}")
        
        # Evaluate baseline model
        results = self.evaluate_model_performance(model, self.valid_loader, model_name)
        return results
    
    def create_comparison_plots(self, results: List[Dict], save_name: str = "performance_comparison.png") -> str:
        """Create comprehensive comparison plots."""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Extract data
        models = [r['model_name'] for r in results]
        aucs = [r['performance_metrics']['auc'] for r in results]
        accuracies = [r['performance_metrics']['accuracy'] for r in results]
        
        # Consistency metrics (may be None for baseline)
        monotonicity_violations = []
        bounds_violations = []
        gain_violations = []
        mastery_correlations = []
        
        for r in results:
            cm = r['consistency_metrics']
            monotonicity_violations.append(cm.get('monotonicity_violation_rate', 0) or 0)
            bounds_violations.append(cm.get('bounds_violation_rate', 0) or 0)
            gain_violations.append(cm.get('negative_gain_rate', 0) or 0)
            
            corr = r['correlation_metrics'].get('mastery_performance_mean')
            mastery_correlations.append(corr if corr is not None else 0)
        
        # Create subplots
        gs = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Performance Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = np.arange(len(models))
        bars1 = ax1.bar(x_pos, aucs, alpha=0.8, color=['#2E8B57', '#DC143C', '#4169E1'][:len(models)])
        ax1.set_xlabel('Model')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Predictive Performance (AUC)', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, auc_score in zip(bars1, aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc_score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Accuracy Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(x_pos, accuracies, alpha=0.8, color=['#2E8B57', '#DC143C', '#4169E1'][:len(models)])
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Prediction Accuracy', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars2, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Consistency Violations Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        violation_types = ['Monotonicity', 'Bounds', 'Negative Gains']
        violation_data = np.array([monotonicity_violations, bounds_violations, gain_violations])
        
        width = 0.25
        x_violation = np.arange(len(violation_types))
        
        for i, model in enumerate(models):
            ax3.bar(x_violation + i * width, violation_data[:, i], width, 
                   label=model, alpha=0.8, 
                   color=['#2E8B57', '#DC143C', '#4169E1'][i])
        
        ax3.set_xlabel('Violation Type')
        ax3.set_ylabel('Violation Rate')
        ax3.set_title('Educational Consistency Violations', fontweight='bold')
        ax3.set_xticks(x_violation + width)
        ax3.set_xticklabels(violation_types, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance vs Consistency Trade-off
        ax4 = fig.add_subplot(gs[1, :])
        
        # Calculate overall consistency score (lower is better)
        consistency_scores = [sum([monotonicity_violations[i], bounds_violations[i], gain_violations[i]]) 
                             for i in range(len(models))]
        
        # Scatter plot with model labels
        colors = ['#2E8B57', '#DC143C', '#4169E1'][:len(models)]
        ax4.scatter(consistency_scores, aucs, s=200, alpha=0.7, c=colors, edgecolors='black', linewidth=2)
        
        # Add model labels
        for i, model in enumerate(models):
            ax4.annotate(model, (consistency_scores[i], aucs[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
        
        ax4.set_xlabel('Total Consistency Violations (Lower is Better)', fontsize=12)
        ax4.set_ylabel('AUC Performance (Higher is Better)', fontsize=12)
        ax4.set_title('Performance vs Educational Consistency Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add ideal region
        if min(consistency_scores) < max(consistency_scores):
            ax4.axvspan(min(consistency_scores), min(consistency_scores) + 0.01, alpha=0.2, color='green', 
                       label='Ideal Consistency')
        ax4.axhspan(max(aucs) * 0.95, max(aucs), alpha=0.2, color='blue', 
                   label='High Performance')
        
        # 5. Summary Statistics Table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary table
        table_data = []
        headers = ['Model', 'AUC', 'Accuracy', 'Monotonicity\\nViolations', 'Bounds\\nViolations', 
                  'Negative Gains', 'Mastery-Perf\\nCorrelation', 'Overall\\nGrade']
        
        for i, r in enumerate(results):
            pm = r['performance_metrics']
            cm = r['consistency_metrics']
            corr_m = r['correlation_metrics']
            
            # Calculate overall grade
            auc_grade = 'A' if pm['auc'] > 0.7 else 'B' if pm['auc'] > 0.65 else 'C'
            consistency_grade = 'A+' if consistency_scores[i] < 0.01 else 'A' if consistency_scores[i] < 0.05 else 'B'
            overall_grade = f"{auc_grade}/{consistency_grade}"
            
            row = [
                r['model_name'],
                f"{pm['auc']:.3f}",
                f"{pm['accuracy']:.3f}",
                f"{cm.get('monotonicity_violation_rate', 0) or 0:.1%}",
                f"{cm.get('bounds_violation_rate', 0) or 0:.1%}",
                f"{cm.get('negative_gain_rate', 0) or 0:.1%}",
                f"{corr_m.get('mastery_performance_mean', 0) or 0:.3f}",
                overall_grade
            ]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the table
        for i in range(len(table_data)):
            for j in range(len(headers)):
                if j == 0:  # Model name
                    table[(i+1, j)].set_facecolor(['#E8F5E8', '#FFE8E8', '#E8E8FF'][i])
                else:
                    table[(i+1, j)].set_facecolor('#F5F5F5')
        
        plt.suptitle('Comprehensive Performance Benchmarking\\n'
                    'Educational Consistency vs Predictive Performance', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Comparison plots saved: {save_path}")
        return save_path
    
    def run_comprehensive_benchmark(self, cumulative_mastery_model_path: str) -> Dict:
        """Run comprehensive benchmark comparing cumulative mastery vs baseline."""
        
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE PERFORMANCE BENCHMARKING")
        self.logger.info("=" * 80)
        
        # Load data
        self.load_data()
        
        # Benchmark cumulative mastery model
        self.logger.info("\\n1. Evaluating Cumulative Mastery Model...")
        cumulative_results = self.benchmark_cumulative_mastery_model(cumulative_mastery_model_path)
        
        # Get model config for baseline comparison
        checkpoint = torch.load(cumulative_mastery_model_path, map_location=self.device)
        base_config = checkpoint['model_config']
        
        # Benchmark baseline model
        self.logger.info("\\n2. Training and Evaluating Baseline Model...")
        baseline_results = self.benchmark_baseline_model(base_config, "Baseline_NoConstraints")
        
        # Combine results
        all_results = [cumulative_results, baseline_results]
        
        # Create comparison visualizations
        self.logger.info("\\n3. Creating Comparison Visualizations...")
        comparison_plot = self.create_comparison_plots(all_results)
        
        # Generate comprehensive report
        self.logger.info("\\n4. Generating Comprehensive Report...")
        report = self._generate_benchmark_report(all_results)
        
        # Save detailed results
        benchmark_data = {
            'timestamp': datetime.now().isoformat(),
            'cumulative_mastery_results': cumulative_results,
            'baseline_results': baseline_results,
            'comparison_plot': comparison_plot,
            'comprehensive_report': report
        }
        
        results_file = os.path.join(self.output_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        benchmark_data = convert_numpy_types(benchmark_data)
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        self.logger.info("=" * 80)
        self.logger.info("✅ BENCHMARKING COMPLETED!")
        self.logger.info(f"📊 Results saved to: {results_file}")
        self.logger.info(f"📈 Comparison plot: {comparison_plot}")
        self.logger.info("=" * 80)
        
        return benchmark_data
    
    def _generate_benchmark_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive benchmark report."""
        
        cumulative = next(r for r in results if 'Cumulative' in r['model_name'])
        baseline = next(r for r in results if 'Baseline' in r['model_name'])
        
        # Performance comparison
        auc_diff = cumulative['performance_metrics']['auc'] - baseline['performance_metrics']['auc']
        acc_diff = cumulative['performance_metrics']['accuracy'] - baseline['performance_metrics']['accuracy']
        
        # Consistency comparison
        cumulative_violations = (
            cumulative['consistency_metrics'].get('monotonicity_violation_rate', 0) +
            cumulative['consistency_metrics'].get('bounds_violation_rate', 0) +
            cumulative['consistency_metrics'].get('negative_gain_rate', 0)
        )
        
        baseline_violations = (
            baseline['consistency_metrics'].get('monotonicity_violation_rate', float('inf')) +
            baseline['consistency_metrics'].get('bounds_violation_rate', float('inf')) +
            baseline['consistency_metrics'].get('negative_gain_rate', float('inf'))
        )
        
        report = {
            'executive_summary': {
                'cumulative_mastery_achieves_perfect_consistency': cumulative_violations == 0,
                'performance_maintained': auc_diff >= -0.05,  # Less than 5% AUC drop acceptable
                'significant_consistency_improvement': cumulative_violations < baseline_violations * 0.1,
                'recommendation': 'ADOPT' if (cumulative_violations == 0 and auc_diff >= -0.05) else 'CONSIDER'
            },
            'performance_analysis': {
                'auc_difference': auc_diff,
                'auc_percentage_change': (auc_diff / baseline['performance_metrics']['auc']) * 100,
                'accuracy_difference': acc_diff,
                'performance_verdict': 'MAINTAINED' if auc_diff >= -0.02 else 'SLIGHT_DECREASE' if auc_diff >= -0.05 else 'SIGNIFICANT_DECREASE'
            },
            'consistency_analysis': {
                'cumulative_mastery_violations': cumulative_violations,
                'baseline_violations': baseline_violations if baseline_violations != float('inf') else 'HIGH',
                'consistency_improvement': 'PERFECT' if cumulative_violations == 0 else 'SIGNIFICANT' if cumulative_violations < baseline_violations * 0.5 else 'MODERATE',
                'educational_validity': 'EXCELLENT' if cumulative_violations == 0 else 'GOOD' if cumulative_violations < 0.05 else 'POOR'
            },
            'trade_off_analysis': {
                'auc_cost_per_consistency_improvement': abs(auc_diff) / max(baseline_violations - cumulative_violations, 0.001),
                'worth_trade_off': auc_diff >= -0.05 and cumulative_violations < baseline_violations * 0.5,
                'educational_value': 'HIGH' if cumulative_violations == 0 else 'MEDIUM' if cumulative_violations < 0.1 else 'LOW'
            }
        }
        
        return report


def main():
    """Main function for command-line benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark cumulative mastery model performance')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to cumulative mastery model')
    parser.add_argument('--output_dir', type=str, 
                       default=f"performance_benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Output directory for benchmark results')
    
    args = parser.parse_args()
    
    # Create benchmarker
    benchmarker = PerformanceBenchmarker(args.output_dir)
    
    # Run comprehensive benchmark
    results = benchmarker.run_comprehensive_benchmark(args.model_path)
    
    # Print summary
    print("\\n🎯 BENCHMARKING SUMMARY:")
    print("=" * 50)
    
    report = results['comprehensive_report']
    exec_summary = report['executive_summary']
    
    print(f"✅ Perfect Consistency: {exec_summary['cumulative_mastery_achieves_perfect_consistency']}")
    print(f"✅ Performance Maintained: {exec_summary['performance_maintained']}")  
    print(f"✅ Consistency Improvement: {exec_summary['significant_consistency_improvement']}")
    print(f"🎯 Recommendation: {exec_summary['recommendation']}")
    
    perf_analysis = report['performance_analysis']
    print(f"\\n📊 Performance Impact:")
    print(f"  AUC Change: {perf_analysis['auc_percentage_change']:+.2f}%")
    print(f"  Verdict: {perf_analysis['performance_verdict']}")
    
    cons_analysis = report['consistency_analysis']
    print(f"\\n🎓 Educational Validity:")
    print(f"  Consistency Grade: {cons_analysis['educational_validity']}")
    print(f"  Improvement Level: {cons_analysis['consistency_improvement']}")


if __name__ == "__main__":
    main()