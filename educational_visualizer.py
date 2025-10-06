#!/usr/bin/env python3
"""
Educational Visualization Tools for GainAKT2Monitored with Cumulative Mastery.
Creates comprehensive visualizations to demonstrate educational interpretability.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model


class EducationalVisualizer:
    """Comprehensive visualization toolkit for educational interpretability."""
    
    def __init__(self, model_path: str, output_dir: str = "educational_visualizations"):
        """Initialize the visualizer with a trained model."""
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
        
        # Setup visualization style
        self._setup_style()
    
    def _load_model(self):
        """Load the trained model."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = checkpoint['model_config']
        
        # Create and load model
        self.model = create_monitored_model(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("✓ Model loaded successfully")
    
    def _setup_style(self):
        """Setup matplotlib and seaborn styling."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Custom color schemes
        self.colors = {
            'mastery': '#2E8B57',      # Sea green
            'gains': '#FF6347',        # Tomato red  
            'performance': '#4169E1',   # Royal blue
            'correct': '#32CD32',       # Lime green
            'incorrect': '#DC143C',     # Crimson
            'monotonicity': '#9370DB',  # Medium purple
            'background': '#F5F5F5',    # White smoke
            'grid': '#D3D3D3'          # Light gray
        }
    
    def load_data(self, dataset_name: str = "assist2015", fold: int = 0, batch_size: int = 32):
        """Load and prepare dataset for visualization."""
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
        
        self.train_loader, self.valid_loader = init_dataset4train(
            dataset_name, "gainakt2", data_config, fold, batch_size
        )
        
        self.logger.info("✓ Dataset loaded successfully")
    
    def extract_learning_trajectories(self, num_students: int = 20) -> List[Dict]:
        """Extract detailed learning trajectories from multiple students."""
        self.logger.info(f"Extracting learning trajectories from {num_students} students...")
        
        trajectories = []
        student_count = 0
        
        with torch.no_grad():
            for batch in self.valid_loader:
                if student_count >= num_students:
                    break
                
                questions = batch['cseqs'].to(self.device)
                responses = batch['rseqs'].to(self.device)
                questions_shifted = batch['shft_cseqs'].to(self.device)
                responses_shifted = batch['shft_rseqs'].to(self.device)
                mask = batch['masks'].to(self.device)
                
                # Get model outputs
                outputs = self.model.forward_with_states(
                    q=questions, r=responses, qry=questions_shifted
                )
                
                skill_mastery = outputs['projected_mastery']
                skill_gains = outputs['projected_gains']
                predictions = outputs['predictions']
                
                batch_size_actual = questions.size(0)
                
                for i in range(batch_size_actual):
                    if student_count >= num_students:
                        break
                    
                    # Extract valid sequence
                    student_mask = mask[i].bool()
                    valid_indices = torch.where(student_mask)[0]
                    
                    if len(valid_indices) < 3:  # Need minimum sequence length
                        continue
                    
                    # Extract data
                    seq_len = len(valid_indices)
                    student_questions = questions_shifted[i][student_mask].cpu().numpy()
                    student_responses = responses_shifted[i][student_mask].cpu().numpy()
                    student_mastery = skill_mastery[i][student_mask].cpu().numpy()
                    student_gains = skill_gains[i][student_mask].cpu().numpy()
                    student_predictions = predictions[i][student_mask].cpu().numpy()
                    
                    # Create trajectory
                    trajectory = {
                        'student_id': student_count,
                        'sequence_length': seq_len,
                        'questions': student_questions,
                        'responses': student_responses,
                        'mastery_states': student_mastery,  # Shape: (seq_len, num_concepts)
                        'learning_gains': student_gains,   # Shape: (seq_len, num_concepts)
                        'predictions': student_predictions,
                        'mean_mastery': np.mean(student_mastery, axis=1),
                        'mean_gains': np.mean(student_gains, axis=1),
                        'performance_rate': np.mean(student_responses),
                        'monotonicity_check': self._check_monotonicity(student_mastery),
                        'bounds_check': self._check_bounds(student_mastery, student_gains)
                    }
                    
                    trajectories.append(trajectory)
                    student_count += 1
                    
                    if student_count % 10 == 0:
                        self.logger.info(f"  Extracted {student_count} trajectories...")
        
        self.logger.info(f"✓ Extracted {len(trajectories)} learning trajectories")
        return trajectories
    
    def _check_monotonicity(self, mastery_states: np.ndarray) -> Dict:
        """Check monotonicity violations in mastery progression."""
        seq_len, num_concepts = mastery_states.shape
        violations = 0
        total_checks = 0
        
        for concept in range(num_concepts):
            for t in range(1, seq_len):
                if mastery_states[t, concept] < mastery_states[t-1, concept]:
                    violations += 1
                total_checks += 1
        
        violation_rate = violations / total_checks if total_checks > 0 else 0.0
        
        return {
            'violations': violations,
            'total_checks': total_checks,
            'violation_rate': violation_rate,
            'is_monotonic': violation_rate == 0.0
        }
    
    def _check_bounds(self, mastery_states: np.ndarray, learning_gains: np.ndarray) -> Dict:
        """Check bounds violations in mastery and gains."""
        mastery_violations = np.sum((mastery_states < 0) | (mastery_states > 1))
        gain_violations = np.sum(learning_gains < 0)
        
        total_mastery_entries = mastery_states.size
        total_gain_entries = learning_gains.size
        
        return {
            'mastery_violations': mastery_violations,
            'mastery_violation_rate': mastery_violations / total_mastery_entries,
            'gain_violations': gain_violations,
            'gain_violation_rate': gain_violations / total_gain_entries,
            'bounds_satisfied': mastery_violations == 0 and gain_violations == 0
        }
    
    def plot_individual_trajectory(self, trajectory: Dict, save_name: Optional[str] = None) -> str:
        """Create detailed visualization of a single student's learning trajectory."""
        student_id = trajectory['student_id']
        
        if save_name is None:
            save_name = f"student_{student_id}_trajectory.png"
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)
        
        seq_len = trajectory['sequence_length']
        timesteps = range(seq_len)
        
        # 1. Mastery Evolution Over Time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(timesteps, trajectory['mean_mastery'], 
                marker='o', linewidth=3, markersize=8, color=self.colors['mastery'],
                label='Mean Mastery Level')
        
        # Highlight correct/incorrect responses
        for t, (response, prediction) in enumerate(zip(trajectory['responses'], trajectory['predictions'])):
            color = self.colors['correct'] if response == 1 else self.colors['incorrect']
            alpha = 0.3 + 0.4 * prediction  # Opacity based on confidence
            ax1.axvline(x=t, color=color, alpha=alpha, linewidth=2)
        
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Mastery Level', fontsize=12)
        ax1.set_title(f'Student {student_id}: Skill Mastery Evolution\\n'
                     f'(Performance Rate: {trajectory["performance_rate"]:.1%})', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, color=self.colors['grid'])
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()
        
        # Add monotonicity indicator
        monotonic_status = "✓ Monotonic" if trajectory['monotonicity_check']['is_monotonic'] else "✗ Non-monotonic"
        ax1.text(0.02, 0.95, monotonic_status, transform=ax1.transAxes, 
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='lightgreen' if trajectory['monotonicity_check']['is_monotonic'] else 'lightcoral'))
        
        # 2. Learning Gains Over Time
        ax2 = fig.add_subplot(gs[0, 1])
        gains = trajectory['mean_gains']
        bars = ax2.bar(timesteps, gains, color=self.colors['gains'], alpha=0.7, 
                      edgecolor='darkred', linewidth=1)
        
        # Color bars by performance
        for bar, response in zip(bars, trajectory['responses']):
            if response == 1:
                bar.set_color(self.colors['correct'])
                bar.set_alpha(0.8)
            else:
                bar.set_color(self.colors['incorrect'])
                bar.set_alpha(0.6)
        
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Learning Gains', fontsize=12)
        ax2.set_title('Learning Gains per Time Step\\n(Green: Correct, Red: Incorrect)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, color=self.colors['grid'])
        ax2.set_ylim(bottom=0)
        
        # 3. Mastery vs Performance Correlation
        ax3 = fig.add_subplot(gs[1, :])
        
        # Scatter plot of mastery vs performance
        colors_scatter = [self.colors['correct'] if r == 1 else self.colors['incorrect'] 
                         for r in trajectory['responses']]
        
        scatter = ax3.scatter(trajectory['mean_mastery'], trajectory['predictions'], 
                            c=colors_scatter, s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add correlation line
        correlation = np.corrcoef(trajectory['mean_mastery'], trajectory['predictions'])[0, 1]
        if not np.isnan(correlation):
            z = np.polyfit(trajectory['mean_mastery'], trajectory['predictions'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(trajectory['mean_mastery'].min(), trajectory['mean_mastery'].max(), 100)
            ax3.plot(x_line, p(x_line), "--", color='purple', linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Mastery Level', fontsize=12)
        ax3.set_ylabel('Performance Prediction', fontsize=12)
        ax3.set_title(f'Mastery-Performance Relationship (r={correlation:.3f})', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, color=self.colors['grid'])
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        
        # 4. Summary Statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create summary table
        summary_data = [
            ['Sequence Length', f"{seq_len} steps"],
            ['Performance Rate', f"{trajectory['performance_rate']:.1%}"],
            ['Mean Mastery', f"{np.mean(trajectory['mean_mastery']):.3f}"],
            ['Mean Gains', f"{np.mean(trajectory['mean_gains']):.3f}"],
            ['Monotonicity', "✓ Perfect" if trajectory['monotonicity_check']['is_monotonic'] else "✗ Violations"],
            ['Bounds Check', "✓ Valid" if trajectory['bounds_check']['bounds_satisfied'] else "✗ Violations"],
            ['Mastery-Perf Correlation', f"{correlation:.3f}" if not np.isnan(correlation) else "N/A"]
        ]
        
        table = ax4.table(cellText=summary_data, 
                         colWidths=[0.3, 0.2],
                         cellLoc='left',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            table[(i, 0)].set_facecolor('#E6E6FA')  # Lavender
            table[(i, 1)].set_facecolor('#F0F8FF')  # Alice blue
        
        plt.suptitle(f'Educational Learning Trajectory Analysis - Student {student_id}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Individual trajectory saved: {save_path}")
        return save_path
    
    def plot_consistency_dashboard(self, trajectories: List[Dict], save_name: str = "consistency_dashboard.png") -> str:
        """Create comprehensive consistency dashboard across all students."""
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # Extract metrics
        monotonicity_rates = [t['monotonicity_check']['violation_rate'] for t in trajectories]
        mastery_violation_rates = [t['bounds_check']['mastery_violation_rate'] for t in trajectories]
        gain_violation_rates = [t['bounds_check']['gain_violation_rate'] for t in trajectories]
        performance_rates = [t['performance_rate'] for t in trajectories]
        mean_masteries = [np.mean(t['mean_mastery']) for t in trajectories]
        
        # Calculate correlations
        correlations = []
        for t in trajectories:
            corr = np.corrcoef(t['mean_mastery'], t['predictions'])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        # 1. Monotonicity Violations Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(monotonicity_rates, bins=20, color=self.colors['monotonicity'], 
                alpha=0.7, edgecolor='black', linewidth=1)
        ax1.axvline(x=np.mean(monotonicity_rates), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(monotonicity_rates):.1%}')
        ax1.set_xlabel('Monotonicity Violation Rate', fontsize=11)
        ax1.set_ylabel('Number of Students', fontsize=11)
        ax1.set_title('Monotonicity Consistency\\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Mastery Bounds Violations
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(mastery_violation_rates, bins=20, color=self.colors['mastery'], 
                alpha=0.7, edgecolor='black', linewidth=1)
        ax2.axvline(x=np.mean(mastery_violation_rates), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(mastery_violation_rates):.1%}')
        ax2.set_xlabel('Mastery Bounds Violation Rate', fontsize=11)
        ax2.set_ylabel('Number of Students', fontsize=11)
        ax2.set_title('Mastery Bounds [0,1]\\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Gain Non-negativity Violations
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(gain_violation_rates, bins=20, color=self.colors['gains'], 
                alpha=0.7, edgecolor='black', linewidth=1)
        ax3.axvline(x=np.mean(gain_violation_rates), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(gain_violation_rates):.1%}')
        ax3.set_xlabel('Negative Gains Rate', fontsize=11)
        ax3.set_ylabel('Number of Students', fontsize=11)
        ax3.set_title('Non-negative Gains\\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Mastery-Performance Correlations
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(correlations, bins=20, color=self.colors['performance'], 
                alpha=0.7, edgecolor='black', linewidth=1)
        ax4.axvline(x=np.mean(correlations), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(correlations):.3f}')
        ax4.set_xlabel('Mastery-Performance Correlation', fontsize=11)
        ax4.set_ylabel('Number of Students', fontsize=11)
        ax4.set_title('Performance Alignment\\n(Higher is Better)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance vs Mastery Scatter
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(mean_masteries, performance_rates, 
                            c=correlations, cmap='RdYlGn', s=80, alpha=0.7, 
                            edgecolors='black', linewidth=1)
        
        # Add correlation line
        overall_corr = np.corrcoef(mean_masteries, performance_rates)[0, 1]
        if not np.isnan(overall_corr):
            z = np.polyfit(mean_masteries, performance_rates, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(mean_masteries), max(mean_masteries), 100)
            ax5.plot(x_line, p(x_line), "--", color='purple', linewidth=2)
        
        ax5.set_xlabel('Mean Mastery Level', fontsize=11)
        ax5.set_ylabel('Performance Rate', fontsize=11)
        ax5.set_title(f'Mastery vs Performance\\n(Overall r={overall_corr:.3f})', 
                     fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Individual Correlation', rotation=270, labelpad=20)
        
        # 6. Consistency Summary Metrics
        ax6 = fig.add_subplot(gs[1, 2])
        
        perfect_students = sum(1 for t in trajectories 
                              if t['monotonicity_check']['is_monotonic'] 
                              and t['bounds_check']['bounds_satisfied'])
        
        metrics = [
            f"Students Analyzed: {len(trajectories)}",
            f"Perfect Consistency: {perfect_students}/{len(trajectories)} ({perfect_students/len(trajectories):.1%})",
            f"Mean Monotonicity Violations: {np.mean(monotonicity_rates):.1%}",
            f"Mean Bounds Violations: {np.mean(mastery_violation_rates):.1%}",
            f"Mean Negative Gains: {np.mean(gain_violation_rates):.1%}",
            f"Mean Mastery-Perf Correlation: {np.mean(correlations):.3f}",
            f"Strong Correlations (>0.3): {sum(1 for c in correlations if c > 0.3)}/{len(correlations)}"
        ]
        
        ax6.text(0.05, 0.95, '\\n'.join(metrics), transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        # 7. Learning Trajectory Heatmap
        ax7 = fig.add_subplot(gs[2, :])
        
        # Create trajectory matrix for heatmap
        max_len = max(t['sequence_length'] for t in trajectories)
        trajectory_matrix = np.full((len(trajectories), max_len), np.nan)
        
        for i, trajectory in enumerate(trajectories):
            seq_len = trajectory['sequence_length']
            trajectory_matrix[i, :seq_len] = trajectory['mean_mastery']
        
        # Create heatmap
        im = ax7.imshow(trajectory_matrix, aspect='auto', cmap='RdYlBu_r', 
                       vmin=0, vmax=1, interpolation='nearest')
        
        ax7.set_xlabel('Time Step', fontsize=11)
        ax7.set_ylabel('Student ID', fontsize=11)
        ax7.set_title('Learning Trajectories Heatmap (Mastery Evolution)', 
                     fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('Mastery Level', rotation=270, labelpad=20)
        
        plt.suptitle('Educational Consistency Dashboard\\n'
                    'GainAKT2Monitored with Cumulative Mastery', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Consistency dashboard saved: {save_path}")
        return save_path
    
    def plot_concept_mastery_evolution(self, trajectories: List[Dict], save_name: str = "concept_mastery_evolution.png") -> str:
        """Visualize mastery evolution for different concepts/skills."""
        
        # Select a representative student with good sequence length
        best_trajectory = max(trajectories, key=lambda t: t['sequence_length'])
        student_id = best_trajectory['student_id']
        
        mastery_states = best_trajectory['mastery_states']  # (seq_len, num_concepts)
        seq_len, num_concepts = mastery_states.shape
        
        # Select top concepts (most variable or highest mastery)
        concept_variance = np.var(mastery_states, axis=0)
        top_concepts = np.argsort(concept_variance)[-10:]  # Top 10 most variable
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Individual Concept Evolution
        ax1 = fig.add_subplot(gs[0, :])
        
        timesteps = range(seq_len)
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_concepts)))
        
        for i, concept_idx in enumerate(top_concepts):
            mastery_curve = mastery_states[:, concept_idx]
            ax1.plot(timesteps, mastery_curve, marker='o', linewidth=2, 
                    markersize=4, color=colors[i], 
                    label=f'Concept {concept_idx}', alpha=0.8)
        
        # Highlight correct/incorrect responses
        for t, response in enumerate(best_trajectory['responses']):
            color = self.colors['correct'] if response == 1 else self.colors['incorrect']
            ax1.axvline(x=t, color=color, alpha=0.2, linewidth=3)
        
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Mastery Level', fontsize=12)
        ax1.set_title(f'Individual Concept Mastery Evolution - Student {student_id}\\n'
                     f'(Green bg: Correct answers, Red bg: Incorrect answers)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(-0.05, 1.05)
        
        # 2. Concept Mastery Heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        
        im = ax2.imshow(mastery_states[:, top_concepts].T, aspect='auto', 
                       cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
        
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Concept ID', fontsize=12)
        ax2.set_title('Concept Mastery Heatmap', fontsize=14, fontweight='bold')
        ax2.set_yticks(range(len(top_concepts)))
        ax2.set_yticklabels([f'C{idx}' for idx in top_concepts])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Mastery Level', rotation=270, labelpad=20)
        
        # 3. Mastery Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate mastery statistics
        initial_mastery = mastery_states[0, top_concepts]
        final_mastery = mastery_states[-1, top_concepts]
        mastery_gains = final_mastery - initial_mastery
        
        x_pos = np.arange(len(top_concepts))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, initial_mastery, width, 
                       label='Initial Mastery', color='lightblue', 
                       edgecolor='navy', linewidth=1)
        bars2 = ax3.bar(x_pos + width/2, final_mastery, width,
                       label='Final Mastery', color='darkblue', 
                       edgecolor='navy', linewidth=1)
        
        ax3.set_xlabel('Concept ID', fontsize=12)
        ax3.set_ylabel('Mastery Level', fontsize=12)
        ax3.set_title('Initial vs Final Mastery', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'C{idx}' for idx in top_concepts], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Concept-Level Learning Analysis\\n'
                    'Demonstrating Monotonic Skill Progression', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Concept mastery evolution saved: {save_path}")
        return save_path
    
    def generate_educational_report(self, trajectories: List[Dict], 
                                  save_name: str = "educational_interpretability_report.json") -> str:
        """Generate comprehensive educational interpretability report."""
        
        self.logger.info("Generating comprehensive educational report...")
        
        # Calculate aggregate statistics
        total_students = len(trajectories)
        
        # Consistency metrics
        perfect_monotonic = sum(1 for t in trajectories if t['monotonicity_check']['is_monotonic'])
        perfect_bounds = sum(1 for t in trajectories if t['bounds_check']['bounds_satisfied'])
        
        monotonicity_rates = [t['monotonicity_check']['violation_rate'] for t in trajectories]
        mastery_violation_rates = [t['bounds_check']['mastery_violation_rate'] for t in trajectories]
        gain_violation_rates = [t['bounds_check']['gain_violation_rate'] for t in trajectories]
        
        # Performance metrics
        correlations = []
        for t in trajectories:
            corr = np.corrcoef(t['mean_mastery'], t['predictions'])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        strong_correlations = sum(1 for c in correlations if c > 0.3)
        performance_rates = [t['performance_rate'] for t in trajectories]
        
        # Create comprehensive report
        report = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'total_students_analyzed': total_students,
                'model_type': 'GainAKT2Monitored_CumulativeMastery'
            },
            'consistency_analysis': {
                'monotonicity': {
                    'perfect_students': perfect_monotonic,
                    'perfect_percentage': perfect_monotonic / total_students,
                    'mean_violation_rate': np.mean(monotonicity_rates),
                    'std_violation_rate': np.std(monotonicity_rates),
                    'max_violation_rate': np.max(monotonicity_rates),
                    'assessment': 'PERFECT' if np.mean(monotonicity_rates) == 0 else 'NEEDS_IMPROVEMENT'
                },
                'mastery_bounds': {
                    'perfect_students': perfect_bounds,
                    'perfect_percentage': perfect_bounds / total_students,
                    'mean_violation_rate': np.mean(mastery_violation_rates),
                    'std_violation_rate': np.std(mastery_violation_rates),
                    'max_violation_rate': np.max(mastery_violation_rates),
                    'assessment': 'PERFECT' if np.mean(mastery_violation_rates) == 0 else 'NEEDS_IMPROVEMENT'
                },
                'non_negative_gains': {
                    'perfect_students': sum(1 for r in gain_violation_rates if r == 0),
                    'perfect_percentage': sum(1 for r in gain_violation_rates if r == 0) / total_students,
                    'mean_violation_rate': np.mean(gain_violation_rates),
                    'std_violation_rate': np.std(gain_violation_rates),
                    'max_violation_rate': np.max(gain_violation_rates),
                    'assessment': 'PERFECT' if np.mean(gain_violation_rates) == 0 else 'NEEDS_IMPROVEMENT'
                }
            },
            'performance_analysis': {
                'mastery_performance_correlation': {
                    'mean_correlation': np.mean(correlations),
                    'std_correlation': np.std(correlations),
                    'strong_correlations_count': strong_correlations,
                    'strong_correlations_percentage': strong_correlations / total_students,
                    'assessment': 'STRONG' if np.mean(correlations) > 0.3 else 'MODERATE' if np.mean(correlations) > 0.1 else 'WEAK'
                },
                'overall_performance': {
                    'mean_performance_rate': np.mean(performance_rates),
                    'std_performance_rate': np.std(performance_rates),
                    'min_performance_rate': np.min(performance_rates),
                    'max_performance_rate': np.max(performance_rates)
                }
            },
            'educational_validity': {
                'interpretability_score': self._calculate_interpretability_score(
                    np.mean(monotonicity_rates), np.mean(mastery_violation_rates), 
                    np.mean(gain_violation_rates), np.mean(correlations)
                ),
                'consistency_grade': self._grade_consistency(
                    np.mean(monotonicity_rates), np.mean(mastery_violation_rates), 
                    np.mean(gain_violation_rates)
                ),
                'correlation_grade': self._grade_correlations(np.mean(correlations)),
                'overall_assessment': self._overall_assessment(trajectories)
            },
            'detailed_statistics': {
                'sequence_lengths': [t['sequence_length'] for t in trajectories],
                'individual_correlations': correlations,
                'individual_performance_rates': performance_rates,
                'monotonicity_violation_rates': monotonicity_rates,
                'mastery_bounds_violation_rates': mastery_violation_rates,
                'gain_violation_rates': gain_violation_rates
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
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
        
        report = convert_numpy_types(report)
        
        # Save report
        save_path = os.path.join(self.output_dir, save_name)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✓ Educational report saved: {save_path}")
        return save_path
    
    def _calculate_interpretability_score(self, mono_rate: float, bounds_rate: float, 
                                        gains_rate: float, correlation: float) -> float:
        """Calculate overall interpretability score (0-100)."""
        consistency_score = 100 * (1 - (mono_rate + bounds_rate + gains_rate) / 3)
        correlation_score = 100 * max(0, correlation)  # Normalize positive correlations
        
        return 0.7 * consistency_score + 0.3 * correlation_score
    
    def _grade_consistency(self, mono_rate: float, bounds_rate: float, gains_rate: float) -> str:
        """Grade consistency performance."""
        total_violation_rate = (mono_rate + bounds_rate + gains_rate) / 3
        
        if total_violation_rate == 0:
            return 'A+'
        elif total_violation_rate < 0.01:
            return 'A'
        elif total_violation_rate < 0.05:
            return 'B'
        elif total_violation_rate < 0.10:
            return 'C'
        else:
            return 'F'
    
    def _grade_correlations(self, correlation: float) -> str:
        """Grade correlation strength."""
        if correlation >= 0.5:
            return 'A+'
        elif correlation >= 0.4:
            return 'A'
        elif correlation >= 0.3:
            return 'B'
        elif correlation >= 0.2:
            return 'C'
        else:
            return 'F'
    
    def _overall_assessment(self, trajectories: List[Dict]) -> str:
        """Provide overall educational assessment."""
        total_students = len(trajectories)
        
        perfect_consistency = sum(1 for t in trajectories 
                                 if t['monotonicity_check']['is_monotonic'] 
                                 and t['bounds_check']['bounds_satisfied'])
        
        correlations = []
        for t in trajectories:
            corr = np.corrcoef(t['mean_mastery'], t['predictions'])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        strong_correlations = sum(1 for c in correlations if c > 0.3)
        
        if perfect_consistency == total_students and strong_correlations / total_students > 0.7:
            return "EXCELLENT: Perfect consistency with strong correlations"
        elif perfect_consistency == total_students:
            return "VERY_GOOD: Perfect consistency achieved"
        elif perfect_consistency / total_students > 0.8:
            return "GOOD: High consistency with room for improvement"
        else:
            return "NEEDS_IMPROVEMENT: Consistency violations present"
    
    def create_complete_visualization_suite(self, num_students: int = 20) -> Dict[str, str]:
        """Generate complete suite of educational visualizations."""
        
        self.logger.info("="*60)
        self.logger.info("GENERATING COMPLETE EDUCATIONAL VISUALIZATION SUITE")
        self.logger.info("="*60)
        
        # Load data and extract trajectories
        self.load_data()
        trajectories = self.extract_learning_trajectories(num_students)
        
        if len(trajectories) == 0:
            raise ValueError("No valid learning trajectories found")
        
        # Generate all visualizations
        outputs = {}
        
        # 1. Individual trajectory examples (first 3 students)
        self.logger.info("Creating individual trajectory visualizations...")
        for i in range(min(3, len(trajectories))):
            trajectory = trajectories[i]
            path = self.plot_individual_trajectory(trajectory, 
                                                  f"student_{trajectory['student_id']}_detailed_trajectory.png")
            outputs[f'individual_trajectory_{i+1}'] = path
        
        # 2. Consistency dashboard
        self.logger.info("Creating consistency dashboard...")
        outputs['consistency_dashboard'] = self.plot_consistency_dashboard(trajectories)
        
        # 3. Concept mastery evolution
        self.logger.info("Creating concept mastery evolution...")
        outputs['concept_mastery_evolution'] = self.plot_concept_mastery_evolution(trajectories)
        
        # 4. Educational report
        self.logger.info("Generating educational interpretability report...")
        outputs['educational_report'] = self.generate_educational_report(trajectories)
        
        self.logger.info("="*60)
        self.logger.info("✅ COMPLETE VISUALIZATION SUITE GENERATED!")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info(f"📊 Visualizations created: {len(outputs)}")
        self.logger.info(f"👥 Students analyzed: {len(trajectories)}")
        self.logger.info("="*60)
        
        return outputs


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate educational visualizations')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='educational_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_students', type=int, default=20,
                       help='Number of students to analyze')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EducationalVisualizer(args.model_path, args.output_dir)
    
    # Generate complete suite
    outputs = visualizer.create_complete_visualization_suite(args.num_students)
    
    print("\\n🎉 Visualization generation completed!")
    print(f"📁 Check output directory: {args.output_dir}")
    for name, path in outputs.items():
        print(f"  📊 {name}: {path}")


if __name__ == "__main__":
    main()