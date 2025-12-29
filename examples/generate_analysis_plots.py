#!/usr/bin/env python3
"""
Generate comprehensive analysis plots for iKT experiments.

Plots generated:
1. Loss Evolution: L_total, L1, L2_penalty, L2 over epochs (4 subplots)
2. AUC vs Violation Rate: Pareto frontier visualization
3. Deviation Histogram: Distribution of |Mi - M_rasch| with epsilon threshold
4. Per-Skill Alignment: Heatmap of MSE per skill across students

Usage:
    python examples/generate_analysis_plots.py --run_dir experiments/20251128_123456_ikt_test_abcdef
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import pearsonr
import pickle

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')


def load_metrics_csv(run_dir):
    """
    Load metrics CSV. Supports:
    1. legacy iKT: metrics_validation.csv
    2. modern iDKT: metrics_epoch.csv
    """
    # Try multiple possible file names
    possible_files = ['metrics_validation.csv', 'metrics_epoch.csv']
    csv_path = None
    for f in possible_files:
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        raise FileNotFoundError(f"Neither metrics_validation.csv nor metrics_epoch.csv found in {run_dir}")
    
    print(f"✓ Loading metrics from: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    
    # Map iDKT names to internal plotting names if necessary
    mapping = {
        'valid_auc': 'val_auc',
        'valid_acc': 'val_accuracy',
        'train_loss': 'train_total_loss',
        # For iDKT, we don't have separate val_total_loss in the same CSV usually, 
        # but we can use train_loss as a proxy or just skip if missing.
    }
    
    for old_col, new_col in mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
            
    # Normalize prefixes for iDKT metrics if present
    idkt_normalization = {
        'train_l_sup': 'l_sup',
        'train_l_ref': 'l_ref',
        'train_l_init': 'l_init',
        'train_l_rate': 'l_rate',
        'l_initmastery': 'l_init' # Backward compatibility
    }
    for old_col, tgt_col in idkt_normalization.items():
        if old_col in df.columns and tgt_col not in df.columns:
            df[tgt_col] = df[old_col]
        elif old_col in df.columns:
             df[tgt_col] = df[old_col] # Ensure target is always present if source is
            
    # Check for comprehensive metrics format (iKT style)
    required_cols_ikt = [
        'epoch', 'phase', 'val_l1_bce', 'val_auc', 'val_accuracy',
        'val_l2_mse', 'val_penalty_loss', 'val_violation_rate',
        'val_total_loss', 'train_l1_bce', 'train_l2_mse', 'train_penalty_loss'
    ]
    
    # Check for comprehensive metrics format (iDKT style)
    required_cols_idkt = [
        'epoch', 'train_loss', 'valid_auc', 'valid_acc', 'l_sup', 'l_ref', 'l_init', 'l_rate'
    ]
    
    has_ikt = all(col in df.columns for col in required_cols_ikt)
    has_idkt = all(col in df.columns for col in required_cols_idkt)
    
    if has_ikt:
        print(f"✓ Detected comprehensive iKT metrics format")
        return df, "ikt"
    elif has_idkt:
        print(f"✓ Detected comprehensive iDKT metrics format")
        return df, "idkt"
    else:
        # Check if we have minimal columns for basic plots
        minimal_cols = ['epoch', 'val_auc']
        if all(col in df.columns for col in minimal_cols):
            print(f"⚠️  Incomplete metrics detected (Missing detailed pillars)")
            print(f"   Available columns: {list(df.columns)}")
            print(f"   Will generate basic plots only (AUC trends)")
            return df, "basic"
        else:
            raise ValueError(
                f"CSV missing required columns for basic plots: {['epoch', 'val_auc']}\n"
                f"Available columns: {list(df.columns)}\n"
                f"This script requires at least 'epoch' and 'val_auc' (or 'valid_auc') columns."
            )


def load_mastery_states(run_dir, split='test'):
    """Load mastery states CSV if it exists (handles multiple naming conventions)."""
    # Map splits to possible file names
    possible_paths = {
        'test': ['traj_initmastery.csv', 'initmastery_trajectory.csv', 'mastery_test.csv'],
        'trajectory': ['traj_predictions.csv', 'predictions_trajectory.csv', 'mastery_trajectory.csv'],
        'rate': ['traj_rate.csv', 'rate_trajectory.csv', 'mastery_rate.csv']
    }
    
    possible_names = possible_paths.get(split, [f'mastery_{split}.csv'])
    mastery_path = None
    
    for name in possible_names:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            mastery_path = path
            break
            
    if not mastery_path:
        print(f"⚠️  Alignment data for '{split}' not found, skipping plot")
        return None
    
    df = pd.read_csv(mastery_path)
    print(f"✓ Loaded {len(df)} records from {os.path.basename(mastery_path)}")
    return df


def load_config(run_dir):
    """Load experiment config to get epsilon value."""
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        print("⚠️  config.json not found, using default epsilon=0.05")
        return {'epsilon': 0.05, 'lambda_penalty': 100.0}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    epsilon = config.get('epsilon', 0.05)
    lambda_penalty = config.get('lambda_penalty', 100.0)
    print(f"✓ Loaded config: epsilon={epsilon}, lambda_penalty={lambda_penalty}")
    return {'epsilon': epsilon, 'lambda_penalty': lambda_penalty}


def load_bkt_params(bkt_params_path):
    """Load BKT parameters from pickle file."""
    if not bkt_params_path or not os.path.exists(bkt_params_path):
        print(f"⚠️  BKT params file not found: {bkt_params_path}")
        return None
    
    try:
        with open(bkt_params_path, 'rb') as f:
            data = pickle.load(f)
        params = data.get('params', {})
        print(f"✓ Loaded BKT parameters for {len(params)} skills")
        return params
    except Exception as e:
        print(f"⚠️  Error loading BKT params: {e}")
        return None


def filter_skills_by_bkt(mastery_df, bkt_params, guess_threshold=0.3, slip_threshold=0.3):
    """Filter out skills with extreme BKT parameters."""
    valid_skills = [s for s, p in bkt_params.items() 
                   if p.get('guesses', 0) <= guess_threshold 
                   and p.get('slips', 0) <= slip_threshold]
    filtered_df = mastery_df[mastery_df['skill_id'].isin(valid_skills)]
    return filtered_df


def plot_auc_trend_simple(df, output_path, config):
    """
    Simple AUC trend plot for minimal data scenarios.
    Works with just 'epoch' and 'val_auc' columns.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = df['epoch'].values
    val_auc = df['val_auc'].values
    
    ax.plot(epochs, val_auc, 'o-', linewidth=2, markersize=8, color='#2ecc71', label='Validation AUC')
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Validation AUC Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Add value annotations if few epochs
    if len(epochs) <= 10:
        for i, (e, auc) in enumerate(zip(epochs, val_auc)):
            ax.annotate(f'{auc:.4f}', 
                       xy=(e, auc), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       ha='center', 
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add best AUC marker
    best_idx = val_auc.argmax()
    best_epoch = epochs[best_idx]
    best_auc = val_auc[best_idx]
    ax.plot(best_epoch, best_auc, '*', markersize=20, color='gold', 
            markeredgecolor='darkgoldenrod', markeredgewidth=2, 
            label=f'Best: {best_auc:.4f} @ Epoch {best_epoch}', zorder=10)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_ikt_loss_evolution(df, output_path, config):
    """
    Plot 1: Loss Evolution (4 subplots) for legacy iKT
    - L_total over epochs
    - L1 (BCE) over epochs
    - L2_penalty over epochs
    - L2_MSE over epochs
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Identify phase transition
    phase_transition = None
    if 'phase' in df.columns:
        phase_changes = df[df['phase'].diff() != 0]
        if len(phase_changes) > 0:
            phase_transition = phase_changes.index[0] if phase_changes.index[0] > 0 else None
    
    # Subplot 1: Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['epoch'], df['val_total_loss'], 'o-', label='Validation', linewidth=2, markersize=4)
    ax1.plot(df['epoch'], df['train_total_loss'], 's-', label='Training', linewidth=2, markersize=3, alpha=0.7)
    if phase_transition:
        ax1.axvline(x=df.loc[phase_transition, 'epoch'], color='red', linestyle='--', alpha=0.5, label='Phase Transition')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Total Loss', fontsize=11)
    ax1.set_title('L_total: Overall Optimization Objective', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: L1 (BCE Loss)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['epoch'], df['val_l1_bce'], 'o-', label='Validation', linewidth=2, markersize=4, color='green')
    ax2.plot(df['epoch'], df['train_l1_bce'], 's-', label='Training', linewidth=2, markersize=3, alpha=0.7, color='lightgreen')
    if phase_transition:
        ax2.axvline(x=df.loc[phase_transition, 'epoch'], color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('BCE Loss', fontsize=11)
    ax2.set_title('L1: Binary Cross-Entropy (Performance)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: L2_penalty
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['epoch'], df['val_penalty_loss'], 'o-', label='Validation', linewidth=2, markersize=4, color='orange')
    ax3.plot(df['epoch'], df['train_penalty_loss'], 's-', label='Training', linewidth=2, markersize=3, alpha=0.7, color='moccasin')
    if phase_transition:
        ax3.axvline(x=df.loc[phase_transition, 'epoch'], color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Penalty Loss', fontsize=11)
    ax3.set_title('L2_penalty: Constraint Violation Penalty', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: L2_MSE
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['epoch'], df['val_l2_mse'], 'o-', label='Validation', linewidth=2, markersize=4, color='purple')
    ax4.plot(df['epoch'], df['train_l2_mse'], 's-', label='Training', linewidth=2, markersize=3, alpha=0.7, color='plum')
    if phase_transition:
        ax4.axvline(x=df.loc[phase_transition, 'epoch'], color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('MSE vs Rasch', fontsize=11)
    ax4.set_title('L2: Alignment with Rasch Targets', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    epsilon = config['epsilon']
    lambda_penalty = config['lambda_penalty']
    fig.suptitle(f'Loss Component Evolution (λ_penalty={lambda_penalty}, ε={epsilon})', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_idkt_loss_evolution(df, output_path, config):
    """
    Plot Theory-Guided Loss Evolution for iDKT
    - L_SUP: Supervised loss
    - L_REF: Prediction alignment loss
    - L_IM: Initial Mastery alignment loss
    - L_RT: Learning Rate alignment loss
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Supervised Loss (L_SUP)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['epoch'], df['l_sup'], 'o-', label='L_SUP (BCE)', linewidth=2, markersize=4, color='#2980b9')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Supervised Performance (L_SUP)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Prediction Alignment (L_REF)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['epoch'], df['l_ref'], 's-', label='L_REF (MSE)', linewidth=2, markersize=4, color='#27ae60')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Prediction Alignment (L_REF)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Initial Mastery Alignment (L_IM)
    ax3 = fig.add_subplot(gs[1, 0])
    # Note: 'l_init' is the unified name for initial mastery alignment
    ax3.plot(df['epoch'], df['l_init'], 'd-', label='L_IM (MSE)', linewidth=2, markersize=4, color='#8e44ad')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Initial Mastery Consistency (L_IM)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Learning Rate Alignment (L_RT)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['epoch'], df['l_rate'], 'v-', label='L_RT (MSE)', linewidth=2, markersize=4, color='#d35400')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Learning Rate Consistency (L_RT)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('iDKT Theory-Guided Loss Component Evolution', fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_auc_vs_violations(df, output_path, config):
    """
    Plot 2: AUC vs Violation Rate
    Pareto frontier showing performance-interpretability trade-off
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert violation_rate to percentage
    df['val_violation_pct'] = df['val_violation_rate'] * 100
    
    # Color by epoch (gradient from early to late)
    scatter = ax.scatter(df['val_violation_pct'], df['val_auc'], 
                        c=df['epoch'], cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch', fontsize=11)
    
    # Highlight best AUC point
    best_auc_idx = df['val_auc'].idxmax()
    ax.scatter(df.loc[best_auc_idx, 'val_violation_pct'], 
              df.loc[best_auc_idx, 'val_auc'],
              s=300, facecolors='none', edgecolors='red', linewidth=3, 
              label=f'Best AUC (Epoch {df.loc[best_auc_idx, "epoch"]})')
    
    # Add target zones
    ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Good AUC Threshold (0.75)')
    ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='Violation Target (5%)')
    
    # Shade optimal region (AUC>0.75, violations<5%)
    ax.fill_between([0, 5], 0.75, 1.0, alpha=0.1, color='green', label='Optimal Region')
    
    ax.set_xlabel('Violation Rate (%)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Performance vs Interpretability Trade-off (λ_penalty={config["lambda_penalty"]}, ε={config["epsilon"]})', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=max(0.5, df['val_auc'].min() - 0.05), top=min(1.0, df['val_auc'].max() + 0.02))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_deviation_histogram(df, output_path, config):
    """
    Plot 3: Deviation Histogram
    Distribution of |Mi - M_rasch| with epsilon threshold line
    Uses validation metrics across all epochs
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epsilon = config['epsilon']
    
    # We'll plot histograms for 4 key epochs: first, phase transition, mid-phase2, final
    phase_transition_idx = None
    if 'phase' in df.columns:
        phase_changes = df[df['phase'].diff() != 0]
        if len(phase_changes) > 0 and phase_changes.index[0] > 0:
            phase_transition_idx = phase_changes.index[0]
    
    # Select 4 epochs to visualize
    epochs_to_plot = []
    labels = []
    
    # Epoch 1 (or earliest)
    epochs_to_plot.append(0)
    labels.append(f"Epoch {df.loc[0, 'epoch']} (Initial)")
    
    # Phase transition (if exists)
    if phase_transition_idx:
        epochs_to_plot.append(phase_transition_idx)
        labels.append(f"Epoch {df.loc[phase_transition_idx, 'epoch']} (Phase Transition)")
    
    # Mid-training
    mid_idx = len(df) // 2
    epochs_to_plot.append(mid_idx)
    labels.append(f"Epoch {df.loc[mid_idx, 'epoch']} (Mid-training)")
    
    # Final epoch
    epochs_to_plot.append(len(df) - 1)
    labels.append(f"Epoch {df.loc[len(df)-1, 'epoch']} (Final)")
    
    # If we don't have 4 unique epochs, pad with duplicates
    while len(epochs_to_plot) < 4:
        epochs_to_plot.append(len(df) - 1)
        labels.append(f"Epoch {df.loc[len(df)-1, 'epoch']}")
    
    # For histogram, we need actual deviation data
    # Since we only have aggregated metrics, we'll simulate distribution
    # based on mean_violation, max_violation, and violation_rate
    
    for idx, (epoch_idx, label) in enumerate(zip(epochs_to_plot[:4], labels[:4])):
        ax = axes[idx // 2, idx % 2]
        
        row = df.iloc[epoch_idx]
        
        # Simulate deviation distribution
        # We know: violation_rate, mean_violation (of violations), max_violation
        violation_rate = row['val_violation_rate']
        mean_violation = row['val_mean_violation']
        max_violation = row['val_max_violation']
        
        # Generate synthetic deviation data for visualization
        # Assumption: most deviations are near 0, with tail extending to max_violation
        n_samples = 10000
        
        # Generate deviations: mix of small deviations (below epsilon) and violations
        n_violations = int(n_samples * violation_rate)
        n_compliant = n_samples - n_violations
        
        # Compliant deviations: uniform/normal distribution below epsilon
        if n_compliant > 0:
            compliant_devs = np.abs(np.random.normal(0, epsilon/3, n_compliant))
            compliant_devs = np.clip(compliant_devs, 0, epsilon)
        else:
            compliant_devs = np.array([])
        
        # Violations: exponential distribution from epsilon to max
        if n_violations > 0 and max_violation > epsilon:
            violation_excess = np.random.exponential(mean_violation if mean_violation > 0 else 0.02, n_violations)
            violation_devs = epsilon + np.clip(violation_excess, 0, max_violation - epsilon)
        else:
            violation_devs = np.array([])
        
        all_devs = np.concatenate([compliant_devs, violation_devs])
        
        # Plot histogram
        ax.hist(all_devs, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add epsilon threshold line
        ax.axvline(x=epsilon, color='red', linestyle='--', linewidth=2, label=f'ε = {epsilon}')
        
        # Shade violation region
        ylim = ax.get_ylim()
        ax.fill_betweenx(ylim, epsilon, ax.get_xlim()[1], alpha=0.2, color='red', label='Violation Region')
        ax.set_ylim(ylim)
        
        # Add statistics text
        stats_text = f'Violation Rate: {violation_rate*100:.1f}%\n'
        stats_text += f'Mean Violation: {mean_violation:.3f}\n'
        stats_text += f'Max Violation: {max_violation:.3f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('|Mi - M_rasch|', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Deviation Distribution Evolution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def calculate_ccc(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient (CCC).
    Combines precision (correlation) and accuracy (bias).
    """
    if len(y_true) < 2: return 0.0
    
    # 1. Pearson Correlation
    from scipy.stats import pearsonr
    try:
        r, _ = pearsonr(y_true, y_pred)
    except:
        r = 0.0 # Occurs if variance is zero
        
    if np.isnan(r): r = 0.0
    
    # 2. Means and Variances
    mu_true, mu_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.sqrt(var_true), np.sqrt(var_pred)
    
    # 3. Formula: 2*r*s1*s2 / (s1^2 + s2^2 + (m1-m2)^2)
    numerator = 2 * r * sd_true * sd_pred
    denominator = var_true + var_pred + (mu_true - mu_pred)**2
    
    if denominator == 0: return 1.0 # Exact match including constant
    return numerator / denominator


def plot_per_skill_alignment(mastery_df, output_path, config, seed=42):
    """
    Plot 4: Pedagogical Confidence Zone Heatmap
    Treats iDKT as the 'Observer' to determine if BKT prediction is in/out of confidence.
    """
    if mastery_df is None:
        print("⚠️  Skipping per-skill alignment plot (no mastery states data)")
        return
    
    # Identify descriptive names
    new_pairs = [('p_idkt', 'p_bkt'), ('idkt_im', 'bkt_im'), ('idkt_rate', 'bkt_rate')]
    mi_col, rasch_col = None, None
    
    for c1, c2 in new_pairs:
        if c1 in mastery_df.columns and c2 in mastery_df.columns:
            mi_col, rasch_col = c1, c2
            break
            
    if mi_col is None: return
    
    print(f"📊 Generating Scientific Concordance Heatmap: {mi_col} vs {rasch_col}")
    
    # 1. Filter for Robust sequences (T >= 8) to ensure longitudinal evidence
    counts = mastery_df.groupby(['student_id', 'skill_id']).size().reset_index(name='T')
    df_robust = pd.merge(mastery_df, counts[counts['T'] >= 8][['student_id', 'skill_id']], on=['student_id', 'skill_id'])
    
    if df_robust.empty:
        print("⚠️  No sequences with T >= 8, using T >= 3 fallback")
        df_robust = pd.merge(mastery_df, counts[counts['T'] >= 3][['student_id', 'skill_id']], on=['student_id', 'skill_id'])

    # 2. Max-Density Selection (Visual Clarity & Best Evidence)
    # Pick Top 50 Skills and Top 30 Students with most interactions
    # This ensures every cell has a deep longitudinal history and minimizes sparsity
    top_skills = mastery_df.groupby('skill_id').size().nlargest(50).index
    top_students = mastery_df.groupby('student_id').size().nlargest(30).index
    
    df_sample = df_robust[df_robust['skill_id'].isin(top_skills) & df_robust['student_id'].isin(top_students)]
    
    print(f"   Max-Density Sampling: {len(top_students)} students, {len(top_skills)} skills")

    # 3. Compute Trajectory Concordance (1 - |MAE|) per (Student, Skill)
    def group_concordance(group):
        mae = (group[rasch_col].values - group[mi_col].values).__abs__().mean()
        return 1.0 - mae
            
    results = df_sample.groupby(['student_id', 'skill_id']).apply(group_concordance).reset_index(name='c')
    pivot = results.pivot_table(index='student_id', columns='skill_id', values='c')
    
    # Create Heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Discrete Colormap for Concordance Zones
    # [Red (<0.65), Orange (0.65-0.80), Yellow (0.80-0.90), Green (0.90-1.0)]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
    
    colors_hex = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]
    cmap_con = ListedColormap(colors_hex)
    bounds = [0.0, 0.65, 0.80, 0.90, 1.0]
    norm = BoundaryNorm(bounds, cmap_con.N)

    # Plot Heatmap
    sns.heatmap(pivot, cmap=cmap_con, norm=norm, ax=ax, 
                cbar_kws={'label': 'Student Error', 'pad': 0.08})
    
    ax.set_xlabel('Knowledge Components (Top 50 by Density)', fontsize=12)
    ax.set_ylabel('Student ID (Top 30 by Density)', fontsize=12)
    ax.set_title(f'Scientific Concordance Heatmap: Longitudinal Validation Scope\n'
                 f'iDKT Observer vs. BKT Baseline ({mi_col.upper()})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 2. Add Discrete Legend
    legend_patches = [
        mpatches.Patch(color='#27ae60', label='[0.90 - 1.0]: Scientific Concordance (Total Alignment)'),
        mpatches.Patch(color='#f1c40f', label='[0.80 - 0.9]: Pedagogical Grounding (Rule Consistent)'),
        mpatches.Patch(color='#e67e22', label='[0.65 - 0.8]: Measured Discovery (Individualized Path)'),
        mpatches.Patch(color='#c0392b', label='[< 0.65]: Discovery Breakout (New Knowledge Structure)')
    ]
    
    legend = plt.legend(handles=legend_patches, title="Interpretability Alignment Scale",
                       loc='center left', bbox_to_anchor=(1.35, 0.5), 
                       fontsize=10, title_fontsize=11, frameon=True, shadow=True)
    legend.get_frame().set_facecolor('#fdfdfd')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8) 
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
def plot_student_intervention_analysis(mastery_df, output_root, config, prefix, seed=42):
    """
    Educator-Focused Plot:
    1. Histogram of % Curriculum Outside Confidence (Population View)
    2. Stacked Bar Chart of Top-Risk Students (Individual Triage)
    """
    if mastery_df is None: return

    # Identify columns
    new_pairs = [('p_idkt', 'p_bkt'), ('idkt_im', 'bkt_im'), ('idkt_rate', 'bkt_rate')]
    mi_col, rasch_col = None, None
    for c1, c2 in new_pairs:
        if c1 in mastery_df.columns and c2 in mastery_df.columns:
            mi_col, rasch_col = c1, c2
            break
    if mi_col is None: return

    # 1. Functional Baseline Filter (The "Long Tail" Problem)
    # Only count skills with enough global evidence to be considered a stable BKT theory
    global_counts = mastery_df.groupby('skill_id').size()
    top_grounded_skills = global_counts.nlargest(100).index.tolist()
    
    # 2. Compute Longitudinal Consensus for every (S, K) pair
    # Use robust filtering (T >= 5) for student-level aggregation
    counts = mastery_df.groupby(['student_id', 'skill_id']).size().reset_index(name='T')
    mask = (counts['T'] >= 5) & (counts['skill_id'].isin(top_grounded_skills))
    df_robust = pd.merge(mastery_df, counts[mask][['student_id', 'skill_id']], on=['student_id', 'skill_id'])
    
    if df_robust.empty:
        print("⚠️  Filtering too strict for triage, using all skills")
        df_robust = pd.merge(mastery_df, counts[counts['T'] >= 3][['student_id', 'skill_id']], on=['student_id', 'skill_id'])

    def get_c(group):
        return 1.0 - (group[rasch_col].values - group[mi_col].values).__abs__().mean()
        
    sk_concordance = df_robust.groupby(['student_id', 'skill_id']).apply(get_c).reset_index(name='c')

    # 3. Assign Zones (Unified Pedagogical Thresholds)
    # MUST MATCH plot_per_skill_alignment for visual consistency
    def categorize(c):
        if c >= 0.90: return 'High'     # Green
        if c >= 0.80: return 'Marginal' # Yellow
        if c >= 0.65: return 'Low'      # Orange
        return 'Breakout'               # Red
        
    sk_concordance['zone'] = sk_concordance['c'].apply(categorize)

    # 4. Aggregate per Student
    student_stats = sk_concordance.groupby(['student_id', 'zone']).size().unstack(fill_value=0)
    # Ensure all columns exist
    for zone in ['High', 'Marginal', 'Low', 'Breakout']:
        if zone not in student_stats.columns: student_stats[zone] = 0
    
    student_stats['total_skills'] = student_stats.sum(axis=1)
    for zone in ['High', 'Marginal', 'Low', 'Breakout']:
        student_stats[f'pct_{zone}'] = (student_stats[zone] / student_stats['total_skills']) * 100
        
    # Attention Needed is defined as Low + Breakout
    student_stats['pct_attention_needed'] = student_stats['pct_Low'] + student_stats['pct_Breakout']
    
    # 5. PLOT A: Population Distribution of Attention Need
    plt.figure(figsize=(10, 6))
    # Filter out students with very little data (less than 3 skills practiced in the Top 100)
    top_distribution = student_stats[student_stats['total_skills'] >= 3]
    
    sns.histplot(top_distribution['pct_attention_needed'], bins=15, kde=True, color='#e67e22')
    mean_val = top_distribution['pct_attention_needed'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean Risk: {mean_val:.1f}%')
    
    plt.title('Educator Population Overview: Intervention Risk\n(% of Core Curriculum Outside Theoretical Confidence)', fontsize=13, fontweight='bold')
    plt.xlabel('% of Core KC Journeys with Low/Breakout Concordance (<0.80)', fontsize=11)
    plt.ylabel('Number of Students', fontsize=11)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    pop_path = os.path.join(output_root, f'student_attention_distribution_{prefix}.png')
    plt.savefig(pop_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Refined Educator Plot: {pop_path}")
    plt.close()

    # 6. PLOT B: Student Triage (Top 30 Students by Interaction Density)
    # Sync with Heatmap sampling: Use total interaction count
    top_risk = student_stats.nlargest(30, 'total_skills')
        
    # Reorder columns for stacked bar
    plot_data = top_risk[['pct_Breakout', 'pct_Low', 'pct_Marginal', 'pct_High']]
    
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]
    ax = plot_data.plot(kind='bar', stacked=True, figsize=(15, 7), color=colors, edgecolor='black', linewidth=0.5)
    
    plt.title('Student Intervention Triage: Top 30 Students by Interaction Density\n(Aggregated Results for High-Confidence Longitudinal Journeys)', fontsize=13, fontweight='bold')
    plt.ylabel('% of student\'s Core Curriculum', fontsize=11)
    plt.xlabel('Student ID (Ranked by Total Interaction Density)', fontsize=11)
    plt.legend(['Breakout (Red, <0.65)', 'Low Confidence (Orange, <0.80)', 'Marginal (Yellow, <0.90)', 'High Fidelity (Green)'], 
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.2)
    
    triage_path = os.path.join(output_root, f'student_triage_risk_profiles_{prefix}.png')
    plt.savefig(triage_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Refined Educator Plot: {triage_path}")
    plt.close()

def plot_per_skill_bar_chart(df_agg, metric_col, output_path, title, ylabel, color='#2980b9', threshold=None, threshold_label="Safe/High"):
    """
    Generic bar chart for per-skill metrics, ranked.
    """
    if df_agg is None or df_agg.empty:
        return
        
    df_sorted = df_agg.sort_values(metric_col, ascending=False)
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(df_sorted)), df_sorted[metric_col], color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 1. Add Threshold Line
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.9, label=f'Threshold: {threshold}')
        # Highlight "Good" area
        # plt.axhspan(threshold, plt.ylim()[1], color='green', alpha=0.05)
        
        # Add labels for Good/Bad sides
        plt.text(len(df_sorted)-1, threshold + (plt.ylim()[1]*0.02), f"   {threshold_label}", color='red', fontweight='bold', verticalalignment='bottom', horizontalalignment='right')
        plt.text(len(df_sorted)-1, threshold - (plt.ylim()[1]*0.02), "   Limited/Discovery", color='gray', verticalalignment='top', horizontalalignment='right')

    # Label top and bottom 5 skills if many
    num_skills = len(df_sorted)
    if num_skills > 60:
        plt.xticks([]) # Hide x labels if too many
        plt.xlabel('Skills (Ranked)', fontsize=12)
    else:
        plt.xticks(range(num_skills), df_sorted.index, rotation=90, fontsize=8)
        plt.xlabel('Skill ID', fontsize=12)
        
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved Bar Chart: {output_path}")
    plt.close()

def plot_per_skill_probing_bars(run_dir, output_path):
    """Plot probing results if they exist."""
    ps_path = os.path.join(run_dir, 'probe_per_skill_results.json')
    if not os.path.exists(ps_path):
         # Try looking in subdirectory 'probing'
         ps_path = os.path.join(run_dir, 'probing', 'probe_per_skill_results.json')
         
    if not os.path.exists(ps_path):
        print(f"⚠️  Probing per-skill results not found at {ps_path}, skipping plot")
        return
        
    with open(ps_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame.from_dict(data, orient='index')
    if df.empty: return
    
    df.index.name = 'skill_id'
    
    # Threshold for probing: r > 0.4 is significant alignment in educational data
    plot_per_skill_bar_chart(df, 'pearson', output_path, 
                            "Per-Skill Latent Grounding (Probing Pearson r)", 
                            "Diagnostic Pearson Correlation", color='#27ae60',
                            threshold=0.4, threshold_label="Strong Grounding")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive analysis plots for iKT experiments')
    parser.add_argument('--run_dir', type=str, required=True,
                       help='Experiment directory containing metrics results')
    parser.add_argument('--filter_bkt', action='store_true',
                       help='Filter out skills with extreme BKT parameters')
    parser.add_argument('--guess_threshold', type=float, default=0.3,
                       help='Max guess rate allowed for a skill to be included (default: 0.3)')
    parser.add_argument('--slip_threshold', type=float, default=0.3,
                       help='Max slip rate allowed for a skill to be included (default: 0.3)')
    parser.add_argument('--bkt_params_path', type=str,
                       help='Path to bkt_skill_params.pkl')
    # New Plot Type Flags
    parser.add_argument('--plot_heatmap', type=int, default=1, help='Generate Student x Skill Heatmaps')
    parser.add_argument('--plot_correlation', type=int, default=1, help='Generate Per-Skill Correlation Bar Charts')
    parser.add_argument('--plot_variance', type=int, default=1, help='Generate Per-Skill Individualization Variance Bar Charts')
    parser.add_argument('--plot_probing', type=int, default=1, help='Generate Per-Skill Probing Bar Charts')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducible sampling')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.run_dir):
        print(f"❌ Error: Directory not found: {args.run_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("Generating Analysis Plots for iKT Experiment")
    print("=" * 80)
    print(f"Run directory: {args.run_dir}\n")
    
    # Create plots subdirectory
    plots_dir = os.path.join(args.run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"✓ Plots directory: {plots_dir}\n")
    
    try:
        metrics_df, format_type = load_metrics_csv(args.run_dir)
        config = load_config(args.run_dir)
        
        # BKT parameters for filtering (if requested)
        bkt_params = None
        if args.filter_bkt:
            bkt_params = load_bkt_params(args.bkt_params_path)
            if bkt_params:
                print(f"✓ Skills filtered by BKT bounds: g <= {args.guess_threshold}, s <= {args.slip_threshold}")
            else:
                print("⚠️  Filtering requested but BKT params could not be loaded. Skipping filter.")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Generating Plots")
    print("=" * 80 + "\n")
    
    # Generate plots
    try:
        if format_type == "ikt":
            # Plot 1: Loss Evolution (iKT version)
            print("1. Loss Evolution (iKT)...")
            plot_ikt_loss_evolution(metrics_df, os.path.join(plots_dir, 'loss_evolution.png'), config)
            
            # Plot 2: AUC vs Violations
            print("2. AUC vs Violation Rate...")
            plot_auc_vs_violations(metrics_df, os.path.join(plots_dir, 'auc_vs_violations.png'), config)
            
            # Plot 3: Deviation Histogram
            print("3. Deviation Histogram...")
            plot_deviation_histogram(metrics_df, os.path.join(plots_dir, 'deviation_histogram.png'), config)
            
            # Plot 4: Per-Skill Alignment
            print("4. Per-Skill Alignment Heatmap...")
            plot_per_skill_alignment(mastery_df, os.path.join(plots_dir, 'per_skill_alignment.png'), config)
            
        elif format_type == "idkt":
            # Plot 1: Loss Evolution (iDKT version)
            print("1. Loss Evolution (iDKT)...")
            plot_idkt_loss_evolution(metrics_df, os.path.join(plots_dir, 'loss_evolution.png'), config)
            
            # Plot 2: Per-Skill Alignment Heatmaps (multiple splits)
            print("2. Per-Skill Alignment Heatmaps...")
            alignment_splits = ['test', 'trajectory', 'rate']
            for split in alignment_splits:
                df_split = load_mastery_states(args.run_dir, split=split)
                if df_split is None:
                    continue
                
                # Apply filtering if requested
                if args.filter_bkt and bkt_params:
                    initial_skills = df_split['skill_id'].nunique()
                    df_split = filter_skills_by_bkt(df_split, bkt_params, 
                                                 args.guess_threshold, args.slip_threshold)
                    final_skills = df_split['skill_id'].nunique()
                    print(f"   Filtering {split}: {final_skills} skills remaining (from {initial_skills})")
                
                suffix = "_filtered" if args.filter_bkt else ""
                # Map internal split name to descriptive file name for plot
                plot_name_map = {
                    'test': 'initmastery',
                    'trajectory': 'predictions',
                    'rate': 'rate'
                }
                base_name = plot_name_map.get(split, split)
                
                # 1. Heatmaps
                if args.plot_heatmap:
                    plot_filename = f'per_skill_alignment_{base_name}{suffix}.png'
                    print(f"   Generating Heatmap: {plot_filename}...")
                    plot_per_skill_alignment(df_split, os.path.join(plots_dir, plot_filename), config, seed=args.seed)
                    
                    # Also generate Student-Level Triage Plots
                    plot_student_intervention_analysis(df_split, plots_dir, config, base_name, seed=args.seed)
                
                # Identify columns for bars
                new_pairs = [('p_idkt', 'p_bkt'), ('idkt_im', 'bkt_im'), ('idkt_rate', 'bkt_rate')]
                mi_col, rasch_col = None, None
                for c1, c2 in new_pairs:
                    if c1 in df_split.columns and c2 in df_split.columns:
                        mi_col, rasch_col = c1, c2; break
                
                if mi_col:
                    # 2. Correlation Bar Charts
                    if args.plot_correlation:
                        corr_agg = df_split.groupby('skill_id').apply(
                            lambda x: pearsonr(x[mi_col], x[rasch_col])[0] if len(x) > 5 and len(np.unique(x[rasch_col])) > 1 else np.nan
                        )
                        corr_agg = corr_agg.dropna()
                        if not corr_agg.empty:
                            plot_filename = f'per_skill_correlation_{base_name}{suffix}.png'
                            plot_per_skill_bar_chart(corr_agg.to_frame('r'), 'r', 
                                                    os.path.join(plots_dir, plot_filename),
                                                    f"Structural Fidelity: {base_name.replace('_',' ').title()} Alignment",
                                                    "Pearson Correlation (r)", color='#2980b9',
                                                    threshold=0.4, threshold_label="Strong Grounding")
                                                    
                    # 3. Individualization Volume Bar Charts
                    if args.plot_variance:
                        var_agg = df_split.groupby('skill_id')[mi_col].std()
                        var_agg = var_agg.dropna()
                        if not var_agg.empty:
                            # Context-aware threshold: Predictions vary more than static parameters
                            is_pred = 'p_idkt' in mi_col
                            v_thresh = 0.05 if is_pred else 0.0001
                            v_label = "Significant Discovery"
                            
                            plot_filename = f'per_skill_variance_{base_name}{suffix}.png'
                            plot_per_skill_bar_chart(var_agg.to_frame('std'), 'std',
                                                    os.path.join(plots_dir, plot_filename),
                                                    f"Individualization Volume: {base_name.replace('_',' ').title()} Nuance",
                                                    "Standard Deviation (σ)", color='#f39c12',
                                                    threshold=v_thresh, threshold_label=v_label)
                
            # 4. Probing Bar Chart
            if args.plot_probing:
                print("   Generating Probing Alignment Bar Chart...")
                plot_per_skill_probing_bars(args.run_dir, os.path.join(plots_dir, 'per_skill_probing_r.png'))
        else:
            print("⚠️  Limited data available - generating basic plots only")
            
            # Generate simple AUC trend if we have the data
            if 'val_auc' in metrics_df.columns and 'epoch' in metrics_df.columns:
                print("1. Simple AUC Trend...")
                plot_auc_trend_simple(metrics_df, os.path.join(plots_dir, 'auc_trend.png'), config)
            else:
                print("   Cannot generate any plots (missing 'epoch' and 'val_auc' columns)")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ All plots generated successfully")
    print("=" * 80)
    print(f"\nPlots saved in: {plots_dir}")


if __name__ == '__main__':
    main()
