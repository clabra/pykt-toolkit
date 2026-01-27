#!/usr/bin/env python3
"""
Generate per-skill prediction CONFIDENCE heatmaps using H1.3 composite metric.

Uses composite confidence score (calibrated + directional + percentile) instead of
simple concordance (1 - MAE) to provide more nuanced trust assessment.

Usage:
    python examples/results/generate_skill_alignment_heatmap_h13.py \
        --exp_dir experiments/20260126_212614_ablation-none-nblocks-4-numattnheads-4_268444/gtransformer/assist2009/fold_0_173167 \
        --output_dir examples/results/confidence_heatmap_268444
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from scipy import stats

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


def load_predictions_from_file(pred_file):
    """
    Load predictions from qid_test_question_predictions_supervised.txt or _reference.txt
    
    Returns DataFrame with columns:
    - orirow (student ID)
    - qidx (question index)
    - concepts (comma-separated skill IDs)
    - concept_preds (comma-separated predictions per skill)
    """
    df = pd.read_csv(pred_file, sep='\t')
    print(f"✓ Loaded {len(df):,} question-level records from {os.path.basename(pred_file)}")
    return df


def explode_skill_predictions(df, pred_col='concept_preds'):
    """
    Explode multi-skill questions into one row per skill.
    
    Input row:
        orirow=0, concepts='20,47,46', concept_preds='0.05,0.08,0.03'
    
    Output rows:
        orirow=0, skill=20, pred=0.05
        orirow=0, skill=47, pred=0.08
        orirow=0, skill=46, pred=0.03
    """
    records = []
    
    for _, row in df.iterrows():
        student_id = row['orirow']
        concepts_str = str(row['concepts'])
        preds_str = str(row[pred_col])
        
        # Parse comma-separated values
        try:
            skills = [int(s) for s in concepts_str.split(',')]
            preds = [float(p) for p in preds_str.split(',')]
            
            if len(skills) != len(preds):
                continue  # Skip malformed rows
            
            for skill, pred in zip(skills, preds):
                records.append({
                    'student_id': student_id,
                    'skill_id': skill,
                    'prediction': pred
                })
        except (ValueError, AttributeError):
            continue  # Skip rows with parsing errors
    
    exploded_df = pd.DataFrame(records)
    print(f"✓ Exploded to {len(exploded_df):,} skill-level predictions")
    return exploded_df


def calculate_composite_confidence(p_ref_values, p_sup_values, all_disagreements_sorted=None):
    """
    Calculate H1.3 composite confidence metric for a group of predictions.
    
    Composite confidence combines:
    1. Calibrated confidence (exponential decay of disagreement)
    2. Directional agreement (binary decision consistency)
    3. Percentile-based confidence (relative ranking)
    
    Args:
        p_ref_values: Array of interpretable predictions
        p_sup_values: Array of supervised predictions  
        all_disagreements_sorted: Pre-sorted array of all disagreements (for percentile calculation)
    
    Returns:
        Mean composite confidence score [0, 1]
    """
    p_ref = np.array(p_ref_values)
    p_sup = np.array(p_sup_values)
    
    # 1. Disagreement and calibrated confidence
    disagreement = np.abs(p_ref - p_sup)
    calibrated_confidence = np.exp(-2 * disagreement)
    
    # 2. Directional agreement (same pass/fail decision)
    p_ref_binary = (p_ref >= 0.5).astype(int)
    p_sup_binary = (p_sup >= 0.5).astype(int)
    directional_agreement = (p_ref_binary == p_sup_binary).astype(float)
    
    # 3. Percentile-based confidence (OPTIMIZED: use searchsorted instead of loop)
    if all_disagreements_sorted is not None:
        # Fast percentile calculation using binary search
        n = len(all_disagreements_sorted)
        percentile_ranks = np.searchsorted(all_disagreements_sorted, disagreement, side='right') / n * 100
        # Invert so high agreement = high percentile
        confidence_percentile = 100 - percentile_ranks
    else:
        # Fallback to simple normalization if sorted array not provided
        confidence_percentile = np.zeros(len(disagreement))
    
    # 4. Composite confidence (weighted combination)
    composite_confidence = (
        0.4 * calibrated_confidence +           # How close to p_sup
        0.3 * directional_agreement +           # Same binary decision
        0.3 * (confidence_percentile / 100.0)   # Relative to other predictions
    )
    
    return np.mean(composite_confidence)


def compute_confidence_heatmap(df_sup, df_ref, min_interactions=5, top_skills=50, top_students=30):
    """
    Compute H1.3 composite confidence between p_sup and p_ref predictions.
    
    Args:
        df_sup: DataFrame with student_id, skill_id, prediction (p_sup)
        df_ref: DataFrame with student_id, skill_id, prediction (p_ref)
        min_interactions: Minimum number of interactions required per student-skill pair
        top_skills: Number of most active skills to include
        top_students: Number of most active students to include
    
    Returns:
        pivot: Heatmap data (students x skills)
        stats: Summary statistics
    """
    # Merge supervised and reference predictions
    df_sup = df_sup.rename(columns={'prediction': 'p_sup'})
    df_ref = df_ref.rename(columns={'prediction': 'p_ref'})
    
    merged = pd.merge(
        df_sup, df_ref,
        on=['student_id', 'skill_id'],
        how='inner'
    )
    
    print(f"✓ Merged {len(merged):,} matched predictions")
    
    # Calculate and sort all disagreements for fast percentile calculation
    all_disagreements = np.abs(merged['p_sup'].values - merged['p_ref'].values)
    all_disagreements_sorted = np.sort(all_disagreements)
    print(f"✓ Pre-sorted {len(all_disagreements_sorted):,} disagreements for percentile calculation")
    
    # Filter for robust sequences (min interactions per student-skill pair)
    counts = merged.groupby(['student_id', 'skill_id']).size().reset_index(name='T')
    df_robust = pd.merge(
        merged,
        counts[counts['T'] >= min_interactions][['student_id', 'skill_id']],
        on=['student_id', 'skill_id']
    )
    
    print(f"✓ Filtered to {len(df_robust):,} predictions with T >= {min_interactions}")
    
    if df_robust.empty:
        print(f"⚠️  No sequences with T >= {min_interactions}, trying T >= 3")
        df_robust = pd.merge(
            merged,
            counts[counts['T'] >= 3][['student_id', 'skill_id']],
            on=['student_id', 'skill_id']
        )
        min_interactions = 3
    
    if df_robust.empty:
        print(f"⚠️  No sequences with T >= 3, using all data")
        df_robust = merged
        min_interactions = 1
    
    # Select top skills and students by interaction count (max density)
    skill_counts = df_robust.groupby('skill_id').size().nlargest(top_skills)
    student_counts = df_robust.groupby('student_id').size().nlargest(top_students)
    
    df_sample = df_robust[
        df_robust['skill_id'].isin(skill_counts.index) &
        df_robust['student_id'].isin(student_counts.index)
    ]
    
    print(f"✓ Max-Density Sampling: {len(student_counts)} students, {len(skill_counts)} skills")
    print(f"✓ Sample size: {len(df_sample):,} predictions")
    
    # Compute composite confidence per (student, skill) pair using optimized calculation
    def compute_confidence(group):
        p_ref_vals = group['p_ref'].values
        p_sup_vals = group['p_sup'].values
        return calculate_composite_confidence(p_ref_vals, p_sup_vals, all_disagreements_sorted)
    
    results = df_sample.groupby(['student_id', 'skill_id']).apply(
        compute_confidence
    ).reset_index(name='confidence')
    
    # Create pivot table for heatmap
    pivot = results.pivot_table(
        index='student_id',
        columns='skill_id',
        values='confidence'
    )
    
    # Compute summary statistics
    stats = {
        'mean_confidence': results['confidence'].mean(),
        'median_confidence': results['confidence'].median(),
        'std_confidence': results['confidence'].std(),
        'min_confidence': results['confidence'].min(),
        'max_confidence': results['confidence'].max(),
        'n_students': len(student_counts),
        'n_skills': len(skill_counts),
        'n_pairs': len(results),
        'min_interactions': min_interactions,
        'high_confidence_pct': (results['confidence'] >= 0.8).mean() * 100,
        'medium_confidence_pct': ((results['confidence'] >= 0.5) & (results['confidence'] < 0.8)).mean() * 100,
        'low_confidence_pct': (results['confidence'] < 0.5).mean() * 100,
    }
    
    print(f"\n{'='*80}")
    print("CONFIDENCE STATISTICS (H1.3 Composite Metric)")
    print(f"{'='*80}")
    print(f"Mean Confidence:   {stats['mean_confidence']:.3f}")
    print(f"Median Confidence: {stats['median_confidence']:.3f}")
    print(f"Std Confidence:    {stats['std_confidence']:.3f}")
    print(f"Range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
    print(f"\nConfidence Categories:")
    print(f"  🟢 High (≥0.8):   {stats['high_confidence_pct']:>5.1f}%")
    print(f"  🟡 Medium (0.5-0.8): {stats['medium_confidence_pct']:>5.1f}%")
    print(f"  🔴 Low (<0.5):    {stats['low_confidence_pct']:>5.1f}%")
    print(f"\nStudent-Skill Pairs: {stats['n_pairs']:,}")
    
    return pivot, stats


def plot_confidence_heatmap(pivot, stats, output_path):
    """
    Create confidence heatmap with discrete color zones.
    
    Color zones:
    - Green [0.80-1.0]: High confidence (p_ref trustworthy)
    - Yellow [0.65-0.80]: Medium confidence (use with caution)
    - Orange [0.50-0.65]: Low-medium confidence  
    - Red [<0.50]: Low confidence (p_ref unreliable)
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Discrete colormap for confidence zones
    colors_hex = ["#c0392b", "#e67e22", "#f39c12", "#27ae60"]
    cmap_conf = ListedColormap(colors_hex)
    bounds = [0.0, 0.50, 0.65, 0.80, 1.0]
    norm = BoundaryNorm(bounds, cmap_conf.N)
    
    # Plot heatmap
    sns.heatmap(
        pivot,
        cmap=cmap_conf,
        norm=norm,
        ax=ax,
        cbar_kws={'label': 'H1.3 Composite Confidence', 'pad': 0.08}
    )
    
    ax.set_xlabel(f'Skills (Top {stats["n_skills"]} by Density)', fontsize=12)
    ax.set_ylabel(f'Students (Top {stats["n_students"]} by Density)', fontsize=12)
    ax.set_title(
        'Prediction Confidence Heatmap',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Add discrete legend
    legend_patches = [
        mpatches.Patch(color='#27ae60', label='[0.80 - 1.0]: High Confidence'),
        mpatches.Patch(color='#f39c12', label='[0.65 - 0.80]: Medium Confidence'),
        mpatches.Patch(color='#e67e22', label='[0.50 - 0.65]: Low-Medium Confidence'),
        mpatches.Patch(color='#c0392b', label='[< 0.50]: Low Confidence')
    ]
    
    legend = plt.legend(
        handles=legend_patches,
        title="Prediction Trust Scale",
        loc='center left',
        bbox_to_anchor=(1.25, 0.5),
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        shadow=True
    )
    legend.get_frame().set_facecolor('#fdfdfd')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved heatmap to {output_path}")
    plt.close()


def plot_confidence_distribution(df_sup, df_ref, output_path):
    """
    Plot distribution of confidence scores and disagreements.
    """
    # Merge predictions
    df_sup = df_sup.rename(columns={'prediction': 'p_sup'})
    df_ref = df_ref.rename(columns={'prediction': 'p_ref'})
    
    merged = pd.merge(
        df_sup, df_ref,
        on=['student_id', 'skill_id'],
        how='inner'
    )
    
    # Calculate metrics (OPTIMIZED: vectorized calculation)
    disagreement = np.abs(merged['p_sup'] - merged['p_ref'])
    all_disagreements_sorted = np.sort(disagreement.values)
    
    # Calculate composite confidence for all predictions (vectorized)
    p_ref_arr = merged['p_ref'].values
    p_sup_arr = merged['p_sup'].values
    
    # 1. Calibrated confidence
    calibrated_confidence = np.exp(-2 * disagreement.values)
    
    # 2. Directional agreement
    p_ref_binary = (p_ref_arr >= 0.5).astype(int)
    p_sup_binary = (p_sup_arr >= 0.5).astype(int)
    directional_agreement = (p_ref_binary == p_sup_binary).astype(float)
    
    # 3. Percentile-based confidence
    n = len(all_disagreements_sorted)
    percentile_ranks = np.searchsorted(all_disagreements_sorted, disagreement.values, side='right') / n * 100
    confidence_percentile = 100 - percentile_ranks
    
    # 4. Composite confidence
    confidence_scores = (
        0.4 * calibrated_confidence +
        0.3 * directional_agreement +
        0.3 * (confidence_percentile / 100.0)
    )
    
    merged['confidence'] = confidence_scores
    merged['disagreement'] = disagreement
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram of confidence scores
    ax1.hist(merged['confidence'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(merged['confidence'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {merged["confidence"].mean():.3f}')
    ax1.axvline(merged['confidence'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {merged["confidence"].median():.3f}')
    # Add threshold lines
    ax1.axvline(0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='High threshold (0.8)')
    ax1.axvline(0.5, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7, label='Low threshold (0.5)')
    ax1.set_xlabel('H1.3 Composite Confidence', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Confidence Scores', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of disagreement
    ax2.hist(merged['disagreement'], bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax2.axvline(merged['disagreement'].mean(), color='darkred', linestyle='--', linewidth=2, 
                label=f'Mean: {merged["disagreement"].mean():.3f}')
    ax2.axvline(merged['disagreement'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {merged["disagreement"].median():.3f}')
    ax2.set_xlabel('Disagreement |p_ref - p_sup|', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Prediction Disagreement', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: p_sup vs p_ref (colored by confidence)
    sample = merged.sample(min(10000, len(merged)), random_state=42)
    scatter = ax3.scatter(sample['p_ref'], sample['p_sup'], 
                         c=sample['confidence'], cmap='RdYlGn', 
                         alpha=0.5, s=20, vmin=0, vmax=1)
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Agreement')
    ax3.set_xlabel('$p_{ref}$ (Interpretable)', fontsize=12)
    ax3.set_ylabel('$p_{sup}$ (Supervised)', fontsize=12)
    ax3.set_title('Prediction Agreement (colored by confidence)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Confidence', rotation=270, labelpad=15)
    
    # 4. Confidence vs Disagreement scatter
    sample2 = merged.sample(min(10000, len(merged)), random_state=42)
    ax4.scatter(sample2['disagreement'], sample2['confidence'], alpha=0.3, s=10, color='#9b59b6')
    ax4.set_xlabel('Disagreement |p_ref - p_sup|', fontsize=12)
    ax4.set_ylabel('H1.3 Composite Confidence', fontsize=12)
    ax4.set_title('Confidence vs Disagreement', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0.8, color='green', linestyle=':', alpha=0.7, label='High threshold')
    ax4.axhline(0.5, color='red', linestyle=':', alpha=0.7, label='Low threshold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate per-skill prediction confidence heatmaps (H1.3 metric)')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Experiment directory (e.g., experiments/.../fold_0_173167)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for heatmaps')
    parser.add_argument('--min_interactions', type=int, default=5,
                       help='Minimum interactions per student-skill pair (default: 5)')
    parser.add_argument('--top_skills', type=int, default=50,
                       help='Number of top skills to include (default: 50)')
    parser.add_argument('--top_students', type=int, default=30,
                       help='Number of top students to include (default: 30)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("PER-SKILL PREDICTION CONFIDENCE ANALYSIS (H1.3 Metric)")
    print(f"{'='*80}")
    print(f"Experiment: {args.exp_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Min interactions: {args.min_interactions}")
    print(f"Top skills: {args.top_skills}")
    print(f"Top students: {args.top_students}")
    print(f"\nMetric: H1.3 Composite Confidence")
    print(f"  - 40% Calibrated confidence (exp decay of disagreement)")
    print(f"  - 30% Directional agreement (same pass/fail decision)")
    print(f"  - 30% Percentile-based confidence (relative ranking)")
    print(f"{'='*80}\n")
    
    # Load prediction files
    sup_file = os.path.join(args.exp_dir, 'qid_test_question_predictions_supervised.txt')
    ref_file = os.path.join(args.exp_dir, 'qid_test_question_predictions_reference.txt')
    
    if not os.path.exists(sup_file):
        raise FileNotFoundError(f"Supervised predictions not found: {sup_file}")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference predictions not found: {ref_file}")
    
    # Load and explode predictions
    print("Loading supervised predictions (p_sup)...")
    df_sup_raw = load_predictions_from_file(sup_file)
    df_sup = explode_skill_predictions(df_sup_raw, pred_col='concept_preds')
    
    print("\nLoading reference predictions (p_ref)...")
    df_ref_raw = load_predictions_from_file(ref_file)
    df_ref = explode_skill_predictions(df_ref_raw, pred_col='concept_preds')
    
    # Compute confidence heatmap
    print("\nComputing confidence heatmap...")
    pivot, stats = compute_confidence_heatmap(
        df_sup, df_ref,
        min_interactions=args.min_interactions,
        top_skills=args.top_skills,
        top_students=args.top_students
    )
    
    # Plot heatmap
    heatmap_path = os.path.join(args.output_dir, 'h13_skill_confidence_heatmap.png')
    plot_confidence_heatmap(pivot, stats, heatmap_path)
    
    # Plot distribution
    dist_path = os.path.join(args.output_dir, 'h13_skill_confidence_distribution.png')
    plot_confidence_distribution(df_sup, df_ref, dist_path)
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, 'h13_confidence_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - h13_skill_confidence_heatmap.png")
    print(f"  - h13_skill_confidence_distribution.png")
    print(f"  - h13_confidence_statistics.json")
    print(f"\n💡 Interpretation:")
    print(f"  🟢 Green cells: High confidence - p_ref is trustworthy")
    print(f"  🟡 Yellow cells: Medium confidence - use with caution")
    print(f"  🔴 Red cells: Low confidence - p_ref unreliable, consider p_sup")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
