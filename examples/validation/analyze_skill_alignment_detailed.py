#!/usr/bin/env python3
"""
Detailed per-skill alignment analysis for GTransformer.

Generates:
1. Per-skill alignment bar chart (sorted by concordance)
2. Per-skill envelope width statistics
3. Skill-level interpretability assessment

Usage:
    python examples/validation/analyze_skill_alignment_detailed.py \
        --exp_dir experiments/20260119_110013_orthogonal_diversity_baseline_801184/gtransformer/assist2009/fold_0_955042 \
        --output_dir examples/validation/results_exp801184_alignment
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


def load_and_merge_predictions(sup_file, ref_file):
    """Load and merge supervised and reference predictions at skill level."""
    # Load supervised
    df_sup_raw = pd.read_csv(sup_file, sep='\t')
    
    # Explode to skill level
    sup_records = []
    for _, row in df_sup_raw.iterrows():
        student_id = row['orirow']
        try:
            skills = [int(s) for s in str(row['concepts']).split(',')]
            preds = [float(p) for p in str(row['concept_preds']).split(',')]
            
            for skill, pred in zip(skills, preds):
                sup_records.append({
                    'student_id': student_id,
                    'skill_id': skill,
                    'p_sup': pred
                })
        except (ValueError, AttributeError):
            continue
    
    df_sup = pd.DataFrame(sup_records)
    
    # Load reference
    df_ref_raw = pd.read_csv(ref_file, sep='\t')
    
    # Explode to skill level
    ref_records = []
    for _, row in df_ref_raw.iterrows():
        student_id = row['orirow']
        try:
            skills = [int(s) for s in str(row['concepts']).split(',')]
            preds = [float(p) for p in str(row['concept_preds']).split(',')]
            
            for skill, pred in zip(skills, preds):
                ref_records.append({
                    'student_id': student_id,
                    'skill_id': skill,
                    'p_ref': pred
                })
        except (ValueError, AttributeError):
            continue
    
    df_ref = pd.DataFrame(ref_records)
    
    # Merge
    merged = pd.merge(df_sup, df_ref, on=['student_id', 'skill_id'], how='inner')
    merged['envelope'] = np.abs(merged['p_sup'] - merged['p_ref'])
    
    print(f"✓ Merged {len(merged):,} skill-level predictions")
    return merged


def analyze_per_skill(merged_df, min_samples=100):
    """
    Compute per-skill alignment metrics.
    
    For each skill:
    - Mean concordance (1 - MAE)
    - Mean envelope width
    - Std envelope width
    - Number of predictions
    """
    skill_stats = []
    
    for skill_id, group in merged_df.groupby('skill_id'):
        if len(group) < min_samples:
            continue
        
        mae = np.abs(group['p_sup'] - group['p_ref']).mean()
        concordance = 1.0 - mae
        
        skill_stats.append({
            'skill_id': skill_id,
            'concordance': concordance,
            'mae': mae,
            'envelope_mean': group['envelope'].mean(),
            'envelope_std': group['envelope'].std(),
            'envelope_median': group['envelope'].median(),
            'n_predictions': len(group)
        })
    
    df_stats = pd.DataFrame(skill_stats)
    df_stats = df_stats.sort_values('concordance', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"PER-SKILL ALIGNMENT STATISTICS ({len(df_stats)} skills with >= {min_samples} predictions)")
    print(f"{'='*80}")
    print(f"Mean Concordance: {df_stats['concordance'].mean():.3f}")
    print(f"Median Concordance: {df_stats['concordance'].median():.3f}")
    print(f"Std Concordance: {df_stats['concordance'].std():.3f}")
    print()
    print("Top 10 Best Aligned Skills:")
    print(df_stats.head(10)[['skill_id', 'concordance', 'envelope_mean', 'n_predictions']].to_string(index=False))
    print()
    print("Top 10 Worst Aligned Skills:")
    print(df_stats.tail(10)[['skill_id', 'concordance', 'envelope_mean', 'n_predictions']].to_string(index=False))
    
    return df_stats


def plot_per_skill_concordance(df_stats, output_path, top_n=50):
    """Bar chart of per-skill concordance."""
    df_top = df_stats.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color code by concordance level
    colors = []
    for c in df_top['concordance']:
        if c >= 0.90:
            colors.append('#27ae60')  # Green
        elif c >= 0.80:
            colors.append('#f1c40f')  # Yellow
        elif c >= 0.65:
            colors.append('#e67e22')  # Orange
        else:
            colors.append('#c0392b')  # Red
    
    bars = ax.barh(range(len(df_top)), df_top['concordance'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Set y-ticks to skill IDs
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels([f"Skill {int(sid)}" for sid in df_top['skill_id']], fontsize=8)
    
    ax.set_xlabel('Concordance (1 - MAE)', fontsize=12)
    ax.set_ylabel('Skill ID', fontsize=12)
    ax.set_title(
        f'Per-Skill Prediction Alignment (Top {top_n} by Concordance)\n'
        f'$p_{{sup}}$ (Supervised) vs $p_{{ref}}$ (Interpretable)',
        fontsize=14, fontweight='bold'
    )
    
    # Add vertical lines for thresholds
    ax.axvline(0.90, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent (0.90)')
    ax.axvline(0.80, color='#f1c40f', linestyle='--', linewidth=1.5, alpha=0.7, label='Good (0.80)')
    ax.axvline(0.65, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.7, label='Moderate (0.65)')
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-skill concordance chart to {output_path}")
    plt.close()


def plot_envelope_distribution_by_skill(merged_df, output_path, top_skills=20):
    """Box plot of envelope widths for top skills."""
    # Get top skills by prediction count
    skill_counts = merged_df.groupby('skill_id').size().nlargest(top_skills)
    df_top = merged_df[merged_df['skill_id'].isin(skill_counts.index)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create box plot
    skill_data = [df_top[df_top['skill_id'] == sid]['envelope'].values 
                  for sid in skill_counts.index]
    
    bp = ax.boxplot(skill_data, labels=[f"S{int(sid)}" for sid in skill_counts.index],
                    patch_artist=True, showmeans=True)
    
    # Color boxes by mean envelope
    for patch, skill_id in zip(bp['boxes'], skill_counts.index):
        mean_env = df_top[df_top['skill_id'] == skill_id]['envelope'].mean()
        if mean_env < 0.10:
            patch.set_facecolor('#27ae60')
        elif mean_env < 0.20:
            patch.set_facecolor('#f1c40f')
        else:
            patch.set_facecolor('#e67e22')
    
    ax.set_xlabel('Skill ID', fontsize=12)
    ax.set_ylabel('Envelope Width |p_sup - p_ref|', fontsize=12)
    ax.set_title(
        f'Prediction Envelope Distribution (Top {top_skills} Most Active Skills)\n'
        f'Lower values = Better alignment between supervised and interpretable predictions',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved envelope distribution chart to {output_path}")
    plt.close()


def categorize_skills(df_stats):
    """Categorize skills by alignment quality."""
    excellent = df_stats[df_stats['concordance'] >= 0.90]
    good = df_stats[(df_stats['concordance'] >= 0.80) & (df_stats['concordance'] < 0.90)]
    moderate = df_stats[(df_stats['concordance'] >= 0.65) & (df_stats['concordance'] < 0.80)]
    poor = df_stats[df_stats['concordance'] < 0.65]
    
    print(f"\n{'='*80}")
    print("SKILL ALIGNMENT CATEGORIES")
    print(f"{'='*80}")
    print(f"Excellent (≥0.90): {len(excellent)} skills ({len(excellent)/len(df_stats)*100:.1f}%)")
    print(f"Good (0.80-0.90): {len(good)} skills ({len(good)/len(df_stats)*100:.1f}%)")
    print(f"Moderate (0.65-0.80): {len(moderate)} skills ({len(moderate)/len(df_stats)*100:.1f}%)")
    print(f"Poor (<0.65): {len(poor)} skills ({len(poor)/len(df_stats)*100:.1f}%)")
    print(f"{'='*80}")
    
    return {
        'excellent': excellent,
        'good': good,
        'moderate': moderate,
        'poor': poor
    }


def main():
    parser = argparse.ArgumentParser(description='Detailed per-skill alignment analysis')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Experiment directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--min_samples', type=int, default=100,
                       help='Minimum predictions per skill (default: 100)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("DETAILED PER-SKILL ALIGNMENT ANALYSIS")
    print(f"{'='*80}\n")
    
    # Load predictions
    sup_file = os.path.join(args.exp_dir, 'qid_test_question_predictions_supervised.txt')
    ref_file = os.path.join(args.exp_dir, 'qid_test_question_predictions_reference.txt')
    
    print("Loading predictions...")
    merged_df = load_and_merge_predictions(sup_file, ref_file)
    
    # Compute per-skill statistics
    print("\nComputing per-skill statistics...")
    df_stats = analyze_per_skill(merged_df, min_samples=args.min_samples)
    
    # Categorize skills
    categories = categorize_skills(df_stats)
    
    # Save detailed statistics
    stats_path = os.path.join(args.output_dir, 'per_skill_alignment_stats.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"\n✓ Saved per-skill statistics to {stats_path}")
    
    # Plot per-skill concordance
    concordance_path = os.path.join(args.output_dir, 'per_skill_concordance.png')
    plot_per_skill_concordance(df_stats, concordance_path, top_n=50)
    
    # Plot envelope distribution
    envelope_path = os.path.join(args.output_dir, 'per_skill_envelope_distribution.png')
    plot_envelope_distribution_by_skill(merged_df, envelope_path, top_skills=20)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - per_skill_alignment_stats.csv")
    print(f"  - per_skill_concordance.png")
    print(f"  - per_skill_envelope_distribution.png")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
