#!/usr/bin/env python3
"""
Generate per-skill prediction alignment heatmaps.

Shows concordance between $p_{sup}$ (supervised predictions) and $p_{ref}$ (interpretable predictions)
across skills and students.

Usage:
    python examples/validation/generate_skill_alignment_heatmap.py \
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
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

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


def compute_alignment_heatmap(df_sup, df_ref, min_interactions=8, top_skills=50, top_students=30):
    """
    Compute alignment (concordance) between p_sup and p_ref predictions.
    
    Alignment metric: 1 - MAE (Mean Absolute Error) per student-skill pair.
    Higher values = better alignment between neural and BKT logic predictions.
    
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
    
    # Select top skills and students by interaction count (max density)
    skill_counts = df_robust.groupby('skill_id').size().nlargest(top_skills)
    student_counts = df_robust.groupby('student_id').size().nlargest(top_students)
    
    df_sample = df_robust[
        df_robust['skill_id'].isin(skill_counts.index) &
        df_robust['student_id'].isin(student_counts.index)
    ]
    
    print(f"✓ Max-Density Sampling: {len(student_counts)} students, {len(skill_counts)} skills")
    print(f"✓ Sample size: {len(df_sample):,} predictions")
    
    # Compute concordance per (student, skill) pair
    def compute_concordance(group):
        mae = np.abs(group['p_sup'].values - group['p_ref'].values).mean()
        return 1.0 - mae  # Concordance: 1 = perfect alignment, 0 = maximum disagreement
    
    results = df_sample.groupby(['student_id', 'skill_id']).apply(
        compute_concordance
    ).reset_index(name='concordance')
    
    # Create pivot table for heatmap
    pivot = results.pivot_table(
        index='student_id',
        columns='skill_id',
        values='concordance'
    )
    
    # Compute summary statistics
    stats = {
        'mean_concordance': results['concordance'].mean(),
        'median_concordance': results['concordance'].median(),
        'std_concordance': results['concordance'].std(),
        'min_concordance': results['concordance'].min(),
        'max_concordance': results['concordance'].max(),
        'n_students': len(student_counts),
        'n_skills': len(skill_counts),
        'n_pairs': len(results),
        'min_interactions': min_interactions
    }
    
    print(f"\n{'='*80}")
    print("ALIGNMENT STATISTICS")
    print(f"{'='*80}")
    print(f"Mean Concordance: {stats['mean_concordance']:.3f}")
    print(f"Median Concordance: {stats['median_concordance']:.3f}")
    print(f"Std Concordance: {stats['std_concordance']:.3f}")
    print(f"Range: [{stats['min_concordance']:.3f}, {stats['max_concordance']:.3f}]")
    print(f"Student-Skill Pairs: {stats['n_pairs']:,}")
    
    return pivot, stats


def plot_alignment_heatmap(pivot, stats, output_path):
    """
    Create concordance heatmap with discrete color zones.
    
    Color zones:
    - Green [0.90-1.0]: Excellent alignment (p_sup ≈ p_ref)
    - Yellow [0.80-0.90]: Good alignment
    - Orange [0.65-0.80]: Moderate alignment
    - Red [<0.65]: Poor alignment (p_sup diverges from p_ref)
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Discrete colormap for concordance zones
    colors_hex = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]
    cmap_con = ListedColormap(colors_hex)
    bounds = [0.0, 0.65, 0.80, 0.90, 1.0]
    norm = BoundaryNorm(bounds, cmap_con.N)
    
    # Plot heatmap
    sns.heatmap(
        pivot,
        cmap=cmap_con,
        norm=norm,
        ax=ax,
        cbar_kws={'label': 'Prediction Alignment (1 - MAE)', 'pad': 0.08}
    )
    
    ax.set_xlabel(f'Skills (Top {stats["n_skills"]} by Density)', fontsize=12)
    ax.set_ylabel(f'Students (Top {stats["n_students"]} by Density)', fontsize=12)
    ax.set_title(
        f'Prediction Alignment Heatmap\n'
        f'$p_{{sup}}$ (Supervised) vs $p_{{ref}}$ (Interpretable) - Mean Concordance: {stats["mean_concordance"]:.3f}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Add discrete legend
    legend_patches = [
        mpatches.Patch(color='#27ae60', label='[0.90 - 1.0]: Excellent Alignment (Supervised ≈ Interpretable)'),
        mpatches.Patch(color='#f1c40f', label='[0.80 - 0.90]: Good Alignment (Consistent Predictions)'),
        mpatches.Patch(color='#e67e22', label='[0.65 - 0.80]: Moderate Alignment (Supervised Refinement)'),
        mpatches.Patch(color='#c0392b', label='[< 0.65]: Poor Alignment (Supervised Divergence)')
    ]
    
    legend = plt.legend(
        handles=legend_patches,
        title="Interpretability Alignment Scale",
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


def plot_alignment_distribution(df_sup, df_ref, output_path):
    """
    Plot distribution of envelope widths (disagreement between p_sup and p_ref).
    """
    # Merge predictions
    df_sup = df_sup.rename(columns={'prediction': 'p_sup'})
    df_ref = df_ref.rename(columns={'prediction': 'p_ref'})
    
    merged = pd.merge(
        df_sup, df_ref,
        on=['student_id', 'skill_id'],
        how='inner'
    )
    
    # Calculate envelope width
    merged['envelope'] = np.abs(merged['p_sup'] - merged['p_ref'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of envelope widths
    ax1.hist(merged['envelope'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(merged['envelope'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {merged["envelope"].mean():.3f}')
    ax1.axvline(merged['envelope'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {merged["envelope"].median():.3f}')
    ax1.set_xlabel('Envelope Width |p_sup - p_ref|', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Prediction Disagreement', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot: p_sup vs p_ref
    sample = merged.sample(min(10000, len(merged)), random_state=42)
    ax2.scatter(sample['p_ref'], sample['p_sup'], alpha=0.3, s=10, color='#2ecc71')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
    ax2.set_xlabel('$p_{ref}$ (Interpretable)', fontsize=12)
    ax2.set_ylabel('$p_{sup}$ (Supervised)', fontsize=12)
    ax2.set_title('Prediction Agreement Scatter', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate per-skill prediction alignment heatmaps')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Experiment directory (e.g., experiments/.../fold_0_955042)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for heatmaps')
    parser.add_argument('--min_interactions', type=int, default=8,
                       help='Minimum interactions per student-skill pair (default: 8)')
    parser.add_argument('--top_skills', type=int, default=50,
                       help='Number of top skills to include (default: 50)')
    parser.add_argument('--top_students', type=int, default=30,
                       help='Number of top students to include (default: 30)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("PER-SKILL PREDICTION ALIGNMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Experiment: {args.exp_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Min interactions: {args.min_interactions}")
    print(f"Top skills: {args.top_skills}")
    print(f"Top students: {args.top_students}")
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
    
    # Compute alignment heatmap
    print("\nComputing alignment heatmap...")
    pivot, stats = compute_alignment_heatmap(
        df_sup, df_ref,
        min_interactions=args.min_interactions,
        top_skills=args.top_skills,
        top_students=args.top_students
    )
    
    # Plot heatmap
    heatmap_path = os.path.join(args.output_dir, 'skill_alignment_heatmap.png')
    plot_alignment_heatmap(pivot, stats, heatmap_path)
    
    # Plot distribution
    dist_path = os.path.join(args.output_dir, 'skill_alignment_distribution.png')
    plot_alignment_distribution(df_sup, df_ref, dist_path)
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, 'alignment_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - skill_alignment_heatmap.png")
    print(f"  - skill_alignment_distribution.png")
    print(f"  - alignment_statistics.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
