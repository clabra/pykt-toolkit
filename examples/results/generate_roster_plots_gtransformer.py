#!/usr/bin/env python3
"""
Generate 3D Roster Plots for GTransformer.
Visualizes student learning trajectories in the (Initial Mastery, Learn Rate, Time) space.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def load_data(run_dir):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    
    if not all(os.path.exists(p) for p in [rate_path, init_path]):
        print(f"Missing required CSVs in {run_dir}")
        return None, None
        
    df_rate = pd.read_csv(rate_path)
    df_init = pd.read_csv(init_path)
    
    return df_rate, df_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help="Path to experiment directory containing traj_*.csv files")
    parser.add_argument('--output_dir', default=None, help="Where to save plot")
    parser.add_argument('--timestep', type=int, default=10,
                        help="Subsample: keep every N-th interaction for plotting (default: 10)")
    parser.add_argument('--min_interactions', type=int, default=20,
                        help="Minimum interactions required for a student to be a quadrant representative (default: 20)")
    parser.add_argument('--weight_centroid', type=float, default=0.5,
                        help="Weight for centroid proximity in [0,1]; complement goes to interaction count (default: 0.5)")
    parser.add_argument('--count_iqr_threshold', type=float, default=1.5,
                        help="IQR multiplier for outlier detection on interaction count; students above Q3 + k*IQR are excluded (default: 1.5)")
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.output_dir else args.run_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating 3D Roster for: {args.run_dir}")
    df_rate, df_init = load_data(args.run_dir)
    if df_rate is None: return

    # Map column names
    rate_col = 'idkt_rate' if 'idkt_rate' in df_rate.columns else 'ts'
    init_col = 'idkt_im' if 'idkt_im' in df_init.columns else 'lc'
    
    # Merge parameters — join on student + interaction index to avoid
    # many-to-many cartesian product when a student repeats the same skill
    merge_keys = ['student_id', 'interaction_idx'] if 'interaction_idx' in df_rate.columns else ['student_id', 'skill_id']
    df = df_rate.merge(df_init[merge_keys + [init_col]], on=merge_keys)
    
    # Calculate global medians for quadrant assignment
    l0_med = df[init_col].median()
    t_med = df[rate_col].median()
    
    print(f"Medians: L0={l0_med:.4f}, T={t_med:.4f}")
    
    def get_quadrant(row):
        l0, t = row[init_col], row[rate_col]
        if l0 <= l0_med and t <= t_med: return 0 # Slow Starters
        if l0 > l0_med and t <= t_med: return 1  # Plateaued
        if l0 <= l0_med and t > t_med: return 2  # Diligent Beginners
        return 3 # Fast Masters

    # Assign quadrant per student based on their mean L0 and T across all interactions
    student_means = df.groupby('student_id')[[init_col, rate_col]].mean()
    student_means['quadrant'] = student_means.apply(
        lambda row: get_quadrant({init_col: row[init_col], rate_col: row[rate_col]}),
        axis=1
    )

    # Also propagate quadrant back to per-interaction df (for heatmap use)
    df = df.merge(student_means[['quadrant']], on='student_id')

    # Global max interaction count — used for consistent y-axis across all plots
    interaction_counts = df.groupby('student_id').size()
    global_max_interactions = min(int(interaction_counts.max()), 500)
    print(f"Global max interactions (capped at 500): {global_max_interactions}")

    # Detect and exclude count outliers using IQR before centroid/score computation
    q1 = interaction_counts.quantile(0.25)
    q3 = interaction_counts.quantile(0.75)
    iqr = q3 - q1
    count_upper = q3 + args.count_iqr_threshold * iqr
    outlier_ids = interaction_counts[interaction_counts > count_upper].index
    if len(outlier_ids):
        print(f"Excluding {len(outlier_ids)} outlier student(s) with >{count_upper:.0f} interactions "
              f"(Q3={q3:.0f}, IQR={iqr:.0f}, threshold={args.count_iqr_threshold}): "
              f"{list(outlier_ids)}")
    non_outlier_ids = interaction_counts[interaction_counts <= count_upper].index

    representatives = []
    for quad in [0, 1, 2, 3]:
        candidates = student_means[student_means['quadrant'] == quad]
        if candidates.empty:
            print(f"Warning: No candidates for quadrant {quad}")
            continue
        # Filter by minimum interaction count, excluding outliers
        eligible = candidates[candidates.index.isin(
            interaction_counts[
                (interaction_counts >= args.min_interactions) &
                (interaction_counts.index.isin(non_outlier_ids))
            ].index
        )]
        if eligible.empty:
            # Relax outlier filter but keep min_interactions
            eligible = candidates[candidates.index.isin(
                interaction_counts[interaction_counts >= args.min_interactions].index
            )]
        if eligible.empty:
            print(f"Warning: No candidates with >={args.min_interactions} interactions for quadrant {quad}, relaxing all filters")
            eligible = candidates
        # Weighted score: balance centroid proximity with interaction count
        # Both components are min-max normalised to [0,1] before weighting
        centroid_l0 = eligible[init_col].mean()
        centroid_t  = eligible[rate_col].mean()
        dist = ((eligible[init_col] - centroid_l0) ** 2 +
                (eligible[rate_col] - centroid_t) ** 2) ** 0.5
        # Proximity = 1 - normalised distance (higher is better)
        dist_range = dist.max() - dist.min()
        proximity = 1.0 - (dist - dist.min()) / dist_range if dist_range > 0 else pd.Series(1.0, index=dist.index)
        # Interaction count normalised to [0,1] (higher is better)
        counts_eligible = interaction_counts[eligible.index]
        count_range = counts_eligible.max() - counts_eligible.min()
        count_norm = (counts_eligible - counts_eligible.min()) / count_range if count_range > 0 else pd.Series(1.0, index=counts_eligible.index)
        w_c = args.weight_centroid
        score = w_c * proximity + (1.0 - w_c) * count_norm
        best_uid = score.idxmax()
        representatives.append(best_uid)
        print(f"Quad {quad} representative: student {best_uid}  "
              f"n={interaction_counts[best_uid]}  "
              f"mean L0={eligible.loc[best_uid, init_col]:.4f}  "
              f"mean T={eligible.loc[best_uid, rate_col]:.4f}")

    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    quad_labels = [
        'Slow Starters (Low $p_{L_0}$, Low $p_T$)',
        'Plateaued (High $p_{L_0}$, Low $p_T$)',
        'Diligent Beginners (Low $p_{L_0}$, High $p_T$)',
        'Fast Masters (High $p_{L_0}$, High $p_T$)',
    ]
    quad_names = ['slow_starters', 'plateaued', 'diligent_beginners', 'fast_masters']

    # --- Individual 3D trajectory plot per representative student ---
    for i, uid in enumerate(representatives):
        full_subset = df[df['student_id'] == uid].reset_index(drop=True)
        # Cap to global_max_interactions
        full_subset = full_subset.iloc[:global_max_interactions]

        # Select transition points: first interaction, then only when quadrant changes
        def row_quadrant(row):
            return get_quadrant({init_col: row[init_col], rate_col: row[rate_col]})

        full_subset = full_subset.copy()
        full_subset['pt_quad'] = full_subset.apply(row_quadrant, axis=1)

        transition_indices = [0]
        last_quad = full_subset['pt_quad'].iloc[0]
        for idx in range(1, len(full_subset)):
            q = full_subset['pt_quad'].iloc[idx]
            if q != last_quad:
                transition_indices.append(idx)
                last_quad = q

        subset = full_subset.iloc[transition_indices].reset_index(drop=True)
        t = np.array(transition_indices)

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Colored quadrant regions on the back wall (x = global_max_interactions) ---
        quad_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        quad_yz = [
            ([0.0, t_med],   [0.0, l0_med]),   # Q0 Slow Starters
            ([0.0, t_med],   [l0_med, 1.0]),    # Q1 Plateaued
            ([t_med, 1.0],   [0.0, l0_med]),    # Q2 Diligent Beginners
            ([t_med, 1.0],   [l0_med, 1.0]),    # Q3 Fast Masters
        ]
        xback = global_max_interactions
        for qc, (yr, zr) in zip(quad_colors, quad_yz):
            yy, zz = np.meshgrid(yr, zr)
            xx = np.full_like(yy, xback, dtype=float)
            ax.plot_surface(xx, yy, zz, color=qc, alpha=0.18, zorder=1)

        # --- Two boundary planes spanning full time range ---
        t_plot_range = [0, global_max_interactions]
        tt, ll = np.meshgrid(t_plot_range, [0.0, 1.0])
        ax.plot_surface(tt, np.full_like(tt, t_med),  ll,          alpha=0.10, color='grey', zorder=2)
        rr, tt2 = np.meshgrid([0.0, 1.0], t_plot_range)
        ax.plot_surface(tt2, rr, np.full_like(rr, l0_med),          alpha=0.10, color='grey', zorder=2)

        # --- Connecting line ---
        ax.plot(t, subset[rate_col], subset[init_col],
                color='dimgrey', alpha=0.5, linewidth=1.5, zorder=3)

        # --- Transition points numbered and colored by quadrant ---
        for seq, (idx, row) in enumerate(zip(t, subset.itertuples())):
            pt_color = quad_colors[int(row.pt_quad)]
            ax.scatter([idx], [getattr(row, rate_col)], [getattr(row, init_col)],
                       color=pt_color, s=90, edgecolors='k', zorder=5, alpha=0.95)
            ax.text(idx, getattr(row, rate_col), getattr(row, init_col),
                    f' {seq}', fontsize=8, color='black', zorder=6)

        ax.set_xlabel('Interaction time-step ($t$)', fontsize=11, labelpad=10)
        ax.set_ylabel('Learning Rate ($p_T$)', fontsize=11, labelpad=10)
        ax.set_zlabel('Initial Mastery ($p_{L_0}$)', fontsize=11, labelpad=10)

        ax.set_xlim(global_max_interactions, 0)
        max_ticks = 10
        tick_step = max(1, global_max_interactions // max_ticks)
        xticks = list(range(0, global_max_interactions + 1, tick_step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(v) for v in xticks], fontsize=8)

        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # --- Legend ---
        legend_handles = [
            mpatches.Patch(color=quad_colors[q], alpha=0.7, label=quad_labels[q])
            for q in range(4)
        ]
        ax.legend(handles=legend_handles, loc='upper left',
                  bbox_to_anchor=(0.0, 1.0), fontsize=8, framealpha=0.7)

        mean_l0 = full_subset[init_col].mean()
        mean_t  = full_subset[rate_col].mean()
        n_transitions = len(transition_indices) - 1
        ax.set_title(
            f'Trajectory: {quad_labels[i]}\n'
            f'student {uid}  |  {len(full_subset)} interactions  |  {n_transitions} transition(s)\n'
            f'mean $p_{{L_0}}$ = {mean_l0:.3f}  |  mean $p_T$ = {mean_t:.3f}',
            fontsize=10, pad=15
        )
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"roster_3d_{quad_names[i]}_893468.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {quad_names[i]}: {len(transition_indices)} points ({n_transitions} transitions) → {out_path}")

    # --- Combined plot (kept for backward compatibility) ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, uid in enumerate(representatives):
        full_subset = df[df['student_id'] == uid].reset_index(drop=True)
        full_subset = full_subset.iloc[:global_max_interactions]
        indices = list(range(0, len(full_subset), args.timestep))
        if (len(full_subset) - 1) not in indices:
            indices.append(len(full_subset) - 1)
        subset = full_subset.iloc[indices].reset_index(drop=True)
        t = np.array(indices)

        ax.scatter(t, subset[rate_col], subset[init_col],
                   color=colors[i], s=50, label=quad_labels[i], alpha=0.8, edgecolors='k')
        ax.plot(t, subset[rate_col], subset[init_col],
                color=colors[i], alpha=0.5, linewidth=2)

    ax.set_xlabel('Interaction Time-step ($t$)', fontsize=12, labelpad=10)
    ax.set_ylabel('Learning Rate ($p_{T}$)', fontsize=12, labelpad=10)
    ax.set_zlabel('Initial Mastery ($p_{L_0}$)', fontsize=12, labelpad=10)
    ax.set_xlim(global_max_interactions, 0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0])
    max_ticks = 10
    tick_step = max(1, global_max_interactions // max_ticks)
    xticks = list(range(0, global_max_interactions + 1, tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(v) for v in xticks], fontsize=8)
    plt.title('3D Cognitive Roster: Longitudinal Trajectories across Learning Situations',
              fontsize=16, fontweight='bold', pad=20)
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.9), fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"roster_3d_situations_893468.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved combined 3D roster plot to {output_path}")

    # --- Conventional Student x Skill Roster Heatmap ---
    print("Generating Student x Skill Roster Heatmap...")
    # Sample students (25 from each quadrant) using student_means quadrant assignment
    sample_uids = []
    for quad in [0, 1, 2, 3]:
        candidates = student_means[student_means['quadrant'] == quad].index.tolist()
        if len(candidates) > 0:
            sample_uids.extend(np.random.choice(candidates, min(25, len(candidates)), replace=False))
    
    # Filter for top skills
    top_skills = df['skill_id'].value_counts().nlargest(20).index
    
    # Calculate Mean Mastery per Student-Skill
    roster_data = df[df['student_id'].isin(sample_uids) & df['skill_id'].isin(top_skills)]
    # Use p_ref logic if available, otherwise just use init_col as proxy for state
    # Actually, we should use the average mastery across interactions for that skill
    # (Since gTransformer provides contextual params, the BKT mastery L_t is a better proxy)
    
    # Pivot
    pivot = roster_data.groupby(['student_id', 'skill_id'])[init_col].mean().unstack(fill_value=0.5)
    
    # Reorder students by quadrant for visual grouping
    quad_desc = {uid: df[df['student_id'] == uid]['quadrant'].iloc[-1] for uid in sample_uids}
    pivot['quad'] = pivot.index.map(quad_desc)
    pivot = pivot.sort_values('quad')
    quad_boundaries = pivot['quad'].value_counts().sort_index().cumsum().values
    pivot = pivot.drop(columns='quad')

    plt.figure(figsize=(16, 12))
    ax_heat = sns.heatmap(pivot, cmap='RdYlGn', center=0.5, annot=False, 
                         cbar_kws={'label': 'Mean Pedagogical Mastery ($\hat{L}$)'})
    
    # Add horizontal lines to separate quadrants
    for boundary in quad_boundaries[:-1]:
        plt.axhline(boundary, color='black', linewidth=1, linestyle='--')
    
    plt.title('Cognitive Roster: Student x Skill Mastery (Grouped by Learning Situation)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Students (Sampled and Grouped by Situation)', fontsize=12)
    plt.xlabel('Knowledge Components (Top 20 by Interaction Density)', fontsize=12)
    
    heatmap_path = os.path.join(output_dir, f"roster_heatmap_students_893468.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved roster heatmap to {heatmap_path}")

if __name__ == "__main__":
    main()
