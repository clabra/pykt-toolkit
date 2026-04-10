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
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
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
    parser.add_argument('--timestep', type=int, default=5,
                        help="Subsample: keep every N-th interaction as a candidate before min_move filtering (default: 5)")
    parser.add_argument('--min_interactions', type=int, default=20,
                        help="Minimum interactions required for a student to be a quadrant representative (default: 20)")
    parser.add_argument('--weight_centroid', type=float, default=0.5,
                        help="Weight for centroid proximity in [0,1]; complement goes to interaction count (default: 0.5)")
    parser.add_argument('--count_iqr_threshold', type=float, default=1.5,
                        help="IQR multiplier for outlier detection on interaction count; students above Q3 + k*IQR are excluded (default: 1.5)")
    parser.add_argument('--min_move', type=float, default=None,
                        help="Override the auto-computed min_move threshold (Euclidean distance in [0,1]x[0,1]). "
                             "If not set, it is derived so that the most-active representative has at most --max_points points.")
    parser.add_argument('--max_points', type=int, default=20,
                        help="Target maximum number of sampled points per plot; used to auto-compute min_move threshold (default: 20)")
    parser.add_argument('--gif', action='store_true',
                        help="Also save an animated GIF showing trajectory progression")
    parser.add_argument('--gif_fps', type=int, default=2,
                        help="Frames per second for the GIF (default: 2)")
    parser.add_argument('--gif_hold', type=int, default=4,
                        help="Extra repeated frames on the final frame so the end is visible (default: 4)")
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

    def sample_stride_then_filter(data, rate_c, init_c, stride, min_d):
        """Take every `stride`-th interaction, then drop any that haven't moved
        >= min_d from the previously kept point. First and last are always kept."""
        candidates = list(range(0, len(data), stride))
        if candidates[-1] != len(data) - 1:
            candidates.append(len(data) - 1)
        sel = [candidates[0]]
        last_x = data[rate_c].iloc[candidates[0]]
        last_y = data[init_c].iloc[candidates[0]]
        for j in candidates[1:]:
            dx = data[rate_c].iloc[j] - last_x
            dy = data[init_c].iloc[j] - last_y
            if (dx*dx + dy*dy) ** 0.5 >= min_d or j == candidates[-1]:
                sel.append(j)
                last_x = data[rate_c].iloc[j]
                last_y = data[init_c].iloc[j]
        return sel

    # Auto-compute the global min_move threshold via binary search so that the
    # representative with the most movement has at most --max_points plotted points.
    if args.min_move is not None:
        min_move = args.min_move
        print(f"Using user-specified min_move={min_move:.4f}")
    else:
        def _max_count(min_d):
            mx = 0
            for uid in representatives:
                fs = df[df['student_id'] == uid].reset_index(drop=True).iloc[:global_max_interactions]
                mx = max(mx, len(sample_stride_then_filter(fs, rate_col, init_col, args.timestep, min_d)))
            return mx
        if _max_count(0.0) <= args.max_points:
            min_move = 0.0
        else:
            lo, hi = 0.0, 1.4143  # sqrt(2)
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if _max_count(mid) <= args.max_points:
                    hi = mid
                else:
                    lo = mid
            min_move = hi
        print(f"Auto-computed min_move={min_move:.4f} (max_points={args.max_points})")

    # --- Individual 2D trajectory plot per representative student ---
    for i, uid in enumerate(representatives):
        full_subset = df[df['student_id'] == uid].reset_index(drop=True)
        full_subset = full_subset.iloc[:global_max_interactions]

        def row_quadrant(row):
            return get_quadrant({init_col: row[init_col], rate_col: row[rate_col]})

        full_subset = full_subset.copy()
        full_subset['pt_quad'] = full_subset.apply(row_quadrant, axis=1)

        indices = sample_stride_then_filter(full_subset, rate_col, init_col, args.timestep, min_move)
        subset = full_subset.iloc[indices].reset_index(drop=True)

        quad_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

        fig, ax = plt.subplots(figsize=(8, 7))

        # --- Colored quadrant background regions ---
        ax.add_patch(Rectangle((0.0, 0.0),    t_med,        l0_med,        color=quad_colors[0], alpha=0.15, zorder=0))
        ax.add_patch(Rectangle((0.0, l0_med), t_med,        1.0 - l0_med,  color=quad_colors[1], alpha=0.15, zorder=0))
        ax.add_patch(Rectangle((t_med, 0.0),  1.0 - t_med,  l0_med,        color=quad_colors[2], alpha=0.15, zorder=0))
        ax.add_patch(Rectangle((t_med, l0_med), 1.0 - t_med, 1.0 - l0_med, color=quad_colors[3], alpha=0.15, zorder=0))

        # --- Boundary lines ---
        ax.axvline(t_med,  color='grey', linewidth=1.0, linestyle='--', alpha=0.6, zorder=1)
        ax.axhline(l0_med, color='grey', linewidth=1.0, linestyle='--', alpha=0.6, zorder=1)

        # --- Connecting line with directional arrows between sampled points ---
        for k in range(len(subset) - 1):
            x0, y0 = subset[rate_col].iloc[k],   subset[init_col].iloc[k]
            x1, y1 = subset[rate_col].iloc[k+1], subset[init_col].iloc[k+1]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='dimgrey',
                                        lw=2.0, alpha=0.35,
                                        mutation_scale=20),
                        zorder=2)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, str(k + 1), fontsize=12, color='dimgrey',
                    ha='center', va='center', zorder=3,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))

        # --- Transition points numbered by actual interaction index ---
        for actual_idx, row in zip(indices, subset.itertuples()):
            pt_color = quad_colors[int(row.pt_quad)]
            ax.scatter(getattr(row, rate_col), getattr(row, init_col),
                       color=pt_color, s=160, edgecolors='k', linewidths=0.9,
                       zorder=4, alpha=0.95)
            ax.annotate(str(actual_idx),
                        xy=(getattr(row, rate_col), getattr(row, init_col)),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color='black', zorder=5)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Learning Rate ($p_T$)', fontsize=12)
        ax.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=12)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # Quadrant labels inside the regions
        ax.text(t_med / 2,       l0_med / 2,       'Slow\nStarters',      ha='center', va='center', fontsize=8, color=quad_colors[0], alpha=0.7)
        ax.text(t_med / 2,       (l0_med + 1) / 2,  'Plateaued',           ha='center', va='center', fontsize=8, color=quad_colors[1], alpha=0.7)
        ax.text((t_med + 1) / 2, l0_med / 2,        'Diligent\nBeginners', ha='center', va='center', fontsize=8, color=quad_colors[2], alpha=0.7)
        ax.text((t_med + 1) / 2, (l0_med + 1) / 2,  'Fast\nMasters',       ha='center', va='center', fontsize=8, color=quad_colors[3], alpha=0.7)

        # --- Legend ---
        legend_handles = [
            mpatches.Patch(color=quad_colors[q], alpha=0.7, label=quad_labels[q])
            for q in range(4)
        ]
        ax.legend(handles=legend_handles, fontsize=8, framealpha=0.8,
                  loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=2, borderaxespad=0)

        mean_l0 = full_subset[init_col].mean()
        mean_t  = full_subset[rate_col].mean()
        ax.set_title(
            f'Trajectory: {quad_labels[i]}\n'
            f'student {uid}  |  {len(full_subset)} interactions  |  {len(subset)} sampled (every {args.timestep}, min_move={min_move:.3f})\n'
            f'mean $p_{{L_0}}$ = {mean_l0:.3f}  |  mean $p_T$ = {mean_t:.3f}',
            fontsize=10, pad=10
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"roster_3d_{quad_names[i]}_893468.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {quad_names[i]}: {len(subset)} points (every {args.timestep}, min_move={min_move:.3f}) → {out_path}")

        # --- Animated GIF ---
        if args.gif:
            n_pts = len(subset)
            # Total frames: one per point reveal, plus hold frames at the end
            total_frames = n_pts + args.gif_hold

            fig_g, ax_g = plt.subplots(figsize=(8, 7))

            def _draw_static_bg(ax_):
                ax_.add_patch(Rectangle((0.0, 0.0),    t_med,        l0_med,        color=quad_colors[0], alpha=0.15, zorder=0))
                ax_.add_patch(Rectangle((0.0, l0_med), t_med,        1.0 - l0_med,  color=quad_colors[1], alpha=0.15, zorder=0))
                ax_.add_patch(Rectangle((t_med, 0.0),  1.0 - t_med,  l0_med,        color=quad_colors[2], alpha=0.15, zorder=0))
                ax_.add_patch(Rectangle((t_med, l0_med), 1.0 - t_med, 1.0 - l0_med, color=quad_colors[3], alpha=0.15, zorder=0))
                ax_.axvline(t_med,  color='grey', linewidth=1.0, linestyle='--', alpha=0.6, zorder=1)
                ax_.axhline(l0_med, color='grey', linewidth=1.0, linestyle='--', alpha=0.6, zorder=1)
                ax_.text(t_med / 2,       l0_med / 2,       'Slow\nStarters',      ha='center', va='center', fontsize=8, color=quad_colors[0], alpha=0.7)
                ax_.text(t_med / 2,       (l0_med + 1) / 2,  'Plateaued',           ha='center', va='center', fontsize=8, color=quad_colors[1], alpha=0.7)
                ax_.text((t_med + 1) / 2, l0_med / 2,        'Diligent\nBeginners', ha='center', va='center', fontsize=8, color=quad_colors[2], alpha=0.7)
                ax_.text((t_med + 1) / 2, (l0_med + 1) / 2,  'Fast\nMasters',       ha='center', va='center', fontsize=8, color=quad_colors[3], alpha=0.7)
                ax_.set_xlim(0.0, 1.0)
                ax_.set_ylim(0.0, 1.0)
                ax_.set_xlabel('Learning Rate ($p_T$)', fontsize=12)
                ax_.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=12)
                ax_.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax_.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                legend_handles_ = [mpatches.Patch(color=quad_colors[q], alpha=0.7, label=quad_labels[q]) for q in range(4)]
                ax_.legend(handles=legend_handles_, fontsize=8, framealpha=0.8,
                           loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, borderaxespad=0)

            _draw_static_bg(ax_g)

            def _animate(frame):
                # Clamp to last data frame during hold period
                f = min(frame, n_pts - 1)
                ax_g.cla()
                _draw_static_bg(ax_g)
                revealed = subset.iloc[:f + 1]
                r_indices = indices[:f + 1]
                # Past points (faded)
                for k in range(f):
                    x0 = subset[rate_col].iloc[k];  y0 = subset[init_col].iloc[k]
                    x1 = subset[rate_col].iloc[k+1]; y1 = subset[init_col].iloc[k+1]
                    ax_g.annotate('', xy=(x1, y1), xytext=(x0, y0),
                                  arrowprops=dict(arrowstyle='->', color='dimgrey', lw=2.0, alpha=0.25, mutation_scale=20), zorder=2)
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    ax_g.text(mx, my, str(k + 1), fontsize=12, color='dimgrey', ha='center', va='center', zorder=3,
                              bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))
                # Past points
                for k in range(f):
                    row_ = revealed.iloc[k]
                    ax_g.scatter(row_[rate_col], row_[init_col],
                                 color=quad_colors[int(row_['pt_quad'])], s=120, edgecolors='k',
                                 linewidths=0.7, zorder=4, alpha=0.5)
                    ax_g.annotate(str(r_indices[k]),
                                  xy=(row_[rate_col], row_[init_col]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=10, color='grey', zorder=5)
                # Current (highlighted) point
                cur = revealed.iloc[f]
                ax_g.scatter(cur[rate_col], cur[init_col],
                             color=quad_colors[int(cur['pt_quad'])], s=260, edgecolors='k',
                             linewidths=1.5, zorder=6, alpha=1.0)
                ax_g.annotate(str(r_indices[f]),
                              xy=(cur[rate_col], cur[init_col]),
                              xytext=(6, 6), textcoords='offset points',
                              fontsize=11, color='black', fontweight='bold', zorder=7)
                ax_g.set_title(
                    f'Trajectory: {quad_labels[i]}\n'
                    f'student {uid}  |  step {r_indices[f]} / {indices[-1]}',
                    fontsize=10, pad=10
                )

            anim = FuncAnimation(fig_g, _animate, frames=total_frames, interval=1000 // args.gif_fps, repeat=False)
            gif_path = os.path.join(output_dir, f"roster_traj_{quad_names[i]}_893468.gif")
            anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
            plt.close(fig_g)
            print(f"Saved GIF {quad_names[i]}: {total_frames} frames → {gif_path}")

    # --- Combined plot (kept for backward compatibility) ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, uid in enumerate(representatives):
        full_subset = df[df['student_id'] == uid].reset_index(drop=True)
        full_subset = full_subset.iloc[:global_max_interactions]
        indices = sample_stride_then_filter(full_subset, rate_col, init_col, args.timestep, min_move)
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
