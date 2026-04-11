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
    parser.add_argument('--rate_max', type=float, default=0.5,
                        help="Upper bound on the learning-rate axis; points above this value are treated as outliers "
                             "and excluded from the 2D trajectory plots (default: 0.5)")
    parser.add_argument('--rate_outlier_warn_pct', type=float, default=5.0,
                        help="Percentage threshold: emit a warning when more than this fraction of interactions "
                             "exceed --rate_max, suggesting they may not be outliers (default: 5.0)")
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

    # --- Rate-axis outlier report ---
    rate_max = args.rate_max
    n_total = len(df)
    n_above = int((df[rate_col] > rate_max).sum())
    pct_above = 100.0 * n_above / n_total if n_total > 0 else 0.0
    print(f"Rate-axis range check: {n_above}/{n_total} interactions ({pct_above:.2f}%) "
          f"have {rate_col} > {rate_max} (--rate_max).")
    if pct_above > args.rate_outlier_warn_pct:
        import warnings
        warnings.warn(
            f"WARNING: {pct_above:.1f}% of interactions exceed rate_max={rate_max}, "
            f"which is above the --rate_outlier_warn_pct={args.rate_outlier_warn_pct}% threshold. "
            f"These points may not be outliers; consider raising --rate_max.",
            UserWarning, stacklevel=2
        )
    
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
        'Foundational (Low $p_{L_0}$, Low $p_T$)',
        'Consolidating (High $p_{L_0}$, Low $p_T$)',
        'Emerging (Low $p_{L_0}$, High $p_T$)',
        'Advancing (High $p_{L_0}$, High $p_T$)',
    ]
    quad_names = ['foundational', 'consolidating', 'emerging', 'advancing']

    def sample_stride_then_filter(data, rate_c, init_c, stride, min_d):
        """Take every `stride`-th candidate. A candidate is plotted when its
        Euclidean distance from the *last plotted point* exceeds min_d.
        Skipped candidates increment a counter on the last plotted point.
        First and last candidates are always kept.
        Returns (kept_indices, skip_counts) where skip_counts[idx] is the
        number of candidates absorbed by that plotted point before it."""
        candidates = list(range(0, len(data), stride))
        if not candidates:
            return [], {}
        if candidates[-1] != len(data) - 1:
            candidates.append(len(data) - 1)
        kept = [candidates[0]]
        skip_counts = {candidates[0]: 0}
        pending = 0  # skipped candidates since last plotted point
        last_x = float(data[rate_c].iloc[candidates[0]])
        last_y = float(data[init_c].iloc[candidates[0]])
        for k in range(1, len(candidates)):
            curr = candidates[k]
            cx = float(data[rate_c].iloc[curr])
            cy = float(data[init_c].iloc[curr])
            dist = ((cx - last_x) ** 2 + (cy - last_y) ** 2) ** 0.5
            is_forced = (k == len(candidates) - 1)  # always keep last
            if dist >= min_d or is_forced:
                kept.append(curr)
                skip_counts[curr] = pending
                pending = 0
                last_x, last_y = cx, cy
            else:
                pending += 1
        # If the forced-last point is too close to its predecessor, discard it
        # and use the predecessor as the final point instead.
        if len(kept) >= 2:
            prev_idx = kept[-2]
            last_idx = kept[-1]
            px = float(data[rate_c].iloc[prev_idx])
            py = float(data[init_c].iloc[prev_idx])
            lx = float(data[rate_c].iloc[last_idx])
            ly = float(data[init_c].iloc[last_idx])
            if ((lx - px) ** 2 + (ly - py) ** 2) ** 0.5 < min_d:
                absorbed = skip_counts.pop(last_idx, 0)
                kept.pop()
                skip_counts[prev_idx] = skip_counts.get(prev_idx, 0) + absorbed + 1
        return kept, skip_counts

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
                idxs, _ = sample_stride_then_filter(fs, rate_col, init_col, args.timestep, min_d)
                mx = max(mx, len(idxs))
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

        # Drop per-student points outside the rate axis range
        n_before = len(full_subset)
        full_subset = full_subset[full_subset[rate_col] <= rate_max].reset_index(drop=True)
        n_dropped = n_before - len(full_subset)
        if n_dropped > 0:
            pct_dropped = 100.0 * n_dropped / n_before
            print(f"  Student {uid}: dropped {n_dropped}/{n_before} ({pct_dropped:.1f}%) "
                  f"interactions with {rate_col} > {rate_max}.")
            if pct_dropped > args.rate_outlier_warn_pct:
                import warnings
                warnings.warn(
                    f"WARNING: student {uid} has {pct_dropped:.1f}% of interactions above "
                    f"rate_max={rate_max}; trajectory may be distorted.",
                    UserWarning, stacklevel=2
                )

        def row_quadrant(row):
            return get_quadrant({init_col: row[init_col], rate_col: row[rate_col]})

        full_subset = full_subset.copy()
        full_subset['pt_quad'] = full_subset.apply(row_quadrant, axis=1)

        indices, skip_counts = sample_stride_then_filter(full_subset, rate_col, init_col, args.timestep, min_move)
        subset = full_subset.iloc[indices].reset_index(drop=True)

        quad_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

        fig, ax = plt.subplots(figsize=(8, 7))

        # --- Colored quadrant background regions (clipped to rate_max) ---
        t_right = min(t_med, rate_max)  # left half never exceeds rate_max
        e_right = rate_max - t_med if t_med < rate_max else 0.0  # emerging/advancing width
        ax.add_patch(Rectangle((0.0,   0.0),    t_right,  l0_med,       color=quad_colors[0], alpha=0.15, zorder=0))
        ax.add_patch(Rectangle((0.0,   l0_med), t_right,  1.0 - l0_med, color=quad_colors[1], alpha=0.15, zorder=0))
        if e_right > 0:
            ax.add_patch(Rectangle((t_med, 0.0),    e_right, l0_med,       color=quad_colors[2], alpha=0.15, zorder=0))
            ax.add_patch(Rectangle((t_med, l0_med), e_right, 1.0 - l0_med, color=quad_colors[3], alpha=0.15, zorder=0))

        # --- Boundary lines ---
        if t_med < rate_max:
            ax.axvline(t_med,  color='grey', linewidth=1.0, linestyle='--', alpha=0.6, zorder=1)
        ax.axhline(l0_med, color='grey', linewidth=1.0, linestyle='--', alpha=0.6, zorder=1)

        # --- Connecting line with directional arrows between sampled points ---
        _MIN_SEG_LEN = 0.008  # segments shorter than this are skipped (invisible arrow)
        n_segments = len(subset) - 1

        # Pre-compute segment lengths; find the effective first/last visible segments
        seg_lens = []
        for k in range(n_segments):
            x0_ = subset[rate_col].iloc[k];  y0_ = subset[init_col].iloc[k]
            x1_ = subset[rate_col].iloc[k+1]; y1_ = subset[init_col].iloc[k+1]
            seg_lens.append(((x1_ - x0_) ** 2 + (y1_ - y0_) ** 2) ** 0.5)
        visible_ks = [k for k, sl in enumerate(seg_lens) if sl >= _MIN_SEG_LEN]
        eff_first = visible_ks[0]  if visible_ks else None
        eff_last  = visible_ks[-1] if visible_ks else None

        for k in range(n_segments):
            if seg_lens[k] < _MIN_SEG_LEN:
                continue
            x0, y0 = subset[rate_col].iloc[k],   subset[init_col].iloc[k]
            x1, y1 = subset[rate_col].iloc[k+1], subset[init_col].iloc[k+1]
            is_endpoint_seg = (k == eff_first or k == eff_last)
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='#444444' if is_endpoint_seg else '#aaaaaa',
                                        lw=2.0 if is_endpoint_seg else 1.5,
                                        alpha=0.80 if is_endpoint_seg else 0.35,
                                        mutation_scale=20),
                        zorder=6 if is_endpoint_seg else 2)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, str(k + 1), fontsize=12,
                    color='#444444' if is_endpoint_seg else '#aaaaaa',
                    ha='center', va='center', zorder=3,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))

        # --- Transition points: size encodes number of absorbed skipped candidates ---
        _BASE_S = 100
        _S_PER_SKIP = 22
        first_idx, last_idx = indices[0], indices[-1]
        for actual_idx, row in zip(indices, subset.itertuples()):
            pt_color = quad_colors[int(row.pt_quad)]
            n_skips = skip_counts.get(actual_idx, 0)
            marker_s = _BASE_S + n_skips * _S_PER_SKIP
            is_endpoint = (actual_idx == first_idx or actual_idx == last_idx)
            ax.scatter(getattr(row, rate_col), getattr(row, init_col),
                       color=pt_color, s=marker_s, edgecolors='k',
                       linewidths=1.8 if is_endpoint else 0.9,
                       zorder=4, alpha=0.95)
            ax.annotate(str(actual_idx),
                        xy=(getattr(row, rate_col), getattr(row, init_col)),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=11 if is_endpoint else 10,
                        color='#1a1a1a' if is_endpoint else '#888888', zorder=5)

        ax.set_xlim(0.0, rate_max)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Learning Rate ($p_T$)', fontsize=12)
        ax.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=12)
        x_ticks = [v for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0] if v <= rate_max + 1e-9]
        ax.set_xticks(x_ticks)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # Quadrant labels inside the regions (only where visible)
        ax.text(t_med / 2,                    l0_med / 2,       'Foundational',  ha='center', va='center', fontsize=8, color=quad_colors[0], alpha=0.7)
        ax.text(t_med / 2,                    (l0_med + 1) / 2, 'Consolidating', ha='center', va='center', fontsize=8, color=quad_colors[1], alpha=0.7)
        if t_med < rate_max:
            ax.text((t_med + rate_max) / 2, l0_med / 2,       'Emerging',  ha='center', va='center', fontsize=8, color=quad_colors[2], alpha=0.7)
            ax.text((t_med + rate_max) / 2, (l0_med + 1) / 2, 'Advancing', ha='center', va='center', fontsize=8, color=quad_colors[3], alpha=0.7)

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
            f'{quad_labels[i]}\n'
            f'student {uid}  |  {len(full_subset)} interactions  |  {len(subset)} sampled (every {args.timestep}, min_move={min_move:.3f})\n'
            f'mean $p_{{L_0}}$ = {mean_l0:.3f}  |  mean $p_T$ = {mean_t:.3f}',
            fontsize=10, pad=10
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"roster_2d_{quad_names[i]}_893468.png")
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
                ax_.text(t_med / 2,       l0_med / 2,       'Foundational',  ha='center', va='center', fontsize=8, color=quad_colors[0], alpha=0.7)
                ax_.text(t_med / 2,       (l0_med + 1) / 2,  'Consolidating', ha='center', va='center', fontsize=8, color=quad_colors[1], alpha=0.7)
                ax_.text((t_med + 1) / 2, l0_med / 2,        'Emerging',      ha='center', va='center', fontsize=8, color=quad_colors[2], alpha=0.7)
                ax_.text((t_med + 1) / 2, (l0_med + 1) / 2,  'Advancing',     ha='center', va='center', fontsize=8, color=quad_colors[3], alpha=0.7)
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
                    f'{quad_labels[i]}\n'
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
        indices, _ = sample_stride_then_filter(full_subset, rate_col, init_col, args.timestep, min_move)
        subset = full_subset.iloc[indices].reset_index(drop=True)
        t = np.array(indices)

        ax.scatter(t, subset[rate_col], subset[init_col],
                   color=colors[i], s=50, label=quad_labels[i], alpha=0.8, edgecolors='k')
        ax.plot(t, subset[rate_col], subset[init_col],
                color=colors[i], alpha=0.5, linewidth=2)

    # --- Quadrant dividing planes ---
    # Axes: X=t (reversed), Y=p_T (0→1), Z=p_L0
    # Quadrants: Q0=Foundational(low L0,low T), Q1=Consolidating(high L0,low T),
    #            Q2=Emerging(low L0,high T),    Q3=Advancing(high L0,high T)
    plane_alpha = 0.13
    t_range = np.array([0, global_max_interactions])
    t_grid  = np.array([[0, 0], [global_max_interactions, global_max_interactions]])

    # Plane 1a: Y=t_med, Z in [0, l0_med]  → Foundational (Q0) color
    xx1a = t_grid.copy()
    yy1a = np.full_like(xx1a, t_med, dtype=float)
    zz1a = np.array([[0, l0_med], [0, l0_med]], dtype=float)
    ax.plot_surface(xx1a, yy1a, zz1a, color=quad_colors[0], alpha=plane_alpha, linewidth=0, antialiased=False)

    # Plane 1b: Y=t_med, Z in [l0_med, 1]  → Consolidating (Q1) color
    xx1b = t_grid.copy()
    yy1b = np.full_like(xx1b, t_med, dtype=float)
    zz1b = np.array([[l0_med, 1.0], [l0_med, 1.0]], dtype=float)
    ax.plot_surface(xx1b, yy1b, zz1b, color=quad_colors[1], alpha=plane_alpha, linewidth=0, antialiased=False)

    # Plane 2a: Z=l0_med, Y in [0, t_med]  → Foundational (Q0) color
    xx2a = t_grid.copy()
    yy2a = np.array([[0, t_med], [0, t_med]], dtype=float)
    zz2a = np.full_like(xx2a, l0_med, dtype=float)
    ax.plot_surface(xx2a, yy2a, zz2a, color=quad_colors[0], alpha=plane_alpha, linewidth=0, antialiased=False)

    # Plane 2b: Z=l0_med, Y in [t_med, 1]  → Emerging (Q2) color
    xx2b = t_grid.copy()
    yy2b = np.array([[t_med, 1.0], [t_med, 1.0]], dtype=float)
    zz2b = np.full_like(xx2b, l0_med, dtype=float)
    ax.plot_surface(xx2b, yy2b, zz2b, color=quad_colors[2], alpha=plane_alpha, linewidth=0, antialiased=False)

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
