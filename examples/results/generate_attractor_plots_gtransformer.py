#!/usr/bin/env python3
"""
Attractor visualisations for GTransformer learning trajectories.

Produces covariance-ellipse plots (one per quadrant representative):
the student's full trajectory is shown as a faded scatter, with 1-sigma
and 2-sigma covariance ellipses centred on the mean position, visualising
the shape and size of the attractor orbit.

Output files (written to --output_dir, which defaults to --run_dir):
  attractor_ellipse_{quad_name}_{uid_suffix}.png   (4 files)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Ellipse

def lighten_color(hex_color, factor=0.55):
    """Blend hex_color toward white by the given factor (0=original, 1=white)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

# ── helpers ─────────────────────────────────────────────────────────────────

def load_data(run_dir):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    if not all(os.path.exists(p) for p in [rate_path, init_path]):
        raise FileNotFoundError(f"Missing traj_rate.csv or traj_initmastery.csv in {run_dir}")
    return pd.read_csv(rate_path), pd.read_csv(init_path)


def draw_quadrant_bg(ax, t_med, l0_med, quad_colors, quad_names_short):
    ax.add_patch(Rectangle((0.0, 0.0),    t_med,       l0_med,       color=quad_colors[0], alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((0.0, l0_med), t_med,       1.0 - l0_med, color=quad_colors[1], alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((t_med, 0.0),  1.0 - t_med, l0_med,       color=quad_colors[2], alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((t_med, l0_med), 1.0 - t_med, 1.0 - l0_med, color=quad_colors[3], alpha=0.10, zorder=0))
    ax.axvline(t_med,  color='grey', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(l0_med, color='grey', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)
    positions = [
        (t_med / 2,         l0_med / 2),
        (t_med / 2,         (l0_med + 1) / 2),
        ((t_med + 1) / 2,   l0_med / 2),
        ((t_med + 1) / 2,   (l0_med + 1) / 2),
    ]
    for q, (px, py) in enumerate(positions):
        ax.text(px, py, quad_names_short[q], ha='center', va='center',
                fontsize=18, color=quad_colors[q], alpha=0.65, zorder=1)


def covariance_ellipse(ax, x, y, n_std=2.0, **kwargs):
    """Draw a covariance ellipse for the 2-D data (x, y)."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    # Largest eigenvalue first
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                  angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help="Experiment directory containing traj_*.csv files")
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--min_interactions', type=int, default=20)
    parser.add_argument('--weight_centroid', type=float, default=0.3)
    parser.add_argument('--count_iqr_threshold', type=float, default=1.5)
    parser.add_argument('--uid_suffix', type=str, default='893468',
                        help="Suffix appended to output filenames (default: 893468)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── load & merge ────────────────────────────────────────────────────────
    df_rate, df_init = load_data(args.run_dir)
    rate_col = 'idkt_rate' if 'idkt_rate' in df_rate.columns else 'ts'
    init_col = 'idkt_im'   if 'idkt_im'   in df_init.columns else 'lc'
    merge_keys = (['student_id', 'interaction_idx']
                  if 'interaction_idx' in df_rate.columns
                  else ['student_id', 'skill_id'])
    df = df_rate.merge(df_init[merge_keys + [init_col]], on=merge_keys)

    l0_med = df[init_col].median()
    t_med  = df[rate_col].median()
    print(f"Medians: L0={l0_med:.4f}, T={t_med:.4f}")

    def get_quadrant(l0, t):
        if l0 <= l0_med and t <= t_med: return 0
        if l0 >  l0_med and t <= t_med: return 1
        if l0 <= l0_med and t >  t_med: return 2
        return 3

    student_means = df.groupby('student_id')[[init_col, rate_col]].mean()
    student_means['quadrant'] = student_means.apply(
        lambda r: get_quadrant(r[init_col], r[rate_col]), axis=1)
    df = df.merge(student_means[['quadrant']], on='student_id')

    interaction_counts = df.groupby('student_id').size()
    q1_ = interaction_counts.quantile(0.25)
    q3_ = interaction_counts.quantile(0.75)
    iqr_ = q3_ - q1_
    count_upper = q3_ + args.count_iqr_threshold * iqr_
    non_outlier_ids = interaction_counts[interaction_counts <= count_upper].index

    # ── find representatives ─────────────────────────────────────────────────
    quad_colors     = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    quad_names      = ['foundational', 'consolidating', 'emerging', 'advancing']
    quad_names_short = ['Foundational', 'Consolidating', 'Emerging', 'Advancing']
    quad_labels     = [
        'Foundational (Low $p_{L_0}$, Low $p_T$)',
        'Consolidating (High $p_{L_0}$, Low $p_T$)',
        'Emerging (Low $p_{L_0}$, High $p_T$)',
        'Advancing (High $p_{L_0}$, High $p_T$)',
    ]

    representatives = []
    for quad in range(4):
        candidates = student_means[student_means['quadrant'] == quad]
        eligible = candidates[candidates.index.isin(
            interaction_counts[(interaction_counts >= args.min_interactions) &
                               (interaction_counts.index.isin(non_outlier_ids))].index)]
        if eligible.empty:
            eligible = candidates[candidates.index.isin(
                interaction_counts[interaction_counts >= args.min_interactions].index)]
        if eligible.empty:
            eligible = candidates
        centroid_l0 = eligible[init_col].mean()
        centroid_t  = eligible[rate_col].mean()
        dist = ((eligible[init_col] - centroid_l0)**2 +
                (eligible[rate_col] - centroid_t)**2)**0.5
        dr = dist.max() - dist.min()
        proximity = 1.0 - (dist - dist.min()) / dr if dr > 0 else pd.Series(1.0, index=dist.index)
        ce = interaction_counts[eligible.index]
        cr = ce.max() - ce.min()
        count_norm = (ce - ce.min()) / cr if cr > 0 else pd.Series(1.0, index=ce.index)
        score = args.weight_centroid * proximity + (1.0 - args.weight_centroid) * count_norm
        representatives.append(score.idxmax())

    # ════════════════════════════════════════════════════════════════════════
    # Trajectory + covariance ellipses  (one per representative)
    # ════════════════════════════════════════════════════════════════════════
    print("\n── Covariance ellipse plots ────────────────────────")
    for i, uid in enumerate(representatives):
        traj = df[df['student_id'] == uid].reset_index(drop=True)
        xs = traj[rate_col].values
        ys = traj[init_col].values

        mean_x, mean_y = xs.mean(), ys.mean()

        fig, ax = plt.subplots(figsize=(8, 7))
        draw_quadrant_bg(ax, t_med, l0_med, quad_colors, quad_names_short)

        # ── Trajectory (thin, faded) ──────────────────────────────────────
        ax.plot(xs, ys, color='dimgrey', lw=0.8, alpha=0.25, zorder=2)
        ax.scatter(xs, ys, c=quad_colors[i], s=20, alpha=0.35,
                   edgecolors='none', zorder=3)

        # ── Covariance ellipses ────────────────────────────────────────────
        if len(xs) >= 3:
            # 1σ: richer fill; 2σ: very light fill with desaturated edge
            covariance_ellipse(ax, xs, ys, n_std=1.0,
                               edgecolor=quad_colors[i], facecolor=quad_colors[i],
                               linewidth=2.0, linestyle='-',
                               alpha=0.35, fill=True, zorder=4)
            covariance_ellipse(ax, xs, ys, n_std=2.0,
                               edgecolor=lighten_color(quad_colors[i], 0.45),
                               facecolor=lighten_color(quad_colors[i], 0.60),
                               linewidth=1.5, linestyle='--',
                               alpha=0.30, fill=True, zorder=3)

        # Overplot facecolor=white to make it look like contour lines
        # (separate clean patches on top)
        if len(xs) >= 3:
            covariance_ellipse(ax, xs, ys, n_std=1.0,
                               edgecolor=quad_colors[i], facecolor='none',
                               linewidth=2.0, linestyle='-',
                               alpha=0.95, fill=False, zorder=5)
            covariance_ellipse(ax, xs, ys, n_std=2.0,
                               edgecolor=lighten_color(quad_colors[i], 0.35),
                               linewidth=1.5, linestyle='--',
                               alpha=0.75, fill=False, zorder=5)

        # ── Mean marker ───────────────────────────────────────────────────
        ax.scatter(mean_x, mean_y, s=200, marker='+',
                   color=quad_colors[i], linewidths=2.5, zorder=6,
                   label=f'Mean ({mean_x:.3f}, {mean_y:.3f})')

        # ── First and last point ───────────────────────────────────────────
        ax.scatter(xs[0], ys[0], s=120, marker='o', color='white',
                   edgecolors=quad_colors[i], linewidths=1.5, zorder=7)
        ax.annotate('start', xy=(xs[0], ys[0]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='dimgrey')
        ax.scatter(xs[-1], ys[-1], s=120, marker='s', color=quad_colors[i],
                   edgecolors='k', linewidths=1.0, zorder=7)
        ax.annotate('end', xy=(xs[-1], ys[-1]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='dimgrey')

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Learning Rate ($p_T$)', fontsize=22)
        ax.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=22)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # Legend: situation name + ellipses + mean
        import matplotlib.lines as mlines
        situation_handle = mpatches.Patch(facecolor=quad_colors[i], edgecolor='k',
                                          linewidth=0.8, alpha=0.6,
                                          label=quad_names_short[i])
        sigma_handles = [
            situation_handle,
            mpatches.Patch(edgecolor=quad_colors[i], facecolor=quad_colors[i],
                           linewidth=2.0, linestyle='-',  alpha=0.5, label='1\u03c3 region'),
            mpatches.Patch(edgecolor=lighten_color(quad_colors[i], 0.35),
                           facecolor=lighten_color(quad_colors[i], 0.60),
                           linewidth=1.5, linestyle='--', alpha=0.5, label='2\u03c3 region'),
        ]
        ax.legend(handles=sigma_handles + [
            plt.scatter([], [], s=200, marker='+', color=quad_colors[i],
                        linewidths=2.5, label=f'Mean ({mean_x:.3f}, {mean_y:.3f})')
        ], fontsize=14, loc='lower right', framealpha=0.8)

        ax.set_title(
            f'{quad_labels[i]}\n'
            f'student {uid}  |  {len(xs)} interactions  '
            f'|  σ₁=({np.sqrt(np.cov(xs, ys)[0,0]):.3f}, '
            f'{np.sqrt(np.cov(xs, ys)[1,1]):.3f})',
            fontsize=18, pad=10
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir,
                                f"attractor_ellipse_{quad_names[i]}_{args.uid_suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ellipse [{quad_names[i]}]: student {uid}, {len(xs)} pts → {out_path}")


if __name__ == '__main__':
    main()
