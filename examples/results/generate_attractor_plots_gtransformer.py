#!/usr/bin/env python3
"""
Attractor visualisations for GTransformer learning trajectories.

Produces two plot families:
  1. Vector field  -- one plot per quadrant, using ALL students in that quadrant.
     Transitions are binned onto a grid; mean displacement vectors are drawn as
     quiver arrows.  KDE contours overlay the probability mass (the attractor
     is the innermost contour / sink of arrows).

  2. Covariance ellipse  -- one plot per quadrant representative.
     The student's trajectory is shown as in the roster plots, with 1-sigma and
     2-sigma covariance ellipses centred on the mean position, visualising the
     shape and size of the attractor orbit.

Output files (written to --output_dir, which defaults to --run_dir):
  attractor_vectorfield_{quad_name}_{uid_suffix}.png   (4 files)
  attractor_ellipse_{quad_name}_{uid_suffix}.png       (4 files)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Ellipse
from scipy.stats import gaussian_kde

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
                fontsize=8, color=quad_colors[q], alpha=0.65, zorder=1)


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
    parser.add_argument('--grid_size', type=int, default=10,
                        help="Number of cells per axis for the vector-field grid (default: 10)")
    parser.add_argument('--min_transitions', type=int, default=3,
                        help="Minimum transitions in a grid cell to draw an arrow (default: 3)")
    parser.add_argument('--timestep', type=int, default=1,
                        help="Stride for building vector field transitions (default: 1, use all). "
                             "Increase to subsample dense trajectories.")
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
    quad_names      = ['slow_starters', 'plateaued', 'diligent_beginners', 'fast_masters']
    quad_names_short = ['Slow\nStarters', 'Plateaued', 'Diligent\nBeginners', 'Fast\nMasters']
    quad_labels     = [
        'Slow Starters (Low $p_{L_0}$, Low $p_T$)',
        'Plateaued (High $p_{L_0}$, Low $p_T$)',
        'Diligent Beginners (Low $p_{L_0}$, High $p_T$)',
        'Fast Masters (High $p_{L_0}$, High $p_T$)',
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
    # Plot 1: Vector field  (one per quadrant)
    # ════════════════════════════════════════════════════════════════════════
    print("\n── Vector field plots ──────────────────────────────")
    G = args.grid_size
    edges = np.linspace(0.0, 1.0, G + 1)
    cell_centres = 0.5 * (edges[:-1] + edges[1:])

    for quad in range(4):
        quad_student_ids = student_means[student_means['quadrant'] == quad].index
        quad_df = df[df['student_id'].isin(quad_student_ids)].copy()

        # Accumulate displacement vectors on the grid
        sum_dx   = np.zeros((G, G))
        sum_dy   = np.zeros((G, G))
        count_tr = np.zeros((G, G), dtype=int)
        all_x    = []   # for KDE
        all_y    = []

        for uid, traj in quad_df.groupby('student_id'):
            traj = traj.reset_index(drop=True)
            xs = traj[rate_col].values
            ys = traj[init_col].values
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())
            # Stride
            origins = range(0, len(traj) - 1, args.timestep)
            for k in origins:
                x0, y0 = xs[k],   ys[k]
                x1, y1 = xs[k+1], ys[k+1]
                ci = np.searchsorted(edges, x0, side='right') - 1
                ri = np.searchsorted(edges, y0, side='right') - 1
                ci = max(0, min(G - 1, ci))
                ri = max(0, min(G - 1, ri))
                sum_dx[ri, ci]   += x1 - x0
                sum_dy[ri, ci]   += y1 - y0
                count_tr[ri, ci] += 1

        # Average and mask sparse cells
        valid = count_tr >= args.min_transitions
        avg_dx = np.where(valid, sum_dx / np.where(count_tr > 0, count_tr, 1), np.nan)
        avg_dy = np.where(valid, sum_dy / np.where(count_tr > 0, count_tr, 1), np.nan)

        Xg, Yg = np.meshgrid(cell_centres, cell_centres)

        fig, ax = plt.subplots(figsize=(8, 7))
        draw_quadrant_bg(ax, t_med, l0_med, quad_colors, quad_names_short)

        # ── KDE contours ──────────────────────────────────────────────────
        ax_x = np.array(all_x)
        ax_y = np.array(all_y)
        # Clip to [0,1] — a few model outputs can marginally exceed bounds
        ax_x = np.clip(ax_x, 0.0, 1.0)
        ax_y = np.clip(ax_y, 0.0, 1.0)
        if len(ax_x) > 20:
            try:
                kde = gaussian_kde(np.vstack([ax_x, ax_y]))
                xgrid = np.linspace(0.0, 1.0, 80)
                ygrid = np.linspace(0.0, 1.0, 80)
                Xkde, Ykde = np.meshgrid(xgrid, ygrid)
                Zkde = kde(np.vstack([Xkde.ravel(), Ykde.ravel()])).reshape(Xkde.shape)
                levels = 5
                ax.contourf(Xkde, Ykde, Zkde, levels=levels,
                            cmap='Greys', alpha=0.25, zorder=1)
                ax.contour(Xkde, Ykde, Zkde, levels=levels,
                           colors='grey', linewidths=0.6, alpha=0.5, zorder=2)
            except Exception:
                pass  # KDE can fail with degenerate data

        # ── Quiver ────────────────────────────────────────────────────────
        # Scale arrows so the largest does not exceed one cell width
        mag = np.sqrt(avg_dx**2 + avg_dy**2)
        max_mag = np.nanmax(mag) if np.any(~np.isnan(mag)) else 1.0
        cell_w = edges[1] - edges[0]
        scale_factor = cell_w / max_mag if max_mag > 0 else 1.0

        # Build flat arrays of valid cells only
        rows_v, cols_v = np.where(valid)
        qx  = Xg[rows_v, cols_v]
        qy  = Yg[rows_v, cols_v]
        qdx = avg_dx[rows_v, cols_v] * scale_factor
        qdy = avg_dy[rows_v, cols_v] * scale_factor
        qmag = mag[rows_v, cols_v]

        sc = ax.quiver(qx, qy, qdx, qdy,
                       qmag,
                       cmap='plasma', clim=(0, max_mag),
                       angles='xy', scale_units='xy', scale=1.0,
                       width=0.004, headwidth=4, headlength=5,
                       alpha=0.85, zorder=3)
        plt.colorbar(sc, ax=ax, label='Mean transition magnitude', fraction=0.035, pad=0.02)

        # ── Quadrant centroid marker ───────────────────────────────────────
        cx = student_means.loc[student_means['quadrant'] == quad, rate_col].mean()
        cy = student_means.loc[student_means['quadrant'] == quad, init_col].mean()
        ax.scatter(cx, cy, s=180, marker='*', color=quad_colors[quad],
                   edgecolors='k', linewidths=0.8, zorder=5,
                   label=f'Quadrant centroid ({cx:.2f}, {cy:.2f})')

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Learning Rate ($p_T$)', fontsize=12)
        ax.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=12)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        n_students = len(quad_student_ids)
        ax.set_title(
            f'Vector Field: {quad_labels[quad]}\n'
            f'{n_students} students  |  grid {G}×{G}  |  min transitions/cell = {args.min_transitions}',
            fontsize=10, pad=10
        )
        ax.legend(fontsize=8, loc='lower right', framealpha=0.8)

        plt.tight_layout()
        out_path = os.path.join(output_dir,
                                f"attractor_vectorfield_{quad_names[quad]}_{args.uid_suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved vector field [{quad_names[quad]}]: {n_students} students → {out_path}")

    # ════════════════════════════════════════════════════════════════════════
    # Plot 2: Trajectory + covariance ellipses  (one per representative)
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
            for n_std, lw, alpha_, ls in [(1.0, 2.0, 0.80, '-'),
                                          (2.0, 1.5, 0.50, '--')]:
                covariance_ellipse(ax, xs, ys, n_std=n_std,
                                   edgecolor=quad_colors[i],
                                   facecolor=quad_colors[i],
                                   linewidth=lw, linestyle=ls,
                                   alpha=alpha_, fill=True,
                                   zorder=4, label=f'{n_std:.0f}σ ellipse')

        # Overplot facecolor=white to make it look like contour lines
        # (separate clean patches on top)
        if len(xs) >= 3:
            for n_std, lw, ls in [(1.0, 2.0, '-'), (2.0, 1.5, '--')]:
                covariance_ellipse(ax, xs, ys, n_std=n_std,
                                   edgecolor=quad_colors[i],
                                   facecolor='none',
                                   linewidth=lw, linestyle=ls,
                                   alpha=0.9, fill=False,
                                   zorder=5)

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
        ax.set_xlabel('Learning Rate ($p_T$)', fontsize=12)
        ax.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=12)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # Legend: deduplicate (add ellipses manually)
        sigma_handles = [
            mpatches.Patch(edgecolor=quad_colors[i], facecolor='none',
                           linewidth=2.0, linestyle='-',  label='1σ ellipse'),
            mpatches.Patch(edgecolor=quad_colors[i], facecolor='none',
                           linewidth=1.5, linestyle='--', label='2σ ellipse'),
        ]
        ax.legend(handles=sigma_handles + [
            plt.scatter([], [], s=200, marker='+', color=quad_colors[i],
                        linewidths=2.5, label=f'Mean ({mean_x:.3f}, {mean_y:.3f})')
        ], fontsize=8, loc='lower right', framealpha=0.8)

        ax.set_title(
            f'Attractor Orbit: {quad_labels[i]}\n'
            f'student {uid}  |  {len(xs)} interactions  '
            f'|  orbit σ₁=({np.sqrt(np.cov(xs, ys)[0,0]):.3f}, '
            f'{np.sqrt(np.cov(xs, ys)[1,1]):.3f})',
            fontsize=10, pad=10
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir,
                                f"attractor_ellipse_{quad_names[i]}_{args.uid_suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ellipse [{quad_names[i]}]: student {uid}, {len(xs)} pts → {out_path}")


if __name__ == '__main__':
    main()
