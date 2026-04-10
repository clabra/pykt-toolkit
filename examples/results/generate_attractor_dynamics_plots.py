#!/usr/bin/env python3
"""
Dynamic attractor visualisations for GTransformer learning trajectories.

Produces three plot families per quadrant:

  Option 2 — KDE density contours + representative trajectory overlay
    attractor_kde_{quad}_{suffix}.png

  Option 4 — Return map (lag plot)
    attractor_returnmap_{quad}_{suffix}.png
    Two sub-panels: p_T(t+1) vs p_T(t)  and  p_L0(t+1) vs p_L0(t).
    A fixed-point attractor appears as a cluster near the diagonal.

  Option 5 — State transition graph
    attractor_transgraph_{quad}_{suffix}.png
    Space is divided into a coarse grid of named cells; directed edges show
    the most frequent transitions within a quadrant's students.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde

# ── shared helpers ───────────────────────────────────────────────────────────

def load_and_merge(run_dir):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    if not all(os.path.exists(p) for p in [rate_path, init_path]):
        raise FileNotFoundError(f"Missing traj_*.csv in {run_dir}")
    df_r = pd.read_csv(rate_path)
    df_i = pd.read_csv(init_path)
    rate_col = 'idkt_rate' if 'idkt_rate' in df_r.columns else 'ts'
    init_col = 'idkt_im'   if 'idkt_im'   in df_i.columns else 'lc'
    mk = (['student_id', 'interaction_idx'] if 'interaction_idx' in df_r.columns
          else ['student_id', 'skill_id'])
    df = df_r.merge(df_i[mk + [init_col]], on=mk)
    return df, rate_col, init_col


def assign_quadrants(df, rate_col, init_col):
    l0_med = df[init_col].median()
    t_med  = df[rate_col].median()

    def qid(l0, t):
        if l0 <= l0_med and t <= t_med: return 0
        if l0 >  l0_med and t <= t_med: return 1
        if l0 <= l0_med and t >  t_med: return 2
        return 3

    sm = df.groupby('student_id')[[init_col, rate_col]].mean()
    sm['quadrant'] = sm.apply(lambda r: qid(r[init_col], r[rate_col]), axis=1)
    df = df.merge(sm[['quadrant']], on='student_id')
    return df, sm, l0_med, t_med


def pick_representatives(sm, interaction_counts, init_col, rate_col,
                         min_interactions, weight_centroid, non_outlier_ids):
    reps = []
    for quad in range(4):
        cands = sm[sm['quadrant'] == quad]
        eligible = cands[cands.index.isin(
            interaction_counts[(interaction_counts >= min_interactions) &
                               (interaction_counts.index.isin(non_outlier_ids))].index)]
        if eligible.empty:
            eligible = cands[cands.index.isin(
                interaction_counts[interaction_counts >= min_interactions].index)]
        if eligible.empty:
            eligible = cands
        cl0 = eligible[init_col].mean(); ct = eligible[rate_col].mean()
        dist = ((eligible[init_col]-cl0)**2 + (eligible[rate_col]-ct)**2)**0.5
        dr = dist.max() - dist.min()
        prox = 1.0 - (dist - dist.min())/dr if dr > 0 else pd.Series(1.0, index=dist.index)
        ce = interaction_counts[eligible.index]
        cr = ce.max() - ce.min()
        cn = (ce - ce.min())/cr if cr > 0 else pd.Series(1.0, index=ce.index)
        score = weight_centroid*prox + (1.0-weight_centroid)*cn
        reps.append(score.idxmax())
    return reps


def draw_bg(ax, t_med, l0_med, colors, short_names):
    ax.add_patch(Rectangle((0,0), t_med, l0_med,            color=colors[0], alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((0,l0_med), t_med, 1-l0_med,     color=colors[1], alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((t_med,0), 1-t_med, l0_med,      color=colors[2], alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((t_med,l0_med), 1-t_med, 1-l0_med, color=colors[3], alpha=0.10, zorder=0))
    ax.axvline(t_med,  color='grey', lw=1.0, ls='--', alpha=0.5, zorder=1)
    ax.axhline(l0_med, color='grey', lw=1.0, ls='--', alpha=0.5, zorder=1)
    for q, (px, py) in enumerate([
        (t_med/2, l0_med/2), (t_med/2, (l0_med+1)/2),
        ((t_med+1)/2, l0_med/2), ((t_med+1)/2, (l0_med+1)/2)
    ]):
        ax.text(px, py, short_names[q], ha='center', va='center',
                fontsize=8, color=colors[q], alpha=0.65, zorder=1)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Learning Rate ($p_T$)', fontsize=11)
    ax.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=11)
    ax.set_xticks([0,.25,.5,.75,1]); ax.set_yticks([0,.25,.5,.75,1])


# ── Option 2: KDE + trajectory overlay ──────────────────────────────────────

def plot_kde_overlay(df, sm, rate_col, init_col, representatives,
                     t_med, l0_med, colors, short_names, labels, names,
                     output_dir, suffix):
    print("\n── Option 2: KDE density + trajectory overlay ──────")
    for i, uid in enumerate(representatives):
        quad = i
        quad_ids = sm[sm['quadrant'] == quad].index
        quad_df  = df[df['student_id'].isin(quad_ids)]
        xs_all = np.clip(quad_df[rate_col].values, 0, 1)
        ys_all = np.clip(quad_df[init_col].values, 0, 1)

        traj    = df[df['student_id'] == uid].reset_index(drop=True)
        xs_rep  = traj[rate_col].values
        ys_rep  = traj[init_col].values

        fig, ax = plt.subplots(figsize=(8, 7))
        draw_bg(ax, t_med, l0_med, colors, short_names)

        # KDE filled contours (all students in quadrant)
        if len(xs_all) > 20:
            try:
                kde = gaussian_kde(np.vstack([xs_all, ys_all]))
                gx = np.linspace(0, 1, 100)
                gy = np.linspace(0, 1, 100)
                Gx, Gy = np.meshgrid(gx, gy)
                Z = kde(np.vstack([Gx.ravel(), Gy.ravel()])).reshape(Gx.shape)
                cf = ax.contourf(Gx, Gy, Z, levels=8, cmap='YlOrRd', alpha=0.45, zorder=2)
                ax.contour(Gx, Gy, Z, levels=8,
                           colors=colors[quad], linewidths=0.7, alpha=0.6, zorder=3)
                plt.colorbar(cf, ax=ax, label='Interaction density', fraction=0.035, pad=0.02)
            except Exception as e:
                print(f"  KDE failed for quad {quad}: {e}")

        # Representative trajectory on top
        ax.plot(xs_rep, ys_rep, color='black', lw=1.2, alpha=0.55, zorder=4)
        ax.scatter(xs_rep, ys_rep, c=colors[quad], s=25, alpha=0.7,
                   edgecolors='k', linewidths=0.4, zorder=5)
        # Start / end markers
        ax.scatter(xs_rep[0],  ys_rep[0],  s=120, marker='o', color='white',
                   edgecolors=colors[quad], linewidths=1.5, zorder=6)
        ax.scatter(xs_rep[-1], ys_rep[-1], s=120, marker='s', color=colors[quad],
                   edgecolors='k', linewidths=1.0, zorder=6)
        ax.annotate('start', (xs_rep[0],  ys_rep[0]),  xytext=(5,5),
                    textcoords='offset points', fontsize=8, color='dimgrey')
        ax.annotate('end',   (xs_rep[-1], ys_rep[-1]), xytext=(5,5),
                    textcoords='offset points', fontsize=8, color='dimgrey')

        n_quad = len(quad_ids)
        ax.set_title(
            f'KDE Attractor: {labels[i]}\n'
            f'{n_quad} students in quadrant  |  representative: student {uid}  ({len(xs_rep)} interactions)',
            fontsize=10, pad=10
        )

        legend_elems = [
            mpatches.Patch(facecolor=colors[quad], alpha=0.5, label='KDE density (all quadrant students)'),
            plt.Line2D([0],[0], color='black', lw=1.2, label=f'Rep. trajectory (student {uid})'),
        ]
        ax.legend(handles=legend_elems, fontsize=8, loc='lower right', framealpha=0.8)

        plt.tight_layout()
        out = os.path.join(output_dir, f"attractor_kde_{names[i]}_{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved [{names[i]}] → {out}")


# ── Option 4: Return map (lag plot) ─────────────────────────────────────────

def plot_return_map(df, sm, rate_col, init_col, representatives,
                    colors, labels, names, output_dir, suffix):
    print("\n── Option 4: Return map (lag plot) ─────────────────")
    for i, uid in enumerate(representatives):
        traj = df[df['student_id'] == uid].reset_index(drop=True)
        t_vals  = traj[rate_col].values
        l0_vals = traj[init_col].values

        # lag-1 pairs
        t0,  t1  = t_vals[:-1],  t_vals[1:]
        l0_0, l0_1 = l0_vals[:-1], l0_vals[1:]
        # time colouring (early→late)
        n = len(t0)
        cmap_vals = np.linspace(0, 1, n)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f'Return Map (Lag-1): {labels[i]}\n'
            f'student {uid}  |  {len(traj)} interactions',
            fontsize=11)

        for ax, x0, x1, col_label, param_label in [
            (axes[0], t0,   t1,   colors[i], '$p_T$'),
            (axes[1], l0_0, l0_1, colors[i], '$p_{L_0}$'),
        ]:
            sc = ax.scatter(x0, x1, c=cmap_vals, cmap='plasma',
                            s=25, alpha=0.7, edgecolors='none', zorder=3)
            # Identity line (fixed-point attractor lies here)
            lims = [0, 1]
            ax.plot(lims, lims, 'k--', lw=1.0, alpha=0.4, zorder=2,
                    label='Identity (fixed point)')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel(f'{param_label}(t)',   fontsize=11)
            ax.set_ylabel(f'{param_label}(t+1)', fontsize=11)
            ax.set_xticks([0,.25,.5,.75,1]); ax.set_yticks([0,.25,.5,.75,1])
            ax.set_aspect('equal')
            ax.set_title(f'{param_label} lag-1', fontsize=10)
            ax.legend(fontsize=8, loc='upper left')

        plt.colorbar(sc, ax=axes[1], label='Interaction index (early→late)',
                     fraction=0.04, pad=0.04)
        plt.tight_layout()
        out = os.path.join(output_dir, f"attractor_returnmap_{names[i]}_{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved [{names[i]}] → {out}")


# ── Option 5: State transition graph ────────────────────────────────────────

def plot_transition_graph(df, sm, rate_col, init_col, representatives,
                          t_med, l0_med, colors, short_names, labels, names,
                          output_dir, suffix, grid_n=5, top_k=12):
    """
    Divide [0,1]² into grid_n×grid_n cells.  For every consecutive pair of
    interactions within each quadrant's students, count cell→cell transitions.
    Draw the top_k most frequent inter-cell transitions as directed arrows,
    scaled by frequency.  Self-loops (cell→same cell) are shown as thicker node
    outlines.
    """
    print(f"\n── Option 5: State transition graph ({grid_n}×{grid_n} grid) ──")

    for i, uid in enumerate(representatives):
        quad     = i
        quad_ids = sm[sm['quadrant'] == quad].index
        quad_df  = df[df['student_id'].isin(quad_ids)]

        # ── Data-driven axis bounds with 5% padding ────────────────────────
        pad = 0.05
        x_all = quad_df[rate_col].values
        y_all = quad_df[init_col].values
        xlo = max(0.0, x_all.min() - pad * (x_all.max() - x_all.min()))
        xhi = min(1.0, x_all.max() + pad * (x_all.max() - x_all.min()))
        ylo = max(0.0, y_all.min() - pad * (y_all.max() - y_all.min()))
        yhi = min(1.0, y_all.max() + pad * (y_all.max() - y_all.min()))
        if xhi - xlo < 1e-6: xlo, xhi = 0.0, 1.0
        if yhi - ylo < 1e-6: ylo, yhi = 0.0, 1.0

        # Per-quadrant grid edges in data range
        q_edges_x = np.linspace(xlo, xhi, grid_n + 1)
        q_edges_y = np.linspace(ylo, yhi, grid_n + 1)
        q_cx = 0.5 * (q_edges_x[:-1] + q_edges_x[1:])
        q_cy = 0.5 * (q_edges_y[:-1] + q_edges_y[1:])

        def cell_id(x, y):
            ci = np.searchsorted(q_edges_x, x, side='right') - 1
            ri = np.searchsorted(q_edges_y, y, side='right') - 1
            ci = int(max(0, min(grid_n - 1, ci)))
            ri = int(max(0, min(grid_n - 1, ri)))
            return ri, ci

        # Count transitions
        from collections import defaultdict
        trans = defaultdict(int)
        self_loops = defaultdict(int)
        visit_count = defaultdict(int)

        for _, traj in quad_df.groupby('student_id'):
            traj = traj.reset_index(drop=True)
            xs = traj[rate_col].values
            ys = traj[init_col].values
            for k in range(len(traj)):
                visit_count[cell_id(xs[k], ys[k])] += 1
            for k in range(len(traj) - 1):
                src = cell_id(xs[k],   ys[k])
                dst = cell_id(xs[k+1], ys[k+1])
                if src == dst:
                    self_loops[src] += 1
                else:
                    trans[(src, dst)] += 1

        # Top inter-cell transitions
        top_trans = sorted(trans.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_freq  = top_trans[0][1] if top_trans else 1

        fig, ax = plt.subplots(figsize=(9, 8))

        # ── Quadrant background regions clipped to data range ──────────────
        def clipped_rect(ax_, x0, y0, w, h, color, alpha):
            rx0 = max(x0, xlo); ry0 = max(y0, ylo)
            rx1 = min(x0+w, xhi); ry1 = min(y0+h, yhi)
            if rx1 > rx0 and ry1 > ry0:
                ax_.add_patch(Rectangle((rx0, ry0), rx1-rx0, ry1-ry0,
                                        color=color, alpha=alpha, zorder=0))

        clipped_rect(ax, 0,     0,     t_med,     l0_med,     colors[0], 0.10)
        clipped_rect(ax, 0,     l0_med, t_med,    1-l0_med,   colors[1], 0.10)
        clipped_rect(ax, t_med, 0,     1-t_med,   l0_med,     colors[2], 0.10)
        clipped_rect(ax, t_med, l0_med, 1-t_med,  1-l0_med,   colors[3], 0.10)

        if xlo <= t_med <= xhi:
            ax.axvline(t_med,  color='grey', lw=1.0, ls='--', alpha=0.5, zorder=1)
        if ylo <= l0_med <= yhi:
            ax.axhline(l0_med, color='grey', lw=1.0, ls='--', alpha=0.5, zorder=1)

        # Quadrant name labels (only if median is within view)
        for q_l, (px, py) in enumerate([
            (t_med/2,        l0_med/2),
            (t_med/2,        (l0_med+1)/2),
            ((t_med+1)/2,    l0_med/2),
            ((t_med+1)/2,    (l0_med+1)/2),
        ]):
            if xlo <= px <= xhi and ylo <= py <= yhi:
                ax.text(px, py, short_names[q_l], ha='center', va='center',
                        fontsize=8, color=colors[q_l], alpha=0.65, zorder=1)

        # ── Draw grid ──────────────────────────────────────────────────────
        for v in q_edges_x:
            ax.axvline(v, color='lightgrey', lw=0.4, zorder=1)
        for v in q_edges_y:
            ax.axhline(v, color='lightgrey', lw=0.4, zorder=1)

        # ── Node circles sized by visit count ─────────────────────────────
        max_visits = max(visit_count.values()) if visit_count else 1
        cell_w = (xhi - xlo) / grid_n
        cell_h = (yhi - ylo) / grid_n
        base_r = 0.3 * min(cell_w, cell_h)
        for (r, c), cnt in visit_count.items():
            cx_n = q_cx[c]; cy_n = q_cy[r]
            radius = 0.2 * base_r + 0.8 * base_r * cnt / max_visits
            sl_lw  = 2.0 + 4.0 * self_loops.get((r,c), 0) / max_freq
            circle = plt.Circle((cx_n, cy_n), radius,
                                 color=colors[quad], alpha=0.35,
                                 linewidth=sl_lw,
                                 edgecolor=colors[quad], zorder=3)
            ax.add_patch(circle)

        # ── Arrows for top transitions ─────────────────────────────────────
        for (src, dst), freq in top_trans:
            sr, sc_ = src; dr, dc = dst
            x0 = q_cx[sc_]; y0 = q_cy[sr]
            x1 = q_cx[dc];  y1 = q_cy[dr]
            lw   = 0.8 + 3.5 * freq / max_freq
            alpha = 0.4 + 0.5 * freq / max_freq
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=colors[quad],
                                        lw=lw, alpha=alpha, mutation_scale=14),
                        zorder=4)
            # Frequency label at arrow midpoint
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my, str(freq), fontsize=6, color='dimgrey',
                    ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel('Learning Rate ($p_T$)', fontsize=11)
        ax.set_ylabel('Initial Mastery ($p_{L_0}$)', fontsize=11)

        n_quad = len(quad_ids)
        quad_mean_l0 = sm.loc[quad_ids, init_col].mean()
        quad_mean_t  = sm.loc[quad_ids, rate_col].mean()
        rep_mean_l0  = sm.loc[uid, init_col]
        rep_mean_t   = sm.loc[uid, rate_col]
        ax.set_title(
            f'Transition Graph: {labels[i]}\n'
            f'All {n_quad} students in quadrant  |  {grid_n}×{grid_n} grid  |  top {top_k} transitions shown\n'
            f'Quadrant mean: $p_{{L_0}}$={quad_mean_l0:.3f}, $p_T$={quad_mean_t:.3f}',
            fontsize=10, pad=10
        )

        plt.tight_layout()
        out = os.path.join(output_dir, f"attractor_transgraph_{names[i]}_{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved [{names[i]}] → {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--min_interactions', type=int, default=20)
    parser.add_argument('--weight_centroid',  type=float, default=0.3)
    parser.add_argument('--count_iqr_threshold', type=float, default=1.5)
    parser.add_argument('--grid_n', type=int, default=6,
                        help="Grid resolution for transition graph (default: 6)")
    parser.add_argument('--top_k', type=int, default=15,
                        help="Top-k transitions to draw in transition graph (default: 15)")
    parser.add_argument('--uid_suffix', type=str, default='893468')
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir
    os.makedirs(output_dir, exist_ok=True)

    df, rate_col, init_col = load_and_merge(args.run_dir)
    df, sm, l0_med, t_med  = assign_quadrants(df, rate_col, init_col)
    print(f"Medians: L0={l0_med:.4f}, T={t_med:.4f}")

    ic = df.groupby('student_id').size()
    q1 = ic.quantile(0.25); q3 = ic.quantile(0.75)
    non_outlier_ids = ic[ic <= q3 + args.count_iqr_threshold*(q3-q1)].index

    reps = pick_representatives(sm, ic, init_col, rate_col,
                                args.min_interactions, args.weight_centroid,
                                non_outlier_ids)

    colors      = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    short_names = ['Slow\nStarters', 'Plateaued', 'Diligent\nBeginners', 'Fast\nMasters']
    labels      = [
        'Slow Starters (Low $p_{L_0}$, Low $p_T$)',
        'Plateaued (High $p_{L_0}$, Low $p_T$)',
        'Diligent Beginners (Low $p_{L_0}$, High $p_T$)',
        'Fast Masters (High $p_{L_0}$, High $p_T$)',
    ]
    names = ['slow_starters', 'plateaued', 'diligent_beginners', 'fast_masters']

    plot_kde_overlay(df, sm, rate_col, init_col, reps, t_med, l0_med,
                     colors, short_names, labels, names, output_dir, args.uid_suffix)

    plot_return_map(df, sm, rate_col, init_col, reps,
                    colors, labels, names, output_dir, args.uid_suffix)

    plot_transition_graph(df, sm, rate_col, init_col, reps, t_med, l0_med,
                          colors, short_names, labels, names, output_dir,
                          args.uid_suffix, grid_n=args.grid_n, top_k=args.top_k)


if __name__ == '__main__':
    main()
