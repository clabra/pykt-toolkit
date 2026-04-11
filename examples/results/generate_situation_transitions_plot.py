#!/usr/bin/env python3
"""
Generate a figure that directly evidences the dynamic, non-static nature of
learning situation assignments, avoiding the masking problem of net-churn:

  Left / top panel  — stacked bar chart of absolute student counts per
      learning situation at each snapshot (same style as the preferred
      situation_dist_counts plot).

  Right / bottom panel — aggregate transition matrix (4 × 4 heatmap) counting
      every individual student-level A→B reassignment observed across all
      snapshots.  Diagonal = students who stayed; off-diagonal = genuine
      transitions.  Displayed as row-normalised probabilities so each row
      reads as "given a student was in situation X, probability of being in
      situation Y at the next snapshot".

A text annotation reports the total number of gross transitions to anchor the
narrative: "X% of consecutive student-snapshot pairs involved a situation
change."

Output:
  situation_transitions_<uid_suffix>.png

Reads:
  <run_dir>/traj_rate.csv
  <run_dir>/traj_initmastery.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# ── shared palette ─────────────────────────────────────────────────────────
QUAD_COLORS = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
QUAD_LABELS = ['Foundational', 'Consolidating', 'Emerging', 'Advancing']
QUAD_SHORT  = ['Found.', 'Consol.', 'Emerg.', 'Advanc.']


def load_data(run_dir: str):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    for p in [rate_path, init_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    return pd.read_csv(rate_path), pd.read_csv(init_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning-situation transition matrix + distribution."
    )
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--min_interactions', type=int, default=5)
    parser.add_argument('--max_interactions', type=int, default=None,
                        help="Cap trajectory length. Defaults to p90.")
    parser.add_argument('--smooth_window', type=int, default=3,
                        help="Smoothing window for bar trend lines (default 3).")
    parser.add_argument('--uid_suffix', type=str, default='893468')
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.run_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── load & merge ──────────────────────────────────────────────────────
    df_rate, df_init = load_data(args.run_dir)

    rate_col = 'idkt_rate' if 'idkt_rate' in df_rate.columns else 'ts'
    init_col = 'idkt_im'   if 'idkt_im'   in df_init.columns else 'lc'
    merge_keys = (['student_id', 'interaction_idx']
                  if 'interaction_idx' in df_rate.columns
                  else ['student_id', 'skill_id'])

    df = df_rate.merge(df_init[merge_keys + [init_col]], on=merge_keys)

    l0_med = df[init_col].median()
    t_med  = df[rate_col].median()
    print(f"Population medians: {init_col}={l0_med:.4f}  {rate_col}={t_med:.4f}")

    def assign_quadrant(l0_mean, t_mean):
        if l0_mean <= l0_med and t_mean <= t_med:  return 0
        if l0_mean >  l0_med and t_mean <= t_med:  return 1
        if l0_mean <= l0_med and t_mean >  t_med:  return 2
        return 3

    ic_all   = df.groupby('student_id').size()
    eligible = ic_all[ic_all >= args.min_interactions].index
    df = df[df['student_id'].isin(eligible)]
    n_students = df['student_id'].nunique()
    print(f"Eligible students: {n_students}")

    ic_elig = df.groupby('student_id').size()
    p90 = int(ic_elig.quantile(0.90))
    cap = args.max_interactions if args.max_interactions is not None else p90
    print(f"Interaction cap: {cap}  (p90={p90})")

    df = df.sort_values(['student_id', 'interaction_idx']).reset_index(drop=True)
    max_idx   = min(int(ic_elig.max()), cap)
    snapshots = list(range(args.stride, max_idx + 1, args.stride))
    if not snapshots:
        raise ValueError(f"No snapshots with stride={args.stride}, cap={cap}.")
    print(f"Snapshots: {snapshots[0]} … {snapshots[-1]} ({len(snapshots)} steps)")

    all_uids  = sorted(df['student_id'].unique())
    uid_index = {uid: i for i, uid in enumerate(all_uids)}
    n_uids    = len(all_uids)

    dist_counts  = np.zeros((len(snapshots), 4), dtype=int)
    dist_present = np.zeros(len(snapshots), dtype=int)
    student_quad = np.full((n_uids, len(snapshots)), np.nan)

    for si, snap in enumerate(snapshots):
        sub = df[df['interaction_idx'] < snap]
        if sub.empty:
            continue
        means = sub.groupby('student_id')[[init_col, rate_col]].mean()
        for uid, row in means.iterrows():
            q  = assign_quadrant(row[init_col], row[rate_col])
            ui = uid_index[uid]
            student_quad[ui, si] = q
            dist_counts[si, q]  += 1
            dist_present[si]    += 1

    # ── transition matrix (gross, across all consecutive snapshot pairs) ───
    trans_matrix = np.zeros((4, 4), dtype=int)
    gross_total   = 0
    gross_changed = 0

    for si in range(1, len(snapshots)):
        prev = student_quad[:, si - 1]
        curr = student_quad[:, si]
        both = (~np.isnan(prev)) & (~np.isnan(curr))
        for ui in np.where(both)[0]:
            p, c = int(prev[ui]), int(curr[ui])
            trans_matrix[p, c] += 1
            gross_total   += 1
            if p != c:
                gross_changed += 1

    pct_changed = gross_changed / gross_total * 100 if gross_total > 0 else 0.0
    print(f"Gross transitions: {gross_changed} / {gross_total} "
          f"({pct_changed:.1f}%) involved a situation change")

    # row-normalised probability matrix
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_prob = np.where(row_sums > 0, trans_matrix / row_sums, 0.0)

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1])
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # ── left: stacked bar chart with trend lines ──────────────────────────
    s_arr  = np.array(snapshots)
    bar_w  = args.stride * 0.82
    bottom = np.zeros(len(snapshots))
    bar_bottoms = []

    for q in range(4):
        bar_bottoms.append(bottom.copy())
        ax_bar.bar(s_arr, dist_counts[:, q], bottom=bottom,
                   color=QUAD_COLORS[q], alpha=0.45, width=bar_w, linewidth=0)
        bottom += dist_counts[:, q]

    from scipy.ndimage import uniform_filter1d
    w = max(1, args.smooth_window)

    def smooth(arr):
        return uniform_filter1d(arr.astype(float), size=w, mode='nearest')

    for q in range(4):
        mid = bar_bottoms[q] + dist_counts[:, q] / 2.0
        ax_bar.plot(s_arr, smooth(mid), color=QUAD_COLORS[q],
                    linewidth=2.0, alpha=0.95, zorder=5)

    ax_bar.set_xlim(s_arr[0] - args.stride * 0.6,
                    s_arr[-1] + args.stride * 0.6)
    ax_bar.set_xlabel('Interaction index', fontsize=11)
    ax_bar.set_ylabel('Number of students', fontsize=11)
    ax_bar.set_title(
        f'Learning situation distribution over time\n'
        f'(n = {n_students} students, snapshot every {args.stride} interactions)',
        fontsize=10, pad=6
    )

    # gross-transition annotation
    ax_bar.text(0.02, 0.97,
                f'{pct_changed:.1f}% of student-snapshot pairs\ninvolved a '
                f'situation change\n({gross_changed:,} of {gross_total:,})',
                transform=ax_bar.transAxes,
                va='top', ha='left', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#aaaaaa', alpha=0.85))

    legend_patches = [
        mpatches.Patch(color=QUAD_COLORS[q], alpha=0.7, label=QUAD_LABELS[q])
        for q in range(4)
    ]
    legend_patches.append(
        mpatches.Patch(color='none',
                       label=f'x-axis capped at p90 ({cap} interactions):\n'
                             '90% of students have fewer interactions')
    )
    ax_bar.legend(handles=legend_patches, loc='upper right',
                  fontsize=8, framealpha=0.85, ncol=2)

    # ── right: transition probability heatmap ────────────────────────────
    # colour: white (stay) → deep purple (strong transition)
    cmap = plt.cm.YlOrRd
    im = ax_heat.imshow(trans_prob, cmap=cmap, vmin=0, vmax=1,
                        aspect='auto', interpolation='nearest')

    # annotate each cell with the probability and raw count
    for i in range(4):
        for j in range(4):
            prob = trans_prob[i, j]
            cnt  = trans_matrix[i, j]
            text_color = 'white' if prob > 0.55 else 'black'
            cell_text  = f'{prob:.2f}\n({cnt:,})'
            ax_heat.text(j, i, cell_text, ha='center', va='center',
                         fontsize=7.5, color=text_color)

    ax_heat.set_xticks(range(4))
    ax_heat.set_yticks(range(4))
    ax_heat.set_xticklabels(QUAD_SHORT, fontsize=8.5, rotation=20, ha='right')
    ax_heat.set_yticklabels(QUAD_SHORT, fontsize=8.5)
    ax_heat.set_xlabel('Situation at next snapshot', fontsize=9)
    ax_heat.set_ylabel('Situation at current snapshot', fontsize=9)
    ax_heat.set_title(
        'Transition probabilities\n(row-normalised; counts in parentheses)',
        fontsize=9, pad=6
    )

    # diagonal border to distinguish stay vs change
    for spine in ax_heat.spines.values():
        spine.set_linewidth(0.5)
    for k in range(4):
        ax_heat.add_patch(
            plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                           fill=False, edgecolor='#2c3e50',
                           linewidth=2.0, zorder=5)
        )

    cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label('Transition probability', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    out_path = os.path.join(output_dir,
                            f'situation_transitions_{args.uid_suffix}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
