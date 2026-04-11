#!/usr/bin/env python3
"""
Generate plots showing how the distribution of students across the four
learning situations evolves over time.

For every interaction snapshot at multiples of --stride, each student is
assigned to a learning situation based on their mean parameter values up to
that snapshot.  The plots produced are:

  situation_dist_stacked_<uid_suffix>.png   – stacked area chart: fraction
      in each situation vs interaction index.
  situation_dist_counts_<uid_suffix>.png    – absolute student counts in each
      situation vs interaction index.
  situation_dist_heatmap_<uid_suffix>.png   – student × snapshot heatmap of
      assigned learning situation (colour).

Reads:
  <run_dir>/traj_rate.csv
  <run_dir>/traj_initmastery.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# ── colour palette shared with the roster plots ───────────────────────────────
QUAD_COLORS  = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
QUAD_NAMES   = ['foundational', 'consolidating', 'emerging', 'advancing']
QUAD_LABELS  = [
    'Foundational',
    'Consolidating',
    'Emerging',
    'Advancing',
]


def load_data(run_dir: str):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    for p in [rate_path, init_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    df_rate = pd.read_csv(rate_path)
    df_init = pd.read_csv(init_path)
    return df_rate, df_init


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning-situation distribution over time."
    )
    parser.add_argument('--run_dir', required=True,
                        help="Fold directory containing traj_rate.csv and "
                             "traj_initmastery.csv")
    parser.add_argument('--output_dir', default=None,
                        help="Where to save plots (default: --run_dir)")
    parser.add_argument('--stride', type=int, default=10,
                        help="Snapshot every N interactions (default: 10)")
    parser.add_argument('--min_interactions', type=int, default=5,
                        help="Students with fewer interactions are excluded "
                             "(default: 5)")
    parser.add_argument('--max_interactions', type=int, default=None,
                        help="Cap trajectories at this length.  Defaults to "
                             "the 90th percentile of student interaction "
                             "counts (covers the region where at least 10%% "
                             "of students contribute data).")
    parser.add_argument('--uid_suffix', type=str, default='893468',
                        help="Suffix appended to output filenames "
                             "(default: 893468)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.run_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── load & merge ──────────────────────────────────────────────────────────
    df_rate, df_init = load_data(args.run_dir)

    rate_col = 'idkt_rate' if 'idkt_rate' in df_rate.columns else 'ts'
    init_col = 'idkt_im'   if 'idkt_im'   in df_init.columns else 'lc'
    merge_keys = (['student_id', 'interaction_idx']
                  if 'interaction_idx' in df_rate.columns
                  else ['student_id', 'skill_id'])

    df = df_rate.merge(df_init[merge_keys + [init_col]], on=merge_keys)

    # ── population medians (computed once on the full dataset) ────────────────
    l0_med = df[init_col].median()
    t_med  = df[rate_col].median()
    print(f"Population medians: {init_col}={l0_med:.4f}  {rate_col}={t_med:.4f}")

    def assign_quadrant(l0_mean, t_mean):
        if l0_mean <= l0_med and t_mean <= t_med: return 0   # foundational
        if l0_mean >  l0_med and t_mean <= t_med: return 1   # consolidating
        if l0_mean <= l0_med and t_mean >  t_med: return 2   # emerging
        return 3                                              # advancing

    # ── filter students ───────────────────────────────────────────────────────
    interaction_counts = df.groupby('student_id').size()
    eligible_ids = interaction_counts[
        interaction_counts >= args.min_interactions
    ].index
    df = df[df['student_id'].isin(eligible_ids)]
    n_students = df['student_id'].nunique()
    print(f"Students with >= {args.min_interactions} interactions: {n_students}")

    # ── build snapshots ───────────────────────────────────────────────────────
    # For each student, sort interactions and compute cumulative mean up to
    # each snapshot index.
    df = df.sort_values(['student_id', 'interaction_idx']).reset_index(drop=True)

    ic_all = df.groupby('student_id').size()
    p90    = int(ic_all.quantile(0.90))
    cap    = args.max_interactions if args.max_interactions is not None else p90
    print(f"Interaction cap: {cap}  (p90={p90})")
    max_idx = min(int(ic_all.max()), cap)
    snapshots = list(range(args.stride, max_idx + 1, args.stride))
    if not snapshots:
        raise ValueError(
            f"No snapshots produced with stride={args.stride} and "
            f"max_interactions={args.max_interactions}. Reduce --stride."
        )

    print(f"Snapshots at: {snapshots}")

    # ── per-snapshot quadrant assignment ─────────────────────────────────────
    # For each snapshot index s, assign each student to a quadrant based on
    # the mean of their parameters across all interactions up to s.
    all_uids = sorted(df['student_id'].unique())
    n_uids   = len(all_uids)
    uid_index = {uid: i for i, uid in enumerate(all_uids)}

    dist_counts  = np.zeros((len(snapshots), 4), dtype=int)
    dist_present = np.zeros(len(snapshots), dtype=int)
    student_quad = np.full((n_uids, len(snapshots)), np.nan)

    for si, snap in enumerate(snapshots):
        sub = df[df['interaction_idx'] < snap]
        if sub.empty:
            continue
        means = sub.groupby('student_id')[[init_col, rate_col]].mean()
        for uid, row in means.iterrows():
            q = assign_quadrant(row[init_col], row[rate_col])
            ui = uid_index[uid]
            student_quad[ui, si] = q
            dist_counts[si, q] += 1
            dist_present[si] += 1

    dist_frac = np.where(
        dist_present[:, None] > 0,
        dist_counts / dist_present[:, None],
        0.0
    )

    s_arr = np.array(snapshots)

    # ── Plot 1: stacked area chart (fractions) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(snapshots))
    for q in range(4):
        ax.fill_between(s_arr, bottom, bottom + dist_frac[:, q],
                        color=QUAD_COLORS[q], alpha=0.75, label=QUAD_LABELS[q])
        bottom += dist_frac[:, q]
    ax.set_xlim(s_arr[0], s_arr[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Interaction index', fontsize=12)
    ax.set_ylabel('Fraction of students', fontsize=12)
    ax.set_title('Learning situation distribution over time\n'
                 f'(snapshot every {args.stride} interactions, '
                 f'n={n_students} students)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85,
              title=f'x-axis capped at p90 ({cap} interactions):\n'
                    '90% of students have fewer interactions',
              title_fontsize=7.5)
    plt.tight_layout()
    path1 = os.path.join(output_dir,
                         f'situation_dist_stacked_{args.uid_suffix}.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved stacked area chart → {path1}")

    # ── Plot 2: absolute counts ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(snapshots))
    for q in range(4):
        ax.bar(s_arr, dist_counts[:, q], bottom=bottom,
               color=QUAD_COLORS[q], alpha=0.80,
               label=QUAD_LABELS[q], width=args.stride * 0.85)
        bottom += dist_counts[:, q]
    ax.set_xlim(s_arr[0] - args.stride, s_arr[-1] + args.stride)
    ax.set_xlabel('Interaction index', fontsize=12)
    ax.set_ylabel('Number of students', fontsize=12)
    ax.set_title('Learning situation distribution over time (absolute counts)\n'
                 f'(snapshot every {args.stride} interactions, '
                 f'n={n_students} students)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85,
              title=f'x-axis capped at p90 ({cap} interactions):\n'
                    '90% of students have fewer interactions',
              title_fontsize=7.5)
    plt.tight_layout()
    path2 = os.path.join(output_dir,
                         f'situation_dist_counts_{args.uid_suffix}.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved counts bar chart → {path2}")

    # ── Plot 3: student × snapshot heatmap ───────────────────────────────────
    # Sort students by their most frequent quadrant for a cleaner visual
    valid_mask = ~np.all(np.isnan(student_quad), axis=1)
    sq_valid   = student_quad[valid_mask]

    def dominant_quad(row):
        vals = row[~np.isnan(row)]
        if len(vals) == 0:
            return 4  # unknown
        counts = np.bincount(vals.astype(int), minlength=4)
        return int(np.argmax(counts))

    order = sorted(range(len(sq_valid)),
                   key=lambda i: dominant_quad(sq_valid[i]))
    sq_sorted = sq_valid[order]

    cmap = ListedColormap(QUAD_COLORS + ['#cccccc'])  # grey for NaN
    sq_display = np.where(np.isnan(sq_sorted), 4.0, sq_sorted)

    fig_h, ax_h = plt.subplots(figsize=(12, max(4, len(sq_sorted) // 20)))
    im = ax_h.imshow(sq_display, aspect='auto', cmap=cmap,
                     vmin=0, vmax=4, interpolation='nearest')
    ax_h.set_xlabel('Snapshot index', fontsize=11)
    ax_h.set_ylabel('Students (sorted by dominant situation)', fontsize=11)
    tick_step = max(1, len(snapshots) // 10)
    ax_h.set_xticks(range(0, len(snapshots), tick_step))
    ax_h.set_xticklabels([str(snapshots[i])
                          for i in range(0, len(snapshots), tick_step)],
                         fontsize=8)
    ax_h.set_yticks([])
    ax_h.set_title('Per-student learning situation across snapshots\n'
                   f'(snapshot every {args.stride} interactions, '
                   f'n={len(sq_sorted)} students)',
                   fontsize=12)
    legend_handles = [
        mpatches.Patch(color=QUAD_COLORS[q], label=QUAD_LABELS[q])
        for q in range(4)
    ] + [mpatches.Patch(color='#cccccc', label='No data')]
    ax_h.legend(handles=legend_handles, loc='lower right',
                fontsize=8, framealpha=0.85, ncol=2,
                bbox_to_anchor=(1.0, -0.18))
    plt.tight_layout()
    path3 = os.path.join(output_dir,
                         f'situation_dist_heatmap_{args.uid_suffix}.png')
    fig_h.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig_h)
    print(f"Saved student heatmap → {path3}")


if __name__ == '__main__':
    main()
