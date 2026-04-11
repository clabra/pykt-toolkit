#!/usr/bin/env python3
"""
Generate a two-panel figure that highlights the dynamic nature of learning
situation assignments over time.

Top panel — stacked bar chart (absolute student counts per learning situation
    at each interaction snapshot), with per-situation trend lines overlaid to
    make the temporal evolution salient.

Bottom panel — "transition churn": the fraction of students whose assigned
    learning situation changed relative to the previous snapshot.  This panel
    provides direct empirical evidence that situations are not static labels.

Output:
  situation_dynamics_<uid_suffix>.png

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
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d


# ── shared palette ─────────────────────────────────────────────────────────
QUAD_COLORS = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
QUAD_LABELS = ['Foundational', 'Consolidating', 'Emerging', 'Advancing']


def load_data(run_dir: str):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    for p in [rate_path, init_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    return pd.read_csv(rate_path), pd.read_csv(init_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning-situation dynamics (counts + churn)."
    )
    parser.add_argument('--run_dir', required=True,
                        help="Fold directory with traj_rate.csv / "
                             "traj_initmastery.csv")
    parser.add_argument('--output_dir', default=None,
                        help="Output directory (default: --run_dir)")
    parser.add_argument('--stride', type=int, default=10,
                        help="Snapshot every N interactions (default: 10)")
    parser.add_argument('--min_interactions', type=int, default=5,
                        help="Exclude students with fewer interactions "
                             "(default: 5)")
    parser.add_argument('--max_interactions', type=int, default=None,
                        help="Cap trajectory length.  Defaults to the 90th "
                             "percentile of student interaction counts, so the "
                             "x-axis covers the region where at least 10%% of "
                             "students contribute data.")
    parser.add_argument('--smooth_window', type=int, default=5,
                        help="Uniform smoothing window applied to trend lines "
                             "and churn curve (default: 5; set 1 to disable)")
    parser.add_argument('--uid_suffix', type=str, default='893468',
                        help="Filename suffix (default: 893468)")
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

    # ── population medians ────────────────────────────────────────────────
    l0_med = df[init_col].median()
    t_med  = df[rate_col].median()
    print(f"Population medians: {init_col}={l0_med:.4f}  {rate_col}={t_med:.4f}")

    def assign_quadrant(l0_mean, t_mean):
        if l0_mean <= l0_med and t_mean <= t_med:  return 0  # foundational
        if l0_mean >  l0_med and t_mean <= t_med:  return 1  # consolidating
        if l0_mean <= l0_med and t_mean >  t_med:  return 2  # emerging
        return 3                                              # advancing

    # ── filter students ───────────────────────────────────────────────────
    ic = df.groupby('student_id').size()
    eligible = ic[ic >= args.min_interactions].index
    df = df[df['student_id'].isin(eligible)]
    n_students = df['student_id'].nunique()
    print(f"Students with >= {args.min_interactions} interactions: {n_students}")

    df = df.sort_values(['student_id', 'interaction_idx']).reset_index(drop=True)

    ic_all = df.groupby('student_id').size()
    p90    = int(ic_all.quantile(0.90))
    cap    = args.max_interactions if args.max_interactions is not None else p90
    print(f"Interaction cap: {cap}  (p90={p90})")
    max_idx   = min(int(ic_all.max()), cap)
    snapshots = list(range(args.stride, max_idx + 1, args.stride))
    if not snapshots:
        raise ValueError(
            f"No snapshots with stride={args.stride} and "
            f"max_interactions={args.max_interactions}."
        )
    print(f"Snapshots: {snapshots[0]} … {snapshots[-1]} "
          f"({len(snapshots)} steps)")

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

    s_arr = np.array(snapshots)

    # ── churn: fraction of students who changed situation vs previous snap ─
    churn = np.full(len(snapshots), np.nan)
    for si in range(1, len(snapshots)):
        col_prev = student_quad[:, si - 1]
        col_curr = student_quad[:, si]
        both_valid = (~np.isnan(col_prev)) & (~np.isnan(col_curr))
        if both_valid.sum() > 0:
            changed = (col_prev[both_valid] != col_curr[both_valid]).sum()
            churn[si] = changed / both_valid.sum()

    # ── smooth curves for trend lines and churn ───────────────────────────
    w = max(1, args.smooth_window)

    def smooth(arr):
        if w <= 1:
            return arr.copy()
        # only smooth where we have data; pad edges with edge values
        return uniform_filter1d(arr.astype(float), size=w, mode='nearest')

    counts_smooth = np.stack(
        [smooth(dist_counts[:, q].astype(float)) for q in range(4)], axis=1
    )
    churn_smooth = np.where(np.isnan(churn), np.nan,
                            smooth(np.nan_to_num(churn, nan=0.0)))

    # descriptive statistics for annotation
    mean_churn = float(np.nanmean(churn[1:]))
    peak_churn_idx = int(np.nanargmax(churn[1:])) + 1
    peak_churn_val = float(churn[peak_churn_idx])

    # ── figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.2], hspace=0.38)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    bar_w = args.stride * 0.82

    # ── top panel: stacked bars + trend lines ─────────────────────────────
    bottom = np.zeros(len(snapshots))
    bar_bottoms = []
    for q in range(4):
        bar_bottoms.append(bottom.copy())
        ax_top.bar(s_arr, dist_counts[:, q], bottom=bottom,
                   color=QUAD_COLORS[q], alpha=0.45, width=bar_w,
                   linewidth=0)
        bottom += dist_counts[:, q]

    # trend lines: centre of each segment
    for q in range(4):
        mid = bar_bottoms[q] + dist_counts[:, q] / 2.0
        mid_s = bar_bottoms[q] + counts_smooth[:, q] / 2.0
        ax_top.plot(s_arr, smooth(mid), color=QUAD_COLORS[q],
                    linewidth=2.2, alpha=0.95, zorder=5)

    # total students curve (right axis)
    ax_r = ax_top.twinx()
    ax_r.plot(s_arr, dist_present, color='#555555', linewidth=1.2,
              linestyle='--', alpha=0.5, label='Students present')
    ax_r.set_ylabel('Students present', fontsize=9, color='#555555')
    ax_r.tick_params(axis='y', labelcolor='#555555', labelsize=8)
    ax_r.set_ylim(0, dist_present.max() * 1.3)

    ax_top.set_xlim(s_arr[0] - args.stride * 0.6,
                    s_arr[-1] + args.stride * 0.6)
    ax_top.set_ylabel('Number of students', fontsize=11)
    ax_top.set_title(
        'Learning situation distribution over time  '
        f'(n = {n_students} students, snapshot every {args.stride} interactions)',
        fontsize=11, pad=8
    )
    legend_patches = [
        mpatches.Patch(color=QUAD_COLORS[q], alpha=0.7, label=QUAD_LABELS[q])
        for q in range(4)
    ]
    legend_patches.append(
        mpatches.Patch(color='none',
                       label=f'x-axis capped at p90 ({cap} interactions):\n'
                             f'90% of students have fewer interactions')
    )
    ax_top.legend(handles=legend_patches, loc='upper right',
                  fontsize=8.5, framealpha=0.85, ncol=2)
    ax_top.tick_params(axis='x', labelbottom=False)

    # ── bottom panel: churn ───────────────────────────────────────────────
    ax_bot.fill_between(s_arr[1:], 0, churn_smooth[1:] * 100,
                        color='#7f8c8d', alpha=0.25)
    ax_bot.plot(s_arr[1:], churn_smooth[1:] * 100,
                color='#2c3e50', linewidth=2.0)

    # mean churn reference line
    ax_bot.axhline(mean_churn * 100, color='#c0392b', linewidth=1.0,
                   linestyle=':', alpha=0.85)
    ax_bot.annotate(
        f'mean {mean_churn * 100:.1f}%',
        xy=(s_arr[-1], mean_churn * 100),
        xytext=(-6, 4), textcoords='offset points',
        ha='right', va='bottom', fontsize=8, color='#c0392b'
    )

    # annotate peak
    ax_bot.annotate(
        f'peak\n{peak_churn_val * 100:.1f}%',
        xy=(s_arr[peak_churn_idx], churn_smooth[peak_churn_idx] * 100),
        xytext=(10, 6), textcoords='offset points',
        ha='left', va='bottom', fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color='#555', lw=0.8),
        color='#2c3e50'
    )

    ax_bot.set_xlabel('Interaction index', fontsize=11)
    ax_bot.set_ylabel('Students\nchanging\nsituation (%)', fontsize=9)
    ax_bot.set_ylim(0, max(churn_smooth[1:]) * 100 * 1.35)
    ax_bot.set_title(
        'Transition churn — fraction of students reassigned '
        'at each snapshot',
        fontsize=10, pad=4
    )
    ax_bot.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.0f}%')
    )

    out_path = os.path.join(output_dir,
                            f'situation_dynamics_{args.uid_suffix}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved dynamics plot → {out_path}")
    print(f"  mean churn: {mean_churn*100:.1f}%  "
          f"  peak churn: {peak_churn_val*100:.1f}% "
          f"at interaction {s_arr[peak_churn_idx]}")


if __name__ == '__main__':
    main()
