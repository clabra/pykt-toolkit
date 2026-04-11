#!/usr/bin/env python3
"""
Visualise learning situation change from session entry to session exit.

For each student, two situation assignments are computed:
  - Entry: mean parameters over the first --window interactions
  - Exit:  mean parameters over the last --window interactions

The figure has two panels:
  Left  — grouped bar chart comparing the entry and exit distributions
          across the four learning situations.
  Right — 4×4 transition matrix (entry situation → exit situation),
          row-normalised to transition probabilities; raw counts shown
          in each cell.

A text annotation reports the fraction of students whose exit situation
differs from their entry situation, providing evidence that the model
tracks genuine learning progression rather than assigning static labels.

Output:
  situation_entry_exit_<uid_suffix>.png

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


QUAD_COLORS = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
QUAD_LABELS = ['Foundational', 'Consolidating', 'Emerging', 'Advancing']
QUAD_SHORT  = ['Found.', 'Consol.', 'Emerg.', 'Advanc.']


def load_data(run_dir):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    for p in [rate_path, init_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    return pd.read_csv(rate_path), pd.read_csv(init_path)


def main():
    parser = argparse.ArgumentParser(
        description="Entry-vs-exit learning situation transition figure."
    )
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--window', type=int, default=30,
                        help="Number of interactions used for entry and exit "
                             "windows (default: 30).  Students with fewer "
                             "than 2 × window interactions are excluded so "
                             "the two windows do not overlap.")
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

    df = df.sort_values(['student_id', 'interaction_idx']).reset_index(drop=True)

    # ── entry / exit assignment per student ───────────────────────────────
    min_required = 2 * args.window
    records = []
    for uid, grp in df.groupby('student_id'):
        grp = grp.reset_index(drop=True)
        if len(grp) < min_required:
            continue
        entry_rows = grp.iloc[:args.window]
        exit_rows  = grp.iloc[-args.window:]
        q_entry = assign_quadrant(entry_rows[init_col].mean(),
                                  entry_rows[rate_col].mean())
        q_exit  = assign_quadrant(exit_rows[init_col].mean(),
                                  exit_rows[rate_col].mean())
        records.append({'uid': uid, 'entry': q_entry, 'exit': q_exit,
                        'n': len(grp)})

    if not records:
        raise ValueError(
            f"No students have >= {min_required} interactions. "
            f"Reduce --window."
        )

    result = pd.DataFrame(records)
    n_students = len(result)
    n_changed  = (result['entry'] != result['exit']).sum()
    pct_changed = n_changed / n_students * 100
    print(f"Students with >= {min_required} interactions: {n_students}")
    print(f"Students who changed situation: {n_changed} / {n_students} "
          f"({pct_changed:.1f}%)")

    # ── transition matrix ─────────────────────────────────────────────────
    trans = np.zeros((4, 4), dtype=int)
    for _, row in result.iterrows():
        trans[int(row['entry']), int(row['exit'])] += 1

    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = np.where(row_sums > 0, trans / row_sums, 0.0)

    # ── entry / exit marginal percentages ─────────────────────────────────
    entry_counts = np.array([
        (result['entry'] == q).sum() for q in range(4)
    ])
    exit_counts = np.array([
        (result['exit'] == q).sum() for q in range(4)
    ])
    entry_pct = entry_counts / n_students * 100
    exit_pct  = exit_counts  / n_students * 100

    # ── Figure 1: alluvial diagram ────────────────────────────────────────
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    GAP   = 1.2
    BW    = 0.08

    def stacked_bottoms(pcts, gap):
        bottoms = []
        y = 0.0
        for p in pcts:
            bottoms.append(y)
            y += p + gap
        return bottoms

    entry_pct_arr = entry_counts / n_students * 100
    exit_pct_arr  = exit_counts  / n_students * 100

    entry_bottoms = stacked_bottoms(entry_pct_arr, GAP)
    exit_bottoms  = stacked_bottoms(exit_pct_arr,  GAP)

    total_height = max(
        entry_bottoms[-1] + entry_pct_arr[-1],
        exit_bottoms[-1]  + exit_pct_arr[-1],
    ) + GAP

    fig1, ax_al = plt.subplots(figsize=(7, 6), constrained_layout=True)

    for q in range(4):
        ax_al.add_patch(plt.Rectangle(
            (0 - BW, entry_bottoms[q]), BW * 2, entry_pct_arr[q],
            facecolor=QUAD_COLORS[q], edgecolor='none', zorder=3, alpha=0.9
        ))
        ax_al.add_patch(plt.Rectangle(
            (1 - BW, exit_bottoms[q]), BW * 2, exit_pct_arr[q],
            facecolor=QUAD_COLORS[q], edgecolor='none', zorder=3, alpha=0.9
        ))

    entry_offsets = [entry_bottoms[q] for q in range(4)]
    exit_offsets  = [exit_bottoms[q]  for q in range(4)]

    draw_order = sorted(
        [(i, j) for i in range(4) for j in range(4) if trans[i, j] > 0],
        key=lambda ij: (ij[0] == ij[1], trans[ij[0], ij[1]])
    )

    for i, j in draw_order:
        cnt  = trans[i, j]
        if cnt == 0:
            continue
        height = cnt / n_students * 100
        y0_bot = entry_offsets[i]
        y0_top = y0_bot + height
        y1_bot = exit_offsets[j]
        y1_top = y1_bot + height
        entry_offsets[i] += height
        exit_offsets[j]  += height

        ctrl_x = 0.42
        verts = [
            (BW,         y0_bot),
            (ctrl_x,     y0_bot),
            (1-ctrl_x,   y1_bot),
            (1-BW,       y1_bot),
            (1-BW,       y1_top),
            (1-ctrl_x,   y1_top),
            (ctrl_x,     y0_top),
            (BW,         y0_top),
            (BW,         y0_bot),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        is_stay = (i == j)
        alpha   = 0.25 + 0.45 * (height / entry_pct_arr.max())
        ax_al.add_patch(PathPatch(
            Path(verts, codes),
            facecolor=QUAD_COLORS[i],
            edgecolor=QUAD_COLORS[i] if is_stay else 'none',
            linewidth=0.4 if is_stay else 0,
            alpha=0.65 if is_stay else alpha,
            zorder=2 if is_stay else 1,
        ))

    for q in range(4):
        mid_entry = entry_bottoms[q] + entry_pct_arr[q] / 2
        mid_exit  = exit_bottoms[q]  + exit_pct_arr[q]  / 2
        ax_al.text(-BW - 0.02, mid_entry,
                   f'{QUAD_SHORT[q]}\n{entry_pct_arr[q]:.1f}%',
                   ha='right', va='center', fontsize=8,
                   color=QUAD_COLORS[q], fontweight='bold')
        ax_al.text(1 + BW + 0.02, mid_exit,
                   f'{QUAD_SHORT[q]}\n{exit_pct_arr[q]:.1f}%',
                   ha='left', va='center', fontsize=8,
                   color=QUAD_COLORS[q], fontweight='bold')

    ax_al.set_xlim(-0.35, 1.35)
    ax_al.set_ylim(-GAP, total_height)
    ax_al.axis('off')
    ax_al.text(0, -GAP * 0.6, f'Entry\n(first {args.window} interactions)',
               ha='center', va='top', fontsize=9, style='italic')
    ax_al.text(1, -GAP * 0.6, f'Exit\n(last {args.window} interactions)',
               ha='center', va='top', fontsize=9, style='italic')
    ax_al.set_title(
        f'Learning situation: entry vs. exit\n'
        f'(n = {n_students} students with \u2265 {min_required} interactions; '
        f'{pct_changed:.1f}% changed situation)',
        fontsize=10, pad=8
    )

    out_alluvial = os.path.join(output_dir,
                                f'situation_alluvial_{args.uid_suffix}.png')
    fig1.savefig(out_alluvial, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved \u2192 {out_alluvial}")

    # ── Figure 2: transition probability heatmap ──────────────────────────
    fig2, ax_heat = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)

    im = ax_heat.imshow(trans_prob, cmap='YlOrRd', vmin=0, vmax=1,
                        aspect='auto', interpolation='nearest')

    for i in range(4):
        for j in range(4):
            prob = trans_prob[i, j]
            cnt  = trans[i, j]
            text_color = 'white' if prob > 0.55 else 'black'
            ax_heat.text(j, i, f'{prob*100:.1f}%\n({cnt})',
                         ha='center', va='center',
                         fontsize=8, color=text_color)

    ax_heat.set_xticks(range(4))
    ax_heat.set_yticks(range(4))
    ax_heat.set_xticklabels(QUAD_SHORT, fontsize=9, rotation=20, ha='right')
    ax_heat.set_yticklabels(QUAD_SHORT, fontsize=9)
    ax_heat.set_xlabel(f'Exit situation (last {args.window} interactions)', fontsize=10)
    ax_heat.set_ylabel(f'Entry situation (first {args.window} interactions)', fontsize=10)
    ax_heat.set_title(
        f'Entry \u2192 exit transition probabilities\n'
        f'(n = {n_students} students with \u2265 {min_required} interactions; '
        f'each row sums to 100%, counts in parentheses)',
        fontsize=9.5, pad=6
    )

    cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label('Transition probability', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    out_matrix = os.path.join(output_dir,
                              f'situation_transition_matrix_{args.uid_suffix}.png')
    fig2.savefig(out_matrix, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved \u2192 {out_matrix}")

    # ── print dominant off-diagonal transitions for paper narrative ───────
    print("\nDominant off-diagonal transitions (entry \u2192 exit, count):")
    off_diag = [(trans[i, j], QUAD_LABELS[i], QUAD_LABELS[j])
                for i in range(4) for j in range(4) if i != j]
    off_diag.sort(reverse=True)
    for cnt, src, dst in off_diag[:8]:
        print(f"  {src} \u2192 {dst}: {cnt}")


if __name__ == '__main__':
    main()

