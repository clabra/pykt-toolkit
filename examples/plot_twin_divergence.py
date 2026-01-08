import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_enhanced_twins(csv_path, skill_id, uids, output_path):
    df = pd.read_csv(csv_path)
    df['student_id'] = df['student_id'].astype(str)
    
    # Calculate global accuracy for context
    global_stats = {}
    for uid in uids:
        u_data = df[df['student_id'] == uid]
        global_stats[uid] = u_data['y_true'].mean()

    plt.figure(figsize=(14, 7))
    
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red for Low/Fast, Blue for High/Steady
    markers = {0: 'x', 1: 'o'}
    marker_sizes = {0: 100, 1: 100}
    
    # Plot BKT (Identical for both)
    first_uid = uids[0]
    first_data = df[(df['student_id'] == first_uid) & (df['skill_id'] == skill_id)].reset_index()
    plt.plot(first_data.index, first_data['p_bkt'], label='BKT State (Markovian)', 
             color='gray', linestyle='--', linewidth=3, alpha=0.4, zorder=1)
    
    # Sort UIDs by global accuracy to assign descriptive roles
    sorted_uids = sorted(uids, key=lambda x: global_stats[x], reverse=True)
    
    # Plot iDKT for each student
    for i, uid in enumerate(uids):
        student_data = df[(df['student_id'] == uid) & (df['skill_id'] == skill_id)].reset_index()
        g_acc = global_stats[uid]
        
        # Determine Profile Label based on relative performance
        if uid == sorted_uids[0]:
            profile = f"Student {uid}: 'High-Velocity Generalist'\n(Global Success: {g_acc:.1%})\nStarts low here, but masters rapidly."
        else:
            profile = f"Student {uid}: 'Struggling Specialist'\n(Global Success: {g_acc:.1%})\nKnows the basics here, but learning is slow."

        # Line plot
        plt.plot(student_data.index, student_data['p_idkt'], label=profile, 
                 color=colors[i % len(colors)], linewidth=3, zorder=2)
        
        # Markers for Outcomes
        for idx, row in student_data.iterrows():
            m = 'o' if row['y_true'] == 1 else 'x'
            plt.scatter(idx, row['p_idkt'], marker=m, s=120, edgecolors='black' if m=='o' else None,
                        c=colors[i % len(colors)], zorder=3, linewidths=2)
            # Duplicate for BKT to show they are same inputs
            plt.scatter(idx, row['p_bkt'], marker=m, s=80, c='gray', alpha=0.3, zorder=1)

    plt.xlabel('Interaction Sequence (Skill 4)', fontsize=13, fontweight='bold')
    plt.ylabel('Knowledge Mastery Estimation', fontsize=13, fontweight='bold')
    plt.title('The iDKT Advantage: Contextualization vs. Markovian Baselines\n(Identical Response Sequences yielding Divergent Pedagogical Insights)', fontsize=15, pad=20)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-0.05, 1.1)
    plt.xticks(range(len(first_data)))
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
    
    # Add explanatory annotations
    # Focus on the divergent start (Contextualization)
    plt.annotate('Non-Markovian Initialization:\niDKT differentiates starting points\nvia prior context.', 
                 xy=(0, 0.3), xytext=(0.5, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.1))

    # Focus on the cross-over (Velocity)
    plt.annotate('Cross-Over Point:\nLower baseline but higher\nlearning velocity.', 
                 xy=(1.5, 0.32), xytext=(2.5, 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Enhanced plot saved to {output_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot twin divergence for iDKT paper.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing traj_predictions.csv")
    parser.add_argument("--skill_id", type=int, default=11, help="Skill ID to plot")
    parser.add_argument("--uids", type=str, default="1154,306", help="Comma-separated student IDs to plot")
    parser.add_argument("--output", type=str, default="paper/latex/img/twin_divergence.png", help="Output path for the plot")
    args = parser.parse_args()

    csv_path = os.path.join(args.run_dir, "traj_predictions.csv")
    uids = args.uids.split(",")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    # Verify UIDs exist in the CSV
    df_check = pd.read_csv(csv_path)
    df_check['student_id'] = df_check['student_id'].astype(str)
    available_uids = df_check['student_id'].unique()
    
    final_uids = []
    for uid in uids:
        if uid in available_uids:
            final_uids.append(uid)
        else:
            print(f"Warning: Student {uid} not found in {csv_path}. Skipping.")
    
    if not final_uids:
        print("Error: No valid student IDs found. Available IDs:", available_uids[:20], "...")
        sys.exit(1)

    plot_enhanced_twins(csv_path, skill_id=args.skill_id, uids=final_uids, output_path=args.output)
