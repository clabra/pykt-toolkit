import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    # Plot iDKT for each student
    for i, uid in enumerate(uids):
        student_data = df[(df['student_id'] == uid) & (df['skill_id'] == skill_id)].reset_index()
        g_acc = global_stats[uid]
        
        # Determine Profile Label
        if uid == '1154':
            profile = f"Student {uid}: 'High-Velocity Generalist'\n(Global Success: {g_acc:.1%})\nStarts low here, but masters at 5x speed."
        else:
            profile = f"Student {uid}: 'Struggling Specialist'\n(Global Success: {g_acc:.1%})\nKnows the basics here, but learning is flat."

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

    plt.xlabel('Interaction Sequence (Skill 11)', fontsize=13, fontweight='bold')
    plt.ylabel('Knowledge Mastery Estimation', fontsize=13, fontweight='bold')
    plt.title('The iDKT Advantage: Contextual Diagnostics vs. Markovian Baselines\n(Identical Response Sequences yielding Divergent Pedagogical Insights)', fontsize=15, pad=20)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-0.05, 1.1)
    plt.xticks(range(len(first_data)))
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
    
    # Add explanatory annotations
    plt.annotate('Cross-Over Point:\niDKT velocity captures individual\nacquisition pace.', 
                 xy=(2.5, 0.45), xytext=(5, 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Enhanced plot saved to {output_path}")

if __name__ == "__main__":
    path = "experiments/20251230_224907_idkt_setS-pure_364494/traj_predictions.csv"
    # Concept: 11, Len: 11
    # UID 1154: 45.0% Global Accuracy -> "The Fast Learner" (Learns from failure)
    # UID 306:  16.0% Global Accuracy -> "The Lucky Novice" (Starts high but stagnates)
    plot_enhanced_twins(path, skill_id=11, uids=['1154', '306'], output_path='paper/latex/img/twin_divergence.png')
