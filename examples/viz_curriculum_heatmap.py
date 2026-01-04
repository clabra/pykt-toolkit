
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Config ---
EXP_DIR = "/workspaces/pykt-toolkit/experiments/20251230_224907_idkt_setS-pure_assist2009_baseline_364494"
OUTPUT_DIR = os.path.join(EXP_DIR, "probing_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="white", context="paper")
plt.rcParams['font.family'] = 'serif'

def main():
    print("Loading data for curriculum calibration...")
    df_traj = pd.read_csv(os.path.join(EXP_DIR, "traj_predictions.csv"))
    df_roster = pd.read_csv(os.path.join(EXP_DIR, "roster_idkt.csv"))

    # 1. Calculate Skill Difficulty (Global)
    # Difficulty = 1 - Accuracy
    skill_difficulty = df_traj.groupby('skill_id')['y_true'].mean().reset_index()
    skill_difficulty['difficulty'] = 1 - skill_difficulty['y_true']
    skill_difficulty = skill_difficulty[['skill_id', 'difficulty']]

    # 2. Get Student Clusters (Knowledge Space Profile)
    last_states = df_roster.groupby('student_id').tail(1)
    skill_cols = [c for c in last_states.columns if c.startswith('S')]
    profiles = last_states[skill_cols].values
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    original_clusters = kmeans.fit_predict(profiles)

    # Re-order logic: consistent Left -> Right on PC1
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(profiles).flatten()
    df_temp = pd.DataFrame({'pc1': pc1, 'orig': original_clusters})
    cluster_order = df_temp.groupby('orig')['pc1'].mean().sort_values().index.tolist()
    remap = {orig: new for new, orig in enumerate(cluster_order)}
    last_states = last_states.copy()
    last_states['Cluster'] = [f'Cluster {remap[c]}' for c in original_clusters]

    # 3. Aggregate Mastery per Skill per Cluster
    cluster_mastery = last_states.melt(id_vars=['student_id', 'Cluster'], value_vars=skill_cols, 
                                       var_name='skill_idx', value_name='mastery')
    cluster_mastery['skill_id_num'] = cluster_mastery['skill_idx'].str[1:].astype(int)

    # 4. Join with Difficulty
    merged = pd.merge(cluster_mastery, skill_difficulty, left_on='skill_id_num', right_on='skill_id')

    # Group skills into difficulty deciles
    merged['difficulty_bin'] = pd.qcut(merged['difficulty'], q=10, labels=False, duplicates='drop')

    # 5. Pivot for Heatmap
    pivot = merged.pivot_table(index='Cluster', columns='difficulty_bin', values='mastery', aggfunc='mean')
    
    # 6. Plotting
    plt.figure(figsize=(14, 7))
    ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                     cbar_kws={'label': 'Mean Latent Mastery ($\mathbf{h}_T$)'},
                     annot_kws={"size": 11})
    
    plt.title('Curriculum Calibration: Cluster Mastery vs. Global Skill Difficulty', fontsize=18, pad=20)
    plt.ylabel('Diagnostic Cluster Standing', fontsize=14)
    plt.xlabel('Skill Difficulty Decile (0=Easiest $\\rightarrow$ 9=Hardest)', fontsize=14)
    
    # Add cluster color indicators if possible or just rely on IDs 0-3
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "skill_difficulty_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved curriculum calibration heatmap to {output_path}")

if __name__ == "__main__":
    main()
