
import os
import sys
import json
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
ROSTER_PATH = os.path.join(EXP_DIR, "roster_idkt.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="white", context="paper")
plt.rcParams['font.family'] = 'serif'

def get_skill_names():
    """Maps skill indices to names using the mapping files."""
    try:
        names_df = pd.read_csv('/workspaces/pykt-toolkit/data/assist2009/skill_builder_data_corrected_collapsed.csv', 
                               low_memory=False, encoding='ISO-8859-1')
        def clean_id(x):
            try: return str(int(float(x)))
            except: return str(x)
        names_df['skill_id_str'] = names_df['skill_id'].apply(clean_id)
        id_to_name = names_df.dropna(subset=['skill_id_str', 'skill_name']).groupby('skill_id_str')['skill_name'].first().to_dict()

        with open('/workspaces/pykt-toolkit/data/assist2009_S/keyid2idx.json') as f:
            keyid_data = json.load(f)
        keyid2idx = keyid_data['concepts']

        idx_to_name = {}
        for kid, idx in keyid2idx.items():
            kid_str = clean_id(kid)
            name = id_to_name.get(kid_str, f'Skill {kid}')
            
            # Manual Overrides for specific IDs requested by user
            if kid == '367': name = 'Rounding'
            if kid == '102': name = 'Nets of 3D Figures'
            
            idx_to_name[idx] = name
        return idx_to_name
    except Exception as e:
        print(f"Skill name mapping failed: {e}")
        return {}

def main():
    print("Loading data for skill-by-cluster map...")
    df_roster = pd.read_csv(ROSTER_PATH)
    idx_to_name = get_skill_names()

    # 1. Consistent Clustering Logic
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

    # 2. Calculate Mean Mastery per Skill per Cluster
    cluster_skill_means = last_states.groupby('Cluster')[skill_cols].mean()
    
    # Rename columns to skill names
    new_cols = []
    for col in skill_cols:
        idx = int(col[1:])
        new_cols.append(idx_to_name.get(idx, col))
    cluster_skill_means.columns = new_cols

    # 3. Transpose for better display (Skills on Y, Clusters on X)
    # Educators usually prefer scrolling vertically through a list of skills
    pivot = cluster_skill_means.transpose()
    
    # 4. Sort skills by global mastery to show a gradient
    pivot['Global_Mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Global_Mean', ascending=False)
    pivot = pivot.drop(columns='Global_Mean')

    print(f"Generating full Skill-by-Cluster Map with {len(pivot)} skills...")
    
    # Calculate figure height dynamically based on number of skills
    fig_height = len(pivot) * 0.25 # Adjust spacing for readability
    plt.figure(figsize=(12, max(10, fig_height)))
    
    ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                     cbar_kws={'label': 'Latent Mastery ($\mathbf{h}_T$)', 'location': 'top'},
                     annot_kws={"size": 8})
    
    # plt.title('Pedagogical Diagnostic Map: Skill Proficiency by Student Cluster', fontsize=18, pad=40)
    plt.xlabel('Diagnostic Cluster', fontsize=14)
    plt.ylabel('Curriculum Skills (Sorted by Global Mastery)', fontsize=14)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "pedagogical_skill_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Generate a 'Top/Bottom' version for easier inclusion in a paper
    print("Generating concise Skill-by-Cluster Map...")
    concise_pivot = pd.concat([pivot.head(15), pivot.tail(15)])
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(concise_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Latent Mastery ($\mathbf{h}_T$)'})
    # plt.title('Concise Pedagogical Map: Extremes of Mastery', fontsize=16)
    plt.tight_layout()
    output_path_concise = os.path.join(OUTPUT_DIR, "pedagogical_skill_map_concise.png")
    plt.savefig(output_path_concise, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved skill maps to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
