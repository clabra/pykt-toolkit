
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Config ---
EXP_DIR = "/workspaces/pykt-toolkit/experiments/20251230_224907_idkt_setS-pure_assist2009_baseline_364494"
CHECKPOINT = os.path.join(EXP_DIR, "best_model.pt")
ROSTER_PATH = os.path.join(EXP_DIR, "roster_idkt.csv")
OUTPUT_DIR = os.path.join(EXP_DIR, "probing_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'

def cluster_placement_vs_pacing():
    """Clusters students based on Alpha (Init) and Beta (Learning Rate)."""
    print("Extracting individualized parameters...")
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Indices 1 to N (skip padding at 0)
    alpha = state_dict['student_param.weight'].numpy().flatten()[1:]
    beta = state_dict['student_gap_param.weight'].numpy().flatten()[1:]
    
    # Filter out unused weights (if n_uid was large but dataset small)
    # Actually, iDKT initializes based on n_uid.
    
    # Scale for clustering
    X = np.stack([alpha, beta], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    original_clusters = kmeans.fit_predict(X_scaled)
    
    # Re-order based on overall proficiency (alpha + beta)
    df_temp = pd.DataFrame({'a': X_scaled[:, 0], 'b': X_scaled[:, 1], 'orig_cluster': original_clusters})
    # Proficiency index: lower is more 'at-risk', higher is 'accelerated'
    cluster_order = df_temp.groupby('orig_cluster').apply(lambda g: g['a'].mean() + g['b'].mean()).sort_values().index.tolist()
    
    # Map original cluster ID to sorted ID
    remap = {orig: new for new, orig in enumerate(cluster_order)}
    sorted_clusters = np.array([remap[c] for c in original_clusters])
    
    label_map = {
        0: 'Initial Standing / Consolidating Pacing',
        1: 'Initial Standing / Developing Pacing',
        2: 'Initial Standing / High-Velocity Growth',
        3: 'Advanced Standing / Sustained Acceleration'
    }
    
    df = pd.DataFrame({
        'Initial Mastery ($\\alpha_s$)': alpha,
        'Learning Velocity ($\\beta_s$)': beta,
        'Cluster': [f'Cluster {c}' for c in sorted_clusters]
    })
    
    # Sort for legend
    df = df.sort_values('Cluster')
    
    # Custom palette matching Skill Profile
    custom_palette = {
        'Cluster 0': '#e74c3c', # Red
        'Cluster 1': '#f39c12', # Orange
        'Cluster 2': '#3498db', # Blue
        'Cluster 3': '#2ecc71'  # Green
    }
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Initial Mastery ($\\alpha_s$)', y='Learning Velocity ($\\beta_s$)', 
                    hue='Cluster', palette=custom_palette, s=60, alpha=0.7, edgecolors='black', linewidth=0.3)
    
    # plt.title("Student Clusters: Placement vs. Pacing", fontsize=16)
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_placement_pacing.png"), dpi=300)
    plt.close()
    print(f"Saved cluster plot to {OUTPUT_DIR}/cluster_placement_pacing.png")

def cluster_skill_profiles():
    """Clusters students based on their final mastery across all skills."""
    print("Loading skill profiles from roster...")
    df_roster = pd.read_csv(ROSTER_PATH)
    
    # Get last state for each student
    last_states = df_roster.groupby('student_id').tail(1)
    
    # Select skill columns (S0, S1, ...)
    skill_cols = [c for c in last_states.columns if c.startswith('S')]
    profiles = last_states[skill_cols].values
    
    print(f"Clustering {len(profiles)} students based on {len(skill_cols)} skill dimensions...")
    
    # PCA to reduce skill space for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(profiles)
    
    # Clustering on the full profile space
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    original_clusters = kmeans.fit_predict(profiles)
    
    # Re-order clusters from Left (min PC1) to Right (max PC1)
    df_temp = pd.DataFrame({'PC1': X_pca[:, 0], 'orig_cluster': original_clusters})
    cluster_order = df_temp.groupby('orig_cluster')['PC1'].mean().sort_values().index.tolist()
    # cluster_order[0] is the index of the leftmost cluster
    
    # Map original cluster ID to sorted ID
    remap = {orig: new for new, orig in enumerate(cluster_order)}
    sorted_clusters = np.array([remap[c] for c in original_clusters])
    
    label_map = {
        0: 'Focus Area: Foundational Concepts',
        1: 'Profile: Arithmetic Development',
        2: 'Profile: Conceptual & Spatial Reasoning',
        3: 'Profile: Advanced Numerical Fluency'
    }
    
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f'Cluster {c}' for c in sorted_clusters]
    })
    
    # Ensure legend is sorted by Cluster ID
    df_pca = df_pca.sort_values('Cluster')
    
    # Define custom palette: Red, Orange, Blue, Green (Left-to-Right)
    custom_palette = {
        'Cluster 0': '#e74c3c', # Red
        'Cluster 1': '#f39c12', # Orange
        'Cluster 2': '#3498db', # Blue
        'Cluster 3': '#2ecc71'  # Green
    }
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette=custom_palette, 
                    s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # plt.title("Curriculum Fingerprints (Skill Profile PCA)", fontsize=16)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_skill_profiles.png"), dpi=300)
    plt.close()
    print(f"Saved skill profiles plot to {OUTPUT_DIR}/cluster_skill_profiles.png")

def cluster_skill_profiles_3d():
    """Clusters students based on their final mastery across all skills in 3D."""
    from mpl_toolkits.mplot3d import Axes3D
    print("Loading skill profiles from roster for 3D plot...")
    df_roster = pd.read_csv(ROSTER_PATH)
    last_states = df_roster.groupby('student_id').tail(1)
    skill_cols = [c for c in last_states.columns if c.startswith('S')]
    profiles = last_states[skill_cols].values
    
    # 3D PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(profiles)
    
    # Clustering (same logic as 2D)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(profiles)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use distinct colors from tab10
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for c in range(4):
        mask = (clusters == c)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                   label=f'Cluster {c}', alpha=0.7, s=60, edgecolors='black', linewidth=0.3)
    
    # ax.set_title("Curriculum Fingerprints (3D Latent Manifold)", fontsize=16)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} var)")
    
    # Set background color to white for academic style
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    
    ax.legend(title="Student Groups", loc='best')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "cluster_skill_profiles_3d.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved 3D skill profiles plot to {output_path}")

if __name__ == "__main__":
    cluster_placement_vs_pacing()
    cluster_skill_profiles()
    cluster_skill_profiles_3d()
