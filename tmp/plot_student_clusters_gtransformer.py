import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def extract_student_embeddings(model, loader, device, n_uid):
    """
    Extracts student-specific parameters from learned embeddings using PCA.
    Only works when n_uid > 0 (personalization enabled).
    """
    from sklearn.decomposition import PCA
    
    model.eval()
    
    # Get student embeddings from model
    student_param = model.student_param.weight.detach().cpu().numpy()  # [n_uid, d_model]
    student_gap_param = model.student_gap_param.weight.detach().cpu().numpy()  # [n_uid, d_model]
    
    # Concatenate both embeddings for joint PCA
    combined_embeddings = np.concatenate([student_param, student_gap_param], axis=1)  # [n_uid, 2*d_model]
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(combined_embeddings)  # [n_uid, 2]
    
    alpha = embeddings_2d[:, 0]  # First principal component (placement proxy)
    beta = embeddings_2d[:, 1]   # Second principal component (pacing proxy)
    
    # Normalize to [0, 1] range for interpretability
    alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
    beta = (beta - beta.min()) / (beta.max() - beta.min() + 1e-8)
    
    uids = np.arange(n_uid)
    
    print(f"[PCA] Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}, Total={pca.explained_variance_ratio_.sum():.3f}")
    
    return alpha, beta, uids

def extract_student_metrics_from_probes(model, loader, device):
    """
    Extracts mean predicted L0 and T per student using probes.
    Used when personalization is disabled (n_uid = 0).
    """
    model.eval()
    student_metrics = {}
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            uids = data["uids"].numpy()
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            
            # Forward to get probe predictions
            outputs, _, _ = model(c, r, pid_data=q, qtest=True)
            
            l0_preds = outputs["p_l0_probe"] # [BS, seqlen]
            t_preds = outputs["p_t_probe"]   # [BS, seqlen]
            
            mask = sm.bool()
            
            for b in range(uids.shape[0]):
                uid = uids[b]
                m = mask[b]
                if m.sum() == 0: continue
                
                l0_vals = l0_preds[b][m].cpu().numpy()
                t_vals = t_preds[b][m].cpu().numpy()
                
                if uid not in student_metrics:
                    student_metrics[uid] = {'l0': [], 't': []}
                
                student_metrics[uid]['l0'].extend(l0_vals)
                student_metrics[uid]['t'].extend(t_vals)

    # Compute means per student
    uids_list = []
    l0_means = []
    t_means = []
    
    for uid, vals in student_metrics.items():
        uids_list.append(uid)
        l0_means.append(np.mean(vals['l0']))
        t_means.append(np.mean(vals['t']))
        
    return np.array(l0_means), np.array(t_means), np.array(uids_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Experiment fold directory")
    parser.add_argument("--output_dir", type=str, help="Where to save plots")
    args = parser.parse_args()

    # Paths
    EXP_DIR = args.exp_dir if args.exp_dir else "/workspaces/pykt-toolkit/experiments/20260116_101107_benchpaper_oraclecorrect_baseline_334772/gtransformer/assist2009/fold_0_536546"
    OUTPUT_DIR = args.output_dir if args.output_dir else EXP_DIR
    
    # Checkpoint Discovery
    found_ckpt = None
    for root, dirs, files in os.walk(EXP_DIR):
        for file in files:
            if file.endswith(".ckpt"):
                found_ckpt = os.path.join(root, file)
                break
        if found_ckpt: break
        
    CHECKPOINT = found_ckpt
    CONFIG_PATH = os.path.join(EXP_DIR, "config.json")
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Robust Config Lookup
    model_config = config.get('params', config.get('train_config', config.get('model_config', config.get('defaults', config))))
    dataset_name = model_config.get("dataset", config.get("params", {}).get("dataset_name", config.get("input", {}).get("dataset", "assist2009")))
    fold = model_config.get("fold", config.get("params", {}).get("fold", config.get("input", {}).get("fold", 0)))
    batch_size = model_config.get("batch_size", 64)
    
    # Determine personalization: check flag first, then fall back to n_uid
    personalization = model_config.get("personalization", None)
    n_uid = model_config.get("n_uid", 0)
    if personalization is None:
        # Backward compatibility: infer from n_uid
        has_personalization = (n_uid > 0)
    else:
        has_personalization = personalization
    
    # Load data config
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    dpath = os.path.join(project_root, data_config[dataset_name]['dpath'].replace("../", ""))
    
    # Init Dataset (Test set)
    target_path = os.path.join(dpath, "bkt_targets_test.npz")
    test_file = os.path.join(dpath, "test_question_sequences.csv")
    test_dataset = GTransformerDataset(test_file, data_config[dataset_name]["input_type"], {-1}, target_path=target_path)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    bkt_params_path = os.path.join(dpath, "bkt_skill_params.pkl")
    with open(bkt_params_path, "rb") as f:
        bkt_skill_params = pickle.load(f)
    
    model = init_model('gtransformer', model_config, data_config[dataset_name], model_config['emb_type'])
    model.load_theory_params(bkt_skill_params)
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Use has_personalization determined from config above
    if has_personalization:
        print(f"[PERSONALIZATION DETECTED] Using learned student embeddings (n_uid={n_uid})")
        alpha, beta, uids = extract_student_embeddings(model, loader, device, n_uid)
        plot_subtitle = "Personalized (Learned Student Embeddings)"
        filename = "cluster_placement_pacing_personalized.png"
    else:
        print(f"[NO PERSONALIZATION] Using probe predictions from temporal context (n_uid=0)")
        alpha, beta, uids = extract_student_metrics_from_probes(model, loader, device)
        plot_subtitle = "Contextual (Probe Predictions)"
        filename = "cluster_placement_pacing_contextual.png"
    
    print(f"--- STUDENT METRICS DEBUG ---")
    print(f"Alpha (Placement) - Mean: {alpha.mean():.4f}, Std: {alpha.std():.4f}, Range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"Beta  (Pacing)    - Mean: {beta.mean():.4f}, Std: {beta.std():.4f}, Range: [{beta.min():.4f}, {beta.max():.4f}]")
    print(f"-----------------------------")
    X = np.stack([alpha, beta], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    original_clusters = kmeans.fit_predict(X_scaled)
    
    # Re-order based on overall proficiency
    df_temp = pd.DataFrame({'a': X_scaled[:, 0], 'b': X_scaled[:, 1], 'orig_cluster': original_clusters})
    cluster_order = df_temp.groupby('orig_cluster').apply(lambda g: g['a'].mean() + g['b'].mean()).sort_values().index.tolist()
    remap = {orig: new for new, orig in enumerate(cluster_order)}
    sorted_clusters = np.array([remap[c] for c in original_clusters])
    
    df = pd.DataFrame({
        'Placement (Mean Probe L0)': alpha,
        'Pacing (Mean Probe T)': beta,
        'Cluster': [f'Cluster {c}' for c in sorted_clusters]
    })
    
    print("--- CLUSTER MEANS ---")
    print(df.groupby('Cluster').mean())
    print("\n--- CLUSTER COUNTS ---")
    print(df['Cluster'].value_counts())
    print("----------------------")
    
    # Plot
    sns.set_theme(style="whitegrid")
    
    # High-contrast color palette
    if has_personalization:
        custom_palette = {
            'Cluster 0': '#d62728',  # Bright red
            'Cluster 1': '#ff7f0e',  # Bright orange
            'Cluster 2': '#2ca02c',  # Bright green
            'Cluster 3': '#1f77b4'   # Bright blue
        }
        cluster_labels = {
            'Cluster 0': 'Foundational',
            'Cluster 1': 'Rapid Progression',
            'Cluster 2': 'Steady Advancement',
            'Cluster 3': 'High Performance'
        }
    else:
        custom_palette = {
            'Cluster 0': '#e74c3c',
            'Cluster 1': '#f39c12',
            'Cluster 2': '#3498db',
            'Cluster 3': '#2ecc71'
        }
        cluster_labels = None
    
    # Calculate cluster percentages
    cluster_counts = df['Cluster'].value_counts()
    total_students = len(df)
    cluster_pcts = {cluster: (count / total_students * 100) for cluster, count in cluster_counts.items()}
    
    # Create labels with percentages and descriptions
    if cluster_labels:
        legend_labels = {
            cluster: f"{cluster} ({cluster_pcts[cluster]:.0f}%): {cluster_labels[cluster]}"
            for cluster in df['Cluster'].unique()
        }
        df['Cluster_Label'] = df['Cluster'].map(legend_labels)
    else:
        legend_labels = {
            cluster: f"{cluster} ({cluster_pcts[cluster]:.0f}%)"
            for cluster in df['Cluster'].unique()
        }
        df['Cluster_Label'] = df['Cluster'].map(legend_labels)
    
    plt.figure(figsize=(12, 9))
    
    # Adjust point size and transparency based on personalization
    if has_personalization:
        point_size = 60  # Slightly larger for better visibility
        point_alpha = 0.6  # Slightly more opaque
    else:
        point_size = 100
        point_alpha = 0.7
    
    # 1. Plot Density Contours (KDE) to show the underlying mass
    sns.kdeplot(data=df, x='Placement (Mean Probe L0)', y='Pacing (Mean Probe T)',
                hue='Cluster_Label', palette={legend_labels[k]: v for k, v in custom_palette.items()},
                alpha=0.3, levels=5, thresh=0.1, fill=True, legend=False)

    # 2. Plot the individual student points
    sns.scatterplot(data=df.sort_values('Cluster'), x='Placement (Mean Probe L0)', y='Pacing (Mean Probe T)', 
                    hue='Cluster_Label', palette={legend_labels[k]: v for k, v in custom_palette.items()},
                    s=point_size, alpha=point_alpha, edgecolors='black', linewidth=0.5)

    # 3. Apply Power Transform (Sqrt) to y-axis to expand the dense [0.0 - 0.4] region
    # This addresses the sparse space issue while keeping 0 visible.
    def forward(x): return np.sqrt(np.maximum(x, 0))
    def inverse(x): return x**2
    ax = plt.gca()
    ax.set_yscale('function', functions=(forward, inverse))

    # Set major ticks for the power scale
    ax.yaxis.set_major_locator(plt.FixedLocator([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]))
    plt.xlim(-0.02, 1.05)
    plt.ylim(0.0, 1.05)
    
    # Remove log scale for the y-axis (Pacing/T is in [0,1] range)
    # if not has_personalization:
    #     plt.yscale('log')

    plt.title("Student Situational Clustering (Sqrt Scale)", fontsize=16, fontweight='bold', pad=20)
    
    if has_personalization:
        plt.xlabel("Student Embedding PC1 (Placement-related)", fontsize=14)
        plt.ylabel("Student Embedding PC2 (Pacing-related)", fontsize=14)
    else:
        plt.xlabel("Student Placement (Estimated Mean Difficulty Encountered)", fontsize=14)
        plt.ylabel("Student Pacing (Estimated Mean Learning Rate)", fontsize=14)
    
    # Remove explicit xticks/yticks to allow the new scale to handle formatting
    # plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
    plt.xlim(-0.02, 1.05)
    plt.ylim(-0.02, 1.05)

    # Disable scientific notation for axes
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')
    
    # Improve legend
    plt.legend(title='Learning Patterns', title_fontsize=12, fontsize=11, 
               loc='upper left', framealpha=0.95, edgecolor='black')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster plot to {output_path}")

if __name__ == "__main__":
    main()
