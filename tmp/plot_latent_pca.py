
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def plot_manifold_pca(X, y, output_path, title="Latent Space (PCA)"):
    """Projects embeddings to 2D and colors by BKT Difficulty (L0)."""
    print(f"Computing PCA on {len(X)} samples...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    # Filter y to stay in [0, 1] for coloring
    y_plot = np.clip(y, 0, 1)
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_plot, cmap='magma', alpha=0.6, s=15)
    cbar = plt.colorbar(sc)
    cbar.set_label('BKT Difficulty (L0)', fontsize=12)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)", fontsize=14)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved PCA manifold plot to {output_path}")

def plot_manifold_tsne(X, y, skills, output_path, skill_names=None, title="Latent Space (t-SNE)"):
    """Projects embeddings to 2D using t-SNE for richer semantic mapping."""
    print(f"Computing t-SNE on {len(X)} samples...")
    
    # Pre-reduce with PCA for t-SNE stability
    X_reduced = PCA(n_components=50).fit_transform(X) if X.shape[1] > 50 else X
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_reduced)
    
    # 1. Plot colored by Difficulty
    plt.figure(figsize=(10, 8))
    y_plot = np.clip(y, 0, 1)
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_plot, cmap='magma', alpha=0.6, s=15)
    cbar = plt.colorbar(sc)
    cbar.set_label('BKT Difficulty (L0)', fontsize=12)
    
    plt.title("Latent Space (t-SNE)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("t-SNE dimension 1", fontsize=14)
    plt.ylabel("t-SNE dimension 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # 2. Plot colored by Top 10 Skills
    top_skills_idx = pd.Series(skills).value_counts().nlargest(10).index
    mask_top = np.isin(skills, top_skills_idx)
    
    plt.figure(figsize=(13, 8)) # Wider for legend
    # Grey out background points
    plt.scatter(X_tsne[~mask_top, 0], X_tsne[~mask_top, 1], c='grey', alpha=0.1, s=5, label='Other Skills')
    
    # Map index to Name if available
    # legend should show "ID: Name"
    # skill_names received here is already indexed by index (not ID) from main
    labels = np.array([skill_names.get(int(idx), str(idx)) for idx in skills[mask_top]])
    
    # Plot top skills
    sns.scatterplot(x=X_tsne[mask_top, 0], y=X_tsne[mask_top, 1], hue=labels, 
                    palette="tab10", alpha=0.8, s=25, legend='full')
    
    plt.title("Latent Space by Skill", fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., 
               title="Skill (ID: Name)", title_fontsize=12, fontsize=11)
    plt.xlabel("t-SNE dimension 1", fontsize=14)
    plt.ylabel("t-SNE dimension 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    skill_path = output_path.replace(".png", "_by_skill.png")
    plt.savefig(skill_path, dpi=300)
    plt.close()
    
    print(f"Saved t-SNE manifold plots to {output_path} and {skill_path}")

def plot_parity(y_true, y_pred, output_path, title="Recovery Diagonal"):
    """Generates X=BKT Estimation, Y=Probe Parity Plot with x-binned aggregation."""
    from scipy.stats import pearsonr
    import pandas as pd
    
    print(f"Generating parity plot...")
    
    # Calculate correlation
    r, _ = pearsonr(y_true, y_pred)
    
    # Binned aggregation
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['x_bin'] = pd.cut(df['y_true'], bins=40)
    
    bin_stats = df.groupby('x_bin').agg({'y_pred': 'mean', 'y_true': 'count'}).reset_index()
    # Use midpoint of bin for x position
    bin_stats['x_pos'] = bin_stats['x_bin'].apply(lambda x: x.mid).astype(float)
    
    # Filter out empty bins
    bin_stats = bin_stats[bin_stats['y_true'] > 0]
    
    # Normalize sizes for plot
    sizes = bin_stats['y_true'].values
    sizes_normalized = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6) * 450
    
    plt.figure(figsize=(10, 10))
    plt.scatter(bin_stats['x_pos'], bin_stats['y_pred'], s=sizes_normalized, 
                alpha=0.6, c='blue', edgecolors='black', linewidth=0.8)
    
    # Diagonal
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("BKT Estimation", fontsize=14)
    plt.ylabel("Diagnostic Probe Prediction (Mean)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Stats box
    textstr = f"$R^2 = {r**2:.3f}$"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, textstr, fontsize=14, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved parity plot to {output_path} (R² = {r**2:.3f}, {len(bin_stats)} x-bins)")

def extract_latent_and_l0(model, loader, device):
    """
    Extracts z_context, target_l0, probe_l0, and skills from the model and loader.
    """
    model.eval()
    all_z = []
    all_l0 = []
    all_l0_pred = []
    all_skills = []
    
    steps = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            
            # For GTransformer, forward with qtest=True returns z_context
            outputs, _, z_context = model(c, r, pid_data=q, qtest=True)
            
            # target_l0 is provided by GTransformerDataset
            l0_targets = data["target_l0"].to(device) # [BS, seqlen]
            
            # Extract probe prediction from outputs
            l0_preds = outputs["p_l0_probe"] # [BS, seqlen]
            
            # Masked extraction
            mask = sm.bool() # [BS, seqlen]
            
            z_flat = z_context[mask] # [N, z_dim]
            l0_flat = l0_targets[mask] # [N]
            l0_pred_flat = l0_preds[mask] # [N]
            
            # Concepts (c) are already [BS, seqlen] matching mask
            skills_flat = c[mask] # [N]
            
            all_z.append(z_flat.cpu().numpy())
            all_l0.append(l0_flat.cpu().numpy())
            all_l0_pred.append(l0_pred_flat.cpu().numpy())
            all_skills.append(skills_flat.cpu().numpy())
            
            steps += 1
            if steps > 150: # More samples for better skill mapping
                break
                
    X = np.concatenate(all_z, axis=0)
    y = np.concatenate(all_l0, axis=0)
    y_pred = np.concatenate(all_l0_pred, axis=0)
    skills = np.concatenate(all_skills, axis=0)
    
    return X, y, y_pred, skills

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Experiment fold directory")
    parser.add_argument("--output_dir", type=str, help="Where to save plots")
    args = parser.parse_args()

    # Paths 
    EXP_DIR = args.exp_dir if args.exp_dir else "/workspaces/pykt-toolkit/experiments/20260116_101107_benchpaper_oraclecorrect_baseline_334772/gtransformer/assist2009/fold_0_536546"
    OUTPUT_DIR = args.output_dir if args.output_dir else EXP_DIR
    
    # Checkpoint Discovery Logic
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
    
    # Handle different config formats - params is the most complete for flat lookups
    model_config = config.get('params', config.get('train_config', config.get('model_config', config.get('defaults', config))))

    # Dataset and Fold configuration
    dataset_name = model_config.get("dataset", config.get("params", {}).get("dataset_name", config.get("input", {}).get("dataset", "assist2009")))
    fold = model_config.get("fold", config.get("params", {}).get("fold", config.get("input", {}).get("fold", 0)))
    
    # Load data config
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    # Absolute paths for data
    dpath = data_config[dataset_name]['dpath'].replace("../", "")
    dpath = os.path.join(project_root, dpath)
    data_config[dataset_name]['dpath'] = dpath
            
    # Init Dataset
    target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
    train_file = os.path.join(dpath, "train_valid_sequences.csv")
    test_dataset = GTransformerDataset(train_file, data_config[dataset_name]["input_type"], {fold}, target_path=target_path)
    loader = DataLoader(test_dataset, batch_size=model_config.get('batch_size', 64), shuffle=False, num_workers=4)
    
    # Model Setup
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = checkpoint
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Skill params for initialization
    bkt_params_path = os.path.join(dpath, "bkt_skill_params.pkl")
    with open(bkt_params_path, "rb") as f:
        bkt_skill_params = pickle.load(f)
    
    model = init_model('gtransformer', model_config, data_config[dataset_name], model_config['emb_type'])
    model.load_theory_params(bkt_skill_params)
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load skill mapping for legends
    keyid2idx_path = os.path.join(dpath, "keyid2idx.json")
    skill_names_map = {}
    if os.path.exists(keyid2idx_path):
        with open(keyid2idx_path, 'r') as f:
            mapping_data = json.load(f)
            # keyid2idx has {"concepts": {"orig_id": index}}
            # We want index -> orig_id first
            idx_to_orig = {int(v): str(k) for k, v in mapping_data.get("concepts", {}).items()}
            
            # Now try to find names from original CSV (dataset specific)
            if "assist2009" in dataset_name:
                csv_path = os.path.join(dpath, "skill_builder_data_corrected_collapsed.csv")
                if os.path.exists(csv_path):
                    import csv
                    # orig_id -> name
                    id_to_name = {}
                    with open(csv_path, 'r', encoding='ISO-8859-1') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            sid = row.get("skill_id", "").replace('"', '')
                            name = row.get("skill_name", "")
                            if sid and name:
                                id_to_name[sid] = name
                    
                    # Known fixes for assist2009
                    id_to_name["2"] = "Circle Graph"
                    id_to_name["70"] = "Percent Of (a number)"
                    
                    # combine to get index -> name (Format: "OriginalID: Name")
                    for idx, sid in idx_to_orig.items():
                        name = id_to_name.get(sid, "")
                        if name:
                            skill_names_map[idx] = f"{sid}: {name}"
                        else:
                            skill_names_map[idx] = sid
            else:
                # Default to orig_id as name for other datasets
                skill_names_map = idx_to_orig

    X, y, y_pred, s = extract_latent_and_l0(model, loader, device)
    
    # Downsample if too many
    if len(X) > 8000:
        idx = np.random.choice(len(X), 8000, replace=False)
        X, y, y_pred, s = X[idx], y[idx], y_pred[idx], s[idx]
        
    output_pca = os.path.join(OUTPUT_DIR, "latent_pca_map.png")
    plot_manifold_pca(X, y, output_pca)
    
    output_tsne = os.path.join(OUTPUT_DIR, "latent_tsne_map.png")
    plot_manifold_tsne(X, y, s, output_tsne, skill_names=skill_names_map)
    
    output_parity = os.path.join(OUTPUT_DIR, "probe_parity_plot.png")
    plot_parity(y, y_pred, output_parity)

if __name__ == "__main__":
    main()
