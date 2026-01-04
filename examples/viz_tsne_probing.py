
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.models import init_model
from pykt.datasets import init_dataset4train
from examples.train_probe import extract_embeddings_and_targets

# --- Config ---
EXP_DIR = "/workspaces/pykt-toolkit/experiments/20251230_224907_idkt_setS-pure_assist2009_baseline_364494"
CHECKPOINT = os.path.join(EXP_DIR, "best_model.pt")
BKT_PREDS = os.path.join(EXP_DIR, "traj_predictions.csv")
OUTPUT_DIR = os.path.join(EXP_DIR, "probing_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'

def plot_tsne_manifold(X, y, skills, output_path, title="iDKT Latent Space t-SNE"):
    """Generates a t-SNE plot colored by Mastery."""
    print(f"Running t-SNE on {len(X)} samples...")
    
    # Pre-reduce with PCA to speed up and denoise
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_reduced)
    
    # 1. Plot colored by Mastery
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(sc, label='BKT Mastery')
    plt.title(title + " (Colored by Mastery)", fontsize=15)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # 2. Plot colored by Skill (top 10 skills only for clarity)
    top_skills = pd.Series(skills).value_counts().nlargest(10).index
    mask = np.isin(skills, top_skills)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[mask, 0], y=X_tsne[mask, 1], hue=skills[mask], 
                    palette="tab10", alpha=0.7, s=30, legend='full')
    plt.title(title + " (Colored by Skill - Top 10)", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Skill ID")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    skill_path = output_path.replace(".png", "_by_skill.png")
    plt.savefig(skill_path, dpi=300)
    plt.close()
    
    print(f"Saved t-SNE plots to {OUTPUT_DIR}")

def main():
    # Load config from experiment
    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config['defaults']
    dataset_name = config['input']['dataset']
    
    # Data Setup
    project_root = "/workspaces/pykt-toolkit"
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
        
    for d in data_config:
        if 'dpath' in data_config[d]:
            dpath = data_config[d]['dpath'].replace("../", "")
            data_config[d]['dpath'] = os.path.join(project_root, dpath)

    from pykt.datasets.init_dataset import init_test_datasets
    cur_config = data_config[dataset_name].copy()
    cur_config["dataset_name"] = dataset_name
    test_loader, _, _, _ = init_test_datasets(cur_config, 'idkt', params['batch_size'])
    loader = test_loader
    
    # Model Setup
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    if 'student_param.weight' in state_dict:
        n_uid = state_dict['student_param.weight'].shape[0] - 1
    else:
        n_uid = 0
    
    model_config = {
        'd_model': params['d_model'],
        'd_ff': params['d_ff'],
        'num_attn_heads': params['n_heads'],
        'n_blocks': params['n_blocks'],
        'dropout': params['dropout'],
        'emb_type': params['emb_type'],
        'final_fc_dim': params['final_fc_dim'],
        'l2': params['l2'],
        'lambda_student': params['lambda_student'],
        'lambda_gap': params['lambda_gap'],
        'n_uid': n_uid
    }
    
    model = init_model('idkt', model_config, data_config[dataset_name], params['emb_type'])
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extract
    bkt_df = pd.read_csv(BKT_PREDS)
    X, y, skills = extract_embeddings_and_targets(model, loader, bkt_df, device)
    
    # Downsample for t-SNE visualization (max 5000 points)
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X, y, skills = X[idx], y[idx], skills[idx]
        
    plot_tsne_manifold(X, y, skills, os.path.join(OUTPUT_DIR, "probing_tsne.png"))

if __name__ == "__main__":
    main()
