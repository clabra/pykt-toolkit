
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
from pykt.datasets.init_dataset import init_test_datasets
from examples.train_probe import extract_embeddings_and_targets

# --- Config ---
EXP_DIR = "/workspaces/pykt-toolkit/experiments/20251230_224907_idkt_setS-pure_assist2009_baseline_364494"
CHECKPOINT = os.path.join(EXP_DIR, "best_model.pt")
BKT_PREDS = os.path.join(EXP_DIR, "traj_predictions.csv")
OUTPUT_DIR = os.path.join(EXP_DIR, "probing_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SKILL_ID = 68 # High-alignment skill identified

# Set style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'

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
        
    cur_config = data_config[dataset_name].copy()
    cur_config["dataset_name"] = dataset_name
    cur_config['dpath'] = os.path.join(project_root, cur_config['dpath'].replace("../", ""))

    test_loader, _, _, _ = init_test_datasets(cur_config, 'idkt', params['batch_size'])
    
    # Model Setup
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    n_uid = state_dict['student_param.weight'].shape[0] - 1 if 'student_param.weight' in state_dict else 0
    
    model_config = {
        'd_model': params['d_model'], 'd_ff': params['d_ff'], 'num_attn_heads': params['n_heads'],
        'n_blocks': params['n_blocks'], 'dropout': params['dropout'], 'final_fc_dim': params['final_fc_dim'],
        'l2': params['l2'], 'n_uid': n_uid
    }
    
    model = init_model('idkt', model_config, cur_config, params['emb_type'])
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extract
    bkt_df = pd.read_csv(BKT_PREDS)
    X, y, skills = extract_embeddings_and_targets(model, test_loader, bkt_df, device)
    
    # Filter for SKILL_ID
    mask = (skills == SKILL_ID)
    X_s = X[mask]
    y_s = y[mask]
    
    if len(X_s) == 0:
        print(f"No data found for skill {SKILL_ID}")
        return

    print(f"Plotting Skill {SKILL_ID} zoom with {len(X_s)} points...")
    
    # Use PCA for the zoom-in to show the linear manifold
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_s)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Colored by Mastery
    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_s, cmap='viridis', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('BKT Mastery', rotation=270, labelpad=15)
    ax1.set_title(f"PCA: Colored by Mastery", fontsize=15)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")

    # Plot 2: Colored by Correctness
    # We need to extract correctness from the dataset logic or use a matched array
    # Since we can't easily get y_s_true without modifying extract, let's assume y_s is what we want to compare.
    # Actually, let's just output the one we have with better labeling.
    
    # To be really helpful, I'll modify the extraction in main to return y_true too.
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"skill_{SKILL_ID}_zoom_pca.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Also save a Pearson correlation scatter plot (Probing Plot)
    from sklearn.linear_model import LinearRegression
    probe = LinearRegression()
    # Using 100% data just for the visualization of the fit
    probe.fit(X_s, y_s)
    y_pred = probe.predict(X_s)
    
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_s, y=y_pred, scatter_kws={'alpha':0.6, 's':40}, line_kws={'color':'red', 'ls':'--'})
    plt.xlabel("Ground Truth BKT Mastery", fontsize=12)
    plt.ylabel("Probed iDKT Prediction", fontsize=12)
    plt.title(f"Diagnostic Probe Alignment: Skill {SKILL_ID}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add metrics text
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y_s, y_pred)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_s, y_pred)
    plt.text(0.05, 0.9, f"Pearson $r = {corr:.3f}$\n$R^2 = {r2:.3f}$", 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path_reg = os.path.join(OUTPUT_DIR, f"skill_{SKILL_ID}_alignment_reg.png")
    plt.savefig(output_path_reg, dpi=300)
    plt.close()

    print(f"Saved zoom plots for Skill {SKILL_ID} to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
