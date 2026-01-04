
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

SKILL_ID = 68 

# Set style
sns.set_theme(style="white", context="paper")
plt.rcParams['font.family'] = 'serif'

def main():
    # Load config
    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    params = config['defaults']
    dataset_name = config['input']['dataset']
    
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
    X, y, skills, r_true = extract_embeddings_and_targets(model, test_loader, bkt_df, device)
    
    # Filter for SKILL_ID
    mask = (skills == SKILL_ID)
    X_s = X[mask]
    y_s = y[mask]
    r_s = r_true[mask]
    
    print(f"Plotting Skill {SKILL_ID} Bifurcation check with {len(X_s)} points...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_s)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Colored by Mastery
    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_s, cmap='viridis', s=100, alpha=0.9, edgecolors='black', linewidth=0.5)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('BKT Mastery', rotation=270, labelpad=15)
    ax1.set_title(f"A. Colored by Mastery ($r=0.993$)", fontsize=16)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax1.grid(True, alpha=0.2)

    # Plot 2: Colored by Correctness
    # 0 = Incorrect (Red), 1 = Correct (Blue)
    colors = ['#e74c3c' if r == 0 else '#3498db' for r in r_s]
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=100, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    # Add dummy legend for correctness
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Incorrect (r=0)', markerfacecolor='#e74c3c', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Correct (r=1)', markerfacecolor='#3498db', markersize=10)]
    ax2.legend(handles=legend_elements, loc='best', title="Interaction Result")
    
    ax2.set_title(f"B. Colored by Success/Failure", fontsize=16)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax2.grid(True, alpha=0.2)
    
    plt.suptitle(f"Bifurcation of Latent Manifold for Skill {SKILL_ID}", fontsize=20, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"skill_{SKILL_ID}_bifurcation_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved bifurcation analysis to {output_path}")

if __name__ == "__main__":
    main()
