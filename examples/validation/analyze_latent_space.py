
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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def extract_latent(model, loader, device, n_samples=5000):
    model.eval()
    all_z = []
    all_l0 = []
    all_skills = []
    
    count = 0
    with torch.no_grad():
        for data in loader:
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            
            # Forward returns z_context at index 2
            _, _, z_context = model(c, r, pid_data=q, qtest=True)
            mask = sm.bool()
            
            all_z.append(z_context[mask].cpu().numpy())
            all_l0.append(data["target_l0"].to(device)[mask].cpu().numpy())
            all_skills.append(c[mask].cpu().numpy())
            
            count += mask.sum().item()
            if count >= n_samples: break
                
    X = np.concatenate(all_z, axis=0)[:n_samples]
    y = np.concatenate(all_l0, axis=0)[:n_samples]
    skills = np.concatenate(all_skills, axis=0)[:n_samples]
    
    return X, y, skills

def get_clustering_metrics(X, labels):
    # Only use skills with enough samples
    counts = pd.Series(labels).value_counts()
    valid_skills = counts[counts >= 10].index
    mask = np.isin(labels, valid_skills)
    
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        return {"silhouette": 0, "db_index": 0, "ch_score": 0}

    X_sub = X[mask]
    L_sub = labels[mask]
    
    return {
        "silhouette": float(silhouette_score(X_sub, L_sub)),
        "db_index": float(davies_bouldin_score(X_sub, L_sub)),
        "ch_score": float(calinski_harabasz_score(X_sub, L_sub))
    }

def plot_scree(pca_g, pca_b, output_path):
    plt.figure(figsize=(10, 6))
    
    evr_g = np.cumsum(pca_g.explained_variance_ratio_)
    evr_b = np.cumsum(pca_b.explained_variance_ratio_)
    
    plt.plot(range(1, len(evr_g)+1), evr_g, marker='o', label='Proposed Architecture', color='royalblue', linewidth=2)
    plt.plot(range(1, len(evr_b)+1), evr_b, marker='s', label='Baseline', color='grey', linestyle='--')
    
    # 90% Variance threshold
    plt.axhline(y=0.9, color='r', linestyle=':', label='90% Variance Threshold')
    
    rank_g = np.argmax(evr_g >= 0.9) + 1
    rank_b = np.argmax(evr_b >= 0.9) + 1
    
    plt.annotate(f'Eff. Rank: {rank_g}', xy=(rank_g, 0.9), xytext=(rank_g+2, 0.8),
                 arrowprops=dict(facecolor='royalblue', shrink=0.05))
    
    plt.title("Latent Space Parsimony: Elbow Plot", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.ylabel("Cumulative Explained Variance", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return rank_g, rank_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded_exp", type=str, required=True)
    parser.add_argument("--baseline_exp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="examples/validation/results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config_path = os.path.join(PROJECT_ROOT, "configs/data_config.json")
    with open(data_config_path, 'r') as f:
        dc = json.load(f)

    def load_model_from_dir(exp_dir):
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, 'r') as f: config = json.load(f)
        
        mc = config.get('params', config.get('model_config', {}))
        # Robust arch migration (copied from parameter_recovery logic)
        for k in ['d_model', 'n_blocks', 'dropout', 'd_ff', 'final_fc_dim', 'num_attn_heads', 'n_heads', 'ablation']:
            if k not in mc:
                for src in ['params', 'train_config', 'defaults']:
                    if src in config and k in config[src]:
                        mc[k] = config[src][k]; break
        if 'n_heads' in mc: mc['num_attn_heads'] = mc['n_heads']
        
        # Add required defaults for older configs
        defaults = {'kq_same': 1, 'separate_qa': 0, 'l2_rasch': 0.0, 'pretrain_dim': 768, 'ablation': 'none'}
        for k, v in defaults.items():
            if k not in mc: mc[k] = v
        
        dataset_name = config.get('dataset_name', mc.get('dataset_name', 'assist2009'))
        fold = mc.get('fold', 0)
        
        checkpoint_path = None
        for root, _, files in os.walk(exp_dir):
            for f in files:
                if f.endswith(".ckpt"):
                    checkpoint_path = os.path.join(root, f); break
            if checkpoint_path: break
            
        model = init_model('gtransformer', mc, dc[dataset_name], mc.get('emb_type', 'qid'))
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        model.to(device)
        
        dpath = os.path.join(PROJECT_ROOT, dc[dataset_name]['dpath'].replace("../", ""))
        test_file = os.path.join(dpath, "train_valid_sequences.csv")
        target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
        dataset = GTransformerDataset(test_file, dc[dataset_name]["input_type"], {fold}, target_path=target_path)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        return model, loader, mc

    print("Processing Grounded model...")
    model_g, loader_g, mc_g = load_model_from_dir(args.grounded_exp)
    X_g, y_g, s_g = extract_latent(model_g, loader_g, device)
    
    print("Processing Baseline model...")
    model_b, loader_b, mc_b = load_model_from_dir(args.baseline_exp)
    X_b, y_b, s_b = extract_latent(model_b, loader_b, device)
    
    # 1. PCA Comparison
    pca_g = PCA().fit(X_g)
    pca_b = PCA().fit(X_b)
    rank_g, rank_b = plot_scree(pca_g, pca_b, os.path.join(args.output_dir, "elbow_plot_comparison.png"))
    
    # 2. Clustering Metrics
    metrics_g = get_clustering_metrics(X_g, s_g)
    metrics_b = get_clustering_metrics(X_b, s_b)
    
    # 3. TSNE for Grounded (Standard Validation Visualization)
    print("Computing t-SNE for Grounded visualization...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_g)
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(y_g, 0, 1), cmap='magma', alpha=0.6, s=15)
    plt.colorbar(sc).set_label('BKT Difficulty ($L_0$)', fontsize=12)
    plt.title("Latent Space Organization (Proposed)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(args.output_dir, "latent_organization_tsne.png"), dpi=300)
    plt.close()
    
    summary = {
        "grounded": {**metrics_g, "effective_rank": int(rank_g)},
        "baseline": {**metrics_b, "effective_rank": int(rank_b)}
    }
    
    with open(os.path.join(args.output_dir, "latent_clustering_metrics.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
