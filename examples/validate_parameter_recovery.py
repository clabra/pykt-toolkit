
import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pickle

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def compute_metrics(y_true, y_pred):
    """Compute standard statistical metrics."""
    # Handle NaNs if any
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        "pearson_r": float(r),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "count": int(len(y_true))
    }

def plot_recovery(y_true, y_pred, title, output_path, color='blue'):
    """Generate a publication-quality scatter plot with regression line."""
    plt.figure(figsize=(8, 8))
    
    # Subsample for plot clarity if needed
    if len(y_true) > 5000:
        idx = np.random.choice(len(y_true), 5000, replace=False)
        y_t_plot, y_p_plot = y_true[idx], y_pred[idx]
    else:
        y_t_plot, y_p_plot = y_true, y_pred

    # Scatter with transparency
    sns.regplot(x=y_t_plot, y=y_p_plot, 
                scatter_kws={'alpha':0.2, 's':10, 'color': color},
                line_kws={'color': 'red', 'label': 'Linear Fit'})
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3", label='Ideal (y=x)')
    
    metrics = compute_metrics(y_true, y_pred)
    stats_text = (f"$r = {metrics['pearson_r']:.3f}$\n"
                  f"$R^2 = {metrics['r2']:.3f}$\n"
                  f"MAE = {metrics['mae']:.3f}")
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("Oracle BKT Calculation", fontsize=12)
    plt.ylabel("GTransformer Estimation", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="validation_results/parameter_recovery")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Config & Model
    config_path = os.path.join(args.exp_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config.get('params', config.get('model_config', {}))
    # Standard GTransformer defaults for init compatibility
    defaults = {'kq_same': 1, 'separate_qa': 0, 'l2_rasch': 0.0, 'pretrain_dim': 768, 'ablation': 'all', 'n_uid': 0}
    for k, v in defaults.items():
        if k not in model_config: model_config[k] = v
        
    dataset_name = config.get('dataset_name', model_config.get('dataset_name', 'assist2009'))
    fold = model_config.get('fold', 0)
    
    # Discovery
    checkpoint_path = None
    for root, _, files in os.walk(args.exp_dir):
        for f in files:
            if f.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, f)
                break
        if checkpoint_path: break

    # Data config
    data_config_path = os.path.join(PROJECT_ROOT, "configs/data_config.json")
    with open(data_config_path, 'r') as f:
        dc = json.load(f)
    dpath = os.path.join(PROJECT_ROOT, dc[dataset_name]['dpath'].replace("../", ""))
    
    # 2. Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model('gtransformer', model_config, dc[dataset_name], model_config.get('emb_type', 'qid'))
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    
    with open(os.path.join(dpath, "bkt_skill_params.pkl"), "rb") as f:
        bkt_params = pickle.load(f)
    model.load_theory_params(bkt_params)
    model.to(device)
    model.eval()
    
    # 3. Data
    test_file = os.path.join(dpath, "train_valid_sequences.csv")
    target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
    dataset = GTransformerDataset(test_file, dc[dataset_name]["input_type"], {fold}, target_path=target_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 4. Extract
    print(f"Running validation on {dataset_name} (Fold {fold})...")
    results = {
        "l0_true": [], "l0_grounded": [], "l0_probe": [],
        "t_true": [], "t_grounded": [], "t_probe": [],
        "skill": []
    }
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            
            # Forward
            outputs, _, _ = model(c, r, pid_data=q, qtest=True)
            
            # Mask
            mask = sm.bool()
            
            # Grounded (From Semantic Axis)
            results["l0_grounded"].append(outputs["p_l0"][mask].cpu().numpy())
            results["t_grounded"].append(outputs["p_t"][mask].cpu().numpy())
            
            # Probe (From Active Grounding linear head)
            results["l0_probe"].append(outputs["p_l0_probe"][mask].cpu().numpy())
            results["t_probe"].append(outputs["p_t_probe"][mask].cpu().numpy())
            
            # Oracle - Ensure on same device as mask
            results["l0_true"].append(data["target_l0"].to(device)[mask].cpu().numpy())
            results["t_true"].append(data["target_t"].to(device)[mask].cpu().numpy())
            results["skill"].append(c[mask].cpu().numpy())
            
            if i > 100: break # Sufficient for validation statistics

    # Flatten
    for k in results: results[k] = np.concatenate(results[k])
    
    # 5. Global Metrics
    summary = {
        "l0_grounded": compute_metrics(results["l0_true"], results["l0_grounded"]),
        "l0_probe": compute_metrics(results["l0_true"], results["l0_probe"]),
        "t_grounded": compute_metrics(results["t_true"], results["t_grounded"]),
        "t_probe": compute_metrics(results["t_true"], results["t_probe"])
    }
    
    # 6. Per-Skill Analysis
    df = pd.DataFrame({
        "skill": results["skill"],
        "l0_true": results["l0_true"],
        "l0_grounded": results["l0_grounded"]
    })
    
    skill_metrics = []
    for skill_id, group in df.groupby("skill"):
        if len(group) > 50: # Only skills with enough samples
            m = compute_metrics(group["l0_true"].values, group["l0_grounded"].values)
            skill_metrics.append({"skill_id": int(skill_id), **m})
    
    skill_df = pd.DataFrame(skill_metrics).sort_values("r2", ascending=False)
    skill_df.to_csv(os.path.join(args.output_dir, "per_skill_metrics.csv"), index=False)
    
    # 7. Plots
    plot_recovery(results["l0_true"], results["l0_grounded"], "Initial Mastery Recovery ($P(L_0)$ - Grounded)", 
                  os.path.join(args.output_dir, "recovery_l0_grounded.png"), color='teal')
    plot_recovery(results["t_true"], results["t_grounded"], "Learning Rate Recovery ($P(T)$ - Grounded)", 
                  os.path.join(args.output_dir, "recovery_t_grounded.png"), color='darkorange')
    
    # Add Probe Plots for structural validity proof
    plot_recovery(results["l0_true"], results["l0_probe"], "Initial Mastery Recovery ($P(L_0)$ - Probe)", 
                  os.path.join(args.output_dir, "recovery_l0_probe.png"), color='blue')
    plot_recovery(results["t_true"], results["t_probe"], "Learning Rate Recovery ($P(T)$ - Probe)", 
                  os.path.join(args.output_dir, "recovery_t_probe.png"), color='purple')

    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nStep 1 Complete. Results saved to {args.output_dir}")
    print(f"L0 Grounded R2: {summary['l0_grounded']['r2']:.4f}")
    print(f"T Grounded R2: {summary['t_grounded']['r2']:.4f}")

if __name__ == "__main__":
    main()
