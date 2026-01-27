
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
from scipy.stats import pearsonr, spearmanr
import pickle

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def compute_metrics(y_true, y_pred, weights=None):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) < 2:
        return {"pearson_r": 0, "spearman_r": 0, "weighted_r": 0, "r2": 0, "mae": 0, "rmse": 0, "count": len(y_t)}
    
    # Standard Pearson correlation (sensitive to outliers)
    r, _ = pearsonr(y_t, y_p)
    
    # Spearman rank correlation (robust to outliers - uses ranks instead of values)
    rho, _ = spearmanr(y_t, y_p)
    
    # Weighted Pearson correlation (if weights provided, e.g., by sample frequency)
    if weights is not None:
        w = weights[mask]
        w = w / w.sum()  # Normalize weights
        mean_t = np.average(y_t, weights=w)
        mean_p = np.average(y_p, weights=w)
        cov = np.average((y_t - mean_t) * (y_p - mean_p), weights=w)
        std_t = np.sqrt(np.average((y_t - mean_t)**2, weights=w))
        std_p = np.sqrt(np.average((y_p - mean_p)**2, weights=w))
        weighted_r = cov / (std_t * std_p) if (std_t > 0 and std_p > 0) else 0
    else:
        weighted_r = r  # Fallback to unweighted if no weights provided
    
    r2 = r2_score(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    
    return {
        "pearson_r": float(r),
        "spearman_r": float(rho),
        "weighted_r": float(weighted_r),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "count": int(len(y_t))
    }

def plot_recovery(y_true, y_pred, title, output_path, color='royalblue', label_prefix="", show_metrics=False):
    """Generate parity plot using binned aggregates with bubble sizes for sample density."""
    
    # Create DataFrame and bin the true values
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['x_bin'] = pd.cut(df['y_true'], bins=40)
    
    # Aggregate: mean prediction and count per bin
    bin_stats = df.groupby('x_bin').agg({'y_pred': 'mean', 'y_true': 'count'}).reset_index()
    bin_stats['x_pos'] = bin_stats['x_bin'].apply(lambda x: x.mid).astype(float)
    bin_stats = bin_stats[bin_stats['y_true'] > 0]  # Remove empty bins
    
    # Compute metrics (simple, no weighted version to avoid performance issues)
    metrics = compute_metrics(y_true, y_pred, weights=None)
    
    # Normalize bubble sizes
    sizes = bin_stats['y_true'].values
    sizes_norm = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6) * 450
    
    plt.figure(figsize=(10, 10))
    
    # Scatter with bubble sizes
    plt.scatter(bin_stats['x_pos'], bin_stats['y_pred'], s=sizes_norm, 
                alpha=0.6, c=color, edgecolors='black', linewidth=0.8)
    
    # Perfect alignment diagonal
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5, label='Theoretical Ideal (y=x)')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("BKT Theoretical Prior", fontsize=14)
    plt.ylabel(f"{label_prefix} Parameter Estimation", fontsize=14)
    
    # Statistics box with all correlation metrics (optional)
    if show_metrics:
        textstr = (f"Pearson $r = {metrics['pearson_r']:.3f}$\n"
                   f"Spearman $\\rho = {metrics['spearman_r']:.3f}$\n"
                   f"Weighted $r = {metrics['weighted_r']:.3f}$\n"
                   f"$R^2 = {metrics['r2']:.3f}$\n"
                   f"MAE = {metrics['mae']:.3f}")
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.05, 0.95, textstr, fontsize=12, verticalalignment='top', bbox=props)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="examples/validation/results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_path = os.path.join(args.exp_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config.get('params', config.get('model_config', {}))
    
    # Robust migration of keys: Prioritize the 'params' or 'model_config' values from the checkpoint
    # Especially for architecture parameters that cause size mismatches
    arch_keys = ['d_model', 'n_blocks', 'dropout', 'd_ff', 'final_fc_dim', 'num_attn_heads', 'n_heads', 'ablation']
    for k in arch_keys:
        if k in model_config:
            continue # already there
        if 'params' in config and k in config['params']:
            model_config[k] = config['params'][k]
        elif 'train_config' in config and k in config['train_config']:
            model_config[k] = config['train_config'][k]
        elif 'defaults' in config and k in config['defaults']:
            model_config[k] = config['defaults'][k]
    
    # Force alignment between num_attn_heads and n_heads
    if 'n_heads' in model_config:
        model_config['num_attn_heads'] = model_config['n_heads']
    elif 'num_attn_heads' in model_config:
        model_config['n_heads'] = model_config['num_attn_heads']

    print(f"Detected Config for {args.exp_dir}:")
    print(f" > n_blocks: {model_config.get('n_blocks')}, n_heads: {model_config.get('num_attn_heads')}, ablation: {model_config.get('ablation')}")

    defaults = {
        'kq_same': 1, 'separate_qa': 0, 'l2_rasch': 0.0, 'pretrain_dim': 768, 
        'ablation': 'none', 'n_uid': 0, 'num_attn_heads': 8, 'final_fc_dim': 512,
        'd_model': 64, 'n_blocks': 2, 'd_ff': 256, 'dropout': 0.1
    }
    for k, v in defaults.items():
        if k not in model_config: model_config[k] = v
    
    # Infer dataset name from experiment directory path (e.g., .../nips_task34/fold_0_...)
    # This is more reliable than trusting config.json which may not have dataset_name
    exp_path_parts = os.path.normpath(args.exp_dir).split(os.sep)
    dataset_name = None
    for part in exp_path_parts:
        if part.startswith('fold_'):
            # The dataset name is the parent of the fold directory
            idx = exp_path_parts.index(part)
            if idx > 0:
                dataset_name = exp_path_parts[idx - 1]
                break
    if not dataset_name:
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

    data_config_path = os.path.join(PROJECT_ROOT, "configs/data_config.json")
    with open(data_config_path, 'r') as f:
        dc = json.load(f)
    dpath = os.path.join(PROJECT_ROOT, dc[dataset_name]['dpath'].replace("../", ""))
    
    # CRITICAL: Load checkpoint first to infer dimensions from the trained model
    # The global data_config.json may have been updated since training, so we must
    # use the checkpoint's actual dimensions to initialize the model correctly
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Infer dimensions from checkpoint parameter shapes
    # These represent the actual dimensions the model was trained with
    num_c = state_dict['bkt_guess'].shape[0] - 1  # BKT params have num_c+1 size
    num_q = state_dict['difficult_param.weight'].shape[0] - 1 if 'difficult_param.weight' in state_dict else state_dict['q_embed.weight'].shape[0]
    
    # Use dataset info from data_config but override with checkpoint dimensions
    dataset_info = dc[dataset_name].copy()
    dataset_info['num_q'] = num_q
    dataset_info['num_c'] = num_c
    dataset_info['input_type'] = model_config.get('input_type', dataset_info.get('input_type'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model('gtransformer', model_config, dataset_info, model_config.get('emb_type', 'qid'))
    model.load_state_dict(state_dict, strict=False)
    
    # Load BKT parameters - check both dpath and its parent directory
    bkt_params_path = os.path.join(dpath, "bkt_skill_params.pkl")
    if not os.path.exists(bkt_params_path):
        # Try parent directory (some datasets have BKT params at parent level)
        bkt_params_path = os.path.join(os.path.dirname(dpath), "bkt_skill_params.pkl")
    
    with open(bkt_params_path, "rb") as f:
        bkt_params = pickle.load(f)
    model.load_theory_params(bkt_params)
    model.to(device)
    model.eval()
    
    test_file = os.path.join(dpath, "train_valid_sequences.csv")
    target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
    dataset = GTransformerDataset(test_file, dc[dataset_name]["input_type"], {fold}, target_path=target_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    print(f"Executing Parameter Recovery Validation for {dataset_name}...")
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
            
            outputs, _, _ = model(c, r, pid_data=q, qtest=True)
            mask = sm.bool()
            
            # Use .get() to handle cases where these keys might be missing (e.g. baseline)
            # Fill with zeros of the correct shape if missing, so indices won't fail
            p_l0 = outputs.get("p_l0", torch.zeros_like(mask).float())
            p_t = outputs.get("p_t", torch.zeros_like(mask).float())
            p_l0_probe = outputs.get("p_l0_probe", torch.zeros_like(mask).float())
            p_t_probe = outputs.get("p_t_probe", torch.zeros_like(mask).float())

            results["l0_grounded"].append(p_l0[mask].cpu().numpy())
            results["t_grounded"].append(p_t[mask].cpu().numpy())
            results["l0_probe"].append(p_l0_probe[mask].cpu().numpy())
            results["t_probe"].append(p_t_probe[mask].cpu().numpy())
            results["l0_true"].append(data["target_l0"].to(device)[mask].cpu().numpy())
            results["t_true"].append(data["target_t"].to(device)[mask].cpu().numpy())
            results["skill"].append(c[mask].cpu().numpy())
            
            if i > 500: break # Ensure high statistical significance

    for k in results: results[k] = np.concatenate(results[k])
    
    summary = {
        "l0_grounded": compute_metrics(results["l0_true"], results["l0_grounded"]),
        "l0_probe": compute_metrics(results["l0_true"], results["l0_probe"]),
        "t_grounded": compute_metrics(results["t_true"], results["t_grounded"]),
        "t_probe": compute_metrics(results["t_true"], results["t_probe"]),
        "n_samples": int(len(results["l0_true"]))
    }
    
    # Per-Skill Analysis for Paper Table
    df = pd.DataFrame({
        "skill": results["skill"],
        "l0_true": results["l0_true"],
        "l0_grounded": results["l0_grounded"]
    })
    
    skill_metrics = []
    for skill_id, group in df.groupby("skill"):
        if len(group) >= 10:
            m = compute_metrics(group["l0_true"].values, group["l0_grounded"].values)
            skill_metrics.append({"skill_id": int(skill_id), **m})
    
    pd.DataFrame(skill_metrics).to_csv(os.path.join(args.output_dir, "h12_skill_recovery_metrics.csv"), index=False)
    
    # Generate Plots (without metrics legend for paper)
    plot_recovery(results["l0_true"], results["l0_probe"], 
                  "Initial Mastery Recovery ($P(L_0)$ Probing)", 
                  os.path.join(args.output_dir, "h12_recovery_l0_probe.png"), color='royalblue', label_prefix="Probe", show_metrics=False)
    
    plot_recovery(results["t_true"], results["t_probe"], 
                  "Learning Rate Recovery ($P(T)$ Probing)", 
                  os.path.join(args.output_dir, "h12_recovery_t_probe.png"), color='royalblue', label_prefix="Probe", show_metrics=False)
    
    # Generate Grounded Parameter Plots (for H1.2 Semantic Alignment)
    plot_recovery(results["l0_true"], results["l0_grounded"], 
                  "Initial Mastery Preservation ($P(L_0)$ Grounded)", 
                  os.path.join(args.output_dir, "h12_recovery_l0_grounded.png"), 
                  color='royalblue', label_prefix="Grounded", show_metrics=False)
    
    plot_recovery(results["t_true"], results["t_grounded"], 
                  "Learning Rate Preservation ($P(T)$ Grounded)", 
                  os.path.join(args.output_dir, "h12_recovery_t_grounded.png"), 
                  color='royalblue', label_prefix="Grounded", show_metrics=False)

    with open(os.path.join(args.output_dir, "h12_recovery_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Validation results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
