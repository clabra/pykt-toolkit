
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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

def compute_control_and_selectivity(y_true, y_pred):
    """
    Compute control R² and selectivity metrics for probing analysis.
    Matches the methodology from structural_encoding_validation.py
    
    Returns:
        dict with fidelity_r2, control_r2, and selectivity
    """
    # Use predictions as features (similar to latent representations in structural validation)
    X = y_pred.reshape(-1, 1)
    y = y_true
    
    if len(y) < 10:
        return {"fidelity_r2": 0, "control_r2": 0, "selectivity": 0}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Fidelity: R² between true and predicted values (on full dataset)
    fidelity_r2 = float(r2_score(y_true, y_pred))
    
    # 2. Control: R² with observation-level shuffled labels (both train and test)
    # Train probe on shuffled training labels
    probe_control = Ridge(alpha=1.0)
    y_train_shuffled = np.random.RandomState(42).permutation(y_train)
    probe_control.fit(X_train, y_train_shuffled)
    
    # Evaluate on shuffled test labels (this creates the control baseline)
    y_test_shuffled = np.random.RandomState(43).permutation(y_test)
    control_r2 = float(r2_score(y_test_shuffled, probe_control.predict(X_test)))
    
    # 3. Selectivity: Fidelity - Control
    selectivity = fidelity_r2 - control_r2
    
    return {
        "fidelity_r2": fidelity_r2,
        "control_r2": control_r2,
        "selectivity": selectivity
    }

def plot_recovery(y_true, y_pred, title, output_path, color='royalblue', label_prefix="", show_metrics=False, 
                  fidelity_r2=None, fidelity_r2_std=None, pearson_r=None, pearson_r_std=None,
                  control_r2=None, selectivity=None, selectivity_std=None, 
                  spearman_rho=None, spearman_rho_std=None):
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
    
    # Add Fidelity, Control, and Selectivity metrics if provided (for probe plots)
    if fidelity_r2 is not None and control_r2 is not None and selectivity is not None:
        # Format with uncertainties if provided
        r2_str = f"{fidelity_r2:.3f} ± {fidelity_r2_std:.3f}" if fidelity_r2_std is not None else f"{fidelity_r2:.3f}"
        r_str = f"{pearson_r:.3f} ± {pearson_r_std:.3f}" if pearson_r is not None and pearson_r_std is not None else ""
        sel_str = f"{selectivity:.3f} ± {selectivity_std:.3f}" if selectivity_std is not None else f"{selectivity:.3f}"
        
        metrics_text = f"$R^2$ = {r2_str}\n"
        if r_str:
            metrics_text += f"Pearson $r$ = {r_str}\n"
        metrics_text += f"$\\Delta R^2$ = {sel_str}"
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1.2)
        plt.text(0.05, 0.95, metrics_text, fontsize=12, verticalalignment='top', 
                bbox=props, transform=plt.gca().transAxes)
    
    # Add Spearman correlation if provided (for grounded parameter plots)
    if spearman_rho is not None:
        rho_str = f"{spearman_rho:.3f} ± {spearman_rho_std:.3f}" if spearman_rho_std is not None else f"{spearman_rho:.3f}"
        spearman_text = f"Spearman $\\rho$ = {rho_str}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1.2)
        plt.text(0.05, 0.95, spearman_text, fontsize=12, verticalalignment='top', 
                bbox=props, transform=plt.gca().transAxes)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return metrics

def process_single_fold(exp_dir, dataset_name, fold, output_dir):
    """Process a single fold and return metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(exp_dir, "config.json")
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

    print(f"Detected Config for {exp_dir}:")
    print(f" > n_blocks: {model_config.get('n_blocks')}, n_heads: {model_config.get('num_attn_heads')}, ablation: {model_config.get('ablation')}")

    defaults = {
        'kq_same': 1, 'separate_qa': 0, 'l2_rasch': 0.0, 'pretrain_dim': 768, 
        'ablation': 'none', 'n_uid': 0, 'num_attn_heads': 8, 'final_fc_dim': 512,
        'd_model': 64, 'n_blocks': 2, 'd_ff': 256, 'dropout': 0.1
    }
    for k, v in defaults.items():
        if k not in model_config: model_config[k] = v
    
    # Discovery
    checkpoint_path = None
    for root, _, files in os.walk(exp_dir):
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
    
    print(f"Processing fold {fold} for {dataset_name}...")
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
        "fold": fold,
        "l0_grounded": compute_metrics(results["l0_true"], results["l0_grounded"]),
        "l0_probe": compute_metrics(results["l0_true"], results["l0_probe"]),
        "t_grounded": compute_metrics(results["t_true"], results["t_grounded"]),
        "t_probe": compute_metrics(results["t_true"], results["t_probe"]),
        "n_samples": int(len(results["l0_true"]))
    }
    
    return summary, results
    return summary, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, 
                       help="Path to experiment directory containing gtransformer/<dataset>/ subdirectory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save validation results (default: <exp_dir>/validation)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset name (auto-detected if not provided)")
    args = parser.parse_args()
    
    # Auto-detect dataset name from directory structure
    exp_path_parts = os.path.normpath(args.exp_dir).split(os.sep)
    dataset_name = args.dataset
    
    if not dataset_name:
        # Try to find dataset name from path (e.g., .../gtransformer/assist2009/...)
        for i, part in enumerate(exp_path_parts):
            if part == 'gtransformer' and i + 1 < len(exp_path_parts):
                dataset_name = exp_path_parts[i + 1]
                break
    
    if not dataset_name:
        raise ValueError("Could not auto-detect dataset name. Please provide --dataset argument")
    
    # Set output directory
    if args.output_dir is None:
        # Default: create validation folder at gtransformer/<dataset>/validation
        # Find the gtransformer/<dataset> parent directory
        for i, part in enumerate(exp_path_parts):
            if part == dataset_name and i > 0 and exp_path_parts[i-1] == 'gtransformer':
                parent_dir = os.sep.join(exp_path_parts[:i+1])
                args.output_dir = os.path.join(parent_dir, 'validation')
                break
        if args.output_dir is None:
            args.output_dir = os.path.join(args.exp_dir, 'validation')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all fold directories
    fold_dirs = []
    for item in os.listdir(args.exp_dir):
        item_path = os.path.join(args.exp_dir, item)
        if os.path.isdir(item_path) and item.startswith('fold_'):
            try:
                fold_num = int(item.split('_')[1])
                fold_dirs.append((fold_num, item_path))
            except (IndexError, ValueError):
                continue
    
    fold_dirs.sort()  # Sort by fold number
    
    if not fold_dirs:
        print(f"No fold directories found in {args.exp_dir}")
        return
    
    print(f"\nFound {len(fold_dirs)} folds for dataset {dataset_name}")
    print("="*80)
    
    # Process each fold
    all_fold_summaries = []
    all_fold_results = []
    
    for fold_num, fold_dir in fold_dirs:
        print(f"\nProcessing fold {fold_num}: {fold_dir}")
        try:
            summary, results = process_single_fold(fold_dir, dataset_name, fold_num, args.output_dir)
            all_fold_summaries.append(summary)
            all_fold_results.append(results)
            
            # Save per-fold results
            fold_output = os.path.join(args.output_dir, f"h12_recovery_fold{fold_num}_summary.json")
            with open(fold_output, 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"  L0 Grounded: Spearman ρ={summary['l0_grounded']['spearman_r']:.3f}, MAE={summary['l0_grounded']['mae']:.3f}")
            print(f"  T Grounded:  Spearman ρ={summary['t_grounded']['spearman_r']:.3f}, MAE={summary['t_grounded']['mae']:.3f}")
            
        except Exception as e:
            print(f"  ERROR processing fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_fold_summaries:
        print("\nNo folds were successfully processed!")
        return
    
    print("\n" + "="*80)
    print("Computing 5-fold CV Statistics")
    print("="*80)
    
    # Aggregate metrics across folds
    def aggregate_metrics(summaries, metric_path):
        """Extract metric values across folds and compute mean ± std."""
        values = []
        for s in summaries:
            val = s
            for key in metric_path.split('.'):
                val = val[key]
            values.append(val)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    aggregated = {
        'dataset': dataset_name,
        'n_folds': len(all_fold_summaries),
        'l0_grounded': {
            'spearman_r': aggregate_metrics(all_fold_summaries, 'l0_grounded.spearman_r'),
            'pearson_r': aggregate_metrics(all_fold_summaries, 'l0_grounded.pearson_r'),
            'mae': aggregate_metrics(all_fold_summaries, 'l0_grounded.mae'),
            'rmse': aggregate_metrics(all_fold_summaries, 'l0_grounded.rmse'),
            'r2': aggregate_metrics(all_fold_summaries, 'l0_grounded.r2'),
        },
        't_grounded': {
            'spearman_r': aggregate_metrics(all_fold_summaries, 't_grounded.spearman_r'),
            'pearson_r': aggregate_metrics(all_fold_summaries, 't_grounded.pearson_r'),
            'mae': aggregate_metrics(all_fold_summaries, 't_grounded.mae'),
            'rmse': aggregate_metrics(all_fold_summaries, 't_grounded.rmse'),
            'r2': aggregate_metrics(all_fold_summaries, 't_grounded.r2'),
        },
        'l0_probe': {
            'spearman_r': aggregate_metrics(all_fold_summaries, 'l0_probe.spearman_r'),
            'pearson_r': aggregate_metrics(all_fold_summaries, 'l0_probe.pearson_r'),
            'mae': aggregate_metrics(all_fold_summaries, 'l0_probe.mae'),
            'rmse': aggregate_metrics(all_fold_summaries, 'l0_probe.rmse'),
            'r2': aggregate_metrics(all_fold_summaries, 'l0_probe.r2'),
        },
        't_probe': {
            'spearman_r': aggregate_metrics(all_fold_summaries, 't_probe.spearman_r'),
            'pearson_r': aggregate_metrics(all_fold_summaries, 't_probe.pearson_r'),
            'mae': aggregate_metrics(all_fold_summaries, 't_probe.mae'),
            'rmse': aggregate_metrics(all_fold_summaries, 't_probe.rmse'),
            'r2': aggregate_metrics(all_fold_summaries, 't_probe.r2'),
        },
        'total_samples': sum(s['n_samples'] for s in all_fold_summaries)
    }
    
    # Save aggregated results
    aggregated_path = os.path.join(args.output_dir, "h12_recovery_aggregated.json")
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=4)
    
    print(f"\n5-Fold CV Results for {dataset_name}:")
    print("-" * 80)
    print(f"Total samples across folds: {aggregated['total_samples']:,}")
    print("\nL0 Grounded (Initial Mastery):")
    print(f"  Spearman ρ: {aggregated['l0_grounded']['spearman_r']['mean']:.3f} ± {aggregated['l0_grounded']['spearman_r']['std']:.3f}")
    print(f"  Pearson r:  {aggregated['l0_grounded']['pearson_r']['mean']:.3f} ± {aggregated['l0_grounded']['pearson_r']['std']:.3f}")
    print(f"  MAE:        {aggregated['l0_grounded']['mae']['mean']:.3f} ± {aggregated['l0_grounded']['mae']['std']:.3f}")
    print(f"  RMSE:       {aggregated['l0_grounded']['rmse']['mean']:.3f} ± {aggregated['l0_grounded']['rmse']['std']:.3f}")
    
    print("\nT Grounded (Learning Rate):")
    print(f"  Spearman ρ: {aggregated['t_grounded']['spearman_r']['mean']:.3f} ± {aggregated['t_grounded']['spearman_r']['std']:.3f}")
    print(f"  Pearson r:  {aggregated['t_grounded']['pearson_r']['mean']:.3f} ± {aggregated['t_grounded']['pearson_r']['std']:.3f}")
    print(f"  MAE:        {aggregated['t_grounded']['mae']['mean']:.3f} ± {aggregated['t_grounded']['mae']['std']:.3f}")
    print(f"  RMSE:       {aggregated['t_grounded']['rmse']['mean']:.3f} ± {aggregated['t_grounded']['rmse']['std']:.3f}")
    
    print("\nL0 Probe (Initial Mastery):")
    print(f"  Spearman ρ: {aggregated['l0_probe']['spearman_r']['mean']:.3f} ± {aggregated['l0_probe']['spearman_r']['std']:.3f}")
    print(f"  Pearson r:  {aggregated['l0_probe']['pearson_r']['mean']:.3f} ± {aggregated['l0_probe']['pearson_r']['std']:.3f}")
    
    print("\nT Probe (Learning Rate):")
    print(f"  Spearman ρ: {aggregated['t_probe']['spearman_r']['mean']:.3f} ± {aggregated['t_probe']['spearman_r']['std']:.3f}")
    print(f"  Pearson r:  {aggregated['t_probe']['pearson_r']['mean']:.3f} ± {aggregated['t_probe']['pearson_r']['std']:.3f}")
    
    print(f"\n✅ Aggregated results saved to: {aggregated_path}")
    print(f"✅ Per-fold results saved to: {args.output_dir}/h12_recovery_fold*_summary.json")
    
    # Generate plots using combined data from all folds
    print("\nGenerating 5-fold aggregated plots...")
    combined_results = {
        "l0_true": np.concatenate([r["l0_true"] for r in all_fold_results]),
        "l0_grounded": np.concatenate([r["l0_grounded"] for r in all_fold_results]),
        "l0_probe": np.concatenate([r["l0_probe"] for r in all_fold_results]),
        "t_true": np.concatenate([r["t_true"] for r in all_fold_results]),
        "t_grounded": np.concatenate([r["t_grounded"] for r in all_fold_results]),
        "t_probe": np.concatenate([r["t_probe"] for r in all_fold_results]),
        "skill": np.concatenate([r["skill"] for r in all_fold_results]),
    }
    
    # Per-Skill Analysis for Paper Table
    df = pd.DataFrame({
        "skill": combined_results["skill"],
        "l0_true": combined_results["l0_true"],
        "l0_grounded": combined_results["l0_grounded"]
    })
    
    skill_metrics = []
    for skill_id, group in df.groupby("skill"):
        if len(group) >= 10:
            m = compute_metrics(group["l0_true"].values, group["l0_grounded"].values)
            skill_metrics.append({"skill_id": int(skill_id), **m})
    
    pd.DataFrame(skill_metrics).to_csv(os.path.join(args.output_dir, "h12_skill_recovery_metrics.csv"), index=False)
    
    # Load probe metrics from structural encoding validation results
    structural_file = os.path.join(args.output_dir, "structural_encoding_aggregated.json")
    if os.path.exists(structural_file):
        with open(structural_file, 'r') as f:
            structural_results = json.load(f)
        
        # Compute means AND stds from fold arrays - use selectivity_std (standard) not selectivity_strict
        l0_fidelity_r2_mean = np.mean(structural_results['results']['l0']['fidelity_r2'])
        l0_fidelity_r2_std = np.std(structural_results['results']['l0']['fidelity_r2'])
        l0_pearson_mean = np.mean(structural_results['results']['l0']['fidelity_pearson'])
        l0_pearson_std = np.std(structural_results['results']['l0']['fidelity_pearson'])
        l0_selectivity_mean = np.mean(structural_results['results']['l0']['selectivity_std'])
        l0_selectivity_std = np.std(structural_results['results']['l0']['selectivity_std'])
        # Control R² = Fidelity R² - Selectivity
        l0_control = l0_fidelity_r2_mean - l0_selectivity_mean
        
        t_fidelity_r2_mean = np.mean(structural_results['results']['t']['fidelity_r2'])
        t_fidelity_r2_std = np.std(structural_results['results']['t']['fidelity_r2'])
        t_pearson_mean = np.mean(structural_results['results']['t']['fidelity_pearson'])
        t_pearson_std = np.std(structural_results['results']['t']['fidelity_pearson'])
        t_selectivity_mean = np.mean(structural_results['results']['t']['selectivity_std'])
        t_selectivity_std = np.std(structural_results['results']['t']['selectivity_std'])
        t_control = t_fidelity_r2_mean - t_selectivity_mean
        
        l0_probe_metrics = {
            'fidelity_r2': l0_fidelity_r2_mean,
            'fidelity_r2_std': l0_fidelity_r2_std,
            'pearson_r': l0_pearson_mean,
            'pearson_r_std': l0_pearson_std,
            'control_r2': l0_control,
            'selectivity': l0_selectivity_mean,
            'selectivity_std': l0_selectivity_std
        }
        t_probe_metrics = {
            'fidelity_r2': t_fidelity_r2_mean,
            'fidelity_r2_std': t_fidelity_r2_std,
            'pearson_r': t_pearson_mean,
            'pearson_r_std': t_pearson_std,
            'control_r2': t_control,
            'selectivity': t_selectivity_mean,
            'selectivity_std': t_selectivity_std
        }
        
        print("\n" + "="*80)
        print("Probe Metrics (from structural_encoding_aggregated.json)")
        print("="*80)
        print(f"L0 Probe - Fidelity R²: {l0_probe_metrics['fidelity_r2']:.3f} ± {l0_probe_metrics['fidelity_r2_std']:.3f}, "
              f"Pearson r: {l0_probe_metrics['pearson_r']:.3f} ± {l0_probe_metrics['pearson_r_std']:.3f}, "
              f"Selectivity: {l0_probe_metrics['selectivity']:.3f} ± {l0_probe_metrics['selectivity_std']:.3f}")
        print(f"T Probe  - Fidelity R²: {t_probe_metrics['fidelity_r2']:.3f} ± {t_probe_metrics['fidelity_r2_std']:.3f}, "
              f"Pearson r: {t_probe_metrics['pearson_r']:.3f} ± {t_probe_metrics['pearson_r_std']:.3f}, "
              f"Selectivity: {t_probe_metrics['selectivity']:.3f} ± {t_probe_metrics['selectivity_std']:.3f}")
    else:
        print(f"⚠️  Warning: {structural_file} not found. Computing probe metrics on-the-fly...")
        l0_probe_metrics = compute_control_and_selectivity(combined_results["l0_true"], combined_results["l0_probe"])
        t_probe_metrics = compute_control_and_selectivity(combined_results["t_true"], combined_results["t_probe"])
        
        print("\n" + "="*80)
        print("Probe Metrics (computed on-the-fly)")
        print("="*80)
        print(f"L0 Probe - Fidelity R²: {l0_probe_metrics['fidelity_r2']:.3f}, "
              f"Control R²: {l0_probe_metrics['control_r2']:.3f}, "
              f"Selectivity: {l0_probe_metrics['selectivity']:.3f}")
        print(f"T Probe  - Fidelity R²: {t_probe_metrics['fidelity_r2']:.3f}, "
              f"Control R²: {t_probe_metrics['control_r2']:.3f}, "
              f"Selectivity: {t_probe_metrics['selectivity']:.3f}")
    
    # Generate Plots with probe metrics displayed
    plot_recovery(combined_results["l0_true"], combined_results["l0_probe"], 
                  "Initial Mastery Structural Encoding P(L0)", 
                  os.path.join(args.output_dir, "h12_recovery_l0_probe.png"), 
                  color='royalblue', label_prefix="Probe", show_metrics=False,
                  fidelity_r2=l0_probe_metrics['fidelity_r2'],
                  fidelity_r2_std=l0_probe_metrics.get('fidelity_r2_std'),
                  pearson_r=l0_probe_metrics.get('pearson_r'),
                  pearson_r_std=l0_probe_metrics.get('pearson_r_std'),
                  control_r2=l0_probe_metrics['control_r2'],
                  selectivity=l0_probe_metrics['selectivity'],
                  selectivity_std=l0_probe_metrics.get('selectivity_std'))
    
    plot_recovery(combined_results["t_true"], combined_results["t_probe"], 
                  "Learning Rate Structural Encoding P(T)", 
                  os.path.join(args.output_dir, "h12_recovery_t_probe.png"), 
                  color='royalblue', label_prefix="Probe", show_metrics=False,
                  fidelity_r2=t_probe_metrics['fidelity_r2'],
                  fidelity_r2_std=t_probe_metrics.get('fidelity_r2_std'),
                  pearson_r=t_probe_metrics.get('pearson_r'),
                  pearson_r_std=t_probe_metrics.get('pearson_r_std'),
                  control_r2=t_probe_metrics['control_r2'],
                  selectivity=t_probe_metrics['selectivity'],
                  selectivity_std=t_probe_metrics.get('selectivity_std'))
    
    # Generate Grounded Parameter Plots (for H1.2 Semantic Alignment) with Spearman ρ
    plot_recovery(combined_results["l0_true"], combined_results["l0_grounded"], 
                  "Initial Mastery Semantic Alignment P(L0)", 
                  os.path.join(args.output_dir, "h12_recovery_l0_grounded.png"), 
                  color='royalblue', label_prefix="Grounded", show_metrics=False,
                  spearman_rho=aggregated['l0_grounded']['spearman_r']['mean'],
                  spearman_rho_std=aggregated['l0_grounded']['spearman_r']['std'])
    
    plot_recovery(combined_results["t_true"], combined_results["t_grounded"], 
                  "Learning Rate Semantic Alignment P(T)", 
                  os.path.join(args.output_dir, "h12_recovery_t_grounded.png"), 
                  color='royalblue', label_prefix="Grounded", show_metrics=False,
                  spearman_rho=aggregated['t_grounded']['spearman_r']['mean'],
                  spearman_rho_std=aggregated['t_grounded']['spearman_r']['std'])

    print(f"✅ Plots saved to: {args.output_dir}/h12_recovery_*.png")
    print("\n" + "="*80)
    print("Validation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
