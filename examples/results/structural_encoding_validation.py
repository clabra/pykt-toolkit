
import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
import csv

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def calculate_metrics(X, y, skills, construct_name="L0"):
    """
    Calculates Fidelity (R2, Pearson) and Selectivity (Standard, Strict).
    """
    print(f"\n--- Analyzing {construct_name} ---")
    
    # 1. Fidelity (True Task)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    
    r2_true = float(r2_score(y_test, y_pred))
    corr, _ = pearsonr(y_test, y_pred)
    
    # 2. Selectivity (Standard): Observation-level shuffling
    control_probe_obs = Ridge(alpha=1.0)
    y_train_shuffled = np.random.permutation(y_train)
    control_probe_obs.fit(X_train, y_train_shuffled)
    y_test_shuffled = np.random.permutation(y_test)
    r2_control_obs = float(r2_score(y_test_shuffled, control_probe_obs.predict(X_test)))

    # 3. Selectivity (Strict): Skill-consistent shuffling
    unique_skills = np.unique(skills)
    skill_to_true_val = {s: np.mean(y[skills == s]) for s in unique_skills}
    shuffled_vals = list(skill_to_true_val.values())
    np.random.seed(42)
    np.random.shuffle(shuffled_vals)
    skill_to_shuffled_val = dict(zip(unique_skills, shuffled_vals))
    y_shuffled_skills = np.array([skill_to_shuffled_val[s] for s in skills])
    
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y_shuffled_skills, test_size=0.2, random_state=42)
    control_probe_skill = Ridge(alpha=1.0)
    control_probe_skill.fit(X_train_k, y_train_k)
    r2_control_skill = float(r2_score(y_test_k, control_probe_skill.predict(X_test_k)))
    
    selectivity_std = r2_true - r2_control_obs
    selectivity_strict = r2_true - r2_control_skill
    
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 40)
    print(f"{'R^2 (Fidelity)':<25} | {r2_true:>10.4f}")
    print(f"{'Pearson r':<25} | {corr:>10.4f}")
    print(f"{'Selectivity (Standard)':<25} | {selectivity_std:>10.4f}")
    print(f"{'Selectivity (Strict)':<25} | {selectivity_strict:>10.4f}")
    
    return {
        "fidelity": {
            "r2": r2_true,
            "pearson_r": float(corr)
        },
        "selectivity": {
            "standard_delta_r2": selectivity_std,
            "strict_delta_r2": selectivity_strict,
            "control_r2_obs": r2_control_obs,
            "control_r2_skill": r2_control_skill
        }
    }

def plot_parity_recovery(y_true, y_pred, output_path, title="Structural Fidelity: Recovery Diagonal"):
    """Generates X=BKT Estimation, Y=Probe Prediction Parity Plot."""
    corr, _ = pearsonr(y_true, y_pred)
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['x_bin'] = pd.cut(df['y_true'], bins=40)
    bin_stats = df.groupby('x_bin').agg({'y_pred': 'mean', 'y_true': 'count'}).reset_index()
    bin_stats['x_pos'] = bin_stats['x_bin'].apply(lambda x: x.mid).astype(float)
    bin_stats = bin_stats[bin_stats['y_true'] > 0]
    
    sizes = bin_stats['y_true'].values
    sizes_norm = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6) * 450
    
    plt.figure(figsize=(10, 10))
    plt.scatter(bin_stats['x_pos'], bin_stats['y_pred'], s=sizes_norm, 
                alpha=0.6, c='royalblue', edgecolors='black', linewidth=0.8)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("BKT Theoretical Prior", fontsize=14)
    plt.ylabel("Latent-Extracted Parameter (Probe Mean)", fontsize=14)
    
    textstr = f"$r = {corr:.3f}$\n$R^2 = {r2_score(y_true, y_pred):.3f}$"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, textstr, fontsize=14, verticalalignment='top', bbox=props)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_manifold_pca(X, y, output_path, title="Latent Space (PCA)", param_label="L0"):
    """Projects embeddings to 2D and colors by BKT parameter."""
    print(f"Computing PCA for {param_label}...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.clip(y, 0, 1), cmap='magma', alpha=0.6, s=15)
    plt.colorbar(sc).set_label(f'BKT {param_label}', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_manifold_tsne(X, y, skills, skill_names, output_path, param_label="L0"):
    """Projects embeddings to 2D using t-SNE with parameter and skill colorings."""
    print(f"Computing t-SNE for {param_label}...")
    # Pre-reduce with PCA for stability
    X_red = PCA(n_components=50).fit_transform(X) if X.shape[1] > 50 else X
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_red)

    # 1. By BKT Parameter
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(y, 0, 1), cmap='magma', alpha=0.6, s=15)
    plt.colorbar(sc).set_label(f'BKT {param_label}', fontsize=12)
    plt.title("Latent Space Organization (t-SNE)", fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    # 2. By Top Skills
    top_skills = pd.Series(skills).value_counts().nlargest(10).index
    mask = np.isin(skills, top_skills)
    plt.figure(figsize=(13, 8))
    plt.scatter(X_tsne[~mask, 0], X_tsne[~mask, 1], c='grey', alpha=0.1, s=5, label='Other Skills')
    labels = np.array([skill_names.get(int(idx), str(idx)) for idx in skills[mask]])
    sns.scatterplot(x=X_tsne[mask, 0], y=X_tsne[mask, 1], hue=labels, palette="tab10", alpha=0.8, s=25)
    plt.title("Latent Space by Skill", fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Skill (ID: Name)")
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_by_skill.png"), dpi=300)
    plt.close()

def get_skill_names(dpath):
    """Loads skill ID to Name mapping."""
    keyid2idx_path = os.path.join(dpath, "keyid2idx.json")
    mapping = {}
    if os.path.exists(keyid2idx_path):
        with open(keyid2idx_path, 'r') as f:
            data = json.load(f).get("concepts", {})
            idx_to_orig = {int(v): str(k) for k, v in data.items()}
        
        csv_path = os.path.join(dpath, "skill_builder_data_corrected_collapsed.csv")
        if os.path.exists(csv_path):
            id_to_name = {}
            with open(csv_path, 'r', encoding='ISO-8859-1') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid, name = row.get("skill_id", "").replace('"', ''), row.get("skill_name", "")
                    if sid and name: id_to_name[sid] = name
            for idx, sid in idx_to_orig.items():
                name = id_to_name.get(sid, "")
                mapping[idx] = f"{sid}: {name}" if name else sid
        else:
            mapping = idx_to_orig
    return mapping

def extract_all(model, loader, device, n_samples=8000):
    model.eval()
    all_z, all_l0, all_t, all_skills = [], [], [], []
    collected = 0
    with torch.no_grad():
        for data in loader:
            q, c, r, sm = data["qseqs"].long().to(device), data["cseqs"].long().to(device), \
                          data["rseqs"].long().to(device), data["smasks"].long().to(device)
            _, _, z_context = model(c, r, pid_data=q, qtest=True)
            mask = sm.bool()
            
            all_z.append(z_context[mask].cpu().numpy())
            all_l0.append(data["target_l0"].to(device)[mask].cpu().numpy())
            all_t.append(data["target_t"].to(device)[mask].cpu().numpy())
            all_skills.append(c[mask].cpu().numpy())
            
            collected += mask.sum().item()
            if collected >= n_samples: break
                
    X = np.concatenate(all_z, axis=0)[:n_samples]
    y_l0 = np.concatenate(all_l0, axis=0)[:n_samples]
    y_t = np.concatenate(all_t, axis=0)[:n_samples]
    skills = np.concatenate(all_skills, axis=0)[:n_samples]
    return X, y_l0, y_t, skills

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to fold directory")
    args = parser.parse_args()

    EXP_DIR = args.exp_dir.rstrip('/')
    # Campaign level is 3 levels up from fold: Campaign/model/dataset/fold
    # But usually user passes something like experiments/CAMPAIGN/gtransformer/assist2009/fold_0_...
    # So we find the campaign root by looking for 'experiments' in path or going back
    parts = EXP_DIR.split('/')
    if 'experiments' in parts:
        idx = parts.index('experiments')
        campaign_dir = '/'.join(parts[:idx+2])
    else:
        # Fallback to model level validation if campaign can't be inferred safely
        campaign_dir = os.path.dirname(os.path.dirname(os.path.dirname(EXP_DIR)))
    
    VALIDATION_DIR = os.path.join(campaign_dir, "validation")
    PLOT_DIR = os.path.join(campaign_dir, "plots")
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    CONFIG_PATH = os.path.join(EXP_DIR, "config.json")
    with open(CONFIG_PATH, 'r') as f: config = json.load(f)
    mc = config.get('params', config.get('train_config', {}))
    
    # Infer dataset name from experiment directory path (e.g., .../algebra2005/fold_0_...)
    # This is more reliable than trusting config.json which may not have dataset_name
    exp_path_parts = os.path.normpath(EXP_DIR).split(os.sep)
    dataset_name = None
    for part in exp_path_parts:
        if part.startswith('fold_'):
            # The dataset name is the parent of the fold directory
            idx = exp_path_parts.index(part)
            if idx > 0:
                dataset_name = exp_path_parts[idx - 1]
                break
    if not dataset_name:
        dataset_name = mc.get("dataset", "assist2009")
    
    fold = mc.get("fold", 0)

    # Resolve Data Path
    with open(os.path.join(PROJECT_ROOT, 'configs/data_config.json'), 'r') as f: dc = json.load(f)
    dpath = os.path.join(PROJECT_ROOT, dc[dataset_name]['dpath'].replace("../", ""))
    
    # Load Model
    found_ckpt = None
    for r, _, files in os.walk(EXP_DIR):
        for f in files:
            if f.endswith(".ckpt"):
                found_ckpt = os.path.join(r, f); break
        if found_ckpt: break
    
    # CRITICAL: Load checkpoint first to infer dimensions from the trained model
    # The global data_config.json may have been updated since training, so we must
    # use the checkpoint's actual dimensions to initialize the model correctly
    ckpt = torch.load(found_ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Infer dimensions from checkpoint parameter shapes
    # These represent the actual dimensions the model was trained with
    num_c = state_dict['bkt_guess'].shape[0] - 1  # BKT params have num_c+1 size
    num_q = state_dict['difficult_param.weight'].shape[0] - 1 if 'difficult_param.weight' in state_dict else state_dict['q_embed.weight'].shape[0]
    
    # Use dataset info from data_config but override with checkpoint dimensions
    dataset_info = dc[dataset_name].copy()
    dataset_info['num_q'] = num_q
    dataset_info['num_c'] = num_c
    dataset_info['input_type'] = mc.get('input_type', dataset_info.get('input_type'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model('gtransformer', mc, dataset_info, mc.get('emb_type', 'qid'))
    model.load_state_dict(state_dict, strict=False)
    
    # Load BKT Params for initialization (GTransformer requirement)
    bkt_params_path = os.path.join(dpath, "bkt_skill_params.pkl")
    if not os.path.exists(bkt_params_path):
        # Try parent directory (some datasets have BKT params at parent level)
        bkt_params_path = os.path.join(os.path.dirname(dpath), "bkt_skill_params.pkl")
    if not os.path.exists(bkt_params_path):
        raise FileNotFoundError(
            f"BKT skill params not found at {bkt_params_path}\n"
            f"Structural encoding validation requires BKT oracle parameters.\n"
            f"Datasets with BKT params: assist2009, assist2015, nips_task34, algebra2005, bridge2algebra2006\n"
            f"Dataset '{dataset_name}' does not have pre-computed BKT parameters."
        )
    with open(bkt_params_path, "rb") as f:
        model.load_theory_params(pickle.load(f))
    
    model.to(device)

    # Load Data
    test_file = os.path.join(dpath, "train_valid_sequences.csv")
    target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
    dataset = GTransformerDataset(test_file, dc[dataset_name]["input_type"], {fold}, target_path=target_path)
    # Use num_workers=0 to avoid multiprocessing deadlocks in validation scripts
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Use 3000 samples for faster and more reliable t-SNE computation
    X, y_l0, y_t, skills = extract_all(model, loader, device, n_samples=3000)

    # 2. Metrics (H1.1 Validation)
    results_l0 = calculate_metrics(X, y_l0, skills, "Initial Mastery (L0)")
    results_t = calculate_metrics(X, y_t, skills, "Learning Rate (T)")
    
    # 3. Visualizations (H1.1)
    print("\nGenerating Visualizations...")
    # Parity Plots (saved to validation folder)
    # NOTE: Commented out - using validate_parameter_recovery.py for H1.2 alignment plots instead
    # X_tr, X_te, y_tr_l0, y_te_l0 = train_test_split(X, y_l0, test_size=0.2, random_state=42)
    # p_l0 = Ridge().fit(X_tr, y_tr_l0).predict(X_te)
    # plot_parity_recovery(y_te_l0, p_l0, os.path.join(VALIDATION_DIR, f"h1_structural_fidelity_l0_fold_{fold}.png"), f"Structural Fidelity (L0)")
    # 
    # X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X, y_t, test_size=0.2, random_state=42)
    # p_t = Ridge().fit(X_tr_t, y_tr_t).predict(X_te_t)
    # plot_parity_recovery(y_te_t, p_t, os.path.join(VALIDATION_DIR, f"h1_structural_fidelity_t_fold_{fold}.png"), f"Structural Fidelity (T)")
    
    # PCA and t-SNE Maps (saved to plots folder)
    skill_names = get_skill_names(dpath)
    # Generate for L0
    plot_manifold_pca(X, y_l0, os.path.join(PLOT_DIR, f"h1_latent_pca_l0_fold_{fold}.png"), f"Latent Space PCA (L0)", param_label="L0")
    plot_manifold_tsne(X, y_l0, skills, skill_names, os.path.join(PLOT_DIR, f"h1_latent_tsne_l0_fold_{fold}.png"), param_label="L0")
    # Generate for T
    plot_manifold_pca(X, y_t, os.path.join(PLOT_DIR, f"h1_latent_pca_t_fold_{fold}.png"), f"Latent Space PCA (T)", param_label="T")
    plot_manifold_tsne(X, y_t, skills, skill_names, os.path.join(PLOT_DIR, f"h1_latent_tsne_t_fold_{fold}.png"), param_label="T")

    combined = {
        "fold": fold,
        "dataset": dataset_name,
        "h1_structural_encoding": {
            "l0": results_l0,
            "t": results_t
        }
    }

    # 4. Save results
    output_path = os.path.join(VALIDATION_DIR, f"structural_encoding_fold_{fold}.json")
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=4)
    
    print(f"\nCOMPLETED Structural Encoding Validation for Fold {fold}")
    print(f"Metrics saved to: {output_path}")
    # print(f"Parity plots saved to:")
    # print(f"  - {VALIDATION_DIR}/h1_structural_fidelity_l0_fold_{fold}.png")
    # print(f"  - {VALIDATION_DIR}/h1_structural_fidelity_t_fold_{fold}.png")
    print(f"Latent space visualizations saved to:")
    print(f"  - {PLOT_DIR}/h1_latent_pca_l0_fold_{fold}.png")
    print(f"  - {PLOT_DIR}/h1_latent_pca_t_fold_{fold}.png")
    print(f"  - {PLOT_DIR}/h1_latent_tsne_l0_fold_{fold}.png")
    print(f"  - {PLOT_DIR}/h1_latent_tsne_t_fold_{fold}.png")

if __name__ == "__main__":
    main()
