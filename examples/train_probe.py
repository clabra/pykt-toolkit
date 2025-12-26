#!/usr/bin/env python3
"""
Diagnostic Probing Script for iDKT

This script implements the "Diagnostic Probing with Control Tasks" methodology
(Hewitt & Liang, 2019; Belinkov, 2022).

Goal: Validate that iDKT embeddings structurally encode BKT parameters (Mastery).

Methodology:
1. Extraction: Pass validation data through frozen iDKT -> Get embeddings H (concat_q).
2. Alignment: Match H with pre-computed BKT Mastery predictions (p_bkt).
3. Selectivity Analysis:
   - Task A (True): Train Linear Probe H -> p_bkt. Measure R^2_true.
   - Task B (Control): Train Linear Probe H -> shuffled(p_bkt). Measure R^2_control.
   - Metric: Selectivity = R^2_true - R^2_control.

4. Dataset Specifics (ASSIST2009):
   - Format: "One Question -> Multiple Concepts".
   - pyKT Behavior: Trains on expanded Concept sequences (length T_kc) but evaluates on original Question level (length T_q).
   - Probing Alignment: We implement a "Robust QID-Alignment" that matches the expanded concept embeddings H (one per KC) 
     with the corresponding single BKT target p_bkt (one per Question) by inferring repeats from Question ID continuity.

Usage:
    python examples/train_probe.py \
        --checkpoint experiments/.../best_model.pt \
        --bkt_preds data/assist2015/traj_predictions_bkt.csv \
        --dataset assist2015 \
        --fold 0 \
        --output_dir experiments/.../probing

Note that data/[DATASET]/keyid2idx.json is a bidirectional mapping dictionary that converts between original dataset IDs and zero-based 
sequential indices used internally by the pykt framework
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.models import init_model
from pykt.datasets import init_dataset4train
from pykt.utils import set_seed
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_correlation(y_true, y_pred, output_path, title="Probing Correlation"):
    """Generates a scatter plot of True vs Predicted Mastery."""
    plt.figure(figsize=(8, 8))
    # Downsample for scatter plot if too large
    if len(y_true) > 5000:
        idx = np.random.choice(len(y_true), 5000, replace=False)
        y_t = y_true[idx]
        y_p = y_pred[idx]
    else:
        y_t, y_p = y_true, y_pred
        
    plt.scatter(y_t, y_p, alpha=0.1, s=10, c='blue')
    
    # Plot diagonal ideal
    lims = [0, 1]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('True BKT Mastery')
    plt.ylabel('Probed Prediction (iDKT)')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved correlation plot to {output_path}")

def plot_manifold(X, y, output_path, title="Latent Space PCA"):
    """Projects embeddings to 2D and colors by Mastery."""
    print("Computing PCA...")
    # Downsample for PCA if too large
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub, y_sub = X, y
        
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sub, cmap='viridis', alpha=0.5, s=15)
    plt.colorbar(sc, label='BKT Mastery')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    plt.title(title)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved manifold plot to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diagnostic Probes for iDKT")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to iDKT checkpoint")
    parser.add_argument("--bkt_preds", type=str, required=True, help="Path to BKT predictions CSV (traj_predictions.csv)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    
    # Architecture params (must match training)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--emb_type", type=str, default='qid')
    parser.add_argument("--final_fc_dim", type=int, default=512)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--lambda_student", type=float, default=1e-5)
    parser.add_argument("--lambda_gap", type=float, default=1e-5)
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Run on small subset")
    
    return parser.parse_args()

def extract_embeddings_and_targets(model, loader, bkt_df, device):
    """
    Runs model on loader, extracts embeddings, and aligns with BKT targets using robust signatures.
    Returns: (X, y, skills)
    """
    model.eval()
    
    embeddings_list = []
    targets_list = []
    skills_list = []
    
    # Build Index -> Raw UID Mapping for current loader
    idx_to_uid = {}
    try:
        ds = loader.dataset
        if hasattr(ds, 'dataset'): ds = ds.dataset # Handle Subset
        if hasattr(ds, 'dori') and 'uid_to_index' in ds.dori:
            uid_to_index = ds.dori['uid_to_index']
            idx_to_uid = {v: k for k, v in uid_to_index.items()}
            print(f"Loaded student ID mapping for {len(idx_to_uid)} students.")
    except Exception as e:
        print(f"Dataset UID mapping failed: {e}")

    # 1. Index BKT Predictions with Signatures
    print("Indexing BKT predictions with signatures...")
    bkt_indexed = {} # raw_uid -> list of (sig, p_bkt)
    grouped = bkt_df.groupby('student_id')
    for uid, group in grouped:
        sigs = []
        for _, row in group.iterrows():
            # Signature: (concept_id, y_true, round_p_idkt)
            sig = (int(row['skill_id']), int(row['y_true']), round(float(row['p_idkt']), 6))
            sigs.append((sig, float(row['p_bkt'])))
        bkt_indexed[uid] = sigs
    print(f"Loaded BKT paths for {len(bkt_indexed)} students.")

    # 2. Process Loader with Signature Matching
    steps_processed = 0
    resync_events = 0
    total_samples = 0
    match_count = 0
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            uids_batch = data["uids"].cpu().numpy().flatten()
            
            # Prepare full sequences for model matching (must match eval_idkt_interpretability)
            cq_full = torch.cat((q[:, 0:1], data["shft_qseqs"].long().to(device)), dim=1)
            cc_full = torch.cat((c[:, 0:1], data["shft_cseqs"].long().to(device)), dim=1)
            cr_full = torch.cat((r[:, 0:1], data["shft_rseqs"].long().to(device)), dim=1)

            # Forward pass
            # y[i] is prediction for interaction i (which is cc_full[i])
            # So y[1+t] is prediction for cshft[t]
            preds, _, _, _, concat_q, _ = model(cc_full, cr_full, pid_data=cq_full, qtest=True, uid_data=data["uids"].to(device))
            
            bs, seq_len, dim = concat_q.shape
            concat_q_np = concat_q.cpu().numpy()
            preds_np = preds.cpu().numpy()
            mask_np = sm.cpu().numpy() 
            # Extraction for signatures
            cshft_np = data["shft_cseqs"].cpu().numpy()
            rshft_np = data["shft_rseqs"].cpu().numpy()

            for b_idx in range(bs):
                uid_idx = int(uids_batch[b_idx])
                raw_uid = idx_to_uid.get(uid_idx, uid_idx)
                
                if raw_uid not in bkt_indexed:
                    continue
                
                student_sigs = bkt_indexed[raw_uid]
                valid_indices = np.where(mask_np[b_idx] == 1)[0]
                
                for t in valid_indices:
                    # Model signature for interaction at step t in shft arrays
                    skill_id = int(cshft_np[b_idx, t])
                    y_true = int(rshft_np[b_idx, t])
                    p_idkt = round(float(preds_np[b_idx, 1+t]), 6) # Correctly shifted
                    model_sig = (skill_id, y_true, p_idkt)
                    
                    found = False
                    for b_sig, p_bkt in student_sigs:
                        if model_sig == b_sig:
                            # Success: Use the correctly shifted latent state and prediction
                            embeddings_list.append(concat_q_np[b_idx, 1+t])
                            targets_list.append(p_bkt)
                            skills_list.append(skill_id)
                            found = True
                            match_count += 1
                            break
                    
                    if not found:
                        resync_events += 1
                        if resync_events <= 3:
                            print(f"  [DEBUG] No match for student {raw_uid}, sig {model_sig}")
                            print(f"  [DEBUG] Sample BKT sigs: {student_sigs[:2]}")
                    
                    total_samples += 1
                
                steps_processed += len(valid_indices)
            
            if steps_processed > 20000 and "debug" in sys.argv:
                break
                
    if not embeddings_list:
        print("No embeddings matched.")
        return None, None
        
    fidelity = match_count / max(1, total_samples)
    print(f"Extraction complete. Matched {match_count}/{total_samples} samples ({fidelity:.1%} fidelity).")
    
    X = np.vstack(embeddings_list)
    y = np.array(targets_list)
    skills = np.array(skills_list)
    
    return X, y, skills

def run_probing_experiment(X, y, seed, output_dir=None):
    """
    Trains True and Control probes. Returns metrics.
    """
    # Split Train/Test (80/20)
    indices = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # --- Task A: True Probe ---
    print("Training True Probe (Target: BKT Mastery)...")
    probe_true = LinearRegression()
    probe_true.fit(X_train, y_train)
    y_pred_true = probe_true.predict(X_test)
    
    r2_true = r2_score(y_test, y_pred_true)
    pearson_true, _ = pearsonr(y_test, y_pred_true)
    
    if output_dir:
        # Visualize True Probe
        try:
            plot_correlation(y_test, y_pred_true, 
                            os.path.join(output_dir, 'probe_scatter_true.png'),
                            title=f"True Alignment (R^2={r2_true:.2f})")
            # Visualize Manifold
            plot_manifold(X_test, y_test, 
                         os.path.join(output_dir, 'probe_pca.png'),
                         title="iDKT Latent Space (Colored by Mastery)")
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    # --- Task B: Control Probe (Selectivity) ---
    print("Training Control Probe (Target: Shuffled labels)...")
    y_train_shuffled = np.random.permutation(y_train)
    # Control target randomization must be consistent on test too for "random function"
    # But usually we just shuffle train labels and test on shuffled labels?
    # No, simple shuffle of everything is safer.
    y_control_full = np.random.permutation(y)
    y_train_c, y_test_c = y_control_full[train_idx], y_control_full[test_idx]
    
    probe_control = LinearRegression()
    probe_control.fit(X_train, y_train_c)
    y_pred_control = probe_control.predict(X_test)
    
    r2_control = r2_score(y_test_c, y_pred_control)
    pearson_control, _ = pearsonr(y_test_c, y_pred_control)
    
    return {
        "r2_true": r2_true,
        "pearson_true": pearson_true,
        "r2_control": r2_control,
        "pearson_control": pearson_control,
        "selectivity_r2": r2_true - r2_control
    }

def run_per_skill_probing(X, y, skills, seed):
    """
    Computes probing metrics for EACH skill individually.
    """
    unique_skills = np.unique(skills)
    per_skill_results = {}
    
    print(f"Running per-skill probing for {len(unique_skills)} skills...")
    
    for sid in unique_skills:
        skill_mask = (skills == sid)
        if np.sum(skill_mask) < 20: # Min samples for a meaningful probe
            continue
            
        X_s = X[skill_mask]
        y_s = y[skill_mask]
        
        # Split 80/20
        indices = np.arange(len(X_s))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(X_s))
        if split_idx == 0: continue
        
        X_train, X_test = X_s[indices[:split_idx]], X_s[indices[split_idx:]]
        y_train, y_test = y_s[indices[:split_idx]], y_s[indices[split_idx:]]
        
        if len(y_test) < 5 or len(np.unique(y_train)) < 2:
             continue

        probe = LinearRegression()
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        
        try:
            r2 = r2_score(y_test, y_pred)
            corr, _ = pearsonr(y_test, y_pred)
            per_skill_results[int(sid)] = {
                "r2": float(r2),
                "pearson": float(corr),
                "count": int(np.sum(skill_mask))
            }
        except:
            continue
            
    return per_skill_results

def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data Config
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
        
    # Fix paths
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name]:
            dpath = data_config[dataset_name]['dpath']
            # Robustly handle relative paths starting with ../
            if dpath.startswith("../"):
                dpath = dpath.replace("../", "")
            
            # Ensure we use absolute path joined with project root
            if not os.path.isabs(dpath):
                data_config[dataset_name]['dpath'] = os.path.abspath(os.path.join(project_root, dpath))

    # 2. Load Dataset (Validation Set)
    print(f"Loading validation set for {args.dataset}, fold {args.fold}...")
    
    # Ensure we use augmented BKT file for consistency with interpretability outputs
    orig_file = data_config[args.dataset]['train_valid_file']
    bkt_file = orig_file.replace('.csv', '_bkt.csv')
    if os.path.exists(os.path.join(data_config[args.dataset]['dpath'], bkt_file)):
        data_config[args.dataset]['train_valid_file'] = bkt_file
        print(f"  Using augmented training file: {bkt_file}")

    train_loader, valid_loader = init_dataset4train(
        args.dataset, 'idkt', data_config, args.fold, args.batch_size)
    
    # 3. Load Model
    print(f"Loading iDKT model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint.get('model_config', None)
    
    # Fallback if config not in checkpoint (older saves)
    if model_config is None:
        state_dict = checkpoint
        # Reconstruct config from args
        model_config = vars(args)
    else:
        state_dict = checkpoint['model_state_dict']
    
    # Ensure n_uid detection from checkpoint if valid
    if 'student_param.weight' in state_dict:
        model_config['n_uid'] = state_dict['student_param.weight'].shape[0] - 1
    
    # Filter model_config to only include valid iDKT args to avoid TypeError
    valid_idkt_keys = {
        'd_model', 'd_ff', 'num_attn_heads', 'n_blocks', 'dropout', 'emb_type', 
        'final_fc_dim', 'l2', 'lambda_student', 'lambda_gap', 'n_uid', 'kq_same', 
        'separate_qa', 'emb_path', 'pretrain_dim', 'n_pid'
    }
    # Also handle some generic args that might be in config but mapped differently?
    # Actually, init_model does: iDKT(num_c, num_q, **model_config).
    # so we just need to keep the kwargs that iDKT accepts.
    filtered_config = {k: v for k, v in model_config.items() if k in valid_idkt_keys}
    
    model = init_model('idkt', filtered_config, data_config[args.dataset], args.emb_type)
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 4. Load BKT Predictions
    print(f"Loading BKT predictions from {args.bkt_preds}...")
    bkt_df = pd.read_csv(args.bkt_preds)
    
    # 5. Extract & Align
    print("Starting Extraction Phase...")
    X, y, skills = extract_embeddings_and_targets(model, valid_loader, bkt_df, device)
    
    if X is None:
        print("Error: No aligned data found. Check BKT file vs Dataset UIDs.")
        return
        
    print(f"Data Ready: X shape {X.shape}, y shape {y.shape}")
    
    # 6. Run Experiment
    results = run_probing_experiment(X, y, args.seed, output_dir=args.output_dir)
    
    print("\n=== Probing Results ===")
    print(f"True Task R^2:    {results['r2_true']:.4f}")
    print(f"Control Task R^2: {results['r2_control']:.4f}")
    print(f"Selectivity:      {results['selectivity_r2']:.4f}")
    
    if results['selectivity_r2'] > 0.1:
        print("\nSUCCESS: Strong evidence of structural encoding (High Selectivity).")
    else:
        print("\nWARNING: Low selectivity. Model may differ structurally or probe is too weak/strong.")
        
    # 7. Save
    out_path = os.path.join(args.output_dir, 'probe_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved global results to {out_path}")
    
    # 8. Per-Skill Probing
    per_skill_results = run_per_skill_probing(X, y, skills, args.seed)
    ps_path = os.path.join(args.output_dir, 'probe_per_skill_results.json')
    with open(ps_path, 'w') as f:
        json.dump(per_skill_results, f, indent=2)
    print(f"Saved per-skill results to {ps_path}")

if __name__ == "__main__":
    main()
