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

Usage:
    python examples/train_probe.py \
        --checkpoint experiments/.../best_model.pt \
        --bkt_preds data/assist2015/traj_predictions_bkt.csv \
        --dataset assist2015 \
        --fold 0 \
        --output_dir experiments/.../probing
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
    Runs model on loader, extracts embeddings, and aligns with BKT targets.
    """
    model.eval()
    
    embeddings_list = []
    targets_list = []
    
    # 1. Load BKT Predictions (Full Sequences)
    # BKT CSV format: student_id, skill_id, y_true, p_idkt, y_idkt, p_bkt, ...
    print("Indexing BKT predictions...")
    bkt_lookup = {}
    grouped = bkt_df.groupby('student_id')
    for uid, group in grouped:
        bkt_lookup[uid] = group['p_bkt'].values
    print(f"Loaded BKT paths for {len(bkt_lookup)} students.")

    # 2. Build Index -> Raw UID Mapping via keyid2idx.json
    idx_to_uid = {}
    try:
        # Infer path from loader or data_config
        # loader.dataset.dpath usually points to the processed data folder
        if hasattr(loader.dataset, 'dpath'):
            dpath = loader.dataset.dpath
        else:
            # Fallback: assume typical pykt structure relative to script/args
            # We don't have easy access to data_config here inside the function unless passed
            # But we can try to guess from dataset name if passed?
            # Better: pass 'dataset_name' to this function.
            pass
            
        # Try to find keyid2idx.json in the dataset folder
        # We can reach into the dataset object to find the directory
        if hasattr(loader.dataset, 'input_type'):
             # Standard KTDataset
             # Checking common locations
             # The loader doesn't store the full path easily accessible sometimes
             pass
    except Exception:
        pass

    # Actually, let's load it from the known path if possible or rely on the internal dict if loaded
    # The previous attempt relied on 'dori' which might be subsetted.
    # Let's load the JSON directly if we can find it.
    
    mapping_loaded = False
    
    # Try getting it from the internal dataset dictionary first (fastest)
    try:
        ds = loader.dataset
        if hasattr(ds, 'dataset'): ds = ds.dataset # Handle Subset
        if hasattr(ds, 'dori') and 'uid_to_index' in ds.dori:
            print("Using dori.uid_to_index for mapping...")
            uid_to_index = ds.dori['uid_to_index']
            # keys are original IDs (str), values are indices (int)
            idx_to_uid = {v: k for k, v in uid_to_index.items()}
            mapping_loaded = True
    except Exception as e:
        print(f"Dori mapping failed: {e}")
        
    if not mapping_loaded:
        # Try loading file from likely location
        try:
            # We assume the code is running in project root
            # We need the dataset name. We'll pass it in or infer.
            # This function signature change is risky.
            # Let's try to infer from BKT predictions if they match indices
            pass
        except:
            pass
            
    if idx_to_uid:
        print(f"Built Index->Original ID mapping for {len(idx_to_uid)} students.")
        # Check sample
        first_idx = next(iter(idx_to_uid))
        print(f"Sample: Index {first_idx} -> ID {idx_to_uid[first_idx]}")
        
        # Check type of Original ID in BKT Lookup
        # CSV reading usually infers types (int for 590)
        # JSON keys are always strings ("590")
        # We need to handle this type mismatch!
        
        sample_bkt_key = next(iter(bkt_lookup))
        print(f"Sample BKT Key: {sample_bkt_key} (Type: {type(sample_bkt_key)})")
        
    # 3. Track Consumption Offsets per Student (for Windowing)
    offsets = {} # uid -> int (number of steps consumed)

    steps_processed = 0
    students_matched = 0
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            # Unpack inputs
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            t = data["tseqs"].long().to(device) if "tseqs" in data else None
            m = data["masks"].long().to(device)
            sm = data["smasks"].long().to(device)
            
            # Extract UIDs (batch indices)
            if "uids" in data:
                uids_batch = data["uids"].cpu().numpy().flatten()
            else:
                raise ValueError("DataLoader must return UIDs.")

            # Forward pass (concept-based input for iDKT with pid_data=q)
            # Returns: preds, initmastery, rate, c_reg_loss, concat_q, reg_losses
            preds, _, _, _, concat_q, _ = model(c, r, pid_data=q, qtest=True, uid_data=data["uids"].to(device))
            
            bs, seq_len, dim = concat_q.shape
            
            concat_q_np = concat_q.cpu().numpy()
            mask_np = sm.cpu().numpy() # [BS, SeqLen]
            
            for b_idx in range(bs):
                uid_idx = int(uids_batch[b_idx])
                
                # Convert Index -> Raw UID
                if uid_idx in idx_to_uid:
                    raw_uid = idx_to_uid[uid_idx]
                    
                    # Robust Lookup: Handle String vs Int mismatch between JSON (str) and CSV (int)
                    if raw_uid not in bkt_lookup:
                        # Try int conversion
                        try:
                            raw_uid_int = int(raw_uid)
                            if raw_uid_int in bkt_lookup:
                                raw_uid = raw_uid_int
                        except:
                            pass
                            
                        # Try str conversion
                        if raw_uid not in bkt_lookup:
                            raw_uid_str = str(raw_uid)
                            if raw_uid_str in bkt_lookup:
                                raw_uid = raw_uid_str
                else:
                    raw_uid = uid_idx # Fallback
                
                if raw_uid not in bkt_lookup:
                    continue 
                    
                bkt_full_seq = bkt_lookup[raw_uid]
                
                # Identify valid steps in this window
                valid_indices = np.where(mask_np[b_idx] == 1)[0]
                n_valid = len(valid_indices)
                
                if n_valid == 0:
                    continue
                    
                # State checking
                current_offset = offsets.get(raw_uid, 0)
                end_offset = current_offset + n_valid
                
                # Check if we go out of bounds of BKT data
                if end_offset > len(bkt_full_seq):
                    # Truncate? This implies BKT CSV has fewer steps than Loader
                    # Or we just take what's available
                    available = len(bkt_full_seq) - current_offset
                    if available <= 0:
                        continue 
                    n_valid = available
                    valid_indices = valid_indices[:n_valid]
                    end_offset = current_offset + n_valid
                
                # Extract Aligned Data
                # Model Embeddings
                student_embeddings = concat_q_np[b_idx][valid_indices]
                
                # Target Values (Windowed Slice)
                target_window = bkt_full_seq[current_offset : end_offset]
                
                # Verify shapes
                if len(student_embeddings) != len(target_window):
                    print(f"Shape mismatch error for uid {raw_uid}")
                    continue
                    
                embeddings_list.append(student_embeddings)
                targets_list.append(target_window)
                
                # Update State
                offsets[raw_uid] = end_offset
                steps_processed += n_valid
                students_matched += 1
            
            if steps_processed > 20000 and "debug" in sys.argv:
                break
                
    if not embeddings_list:
        print("No embeddings extracted.")
        return None, None
        
    print(f"Extraction complete. Consumed {steps_processed} steps across {students_matched} windows.")
    
    X = np.vstack(embeddings_list)
    y = np.concatenate(targets_list)
    
    return X, y

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
    # We need the loader that corresponds to the BKT predictions.
    # Usually we validate on the VALIDATION split.
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
    X, y = extract_embeddings_and_targets(model, valid_loader, bkt_df, device)
    
    if X is None:
        print("Error: No aligned data found. Check BKT file vs Dataset UIDs.")
        return
        
    print(f"Data Ready: X shape {X.shape}, y shape {y.shape}")
    
    # 6. Run Experiment
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
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
