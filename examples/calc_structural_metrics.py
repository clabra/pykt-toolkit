
import os
import sys
import argparse
import json
import torch
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def extract_latent_and_oracle(model, loader, device, active_grounding=False):
    """
    Extracts latent vector z, Oracle parameters (Targets), and Probe Weights.
    """
    model.eval()
    all_z = []
    all_l0_target = []
    all_t_target = []
    
    # Store probe weights if active
    probe_w_l0 = None
    probe_w_t = None
    
    if active_grounding and hasattr(model, 'probe_l0'):
        probe_w_l0 = model.probe_l0.weight.data.cpu().numpy().flatten()
        probe_w_t = model.probe_t.weight.data.cpu().numpy().flatten()
        
    print(f"Extracting Latent Representations and Oracle Targets...")
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            
            # Forward pass with qtest=True to get z_context
            # Note: gtransformer returns (outputs, loss, z_context)
            _, _, z_context = model(c, r, pid_data=q, qtest=True)
            
            # Extract targets (Oracle BKT)
            target_l0 = data["target_l0"].to(device)
            target_t = data["target_t"].to(device)
            
            # Mask and Flatten
            mask = sm.bool()
            
            z_flat = z_context[mask] # [N, z_dim]
            l0_flat = target_l0[mask] # [N]
            t_flat = target_t[mask] # [N]
            
            all_z.append(z_flat.cpu().numpy())
            all_l0_target.append(l0_flat.cpu().numpy())
            all_t_target.append(t_flat.cpu().numpy())
            
            if i > 50: # Limit sample size for speed (e.g. ~10k samples)
                break
                
    X = np.concatenate(all_z, axis=0)
    y_l0 = np.concatenate(all_l0_target, axis=0)
    y_t = np.concatenate(all_t_target, axis=0)
    
    return X, y_l0, y_t, probe_w_l0, probe_w_t

def compute_structural_alignment(X, y_oracle, probe_w_intrinsic=None):
    """
    Computes S_align: Cosine Similarity between PC1 and Probe Vector.
    If probe_w_intrinsic is None (Baseline), we train a Ridge probe first.
    """
    # 1. PCA Analysis
    pca = PCA(n_components=10)
    pca.fit(X)
    
    # PC1 Vector (Data Axis)
    v_pc1 = pca.components_[0]
    
    # Explained Variance
    explained_variance = pca.explained_variance_ratio_
    effective_rank_90 = np.argmax(np.cumsum(explained_variance) >= 0.90) + 1
    
    # 2. Probe Identification
    if probe_w_intrinsic is not None:
        # Active Grounding: Use internal probe
        w_probe = probe_w_intrinsic
        probe_type = "Intrinsic"
    else:
        # Baseline: Train post-hoc Linear Probe
        # Ridge Regression to be robust
        clf = Ridge(alpha=1.0)
        clf.fit(X, y_oracle)
        w_probe = clf.coef_
        probe_type = "Post-Hoc"
        
    # 3. Alignment Calculation
    # Normalize vectors
    v_pc1_norm = v_pc1 / np.linalg.norm(v_pc1)
    w_probe_norm = w_probe / np.linalg.norm(w_probe)
    
    # Absolute Cosine Similarity (Orientation doesn't matter, only axis alignment)
    s_align = np.abs(np.dot(v_pc1_norm, w_probe_norm))
    
    return {
        "s_align": s_align,
        "effective_rank_90": effective_rank_90,
        "explained_variance": explained_variance,
        "probe_type": probe_type,
        "v_pc1": v_pc1_norm,
        "w_probe": w_probe_norm
    }

def compute_causal_sensitivity(model, loader, device, w_probe, axis_name="L0"):
    """
    Performs Causal Intervention: z' = z + delta * w_probe
    Checks if Output Probability increases monotonically.
    """
    print(f"Running Causal Sensitivity Analysis for {axis_name}...")
    model.eval()
    
    # Define perturbation range (in Standard Deviations of the latent space)
    deltas = np.linspace(-3.0, 3.0, 7) # -3, -2, -1, 0, 1, 2, 3
    
    # We will average the predicted probability across a batch for each delta
    # ideally keeping everything else constant.
    
    results = {d: [] for d in deltas}
    
    # Use a small fixed batch for sensitivity to ensure stability
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i > 5: break # Only need a few batches to establish the causal law
            
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            # sm = data["smasks"].long().to(device) # Not strictly needed for sensitivity average
            
            # 1. Get Base Latent
            # We need to hook into the model to inject z.
            # gtransformer structure makes this tricky without modifying forward().
            # Luckily, we can use the `qtest=True` return to get z, but we can't easily re-inject it 
            # into `model.out` because `out` is just a Sequential layer, but `_bkt_ref_output` needs parameters.
            
            # Re-implementation of the model head logic for sensitivity
            # Step 1: Get z_context
            _, _, z_context = model(c, r, pid_data=q, qtest=True) # [BS, Seq, Z_dim]
            
            # Step 2: Perturb and Predict Loop
            w_probe_tensor = torch.tensor(w_probe, dtype=torch.float32).to(device)
            w_probe_norm = w_probe_tensor / torch.norm(w_probe_tensor)
            
            for delta in deltas:
                # Intervention: z' = z + delta * w
                z_prime = z_context + (delta * w_probe_norm)
                
                # Step 3: Pass through Output Layers
                if not hasattr(model, 'knowledge_axis_emb'):
                     # Baseline (Ungrounded) Path: Direct Prediction
                     # model.out expects [BS, Seq, Z_dim]
                     output = model.out(z_prime).squeeze(-1)
                     ref_preds = torch.sigmoid(output)
                else: 
                    # Grounded Path: Projection Heads -> BKT Logic
                    # Retrieve embeddings required for projection
                    k_axis = model.knowledge_axis_emb(c)
                    v_axis = model.velocity_axis_emb(c)
                    l0_base = model.l0_base_emb(c).squeeze(-1)
                    t_base = model.t_base_emb(c).squeeze(-1)
                    
                    # Re-compute Parameters using perturbed z
                    if axis_name == "L0":
                        # Perturbing L0 axis primarily affects L0
                        l0_logits = l0_base + (z_prime * k_axis).sum(dim=-1)
                        t_logits = t_base + (z_context * v_axis).sum(dim=-1)
                    else: 
                        l0_logits = l0_base + (z_context * k_axis).sum(dim=-1)
                        t_logits = t_base + (z_prime * v_axis).sum(dim=-1)
                        
                    p_l0 = torch.sigmoid(l0_logits)
                    p_t = torch.sigmoid(t_logits)
                    
                    # Step 4: Pass through BKT Logic
                    ref_preds = model._bkt_ref_output(c, r, p_l0, p_t)
                
                # Metric: Average Predicted Probability of Correctness
                # We only care about the *change*, so average is a fine proxy.
                avg_prob = ref_preds.mean().item()
                results[delta].append(avg_prob)

    # Aggregate
    avg_results = {k: np.mean(v) for k, v in results.items()}
    
    # Calculate Monotonicity (Spearman Correlation between Delta and Output)
    x = list(avg_results.keys())
    y = list(avg_results.values())
    corr, _ = spearmanr(x, y)
    
    return avg_results, corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment folder")
    parser.add_argument("--output_file", type=str, default="structural_metrics.pkl", help="Output pickle file")
    args = parser.parse_args()
    
    print(f"--- Structural Validation: {args.exp_dir} ---")
    
    # 1. Load Config & Model
    config_path = os.path.join(args.exp_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model_config = config.get('model_config', config.get('train_config', config.get('params', {})))
    dataset_name = config.get('dataset_name', config.get('params', {}).get('dataset_name', 'assist2009'))
    fold = model_config.get('fold', 0)
    
    # Checkpoint
    checkpoint_path = None
    for root, _, files in os.walk(args.exp_dir):
        for f in files:
            if f.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, f)
                break
        if checkpoint_path: break
        
    if not checkpoint_path:
        print("No checkpoint found.")
        return

    # Check if Active Grounding
    active_grounding = model_config.get('active_grounding', 0) == 1
    print(f"Configuration: Active Grounding = {active_grounding}")
    
    # Load Data Config
    data_config_path = os.path.join(PROJECT_ROOT, "configs/data_config.json")
    with open(data_config_path, 'r') as f:
        dc = json.load(f)
    
    # Fix paths
    dpath = dc[dataset_name]['dpath'].replace("../", "")
    dpath = os.path.join(PROJECT_ROOT, dpath)
    dc[dataset_name]['dpath'] = dpath
    
    # Sanitize config for GTransformer
    defaults = {
        'kq_same': 1,
        'separate_qa': 0,
        'l2_rasch': 0.0,
        'pretrain_dim': 768,
        'ablation': 'all', # Default to 'all' if not specified (likely baseline)
        'n_uid': 0
    }
    for k, v in defaults.items():
        if k not in model_config:
            model_config[k] = v
            
    # 2. Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model('gtransformer', model_config, dc[dataset_name], model_config.get('emb_type', 'qid'))
    
    # Load Weights
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.to(device)
    
    # Load Theory Params (for projection layers)
    with open(os.path.join(dpath, "bkt_skill_params.pkl"), "rb") as f:
        bkt_params = pickle.load(f)
    model.load_theory_params(bkt_params)
    
    # 3. Init Dataloader
    test_file = os.path.join(dpath, "train_valid_sequences.csv")
    target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
    dataset = GTransformerDataset(test_file, dc[dataset_name]["input_type"], {fold}, target_path=target_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 4. Extract Data
    X, y_l0, y_t, w_l0, w_t = extract_latent_and_oracle(model, loader, device, active_grounding)
    
    # 5. Compute Alignment Metrics
    print("\n[1/2] Computing Structural Alignment...")
    res_l0 = compute_structural_alignment(X, y_l0, w_l0)
    print(f" > L0 Alignment (S_align): {res_l0['s_align']:.4f}")
    print(f" > Effective Rank (90%): {res_l0['effective_rank_90']}")
    
    res_t = compute_structural_alignment(X, y_t, w_t)
    print(f" > T Alignment (S_align): {res_t['s_align']:.4f}")
    
    # 6. Compute Sensitivity Metrics
    print("\n[2/2] Computing Causal Sensitivity...")
    # Use the discovered probe (either intrinsic or post-hoc)
    sens_l0, corr_l0 = compute_causal_sensitivity(model, loader, device, res_l0['w_probe'], "L0")
    print(f" > L0 Causal Correlation: {corr_l0:.4f}")
    
    results_payload = {
        "exp_name": args.exp_dir.split("/")[-1],
        "active_grounding": active_grounding,
        "alignment_l0": res_l0,
        "alignment_t": res_t,
        "sensitivity_l0": sens_l0,
        "corr_l0": corr_l0
    }
    
    with open(args.output_file, "wb") as f:
        pickle.dump(results_payload, f)
    print(f"\nSaved full metrics to {args.output_file}")

if __name__ == "__main__":
    main()
