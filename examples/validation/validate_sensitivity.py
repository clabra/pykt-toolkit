
import os
import sys
import argparse
import json
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

def load_model_from_dir(exp_dir, device):
    """
    Helper to load a GTransformer model from an experiment directory.
    """
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    mc = config.get('model_config', config.get('train_config', config.get('params', {})))
    dataset_name = config.get('dataset_name', config.get('params', {}).get('dataset_name', 'assist2009'))
    
    # Robust defaults for older configs
    defaults = {
        'kq_same': 1, 'separate_qa': 0, 'l2_rasch': 0.0, 
        'pretrain_dim': 768, 'ablation': 'all', 'n_uid': 0
    }
    for k, v in defaults.items():
        if k not in mc: mc[k] = v
        
    # Data config for model init (dimensions)
    data_config_path = os.path.join(PROJECT_ROOT, "configs/data_config.json")
    with open(data_config_path, 'r') as f:
        dc = json.load(f)
    
    dpath = dc[dataset_name]['dpath'].replace("../", "")
    dpath = os.path.join(PROJECT_ROOT, dpath)
    dc[dataset_name]['dpath'] = dpath

    model = init_model('gtransformer', mc, dc[dataset_name], mc.get('emb_type', 'qid'))
    
    # Checkpoint discovery
    checkpoint_path = None
    for root, _, files in os.walk(exp_dir):
        for f in files:
            if f.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, f)
                break
        if checkpoint_path: break
        
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in {exp_dir}")

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.to(device)
    model.eval()

    # Load Theory Params
    theory_path = os.path.join(dpath, "bkt_skill_params.pkl")
    if os.path.exists(theory_path):
        with open(theory_path, "rb") as f:
            bkt_params = pickle.load(f)
        model.load_theory_params(bkt_params)
    
    return model, dc[dataset_name], mc

def extract_causal_axis(model, loader, device, axis_name="L0"):
    """
    Identifies the pedagogical axis in latent space.
    If model has intrinsic probe, use its weights.
    If not (baseline), train a post-hoc probe on limited data.
    """
    if axis_name == "L0" and hasattr(model, 'probe_l0'):
        return model.probe_l0.weight.data.cpu().numpy().flatten()
    if axis_name == "T" and hasattr(model, 'probe_t'):
        return model.probe_t.weight.data.cpu().numpy().flatten()
    
    # Fallback: Post-hoc probe training (approximate axis)
    print(f"  Training post-hoc probe for {axis_name}...")
    zs, targets = [], []
    with torch.no_grad():
        for i, data in enumerate(loader):
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            q = data["qseqs"].long().to(device)
            sm = data["smasks"].bool().to(device)
            
            _, _, z = model(c, r, pid_data=q, qtest=True)
            t_key = 'target_l0' if axis_name == "L0" else 'target_t'
            t = data[t_key].to(device)
            
            zs.append(z[sm].cpu().numpy())
            targets.append(t[sm].cpu().numpy())
            if i > 20: break # Small sample sufficient for axis identification
            
    X = np.concatenate(zs, axis=0)
    y = np.concatenate(targets, axis=0)
    clf = Ridge(alpha=1.0).fit(X, y)
    return clf.coef_.flatten()

def run_causal_sweep(model, loader, device, axis_vec, axis_name="L0", n_batches=10):
    """
    Perturbs latent space along axis_vec and measures average change in P(correct).
    """
    model.eval()
    deltas = np.linspace(-3.0, 3.0, 13) # Detailed sweep
    results = {d: [] for d in deltas}
    
    w_axis = torch.tensor(axis_vec, dtype=torch.float32).to(device)
    w_norm = w_axis / torch.norm(w_axis)
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= n_batches: break
            
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            
            # Base context
            _, _, z_context = model(c, r, pid_data=q, qtest=True)
            
            for delta in deltas:
                # Intervene
                z_prime = z_context + (delta * w_norm)
                
                # Pass through output logic
                if not hasattr(model, 'knowledge_axis_emb'):
                    # Standard AKT Prediction
                    output = model.out(z_prime).squeeze(-1)
                    preds = torch.sigmoid(output)
                else: 
                    # Grounded BKT Logic
                    k_axis = model.knowledge_axis_emb(c)
                    v_axis = model.velocity_axis_emb(c)
                    l0_base = model.l0_base_emb(c).squeeze(-1)
                    t_base = model.t_base_emb(c).squeeze(-1)
                    
                    if axis_name == "L0":
                        l0_logits = l0_base + (z_prime * k_axis).sum(dim=-1)
                        t_logits = t_base + (z_context * v_axis).sum(dim=-1)
                    else: 
                        l0_logits = l0_base + (z_context * k_axis).sum(dim=-1)
                        t_logits = t_base + (z_prime * v_axis).sum(dim=-1)
                        
                    p_l0 = torch.sigmoid(l0_logits)
                    p_t = torch.sigmoid(t_logits)
                    preds = model._bkt_ref_output(c, r, p_l0, p_t)
                
                results[delta].append(preds.mean().item())
                
    avg_y = [np.mean(results[d]) for d in deltas]
    
    # Check monotonicity and flip if anti-correlated
    # We want +delta to mean +mastery/growth effect
    corr, _ = spearmanr(deltas, avg_y)
    if corr < 0:
        print(f"  [Sweep] Flipping axis for {axis_name} to align with prediction increase.")
        deltas = -deltas
        deltas = deltas[::-1]
        avg_y = avg_y[::-1]
        corr, _ = spearmanr(deltas, avg_y)

    # Center y around original state (delta=0)
    # delta=0 is at index len(deltas)//2
    idx0 = len(deltas)//2
    y0 = avg_y[idx0]
    rel_y = [y - y0 for y in avg_y]
    
    return deltas, avg_y, rel_y, corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded_exp", type=str, required=True)
    parser.add_argument("--baseline_exp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="examples/validation/results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Sensitivity Analysis ---")
    
    # 1. Load Models
    model_g, dc_g, mc_g = load_model_from_dir(args.grounded_exp, device)
    model_b, dc_b, mc_b = load_model_from_dir(args.baseline_exp, device)
    
    # 2. Setup Dataloader (use same test set)
    test_file = os.path.join(dc_g['dpath'], "train_valid_sequences.csv")
    target_path = os.path.join(dc_g['dpath'], "bkt_targets_train_valid.npz")
    dataset = GTransformerDataset(test_file, dc_g["input_type"], {0}, target_path=target_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 3. Analyze Mastery Axis (L0)
    print("\n[1/2] Analyzing Mastery Sensitivity (L0 Axis)...")
    w_g_l0 = extract_causal_axis(model_g, loader, device, "L0")
    w_b_l0 = extract_causal_axis(model_b, loader, device, "L0")
    
    d_g, y_g, r_g, c_g = run_causal_sweep(model_g, loader, device, w_g_l0, "L0")
    print(f"  Proposed Model Spearman ρ: {c_g:.4f}")
    
    d_b, y_b, r_b, c_b = run_causal_sweep(model_b, loader, device, w_b_l0, "L0")
    print(f"  Baseline Spearman ρ: {c_b:.4f}")
    
    # 4. Analyze Learning Rate Axis (T)
    print("\n[2/2] Analyzing Learning Rate Sensitivity (T Axis)...")
    w_g_t = extract_causal_axis(model_g, loader, device, "T")
    w_b_t = extract_causal_axis(model_b, loader, device, "T")
    
    d_g_t, y_g_t, r_g_t, c_g_t = run_causal_sweep(model_g, loader, device, w_g_t, "T")
    d_b_t, y_b_t, r_b_t, c_b_t = run_causal_sweep(model_b, loader, device, w_b_t, "T")

    # 5. Plot Comparison
    plt.figure(figsize=(12, 5))
    
    # Panel A: Mastery Sensitivity
    plt.subplot(1, 2, 1)
    plt.plot(d_g, r_g, marker='o', label=f'Proposed (ρ={c_g:.3f})', color='royalblue', linewidth=2)
    plt.plot(d_b, r_b, marker='s', label=f'Baseline (ρ={c_b:.3f})', color='grey', linestyle='--')
    plt.axhline(0, color='black', alpha=0.2)
    plt.axvline(0, color='black', alpha=0.2)
    plt.title("Mastery Intervention ($L_0$ Axis)", fontsize=13, fontweight='bold')
    plt.xlabel("Perturbation Size (Std Dev)", fontsize=11)
    plt.ylabel("Relative Change in P(Correct)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel B: Learning Rate Sensitivity
    plt.subplot(1, 2, 2)
    plt.plot(d_g_t, r_g_t, marker='o', label=f'Proposed (ρ={c_g_t:.3f})', color='darkorange', linewidth=2)
    plt.plot(d_b_t, r_b_t, marker='s', label=f'Baseline (ρ={c_b_t:.3f})', color='grey', linestyle='--')
    plt.axhline(0, color='black', alpha=0.2)
    plt.axvline(0, color='black', alpha=0.2)
    plt.title("Growth Intervention ($T$ Axis)", fontsize=13, fontweight='bold')
    plt.xlabel("Perturbation Size (Std Dev)", fontsize=11)
    plt.ylabel("Relative Change in P(Correct)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "sensitivity_curves.png"), dpi=300)
    print(f"\nSaved sensitivity curves to {args.output_dir}/sensitivity_curves.png")
    
    # Save metrics JSON
    metrics = {
        "l0": {"rho_proposed": float(c_g), "rho_baseline": float(c_b)},
        "t": {"rho_proposed": float(c_g_t), "rho_baseline": float(c_b_t)}
    }
    with open(os.path.join(args.output_dir, "sensitivity_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
