
import os
import sys
import argparse
import json
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pykt.models import init_model
from pykt.datasets import init_test_datasets

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

def load_model_from_dir(exp_dir, device):
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    mc = config.get('model_config', config.get('train_config', config.get('params', {})))
    dataset_name = config.get('dataset_name', config.get('params', {}).get('dataset_name', 'assist2009'))
    
    # Robust defaults
    defaults = {
        'kq_same': 1, 'separate_qa': 0, 'l2_rasch': 0.0, 
        'pretrain_dim': 768, 'ablation': 'none', 'n_uid': 0
    }
    for k, v in defaults.items():
        if k not in mc: mc[k] = v
        
    data_config_path = os.path.join(PROJECT_ROOT, "configs/data_config.json")
    with open(data_config_path, 'r') as f:
        dc = json.load(f)
    
    dpath = dc[dataset_name]['dpath'].replace("../", "")
    dpath = os.path.join(PROJECT_ROOT, dpath)
    dc[dataset_name]['dpath'] = dpath

    model = init_model('gtransformer', mc, dc[dataset_name], mc.get('emb_type', 'qid'))
    
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

    theory_path = os.path.join(dpath, "bkt_skill_params.pkl")
    bkt_params = None
    if os.path.exists(theory_path):
        with open(theory_path, "rb") as f:
            bkt_params = pickle.load(f)
        model.load_theory_params(bkt_params)
    
    return model, dc[dataset_name], mc, dpath, bkt_params

def collect_trajectories(model, loader, device, n_students=50):
    """
    Collects student trajectories for visualization.
    """
    all_trajs = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].bool().to(device)
            
            outputs, _ = model(c, r, pid_data=q)
            
            p_l0 = outputs['p_l0'].cpu().numpy()
            p_t = outputs['p_t'].cpu().numpy()
            preds = outputs['predictions'].cpu().numpy()
            
            for b in range(c.shape[0]):
                m = sm[b].cpu().numpy()
                if np.sum(m) < 10: continue # Only longer sequences
                
                traj = {
                    'c': c[b].cpu().numpy()[m],
                    'r': r[b].cpu().numpy()[m],
                    'p_l0': p_l0[b][m],
                    'p_t': p_t[b][m],
                    'preds': preds[b][m]
                }
                all_trajs.append(traj)
                if len(all_trajs) >= n_students:
                    return all_trajs
    return all_trajs

def plot_case_studies(trajs, output_path):
    """
    Plots a composite 2x2 figure of representative students.
    """
    # 4 archetypes selection loop
    # 1. Struggling: Low correctness, low mastery
    # 2. Fast Learner: Early fails, late pass, high p_t
    # 3. Advanced: High early correctness, high p_l0
    # 4. Consistent/Average
    
    def get_struggling(ts):
        # Low accuracy, low final mastery, long sequence
        candidates = [t for t in ts if np.mean(t['r']) < 0.3 and t['p_l0'][-1] < 0.6]
        if not candidates: return sorted(ts, key=lambda x: np.mean(x['r']))[0]
        return sorted(candidates, key=lambda x: len(x['r']), reverse=True)[0]

    def get_fast_learner(ts):
        # Low initial mastery, high final mastery, positive trend
        candidates = [t for t in ts if t['p_l0'][0] < 0.4 and t['p_l0'][-1] > 0.7]
        if not candidates: return sorted(ts, key=lambda x: x['p_l0'][-1] - x['p_l0'][0], reverse=True)[0]
        return sorted(candidates, key=lambda x: x['p_l0'][-1] - x['p_l0'][0], reverse=True)[0]

    def get_advanced(ts):
        # High initial mastery, high accuracy
        candidates = [t for t in ts if t['p_l0'][0] > 0.8 and np.mean(t['r']) > 0.8]
        if not candidates: return sorted(ts, key=lambda x: x['p_l0'][0], reverse=True)[0]
        return candidates[0]

    def get_steady(ts):
        # Gradual growth, moderate initial mastery
        candidates = [t for t in ts if 0.4 < t['p_l0'][0] < 0.7 and t['p_l0'][-1] > t['p_l0'][0]]
        if not candidates: return ts[len(ts)//2]
        return candidates[len(candidates)//2]

    cases = [
        (get_struggling(trajs), "Struggling Student (Low Mastery/Growth)"),
        (get_fast_learner(trajs), "Fast Learner (High Learning Rate)"),
        (get_advanced(trajs), "Advanced Student (High Initial Mastery)"),
        (get_steady(trajs), "Steady Progress Student")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (traj, title) in enumerate(cases):
        ax = axes[i]
        x = np.arange(len(traj['r']))
        
        # Plot Correctness (bars)
        ax.bar(x, traj['r'], alpha=0.2, color='green', label='Actual Response (1=Correct)')
        ax.set_ylim(-0.1, 1.1)
        
        # Plot Mastery and Prediction
        ax.plot(x, traj['preds'], label='P(Correct)', color='royalblue', linewidth=2, marker='.')
        ax.plot(x, traj['p_l0'], label='Inferred Mastery $P(L_0)$', color='darkorange', linestyle='--', alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Practice Step", fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved case studies to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="examples/validation/results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model, dc, mc, dpath, bkt_params = load_model_from_dir(args.exp_dir, device)
    
    # 2. Setup Dataloader (Standard PyKT init)
    # This handles padding and format alignment automatically
    print("Initializing test dataset...")
    model_name = mc.get('model_name', mc.get('model', 'gtransformer'))
    dataset_name = mc.get('dataset_name', mc.get('dataset', 'assist2009'))
    dc['dataset_name'] = dataset_name # init_test_datasets expects this
    test_loader, _, _, _ = init_test_datasets(dc, model_name, 64)
    loader = test_loader
    
    # 3. Collect Trajectories
    print("Collecting student trajectories...")
    trajs = collect_trajectories(model, loader, device, n_students=200)
    
    # 4. Plot
    plot_case_studies(trajs, os.path.join(args.output_dir, "case_study_composite.png"))

if __name__ == "__main__":
    main()
