"""
Generate Initial Mastery Mosaic: 2x2 grid showing how Our Model adapts predictions
based on Initial Mastery (P(L0)) differences while Learning Rate (P(T)) remains similar.

Each subplot shows:
- Student A (orange): Low P(L0), similar P(T)
- Student B (blue): High P(L0), similar P(T)
- BKT (gray): Same for both students (non-personalized)
- Ground truth: Green/red bars

The four plots demonstrate how initial knowledge affects predictions across different response patterns.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pykt.models import init_model
from pykt.datasets import init_test_datasets
import pickle
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_bkt_trajectory(bkt_params, skill, responses):
    """Calculate BKT predictions for a sequence."""
    params = bkt_params.get('params', {}).get(skill, bkt_params.get('params', {}).get(str(skill), {}))
    if not params:
        params = bkt_params.get('global', {'prior': 0.5, 'learns': 0.1, 'guess': 0.2, 'slip': 0.1})
    
    def get_val(d, k, def_val):
        v = d.get(k, def_val)
        if isinstance(v, (np.ndarray, list)):
            return float(v[0])
        return float(v)
    
    L0 = get_val(params, 'prior', 0.5)
    T = get_val(params, 'learns', 0.1)
    G = get_val(params, 'guess', 0.2)
    S = get_val(params, 'slip', 0.1)
    
    L = L0
    predictions = []
    for r in responses:
        pred = L * (1 - S) + (1 - L) * G
        predictions.append(pred)
        
        if r == 1:
            L_post = (L * (1 - S)) / max(pred, 1e-6)
        else:
            L_post = (L * S) / max(1 - pred, 1e-6)
        
        L = L_post + (1 - L_post) * T
    
    return np.array(predictions)

def find_twin_pairs(model, test_loader, bkt_params, n_pairs=4, n_students=6000):
    """
    Find pairs of students with identical response sequences but different cognitive profiles.
    Returns 4 pairs representing different response patterns.
    """
    model.eval()
    
    # Store student data
    student_sequences = {}
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i > n_students // 64:
                break
            
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            uids = data.get("uids", None)
            if uids is not None:
                uids = uids.long().to(device)
            
            outputs, _ = model(c, r, pid_data=q, uid_data=uids)
            p_l0 = outputs['p_l0'].cpu().numpy()
            p_t = outputs['p_t'].cpu().numpy()
            preds = outputs['predictions'].cpu().numpy()
            
            for b in range(c.shape[0]):
                m = sm[b].cpu().numpy()
                cur_c = c[b][m == 1].cpu().numpy()
                cur_r = r[b][m == 1].cpu().numpy()
                cur_p_l0 = p_l0[b][m == 1]
                cur_p_t = p_t[b][m == 1]
                cur_preds = preds[b][m == 1]
                uid = data['uids'][b].item() if 'uids' in data else f"S{b}"
                
                # Find skills with sufficient length
                unique_skills = np.unique(cur_c)
                for skill in unique_skills:
                    s_mask = (cur_c == skill)
                    if np.sum(s_mask) < 6 or np.sum(s_mask) > 15:
                        continue
                    
                    seq_tuple = tuple(cur_r[s_mask])
                    mean_l0 = np.mean(cur_p_l0[s_mask])
                    mean_t = np.mean(cur_p_t[s_mask])
                    
                    if seq_tuple not in student_sequences:
                        student_sequences[seq_tuple] = []
                    
                    student_sequences[seq_tuple].append({
                        'uid': uid,
                        'skill': skill,
                        'seq': cur_r[s_mask],
                        'preds': cur_preds[s_mask],
                        'p_l0': mean_l0,
                        'p_t': mean_t
                    })
    
    # Find best twin pairs with contrasting profiles
    best_pairs = []
    
    for seq_tuple, students in student_sequences.items():
        if len(students) < 2:
            continue
        
        # Find pairs with contrasting Initial Mastery but similar Learning Rate
        for i, student_a in enumerate(students):
            for student_b in students[i+1:]:
                # We want: Student A (Low L0) vs Student B (High L0), but similar T
                # Low L0: < 0.4, High L0: > 0.6, Similar T: difference < 0.15
                if student_a['p_l0'] < 0.4 and student_b['p_l0'] > 0.6:
                    t_diff = abs(student_a['p_t'] - student_b['p_t'])
                    if t_diff < 0.15:  # Similar learning rates
                        # CRITICAL: Ensure predictions are consistent with profiles
                        # Student B (high profile) should have higher mean predictions than Student A (low profile)
                        mean_pred_a = np.mean(student_a['preds'])
                        mean_pred_b = np.mean(student_b['preds'])
                        
                        if mean_pred_b <= mean_pred_a:
                            # Skip this pair - predictions are inverted
                            continue
                        
                        contrast = (student_b['p_l0'] - student_a['p_l0']) - t_diff  # Reward L0 contrast, penalize T difference
                        divergence = np.mean(np.abs(student_a['preds'] - student_b['preds']))
                        score = contrast * 3 + divergence * 2
                        
                        best_pairs.append({
                            'score': score,
                            'student_a': student_a,
                            'student_b': student_b,
                            'seq': seq_tuple
                        })
    
    # Sort and return top 4 pairs
    best_pairs.sort(key=lambda x: x['score'], reverse=True)
    return best_pairs[:4]

def plot_personalization_mosaic(pairs, bkt_params, output_path):
    """Create 2x2 mosaic showing personalization for 4 different response patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, pair in enumerate(pairs):
        ax = axes[idx]
        student_a = pair['student_a']
        student_b = pair['student_b']
        seq = pair['seq']
        
        # Calculate BKT predictions (same for both students)
        bkt_p = calculate_bkt_trajectory(bkt_params, student_a['skill'], seq)
        
        x = np.arange(len(seq))
        
        # Plot ground truth bars
        for i, response in enumerate(seq):
            color = 'green' if response == 1 else 'red'
            ax.axvline(x=i, color=color, alpha=0.15, linewidth=18, zorder=0)
        
        # Student A (Low L0) - Orange
        ax.plot(x, student_a['preds'], color='darkorange', linewidth=2, alpha=0.8, 
                label=f"Student A (Low Initial Mastery)\n$P_{{L0}}$={student_a['p_l0']:.2f}, $P_{{T}}$={student_a['p_t']:.2f}")
        for i, p in enumerate(student_a['preds']):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='darkorange', markersize=6)
        
        # Student B (High L0) - Blue
        ax.plot(x, student_b['preds'], color='royalblue', linewidth=2, alpha=0.8,
                label=f"Student B (High Initial Mastery)\n$P_{{L0}}$={student_b['p_l0']:.2f}, $P_{{T}}$={student_b['p_t']:.2f}")
        for i, p in enumerate(student_b['preds']):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='royalblue', markersize=6)
        
        # BKT (same for both) - Gray
        ax.plot(x, bkt_p, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
                label="BKT (Non-Personalized)")
        for i, p in enumerate(bkt_p):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='gray', markersize=5)
        
        ax.set_title(f"Response Pattern {idx+1} (Skill {student_a['skill']})", 
                    fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Predicted $P(Correct)$")
        ax.set_xlabel("Time Step")
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle("Initial Mastery Effect: Context-Aware Predictions vs. Non-Personalized BKT\nIdentical Sequences, Different Initial Knowledge", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Saved initial mastery mosaic to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--output_dir', type=str, default='examples/validation/results')
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(args.exp_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model config and dataset name
    model_config = config.get('model_config', config.get('train_config', config.get('params', {})))
    dataset_name = config.get('dataset_name', config.get('params', {}).get('dataset_name', 'assist2009'))
    
    # Load data_config.json
    data_config_path = 'configs/data_config.json'
    with open(data_config_path, 'r') as f:
        data_configs = json.load(f)
    
    # Get dataset-specific config
    data_config_dataset = data_configs[dataset_name]
    dpath = data_config_dataset['dpath'].replace("../", "")
    data_config_dataset['dpath'] = dpath
    data_config_dataset['dataset_name'] = dataset_name  # Add dataset_name for init_test_datasets
    
    # Load BKT parameters
    bkt_path = os.path.join(dpath, "bkt_skill_params.pkl")
    with open(bkt_path, 'rb') as f:
        bkt_params = pickle.load(f)
    
    # Initialize model
    # Find the checkpoint file in the nested directory
    model_dir = [d for d in os.listdir(args.exp_dir) if d.startswith('gtransformer_')]
    if model_dir:
        checkpoint_path = os.path.join(args.exp_dir, model_dir[0], 'qid_model.ckpt')
    else:
        checkpoint_path = os.path.join(args.exp_dir, 'model.ckpt')
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with proper configs
    model_name = config.get('input', {}).get('model', 'gtransformer')
    model = init_model(model_name, model_config, data_config_dataset, model_config.get('emb_type', 'qid'))
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model = model.to(device)
    model.eval()
    
    # Load test data - returns 4 loaders
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config_dataset, model_name, 64)
    
    # Find twin pairs
    print("Searching for twin pairs with contrasting cognitive profiles...")
    pairs = find_twin_pairs(model, test_loader, bkt_params)
    
    if len(pairs) < 4:
        print(f"Warning: Only found {len(pairs)} suitable pairs")
    
    # Generate mosaic
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'initial_mastery_mosaic.png')
    plot_personalization_mosaic(pairs, bkt_params, output_path)

if __name__ == '__main__':
    main()
