
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

from examples.validation.generate_case_studies import load_model_from_dir

def calculate_bkt_trajectory(skill_params, skill_id, sequence):
    """
    Calculates BKT predictions for a given skill and response sequence.
    """
    params_dict = skill_params.get('params', {})
    if skill_id not in params_dict:
        return np.full(len(sequence), 0.5)
        
    p = params_dict[skill_id]
    l0, t, s, g = p.get('pl0', 0.5), p.get('pt', 0.1), p.get('ps', 0.1), p.get('pg', 0.1)
    
    # BKT Formula: P(Correct) = L * (1-S) + (1-L) * G
    preds = []
    l_curr = l0
    
    for r in sequence:
        # Prediction
        p_correct = l_curr * (1 - s) + (1 - l_curr) * g
        preds.append(p_correct)
        
        # Bayes Update
        if r == 1:
            l_post = (l_curr * (1 - s)) / (l_curr * (1 - s) + (1 - l_curr) * g)
        else:
            l_post = (l_curr * s) / (l_curr * s + (1 - l_curr) * (1 - g))
            
        # Growth
        l_curr = l_post + (1 - l_post) * t
        
    return np.array(preds)

def find_all_twins(model, loader, device, bkt_params, max_pairs=30, n_students=3000):
    """
    Finds multiple twin pairs across all skills and ranks them by 'impressiveness'.
    """
    skill_trajs = {} 
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].bool().to(device)
            uids = data.get("uids", None)
            if uids is not None:
                uids = uids.long().to(device)
            
            outputs, _ = model(c, r, pid_data=q, uid_data=uids)
            p_l0 = outputs['p_l0'].cpu().numpy()
            p_t_out = outputs['p_t'].cpu().numpy()
            preds = outputs['predictions'].cpu().numpy()
            
            for b in range(c.shape[0]):
                m = sm[b].cpu().numpy()
                cur_c = c[b].cpu().numpy()[m]
                cur_r = r[b].cpu().numpy()[m]
                cur_p_l0 = p_l0[b][m]
                cur_p_t = p_t_out[b][m]
                cur_preds = preds[b][m]
                
                unique_skills = np.unique(cur_c)
                for skill in unique_skills:
                    skill_mask = (cur_c == skill)
                    if np.sum(skill_mask) < 6: continue 
                    
                    skill_r_tuple = tuple(cur_r[skill_mask].tolist())
                    
                    # Heuristic: Start with Failures
                    if skill_r_tuple[0] != 0 or skill_r_tuple[1] != 0: continue
                        
                    skill_p_l0 = cur_p_l0[skill_mask]
                    skill_p_t = cur_p_t[skill_mask] # Extract T
                    skill_preds = cur_preds[skill_mask]
                    
                    if skill not in skill_trajs:
                        skill_trajs[skill] = {}
                    
                    if skill_r_tuple not in skill_trajs[skill]:
                        skill_trajs[skill][skill_r_tuple] = []
                    
                    skill_trajs[skill][skill_r_tuple].append({
                        'uid': data['uids'][b].item() if 'uids' in data else f"S{b}",
                        'p_l0': skill_p_l0,
                        'p_t': skill_p_t,
                        'preds': skill_preds,
                        'r': skill_r_tuple,
                        'skill': skill,
                    })
            
            if i > n_students // 64: break

    all_pairs = []
    for skill, sequences in skill_trajs.items():
        for seq, trajs in sequences.items():
            if len(trajs) < 2: continue
            
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    # Similarity at start
                    init_diff = np.abs(trajs[i]['preds'][0] - trajs[j]['preds'][0])
                    # Divergence at any point
                    max_div = np.max(np.abs(trajs[i]['preds'] - trajs[j]['preds']))
                    l0_diff = np.abs(np.mean(trajs[i]['p_l0']) - np.mean(trajs[j]['p_l0']))
                    t_diff = np.abs(np.mean(trajs[i]['p_t']) - np.mean(trajs[j]['p_t']))
                    
                    # Increased thresholds to find more 'Extreme' cases
                    # We want high internal param difference AND high late divergence
                    if init_diff < 0.2 and (l0_diff > 0.15 or t_diff > 0.15):
                        bkt_t = calculate_bkt_trajectory(bkt_params, skill, seq)
                        bkt_var = np.std(bkt_t)
                        
                        # Multiplicative score to favor cases where ALL dimensions diverge
                        # We want: Similar start, High L0 diff, High T diff, High Prediction Div
                        score = (max_div * (1 + l0_diff) * (1 + t_diff) + bkt_var) / (init_diff + 0.1)
                        all_pairs.append({
                            'pair': (trajs[i], trajs[j], skill, seq),
                            'score': score
                        })
                        
    all_pairs = sorted(all_pairs, key=lambda x: x['score'], reverse=True)
    return [p['pair'] for p in all_pairs[:max_pairs]]

def plot_twin_mosaic(twin_pairs, bkt_params, output_path):
    """
    Plots a 3x3 mosaic of twin divergence cases.
    """
    n = min(len(twin_pairs), 9)
    if n == 0:
        print("Error: No twin pairs found to plot.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx in range(9):
        ax = axes[idx]
        if idx >= n:
            ax.axis('off')
            continue
            
        student_a, student_b, skill, seq = twin_pairs[idx]
        bkt_preds = calculate_bkt_trajectory(bkt_params, skill, seq)
        x = np.arange(len(seq))
        
        # Labels with UID and Cognitive Parameters
        # Using mean p_l0 and p_t for the sequence to characterize the student
        label_a = f"ID:{student_a['uid']} | L0:{np.mean(student_a['p_l0']):.2f} T:{np.mean(student_a['p_t']):.2f}"
        label_b = f"ID:{student_b['uid']} | L0:{np.mean(student_b['p_l0']):.2f} T:{np.mean(student_b['p_t']):.2f}"

        # Ground Truth Responses (Vertical Bars)
        for i, val in enumerate(seq):
            color = 'green' if val == 1 else 'red'
            ax.axvline(x=i, color=color, alpha=0.15, linewidth=18, zorder=0)

        # GTransformer Lines and Dynamic Markers
        ax.plot(x, student_a['preds'], color='royalblue', linewidth=1.5, alpha=0.7, label=label_a)
        for i, p in enumerate(student_a['preds']):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='royalblue', markersize=5)

        ax.plot(x, student_b['preds'], color='crimson', linewidth=1.5, alpha=0.7, label=label_b)
        for i, p in enumerate(student_b['preds']):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='crimson', markersize=5)

        # BKT Line
        ax.plot(x, bkt_preds, color='gray', linestyle=':', linewidth=1.2, alpha=0.6, label='BKT Baseline')
        for i, p in enumerate(bkt_preds):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='gray', markersize=4)
            
        ax.set_title(f"Skill {skill} | Seq: {list(seq)[:4]}...", fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.8)

    plt.suptitle("Interpretability Mosaic: 3x3 Twin Sequence Divergence Analysis\n(Same local response sequence, divergent global curriculum context)", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Saved 3x3 mosaic to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="examples/validation/results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, dc, mc, dpath, bkt_params = load_model_from_dir(args.exp_dir, device)
    
    print("Initializing test dataset...")
    model_name = mc.get('model_name', mc.get('model', 'gtransformer'))
    dataset_name = mc.get('dataset_name', mc.get('dataset', 'assist2009'))
    dc['dataset_name'] = dataset_name 
    test_loader, _, _, _ = init_test_datasets(dc, model_name, 64)
    
    print(f"Searching for diverse twin sequences (3x3 mosaic)...")
    twin_pairs = find_all_twins(model, test_loader, device, bkt_params, max_pairs=50, n_students=5000)
    
    if twin_pairs:
        plot_twin_mosaic(twin_pairs, bkt_params, os.path.join(args.output_dir, "twin_divergence_mosaic.png"))
    else:
        print("Error: Could not find any twin sequences with diverging mastery.")

if __name__ == "__main__":
    main()
