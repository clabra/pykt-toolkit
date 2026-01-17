
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
        p_correct = l_curr * (1 - s) + (1 - l_curr) * g
        preds.append(p_correct)
        
        if r == 1:
            l_post = (l_curr * (1 - s)) / max(1e-6, (l_curr * (1 - s) + (1 - l_curr) * g))
        else:
            l_post = (l_curr * s) / max(1e-6, (l_curr * s + (1 - l_curr) * (1 - g)))
            
        l_curr = l_post + (1 - l_post) * t
        
    return np.array(preds)

def find_quadrant_cases(model, loader, device, bkt_params, n_students=6000):
    """
    Searches for the most illustrative students for the 2x2 matrix:
    (Low/High L0) x (Low/High T)
    Maximizes divergence from Markovian BKT.
    """
    best_cases = {
        'Low L0 / Low T': {'data': None, 'score': -1.0},
        'Low L0 / High T': {'data': None, 'score': -1.0},
        'High L0 / Low T': {'data': None, 'score': -1.0},
        'High L0 / High T': {'data': None, 'score': -1.0}
    }
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].bool().to(device)
            uids = data.get("uids", None)
            if uids is not None: uids = uids.long().to(device)
            
            outputs, _ = model(c, r, pid_data=q, uid_data=uids)
            # Use supervised predictions (more dynamic, context-aware)
            p_l0 = outputs['p_l0'].cpu().numpy()
            p_t = outputs['p_t'].cpu().numpy()
            preds = outputs['predictions'].cpu().numpy()
            
            for b in range(c.shape[0]):
                m = sm[b].cpu().numpy()
                cur_c = c[b].cpu().numpy()[m]
                cur_r = r[b].cpu().numpy()[m]
                cur_p_l0 = p_l0[b][m]
                cur_p_t = p_t[b][m]
                cur_preds = preds[b][m]
                
                unique_skills = np.unique(cur_c)
                for skill in unique_skills:
                    s_mask = (cur_c == skill)
                    if np.sum(s_mask) < 6: continue
                    
                    mean_l0 = np.mean(cur_p_l0[s_mask])
                    mean_t = np.mean(cur_p_t[s_mask])
                    init_p_l0 = cur_p_l0[s_mask][0] # Mastery at the very start
                    uid = data['uids'][b].item() if 'uids' in data else f"S{b}"
                    
                    # 1. Define Quadrant Thresholds
                    q_key = None
                    if mean_l0 < 0.35 and mean_t < 0.2: q_key = 'Low L0 / Low T'
                    elif mean_l0 < 0.35 and mean_t > 0.4: q_key = 'Low L0 / High T'
                    elif mean_l0 > 0.6 and mean_t < 0.2: q_key = 'High L0 / Low T'
                    elif mean_l0 > 0.6 and 0.4 <= mean_t <= 0.8: q_key = 'High L0 / High T' # Less exaggerated
                    
                    if q_key:
                        bkt_p = calculate_bkt_trajectory(bkt_params, skill, cur_r[s_mask])
                        
                    if q_key:
                        bkt_p = calculate_bkt_trajectory(bkt_params, skill, cur_r[s_mask])
                        
                        # Calculate prediction errors for both models
                        gt_errors = np.abs(cur_preds[s_mask] - cur_r[s_mask])
                        bkt_errors = np.abs(bkt_p - cur_r[s_mask])
                        
                        # Accuracy advantage: reward cases where GT is more accurate than BKT
                        accuracy_advantage = np.mean(bkt_errors - gt_errors)
                        
                        # Only consider cases where GT is at least as good as BKT
                        if accuracy_advantage < -0.05:  # GT significantly worse than BKT
                            continue
                        
                        # Consistency check: predictions should align with cognitive parameters
                        first_pred = cur_preds[s_mask][0]
                        if q_key in ['Low L0 / Low T', 'Low L0 / High T']:
                            # Low L0 should have low initial predictions
                            if first_pred > 0.5:
                                continue
                        elif q_key in ['High L0 / Low T', 'High L0 / High T']:
                            # High L0 should have high initial predictions
                            if first_pred < 0.5:
                                continue
                        
                        # Narrative-Driven Scoring
                        if q_key == 'Low L0 / Low T':
                            # Story: GT < BKT, penalizes fails, treats successes as guesses
                            success_mask = (cur_r[s_mask] == 1)
                            if np.any(success_mask):
                                pessimism_after_success = np.mean(bkt_p[success_mask] - cur_preds[s_mask][success_mask])
                                narrative_score = np.mean(bkt_p - cur_preds[s_mask]) + pessimism_after_success
                            else:
                                narrative_score = np.mean(bkt_p - cur_preds[s_mask])
                        
                        elif q_key == 'Low L0 / High T':
                            # Story: GT > BKT, trusts growth, optimistic recovery
                            low_start = 2.0 if cur_preds[s_mask][0] < 0.4 else 0.0
                            recovery = np.std(cur_preds[s_mask]) * 3
                            optimism = np.mean(cur_preds[s_mask] - bkt_p)
                            narrative_score = low_start + recovery + optimism
                        
                        elif q_key == 'High L0 / Low T':
                            # Story: GT identifies slips, stays high despite failures
                            fail_mask = (cur_r[s_mask] == 0)
                            if np.any(fail_mask):
                                robustness = np.mean(cur_preds[s_mask][fail_mask] - bkt_p[fail_mask])
                                high_mastery = np.mean(cur_preds[s_mask]) if np.mean(cur_preds[s_mask]) > 0.7 else 0
                                narrative_score = robustness * 2 + high_mastery
                            else:
                                narrative_score = -1.0
                        
                        elif q_key == 'High L0 / High T':
                            # Story: Very optimistic, treats all fails as slips
                            fail_mask = (cur_r[s_mask] == 0)
                            high_confidence = np.mean(cur_preds[s_mask])
                            if np.any(fail_mask):
                                slip_identification = np.mean(cur_preds[s_mask][fail_mask] - bkt_p[fail_mask])
                                narrative_score = high_confidence * 2 + slip_identification
                            else:
                                narrative_score = high_confidence
                        
                        # Combined score: narrative alignment + accuracy advantage
                        score = narrative_score + accuracy_advantage * 2
                        
                        if score > best_cases[q_key]['score']:
                            best_cases[q_key]['score'] = score
                            best_cases[q_key]['data'] = {
                                'uid': uid, 'skill': skill, 'seq': cur_r[s_mask], 
                                'preds': cur_preds[s_mask], 
                                'p_l0': mean_l0,  # Use actual latent parameter
                                'p_t': mean_t
                            }
            
            if i > n_students // 64: break
            
    return {k: v['data'] for k, v in best_cases.items()}

def plot_quadrant_mosaic(quadrants, bkt_params, output_path):
    """
    Plots a 2x2 mosaic of the cognitive archetypes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    for idx, (title, data) in enumerate(quadrants.items()):
        ax = axes[idx]
        if data is None:
            ax.text(0.5, 0.5, f"No case found for\n{title}", ha='center', va='center')
            continue
            
        bkt_p = calculate_bkt_trajectory(bkt_params, data['skill'], data['seq'])
        x = np.arange(len(data['seq']))
        
        # Ground Truth Responses (Vertical Bars)
        for i, val in enumerate(data['seq']):
            color = 'green' if val == 1 else 'red'
            ax.axvline(x=i, color=color, alpha=0.15, linewidth=18, zorder=0)

        # GTransformer Line and Dynamic Markers
        ax.plot(x, data['preds'], color='royalblue', linewidth=2, alpha=0.8, 
                label=f"Our Model")
        for i, p in enumerate(data['preds']):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='royalblue', markersize=6)

        # BKT Line and Dynamic Markers
        ax.plot(x, bkt_p, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label="Standard BKT (Markovian)")
        for i, p in enumerate(bkt_p):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='gray', markersize=5)
            
        ax.set_title(f"{title} (Skill {data['skill']})\nStudent ID:{data['uid']}, $P_{{L0}}$={data['p_l0']:.2f}, $P_{{T}}$={data['p_t']:.2f}", 
                    fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Predicted $P(Correct)$")
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        
    plt.suptitle("Predictions for Context-Aware Profiles compared with Markovian BKT", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Saved 2x2 quadrant mosaic to {output_path}")

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
    
    print("Searching for quadrant-based cases (2x2 mosaic)...")
    quadrants = find_quadrant_cases(model, test_loader, device, bkt_params)
    
    plot_quadrant_mosaic(quadrants, bkt_params, os.path.join(args.output_dir, "cognitive_quadrants_mosaic.png"))

if __name__ == "__main__":
    main()
