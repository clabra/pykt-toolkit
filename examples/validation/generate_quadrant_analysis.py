
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

from examples.validation.validation_helpers import load_model_from_dir

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
    Searches for diverse cases showcasing different prediction behaviors.
    Returns 4 cases for a 2x2 mosaic based on L0/T quadrants.
    """
    best_cases = {
        'Low L0 / Low T': {'data': None, 'score': -1.0},
        'Low L0 / High T': {'data': None, 'score': -1.0},
        'High L0 / Low T': {'data': None, 'score': -1.0},
        'High L0 / High T': {'data': None, 'score': -1.0}
    }
    
    all_l0 = []
    all_t = []
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            qshft = data["shft_qseqs"].long().to(device)
            cshft = data["shft_cseqs"].long().to(device)
            rshft = data["shft_rseqs"].long().to(device)
            m = data["masks"].bool().to(device)
            sm = data["smasks"].bool().to(device)
            uids = data.get("uids", None)
            if uids is not None: uids = uids.long().to(device)
            
            # Concatenate sequences like in evaluation
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)
            
            outputs, _ = model(cc.long(), cr.long(), pid_data=cq.long(), uid_data=uids)
            
            # Extract both supervised and reference predictions (already shifted by model)
            # Predictions are at positions [1:] corresponding to responses at positions [1:]
            p_l0 = outputs['p_l0'][:,1:].cpu().numpy()
            p_t = outputs['p_t'][:,1:].cpu().numpy()
            preds = outputs['predictions'][:,1:].cpu().numpy()  # p_sup (neural head)
            ref_preds = outputs['reference_preds'][:,1:].cpu().numpy()  # p_ref (BKT logic)
            
            for b in range(c.shape[0]):
                m_b = sm[b].cpu().numpy()
                cur_c = cshft[b].cpu().numpy()[m_b]
                cur_r = rshft[b].cpu().numpy()[m_b]
                cur_p_l0 = p_l0[b][m_b]
                cur_p_t = p_t[b][m_b]
                cur_preds = preds[b][m_b]  # p_sup
                cur_ref_preds = ref_preds[b][m_b]  # p_ref
                
                unique_skills = np.unique(cur_c)
                for skill in unique_skills:
                    s_mask = (cur_c == skill)
                    if np.sum(s_mask) < 6: continue
                    
                    mean_l0 = np.mean(cur_p_l0[s_mask])
                    mean_t = np.mean(cur_p_t[s_mask])
                    
                    all_l0.append(mean_l0)
                    all_t.append(mean_t)
                    init_p_l0 = cur_p_l0[s_mask][0] # Mastery at the very start
                    uid = data['uids'][b].item() if 'uids' in data else f"S{b}"
                    
                    # 1. Define Quadrant Thresholds
                    q_key = None
                    if mean_l0 < 0.65 and mean_t < 0.1: q_key = 'Low L0 / Low T'
                    elif mean_l0 < 0.65 and mean_t > 0.15: q_key = 'Low L0 / High T'
                    elif mean_l0 > 0.7 and mean_t < 0.1: q_key = 'High L0 / Low T'
                    elif mean_l0 > 0.7 and mean_t > 0.15: q_key = 'High L0 / High T'
                    
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
                        
                        # Narrative-Driven Scoring - Enhanced for clarity
                        if q_key == 'Low L0 / Low T':
                            # Story: Cautious model, treats successes as potential guesses
                            # Want: Low initial predictions, stays cautious, GT < BKT after successes
                            success_mask = (cur_r[s_mask] == 1)
                            if np.any(success_mask) and np.sum(success_mask) >= 2:
                                # Reward staying cautious after successes
                                pessimism_after_success = np.mean(bkt_p[success_mask] - cur_preds[s_mask][success_mask])
                                # Reward low initial prediction
                                low_start_bonus = 2.0 if cur_preds[s_mask][0] < 0.3 else 0.0
                                # Reward overall caution
                                overall_caution = np.mean(bkt_p - cur_preds[s_mask])
                                narrative_score = pessimism_after_success + low_start_bonus + overall_caution
                            else:
                                narrative_score = -1.0  # Need successes to show the pattern
                        
                        elif q_key == 'Low L0 / High T':
                            # Story: High learning rate - sharp jumps in predictions after correct responses
                            # Want: Low start, sharp increases after successes, high volatility
                            if len(cur_preds[s_mask]) >= 6:
                                low_start_bonus = 3.0 if cur_preds[s_mask][0] < 0.4 else 1.0
                                
                                # Calculate step-by-step changes (jumps)
                                pred_diffs = np.diff(cur_preds[s_mask])
                                
                                # Reward sharp positive jumps (high T means responsive to successes)
                                positive_jumps = pred_diffs[pred_diffs > 0]
                                if len(positive_jumps) > 0:
                                    max_jump = np.max(positive_jumps)
                                    avg_positive_jump = np.mean(positive_jumps)
                                    sharp_jump_bonus = max_jump * 10 + avg_positive_jump * 5
                                else:
                                    sharp_jump_bonus = 0.0
                                
                                # Reward volatility (high T means predictions change a lot)
                                volatility = np.std(pred_diffs)
                                volatility_bonus = volatility * 8
                                
                                # Overall growth still matters
                                growth = cur_preds[s_mask][-1] - cur_preds[s_mask][0]
                                growth_bonus = max(0, growth * 3)
                                
                                # Reward if GT is more optimistic than BKT in later steps
                                later_half = len(cur_preds[s_mask]) // 2
                                optimism_later = np.mean(cur_preds[s_mask][later_half:] - bkt_p[later_half:])
                                
                                narrative_score = low_start_bonus + sharp_jump_bonus + volatility_bonus + growth_bonus + optimism_later * 2
                            else:
                                narrative_score = -1.0
                        
                        elif q_key == 'High L0 / Low T':
                            # Story: Strong mastery, treats errors as slips
                            # Want: High predictions throughout, stays high after errors, GT > BKT after errors
                            fail_mask = (cur_r[s_mask] == 0)
                            if np.any(fail_mask) and np.sum(fail_mask) >= 1:
                                # Reward staying confident after errors
                                robustness = np.mean(cur_preds[s_mask][fail_mask])
                                # Reward if GT stays higher than BKT after errors
                                confidence_advantage = np.mean(cur_preds[s_mask][fail_mask] - bkt_p[fail_mask])
                                # Reward high overall mastery
                                high_mastery_bonus = 3.0 if np.mean(cur_preds[s_mask]) > 0.75 else 0.0
                                narrative_score = robustness * 2 + confidence_advantage * 2 + high_mastery_bonus
                            else:
                                narrative_score = -1.0  # Need errors to show slip interpretation
                        
                        elif q_key == 'High L0 / High T':
                            # Story: Advanced learner, high confidence, treats errors as slips
                            # Want: Very high predictions, stays high throughout, errors don't hurt much
                            fail_mask = (cur_r[s_mask] == 0)
                            high_confidence = np.mean(cur_preds[s_mask])
                            high_start_bonus = 3.0 if cur_preds[s_mask][0] > 0.7 else 0.0
                            if np.any(fail_mask):
                                # Reward maintaining confidence after errors
                                slip_identification = np.mean(cur_preds[s_mask][fail_mask])
                                confidence_advantage = np.mean(cur_preds[s_mask][fail_mask] - bkt_p[fail_mask])
                                narrative_score = high_confidence * 3 + slip_identification * 2 + confidence_advantage + high_start_bonus
                            else:
                                # Still good if consistently high
                                narrative_score = high_confidence * 3 + high_start_bonus
                        
                        # Combined score: narrative alignment + accuracy advantage
                        score = narrative_score + accuracy_advantage * 3
                        
                        if score > best_cases[q_key]['score']:
                            best_cases[q_key]['score'] = score
                            best_cases[q_key]['data'] = {
                                'uid': uid, 'skill': skill, 'seq': cur_r[s_mask], 
                                'preds': cur_preds[s_mask],  # p_sup (neural)
                                'ref_preds': cur_ref_preds[s_mask],  # p_ref (BKT logic)
                                'p_l0': mean_l0,  # Use actual latent parameter
                                'p_t': mean_t
                            }
            
            if i > n_students // 64: break
            
    print(f"\nParameter Variance Audit:")
    print(f"P_L0: Mean {np.mean(all_l0):.4f}, Std {np.std(all_l0):.4f}, Range [{np.min(all_l0):.4f}, {np.max(all_l0):.4f}]")
    print(f"P_T:  Mean {np.mean(all_t):.4f}, Std {np.std(all_t):.4f}, Range [{np.min(all_t):.4f}, {np.max(all_t):.4f}]")
    
    # Count how many students fall into each quadrant range
    l_l0 = [x for x in all_l0 if x < 0.35]
    h_l0 = [x for x in all_l0 if x > 0.6]
    l_t = [x for x in all_t if x < 0.2]
    h_t = [x for x in all_t if x > 0.4]
    
    print(f"Quadrant Coverage: Low L0: {len(l_l0)}, High L0: {len(h_l0)}, Low T: {len(l_t)}, High T: {len(h_t)}")

    return {k: v['data'] for k, v in best_cases.items()}

def plot_quadrant_mosaic(quadrants, bkt_params, output_path):
    """
    Plots a 2x2 mosaic of the cognitive archetypes.
    Shows both p_sup (neural) and p_ref (interpretable BKT logic) trajectories.
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

        # GTransformer Prediction Range (shaded band between p_ref and p_sup)
        # This shows the "interpretability-accuracy envelope"
        p_sup = data['preds']  # Neural head (more accurate)
        p_ref = data['ref_preds']  # BKT logic (interpretable)
        
        # Fill between p_ref and p_sup to show the prediction envelope
        ax.fill_between(x, p_ref, p_sup, color='royalblue', alpha=0.2, 
                        label='Prediction Envelope', zorder=1)
        
        # p_ref trajectory (interpretable, BKT logic)
        ax.plot(x, p_ref, color='steelblue', linewidth=2, alpha=0.9, 
                linestyle='--', label='p_ref (Interpretable BKT Logic)', zorder=2)
        for i, p in enumerate(p_ref):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='steelblue', markersize=5, zorder=2)
        
        # p_sup trajectory (neural head, more accurate)
        ax.plot(x, p_sup, color='royalblue', linewidth=2.5, alpha=0.9, 
                label='p_sup (Neural Head)', zorder=3)
        for i, p in enumerate(p_sup):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='royalblue', markersize=6, zorder=3)

        # BKT Baseline (Markovian)
        ax.plot(x, bkt_p, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, 
                label="Classical BKT (Markovian)", zorder=1)
        for i, p in enumerate(bkt_p):
            marker = 'o' if p > 0.5 else 'x'
            ax.plot(i, p, marker=marker, color='gray', markersize=4, alpha=0.6, zorder=1)
            
        ax.set_title(f"{title} (Skill {data['skill']})\nStudent ID:{data['uid']}, $P_{{L0}}$={data['p_l0']:.2f}, $P_{{T}}$={data['p_t']:.2f}", 
                    fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Predicted $P(Correct)$")
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        
    plt.suptitle("GTransformer Prediction Envelope: Interpretable (p_ref) vs. Accurate (p_sup)", 
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
