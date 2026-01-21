
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

def find_diverse_cases(model, loader, device, bkt_params, n_cases=9):
    """
    Find diverse student-skill cases that illustrate different envelope behaviors:
    1. Narrow envelope (high agreement)
    2. Wide envelope (high disagreement)
    3. p_sup > p_ref (neural more optimistic)
    4. p_ref > p_sup (interpretable more optimistic)
    5. Dynamic p_sup, flat p_ref
    6. Both dynamic
    7. Both converge over time
    8. Diverge over time
    9. Mixed patterns
    """
    
    case_criteria = {
        'Narrow Envelope\n(High Agreement)': {'score': -1.0, 'data': None, 'metric': 'min_envelope'},
        'Wide Envelope\n(High Disagreement)': {'score': -1.0, 'data': None, 'metric': 'max_envelope'},
        'p_sup Optimistic\n(Supervised > Interpretable)': {'score': -1.0, 'data': None, 'metric': 'psup_higher'},
        'p_ref Optimistic\n(Interpretable > Supervised)': {'score': -1.0, 'data': None, 'metric': 'pref_higher'},
        'Dynamic p_sup\nStable p_ref': {'score': -1.0, 'data': None, 'metric': 'psup_volatile'},
        'Both Dynamic\n(Covarying)': {'score': -1.0, 'data': None, 'metric': 'both_dynamic'},
        'Converging\nEnvelope': {'score': -1.0, 'data': None, 'metric': 'converging'},
        'Diverging\nEnvelope': {'score': -1.0, 'data': None, 'metric': 'diverging'},
        'Oscillating\nEnvelope': {'score': -1.0, 'data': None, 'metric': 'oscillating'}
    }
    
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
            
            # Extract predictions (already shifted by model)
            p_l0 = outputs['p_l0'][:,1:].cpu().numpy()
            p_t = outputs['p_t'][:,1:].cpu().numpy()
            preds = outputs['predictions'][:,1:].cpu().numpy()  # p_sup
            ref_preds = outputs['reference_preds'][:,1:].cpu().numpy()  # p_ref
            
            for b in range(c.shape[0]):
                m_b = sm[b].cpu().numpy()
                cur_c = cshft[b].cpu().numpy()[m_b]
                cur_r = rshft[b].cpu().numpy()[m_b]
                cur_p_l0 = p_l0[b][m_b]
                cur_p_t = p_t[b][m_b]
                cur_preds = preds[b][m_b]
                cur_ref_preds = ref_preds[b][m_b]
                
                unique_skills = np.unique(cur_c)
                for skill in unique_skills:
                    s_mask = (cur_c == skill)
                    if np.sum(s_mask) < 6:  # Need minimum sequence length
                        continue
                    
                    seq_preds = cur_preds[s_mask]
                    seq_ref = cur_ref_preds[s_mask]
                    seq_r = cur_r[s_mask]
                    
                    # Calculate envelope metrics
                    envelope_width = np.abs(seq_preds - seq_ref)
                    mean_envelope = np.mean(envelope_width)
                    
                    # Calculate volatility (how much predictions change)
                    psup_volatility = np.std(np.diff(seq_preds)) if len(seq_preds) > 1 else 0
                    pref_volatility = np.std(np.diff(seq_ref)) if len(seq_ref) > 1 else 0
                    
                    # Who's more optimistic on average
                    psup_advantage = np.mean(seq_preds - seq_ref)
                    
                    # Convergence/divergence
                    first_half_env = np.mean(envelope_width[:len(envelope_width)//2])
                    second_half_env = np.mean(envelope_width[len(envelope_width)//2:])
                    convergence = first_half_env - second_half_env  # Positive = converging
                    
                    # Oscillation (envelope changes direction)
                    env_diffs = np.diff(envelope_width)
                    oscillations = np.sum(env_diffs[:-1] * env_diffs[1:] < 0) if len(env_diffs) > 1 else 0
                    
                    uid = data['uids'][b].item() if 'uids' in data else f"S{b}"
                    mean_l0 = np.mean(cur_p_l0[s_mask])
                    mean_t = np.mean(cur_p_t[s_mask])
                    
                    candidate = {
                        'uid': uid, 'skill': skill, 'seq': seq_r,
                        'preds': seq_preds, 'ref_preds': seq_ref,
                        'p_l0': mean_l0, 'p_t': mean_t
                    }
                    
                    # Calculate p_ref variance (secondary criterion for visual interest)
                    pref_variance = np.var(seq_ref)
                    
                    # Score each case type
                    scores = {
                        'min_envelope': -mean_envelope if mean_envelope < 0.08 else -100,
                        'max_envelope': mean_envelope if mean_envelope > 0.15 else -100,
                        'psup_higher': psup_advantage if psup_advantage > 0.05 else -100,
                        'pref_higher': -psup_advantage if psup_advantage < -0.05 else -100,
                        'psup_volatile': psup_volatility / max(pref_volatility, 0.01) if pref_volatility < 0.02 and psup_volatility > 0.05 else -100,
                        'both_dynamic': min(psup_volatility, pref_volatility) if psup_volatility > 0.04 and pref_volatility > 0.04 else -100,
                        'converging': convergence if convergence > 0.05 else -100,
                        'diverging': -convergence if convergence < -0.05 else -100,
                        'oscillating': oscillations / len(seq_preds) if len(seq_preds) > 3 else -100
                    }
                    
                    # Update best cases with secondary criterion: prefer higher p_ref variance
                    for case_name, case_info in case_criteria.items():
                        metric = case_info['metric']
                        primary_score = scores[metric]
                        
                        if primary_score > -99:  # Valid candidate
                            # Combine primary score with secondary (p_ref variance bonus)
                            # Primary criterion has priority, variance is tie-breaker with stronger weight
                            combined_score = primary_score + 0.5 * pref_variance  # 50% weight to variance for visual interest
                            
                            if combined_score > case_info['score']:
                                case_criteria[case_name]['score'] = combined_score
                                case_criteria[case_name]['data'] = candidate
                                case_criteria[case_name]['pref_var'] = pref_variance
            
            if i > 300:  # Search through enough data
                break
    
    # Report findings
    print("\n=== Diversity Search Results ===")
    for case_name, case_info in case_criteria.items():
        if case_info['data']:
            pref_var = case_info.get('pref_var', 0)
            print(f"{case_name}: Score={case_info['score']:.4f}, p_ref_var={pref_var:.4f}, Skill={case_info['data']['skill']}, SeqLen={len(case_info['data']['seq'])}")
        else:
            print(f"{case_name}: NO CASE FOUND")
    
    return {k: v['data'] for k, v in case_criteria.items()}

def plot_envelope_gallery(cases, bkt_params, output_path):
    """
    Plots a 3x3 gallery of diverse envelope behaviors.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, (title, data) in enumerate(cases.items()):
        ax = axes[idx]
        if data is None:
            ax.text(0.5, 0.5, f"No case found for\n{title}", ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=10, fontweight='bold')
            continue
            
        bkt_p = calculate_bkt_trajectory(bkt_params, data['skill'], data['seq'])
        x = np.arange(len(data['seq']))
        
        # Ground Truth Responses (Vertical Bars)
        for i, val in enumerate(data['seq']):
            color = 'green' if val == 1 else 'red'
            ax.axvline(x=i, color=color, alpha=0.12, linewidth=14, zorder=0)

        # Prediction Envelope (shaded band)
        p_sup = data['preds']
        p_ref = data['ref_preds']
        
        ax.fill_between(x, p_ref, p_sup, color='royalblue', alpha=0.2, 
                        label='Envelope', zorder=1)
        
        # Classical BKT Model (population-level baseline)
        ax.plot(x, bkt_p, color='black', linestyle=':', linewidth=2, alpha=0.7, 
                label="BKT Model", zorder=2, marker='x', markersize=4)
        
        # p_ref trajectory (interpretable)
        ax.plot(x, p_ref, color='steelblue', linewidth=2, alpha=0.85, 
                linestyle='--', label='$p_{ref}$ (Interpretable)', zorder=3, marker='s', markersize=4)
        
        # p_sup trajectory (supervised)
        ax.plot(x, p_sup, color='darkblue', linewidth=2.5, alpha=0.9, 
                label='$p_{sup}$ (Supervised)', zorder=4, marker='o', markersize=5)
        
        # Calculate envelope stats for subtitle
        env_width = np.mean(np.abs(p_sup - p_ref))
        psup_mean = np.mean(p_sup)
        pref_mean = np.mean(p_ref)
        
        ax.set_title(f"{title}\n(Skill {data['skill']}, Env={env_width:.3f})", 
                    fontsize=9, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(Correct)", fontsize=8)
        ax.set_xlabel("Interaction", fontsize=8)
        ax.legend(loc='best', fontsize=7, framealpha=0.85)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        
    plt.suptitle("Prediction Envelope Gallery: Diverse Behaviors of $p_{sup}$ vs $p_{ref}$", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 3x3 envelope gallery to {output_path}")

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
    
    print("Searching for diverse envelope behaviors (3x3 gallery)...")
    cases = find_diverse_cases(model, test_loader, device, bkt_params, n_cases=9)
    
    plot_envelope_gallery(cases, bkt_params, os.path.join(args.output_dir, "prediction_envelope_gallery.png"))

if __name__ == "__main__":
    main()
