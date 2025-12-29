
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

def bkt_update(mp, y, T, G, S):
    if y == 1:
        likelihood = mp * (1-S) / (mp * (1-S) + (1-mp) * G)
    else:
        likelihood = mp * S / (mp * S + (1-mp) * (1-G))
    m_next = likelihood + (1-likelihood) * T
    return m_next

def calculate_divergence(seq, idkt_params, bkt_params):
    G, S = bkt_params['guesses'], bkt_params['slips']
    T_bkt, L0_bkt = bkt_params['transition'], bkt_params['prior']
    ts_idkt, lc_idkt = idkt_params['rate'], idkt_params['init']
    
    m_idkt, m_bkt = [lc_idkt], [L0_bkt]
    for y in seq:
        m_idkt.append(bkt_update(m_idkt[-1], y, ts_idkt, G, S))
        m_bkt.append(bkt_update(m_bkt[-1], y, T_bkt, G, S))
    return np.mean(np.abs(np.array(m_idkt) - np.array(m_bkt))), m_idkt, m_bkt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    
    dataset = "assist2009"
    bkt_params_path = f"data/{dataset}/bkt_skill_params.pkl"
    with open(bkt_params_path, 'rb') as f:
        bkt_raw = pickle.load(f)['params']
        
    pred = pd.read_csv(os.path.join(args.run_dir, 'traj_predictions.csv'))
    rate = pd.read_csv(os.path.join(args.run_dir, 'traj_rate.csv'))
    init = pd.read_csv(os.path.join(args.run_dir, 'traj_initmastery.csv'))
    
    rate_col = 'idkt_rate' if 'idkt_rate' in rate.columns else 'ts'
    init_col = 'idkt_im' if 'idkt_im' in init.columns else 'lc'
    
    # Selection: Find a skill with many diverse students
    # 1. Group rates by skill to find high variance skills
    skill_stats = rate.groupby('skill_id')[rate_col].agg(['std', 'count', 'mean'])
    # Filter for skills with enough students and good variance
    candidates = skill_stats[(skill_stats['count'] > 20) & (skill_stats['std'] > 0.05)]
    if candidates.empty:
        target_sid = skill_stats.sort_values('count', ascending=False).index[0]
    else:
        target_sid = candidates.sort_values('std', ascending=False).index[0]
    
    print(f"Targeting Skill {target_sid} for comparative individualization analysis.")
    
    # 2. Get all students for this skill
    skill_rates = rate[rate['skill_id'] == target_sid].sort_values(rate_col)
    
    # Pick 4 students: Slowest, Medium, Fast, Fastest
    n = len(skill_rates)
    indices = [0, n//3, 2*n//3, n-1]
    selected_uids = skill_rates.iloc[indices]['student_id'].tolist()
    
    # 3. Process trajectories
    bkt_p = bkt_raw[target_sid]
    bkt_params = {
        'guesses': bkt_p.get('guesses', 0.1),
        'slips': bkt_p.get('slips', 0.1),
        'transition': bkt_p.get('transition', 0.1) if 'transition' in bkt_p else bkt_p.get('T', 0.1),
        'prior': bkt_p.get('prior', 0.5) if 'prior' in bkt_p else bkt_p.get('L0', 0.5)
    }
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    
    for i, uid in enumerate(selected_uids):
        ax = axes[i]
        
        upred = pred[(pred['student_id'] == uid) & (pred['skill_id'] == target_sid)]
        urate = rate[(rate['student_id'] == uid) & (rate['skill_id'] == target_sid)][rate_col].values[0]
        uinit = init[(init['student_id'] == uid) & (init['skill_id'] == target_sid)][init_col].values[0]
        
        if upred.empty: continue
        
        idkt_params = {'rate': urate, 'init': uinit}
        y = upred['y_true'].values
        div, m_idkt, m_bkt = calculate_divergence(y, idkt_params, bkt_params)
        
        t_axis = np.arange(len(m_idkt))
        ax.plot(t_axis, m_bkt, color='gray', linestyle='--', linewidth=3, alpha=0.5, label='BKT (Unified Policy)')
        ax.plot(t_axis, m_idkt, color='blue', linestyle='-', linewidth=3, label='iDKT (Individualized)')
        
        for t, val in enumerate(y):
            marker, color = ('o', 'green') if val == 1 else ('x', 'red')
            ax.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=80, zorder=5)
            
        bkt_t = bkt_params['transition']
        accel_pct = ((urate - bkt_t) / bkt_t) * 100
        
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Student {uid}\nIndividual Velocity: {urate:.3f}\n({accel_pct:+.1f}% vs BKT Prior)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Interaction Step", fontsize=12)
        if i == 0: ax.set_ylabel("Mastery Probability", fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'divergence_comparative_velocity.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Comparative Velocity Plot saved to {out_path}")

if __name__ == "__main__":
    main()
