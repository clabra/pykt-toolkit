
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
    
    uid, sid = 866, 63
    
    upred = pred[(pred['student_id'] == uid) & (pred['skill_id'] == sid)]
    urate = rate[(rate['student_id'] == uid) & (rate['skill_id'] == sid)][rate_col].values[0]
    uinit = init[(init['student_id'] == uid) & (init['skill_id'] == sid)][init_col].values[0]
    
    bp = bkt_raw[sid]
    bkt_params = {
        'guesses': bp.get('guesses', 0.1),
        'slips': bp.get('slips', 0.1),
        'transition': bp.get('transition', 0.1) if 'transition' in bp else bp.get('T', 0.1),
        'prior': bp.get('prior', 0.5) if 'prior' in bp else bp.get('L0', 0.5)
    }
    
    y = upred['y_true'].values
    div, m_idkt, m_bkt = calculate_divergence(y, {'rate': urate, 'init': uinit}, bkt_params)
    
    plt.figure(figsize=(10, 6))
    t_axis = np.arange(len(m_idkt))
    plt.plot(t_axis, m_bkt, color='gray', linestyle='--', linewidth=3, alpha=0.5, label='BKT')
    plt.plot(t_axis, m_idkt, color='blue', linestyle='-', linewidth=3, label='iDKT')
    
    for t, val in enumerate(y):
        marker, color = ('o', 'green') if val == 1 else ('x', 'red')
        plt.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=100, zorder=5)
        
    plt.ylim(0, 1.05)
    plt.title(f"Stud {uid}, Skill {sid}, Indiv Vel={urate:.4f} (High)", fontsize=16, fontweight='bold')
    plt.xlabel("Interaction Step", fontsize=14)
    plt.ylabel("Mastery Probability", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'mastery_single_866.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Single Mastery Plot saved to {out_path}")

if __name__ == "__main__":
    main()
