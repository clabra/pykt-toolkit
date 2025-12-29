
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
    parser.add_argument('--target_sid', type=int, default=63)
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
    
    # 1. Filter students for the target skill with medium length sequences (10-30)
    seq_lengths = pred[pred['skill_id'] == args.target_sid].groupby('student_id').size().reset_index(name='length')
    medium_stus = seq_lengths[(seq_lengths['length'] >= 10) & (seq_lengths['length'] <= 30)]['student_id'].tolist()
    
    if len(medium_stus) < 4:
        print(f"Not enough medium-length students for skill {args.target_sid}. Retrying with broader range.")
        medium_stus = seq_lengths[(seq_lengths['length'] >= 8) & (seq_lengths['length'] <= 40)]['student_id'].tolist()
        
    # 2. Sort these students by their individualized velocity
    skill_rates = rate[(rate['skill_id'] == args.target_sid) & (rate['student_id'].isin(medium_stus))].sort_values(rate_col)
    
    # 3. Select 4 extreme/diverse cases
    n = len(skill_rates)
    indices = [0, n//3, 2*n//3, n-1]
    selected_uids = skill_rates.iloc[indices]['student_id'].tolist()
    
    # 4. BKT Params
    bp = bkt_raw[args.target_sid]
    bkt_params = {
        'guesses': bp.get('guesses', 0.1),
        'slips': bp.get('slips', 0.1),
        'transition': bp.get('transition', 0.1) if 'transition' in bp else bp.get('T', 0.1),
        'prior': bp.get('prior', 0.5) if 'prior' in bp else bp.get('L0', 0.5)
    }
    
    # 5. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    
    for i, uid in enumerate(selected_uids):
        ax = axes[i]
        
        upred = pred[(pred['student_id'] == uid) & (pred['skill_id'] == args.target_sid)]
        urate = rate[(rate['student_id'] == uid) & (rate['skill_id'] == args.target_sid)][rate_col].values[0]
        uinit = init[(init['student_id'] == uid) & (init['skill_id'] == args.target_sid)][init_col].values[0]
        
        y = upred['y_true'].values
        div, m_idkt, m_bkt = calculate_divergence(y, {'rate': urate, 'init': uinit}, bkt_params)
        
        t_axis = np.arange(len(m_idkt))
        ax.plot(t_axis, m_bkt, color='gray', linestyle='--', linewidth=3, alpha=0.5, label='BKT (Fixed Rate)')
        ax.plot(t_axis, m_idkt, color='blue', linestyle='-', linewidth=3, label='iDKT (Indiv. Rate)')
        
        # Interactions
        for t, val in enumerate(y):
            marker, color = ('o', 'green') if val == 1 else ('x', 'red')
            ax.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=70, zorder=5)
            
        bkt_t = bkt_params['transition']
        accel_pct = ((urate - bkt_t) / bkt_t) * 100
        
        if i == 0: profile = "Slowest (in range)"
        elif i == 3: profile = "Fastest (in range)"
        else: profile = "Average"
            
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Stud {uid}: {profile}\nTS: {urate:.4f} ({accel_pct:+.1f}%)", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"Steps (Total: {len(y)})", fontsize=11)
        if i == 0: ax.set_ylabel("Mastery Probability", fontsize=11)
        ax.grid(True, alpha=0.2)
        if i == 0: ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'divergence_comparative_medium_seq.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Comparative (Medium Seq) saved to {out_path}")

if __name__ == "__main__":
    main()
