
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
    
    dataset = "assist2015" if "assist2015" in args.run_dir else "assist2009"
    bkt_params_path = f"data/{dataset}/bkt_skill_params.pkl"
    with open(bkt_params_path, 'rb') as f:
        bkt_raw = pickle.load(f)['params']
        
    pred = pd.read_csv(os.path.join(args.run_dir, 'traj_predictions.csv'))
    rate = pd.read_csv(os.path.join(args.run_dir, 'traj_rate.csv'))
    init = pd.read_csv(os.path.join(args.run_dir, 'traj_initmastery.csv'))
    
    rate_col = 'idkt_rate' if 'idkt_rate' in rate.columns else 'ts'
    init_col = 'idkt_im' if 'idkt_im' in init.columns else 'lc'
    
    all_pairs = []
    grouped = pred.groupby(['student_id', 'skill_id'])
    for (uid, sid), group in grouped:
        if len(group) < 8: continue # Stricter for high quality trajectories
        r_row = rate[(rate['student_id'] == uid) & (rate['skill_id'] == sid)]
        i_row = init[(init['student_id'] == uid) & (init['skill_id'] == sid)]
        if r_row.empty or i_row.empty or sid not in bkt_raw: continue
        
        idkt_p = {'rate': r_row[rate_col].values[0], 'init': i_row[init_col].values[0]}
        bp = bkt_raw[sid]
        bkt_p = {'guesses': bp.get('guesses', 0.1), 'slips': bp.get('slips', 0.1),
                 'transition': bp.get('transition', 0.1) if 'transition' in bp else bp.get('T', 0.1),
                 'prior': bp.get('prior', 0.5) if 'prior' in bp else bp.get('L0', 0.5)}
        
        div, m_idkt, m_bkt = calculate_divergence(group['y_true'].values, idkt_p, bkt_p)
        all_pairs.append({'uid': uid, 'sid': sid, 'div': div, 'm_idkt': m_idkt, 'm_bkt': m_bkt, 'y': group['y_true'].values})

    df = pd.DataFrame(all_pairs)
    
    # Selection: Find skill with highest "Individualization Contrast"
    skill_contrasts = []
    for sid, group in df.groupby('sid'):
        if len(group) < 10: continue
        best_conforming = group.sort_values('div').iloc[0]
        best_discovery = group.sort_values('div', ascending=False).iloc[0]
        contrast = best_discovery['div'] - best_conforming['div']
        skill_contrasts.append({'sid': sid, 'contrast': contrast, 'item_a': best_conforming, 'item_b': best_discovery})
        
    top_skill = sorted(skill_contrasts, key=lambda x: x['contrast'], reverse=True)[0]
    target_sid = top_skill['sid']
    item_a = top_skill['item_a']
    item_b = top_skill['item_b']

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (item, label) in enumerate([(item_a, "Conforming"), (item_b, "Discovery")]):
        ax = axes[i]
        m_idkt, m_bkt, y = item['m_idkt'], item['m_bkt'], item['y']
        
        t_axis = np.arange(len(m_idkt))
        ax.plot(t_axis, m_bkt, color='gray', linestyle='--', linewidth=3, alpha=0.5, label='BKT (Theory Prior)')
        ax.plot(t_axis, m_idkt, color='blue', linestyle='-', linewidth=3, label='iDKT (Individualized)')
        
        # Interactions
        for t, val in enumerate(y):
            marker = 'o' if val == 1 else 'x'
            color = 'green' if val == 1 else 'red'
            ax.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=80, zorder=5)
            
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label} Individualization: Student {item['uid']}\n(Skill {target_sid}, Divergence: {item['div']:.3f})", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Interaction Step", fontsize=12)
        if i == 0: ax.set_ylabel("Mastery Probability", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'mastery_alignment_contrast.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Saved Contrast Plot to {out_path}")

if __name__ == "__main__":
    main()
