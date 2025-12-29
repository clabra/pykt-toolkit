
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
    
    all_pairs = []
    grouped = pred.groupby(['student_id', 'skill_id'])
    for (uid, sid), group in grouped:
        if len(group) < 8: continue
        r_row = rate[(rate['student_id'] == uid) & (rate['skill_id'] == sid)]
        i_row = init[(init['student_id'] == uid) & (init['skill_id'] == sid)]
        if r_row.empty or i_row.empty or sid not in bkt_raw: continue
        
        idkt_p = {'rate': r_row[rate_col].values[0], 'init': i_row[init_col].values[0]}
        bp = bkt_raw[sid]
        bkt_p = {'guesses': bp.get('guesses', 0.1), 'slips': bp.get('slips', 0.1),
                 'transition': bp.get('transition', 0.1) if 'transition' in bp else bp.get('T', 0.1),
                 'prior': bp.get('prior', 0.5) if 'prior' in bp else bp.get('L0', 0.5)}
        
        div_mean, m_idkt, m_bkt = calculate_divergence(group['y_true'].values, idkt_p, bkt_p)
        
        m_idkt_np = np.array(m_idkt)
        m_bkt_np = np.array(m_bkt)
        y_true = group['y_true'].values
        success_rate = np.mean(y_true)
        
        # 1. Acceleration: iDKT reached 0.8 much faster than BKT
        idkt_reach8 = np.where(m_idkt_np >= 0.8)[0][0] if any(m_idkt_np >= 0.8) else 999
        bkt_reach8 = np.where(m_bkt_np >= 0.8)[0][0] if any(m_bkt_np >= 0.8) else 999
        acceleration = bkt_reach8 - idkt_reach8
        
        # 2. Stability/Filtering: Std of BKT changes vs Std of iDKT changes
        bkt_vol = np.std(np.diff(m_bkt_np))
        idkt_vol = np.std(np.diff(m_idkt_np))
        vol_ratio = bkt_vol / (idkt_vol + 1e-6)
        
        # 3. Resilience: Difference in minimum mastery (after initial steps)
        resil = (np.min(m_idkt_np[3:]) - np.min(m_bkt_np[3:])) if len(y_true) > 5 else 0
        
        all_pairs.append({
            'uid': uid, 'sid': sid, 'div': div_mean, 
            'm_idkt': m_idkt, 'm_bkt': m_bkt, 'y': y_true.tolist(),
            'accel': acceleration, 'vol_ratio': vol_ratio, 'resil': resil, 'success_rate': success_rate
        })

    df = pd.DataFrame(all_pairs)
    picked_sids = set()
    picked_uids = set()

    def select_best(filtered_df, sort_col, reverse=True, min_success=0.2):
        nonlocal picked_sids, picked_uids
        # Ensure we have some evidence of learning for positive patterns
        candidates = filtered_df[
            ~filtered_df['sid'].isin(picked_sids) & 
            ~filtered_df['uid'].isin(picked_uids) &
            (filtered_df['success_rate'] >= min_success)
        ]
        if candidates.empty: 
            return filtered_df.sort_values(sort_col, ascending=not reverse).iloc[0]
        match = candidates.sort_values(sort_col, ascending=not reverse).iloc[0]
        picked_sids.add(match['sid'])
        picked_uids.add(match['uid'])
        return match

    # Pattern Selection with sanity checks
    case1 = select_best(df[df['accel'] > 3], 'accel', min_success=0.4) # Must be a good student
    case2 = select_best(df[df['vol_ratio'] > 1.2], 'vol_ratio', min_success=0.3) 
    case3 = select_best(df[df['resil'] > 0.25], 'resil', min_success=0.3)
    case4 = select_best(df, 'div', min_success=0.5) # High confidence discovery must have >50% success
    
    selected_cases = [
        (case1, "Empirical Acceleration"),
        (case2, "Noise Filtering"),
        (case3, "Diagnostic Resilience"),
        (case4, "Confidence Discovery")
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for i, (item, label) in enumerate(selected_cases):
        ax = axes[i]
        m_idkt, m_bkt, y = item['m_idkt'], item['m_bkt'], item['y']
        t_axis = np.arange(len(m_idkt))
        ax.plot(t_axis, m_bkt, color='gray', linestyle='--', linewidth=3, alpha=0.5, label='BKT (Theory)')
        ax.plot(t_axis, m_idkt, color='blue', linestyle='-', linewidth=3, label='iDKT (Individualized)')
        for t, val in enumerate(y):
            marker, color = ('o', 'green') if val == 1 else ('x', 'red')
            ax.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=80, zorder=5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Pattern {i+1}: {label}\n(Skill {item['sid']}, Stud {item['uid']})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Steps", fontsize=12)
        if i == 0: ax.set_ylabel("Mastery Probability", fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'divergence_quad_patterns.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Quadrant Divergence Plot saved to {out_path}")

if __name__ == "__main__":
    main()
