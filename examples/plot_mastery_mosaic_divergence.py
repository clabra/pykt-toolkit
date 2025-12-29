
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def bkt_update(mp, y, T, G, S):
    # Posterior
    if y == 1:
        likelihood = mp * (1-S) / (mp * (1-S) + (1-mp) * G)
    else:
        likelihood = mp * S / (mp * S + (1-mp) * (1-G))
    # Prediction (Next state)
    m_next = likelihood + (1-likelihood) * T
    return m_next

def calculate_divergence(seq, idkt_params, bkt_params):
    G, S = bkt_params['guesses'], bkt_params['slips']
    T_bkt, L0_bkt = bkt_params['transition'], bkt_params['prior']
    
    ts_idkt = idkt_params['rate']
    lc_idkt = idkt_params['init']
    
    m_idkt = [lc_idkt]
    m_bkt = [L0_bkt]
    
    for y in seq:
        m_idkt.append(bkt_update(m_idkt[-1], y, ts_idkt, G, S))
        m_bkt.append(bkt_update(m_bkt[-1], y, T_bkt, G, S))
        
    return np.mean(np.abs(np.array(m_idkt) - np.array(m_bkt))), m_idkt, m_bkt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--n_students', type=int, default=5)
    parser.add_argument('--m_skills', type=int, default=3)
    args = parser.parse_args()
    
    # 1. Determine dataset from run_dir name or config
    dataset = "assist2015" if "assist2015" in args.run_dir else "assist2009"
    bkt_params_path = f"data/{dataset}/bkt_skill_params.pkl"
    
    if not os.path.exists(bkt_params_path):
        print(f"Error: BKT params not found at {bkt_params_path}")
        return

    with open(bkt_params_path, 'rb') as f:
        bkt_raw = pickle.load(f)['params']
        
    # 2. Load iDKT Data
    pred = pd.read_csv(os.path.join(args.run_dir, 'traj_predictions.csv'))
    rate = pd.read_csv(os.path.join(args.run_dir, 'traj_rate.csv'))
    init = pd.read_csv(os.path.join(args.run_dir, 'traj_initmastery.csv'))
    
    # Map rate and init columns properly
    rate_col = 'idkt_rate' if 'idkt_rate' in rate.columns else 'ts'
    init_col = 'idkt_im' if 'idkt_im' in init.columns else 'lc'
    
    # 3. Process Divergences
    print("Calculating trajectory clinical divergence...")
    student_skill_data = []
    
    # Group by student and skill
    grouped = pred.groupby(['student_id', 'skill_id'])
    
    for (uid, sid), group in grouped:
        if len(group) < 5: continue # Only long trajectories
        
        # Get iDKT params for this pair
        r_row = rate[(rate['student_id'] == uid) & (rate['skill_id'] == sid)]
        i_row = init[(init['student_id'] == uid) & (init['skill_id'] == sid)]
        
        if r_row.empty or i_row.empty: continue
        
        idkt_p = {'rate': r_row[rate_col].values[0], 'init': i_row[init_col].values[0]}
        
        # Get BKT params
        if sid not in bkt_raw: continue
        bp = bkt_raw[sid]
        # In assist2015, params might be named differently (transition vs T)
        bkt_p = {
            'guesses': bp.get('guesses', 0.1),
            'slips': bp.get('slips', 0.1),
            'transition': bp.get('transition', 0.1) if 'transition' in bp else bp.get('T', 0.1),
            'prior': bp.get('prior', 0.5) if 'prior' in bp else bp.get('L0', 0.5)
        }
        
        div, m_idkt, m_bkt = calculate_divergence(group['y_true'].values, idkt_p, bkt_p)
        
        student_skill_data.append({
            'student_id': uid,
            'skill_id': sid,
            'divergence': div,
            'm_idkt': m_idkt,
            'm_bkt': m_bkt,
            'y_true': group['y_true'].values.tolist()
        })
        
    df_div = pd.DataFrame(student_skill_data)
    
    # 4. Selection Logic
    # For each student, find top M skills by divergence
    student_scores = []
    for uid, group in df_div.groupby('student_id'):
        if len(group) < args.m_skills: continue
        top_m = group.sort_values('divergence', ascending=False).head(args.m_skills)
        avg_div = top_m['divergence'].mean()
        student_scores.append({'student_id': uid, 'score': avg_div, 'skills': top_m})
        
    # Pick Top N students
    top_n_students = sorted(student_scores, key=lambda x: x['score'], reverse=True)[:args.n_students]
    
    # 5. Plotting
    fig, axes = plt.subplots(args.n_students, args.m_skills, figsize=(args.m_skills*4, args.n_students*3))
    
    for r, student_data in enumerate(top_n_students):
        uid = student_data['student_id']
        skill_group = student_data['skills']
        
        for c, (_, row) in enumerate(skill_group.iterrows()):
            ax = axes[r, c]
            
            m_idkt = row['m_idkt']
            m_bkt = row['m_bkt']
            y_true = row['y_true']
            
            # Plot
            t_axis = np.arange(len(m_idkt))
            ax.plot(t_axis, m_bkt, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='BKT')
            ax.plot(t_axis, m_idkt, color='blue', linestyle='-', linewidth=2, label='iDKT')
            
            # Scatter interaction outcomes
            for t, y in enumerate(y_true):
                marker = 'o' if y == 1 else 'x'
                color = 'green' if y == 1 else 'red'
                ax.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=40, zorder=3)
                
            ax.set_ylim(0, 1.05)
            ax.set_title(f"S:{uid} | K:{row['skill_id']}\nDiv: {row['divergence']:.3f}", fontsize=10)
            ax.grid(True, alpha=0.2)
            
            if r == 0 and c == 0:
                ax.legend(loc='lower right', fontsize=8)
            if c > 0: ax.set_yticklabels([])
            if r < args.n_students - 1: ax.set_xticklabels([])

    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'mastery_alignment_mosaic_divergence.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Saved Divergence Mosaic TO {out_path}")

if __name__ == "__main__":
    main()
