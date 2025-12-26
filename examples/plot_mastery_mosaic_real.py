
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
BKT_COLS = ['p_bkt', 'p_idkt', 'bkt_m', 'idkt_m']

def load_data(run_dir):
    # Need predictions (for sequences) and roster/rate for parameters if needed
    # Ideally, we can reconstruct from traj_predictions.csv if it has the history
    # Or traj_rate.csv for parameters.
    
    # For this mosaic, we need P(m) trajectories. 
    # If they are not pre-computed, we might need to compute them.
    # However, 'traj_initmastery.csv' exists, 'traj_rate.csv' exists.
    # We can infer the P(m) trajectory using the BKT recurrence:
    # m_t+1 = m_t + (1-m_t)*T (if s_t=1) ?? No, BKT update is more complex based on obs.
    
    # Assumption: The user has a script that does this.
    # Since I cannot find it, I will implement a reconstruction based on the available CSVs.
    
    pred_path = os.path.join(run_dir, 'traj_predictions.csv')
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    
    if not all(os.path.exists(p) for p in [pred_path, rate_path, init_path]):
        print("Missing required CSVs")
        return None
        
    df_pred = pd.read_csv(pred_path) # uid, skill, y_true, etc.
    df_rate = pd.read_csv(rate_path) # uid, skill, idkt_rate, bkt_rate
    df_init = pd.read_csv(init_path) # uid, skill, idkt_im, bkt_im
    
    # Merge
    # Note: pred is longitudinal (many rows per uid-skill). rate/init are usually 1 row per uid-skill (or seq).
    # We need to map rate/init to the sequences.
    
    # Simplified: Get average rate/init per uid-skill
    # Check if rate/init are per-interaction or per-student-skill
    # For idkt, they are static per student-skill in the output (usually)
    
    # Let's focus on reconstructing the "Induced Mastery"
    # m_0 = l_c
    # m_{t+1} depends on y_t.
    # If correct: m_{t+1} = P(L|Correct) + (1-P(L|Correct))*T
    # If incorrect: m_{t+1} = P(L|Incorrect) + (1-P(L|Incorrect))*T
    # Where P(L|Obs) is the posterior.
    
    # This is complex to re-implement perfectly without the BKT logic details (guesses/slips).
    # BUT, the user prompt mentions "iDKT Induced (Solid)" vs "BKT Baseline (Dashed)".
    # If these are not in the CSVs, I must calculate them.
    # The BKT formula requires Guess (G) and Slip (S) params per skill.
    # I can load bkt_skill_params.pkl if available.
    
    return df_pred, df_rate, df_init

# Minimal BKT Update logic
def bkt_update(mp, y, T, G, S):
    # Posterior
    if y == 1:
        likelihood = mp * (1-S) / (mp * (1-S) + (1-mp) * G)
    else:
        likelihood = mp * S / (mp * S + (1-mp) * (1-G))
        
    # Prediction (Next state)
    m_next = likelihood + (1-likelihood) * T
    return m_next

def get_representatives(df_skill, idkt_rate_col, idkt_im_col, preds):
    # Calculate accuracy and count per student for this skill
    df_perf = preds.groupby('student_id').agg({'y_true': ['mean', 'count']}).reset_index()
    df_perf.columns = ['student_id', 'accuracy', 'count']
    
    # Filter out very short sequences (< 5) if possible
    df_perf = df_perf[df_perf['count'] >= 5]
    if len(df_perf) < 3:
        # Fallback to whatever we have
        df_perf = preds.groupby('student_id').agg({'y_true': ['mean', 'count']}).reset_index()
        df_perf.columns = ['student_id', 'accuracy', 'count']

    # Merge with parameters
    merged = df_perf.merge(df_skill[['student_id', idkt_rate_col, idkt_im_col]].drop_duplicates(), on='student_id')
    
    # Selection criteria:
    # 1. "Quick" (Green): High rate, High accuracy
    # 2. "Slow/Struggling" (Red): Low rate, Low accuracy
    # 3. "Knowledge Gap" (Orange/Yellow): Low IM, but some accuracy variation
    
    merged['score_quick'] = 0.5 * merged['accuracy'] + 0.5 * (merged[idkt_rate_col] / (merged[idkt_rate_col].max() + 1e-6))
    merged['score_slow'] = 0.5 * (1 - merged['accuracy']) + 0.5 * (1 - (merged[idkt_rate_col] / (merged[idkt_rate_col].max() + 1e-6)))
    merged['score_gap'] = 1 - merged[idkt_im_col]
    
    fast = merged.sort_values('score_quick', ascending=False).iloc[[0]]
    slow = merged.sort_values('score_slow', ascending=False).iloc[[0]]
    
    # For gap/med, try to pick one that is NOT slow or fast if possible
    others = merged[~merged['student_id'].isin([fast['student_id'].values[0], slow['student_id'].values[0]])]
    if not others.empty:
        med = others.sort_values('score_gap', ascending=False).iloc[[0]]
    else:
        # Fallback to median score
        med = merged.iloc[[len(merged)//2]]
        
    return [slow, med, fast]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    
    # Load bkt params
    import pickle
    bkt_params_path = "data/assist2009/bkt_skill_params.pkl" 
    
    if not os.path.exists(bkt_params_path):
        bkt_params_path = os.path.join(os.path.dirname(args.run_dir), '../../data/assist2009/bkt_skill_params.pkl')
        
    if not os.path.exists(bkt_params_path):
         print("Warning: BKT params not found. Using defaults G=0.1, S=0.1 for plot.")
         bkt_params = {}
    else:
        with open(bkt_params_path, 'rb') as f:
            bkt_params = pickle.load(f)['params']

    # Load Data
    pred, rate, init = load_data(args.run_dir)
    if pred is None: return

    # Merge Params for unified access
    rate_col = 'idkt_rate' if 'idkt_rate' in rate.columns else 'ts'
    init_col = 'idkt_im' if 'idkt_im' in init.columns else 'lc'
    
    params = rate.merge(init[['student_id', 'skill_id', init_col, 'bkt_im']], on=['student_id', 'skill_id'])
    
    bkt_rate_col = 'bkt_rate'
    bkt_im_col = 'bkt_im'
    
    # --- Advanced Skill Selection ---
    skill_metrics = []
    for skill_id in params['skill_id'].unique():
        s_params = params[params['skill_id'] == skill_id]
        if len(s_params) < 10: continue
        
        metrics = {
            'skill_id': skill_id,
            'v_rate': s_params[rate_col].std(),
            'v_im': s_params[init_col].std(),
            'div_rate': (s_params[rate_col] - s_params[bkt_rate_col]).abs().mean(),
            'count': len(s_params)
        }
        skill_metrics.append(metrics)
    
    df_m = pd.DataFrame(skill_metrics)
    
    # Pick 5 High Variance Rate (Individualization)
    top_v_rate = df_m.sort_values('v_rate', ascending=False).head(5)['skill_id'].tolist()
    # Pick 5 High Variance IM (Knowledge Gaps)
    top_v_im = df_m[~df_m['skill_id'].isin(top_v_rate)].sort_values('v_im', ascending=False).head(5)['skill_id'].tolist()
    # Pick 5 High Divergence (DL vs Theory)
    top_div = df_m[~df_m['skill_id'].isin(top_v_rate + top_v_im)].sort_values('div_rate', ascending=False).head(5)['skill_id'].tolist()
    
    diverse_skills = top_v_rate + top_v_im + top_div
    categories = ['Indiv'] * 5 + ['Gap'] * 5 + ['Div'] * 5
    
    # Plotting
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (skill, cat) in enumerate(zip(diverse_skills, categories)):
        ax = axes[i]
        
        sp = bkt_params.get(skill, {'guesses': 0.1, 'slips': 0.1})
        G, S = sp['guesses'], sp['slips']
        
        skill_preds = pred[pred['skill_id'] == skill]
        skill_params = params[params['skill_id'] == skill]
        
        reps = get_representatives(skill_params, rate_col, init_col, skill_preds)
        colors = ['red', 'orange', 'green']
        
        for rep, color in zip(reps, colors):
            uid = rep['student_id'].values[0]
            
            traj = skill_preds[skill_preds['student_id'] == uid]
            if len(traj) == 0: continue
            y_true = traj['y_true'].values
            
            ts = rep[rate_col].values[0]
            lc = rep[init_col].values[0]
            
            try:
                T = float(rep[bkt_rate_col].values[0])
                L0 = float(rep[bkt_im_col].values[0])
            except:
                T, L0 = 0.1, 0.5
            
            m_idkt, m_bkt = [lc], [L0]
            for y in y_true:
                m_idkt.append(bkt_update(m_idkt[-1], y, ts, G, S))
                m_bkt.append(bkt_update(m_bkt[-1], y, T, G, S))
                
            ax.plot(m_bkt, color=color, linestyle='--', linewidth=2.0, alpha=0.35, zorder=1)
            ax.plot(m_idkt, color=color, linestyle='-', alpha=1.0, linewidth=1.5, marker='o', markersize=3, zorder=2)
            
            for t, y in enumerate(y_true):
                marker = 'o' if y == 1 else 'x'
                ax.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=25, zorder=3)
        
        ax.set_title(f"Skill {skill} ({cat})", fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.15)
        if i % 5 != 0: ax.set_yticklabels([])
        if i < 10: ax.set_xticklabels([])
        
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='gray', lw=2, ls='--'),
                Line2D([0], [0], color='gray', lw=2, ls='-'),
                Line2D([0], [0], marker='o', color='gray', label='Correct', markersize=5, ls=''),
                Line2D([0], [0], marker='x', color='gray', label='Incorrect', markersize=5, ls='')
            ]
            ax.legend(custom_lines, ['BKT', 'iDKT', 'Corr', 'Incorr'], loc='lower right', fontsize=7, framealpha=0.8)
    
    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'mastery_alignment_mosaic_real.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved mosaic TO {out_path}")

if __name__ == "__main__":
    main()
