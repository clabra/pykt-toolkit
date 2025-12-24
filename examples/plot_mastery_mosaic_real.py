
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

def get_representatives(df_skill, idkt_rate_col, preds):
    # Calculate accuracy per student for this skill
    # preds contains y_true
    df_perf = preds.groupby('student_id')['y_true'].mean().reset_index()
    df_perf.columns = ['student_id', 'accuracy']
    
    # Score = Accuracy (0.7) + Normalized Rate (0.3)
    # We want "Green" to be high perf, high rate.
    # "Red" to be low perf, low rate.
    # "Yellow" in between.
    
    # Join rate info
    merged = df_perf.merge(df_skill[['student_id', idkt_rate_col]].drop_duplicates(), on='student_id')
    
    # Normalize
    merged['norm_acc'] = (merged['accuracy'] - merged['accuracy'].min()) / (merged['accuracy'].max() - merged['accuracy'].min() + 1e-6)
    merged['norm_rate'] = (merged[idkt_rate_col] - merged[idkt_rate_col].min()) / (merged[idkt_rate_col].max() - merged[idkt_rate_col].min() + 1e-6)
    
    merged['score'] = 0.7 * merged['norm_acc'] + 0.3 * merged['norm_rate']
    
    # Sort by score
    sorted_df = merged.sort_values('score')
    
    if len(sorted_df) < 3:
        return [sorted_df.iloc[[0]], sorted_df.iloc[[0]], sorted_df.iloc[[0]]] # Fallback
        
    n = len(sorted_df)
    slow = sorted_df.iloc[[0]] # Lowest score
    med = sorted_df.iloc[[n//2]] # Median score
    fast = sorted_df.iloc[[-1]] # Highest score
    
    return [slow, med, fast]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    # Assuming standard pykt locations
    args = parser.parse_args()
    
    # Load bkt params
    import pickle
    bkt_params_path = "data/assist2009/bkt_skill_params.pkl" # Hardcoded assumption for now based on context
    
    # Verify paths
    if not os.path.exists(bkt_params_path):
        # fallback try generic path
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

    # Merge Params into Pred
    # Assume 1-to-1 mapping logic exists or just process per uid
    
    # Select 15 Diverse Skills (High Variance in Rate)
    # Using 'idkt_rate'
    rate_col = 'idkt_rate' if 'idkt_rate' in rate.columns else 'ts'
    init_col = 'idkt_im' if 'idkt_im' in init.columns else 'lc'
    
    # Group by skill
    skill_stats = rate.groupby('skill_id')[rate_col].std().sort_values(ascending=False)
    diverse_skills = skill_stats.head(15).index.tolist()
    
    # Plotting
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, skill in enumerate(diverse_skills):
        ax = axes[i]
        
        # Get skill params
        sp = bkt_params.get(skill, {'guesses': 0.1, 'slips': 0.1})
        G, S = sp['guesses'], sp['slips']
        
        # Get Representatives
        # Combine rate info with pred info
        skill_preds = pred[pred['skill_id'] == skill]
        skill_rates = rate[rate['skill_id'] == skill]
        skill_inits = init[init['skill_id'] == skill]
        
        # Find reps
        reps = get_representatives(skill_rates, rate_col, skill_preds)
        colors = ['red', 'orange', 'green']
        
        for rep, color in zip(reps, colors):
            uid = rep['student_id'].values[0]
            
            # Get Trajectory
            traj = skill_preds[skill_preds['student_id'] == uid] # assumption of index order
            if len(traj) == 0: continue
            
            y_true = traj['y_true'].values
            
            # Get Params
            ts = rep[rate_col].values[0]
            lc = skill_inits[skill_inits['student_id'] == uid][init_col].values[0]
            
            # BKT Params (Global)
            # Safe extraction of scalar values
            try:
                # Filter first for safety
                row_rate = skill_rates[skill_rates['student_id'] == uid]
                if not row_rate.empty:
                    T = float(row_rate['bkt_rate'].values[0])
                else:
                    T = 0.1
                    
                row_init = skill_inits[skill_inits['student_id'] == uid]
                if not row_init.empty:
                    L0 = float(row_init['bkt_im'].values[0])
                else:
                    L0 = 0.5
            except Exception as e:
                print(f"Error extracting params for uid {uid}: {e}")
                T, L0 = 0.1, 0.5
            
            # Simulate Paths
            m_idkt = [lc]
            m_bkt = [L0]
            
            for y in y_true:
                m_idkt.append(bkt_update(m_idkt[-1], y, ts, G, S))
                m_bkt.append(bkt_update(m_bkt[-1], y, T, G, S))
                
            # Plot BKT FIRST (Background) - VERY LIGHT BLACK DASHED
            ax.plot(m_bkt, color='black', linestyle='--', linewidth=2.0, alpha=0.2, zorder=1)
            
            # Plot iDKT SECOND (Foreground) - COLORED SOLID WITH MARKERS
            ax.plot(m_idkt, color=color, linestyle='-', alpha=1.0, linewidth=1.2, marker='o', markersize=3, label='iDKT', zorder=2)
            
            # Plot Outcomes
            for t, y in enumerate(y_true):
                marker = 'o' if y == 1 else 'x'
                ax.scatter(t+1, m_idkt[t+1], marker=marker, color=color, s=20, zorder=3)
        
        ax.set_title(f"Skill {skill}", fontsize=10)
        ax.set_ylim(0, 1.05)
        if i % 5 != 0: ax.set_yticklabels([])
        
        # Add legend only to the last plot or top right
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='black', lw=1, ls='--', alpha=0.5),
                            Line2D([0], [0], color='gray', lw=1.5, ls='-')]
            ax.legend(custom_lines, ['Theory (BKT)', 'iDKT'], loc='lower right', fontsize=8)
    
    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'plots', 'mastery_alignment_mosaic_real.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved mosaic to {out_path}")

if __name__ == "__main__":
    main()
