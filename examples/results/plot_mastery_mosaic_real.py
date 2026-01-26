
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(run_dir):
    pred_path = os.path.join(run_dir, 'traj_predictions.csv')
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    
    if not all(os.path.exists(p) for p in [pred_path, rate_path, init_path]):
        print(f"Missing required CSVs in {run_dir}")
        return None, None, None
        
    df_pred = pd.read_csv(pred_path)
    df_rate = pd.read_csv(rate_path)
    df_init = pd.read_csv(init_path)
    
    return df_pred, df_rate, df_init


def get_representatives(skill_params, rate_col, init_col, skill_preds, theoretical_p_start):
    # Calculate group metrics
    df_perf = skill_preds.groupby('student_id').agg({'y_true': ['mean', 'count'], 'p_bkt': 'first'}).reset_index()
    df_perf.columns = ['student_id', 'accuracy', 'count', 'p_bkt_first']
    
    # Priority 1: Students whose BKT starts at the Prior (Beginners)
    # We look for students where p_bkt_first is within 0.05 of theoretical_p_start
    df_perf['is_beginner'] = (df_perf['p_bkt_first'] - theoretical_p_start).abs() < 0.05
    
    # Selection: We still want a mix of Fast, Slow, and Gap, but prioritized from 'Beginners'
    beginners = df_perf[df_perf['is_beginner']]
    if len(beginners) < 3:
        # Fallback to everyone if we can't find enough absolute beginners
        pool = df_perf
    else:
        pool = beginners

    # Filter out very short sequences for better visualization (> 5 steps)
    pool = pool[pool['count'] >= 5]
    if pool.empty: pool = df_perf

    # Merge with parameters
    merged = pool.merge(skill_params[['student_id', rate_col, init_col]].drop_duplicates(), on='student_id')
    
    merged['score_quick'] = 0.5 * merged['accuracy'] + 0.5 * (merged[rate_col] / (merged[rate_col].max() + 1e-6))
    merged['score_slow'] = 0.5 * (1 - merged['accuracy']) + 0.5 * (1 - (merged[rate_col] / (merged[rate_col].max() + 1e-6)))
    merged['score_gap'] = (merged[init_col].mean() - merged[init_col]).abs()
    
    fast = merged.sort_values('score_quick', ascending=False).iloc[[0]]
    slow = merged.sort_values('score_slow', ascending=False).iloc[[0]]
    
    others = merged[~merged['student_id'].isin([fast['student_id'].values[0], slow['student_id'].values[0]])]
    if not others.empty:
        gap = others.sort_values('score_gap', ascending=False).iloc[[0]]
    else:
        gap = merged.iloc[[len(merged)//2]]
        
    return [slow, gap, fast]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    
    # Load bkt params for theoretical start alignment
    import pickle
    bkt_params_path = "data/assist2009/bkt_skill_params.pkl" 
    if not os.path.exists(bkt_params_path):
        bkt_params_path = os.path.join(os.path.dirname(args.run_dir), '../../data/assist2009/bkt_skill_params.pkl')
    
    if os.path.exists(bkt_params_path):
        with open(bkt_params_path, 'rb') as f:
            bkt_param_data = pickle.load(f)['params']
    else:
        bkt_param_data = None

    # Load Data
    pred, rate, init = load_data(args.run_dir)
    if pred is None: return

    # Map column names
    rate_col = 'idkt_rate' if 'idkt_rate' in rate.columns else 'ts'
    init_col = 'idkt_im' if 'idkt_im' in init.columns else 'lc'
    
    # Merge parameters for selection metadata
    params = rate.merge(init[['student_id', 'skill_id', init_col, 'bkt_im']], on=['student_id', 'skill_id'])
    
    # Select Skills
    skill_metrics = []
    # Pre-calculate counts per student-skill to speed up filtering
    student_skill_counts = pred.groupby(['skill_id', 'student_id']).size().reset_index(name='count')
    
    for skill_id in params['skill_id'].unique():
        # Filter for students who have > 10 interactions for THIS skill
        long_history_uids = student_skill_counts[(student_skill_counts['skill_id'] == skill_id) & 
                                                (student_skill_counts['count'] > 10)]['student_id']
        
        if len(long_history_uids) < 5: continue # Ensure we have enough candidates
        
        s_params = params[(params['skill_id'] == skill_id) & (params['student_id'].isin(long_history_uids))]
        
        # Empirical Mode Alignment (for beginners check)
        skill_preds_subset = pred[(pred['skill_id'] == skill_id) & (pred['student_id'].isin(long_history_uids))]
        first_bkt_values = skill_preds_subset.groupby('student_id')['p_bkt'].first()
        empirical_prior = first_bkt_values.mode().iloc[0] if not first_bkt_values.empty else 0.5
        
        metrics = {
            'skill_id': skill_id,
            'v_rate': s_params[rate_col].std(),
            'v_im': s_params[init_col].std(),
            'p_start': empirical_prior,
            'count': len(s_params)
        }
        skill_metrics.append(metrics)
    
    df_m = pd.DataFrame(skill_metrics)
    if df_m.empty: 
        print("No skills found with students having > 10 interactions.")
        return

    diverse_skills = df_m.sort_values('v_rate', ascending=False).head(15)['skill_id'].tolist()
    
    # Plotting
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    axes = axes.flatten()
    
    for i, skill in enumerate(diverse_skills):
        ax = axes[i]
        skill_preds = pred[pred['skill_id'] == skill]
        skill_params = params[params['skill_id'] == skill]
        t_start = df_m[df_m['skill_id'] == skill]['p_start'].values[0]
        
        # Re-Selection using empirical prior alignment
        df_perf = skill_preds.groupby('student_id').agg({'y_true': ['mean', 'count'], 'p_bkt': 'first'}).reset_index()
        df_perf.columns = ['student_id', 'accuracy', 'count', 'p_bkt_first']
        
        # Filter for students aligned to the mode (prior) AND having > 10 points
        beginners = df_perf[(df_perf['p_bkt_first'] - t_start).abs() < 1e-4]
        pool = beginners[beginners['count'] > 10]
        
        if len(pool) < 3:
            # Fallback: Just get any student with > 10 points for this skill
            pool = df_perf[df_perf['count'] > 10]
        
        if pool.empty: # Final fallback if no one has > 10 points (should be rare given skill selection)
             pool = df_perf.sort_values('count', ascending=False).head(5)
        
        merged = pool.merge(skill_params[['student_id', rate_col, init_col]].drop_duplicates(), on='student_id')
        merged['score_quick'] = 0.5 * merged['accuracy'] + 0.5 * (merged[rate_col] / (merged[rate_col].max() + 1e-6))
        merged['score_slow'] = 0.5 * (1 - merged['accuracy']) + 0.5 * (1 - (merged[rate_col] / (merged[rate_col].max() + 1e-6)))
        merged['score_gap'] = (merged[init_col].mean() - merged[init_col]).abs()
        
        fast = merged.sort_values('score_quick', ascending=False).iloc[[0]]
        slow = merged.sort_values('score_slow', ascending=False).iloc[[0]]
        others = merged[~merged['student_id'].isin([fast['student_id'].values[0], slow['student_id'].values[0]])]
        gap = others.sort_values('score_gap', ascending=False).iloc[[0]] if not others.empty else merged.iloc[[len(merged)//2]]
        
        reps = [slow, gap, fast]
        colors = ['red', 'orange', 'green']
        
        for rep, color in zip(reps, colors):
            uid = rep['student_id'].values[0]
            traj = skill_preds[skill_preds['student_id'] == uid]
            if len(traj) == 0: continue
            
            # Use actual model predictions
            y_bkt = traj['p_bkt'].values
            y_idkt = traj['p_idkt'].values
            y_true = traj['y_true'].values
            steps = np.arange(1, len(y_true) + 1)
            
            # Plot actual trajectories
            ax.plot(steps, y_bkt, color=color, linestyle='--', linewidth=1.5, alpha=0.4, label='BKT' if i==0 else "")
            ax.plot(steps, y_idkt, color=color, linestyle='-', linewidth=2.0, alpha=0.9, marker='o', markersize=4, label='iDKT' if i==0 else "")
            
            # Plot outcome markers
            for t, val in enumerate(y_true):
                marker = 'o' if val == 1 else 'x'
                ax.scatter(t+1, y_idkt[t], marker=marker, color=color, s=40, zorder=5)
        
        ax.set_title(f"Skill {skill}", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        if i % 5 != 0: ax.set_yticklabels([])
        if i < 10: ax.set_xticklabels([])
        
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='gray', lw=2, ls='--'),
                Line2D([0], [0], color='gray', lw=2, ls='-'),
                Line2D([0], [0], marker='o', color='gray', label='Correct', markersize=6, ls=''),
                Line2D([0], [0], marker='x', color='gray', label='Incorrect', markersize=6, ls='')
            ]
            ax.legend(custom_lines, ['BKT', 'iDKT', 'Correct', 'Incorrect'], loc='upper left', fontsize=8)

    plt.suptitle("Mastery Trajectory Alignment: iDKT (Solid) vs BKT (Dashed)\n(Using Post-Fix Individualized Parameters)", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(args.run_dir, 'plots', 'mastery_alignment_mosaic_real.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Fixed Mosaic Plot saved to {out_path}")

if __name__ == "__main__":
    main()
