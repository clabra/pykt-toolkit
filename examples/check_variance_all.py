
import pandas as pd
import numpy as np
import os

# List of interesting experiments to check for individualization variance
exp_to_check = [
    'experiments/20251223_193204_idkt_assist2009_baseline_742098',
    'experiments/20251221_101701_idkt_pareto_v2_l0.10_assist2009_839009',
    'experiments/20251221_021700_idkt_individual_assist2009_623104',
    'experiments/20251219_165430_idkt_pareto_highres_l0.00_854257'
]

def check_variance(run_dir):
    print(f"\n=== Checking: {run_dir} ===")
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    if not os.path.exists(rate_path):
        print("traj_rate.csv not found")
        return
    
    df = pd.read_csv(rate_path)
    idkt_col = 'idkt_rate' if 'idkt_rate' in df.columns else 'ts'
    bkt_col = 'bkt_rate' if 'bkt_rate' in df.columns else 'T'
    
    if idkt_col not in df.columns or bkt_col not in df.columns:
        print(f"Columns not found: {idkt_col}, {bkt_col}")
        return
        
    df['diff'] = (df[idkt_col] - df[bkt_col]).abs()
    print(f"Mean Absolute Difference (iDKT vs BKT): {df['diff'].mean():.8f}")
    print(f"Max Absolute Difference: {df['diff'].max():.8f}")
    
    # Ribbon width (90% range)
    stats = df.groupby('skill_id')[idkt_col].agg(lambda x: np.percentile(x, 95) - np.percentile(x, 5))
    print(f"Mean Ribbon Width: {stats.mean():.8f}")
    print(f"Max Ribbon Width: {stats.max():.8f}")

for exp in exp_to_check:
    check_variance(exp)
