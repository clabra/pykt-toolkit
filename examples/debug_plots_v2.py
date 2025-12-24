
import pandas as pd
import numpy as np
import os

run_dir = 'experiments/20251223_193204_idkt_assist2009_baseline_742098'

def check_rate_variance():
    path = os.path.join(run_dir, 'traj_rate.csv')
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    idkt_col = 'idkt_rate'
    bkt_col = 'bkt_rate'
    
    # Difference stats
    df['diff'] = df[idkt_col] - df[bkt_col]
    print(f"--- Rate Difference Stats (iDKT - BKT) ---")
    print(df['diff'].describe())
    
    # Ribbon width stats per skill
    stats = df.groupby('skill_id')[idkt_col].agg(
        low=lambda x: np.percentile(x, 5),
        high=lambda x: np.percentile(x, 95)
    )
    stats['width'] = stats['high'] - stats['low']
    print(f"\n--- Ribbon Width Stats (90% range per skill) ---")
    print(stats['width'].describe())
    
    # Check a few specific skills
    print(f"\nTop 5 skills by ribbon width:")
    print(stats.sort_values('width', ascending=False).head(5))

def check_im_variance():
    path = os.path.join(run_dir, 'traj_initmastery.csv')
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    idkt_col = 'idkt_im'
    bkt_col = 'bkt_im'
    
    df['diff'] = df[idkt_col] - df[bkt_col]
    print(f"\n--- IM Difference Stats (iDKT - BKT) ---")
    print(df['diff'].describe())
    
    stats = df.groupby('skill_id')[idkt_col].agg(
        low=lambda x: np.percentile(x, 5),
        high=lambda x: np.percentile(x, 95)
    )
    stats['width'] = stats['high'] - stats['low']
    print(f"\n--- IM Ribbon Width Stats ---")
    print(stats['width'].describe())

check_rate_variance()
check_im_variance()
