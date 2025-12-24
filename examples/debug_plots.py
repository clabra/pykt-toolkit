
import pandas as pd
import numpy as np
import os

run_dir = 'experiments/20251223_193204_idkt_assist2009_baseline_742098'

def check_rate():
    path = os.path.join(run_dir, 'traj_rate.csv')
    if not os.path.exists(path):
        print(f"{path} not found")
        return
    df = pd.read_csv(path)
    print(f"--- Rate Stats ({path}) ---")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check idkt_rate and bkt_rate
    cols = [c for c in ['idkt_rate', 'bkt_rate', 'ts', 'T'] if c in df.columns]
    print(df[cols].describe())
    
    idkt_col = 'idkt_rate' if 'idkt_rate' in df.columns else 'ts'
    bkt_col = 'bkt_rate' if 'bkt_rate' in df.columns else 'T'
    
    mean_bkt = df.groupby('skill_id')[bkt_col].mean()
    sorted_skills = mean_bkt.sort_values().index.tolist()
    
    stats = df.groupby('skill_id')[idkt_col].agg(
        low=lambda x: np.percentile(x, 5),
        median='median',
        high=lambda x: np.percentile(x, 95)
    )
    
    stats_reindexed = stats.reindex(sorted_skills)
    print(f"Stats shape: {stats.shape}, Reindexed shape: {stats_reindexed.shape}")
    print(f"NaNs in reindexed median: {stats_reindexed['median'].isnull().sum()}")
    print(f"Zero values in reindexed median: {(stats_reindexed['median'] == 0).sum()}")
    
    if stats_reindexed['median'].isnull().sum() > 0:
        missing = stats_reindexed[stats_reindexed['median'].isnull()].index.tolist()
        print(f"Missing skills in iDKT stats: {missing[:5]}...")

def check_im():
    path = os.path.join(run_dir, 'traj_initmastery.csv')
    if not os.path.exists(path):
        print(f"{path} not found")
        return
    df = pd.read_csv(path)
    print(f"\n--- IM Stats ({path}) ---")
    print(f"Columns: {df.columns.tolist()}")
    cols = [c for c in ['idkt_im', 'bkt_im', 'im', 'L0'] if c in df.columns]
    print(df[cols].describe())

check_rate()
check_im()
