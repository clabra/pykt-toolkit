
import pandas as pd
import numpy as np
import os

run_dir = 'experiments/20251223_193204_idkt_assist2009_baseline_742098'

def find_extreme_skill():
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    df = pd.read_csv(rate_path)
    
    # Calculate ribbon width per skill
    stats = df.groupby('skill_id')['idkt_rate'].agg(lambda x: np.percentile(x, 95) - np.percentile(x, 5)).sort_values(ascending=False)
    
    top_skill = stats.index[0]
    width = stats.iloc[0]
    
    print(f"Top Variance Skill: {top_skill} (Width: {width:.6f})")
    
    # Get students for this skill
    df_skill = df[df['skill_id'] == top_skill]
    
    # Pick 5 students with highest and 5 with lowest idkt_rate
    unique_students = df_skill[['student_id', 'idkt_rate', 'bkt_rate']].drop_duplicates()
    
    print("\nSample Students (Sorted by iDKT Rate):")
    print(unique_students.sort_values('idkt_rate').head(5))
    print("...")
    print(unique_students.sort_values('idkt_rate', ascending=False).head(5))

find_extreme_skill()
