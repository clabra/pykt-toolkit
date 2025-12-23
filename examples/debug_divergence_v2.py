
import pandas as pd
import numpy as np
import os

run_dir = 'experiments/20251223_193204_idkt_assist2009_baseline_742098'

def find_most_divergent_students():
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    df = pd.read_csv(rate_path)
    
    # Calculate absolute difference
    df['delta'] = np.abs(df['idkt_rate'] - df['bkt_rate'])
    
    # Get top 20 most divergent interactions
    top_divergent = df.sort_values('delta', ascending=False).head(20)
    
    print("Most Divergent Student-Skill Pairs (Rate):")
    print(top_divergent[['student_id', 'skill_id', 'idkt_rate', 'bkt_rate', 'delta']])

    init_path = os.path.join(run_dir, 'traj_initmastery.csv')
    df_im = pd.read_csv(init_path)
    df_im['delta'] = np.abs(df_im['idkt_im'] - df_im['bkt_im'])
    print("\nMost Divergent Student-Skill Pairs (InitMastery):")
    print(df_im.sort_values('delta', ascending=False).head(20))

find_most_divergent_students()
