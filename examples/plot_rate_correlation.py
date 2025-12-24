
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def load_rate_data(run_dir):
    rate_path = os.path.join(run_dir, 'traj_rate.csv')
    if not os.path.exists(rate_path):
        print(f"Error: {rate_path} not found.")
        return None
    
    # Check column names
    df = pd.read_csv(rate_path, nrows=5)
    cols = df.columns
    
    # Mapping
    idkt_col = 'idkt_rate' if 'idkt_rate' in df.columns else 'ts'
    bkt_col = 'bkt_rate' if 'bkt_rate' in df.columns else 'T'
    
    if idkt_col not in df.columns or bkt_col not in df.columns:
         # Try legacy names
         if 't_s' in df.columns: idkt_col = 't_s'
         if 'T' in df.columns: bkt_col = 'T'
         
    print(f"Loading data using columns: {idkt_col} (iDKT) vs {bkt_col} (BKT)")
    
    df = pd.read_csv(rate_path, usecols=['skill_id', idkt_col, bkt_col])
    return df, idkt_col, bkt_col

def plot_correlation(df, idkt_col, bkt_col, output_path):
    # Aggregating by skill
    skill_agg = df.groupby('skill_id').agg({
        idkt_col: 'mean',
        bkt_col: 'mean'
    }).reset_index()
    
    # Calculate Correlation
    corr, p = pearsonr(skill_agg[bkt_col], skill_agg[idkt_col])
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=skill_agg, x=bkt_col, y=idkt_col, alpha=0.6, s=80, color='#2980b9')
    
    # Layout
    min_val = min(skill_agg[bkt_col].min(), skill_agg[idkt_col].min())
    max_val = max(skill_agg[bkt_col].max(), skill_agg[idkt_col].max())
    
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Identity Line')
    
    plt.title(f"Learning Rate Alignment (r={corr:.3f})\nBKT (T) vs iDKT (mean $t_s$) per Skill", fontsize=14)
    plt.xlabel("Theoretical BKT Learning Rate (T)", fontsize=12)
    plt.ylabel("iDKT Learning Velocity ($v_s$)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_val * 1.1)
    plt.ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved correlation plot to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    
    df, c1, c2 = load_rate_data(args.run_dir)
    if df is not None:
        os.makedirs(os.path.join(args.run_dir, 'plots'), exist_ok=True)
        out_path = os.path.join(args.run_dir, 'plots', 'idkt_rate_correlation.png')
        plot_correlation(df, c1, c2, out_path)

if __name__ == "__main__":
    main()
