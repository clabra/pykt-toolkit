
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_param_data(run_dir, param_type='rate'):
    if param_type == 'rate':
        path = os.path.join(run_dir, 'traj_rate.csv')
        idkt_keywords = ['idkt_rate', 'ts', 't_s']
        bkt_keywords = ['bkt_rate', 'T', 't_bkt']
        title_tag = "Learning Velocity"
        label_tag = "Learning Rate"
    elif param_type == 'im':
        path = os.path.join(run_dir, 'traj_initmastery.csv')
        idkt_keywords = ['idkt_im', 'im', 'lc']
        bkt_keywords = ['bkt_im', 'L0', 'l0_bkt']
        title_tag = "Initial Mastery"
        label_tag = "Mastery Prior"
    else:
        return None, None, None, None, None
        
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None, None, None, None, None
        
    df = pd.read_csv(path)
    
    # Identify columns
    idkt_col = next((c for c in idkt_keywords if c in df.columns), None)
    bkt_col = next((c for c in bkt_keywords if c in df.columns), None)
    
    return df, idkt_col, bkt_col, title_tag, label_tag

def plot_ribbon(df, idkt_col, bkt_col, output_path, title_tag, label_tag):
    # Sort based on BKT value
    mean_bkt = df.groupby('skill_id')[bkt_col].mean()
    sorted_skills = mean_bkt.sort_values().index.tolist()
    
    # Calculate stats
    stats = df.groupby('skill_id')[idkt_col].agg(
        low=lambda x: np.percentile(x, 5),
        median='median',
        high=lambda x: np.percentile(x, 95)
    )
    
    # Enforce strict alignment
    stats = stats.reindex(sorted_skills)
    bkt_vals = mean_bkt.reindex(sorted_skills)
    
    plt.figure(figsize=(14, 6))
    x = np.arange(len(sorted_skills))
    
    y_low = stats['low'].values
    y_high = stats['high'].values
    y_median = stats['median'].values
    y_bkt = bkt_vals.values
    
    # 1. Plot individualization sample (Scatter) - to show the "points with variance"
    # To avoid overwhelming, sample students or use alpha
    # We'll take a sample of interaction records to show the spread
    sample_df = df.sample(min(len(df), 5000), random_state=42)
    # Map skill_id to its x-index
    skill_to_x = {s: i for i, s in enumerate(sorted_skills)}
    sample_x = sample_df['skill_id'].map(skill_to_x).values
    sample_y = sample_df[idkt_col].values
    
    # Add horizontal jitter to avoid vertical lines
    jitter = np.random.uniform(-0.3, 0.3, size=len(sample_x))
    plt.scatter(sample_x + jitter, sample_y, color='#3498db', s=1, alpha=0.15, label='Individual Students', zorder=0)

    # 2. Plot BKT FIRST (Background) - THICK BLACK DASHED
    plt.step(x, y_bkt, where='mid', color='black', linestyle='--', linewidth=2.5, label=f'Theoretical BKT ({bkt_col})', alpha=0.5, zorder=1)
    
    # 3. Plot iDKT SECOND (Foreground Envelope)
    plt.fill_between(x, y_low, y_high, alpha=0.4, color='#3498db', label=f'iDKT 5th-95th Percentile Range', zorder=2)
    plt.plot(x, y_median, color='#2980b9', linewidth=1.5, label='iDKT Median', zorder=3)
    
    plt.title(f"Option 1: Quantile Ribbon ({title_tag} Envelope)", fontsize=14)
    plt.xlabel(f"Skills (Sorted by Theoretical {bkt_col})", fontsize=12)
    plt.ylabel(label_tag, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved ribbon plot to {output_path}")

def plot_delta(df, idkt_col, bkt_col, output_path, title_tag, label_tag):
    # Calculate delta
    df['delta'] = df[idkt_col] - df[bkt_col]
    
    plt.figure(figsize=(14, 6))
    
    sns.histplot(data=df, x='delta', bins=100, kde=True, color='#8e44ad', alpha=0.6)
    
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Theoretical Baseline (Delta=0)')
    
    mu = df['delta'].mean()
    plt.axvline(mu, color='blue', linestyle=':', linewidth=2, label=f'Mean Delta ({mu:.3f})')
    
    plt.title(f"Option 2: Delta Distribution ({title_tag} Bias)", fontsize=14)
    plt.xlabel(f"Delta (iDKT {label_tag} - BKT {label_tag})", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved delta plot to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    
    plots_dir = os.path.join(args.run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Handle Rate
    df_rate, c1, c2, t_tag, l_tag = load_param_data(args.run_dir, 'rate')
    if df_rate is not None:
        plot_ribbon(df_rate, c1, c2, os.path.join(plots_dir, 'param_rate_alt_ribbon.png'), t_tag, l_tag)
        plot_delta(df_rate, c1, c2, os.path.join(plots_dir, 'param_rate_alt_delta.png'), t_tag, l_tag)
        
    # Handle Initial Mastery
    df_im, c1, c2, t_tag, l_tag = load_param_data(args.run_dir, 'im')
    if df_im is not None:
        plot_ribbon(df_im, c1, c2, os.path.join(plots_dir, 'param_im_alt_ribbon.png'), t_tag, l_tag)
        plot_delta(df_im, c1, c2, os.path.join(plots_dir, 'param_im_alt_delta.png'), t_tag, l_tag)

if __name__ == "__main__":
    main()
