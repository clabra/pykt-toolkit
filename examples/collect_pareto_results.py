#!/usr/bin/env python3
"""
Collect and visualize results from the iDKT Pareto Frontier sweep.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse

def collect_results(exp_root="experiments", pattern="*idkt_pareto_v2*"):
    results = []
    exp_dirs = glob.glob(os.path.join(exp_root, pattern))
    
    print(f"Found {len(exp_dirs)} experiment directories matching pattern.")
    
    for run_dir in exp_dirs:
        config_path = os.path.join(run_dir, "config.json")
        eval_path = os.path.join(run_dir, "eval_results.json")
        align_path = os.path.join(run_dir, "interpretability_alignment.json")
        
        if not (os.path.exists(config_path) and os.path.exists(eval_path)):
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            with open(eval_path, 'r') as f:
                eval_res = json.load(f)
                
            # Basic metrics
            # Look for lambda_ref in overrides first, then input, then top level
            lambda_val = config.get('overrides', {}).get('lambda_ref')
            if lambda_val is None:
                lambda_val = config.get('input', {}).get('lambda_ref')
            if lambda_val is None:
                lambda_val = config.get('lambda_ref', 0.0)
                
            row = {
                'lambda': float(lambda_val),
                'test_auc': eval_res.get('test_auc', 0.0),
                'valid_auc': eval_res.get('valid_auc', 0.0),
                'exp_id': config.get('experiment', {}).get('experiment_id', 'unknown'),
                'dir': os.path.basename(run_dir)
            }
            
            # Interpretability metrics if available
            if os.path.exists(align_path):
                with open(align_path, 'r') as f:
                    align = json.load(f)
                row['pred_corr'] = align.get('prediction_corr', 0.0)
                row['init_corr'] = align.get('initmastery_corr', 0.0)
                row['rate_corr'] = align.get('learning_rate_corr', 0.0)
                row['h2_functional'] = align.get('h2_functional_alignment', 0.0)
                row['h3_discriminant'] = align.get('h3_discriminant_overlap', 0.0)
                row['h3_latent'] = align.get('h3_latent_overlap', 0.0)
                row['mean_corr'] = (row['pred_corr'] + row['init_corr'] + row['rate_corr']) / 3.0
            else:
                row['pred_corr'] = row['init_corr'] = row['rate_corr'] = row['mean_corr'] = None
                row['h2_functional'] = row['h3_discriminant'] = row['h3_latent'] = None
                
            results.append(row)
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('lambda').reset_index(drop=True)
    return df

def plot_pareto(df, output_dir="assistant"):
    if df.empty:
        print("No results to plot.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Performance vs Interpretability Pareto Frontier
    plt.figure(figsize=(10, 7))
    
    # Filter for completed alignment data
    plot_df = df.dropna(subset=['mean_corr'])
    
    if plot_df.empty:
        print("No alignment data found for Pareto plot. Generating AUC vs Lambda only.")
    else:
        scatter = plt.scatter(plot_df['mean_corr'], plot_df['test_auc'], 
                             c=plot_df['lambda'], cmap='viridis', s=100, edgecolors='black')
        plt.colorbar(scatter, label='Grounding Strength (λ)')
        plt.xlabel('Theoretic Fidelity (Mean Correlation with BKT)', fontsize=12)
        plt.ylabel('Predictive Performance (Test AUC)', fontsize=12)
        plt.title('iDKT Pareto Frontier: Performance vs. Interpretability', fontsize=14, fontweight='bold')
        
        # Annotate points with lambda
        for idx, row in plot_df.iterrows():
            plt.annotate(f"λ={row['lambda']:.2f}", (row['mean_corr'], row['test_auc']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
                        
        plt.savefig(os.path.join(output_dir, "idkt_pareto_frontier.png"), dpi=300, bbox_inches='tight')
        print(f"✓ Saved Pareto plot to {output_dir}/idkt_pareto_frontier.png")

    # 2. Performance & Fidelity vs Lambda Trend
    plt.figure(figsize=(10, 6))
    plt.plot(df['lambda'], df['test_auc'], 'o-', label='Test AUC', color='tab:blue', linewidth=2)
    if 'mean_corr' in df.columns and not df['mean_corr'].isnull().all():
        plt.plot(df['lambda'], df['mean_corr'], 's--', label='Mean Fidelity', color='tab:green', linewidth=2)
        
    plt.xlabel('Grounding Weight (λ)', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('iDKT Sensitivity Analysis: AUC and Fidelity vs λ', fontsize=14, fontweight='bold')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(output_dir, "idkt_sensitivity_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved Sensitivity plot to {output_dir}/idkt_sensitivity_analysis.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="experiments")
    parser.add_argument("--pattern", default="*idkt_pareto_v2*")
    parser.add_argument("--output", default="assistant/pareto_results.csv")
    args = parser.parse_args()
    
    df = collect_results(args.exp_root, args.pattern)
    if not df.empty:
        df.to_csv(args.output, index=False)
        print(f"✓ Saved results CSV to {args.output}")
        plot_pareto(df)
    else:
        print("No matching results found.")

if __name__ == "__main__":
    main()
