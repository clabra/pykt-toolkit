import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, mean_absolute_error

def generate_validation_plots(run_dir):
    print(f"Generating validation plots for {run_dir}...")
    
    plots_dir = os.path.join(run_dir, "plots", "validation")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    pred_path = os.path.join(run_dir, "traj_predictions.csv")
    roster_path = os.path.join(run_dir, "roster_bkt.csv")
    
    if not os.path.exists(pred_path) or not os.path.exists(roster_path):
        print(f"Error: Required files missing. (pred: {os.path.exists(pred_path)}, roster: {os.path.exists(roster_path)})")
        return

    df_pred = pd.read_csv(pred_path)
    df_roster = pd.read_csv(roster_path)
    
    # Standardize column naming if necessary (handling NEW/OLD names)
    if 'Mi' in df_pred.columns: df_pred = df_pred.rename(columns={'Mi': 'p_idkt', 'M_rasch': 'p_bkt'})
    
    # Reconstruct 'step' in df_pred because it was missing in the export
    # We assume the CSV order is step-sequential per student_id
    df_pred['step'] = df_pred.groupby('student_id').cumcount() + 1
    
    # Melt roster to get mastery for the ACTIVE skill
    # roster_bkt.csv format: student_id, step, skill_id, correct, S0, S1, ...
    # We want a 'mastery' column that corresponds to the skill_id of that interaction
    skills = [col for col in df_roster.columns if col.startswith('S')]
    
    # To optimize, we can pivot or just iterate
    def get_latent_mastery(row):
        col_name = f"S{int(row['skill_id'])}"
        if col_name in row:
            return row[col_name]
        return None
        
    df_roster['bkt_latent_mastery'] = df_roster.apply(get_latent_mastery, axis=1)
    
    # Merge
    df = pd.merge(df_pred, df_roster[['student_id', 'step', 'skill_id', 'correct', 'bkt_latent_mastery']], 
                  on=['student_id', 'step', 'skill_id'], how='inner')
    
    if df.empty:
        print("Error: Merged dataframe is empty. Check student_id/step/skill_id alignment.")
        return

    # --- 1. Multi-Model Consensus Agreement (Confidence Mapping) ---
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x='bkt_latent_mastery', y='p_idkt', fill=True, cmap="Greens", thresh=0, levels=15)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', alpha=0.5, label='Identity Line')
    plt.xlabel('BKT Latent Mastery P(L_t)')
    plt.ylabel('iDKT Predicted Correctness P(r_t+1)')
    plt.title('Multi-Model Consensus Agreement (Density Plot)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(plots_dir, "consensus_agreement_density.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved Consensus Agreement: {save_path}")

    # --- 2. Theoretical Residual vs. AUC (Validating BKT Skill-Fit) ---
    skill_metrics = []
    for skill_id, group in df.groupby('skill_id'):
        if len(group) < 10 or len(group['correct'].unique()) < 2:
            continue
            
        mae = mean_absolute_error(group['p_bkt'], group['p_idkt'])
        try:
            auc = roc_auc_score(group['correct'], group['p_bkt'])
        except:
            continue
            
        skill_metrics.append({
            'skill_id': skill_id,
            'mae': mae,
            'bkt_auc': auc
        })
        
    df_skills = pd.DataFrame(skill_metrics)
    if not df_skills.empty:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_skills, x='bkt_auc', y='mae', size='mae', hue='mae', palette='viridis', legend=False)
        plt.xlabel('BKT Prediction Performance (AUC per Skill)')
        plt.ylabel('Model-Theory Divergence (MAE between iDKT and BKT)')
        plt.title('Theoretical Residual vs. AUC (Skill-Fit Validation)')
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(plots_dir, "theoretical_residual_vs_auc.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved Theoretical Residual vs AUC: {save_path}")
    else:
        print("Skipped Theoretical Residual plot (insufficient per-skill data)")

    # --- 3. Cross-Model Uncertainty Intervals (iDKT-based BKT Bounds) ---
    # Bin BKT mastery into 10 intervals
    df['mastery_bin'] = pd.cut(df['bkt_latent_mastery'], bins=np.linspace(0, 1, 11), labels=False)
    
    bin_stats = df.groupby('mastery_bin')['p_idkt'].agg(['mean', 'std']).reset_index()
    bin_stats['bin_center'] = bin_stats['mastery_bin'] / 10 + 0.05
    
    plt.figure(figsize=(10, 8))
    # Plot BKT Theoretical Line (assuming p_bkt tracks latent mastery closely, or just the identity as a reference)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Theoretical BKT Identity')
    
    # Plot iDKT Mean and Variance
    plt.errorbar(bin_stats['bin_center'], bin_stats['mean'], yerr=bin_stats['std'], fmt='o', color='forestgreen', 
                 ecolor='lightgreen', elinewidth=3, capsize=5, label='iDKT Mean ± Std')
    
    plt.fill_between(bin_stats['bin_center'], bin_stats['mean'] - bin_stats['std'], 
                     bin_stats['mean'] + bin_stats['std'], color='lightgreen', alpha=0.3)
    
    plt.xlabel('BKT Latent Mastery Level')
    plt.ylabel('iDKT Predicted Correctness Distribution')
    plt.title('Cross-Model Uncertainty Intervals (iDKT-based BKT Bounds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(plots_dir, "uncertainty_intervals.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved Uncertainty Intervals: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Experiment results directory")
    args = parser.parse_args()
    
    generate_validation_plots(args.run_dir)
