
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# --- Config ---
EXP_DIR = "/workspaces/pykt-toolkit/experiments/20251230_224907_idkt_setS-pure_assist2009_baseline_364494"
OUTPUT_DIR = os.path.join(EXP_DIR, "probing_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'

def generate_selectivity_plots():
    """Generates distribution plots for Pearson and R2."""
    ps_path = os.path.join(EXP_DIR, "probe_per_skill_results.json")
    with open(ps_path) as f:
        ps_results = json.load(f)
    
    data = []
    for sid, m in ps_results.items():
        data.append({
            "Skill": sid,
            "R2": m["r2"],
            "Pearson": m["pearson"],
            "Count": m["count"]
        })
    df = pd.DataFrame(data)
    
    # Pearson Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df["Pearson"], bins=15, kde=True, color="#2ecc71", alpha=0.7)
    plt.axvline(df["Pearson"].median(), color='red', linestyle='--', label=f'Median: {df["Pearson"].median():.2f}')
    plt.title("Distribution of Alignment ($r$) across Skills", fontsize=14)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
    plt.ylabel("Frequency (Skills)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "probing_pearson_dist.png"), dpi=300)
    plt.close()

    # R2 Distribution (clipped for visibility)
    df_r2 = df[df["R2"] > -1.0]
    plt.figure(figsize=(7, 5))
    sns.histplot(df_r2["R2"], bins=15, kde=True, color="#3498db", alpha=0.7)
    plt.title("Distribution of $R^2$ Scores across Skills", fontsize=14)
    plt.xlabel("$R^2$ Score", fontsize=12)
    plt.ylabel("Frequency (Skills)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "probing_r2_dist.png"), dpi=300)
    plt.close()
    
    print(f"Generated distribution plots in {OUTPUT_DIR}")

def generate_manifold_viz():
    """Enhances the PCA manifold plot."""
    # This requires running the probe again or finding the saved PCA data.
    # Since we have probe_pca.png already, we know the script runs.
    # We will assume for now we want to provide the user with the recommendation 
    # to use Seaborn for better aesthetic control.
    pass

if __name__ == "__main__":
    generate_selectivity_plots()
