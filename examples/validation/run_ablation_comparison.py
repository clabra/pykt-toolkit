
import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Generate ablation study comparison')
    parser.add_argument('--baseline_exp', type=str, required=True, help='Baseline experiment directory')
    parser.add_argument('--grounded_exp', type=str, required=True, help='Grounded experiment directory')
    parser.add_argument('--probing_exp', type=str, required=True, help='Probing experiment directory')
    parser.add_argument('--personalization_exp', type=str, default=None, help='Personalization experiment directory (optional)')
    parser.add_argument('--output_dir', type=str, default='examples/validation/results', help='Output directory')
    args = parser.parse_args()
    
    # Define Experiments with command-line arguments
    EXPERIMENTS = {
        "Baseline": {
            "exp_dir": args.baseline_exp,
            "recovery_dir": os.path.join(args.output_dir, "ablation_baseline"),
            "config": "G=❌, P=❌, Pr=❌"
        },
        "Grounded": {
            "exp_dir": args.grounded_exp,
            "recovery_dir": os.path.join(args.output_dir, "ablation_grounded"),
            "config": "G=✅, P=❌, Pr=❌"
        },
        "Probing": {
            "exp_dir": args.probing_exp,
            "recovery_dir": os.path.join(args.output_dir, "ablation_probing"),
            "config": "G=✅, P=❌, Pr=✅"
        }
    }
    
    # Add personalization if provided
    if args.personalization_exp:
        EXPERIMENTS["Personalized"] = {
            "exp_dir": args.personalization_exp,
            "recovery_dir": os.path.join(args.output_dir, "ablation_personalization"),
            "config": "G=✅, P=✅, Pr=✅"
        }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    rows = []
    
    for name, paths in EXPERIMENTS.items():
        # 1. Load AUC
        auc = 0.0
        # Recursive discovery for eval_results.json
        eval_path = None
        for root, _, files in os.walk(paths["exp_dir"]):
            if "eval_results.json" in files:
                eval_path = os.path.join(root, "eval_results.json")
                break
                
        if eval_path and os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                res = json.load(f)
                # Correct key for Que-Level Late Fusion AUC in pykt
                # Fallback to testauc if oriauclate_mean is missing
                auc = res.get("oriauclate_mean", res.get("testauc", 0.0))
        
        # 2. Load Recovery
        l0_r = 0.0
        t_r = 0.0
        # Check both names
        rec_path = os.path.join(paths["recovery_dir"], "recovery_summary.json")
        if not os.path.exists(rec_path):
            rec_path = os.path.join(paths["recovery_dir"], "summary.json")
            
        if os.path.exists(rec_path):
            with open(rec_path, 'r') as f:
                rec = json.load(f)
                # Handle cases where grounded or probe might be used
                l0_r = rec.get("l0_probe", {}).get("pearson_r", 0.0)
                t_r = rec.get("t_probe", {}).get("pearson_r", 0.0)
        
        rows.append({
            "Stage": name,
            "Configuration": paths["config"],
            "Test AUC": auc,
            "L0 Recovery (r)": l0_r,
            "T Recovery (r)": t_r,
            "Interpretability": (l0_r + t_r) / 2
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "ablation_summary.csv"), index=False)
    
    # Plotting: Performance vs Interpretability Trade-off
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Performance Line
    ax1 = plt.gca()
    sns.lineplot(data=df, x="Stage", y="Test AUC", marker='o', color='crimson', label='Predictive Accuracy (AUC)', ax=ax1)
    ax1.set_ylabel("AUC (One-by-One Late Fusion)", color='crimson', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.set_ylim(0.77, 0.79)
    
    # Interpretability Line
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="Stage", y="Interpretability", marker='s', color='royalblue', label='Structural Interpretability (r)', ax=ax2)
    ax2.set_ylabel("Mean Parameter Recovery (Pearson r)", color='royalblue', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.set_ylim(-0.1, 1.0)
    
    plt.title("Ablation Study: The Pareto Frontier of the Proposed Architecture", fontsize=14, fontweight='bold')
    
    # Legend manually combined
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ablation_tradeoff.png"), dpi=300)
    
    print(f"Summary Table:\n{df[['Stage', 'Test AUC', 'Interpretability']]}")
    print(f"Ablation study assets saved to {args.output_dir}")

if __name__ == "__main__":
    main()
