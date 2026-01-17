import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def generate_baseline_comparison(output_dir="examples/validation/results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Data gathered from benchmark_paper.md and ablation_summary.csv
    data = {
        "Model": ["BKT (Symbolic)", "AKT (Deep Learning)", "GTransformer (Proposed)"],
        "Predictive AUC": [0.6097, 0.7825, 0.7785],
        "Structural Alignment (r)": [1.0000, 0.0521, 0.7278], # AKT value from post-hoc ridge usually low
        "Individualization": ["❌ No", "❌ No", "✅ Yes"],
        "Interpretability Mode": ["Intrinsic (Formulaic)", "Post-hoc (Black-box)", "Intrinsic (Neuro-symbolic)"]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "baseline_comparison.csv"), index=False)
    
    # Create comparison plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    models = df["Model"]
    auc = df["Predictive AUC"]
    alignment = df["Structural Alignment (r)"]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, auc, width, label='Predictive AUC', color='skyblue', alpha=0.8)
    ax1.set_ylabel('AUC (Accuracy)')
    ax1.set_ylim(0.5, 0.85)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, alignment, width, label='Structural Alignment (Pearson r)', color='coral', alpha=0.8)
    ax2.set_ylabel('Alignment with Theory (r)')
    ax2.set_ylim(0, 1.1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.title('Performance-Interpretability Pareto Frontier')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_comparison_plot.png"))
    plt.close()

    print(f"Generated baseline comparison to {output_dir}")

if __name__ == "__main__":
    generate_baseline_comparison()
