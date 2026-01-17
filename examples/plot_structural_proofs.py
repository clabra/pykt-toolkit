
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import seaborn as sns

def plot_compass(ax, v_data, w_theory, title, color_data='blue', color_theory='red'):
    """Draws the Unit Circle Compass Plot for Structural Alignment."""
    # Circle
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_artist(circle)
    
    # Vectors
    # Data Axis (PC1)
    ax.arrow(0, 0, v_data[0], v_data[1], head_width=0.05, head_length=0.1, 
             fc=color_data, ec=color_data, linewidth=2, label='Data Axis (PC1)')
    
    # Theory Axis (Probe)
    ax.arrow(0, 0, w_theory[0], w_theory[1], head_width=0.05, head_length=0.1, 
             fc=color_theory, ec=color_theory, linewidth=2, label='Theory Axis (Probe)')
    
    # Calculate Angle
    cos_sim = np.abs(np.dot(v_data, w_theory))
    angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    
    # Text Annotation
    ax.text(-0.9, 0.9, f"$\\theta = {angle_deg:.1f}^\\circ$\n$S_{{align}}={cos_sim:.3f}$", 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    # Only show legend on the first one usually, but here we can just skip or add custom
    
def plot_sensitivity(ax, curves, labels, colors):
    """Plots the Causal Sensitivity Curves (Delta vs Probability)."""
    for curve, label, color in zip(curves, labels, colors):
        deltas = sorted(curve.keys())
        probs = [curve[d] for d in deltas]
        
        # Normalize probs (delta from mean) for fair comparison?
        # Or raw probability? Raw is better to show "it actually increases".
        
        sns.lineplot(x=deltas, y=probs, ax=ax, label=label, color=color, marker='o', linewidth=2.5)
        
    ax.set_title("Causal Sensitivity (Intervention)", fontsize=14, pad=10)
    ax.set_xlabel("Intervention Magnitude ($\\delta$ stds)", fontsize=12)
    ax.set_ylabel("Predicted Probability $P(Correct)$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_scree(ax, variances, labels, colors):
    """Plots the Scree Plot (Dimensional Collapse)."""
    for var, label, color in zip(variances, labels, colors):
        comps = np.arange(1, len(var) + 1)
        cum_var = np.cumsum(var)
        
        sns.lineplot(x=comps, y=cum_var, ax=ax, label=label, color=color, marker='s', linewidth=2)
        
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='90% Variance')
    ax.set_title("Dimensional Collapse (Scree Plot)", fontsize=14, pad=10)
    ax.set_xlabel("Principal Component Index", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(comps)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", type=str, required=True, help="Pickle file for Active Grounding model")
    parser.add_argument("--baseline", type=str, required=True, help="Pickle file for Baseline model")
    parser.add_argument("--output_file", type=str, default="structural_proofs_panel.png", help="Output PNG")
    args = parser.parse_args()
    
    # Load Data
    with open(args.active, 'rb') as f:
        d_active = pickle.load(f)
    with open(args.baseline, 'rb') as f:
        d_baseline = pickle.load(f)
        
    sns.set_style("whitegrid")
    
    # Create Panel Figure
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 4)
    
    # 1. Compass Plots (L0) - Active vs Baseline
    ax1 = fig.add_subplot(gs[0, 0])
    
    # For 2D plotting, we need to project the high-dim vectors onto the PC1-Probe Plane.
    # Actually, S_align defines the relationship directly.
    # We can simulate the 2D view by having Vector A = (1, 0) and Vector B = (cos_theta, sin_theta).
    
    def get_2d_vectors(s_align):
        theta = np.arccos(np.clip(s_align, -1, 1))
        return (1.0, 0.0), (np.cos(theta), np.sin(theta))
    
    # Active
    v_a, w_a = get_2d_vectors(d_active['alignment_l0']['s_align'])
    plot_compass(ax1, v_a, w_a, "Active Grounding\n(Structural Alignment)")
    
    # Baseline
    ax2 = fig.add_subplot(gs[0, 1])
    v_b, w_b = get_2d_vectors(d_baseline['alignment_l0']['s_align'])
    plot_compass(ax2, v_b, w_b, "Baseline Model\n(Random Orientation)")
    
    # 3. Dimensional Collapse (Scree)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_scree(ax3, 
               [d_active['alignment_l0']['explained_variance'], d_baseline['alignment_l0']['explained_variance']],
               ['Active', 'Baseline'],
               ['#2ca02c', '#d62728']) # Green, Red
               
    # 4. Sensitivity
    ax4 = fig.add_subplot(gs[0, 3])
    plot_sensitivity(ax4, 
                     [d_active['sensitivity_l0'], d_baseline['sensitivity_l0']],
                     ['Active', 'Baseline'],
                     ['#2ca02c', '#d62728'])
                     
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Generated Structural Proof Panel: {args.output_file}")

if __name__ == "__main__":
    main()
