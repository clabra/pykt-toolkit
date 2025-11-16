#!/usr/bin/env python3
"""
Generate visualizations for Phase 1 sweep results
Shows parameter impact on Encoder2 AUC
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read results
df = pd.read_csv('/workspaces/pykt-toolkit/examples/sweep_results/phase1_sweep_20251116_174852.csv')
success_df = df[(df['status'] == 'success') & (df['encoder2_test_auc'] > 0)]

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Phase 1 Learning Curve Parameter Sweep - Encoder2 AUC Analysis', 
             fontsize=16, fontweight='bold')

# 1. Distribution of E2 AUC
ax1 = axes[0, 0]
ax1.hist(success_df['encoder2_test_auc'], bins=20, edgecolor='black', alpha=0.7)
ax1.axvline(success_df['encoder2_test_auc'].mean(), color='red', linestyle='--', 
            label=f'Mean: {success_df["encoder2_test_auc"].mean():.4f}')
ax1.axvline(success_df['encoder2_test_auc'].median(), color='blue', linestyle='--',
            label=f'Median: {success_df["encoder2_test_auc"].median():.4f}')
ax1.set_xlabel('Encoder2 Test AUC')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Encoder2 AUC Across All Configurations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Beta impact
ax2 = axes[0, 1]
beta_stats = success_df.groupby('beta_skill_init')['encoder2_test_auc'].agg(['mean', 'std'])
ax2.bar(beta_stats.index, beta_stats['mean'], yerr=beta_stats['std'], 
        capsize=5, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Beta (Learning Rate Amplification)')
ax2.set_ylabel('Mean Encoder2 AUC')
ax2.set_title('Beta Impact (Correlation: +0.72)')
ax2.grid(True, alpha=0.3)
for i, (idx, row) in enumerate(beta_stats.iterrows()):
    ax2.text(idx, row['mean'] + 0.002, f"{row['mean']:.4f}", 
            ha='center', va='bottom', fontweight='bold')

# 3. Offset impact
ax3 = axes[0, 2]
offset_stats = success_df.groupby('sigmoid_offset')['encoder2_test_auc'].agg(['mean', 'std'])
ax3.bar(offset_stats.index, offset_stats['mean'], yerr=offset_stats['std'],
        capsize=5, alpha=0.7, edgecolor='black', color='orange')
ax3.set_xlabel('Sigmoid Offset (Inflection Point)')
ax3.set_ylabel('Mean Encoder2 AUC')
ax3.set_title('Offset Impact (Correlation: -0.54)')
ax3.grid(True, alpha=0.3)
for i, (idx, row) in enumerate(offset_stats.iterrows()):
    ax3.text(idx, row['mean'] + 0.002, f"{row['mean']:.4f}",
            ha='center', va='bottom', fontweight='bold')

# 4. Gamma impact
ax4 = axes[1, 0]
gamma_stats = success_df.groupby('gamma_student_init')['encoder2_test_auc'].agg(['mean', 'std'])
ax4.bar(gamma_stats.index, gamma_stats['mean'], yerr=gamma_stats['std'],
        capsize=5, alpha=0.7, edgecolor='black', color='green')
ax4.set_xlabel('Gamma (Student Learning Velocity)')
ax4.set_ylabel('Mean Encoder2 AUC')
ax4.set_title('Gamma Impact (Correlation: +0.41)')
ax4.grid(True, alpha=0.3)
for i, (idx, row) in enumerate(gamma_stats.iterrows()):
    ax4.text(idx, row['mean'] + 0.002, f"{row['mean']:.4f}",
            ha='center', va='bottom', fontweight='bold')

# 5. M_sat impact
ax5 = axes[1, 1]
msat_stats = success_df.groupby('m_sat_init')['encoder2_test_auc'].agg(['mean', 'std'])
ax5.bar(msat_stats.index, msat_stats['mean'], yerr=msat_stats['std'],
        capsize=5, alpha=0.7, edgecolor='black', color='purple')
ax5.set_xlabel('M_sat (Maximum Mastery Saturation)')
ax5.set_ylabel('Mean Encoder2 AUC')
ax5.set_title('M_sat Impact (Correlation: -0.10)')
ax5.grid(True, alpha=0.3)
for i, (idx, row) in enumerate(msat_stats.iterrows()):
    ax5.text(idx, row['mean'] + 0.002, f"{row['mean']:.4f}",
            ha='center', va='bottom', fontweight='bold')

# 6. Heatmap: Beta vs Offset (most important interaction)
ax6 = axes[1, 2]
pivot = success_df.pivot_table(values='encoder2_test_auc', 
                                index='sigmoid_offset', 
                                columns='beta_skill_init', 
                                aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
            ax=ax6, cbar_kws={'label': 'Encoder2 AUC'})
ax6.set_title('Beta × Offset Interaction')
ax6.set_xlabel('Beta (Learning Rate)')
ax6.set_ylabel('Offset (Inflection Point)')

plt.tight_layout()
plt.savefig('/workspaces/pykt-toolkit/paper/phase1_sweep_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: /workspaces/pykt-toolkit/paper/phase1_sweep_analysis.png")

# Create second figure: Top 10 configurations
fig2, ax = plt.subplots(figsize=(14, 8))
top10 = success_df.nlargest(10, 'encoder2_test_auc')
labels = [f"β={row.beta_skill_init:.1f}, M={row.m_sat_init:.1f}, γ={row.gamma_student_init:.1f}, O={row.sigmoid_offset:.1f}"
          for row in top10.itertuples()]
colors = plt.cm.viridis(np.linspace(0, 1, 10))

bars = ax.barh(range(len(top10)), top10['encoder2_test_auc'], color=colors, edgecolor='black')
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(labels)
ax.set_xlabel('Encoder2 Test AUC', fontsize=12)
ax.set_title('Top 10 Configurations by Encoder2 AUC', fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top10['encoder2_test_auc'])):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2, 
            f'{val:.4f}', va='center', fontweight='bold')

# Add baseline reference
baseline = 0.51
ax.axvline(baseline, color='red', linestyle='--', linewidth=2, 
           label=f'Baseline (~{baseline:.2f})')
ax.legend()

plt.tight_layout()
plt.savefig('/workspaces/pykt-toolkit/paper/phase1_sweep_top10.png', dpi=300, bbox_inches='tight')
print("✅ Saved: /workspaces/pykt-toolkit/paper/phase1_sweep_top10.png")

# Create third figure: Parameter correlations
fig3, ax = plt.subplots(figsize=(10, 6))
params = ['beta_skill_init', 'm_sat_init', 'gamma_student_init', 'sigmoid_offset']
correlations = [success_df[param].corr(success_df['encoder2_test_auc']) for param in params]
param_labels = ['Beta\n(Learning Rate)', 'M_sat\n(Saturation)', 
                'Gamma\n(Student Velocity)', 'Offset\n(Inflection)']

colors_corr = ['green' if c > 0 else 'red' for c in correlations]
bars = ax.bar(param_labels, correlations, color=colors_corr, alpha=0.7, edgecolor='black')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Correlation with Encoder2 AUC', fontsize=12)
ax.set_title('Parameter Correlations with Encoder2 Performance', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(-0.7, 0.8)

# Add value labels
for bar, val in zip(bars, correlations):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + (0.03 if height > 0 else -0.03),
            f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('/workspaces/pykt-toolkit/paper/phase1_sweep_correlations.png', dpi=300, bbox_inches='tight')
print("✅ Saved: /workspaces/pykt-toolkit/paper/phase1_sweep_correlations.png")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated 3 figures:")
print("  1. phase1_sweep_analysis.png - Comprehensive parameter analysis (6 panels)")
print("  2. phase1_sweep_top10.png - Top 10 configurations comparison")
print("  3. phase1_sweep_correlations.png - Parameter correlation summary")
print("\nAll figures saved to: /workspaces/pykt-toolkit/paper/")
