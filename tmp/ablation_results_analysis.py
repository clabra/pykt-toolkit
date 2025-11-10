#!/usr/bin/env python3
"""
Ablation Study Results Analysis: Loss Function Necessity
Compares three configurations to determine which losses are redundant.
"""

import json
import numpy as np
from scipy import stats

# Configuration paths
configs = {
    'A_current': '/workspaces/pykt-toolkit/examples/experiments/20251110_201758_gainakt2exp_ablation_current_401632/eval_results.json',
    'B_simplified': '/workspaces/pykt-toolkit/examples/experiments/20251110_201954_gainakt2exp_ablation_simplified_591617/eval_results.json',
    'C_hybrid': '/workspaces/pykt-toolkit/examples/experiments/20251110_202102_gainakt2exp_ablation_hybrid_375411/eval_results.json'
}

# Load results
results = {}
for name, path in configs.items():
    with open(path) as f:
        results[name] = json.load(f)

# Extract key metrics
metrics = ['test_auc', 'test_acc', 'test_mastery_correlation', 'test_gain_correlation']

print("=" * 100)
print("ABLATION STUDY RESULTS: LOSS FUNCTION NECESSITY")
print("=" * 100)
print()
print("Configuration Details:")
print("  A (Current):    mastery_weight=1.5, gain_weight=0.8, consistency_weight=0.3")
print("  B (Simplified): mastery_weight=1.5, gain_weight=0.0, consistency_weight=0.0")
print("  C (Hybrid):     mastery_weight=1.5, gain_weight=0.8, consistency_weight=0.0")
print()

# Print comparison table
print("=" * 100)
print("METRIC COMPARISON")
print("=" * 100)
print()
print(f"{'Metric':<30} {'A: Current':<18} {'B: Simplified':<18} {'C: Hybrid':<18}")
print("-" * 100)
for metric in metrics:
    a_val = results['A_current'][metric]
    b_val = results['B_simplified'][metric]
    c_val = results['C_hybrid'][metric]
    print(f"{metric:<30} {a_val:<18.6f} {b_val:<18.6f} {c_val:<18.6f}")

print()
print("=" * 100)
print("DELTA ANALYSIS (relative to A: Current)")
print("=" * 100)
print()

print(f"{'Metric':<30} {'B - A (Δ abs)':<20} {'B - A (Δ %)':<15} {'C - A (Δ abs)':<20} {'C - A (Δ %)':<15}")
print("-" * 100)
for metric in metrics:
    a_val = results['A_current'][metric]
    b_val = results['B_simplified'][metric]
    c_val = results['C_hybrid'][metric]
    
    delta_b_abs = b_val - a_val
    delta_b_pct = (delta_b_abs / a_val * 100) if a_val != 0 else 0
    
    delta_c_abs = c_val - a_val
    delta_c_pct = (delta_c_abs / a_val * 100) if a_val != 0 else 0
    
    # Mark significant differences
    sig_b = "**" if abs(delta_b_abs) > 0.005 else ""
    sig_c = "**" if abs(delta_c_abs) > 0.005 else ""
    
    print(f"{metric:<30} {delta_b_abs:+.6f}{sig_b:<14} {delta_b_pct:+.2f}%{'':<9} {delta_c_abs:+.6f}{sig_c:<14} {delta_c_pct:+.2f}%")

print()
print("** = Difference > 0.005 (threshold for significance)")
print()

# Statistical tests for correlation differences
def fisher_z_transform(r):
    """Fisher's z-transformation"""
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, r2, n):
    """Compare two correlations with Fisher's z-transformation"""
    z1 = fisher_z_transform(r1)
    z2 = fisher_z_transform(r2)
    se = np.sqrt(2 / (n - 3))
    z_stat = (z2 - z1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed
    return z_stat, p_value

# Sample size
n = 3177

print("=" * 100)
print("STATISTICAL SIGNIFICANCE TESTS (Fisher's z-transformation, n=3177)")
print("=" * 100)
print()

correlation_metrics = ['test_mastery_correlation', 'test_gain_correlation']

for metric in correlation_metrics:
    a_val = results['A_current'][metric]
    b_val = results['B_simplified'][metric]
    c_val = results['C_hybrid'][metric]
    
    print(f"{metric}:")
    print(f"  A (Current):    {a_val:.6f}")
    print(f"  B (Simplified): {b_val:.6f}")
    print(f"  C (Hybrid):     {c_val:.6f}")
    print()
    
    # A vs B
    if a_val != b_val:
        z_ab, p_ab = compare_correlations(a_val, b_val, n)
        sig_ab = "***" if p_ab < 0.001 else "**" if p_ab < 0.01 else "*" if p_ab < 0.05 else "n.s."
        print(f"  A vs B: Z = {z_ab:+.3f}, p = {p_ab:.6f} ({sig_ab})")
    else:
        print(f"  A vs B: IDENTICAL VALUES (no test needed)")
    
    # A vs C
    if a_val != c_val:
        z_ac, p_ac = compare_correlations(a_val, c_val, n)
        sig_ac = "***" if p_ac < 0.001 else "**" if p_ac < 0.01 else "*" if p_ac < 0.05 else "n.s."
        print(f"  A vs C: Z = {z_ac:+.3f}, p = {p_ac:.6f} ({sig_ac})")
    else:
        print(f"  A vs C: IDENTICAL VALUES (no test needed)")
    
    print()

print()
print("=" * 100)
print("HYPOTHESIS TESTS")
print("=" * 100)
print()

# Check H1: Consistency loss redundant (A ≈ C)
auc_a = results['A_current']['test_auc']
auc_b = results['B_simplified']['test_auc']
auc_c = results['C_hybrid']['test_auc']

mastery_a = results['A_current']['test_mastery_correlation']
mastery_b = results['B_simplified']['test_mastery_correlation']
mastery_c = results['C_hybrid']['test_mastery_correlation']

gain_a = results['A_current']['test_gain_correlation']
gain_b = results['B_simplified']['test_gain_correlation']
gain_c = results['C_hybrid']['test_gain_correlation']

print("H1: Consistency Loss Redundant (A ≈ C)")
print("-" * 100)
h1_auc = abs(auc_a - auc_c) < 0.005
h1_mastery = abs(mastery_a - mastery_c) < 0.005
h1_gain = abs(gain_a - gain_c) < 0.005
h1_support = h1_auc and h1_mastery and h1_gain

print(f"  AUC:     |{auc_a:.6f} - {auc_c:.6f}| = {abs(auc_a - auc_c):.6f} < 0.005? {h1_auc}")
print(f"  Mastery: |{mastery_a:.6f} - {mastery_c:.6f}| = {abs(mastery_a - mastery_c):.6f} < 0.005? {h1_mastery}")
print(f"  Gain:    |{gain_a:.6f} - {gain_c:.6f}| = {abs(gain_a - gain_c):.6f} < 0.005? {h1_gain}")
print()
print(f"  → H1 SUPPORTED: {h1_support}")
if h1_support:
    print(f"  → DECISION: ✅ Remove consistency_loss_weight (redundant)")
else:
    print(f"  → DECISION: ❌ Keep consistency_loss (contributes to performance)")
print()

print("H2: Gain Supervision Redundant (A ≈ B)")
print("-" * 100)
h2_auc = abs(auc_a - auc_b) < 0.005
h2_mastery = abs(mastery_a - mastery_b) < 0.005
h2_gain = abs(gain_a - gain_b) < 0.005
h2_support = h2_auc and h2_mastery and h2_gain

print(f"  AUC:     |{auc_a:.6f} - {auc_b:.6f}| = {abs(auc_a - auc_b):.6f} < 0.005? {h2_auc}")
print(f"  Mastery: |{mastery_a:.6f} - {mastery_b:.6f}| = {abs(mastery_a - mastery_b):.6f} < 0.005? {h2_mastery}")
print(f"  Gain:    |{gain_a:.6f} - {gain_b:.6f}| = {abs(gain_a - gain_b):.6f} < 0.005? {h2_gain}")
print()
print(f"  → H2 SUPPORTED: {h2_support}")
if h2_support:
    print(f"  → DECISION: ✅ Remove gain_performance_loss_weight (redundant)")
else:
    print(f"  → DECISION: ❌ Keep gain_performance_loss (contributes to performance)")
print()

# Special case: B and C are identical
if auc_b == auc_c and mastery_b == mastery_c and gain_b == gain_c:
    print("=" * 100)
    print("⚠️  SPECIAL OBSERVATION: B and C are IDENTICAL")
    print("=" * 100)
    print()
    print("Configs B (Simplified) and C (Hybrid) produced identical results:")
    print(f"  - Same AUC:     {auc_b:.6f}")
    print(f"  - Same Mastery: {mastery_b:.6f}")
    print(f"  - Same Gain:    {gain_b:.6f}")
    print()
    print("This suggests that BOTH consistency_loss AND gain_performance_loss are redundant.")
    print("Removing either one or both produces the same degraded performance relative to A.")
    print()

print()
print("=" * 100)
print("RECOMMENDED ARCHITECTURE DECISION")
print("=" * 100)
print()

if h1_support and h2_support:
    print("✅ SIMPLIFY TO 3 LOSSES (both redundant):")
    print("   - mastery_performance_loss (weight: 1.5)")
    print("   - monotonicity_loss (weight: 0.1)")
    print("   - sparsity_loss (weight: 0.2)")
    print()
    print("❌ REMOVE:")
    print("   - gain_performance_loss_weight (mastery spillover sufficient)")
    print("   - consistency_loss_weight (architectural constraint redundant)")
elif h1_support and not h2_support:
    print("✅ SIMPLIFY TO 4 LOSSES (only consistency redundant):")
    print("   - mastery_performance_loss (weight: 1.5)")
    print("   - gain_performance_loss (weight: 0.8)")
    print("   - monotonicity_loss (weight: 0.1)")
    print("   - sparsity_loss (weight: 0.2)")
    print()
    print("❌ REMOVE:")
    print("   - consistency_loss_weight (architectural constraint redundant)")
elif not h1_support and h2_support:
    print("⚠️  UNEXPECTED RESULT:")
    print("   Gain supervision redundant but consistency loss needed?")
    print("   This suggests consistency loss provides value beyond architectural constraint.")
    print("   Recommend further investigation.")
else:
    print("❌ KEEP CURRENT ARCHITECTURE (5 losses):")
    print("   Both losses provide measurable value.")
    print("   No simplification recommended based on these results.")
    print()
    print("   Current architecture:")
    print("   - mastery_performance_loss (weight: 1.5)")
    print("   - gain_performance_loss (weight: 0.8)")
    print("   - consistency_loss (weight: 0.3)")
    print("   - monotonicity_loss (weight: 0.1)")
    print("   - sparsity_loss (weight: 0.2)")

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print()

delta_mastery_b = mastery_a - mastery_b
delta_gain_b = gain_a - gain_b

if delta_mastery_b > 0.01:
    print(f"⚠️  DEGRADATION OBSERVED:")
    print(f"   Removing gain/consistency losses causes:")
    print(f"   - Mastery correlation drops by {delta_mastery_b:.4f} ({delta_mastery_b/mastery_a*100:.1f}%)")
    print(f"   - Gain correlation drops by {delta_gain_b:.4f} ({delta_gain_b/gain_a*100:.1f}%)")
    print()
    print("   This suggests that explicit supervision on gains and consistency enforcement")
    print("   are NECESSARY despite the architectural constraint. The gradient spillover")
    print("   from mastery supervision alone is INSUFFICIENT to maintain interpretability.")
    print()
    print("   CONCLUSION: Keep all current losses. The hypothesis that losses are")
    print("   redundant is REJECTED by the data.")
else:
    print(f"✅ NO DEGRADATION:")
    print(f"   Simplified architecture maintains performance.")
    print(f"   Losses can be safely removed.")

print()
print("=" * 100)
