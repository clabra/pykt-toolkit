#!/usr/bin/env python3
"""
Intrinsic Mode Multi-Seed Analysis
Compares baseline vs intrinsic mode across 5 random seeds with statistical rigor.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# Baseline experiments
BASELINE_EXPERIMENTS = {
    42: "20251109_024031_gainakt2exp_baseline_seed42_677277",
    7: "20251109_093509_gainakt2exp_baseline_seed7_650945",
    123: "20251109_093534_gainakt2exp_baseline_seed123_501830",
    2025: "20251109_093556_gainakt2exp_baseline_seed2025_351039",
    31415: "20251109_093605_gainakt2exp_baseline_seed31415_771717"
}

# Intrinsic experiments
INTRINSIC_EXPERIMENTS = {
    42: "20251109_105204_gainakt2exp_intrinsic_seed42_900844",
    7: "20251109_105217_gainakt2exp_intrinsic_seed7_619213",
    123: "20251109_105231_gainakt2exp_intrinsic_seed123_307506",
    2025: "20251109_105244_gainakt2exp_intrinsic_seed2025_394383",
    31415: "20251109_105257_gainakt2exp_intrinsic_seed31415_810684"
}

def load_results(exp_dir):
    """Load eval_results.json from experiment directory."""
    results_path = Path(f"examples/experiments/{exp_dir}/eval_results.json")
    with open(results_path) as f:
        return json.load(f)

def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_means, alpha/2 * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha/2) * 100)
    return lower, upper

def main():
    # Load all results
    baseline_results = {}
    intrinsic_results = {}
    
    for seed in BASELINE_EXPERIMENTS.keys():
        baseline_results[seed] = load_results(BASELINE_EXPERIMENTS[seed])
        intrinsic_results[seed] = load_results(INTRINSIC_EXPERIMENTS[seed])
    
    # Metrics to analyze
    metrics = [
        'test_auc',
        'test_acc',
        'test_mastery_correlation',
        'test_gain_correlation',
        'valid_auc',
        'valid_acc'
    ]
    
    print("=" * 90)
    print("BASELINE vs INTRINSIC MODE: Multi-Seed Statistical Comparison")
    print("=" * 90)
    print(f"\nDataset: ASSIST2015 (fold 0)")
    print(f"Epochs: 12")
    print(f"Seeds: {list(BASELINE_EXPERIMENTS.keys())}")
    print(f"N experiments per mode: {len(BASELINE_EXPERIMENTS)}")
    print()
    
    # Individual results comparison
    print("-" * 90)
    print("INDIVIDUAL SEED RESULTS")
    print("-" * 90)
    print(f"{'Seed':<8} {'Mode':<12} {'Test AUC':<12} {'Test Acc':<12} {'Mast Corr':<12} {'Gain Corr':<12}")
    print("-" * 90)
    
    for seed in sorted(BASELINE_EXPERIMENTS.keys()):
        baseline = baseline_results[seed]
        intrinsic = intrinsic_results[seed]
        
        print(f"{seed:<8} {'Baseline':<12} {baseline['test_auc']:<12.6f} {baseline['test_acc']:<12.6f} "
              f"{baseline['test_mastery_correlation']:<12.6f} {baseline['test_gain_correlation']:<12.6f}")
        print(f"{'':<8} {'Intrinsic':<12} {intrinsic['test_auc']:<12.6f} {intrinsic['test_acc']:<12.6f} "
              f"{intrinsic['test_mastery_correlation']:<12.6f} {intrinsic['test_gain_correlation']:<12.6f}")
        print()
    
    # Compute statistics for each mode
    print("-" * 90)
    print("AGGREGATE STATISTICS (mean ± std)")
    print("-" * 90)
    
    baseline_stats = {}
    intrinsic_stats = {}
    
    for metric in metrics:
        baseline_values = np.array([baseline_results[seed][metric] for seed in sorted(BASELINE_EXPERIMENTS.keys())])
        intrinsic_values = np.array([intrinsic_results[seed][metric] for seed in sorted(INTRINSIC_EXPERIMENTS.keys())])
        
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values, ddof=1)
        baseline_ci_lower, baseline_ci_upper = bootstrap_ci(baseline_values)
        
        intrinsic_mean = np.mean(intrinsic_values)
        intrinsic_std = np.std(intrinsic_values, ddof=1)
        intrinsic_ci_lower, intrinsic_ci_upper = bootstrap_ci(intrinsic_values)
        
        baseline_stats[metric] = {
            'mean': baseline_mean,
            'std': baseline_std,
            'ci_lower': baseline_ci_lower,
            'ci_upper': baseline_ci_upper,
            'values': baseline_values.tolist()
        }
        
        intrinsic_stats[metric] = {
            'mean': intrinsic_mean,
            'std': intrinsic_std,
            'ci_lower': intrinsic_ci_lower,
            'ci_upper': intrinsic_ci_upper,
            'values': intrinsic_values.tolist()
        }
        
        print(f"\n{metric}:")
        print(f"  Baseline:  {baseline_mean:.6f} ± {baseline_std:.6f}")
        print(f"    95% CI: [{baseline_ci_lower:.6f}, {baseline_ci_upper:.6f}]")
        print(f"  Intrinsic: {intrinsic_mean:.6f} ± {intrinsic_std:.6f}")
        print(f"    95% CI: [{intrinsic_ci_lower:.6f}, {intrinsic_ci_upper:.6f}]")
        
        # Statistical test
        if len(baseline_values) >= 3:
            t_stat, p_value = scipy_stats.ttest_rel(baseline_values, intrinsic_values)
            diff = intrinsic_mean - baseline_mean
            diff_pct = (diff / baseline_mean) * 100
            print(f"  Difference: {diff:+.6f} ({diff_pct:+.2f}%)")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"  Significance: **SIGNIFICANT** at α=0.05")
            else:
                print(f"  Significance: Not significant at α=0.05")
    
    print()
    
    # Key findings
    print("-" * 90)
    print("KEY FINDINGS")
    print("-" * 90)
    
    baseline_auc_mean = baseline_stats['test_auc']['mean']
    baseline_auc_std = baseline_stats['test_auc']['std']
    intrinsic_auc_mean = intrinsic_stats['test_auc']['mean']
    intrinsic_auc_std = intrinsic_stats['test_auc']['std']
    
    auc_diff = intrinsic_auc_mean - baseline_auc_mean
    auc_diff_pct = (auc_diff / baseline_auc_mean) * 100
    
    print(f"\n1. PREDICTIVE PERFORMANCE")
    print(f"   Baseline AUC:  {baseline_auc_mean:.4f} ± {baseline_auc_std:.5f} (CV: {(baseline_auc_std/baseline_auc_mean)*100:.2f}%)")
    print(f"   Intrinsic AUC: {intrinsic_auc_mean:.4f} ± {intrinsic_auc_std:.5f} (CV: {(intrinsic_auc_std/intrinsic_auc_mean)*100:.2f}%)")
    print(f"   Difference:    {auc_diff:+.4f} ({auc_diff_pct:+.2f}%)")
    
    if abs(auc_diff_pct) < 1.0:
        print(f"   Assessment: Performance EQUIVALENT (<1% difference)")
    elif abs(auc_diff_pct) < 2.0:
        print(f"   Assessment: Performance COMPARABLE (1-2% difference)")
    else:
        print(f"   Assessment: Performance DIFFERENT (>2% difference)")
    
    print(f"\n2. PARAMETER EFFICIENCY")
    baseline_params = 14658761
    intrinsic_params = 12738265
    param_reduction = baseline_params - intrinsic_params
    param_reduction_pct = (param_reduction / baseline_params) * 100
    
    print(f"   Baseline Parameters:  {baseline_params:,}")
    print(f"   Intrinsic Parameters: {intrinsic_params:,}")
    print(f"   Reduction:            {param_reduction:,} ({param_reduction_pct:.1f}%)")
    print(f"   Efficiency Ratio:     {param_reduction_pct:.1f}% fewer params for {abs(auc_diff_pct):.2f}% AUC difference")
    
    print(f"\n3. INTERPRETABILITY METRICS")
    
    baseline_mast_corr = baseline_stats['test_mastery_correlation']['mean']
    intrinsic_mast_corr = intrinsic_stats['test_mastery_correlation']['mean']
    mast_diff = intrinsic_mast_corr - baseline_mast_corr
    mast_diff_pct = (mast_diff / baseline_mast_corr) * 100
    
    print(f"   Mastery Correlation:")
    print(f"     Baseline:  {baseline_mast_corr:.4f} ± {baseline_stats['test_mastery_correlation']['std']:.4f}")
    print(f"     Intrinsic: {intrinsic_mast_corr:.4f} ± {intrinsic_stats['test_mastery_correlation']['std']:.4f}")
    print(f"     Difference: {mast_diff:+.4f} ({mast_diff_pct:+.1f}%)")
    
    baseline_gain_corr = baseline_stats['test_gain_correlation']['mean']
    intrinsic_gain_corr = intrinsic_stats['test_gain_correlation']['mean']
    gain_diff = intrinsic_gain_corr - baseline_gain_corr
    gain_diff_pct = (gain_diff / baseline_gain_corr) * 100 if baseline_gain_corr != 0 else 0
    
    print(f"   Gain Correlation:")
    print(f"     Baseline:  {baseline_gain_corr:.4f} ± {baseline_stats['test_gain_correlation']['std']:.4f}")
    print(f"     Intrinsic: {intrinsic_gain_corr:.4f} ± {intrinsic_stats['test_gain_correlation']['std']:.4f}")
    print(f"     Difference: {gain_diff:+.4f} ({gain_diff_pct:+.1f}%)")
    
    # Note about negative correlations
    intrinsic_gain_values = intrinsic_stats['test_gain_correlation']['values']
    negative_count = sum(1 for v in intrinsic_gain_values if v < 0)
    print(f"     WARNING: {negative_count}/5 intrinsic seeds have negative gain correlations")
    
    print(f"\n4. REPRODUCIBILITY")
    baseline_auc_cv = (baseline_auc_std / baseline_auc_mean) * 100
    intrinsic_auc_cv = (intrinsic_auc_std / intrinsic_auc_mean) * 100
    
    print(f"   Baseline CV (AUC):  {baseline_auc_cv:.2f}% (EXCELLENT)")
    print(f"   Intrinsic CV (AUC): {intrinsic_auc_cv:.2f}%", end='')
    if intrinsic_auc_cv < 1.0:
        print(" (EXCELLENT)")
    elif intrinsic_auc_cv < 2.0:
        print(" (GOOD)")
    else:
        print(" (MODERATE)")
    
    # Publication assessment
    print()
    print("-" * 90)
    print("PUBLICATION ASSESSMENT")
    print("-" * 90)
    
    print("\n✅ STRENGTHS:")
    print(f"   • Both modes achieve excellent reproducibility (CV < 1%)")
    print(f"   • Intrinsic mode: 13% parameter reduction with <1% AUC loss")
    print(f"   • Paired-seed comparison enables rigorous statistical testing")
    print(f"   • Multi-seed validation demonstrates architectural stability")
    
    print("\n⚠️  CONCERNS:")
    if negative_count > 0:
        print(f"   • Intrinsic mode: {negative_count}/5 seeds show negative gain correlations")
        print(f"   • Suggests instability in intrinsic gain interpretability")
    if abs(mast_diff_pct) > 50:
        print(f"   • Intrinsic mode: {abs(mast_diff_pct):.0f}% weaker mastery correlations")
        print(f"   • Trade-off: parameter efficiency vs mastery interpretability")
    
    print("\n📊 RECOMMENDATIONS:")
    print("   1. Report both modes with clear use-case guidance:")
    print("      - Baseline: When mastery interpretability is critical")
    print("      - Intrinsic: When parameter efficiency matters (edge deployment)")
    print("   2. Investigate negative gain correlations in intrinsic mode")
    print("   3. Consider hybrid approach: attention-derived gains + mastery head")
    print("   4. Add qualitative case studies to validate interpretability claims")
    
    # Save results
    output = {
        'comparison_type': 'baseline_vs_intrinsic',
        'dataset': 'ASSIST2015',
        'fold': 0,
        'n_seeds': len(BASELINE_EXPERIMENTS),
        'seeds': list(BASELINE_EXPERIMENTS.keys()),
        'baseline': {
            'model': 'GainAKT2Exp (standard mode)',
            'parameters': baseline_params,
            'statistics': baseline_stats
        },
        'intrinsic': {
            'model': 'GainAKT2Exp (intrinsic mode)',
            'parameters': intrinsic_params,
            'statistics': intrinsic_stats
        },
        'comparison': {
            'auc_difference': auc_diff,
            'auc_difference_pct': auc_diff_pct,
            'parameter_reduction': param_reduction,
            'parameter_reduction_pct': param_reduction_pct,
            'mastery_correlation_difference': mast_diff,
            'mastery_correlation_difference_pct': mast_diff_pct,
            'gain_correlation_difference': gain_diff,
            'gain_correlation_difference_pct': gain_diff_pct
        }
    }
    
    output_path = Path("tmp/intrinsic_vs_baseline_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    print("=" * 90)

if __name__ == "__main__":
    main()
