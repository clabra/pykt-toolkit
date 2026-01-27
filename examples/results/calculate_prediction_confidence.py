#!/usr/bin/env python3
"""Calculate prediction-level confidence scores."""
import numpy as np
import ast
from scipy import stats
from pathlib import Path
import pandas as pd

def parse_prediction_file(filepath):
    all_predictions = []
    all_labels = []
    all_qids = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = ast.literal_eval(line)
                if isinstance(data, list) and len(data) >= 5:
                    qids = data[2]
                    predictions = data[4]
                    shifted_answers = data[3]
                    if isinstance(predictions, (list, tuple)):
                        all_predictions.extend(predictions)
                        all_labels.extend(shifted_answers)
                        all_qids.extend(qids)
            except Exception as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    return np.array(all_predictions), np.array(all_labels), np.array(all_qids)

def calculate_prediction_confidence(p_ref, p_sup):
    n = len(p_ref)
    disagreement = np.abs(p_ref - p_sup)
    agreement_confidence = 1.0 - disagreement
    calibrated_confidence = np.exp(-2 * disagreement)
    agreement_percentiles = np.zeros(n)
    for i in range(n):
        agreement_percentiles[i] = stats.percentileofscore(disagreement, disagreement[i])
    confidence_percentile = 100 - agreement_percentiles
    p_ref_binary = (p_ref >= 0.5).astype(int)
    p_sup_binary = (p_sup >= 0.5).astype(int)
    directional_agreement = (p_ref_binary == p_sup_binary).astype(float)
    lower_bound = np.clip(p_ref - disagreement, 0, 1)
    upper_bound = np.clip(p_ref + disagreement, 0, 1)
    interval_width = upper_bound - lower_bound
    composite_confidence = (
        0.4 * calibrated_confidence +
        0.3 * directional_agreement +
        0.3 * (confidence_percentile / 100.0)
    )
    results = {
        'disagreement': disagreement,
        'agreement_confidence': agreement_confidence,
        'calibrated_confidence': calibrated_confidence,
        'confidence_percentile': confidence_percentile,
        'directional_agreement': directional_agreement,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'interval_width': interval_width,
        'composite_confidence': composite_confidence,
    }
    return results

def analyze_confidence_distribution(confidence_results):
    composite = confidence_results['composite_confidence']
    disagreement = confidence_results['disagreement']
    directional = confidence_results['directional_agreement']
    high_confidence = composite >= 0.8
    medium_confidence = (composite >= 0.5) & (composite < 0.8)
    low_confidence = composite < 0.5
    high_disagreement = disagreement > 0.3
    directional_mismatch = directional == 0
    stats_dict = {
        'total_predictions': len(composite),
        'high_confidence_count': np.sum(high_confidence),
        'high_confidence_pct': np.mean(high_confidence) * 100,
        'medium_confidence_count': np.sum(medium_confidence),
        'medium_confidence_pct': np.mean(medium_confidence) * 100,
        'low_confidence_count': np.sum(low_confidence),
        'low_confidence_pct': np.mean(low_confidence) * 100,
        'high_disagreement_count': np.sum(high_disagreement),
        'high_disagreement_pct': np.mean(high_disagreement) * 100,
        'directional_mismatch_count': np.sum(directional_mismatch),
        'directional_mismatch_pct': np.mean(directional_mismatch) * 100,
    }
    return stats_dict

def main():
    exp_id = "268444"
    exp_name = "20260126_212614_ablation-none-nblocks-4-numattnheads-4_268444"
    exp_dir = Path(f"/workspaces/pykt-toolkit/experiments/{exp_name}/gtransformer/assist2009")
    fold_dirs = list(exp_dir.glob("fold_0_*"))
    if not fold_dirs:
        print(f"❌ No fold_0_* directory found")
        return
    fold_dir = fold_dirs[0]
    print("=" * 80)
    print(f"Prediction Confidence Analysis for Experiment {exp_id}")
    print("=" * 80)
    print(f"\nApproach:")
    print(f"  • p_ref (interpretable) = Primary prediction to show users")
    print(f"  • p_sup (supervised) = Trust anchor (more accurate)")
    print(f"  • Confidence = How much can we trust p_ref based on p_sup agreement")
    print("=" * 80)
    print()
    supervised_file = fold_dir / "qid_test_window_predictions_supervised.txt"
    reference_file = fold_dir / "qid_test_window_predictions_reference.txt"
    print(f"Loading predictions from {fold_dir.name}...")
    p_sup, labels_sup, qids_sup = parse_prediction_file(supervised_file)
    p_ref, labels_ref, qids_ref = parse_prediction_file(reference_file)
    print(f"  Supervised predictions: {len(p_sup):,} values")
    print(f"  Reference predictions:  {len(p_ref):,} values")
    print("\nCalculating confidence scores...")
    confidence_results = calculate_prediction_confidence(p_ref, p_sup)
    dist_stats = analyze_confidence_distribution(confidence_results)
    print("\n" + "=" * 80)
    print("CONFIDENCE SCORE DISTRIBUTION")
    print("=" * 80)
    composite = confidence_results['composite_confidence']
    print(f"\nComposite Confidence Score:")
    print(f"  Mean:       {np.mean(composite):.4f}")
    print(f"  Median:     {np.median(composite):.4f}")
    print(f"  Std Dev:    {np.std(composite):.4f}")
    print(f"  Min:        {np.min(composite):.4f}")
    print(f"  Max:        {np.max(composite):.4f}")
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th:      {np.percentile(composite, p):.4f}")
    print("\n" + "=" * 80)
    print("PREDICTION CATEGORIZATION")
    print("=" * 80)
    print(f"\n🟢 High Confidence (≥0.8):      {dist_stats['high_confidence_count']:>6,}  ({dist_stats['high_confidence_pct']:>5.1f}%)")
    print(f"   → p_ref closely matches p_sup, very trustworthy")
    print(f"\n🟡 Medium Confidence (0.5-0.8): {dist_stats['medium_confidence_count']:>6,}  ({dist_stats['medium_confidence_pct']:>5.1f}%)")
    print(f"   → Moderate agreement, use with caution")
    print(f"\n🔴 Low Confidence (<0.5):       {dist_stats['low_confidence_count']:>6,}  ({dist_stats['low_confidence_pct']:>5.1f}%)")
    print(f"   → p_ref diverges from p_sup, less trustworthy")
    print("\n" + "=" * 80)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 80)
    disagreement = confidence_results['disagreement']
    print(f"\nDisagreement (|p_ref - p_sup|):")
    print(f"  Mean (MAE): {np.mean(disagreement):.4f}")
    print(f"  Median:     {np.median(disagreement):.4f}")
    print(f"  Std Dev:    {np.std(disagreement):.4f}")
    print(f"  95th %ile:  {np.percentile(disagreement, 95):.4f}")
    print(f"\n⚠️  High Disagreement (>0.3):    {dist_stats['high_disagreement_count']:>6,}  ({dist_stats['high_disagreement_pct']:>5.1f}%)")
    print(f"   → Predictions differ by more than 30 percentage points")
    print(f"\n❌ Directional Mismatch:        {dist_stats['directional_mismatch_count']:>6,}  ({dist_stats['directional_mismatch_pct']:>5.1f}%)")
    print(f"   → p_ref and p_sup disagree on pass/fail (<0.5 vs ≥0.5)")
    print("\n" + "=" * 80)
    print("CONFIDENCE INTERVALS")
    print("=" * 80)
    interval_width = confidence_results['interval_width']
    print(f"\nInterval Width (uncertainty):")
    print(f"  Mean:       {np.mean(interval_width):.4f}")
    print(f"  Median:     {np.median(interval_width):.4f}")
    print(f"  95th %ile:  {np.percentile(interval_width, 95):.4f}")
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS WITH CONFIDENCE SCORES")
    print("=" * 80)
    high_conf_idx = np.where(composite >= 0.8)[0]
    low_conf_idx = np.where(composite < 0.5)[0]
    print("\n🟢 High Confidence Examples (first 5):")
    print(f"{'p_ref':>8} {'p_sup':>8} {'Disagree':>10} {'Confidence':>12} {'Decision':>10}")
    print("-" * 60)
    for i in high_conf_idx[:5]:
        decision = "✓ Match" if confidence_results['directional_agreement'][i] else "✗ Differ"
        print(f"{p_ref[i]:>8.4f} {p_sup[i]:>8.4f} {disagreement[i]:>10.4f} {composite[i]:>12.4f} {decision:>10}")
    print("\n🔴 Low Confidence Examples (first 5):")
    print(f"{'p_ref':>8} {'p_sup':>8} {'Disagree':>10} {'Confidence':>12} {'Decision':>10}")
    print("-" * 60)
    for i in low_conf_idx[:5]:
        decision = "✓ Match" if confidence_results['directional_agreement'][i] else "✗ Differ"
        print(f"{p_ref[i]:>8.4f} {p_sup[i]:>8.4f} {disagreement[i]:>10.4f} {composite[i]:>12.4f} {decision:>10}")
    print("\n" + "=" * 80)
    print("PRACTICAL USAGE RECOMMENDATIONS")
    print("=" * 80)
    print(f"""
When presenting interpretable predictions (p_ref) to users:

   → Show prediction with high certainty1. 
   → "The model predicts X with high confidence"

2. 🟡 MEDIUM CONFIDENCE ({dist_stats['medium_confidence_pct']:.1f}% of predictions)
   → Show prediction with caveat
   → "The model predicts X (moderate confidence)"

3. 🔴 LOW CONFIDENCE ({dist_stats['low_confidence_pct']:.1f}% of predictions)
   → Flag as uncertain
   → "The model is uncertain about this prediction"
   → Recommend human review for critical decisions
    """)
    print("\n" + "=" * 80)
    print("EXPORT OPTIONS")
    print("=" * 80)
    df = pd.DataFrame({
        'qid': qids_ref,
        'p_ref': p_ref,
        'p_sup': p_sup,
        'true_label': labels_ref,
        'disagreement': disagreement,
        'composite_confidence': composite,
        'calibrated_confidence': confidence_results['calibrated_confidence'],
        'directional_agreement': confidence_results['directional_agreement'],
        'lower_bound': confidence_results['lower_bound'],
        'upper_bound': confidence_results['upper_bound'],
        'interval_width': interval_width,
    })
    df_sorted = df.sort_values('composite_confidence')
    output_file = fold_dir / "prediction_confidence_scores.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"\n✅ Detailed confidence scores exported to:")
    print(f"   {output_file}")
    low_conf_df = df_sorted[df_sorted['composite_confidence'] < 0.5]
    if len(low_conf_df) > 0:
        review_file = fold_dir / "low_confidence_predictions_for_review.csv"
        low_conf_df.to_csv(review_file, index=False)
        print(f"\n⚠️  Low confidence predictions ({len(low_conf_df):,}) exported for review:")
        print(f"   {review_file}")
    print("\n" + "=" * 80)
    print()
    return confidence_results, df

if __name__ == "__main__":
    main()
