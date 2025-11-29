"""
Compute Head Agreement: Correlation between M_IRT and p_correct predictions.

This metric validates that the IRT mastery head produces predictions consistent
with the performance prediction head, demonstrating internal IRT structure validity.

Usage:
    python examples/compute_head_agreement.py --experiment_dir <path>
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr


def compute_head_agreement(experiment_dir, output_file=None):
    """
    Compute correlation between M_IRT (IRT mastery) and p_correct (BCE predictions).
    
    This validates internal IRT consistency: do the two prediction heads agree?
    
    Returns:
        dict: Head agreement results including correlation and statistics
    """
    exp_path = Path(experiment_dir)
    
    print("\n" + "=" * 80)
    print("HEAD AGREEMENT (IRT CONSISTENCY)")
    print("=" * 80)
    
    # Load mastery data (contains both M_IRT and BCE predictions)
    print("\n1. Loading model predictions from mastery_test.csv...")
    mastery_file = exp_path / 'mastery_test.csv'
    
    if not mastery_file.exists():
        raise FileNotFoundError(
            f"Mastery file not found: {mastery_file}\n"
            f"Please run mastery_states.py first"
        )
    
    df = pd.read_csv(mastery_file)
    print(f"   Loaded {len(df)} interactions")
    
    # Extract predictions
    # mi_value = M_IRT = σ(θ - β) from IRT head
    # bce_prediction = p_correct from performance head
    m_irt = df['mi_value'].values
    p_correct = df['bce_prediction'].values
    
    print(f"   Students: {df['student_id'].nunique()}")
    print(f"   Skills: {df['skill_id'].nunique()}")
    
    # Compute correlation
    print("\n2. Computing Pearson correlation between M_IRT and p_correct...")
    
    # Filter valid values (non-NaN)
    valid_mask = ~(np.isnan(m_irt) | np.isnan(p_correct))
    m_irt_clean = m_irt[valid_mask]
    p_correct_clean = p_correct[valid_mask]
    
    if len(m_irt_clean) < 2:
        raise ValueError("Insufficient valid data points for correlation")
    
    correlation, p_value = pearsonr(m_irt_clean, p_correct_clean)
    
    # Compute additional statistics
    mse = np.mean((m_irt_clean - p_correct_clean) ** 2)
    mae = np.mean(np.abs(m_irt_clean - p_correct_clean))
    
    # Results dictionary
    results = {
        'head_agreement': float(correlation),
        'p_value': float(p_value),
        'num_interactions': len(m_irt_clean),
        'mse': float(mse),
        'mae': float(mae),
        'm_irt_stats': {
            'mean': float(m_irt_clean.mean()),
            'std': float(m_irt_clean.std()),
            'min': float(m_irt_clean.min()),
            'max': float(m_irt_clean.max()),
        },
        'p_correct_stats': {
            'mean': float(p_correct_clean.mean()),
            'std': float(p_correct_clean.std()),
            'min': float(p_correct_clean.min()),
            'max': float(p_correct_clean.max()),
        },
        'metadata': {
            'experiment_dir': str(experiment_dir),
            'metric_type': 'head_agreement',
            'description': 'Correlation between IRT mastery (M_IRT) and performance prediction (p_correct)'
        }
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nHEAD AGREEMENT (M_IRT vs p_correct):")
    print(f"  r = {correlation:.4f}  (p = {p_value:.4e})")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    print(f"\nDISTRIBUTION STATISTICS:")
    print(f"  M_IRT:     mean={m_irt_clean.mean():.4f}, std={m_irt_clean.std():.4f}")
    print(f"  p_correct: mean={p_correct_clean.mean():.4f}, std={p_correct_clean.std():.4f}")
    
    print(f"\nDATA SUMMARY:")
    print(f"  Interactions: {len(m_irt_clean):,}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if correlation >= 0.8:
        interpretation = "EXCELLENT internal consistency - IRT structure is highly meaningful"
        rating = "✓✓✓"
    elif correlation >= 0.7:
        interpretation = "STRONG internal consistency - IRT formulation is valid"
        rating = "✓✓"
    elif correlation >= 0.6:
        interpretation = "GOOD internal consistency - reasonable IRT alignment"
        rating = "✓"
    elif correlation >= 0.5:
        interpretation = "MODERATE internal consistency - IRT head may be weakly aligned"
        rating = "~"
    else:
        interpretation = "WEAK internal consistency - IRT head diverges from predictions"
        rating = "✗"
    
    print(f"\nHEAD AGREEMENT (r = {correlation:.4f}): {rating}")
    print(f"  {interpretation}")
    print(f"\n  This metric validates that the IRT mastery head M_IRT = σ(θ - β)")
    print(f"  produces predictions consistent with the performance head p_correct,")
    print(f"  demonstrating that learned ability (θ) and difficulty (β) parameters")
    print(f"  are meaningful rather than arbitrary values.")
    
    # Save results
    if output_file:
        output_path = exp_path / output_file
    else:
        output_path = exp_path / 'head_agreement_test.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


def update_metrics_csv(experiment_dir, correlation_value):
    """Update metrics_test.csv with head_agreement column."""
    import pandas as pd
    csv_path = Path(experiment_dir) / 'metrics_test.csv'
    
    if not csv_path.exists():
        print(f"\n⚠ Warning: {csv_path} not found, skipping CSV update")
        return
    
    try:
        df = pd.read_csv(csv_path)
        df['head_agreement'] = correlation_value
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Updated {csv_path} with head_agreement={correlation_value:.6f}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not update metrics_test.csv: {e}")


def main():
    parser = argparse.ArgumentParser(description='Compute head agreement (IRT consistency)')
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Path to experiment directory')
    parser.add_argument('--output_file', type=str, default='head_agreement_test.json',
                      help='Output JSON filename')
    parser.add_argument('--update_csv', action='store_true',
                      help='Update metrics_test.csv with correlation (default: True)')
    
    args = parser.parse_args()
    
    results = compute_head_agreement(args.experiment_dir, args.output_file)
    
    # Update metrics_test.csv by default
    if args.update_csv or args.update_csv is None:
        update_metrics_csv(args.experiment_dir, results['head_agreement'])


if __name__ == '__main__':
    main()
