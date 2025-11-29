"""
Compute IRT correlation between model's learned difficulty (β) and IRT-calibrated priors.

This validates that skill difficulty embeddings remain aligned with IRT theory
despite being trainable parameters.

Usage:
    python examples/compute_irt_correlation.py --experiment_dir <path> --dataset <name>
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import pickle


def load_rasch_targets(dataset_name):
    """Load pre-computed IRT difficulty parameters."""
    rasch_file = Path(f"data/{dataset_name}/rasch_targets.pkl")
    
    if not rasch_file.exists():
        raise FileNotFoundError(
            f"Rasch targets not found: {rasch_file}\n"
            f"Please run: python examples/compute_rasch_targets.py --dataset {dataset_name}"
        )
    
    with open(rasch_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract beta from skill_difficulties dict
    beta_rasch = data.get('skill_difficulties', data.get('beta_rasch', {}))
    
    print(f"Loaded Rasch targets for {len(beta_rasch)} skills")
    print(f"Rasch metadata: {data.get('metadata', {})}")
    
    # Return dict with standardized key
    return {
        'beta_rasch': beta_rasch,
        'metadata': data.get('metadata', {})
    }


def compute_irt_correlation(experiment_dir, dataset_name, output_file=None):
    """
    Compute correlation between learned β and IRT-calibrated β.
    
    Returns:
        dict: Correlation results including r, p-value, and statistics
    """
    exp_path = Path(experiment_dir)
    
    print("\n" + "=" * 80)
    print("IRT DIFFICULTY CORRELATION")
    print("=" * 80)
    
    # 1. Load IRT-calibrated difficulties (ground truth)
    print("\n1. Loading IRT-calibrated difficulties (β_IRT)...")
    rasch_data = load_rasch_targets(dataset_name)
    beta_irt = rasch_data['beta_rasch']  # Dict: skill_id -> difficulty
    
    # 2. Load model's learned difficulties from mastery_test.csv
    print("\n2. Loading model's learned difficulties (β_learned)...")
    mastery_file = exp_path / 'mastery_test.csv'
    
    if not mastery_file.exists():
        raise FileNotFoundError(
            f"Mastery file not found: {mastery_file}\n"
            f"Please run evaluation first: python examples/eval_ikt2.py"
        )
    
    df = pd.read_csv(mastery_file)
    
    # Extract unique skill -> beta mapping from model
    skill_beta = df.groupby('skill_id')['beta'].first().to_dict()
    
    print(f"   Found {len(skill_beta)} skills in test data")
    
    # 3. Align skills present in both datasets
    print("\n3. Aligning skills...")
    common_skills = sorted(set(skill_beta.keys()) & set(beta_irt.keys()))
    
    if len(common_skills) == 0:
        raise ValueError("No common skills between model and IRT calibration!")
    
    print(f"   Common skills: {len(common_skills)}")
    
    # Create aligned arrays
    beta_model = np.array([skill_beta[s] for s in common_skills])
    beta_rasch = np.array([beta_irt[s] for s in common_skills])
    
    # 4. Compute correlation
    print("\n4. Computing Pearson correlation...")
    correlation, p_value = pearsonr(beta_model, beta_rasch)
    
    # 5. Compute additional statistics
    mse = np.mean((beta_model - beta_rasch) ** 2)
    mae = np.mean(np.abs(beta_model - beta_rasch))
    
    # Results dictionary
    results = {
        'difficulty_fidelity': float(correlation),
        'p_value': float(p_value),
        'num_skills': len(common_skills),
        'mse': float(mse),
        'mae': float(mae),
        'model_beta_stats': {
            'mean': float(beta_model.mean()),
            'std': float(beta_model.std()),
            'min': float(beta_model.min()),
            'max': float(beta_model.max()),
        },
        'rasch_beta_stats': {
            'mean': float(beta_rasch.mean()),
            'std': float(beta_rasch.std()),
            'min': float(beta_rasch.min()),
            'max': float(beta_rasch.max()),
        },
        'metadata': {
            'experiment_dir': str(experiment_dir),
            'dataset': dataset_name,
            'rasch_file': str(rasch_data.get('metadata', {})),
        }
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("\nDIFFICULTY FIDELITY (β_learned vs β_IRT):")
    print(f"  r = {correlation:.4f}  (p = {p_value:.4e})")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    print(f"\nDISTRIBUTION STATISTICS:")
    print(f"  Model β:  mean={beta_model.mean():.4f}, std={beta_model.std():.4f}")
    print(f"  Rasch β:  mean={beta_rasch.mean():.4f}, std={beta_rasch.std():.4f}")
    
    print(f"\nDATA SUMMARY:")
    print(f"  Skills: {len(common_skills)}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if correlation >= 0.9:
        interpretation = "EXCELLENT alignment - difficulties remain strongly anchored to IRT"
    elif correlation >= 0.8:
        interpretation = "STRONG alignment - good preservation of IRT semantics"
    elif correlation >= 0.7:
        interpretation = "GOOD alignment - reasonable preservation with some drift"
    elif correlation >= 0.6:
        interpretation = "MODERATE alignment - consider increasing λ_reg"
    else:
        interpretation = "WEAK alignment - embeddings have drifted significantly from IRT"
    
    print(f"\nDIFFICULTY FIDELITY (r = {correlation:.4f}):")
    print(f"  {interpretation}")
    
    if correlation < 0.8:
        print(f"\nRECOMMENDATION:")
        print(f"  Consider increasing λ_reg (difficulty regularization) to better preserve")
        print(f"  IRT alignment while maintaining prediction performance")
    
    # Save results
    if output_file:
        output_path = exp_path / output_file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        output_path = exp_path / 'irt_correlation_test.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def update_metrics_csv(experiment_dir, correlation_value):
    """Update metrics_test.csv with difficulty_fidelity column."""
    import pandas as pd
    csv_path = Path(experiment_dir) / 'metrics_test.csv'
    
    if not csv_path.exists():
        print(f"\n⚠ Warning: {csv_path} not found, skipping CSV update")
        return
    
    try:
        df = pd.read_csv(csv_path)
        df['difficulty_fidelity'] = correlation_value
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Updated {csv_path} with difficulty_fidelity={correlation_value:.6f}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not update metrics_test.csv: {e}")


def main():
    parser = argparse.ArgumentParser(description='Compute IRT difficulty correlation')
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Path to experiment directory')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., assist2015)')
    parser.add_argument('--output_file', type=str, default='irt_correlation_test.json',
                      help='Output JSON filename')
    parser.add_argument('--update_csv', action='store_true',
                      help='Update metrics_test.csv with correlation (default: True)')
    
    args = parser.parse_args()
    
    results = compute_irt_correlation(args.experiment_dir, args.dataset, args.output_file)
    
    # Update metrics_test.csv by default
    if args.update_csv or args.update_csv is None:
        update_metrics_csv(args.experiment_dir, results['difficulty_fidelity'])


if __name__ == '__main__':
    main()
