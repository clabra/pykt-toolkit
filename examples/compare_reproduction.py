#!/usr/bin/env python3
"""
Compare metrics between original and reproduced experiments.

This script loads metrics_epoch.csv from both experiments and performs
detailed comparison to verify reproducibility.

Usage:
    python examples/compare_reproduction.py <original_experiment_id> <repro_experiment_id>
    
    OR
    
    python examples/compare_reproduction.py <path_to_original> <path_to_repro>

Examples:
    # Compare by experiment IDs
    python examples/compare_reproduction.py 983383 983383_repro
    
    # Compare by full paths
    python examples/compare_reproduction.py \
        examples/experiments/20251102_182023_gainakt2exp_real_test_983383 \
        examples/experiments/20251102_183045_gainakt2exp_real_test_983383_repro
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def find_experiment_folder(identifier):
    """
    Find experiment folder by ID or return the path if it's a valid directory.
    
    Args:
        identifier: Either an experiment ID or a path to experiment folder
        
    Returns:
        Path object to the experiment folder
    """
    # Check if it's already a valid path
    path = Path(identifier)
    if path.exists() and path.is_dir():
        return path
    
    # Try to find by experiment ID
    experiments_dir = Path(__file__).parent / "experiments"
    if not experiments_dir.exists():
        raise ValueError(f"Experiments directory not found: {experiments_dir}")
    
    # Search for folder containing the ID
    matching_folders = list(experiments_dir.glob(f"*_{identifier}"))
    
    if not matching_folders:
        raise ValueError(f"No experiment found with ID: {identifier}")
    
    if len(matching_folders) > 1:
        raise ValueError(f"Multiple experiments found with ID {identifier}:\n" + 
                        "\n".join(f"  - {f.name}" for f in matching_folders))
    
    return matching_folders[0]


def compare_metrics(original_folder, repro_folder, tolerance=1e-4):
    """
    Compare metrics between original and reproduced experiments.
    
    Args:
        original_folder: Path to original experiment folder
        repro_folder: Path to reproduction experiment folder
        tolerance: Acceptable numeric difference (default: 1e-4)
        
    Returns:
        dict with comparison results
    """
    original_metrics = original_folder / "metrics_epoch.csv"
    repro_metrics = repro_folder / "metrics_epoch.csv"
    
    # Check if both files exist
    if not original_metrics.exists():
        return {
            'status': 'ERROR',
            'message': f'Original metrics file not found: {original_metrics}'
        }
    
    if not repro_metrics.exists():
        return {
            'status': 'ERROR',
            'message': f'Reproduction metrics file not found: {repro_metrics}'
        }
    
    try:
        # Load both CSVs
        df_original = pd.read_csv(original_metrics)
        df_repro = pd.read_csv(repro_metrics)
        
        # Check if same number of rows
        if len(df_original) != len(df_repro):
            return {
                'status': 'ERROR',
                'message': f'Different number of epochs: original={len(df_original)}, repro={len(df_repro)}'
            }
        
        # Check if same columns
        if set(df_original.columns) != set(df_repro.columns):
            missing_in_repro = set(df_original.columns) - set(df_repro.columns)
            missing_in_orig = set(df_repro.columns) - set(df_original.columns)
            msg = "Column mismatch:"
            if missing_in_repro:
                msg += f"\n  Missing in repro: {missing_in_repro}"
            if missing_in_orig:
                msg += f"\n  Missing in original: {missing_in_orig}"
            return {
                'status': 'ERROR',
                'message': msg
            }
        
        # Compare numeric columns
        exact_matches = []
        within_tolerance = []
        outside_tolerance = []
        
        for col in df_original.columns:
            if col == 'epoch':
                continue  # Skip epoch column
            
            # Check if numeric
            if df_original[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Compute differences
                diff = np.abs(df_original[col].values - df_repro[col].values)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                if max_diff == 0:
                    exact_matches.append(col)
                elif max_diff <= tolerance:
                    within_tolerance.append({
                        'column': col,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff
                    })
                else:
                    outside_tolerance.append({
                        'column': col,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'differences': diff.tolist()
                    })
        
        # Determine overall status
        if outside_tolerance:
            status = 'ERROR'
        elif within_tolerance:
            status = 'WARNING'
        else:
            status = 'SUCCESS'
        
        return {
            'status': status,
            'exact_matches': exact_matches,
            'within_tolerance': within_tolerance,
            'outside_tolerance': outside_tolerance,
            'tolerance': tolerance,
            'num_epochs': len(df_original)
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Error comparing metrics: {str(e)}'
        }


def print_results(results, original_folder, repro_folder):
    """Print formatted comparison results."""
    print("\n" + "=" * 80)
    print("REPRODUCTION COMPARISON")
    print("=" * 80)
    print(f"\nOriginal:     {original_folder.name}")
    print(f"Reproduction: {repro_folder.name}")
    print(f"Tolerance:    {results.get('tolerance', 'N/A')}")
    print(f"Epochs:       {results.get('num_epochs', 'N/A')}")
    print("=" * 80)
    
    # Check for errors first
    if 'message' in results:
        print(f"\n❌ {results['message']}")
        return
    
    status = results['status']
    tolerance = results['tolerance']
    
    # Print exact matches
    if results['exact_matches']:
        print(f"\n✓ EXACT MATCHES ({len(results['exact_matches'])} metrics):")
        for col in results['exact_matches']:
            print(f"  - {col}")
    
    # Print within-tolerance differences
    if results['within_tolerance']:
        print(f"\n⚠️  WITHIN TOLERANCE ({len(results['within_tolerance'])} metrics, tolerance={tolerance}):")
        for item in results['within_tolerance']:
            print(f"  - {item['column']}: max_diff={item['max_diff']:.2e}, mean_diff={item['mean_diff']:.2e}")
    
    # Print outside-tolerance differences
    if results['outside_tolerance']:
        print(f"\n❌ OUTSIDE TOLERANCE ({len(results['outside_tolerance'])} metrics, tolerance={tolerance}):")
        for item in results['outside_tolerance']:
            print(f"  - {item['column']}: max_diff={item['max_diff']:.2e}, mean_diff={item['mean_diff']:.2e}")
            # Show per-epoch differences
            if len(item['differences']) <= 10:
                # Show all if few epochs
                print(f"    Per-epoch diffs: {item['differences']}")
            else:
                # Show first 5 and last 5 if many epochs
                diffs = item['differences']
                print(f"    Per-epoch diffs (first 5): {diffs[:5]}")
                print(f"    Per-epoch diffs (last 5):  {diffs[-5:]}")
    
    # Print overall status
    print("\n" + "-" * 80)
    if status == 'SUCCESS':
        print("✓ REPRODUCTION VERIFIED: All metrics match exactly!")
        print("\nPerfect reproducibility achieved:")
        print("  - Same random seed was used")
        print("  - Completely deterministic execution")
        print("  - Identical hardware/software environment")
    elif status == 'WARNING':
        print(f"⚠️  WARNING: Some metrics differ but within tolerance ({tolerance})")
        print("\nThis may be due to:")
        print("  - Floating-point arithmetic differences")
        print("  - Random number generator implementation changes")
        print("  - GPU/CUDA version differences")
        print("\nAction: Review the differences. If tiny (< 1e-4), usually acceptable for research.")
    elif status == 'ERROR':
        print(f"❌ ERROR: Some metrics differ beyond tolerance ({tolerance})")
        print("\nPossible causes:")
        print("  - Different random seeds (check CUDA/PyTorch determinism)")
        print("  - Code changes in model/training logic")
        print("  - Different hardware/environment")
        print("  - Data preprocessing differences")
        print("\nAction: Investigate immediately. Large differences indicate a reproducibility problem.")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare metrics between original and reproduced experiments.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare by experiment ID (auto-finds reproduction folder)
  %(prog)s 983383
  
  # Compare by specifying both IDs
  %(prog)s 983383 983383_repro
  
  # Compare by full paths
  %(prog)s examples/experiments/20251102_182023_gainakt2exp_real_test_983383 \\
           examples/experiments/20251102_183045_gainakt2exp_real_test_983383_repro
  
  # Adjust tolerance
  %(prog)s 983383 --tolerance 1e-6
        """
    )
    
    parser.add_argument('original', type=str,
                       help='Original experiment ID or path to experiment folder')
    parser.add_argument('reproduction', type=str, nargs='?', default=None,
                       help='Reproduction experiment ID or path (optional, auto-detected if not provided)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Acceptable numeric difference threshold (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Find experiment folders
    try:
        original_folder = find_experiment_folder(args.original)
        
        # Auto-detect reproduction folder if not provided
        if args.reproduction is None:
            original_name = original_folder.name
            if original_name.endswith('_repro'):
                raise ValueError(f"'{args.original}' appears to be a reproduction folder. Please specify the original experiment ID.")
            
            experiments_dir = original_folder.parent
            
            # Extract experiment ID from original folder name (last component)
            # Format: YYYYMMDD_HHMMSS_modelname_shortname_EXPID
            parts = original_name.split('_')
            exp_id = parts[-1]  # e.g., "983383"
            
            # Find reproduction folder - look for folders containing the exp_id and ending with _repro
            matching_repro = [f for f in experiments_dir.iterdir() 
                            if f.is_dir() and exp_id in f.name and f.name.endswith('_repro') and f != original_folder]
            
            if not matching_repro:
                raise ValueError(f"Could not find reproduction folder for experiment ID '{exp_id}' (ending with '_repro')")
            
            if len(matching_repro) > 1:
                raise ValueError(f"Multiple reproduction folders found: {', '.join([f.name for f in matching_repro])}. Please specify explicitly.")
            
            repro_folder = matching_repro[0]
        else:
            repro_folder = find_experiment_folder(args.reproduction)
    except ValueError as e:
        print(f"❌ ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Compare metrics
    results = compare_metrics(original_folder, repro_folder, tolerance=args.tolerance)
    
    # Print results
    print_results(results, original_folder, repro_folder)
    
    # Exit with appropriate code
    if results['status'] == 'ERROR':
        sys.exit(1)
    elif results['status'] == 'WARNING':
        sys.exit(0)  # Accept within-tolerance differences
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
