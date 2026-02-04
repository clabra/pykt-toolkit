#!/usr/bin/env python
"""
Generate data for LaTeX table \label{tab_probing} from experiment campaigns.

This script extracts H1.1 structural encoding metrics (probe fidelity and selectivity)
from completed experiment campaigns and formats them for the probing hypothesis table.

Usage:
    python examples/validation/generate_probing_table.py --campaign "20260203_205149_probe_algebra2005_382974"
    python examples/validation/generate_probing_table.py --campaign "*probe*" --output results.csv
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def extract_structural_encoding(exp_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Extract H1.1 structural encoding metrics from experiment directory.
    
    Args:
        exp_dir: Path to experiment directory (e.g., experiments/.../gtransformer/algebra2005)
    
    Returns:
        Dictionary with mean and std for each metric, or None if not available
    """
    validation_dir = exp_dir / 'validation'
    
    # Try to load aggregated file first (5-fold)
    aggregated_file = validation_dir / 'structural_encoding_aggregated.json'
    if aggregated_file.exists():
        with open(aggregated_file) as f:
            data = json.load(f)
        
        return {
            'l0_fidelity_r2': {
                'mean': data['h1_structural_encoding']['l0']['fidelity']['r2']['mean'],
                'std': data['h1_structural_encoding']['l0']['fidelity']['r2']['std']
            },
            'l0_fidelity_pearson': {
                'mean': data['h1_structural_encoding']['l0']['fidelity']['pearson_r']['mean'],
                'std': data['h1_structural_encoding']['l0']['fidelity']['pearson_r']['std']
            },
            'l0_control_r2': {
                'mean': data['h1_structural_encoding']['l0']['selectivity']['control_r2_obs']['mean'],
                'std': data['h1_structural_encoding']['l0']['selectivity']['control_r2_obs']['std']
            },
            'l0_delta_r2': {
                'mean': data['h1_structural_encoding']['l0']['selectivity']['standard_delta_r2']['mean'],
                'std': data['h1_structural_encoding']['l0']['selectivity']['standard_delta_r2']['std']
            },
            't_fidelity_r2': {
                'mean': data['h1_structural_encoding']['t']['fidelity']['r2']['mean'],
                'std': data['h1_structural_encoding']['t']['fidelity']['r2']['std']
            },
            't_fidelity_pearson': {
                'mean': data['h1_structural_encoding']['t']['fidelity']['pearson_r']['mean'],
                'std': data['h1_structural_encoding']['t']['fidelity']['pearson_r']['std']
            },
            't_control_r2': {
                'mean': data['h1_structural_encoding']['t']['selectivity']['control_r2_obs']['mean'],
                'std': data['h1_structural_encoding']['t']['selectivity']['control_r2_obs']['std']
            },
            't_delta_r2': {
                'mean': data['h1_structural_encoding']['t']['selectivity']['standard_delta_r2']['mean'],
                'std': data['h1_structural_encoding']['t']['selectivity']['standard_delta_r2']['std']
            },
            'n_folds': data.get('n_folds', 5)
        }
    
    # Otherwise, try to aggregate individual fold files
    fold_files = sorted(validation_dir.glob('structural_encoding_fold_*.json'))
    if fold_files:
        metrics = {
            'l0_fidelity_r2': [],
            'l0_fidelity_pearson': [],
            'l0_control_r2': [],
            'l0_delta_r2': [],
            't_fidelity_r2': [],
            't_fidelity_pearson': [],
            't_control_r2': [],
            't_delta_r2': []
        }
        
        for fold_file in fold_files:
            with open(fold_file) as f:
                fold_data = json.load(f)['h1_structural_encoding']
            
            metrics['l0_fidelity_r2'].append(fold_data['l0']['fidelity']['r2'])
            metrics['l0_fidelity_pearson'].append(fold_data['l0']['fidelity']['pearson_r'])
            metrics['l0_control_r2'].append(fold_data['l0']['selectivity']['control_r2_obs'])
            metrics['l0_delta_r2'].append(fold_data['l0']['selectivity']['standard_delta_r2'])
            metrics['t_fidelity_r2'].append(fold_data['t']['fidelity']['r2'])
            metrics['t_fidelity_pearson'].append(fold_data['t']['fidelity']['pearson_r'])
            metrics['t_control_r2'].append(fold_data['t']['selectivity']['control_r2_obs'])
            metrics['t_delta_r2'].append(fold_data['t']['selectivity']['standard_delta_r2'])
        
        # Compute means and stds
        result = {}
        for key, values in metrics.items():
            result[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        result['n_folds'] = len(fold_files)
        return result
    
    return None


def extract_dataset_config(exp_dir: Path) -> Tuple[int, float, float]:
    """
    Extract model configuration from experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
    
    Returns:
        Tuple of (num_attn_heads, lambda_ref, lambda_probe)
    """
    # Check first fold config
    fold_dirs = sorted(exp_dir.glob('fold_*'))
    if not fold_dirs:
        return None, None, None
    
    config_file = fold_dirs[0] / 'config.json'
    if not config_file.exists():
        return None, None, None
    
    with open(config_file) as f:
        config = json.load(f)
    
    params = config.get('params', {})
    return (
        params.get('num_attn_heads', 4),
        params.get('lambda_ref', 0.5),
        params.get('lambda_probe', 1.0)
    )


def find_experiment_dirs(base_dir: Path, campaign_patterns: str) -> List[Tuple[str, Path]]:
    """
    Find all dataset experiment directories matching campaign pattern(s).
    
    Args:
        base_dir: Base experiments directory
        campaign_patterns: Comma-separated list of glob patterns for campaign folders
    
    Returns:
        List of (dataset_name, exp_dir) tuples
    """
    results = []
    
    # Split by comma and process each pattern
    patterns = [p.strip() for p in campaign_patterns.split(',')]
    
    for campaign_pattern in patterns:
        # Find matching campaign directories
        campaign_dirs = sorted(base_dir.glob(campaign_pattern))
        
        for campaign_dir in campaign_dirs:
            # Look for gtransformer/dataset structure
            gtrans_dir = campaign_dir / 'gtransformer'
            if not gtrans_dir.exists():
                continue
            
            # Each subdirectory is a dataset
            for dataset_dir in sorted(gtrans_dir.iterdir()):
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    results.append((dataset_name, dataset_dir))
    
    return results


def format_metric(mean: float, std: float, n_folds: int) -> str:
    """Format metric as mean ± std or just mean if single fold."""
    if n_folds > 1:
        return f"{mean:.3f} ± {std:.3f}"
    else:
        return f"{mean:.3f}"


def generate_latex_table(results: Dict[str, Dict]) -> str:
    """
    Generate LaTeX table from results.
    
    Args:
        results: Dictionary mapping dataset names to metrics
    
    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{H1.1 Structural Encoding: Probe Fidelity and Selectivity}")
    lines.append("\\label{tab_probing}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Dataset & \\multicolumn{2}{c}{$L_0$ (Init. Mastery)} & \\multicolumn{2}{c}{$T$ (Learning Rate)} \\\\")
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    lines.append(" & Fidelity $R^2$ & Selectivity $\\Delta R^2$ & Fidelity $R^2$ & Selectivity $\\Delta R^2$ \\\\")
    lines.append("\\midrule")
    
    # Sort datasets alphabetically
    for dataset in sorted(results.keys()):
        data = results[dataset]
        n_folds = data['n_folds']
        
        # Format dataset name (capitalize first letter)
        dataset_display = dataset.replace('_', ' ').title()
        
        lines.append(
            f"{dataset_display} & "
            f"{format_metric(data['l0_fidelity_r2']['mean'], data['l0_fidelity_r2']['std'], n_folds)} & "
            f"{format_metric(data['l0_delta_r2']['mean'], data['l0_delta_r2']['std'], n_folds)} & "
            f"{format_metric(data['t_fidelity_r2']['mean'], data['t_fidelity_r2']['std'], n_folds)} & "
            f"{format_metric(data['t_delta_r2']['mean'], data['t_delta_r2']['std'], n_folds)} \\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return '\n'.join(lines)


def generate_markdown_table(results: Dict[str, Dict]) -> str:
    """
    Generate Markdown table from results.
    
    Args:
        results: Dictionary mapping dataset names to metrics
    
    Returns:
        Markdown table string
    """
    lines = []
    lines.append("# H1.1 Structural Encoding: Probe Fidelity and Selectivity")
    lines.append("")
    lines.append("| Dataset | L₀ Fidelity R² | L₀ Selectivity ΔR² | T Fidelity R² | T Selectivity ΔR² | Config |")
    lines.append("|---------|----------------|-------------------|---------------|-------------------|--------|")
    
    # Sort datasets alphabetically
    for dataset in sorted(results.keys()):
        data = results[dataset]
        n_folds = data['n_folds']
        config_str = f"{data['num_heads']}h, λ_ref={data['lambda_ref']}, λ_probe={data['lambda_probe']}"
        
        lines.append(
            f"| {dataset} | "
            f"{format_metric(data['l0_fidelity_r2']['mean'], data['l0_fidelity_r2']['std'], n_folds)} | "
            f"{format_metric(data['l0_delta_r2']['mean'], data['l0_delta_r2']['std'], n_folds)} | "
            f"{format_metric(data['t_fidelity_r2']['mean'], data['t_fidelity_r2']['std'], n_folds)} | "
            f"{format_metric(data['t_delta_r2']['mean'], data['t_delta_r2']['std'], n_folds)} | "
            f"{config_str} |"
        )
    
    return '\n'.join(lines)


def generate_csv_table(results: Dict[str, Dict]) -> str:
    """
    Generate CSV table from results.
    
    Args:
        results: Dictionary mapping dataset names to metrics
    
    Returns:
        CSV table string
    """
    lines = []
    lines.append("dataset,num_heads,lambda_ref,lambda_probe,n_folds,"
                "l0_fidelity_r2_mean,l0_fidelity_r2_std,l0_delta_r2_mean,l0_delta_r2_std,"
                "t_fidelity_r2_mean,t_fidelity_r2_std,t_delta_r2_mean,t_delta_r2_std")
    
    # Sort datasets alphabetically
    for dataset in sorted(results.keys()):
        data = results[dataset]
        lines.append(
            f"{dataset},"
            f"{data['num_heads']},{data['lambda_ref']},{data['lambda_probe']},{data['n_folds']},"
            f"{data['l0_fidelity_r2']['mean']:.6f},{data['l0_fidelity_r2']['std']:.6f},"
            f"{data['l0_delta_r2']['mean']:.6f},{data['l0_delta_r2']['std']:.6f},"
            f"{data['t_fidelity_r2']['mean']:.6f},{data['t_fidelity_r2']['std']:.6f},"
            f"{data['t_delta_r2']['mean']:.6f},{data['t_delta_r2']['std']:.6f}"
        )
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate probing table data from experiment campaigns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single campaign
  python examples/validation/generate_probing_table.py --campaign "20260203_205149_probe_algebra2005_382974"
  
  # Multiple campaigns (comma-separated)
  python examples/validation/generate_probing_table.py --campaign "20260203_205149_probe_algebra2005_382974,20260203_233447_probe_datasets_498219"
  
  # Multiple campaigns with wildcard
  python examples/validation/generate_probing_table.py --campaign "*probe*"
  
  # Save to CSV
  python examples/validation/generate_probing_table.py --campaign "*probe*" --format csv --output probing_results.csv
  
  # LaTeX table
  python examples/validation/generate_probing_table.py --campaign "*probe*" --format latex
        """
    )
    
    parser.add_argument(
        '--campaign',
        type=str,
        required=True,
        help='Campaign folder pattern(s) under experiments/ (comma-separated, supports wildcards)'
    )
    parser.add_argument(
        '--experiments-dir',
        type=Path,
        default=Path('experiments'),
        help='Base experiments directory (default: experiments)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'latex', 'csv'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information'
    )
    
    args = parser.parse_args()
    
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(args.experiments_dir, args.campaign)
    
    if not exp_dirs:
        print(f"ERROR: No experiment directories found matching pattern: {args.campaign}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(exp_dirs)} dataset experiment(s):", file=sys.stderr)
        for dataset, path in exp_dirs:
            print(f"  - {dataset}: {path}", file=sys.stderr)
        print(file=sys.stderr)
    
    # Extract metrics from each dataset
    results = {}
    missing = []
    
    for dataset, exp_dir in exp_dirs:
        if args.verbose:
            print(f"Processing {dataset}...", file=sys.stderr)
        
        # Extract structural encoding metrics
        metrics = extract_structural_encoding(exp_dir)
        
        if metrics is None:
            missing.append(dataset)
            if args.verbose:
                print(f"  WARNING: No structural encoding data found", file=sys.stderr)
            continue
        
        # Extract configuration
        num_heads, lambda_ref, lambda_probe = extract_dataset_config(exp_dir)
        
        # Store results
        results[dataset] = {
            **metrics,
            'num_heads': num_heads,
            'lambda_ref': lambda_ref,
            'lambda_probe': lambda_probe
        }
        
        if args.verbose:
            print(f"  ✓ Loaded {metrics['n_folds']} fold(s)", file=sys.stderr)
    
    if not results:
        print("ERROR: No valid structural encoding data found in any experiment", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose and missing:
        print(f"\nWARNING: Skipped {len(missing)} dataset(s) with missing data: {', '.join(missing)}", file=sys.stderr)
        print(file=sys.stderr)
    
    # Generate output in requested format
    if args.format == 'markdown':
        output = generate_markdown_table(results)
    elif args.format == 'latex':
        output = generate_latex_table(results)
    elif args.format == 'csv':
        output = generate_csv_table(results)
    
    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(output)
        if args.verbose:
            print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
