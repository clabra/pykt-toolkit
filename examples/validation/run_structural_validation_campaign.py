#!/usr/bin/env python
"""
Run H1.1 Structural Encoding Validation for All Datasets in a Campaign

This script automates structural encoding validation across all datasets in a campaign directory.
For each dataset, it:
1. Detects all fold directories
2. Runs structural_encoding_validation.py for each fold
3. Aggregates results across folds
4. Generates campaign-level summary table

Usage:
    python examples/validation/run_structural_validation_campaign.py \
        --campaign_dir experiments/20260202_222106_benchpaper_698838 \
        --datasets algebra2005,assist2015,bridge2algebra2006,nips_task34

    # Or process all datasets found in campaign:
    python examples/validation/run_structural_validation_campaign.py \
        --campaign_dir experiments/20260202_222106_benchpaper_698838

Output:
    - Per-fold validation results in each dataset's validation/ folder
    - Aggregated results: validation/structural_encoding_aggregated.json
    - Campaign-level summary: campaign_dir/structural_validation_summary.csv
    - Campaign-level table for paper: campaign_dir/h11_structural_encoding_table.md
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def find_datasets(campaign_dir):
    """Automatically detect all datasets in campaign directory."""
    datasets = []
    gtransformer_dir = Path(campaign_dir) / "gtransformer"
    
    if not gtransformer_dir.exists():
        print(f"Warning: {gtransformer_dir} does not exist")
        return datasets
    
    for dataset_dir in gtransformer_dir.iterdir():
        if dataset_dir.is_dir():
            datasets.append(dataset_dir.name)
    
    return sorted(datasets)


def find_fold_dirs(campaign_dir, dataset):
    """Find all fold directories for a given dataset."""
    dataset_dir = Path(campaign_dir) / "gtransformer" / dataset
    
    if not dataset_dir.exists():
        print(f"Warning: {dataset_dir} does not exist")
        return []
    
    fold_dirs = []
    for fold_dir in dataset_dir.iterdir():
        if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
            fold_dirs.append(fold_dir)
    
    return sorted(fold_dirs, key=lambda x: x.name)


def run_structural_validation_fold(exp_dir):
    """Run structural encoding validation for a single fold."""
    validation_script = Path(PROJECT_ROOT) / "examples" / "results" / "structural_encoding_validation.py"
    
    cmd = [
        sys.executable,
        str(validation_script),
        "--exp_dir", str(exp_dir)
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: Validation failed for {exp_dir}")
        print(f"  STDERR: {result.stderr}")
        return False
    
    print(f"  ✓ Completed validation for {exp_dir.name}")
    return True


def aggregate_fold_results(campaign_dir, dataset):
    """Aggregate results across all folds for a dataset."""
    aggregation_script = Path(PROJECT_ROOT) / "examples" / "results" / "aggregate_structural_validation.py"
    
    cmd = [
        sys.executable,
        str(aggregation_script),
        "--campaign_dir", str(Path(campaign_dir) / "gtransformer" / dataset)
    ]
    
    print(f"  Aggregating results: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: Aggregation failed for {dataset}")
        print(f"  STDERR: {result.stderr}")
        return None
    
    # Load aggregated results
    agg_file = Path(campaign_dir) / "gtransformer" / dataset / "validation" / "structural_encoding_aggregated.json"
    if agg_file.exists():
        with open(agg_file, 'r') as f:
            return json.load(f)
    
    return None


def generate_campaign_summary(campaign_dir, results_by_dataset):
    """Generate campaign-level summary CSV and markdown table."""
    
    summary_data = []
    
    for dataset, agg_results in results_by_dataset.items():
        if agg_results is None:
            continue
        
        for construct in ["l0", "t"]:
            if construct not in agg_results["results"]:
                continue
            
            data = agg_results["results"][construct]
            
            row = {
                "Dataset": dataset,
                "Construct": "Initial Mastery (L₀)" if construct == "l0" else "Learning Rate (T)",
                "Fidelity_R2_Mean": np.mean(data["fidelity_r2"]),
                "Fidelity_R2_Std": np.std(data["fidelity_r2"]),
                "Pearson_r_Mean": np.mean(data["fidelity_pearson"]),
                "Pearson_r_Std": np.std(data["fidelity_pearson"]),
                "Selectivity_Std_Mean": np.mean(data["selectivity_std"]),
                "Selectivity_Std_Std": np.std(data["selectivity_std"]),
                "Selectivity_Strict_Mean": np.mean(data["selectivity_strict"]),
                "Selectivity_Strict_Std": np.std(data["selectivity_strict"]),
                "Num_Folds": agg_results["num_folds"]
            }
            summary_data.append(row)
    
    if not summary_data:
        print("Warning: No results to summarize")
        return
    
    df = pd.DataFrame(summary_data)
    
    # Save CSV
    csv_path = Path(campaign_dir) / "structural_validation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Campaign summary saved to: {csv_path}")
    
    # Generate markdown table for paper
    generate_paper_table(campaign_dir, df)


def generate_paper_table(campaign_dir, df):
    """Generate formatted markdown table for Paper Table 4."""
    
    campaign_id = Path(campaign_dir).name.split("_")[-1]
    
    md_lines = [
        "## H1.1 Structural Encoding - Diagnostic Probing Results\n",
        f"**Campaign**: {Path(campaign_dir).name}\n",
        f"**Experiment ID**: {campaign_id}\n",
        "",
        "| Dataset | Construct | Fidelity (R²) | Pearson (r) | Selectivity (Δ R² Standard) | Selectivity (Δ R² Strict) |",
        "|---------|-----------|---------------|-------------|----------------------------|---------------------------|"
    ]
    
    for _, row in df.iterrows():
        dataset = row["Dataset"]
        construct = row["Construct"]
        fid_r2 = f"{row['Fidelity_R2_Mean']:.3f} ± {row['Fidelity_R2_Std']:.3f}"
        pearson = f"{row['Pearson_r_Mean']:.3f} ± {row['Pearson_r_Std']:.3f}"
        sel_std = f"{row['Selectivity_Std_Mean']:.3f} ± {row['Selectivity_Std_Std']:.3f}"
        sel_strict = f"{row['Selectivity_Strict_Mean']:.3f} ± {row['Selectivity_Strict_Std']:.3f}"
        
        # Highlight strong selectivity (> 0.5)
        sel_std_val = row['Selectivity_Std_Mean']
        sel_std_display = f"**{sel_std}**" if sel_std_val > 0.5 else sel_std
        
        md_lines.append(
            f"| {dataset} | {construct} | {fid_r2} | {pearson} | {sel_std_display} | {sel_strict} |"
        )
    
    md_lines.extend([
        "",
        "**Notes**:",
        f"- Results computed from campaign: {Path(campaign_dir).name}",
        "- All metrics averaged across 5-fold cross-validation",
        "- **Selectivity (Δ R²)**: Difference between fidelity and control task R²",
        "  - **Standard**: Observation-level shuffling (baseline control)",
        "  - **Strict**: Skill-consistent shuffling (stronger control, prevents skill-level artifacts)",
        "- **Threshold > 0.5 indicates strong structural encoding** (marked in bold)",
        "- Control R² < 0 confirms probes cannot recover shuffled targets",
        ""
    ])
    
    md_path = Path(campaign_dir) / "h11_structural_encoding_table.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
    
    print(f"✓ Paper table saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run H1.1 structural encoding validation for all datasets in a campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--campaign_dir", 
        type=str, 
        required=True,
        help="Path to campaign directory (e.g., experiments/20260202_222106_benchpaper_698838)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to process (default: auto-detect all)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip folds that already have validation results"
    )
    
    args = parser.parse_args()
    
    campaign_dir = Path(args.campaign_dir)
    if not campaign_dir.exists():
        print(f"Error: Campaign directory does not exist: {campaign_dir}")
        return
    
    # Determine datasets to process
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = find_datasets(campaign_dir)
        if not datasets:
            print(f"Error: No datasets found in {campaign_dir}")
            return
        print(f"Auto-detected datasets: {', '.join(datasets)}")
    
    print(f"\n{'='*70}")
    print(f"H1.1 Structural Encoding Validation Campaign")
    print(f"{'='*70}")
    print(f"Campaign: {campaign_dir.name}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"{'='*70}\n")
    
    results_by_dataset = {}
    
    for dataset in datasets:
        print(f"\n{'─'*70}")
        print(f"Processing dataset: {dataset}")
        print(f"{'─'*70}")
        
        fold_dirs = find_fold_dirs(campaign_dir, dataset)
        
        if not fold_dirs:
            print(f"  Warning: No fold directories found for {dataset}")
            continue
        
        print(f"  Found {len(fold_dirs)} folds")
        
        success_count = 0
        
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            
            # Check if results already exist
            # The script creates validation dir at campaign level, so check there
            validation_dir = Path(campaign_dir) / "gtransformer" / dataset / "validation"
            json_output = validation_dir / f"structural_encoding_{fold_name}.json"
            
            if args.skip_existing and json_output.exists():
                print(f"  ⊘ Skipping {fold_name} (results already exist)")
                success_count += 1
                continue
            
            print(f"\n  [{fold_name}]")
            if run_structural_validation_fold(fold_dir):
                success_count += 1
        
        print(f"\n  Completed {success_count}/{len(fold_dirs)} folds")
        
        # Aggregate results across folds
        if success_count > 0:
            print(f"\n  Aggregating results for {dataset}...")
            agg_results = aggregate_fold_results(campaign_dir, dataset)
            results_by_dataset[dataset] = agg_results
    
    # Generate campaign-level summary
    if results_by_dataset:
        print(f"\n{'='*70}")
        print("Generating campaign-level summary...")
        print(f"{'='*70}")
        generate_campaign_summary(campaign_dir, results_by_dataset)
    
    print(f"\n{'='*70}")
    print("Campaign validation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
