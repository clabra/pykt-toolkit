#!/usr/bin/env python
"""
Compute 5-fold CV statistics for H1.1 Structural Encoding metrics
by reading individual fold validation results.
"""

import json
import numpy as np
from pathlib import Path
import sys

def compute_fold_stats(campaign_dir, dataset):
    """Compute 5-fold CV stats for a dataset."""
    dataset_dir = Path(campaign_dir) / "gtransformer" / dataset
    
    # Check if all folds have h12_recovery_summary.json (contains probe metrics)
    fold_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    if len(fold_dirs) != 5:
        print(f"Warning: Expected 5 folds, found {len(fold_dirs)} for {dataset}")
        return None
    
    # Collect probe metrics from each fold
    l0_fidelity, l0_pearson, l0_selectivity = [], [], []
    t_fidelity, t_pearson, t_selectivity = [], [], []
    control_l0, control_t = [], []
    
    for fold_dir in fold_dirs:
        h12_file = fold_dir / "validation" / "h12_recovery_summary.json"
        if not h12_file.exists():
            print(f"Missing {h12_file}")
            continue
            
        with open(h12_file) as f:
            data = json.load(f)
        
        # Extract probe metrics (structural encoding)
        l0_probe = data.get("l0_probe", {})
        t_probe = data.get("t_probe", {})
        
        l0_fidelity.append(l0_probe.get("r2", 0))
        l0_pearson.append(l0_probe.get("pearson_r", 0))
        t_fidelity.append(t_probe.get("r2", 0))
        t_pearson.append(t_probe.get("pearson_r", 0))
        
        # For selectivity, we need control R² - check if it exists in the file
        # If not, we'll use the aggregated validation folder results
    
    if len(l0_fidelity) != 5:
        print(f"Incomplete fold data for {dataset}")
        return None
    
    # Compute statistics
    results = {
        "dataset": dataset,
        "num_folds": 5,
        "l0": {
            "fidelity_r2_mean": np.mean(l0_fidelity),
            "fidelity_r2_std": np.std(l0_fidelity, ddof=1),
            "pearson_r_mean": np.mean(l0_pearson),
            "pearson_r_std": np.std(l0_pearson, ddof=1),
        },
        "t": {
            "fidelity_r2_mean": np.mean(t_fidelity),
            "fidelity_r2_std": np.std(t_fidelity, ddof=1),
            "pearson_r_mean": np.mean(t_pearson),
            "pearson_r_std": np.std(t_pearson, ddof=1),
        }
    }
    
    return results


if __name__ == "__main__":
    campaign_dir = "experiments/20260202_222106_benchpaper_698838"
    datasets = ["algebra2005", "bridge2algebra2006", "nips_task34"]
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}")
        print('='*70)
        
        results = compute_fold_stats(campaign_dir, dataset)
        if results:
            print(f"\nL₀ (Initial Mastery):")
            print(f"  Fidelity (R²): {results['l0']['fidelity_r2_mean']:.3f} ± {results['l0']['fidelity_r2_std']:.3f}")
            print(f"  Pearson (r):   {results['l0']['pearson_r_mean']:.3f} ± {results['l0']['pearson_r_std']:.3f}")
            
            print(f"\nT (Learning Rate):")
            print(f"  Fidelity (R²): {results['t']['fidelity_r2_mean']:.3f} ± {results['t']['fidelity_r2_std']:.3f}")
            print(f"  Pearson (r):   {results['t']['pearson_r_mean']:.3f} ± {results['t']['pearson_r_std']:.3f}")
