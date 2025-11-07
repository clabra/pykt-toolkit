#!/usr/bin/env python3
"""
Update RESULTS.csv with all experiment results from examples/experiments.

Usage:
    python examples/update_results.py

Scans all experiment directories, extracts best epoch metrics, and writes to RESULTS.csv
"""
import json
from pathlib import Path
import csv
import sys

def update_results():
    """Scan experiments and update RESULTS.csv"""
    experiments_dir = Path(__file__).parent / "experiments"
    results = []
    
    print("Scanning experiments in:", experiments_dir)
    print("=" * 80)
    
    # Scan all experiment directories
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Skip RESULTS.csv and summary files
        if exp_dir.name in ["RESULTS.csv", "RESULTS_SUMMARY.md"]:
            continue
        
        metrics_file = exp_dir / "metrics_epoch.csv"
        config_file = exp_dir / "config.json"
        
        if not metrics_file.exists():
            print(f"⚠️  No metrics: {exp_dir.name}")
            continue
        
        # Read config for experiment details
        config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"⚠️  Config error in {exp_dir.name}: {e}")
        
        # Read metrics to find best epoch
        try:
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                print(f"⚠️  Empty metrics: {exp_dir.name}")
                continue
            
            # Find best epoch by val_auc
            best_epoch = max(rows, key=lambda x: float(x.get('val_auc', 0)))
            
            # Extract experiment info
            exp_id = config.get('experiment', {}).get('experiment_id', 'unknown')
            dataset = config.get('defaults', {}).get('dataset', 'unknown')
            fold = config.get('defaults', {}).get('fold', 'unknown')
            intrinsic = config.get('defaults', {}).get('intrinsic_gain_attention', False)
            
            # Handle intrinsic flag in overrides
            if 'overrides' in config and 'intrinsic_gain_attention' in config['overrides']:
                intrinsic = config['overrides']['intrinsic_gain_attention']
            
            results.append({
                'experiment_id': exp_id,
                'experiment_dir': exp_dir.name,
                'dataset': dataset,
                'fold': fold,
                'intrinsic_gain_attention': intrinsic,
                'best_epoch': best_epoch.get('epoch', '?'),
                'best_val_auc': float(best_epoch.get('val_auc', 0)),
                'best_val_accuracy': float(best_epoch.get('val_accuracy', 0)),
                'train_loss': float(best_epoch.get('train_loss', 0)),
                'train_auc': float(best_epoch.get('train_auc', 0)),
                'mastery_correlation': float(best_epoch.get('mastery_correlation', 0)),
                'gain_correlation': float(best_epoch.get('gain_correlation', 0)),
                'monotonicity_violation_rate': float(best_epoch.get('monotonicity_violation_rate', 0)),
                'negative_gain_rate': float(best_epoch.get('negative_gain_rate', 0)),
                'bounds_violation_rate': float(best_epoch.get('bounds_violation_rate', 0)),
            })
            
            print(f"✅ {exp_dir.name}: Epoch {best_epoch.get('epoch')} - Val AUC {best_epoch.get('val_auc')}")
        
        except Exception as e:
            print(f"❌ Error processing {exp_dir.name}: {e}")
            continue
    
    # Sort by experiment_dir (timestamp)
    results.sort(key=lambda x: x['experiment_dir'])
    
    # Write to RESULTS.csv
    output_file = experiments_dir / "RESULTS.csv"
    if results:
        fieldnames = results[0].keys()
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print("=" * 80)
        print(f"✅ Wrote {len(results)} experiments to {output_file}")
        print("\nTop 5 by Val AUC:")
        sorted_by_auc = sorted(results, key=lambda x: x['best_val_auc'], reverse=True)
        for i, r in enumerate(sorted_by_auc[:5], 1):
            mode = "Intrinsic" if r['intrinsic_gain_attention'] else "Legacy"
            print(f"  {i}. [{mode}] {r['experiment_dir']}: {r['best_val_auc']:.4f} (epoch {r['best_epoch']})")
        
        print("\nIntrinsic mode experiments:", sum(1 for r in results if r['intrinsic_gain_attention']))
        print("=" * 80)
        return 0
    else:
        print("❌ No results found!")
        return 1

if __name__ == "__main__":
    sys.exit(update_results())
