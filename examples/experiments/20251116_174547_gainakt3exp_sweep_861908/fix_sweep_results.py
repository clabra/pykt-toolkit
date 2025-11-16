#!/usr/bin/env python3
"""
Fix Phase 1 Sweep Results CSV
Reprocess metrics from experiment directories that have zeros
"""

import csv
import os
import sys

def extract_metrics_from_dir(exp_dir):
    """Extract metrics from experiment directory."""
    metrics = {}
    metrics_file = os.path.join(exp_dir, 'metrics_epoch_eval.csv')
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Find the test row
                for row in rows:
                    if row.get('split', '').lower() == 'test':
                        metrics['encoder1_test_auc'] = float(row.get('encoder1_auc', 0))
                        metrics['encoder2_test_auc'] = float(row.get('encoder2_auc', 0))
                        metrics['test_auc'] = float(row.get('auc', 0))
                        metrics['test_acc'] = float(row.get('accuracy', 0))
                        break
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
    
    return metrics

def fix_csv(csv_path):
    """Fix the CSV by re-extracting metrics from experiment directories."""
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    # Read existing CSV
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Found {len(rows)} rows in CSV")
    
    # Update rows with correct metrics
    updated = 0
    for row in rows:
        if row['status'] == 'success' and row['exp_dir']:
            # Check if metrics are zero or close to zero (need fixing)
            try:
                current_e2_auc = float(row.get('encoder2_test_auc', '0'))
            except (ValueError, TypeError):
                current_e2_auc = 0.0
            if current_e2_auc < 0.001:  # Essentially zero
                metrics = extract_metrics_from_dir(row['exp_dir'])
                if metrics:
                    row['encoder1_test_auc'] = f"{metrics.get('encoder1_test_auc', 0):.4f}"
                    row['encoder2_test_auc'] = f"{metrics.get('encoder2_test_auc', 0):.4f}"
                    row['test_auc'] = f"{metrics.get('test_auc', 0):.4f}"
                    row['test_acc'] = f"{metrics.get('test_acc', 0):.4f}"
                    updated += 1
                    print(f"✓ Updated {row['short_title']}: E2_AUC={metrics.get('encoder2_test_auc', 0):.4f}")
    
    # Write updated CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✅ Updated {updated} rows in {csv_path}")
    
    # Print top 5
    successful = [r for r in rows if r['status'] == 'success' and float(r.get('encoder2_test_auc', 0)) > 0]
    if successful:
        sorted_rows = sorted(successful, key=lambda x: float(x['encoder2_test_auc']), reverse=True)
        print("\n🏆 TOP 5 BY ENCODER2 AUC:")
        for i, row in enumerate(sorted_rows[:5], 1):
            print(f"{i}. {row['short_title']}")
            print(f"   Beta={row['beta_skill_init']}, M_sat={row['m_sat_init']}, "
                  f"Gamma={row['gamma_student_init']}, Offset={row['sigmoid_offset']}")
            print(f"   E2_AUC={row['encoder2_test_auc']}, E1_AUC={row['encoder1_test_auc']}, "
                  f"Overall={row['test_auc']}")

if __name__ == '__main__':
    csv_path = '/workspaces/pykt-toolkit/examples/sweep_results/phase1_sweep_20251116_174852.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    fix_csv(csv_path)
