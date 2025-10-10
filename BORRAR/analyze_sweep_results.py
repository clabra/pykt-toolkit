#!/usr/bin/env python3
"""
Analyze Wandb Sweep Results - Find Best Parameter Combinations
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_sweep_results(results_file):
    """Comprehensive analysis of sweep results."""
    
    print("🔍 WANDB SWEEP RESULTS ANALYSIS")
    print("=" * 60)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Total runs analyzed: {len(results)}")
    
    # Convert to DataFrame for easier analysis
    df_data = []
    for result in results:
        row = {
            'run_id': result['run_id'],
            'status': result['status'],
            'duration_minutes': result.get('duration_minutes', 0),
            'gpu_id': result.get('gpu_id', -1),
        }
        
        # Add parameters
        params = result.get('parameters', {})
        for key, value in params.items():
            row[f'param_{key}'] = value
        
        # Add metrics (though they appear empty)
        metrics = result.get('metrics', {})
        for key, value in metrics.items():
            row[f'metric_{key}'] = value
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Basic statistics
    print("\n📈 EXECUTION STATISTICS")
    print("-" * 40)
    successful_runs = df[df['status'] == 'success']
    print(f"✅ Successful runs: {len(successful_runs)}")
    print(f"❌ Failed runs: {len(df) - len(successful_runs)}")
    
    if len(successful_runs) > 0:
        print(f"⏱️  Average duration: {successful_runs['duration_minutes'].mean():.1f} minutes")
        print(f"⏱️  Fastest run: {successful_runs['duration_minutes'].min():.1f} minutes")
        print(f"⏱️  Slowest run: {successful_runs['duration_minutes'].max():.1f} minutes")
    
    # GPU utilization
    print("\n🖥️  GPU UTILIZATION")
    print("-" * 40)
    if 'gpu_id' in df.columns:
        gpu_counts = df['gpu_id'].value_counts().sort_index()
        for gpu_id, count in gpu_counts.items():
            avg_duration = df[df['gpu_id'] == gpu_id]['duration_minutes'].mean()
            print(f"GPU {gpu_id}: {count} runs (avg {avg_duration:.1f}min)")
    
    # Parameter analysis
    print("\n🎯 PARAMETER ANALYSIS")
    print("-" * 40)
    
    # Find parameter columns
    param_cols = [col for col in df.columns if col.startswith('param_')]
    
    for col in param_cols:
        param_name = col.replace('param_', '')
        unique_values = df[col].unique()
        
        if len(unique_values) <= 10:  # Categorical or few values
            print(f"\n{param_name}:")
            for value in sorted(unique_values):
                count = len(df[df[col] == value])
                avg_duration = df[df[col] == value]['duration_minutes'].mean()
                print(f"  {value}: {count} runs (avg {avg_duration:.1f}min)")
        else:  # Continuous values
            print(f"\n{param_name}: {df[col].min():.4f} - {df[col].max():.4f}")
    
    # Performance analysis based on duration (since metrics are empty)
    print("\n🏆 PERFORMANCE ANALYSIS (Based on Duration)")
    print("-" * 40)
    print("💡 Note: Faster completion might indicate early stopping or better convergence")
    
    # Top 5 fastest runs (might indicate good convergence)
    fastest_runs = successful_runs.nsmallest(5, 'duration_minutes')
    
    print("\n⚡ TOP 5 FASTEST RUNS:")
    for idx, (_, row) in enumerate(fastest_runs.iterrows(), 1):
        print(f"\n#{idx}. Run {row['run_id']} - {row['duration_minutes']:.1f} minutes")
        print(f"    GPU: {row['gpu_id']}")
        print(f"    epochs: {row.get('param_epochs', 'N/A')}")
        print(f"    batch_size: {row.get('param_batch_size', 'N/A')}")
        print(f"    lr: {row.get('param_lr', 0):.4f}")
        print(f"    weight_decay: {row.get('param_weight_decay', 0):.5f}")
        print(f"    enhanced_constraints: {row.get('param_enhanced_constraints', 'N/A')}")
    
    # Slowest runs (might indicate poor parameters or convergence issues)
    slowest_runs = successful_runs.nlargest(5, 'duration_minutes')
    
    print("\n🐌 SLOWEST 5 RUNS (potential issues):")
    for idx, (_, row) in enumerate(slowest_runs.iterrows(), 1):
        print(f"\n#{idx}. Run {row['run_id']} - {row['duration_minutes']:.1f} minutes")
        print(f"    epochs: {row.get('param_epochs', 'N/A')}")
        print(f"    batch_size: {row.get('param_batch_size', 'N/A')}")
        print(f"    lr: {row.get('param_lr', 0):.4f}")
        print(f"    enhanced_constraints: {row.get('param_enhanced_constraints', 'N/A')}")
    
    # Parameter correlation with performance (duration)
    print("\n📊 PARAMETER-PERFORMANCE CORRELATIONS")
    print("-" * 40)
    
    correlations = {}
    for col in param_cols:
        param_name = col.replace('param_', '')
        if df[col].dtype in ['int64', 'float64']:
            corr = df[col].corr(df['duration_minutes'])
            if not pd.isna(corr):
                correlations[param_name] = corr
    
    # Sort correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for param, corr in sorted_corr:
        direction = "⬆️ slower" if corr > 0 else "⬇️ faster"
        print(f"{param}: {corr:.3f} ({direction})")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    # Best parameters based on fastest completion
    best_run = fastest_runs.iloc[0]
    print("🎯 RECOMMENDED PARAMETERS (from fastest run):")
    print(f"   epochs: {best_run.get('param_epochs')}")
    print(f"   batch_size: {best_run.get('param_batch_size')}")
    print(f"   lr: {best_run.get('param_lr', 0):.4f}")
    print(f"   weight_decay: {best_run.get('param_weight_decay', 0):.5f}")
    print(f"   patience: {best_run.get('param_patience')}")
    print(f"   enhanced_constraints: {best_run.get('param_enhanced_constraints')}")
    
    # Parameter frequency in top performers
    top_performers = successful_runs.nsmallest(10, 'duration_minutes')
    
    print(f"\n📊 PARAMETER FREQUENCY in TOP 10 FASTEST RUNS:")
    for param_col in param_cols:
        param_name = param_col.replace('param_', '')
        values = top_performers[param_col].value_counts()
        print(f"\n{param_name}:")
        for value, count in values.items():
            percentage = (count / len(top_performers)) * 100
            print(f"   {value}: {count}/10 ({percentage:.0f}%)")
    
    return df, fastest_runs, correlations


def check_missing_metrics_issue(results_file):
    """Analyze why metrics are missing."""
    
    print("\n🔧 DEBUGGING: WHY ARE METRICS EMPTY?")
    print("-" * 40)
    
    # This suggests the metrics extraction from training output failed
    print("❌ All runs show empty metrics: {}")
    print("\n🔍 Possible causes:")
    print("1. Training script doesn't output FINAL_RESULTS properly")
    print("2. Metrics extraction regex doesn't match output format")
    print("3. Training completed but metrics weren't captured")
    print("4. Output buffering prevented metrics capture")
    
    print("\n💡 Solutions:")
    print("1. Check training script FINAL_RESULTS output format")
    print("2. Improve metrics extraction in launch_wandb_sweep.py")
    print("3. Add explicit result file saving in training script")
    print("4. Run a single training job manually to verify output")


if __name__ == "__main__":
    import sys
    
    results_file = "offline_sweep_results_20251006_142433.json"
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    print(f"📁 Analyzing results from: {results_file}")
    
    try:
        df, best_runs, correlations = analyze_sweep_results(results_file)
        check_missing_metrics_issue(results_file)
        
        print(f"\n✅ Analysis complete!")
        print(f"📄 Results DataFrame shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()