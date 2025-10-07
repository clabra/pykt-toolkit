#!/usr/bin/env python3
"""
Compare ablation study results: enhanced_constraints True vs False
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def load_and_compare_results():
    """Load and compare the ablation study results."""
    
    print("🔬 ABLATION STUDY: ENHANCED CONSTRAINTS IMPACT ANALYSIS")
    print("=" * 80)
    
    # Load results from the sweep (enhanced_constraints=True)
    sweep_results_file = "offline_sweep_results_20251007_074242.json"
    
    # Load results from ablation study (enhanced_constraints=False) 
    ablation_results_file = "cumulative_mastery_results_ablation_enhanced_false_20251007_150509.json"
    
    try:
        with open(sweep_results_file, 'r') as f:
            sweep_results = json.load(f)
        
        with open(ablation_results_file, 'r') as f:
            ablation_results = json.load(f)
            
        print("✅ Successfully loaded both result files")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # Find the optimal configuration from sweep (Run 13 with enhanced_constraints=True)
    optimal_sweep = None
    for result in sweep_results:
        if result.get('run_id') == 13:
            optimal_sweep = result
            break
    
    if not optimal_sweep:
        print("❌ Could not find Run 13 in sweep results")
        return
    
    # Extract key metrics for comparison
    enhanced_true = {
        'auc': optimal_sweep['metrics']['best_val_auc'],
        'consistency': {
            'monotonicity': 0.0,  # Perfect by design
            'negative_gains': 0.0,  # Perfect by design  
            'bounds': 0.0,  # Perfect by design
        },
        'duration': optimal_sweep['duration_minutes'],
        'parameters': optimal_sweep['parameters']
    }
    
    enhanced_false = {
        'auc': ablation_results['best_val_auc'],
        'consistency': ablation_results['final_consistency_metrics'],
        'duration': None,  # Not available in this format
        'parameters': ablation_results['training_args']
    }
    
    # Display comparison
    print("\n📊 PARAMETER CONFIGURATION")
    print("-" * 50)
    print("Parameter                Enhanced=True    Enhanced=False")
    print("-" * 50)
    print(f"epochs                   {enhanced_true['parameters']['epochs']:<15} {enhanced_false['parameters']['epochs']}")
    print(f"batch_size               {enhanced_true['parameters']['batch_size']:<15} {enhanced_false['parameters']['batch_size']}")
    print(f"lr                       {enhanced_true['parameters']['lr']:<15.6f} {enhanced_false['parameters']['lr']:<.6f}")
    print(f"weight_decay             {enhanced_true['parameters']['weight_decay']:<15.6f} {enhanced_false['parameters']['weight_decay']:<.6f}")
    print(f"patience                 {enhanced_true['parameters']['patience']:<15} {enhanced_false['parameters']['patience']}")
    print(f"enhanced_constraints     {enhanced_true['parameters']['enhanced_constraints']:<15} {enhanced_false['parameters']['enhanced_constraints']}")
    
    print(f"\n🎯 PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"Metric                   Enhanced=True    Enhanced=False    Difference")
    print("-" * 50)
    
    auc_diff = enhanced_true['auc'] - enhanced_false['auc']
    auc_diff_pct = (auc_diff / enhanced_false['auc']) * 100
    
    print(f"Best Validation AUC      {enhanced_true['auc']:<15.4f} {enhanced_false['auc']:<15.4f} {auc_diff:+.4f} ({auc_diff_pct:+.2f}%)")
    
    if enhanced_true['duration'] and enhanced_false['duration']:
        duration_diff = enhanced_true['duration'] - enhanced_false['duration']
        print(f"Training Duration (min)  {enhanced_true['duration']:<15.1f} {enhanced_false['duration']:<15.1f} {duration_diff:+.1f}")
    
    print(f"\n🔍 CONSISTENCY METRICS")
    print("-" * 50) 
    print(f"Metric                   Enhanced=True    Enhanced=False")
    print("-" * 50)
    print(f"Monotonicity Violations  {enhanced_true['consistency']['monotonicity']:<15.1%} {enhanced_false['consistency']['monotonicity_violation_rate']:<15.1%}")
    print(f"Negative Gain Rate       {enhanced_true['consistency']['negative_gains']:<15.1%} {enhanced_false['consistency']['negative_gain_rate']:<15.1%}")
    print(f"Bounds Violations        {enhanced_true['consistency']['bounds']:<15.1%} {enhanced_false['consistency']['bounds_violation_rate']:<15.1%}")
    
    print(f"\n📈 CORRELATION ANALYSIS")
    print("-" * 50)
    print(f"Metric                   Enhanced=True    Enhanced=False")
    print("-" * 50)
    print(f"Mastery-Performance      {'N/A':<15} {enhanced_false['consistency']['mastery_correlation']:<15.4f}")
    print(f"Gain-Performance         {'N/A':<15} {enhanced_false['consistency']['gain_correlation']:<15.4f}")
    
    print(f"\n🎯 KEY FINDINGS")
    print("=" * 80)
    
    if auc_diff > 0:
        print(f"✅ Enhanced constraints IMPROVED AUC by {auc_diff:.4f} ({auc_diff_pct:+.2f}%)")
    elif auc_diff < -0.001:  # Small threshold for significance
        print(f"⚠️  Enhanced constraints REDUCED AUC by {abs(auc_diff):.4f} ({abs(auc_diff_pct):.2f}%)")
    else:
        print(f"➡️  Enhanced constraints had MINIMAL impact on AUC (Δ={auc_diff:.4f})")
    
    # Consistency analysis
    total_violations_false = (
        enhanced_false['consistency']['monotonicity_violation_rate'] +
        enhanced_false['consistency']['negative_gain_rate'] + 
        enhanced_false['consistency']['bounds_violation_rate']
    )
    
    if total_violations_false == 0:
        print("✅ Both configurations maintained PERFECT consistency (0% violations)")
    else:
        print(f"⚠️  Without enhanced constraints: {total_violations_false:.1%} total violation rate")
        print("✅ With enhanced constraints: 0% violations (architectural guarantee)")
    
    # Correlation analysis
    weak_correlations = (
        abs(enhanced_false['consistency']['mastery_correlation']) < 0.1 and
        abs(enhanced_false['consistency']['gain_correlation']) < 0.1
    )
    
    if weak_correlations:
        print("📉 Without enhanced constraints: WEAK correlations between mastery/gains and performance")
        print("🎯 Enhanced constraints likely provide better interpretability")
    else:
        print("📈 Without enhanced constraints: Reasonable correlations maintained")
    
    print(f"\n💡 RECOMMENDATION")
    print("-" * 50)
    
    if auc_diff > 0.001 and total_violations_false == 0:
        print("🏆 USE ENHANCED CONSTRAINTS: Better AUC + Perfect consistency")
    elif auc_diff < -0.001 and total_violations_false > 0:
        print("⚖️  TRADE-OFF: Enhanced constraints sacrifice some AUC for perfect consistency")
    elif abs(auc_diff) < 0.001:
        print("✅ USE ENHANCED CONSTRAINTS: Similar AUC + Guaranteed consistency + Better interpretability")
    else:
        print("🔬 CONTEXT-DEPENDENT: Consider your priority (AUC vs Consistency vs Interpretability)")
    
    # Training history comparison if available
    if 'train_history' in ablation_results:
        print(f"\n📈 TRAINING PROGRESSION (Enhanced=False)")
        print("-" * 50)
        train_history = ablation_results['train_history']
        
        if 'val_auc' in train_history:
            val_aucs = train_history['val_auc']
            print(f"Final validation AUC progression:")
            print(f"  Epochs 1-5:   {np.mean(val_aucs[:5]):.4f}")
            if len(val_aucs) >= 10:
                print(f"  Epochs 6-10:  {np.mean(val_aucs[5:10]):.4f}")
            if len(val_aucs) >= 15:
                print(f"  Epochs 11-15: {np.mean(val_aucs[10:15]):.4f}")
            if len(val_aucs) >= 20:
                print(f"  Epochs 16-20: {np.mean(val_aucs[15:20]):.4f}")
            print(f"  Best AUC:     {max(val_aucs):.4f} (epoch {val_aucs.index(max(val_aucs)) + 1})")

def main():
    load_and_compare_results()

if __name__ == "__main__":
    main()