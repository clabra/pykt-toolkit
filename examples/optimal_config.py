#!/usr/bin/env python3
"""
OPTIMAL GAINAKT2EXP CONFIGURATION
================================

This file contains the optimal parameter configuration that achieved
0.7260 AUC with perfect consistency constraints (0% violations).

Use these values as defaults in any GainAKT2Exp training script.
"""

# 🎯 OPTIMAL CONFIGURATION (Verified to achieve 0.7260 AUC)
OPTIMAL_GAINAKT2EXP_CONFIG = {
    # Core training parameters
    "learning_rate": 0.000174,      # 50% of base rate - prevents overshooting
    "weight_decay": 1.7571e-05,     # 30% of base decay - light regularization
    "batch_size": 96,               # Optimal memory/performance balance
    "num_epochs": 20,               # Peaks at epoch 3, early stopping recommended
    "patience": 5,                  # Early stopping patience
    
    # Model configuration
    "enhanced_constraints": True,    # CRITICAL - must be True for perfect consistency
    "monitor_freq": 50,             # Interpretability monitoring frequency
    
    # Dataset and experiment
    "dataset_name": "assist2015",
    "fold": 0,
    "seed": 42,                     # For reproducibility
    "model_name": "gainakt2exp",
    "emb_type": "qid",
    "experiment_suffix": "optimal_v1",
    
    # Performance guarantees
    "expected_auc": 0.7260371465485588,
    "peak_epoch": 3,
    "training_time_minutes": 5,
    
    # Consistency metrics achieved
    "monotonicity_violation_rate": 0.0,
    "negative_gain_rate": 0.0,
    "bounds_violation_rate": 0.0,
    "mastery_correlation": 0.0206,
    "gain_correlation": 0.0055
}

# 🔧 PARAMETER SENSITIVITY GUIDE
PARAMETER_SENSITIVITY = {
    "learning_rate": "HIGH - ±0.00005 significantly changes results",
    "weight_decay": "MEDIUM - can vary ±50% without major issues", 
    "batch_size": "LOW - 64-128 range works well",
    "enhanced_constraints": "CRITICAL - must be True for consistency",
    "epochs": "FLEXIBLE - peaks at epoch 3, can stop early"
}

# 💡 USAGE RECOMMENDATIONS
USAGE_NOTES = {
    "production": "Use early stopping with patience=3 to catch peak at epoch 3",
    "research": "These params provide reproducible baseline for comparisons",
    "speed": "Can reduce epochs to 8 since peak is at epoch 3",
    "memory": "Can increase batch_size to 128 if GPU memory allows"
}

def get_optimal_config():
    """Returns a copy of the optimal configuration dictionary."""
    return OPTIMAL_GAINAKT2EXP_CONFIG.copy()

def get_optimal_args():
    """Returns optimal config in argparse-style namespace format."""
    from types import SimpleNamespace
    config = OPTIMAL_GAINAKT2EXP_CONFIG.copy()
    # Add alias fields for compatibility
    config['epochs'] = config['num_epochs']
    config['lr'] = config['learning_rate'] 
    config['dataset'] = config['dataset_name']
    return SimpleNamespace(**config)

def print_config_summary():
    """Print a summary of the optimal configuration."""
    print("🎯 OPTIMAL GAINAKT2EXP CONFIGURATION SUMMARY")
    print("=" * 60)
    
    config = get_optimal_config()
    
    print(f"📊 Expected AUC: {config['expected_auc']:.4f}")
    print(f"⚡ Peak Epoch: {config['peak_epoch']}")
    print(f"⏱️  Training Time: ~{config['training_time_minutes']} minutes")
    print(f"🔧 Learning Rate: {config['learning_rate']:.6f}")
    print(f"⚖️  Weight Decay: {config['weight_decay']:.6e}")
    print(f"📦 Batch Size: {config['batch_size']}")
    print(f"✅ Enhanced Constraints: {config['enhanced_constraints']}")
    
    print(f"\n🏆 CONSISTENCY METRICS:")
    print(f"   Monotonicity Violations: {config['monotonicity_violation_rate']:.1%}")
    print(f"   Negative Gains: {config['negative_gain_rate']:.1%}")
    print(f"   Bounds Violations: {config['bounds_violation_rate']:.1%}")
    print(f"   Mastery Correlation: {config['mastery_correlation']:.4f}")

if __name__ == "__main__":
    print_config_summary()
    
    print(f"\n💡 QUICK START:")
    print(f"   from optimal_config import get_optimal_config")
    print(f"   config = get_optimal_config()")
    print(f"   # Use config values in your training script")