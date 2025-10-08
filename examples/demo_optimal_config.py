#!/usr/bin/env python3
"""
DEMONSTRATION: Using Optimal GainAKT2Exp Configuration
=====================================================

This script shows how to use the optimal parameters that achieved 0.7260 AUC
with perfect consistency in your own training code.
"""

import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

from optimal_config import get_optimal_config, get_optimal_args
from train_gainakt2exp import train_gainakt2exp_model
import argparse

def demo_optimal_training():
    """Demonstrate training with optimal configuration."""
    
    print("🎯 TRAINING GAINAKT2EXP WITH OPTIMAL CONFIGURATION")
    print("=" * 60)
    
    # Method 1: Get config as dictionary
    config = get_optimal_config()
    print("📊 Using optimal configuration:")
    print(f"   Learning Rate: {config['learning_rate']:.6f}")
    print(f"   Weight Decay: {config['weight_decay']:.6e}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Enhanced Constraints: {config['enhanced_constraints']}")
    print(f"   Expected AUC: {config['expected_auc']:.4f}")
    
    # Method 2: Get config as namespace (compatible with argparse)
    args = get_optimal_args()
    
    print(f"\n🚀 Starting training...")
    print(f"   This should achieve ~{config['expected_auc']:.4f} AUC")
    print(f"   Peak expected at epoch {config['peak_epoch']}")
    print(f"   Training time: ~{config['training_time_minutes']} minutes")
    
    # Train the model with optimal parameters
    try:
        train_gainakt2exp_model(args)
        print(f"\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")

def create_custom_training_script():
    """Create a custom training script with optimal defaults pre-set."""
    
    script_content = '''#!/usr/bin/env python3
"""
Custom GainAKT2Exp Training with Optimal Defaults
Automatically uses the configuration that achieved 0.7260 AUC
"""

import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

from optimal_config import get_optimal_config
from train_gainakt2exp import train_gainakt2exp_model
import argparse

def main():
    """Train GainAKT2Exp with optimal defaults, allowing override via command line."""
    
    # Get optimal configuration
    optimal = get_optimal_config()
    
    parser = argparse.ArgumentParser(description='Train GainAKT2Exp with optimal defaults')
    
    # Add arguments with optimal defaults
    parser.add_argument('--learning_rate', type=float, default=optimal['learning_rate'],
                       help=f'Learning rate (optimal: {optimal["learning_rate"]:.6f})')
    parser.add_argument('--weight_decay', type=float, default=optimal['weight_decay'],
                       help=f'Weight decay (optimal: {optimal["weight_decay"]:.6e})')
    parser.add_argument('--batch_size', type=int, default=optimal['batch_size'],
                       help=f'Batch size (optimal: {optimal["batch_size"]})')
    parser.add_argument('--num_epochs', type=int, default=optimal['num_epochs'],
                       help=f'Number of epochs (optimal peaks at epoch {optimal["peak_epoch"]})')
    parser.add_argument('--enhanced_constraints', type=bool, default=optimal['enhanced_constraints'],
                       help='CRITICAL: Enhanced constraints for perfect consistency')
    parser.add_argument('--dataset_name', type=str, default=optimal['dataset_name'],
                       help='Dataset name')
    parser.add_argument('--fold', type=int, default=optimal['fold'],
                       help='Dataset fold')
    parser.add_argument('--seed', type=int, default=optimal['seed'],
                       help='Random seed for reproducibility')
    parser.add_argument('--experiment_suffix', type=str, default=optimal['experiment_suffix'],
                       help='Experiment suffix')
    
    args = parser.parse_args()
    
    print("🎯 OPTIMAL GAINAKT2EXP TRAINING")
    print(f"Expected AUC: ~{optimal['expected_auc']:.4f}")
    print(f"Peak at epoch: {optimal['peak_epoch']}")
    print("=" * 50)
    
    # Train with optimal configuration
    train_gainakt2exp_model(args)

if __name__ == "__main__":
    main()
'''
    
    with open('/workspaces/pykt-toolkit/examples/train_optimal_gainakt2exp.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created train_optimal_gainakt2exp.py")

def show_parameter_comparison():
    """Show comparison between old and new optimal parameters."""
    
    print("🔄 PARAMETER COMPARISON: Old vs OPTIMAL")
    print("=" * 60)
    
    # Old parameters (Run 13)
    old_params = {
        'learning_rate': 0.0003479574665378502,
        'weight_decay': 5.857348235335391e-05,
        'batch_size': 128,
        'auc': 0.7259
    }
    
    # New optimal parameters
    new_params = get_optimal_config()
    
    print(f"📊 Learning Rate:")
    print(f"   Old: {old_params['learning_rate']:.6f}")
    print(f"   NEW: {new_params['learning_rate']:.6f} ({new_params['learning_rate']/old_params['learning_rate']*100:.1f}% of old)")
    
    print(f"\n⚖️  Weight Decay:")
    print(f"   Old: {old_params['weight_decay']:.6e}")
    print(f"   NEW: {new_params['weight_decay']:.6e} ({new_params['weight_decay']/old_params['weight_decay']*100:.1f}% of old)")
    
    print(f"\n📦 Batch Size:")
    print(f"   Old: {old_params['batch_size']}")
    print(f"   NEW: {new_params['batch_size']}")
    
    print(f"\n🎯 Performance:")
    print(f"   Old AUC: {old_params['auc']:.4f}")
    print(f"   NEW AUC: {new_params['expected_auc']:.4f} (+{new_params['expected_auc']-old_params['auc']:.4f})")
    
    print(f"\n✨ Key Improvements:")
    print(f"   • Higher AUC: {new_params['expected_auc']:.4f} vs {old_params['auc']:.4f}")
    print(f"   • Perfect consistency: 0% violations")
    print(f"   • Faster training: peaks at epoch {new_params['peak_epoch']}")
    print(f"   • Lower resource usage: batch size {new_params['batch_size']} vs {old_params['batch_size']}")

if __name__ == "__main__":
    print("🎯 OPTIMAL GAINAKT2EXP CONFIGURATION DEMO")
    print("=" * 60)
    
    show_parameter_comparison()
    print("\\n" + "="*60)
    create_custom_training_script()
    print("\\n" + "="*60)
    
    response = input("\\nWould you like to run a demo training with optimal config? [y/N]: ")
    if response.lower() == 'y':
        demo_optimal_training()
    else:
        print("\\n💡 To use optimal config in your code:")
        print("   from optimal_config import get_optimal_config")
        print("   config = get_optimal_config()")
        print("   # Use config['learning_rate'], config['batch_size'], etc.")
        print("\\n🚀 Or run: python train_optimal_gainakt2exp.py")