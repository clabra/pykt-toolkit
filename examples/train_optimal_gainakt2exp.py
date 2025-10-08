#!/usr/bin/env python3
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
