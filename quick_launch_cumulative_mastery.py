#!/usr/bin/env python3
"""
Quick launch script for training GainAKT2Monitored with cumulative mastery.
This provides easy presets for different training configurations.
"""

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Quick launch cumulative mastery training')
    
    # Preset configurations
    parser.add_argument('--preset', type=str, default='standard', 
                       choices=['quick', 'standard', 'intensive', 'correlation_focused'],
                       help='Training preset configuration')
    
    # Override parameters
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--experiment_suffix', type=str, help='Experiment suffix')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    
    args = parser.parse_args()
    
    # Define presets
    presets = {
        'quick': {
            'epochs': 20,
            'batch_size': 32,
            'lr': 0.001,
            'enhanced_constraints': True,
            'experiment_suffix': 'quick_test'
        },
        'standard': {
            'epochs': 50,
            'batch_size': 32,
            'lr': 0.001,
            'enhanced_constraints': True,
            'experiment_suffix': 'standard_v1'
        },
        'intensive': {
            'epochs': 100,
            'batch_size': 16,  # Smaller batch for better gradients
            'lr': 0.0005,     # Lower learning rate for stability
            'enhanced_constraints': True,
            'experiment_suffix': 'intensive_v1'
        },
        'correlation_focused': {
            'epochs': 75,
            'batch_size': 32,
            'lr': 0.001,
            'enhanced_constraints': True,
            'experiment_suffix': 'corr_focused_v1'
        }
    }
    
    # Get configuration
    config = presets[args.preset].copy()
    
    # Apply overrides
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.experiment_suffix is not None:
        config['experiment_suffix'] = args.experiment_suffix
    
    # Build command
    cmd = [
        sys.executable, 'train_cumulative_mastery_full.py',
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--enhanced_constraints', str(config['enhanced_constraints']),
        '--experiment_suffix', config['experiment_suffix']
    ]
    
    if args.use_wandb:
        cmd.extend(['--use_wandb', 'True'])
    
    print("🚀 LAUNCHING CUMULATIVE MASTERY TRAINING")
    print("=" * 60)
    print(f"Preset: {args.preset}")
    print(f"Configuration: {config}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Launch training
    try:
        subprocess.run(cmd, check=True)
        print("\\n🎉 Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\\n⏹️  Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()