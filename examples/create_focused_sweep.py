#!/usr/bin/env python3
"""
Focused sweep around current defaults to find better parameters for GainAKT2Exp
Target: AUC >= 0.7259
"""

import wandb
import os
import sys
from datetime import datetime

def main():
    # Set offline mode
    os.environ['WANDB_MODE'] = 'offline'
    
    # Sweep configuration focused around current defaults
    sweep_config = {
        'method': 'bayes',
        'name': f'gainakt2exp_focused_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'metric': {'name': 'best_val_auc', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,  # Around 0.0003 base
                'max': 0.0008
            },
            'weight_decay': {
                'distribution': 'log_uniform_values', 
                'min': 0.00002,  # Around 0.000059 base
                'max': 0.0002
            },
            'batch_size': {'values': [64, 96, 128, 160, 192]},
            'num_epochs': {'values': [15, 18, 20, 22, 25, 30]},
            'enhanced_constraints': {'values': [True, False]},
            'patience': {'values': [15, 20, 25]},
            'dataset_name': {'value': 'assist2015'},
            'fold': {'value': 0},
            'use_wandb': {'value': 1},
            'model_name': {'value': 'gainakt2exp'},
            'experiment_suffix': {'value': 'focused_sweep'}
        }
    }
    
    print("🔬 GainAKT2Exp Focused Parameter Sweep")
    print("="*50)
    print("Goal: Find combinations achieving AUC >= 0.7259")
    print("Search ranges:")
    print("  - learning_rate: 0.0001 to 0.0008 (base: 0.0003)")
    print("  - weight_decay: 0.00002 to 0.0002 (base: 0.000059)")
    print("  - batch_size: [64, 96, 128, 160, 192]")
    print("  - epochs: [15, 18, 20, 22, 25, 30]")
    
    try:
        # Create sweep
        sweep_id = wandb.sweep(sweep_config, project='gainakt2exp-focused-sweep')
        print(f"\\n🚀 Created sweep: {sweep_id}")
        print("\\nTo run the sweep, execute:")
        print(f"wandb agent {sweep_id}")
        print("\\nOr run multiple agents in parallel:")
        print(f"wandb agent {sweep_id} & wandb agent {sweep_id} & wandb agent {sweep_id}")
        
    except Exception as e:
        print(f"Error creating sweep: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())