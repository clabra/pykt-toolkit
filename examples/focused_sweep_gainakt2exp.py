#!/usr/bin/env python3
"""
Focused Wandb Sweep for GainAKT2Exp - Parameter Optimization Around Current Defaults

This sweep focuses on fine-tuning parameters around the current defaults to find AUC >= 0.7259
"""

import wandb
import os
import sys
import subprocess
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

def create_focused_sweep_config():
    """Create focused sweep configuration around current defaults."""
    
    sweep_config = {
        'method': 'bayes',
        'name': f'gainakt2exp_focused_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'metric': {
            'name': 'best_val_auc',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 0.0008
            },
            'weight_decay': {
                'distribution': 'log_uniform_values', 
                'min': 0.00002,
                'max': 0.0002
            },
            'batch_size': {
                'values': [64, 96, 128, 160, 192]
            },
            'num_epochs': {
                'values': [15, 18, 20, 22, 25, 30]
            },
            'enhanced_constraints': {
                'values': [True, False]
            },
            'patience': {
                'values': [15, 20, 25]
            },
            'dataset_name': {'value': 'assist2015'},
            'fold': {'value': 0}
        }
    }
    
    return sweep_config

def run_single_experiment(params):
    """Run a single experiment with given parameters."""
    
    # Convert params to command line arguments
    cmd = [
        'python', 'wandb_gainakt2exp_train.py',
        '--learning_rate', str(params['learning_rate']),
        '--weight_decay', str(params['weight_decay']),
        '--batch_size', str(params['batch_size']),
        '--num_epochs', str(params['num_epochs']),
        '--enhanced_constraints', str(params['enhanced_constraints']),
        '--patience', str(params['patience']),
        '--use_wandb', '1',
        '--experiment_suffix', f"focused_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Parse results from stdout
            output_lines = result.stdout.strip().split('\\n')
            
            best_val_auc = None
            for line in output_lines:
                if 'FINAL_RESULTS: best_val_auc:' in line:
                    best_val_auc = float(line.split(':')[-1].strip())
                    break
            
            if best_val_auc is not None:
                # Log results to wandb
                wandb.log({
                    'best_val_auc': best_val_auc,
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay'],
                    'batch_size': params['batch_size'],
                    'num_epochs': params['num_epochs'],
                    'enhanced_constraints': params['enhanced_constraints'],
                    'patience': params['patience']
                })
                
                print(f"✅ Experiment completed - AUC: {best_val_auc:.4f}")
                return best_val_auc
            else:
                print(f"❌ Could not parse AUC from output")
                return None
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ Experiment timed out after 1 hour")
        return None
    except Exception as e:
        print(f"❌ Experiment failed with exception: {e}")
        return None

def sweep_function():
    """Main sweep function called by wandb agent."""
    
    # Initialize wandb run
    with wandb.init() as run:
        # Get parameters from wandb
        params = dict(wandb.config)
        
        print(f"\\n{'='*60}")
        print(f"STARTING EXPERIMENT")
        print(f"{'='*60}")
        print(f"Parameters: {params}")
        
        # Run experiment
        best_val_auc = run_single_experiment(params)
        
        if best_val_auc is not None:
            # Log final metric
            wandb.log({'best_val_auc': best_val_auc})
            print(f"✅ Final AUC: {best_val_auc:.4f}")
        else:
            print(f"❌ Experiment failed")

def main():
    parser = argparse.ArgumentParser(description='Run focused GainAKT2Exp parameter sweep')
    parser.add_argument('--count', type=int, default=15, help='Number of experiments to run')
    parser.add_argument('--project', type=str, default='pykt-gainakt2exp-focused', 
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    # Set wandb to offline mode to avoid network issues
    os.environ['WANDB_MODE'] = 'offline'
    
    # Create sweep configuration
    sweep_config = create_sweep_config()
    
    print("Focused GainAKT2Exp Parameter Sweep")
    print("="*50)
    print(f"Target: Find combinations achieving AUC >= 0.7259")
    print(f"Experiments: {args.count}")
    print(f"Mode: Offline (results saved locally)")
    print(f"Base learning_rate: 0.0003")
    print(f"Base weight_decay: 0.000059")
    print(f"Base batch_size: 128")
    print(f"Base epochs: 20")
    
    # Create sweep  
    try:
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"\\n📊 Created sweep: {sweep_id}")
        
        # Run sweep
        print(f"🚀 Starting {args.count} experiments...")
        wandb.agent(sweep_id, function=sweep_function, count=args.count)
        
        print(f"\\n✅ Sweep completed!")
        print(f"📄 Results saved locally (offline mode)")
        print(f"🔍 Check wandb local files for detailed results")
        
    except Exception as e:
        print(f"❌ Sweep failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)