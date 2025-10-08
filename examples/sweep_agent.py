#!/usr/bin/env python3
"""
Wandb agent for focused GainAKT2Exp sweep
This script runs as a wandb agent to execute individual sweep experiments
"""

import wandb
import sys
import os

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

def run_experiment():
    """Function called by wandb agent for each experiment"""
    
    # Initialize wandb run
    with wandb.init() as run:
        config = wandb.config
        
        print(f"\\n{'='*60}")
        print(f"EXPERIMENT: lr={config.learning_rate:.6f}, wd={config.weight_decay:.6f}")
        print(f"            bs={config.batch_size}, epochs={config.num_epochs}")
        print(f"            enhanced={config.enhanced_constraints}")
        print(f"{'='*60}")
        
        # Import and run training
        from wandb_train import main as wandb_main
        
        # Convert wandb config to params dict
        params = {
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'enhanced_constraints': config.enhanced_constraints,
            'patience': config.patience,
            'dataset_name': config.dataset_name,
            'fold': config.fold,
            'use_wandb': 1,
            'model_name': 'gainakt2exp',
            'emb_type': 'qid',
            'seed': 42,
            'add_uuid': 0,
            'experiment_suffix': 'focused_sweep'
        }
        
        try:
            # Run training through wandb_train.main()
            result = wandb_main(params)
            
            # Log success
            if result and 'best_valid_auc' in result:
                best_auc = result['best_valid_auc']
                wandb.log({'best_val_auc': best_auc})
                print(f"✅ Experiment completed - AUC: {best_auc:.4f}")
                
                # Check if we found a good result
                if best_auc >= 0.7259:
                    print(f"🎉 TARGET ACHIEVED! AUC: {best_auc:.4f} >= 0.7259")
            else:
                print("❌ Training failed - no valid result")
                wandb.log({'best_val_auc': 0.0})
                
        except Exception as e:
            print(f"❌ Experiment failed with error: {e}")
            wandb.log({'best_val_auc': 0.0})

if __name__ == "__main__":
    # Set offline mode
    os.environ['WANDB_MODE'] = 'offline'
    
    print("🤖 GainAKT2Exp Focused Sweep Agent")
    print("Waiting for sweep experiments...")
    
    # This will be called by wandb agent
    run_experiment()