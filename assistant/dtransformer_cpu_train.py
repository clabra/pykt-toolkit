#!/usr/bin/env python3
"""
Memory-Optimized DTransformer Training Script for CPU-Only Execution

This script trains the DTransformer model with reduced memory requirements
by using smaller batch sizes and model dimensions suitable for CPU training.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# Add pykt to path
sys.path.append('/workspaces/pykt-toolkit')

from pykt.datasets.init_dataset import init_dataset4train
from pykt.utils.utils import set_seed
from pykt.models.init_model import init_model
from pykt.models.train_model import train_model

def create_memory_optimized_config():
    """Create configuration optimized for CPU training with reduced memory usage"""
    
    # Force CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    params = {
        'dataset_name': 'assist2015',
        'model_name': 'dtransformer',
        'emb_type': 'qid_cl',
        'save_dir': 'saved_model',
        'seed': 42,
        'fold': 0,
        'dropout': 0.2,  # Reduced from 0.3
        
        # Reduced model dimensions for memory efficiency
        'd_model': 128,      # Reduced from 256
        'd_ff': 128,         # Reduced from 256
        'num_attn_heads': 4, # Reduced from 8
        'n_blocks': 2,       # Reduced from 4
        'learning_rate': 0.001,
        
        # Reduced knowledge components
        'n_know': 8,         # Reduced from 16
        
        'lambda_cl': 0.1,
        'window': 1,
        'proj': True,
        'hard_neg': False,
        'use_wandb': 0,
        'add_uuid': 1
    }
    
    # Memory-optimized training configuration
    train_config = {
        'batch_size': 32,      # Heavily reduced from 1024
        'num_epochs': 5,       # Reduced epochs for faster completion
        'optimizer': 'adam',
        'seq_len': 100         # Reduced sequence length from 200
    }
    
    return params, train_config

def main():
    """Main training function with memory optimizations"""
    
    print("=" * 80)
    print("DTransformer CPU-Only Training (Memory Optimized)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create optimized configuration
    params, train_config = create_memory_optimized_config()
    
    print("\nConfiguration:")
    print("-" * 40)
    print(f"Dataset: {params['dataset_name']}")
    print(f"Model: {params['model_name']}")
    print(f"Device: CPU-only")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Epochs: {train_config['num_epochs']}")
    print(f"Model dimension: {params['d_model']}")
    print(f"Attention heads: {params['num_attn_heads']}")
    print(f"Transformer blocks: {params['n_blocks']}")
    print(f"Knowledge components: {params['n_know']}")
    print(f"Sequence length: {train_config['seq_len']}")
    
    # Set seed for reproducibility
    set_seed(params['seed'])
    
    # Force CPU usage
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    try:
        # Initialize dataset with reduced sequence length
        print("\nInitializing dataset...")
        
        # Temporarily modify config for shorter sequences
        config_path = "/workspaces/pykt-toolkit/configs/kt_config.json"
        original_config = None
        
        with open(config_path, 'r') as f:
            original_config = json.load(f)
        
        # Create memory-optimized config
        modified_config = original_config.copy()
        modified_config["train_config"]["batch_size"] = train_config['batch_size']
        modified_config["train_config"]["num_epochs"] = train_config['num_epochs']
        modified_config["train_config"]["seq_len"] = train_config['seq_len']
        
        with open(config_path, 'w') as f:
            json.dump(modified_config, f, indent=2)
        
        print(f"Modified config: batch_size={train_config['batch_size']}, "
              f"num_epochs={train_config['num_epochs']}, seq_len={train_config['seq_len']}")
        
        # Load data configuration
        with open('/workspaces/pykt-toolkit/configs/data_config.json', 'r') as f:
            data_config = json.load(f)
        
        # Initialize dataset using the correct function
        train_loader, valid_loader = init_dataset4train(
            dataset_name=params['dataset_name'],
            model_name=params['model_name'], 
            data_config=data_config,
            i=params['fold'],
            batch_size=train_config['batch_size']
        )
        
        print(f"Dataset loaded: {len(train_loader)} train batches, {len(valid_loader)} valid batches")
        
        # Get dataset info for model initialization
        num_q = data_config[params['dataset_name']]['num_q']
        num_c = data_config[params['dataset_name']]['num_c']
        print(f"Questions: {num_q}, Concepts: {num_c}")
        
        # Initialize model with reduced parameters
        print("\nInitializing model...")
        
        # Prepare model configuration
        model_config = {
            'dropout': params['dropout'],
            'd_model': params['d_model'],
            'd_ff': params['d_ff'],
            'num_attn_heads': params['num_attn_heads'],
            'n_blocks': params['n_blocks'],
            'n_know': params['n_know'],
            'lambda_cl': params['lambda_cl'],
            'window': params['window'],
            'proj': params['proj'],
            'hard_neg': params['hard_neg']
        }
        
        # Use the correct data config structure for the model
        model_data_config = data_config[params['dataset_name']]
        
        model = init_model(params['model_name'], model_config, model_data_config, params['emb_type'])
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Setup optimizer with weight decay
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=params['learning_rate'], 
                                   weight_decay=1e-5)
        
        # Create save directory
        params_str = f"{params['dataset_name']}_{params['model_name']}_cpu_optimized_{params['seed']}"
        save_path = f"saved_model/{params_str}"
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\nModel will be saved to: {save_path}")
        
        print("\nStarting training...")
        print("=" * 80)
        
        # Train model
        results = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=train_config['num_epochs'],
            opt=optimizer,
            ckpt_path=save_path,
            test_loader=None,
            test_window_loader=None,
            save_model=True
        )
        
        # Extract results
        test_auc, test_acc, window_test_auc, window_test_acc, valid_auc, valid_acc, best_epoch = results
        
        print("=" * 80)
        print("Training completed successfully!")
        print("-" * 40)
        print(f"Best epoch: {best_epoch}")
        print(f"Validation AUC: {valid_auc:.4f}")
        print(f"Validation ACC: {valid_acc:.4f}")
        
        if test_auc != -1:
            print(f"Test AUC: {test_auc:.4f}")
            print(f"Test ACC: {test_acc:.4f}")
        
        # Save final metrics
        final_results = {
            'model': params['model_name'],
            'dataset': params['dataset_name'],
            'device': 'cpu',
            'configuration': 'memory_optimized',
            'parameters': {
                'd_model': params['d_model'],
                'n_blocks': params['n_blocks'],
                'num_attn_heads': params['num_attn_heads'],
                'n_know': params['n_know'],
                'batch_size': train_config['batch_size'],
                'seq_len': train_config['seq_len']
            },
            'results': {
                'best_epoch': best_epoch,
                'valid_auc': float(valid_auc),
                'valid_acc': float(valid_acc),
                'test_auc': float(test_auc) if test_auc != -1 else None,
                'test_acc': float(test_acc) if test_acc != -1 else None
            },
            'model_stats': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = f"{save_path}/final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print("=" * 80)
        
        return valid_auc
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Restore original configuration
        if original_config is not None:
            try:
                with open(config_path, 'w') as f:
                    json.dump(original_config, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not restore original config: {e}")

if __name__ == "__main__":
    main()