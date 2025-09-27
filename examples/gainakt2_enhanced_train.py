#!/usr/bin/env python
"""
Enhanced GainAKT2 Training Script
Launch training for the improved GainAKT2Enhanced model with multi-scale attention and advanced features
Target: AUC ~0.8+ through architectural innovations
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

# Add the pykt path
sys.path.append("/workspaces/pykt-toolkit")

from pykt.models import init_model
from pykt.datasets import init_dataset4train
from pykt.models.train_model import train_model
from pykt.utils.utils import debug_print, set_seed

def enhanced_train(params):
    """Enhanced training function with advanced optimization"""
    set_seed(params["seed"])
    
    # Initialize dataset with correct parameters
    debug_print(f"Initializing dataset: {params['dataset_name']}")
    
    # Load data config
    with open("/workspaces/pykt-toolkit/configs/data_config.json") as f:
        data_config = json.load(f)
    
    # Fix data paths to be absolute
    dataset_config = data_config[params['dataset_name']].copy()
    if dataset_config['dpath'].startswith('../'):
        dataset_config['dpath'] = dataset_config['dpath'].replace('../', '/workspaces/pykt-toolkit/')
    
    train_loader, valid_loader = init_dataset4train(
        params['dataset_name'], 
        params['model_name'], 
        {params['dataset_name']: dataset_config}, 
        params['fold'], 
        params['batch_size']
    )
    
    params_str = "_".join([
        str(params['seed']), 
        str(params['fold']), 
        str(params['dropout']),
        str(params['d_model']), 
        str(params['learning_rate']),
        str(params['n_heads']), 
        str(params['num_encoder_blocks']),
        str(params['d_ff']), 
        str(params['seq_len']),
        str(int(params['use_knowledge_tracking']))
    ])
    
    ckpt_path = os.path.join(params['save_dir'], 
                           f"{params['dataset_name']}_{params['model_name']}_{params['emb_type']}_saved_model_{params_str}")
    
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
        
    print(f"Results will be saved to: {ckpt_path}")
    
    # Enhanced model configuration
    model_config = {
        "d_model": params['d_model'],
        "n_heads": params['n_heads'], 
        "num_encoder_blocks": params['num_encoder_blocks'],
        "d_ff": params['d_ff'],
        "dropout": params['dropout'],
        "seq_len": params['seq_len'],
        "use_knowledge_tracking": params['use_knowledge_tracking'],
        "temperature": params.get('temperature', 1.0)
    }
    
    debug_print(f"Model config: {model_config}")
    debug_print(f"Data config: {dataset_config}")
    
    # Initialize enhanced model
    model = init_model(params["model_name"], model_config, dataset_config, params["emb_type"])
    
    if model is None:
        print(f"Failed to initialize model: {params['model_name']}")
        return None
        
    print(f"Model initialized: {model.model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Enhanced optimizer with weight decay
    if params['optimizer'] == 'adamw':
        optimizer = optimizers.AdamW(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
    else:
        optimizer = optimizers.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params.get('weight_decay', 1e-4)
        )
    
    # Advanced learning rate scheduling
    if params.get('use_scheduler', True):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=params['learning_rate'],
            epochs=params['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Apply scheduler after each batch
        def step_scheduler():
            scheduler.step()
            
        # Monkey patch the optimizer step
        original_step = optimizer.step
        def enhanced_step():
            original_step()
            if params.get('use_scheduler', True):
                step_scheduler()
        optimizer.step = enhanced_step
    
    # Training
    debug_print("Starting enhanced training...")
    testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader,
        num_epochs=params['num_epochs'], 
        opt=optimizer, 
        ckpt_path=ckpt_path, 
        test_loader=None,
        test_window_loader=None, 
        save_model=params['save_model'],
        data_config=dataset_config,
        fold=params['fold']
    )
    
    # Save results
    results = {
        "model_name": params['model_name'],
        "dataset_name": params['dataset_name'],
        "emb_type": params['emb_type'],
        "testauc": testauc,
        "testacc": testacc, 
        "window_testauc": window_testauc,
        "window_testacc": window_testacc,
        "validauc": validauc,
        "validacc": validacc,
        "best_epoch": best_epoch,
        "params": params_str,
        "config": model_config
    }
    
    with open(os.path.join(ckpt_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"Model: {params['model_name']}")
    print(f"Dataset: {params['dataset_name']}")
    print(f"Valid AUC: {validauc:.4f}")
    print(f"Valid ACC: {validacc:.4f}")
    print(f"Test AUC: {testauc:.4f}")
    print(f"Test ACC: {testacc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced GainAKT2 Training")
    
    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="assist2015", 
                       help="Dataset name (assist2015, assist2009, etc.)")
    parser.add_argument("--model_name", type=str, default="gainakt2_enhanced",
                       help="Model name")
    parser.add_argument("--emb_type", type=str, default="qid",
                       help="Embedding type")
    
    # Enhanced model architecture parameters
    parser.add_argument("--d_model", type=int, default=256,
                       help="Model dimension (256 for enhanced performance)")
    parser.add_argument("--n_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num_encoder_blocks", type=int, default=6,
                       help="Number of encoder blocks (6 for enhanced performance)")
    parser.add_argument("--d_ff", type=int, default=768,
                       help="Feed-forward dimension (768 for optimal performance)")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")
    parser.add_argument("--seq_len", type=int, default=200,
                       help="Sequence length")
    
    # Enhanced training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate (2e-4 for enhanced training)")
    parser.add_argument("--num_epochs", type=int, default=200,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--optimizer", type=str, default="adamw",
                       choices=["adam", "adamw"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay for regularization")
    parser.add_argument("--use_scheduler", type=int, default=1,
                       help="Use OneCycleLR scheduler (1 for True)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for calibrated predictions")
    
    # Enhanced features
    parser.add_argument("--use_knowledge_tracking", type=int, default=1,
                       help="Enable knowledge state tracking (1 for True)")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument("--save_model", type=bool, default=False, help="Save model")
    parser.add_argument("--save_dir", type=str, default="saved_model", 
                       help="Save directory")
    
    args = parser.parse_args()
    
    # Convert to dictionary and add derived parameters
    params = vars(args)
    params['use_knowledge_tracking'] = bool(params['use_knowledge_tracking'])
    params['use_scheduler'] = bool(params['use_scheduler'])
    
    # Dataset-specific configurations
    if params['dataset_name'] == 'assist2015':
        params['num_c'] = 100
        params['num_q'] = 100
    elif params['dataset_name'] == 'assist2009':
        params['num_c'] = 123 
        params['num_q'] = 17751
    elif params['dataset_name'] == 'algebra2005':
        params['num_c'] = 112
        params['num_q'] = 211
    elif params['dataset_name'] == 'bridge2algebra2006':
        params['num_c'] = 493
        params['num_q'] = 1146
    else:
        print(f"Warning: Unknown dataset {params['dataset_name']}, using default values")
        params['num_c'] = 100
        params['num_q'] = 100
    
    print("=" * 100)
    print("🚀 LAUNCHING ENHANCED GAINAKT2 TRAINING")
    print("=" * 100)
    print(f"Model: {params['model_name']}")
    print(f"Dataset: {params['dataset_name']}")
    print(f"Architecture: d_model={params['d_model']}, blocks={params['num_encoder_blocks']}, d_ff={params['d_ff']}")
    print(f"Training: lr={params['learning_rate']}, epochs={params['num_epochs']}, scheduler={params['use_scheduler']}")
    print(f"Enhanced Features: knowledge_tracking={params['use_knowledge_tracking']}")
    print("=" * 100)
    
    # Run enhanced training
    results = enhanced_train(params)
    
    if results:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Best Validation AUC: {results['validauc']:.4f}")
        print(f"🎯 Target AUC: 0.8000+ (Enhanced model architecture)")
    else:
        print("\n❌ TRAINING FAILED!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)