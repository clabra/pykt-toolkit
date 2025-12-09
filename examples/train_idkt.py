#!/usr/bin/env python3
"""
Training script for iDKT (Interpretable Deep Knowledge Tracing) model.

This script follows pykt framework patterns for standard KT model training.
Initial version: iDKT is identical to AKT baseline.

╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️  REPRODUCIBILITY WARNING ⚠️                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DO NOT CALL THIS SCRIPT DIRECTLY FOR REPRODUCIBLE EXPERIMENTS!             ║
║                                                                              ║
║  This script requires explicit parameters. For reproducible experiments:    ║
║                                                                              ║
║      python examples/run_repro_experiment.py --model idkt --short_title ... ║
║                                                                              ║
║  The launcher will generate explicit commands with ALL parameters.          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import json
import torch
torch.set_num_threads(32)
from torch.optim import SGD, Adam
import numpy as np
from datetime import datetime

sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.models import train_model, evaluate, init_model
from pykt.datasets import init_dataset4train
from pykt.utils import set_seed

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'


def parse_args():
    parser = argparse.ArgumentParser(description="Train iDKT model")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset name (assist2009, assist2015, assist2017, statics2011)")
    parser.add_argument("--fold", type=int, required=True,
                      help="Cross-validation fold number")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, required=True,
                      help="Dimension of attention block")
    parser.add_argument("--d_ff", type=int, required=True,
                      help="Dimension of feedforward network")
    parser.add_argument("--n_heads", type=int, required=True,
                      help="Number of attention heads")
    parser.add_argument("--n_blocks", type=int, required=True,
                      help="Number of transformer blocks")
    parser.add_argument("--dropout", type=float, required=True,
                      help="Dropout rate")
    parser.add_argument("--emb_type", type=str, required=True,
                      help="Embedding type (qid)")
    parser.add_argument("--final_fc_dim", type=int, required=True,
                      help="Final fully connected layer dimension")
    parser.add_argument("--seq_len", type=int, required=True,
                      help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--seed", type=int, required=True,
                      help="Random seed")
    parser.add_argument("--epochs", type=int, required=True,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True,
                      help="Batch size")
    parser.add_argument("--learning_rate", type=float, required=True,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, required=True,
                      help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, required=True,
                      help="Optimizer (Adam, SGD)")
    parser.add_argument("--gradient_clip", type=float, required=True,
                      help="Gradient clipping value")
    parser.add_argument("--patience", type=int, required=True,
                      help="Early stopping patience")
    parser.add_argument("--l2", type=float, required=True,
                      help="L2 regularization weight for Rasch difficulty parameters")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="saved_model/idkt",
                      help="Directory to save model checkpoints")
    parser.add_argument("--use_wandb", type=int, default=0,
                      help="Use Weights & Biases logging (0 or 1)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load data configuration
    data_config_path = '/workspaces/pykt-toolkit/configs/data_config.json'
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    # Convert relative paths to absolute paths
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name]:
            dpath = data_config[dataset_name]['dpath']
            if dpath.startswith('../'):
                # Convert relative path to absolute
                data_config[dataset_name]['dpath'] = os.path.abspath(os.path.join('/workspaces/pykt-toolkit', dpath.replace('../', '')))
    
    # Prepare parameters dict (pykt convention)
    params = {
        'model_name': 'idkt',
        'dataset_name': args.dataset,
        'fold': args.fold,
        'seed': args.seed,
        'emb_type': args.emb_type,
        'save_dir': args.save_dir,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_attn_heads': args.n_heads,
        'n_blocks': args.n_blocks,
        'dropout': args.dropout,
        'final_fc_dim': args.final_fc_dim,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'optimizer': args.optimizer,
        'gradient_clip': args.gradient_clip,
        'patience': args.patience,
        'seq_len': args.seq_len,
        'l2': args.l2,
        'use_wandb': args.use_wandb,
        'add_uuid': 0  # Don't add UUID to save_dir
    }
    
    # Initialize dataset
    print(f"Loading dataset: {args.dataset}, fold: {args.fold}")
    train_loader, valid_loader = init_dataset4train(
        args.dataset, 'idkt', data_config, args.fold, args.batch_size)
    
    # Initialize model
    print(f"Initializing iDKT model...")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, n_blocks={args.n_blocks}")
    print(f"  dropout={args.dropout}, final_fc_dim={args.final_fc_dim}")
    
    # Prepare model config (use parameter names expected by model __init__)
    model_config = {
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_attn_heads': args.n_heads,  # iDKT model expects num_attn_heads
        'n_blocks': args.n_blocks,
        'dropout': args.dropout,
        'final_fc_dim': args.final_fc_dim,
        'l2': args.l2
    }
    
    model = init_model('idkt', model_config, data_config[args.dataset], args.emb_type)
    
    # Training
    print(f"Starting training for {args.epochs} epochs...")
    print(f"  learning_rate={args.learning_rate}, batch_size={args.batch_size}")
    print(f"  optimizer={args.optimizer}, l2={args.l2}")
    
    # Create optimizer (matching AKT's wandb_train.py behavior exactly)
    if args.optimizer.lower() == "sgd":
        optimizer = SGD(model.parameters(), args.learning_rate, momentum=0.9)
    elif args.optimizer.lower() == "adam":
        optimizer = Adam(model.parameters(), args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create checkpoint directory (not file path)
    ckpt_path = args.save_dir
    os.makedirs(ckpt_path, exist_ok=True)
    
    # Train model (returns: test_auc, test_acc, window_testauc, window_testacc, valid_auc, valid_acc, best_epoch)
    test_auc, test_acc, window_testauc, window_testacc, valid_auc, valid_acc, best_epoch = train_model(
        model, train_loader, valid_loader, args.epochs, optimizer, ckpt_path, 
        test_loader=None, test_window_loader=None, save_model=True
    )
    
    print(f"\nTraining completed!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Valid AUC: {valid_auc:.4f}, Valid Acc: {valid_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")
    if window_testauc != -1:
        print(f"  Window Test AUC: {window_testauc:.4f}, Window Test Acc: {window_testacc:.4f}")
    
    # Rename checkpoint file to match expected naming convention
    # pykt saves as {emb_type}_model.ckpt, but we need best_model.pt
    old_ckpt_path = os.path.join(args.save_dir, f'{args.emb_type}_model.ckpt')
    new_ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(old_ckpt_path):
        import shutil
        shutil.move(old_ckpt_path, new_ckpt_path)
        print(f"✓ Renamed checkpoint: {old_ckpt_path} → {new_ckpt_path}")
    
    # Save validation metrics (matching ikt3 structure)
    metrics_valid_path = os.path.join(args.save_dir, 'metrics_valid.csv')
    with open(metrics_valid_path, 'w') as f:
        f.write('split,auc,acc\n')
        f.write(f'validation,{valid_auc:.6f},{valid_acc:.6f}\n')
    print(f"✓ Saved validation metrics: {metrics_valid_path}")
    
    # Save results (for backward compatibility)
    results = {
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'valid_auc': float(valid_auc),
        'valid_acc': float(valid_acc),
        'window_test_auc': float(window_testauc) if window_testauc != -1 else None,
        'window_test_acc': float(window_testacc) if window_testacc != -1 else None,
        'best_epoch': int(best_epoch),
        'params': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                  for k, v in params.items()}
    }
    
    results_path = os.path.join(args.save_dir, 'results.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved training results: {results_path}")


if __name__ == "__main__":
    main()
