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
from torch.nn.functional import binary_cross_entropy
import numpy as np
import csv
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
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training Loop
    best_valid_auc = 0.0
    best_valid_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    test_auc, test_acc = -1, -1
    window_testauc, window_testacc = -1, -1
    
    # Initialize metrics_epoch.csv
    csv_path = os.path.join(args.save_dir, 'metrics_epoch.csv')
    csv_headers = ['epoch', 'train_loss', 'valid_auc', 'valid_acc']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for data in train_loader:
             # Data preparation
             q, c, r, t = data["qseqs"].to(device), data["cseqs"].to(device), data["rseqs"].to(device), data["tseqs"].to(device)
             qshft, cshft, rshft, tshft = data["shft_qseqs"].to(device), data["shft_cseqs"].to(device), data["shft_rseqs"].to(device), data["shft_tseqs"].to(device)
             m, sm = data["masks"].to(device), data["smasks"].to(device)

             cq = torch.cat((q[:,0:1], qshft), dim=1)
             cc = torch.cat((c[:,0:1], cshft), dim=1)
             cr = torch.cat((r[:,0:1], rshft), dim=1)

             # Forward
             y, reg_loss = model(cc.long(), cr.long(), cq.long())
             
             # Loss
             y_pred = torch.masked_select(y[:, 1:], sm)
             y_true = torch.masked_select(rshft, sm)
             loss = binary_cross_entropy(y_pred.double(), y_true.double()) + reg_loss
             
             optimizer.zero_grad()
             loss.backward()
             if args.gradient_clip > 0:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
             optimizer.step()
             
             train_loss += loss.item()
             train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        
        # Validation
        valid_auc, valid_acc = evaluate(model, valid_loader, 'idkt')
        
        print(f"Epoch {epoch}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Valid AUC={valid_auc:.4f}, Valid Acc={valid_acc:.4f}")
        
        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'valid_auc': valid_auc,
                'valid_acc': valid_acc
            })
            
        # Checkpoint and Early Stopping
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_acc = valid_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model directly as best_model.pt
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (AUC: {valid_auc:.4f})")
            
            # Save validation metrics
            metrics_valid_path = os.path.join(args.save_dir, 'metrics_valid.csv')
            with open(metrics_valid_path, 'w') as f:
                f.write('split,auc,acc\n')
                f.write(f'validation,{valid_auc:.6f},{valid_acc:.6f}\n')
            print(f"  ✓ Saved validation metrics: {metrics_valid_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"\nTraining completed!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Valid AUC: {best_valid_auc:.4f}, Valid Acc: {best_valid_acc:.4f}")
    
    # Save results (for backward compatibility)
    results = {
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'valid_auc': float(best_valid_auc),
        'valid_acc': float(best_valid_acc),
        'window_test_auc': float(window_testauc) if window_testauc != -1 else None,
        'window_test_acc': float(window_testacc) if window_testacc != -1 else None,
        'best_epoch': int(best_epoch),
        'params': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                  for k, v in params.items()}
    }
    
    results_path = os.path.join(args.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved training results: {results_path}")


if __name__ == "__main__":
    main()
