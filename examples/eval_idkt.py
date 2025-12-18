#!/usr/bin/env python3
"""
Evaluation script for iDKT (Interpretable Deep Knowledge Tracing) model.

This script evaluates a trained iDKT model on test data.

CRITICAL ARCHITECTURAL FLAGS (must match training configuration):
    --d_model: Dimension of attention mechanism (default: 256)
    --d_ff: Dimension of feedforward network (default: 512)
    --n_heads: Number of attention heads (default: 8)
    --n_blocks: Number of transformer encoder blocks (default: 4)
               Knowledge retriever uses 2*n_blocks blocks (8 for default)
    --final_fc_dim: Dimension of prediction head FC layer (default: 512)
    --dropout: Dropout rate (default: 0.2)
    --emb_type: Embedding type, e.g., 'qid' (default: 'qid')
    --l2: L2 regularization weight for difficulty parameters (default: 1e-5)
    --seq_len: Maximum sequence length (default: 200)

Usage:
    python examples/eval_idkt.py \\
        --checkpoint experiments/YYYYMMDD_HHMMSS_idkt_title_NNNNNN/best_model.pt \\
        --output_dir experiments/YYYYMMDD_HHMMSS_idkt_title_NNNNNN \\
        --dataset assist2015 \\
        --fold 0 \\
        --batch_size 32 \\
        --d_model 256 \\
        --d_ff 512 \\
        --n_heads 8 \\
        --n_blocks 4 \\
        --dropout 0.2 \\
        --emb_type qid \\
        --final_fc_dim 512 \\
        --seq_len 200 \\
        --l2 1e-5 \\
        --seed 42
"""

import os
import sys
import argparse
import json
import torch
import numpy as np

sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.models import evaluate, init_model
from pykt.datasets import init_dataset4train
from torch.utils.data import DataLoader
from pykt.datasets.data_loader import KTDataset
from pykt.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate iDKT model")
    
    # Model path (using --checkpoint to match ikt3 convention)
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to saved model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save evaluation results")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset name")
    parser.add_argument("--fold", type=int, required=True,
                      help="Cross-validation fold")
    parser.add_argument("--batch_size", type=int, required=True,
                      help="Batch size for evaluation")
    
    # Model architecture (must match training)
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
                      help="Embedding type")
    parser.add_argument("--final_fc_dim", type=int, required=True,
                      help="Final FC dimension")
    parser.add_argument("--seq_len", type=int, required=True,
                      help="Sequence length")
    parser.add_argument("--l2", type=float, required=True,
                      help="L2 regularization")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Prepare parameters
    params = {
        'model_name': 'idkt',
        'dataset_name': args.dataset,
        'fold': args.fold,
        'seed': args.seed,
        'emb_type': args.emb_type,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_attn_heads': args.n_heads,
        'n_blocks': args.n_blocks,
        'dropout': args.dropout,
        'final_fc_dim': args.final_fc_dim,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'l2': args.l2,
        'use_wandb': 0
    }
    
    # Load data config
    data_config_path = '../configs/data_config.json'
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    # Convert relative paths to absolute
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name]:
            dpath = data_config[dataset_name]['dpath']
            if dpath.startswith('../'):
                data_config[dataset_name]['dpath'] = os.path.abspath(
                    os.path.join(os.path.dirname(data_config_path), dpath))
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}, fold: {args.fold}")
    train_loader, valid_loader = init_dataset4train(
        args.dataset, 'idkt', data_config, args.fold, args.batch_size)
    
    # Load test datasets manually
    dataset_config = data_config[args.dataset]
    dpath = dataset_config['dpath']
    
    # Create test loader
    test_file = os.path.join(dpath, dataset_config['test_file'])
    test_dataset = KTDataset(test_file, dataset_config['input_type'], {-1})
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create test window loader if available
    if 'test_window_file' in dataset_config:
        test_window_file = os.path.join(dpath, dataset_config['test_window_file'])
        if os.path.exists(test_window_file):
            test_window_dataset = KTDataset(test_window_file, dataset_config['input_type'], {-1})
            test_window_loader = DataLoader(test_window_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        else:
            test_window_loader = None
    else:
        test_window_loader = None
    
    print(f"✓ Loaded datasets (train: {len(train_loader.dataset)}, valid: {len(valid_loader.dataset)}, test: {len(test_dataset)})")
    
    # Prepare model config
    model_config = {
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_attn_heads': args.n_heads,
        'n_blocks': args.n_blocks,
        'dropout': args.dropout,
        'final_fc_dim': args.final_fc_dim,
        'l2': args.l2
    }
    
    # Initialize model
    print("Initializing model...")
    model = init_model('idkt', model_config, data_config[args.dataset], args.emb_type)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict (from pykt train_model)
        model.load_state_dict(checkpoint)
    
    # Evaluate
    print("Evaluating on validation set...")
    valid_auc, valid_acc = evaluate(model, valid_loader, 'idkt')
    print(f"Valid AUC: {valid_auc:.4f}, Valid Acc: {valid_acc:.4f}")
    
    print("Evaluating on test set...")
    test_auc, test_acc = evaluate(model, test_loader, 'idkt')
    print(f"Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")
    
    if test_window_loader is not None:
        window_test_auc, window_test_acc = evaluate(model, test_window_loader, 'idkt')
        print(f"Window Test AUC: {window_test_auc:.4f}, Window Test Acc: {window_test_acc:.4f}")
    else:
        window_test_auc, window_test_acc = None, None
    
    # Save test metrics CSV (matching ikt3 structure)
    metrics_test_path = os.path.join(args.output_dir, 'metrics_test.csv')
    with open(metrics_test_path, 'w') as f:
        f.write('split,auc,acc\n')
        f.write(f'test,{test_auc:.6f},{test_acc:.6f}\n')
    print(f"✓ Saved test metrics: {metrics_test_path}")
    
    # Save results
    results = {
        'valid_auc': float(valid_auc),
        'valid_acc': float(valid_acc),
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'window_test_auc': float(window_test_auc) if window_test_auc else None,
        'window_test_acc': float(window_test_acc) if window_test_acc else None,
        'checkpoint': args.checkpoint
    }
    
    eval_results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_results_path}")
    
    return results


if __name__ == "__main__":
    main()
