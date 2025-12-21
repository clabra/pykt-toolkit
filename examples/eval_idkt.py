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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.models import evaluate, init_model
from pykt.datasets import init_dataset4train
from torch.utils.data import DataLoader
from pykt.datasets.data_loader import KTDataset
from pykt.utils import set_seed
import pickle
from train_idkt import evaluate_idkt_individualized


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
    parser.add_argument("--lambda_student", type=float, required=True,
                      help="Regularization for student capability parameters")
    parser.add_argument("--lambda_gap", type=float, required=True,
                      help="Regularization for student knowledge gap parameters")
    parser.add_argument("--lambda_ref", type=float, required=True,
                      help="Weight for prediction alignment loss")
    parser.add_argument("--lambda_initmastery", type=float, required=True,
                      help="Weight for initial mastery consistency loss")
    parser.add_argument("--lambda_rate", type=float, required=True,
                      help="Weight for learning rate consistency loss")
    parser.add_argument("--theory_guided", type=int, default=1,
                      help="Whether to use theory-guided evaluation metrics")
    
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
        'lambda_ref': args.lambda_ref,
        'lambda_initmastery': args.lambda_initmastery,
        'lambda_rate': args.lambda_rate,
        'theory_guided': args.theory_guided,
        'use_wandb': 0
    }
    
    # Load data config
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    # Convert relative paths to absolute
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name]:
            dpath = data_config[dataset_name]['dpath']
            if dpath.startswith('../'):
                # Strip '../' and join with project_root
                data_config[dataset_name]['dpath'] = os.path.abspath(os.path.join(project_root, dpath.replace('../', '')))
            elif not os.path.isabs(dpath):
                data_config[dataset_name]['dpath'] = os.path.abspath(os.path.join(project_root, dpath))
    
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
    
    # Load BKT Skill Parameters for grounding losses (L_init, L_rate)
    bkt_skill_params = None
    bkt_params_path = os.path.join(data_config[args.dataset]['dpath'], 'bkt_skill_params.pkl')
    if os.path.exists(bkt_params_path):
        with open(bkt_params_path, 'rb') as f:
            bkt_skill_params = pickle.load(f)
        print(f"  Loaded BKT skill parameters from: {bkt_params_path}")
    else:
        print(f"  WARNING: BKT skill parameters not found at {bkt_params_path}. Grounding losses will be empty.")
    
    # Load checkpoint to detect num_students before model init
    checkpoint_path = args.checkpoint
    print(f"Loading checkpoint from {checkpoint_path} to detect n_uid...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check for student_param in checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    n_uid = 0
    if 'student_param.weight' in state_dict:
        n_uid = state_dict['student_param.weight'].shape[0] - 1
        print(f"Detected {n_uid} students in checkpoint.")

    # Prepare model config
    model_config = {
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_attn_heads': args.n_heads,
        'n_blocks': args.n_blocks,
        'dropout': args.dropout,
        'final_fc_dim': args.final_fc_dim,
        'l2': args.l2,
        'n_uid': n_uid,
        'lambda_student': args.lambda_student,
        'lambda_gap': args.lambda_gap
    }
    
    # Initialize model
    print("Initializing model...")
    model = init_model('idkt', model_config, data_config[args.dataset], args.emb_type)
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    # Evaluate
    print("Evaluating on validation set...")
    valid_auc, valid_acc, valid_metrics = evaluate_idkt_individualized(model, valid_loader, 'cuda' if torch.cuda.is_available() else 'cpu', args, bkt_skill_params)
    print(f"Valid AUC: {valid_auc:.4f}, Valid Acc: {valid_acc:.4f}")
    
    print("Evaluating on test set...")
    test_auc, test_acc, test_metrics = evaluate_idkt_individualized(model, test_loader, 'cuda' if torch.cuda.is_available() else 'cpu', args, bkt_skill_params)
    print(f"Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")
    
    if test_window_loader is not None:
        window_test_auc, window_test_acc, window_metrics = evaluate_idkt_individualized(model, test_window_loader, 'cuda' if torch.cuda.is_available() else 'cpu', args, bkt_skill_params)
        print(f"Window Test AUC: {window_test_auc:.4f}, Window Test Acc: {window_test_acc:.4f}")
    else:
        window_test_auc, window_test_acc, window_metrics = None, None, None
    
    # Save test metrics CSV (matching ikt3 structure)
    metrics_test_path = os.path.join(args.output_dir, 'metrics_test.csv')
    with open(metrics_test_path, 'w') as f:
        f.write('split,auc,acc,l_sup,l_ref,l_init,l_rate,l_reg\n')
        f.write(f"test,{test_auc:.6f},{test_acc:.6f},{test_metrics['l_sup']:.6f},{test_metrics['l_ref']:.6f},{test_metrics['l_init']:.6f},{test_metrics['l_rate']:.6f},{test_metrics['l_reg']:.6f}\n")
    print(f"✓ Saved test metrics: {metrics_test_path}")
    
    # Save results
    results = {
        'valid_auc': float(valid_auc),
        'valid_acc': float(valid_acc),
        'valid_metrics': valid_metrics,
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'test_metrics': test_metrics,
        'window_test_auc': float(window_test_auc) if window_test_auc else None,
        'window_test_acc': float(window_test_acc) if window_test_acc else None,
        'window_metrics': window_metrics,
        'checkpoint': args.checkpoint
    }
    
    eval_results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_results_path}")
    
    return results


if __name__ == "__main__":
    main()
