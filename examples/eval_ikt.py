#!/usr/bin/env python3
"""
Evaluation script for iKT model (Interpretable Knowledge Tracing).

Evaluates concept-level performance on train/valid/test splits.
For iKT, we focus on:
- Performance prediction (BCE) accuracy (AUC, ACC)
- Skill vector {Mi} interpretability (planned future analysis)
- Rasch deviation metrics (planned future analysis)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import csv
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_test_datasets
from pykt.models.ikt import iKT
from examples.experiment_utils import compute_auc_acc


def evaluate_split(model, test_loader, device, split_name='test'):
    """Evaluate model on a data split."""
    model.eval()
    all_preds = []
    all_labels = []
    
    total_bce_loss = 0.0
    total_rasch_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            mask = batch['masks'].to(device)
            labels = batch['shft_rseqs'].to(device)
            
            # Forward pass (no Rasch targets in evaluation)
            outputs = model(q=questions, r=responses, qry=questions_shifted, rasch_targets=None)
            
            # Compute loss
            if hasattr(model, 'module'):
                loss_dict = model.module.compute_loss(outputs, labels)
            else:
                loss_dict = model.compute_loss(outputs, labels)
            
            total_bce_loss += loss_dict['bce_loss'].item()
            total_rasch_loss += loss_dict['rasch_loss'].item()
            num_batches += 1
            
            # Collect predictions
            preds = outputs['bce_predictions'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Flatten and filter by mask
            for i in range(len(preds)):
                valid_indices = mask_np[i] == 1
                all_preds.extend(preds[i][valid_indices])
                all_labels.extend(labels_np[i][valid_indices])
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) == 0:
        print(f"⚠️  Warning: No valid predictions for {split_name} split")
        return {
            'auc': 0.0,
            'acc': 0.0,
            'bce_loss': 0.0,
            'rasch_loss': 0.0,
            'num_samples': 0
        }
    
    metrics = compute_auc_acc(all_labels, all_preds)
    
    return {
        'auc': metrics['auc'],
        'acc': metrics['acc'],
        'bce_loss': total_bce_loss / num_batches if num_batches > 0 else 0.0,
        'rasch_loss': total_rasch_loss / num_batches if num_batches > 0 else 0.0,
        'num_samples': len(all_preds)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate iKT model')
    
    # Checkpoint parameters
    parser.add_argument('--run_dir', type=str, required=True, 
                       help='Directory containing the checkpoint')
    parser.add_argument('--ckpt_name', type=str, required=True, 
                       help='Checkpoint filename (e.g., model_best.pth)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    
    # Model architecture parameters (must match training)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--num_encoder_blocks', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--emb_type', type=str, required=True)
    parser.add_argument('--lambda_bce', type=float, required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--phase', type=int, default=2,
                       help='Phase for evaluation (default: 2 for full model with both heads)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("iKT Evaluation")
    print("="*80)
    print(f"Run directory: {args.run_dir}")
    print(f"Checkpoint: {args.ckpt_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load data config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'data_config.json')
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    # Verify dataset exists in config
    if args.dataset not in data_config:
        raise ValueError(f"Dataset '{args.dataset}' not found in data_config.json. Available: {list(data_config.keys())}")
    
    # Convert relative paths to absolute paths
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name] and data_config[dataset_name]['dpath'].startswith('../'):
            data_config[dataset_name]['dpath'] = os.path.join(project_root, data_config[dataset_name]['dpath'][3:])
    
    # Get dataset config
    dataset_config = data_config[args.dataset]
    num_c = dataset_config['num_c']
    
    print(f"\n📊 Loading datasets...")
    
    # Initialize datasets manually (simpler than init_test_datasets)
    from torch.utils.data import DataLoader
    from pykt.datasets.data_loader import KTDataset
    
    # Load test dataset
    test_file = os.path.join(dataset_config['dpath'], dataset_config['test_file'])
    test_dataset = KTDataset(test_file, dataset_config['input_type'], {-1})
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(os.getenv('PYKT_NUM_WORKERS', '32')),
        pin_memory=True
    )
    
    # For now, we only evaluate on test split
    # TODO: Add train/valid evaluation if needed
    train_loader = None
    valid_loader = None
    
    print(f"✓ Datasets loaded:")
    print(f"  - Test: {len(test_loader)} batches ({num_c} concepts)")
    
    # Initialize model
    print(f"\n🏗️  Initializing model...")
    model = iKT(
        num_c=num_c,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_blocks=args.num_encoder_blocks,
        d_ff=args.d_ff,
        dropout=args.dropout,
        emb_type=args.emb_type,
        lambda_bce=args.lambda_bce,
        epsilon=args.epsilon,
        phase=args.phase
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {num_params:,} parameters")
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.run_dir, args.ckpt_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n📦 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    if 'val_bce_auc' in checkpoint:
        print(f"  - Validation AUC at save time: {checkpoint['val_bce_auc']:.4f}")
    
    # Evaluate on all splits
    results = {}
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    if train_loader:
        print("\n📈 Evaluating on TRAIN split...")
        train_results = evaluate_split(model, train_loader, device, 'train')
        results['train'] = train_results
        print(f"Train - AUC: {train_results['auc']:.4f}, ACC: {train_results['acc']:.4f}, "
              f"BCE Loss: {train_results['bce_loss']:.4f}, Samples: {train_results['num_samples']}")
    
    if valid_loader:
        print("\n📊 Evaluating on VALID split...")
        valid_results = evaluate_split(model, valid_loader, device, 'valid')
        results['valid'] = valid_results
        print(f"Valid - AUC: {valid_results['auc']:.4f}, ACC: {valid_results['acc']:.4f}, "
              f"BCE Loss: {valid_results['bce_loss']:.4f}, Samples: {valid_results['num_samples']}")
    
    if test_loader:
        print("\n🎯 Evaluating on TEST split...")
        test_results = evaluate_split(model, test_loader, device, 'test')
        results['test'] = test_results
        print(f"Test - AUC: {test_results['auc']:.4f}, ACC: {test_results['acc']:.4f}, "
              f"BCE Loss: {test_results['bce_loss']:.4f}, Samples: {test_results['num_samples']}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Print summary table
    print(f"\n{'Split':<10} {'AUC':<10} {'ACC':<10} {'BCE Loss':<12} {'Samples':<10}")
    print("-" * 52)
    for split_name in ['train', 'valid', 'test']:
        if split_name in results:
            r = results[split_name]
            print(f"{split_name.capitalize():<10} {r['auc']:<10.4f} {r['acc']:<10.4f} "
                  f"{r['bce_loss']:<12.4f} {r['num_samples']:<10}")
    
    # Save results to JSON
    results_json_path = os.path.join(args.run_dir, 'eval_results.json')
    eval_results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.ckpt_name,
        'dataset': args.dataset,
        'fold': args.fold,
        'num_concepts': num_c,
        'num_parameters': num_params,
        'results': results
    }
    
    with open(results_json_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_json_path}")
    
    # Save config used for evaluation
    config_eval_path = os.path.join(args.run_dir, 'config_eval.json')
    with open(config_eval_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Evaluation config saved to: {config_eval_path}")
    
    # Append to metrics_epoch_eval.csv for reproducibility tracking
    metrics_eval_csv_path = os.path.join(args.run_dir, 'metrics_epoch_eval.csv')
    csv_exists = os.path.exists(metrics_eval_csv_path)
    
    with open(metrics_eval_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            # Write header
            writer.writerow([
                'timestamp', 'checkpoint', 'split',
                'auc', 'acc', 'bce_loss', 'rasch_loss', 'num_samples'
            ])
        
        # Write results for each split
        timestamp = datetime.now().isoformat()
        for split_name in ['train', 'valid', 'test']:
            if split_name in results:
                r = results[split_name]
                writer.writerow([
                    timestamp,
                    args.ckpt_name,
                    split_name,
                    r['auc'],
                    r['acc'],
                    r['bce_loss'],
                    r['rasch_loss'],
                    r['num_samples']
                ])
    
    print(f"✓ Metrics appended to: {metrics_eval_csv_path}")
    
    print("\n" + "="*80)
    print("✅ Evaluation completed successfully")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
