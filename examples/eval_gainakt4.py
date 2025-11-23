#!/usr/bin/env python3
"""
Evaluation script for GainAKT4 model.
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import csv
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.datasets.data_loader import KTDataset
from torch.utils.data import DataLoader
from pykt.models.gainakt4 import GainAKT4
from examples.experiment_utils import compute_auc_acc


def evaluate(model, test_loader, device, lambda_bce=0.9):
    """Evaluate the model on test data."""
    model.eval()
    all_preds = []
    all_labels = []
    all_mastery_preds = []
    all_total_preds = []
    
    has_mastery = False  # Track if mastery predictions exist
    
    with torch.no_grad():
        for batch in test_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            mask = batch['masks'].to(device)
            labels = batch['shft_rseqs'].to(device)
            
            # Forward pass
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            
            # Collect predictions
            preds = outputs['bce_predictions'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Compute combined predictions (handle None mastery_predictions when λ=1.0)
            if outputs['mastery_predictions'] is not None:
                has_mastery = True
                mastery_preds = outputs['mastery_predictions'].cpu().numpy()
                combined_preds = lambda_bce * preds + (1 - lambda_bce) * mastery_preds
            else:
                mastery_preds = None
                combined_preds = preds  # Pure BCE mode when λ=1.0
            
            # Flatten and filter by mask
            for i in range(len(preds)):
                valid_indices = mask_np[i] == 1
                all_preds.extend(preds[i][valid_indices])
                all_labels.extend(labels_np[i][valid_indices])
                if mastery_preds is not None:
                    all_mastery_preds.extend(mastery_preds[i][valid_indices])
                all_total_preds.extend(combined_preds[i][valid_indices])
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_total_preds = np.array(all_total_preds)
    
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    total_metrics = compute_auc_acc(all_labels, all_total_preds)
    
    # Only compute mastery metrics if mastery predictions exist
    if has_mastery:
        all_mastery_preds = np.array(all_mastery_preds)
        mastery_metrics = compute_auc_acc(all_labels, all_mastery_preds)
    else:
        mastery_metrics = {'auc': 'N/A', 'acc': 'N/A'}
    
    return {
        'total_auc': total_metrics['auc'],
        'total_acc': total_metrics['acc'],
        'bce_auc': bce_metrics['auc'],
        'bce_acc': bce_metrics['acc'],
        'mastery_auc': mastery_metrics['auc'],
        'mastery_acc': mastery_metrics['acc']
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate GainAKT4 model')
    parser.add_argument('--run_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--ckpt_name', type=str, required=True, help='Checkpoint filename')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GainAKT4 Evaluation")
    print("="*80)
    print(f"Run directory: {args.run_dir}")
    print(f"Checkpoint: {args.ckpt_name}")
    print("="*80)
    
    # Load config
    config_path = os.path.join(args.run_dir, 'config.json')
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    # Extract parameters from defaults section
    config = full_config['defaults']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    print("\n📊 Loading dataset...")
    
    # Setup data config following PyKT standards
    data_config = {
        "assist2015": {
            "dpath": "/workspaces/pykt-toolkit/data/assist2015",
            "num_q": 0,
            "num_c": 100,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": 3,
            "maxlen": 200,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1, 2, 3, 4],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv"
        }
    }
    
    # Initialize train and valid loaders
    train_loader, valid_loader = init_dataset4train(
        config['dataset'], 'gainakt4', data_config, config['fold'], config['batch_size']
    )
    
    # Create test loader separately
    test_cfg = data_config[config['dataset']]
    test_dataset = KTDataset(
        os.path.join(test_cfg['dpath'], test_cfg['test_file']),
        test_cfg['input_type'],
        {-1}
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=int(os.getenv('PYKT_NUM_WORKERS', '32')),
        pin_memory=True
    )
    
    num_c = data_config[config['dataset']]['num_c']
    
    print(f"✓ Dataset loaded: {num_c} concepts")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n🏗️  Loading model...")
    model = GainAKT4(
        num_c=num_c,
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_blocks=config['num_encoder_blocks'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        emb_type=config['emb_type'],
        lambda_bce=config['lambda_bce']
    ).to(device)
    
    # Multi-GPU support: wrap model with DataParallel if multiple GPUs available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"✓ Multi-GPU detected: {torch.cuda.device_count()} GPUs available")
        print(f"✓ Wrapping model with DataParallel for multi-GPU evaluation")
        model = torch.nn.DataParallel(model)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.run_dir, args.ckpt_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"✓ Validation AUC at checkpoint: {checkpoint['val_bce_auc']:.4f}")
    
    # Evaluate on all splits
    print("\n🔍 Evaluating on all splits...")
    lambda_bce = config['lambda_bce']
    
    print("Evaluating on training set...")
    train_metrics = evaluate(model, train_loader, device, lambda_bce)
    
    print("Evaluating on validation set...")
    valid_metrics = evaluate(model, valid_loader, device, lambda_bce)
    
    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, lambda_bce)
    
    # Combine results
    results = {
        'train': train_metrics,
        'valid': valid_metrics,
        'test': test_metrics,
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_val_auc': checkpoint['val_bce_auc']
    }
    
    print("\n" + "="*80)
    print("Test Results:")
    print("="*80)
    print(f"Combined (Total) - AUC: {test_metrics['total_auc']:.4f}, Acc: {test_metrics['total_acc']:.4f}")
    print(f"BCE Head - AUC: {test_metrics['bce_auc']:.4f}, Acc: {test_metrics['bce_acc']:.4f}")
    mastery_auc_str = f"{test_metrics['mastery_auc']:.4f}" if isinstance(test_metrics['mastery_auc'], float) else test_metrics['mastery_auc']
    mastery_acc_str = f"{test_metrics['mastery_acc']:.4f}" if isinstance(test_metrics['mastery_acc'], float) else test_metrics['mastery_acc']
    print(f"Mastery Head - AUC: {mastery_auc_str}, Acc: {mastery_acc_str}")
    print("="*80)
    
    # Save results JSON
    results_path = os.path.join(args.run_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save config_eval.json for reproducibility
    config_eval = {
        'runtime': {
            'eval_command': ' '.join(sys.argv),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'checkpoint_path': checkpoint_path
        },
        'evaluation': {
            'batch_size': config['batch_size']
        },
        'data': {
            'dataset': config['dataset'],
            'fold': config['fold']
        },
        'model_config': {
            'seq_len': config['seq_len'],
            'd_model': config['d_model'],
            'n_heads': config['n_heads'],
            'num_encoder_blocks': config['num_encoder_blocks'],
            'd_ff': config['d_ff'],
            'dropout': config['dropout'],
            'emb_type': config['emb_type'],
            'lambda_bce': config['lambda_bce']
        }
    }
    
    config_eval_path = os.path.join(args.run_dir, 'config_eval.json')
    with open(config_eval_path, 'w') as f:
        json.dump(config_eval, f, indent=2)
    
    # Save metrics_epoch_eval.csv for reproducibility
    metrics_csv_path = os.path.join(args.run_dir, 'metrics_epoch_eval.csv')
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'split', 'total_auc', 'bce_auc', 'mastery_auc',
            'total_acc', 'bce_acc', 'mastery_acc'
        ])
        writer.writeheader()
        writer.writerow({
            'split': 'training',
            'total_auc': train_metrics['total_auc'],
            'bce_auc': train_metrics['bce_auc'],
            'mastery_auc': train_metrics['mastery_auc'],
            'total_acc': train_metrics['total_acc'],
            'bce_acc': train_metrics['bce_acc'],
            'mastery_acc': train_metrics['mastery_acc']
        })
        writer.writerow({
            'split': 'validation',
            'total_auc': valid_metrics['total_auc'],
            'bce_auc': valid_metrics['bce_auc'],
            'mastery_auc': valid_metrics['mastery_auc'],
            'total_acc': valid_metrics['total_acc'],
            'bce_acc': valid_metrics['bce_acc'],
            'mastery_acc': valid_metrics['mastery_acc']
        })
        writer.writerow({
            'split': 'test',
            'total_auc': test_metrics['total_auc'],
            'bce_auc': test_metrics['bce_auc'],
            'mastery_auc': test_metrics['mastery_auc'],
            'total_acc': test_metrics['total_acc'],
            'bce_acc': test_metrics['bce_acc'],
            'mastery_acc': test_metrics['mastery_acc']
        })
    
    print(f"\n✅ Results saved to: {results_path}")
    print(f"✅ Config saved to: {config_eval_path}")
    print(f"✅ Metrics CSV saved to: {metrics_csv_path}")


if __name__ == '__main__':
    main()
