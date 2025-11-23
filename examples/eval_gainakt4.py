#!/usr/bin/env python3
"""
Evaluation script for GainAKT4 model.

Performs both concept-level and question-level evaluation (with early/late fusion).
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
from pykt.datasets import init_test_datasets
from torch.utils.data import DataLoader
from pykt.models.gainakt4 import GainAKT4
from pykt.models import load_model, evaluate_question
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
    parser = argparse.ArgumentParser(description='GainAKT4 Evaluation - All parameters explicit')
    
    # Required parameters - experiment location
    parser.add_argument('--run_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--ckpt_name', type=str, required=True, help='Checkpoint filename (default: model_best.pth)')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, required=True, help='Dataset fold')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for evaluation')
    
    # Architecture parameters
    parser.add_argument('--seq_len', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, required=True, help='Model dimension')
    parser.add_argument('--n_heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--num_encoder_blocks', type=int, required=True, help='Number of encoder blocks')
    parser.add_argument('--d_ff', type=int, required=True, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate')
    parser.add_argument('--emb_type', type=str, required=True, help='Embedding type (qid/concept)')
    parser.add_argument('--lambda_bce', type=float, required=True, help='Weight for BCE loss (lambda1)')
    
    # Optional parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--experiment_id', type=str, help='Experiment ID for logging')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GainAKT4 Evaluation")
    print("="*80)
    print(f"Run directory: {args.run_dir}")
    print(f"Checkpoint: {args.ckpt_name}")
    print(f"Dataset: {args.dataset}, Fold: {args.fold}")
    print("="*80)
    
    # Build config from explicit CLI parameters (not from file)
    config = {
        'dataset': args.dataset,
        'fold': args.fold,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'num_encoder_blocks': args.num_encoder_blocks,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'emb_type': args.emb_type,
        'lambda_bce': args.lambda_bce
    }
    
    # Setup device
    device = torch.device(args.device)
    
    # Load dataset
    print("\n📊 Loading dataset...")
    
    # Load data config from standard location (not hardcoded)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    data_config_path = os.path.join(project_root, 'configs', 'data_config.json')
    with open(data_config_path, 'r') as f:
        data_config_all = json.load(f)
    
    if config['dataset'] not in data_config_all:
        raise ValueError(f"Dataset '{config['dataset']}' not found in data_config.json")
    
    data_config = data_config_all[config['dataset']]
    
    # Fix relative paths in data_config (they're relative to project root)
    if 'dpath' in data_config and data_config['dpath'].startswith('../'):
        data_config['dpath'] = os.path.join(project_root, data_config['dpath'][3:])
    
    # Initialize train and valid loaders
    train_loader, valid_loader = init_dataset4train(
        config['dataset'], 'gainakt4', data_config_all, config['fold'], config['batch_size']
    )
    
    # Create test loader separately
    test_dataset = KTDataset(
        os.path.join(data_config['dpath'], data_config['test_file']),
        data_config['input_type'],
        {-1}
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=int(os.getenv('PYKT_NUM_WORKERS', '32')),
        pin_memory=True
    )
    
    num_c = data_config['num_c']
    
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
    
    # ========================================
    # QUESTION-LEVEL EVALUATION (with fusion)
    # ========================================
    print("\n" + "="*80)
    print("QUESTION-LEVEL EVALUATION (Early/Late Fusion)")
    print("="*80)
    
    # Load data config to check if question-level files exist
    # Use relative path from examples/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_config_path = os.path.join(script_dir, '..', 'configs', 'data_config.json')
    with open(data_config_path, 'r') as f:
        data_config_all = json.load(f)
    
    dataset_config = data_config_all.get(config['dataset'], {})
    
    # Check if question-level test files exist
    has_question_level = 'test_question_file' in dataset_config
    
    if has_question_level:
        print(f"✓ Dataset '{config['dataset']}' supports question-level evaluation")
        print("  Loading question-level test data...")
        
        # Initialize test loaders (including question-level)
        try:
            test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(
                dataset_config, 'gainakt4', config['batch_size']
            )
            
            # Load model using pykt's load_model for compatibility with evaluate_question
            model_for_question_eval = load_model(
                'gainakt4', 
                {
                    'num_c': num_c,
                    'seq_len': config['seq_len'],
                    'd_model': config['d_model'],
                    'n_heads': config['n_heads'],
                    'num_encoder_blocks': config['num_encoder_blocks'],
                    'd_ff': config['d_ff'],
                    'dropout': config['dropout'],
                    'emb_type': config['emb_type'],
                    'lambda_bce': config['lambda_bce']
                },
                dataset_config,
                config['emb_type'],
                args.run_dir
            )
            
            # Question-level evaluation with fusion
            if test_question_loader is not None:
                print("\n📊 Evaluating on test_question_sequences (original test set)...")
                save_test_question_path = os.path.join(args.run_dir, f"gainakt4_test_question_predictions.txt")
                fusion_types = ["early_fusion", "late_fusion"]
                
                q_testaucs, q_testaccs = evaluate_question(
                    model_for_question_eval, 
                    test_question_loader, 
                    'gainakt4', 
                    fusion_types, 
                    save_test_question_path
                )
                
                print("\n" + "="*80)
                print("Question-Level Test Results:")
                print("="*80)
                for key in sorted(q_testaucs.keys()):
                    auc_val = q_testaucs[key]
                    acc_val = q_testaccs.get(key, 'N/A')
                    print(f"  {key:30s} - AUC: {auc_val:.4f}, ACC: {acc_val:.4f}" if isinstance(acc_val, float) else f"  {key:30s} - AUC: {auc_val:.4f}")
                print("="*80)
                
                # Add question-level results to results dict
                results['question_level'] = {
                    'aucs': {k: float(v) for k, v in q_testaucs.items()},
                    'accs': {k: float(v) for k, v in q_testaccs.items()}
                }
                
                # Update results JSON with question-level metrics
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n✅ Question-level results added to: {results_path}")
            
            # Window evaluation (if available)
            if test_question_window_loader is not None:
                print("\n📊 Evaluating on test_question_window_sequences (sliding window)...")
                save_test_question_window_path = os.path.join(args.run_dir, f"gainakt4_test_question_window_predictions.txt")
                
                qw_testaucs, qw_testaccs = evaluate_question(
                    model_for_question_eval,
                    test_question_window_loader,
                    'gainakt4',
                    fusion_types,
                    save_test_question_window_path
                )
                
                print("\n" + "="*80)
                print("Question-Level Window Test Results:")
                print("="*80)
                for key in sorted(qw_testaucs.keys()):
                    auc_val = qw_testaucs[key]
                    acc_val = qw_testaccs.get(key, 'N/A')
                    print(f"  {key:30s} - AUC: {auc_val:.4f}, ACC: {acc_val:.4f}" if isinstance(acc_val, float) else f"  {key:30s} - AUC: {auc_val:.4f}")
                print("="*80)
                
                # Add window results
                results['question_level_window'] = {
                    'aucs': {k: float(v) for k, v in qw_testaucs.items()},
                    'accs': {k: float(v) for k, v in qw_testaccs.items()}
                }
                
                # Update results JSON
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n✅ Window results added to: {results_path}")
                
        except Exception as e:
            print(f"⚠️  Warning: Question-level evaluation failed: {e}")
            print("   Concept-level results are still valid.")
    else:
        print(f"ℹ️  Dataset '{config['dataset']}' does not have question-level test files")
        print("   (Single-skill dataset - concept-level evaluation is sufficient)")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"Results directory: {args.run_dir}")
    print(f"  - eval_results.json: Complete evaluation metrics")
    print(f"  - metrics_epoch_eval.csv: Concept-level metrics per split")
    if has_question_level:
        print(f"  - gainakt4_test_question_predictions.txt: Question-level predictions")
        if 'question_level_window' in results:
            print(f"  - gainakt4_test_question_window_predictions.txt: Window predictions")
    print("="*80)


if __name__ == '__main__':
    main()
