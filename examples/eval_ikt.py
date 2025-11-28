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


def evaluate_split(model, test_loader, device, split_name='test', lambda_penalty=None, epsilon=None):
    """
    Evaluate model on a data split with comprehensive metrics.
    
    Args:
            
    Returns 13 metrics:
    - L1 Performance: l1_bce, auc, accuracy
    - L2 Alignment: l2_mse, l2_mae, corr_rasch
    - L2_penalty Violations: penalty_loss, violation_rate, mean_violation, max_violation
    - L_total Combined: total_loss, loss_ratio_l1, loss_ratio_penalty
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_skill_vectors = []
    all_beta_values = []
    
    total_bce_loss = 0.0
    total_penalty_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    # No rasch_targets needed (Option 1b) - model uses internal embeddings
    
    with torch.no_grad():
        for batch in test_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            labels = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Forward pass
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            
            # Compute loss
            if hasattr(model, 'module'):
                loss_dict = model.module.compute_loss(outputs, labels)
            else:
                loss_dict = model.compute_loss(outputs, labels)
            
            total_bce_loss += loss_dict['bce_loss'].item()
            if 'penalty_loss' in loss_dict:
                total_penalty_loss += loss_dict['penalty_loss'].item()
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            # Collect predictions and labels
            preds = outputs['bce_predictions'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Flatten and filter by mask
            for i in range(len(preds)):
                valid_indices = mask_np[i] == 1
                all_preds.extend(preds[i][valid_indices])
                all_labels.extend(labels_np[i][valid_indices])
            
            # Collect skill vectors and Rasch targets if available
            if 'skill_vector' in outputs and outputs.get("beta_targets") is not None:
                skill_vec = outputs['skill_vector'].detach().cpu().numpy()
                rasch_batch_np = outputs.get("beta_targets").cpu().numpy()
                
                # Flatten: (batch_size, seq_len, num_skills) -> list
                for i in range(len(skill_vec)):
                    all_skill_vectors.append(skill_vec[i])
                    all_beta_values.append(rasch_batch_np[i])
    
    # Compute L1 metrics (performance prediction)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) == 0:
        print(f"⚠️  Warning: No valid predictions for {split_name} split")
        return {
            'l1_bce': 0.0, 'auc': 0.0, 'accuracy': 0.0,
            'l2_mse': 0.0, 'l2_mae': 0.0, 'corr_beta': 0.0,
            'penalty_loss': 0.0, 'violation_rate': 0.0, 'mean_violation': 0.0, 'max_violation': 0.0,
            'total_loss': 0.0, 'loss_ratio_l1': 0.0, 'loss_ratio_penalty': 0.0,
            'num_samples': 0
        }
    
    performance_metrics = compute_auc_acc(all_labels, all_preds)
    l1_bce = total_bce_loss / num_batches if num_batches > 0 else 0.0
    auc = performance_metrics['auc']
    accuracy = performance_metrics['acc']
    
    # Compute L2 and L2_penalty metrics if beta targets are available
    if len(all_skill_vectors) > 0 and len(all_beta_values) > 0:
        # Flatten skill vectors and Rasch targets
        skill_flat = np.concatenate([sv.flatten() for sv in all_skill_vectors])
        rasch_flat = np.concatenate([rv.flatten() for rv in all_beta_values])
        
        # L2 alignment metrics
        l2_mse = np.mean((skill_flat - rasch_flat) ** 2)
        l2_mae = np.mean(np.abs(skill_flat - rasch_flat))
        
        # Correlation
        if np.std(skill_flat) > 1e-6 and np.std(rasch_flat) > 1e-6:
            corr_rasch = np.corrcoef(skill_flat, rasch_flat)[0, 1]
        else:
            corr_rasch = 0.0
        
        # L2_penalty violation metrics
        deviations = np.abs(skill_flat - rasch_flat)
        eps = epsilon if epsilon is not None else 0.0
        violations = np.maximum(0, deviations - eps)
        
        violation_mask = violations > 0
        violation_rate = np.mean(violation_mask)
        mean_violation = np.mean(violations[violation_mask]) if np.any(violation_mask) else 0.0
        max_violation = np.max(violations) if len(violations) > 0 else 0.0
        penalty_loss = total_penalty_loss / num_batches if num_batches > 0 else 0.0
    else:
        # No Rasch targets provided - set alignment metrics to 0
        l2_mse = 0.0
        l2_mae = 0.0
        corr_rasch = 0.0
        penalty_loss = 0.0
        violation_rate = 0.0
        mean_violation = 0.0
        max_violation = 0.0
    
    # Compute L_total combined metrics
    avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Loss ratios
    if avg_total_loss > 0:
        loss_ratio_l1 = l1_bce / avg_total_loss
        if lambda_penalty is not None and penalty_loss > 0:
            loss_ratio_penalty = (lambda_penalty * penalty_loss) / avg_total_loss
        else:
            loss_ratio_penalty = 0.0
    else:
        loss_ratio_l1 = 0.0
        loss_ratio_penalty = 0.0
    
    return {
        # L1 Performance
        'l1_bce': float(l1_bce),
        'auc': float(auc),
        'accuracy': float(accuracy),
        
        # L2 Alignment
        'l2_mse': float(l2_mse),
        'l2_mae': float(l2_mae),
        'corr_beta': float(corr_rasch),
        
        # L2_penalty Violations
        'penalty_loss': float(penalty_loss),
        'violation_rate': float(violation_rate),
        'mean_violation': float(mean_violation),
        'max_violation': float(max_violation),
        
        # L_total Combined
        'total_loss': float(avg_total_loss),
        'loss_ratio_l1': float(loss_ratio_l1),
        'loss_ratio_penalty': float(loss_ratio_penalty),
        
        # Meta
        'num_samples': int(len(all_preds))
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
    parser.add_argument('--lambda_penalty', type=float, required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--phase', type=int, default=2,
                       help='Phase for evaluation (default: 2 for full model with both heads)')
    
    # Rasch/IRT parameters
    parser.add_argument('--mastery_method', type=str, default='rasch',
                       help='Method for mastery estimation (rasch or bkt)')
    parser.add_argument('--diff_as_ones', action='store_true',
                       help='Use ones instead of difficulty estimates for Rasch')
    
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
    
    # Load Rasch targets
    print(f"\n📊 Loading Rasch/IRT targets...")
    from examples.train_ikt import load_rasch_targets
    
    # Determine rasch_path based on mastery_method
    if args.mastery_method == 'bkt':
        rasch_filename = 'bkt_targets.pkl'
    elif args.mastery_method == 'irt':
        rasch_filename = 'rasch_targets.pkl'
    elif args.mastery_method == 'bkt_mono':
        rasch_filename = 'bkt_mono_targets.pkl'
    elif args.mastery_method == 'irt_mono':
        rasch_filename = 'rasch_mono_targets.pkl'
    else:
        rasch_filename = 'rasch_targets.pkl'
    
    rasch_path = os.path.join(dataset_config['dpath'], rasch_filename)
    
    rasch_targets_data = load_rasch_targets(
        rasch_path=rasch_path,
        dataset_path=dataset_config['dpath'],
        num_c=num_c,
        mastery_method=args.mastery_method
    )
    print(f"✓ Rasch targets data loaded")
    
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
        lambda_penalty=args.lambda_penalty,
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
        train_results = evaluate_split(model, train_loader, device, 'train', 
                                      rasch_targets_data=rasch_targets_data, 
                                      lambda_penalty=args.lambda_penalty, 
                                      epsilon=args.epsilon)
        results['train'] = train_results
        print(f"Train - AUC: {train_results['auc']:.4f}, ACC: {train_results['accuracy']:.4f}, "
              f"L1: {train_results['l1_bce']:.4f}, L2_MSE: {train_results['l2_mse']:.4f}, "
              f"Violations: {train_results['violation_rate']*100:.1f}%")
    
    if valid_loader:
        print("\n📊 Evaluating on VALID split...")
        valid_results = evaluate_split(model, valid_loader, device, 'valid',
                                       rasch_targets_data=rasch_targets_data,
                                       lambda_penalty=args.lambda_penalty,
                                       epsilon=args.epsilon)
        results['valid'] = valid_results
        print(f"Valid - AUC: {valid_results['auc']:.4f}, ACC: {valid_results['accuracy']:.4f}, "
              f"L1: {valid_results['l1_bce']:.4f}, L2_MSE: {valid_results['l2_mse']:.4f}, "
              f"Violations: {valid_results['violation_rate']*100:.1f}%")
    
    if test_loader:
        print("\n🎯 Evaluating on TEST split...")
        test_results = evaluate_split(model, test_loader, device, 'test',
                                      rasch_targets_data=rasch_targets_data,
                                      lambda_penalty=args.lambda_penalty,
                                      epsilon=args.epsilon)
        results['test'] = test_results
        print(f"Test - AUC: {test_results['auc']:.4f}, ACC: {test_results['accuracy']:.4f}, "
              f"L1: {test_results['l1_bce']:.4f}, L2_MSE: {test_results['l2_mse']:.4f}, "
              f"Violations: {test_results['violation_rate']*100:.1f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Print comprehensive summary table
    print(f"\n{'Split':<10} {'AUC':<10} {'ACC':<10} {'L1_BCE':<10} {'L2_MSE':<10} {'Corr':<10} {'Viol%':<10}")
    print("-" * 70)
    for split_name in ['train', 'valid', 'test']:
        if split_name in results:
            r = results[split_name]
            print(f"{split_name.capitalize():<10} {r['auc']:<10.4f} {r['accuracy']:<10.4f} "
                  f"{r['l1_bce']:<10.4f} {r['l2_mse']:<10.4f} {r['corr_beta']:<10.3f} "
                  f"{r['violation_rate']*100:<10.1f}")
    
    # Save results to JSON
    results_json_path = os.path.join(args.run_dir, 'metrics_test.json')
    eval_results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.ckpt_name,
        'dataset': args.dataset,
        'fold': args.fold,
        'num_concepts': num_c,
        'num_parameters': num_params,
        'hyperparameters': {
            'lambda_penalty': args.lambda_penalty,
            'epsilon': args.epsilon,
            'phase': args.phase
        },
        'results': results
    }
    
    with open(results_json_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_json_path}")
    
    # Write metrics to metrics_test.csv
    if 'test' in results:
        test_csv_path = os.path.join(args.run_dir, 'metrics_test.csv')
        test_csv_exists = os.path.exists(test_csv_path)
        
        with open(test_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not test_csv_exists:
                # Write header
                writer.writerow([
                    'lambda_penalty', 'epsilon',
                    'test_auc', 'test_l1_auc', 'test_l2_mae', 'test_l2_mse', 'test_l2_penalty'
                ])
            
            # Write test results
            r = results['test']
            writer.writerow([
                args.lambda_penalty,
                args.epsilon,
                r['auc'],  # test_auc
                r['auc'],  # test_l1_auc (same as test_auc)
                r['l2_mae'],  # test_l2_mae
                r['l2_mse'],  # test_l2_mse
                r['penalty_loss']  # test_l2_penalty
            ])
        
        print(f"✓ Test metrics saved to: {test_csv_path}")
    
    print("\n" + "="*80)
    print("✅ Evaluation completed successfully")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
