#!/usr/bin/env python3
"""
Evaluation script for iKT3 model.

Usage:
    python examples/eval_ikt3.py --checkpoint saved_model/ikt3/best_model.pt --dataset assist2015
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import pickle

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.ikt3 import iKT3
from examples.experiment_utils import compute_auc_acc


def precompute_p_ref_from_rasch_targets(rasch_targets, dataset_uids, dataset_cseqs, num_skills):
    """
    Pre-compute p_ref for all sequences in dataset, aligned with dataset order.
    
    Args:
        rasch_targets: dict mapping student_id -> tensor [seq_len, num_skills]
        dataset_uids: tensor/list of UIDs for each sequence in dataset
        dataset_cseqs: tensor/list of concept sequences [num_sequences, seq_len]
        num_skills: total number of skills
    
    Returns:
        list of tensors: p_ref[i] is tensor [seq_len] for sequence i
    """
    if rasch_targets is None:
        return None
    
    p_ref_list = []
    num_sequences = len(dataset_uids)
    
    for idx in range(num_sequences):
        uid = int(dataset_uids[idx].item()) if torch.is_tensor(dataset_uids[idx]) else int(dataset_uids[idx])
        cseq = dataset_cseqs[idx]
        seq_len = len(cseq)
        
        # Initialize p_ref for this sequence
        p_ref_seq = torch.zeros(seq_len, dtype=torch.float32)
        
        if uid not in rasch_targets:
            # Student not in rasch_targets, use default 0.5
            p_ref_seq[:] = 0.5
        else:
            # Get rasch targets for this student [student_seq_len, num_skills]
            student_targets = rasch_targets[uid]
            student_seq_len = student_targets.shape[0]
            
            # Extract mastery probabilities for each timestep
            for t in range(seq_len):
                if t < student_seq_len:
                    skill_id = int(cseq[t].item()) if torch.is_tensor(cseq[t]) else int(cseq[t])
                    if skill_id >= 0 and skill_id < student_targets.shape[1]:
                        p_ref_seq[t] = student_targets[t, skill_id]
                    else:
                        p_ref_seq[t] = 0.5  # Default for invalid skill ID
                else:
                    p_ref_seq[t] = 0.5  # Beyond student sequence, use default
        
        p_ref_list.append(p_ref_seq)
    
    return p_ref_list


def extract_p_ref_from_rasch_targets(rasch_targets, batch, device):
    """
    Extract reference predictions (p_ref) from rasch_targets for current batch.
    
    Args:
        rasch_targets: dict mapping student_id -> tensor [seq_len, num_skills]
        batch: dict with 'uids', 'shft_cseqs'
        device: torch device
        
    Returns:
        p_ref: tensor [B, L] with reference mastery probabilities
    """
    if rasch_targets is None:
        return None
        
    uids = batch['uids']
    questions = batch['shft_cseqs']  # Skills being answered
    batch_size = questions.size(0)
    seq_len = questions.size(1)
    
    # Build p_ref tensor
    p_ref = torch.zeros(batch_size, seq_len, device=device)
    
    for i, uid in enumerate(uids):
        if uid not in rasch_targets:
            # Student not in rasch_targets - should not happen in train/valid
            # but might happen in test if test students weren't in IRT calibration
            p_ref[i, :] = 0.5
            continue
            
        student_targets = rasch_targets[uid]
        
        for t in range(seq_len):
            if t < student_targets.size(0):
                skill_id = questions[i, t].item()
                if 0 <= skill_id < student_targets.size(1):
                    # Extract IRT probability for the skill being practiced at timestep t
                    p_ref[i, t] = student_targets[t, skill_id]
                else:
                    # Invalid skill ID - shouldn't happen with proper data
                    p_ref[i, t] = 0.5
            else:
                # Beyond student's original sequence length in rasch_targets
                p_ref[i, t] = 0.5
    
    return p_ref


def evaluate(model, data_loader, device, p_ref_list=None):
    """
    Evaluate model on test set.
    
    Args:
        model: iKT3 model
        data_loader: DataLoader for evaluation
        device: torch device
        p_ref_list: Pre-computed list of p_ref tensors aligned with dataset order
    
    Returns:
        dict with evaluation metrics
    """
    model.eval()
    
    all_predictions_per = []  # p_correct predictions
    all_predictions_ali = []  # m_pred predictions
    all_targets = []  # true responses for auc_per
    all_p_ref = []  # reference IRT predictions for auc_ali
    all_theta = []
    all_beta = []
    all_mastery = []  # m_pred for mastery stats
    
    total_loss_per = 0.0
    total_loss_ali = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            q = batch['cseqs'].long().to(device)
            r = batch['rseqs'].long().to(device)
            qry = batch['shft_cseqs'].long().to(device)
            targets = batch['shft_rseqs'].float().to(device)
            mask = batch['masks'].to(device)
            
            # Extract p_ref from pre-computed list
            p_ref = None
            if p_ref_list is not None:
                batch_size = targets.shape[0]
                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + batch_size
                
                # Stack p_ref tensors for this batch
                batch_p_ref = []
                for idx in range(start_idx, min(end_idx, len(p_ref_list))):
                    # Slice [1:] to match shifted predictions (skip first timestep)
                    batch_p_ref.append(p_ref_list[idx][1:])
                
                if batch_p_ref:
                    # Pad sequences to max length in batch if needed
                    max_len = max(len(p) for p in batch_p_ref)
                    padded_p_ref = []
                    for p in batch_p_ref:
                        if len(p) < max_len:
                            padded = torch.cat([p, torch.full((max_len - len(p),), 0.5)])
                            padded_p_ref.append(padded)
                        else:
                            padded_p_ref.append(p)
                    p_ref = torch.stack(padded_p_ref).to(device)
            
            # Forward pass (use shifted sequences for autoregressive prediction)
            outputs = model(q, r, qry)
            
            # Compute losses
            p_correct = outputs['p_correct']
            m_pred = outputs['m_pred']
            
            # L_per: BCE(p_correct, targets) - compare against true responses
            loss_per = torch.nn.functional.binary_cross_entropy(p_correct, targets, reduction='none')
            loss_per = (loss_per * mask).sum() / mask.sum()
            
            # L_ali: BCE(m_pred, p_ref) - compare against reference IRT predictions
            if p_ref is not None:
                loss_ali = torch.nn.functional.binary_cross_entropy(m_pred, p_ref, reduction='none')
                loss_ali = (loss_ali * mask).sum() / mask.sum()
            else:
                # Fallback: if no p_ref, use true responses (same as L_per)
                loss_ali = torch.nn.functional.binary_cross_entropy(m_pred, targets, reduction='none')
                loss_ali = (loss_ali * mask).sum() / mask.sum()
            
            batch_size = q.size(0)
            total_loss_per += loss_per.item() * batch_size
            total_loss_ali += loss_ali.item() * batch_size
            total_samples += batch_size
            
            # Extract outputs for metrics
            p_correct_np = p_correct.cpu().numpy()
            m_pred_np = m_pred.cpu().numpy()
            targets_np = targets.cpu().numpy()
            theta_t = outputs['theta_t'].cpu().numpy()
            beta_k = outputs['beta_k'].cpu().numpy()
            
            # Extract p_ref for auc_ali computation
            if p_ref is not None:
                p_ref_np = p_ref.cpu().numpy()
            else:
                # Fallback: use targets if no p_ref available
                p_ref_np = targets_np
            
            mask_np = mask.cpu().numpy()
            
            # Only keep valid positions
            for i in range(batch_size):
                valid_mask = mask_np[i] == 1
                all_predictions_per.extend(p_correct_np[i][valid_mask])
                all_predictions_ali.extend(m_pred_np[i][valid_mask])
                all_targets.extend(targets_np[i][valid_mask])
                all_p_ref.extend(p_ref_np[i][valid_mask])
                all_theta.extend(theta_t[i][valid_mask])
                all_beta.extend(beta_k[i][valid_mask])
                all_mastery.extend(m_pred_np[i][valid_mask])
    
    # Compute AUC and accuracy for L_per (p_correct vs true responses)
    metrics_per = compute_auc_acc(all_targets, all_predictions_per)
    auc_per = metrics_per['auc']
    acc_per = metrics_per['acc']
    
    # Compute alignment metrics for L_ali (m_pred vs reference IRT predictions)
    # Use correlation to measure continuous alignment (not binary AUC)
    import numpy as np
    all_p_ref_np = np.array(all_p_ref)
    all_predictions_ali_np = np.array(all_predictions_ali)
    
    # Pearson correlation between model predictions and IRT reference
    if len(all_p_ref) > 1 and np.std(all_p_ref_np) > 0 and np.std(all_predictions_ali_np) > 0:
        corr_ali = np.corrcoef(all_p_ref_np, all_predictions_ali_np)[0, 1]
    else:
        corr_ali = 0.0
    
    # MSE between predictions and reference (lower is better)
    mse_ali = np.mean((all_predictions_ali_np - all_p_ref_np) ** 2)
    
    # For consistency with existing code, keep auc_ali and acc_ali names
    # but store correlation and MSE instead
    auc_ali = float(corr_ali)  # Correlation (higher is better alignment)
    acc_ali = float(1.0 - mse_ali)  # 1-MSE (higher is better, like accuracy)
    
    # Compute interpretability metrics
    all_theta = np.array(all_theta)
    all_beta = np.array(all_beta)
    all_mastery = np.array(all_mastery)
    
    theta_mean = all_theta.mean()
    theta_std = all_theta.std()
    beta_mean = all_beta.mean()
    beta_std = all_beta.std()
    ratio = theta_std / beta_std if beta_std > 0 else 0.0
    
    return {
        'loss_per': total_loss_per / total_samples,
        'loss_ali': total_loss_ali / total_samples,
        'auc_per': auc_per,
        'acc_per': acc_per,
        'auc_ali': auc_ali,
        'acc_ali': acc_ali,
        'auc': auc_per,  # Keep for backward compatibility
        'acc': acc_per,  # Keep for backward compatibility
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'theta_beta_ratio': ratio,
        'mastery_mean': all_mastery.mean(),
        'mastery_std': all_mastery.std(),
        'num_samples': len(all_predictions_per)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate iKT3 model')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length')
    parser.add_argument('--device', type=str, required=True, help='Device')
    parser.add_argument('--split', type=str, required=True, choices=['valid', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load model checkpoint
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Training AUC: {checkpoint.get('best_auc', 'N/A')}")
    
    # Initialize dataset
    print(f"\n{'='*60}")
    print("Initializing dataset...")
    print(f"{'='*60}")
    
    # Load data config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'data_config.json')
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    # Convert relative paths to absolute
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name] and data_config[dataset_name]['dpath'].startswith('../'):
            data_config[dataset_name]['dpath'] = os.path.join(project_root, data_config[dataset_name]['dpath'][3:])
    
    model_name = "ikt3"
    train_loader, valid_loader = init_dataset4train(
        args.dataset,
        model_name,
        data_config,
        args.fold,
        args.batch_size
    )
    
    num_c = data_config[args.dataset]['num_c']
    
    # Use appropriate split
    if args.split == 'valid':
        data_loader = valid_loader
    else:
        # For test split, would need to load test data separately
        # For now, use valid_loader
        data_loader = valid_loader
        print("⚠️  Using validation set (test set loading not implemented)")
    
    print(f"✓ Dataset initialized: {num_c} skills")
    
    # Load IRT difficulties and rasch_targets
    rasch_path = config['rasch_path']
    beta_irt = None
    rasch_targets = None
    if rasch_path and os.path.exists(rasch_path):
        with open(rasch_path, 'rb') as f:
            data = pickle.load(f)
        if 'skill_difficulties' in data:
            skill_diff_dict = data['skill_difficulties']
            skill_difficulties = [skill_diff_dict.get(k, 0.0) for k in range(num_c)]
            beta_irt = torch.tensor(skill_difficulties, dtype=torch.float32)
            print(f"✓ Loaded IRT difficulties from {rasch_path}")
        if 'rasch_targets' in data:
            rasch_targets = data['rasch_targets']
            print(f"✓ Loaded rasch_targets for L_ali: {len(rasch_targets)} students")
    
    # Pre-compute p_ref for test/valid set
    p_ref_list = None
    if rasch_targets is not None:
        print("Pre-computing p_ref for evaluation set...")
        eval_dataset = data_loader.dataset
        if hasattr(eval_dataset, 'dori') and 'uids' in eval_dataset.dori:
            eval_uids = eval_dataset.dori['uids']
            eval_cseqs = eval_dataset.dori['cseqs']
            p_ref_list = precompute_p_ref_from_rasch_targets(
                rasch_targets, eval_uids, eval_cseqs, num_c
            )
            print(f"✓ Pre-computed p_ref for {len(p_ref_list)} evaluation sequences")
        else:
            print("⚠️  Cannot access dataset UIDs for p_ref pre-computation")
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing model...")
    print(f"{'='*60}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = iKT3(
        num_c=num_c,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_blocks=config['num_encoder_blocks'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        seq_len=config['seq_len'],
        beta_irt=beta_irt,
        target_ratio=config['target_ratio']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load IRT difficulties if available
    if beta_irt is not None:
        model.load_irt_difficulties(beta_irt.to(device))
    
    print(f"✓ Model initialized")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    metrics = evaluate(model, data_loader, device, p_ref_list=p_ref_list)
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"\nInterpretability Metrics:")
    print(f"  θ (ability): mean={metrics['theta_mean']:.3f}, std={metrics['theta_std']:.3f}")
    print(f"  β (difficulty): mean={metrics['beta_mean']:.3f}, std={metrics['beta_std']:.3f}")
    print(f"  θ/β ratio: {metrics['theta_beta_ratio']:.3f} (target: 0.3-0.5)")
    print(f"  Mastery: mean={metrics['mastery_mean']:.3f}, std={metrics['mastery_std']:.3f}")
    
    # Check scale health
    if 0.3 <= metrics['theta_beta_ratio'] <= 0.5:
        print(f"  ✅ Scale ratio healthy")
    else:
        print(f"  ⚠️  Scale ratio outside target range")
    
    # Save results
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in metrics.items()}
    
    # Save JSON
    results_path = os.path.join(output_dir, f'eval_{args.split}_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Save CSV for reproducibility standard
    csv_path = os.path.join(output_dir, 'metrics_eval.csv')
    try:
        import csv as csv_module
        with open(csv_path, 'w', newline='') as cf:
            fieldnames = ['split', 'loss_per', 'loss_ali', 'auc_per', 'acc_per', 'auc_ali', 'acc_ali', 
                         'theta_mean', 'theta_std', 'beta_mean', 'beta_std', 'theta_beta_ratio']
            writer = csv_module.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'split': args.split,
                'loss_per': f"{metrics['loss_per']:.6f}",
                'loss_ali': f"{metrics['loss_ali']:.6f}",
                'auc_per': f"{metrics['auc_per']:.6f}",
                'acc_per': f"{metrics['acc_per']:.6f}",
                'auc_ali': f"{metrics['auc_ali']:.6f}",
                'acc_ali': f"{metrics['acc_ali']:.6f}",
                'theta_mean': f"{metrics['theta_mean']:.6f}",
                'theta_std': f"{metrics['theta_std']:.6f}",
                'beta_mean': f"{metrics['beta_mean']:.6f}",
                'beta_std': f"{metrics['beta_std']:.6f}",
                'theta_beta_ratio': f"{metrics['theta_beta_ratio']:.6f}"
            })
        print(f"✓ Saved CSV metrics to {csv_path}")
    except Exception as e:
        print(f"⚠️  Could not write metrics CSV ({e})")


if __name__ == '__main__':
    main()
