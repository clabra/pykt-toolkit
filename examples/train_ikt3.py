#!/usr/bin/env python3
"""
Training script for iKT3 model (Interpretable Knowledge Tracing v3) using PyKT framework patterns.

                         ⚠️  REPRODUCIBILITY WARNING ⚠️

  DO NOT CALL THIS SCRIPT DIRECTLY FOR REPRODUCIBLE EXPERIMENTS!

  Use the experiment launcher:
      python examples/run_repro_experiment.py --short_title "name"

  The launcher will:
    ✓ Load defaults from configs/parameter_default.json
    ✓ Apply your CLI overrides
    ✓ Generate explicit command with ALL parameters
    ✓ Create experiment folder with full audit trail
    ✓ Save config.json for perfect reproducibility

iKT3 Architecture:
- Single encoder with self-attention
- Single output head: Ability encoder θ_i(t) = MLP(h)
- Fixed β_IRT difficulties (no learnable parameters)
- IRT formula: m_pred = σ(θ - β)

Two-Phase Training:
- Phase 1: Alignment learning (L_total = L_ali)
- Phase 2: Performance + Alignment (L_total = L_per + λ_int×L_ali)

Key Simplifications from iKT2:
- No learnable skill difficulties
- No regularization loss (β fixed)
- Single head architecture
- Cleaner implementation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
import csv
import subprocess
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.ikt3 import iKT3
from examples.experiment_utils import compute_auc_acc


def get_model_attr(model, attr_name):
    """Safely get model attribute whether wrapped in DataParallel or not."""
    if isinstance(model, torch.nn.DataParallel):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)


def load_skill_difficulties_from_irt(rasch_path, num_c):
    """
    Load IRT-calibrated skill difficulties.
    
    Args:
        rasch_path: Path to rasch_targets.pkl file
        num_c: Number of skills
    
    Returns:
        torch.Tensor of shape [num_c] with skill difficulties (beta values),
        or None if file doesn't exist
    """
    if not os.path.exists(rasch_path):
        print(f"⚠️  No IRT file found at {rasch_path}")
        print(f"⚠️  Model will use zero-initialized difficulties")
        return None
    
    try:
        with open(rasch_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'skill_difficulties' in data:
            # Extract skill difficulties as a list ordered by skill index
            skill_diff_dict = data['skill_difficulties']
            skill_difficulties = [skill_diff_dict.get(k, 0.0) for k in range(num_c)]
            beta_irt = torch.tensor(skill_difficulties, dtype=torch.float32)
            print(f"✓ Loaded IRT skill difficulties for {num_c} skills")
            print(f"  Difficulty range: [{beta_irt.min():.3f}, {beta_irt.max():.3f}]")
            print(f"  Difficulty mean: {beta_irt.mean():.3f}, std: {beta_irt.std():.3f}")
            return beta_irt
        else:
            print(f"⚠️  No 'skill_difficulties' key in {rasch_path}")
            return None
    except Exception as e:
        print(f"⚠️  Error loading IRT difficulties: {e}")
        return None


def load_rasch_targets_as_reference(rasch_path):
    """
    Load rasch_targets from pickle file for use as reference predictions.
    
    Args:
        rasch_path: Path to rasch_targets.pkl file
    
    Returns:
        dict mapping student_id to tensor [seq_len, num_skills] with mastery probabilities,
        or None if file doesn't exist
    """
    if not os.path.exists(rasch_path):
        print(f"⚠️  No rasch_targets found at {rasch_path}")
        return None
    
    try:
        with open(rasch_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'rasch_targets' in data:
            rasch_targets = data['rasch_targets']
            print(f"✓ Loaded rasch_targets for alignment: {len(rasch_targets)} students")
            
            # Print sample statistics
            sample_uid = list(rasch_targets.keys())[0]
            sample_tensor = rasch_targets[sample_uid]
            print(f"  Reference predictions shape: {sample_tensor.shape}")
            print(f"  Value range: [{sample_tensor.min():.3f}, {sample_tensor.max():.3f}]")
            
            return rasch_targets
        else:
            print(f"⚠️  No 'rasch_targets' key in {rasch_path}")
            return None
    except Exception as e:
        print(f"⚠️  Error loading rasch_targets: {e}")
        return None


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


def train_epoch(model, train_loader, optimizer, device, phase=1, lambda_int=0.0, 
                lambda_scale=0.0, p_ref_list=None, use_amp=False, gradient_clip=None):
    """
    Train for one epoch.
    
    Args:
        model: iKT3 model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: torch device
        phase: Training phase (1 or 2)
        lambda_int: Alignment weight for Phase 2
        p_ref_list: Pre-computed list of p_ref tensors aligned with dataset order
        use_amp: Whether to use automatic mixed precision
        gradient_clip: Gradient clipping value (None to disable)
    
    Phase 1: L_total = L_ali (align m_pred with IRT reference)
    Phase 2: L_total = L_per + λ_int×L_ali (optimize performance + maintain alignment)
    
    Returns:
        dict with epoch metrics
    """
    model.train()
    total_loss = 0.0
    total_per_loss = 0.0
    total_ali_loss = 0.0
    total_samples = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device and cast to appropriate types
        q = batch['cseqs'].long().to(device)  # [B, L] - current concept/question sequences
        r = batch['rseqs'].long().to(device)  # [B, L] - current response sequences
        qry = batch['shft_cseqs'].long().to(device)  # [B, L] - NEXT questions (shifted)
        targets = batch['shft_rseqs'].float().to(device)  # [B, L] - NEXT responses (shifted targets)
        mask = batch.get('masks', None)  # Note: masks are already shifted (199 timesteps)
        if mask is not None:
            mask = mask.to(device)
        
        # Get reference predictions from pre-computed list (required for both phases)
        p_ref = None
        if p_ref_list is not None:
            # Extract p_ref from pre-computed list using batch indices
            batch_size = targets.shape[0]
            start_idx = batch_idx * train_loader.batch_size
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
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(q, r, qry)
                
                # p_ref must be provided for alignment training
                if p_ref is None:
                    raise ValueError("p_ref is None - alignment training requires pre-computed IRT reference predictions")
                
                loss_dict = model.compute_loss(
                    outputs, targets, p_ref=p_ref, 
                    phase=phase, lambda_int=lambda_int, lambda_scale=lambda_scale, mask=mask
                )
            
            # Backward pass with gradient scaling
            scaler.scale(loss_dict['total_loss']).backward()
            
            # Gradient clipping
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal precision training
            outputs = model(q, r)
            
            # p_ref must be provided for alignment training
            if p_ref is None:
                raise ValueError("p_ref is None - alignment training requires pre-computed IRT reference predictions")
            
            loss_dict = model.compute_loss(
                outputs, targets, p_ref=p_ref,
                phase=phase, lambda_int=lambda_int, lambda_scale=lambda_scale, mask=mask
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        # Accumulate metrics
        batch_size = q.size(0)
        total_loss += loss_dict['total_loss'].item() * batch_size
        total_per_loss += loss_dict['per_loss'].item() * batch_size
        total_ali_loss += loss_dict['alignment_loss'].item() * batch_size
        total_samples += batch_size
    
    return {
        'loss': total_loss / total_samples,
        'per_loss': total_per_loss / total_samples,
        'ali_loss': total_ali_loss / total_samples,
    }


def evaluate(model, data_loader, device, phase=1, lambda_int=0.0, lambda_scale=0.0, p_ref_list=None):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: iKT3 model
        data_loader: DataLoader for evaluation
        device: torch device
        phase: Training phase (1 or 2)
        lambda_int: Alignment weight
        p_ref_list: Pre-computed list of p_ref tensors aligned with dataset order
    
    Returns:
        dict with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_per_loss = 0.0
    total_ali_loss = 0.0
    total_samples = 0
    
    all_predictions_per = []  # p_correct predictions
    all_predictions_ali = []  # m_pred predictions
    all_targets = []  # true responses for auc_per
    all_p_ref = []  # reference IRT predictions for auc_ali
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device and cast to appropriate types
            q = batch['cseqs'].long().to(device)  # Current questions
            r = batch['rseqs'].long().to(device)  # Current responses
            qry = batch['shft_cseqs'].long().to(device)  # NEXT questions (shifted)
            targets = batch['shft_rseqs'].float().to(device)  # NEXT responses (shifted targets)
            mask = batch.get('masks', None)  # Already shifted (199 timesteps)
            if mask is not None:
                mask = mask.to(device)
            
            # Get reference predictions from pre-computed list
            p_ref = None
            if p_ref_list is not None:
                # Extract p_ref from pre-computed list using batch indices
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
            
            # Forward pass (with shifted questions for proper autoregressive prediction)
            outputs = model(q, r, qry)
            
            # p_ref must be provided for alignment evaluation
            if p_ref is None:
                raise ValueError("p_ref is None - alignment evaluation requires pre-computed IRT reference predictions")
            
            loss_dict = model.compute_loss(
                outputs, targets, p_ref=p_ref,
                phase=phase, lambda_int=lambda_int, lambda_scale=lambda_scale, mask=mask
            )
            
            # Accumulate metrics
            batch_size = q.size(0)
            total_loss += loss_dict['total_loss'].item() * batch_size
            total_per_loss += loss_dict['per_loss'].item() * batch_size
            total_ali_loss += loss_dict['alignment_loss'].item() * batch_size
            total_samples += batch_size
            
            # Store predictions for AUC calculation
            p_correct = outputs['p_correct'].cpu().numpy()
            m_pred = outputs['m_pred'].cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Extract p_ref for auc_ali computation
            if p_ref is not None:
                p_ref_np = p_ref.cpu().numpy()
            else:
                # Fallback: use targets if no p_ref available
                p_ref_np = targets_np
            
            if mask is not None:
                mask_np = mask.cpu().numpy()
                # Only keep valid positions
                for i in range(batch_size):
                    valid_mask = mask_np[i] == 1
                    all_predictions_per.extend(p_correct[i][valid_mask])
                    all_predictions_ali.extend(m_pred[i][valid_mask])
                    all_targets.extend(targets_np[i][valid_mask])
                    all_p_ref.extend(p_ref_np[i][valid_mask])
            else:
                all_predictions_per.extend(p_correct.flatten())
                all_predictions_ali.extend(m_pred.flatten())
                all_targets.extend(targets_np.flatten())
                all_p_ref.extend(p_ref_np.flatten())
    
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
        if np.isnan(corr_ali):
            corr_ali = 0.0
    else:
        corr_ali = 0.0
    
    # MSE between predictions and reference (lower is better)
    mse_ali = np.mean((all_predictions_ali_np - all_p_ref_np) ** 2)
    
    # For consistency with existing code, keep auc_ali and acc_ali names
    # but store correlation and MSE-based metric instead
    auc_ali = float(corr_ali)  # Correlation (higher is better alignment)
    acc_ali = float(1.0 - mse_ali)  # 1-MSE (higher is better, like accuracy)
    
    return {
        'loss': total_loss / total_samples,
        'per_loss': total_per_loss / total_samples,
        'ali_loss': total_ali_loss / total_samples,
        'auc': auc_per,  # Primary AUC (performance)
        'acc': acc_per,  # Primary accuracy (performance)
        'auc_per': auc_per,
        'acc_per': acc_per,
        'auc_ali': auc_ali,
        'acc_ali': acc_ali,
    }


def main():
    parser = argparse.ArgumentParser(description='Train iKT3 model')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--rasch_path', type=str, required=True, help='Path to rasch_targets.pkl')
    parser.add_argument('--reference_path', type=str, required=True, help='Path to reference predictions (can be "null" or "None")')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, required=True, help='Model dimension')
    parser.add_argument('--n_heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--num_encoder_blocks', type=int, required=True, help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, required=True, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, required=True, help='Max sequence length')
    parser.add_argument('--emb_type', type=str, required=True, help='Embedding type')
    parser.add_argument('--target_ratio', type=float, required=True, help='Target θ/β scale ratio')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer')
    parser.add_argument('--gradient_clip', type=float, required=True, help='Gradient clipping')
    parser.add_argument('--patience', type=int, required=True, help='Early stopping patience')
    parser.add_argument('--phase1_epochs', type=int, required=True, help='Epochs for Phase 1')
    parser.add_argument('--lambda_int', type=float, required=True, help='Alignment weight for Phase 2')
    parser.add_argument('--lambda_scale', type=float, required=True, help='Scale regularization weight for Phase 2')
    
    # Runtime parameters
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--auto_shifted_eval', action='store_true', help='Auto-shifted evaluation')
    parser.add_argument('--device', type=str, required=True, help='Device')
    parser.add_argument('--num_workers', type=int, required=True, help='DataLoader workers')
    
    # Logging parameters
    parser.add_argument('--save_dir', type=str, help='Directory to save model (auto from EXPERIMENT_DIR if not set)')
    parser.add_argument('--monitor_freq', type=int, required=True, help='Scale monitoring frequency')
    
    args = parser.parse_args()
    
    # Handle save_dir: use EXPERIMENT_DIR if save_dir not provided
    if args.save_dir is None:
        experiment_dir = os.environ.get('EXPERIMENT_DIR')
        if experiment_dir is None:
            raise ValueError("Either --save_dir must be provided or EXPERIMENT_DIR must be set")
        args.save_dir = experiment_dir
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize dataset
    print(f"\n{'='*60}")
    print("Initializing dataset...")
    print(f"{'='*60}")
    
    # Load data config from configs/data_config.json
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'data_config.json')
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    # Verify dataset exists in config
    if args.dataset not in data_config:
        raise ValueError(f"Dataset '{args.dataset}' not found in data_config.json. Available: {list(data_config.keys())}")
    
    # Convert relative paths to absolute paths (relative to project root)
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
    
    # Get num_c from data config
    num_c = data_config[args.dataset]['num_c']
    
    print(f"✓ Dataset initialized: {num_c} skills")
    
    # Load IRT difficulties
    beta_irt = load_skill_difficulties_from_irt(args.rasch_path, num_c)
    
    # Load rasch targets and pre-compute p_ref lists aligned with dataset order
    p_ref_train_list = None
    p_ref_valid_list = None
    if args.rasch_path and os.path.exists(args.rasch_path):
        rasch_targets_all = load_rasch_targets_as_reference(args.rasch_path)
        if rasch_targets_all:
            print("Pre-computing p_ref for training set...")
            # Access dataset UIDs and sequences
            train_dataset = train_loader.dataset
            if hasattr(train_dataset, 'dori') and 'uids' in train_dataset.dori:
                train_uids = train_dataset.dori['uids']
                train_cseqs = train_dataset.dori['cseqs']
                p_ref_train_list = precompute_p_ref_from_rasch_targets(
                    rasch_targets_all, train_uids, train_cseqs, num_c
                )
                print(f"✓ Pre-computed p_ref for {len(p_ref_train_list)} training sequences")
            
            print("Pre-computing p_ref for validation set...")
            valid_dataset = valid_loader.dataset
            if hasattr(valid_dataset, 'dori') and 'uids' in valid_dataset.dori:
                valid_uids = valid_dataset.dori['uids']
                valid_cseqs = valid_dataset.dori['cseqs']
                p_ref_valid_list = precompute_p_ref_from_rasch_targets(
                    rasch_targets_all, valid_uids, valid_cseqs, num_c
                )
                print(f"✓ Pre-computed p_ref for {len(p_ref_valid_list)} validation sequences")
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing model...")
    print(f"{'='*60}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = iKT3(
        num_c=num_c,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_blocks=args.num_encoder_blocks,
        d_ff=args.d_ff,
        dropout=args.dropout,
        seq_len=args.seq_len,
        beta_irt=beta_irt,
        target_ratio=args.target_ratio
    ).to(device)
    
    # Load IRT difficulties into model if provided
    if beta_irt is not None:
        model.load_irt_difficulties(beta_irt.to(device))
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Initialize metrics CSV file for epoch-by-epoch tracking
    metrics_csv_path = os.path.join(args.save_dir, 'metrics_valid.csv')
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'lambda_int',
                'train_loss', 'train_per_loss', 'train_ali_loss', 
                'train_auc_per', 'train_acc_per', 'train_corr_ali', 'train_mse_ali',
                'valid_loss', 'valid_per_loss', 'valid_ali_loss', 
                'valid_auc_per', 'valid_acc_per', 'valid_corr_ali', 'valid_mse_ali'
            ])
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    best_auc = 0.0
    patience_counter = 0
    phase1_best_auc = 0.0
    phase1_completed = False
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Determine phase
        if epoch <= args.phase1_epochs:
            phase = 1
            lambda_int = 0.0
            phase_name = "Phase 1 (Alignment)"
        else:
            phase = 2
            lambda_int = args.lambda_int
            phase_name = f"Phase 2 (Performance + Alignment λ={lambda_int:.2f})"
            
            # Reset patience counter when transitioning to Phase 2
            if not phase1_completed:
                phase1_completed = True
                phase1_best_auc = best_auc
                patience_counter = 0
                print(f"\n{'='*60}")
                print(f"Phase 1 complete - Best AUC: {phase1_best_auc:.4f}")
                print(f"Starting Phase 2 with reset patience counter")
                print(f"{'='*60}\n")
        
        print(f"\n--- Epoch {epoch}/{args.epochs} [{phase_name}] ---")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            phase=phase, lambda_int=lambda_int, lambda_scale=args.lambda_scale,
            p_ref_list=p_ref_train_list,
            use_amp=args.use_amp, gradient_clip=args.gradient_clip
        )
        
        # Evaluate
        valid_metrics = evaluate(
            model, valid_loader, device,
            phase=phase, lambda_int=lambda_int, lambda_scale=args.lambda_scale,
            p_ref_list=p_ref_valid_list
        )
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"L_per: {train_metrics['per_loss']:.4f}, "
              f"L_ali: {train_metrics['ali_loss']:.4f}")
        print(f"Valid - Loss: {valid_metrics['loss']:.4f}, "
              f"L_per: {valid_metrics['per_loss']:.4f}, "
              f"L_ali: {valid_metrics['ali_loss']:.4f}, "
              f"AUC_per: {valid_metrics['auc_per']:.4f}, "
              f"Corr_ali: {valid_metrics['auc_ali']:.4f}")
        
        # Write metrics to CSV
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                phase,
                lambda_int,
                train_metrics['loss'],
                train_metrics['per_loss'],
                train_metrics['ali_loss'],
                train_metrics.get('auc_per', 0.0),
                train_metrics.get('acc_per', 0.0),
                train_metrics.get('auc_ali', 0.0),
                train_metrics.get('acc_ali', 0.0),
                valid_metrics['loss'],
                valid_metrics['per_loss'],
                valid_metrics['ali_loss'],
                valid_metrics['auc_per'],
                valid_metrics['acc_per'],
                valid_metrics['auc_ali'],
                valid_metrics['acc_ali']
            ])
        
        # Append to training history
        epoch_results = {
            'epoch': epoch,
            'phase': phase,
            'lambda_int': lambda_int,
            'training': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in train_metrics.items()},
            'validation': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in valid_metrics.items()}
        }
        history.append(epoch_results)
        
        # Monitor scale health
        if epoch % args.monitor_freq == 0:
            with torch.no_grad():
                # Get a batch for monitoring
                batch = next(iter(valid_loader))
                q = batch['cseqs'].long().to(device)
                r = batch['rseqs'].long().to(device)
                qry = batch['shft_cseqs'].long().to(device)
                outputs = model(q, r, qry)
                model.monitor_scale_health(outputs['theta_t'], outputs['beta_k'], epoch)
        
        # Save best model
        if valid_metrics['auc_per'] > best_auc:
            best_auc = valid_metrics['auc_per']
            patience_counter = 0
            
            # Save model
            model_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': vars(args)
            }, model_path)
            print(f"✓ Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            # Only apply early stopping in Phase 1
            # In Phase 2, continue until max epochs to ensure alignment training completes
            if phase == 1 and patience_counter >= args.patience:
                print(f"\nPhase 1 early stopping at epoch {epoch}")
                print(f"Continuing to Phase 2 for alignment training...")
                # Force transition to Phase 2 by adjusting loop
                continue
            elif phase == 2 and patience_counter >= args.patience:
                print(f"\nPhase 2 early stopping triggered after {epoch} epochs")
                break
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best validation AUC: {best_auc:.4f}")
    print(f"{'='*60}")
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")
    
    # Run auto shifted eval if requested
    if args.auto_shifted_eval:
        print("\n" + "="*80)
        print("Running auto shifted evaluation...")
        print("="*80 + "\n")
        
        eval_script = os.path.join('/workspaces/pykt-toolkit/examples', 'eval_ikt3.py')
        checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
        
        eval_cmd = [
            'python', eval_script,
            '--dataset', args.dataset,
            '--fold', str(args.fold),
            '--checkpoint', checkpoint_path,
            '--batch_size', str(args.batch_size),
            '--seq_len', str(args.seq_len),
            '--output_dir', args.save_dir
        ]
        
        try:
            result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            print("✓ Auto shifted evaluation completed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Auto shifted evaluation failed: {e}")
            print(e.stderr)
    
    print("\n✅ Training script completed successfully!")


if __name__ == '__main__':
    main()
