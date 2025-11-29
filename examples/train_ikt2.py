#!/usr/bin/env python3
"""
Training script for iKT2 model (Interpretable Knowledge Tracing v2) using PyKT framework patterns.


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



iKT2 Architecture:
- Single encoder with dual-stream processing
- Head 1: Performance prediction (BCE loss)
- Head 2: IRT-based mastery inference M_IRT = σ(θ_i(t) - β_k)
- Ability encoder: 2-layer MLP extracting θ_i(t) from hidden states

Two-Phase Training:
- Phase 1: Warmup - Performance learning + difficulty regularization (L_total = L_BCE + λ_reg × L_reg)
- Phase 2: IRT Alignment - Add alignment constraint (L_total = L_BCE + λ_align × L_align + λ_reg × L_reg)

Note: Phase 1 establishes good representations before enforcing IRT consistency in Phase 2.
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
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.ikt2 import iKT2
from examples.experiment_utils import compute_auc_acc


def get_model_attr(model, attr_name):
    """Safely get model attribute whether wrapped in DataParallel or not."""
    if isinstance(model, torch.nn.DataParallel):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)


def load_skill_difficulties_from_irt(rasch_path, num_c):
    """
    Load IRT-calibrated skill difficulties for regularization.
    
    Args:
        rasch_path: Path to rasch_targets.pkl file (explicit, no fallback)
        num_c: Number of skills
    
    Returns:
        torch.Tensor of shape [num_c] with skill difficulties (beta values),
        or None if file doesn't exist
    
    Note:
        Path must be explicitly provided via --rasch_path CLI argument.
        No hardcoded defaults allowed (Explicit Parameters, Zero Defaults).
    """
    import pickle
    
    if not os.path.exists(rasch_path):
        print(f"⚠️  No IRT file found at {rasch_path}, skipping skill regularization")
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
            return beta_irt
        else:
            print(f"⚠️  No 'skill_difficulties' key in {rasch_path}, skipping regularization")
            return None
    except Exception as e:
        print(f"⚠️  Failed to load skill difficulties: {e}")
        return None


def train_epoch(model, train_loader, optimizer, device, gradient_clip, beta_irt, lambda_reg):
    """
    Train for one epoch with comprehensive metrics tracking.
    
    Args:
        model: iKT2 model instance
        train_loader: training data loader
        optimizer: optimizer instance
        device: torch device
        gradient_clip: gradient clipping threshold (0 to disable)
        beta_irt: IRT skill difficulties for regularization (required, can be None)
        lambda_reg: regularization coefficient (required, explicit)
    
    Note:
        All parameters REQUIRED. No defaults per reproducibility guidelines.
        Code will fail if parameters not explicitly provided.
    """
    model.train()
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_align_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    all_mastery_irt = []
    all_p_correct = []
    
    for batch in train_loader:
        questions = batch['cseqs'].to(device)
        responses = batch['rseqs'].to(device)
        questions_shifted = batch['shft_cseqs'].to(device)
        mask = batch['masks'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(q=questions, r=responses, qry=questions_shifted)
        
        # Get targets
        targets = batch['shft_rseqs'].to(device)
        
        # Compute loss with skill regularization
        compute_loss_fn = get_model_attr(model, 'compute_loss')
        loss_dict = compute_loss_fn(outputs, targets, beta_irt=beta_irt, lambda_reg=lambda_reg)
        loss = loss_dict['total_loss']
        bce_loss = loss_dict['bce_loss']
        align_loss = loss_dict['align_loss']
        reg_loss = loss_dict['reg_loss']
        
        # Backward pass
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_align_loss += align_loss.item()
        total_reg_loss += reg_loss.item()
        num_batches += 1
        
        # Collect predictions for AUC/accuracy
        preds = outputs['bce_predictions'].detach().cpu().numpy()
        labels_np = targets.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        for i in range(len(preds)):
            valid_indices = mask_np[i] == 1
            all_preds.extend(preds[i][valid_indices])
            all_labels.extend(labels_np[i][valid_indices])
        
        # Collect IRT mastery and p_correct for alignment metrics
        if 'mastery_irt' in outputs and 'p_correct' in outputs:
            mastery_irt = outputs['mastery_irt'].detach().cpu().numpy()
            # p_correct is the sigmoid of performance logits: σ(θ - β)
            # It's already in [0,1] range
            p_correct = torch.sigmoid(outputs['performance_logits']).detach().cpu().numpy()
            
            for i in range(len(mastery_irt)):
                valid_indices = mask_np[i] == 1
                all_mastery_irt.extend(mastery_irt[i][valid_indices])
                all_p_correct.extend(p_correct[i][valid_indices])
    
    # Compute aggregate metrics
    avg_total_loss = total_loss / num_batches
    avg_bce_loss = total_bce_loss / num_batches
    avg_align_loss = total_align_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    # Compute AUC and accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    
    # Compute IRT alignment metrics
    alignment_metrics = {}
    if all_mastery_irt and all_p_correct:
        mastery_np = np.array(all_mastery_irt)
        p_correct_np = np.array(all_p_correct)
        
        # MSE and MAE between M_IRT and actual correctness
        alignment_metrics['mse'] = np.mean((mastery_np - all_labels) ** 2)
        alignment_metrics['mae'] = np.mean(np.abs(mastery_np - all_labels))
        
        # Head Agreement: Correlation between M_IRT and p_correct (IRT consistency)
        if len(np.unique(mastery_np)) > 1 and len(np.unique(p_correct_np)) > 1:
            alignment_metrics['head_agreement'] = np.corrcoef(mastery_np, p_correct_np)[0, 1]
        else:
            alignment_metrics['head_agreement'] = 0.0
        
        # Average IRT mastery
        alignment_metrics['mean_mastery'] = np.mean(mastery_np)
    else:
        alignment_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'head_agreement': 0.0,
            'mean_mastery': 0.0
        }
    
    # Compute loss ratios
    if avg_total_loss > 0:
        loss_ratio_bce = avg_bce_loss / avg_total_loss
        loss_ratio_align = avg_align_loss / avg_total_loss
        loss_ratio_reg = avg_reg_loss / avg_total_loss
    else:
        loss_ratio_bce = 0.0
        loss_ratio_align = 0.0
        loss_ratio_reg = 0.0
    
    return {
        # L1 - Performance
        'l1_bce': avg_bce_loss,
        'auc': bce_metrics['auc'],
        'accuracy': bce_metrics['acc'],
        
        # L_align - IRT Alignment
        'align_loss': avg_align_loss,
        'align_mse': alignment_metrics['mse'],
        'align_mae': alignment_metrics['mae'],
        'head_agreement': alignment_metrics['head_agreement'],
        'mean_mastery': alignment_metrics['mean_mastery'],
        
        # L_reg - Skill Difficulty Regularization
        'reg_loss': avg_reg_loss,
        
        # L_total - Combined
        'total_loss': avg_total_loss,
        'loss_ratio_bce': loss_ratio_bce,
        'loss_ratio_align': loss_ratio_align,
        'loss_ratio_reg': loss_ratio_reg,
        
        # Legacy (for backwards compatibility)
        'loss': avg_total_loss,
        'bce_loss': avg_bce_loss
    }


def validate(model, val_loader, device, beta_irt, lambda_reg):
    """
    Validate the model with comprehensive metrics tracking.
    
    Args:
        model: iKT2 model instance
        val_loader: validation data loader
        device: torch device
        beta_irt: IRT skill difficulties (required, can be None)
        lambda_reg: regularization coefficient (required, explicit)
    
    Note:
        All parameters REQUIRED. No defaults per reproducibility guidelines.
        Code will fail if parameters not explicitly provided.
    """
    model.eval()
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_align_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    all_mastery_irt = []
    all_p_correct = []
    
    with torch.no_grad():
        for batch in val_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Forward pass
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            
            # Get targets
            targets = batch['shft_rseqs'].to(device)
            
            # Compute loss
            compute_loss_fn = get_model_attr(model, 'compute_loss')
            loss_dict = compute_loss_fn(outputs, targets, beta_irt=beta_irt, lambda_reg=lambda_reg)
            loss = loss_dict['total_loss']
            bce_loss = loss_dict['bce_loss']
            align_loss = loss_dict['align_loss']
            reg_loss = loss_dict['reg_loss']
            
            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_align_loss += align_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            
            # Collect predictions for AUC/accuracy
            preds = outputs['bce_predictions'].detach().cpu().numpy()
            labels_np = targets.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for i in range(len(preds)):
                valid_indices = mask_np[i] == 1
                all_preds.extend(preds[i][valid_indices])
                all_labels.extend(labels_np[i][valid_indices])
            
            # Collect IRT mastery and p_correct for alignment metrics
            if 'mastery_irt' in outputs and 'performance_logits' in outputs:
                mastery_irt = outputs['mastery_irt'].detach().cpu().numpy()
                p_correct = torch.sigmoid(outputs['performance_logits']).detach().cpu().numpy()
                
                for i in range(len(mastery_irt)):
                    valid_indices = mask_np[i] == 1
                    all_mastery_irt.extend(mastery_irt[i][valid_indices])
                    all_p_correct.extend(p_correct[i][valid_indices])
    
    # Compute aggregate metrics
    avg_total_loss = total_loss / num_batches
    avg_bce_loss = total_bce_loss / num_batches
    avg_align_loss = total_align_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    # Compute AUC and accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    
    # Compute IRT alignment metrics
    alignment_metrics = {}
    if all_mastery_irt and all_p_correct:
        mastery_np = np.array(all_mastery_irt)
        p_correct_np = np.array(all_p_correct)
        
        # MSE and MAE
        alignment_metrics['mse'] = np.mean((mastery_np - all_labels) ** 2)
        alignment_metrics['mae'] = np.mean(np.abs(mastery_np - all_labels))
        
        # Head Agreement: Correlation between M_IRT and p_correct (IRT consistency)
        if len(np.unique(mastery_np)) > 1 and len(np.unique(p_correct_np)) > 1:
            alignment_metrics['head_agreement'] = np.corrcoef(mastery_np, p_correct_np)[0, 1]
        else:
            alignment_metrics['head_agreement'] = 0.0
        
        alignment_metrics['mean_mastery'] = np.mean(mastery_np)
    else:
        alignment_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'head_agreement': 0.0,
            'mean_mastery': 0.0
        }
    
    # Compute loss ratios
    if avg_total_loss > 0:
        loss_ratio_bce = avg_bce_loss / avg_total_loss
        loss_ratio_align = avg_align_loss / avg_total_loss
        loss_ratio_reg = avg_reg_loss / avg_total_loss
    else:
        loss_ratio_bce = 0.0
        loss_ratio_align = 0.0
        loss_ratio_reg = 0.0
    
    return {
        # L1 - Performance
        'l1_bce': avg_bce_loss,
        'auc': bce_metrics['auc'],
        'accuracy': bce_metrics['acc'],
        
        # L_align - IRT Alignment
        'align_loss': avg_align_loss,
        'align_mse': alignment_metrics['mse'],
        'align_mae': alignment_metrics['mae'],
        'head_agreement': alignment_metrics['head_agreement'],
        'mean_mastery': alignment_metrics['mean_mastery'],
        
        # L_reg - Skill Difficulty Regularization
        'reg_loss': avg_reg_loss,
        
        # L_total - Combined
        'total_loss': avg_total_loss,
        'loss_ratio_bce': loss_ratio_bce,
        'loss_ratio_align': loss_ratio_align,
        'loss_ratio_reg': loss_ratio_reg,
        
        # Legacy (for backwards compatibility)
        'loss': avg_total_loss,
        'bce_loss': avg_bce_loss,
        'bce_auc': bce_metrics['auc'],
        'bce_acc': bce_metrics['acc']
    }


def main():
    parser = argparse.ArgumentParser(description='Train iKT2 model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--gradient_clip', type=float, required=True)
    parser.add_argument('--patience', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    
    # Model architecture
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--num_encoder_blocks', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--emb_type', type=str, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    
    # iKT2-specific parameters
    parser.add_argument('--lambda_align', type=float, required=True,
                        help='Alignment coefficient for IRT mastery (recommended: 1.0)')
    parser.add_argument('--lambda_reg', type=float, required=True,
                        help='Regularization coefficient for skill difficulty embeddings (recommended: 0.1)')
    parser.add_argument('--irt_noise', type=float, required=True,
                        help='Noise level for IRT initialization (0.0=exact, 0.1=10%% std, recommended: 0.0-0.2)')
    parser.add_argument('--phase', type=str, required=True,
                       help='Training phase: "1" (BCE + align), "2" (BCE + align + reg), or "null" (automatic two-phase)')
    parser.add_argument('--rasch_path', type=str, required=True,
                       help='Path to Rasch targets file (explicit, e.g., data/{dataset}/rasch_targets.pkl)')
    
    # Compatibility parameters (for iKT, ignored by iKT2)
    parser.add_argument('--lambda_penalty', type=float, required=True,
                        help='[iKT only] Penalty coefficient - ignored by iKT2')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='[iKT only] Tolerance threshold - ignored by iKT2')
    parser.add_argument('--mastery_method', type=str, required=True,
                        help='[iKT only] Mastery method - ignored by iKT2')
    
    # Monitoring & evaluation
    parser.add_argument('--monitor_freq', type=int, required=True)
    parser.add_argument('--auto_shifted_eval', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    
    # Educational visualizations (not used in this script but required by launcher)
    parser.add_argument('--min_trajectory_steps', type=int, required=True)
    parser.add_argument('--num_trajectories', type=int, required=True)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get experiment directory from environment
    experiment_dir = os.environ.get('EXPERIMENT_DIR')
    if not experiment_dir:
        print("⚠️  WARNING: EXPERIMENT_DIR not set. Using current directory.")
        experiment_dir = '.'
    
    # Initialize metrics_epoch.csv with comprehensive metrics
    metrics_csv_path = os.path.join(experiment_dir, 'metrics_validation.csv')
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'lambda_align', 'lambda_reg',
                'val_auc', 'val_align_mse', 'val_align_mae', 'val_head_agreement', 'val_reg_loss'
            ])
    
    # Determine training mode
    if args.phase == "null" or args.phase is None:
        training_mode = "automatic_two_phase"
        current_phase = 1  # Start with Phase 1
    else:
        training_mode = "manual_single_phase"
        current_phase = int(args.phase)
    
    print("="*80)
    print("iKT2 Training")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Training mode: {training_mode}")
    if training_mode == "manual_single_phase":
        print(f"Phase: {current_phase}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    
    # Load data config from configs/data_config.json
    import json
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
    
    model_name = "ikt2"
    train_loader, valid_loader = init_dataset4train(
        args.dataset, 
        model_name,
        data_config,
        args.fold,
        args.batch_size
    )
    
    # Get num_c from data config
    num_c = data_config[args.dataset]['num_c']
    dataset_path = data_config[args.dataset]['dpath']
    
    print(f"✓ Data loaded")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Valid: {len(valid_loader)} batches")
    print(f"  Skills: {num_c}\n")
    
    # Load skill difficulties from IRT calibration
    # rasch_path comes explicitly from CLI (no hidden defaults)
    skill_difficulties = load_skill_difficulties_from_irt(args.rasch_path, num_c)
    print()
    
    # Create model
    print("Creating model...")
    model = iKT2(
        num_c=num_c,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_blocks=args.num_encoder_blocks,
        d_ff=args.d_ff,
        dropout=args.dropout,
        emb_type=args.emb_type,
        lambda_align=args.lambda_align,
        lambda_reg=args.lambda_reg,
        phase=current_phase
    ).to(device)
    
    print(f"\u2713 Model created")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture: d_model={args.d_model}, n_heads={args.n_heads}, "
          f"num_encoder_blocks={args.num_encoder_blocks}, d_ff={args.d_ff}")
    
    # Initialize skill difficulties from IRT calibration (fixes scale drift)
    # COMMENTED: Direct IRT init creates "lazy learning" - model doesn't learn from data
    # Better approach: Initialize to 0.0 and let regularization pull toward IRT scale
    # Evidence: Exp 947580 (no init) achieves BKT r=0.506, while exp 524632 (with init) 
    # gets only r=0.302 despite identical AUC. Regularization-based learning is superior.
    # if skill_difficulties is not None:
    #     beta_irt_device = skill_difficulties.to(device)
    #     model.load_irt_difficulties(beta_irt_device, noise_epsilon=args.irt_noise)
    # else:
    #     print("⚠️  No IRT difficulties loaded - β parameters initialized at 0.0")
    print("✓ β parameters initialized at 0.0 (will learn via regularization)")
    print(f"  Hyperparameters: lambda_align={args.lambda_align}, lambda_reg={args.lambda_reg}, irt_noise={args.irt_noise}")
    print()
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    
    # Create optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Training loop
    print("\n🚀 Starting training...")
    if training_mode == "automatic_two_phase":
        print(f"Phase 1: BCE + IRT Alignment (will auto-switch to Phase 2 on convergence)")
    else:
        print(f"Phase {current_phase}: {'BCE + IRT Alignment' if current_phase == 1 else 'BCE + IRT Alignment + Regularization'}")
    
    best_val_auc = 0.0
    patience_counter = 0
    history = []
    phase1_converged = False
    phase1_best_epoch = 0
    phase2_started_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        beta_irt_device = skill_difficulties.to(device) if skill_difficulties is not None else None
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.gradient_clip, 
                                     beta_irt=beta_irt_device, lambda_reg=args.lambda_reg)
        print(f"Train - Total Loss: {train_metrics['total_loss']:.4f}, "
              f"BCE: {train_metrics['l1_bce']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, "
              f"Align: {train_metrics['align_loss']:.4f}, "
              f"Head_Agr: {train_metrics['head_agreement']:.3f}")
        
        # Validate
        val_metrics = validate(model, valid_loader, device, beta_irt=beta_irt_device, lambda_reg=args.lambda_reg)
        print(f"Valid - Total Loss: {val_metrics['total_loss']:.4f}, "
              f"BCE: {val_metrics['l1_bce']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"Align: {val_metrics['align_loss']:.4f}, "
              f"Head_Agr: {val_metrics['head_agreement']:.3f}")
        
        # Save comprehensive epoch results with hyperparameters
        epoch_results = {
            'epoch': epoch + 1,
            'phase': current_phase,
            'lambda_align': args.lambda_align,
            'lambda_reg': args.lambda_reg,
            'train': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in train_metrics.items()},
            'validation': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in val_metrics.items()}
        }
        history.append(epoch_results)
        
        # Append to metrics_validation.csv for reproducibility
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                current_phase,
                args.lambda_align,
                args.lambda_reg,
                val_metrics['auc'],
                val_metrics['align_mse'],
                val_metrics['align_mae'],
                val_metrics['head_agreement'],
                val_metrics['reg_loss']
            ])
        
        # Check for improvement (using AUC as primary metric)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            if training_mode == "automatic_two_phase" and current_phase == 1:
                phase1_best_epoch = epoch + 1
            
            # Save best model (handle DataParallel wrapper)
            checkpoint_path = os.path.join(experiment_dir, 'model_best.pth')
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_acc': val_metrics['accuracy'],
                'val_head_agreement': val_metrics['head_agreement'],
                'config': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                # Check if we should switch phases in automatic mode
                if training_mode == "automatic_two_phase" and current_phase == 1 and not phase1_converged:
                    print("\n" + "="*80)
                    print("✓ PHASE 1 CONVERGED - SWITCHING TO PHASE 2")
                    print("="*80)
                    print(f"Phase 1 best epoch: {phase1_best_epoch}")
                    print(f"Phase 1 best AUC: {best_val_auc:.4f}")
                    
                    # Mark Phase 1 as converged
                    phase1_converged = True
                    phase2_started_epoch = epoch + 1
                    
                    # Switch to Phase 2
                    current_phase = 2
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.phase = 2
                    else:
                        model.phase = 2
                    
                    # Reset early stopping for Phase 2
                    patience_counter = 0
                    best_val_auc = 0.0  # Reset to find best in Phase 2
                    
                    print(f"Switching to Phase 2: BCE + IRT Alignment + Regularization")
                    print(f"Remaining epochs: {args.epochs - epoch - 1}")
                    print(f"Loss: L_total = L_BCE + λ_align={args.lambda_align} × L_align + λ_reg={args.lambda_reg} × L_reg")
                    print("="*80 + "\n")
                    continue  # Continue training, don't break
                else:
                    # Either manual mode or Phase 2 converged - stop training
                    if training_mode == "automatic_two_phase" and current_phase == 2:
                        print("\n⏹️  Phase 2 converged - Training complete")
                    else:
                        print("\n⏹️  Early stopping triggered")
                    break
    
    # Check if training completed successfully
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    if training_mode == "automatic_two_phase":
        if phase1_converged and current_phase == 2:
            print("✓ Two-phase training completed successfully")
            print(f"  Phase 1: Converged at epoch {phase1_best_epoch}")
            print(f"  Phase 2: Started at epoch {phase2_started_epoch}, ran for {epoch - phase2_started_epoch + 1} epochs")
        elif not phase1_converged:
            print("⚠️  WARNING: Training stopped but Phase 1 did not converge")
            print(f"   Consider increasing --epochs (current: {args.epochs})")
        elif current_phase == 1:
            print("⚠️  WARNING: Phase 1 converged but no epochs left for Phase 2")
            print(f"   Phase 1 converged at epoch {epoch + 1}")
            print(f"   Consider increasing --epochs (current: {args.epochs})")
    else:
        print(f"✓ Single-phase training (Phase {current_phase}) completed")
    
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print("="*80 + "\n")
    
    # Save training history
    history_path = os.path.join(experiment_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")
    
    # Run auto shifted eval if requested
    if args.auto_shifted_eval:
        print("\n" + "="*80)
        print("Running auto shifted evaluation...")
        print("="*80 + "\n")
        
        eval_script = os.path.join('/workspaces/pykt-toolkit/examples', 'eval_ikt2.py')
        checkpoint_path = os.path.join(experiment_dir, 'model_best.pth')
        
        eval_cmd = [
            'python', eval_script,
            '--dataset', args.dataset,
            '--fold', str(args.fold),
            '--checkpoint', checkpoint_path,
            '--batch_size', str(args.batch_size),
            '--seq_len', str(args.seq_len),
            '--output_dir', experiment_dir
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
