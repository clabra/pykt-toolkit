#!/usr/bin/env python3
"""
Training script for iKT model (Interpretable Knowledge Tracing) using PyKT framework patterns.


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



iKT Architecture:
- Single encoder with dual-stream processing
- Head 1: Performance prediction (BCE loss)
- Head 2: Skill vector {Mi} with Rasch loss (phase-dependent)

Two-Phase Training:
- Phase 1: Pure Rasch alignment (L_total = L2, epsilon=0)
- Phase 2: Constrained optimization (L_total = L1 + λ_penalty × mean(max(0, |Mi-M_rasch|-ε)²), epsilon>0)
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
from pykt.models.ikt import iKT
from examples.experiment_utils import compute_auc_acc


def get_model_attr(model, attr_name):
    """Safely get model attribute whether wrapped in DataParallel or not."""
    if isinstance(model, torch.nn.DataParallel):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)


def load_skill_difficulties_from_irt(rasch_path, num_c):
    """
    Load IRT-calibrated skill difficulties for Option 1b regularization.
    
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


# UNUSED DEAD CODE - Function defined but never called
# train_ikt.py only uses load_skill_difficulties_from_irt() which requires 'skill_difficulties' key
# This function is kept commented for reference but should not be used
#
# def load_rasch_targets(rasch_path, num_c, mastery_method):
#     """
#     Load pre-computed mastery targets (BKT or IRT/Rasch, standard or monotonic).
#     
#     Args:
#         rasch_path: Path to mastery targets pickle file (explicit, no fallback)
#         num_c: Number of concepts/skills
#         mastery_method: Method used ('bkt', 'irt', 'bkt_mono', 'irt_mono')
#     
#     Returns:
#         dict: Dictionary with mastery targets and metadata
#               - If file exists: {'rasch_targets': dict, 'student_abilities': dict, ...}
#               - If not: {'mode': 'random', 'num_c': num_c}
#     
#     Note:
#         Path must be explicitly provided via --rasch_path CLI argument.
#         No hardcoded defaults allowed (Explicit Parameters, Zero Defaults).
#     """
#     import pickle
#     
#     # Determine if monotonic version requested
#     is_monotonic = mastery_method.endswith('_mono')
#     
#     # Try to load from file
#     if os.path.exists(rasch_path):
#         print(f"✓ Loading mastery targets from: {rasch_path}")
#         print(f"  Method: {mastery_method.upper()}")
#         if is_monotonic:
#             print(f"  Monotonic smoothing: ENABLED")
#         
#         try:
#             with open(rasch_path, 'rb') as f:
#                 data = pickle.load(f)
#             
#             # Handle both BKT and IRT formats
#             # Normalize to common 'rasch_targets' key
#             if 'bkt_targets' in data:
#                 data['rasch_targets'] = data['bkt_targets']
#                 print(f"  Loaded BKT targets for {len(data['bkt_targets'])} students")
#                 if 'bkt_params' in data:
#                     print(f"  BKT parameters available for {len(data['bkt_params'])} skills")
#             elif 'rasch_targets' in data:
#                 print(f"  Loaded IRT/Rasch targets for {len(data['rasch_targets'])} students")
#             else:
#                 raise ValueError("Invalid mastery file: missing 'bkt_targets' or 'rasch_targets' key")
#             
#             if 'metadata' in data:
#                 meta = data['metadata']
#                 print(f"  Metadata: {meta}")
#                 
#                 # Verify monotonic status if specified
#                 file_monotonic = meta.get('monotonic', False)
#                 if is_monotonic and not file_monotonic:
#                     print(f"  ⚠️  WARNING: Requested monotonic version but file has monotonic={file_monotonic}")
#                     print(f"             Make sure you're loading the correct file (e.g., *_mono.pkl)")
#                 elif not is_monotonic and file_monotonic:
#                     print(f"  ⚠️  WARNING: Requested standard version but file has monotonic={file_monotonic}")
#                     print(f"             Make sure you're loading the correct file (not *_mono.pkl)")
#             
#             return data
#             
#         except Exception as e:
#             print(f"✗ Failed to load mastery targets: {e}")
#             raise RuntimeError(
#                 f"Failed to load Rasch/IRT targets from {rasch_path}.\n"
#                 f"iKT requires pre-computed mastery targets for training.\n"
#                 f"Please generate them first:\n"
#                 f"  BKT: python examples/train_bkt.py --dataset {{dataset}}\n"
#                 f"  IRT: python examples/compute_rasch_targets.py --dataset {{dataset}} --dynamic"
#             ) from e
#     else:
#         # Rasch targets are REQUIRED - no random fallback
#         raise FileNotFoundError(
#             f"Rasch/IRT mastery targets not found at: {rasch_path}\n"
#             f"iKT requires pre-computed mastery targets for training.\n"
#             f"Please generate them first:\n"
#             f"  BKT: python examples/train_bkt.py --dataset {{dataset}}\n"
#             f"  IRT: python examples/compute_rasch_targets.py --dataset {{dataset}} --dynamic\n\n"
#             f"Expected file location: {rasch_path}"
#         )


def train_epoch(model, train_loader, optimizer, device, gradient_clip, beta_irt, lambda_reg):
    """Train for one epoch with Option 1b skill regularization."""
    model.train()
    total_loss = 0.0
    total_bce_loss = 0.0
    total_reg_loss = 0.0
    total_penalty_loss = 0.0
    num_batches = 0
    
    # For AUC/accuracy computation
    all_preds = []
    all_labels = []
    
    # For skill-target alignment metrics
    all_skill_vectors = []
    all_beta_targets = []
    
    for batch in train_loader:
        questions = batch['cseqs'].to(device)
        responses = batch['rseqs'].to(device)
        questions_shifted = batch['shft_cseqs'].to(device)
        mask = batch['masks'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (no rasch_targets needed - model uses internal embeddings)
        outputs = model(q=questions, r=responses, qry=questions_shifted)
        
        # Get targets
        targets = batch['shft_rseqs'].to(device)
        
        # Compute loss with skill regularization (access through module if DataParallel)
        compute_loss_fn = get_model_attr(model, 'compute_loss')
        loss_dict = compute_loss_fn(outputs, targets, beta_irt=beta_irt, lambda_reg=lambda_reg)
        loss = loss_dict['total_loss']
        bce_loss = loss_dict['bce_loss']
        reg_loss = loss_dict['reg_loss']
        penalty_loss = loss_dict.get('penalty_loss', torch.tensor(0.0))
        
        # Backward pass
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_reg_loss += reg_loss.item()
        total_penalty_loss += penalty_loss.item()
        num_batches += 1
        
        # Collect predictions for AUC/accuracy
        preds = outputs['bce_predictions'].detach().cpu().numpy()
        labels_np = targets.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        for i in range(len(preds)):
            valid_indices = mask_np[i] == 1
            all_preds.extend(preds[i][valid_indices])
            all_labels.extend(labels_np[i][valid_indices])
        
        # Collect skill vectors and beta targets for alignment metrics
        if outputs.get('skill_vector') is not None and outputs.get('beta_targets') is not None:
            all_skill_vectors.append(outputs['skill_vector'].detach().cpu())
            all_beta_targets.append(outputs['beta_targets'].detach().cpu())
    
    # Compute aggregate metrics
    avg_total_loss = total_loss / num_batches
    avg_bce_loss = total_bce_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    avg_penalty_loss = total_penalty_loss / num_batches
    
    # Compute AUC and accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    
    # Compute skill alignment metrics (vs beta targets)
    alignment_metrics = {}
    if all_skill_vectors and all_beta_targets:
        skill_vectors = torch.cat(all_skill_vectors, dim=0)  # [N, L, num_c]
        beta_targets_cat = torch.cat(all_beta_targets, dim=0)  # [N, L, num_c]
        
        # Flatten for metrics
        Mi_flat = skill_vectors.reshape(-1).numpy()
        beta_flat = beta_targets_cat.reshape(-1).numpy()
        
        # MSE and MAE
        alignment_metrics['mse'] = np.mean((Mi_flat - beta_flat) ** 2)
        alignment_metrics['mae'] = np.mean(np.abs(Mi_flat - beta_flat))
        
        # Correlation
        if len(np.unique(Mi_flat)) > 1 and len(np.unique(beta_flat)) > 1:
            alignment_metrics['correlation'] = np.corrcoef(Mi_flat, beta_flat)[0, 1]
        else:
            alignment_metrics['correlation'] = 0.0
        
        # Violation metrics (assuming epsilon from model)
        model_obj = model.module if hasattr(model, 'module') else model
        epsilon = model_obj.epsilon
        lambda_penalty = model_obj.lambda_penalty
        
        deviations = np.abs(Mi_flat - beta_flat)
        violations = np.maximum(0, deviations - epsilon)
        
        alignment_metrics['violation_rate'] = np.mean(deviations > epsilon)
        alignment_metrics['mean_violation'] = np.mean(violations[violations > 0]) if np.any(violations > 0) else 0.0
        alignment_metrics['max_violation'] = np.max(violations)
    else:
        alignment_metrics = {
            'mse': 0.0, 'mae': 0.0, 'correlation': 0.0,
            'violation_rate': 0.0, 'mean_violation': 0.0, 'max_violation': 0.0
        }
        epsilon = 0.0
        lambda_penalty = 1.0
    
    # Compute loss ratios
    if avg_total_loss > 0:
        loss_ratio_l1 = avg_bce_loss / avg_total_loss
        loss_ratio_penalty = (lambda_penalty * avg_penalty_loss) / avg_total_loss if avg_penalty_loss > 0 else 0.0
    else:
        loss_ratio_l1 = 0.0
        loss_ratio_penalty = 0.0
    
    return {
        # L1 - Performance
        'l1_bce': avg_bce_loss,
        'auc': bce_metrics['auc'],
        'accuracy': bce_metrics['acc'],
        
        # L2 - Alignment  
        'l2_mse': alignment_metrics['mse'],
        'l2_mae': alignment_metrics['mae'],
        'corr_beta': alignment_metrics['correlation'],
        
        # L2_penalty - Violations
        'penalty_loss': avg_penalty_loss,
        'violation_rate': alignment_metrics['violation_rate'],
        'mean_violation': alignment_metrics['mean_violation'],
        'max_violation': alignment_metrics['max_violation'],
        
        # L_total - Combined
        'total_loss': avg_total_loss,
        'loss_ratio_l1': loss_ratio_l1,
        'loss_ratio_penalty': loss_ratio_penalty,
        
        # Legacy (for backwards compatibility)
        'loss': avg_total_loss,
        'bce_loss': avg_bce_loss,
        'reg_loss': avg_reg_loss
    }


def validate(model, val_loader, device, lambda_penalty, beta_irt, lambda_reg):
    """
    Validate the model with comprehensive metrics tracking.
    
    Args:
        model: iKT model instance
        val_loader: validation data loader
        device: torch device
        lambda_penalty: penalty coefficient (required, no default per reproducibility guidelines)
        beta_irt: optional IRT skill difficulties, lambda_reg: regularization coefficient
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_reg_loss = 0.0
    total_penalty_loss = 0.0
    num_batches = 0
    
    # For Rasch alignment metrics
    all_skill_vectors = []
    all_beta_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            mask = batch['masks'].to(device)
            labels = batch['shft_rseqs'].to(device)
            
            # Prepare Rasch targets if available
            # rasch_batch removed (Option 1b)
            if False:  # rasch_targets removed (Option 1b)
                if rasch_targets.get('mode') == 'random':
                    # Generate random Rasch targets in [0.0, 1.0] range
                    batch_size, seq_len = questions.shape
                    num_c = rasch_targets['num_c']
                    rasch_batch = torch.rand(batch_size, seq_len, num_c, device=device)
                else:
                    # Load real Rasch targets for this batch
                    rasch_data = rasch_targets.get('rasch_targets', {})
                    uids = batch['uids']
                    batch_size, seq_len = questions.shape
                    num_c = rasch_targets['metadata']['num_skills']
                    
                    rasch_batch = torch.zeros(batch_size, seq_len, num_c, device=device)
                    for i, uid in enumerate(uids):
                        if uid in rasch_data:
                            target_tensor = rasch_data[uid]
                            # Pad or truncate to match seq_len
                            actual_len = min(target_tensor.shape[0], seq_len)
                            rasch_batch[i, :actual_len, :] = target_tensor[:actual_len, :].to(device)
            
            # Forward pass
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            
            # Compute loss (access through module if DataParallel)
            compute_loss_fn = get_model_attr(model, 'compute_loss')
            loss_dict = compute_loss_fn(outputs, labels)
            loss = loss_dict['total_loss']
            penalty_loss = loss_dict.get('penalty_loss', torch.tensor(0.0))
            
            total_loss += loss.item()
            total_bce_loss += loss_dict['bce_loss'].item()
            total_reg_loss += loss_dict['reg_loss'].item()
            total_penalty_loss += penalty_loss.item()
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
            
            # Collect skill vectors and Rasch targets for alignment metrics
            if outputs.get("skill_vector") is not None and outputs.get("beta_targets") is not None:
                all_skill_vectors.append(outputs['skill_vector'].cpu())
                all_beta_targets.append(outputs["beta_targets"].detach().cpu())
    
    # Compute aggregate metrics
    avg_total_loss = total_loss / num_batches
    avg_bce_loss = total_bce_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    avg_penalty_loss = total_penalty_loss / num_batches
    
    # Compute AUC and accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    
    # Compute Rasch alignment metrics
    alignment_metrics = {}
    if all_skill_vectors and all_beta_targets:
        skill_vectors = torch.cat(all_skill_vectors, dim=0)  # [N, L, num_c]
        beta_targets_cat = torch.cat(all_beta_targets, dim=0)  # [N, L, num_c]
        
        # Flatten for metrics
        Mi_flat = skill_vectors.reshape(-1).numpy()
        beta_flat = beta_targets_cat.reshape(-1).numpy()
        
        # MSE and MAE
        alignment_metrics['mse'] = np.mean((Mi_flat - beta_flat) ** 2)
        alignment_metrics['mae'] = np.mean(np.abs(Mi_flat - beta_flat))
        
        # Correlation
        if len(np.unique(Mi_flat)) > 1 and len(np.unique(beta_flat)) > 1:
            alignment_metrics['correlation'] = np.corrcoef(Mi_flat, beta_flat)[0, 1]
        else:
            alignment_metrics['correlation'] = 0.0
        
        # Violation metrics
        model_obj = model.module if hasattr(model, 'module') else model
        epsilon = model_obj.epsilon
        lambda_penalty_val = model_obj.lambda_penalty
        
        deviations = np.abs(Mi_flat - beta_flat)
        violations = np.maximum(0, deviations - epsilon)
        
        alignment_metrics['violation_rate'] = np.mean(deviations > epsilon)
        alignment_metrics['mean_violation'] = np.mean(violations[violations > 0]) if np.any(violations > 0) else 0.0
        alignment_metrics['max_violation'] = np.max(violations)
    else:
        rasch_metrics = {
            'mse': 0.0, 'mae': 0.0, 'correlation': 0.0,
            'violation_rate': 0.0, 'mean_violation': 0.0, 'max_violation': 0.0
        }
        epsilon = 0.0
        lambda_penalty_val = 1.0
    
    # Compute loss ratios
    if avg_total_loss > 0:
        loss_ratio_l1 = avg_bce_loss / avg_total_loss
        loss_ratio_penalty = (lambda_penalty_val * avg_penalty_loss) / avg_total_loss if avg_penalty_loss > 0 else 0.0
    else:
        loss_ratio_l1 = 0.0
        loss_ratio_penalty = 0.0
    
    return {
        # L1 - Performance
        'l1_bce': avg_bce_loss,
        'auc': bce_metrics['auc'],
        'accuracy': bce_metrics['acc'],
        
        # L2 - Alignment  
        'l2_mse': alignment_metrics['mse'],
        'l2_mae': alignment_metrics['mae'],
        'corr_beta': alignment_metrics['correlation'],
        
        # L2_penalty - Violations
        'penalty_loss': avg_penalty_loss,
        'violation_rate': alignment_metrics['violation_rate'],
        'mean_violation': alignment_metrics['mean_violation'],
        'max_violation': alignment_metrics['max_violation'],
        
        # L_total - Combined
        'total_loss': avg_total_loss,
        'loss_ratio_l1': loss_ratio_l1,
        'loss_ratio_penalty': loss_ratio_penalty,
        
        # Legacy (for backwards compatibility)
        'loss': avg_total_loss,
        'bce_loss': avg_bce_loss,
        'reg_loss': avg_reg_loss,
        'bce_auc': bce_metrics['auc'],
        'bce_acc': bce_metrics['acc']
    }


def main():
    parser = argparse.ArgumentParser(description='Train iKT model')
    
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
    
    # iKT-specific parameters
    parser.add_argument('--lambda_penalty', type=float, required=True,
                        help='Penalty coefficient for Phase 2 constraint (recommended: 10.0-1000.0)')
    parser.add_argument('--lambda_reg', type=float, required=True,
                        help='Regularization coefficient for skill difficulty embeddings (recommended: 0.01-1.0)')
    parser.add_argument('--epsilon', type=float, required=True, 
                       help='Rasch tolerance threshold for Phase 2 (0.0 for Phase 1)')
    parser.add_argument('--phase', type=str, required=True,
                       help='Training phase: "1" (Rasch alignment), "2" (constrained optimization), or "null" (automatic two-phase)')
    parser.add_argument('--rasch_path', type=str, required=True,
                       help='Path to Rasch targets file (default: data/{dataset}/rasch_targets.pkl)')
    parser.add_argument('--mastery_method', type=str, required=True, choices=['bkt', 'irt', 'bkt_mono', 'irt_mono'],
                       help='Method for mastery targets: bkt, irt, bkt_mono (monotonic BKT), or irt_mono (monotonic IRT)')
    
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
                'epoch', 'phase', 'lambda_penalty', 'epsilon',
                'val_auc', 'val_l1_auc', 'val_l2_mae', 'val_l2_mse', 'val_l2_penalty'
            ])
    
    # Determine training mode
    # Handle string "null" from CLI (required for reproducibility compliance)
    if args.phase == "null" or args.phase is None:
        training_mode = "automatic_two_phase"
        current_phase = 1  # Start with Phase 1
    else:
        training_mode = "manual_single_phase"
        current_phase = int(args.phase)
    
    print("="*80)
    print("iKT Training")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Training mode: {'Automatic two-phase (Phase 1 → Phase 2)' if training_mode == 'automatic_two_phase' else f'Manual single-phase (Phase {current_phase})'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Lambda_penalty (constraint coefficient): {args.lambda_penalty}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Device: {device}")
    print(f"Experiment dir: {experiment_dir}")
    print("="*80)
    
    # Initialize dataset
    print("\n📊 Loading dataset...")
    
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
    
    model_name = "ikt"
    train_loader, valid_loader = init_dataset4train(
        args.dataset, model_name, data_config, args.fold, args.batch_size
    )
    
    # Get number of concepts from data config
    num_c = data_config[args.dataset]['num_c']
    dataset_path = data_config[args.dataset]['dpath']
    
    print(f"✓ Dataset loaded: {num_c} concepts")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(valid_loader)}")
    
    # Load IRT skill difficulties for regularization (Option 1b)
    print("\n🎯 Loading IRT skill difficulties...")
    # rasch_path comes explicitly from CLI (no hidden defaults)
    skill_difficulties = load_skill_difficulties_from_irt(args.rasch_path, num_c)
    
    # Initialize model
    print("\n🏗️  Initializing model...")
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
    
    # Initialize skill difficulty embeddings with IRT values
    if skill_difficulties is not None:
        with torch.no_grad():
            model.skill_difficulty_emb.weight.copy_(
                skill_difficulties.unsqueeze(1).to(device)
            )
        print(f"✓ Initialized skill difficulty embeddings with IRT values")
    
    # Multi-GPU support: wrap model with DataParallel if multiple GPUs available
    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        visible_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        if gpu_count > 1:
            print(f"✓ Multi-GPU detected: {gpu_count} GPUs available")
            print(f"  CUDA_VISIBLE_DEVICES: {visible_env if visible_env else 'not set (using all)'}")
            try:
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                print(f"  GPU devices: {', '.join(gpu_names)}")
            except Exception:
                pass
            print(f"✓ Wrapping model with DataParallel for multi-GPU training")
            model = torch.nn.DataParallel(model)
        else:
            print(f"✓ Single GPU mode")
    else:
        print(f"✓ CPU mode")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {num_params:,} parameters")
    
    # Initialize optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Training loop
    print("\n🚀 Starting training...")
    if training_mode == "automatic_two_phase":
        print(f"Phase 1: Rasch Alignment (will auto-switch to Phase 2 on convergence)")
    else:
        print(f"Phase {current_phase}: {'Rasch Alignment' if current_phase == 1 else 'Constrained Optimization'}")
    
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
              f"L1 (BCE): {train_metrics['l1_bce']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, "
              f"L_reg: {train_metrics.get('reg_loss', 0.0):.4f}, "
              f"Penalty: {train_metrics['penalty_loss']:.4f}")
        
        # Validate
        val_metrics = validate(model, valid_loader, device, args.lambda_penalty, beta_irt=beta_irt_device, lambda_reg=args.lambda_reg)
        print(f"Valid - Total Loss: {val_metrics['total_loss']:.4f}, "
              f"L1 (BCE): {val_metrics['l1_bce']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"Penalty: {val_metrics['penalty_loss']:.4f}, "
              f"Violations: {val_metrics['violation_rate']:.2%}")
        
        # Save comprehensive epoch results with hyperparameters
        epoch_results = {
            'epoch': epoch + 1,
            'phase': current_phase,
            'lambda_penalty': args.lambda_penalty,
            'epsilon': args.epsilon,
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
                args.lambda_penalty,
                args.epsilon,
                val_metrics['auc'],  # val_auc (AUC from predictions)
                val_metrics['auc'],  # val_l1_auc (same as val_auc, from L1/BCE head)
                val_metrics['l2_mae'],  # val_l2_mae
                val_metrics['l2_mse'],  # val_l2_mse
                val_metrics['penalty_loss']  # val_l2_penalty
            ])
        
        # Check for improvement (using bce_auc as primary metric)
        if val_metrics['bce_auc'] > best_val_auc:
            best_val_auc = val_metrics['bce_auc']
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
                'val_bce_auc': val_metrics['bce_auc'],
                'val_bce_acc': val_metrics['bce_acc'],
                'val_reg_loss': val_metrics.get('reg_loss', 0.0),
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
                    model.phase = 2  # Update model's phase
                    
                    # Reset early stopping for Phase 2
                    patience_counter = 0
                    best_val_auc = 0.0  # Reset to find best in Phase 2
                    
                    print(f"Switching to Phase 2: Constrained Optimization")
                    print(f"Remaining epochs: {args.epochs - epoch - 1}")
                    print(f"Loss: L_total = L1 + λ_penalty={args.lambda_penalty} × L2_penalty")
                    print(f"Epsilon tolerance: {args.epsilon}")
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
    
    # Save training+validation history
    results_path = os.path.join(experiment_dir, 'metrics_validation_training.json')
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Saved training+validation metrics to: {results_path}")
    print(f"✓ Saved CSV metrics to: {metrics_csv_path}")
    
    # Save final metrics
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
    
    final_metrics = {
        'best_val_bce_auc': best_val_auc,
        'final_epoch': epoch + 1,
        'total_params': num_params,
        'phase': current_phase,
        'lambda_penalty': args.lambda_penalty,
        'epsilon': args.epsilon,
        'training_mode': training_mode,
        'phase1_converged': phase1_converged if training_mode == "automatic_two_phase" else None,
        'phase1_best_epoch': phase1_best_epoch if training_mode == "automatic_two_phase" else None,
        'phase2_started_epoch': phase2_started_epoch if training_mode == "automatic_two_phase" and phase1_converged else None
    }
    # Final metrics now stored in metrics_validation_training.json (last epoch)
    
    print("\n" + "="*80)
    print("✅ Training completed successfully")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Results saved to: {experiment_dir}")
    print("="*80)
    
    # Launch evaluation if requested
    if args.auto_shifted_eval:
        print("\n" + "="*80)
        print("📊 LAUNCHING EVALUATION")
        print("="*80)
        
        # Build evaluation command with ALL required parameters (explicit, zero defaults)
        eval_cmd = [
            sys.executable,
            'examples/eval_ikt.py',
            '--run_dir', experiment_dir,
            '--ckpt_name', 'model_best.pth',
            '--dataset', args.dataset,
            '--fold', str(args.fold),
            '--batch_size', str(args.batch_size),
            '--seq_len', str(args.seq_len),
            '--d_model', str(args.d_model),
            '--n_heads', str(args.n_heads),
            '--num_encoder_blocks', str(args.num_encoder_blocks),
            '--d_ff', str(args.d_ff),
            '--dropout', str(args.dropout),
            '--emb_type', args.emb_type,
            '--lambda_penalty', str(args.lambda_penalty),
            '--epsilon', str(args.epsilon),
            '--phase', str(current_phase)
        ]
        
        try:
            result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True, cwd='/workspaces/pykt-toolkit')
            print("✅ Evaluation completed successfully")
            # Print summary lines
            for line in result.stdout.strip().split('\n'):
                if 'Test Results' in line or 'AUC:' in line or '=' in line:
                    print(line)
            if result.stderr:
                print(f"⚠️  Evaluation stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed with exit code {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
        except Exception as e:
            print(f"❌ Evaluation failed with exception: {e}")
        
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
