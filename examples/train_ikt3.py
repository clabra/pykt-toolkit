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
- Single encoder with dual-stream processing
- Head 1: Performance prediction (BCE loss)
- Head 2: Reference model alignment (IRT, BKT, etc.)
- Pluggable reference models via abstract interface

Single-Phase Training with Warm-up:
    L_total = (1 - λ(t)) × L_BCE + c × L_stability + λ(t) × L_align
    where λ(t) = λ_target × min(1, epoch / warmup_epochs)

Key Innovation (vs iKT2):
- External validity: Aligns with validated theoretical reference model
- Pluggable architecture: Easy to swap IRT → BKT → DINA → PFA
- Dynamic loss computation: Reference model defines its own alignment losses
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import argparse
import json
import csv
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


def prepare_batch_ref_targets(batch, ref_targets, device):
    """
    Extract batch-specific reference targets from full dataset targets.
    
    For IRT reference model:
    - beta_irt: skill difficulties (already tensor, just move to device)
    - theta_irt: extract student abilities for batch (dict -> tensor)
    - m_ref: extract reference predictions for batch (dict -> tensor, or zeros if missing)
    
    Args:
        batch: Dictionary with 'uids' key containing student IDs
        ref_targets: Full dataset targets from reference_model.load_targets()
        device: torch device
    
    Returns:
        Dictionary with batch-specific tensors
    """
    if ref_targets is None:
        return None
    
    batch_targets = {}
    
    # Skill difficulties (already a tensor, same for all students)
    if 'beta_irt' in ref_targets:
        batch_targets['beta_irt'] = ref_targets['beta_irt'].to(device)
    
    # Student abilities - extract for current batch
    # Supports both static (scalar per student) and dynamic (trajectory per student)
    if 'theta_irt' in ref_targets:
        uids = batch.get('uids', None)
        is_dynamic = ref_targets.get('is_dynamic', False)
        
        if uids is not None:
            batch_size, seq_len = batch['cseqs'].shape
            
            if is_dynamic:
                # Dynamic theta: {uid: [L]} trajectories
                theta_batch = torch.zeros(batch_size, seq_len, dtype=torch.float32)
                for i, uid in enumerate(uids):
                    uid_key = torch.tensor(uid).item() if isinstance(uid, torch.Tensor) else uid
                    theta_traj = ref_targets['theta_irt'].get(uid_key, None)
                    if theta_traj is not None:
                        actual_len = min(len(theta_traj), seq_len)
                        if isinstance(theta_traj, (list, np.ndarray)):
                            theta_batch[i, :actual_len] = torch.tensor(theta_traj[:actual_len])
                        else:
                            theta_batch[i, :actual_len] = theta_traj[:actual_len]
                batch_targets['theta_irt'] = theta_batch.to(device)  # [B, L]
            else:
                # Static theta: {uid: scalar}
                theta_values = []
                for uid in uids:
                    uid_key = torch.tensor(uid).item() if isinstance(uid, torch.Tensor) else uid
                    theta_val = ref_targets['theta_irt'].get(uid_key, 0.0)
                    theta_values.append(theta_val)
                batch_targets['theta_irt'] = torch.tensor(theta_values, dtype=torch.float32, device=device)  # [B]
        else:
            # No uids available, use zeros
            batch_size = batch['cseqs'].size(0)
            if is_dynamic:
                seq_len = batch['cseqs'].size(1)
                batch_targets['theta_irt'] = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
            else:
                batch_targets['theta_irt'] = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    # Reference predictions - extract for current batch (or zeros if not available)
    if 'm_ref' in ref_targets:
        uids = batch.get('uids', None)
        if uids is not None and len(ref_targets['m_ref']) > 0:
            # Extract m_ref sequences for students in this batch
            batch_size, seq_len = batch['cseqs'].shape
            m_ref_batch = torch.zeros(batch_size, seq_len, dtype=torch.float32)
            
            for i, uid in enumerate(uids):
                uid_tensor = torch.tensor(uid) if not isinstance(uid, torch.Tensor) else uid
                m_ref_seq = ref_targets['m_ref'].get(uid_tensor.item(), None)
                if m_ref_seq is not None:
                    # Convert to tensor if it's a list
                    if isinstance(m_ref_seq, list):
                        m_ref_seq = torch.tensor(m_ref_seq, dtype=torch.float32)
                    # Pad or truncate to match seq_len
                    actual_len = min(len(m_ref_seq), seq_len)
                    m_ref_batch[i, :actual_len] = m_ref_seq[:actual_len]
                # else: leave as zeros (student not in reference predictions)
            
            batch_targets['m_ref'] = m_ref_batch.to(device)
        else:
            # No reference predictions available, use zeros
            batch_size, seq_len = batch['cseqs'].shape
            batch_targets['m_ref'] = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
    
    return batch_targets


def get_lambda_interp(epoch, lambda_target, warmup_epochs):
    """
    Compute λ interpolation weight for warm-up schedule.
    
    λ(t) = λ_target × min(1, epoch / warmup_epochs)
    
    Args:
        epoch: Current epoch (1-indexed)
        lambda_target: Target λ value after warm-up
        warmup_epochs: Number of epochs for warm-up
    
    Returns:
        float: Current λ value in [0, lambda_target]
    """
    return lambda_target * min(1.0, epoch / warmup_epochs)


def train_epoch(model, train_loader, optimizer, device, gradient_clip, ref_targets, lambda_interp):
    """
    Train for one epoch with reference model alignment.
    
    Args:
        model: iKT3 model instance
        train_loader: training data loader
        optimizer: optimizer instance
        device: torch device
        gradient_clip: gradient clipping threshold (0 to disable)
        ref_targets: Reference model targets (from model.load_reference_targets())
        lambda_interp: Current λ value from warm-up schedule
    
    Note:
        All parameters REQUIRED. No defaults per reproducibility guidelines.
    """
    model.train()
    
    total_loss = 0.0
    total_bce = 0.0
    total_stability = 0.0
    total_align = 0.0
    num_batches = 0
    
    # Dynamic loss tracking (depends on reference model)
    reference_model = get_model_attr(model, 'reference_model')
    loss_names = reference_model.get_loss_names()
    loss_accumulators = {name: 0.0 for name in loss_names}
    
    all_preds = []
    all_labels = []
    
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
        
        # Prepare batch-specific reference targets
        batch_ref_targets = prepare_batch_ref_targets(batch, ref_targets, device)
        
        # Compute loss with reference model alignment
        compute_loss_fn = get_model_attr(model, 'compute_loss')
        loss_dict = compute_loss_fn(outputs, targets, batch_ref_targets, lambda_interp)
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_bce += loss_dict['l_bce'].item()
        total_stability += loss_dict['l_stability'].item()
        total_align += loss_dict['l_align_total'].item()
        num_batches += 1
        
        # Accumulate reference model-specific losses
        for loss_name in loss_names:
            if loss_name in loss_dict:
                loss_accumulators[loss_name] += loss_dict[loss_name].item()
        
        # Collect predictions for AUC/accuracy
        preds = outputs['bce_predictions'].detach().cpu().numpy()
        labels_np = targets.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        for i in range(len(preds)):
            valid_indices = mask_np[i] == 1
            all_preds.extend(preds[i][valid_indices])
            all_labels.extend(labels_np[i][valid_indices])
    
    # Compute aggregate metrics
    avg_total_loss = total_loss / num_batches
    avg_bce = total_bce / num_batches
    avg_stability = total_stability / num_batches
    avg_align = total_align / num_batches
    
    # Average reference model-specific losses
    avg_ref_losses = {name: loss_accumulators[name] / num_batches for name in loss_names}
    
    # Compute AUC and accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    
    return {
        'total_loss': avg_total_loss,
        'l_bce': avg_bce,
        'l_stability': avg_stability,
        'l_align_total': avg_align,
        'auc': bce_metrics['auc'],
        'accuracy': bce_metrics['acc'],
        **avg_ref_losses  # Dynamic losses from reference model
    }


def validate(model, val_loader, device, ref_targets, lambda_interp):
    """
    Validate the model with reference model alignment.
    
    Args:
        model: iKT3 model instance
        val_loader: validation data loader
        device: torch device
        ref_targets: Reference model targets
        lambda_interp: Current λ value from warm-up schedule
    
    Note:
        All parameters REQUIRED. No defaults per reproducibility guidelines.
    """
    model.eval()
    
    total_loss = 0.0
    total_bce = 0.0
    total_stability = 0.0
    total_align = 0.0
    num_batches = 0
    
    # Dynamic loss tracking
    reference_model = get_model_attr(model, 'reference_model')
    loss_names = reference_model.get_loss_names()
    loss_accumulators = {name: 0.0 for name in loss_names}
    
    all_preds = []
    all_labels = []
    
    # Collect interpretable factors for validation
    all_factors = {key: [] for key in reference_model.get_interpretable_factors({}).keys()}
    
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
            
            # Prepare batch-specific reference targets
            batch_ref_targets = prepare_batch_ref_targets(batch, ref_targets, device)
            
            # Compute loss
            compute_loss_fn = get_model_attr(model, 'compute_loss')
            loss_dict = compute_loss_fn(outputs, targets, batch_ref_targets, lambda_interp)
            
            loss = loss_dict['total_loss']
            
            # Accumulate losses
            total_loss += loss.item()
            total_bce += loss_dict['l_bce'].item()
            total_stability += loss_dict['l_stability'].item()
            total_align += loss_dict['l_align_total'].item()
            num_batches += 1
            
            # Accumulate reference model-specific losses
            for loss_name in loss_names:
                if loss_name in loss_dict:
                    loss_accumulators[loss_name] += loss_dict[loss_name].item()
            
            # Collect predictions for AUC/accuracy
            preds = outputs['bce_predictions'].detach().cpu().numpy()
            labels_np = targets.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for i in range(len(preds)):
                valid_indices = mask_np[i] == 1
                all_preds.extend(preds[i][valid_indices])
                all_labels.extend(labels_np[i][valid_indices])
            
            # Collect interpretable factors
            factors = reference_model.get_interpretable_factors(outputs)
            for key, values in factors.items():
                if values is not None:
                    values_np = values.detach().cpu().numpy()
                    for i in range(len(values_np)):
                        valid_indices = mask_np[i] == 1
                        all_factors[key].extend(values_np[i][valid_indices])
    
    # Compute aggregate metrics
    avg_total_loss = total_loss / num_batches
    avg_bce = total_bce / num_batches
    avg_stability = total_stability / num_batches
    avg_align = total_align / num_batches
    
    # Average reference model-specific losses
    avg_ref_losses = {name: loss_accumulators[name] / num_batches for name in loss_names}
    
    # Compute AUC and accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    
    # Compute statistics for interpretable factors
    factor_stats = {}
    for key, values in all_factors.items():
        if len(values) > 0:
            values_arr = np.array(values)
            factor_stats[f'{key}_mean'] = float(np.mean(values_arr))
            factor_stats[f'{key}_std'] = float(np.std(values_arr))
    
    return {
        'total_loss': avg_total_loss,
        'l_bce': avg_bce,
        'l_stability': avg_stability,
        'l_align_total': avg_align,
        'auc': bce_metrics['auc'],
        'accuracy': bce_metrics['acc'],
        **avg_ref_losses,
        **factor_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Train iKT3 model')
    
    # EXPLICIT PARAMETERS REQUIRED (No defaults - violations fail fast)
    # All defaults must come from configs/parameter_default.json
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--reference_targets_path', type=str, required=True, help='Path to reference targets file')
    parser.add_argument('--reference_model', type=str, required=True, help='Reference model type (irt, bkt, etc.)')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer type')
    parser.add_argument('--gradient_clip', type=float, required=True, help='Gradient clipping threshold')
    parser.add_argument('--patience', type=int, required=True, help='Early stopping patience')
    
    # Model architecture
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length')
    parser.add_argument('--d_model', type=int, required=True, help='Model dimension')
    parser.add_argument('--n_heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--num_encoder_blocks', type=int, required=True, help='Number of encoder blocks')
    parser.add_argument('--d_ff', type=int, required=True, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate')
    parser.add_argument('--emb_type', type=str, required=True, help='Embedding type')
    
    # iKT3-specific hyperparameters
    parser.add_argument('--lambda_target', type=float, required=True, help='Target λ for alignment after warm-up')
    parser.add_argument('--warmup_epochs', type=int, required=True, help='Number of epochs for λ warm-up')
    parser.add_argument('--c_stability_reg', type=float, required=True, help='Always-on stability regularization weight')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Get experiment directory from environment (set by run_repro_experiment.py)
    experiment_dir = os.environ.get('EXPERIMENT_DIR')
    if not experiment_dir:
        print("⚠️  WARNING: EXPERIMENT_DIR not set. Using current directory.")
        experiment_dir = '.'
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load parameter defaults to build config structure with defaults/overrides
    project_root = '/workspaces/pykt-toolkit'
    defaults_path = os.path.join(project_root, 'configs/parameter_default.json')
    with open(defaults_path, 'r') as f:
        param_defaults = json.load(f)
    
    # Build config with defaults and overrides structure (for consistency with other scripts)
    args_dict = vars(args)
    defaults_for_model = param_defaults.get('defaults', {})
    
    # Compute overrides: parameters that differ from defaults
    overrides = {}
    for key, value in args_dict.items():
        if key in defaults_for_model and value != defaults_for_model.get(key):
            overrides[key] = value
    
    # Extract experiment ID from directory name (format: YYYYMMDD_HHMMSS_model_shorttitle_expid)
    experiment_dir_name = os.path.basename(experiment_dir)
    parts = experiment_dir_name.split('_')
    experiment_id = parts[-1] if len(parts) > 4 else 'unknown'
    short_title = '_'.join(parts[2:-1]) if len(parts) > 4 else 'unknown'
    
    # Build commands (for reproducibility)
    python_path = sys.executable
    train_script = 'examples/train_ikt3.py'
    eval_script = 'examples/eval_ikt3.py'
    
    # Build explicit train command
    train_cmd_parts = [python_path, train_script]
    for key, value in sorted(args_dict.items()):
        if isinstance(value, bool):
            if value:
                train_cmd_parts.append(f"--{key}")
        else:
            train_cmd_parts.append(f"--{key} {value}")
    train_command = " ".join(train_cmd_parts)
    
    # Build eval command
    eval_command = f"{python_path} {eval_script} --experiment_dir {experiment_dir}"
    
    # Build mastery states command
    mastery_states_command = f"{python_path} examples/mastery_states.py --run_dir {experiment_dir} --split test --num_students 15"
    
    # Save config in full structured format (matching run_repro_experiment.py format)
    config = {
        "commands": {
            "train_explicit": train_command,
            "eval_explicit": eval_command,
            "mastery_states": mastery_states_command
        },
        "experiment": {
            "id": experiment_dir_name,
            "short_title": short_title,
            "experiment_id": experiment_id,
            "created": parts[0] + "_" + parts[1] if len(parts) > 1 else "unknown"
        },
        "seeds": {
            "primary": args.seed,
            "all": [args.seed]
        },
        "defaults": defaults_for_model,  # Pristine copy from parameter_default.json
        "overrides": overrides,  # Only parameters that differ from defaults
        "types": param_defaults.get('types', {}),
        "md5": param_defaults.get('md5', ''),  # MD5 of original defaults
        "reference": {
            "parameter_default_json": "configs/parameter_default.json"
        }
    }
    
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 80)
    print("iKT3 TRAINING")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Seed: {args.seed}")
    print(f"Reference Model: {args.reference_model}")
    print(f"Lambda Target: {args.lambda_target}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Experiment directory: {experiment_dir}")
    print()
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    
    # Load data config
    project_root = '/workspaces/pykt-toolkit'
    with open(os.path.join(project_root, 'configs/data_config.json')) as f:
        data_config = json.load(f)
    
    # Fix relative paths
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name] and data_config[dataset_name]['dpath'].startswith('../'):
            data_config[dataset_name]['dpath'] = os.path.join(project_root, data_config[dataset_name]['dpath'][3:])
    
    train_loader, valid_loader = init_dataset4train(
        args.dataset,
        'ikt3',  # model_name
        data_config,
        args.fold,
        args.batch_size
    )
    
    num_c = data_config[args.dataset]['num_c']
    print(f"✓ Loaded dataset with {num_c} skills")
    print(f"  Train: {len(train_loader.dataset)} students")
    print(f"  Valid: {len(valid_loader.dataset)} students")
    print()
    
    # Create model
    print("Creating iKT3 model...")
    model_config = {
        'num_c': num_c,
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'num_encoder_blocks': args.num_encoder_blocks,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'emb_type': args.emb_type,
        'reference_model_type': args.reference_model,
        'lambda_target': args.lambda_target,
        'warmup_epochs': args.warmup_epochs,
        'c_stability_reg': args.c_stability_reg
    }
    
    model = iKT3(**model_config).to(device)
    
    # Load reference targets
    print(f"Loading reference targets from {args.reference_targets_path}...")
    ref_targets = model.load_reference_targets(args.reference_targets_path)
    print(f"✓ Loaded reference targets")
    print()
    
    # Initialize from reference (e.g., skill difficulties from IRT)
    print("Initializing model from reference targets...")
    model.initialize_from_reference(ref_targets)
    print()
    
    # Setup optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    print(f"Optimizer: {args.optimizer}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Weight Decay: {args.weight_decay}")
    print()
    
    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_auc = 0.0
    patience_counter = 0
    
    # CSV logging
    csv_path = os.path.join(experiment_dir, 'metrics_epoch.csv')
    csv_headers = None
    
    for epoch in range(1, args.epochs + 1):
        # Compute λ for warm-up schedule
        lambda_interp = get_lambda_interp(epoch, args.lambda_target, args.warmup_epochs)
        
        print(f"\nEpoch {epoch}/{args.epochs} (λ={lambda_interp:.4f})")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.gradient_clip, ref_targets, lambda_interp
        )
        
        # Validate
        val_metrics = validate(
            model, valid_loader, device, ref_targets, lambda_interp
        )
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['total_loss']:.4f}, AUC: {train_metrics['auc']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Valid - Loss: {val_metrics['total_loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  BCE: {val_metrics['l_bce']:.4f}, Stability: {val_metrics['l_stability']:.4f}, Align: {val_metrics['l_align_total']:.4f}")
        
        # Print reference model-specific losses
        loss_names = model.reference_model.get_loss_names()
        if loss_names:
            loss_str = ", ".join([f"{name}: {val_metrics.get(name, 0.0):.4f}" for name in loss_names])
            print(f"  Reference: {loss_str}")
        
        # Save metrics to CSV
        row_dict = {
            'epoch': epoch,
            'lambda_interp': lambda_interp,
            'train_loss': train_metrics['total_loss'],
            'train_auc': train_metrics['auc'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['total_loss'],
            'val_auc': val_metrics['auc'],
            'val_acc': val_metrics['accuracy'],
            'val_l_bce': val_metrics['l_bce'],
            'val_l_stability': val_metrics['l_stability'],
            'val_l_align_total': val_metrics['l_align_total']
        }
        
        # Add reference model-specific losses
        for loss_name in loss_names:
            row_dict[f'val_{loss_name}'] = val_metrics.get(loss_name, 0.0)
        
        # Add factor statistics
        for key, value in val_metrics.items():
            if key.endswith('_mean') or key.endswith('_std'):
                row_dict[f'val_{key}'] = value
        
        # Write CSV
        if csv_headers is None:
            csv_headers = list(row_dict.keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(row_dict)
        
        # Early stopping check
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(experiment_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': model_config
            }, checkpoint_path)
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered (patience: {args.patience})")
                break
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {experiment_dir}")
    
    # Save final validation metrics summary
    try:
        import csv as csv_module
        from datetime import datetime
        
        # Get final validation metrics from the best epoch
        best_epoch_idx = metrics_df['val_auc'].idxmax() if 'metrics_df' in locals() else len(csv_headers) - 1
        final_metrics_path = os.path.join(experiment_dir, 'metrics_valid.csv')
        
        # Read the last/best epoch metrics from metrics_epoch.csv
        import pandas as pd
        epoch_df = pd.read_csv(csv_path)
        best_row = epoch_df.loc[epoch_df['val_auc'].idxmax()]
        
        fieldnames = ['split', 'auc', 'acc', 'l_bce', 'l_stability', 'l_align_total']
        row_data = {
            'split': 'validation',
            'auc': f"{best_row['val_auc']:.6f}",
            'acc': f"{best_row['val_acc']:.6f}",
            'l_bce': f"{best_row['val_l_bce']:.6f}",
            'l_stability': f"{best_row['val_l_stability']:.6f}",
            'l_align_total': f"{best_row['val_l_align_total']:.6f}"
        }
        
        # Add reference-specific losses if present
        if 'val_l_21_performance' in best_row:
            fieldnames.extend(['l_21_performance', 'l_22_difficulty', 'l_23_ability'])
            row_data['l_21_performance'] = f"{best_row['val_l_21_performance']:.6f}"
            row_data['l_22_difficulty'] = f"{best_row['val_l_22_difficulty']:.6f}"
            row_data['l_23_ability'] = f"{best_row['val_l_23_ability']:.6f}"
        
        with open(final_metrics_path, 'w', newline='') as cf:
            writer = csv_module.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_data)
        print(f"✓ Saved validation metrics to {final_metrics_path}")
    except Exception as e:
        print(f"⚠️  Warning: could not write metrics_valid.csv ({e})")


if __name__ == '__main__':
    main()
