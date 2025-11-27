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
- Phase 2: Constrained optimization (L_total = λ_bce * L1 + λ_mastery * L2, epsilon>0)
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


def load_rasch_targets(rasch_path, dataset_path, num_c, mastery_method='bkt'):
    """
    Load pre-computed mastery targets (BKT or IRT/Rasch, standard or monotonic).
    
    Args:
        rasch_path: Path to mastery targets pickle file (if None, uses default or random)
        dataset_path: Path to dataset directory
        num_c: Number of concepts/skills
        mastery_method: Method used ('bkt', 'irt', 'bkt_mono', 'irt_mono')
    
    Returns:
        dict: Dictionary with mastery targets and metadata
              - If file exists: {'rasch_targets': dict, 'student_abilities': dict, ...}
              - If not: {'mode': 'random', 'num_c': num_c}
    """
    import pickle
    
    # Determine if monotonic version requested
    is_monotonic = mastery_method.endswith('_mono')
    
    # Determine Rasch file path
    if rasch_path is None:
        rasch_path = os.path.join(dataset_path, 'rasch_targets.pkl')
    
    # Try to load from file
    if os.path.exists(rasch_path):
        print(f"✓ Loading mastery targets from: {rasch_path}")
        print(f"  Method: {mastery_method.upper()}")
        if is_monotonic:
            print(f"  Monotonic smoothing: ENABLED")
        
        try:
            with open(rasch_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle both BKT and IRT formats
            # Normalize to common 'rasch_targets' key
            if 'bkt_targets' in data:
                data['rasch_targets'] = data['bkt_targets']
                print(f"  Loaded BKT targets for {len(data['bkt_targets'])} students")
                if 'bkt_params' in data:
                    print(f"  BKT parameters available for {len(data['bkt_params'])} skills")
            elif 'rasch_targets' in data:
                print(f"  Loaded IRT/Rasch targets for {len(data['rasch_targets'])} students")
            else:
                raise ValueError("Invalid mastery file: missing 'bkt_targets' or 'rasch_targets' key")
            
            if 'metadata' in data:
                meta = data['metadata']
                print(f"  Metadata: {meta}")
                
                # Verify monotonic status if specified
                file_monotonic = meta.get('monotonic', False)
                if is_monotonic and not file_monotonic:
                    print(f"  ⚠️  WARNING: Requested monotonic version but file has monotonic={file_monotonic}")
                    print(f"             Make sure you're loading the correct file (e.g., *_mono.pkl)")
                elif not is_monotonic and file_monotonic:
                    print(f"  ⚠️  WARNING: Requested standard version but file has monotonic={file_monotonic}")
                    print(f"             Make sure you're loading the correct file (not *_mono.pkl)")
            
            return data
            
        except Exception as e:
            print(f"✗ Failed to load mastery targets: {e}")
            print("  Falling back to random initialization")
    else:
        print(f"⚠️  Mastery targets not found at: {rasch_path}")
        print("  Using random initialization in [0.0, 1.0] (placeholder)")
        print(f"  To compute real targets:")
        print(f"    BKT: python examples/compute_bkt_targets.py --dataset {{dataset}}")
        print(f"    IRT: python examples/compute_rasch_targets.py --dataset {{dataset}} --dynamic")
    
    # Fallback: Return flag for random generation
    return {'mode': 'random', 'num_c': num_c}


def train_epoch(model, train_loader, optimizer, device, gradient_clip, rasch_targets=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_bce_loss = 0.0
    total_rasch_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        questions = batch['cseqs'].to(device)
        responses = batch['rseqs'].to(device)
        questions_shifted = batch['shft_cseqs'].to(device)
        mask = batch['masks'].to(device)
        
        optimizer.zero_grad()
        
        # Prepare Rasch targets if available
        rasch_batch = None
        if rasch_targets is not None:
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
        outputs = model(q=questions, r=responses, qry=questions_shifted, rasch_targets=rasch_batch)
        
        # Get targets
        targets = batch['shft_rseqs'].to(device)
        
        # Compute loss (access through module if DataParallel)
        compute_loss_fn = get_model_attr(model, 'compute_loss')
        loss_dict = compute_loss_fn(outputs, targets)
        loss = loss_dict['total_loss']
        bce_loss = loss_dict['bce_loss']
        rasch_loss = loss_dict['rasch_loss']
        
        # Backward pass
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_rasch_loss += rasch_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'bce_loss': total_bce_loss / num_batches,
        'rasch_loss': total_rasch_loss / num_batches
    }


def validate(model, val_loader, device, lambda_bce=0.5, rasch_targets=None):
    """Validate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_rasch_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            mask = batch['masks'].to(device)
            labels = batch['shft_rseqs'].to(device)
            
            # Prepare Rasch targets if available
            rasch_batch = None
            if rasch_targets is not None:
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
            outputs = model(q=questions, r=responses, qry=questions_shifted, rasch_targets=rasch_batch)
            
            # Compute loss (access through module if DataParallel)
            compute_loss_fn = get_model_attr(model, 'compute_loss')
            loss_dict = compute_loss_fn(outputs, labels)
            loss = loss_dict['total_loss']
            total_loss += loss.item()
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
    
    bce_metrics = compute_auc_acc(all_labels, all_preds)
    
    return {
        'loss': total_loss / num_batches,
        'bce_loss': total_bce_loss / num_batches,
        'rasch_loss': total_rasch_loss / num_batches,
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
    parser.add_argument('--lambda_bce', type=float, required=True)
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
    
    # Initialize metrics_epoch.csv for reproducibility
    metrics_csv_path = os.path.join(experiment_dir, 'metrics_epoch.csv')
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'lambda_bce', 'epsilon',
                'val_bce_auc', 'val_bce_acc',
                'val_total_loss', 'val_bce_loss', 'val_rasch_loss',
                'train_total_loss', 'train_bce_loss', 'train_rasch_loss'
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
    print(f"Lambda BCE: {args.lambda_bce} (Lambda Mastery: {1.0 - args.lambda_bce})")
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
    
    # Load Rasch targets if available
    print("\n🎯 Loading Mastery targets...")
    rasch_targets = load_rasch_targets(args.rasch_path, dataset_path, num_c, args.mastery_method)
    if rasch_targets is not None:
        if rasch_targets.get('mode') == 'random':
            print(f"✓ Using random Rasch targets (placeholder)")
        else:
            rasch_data = rasch_targets.get('rasch_targets', {})
            print(f"✓ Loaded real Rasch IRT targets for {len(rasch_data)} students")
    else:
        print(f"⚠️  No Rasch targets - training in pure BCE mode (lambda_mastery will be ignored)")
    
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
        lambda_bce=args.lambda_bce,
        epsilon=args.epsilon,
        phase=args.phase
    ).to(device)
    
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
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.gradient_clip, rasch_targets)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"BCE: {train_metrics['bce_loss']:.4f}, "
              f"Rasch: {train_metrics['rasch_loss']:.4f}")
        
        # Validate
        val_metrics = validate(model, valid_loader, device, args.lambda_bce, rasch_targets)
        print(f"Valid - Loss: {val_metrics['loss']:.4f}, "
              f"AUC: {val_metrics['bce_auc']:.4f}, "
              f"ACC: {val_metrics['bce_acc']:.4f}, "
              f"Rasch: {val_metrics['rasch_loss']:.4f}")
        
        # Save history
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_bce_loss': train_metrics['bce_loss'],
            'train_rasch_loss': train_metrics['rasch_loss'],
            'val_loss': val_metrics['loss'],
            'val_bce_auc': val_metrics['bce_auc'],
            'val_bce_acc': val_metrics['bce_acc'],
            'val_rasch_loss': val_metrics['rasch_loss']
        }
        history.append(epoch_results)
        
        # Append to metrics_epoch.csv for reproducibility
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                current_phase,  # Use current_phase instead of args.phase
                args.lambda_bce,
                args.epsilon,
                val_metrics['bce_auc'],
                val_metrics['bce_acc'],
                val_metrics['loss'],
                val_metrics['bce_loss'],
                val_metrics['rasch_loss'],
                train_metrics['loss'],
                train_metrics['bce_loss'],
                train_metrics['rasch_loss']
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
                'val_rasch_loss': val_metrics['rasch_loss'],
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
                    print(f"Loss: λ_bce={args.lambda_bce} × L1 + λ_mastery={(1-args.lambda_bce):.3f} × (L2 + L3)")
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
    
    # Save training history
    history_path = os.path.join(experiment_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
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
        'lambda_bce': args.lambda_bce,
        'epsilon': args.epsilon,
        'training_mode': training_mode,
        'phase1_converged': phase1_converged if training_mode == "automatic_two_phase" else None,
        'phase1_best_epoch': phase1_best_epoch if training_mode == "automatic_two_phase" else None,
        'phase2_started_epoch': phase2_started_epoch if training_mode == "automatic_two_phase" and phase1_converged else None
    }
    metrics_path = os.path.join(experiment_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
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
            '--lambda_bce', str(args.lambda_bce),
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
