#!/usr/bin/env python3
"""
Training script for GainAKT4 model using PyKT framework patterns.


                         ⚠️  REPRODUCIBILITY WARNING ⚠️                        ║
#

                                                                              ║
  DO NOT CALL THIS SCRIPT DIRECTLY FOR REPRODUCIBLE EXPERIMENTS!             ║
                                                                              ║
  Use the experiment launcher:                                               ║
      python examples/run_repro_experiment.py --short_title "name"           ║
                                                                              ║
  The launcher will:                                                         ║
    ✓ Load defaults from configs/parameter_default.json                      ║
    ✓ Apply your CLI overrides                                               ║
    ✓ Generate explicit command with ALL parameters                          ║
    ✓ Create experiment folder with full audit trail                         ║
 Save config.json for perfect reproducibility                           ║    
                                                                              ║
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
from pykt.models.gainakt4 import GainAKT4
from examples.experiment_utils import compute_auc_acc


def get_model_attr(model, attr_name):
    """Safely get model attribute whether wrapped in DataParallel or not."""
    if isinstance(model, torch.nn.DataParallel):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)


def train_epoch(model, train_loader, optimizer, device, gradient_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_bce_loss = 0.0
    total_mastery_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        questions = batch['cseqs'].to(device)
        responses = batch['rseqs'].to(device)
        questions_shifted = batch['shft_cseqs'].to(device)
        mask = batch['masks'].to(device)
        
        optimizer.zero_grad()
        
        # Use per-skill practice counts (how many times each skill practiced)
        attempts = batch['practice_counts'].to(device) if 'practice_counts' in batch else torch.cumsum(responses, dim=1)
        # Targets: shifted practice counts (predict practice intensity from interaction patterns)
        attempts_targets = batch['shft_practice_counts'].to(device).float() if 'shft_practice_counts' in batch else torch.cumsum(batch['shft_rseqs'].to(device), dim=1).float()
        
        # Forward pass
        outputs = model(q=questions, r=responses, qry=questions_shifted, attempts=attempts)
        
        # Get targets
        targets = batch['shft_rseqs'].to(device)
        
        # Compute loss (access through module if DataParallel)
        compute_loss_fn = get_model_attr(model, 'compute_loss')
        loss_dict = compute_loss_fn(outputs, targets, attempts_targets=attempts_targets)
        loss = loss_dict['total_loss']
        bce_loss = loss_dict['bce_loss']
        mastery_loss = loss_dict['mastery_loss']
        
        # Backward pass
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_mastery_loss += mastery_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'bce_loss': total_bce_loss / num_batches,
        'mastery_loss': total_mastery_loss / num_batches
    }


def validate(model, val_loader, device, lambda_bce=0.7):
    """Validate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_mastery_preds = []
    all_total_preds = []
    all_curve_preds = []
    all_curve_targets = []
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_mastery_loss = 0.0
    total_curve_loss = 0.0
    num_batches = 0
    
    has_mastery = False  # Track if mastery predictions exist
    has_curve = False  # Track if curve predictions exist
    
    with torch.no_grad():
        for batch in val_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            mask = batch['masks'].to(device)
            labels = batch['shft_rseqs'].to(device)
            
            # Use per-skill practice counts (how many times each skill practiced)
            attempts = batch['practice_counts'].to(device) if 'practice_counts' in batch else torch.cumsum(responses, dim=1)
            # Targets: shifted practice counts (predict practice intensity from interaction patterns)
            attempts_targets = batch['shft_practice_counts'].to(device).float() if 'shft_practice_counts' in batch else torch.cumsum(batch['shft_rseqs'].to(device), dim=1).float()
            
            # Forward pass
            outputs = model(q=questions, r=responses, qry=questions_shifted, attempts=attempts)
            
            # Compute loss (access through module if DataParallel)
            compute_loss_fn = get_model_attr(model, 'compute_loss')
            loss_dict = compute_loss_fn(outputs, labels, attempts_targets=attempts_targets)
            loss = loss_dict['total_loss']
            total_loss += loss.item()
            total_bce_loss += loss_dict['bce_loss'].item()
            total_mastery_loss += loss_dict['mastery_loss'].item()
            total_curve_loss += loss_dict['curve_loss'].item()
            num_batches += 1
            
            # Collect predictions
            preds = outputs['bce_predictions'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Check for curve predictions
            if outputs['curve_predictions'] is not None:
                has_curve = True
                curve_preds = outputs['curve_predictions'].cpu().numpy()
            else:
                curve_preds = None
            
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
                if curve_preds is not None:
                    all_curve_preds.extend(curve_preds[i][valid_indices])
                    # For curve metrics, use cumsum of responses as proxy target
                    # (in real training, this should come from attempts-to-mastery data)
                    cumsum_responses = np.cumsum(labels_np[i][:np.sum(valid_indices)])
                    all_curve_targets.extend(cumsum_responses)
    
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
    
    # Compute curve metrics if curve predictions exist
    if has_curve:
        all_curve_preds = np.array(all_curve_preds)
        all_curve_targets = np.array(all_curve_targets)
        # For curve: treat as regression
        curve_mae = np.mean(np.abs(all_curve_preds - all_curve_targets))
        curve_rmse = np.sqrt(np.mean((all_curve_preds - all_curve_targets) ** 2))
        curve_within_1 = np.mean(np.abs(all_curve_preds - all_curve_targets) <= 1.0)
        # Compute R² score (coefficient of determination) for regression
        if len(np.unique(all_curve_targets)) > 1:
            try:
                from sklearn.metrics import r2_score
                curve_r2 = r2_score(all_curve_targets, all_curve_preds)
            except:
                curve_r2 = 'N/A'
        else:
            curve_r2 = 'N/A'
        curve_metrics = {'r2': curve_r2, 'acc': curve_within_1, 'mae': curve_mae, 'rmse': curve_rmse}
    else:
        curve_metrics = {'r2': 'N/A', 'acc': 'N/A', 'mae': 'N/A', 'rmse': 'N/A'}
    
    return {
        'loss': total_loss / num_batches,
        'bce_loss': total_bce_loss / num_batches,
        'mastery_loss': total_mastery_loss / num_batches,
        'curve_loss': total_curve_loss / num_batches,
        'bce_auc': bce_metrics['auc'],
        'bce_acc': bce_metrics['acc'],
        'mastery_auc': mastery_metrics['auc'],
        'mastery_acc': mastery_metrics['acc'],
        'curve_r2': curve_metrics['r2'],
        'curve_acc': curve_metrics['acc'],
        'curve_mae': curve_metrics['mae'],
        'curve_rmse': curve_metrics['rmse'],
        'total_auc': total_metrics['auc'],
        'total_acc': total_metrics['acc']
    }


def main():
    parser = argparse.ArgumentParser(description='Train GainAKT4 model')
    
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
    
    # Loss weights
    parser.add_argument('--lambda_bce', type=float, required=True)
    parser.add_argument('--lambda_mastery', type=float, required=True,
                       help='Mastery loss weight (if None, computed as 1.0 - lambda_bce - lambda_curve)')
    parser.add_argument('--lambda_curve', type=float, required=True, 
                       help='Curve loss weight (default: 0.0 for backward compatibility)')
    parser.add_argument('--max_attempts', type=int, required=True)
    
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
                'epoch', 'lambda_bce', 'lambda_mastery', 'lambda_curve',
                'val_total_auc', 'val_bce_auc', 'val_mastery_auc', 'val_curve_r2',
                'val_total_acc', 'val_bce_acc', 'val_mastery_acc', 'val_curve_acc',
                'val_total_loss', 'val_bce_loss', 'val_mastery_loss', 'val_curve_loss',
                'val_curve_mae', 'val_curve_rmse'
            ])
    
    print("="*80)
    print("GainAKT4 Training")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    # Compute lambda_mastery if not provided
    if args.lambda_mastery is None:
        args.lambda_mastery = 1.0 - args.lambda_bce - args.lambda_curve
    
    print(f"Lambda BCE: {args.lambda_bce}")
    print(f"Lambda Mastery: {args.lambda_mastery}")
    print(f"Lambda Curve: {args.lambda_curve}")
    print(f"Lambda sum: {args.lambda_bce + args.lambda_mastery + args.lambda_curve}")
    print(f"Max Attempts: {args.max_attempts}")
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
    
    model_name = "gainakt4"
    train_loader, valid_loader = init_dataset4train(
        args.dataset, model_name, data_config, args.fold, args.batch_size
    )
    
    # Get number of concepts from data config
    num_c = data_config[args.dataset]['num_c']
    
    print(f"✓ Dataset loaded: {num_c} concepts")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(valid_loader)}")
    
    # Initialize model
    print("\n🏗️  Initializing model...")
    model = GainAKT4(
        num_c=num_c,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_blocks=args.num_encoder_blocks,
        d_ff=args.d_ff,
        dropout=args.dropout,
        emb_type=args.emb_type,
        lambda_bce=args.lambda_bce,
        lambda_mastery=args.lambda_mastery,
        lambda_curve=args.lambda_curve,
        max_attempts=args.max_attempts
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
    best_val_auc = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.gradient_clip)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"BCE: {train_metrics['bce_loss']:.4f}, "
              f"Mastery: {train_metrics['mastery_loss']:.4f}")
        
        # Validate
        val_metrics = validate(model, valid_loader, device, args.lambda_bce)
        mastery_auc_str = f"{val_metrics['mastery_auc']:.4f}" if isinstance(val_metrics['mastery_auc'], float) else val_metrics['mastery_auc']
        curve_r2_str = f"{val_metrics['curve_r2']:.4f}" if isinstance(val_metrics['curve_r2'], float) else val_metrics['curve_r2']
        print(f"Valid - Loss: {val_metrics['loss']:.4f}, "
              f"Total AUC: {val_metrics['total_auc']:.4f}, "
              f"BCE AUC: {val_metrics['bce_auc']:.4f}, "
              f"Mastery AUC: {mastery_auc_str}, "
              f"Curve R²: {curve_r2_str}")
        
        # Save history
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_bce_loss': train_metrics['bce_loss'],
            'train_mastery_loss': train_metrics['mastery_loss'],
            'val_loss': val_metrics['loss'],
            'val_bce_auc': val_metrics['bce_auc'],
            'val_bce_acc': val_metrics['bce_acc'],
            'val_mastery_auc': val_metrics['mastery_auc'],
            'val_mastery_acc': val_metrics['mastery_acc']
        }
        history.append(epoch_results)
        
        # Append to metrics_epoch.csv for reproducibility
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                args.lambda_bce,
                args.lambda_mastery,
                args.lambda_curve,
                val_metrics['total_auc'],
                val_metrics['bce_auc'],
                val_metrics['mastery_auc'],
                val_metrics['curve_r2'],
                val_metrics['total_acc'],
                val_metrics['bce_acc'],
                val_metrics['mastery_acc'],
                val_metrics['curve_acc'],
                val_metrics['loss'],
                val_metrics['bce_loss'],
                val_metrics['mastery_loss'],
                val_metrics['curve_loss'],
                val_metrics['curve_mae'],
                val_metrics['curve_rmse']
            ])
        
        # Check for improvement (using total_auc as primary metric)
        if val_metrics['total_auc'] > best_val_auc:
            best_val_auc = val_metrics['total_auc']
            patience_counter = 0
            
            # Save best model (handle DataParallel wrapper)
            checkpoint_path = os.path.join(experiment_dir, 'model_best.pth')
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bce_auc': val_metrics['bce_auc'],
                'val_mastery_auc': val_metrics['mastery_auc'],
                'config': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print("\n⏹️  Early stopping triggered")
                break
    
    # Save training history
    history_path = os.path.join(experiment_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final metrics
    final_metrics = {
        'best_val_bce_auc': best_val_auc,
        'final_epoch': epoch + 1,
        'total_params': num_params
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
            'examples/eval_gainakt4.py',
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
            '--lambda_bce', str(args.lambda_bce)
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
        
        # Auto-launch mastery states analysis
        print("\n" + "="*80)
        print("LAUNCHING MASTERY STATES ANALYSIS")
        print("="*80)
        
        mastery_cmd = [
            sys.executable,
            'examples/mastery_states.py',
            '--run_dir', experiment_dir,
            '--num_students', '20',
            '--split', 'test'
        ]
        
        print(f"Mastery states command: {' '.join(mastery_cmd)}")
        
        try:
            result = subprocess.run(mastery_cmd, check=True, capture_output=True, text=True, cwd='/workspaces/pykt-toolkit')
            print("✅ Mastery states analysis completed successfully")
            # Log only summary lines
            for line in result.stdout.strip().split('\n'):
                if 'saved' in line.lower() or 'complete' in line.lower():
                    print(line)
            if result.stderr:
                print(f"⚠️  Mastery states stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Mastery states analysis failed with exit code {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
        except Exception as e:
            print(f"❌ Mastery states analysis failed with exception: {e}")
        
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
