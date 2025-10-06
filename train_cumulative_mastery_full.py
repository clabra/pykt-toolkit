#!/usr/bin/env python3
"""
Full training script for GainAKT2Monitored with cumulative mastery.
This script trains the model with perfect educational consistency for longer
to achieve strong correlations while maintaining 0% violations.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import wandb

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model
from examples.interpretability_monitor import InterpretabilityMonitor


def setup_logging(log_dir="logs", experiment_name="cumulative_mastery"):
    """Setup comprehensive logging."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_file


def get_data_config():
    """Get standardized data configuration."""
    return {
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


def get_model_config(num_c, enhanced_constraints=True):
    """Get model configuration with enhanced constraints for better correlations."""
    base_config = {
        'num_c': num_c,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 8,
        'num_encoder_blocks': 6,  # Increased depth for better representation
        'd_ff': 1024,  # Increased FFN size
        'dropout': 0.2,
        'emb_type': 'qid',
        'monitor_frequency': 50
    }
    
    if enhanced_constraints:
        # Stronger constraints to encourage meaningful correlations
        # while maintaining architectural guarantees
        base_config.update({
            'non_negative_loss_weight': 0.0,  # Architectural constraint handles this
            'monotonicity_loss_weight': 0.1,  # Light regularization for smoothness
            'mastery_performance_loss_weight': 0.8,  # Strong correlation encouragement
            'gain_performance_loss_weight': 0.8,    # Strong correlation encouragement
            'sparsity_loss_weight': 0.2,
            'consistency_loss_weight': 0.3
        })
    else:
        base_config.update({
            'non_negative_loss_weight': 0.1,
            'monotonicity_loss_weight': 0.1,
            'mastery_performance_loss_weight': 0.3,
            'gain_performance_loss_weight': 0.3,
            'sparsity_loss_weight': 0.1,
            'consistency_loss_weight': 0.1
        })
    
    return base_config


def validate_model_consistency(model, data_loader, device, logger, max_students=100):
    """Quick consistency validation during training."""
    model.eval()
    
    violations = {
        'monotonicity': 0,
        'negative_gains': 0,
        'bounds': 0,
        'total_checks': 0
    }
    
    correlations = {
        'mastery_performance': [],
        'gain_performance': []
    }
    
    student_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if student_count >= max_students:
                break
                
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            outputs = model.forward_with_states(
                q=questions, r=responses, qry=questions_shifted
            )
            
            skill_mastery = outputs['projected_mastery']
            skill_gains = outputs['projected_gains']
            batch_size_actual = questions.size(0)
            
            for i in range(batch_size_actual):
                if student_count >= max_students:
                    break
                    
                student_mask = mask[i].bool()
                student_mastery = skill_mastery[i][student_mask]
                student_gains = skill_gains[i][student_mask]
                student_performance = responses_shifted[i][student_mask].float()
                
                seq_len = student_mastery.size(0)
                if seq_len < 2:
                    continue
                
                # Convert to numpy and aggregate
                mastery_np = student_mastery.cpu().numpy()
                gains_np = student_gains.cpu().numpy()
                performance_np = student_performance.cpu().numpy()
                
                mean_mastery = np.mean(mastery_np, axis=1)
                mean_gains = np.mean(gains_np, axis=1)
                
                # Check violations
                violations['total_checks'] += 1
                
                # Monotonicity
                for t in range(1, seq_len):
                    if mean_mastery[t] < mean_mastery[t-1] - 1e-6:  # Small tolerance
                        violations['monotonicity'] += 1
                        break
                
                # Negative gains
                if np.any(gains_np < -1e-6):
                    violations['negative_gains'] += 1
                
                # Bounds
                if np.any((mastery_np < -1e-6) | (mastery_np > 1 + 1e-6)):
                    violations['bounds'] += 1
                
                # Correlations
                if seq_len >= 3:
                    try:
                        mastery_corr = np.corrcoef(mean_mastery, performance_np)[0, 1]
                        if not np.isnan(mastery_corr):
                            correlations['mastery_performance'].append(mastery_corr)
                        
                        gain_corr = np.corrcoef(mean_gains, performance_np)[0, 1]
                        if not np.isnan(gain_corr):
                            correlations['gain_performance'].append(gain_corr)
                    except (ValueError, IndexError, np.linalg.LinAlgError):
                        # Skip correlation computation for degenerate cases
                        pass
                
                student_count += 1
    
    # Compute statistics
    if violations['total_checks'] > 0:
        mono_rate = violations['monotonicity'] / violations['total_checks']
        neg_rate = violations['negative_gains'] / violations['total_checks']
        bounds_rate = violations['bounds'] / violations['total_checks']
    else:
        mono_rate = neg_rate = bounds_rate = 0.0
    
    mastery_corr = np.mean(correlations['mastery_performance']) if correlations['mastery_performance'] else 0.0
    gain_corr = np.mean(correlations['gain_performance']) if correlations['gain_performance'] else 0.0
    
    logger.info(f"  Consistency Check - Monotonicity: {mono_rate:.1%}, "
                f"Negative gains: {neg_rate:.1%}, Bounds: {bounds_rate:.1%}")
    logger.info(f"  Correlations - Mastery: {mastery_corr:.3f}, Gains: {gain_corr:.3f}")
    
    return {
        'monotonicity_violation_rate': mono_rate,
        'negative_gain_rate': neg_rate,
        'bounds_violation_rate': bounds_rate,
        'mastery_correlation': mastery_corr,
        'gain_correlation': gain_corr
    }


def train_cumulative_mastery_model(args):
    """Main training function."""
    
    # Setup logging
    logger, log_file = setup_logging("logs", f"cumulative_mastery_full_{args.experiment_suffix}")
    
    logger.info("=" * 80)
    logger.info("TRAINING GAINAKT2MONITORED WITH CUMULATIVE MASTERY")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.experiment_suffix}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Enhanced constraints: {args.enhanced_constraints}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="pykt-cumulative-mastery",
            name=f"cumulative_mastery_{args.experiment_suffix}",
            config=vars(args)
        )
    
    # Load dataset
    dataset_name = args.dataset
    model_name = "gainakt2"
    data_config = get_data_config()
    
    logger.info(f"Loading dataset: {dataset_name}")
    train_loader, valid_loader = init_dataset4train(
        dataset_name, model_name, data_config, args.fold, args.batch_size
    )
    logger.info("✓ Dataset loaded successfully")
    
    # Create model with cumulative mastery
    num_c = data_config[dataset_name]['num_c']
    model_config = get_model_config(num_c, args.enhanced_constraints)
    
    logger.info("Creating GainAKT2Monitored with CUMULATIVE MASTERY...")
    model = create_monitored_model(model_config)
    monitor = InterpretabilityMonitor(model, log_frequency=args.monitor_freq)
    model.set_monitor(monitor)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model created with {total_params:,} parameters")
    logger.info("✓ Perfect consistency guaranteed by architectural constraints")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training tracking
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    train_history = {
        'train_loss': [],
        'train_auc': [],
        'val_auc': [],
        'consistency_metrics': []
    }
    
    logger.info(f"\\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        logger.info(f"\\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 50)
        
        # Training phase
        model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_interpretability_loss = 0.0
        total_predictions = []
        total_targets = []
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with interpretability monitoring
            outputs = model.forward_with_states(
                q=questions, r=responses, qry=questions_shifted, batch_idx=batch_idx
            )
            
            predictions = outputs['predictions']
            interpretability_loss = outputs['interpretability_loss']
            
            # Compute main prediction loss
            valid_mask = mask.bool()
            valid_predictions = predictions[valid_mask]
            valid_targets = responses_shifted[valid_mask].float()
            
            main_loss = criterion(valid_predictions, valid_targets)
            total_batch_loss = main_loss + interpretability_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            total_loss += total_batch_loss.item()
            total_main_loss += main_loss.item()
            total_interpretability_loss += interpretability_loss.item()
            
            with torch.no_grad():
                total_predictions.extend(valid_predictions.cpu().numpy())
                total_targets.extend(valid_targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Main': f'{main_loss.item():.4f}',
                'Constraint': f'{interpretability_loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Compute training metrics
        train_loss = total_loss / len(train_loader)
        train_main_loss = total_main_loss / len(train_loader)
        train_constraint_loss = total_interpretability_loss / len(train_loader)
        train_auc = roc_auc_score(total_targets, total_predictions)
        train_acc = accuracy_score(total_targets, np.round(total_predictions))
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='Validation'):
                questions = batch['cseqs'].to(device)
                responses = batch['rseqs'].to(device)
                questions_shifted = batch['shft_cseqs'].to(device)
                responses_shifted = batch['shft_rseqs'].to(device)
                mask = batch['masks'].to(device)
                
                outputs = model(q=questions, r=responses, qry=questions_shifted)
                predictions = outputs['predictions']
                
                valid_mask = mask.bool()
                valid_predictions = predictions[valid_mask]
                valid_targets = responses_shifted[valid_mask].float()
                
                loss = criterion(valid_predictions, valid_targets)
                val_loss += loss.item()
                
                val_predictions.extend(valid_predictions.cpu().numpy())
                val_targets.extend(valid_targets.cpu().numpy())
        
        val_loss = val_loss / len(valid_loader)
        val_auc = roc_auc_score(val_targets, val_predictions)
        val_acc = accuracy_score(val_targets, np.round(val_predictions))
        
        # Consistency validation
        logger.info("  Running consistency validation...")
        consistency_metrics = validate_model_consistency(
            model, valid_loader, device, logger, max_students=50
        )
        
        # Log epoch results
        logger.info(f"  Train - Loss: {train_loss:.4f} (Main: {train_main_loss:.4f}, "
                   f"Constraint: {train_constraint_loss:.4f}), AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Valid - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        
        # Update history
        train_history['train_loss'].append(train_loss)
        train_history['train_auc'].append(train_auc)
        train_history['val_auc'].append(val_auc)
        train_history['consistency_metrics'].append(consistency_metrics)
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_main_loss': train_main_loss,
                'train_constraint_loss': train_constraint_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **{f'consistency_{k}': v for k, v in consistency_metrics.items()}
            })
        
        # Model saving and early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            save_dir = f"saved_model/gainakt2_cumulative_mastery_{args.experiment_suffix}"
            os.makedirs(save_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'model_config': model_config,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'consistency_metrics': consistency_metrics,
                'train_history': train_history
            }
            
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            logger.info(f"  ✓ New best model saved (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"  Early stopping triggered (patience: {args.patience})")
            break
    
    # Final evaluation
    logger.info("\\n" + "=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Final comprehensive consistency check
    logger.info("\\nRunning final consistency validation...")
    final_consistency = validate_model_consistency(
        model, valid_loader, device, logger, max_students=200
    )
    
    # Save final results
    final_results = {
        'experiment_name': args.experiment_suffix,
        'best_val_auc': best_val_auc,
        'final_consistency_metrics': final_consistency,
        'train_history': train_history,
        'model_config': model_config,
        'training_args': vars(args),
        'log_file': log_file,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = f"cumulative_mastery_results_{args.experiment_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\\n📄 Final results saved to: {results_file}")
    
    if args.use_wandb:
        wandb.log({
            'final_best_val_auc': best_val_auc,
            **{f'final_consistency_{k}': v for k, v in final_consistency.items()}
        })
        wandb.finish()
    
    # Assessment
    logger.info("\\n🎯 FINAL ASSESSMENT:")
    perfect_consistency = (
        final_consistency['monotonicity_violation_rate'] == 0.0 and
        final_consistency['negative_gain_rate'] == 0.0 and
        final_consistency['bounds_violation_rate'] == 0.0
    )
    
    strong_correlations = (
        final_consistency['mastery_correlation'] > 0.3 and
        final_consistency['gain_correlation'] > 0.3
    )
    
    if perfect_consistency:
        logger.info("✅ PERFECT EDUCATIONAL CONSISTENCY MAINTAINED!")
    else:
        logger.info("⚠️  Some consistency violations detected")
    
    if strong_correlations:
        logger.info("✅ STRONG PERFORMANCE CORRELATIONS ACHIEVED!")
    else:
        logger.info("⚠️  Correlations need improvement")
    
    if perfect_consistency and strong_correlations:
        logger.info("🎉 COMPLETE SUCCESS: Perfect consistency + Strong correlations!")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Train GainAKT2Monitored with cumulative mastery')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--enhanced_constraints', type=bool, default=True, 
                       help='Use enhanced constraint weights for better correlations')
    parser.add_argument('--monitor_freq', type=int, default=50, 
                       help='Interpretability monitoring frequency')
    
    # Experiment parameters
    parser.add_argument('--dataset', type=str, default='assist2015', help='Dataset name')
    parser.add_argument('--fold', type=int, default=0, help='Data fold')
    parser.add_argument('--experiment_suffix', type=str, default='v1', 
                       help='Experiment name suffix')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Run training
    results = train_cumulative_mastery_model(args)
    
    print(f"\\n🎉 Training completed! Best Val AUC: {results['best_val_auc']:.4f}")
    print(f"📄 Results saved to: cumulative_mastery_results_{args.experiment_suffix}_*.json")


if __name__ == "__main__":
    main()