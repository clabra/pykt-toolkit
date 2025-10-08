#!/usr/bin/env python3
"""
Training-Time Interpretability Monitoring for GainAKT2

This script demonstrates how to train the GainAKT2 model with real-time
interpretability monitoring during the training process.

Key Features:
1. Uses GainAKT2Monitored model with interpretability hooks
2. Tracks 4 interpretability constraints during training
3. Combines standard loss with auxiliary interpretability losses
4. Logs interpretability metrics alongside performance metrics
5. Supports multi-GPU training with monitoring on all devices
"""

import os
import sys
import torch
import torch.nn as nn
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model
from examples.interpretability_monitor import InterpretabilityMonitor


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_interpretability.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_model_config():
    """Create the optimal model configuration based on previous experiments."""
    return {
        'num_c': 100,  # Will be updated based on dataset
        'seq_len': 200,
        'd_model': 256,
        'n_heads': 8, 
        'num_encoder_blocks': 4,
        'd_ff': 768,
        'dropout': 0.2,
        'emb_type': 'qid',
        'non_negative_loss_weight': 0.1,  # Weight for non-negative gains constraint
        'monotonicity_loss_weight': 0.1, # Weight for mastery monotonicity
        'mastery_performance_loss_weight': 0.1, # Weight for mastery-performance correlation
        'gain_performance_loss_weight': 0.1, # Weight for gain-performance correlation
        'sparsity_loss_weight': 0.1, # Weight for gain sparsity
        'consistency_loss_weight': 0.0,  # Weight for consistency constraint (deactivated)
        'monitor_frequency': 25  # Monitor every 25 batches during training
    }


def train_epoch_with_monitoring(model, dataloader, optimizer, criterion, device, epoch, logger, use_wandb=False):
    """
    Train one epoch with interpretability monitoring.
    
    Args:
        model: GainAKT2Monitored model instance
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        criterion: Loss function (BCELoss)
        device: Training device
        epoch: Current epoch number
        logger: Logger instance
        use_wandb: Whether wandb logging is available
    
    Returns:
        dict: Training metrics including interpretability losses
    """
    model.train()
    total_loss = 0.0
    total_interpretability_loss = 0.0
    total_predictions = []
    total_targets = []
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        questions = batch['qseqs'].to(device)
        responses = batch['rseqs'].to(device)
        questions_shifted = batch['shft_qseqs'].to(device)
        responses_shifted = batch['shft_rseqs'].to(device)
        mask = batch['masks'].to(device)
        
        batch_size, seq_len = questions.shape
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with monitoring
        outputs = model.forward_with_states(
            q=questions, 
            r=responses, 
            qry=questions_shifted,
            batch_idx=batch_idx
        )
        
        predictions = outputs['predictions']
        interpretability_loss = outputs['interpretability_loss']
        
        # Apply mask to get valid predictions and targets
        valid_mask = mask.bool()
        valid_predictions = predictions[valid_mask]
        valid_targets = responses_shifted[valid_mask].float()
        
        # Compute main BCE loss
        main_loss = criterion(valid_predictions, valid_targets)
        
        # Total loss combines main loss and interpretability constraints
        total_batch_loss = main_loss + interpretability_loss
        
        # Backward pass
        total_batch_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate losses and predictions
        total_loss += main_loss.item()
        total_interpretability_loss += interpretability_loss.item()
        
        # Collect predictions for metrics
        with torch.no_grad():
            total_predictions.extend(valid_predictions.cpu().numpy())
            total_targets.extend(valid_targets.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'Main Loss': f'{main_loss.item():.4f}',
            'Interp Loss': f'{interpretability_loss.item():.4f}',
            'Total Loss': f'{total_batch_loss.item():.4f}'
        })
        
        # Log to wandb periodically if available
        if batch_idx % 100 == 0 and use_wandb:
            try:
                wandb.log({
                    'batch_main_loss': main_loss.item(),
                    'batch_interpretability_loss': interpretability_loss.item(),
                    'batch_total_loss': total_batch_loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })
            except Exception:
                pass  # Skip if wandb logging fails
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_interpretability_loss = total_interpretability_loss / len(dataloader)
    
    # Compute AUC and accuracy
    train_auc = roc_auc_score(total_targets, total_predictions)
    train_acc = accuracy_score(total_targets, np.array(total_predictions) > 0.5)
    
    metrics = {
        'train_loss': avg_loss,
        'train_interpretability_loss': avg_interpretability_loss,
        'train_auc': train_auc,
        'train_accuracy': train_acc
    }
    
    logger.info(f'Epoch {epoch+1} Training - Loss: {avg_loss:.4f}, '
               f'Interp Loss: {avg_interpretability_loss:.4f}, '
               f'AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}')
    
    return metrics


def evaluate_model(model, dataloader, criterion, device, logger):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: Trained model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Evaluation device
        logger: Logger instance
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_predictions = []
    total_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move batch to device
            questions = batch['qseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_qseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Forward pass (standard forward for evaluation)
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            predictions = outputs['predictions']
            
            # Apply mask
            valid_mask = mask.bool()
            valid_predictions = predictions[valid_mask]
            valid_targets = responses_shifted[valid_mask].float()
            
            # Compute loss
            loss = criterion(valid_predictions, valid_targets)
            total_loss += loss.item()
            
            # Collect predictions
            total_predictions.extend(valid_predictions.cpu().numpy())
            total_targets.extend(valid_targets.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(total_targets, total_predictions)
    acc = accuracy_score(total_targets, np.array(total_predictions) > 0.5)
    
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': acc
    }
    
    logger.info(f'Evaluation - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}')
    
    return metrics


def main():
    """Main training function with interpretability monitoring."""
    
    # Setup
    logger = setup_logging()
    logger.info("Starting GainAKT2 training with interpretability monitoring")
    
    # Initialize wandb with timeout and error handling
    use_wandb = True
    try:
        wandb.init(
            project="pykt-gainakt2-interpretability",
            name="gainakt2_monitored_training",
            config={
                "model": "GainAKT2Monitored",
                "dataset": "assist2015",
                "interpretability_monitoring": True,
                "auxiliary_loss": True
            },
            settings=wandb.Settings(init_timeout=60)  # Reduce timeout to 60 seconds
        )
        logger.info("Wandb initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        logger.info("Continuing training without wandb logging...")
        use_wandb = False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset configuration
    dataset_name = "assist2015"
    model_name = "gainakt2"
    fold = 0
    batch_size = 32
    
    # Load data config from JSON file
    import json
    with open("configs/data_config.json") as f:
        data_config = json.load(f)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_loader, valid_loader = init_dataset4train(
        dataset_name, model_name, data_config, fold, batch_size
    )
    
    # Get number of concepts and questions from data config
    num_c = data_config[dataset_name]['num_c']
    num_q = data_config[dataset_name]['num_q']
    logger.info(f"Dataset loaded - num_concepts: {num_c}, num_questions: {num_q}")
    
    # Create model configuration
    model_config = create_model_config()
    model_config['num_c'] = num_c  # Update with actual number of concepts
    
    # Create monitored model
    logger.info("Creating GainAKT2Monitored model...")
    model = create_monitored_model(model_config)
    
    # Create and set interpretability monitor
    monitor = InterpretabilityMonitor(model, log_frequency=model_config['monitor_frequency'])
    model.set_monitor(monitor)
    
    # Move model to device
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    num_epochs = 15
    best_auc = 0.0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Train epoch with monitoring
        train_metrics = train_epoch_with_monitoring(
            model, train_loader, optimizer, criterion, device, epoch, logger, use_wandb
        )
        
        # Evaluate on validation set
        valid_metrics = evaluate_model(model, valid_loader, criterion, device, logger)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics to wandb if available
        if use_wandb:
            try:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['train_loss'],
                    'train_interpretability_loss': train_metrics['train_interpretability_loss'],
                    'train_auc': train_metrics['train_auc'],
                    'train_accuracy': train_metrics['train_accuracy'],
                    'valid_loss': valid_metrics['loss'],
                    'valid_auc': valid_metrics['auc'],
                    'valid_accuracy': valid_metrics['accuracy'],
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            except Exception:
                pass  # Skip if wandb logging fails
        
        # Save best model
        if valid_metrics['auc'] > best_auc:
            best_auc = valid_metrics['auc']
            
            # Create save directory
            save_dir = f"saved_model/gainakt2_monitored_best_auc_{best_auc:.4f}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(save_dir, "model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_auc': best_auc,
                'model_config': model_config
            }, model_path)
            
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
    
    logger.info(f"Training completed! Best validation AUC: {best_auc:.4f}")
    
    # Final evaluation summary
    logger.info("="*50)
    logger.info("TRAINING SUMMARY WITH INTERPRETABILITY MONITORING")
    logger.info("="*50)
    logger.info(f"Best Validation AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: saved_model/gainakt2_monitored_best_auc_{best_auc:.4f}/")
    logger.info("Interpretability metrics were monitored throughout training")
    logger.info("Check wandb dashboard for detailed interpretability analysis")
    
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
