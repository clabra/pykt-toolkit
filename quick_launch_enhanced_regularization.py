#!/usr/bin/env python3
"""
Enhanced version of quick_launch_monitored.py with regularization improvements
to address overfitting observed in training results.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model
from examples.interpretability_monitor import InterpretabilityMonitor


def main():
    """Enhanced training with anti-overfitting measures."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("ENHANCED GAINAKT2 TRAINING WITH ANTI-OVERFITTING")
    logger.info("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset configuration
    dataset_name = "assist2015"
    model_name = "gainakt2"
    fold = 0
    batch_size = 32
    
    data_config = {
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
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        train_loader, valid_loader = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size
        )
        logger.info("✓ Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False
    
    # Enhanced model configuration to reduce overfitting
    num_c = data_config[dataset_name]['num_c']
    model_config = {
        'num_c': num_c,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 4,
        'num_encoder_blocks': 4,
        'd_ff': 512,
        'dropout': 0.4,                          # Increased: 0.311646 → 0.4
        'emb_type': 'qid',
        'non_negative_loss_weight': 0.485828,
        'consistency_loss_weight': 0.173548,
        'monitor_frequency': 25,
        'optimizer': 'adam'
    }
    
    # Create model
    logger.info("Creating GainAKT2Monitored model with enhanced regularization...")
    model = create_monitored_model(model_config)
    monitor = InterpretabilityMonitor(model, log_frequency=25)
    model.set_monitor(monitor)
    model = model.to(device)
    logger.info(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Enhanced training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.000103,
        weight_decay=0.001  # Increased: 0.000276 → 0.001 for stronger regularization
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)  # More aggressive decay
    
    num_epochs = 200
    best_auc = 0.0
    patience = 5  # Early stopping
    patience_counter = 0
    
    logger.info(f"Starting enhanced training for {num_epochs} epochs...")
    logger.info("Enhanced regularization: higher dropout, weight decay, early stopping")
    
    # Training loop with early stopping
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # Training
        model.train()
        total_loss = 0.0
        total_interp_loss = 0.0
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
            
            outputs = model.forward_with_states(
                q=questions, r=responses, qry=questions_shifted, batch_idx=batch_idx
            )
            
            predictions = outputs['predictions']
            interpretability_loss = outputs['interpretability_loss']
            
            valid_mask = mask.bool()
            valid_predictions = predictions[valid_mask]
            valid_targets = responses_shifted[valid_mask].float()
            
            main_loss = criterion(valid_predictions, valid_targets)
            total_batch_loss = main_loss + interpretability_loss
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tighter clipping
            optimizer.step()
            
            total_loss += main_loss.item()
            total_interp_loss += interpretability_loss.item()
            
            with torch.no_grad():
                total_predictions.extend(valid_predictions.cpu().numpy())
                total_targets.extend(valid_targets.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{main_loss.item():.4f}',
                'Interp': f'{interpretability_loss.item():.4f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
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
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        avg_interp_loss = total_interp_loss / len(train_loader)
        train_auc = roc_auc_score(total_targets, total_predictions)
        train_acc = accuracy_score(total_targets, np.array(total_predictions) > 0.5)
        
        val_loss /= len(valid_loader)
        val_auc = roc_auc_score(val_targets, val_predictions)
        val_acc = accuracy_score(val_targets, np.array(val_predictions) > 0.5)
        
        # Calculate overfitting indicator
        overfitting_gap = train_auc - val_auc
        
        scheduler.step()
        
        # Log results with overfitting analysis
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"  Training   - Loss: {avg_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Validation - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        logger.info(f"  Interpretability Loss: {avg_interp_loss:.4f}")
        logger.info(f"  Overfitting Gap: {overfitting_gap:.4f} (train-val AUC)")
        
        # Early stopping logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            
            save_dir = f"saved_model/gainakt2_enhanced_auc_{best_auc:.4f}"
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_auc': best_auc,
                'model_config': model_config,
                'train_auc': train_auc,
                'overfitting_gap': overfitting_gap
            }, os.path.join(save_dir, "model.pth"))
            
            logger.info(f"  ✓ New best model saved! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Warning for excessive overfitting
        if overfitting_gap > 0.05:
            logger.warning(f"⚠️  High overfitting detected! Gap: {overfitting_gap:.4f}")
    
    logger.info(f"\n{'='*60}")
    logger.info("ENHANCED TRAINING COMPLETED!")
    logger.info(f"Best Validation AUC: {best_auc:.4f}")
    if best_auc >= 0.7250:
        logger.info("🎉 Target AUC (0.7250) achieved or exceeded!")
    else:
        logger.info(f"📊 AUC gap to target: {0.7250 - best_auc:.4f}")
    logger.info("✅ Enhanced regularization and early stopping applied")
    logger.info(f"✅ Model saved to: saved_model/gainakt2_enhanced_auc_{best_auc:.4f}/")
    logger.info(f"{'='*60}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)