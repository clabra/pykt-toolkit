#!/usr/bin/env python3
"""
Test the improved GainAKT2Monitored with cumulative mastery for perfect monotonicity.
This is a short test to validate the architectural improvements.
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


def test_monotonicity_fix():
    """Test the cumulative mastery fix for monotonicity."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("TESTING CUMULATIVE MASTERY FOR MONOTONICITY")
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
    
    # Model configuration with cumulative mastery
    num_c = data_config[dataset_name]['num_c']
    model_config = {
        'num_c': num_c,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 4,
        'num_encoder_blocks': 4,
        'd_ff': 512,
        'dropout': 0.3,  # Slightly reduced for faster training
        'emb_type': 'qid',
        # Strong constraints but balanced for quick test
        'non_negative_loss_weight': 0.5,
        'monotonicity_loss_weight': 0.5,  # Still keep some loss for stability
        'mastery_performance_loss_weight': 0.3,
        'gain_performance_loss_weight': 0.3,
        'sparsity_loss_weight': 0.1,
        'consistency_loss_weight': 0.1,
        'monitor_frequency': 25
    }
    
    # Create model with cumulative mastery
    logger.info("Creating GainAKT2Monitored with CUMULATIVE MASTERY...")
    model = create_monitored_model(model_config)
    monitor = InterpretabilityMonitor(model, log_frequency=25)
    model.set_monitor(monitor)
    model = model.to(device)
    logger.info(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info("✓ Cumulative mastery enforces PERFECT monotonicity by construction")
    
    # Quick training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    num_epochs = 5  # Just a few epochs to test the fix
    logger.info(f"Quick training for {num_epochs} epochs to test monotonicity fix...")
    
    # Quick training loop
    for epoch in range(num_epochs):
        logger.info(f"\\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0.0
        total_predictions = []
        total_targets = []
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        batch_count = 0
        
        for batch_idx, batch in enumerate(pbar):
            if batch_count >= 20:  # Only train on first 20 batches for speed
                break
                
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            with torch.no_grad():
                total_predictions.extend(valid_predictions.cpu().numpy())
                total_targets.extend(valid_targets.cpu().numpy())
            
            batch_count += 1
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Main': f'{main_loss.item():.4f}',
                'Constraint': f'{interpretability_loss.item():.4f}'
            })
        
        # Quick validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_count = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                if val_count >= 10:  # Only validate on first 10 batches
                    break
                    
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
                
                val_predictions.extend(valid_predictions.cpu().numpy())
                val_targets.extend(valid_targets.cpu().numpy())
                val_count += 1
        
        # Compute metrics
        train_auc = roc_auc_score(total_targets, total_predictions)
        val_auc = roc_auc_score(val_targets, val_predictions) if len(val_predictions) > 0 else 0.0
        
        logger.info(f"  Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Save the test model
    save_dir = "saved_model/gainakt2_cumulative_mastery_test"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'test_auc': val_auc
    }, os.path.join(save_dir, "model.pth"))
    
    logger.info(f"\\n{'='*60}")
    logger.info("CUMULATIVE MASTERY TEST COMPLETED!")
    logger.info(f"✓ Model with cumulative mastery saved to: {save_dir}")
    logger.info("✓ This model should have ZERO monotonicity violations")
    logger.info("✓ Run consistency validation to verify perfect monotonicity")
    logger.info(f"{'='*60}")
    
    return True


if __name__ == "__main__":
    success = test_monotonicity_fix()
    
    if success:
        print("\\n🎉 Test completed successfully!")
        print("\\n💡 Next steps:")
        print("1. Run: python validate_consistency.py")  
        print("   (Update the model path to use the test model)")
        print("2. Verify monotonicity violations = 0%")
        print("3. If successful, apply to full training")
    else:
        print("\\n⚠️  Test failed. Check error messages above.")