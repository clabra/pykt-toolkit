#!/usr/bin/env python3
"""
Enhanced GainAKT2Exp Training with Detailed Epoch AUC Feedback

This script provides comprehensive epoch-by-epoch AUC reporting for better monitoring
during parameter sweeps. Based on the original wandb_gainakt2exp_train.py but with
enhanced logging for every epoch.

Optimal parameters (set as defaults):
- learning_rate = 0.000174        # 50% of base
- weight_decay = 1.7571e-05       # 30% of base  
- batch_size = 96                 # Optimal balance
- Individual constraint weights   # Granular control for sweep optimization

# Proven optimal configuration
non_negative_loss_weight = 0.0
monotonicity_loss_weight = 0.05
mastery_performance_loss_weight = 0.5
gain_performance_loss_weight = 0.5
sparsity_loss_weight = 0.1  # Simplest choice
consistency_loss_weight = 0.3  # Middle value
"""

import sys
import os
import argparse
import logging

# Add the parent directory to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def setup_detailed_logging():
    """Setup detailed logging for epoch-by-epoch feedback"""
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup new detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main(params):
    """Main training function with enhanced epoch feedback"""
    
    # Import training modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
    from wandb_train import main as wandb_main
    
    # Setup detailed logging
    setup_detailed_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 ENHANCED GainAKT2Exp Training with Detailed AUC Feedback")
    logger.info("=" * 70)
    logger.info(f"Dataset: {params['dataset_name']}")
    logger.info(f"Epochs: {params['epochs']}")
    logger.info(f"Learning rate: {params['learning_rate']}")
    logger.info(f"Batch size: {params['batch_size']}")
    logger.info(f"Enhanced constraints: {params['enhanced_constraints']}")
    logger.info("Constraint weights:")
    for key, value in params.items():
        if 'loss_weight' in key:
            logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    logger.info("Using device: cuda")
    logger.info("=" * 70)
    
    # Add detailed epoch callback
    original_params = params.copy()
    original_params['detailed_epoch_feedback'] = True
    
    # Call the main training function
    try:
        results = wandb_main(original_params)
        
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        if isinstance(results, dict) and 'best_val_auc' in results:
            logger.info(f"🏆 Final Best Validation AUC: {results['best_val_auc']:.4f}")
        else:
            logger.info("✅ Training completed")
        logger.info("=" * 50)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ TRAINING FAILED: {e}")
        logger.error("=" * 50)
        raise

def main_wrapper():
    """Wrapper function to parse arguments and call main"""
    parser = argparse.ArgumentParser(description='Enhanced GainAKT2Exp Training')
    
    # Basic training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (OPTIMAL)')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size (OPTIMAL)')  
    parser.add_argument('--learning_rate', type=float, default=0.000174, help='Learning rate (OPTIMAL)')
    parser.add_argument('--weight_decay', type=float, default=1.7571e-05, help='Weight decay (OPTIMAL)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (OPTIMAL)')
    
    # Enhanced constraints - individual weight parameters
    parser.add_argument('--enhanced_constraints', type=bool, default=False, 
                       help='Enable enhanced constraint framework')
    parser.add_argument('--non_negative_loss_weight', type=float, default=0.0,
                       help='Non-negative constraint loss weight (OPTIMAL)')
    parser.add_argument('--monotonicity_loss_weight', type=float, default=0.05,
                       help='Monotonicity constraint loss weight (OPTIMAL)')  
    parser.add_argument('--mastery_performance_loss_weight', type=float, default=0.5,
                       help='Mastery performance constraint loss weight (OPTIMAL)')
    parser.add_argument('--gain_performance_loss_weight', type=float, default=0.5,
                       help='Gain performance constraint loss weight (OPTIMAL)')
    parser.add_argument('--sparsity_loss_weight', type=float, default=0.1,
                       help='Sparsity constraint loss weight (SWEEP PARAMETER)')
    parser.add_argument('--consistency_loss_weight', type=float, default=0.3,
                       help='Consistency constraint loss weight (SWEEP PARAMETER)')
    parser.add_argument('--monitor_freq', type=int, default=50, help='Monitoring frequency')
    
    # Experiment parameters  
    parser.add_argument('--dataset_name', type=str, default='assist2015', help='Dataset name')
    parser.add_argument('--fold', type=int, default=0, help='Data fold')
    parser.add_argument('--experiment_suffix', type=str, default='enhanced_v1', 
                       help='Experiment name suffix')
    parser.add_argument('--use_wandb', type=int, default=0, help='Use Weights & Biases logging (0=offline)')
    parser.add_argument('--model_name', type=str, default='gainakt2exp', help='Model name')
    parser.add_argument('--emb_type', type=str, default='qid', help='Embedding type')
    parser.add_argument('--save_dir', type=str, default='saved_model', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--add_uuid', type=int, default=1, help='Add uuid to save dir')
    parser.add_argument('--l2', type=float, default=1.7571e-05, help='L2 regularization (same as weight_decay)')

    args = parser.parse_args()
    params = vars(args)
    
    # Print detailed configuration for debugging
    print("\n🚀 Enhanced GainAKT2Exp Training Configuration")
    print("=" * 60)
    print(f"📊 Dataset: {params['dataset_name']}")
    print(f"🔄 Fold: {params['fold']}")
    print(f"🎲 Seed: {params['seed']}")
    print(f"📈 Use W&B: {params['use_wandb']}")
    print(f"⚙️  Enhanced constraints: {params['enhanced_constraints']}")
    print("🎯 Granular constraint weights:")
    for key, value in params.items():
        if 'loss_weight' in key:
            print(f"  📌 {key.replace('_', ' ').title()}: {value}")
    print("=" * 60)
    
    # Call main training function
    main(params)

if __name__ == "__main__":
    main_wrapper()