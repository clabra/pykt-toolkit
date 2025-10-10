"""
GainAKT2Exp Training with Granular Constraint Control

Best validation AUC: 0.7260

Optimal parameters (now set as defaults):
- learning_rate = 0.000174        # 50% of base
- weight_decay = 1.7571e-05       # 30% of base  
- batch_size = 96                 # Optimal balance
- Individual constraint weights   # Granular control for sweep optimization


Usage:
    python wandb_gainakt2exp_train.py
    python wandb_gainakt2exp_train.py --mastery_performance_loss_weight 0.9 --consistency_loss_weight 0.4
"""

import argparse
from wandb_train import main as wandb_main

def main():
    parser = argparse.ArgumentParser(description='Train GainAKT2Exp with granular constraint control for sweep optimization')
    
    # Core training parameters (OPTIMAL CONFIGURATION - AUC: 0.7260, Perfect Consistency)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (peaks at epoch 3)')
    parser.add_argument('--batch_size', type=int, default=96, help='Optimal batch size')
    parser.add_argument('--learning_rate', type=float, default=0.000174, help='OPTIMAL learning rate (50%% of base)')
    parser.add_argument('--weight_decay', type=float, default=1.7571e-05, help='OPTIMAL weight decay (30%% of base)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (recommended due to early peak)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs (alias for epochs)')
    
    # Enhanced constraints - granular control for sweep optimization
    parser.add_argument('--enhanced_constraints', action='store_true', default=False, 
                       help='Use enhanced constraint preset (overridden by individual parameters)')
    
    # Individual constraint weight parameters for fine-tuning
    parser.add_argument('--non_negative_loss_weight', type=float, default=0.0,
                       help='Weight for non-negative mastery constraint (optimal: 0.0)')
    parser.add_argument('--monotonicity_loss_weight', type=float, default=0.1,
                       help='Weight for monotonicity constraint (optimal: 0.1)')
    parser.add_argument('--mastery_performance_loss_weight', type=float, default=0.8,
                       help='Weight for mastery performance alignment (optimal: 0.8)')
    parser.add_argument('--gain_performance_loss_weight', type=float, default=0.8,
                       help='Weight for gain performance alignment (optimal: 0.8)')
    parser.add_argument('--sparsity_loss_weight', type=float, default=0.2,
                       help='Weight for sparsity regularization (optimal: 0.2)')
    parser.add_argument('--consistency_loss_weight', type=float, default=0.3,
                       help='Weight for consistency constraint (optimal: 0.3)')
    
    # Model parameters  
    parser.add_argument('--monitor_freq', type=int, default=50, 
                       help='Interpretability monitoring frequency')
    
    # Experiment parameters
    parser.add_argument('--dataset_name', type=str, default='assist2015', help='Dataset name')
    parser.add_argument('--fold', type=int, default=0, help='Data fold')
    parser.add_argument('--experiment_suffix', type=str, default='optimal_v1', 
                       help='Experiment name suffix')
    parser.add_argument('--use_wandb', type=int, default=0, help='Use Weights & Biases logging')
    parser.add_argument('--model_name', type=str, default='gainakt2exp', help='Model name')
    parser.add_argument('--emb_type', type=str, default='qid', help='Embedding type')
    parser.add_argument('--save_dir', type=str, default='saved_model', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--add_uuid', type=int, default=1, help='Add uuid to save dir')
    parser.add_argument('--l2', type=float, default=1.7571e-05, help='L2 regularization (OPTIMAL - same as weight_decay)')

    args = parser.parse_args()
    params = vars(args)
    
    wandb_main(params)

if __name__ == "__main__":
    main()