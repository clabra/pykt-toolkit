/*

python wandb_gainakt2exp_train.py

Best validation AUC: 0.7260

params: {'epochs': 20, 'batch_size': 96, 'learning_rate': 0.000174, 'weight_decay': 1.7571e-05, 'patience': 5, 'num_epochs': 20, 'enhanced_constraints': True, 'monitor_freq': 50, 'dataset_name': 'assist2015', 'fold': 0, 'experiment_suffix': 'optimal_v1', 'use_wandb': 0, 'model_name': 'gainakt2exp', 'emb_type': 'qid', 'save_dir': 'saved_model', 'seed': 42, 'add_uuid': 1, 'l2': 1.7571e-05}, params_str: 20_96_0.000174_1.7571e-05_5_20_True_50_assist2015_0_optimal_v1_0_gainakt2exp_qid_saved_model_42_1_1.7571e-05

*/

import argparse
from wandb_train import main as wandb_main

def main():
    parser = argparse.ArgumentParser(description='Train GainAKT2Exp with cumulative mastery')
    
    # Training parameters (OPTIMAL CONFIGURATION - AUC: 0.7260, Perfect Consistency)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (peaks at epoch 3)')
    parser.add_argument('--batch_size', type=int, default=96, help='Optimal batch size')
    parser.add_argument('--learning_rate', type=float, default=0.000174, help='OPTIMAL learning rate (50% of base)')
    parser.add_argument('--weight_decay', type=float, default=1.7571e-05, help='OPTIMAL weight decay (30% of base)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (recommended due to early peak)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs (alias for epochs)')
    
    # Model parameters
    parser.add_argument('--enhanced_constraints', type=bool, default=True, 
                       help='Use enhanced constraint weights for better correlations')
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