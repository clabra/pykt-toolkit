"""
Train the GainAKT2 model using the generic training script.

This script is the entry point for training the GainAKT2 model. It configures the
hyperparameters and passes them to the main training function in `wandb_train.py`.

Example of how to run this script:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/wandb_gainakt2_train.py \
        --dataset_name=assist2015 \
        --use_wandb=0 \
        --d_model=256 \
        --learning_rate=2e-4 \
        --dropout=0.2 \
        --num_encoder_blocks=4 \
        --d_ff=768 \
        --n_heads=8 \
        --seq_len=200 \
        --num_epochs=200

OPTIMIZED CONFIGURATION (Best AUC: 0.7233 from Advanced Parameter Sweep):

TOP 5 PARAMETER COMBINATIONS (Validation AUC):
1. AUC: 0.7233 | d_model: 256 | lr: 0.0002 | dropout: 0.2 | blocks: 4 | d_ff: 768
2. AUC: 0.7229 | d_model: 384 | lr: 0.0002 | dropout: 0.2 | blocks: 4 | d_ff: 768  
3. AUC: 0.7228 | d_model: 384 | lr: 0.0002 | dropout: 0.2 | blocks: 4 | d_ff: 768
4. AUC: 0.7226 | d_model: 384 | lr: 0.0002 | dropout: 0.2 | blocks: 4 | d_ff: 512
5. AUC: 0.7225 | d_model: 256 | lr: 0.0002 | dropout: 0.2 | blocks: 4 | d_ff: 512

KEY INSIGHTS FROM OPTIMIZATION:
- d_ff=768 consistently outperformed d_ff=512 and d_ff=1024
- learning_rate=2e-4 was optimal across all successful configurations
- dropout=0.2 provided best regularization balance
- 4 encoder blocks achieved optimal depth vs. efficiency trade-off
- 12 attention heads caused instability (6/6 failures with n_heads=12)

Best Performance:

Validation AUC: 0.7233 (72.33% - NEW RECORD!)
Validation Accuracy: 0.7531 (75.31%)
Training Loss: 0.4866 (Well converged)

WINNING PARAMETERS:

d_model: 256
learning_rate: 0.0002
dropout: 0.2
num_encoder_blocks: 4
d_ff: 768 (This was the key improvement!)
n_heads: 8

"""

import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="gainakt2")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    
    # GainAKT2 specific parameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_encoder_blocks", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=1)

    # Interpretability enhancements
    parser.add_argument("--use_gain_head", type=int, default=0)
    parser.add_argument("--use_mastery_head", type=int, default=0)
    parser.add_argument("--non_negative_loss_weight", type=float, default=0.0)
    parser.add_argument("--consistency_loss_weight", type=float, default=0.0)
    
    # Wandb configuration
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)
