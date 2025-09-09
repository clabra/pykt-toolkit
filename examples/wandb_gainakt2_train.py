"""
Train the GainAKT2 model using the generic training script.

This script is the entry point for training the GainAKT2 model. It configures the
hyperparameters and passes them to the main training function in `wandb_train.py`.

Example of how to run this script:

.. code-block:: bash

    python examples/wandb_gainakt2_train.py \
        --dataset_name=assist2015 \
        --use_wandb=0 \
        --learning_rate=2e-4 \
        --d_model=256 \
        --num_encoder_blocks=4 \
        --d_ff=1024 \
        --dropout=0.2
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
