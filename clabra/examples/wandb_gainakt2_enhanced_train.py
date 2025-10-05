"""
Train the GainAKT2 Enhanced model using the generic training script.

This script is the entry point for training the GainAKT2 Enhanced model with 
multi-scale attention and advanced architectural features. It configures the
hyperparameters and passes them to the main training function in `wandb_train.py`.

Optimized parameters: 
CUDA_VISIBLE_DEVICES=0 python wandb_gainakt2_enhanced_train.py --dataset_name=assist2015 --use_wandb=0 --num_epochs=10 --d_model=384 --learning_rate=0.0001 --num_encoder_blocks=8 --d_ff=1536 --dropout=0.2 --n_heads=8 --use_knowledge_tracking=1

Epoch: 1, validauc: 0.6416, validacc: 0.7402, best epoch: 1, best auc: 0.6416, train loss: 0.5830016918963211,

Epoch: 2, validauc: 0.6432, validacc: 0.7394, best epoch: 2, best auc: 0.6432, train loss: 0.5579726117674066

Epoch: 3, validauc: 0.6432, validacc: 0.7394, best epoch: 2, best auc: 0.6432, train loss: 0.5555675646000383
____

Example of how to run this script:

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/wandb_gainakt2_enhanced_train.py \
    --dataset_name=assist2015 \
    --use_wandb=0 \
    --d_model=256 \
    --learning_rate=1e-4 \
    --dropout=0.2 \
    --num_encoder_blocks=6 \
    --d_ff=1024 \
    --n_heads=8 \
    --seq_len=200 \
    --num_epochs=300



ENHANCED CONFIGURATION TARGETING AUC 0.8+:

ENHANCED PARAMETER SET (Target Validation AUC: 0.8+):
- d_model: 256 (optimized from baseline)
- learning_rate: 0.0001 (1e-4, slower for enhanced architecture)
- dropout: 0.2 (adaptive gating handles overfitting)
- num_encoder_blocks: 6 (deeper for multi-scale attention)
- d_ff: 1024 (larger capacity for enhanced features)
- n_heads: 8 (standard multi-head configuration)
- num_epochs: 300+ (enhanced model needs more training)
- batch_size: 64 (from config)
- optimizer: adam
- seq_len: 200
- use_knowledge_tracking: True (enhanced feature)
- temperature: 1.0 (calibrated predictions)

ENHANCED FEATURES:
- Multi-scale attention for short/long-term dependencies
- Adaptive gating mechanism between context and value streams
- Knowledge state tracking for interpretability
- Uncertainty estimation head
- GELU activations for smoother gradients
- Enhanced prediction head with deeper architecture
- Better weight initialization strategies

ARCHITECTURAL IMPROVEMENTS:
- MultiScaleAttention with scales [1, 2, 4]
- AdaptiveGatingMechanism for stream balancing
- KnowledgeStateTracker for concept mastery
- Uncertainty estimation for prediction confidence
- Cross-stream attention components
- Temperature scaling for calibration

TARGET PERFORMANCE:
- AUC: 0.8+ (vs 0.7242 baseline)
- Improved interpretability through knowledge tracking
- Better uncertainty quantification
- Enhanced generalization across datasets

REPRODUCTION COMMAND:
cd /workspaces/pykt-toolkit/examples
python wandb_gainakt2_enhanced_train.py --dataset_name=assist2015 --use_wandb=0

"""

import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="gainakt2_enhanced")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    
    # GainAKT2 Enhanced specific parameters (optimized defaults)
    parser.add_argument("--d_model", type=int, default=256)  # Enhanced default
    parser.add_argument("--learning_rate", type=float, default=0.0001)  # Slower for enhanced model
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_encoder_blocks", type=int, default=6)  # Deeper for multi-scale attention
    parser.add_argument("--d_ff", type=int, default=1024)  # Larger capacity for enhanced features
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=300)  # Enhanced model needs more training
    
    # Enhanced model specific parameters
    parser.add_argument("--use_knowledge_tracking", type=int, default=1)  # Enable knowledge tracking
    parser.add_argument("--temperature", type=float, default=1.0)  # Temperature for calibration
    
    # Wandb configuration
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)