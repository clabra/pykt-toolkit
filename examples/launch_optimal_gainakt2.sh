#!/bin/bash
"""
Launch GainAKT2 with Exact Optimal Parameters from Wandb Sweep

This script launches GainAKT2 training with the exact same configuration 
that achieved AUC: 0.7233 in your wandb sweep optimization.

Usage:
    cd /workspaces/pykt-toolkit/examples
    source /home/vscode/.pykt-env/bin/activate  
    bash launch_optimal_gainakt2.sh
"""

echo "🚀 Launching GainAKT2 with Optimal Parameters (Target AUC: 0.7233)"
echo "=================================================================="
echo "Key parameters:"
echo "  - d_model: 256"
echo "  - learning_rate: 0.0002" 
echo "  - dropout: 0.2"
echo "  - num_encoder_blocks: 4"
echo "  - d_ff: 768 (KEY improvement!)"
echo "  - n_heads: 8"
echo "  - num_epochs: 200 (sufficient for convergence)"
echo "  - batch_size: 64 (from config)"
echo "=================================================================="

# Launch training with exact optimal parameters
python wandb_gainakt2_train.py \
    --dataset_name=assist2015 \
    --use_wandb=0 \
    --d_model=256 \
    --learning_rate=0.0002 \
    --dropout=0.2 \
    --num_encoder_blocks=4 \
    --d_ff=768 \
    --n_heads=8 \
    --num_epochs=200 \
    --seed=42 \
    --fold=0

echo "🎯 Expected result: Validation AUC ~0.7233 (if all parameters match exactly)"