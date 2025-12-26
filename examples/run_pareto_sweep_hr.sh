#!/bin/bash
# Path 2 Pareto Sweep V2: --grounded_init 0 --calibrate 1
# This script launches a systematic TITRATION SWEEP across grounding weights.
# Optimized for 10 data points across 8 GPUs.

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

lambdas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0)
gpus=(0 1 2 3 4 5 6 7 0 1) # Distribution across 8 GPUs

echo "Starting High-Resolution Pareto Sweep for $DATASET..."

for i in "${!lambdas[@]}"; do
    l=${lambdas[$i]}
    g=${gpus[$i]}
    
    label="hr_sweep_l$l"
    
    echo "[GPU $g] Launching Grounding Weight λ=$l for $DATASET..."
    
    # Run in background directly (logic for being inside the container)
    export CUDA_VISIBLE_DEVICES=$g
    nohup python examples/run_repro_experiment.py \
        --short_title "${DATASET}_$label" \
        --dataset $DATASET \
        --epochs 100 \
        --lambda_ref $l \
        --lambda_initmastery $l \
        --lambda_rate $l \
        --calibrate 1 \
        --grounded_init 0 \
        --use_wandb 0 > /workspaces/pykt-toolkit/tmp/sweep_${DATASET}_l${l}.log 2>&1 &
done

echo "Launched 10 experiments for $DATASET in the background."
echo "Monitor with: docker exec pinn-dev nvidia-smi"
