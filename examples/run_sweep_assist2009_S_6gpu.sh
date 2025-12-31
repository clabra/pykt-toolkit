#!/bin/bash
# High-Resolution Pareto Sweep for assist2009_S on 6 GPUs (via Docker)
# Usage: ./examples/run_sweep_assist2009_S_6gpu.sh

DATASET="assist2009_S"
lambdas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0)
gpus=(0 1 2 3 4 5 0 1 2 3) # Distribution across 6 GPUs

echo "Starting High-Resolution Pareto Sweep for $DATASET on 6 GPUs (inside pinn-dev container)..."

# Create logs directory inside the container's workspace
docker exec -w /workspaces/pykt-toolkit pinn-dev mkdir -p tmp/logs

for i in "${!lambdas[@]}"; do
    l=${lambdas[$i]}
    g=${gpus[$i]}
    
    label="hr_sweep_l$l"
    
    echo "[GPU $g] Launching Grounding Weight λ=$l for $DATASET..."
    
    # Construct the command to be executed INSIDE the container
    # We use nohup and & inside the container to ensure persistence
    INNER_CMD="source /home/vscode/.pykt-env/bin/activate && export CUDA_VISIBLE_DEVICES=$g && nohup python examples/run_repro_experiment.py --short_title ${DATASET}_$label --dataset $DATASET --epochs 100 --lambda_ref $l --lambda_initmastery $l --lambda_rate $l --calibrate 1 --grounded_init 0 --use_wandb 0 > tmp/logs/sweep_${DATASET}_l${l}.log 2>&1 &"
    
    # Execute via docker exec
    docker exec -w /workspaces/pykt-toolkit pinn-dev /bin/bash -c "$INNER_CMD"
done

echo "Launched 10 experiments for $DATASET in the background inside the container."
echo "Monitor GPU usage with: docker exec pinn-dev nvidia-smi"
echo "Check logs inside container at: /workspaces/pykt-toolkit/tmp/logs/"
