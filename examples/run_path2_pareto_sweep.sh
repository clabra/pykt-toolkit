#!/bin/bash
# Path 2 Pareto Sweep: --grounded_init 0 --calibrate 1
# This script launches a systematic TITRATION SWEEP across grounding weights
# to map the Fidelity-Performance Paradox frontier.

lambdas=(0.0 0.05 0.1 0.2 0.3 0.5)
gpus=(1 2 3 4 5 6) # Use GPUs 1-6 (leaving 0 and 7 free)

echo "Starting iDKT Path 2 Pareto Sweep..."

for i in "${!lambdas[@]}"; do
    l=${lambdas[$i]}
    g=${gpus[$i]}
    
    label="path2sweep_l$l"
    
    echo "[GPU $g] Launching Grounding Weight λ=$l..."
    
    # We use docker exec to run inside the container with proper environment
    # Each run is backgrounded
    docker exec -w /workspaces/pykt-toolkit pinn-dev /bin/bash -c "
        source /home/vscode/.pykt-env/bin/activate && \
        export CUDA_VISIBLE_DEVICES=$g && \
        python examples/run_repro_experiment.py \
            --short_title \"$label\" \
            --dataset assist2009 \
            --epochs 100 \
            --lambda_ref $l \
            --lambda_initmastery $l \
            --lambda_rate $l \
            --calibrate 1 \
            --grounded_init 0 \
            --use_wandb 0
    " &
done

echo "All 6 experiments are running in the background."
echo "Use 'docker exec pinn-dev nvidia-smi' to monitor progress."
