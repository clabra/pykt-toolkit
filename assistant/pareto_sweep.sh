#!/bin/bash
# assistant/pareto_sweep.sh

# Generate 21 values from 0 to 1 with step 0.05
# This varies the over-all "theory-guidance strength"
LAMBDAS=($(seq 0 0.05 1))
GPUS=(0 1 2 3 4 5) # Using 6 GPUs (75% of available 8)
DATASET="assist2009"
FOLD=0
EPOCHS=100
MAX_CONCURRENT=9 # 1.5 jobs per GPU for a total of 6 GPUs

num_gpus=${#GPUS[@]}

echo "Starting High-Resolution iDKT Pareto Frontier Sweep ($DATASET, $EPOCHS epochs)..."
echo "Resources: Using 6 GPUs with max $MAX_CONCURRENT concurrent tasks."
echo "Varying lambda_ref, lambda_initmastery, and lambda_rate in synchronization."

# Set workers to 0 to prevent process explosion and save RAM during high-concurrency sweep
export PYKT_NUM_WORKERS=0

for i in "${!LAMBDAS[@]}"
do
    LAMBDA=${LAMBDAS[$i]}
    
    # Simple concurrency manager
    while : ; do
        # Count running launcher processes (one per experiment)
        running_count=$(pgrep -f "run_repro_experiment.py.*pareto_v2" | wc -l)
        if [ "$running_count" -lt "$MAX_CONCURRENT" ]; then
            break
        fi
        echo "Waiting for a slot... ($running_count experiments running)"
        sleep 30
    done

    # Cycle through available GPUs
    GPU_IDX=$((i % num_gpus))
    GPU_ID=${GPUS[$GPU_IDX]}
    
    TITLE="pareto_v2_l${LAMBDA}"
    LOG_FILE="assistant/log_v2_l${LAMBDA}.txt"
    
    echo "Launching [$((i+1))/${#LAMBDAS[@]}]: lambda_all=$LAMBDA on GPU $GPU_ID"
    
    # nohup + & ensures the process continues if the terminal is closed
    # We vary ALL three theoretical weights to find the optimal grounding/performance elbow
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python examples/run_repro_experiment.py \
        --model idkt \
        --dataset $DATASET \
        --fold $FOLD \
        --epochs $EPOCHS \
        --calibrate 1 \
        --theory_guided 1 \
        --lambda_ref $LAMBDA \
        --lambda_initmastery $LAMBDA \
        --lambda_rate $LAMBDA \
        --short_title $TITLE \
        --num_gpus 1 \
        --skip_roster \
        > "$LOG_FILE" 2>&1 &
    
    # Small stagger to prevent simultaneous data loading peaks
    sleep 5
done

echo "================================================================================"
echo "Sweep orchestration complete."
echo "Experiments will continue running in the background."
echo "Monitor overall load with: nvidia-smi"
echo "================================================================================"
