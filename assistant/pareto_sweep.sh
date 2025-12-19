#!/bin/bash
# assistant/pareto_sweep.sh

LAMBDAS=(0.0 0.1 0.25 0.5 0.75 1.0)
GPUS=(1 2 3 4 5 6) # Using GPUs 1 to 6 as requested
DATASET="assist2015"
FOLD=0
EPOCHS=10

echo "Starting iDKT Pareto Frontier Sweep in Parallel (Background)..."

for i in "${!LAMBDAS[@]}"
do
    LAMBDA=${LAMBDAS[$i]}
    GPU_ID=${GPUS[$i]}
    TITLE="pareto_l${LAMBDA}"
    LOG_FILE="assistant/log_l${LAMBDA}.txt"
    
    echo "Launching: lambda_ref=$LAMBDA on GPU $GPU_ID (Log: $LOG_FILE)"
    
    # nohup + & ensures the process continues if the terminal is closed
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python examples/run_repro_experiment.py --model idkt --dataset $DATASET --fold $FOLD --epochs $EPOCHS --calibrate 1 --theory_guided 1 --lambda_ref $LAMBDA --short_title $TITLE --num_gpus 1 \
        > "$LOG_FILE" 2>&1 &
done

echo "================================================================================"
echo "All 6 experiments launched in parallel on GPUs 1-6."
echo "You can safely close this terminal."
echo "Monitor progress with: tail -f assistant/log_l*.txt"
echo "================================================================================"
