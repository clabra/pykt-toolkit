#!/bin/bash

# Manual hyperparameter sweep for the dynamic GainAKT2 model

LEARNING_RATES=(1e-3 5e-4 2e-4)
DROPOUT_RATES=(0.1 0.2)

for lr in "${LEARNING_RATES[@]}"; do
    for dropout in "${DROPOUT_RATES[@]}"; do
        echo "Running with learning_rate=${lr} and dropout=${dropout}"
        python examples/wandb_gainakt2_train.py \
            --dataset_name=assist2015 \
            --model_name=gainakt2 \
            --use_wandb=0 \
            --learning_rate=${lr} \
            --dropout=${dropout}
    done
done

echo "Sweep complete."
