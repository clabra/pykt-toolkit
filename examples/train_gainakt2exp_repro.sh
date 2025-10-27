#!/usr/bin/env bash
# Reproducible launch script for GainAKT2Exp.
# Usage: bash examples/train_gainakt2exp_repro.sh
# Adjust CUDA devices & key hyperparameters below.

set -euo pipefail

MODEL=gainakt2exp
TITLE=baseline
EPOCHS=12
BATCH=64
LR=0.000174
WD=1.7571e-05
FOLD=0
DATASET=assist2015
SEED=42
DEVICES="0,1,2,3"

export CUDA_VISIBLE_DEVICES="${DEVICES}"

python examples/train_gainakt2exp_repro.py \
  --experiment_title "${TITLE}" \
  --dataset "${DATASET}" \
  --fold ${FOLD} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH} \
  --learning_rate ${LR} \
  --weight_decay ${WD} \
  --seed ${SEED} \
  --enhanced_constraints \
  --monitor_freq 50 \
  --patience 12 \
  --use_mastery_head --use_gain_head \
  "$@"

echo "Launch complete. Check examples/experiments for new folder."