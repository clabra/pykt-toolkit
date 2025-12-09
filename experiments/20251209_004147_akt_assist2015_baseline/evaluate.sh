#!/bin/bash
#
# Evaluation script for benchmark models
# Generated automatically by benchmark_models.py
#
# Dataset: assist2015
# Seed: 42
# Benchmark results: /workspaces/pykt-toolkit/experiments/20251209_004147_benchmark_assist2015/results.csv
#
# This script evaluates all successfully trained models on test data.
# Run manually after all training completes:
#   bash /workspaces/pykt-toolkit/experiments/20251209_004147_benchmark_assist2015/evaluate.sh
#

# Activate virtual environment
source /home/vscode/.pykt-env/bin/activate

# Change to examples directory (required by wandb_predict.py)
cd /workspaces/pykt-toolkit/examples

# Navigate to examples directory
cd /workspaces/pykt-toolkit/examples

echo '========================================'
echo 'Starting Model Evaluations'
echo '========================================'

# Evaluation 1: akt (fold 0)
echo ''
echo '[1] Evaluating akt (fold 0)...'
python wandb_predict.py \
    --save_dir saved_model/assist2015_akt_qid_saved_model_42_0_0.2_256_512_8_4_0.0001_0_0 \
    --use_wandb 0 \
    --bz 256

if [ $? -eq 0 ]; then
    echo '✓ akt (fold 0) evaluation completed'
else
    echo '✗ akt (fold 0) evaluation failed'
fi

echo ''
echo '========================================'
echo 'All Evaluations Complete'
echo '========================================'
echo ''
echo 'Total evaluations run: 1'
echo 'Check individual model directories for test results.'
