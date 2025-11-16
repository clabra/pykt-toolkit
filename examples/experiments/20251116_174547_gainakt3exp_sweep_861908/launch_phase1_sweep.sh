#!/bin/bash
#
# Launch Phase 1 Learning Curve Parameter Sweep
# with intensive GPU and CPU utilization
#
# Copyright (c) 2025 Concha Labra. All Rights Reserved.

set -e

# Configuration
EPOCHS=6
MAX_PARALLEL=7  # Use 7 GPUs (leaving 1 for system)
DATASET="assist2015"  # Default dataset per parameter_default.json
FOLD=0
OUTPUT_DIR="examples/sweep_results"

# CPU optimization: Set dataloader workers
# With 7 parallel jobs, use 4 workers per job = 28 workers total
# This keeps CPU usage below 75% on typical multi-core systems
export PYKT_NUM_WORKERS=4

# PyTorch optimization
export OMP_NUM_THREADS=4  # OpenMP threads per process
export MKL_NUM_THREADS=4  # Intel MKL threads per process

echo "================================================================================"
echo "PHASE 1: LEARNING CURVE PARAMETER SWEEP"
echo "================================================================================"
echo "Configuration:"
echo "  Epochs per experiment: ${EPOCHS}"
echo "  Max parallel jobs: ${MAX_PARALLEL} GPUs"
echo "  Dataset: ${DATASET}"
echo "  Fold: ${FOLD}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""
echo "CPU/GPU Optimization:"
echo "  PYKT_NUM_WORKERS: ${PYKT_NUM_WORKERS} (dataloader workers per job)"
echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "  MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "  Total worker processes: ~$((MAX_PARALLEL * PYKT_NUM_WORKERS)) = ${MAX_PARALLEL} jobs × ${PYKT_NUM_WORKERS} workers"
echo "================================================================================"
echo ""

# Activate virtual environment
source /home/vscode/.pykt-env/bin/activate

# Run sweep (reduced grid: 81 experiments, ~3-3.5 hours with 7 GPUs)
# Auto-confirm with 'y' input
echo "y" | python examples/sweep_learning_curves.py \
    --epochs ${EPOCHS} \
    --max_parallel ${MAX_PARALLEL} \
    --dataset ${DATASET} \
    --fold ${FOLD} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "================================================================================"
echo "SWEEP COMPLETE"
echo "Results saved to: ${OUTPUT_DIR}"
echo "================================================================================"
