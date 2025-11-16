#!/bin/bash
# Phase 2: Loss Weight Balancing Sweep Launcher
# ==============================================
#
# This script launches Phase 2 sweep using optimal learning curve parameters
# from Phase 1 to find the best bce_loss_weight balancing performance and
# interpretability.
#
# Usage:
#   bash examples/launch_phase2_sweep.sh

set -e

cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate

# Set environment variables for optimal performance
export PYKT_NUM_WORKERS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Phase 2 configuration
BCE_WEIGHTS="0.3,0.4,0.5,0.6,0.7,0.8"
EPOCHS=12
MAX_PARALLEL=6
DATASET="assist2015"
FOLD=0

echo "========================================================================"
echo "PHASE 2: LOSS WEIGHT BALANCING SWEEP"
echo "========================================================================"
echo ""
echo "Sweep folder: examples/experiments/20251116_202637_gainakt3exp_sweep_phase2_126402/"
echo ""
echo "Configuration:"
echo "  BCE weights to test: ${BCE_WEIGHTS}"
echo "  Epochs: ${EPOCHS}"
echo "  Max parallel GPUs: ${MAX_PARALLEL}"
echo "  Dataset: ${DATASET}"
echo "  Fold: ${FOLD}"
echo ""
echo "Fixed parameters (Phase 1 optimal):"
echo "  beta_skill_init: 2.5"
echo "  m_sat_init: 0.7"
echo "  gamma_student_init: 1.1"
echo "  sigmoid_offset: 1.5"
echo ""
echo "Estimated time: 2-3 hours"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-confirm (comment out for interactive mode)
echo "y" | python "${SCRIPT_DIR}/sweep_loss_weights_phase2.py" \
  --bce_loss_weights "${BCE_WEIGHTS}" \
  --epochs ${EPOCHS} \
  --dataset ${DATASET} \
  --fold ${FOLD} \
  --max_parallel ${MAX_PARALLEL}

echo ""
echo "✅ Phase 2 sweep completed!"
echo ""
