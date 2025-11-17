#!/bin/bash
# Launch script for Phase 1 Learning Curve Parameter Sweep with BCE=0.9
#
# This script activates the environment and launches the sweep in the background
# with proper logging.

set -e

SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /workspaces/pykt-toolkit

echo "==========================================================================="
echo "Phase 1 Learning Curve Parameter Sweep - BCE Loss Weight = 0.9"
echo "==========================================================================="
echo ""
echo "Sweep Directory: ${SWEEP_DIR}"
echo "Rationale: Find optimal learning curve parameters for bce_loss_weight=0.9"
echo "Baseline:  Experiment 714616 achieved Test AUC=0.7183 with bce=0.9"
echo "Goal:      Match or exceed baseline performance"
echo ""
echo "Grid: 81 experiments (3×3×3×3)"
echo "  - beta_skill_init: [1.5, 2.0, 2.5]"
echo "  - m_sat_init: [0.7, 0.8, 0.9]"
echo "  - gamma_student_init: [0.9, 1.0, 1.1]"
echo "  - sigmoid_offset: [1.5, 2.0, 2.5]"
echo ""
echo "Fixed: bce_loss_weight=0.9, epochs=6, dataset=assist2015, fold=0"
echo ""
echo "Estimated Duration: ~1.5 hours on 5 GPUs (PARALLEL)"
echo "==========================================================================="
echo ""

# Activate environment
source /home/vscode/.pykt-env/bin/activate

# Create log file
LOG_FILE="${SWEEP_DIR}/sweep_execution.log"

echo "Starting parallel sweep at $(date)"
echo "Logs: ${LOG_FILE}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check GPU usage:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "All experiments will be saved inside:"
echo "  ${SWEEP_DIR}/experiments/"
echo ""

# Run parallel sweep
python3 "${SWEEP_DIR}/sweep_phase1_bce09_parallel.py" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "==========================================================================="
echo "Sweep completed at $(date)"
echo "Results: ${SWEEP_DIR}/sweep_results_*.csv"
echo "==========================================================================="
