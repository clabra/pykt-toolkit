#!/bin/bash
# Monitor learnable alpha pilot experiment progress

EXPERIMENT_DIR="/workspaces/pykt-toolkit/examples/experiments/20251110_014931_gainakt2exp_learnable_alpha_seed42_370430"
LOG_FILE="/tmp/learnable_alpha_pilot.log"

echo "========================================"
echo "Learnable Alpha Pilot Experiment Monitor"
echo "========================================"
echo "Experiment ID: 370430"
echo "Started: 2025-11-10 01:49"
echo ""

# Check if process is running
if ps aux | grep -q "[p]ython examples/run_repro_experiment.py.*learnable_alpha"; then
    echo "✓ Training process is RUNNING"
    PID=$(ps aux | grep "[p]ython examples/run_repro_experiment.py.*learnable_alpha" | awk '{print $2}')
    echo "  PID: $PID"
else
    echo "✗ Training process has STOPPED"
fi
echo ""

# Show last 30 lines of log
echo "Recent log output:"
echo "---"
tail -30 "$LOG_FILE"
echo ""

# Show GPU usage
echo "GPU Usage:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -5
echo ""

# Check if metrics file exists
if [ -f "$EXPERIMENT_DIR/metrics_epoch.csv" ]; then
    echo "Training metrics (epochs completed):"
    cat "$EXPERIMENT_DIR/metrics_epoch.csv"
else
    echo "No metrics file yet (training in first epoch)"
fi
