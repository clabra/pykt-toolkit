#!/bin/bash
#
# Real-time Phase 1 Sweep Monitor
# Monitors actual experiment directories and their progress
#
# Copyright (c) 2025 Concha Labra. All Rights Reserved.

echo "================================================================================"
echo "PHASE 1 SWEEP - REAL-TIME MONITOR"
echo "================================================================================"
echo ""

# Show GPU utilization
echo "GPU Status:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %3s%% | %5s/%5s MB | %2s°C\n", $1, $2, $3, $4, $5}'
echo ""

# Count sweep experiment directories (today's date, lc_ prefix)
TODAY=$(date +%Y%m%d)
SWEEP_DIRS=$(ls -d /workspaces/pykt-toolkit/examples/experiments/${TODAY}*_lc_* 2>/dev/null | wc -l)
TOTAL_EXPERIMENTS=81

echo "Experiment Progress:"
echo "  Total planned: $TOTAL_EXPERIMENTS"
echo "  Directories created: $SWEEP_DIRS"
echo "  Progress: $((SWEEP_DIRS * 100 / TOTAL_EXPERIMENTS))%"
echo ""

# Check completion status of recent experiments
COMPLETED=0
RUNNING=0
FAILED=0

for dir in $(ls -td /workspaces/pykt-toolkit/examples/experiments/${TODAY}*_lc_* 2>/dev/null | head -20); do
    if [ -f "$dir/metrics_epoch_eval.csv" ]; then
        ((COMPLETED++))
    elif [ -f "$dir/metrics_epoch.csv" ]; then
        EPOCHS=$(tail -n +2 "$dir/metrics_epoch.csv" | wc -l)
        if [ $EPOCHS -ge 6 ]; then
            ((COMPLETED++))  # Training done, waiting for eval
        else
            ((RUNNING++))
        fi
    elif [ -f "$dir/config.json" ]; then
        ((RUNNING++))
    fi
done

echo "Status (last 20 experiments):"
echo "  ├─ Completed (with eval): $COMPLETED"
echo "  ├─ Training in progress: $RUNNING"
echo "  └─ Failed/Starting: $((20 - COMPLETED - RUNNING))"
echo ""

# Show top performing experiments so far
if [ $COMPLETED -gt 0 ]; then
    echo "Top 5 Completed Experiments by Encoder2 AUC:"
    echo "  Dir | Beta | M_sat | Gamma | Offset | E2_AUC | E1_AUC"
    echo "  ----|------|-------|-------|--------|--------|--------"
    
    for dir in $(ls -td /workspaces/pykt-toolkit/examples/experiments/${TODAY}*_lc_* 2>/dev/null); do
        if [ -f "$dir/metrics_epoch_eval.csv" ]; then
            # Extract parameters from directory name
            BASENAME=$(basename "$dir")
            PARAMS=$(echo "$BASENAME" | grep -oP 'lc_b\K.*' | sed 's/_/ /g')
            
            # Get last row of evaluation metrics
            E2_AUC=$(tail -1 "$dir/metrics_epoch_eval.csv" | cut -d',' -f13)
            E1_AUC=$(tail -1 "$dir/metrics_epoch_eval.csv" | cut -d',' -f11)
            
            echo "$BASENAME $PARAMS $E2_AUC $E1_AUC"
        fi
    done | sort -k7 -rn | head -5 | awk '{printf "  %-4d | %-4s | %-5s | %-5s | %-6s | %6s | %6s\n", NR, $2, $3, $4, $5, $6, $7}'
fi

# Show sample of currently running experiments
RUNNING_COUNT=$(ps aux | grep "run_repro_experiment.py.*lc_b" | grep -v grep | wc -l)
echo ""
echo "Currently Running: $RUNNING_COUNT processes"
if [ $RUNNING_COUNT -gt 0 ]; then
    echo "  Recent active experiments:"
    ps aux | grep "run_repro_experiment.py.*lc_b" | grep -v grep | head -3 | \
        awk '{print "    - " $0}' | grep -oP 'lc_b[^ ]*' | sed 's/^/    /'
fi

echo ""
echo "================================================================================"
echo "Refresh: watch -n 30 'bash examples/monitor_phase1_sweep_realtime.sh'"
echo "================================================================================"
