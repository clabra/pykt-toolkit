#!/bin/bash
#
# Monitor Phase 1 Sweep Progress
# Shows GPU usage, experiment progress, and top results
#
# Copyright (c) 2025 Concha Labra. All Rights Reserved.

# Find latest results file
LATEST_CSV=$(ls -t /workspaces/pykt-toolkit/examples/sweep_results/phase1_sweep_*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_CSV" ]; then
    echo "No sweep results found!"
    exit 1
fi

echo "================================================================================"
echo "PHASE 1 SWEEP PROGRESS MONITOR"
echo "================================================================================"
echo ""

# Show GPU utilization
echo "GPU Utilization:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %3s%% util | %5s/%5s MB | %2s°C\n", $1, $2, $3, $4, $5}'
echo ""

# Count experiments
TOTAL_EXPERIMENTS=81
COMPLETED=$(tail -n +2 "$LATEST_CSV" | wc -l)
SUCCESS=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f2 | grep -c "success")
FAILED=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f2 | grep -c "failed")
TIMEOUT=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f2 | grep -c "timeout")

echo "Experiment Progress:"
echo "  Total: $TOTAL_EXPERIMENTS experiments"
echo "  Completed: $COMPLETED"
echo "  ├─ Success: $SUCCESS"
echo "  ├─ Failed: $FAILED"
echo "  └─ Timeout: $TIMEOUT"
echo "  Remaining: $((TOTAL_EXPERIMENTS - COMPLETED))"

if [ $COMPLETED -gt 0 ]; then
    PROGRESS_PCT=$((COMPLETED * 100 / TOTAL_EXPERIMENTS))
    echo "  Progress: $PROGRESS_PCT%"
fi
echo ""

# Show top 5 results if any successful experiments
if [ $SUCCESS -gt 0 ]; then
    echo "Top 5 Configurations by Encoder2 AUC:"
    echo "  Rank | Beta | M_sat | Gamma | Offset | E2_AUC | E1_AUC | Overall"
    echo "  -----|------|-------|-------|--------|--------|--------|--------"
    tail -n +2 "$LATEST_CSV" | \
        awk -F',' '$2=="success" {print $5","$6","$7","$8","$10","$11","$12}' | \
        sort -t',' -k5 -rn | \
        head -5 | \
        awk -F',' 'BEGIN{i=1} {printf "   %2d  | %4s | %5s | %5s | %6s | %6s | %6s | %7s\n", i++, $1, $2, $3, $4, $5, $6, $7}'
    echo ""
fi

# Show latest log tail
LATEST_LOG=$(ls -t /workspaces/pykt-toolkit/examples/sweep_results/phase1_sweep_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest Log Activity (last 10 lines):"
    tail -10 "$LATEST_LOG" | sed 's/^/  /'
fi

echo ""
echo "================================================================================"
echo "Results file: $LATEST_CSV"
echo "Log file: $LATEST_LOG"
echo "================================================================================"
