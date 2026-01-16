#!/bin/bash
# Re-run key experiments with lambda_ref=1.0 instead of 0.5
# This script launches training campaigns for the critical experiments

set -e

echo "=========================================="
echo "Re-running Experiments with lambda_ref=1.0"
echo "=========================================="
echo ""
echo "Updated configs/parameter_default.json: lambda_ref = 1.0"
echo ""
echo "The following experiments will be re-run:"
echo "  1. Exp 090230 (Grounded, 2 blocks, 8 heads)"
echo "  2. Exp 636452 (Active Grounding/Probing, 2 blocks, 8 heads)"
echo ""
echo "Dataset: assist2009"
echo "Folds: 0-4 (5-fold CV)"
echo "Epochs: 200"
echo "GPUs: 0-4 (parallel execution)"
echo ""

# Confirm before proceeding
read -p "Proceed with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "================================================"
echo "CAMPAIGN 1: Baseline Grounded (lambda_ref=1.0)"
echo "================================================"
echo "Architecture: n_blocks=2, n_heads=8"
echo "Grounding: lambda_ref=1.0, lambda_initmastery=0.1, lambda_rate=0.1"
echo "Active Probing: DISABLED (active_grounding=0, lambda_probe=0.0)"
echo ""

# Note: We need to temporarily modify run_benchmarks_paper.py or create a custom launcher
# For now, we'll use the standard launcher which will pick up the new default

# Option 1: Run via run_benchmarks_paper.py (picks up new defaults)
echo "Launching baseline grounded experiment..."
nohup python3 examples/run_benchmarks_paper.py \
    --mode training \
    --dataset assist2009 \
    --model gtransformer \
    --gpus 0,1,2,3,4 \
    > experiments/rerun_lambda1.0_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

BASELINE_PID=$!
echo "Baseline training launched (PID: $BASELINE_PID)"
echo "Log: experiments/rerun_lambda1.0_baseline_*.log"
echo ""

# Wait a bit before launching next campaign to avoid resource contention
echo "Waiting 60 seconds before launching probing campaign..."
sleep 60

echo ""
echo "========================================================"
echo "CAMPAIGN 2: Active Probing Grounded (lambda_ref=1.0)"
echo "========================================================"
echo "Architecture: n_blocks=2, n_heads=8"
echo "Grounding: lambda_ref=1.0, lambda_initmastery=0.1, lambda_rate=0.1"
echo "Active Probing: ENABLED (active_grounding=1, lambda_probe=1.0)"
echo ""

# For the probing experiment, we need active_grounding=1
# This requires a custom launch - we'll document the manual command

echo "NOTE: The probing experiment requires active_grounding=1"
echo "To launch manually, update examples/run_benchmarks_paper.py line ~150:"
echo "  Add: \"--active_grounding\", \"1\","
echo "  Add: \"--lambda_probe\", \"1.0\","
echo ""
echo "Then run:"
echo "  python3 examples/run_benchmarks_paper.py --mode training --dataset assist2009 --model gtransformer"
echo ""

echo "================================================"
echo "Baseline Campaign Status"
echo "================================================"
echo "PID: $BASELINE_PID"
echo "Monitor with: tail -f experiments/rerun_lambda1.0_baseline_*.log"
echo "Check progress with: ps aux | grep run_benchmarks_paper.py"
echo ""
echo "After training completes, run evaluation:"
echo "  python3 examples/run_benchmarks_paper.py --mode evaluation --dataset assist2009 --model gtransformer"
echo ""
echo "Then gather results:"
echo "  python3 examples/run_benchmarks_paper.py --mode results --dataset assist2009 --model gtransformer"
echo ""
