#!/bin/bash
#
# Dual Evaluation Launcher
# 
# This script launches dual evaluation (p_sup + p_ref) for interpretability validation.
# It uses the existing run_benchmarks_paper.py infrastructure to ensure protocol compliance.
#
# Usage:
#   ./launch_dual_eval.sh [GPU_IDS] [--wait]
#
# Examples:
#   ./launch_dual_eval.sh                        # Use default GPUs 0-5
#   ./launch_dual_eval.sh "0,1,2"                # Use specific GPUs
#   ./launch_dual_eval.sh "0,1,2,3,4,5" --wait   # Use all GPUs and wait for completion
#

# Parse arguments
GPU_IDS=${1:-"0,1,2,3,4,5"}
WAIT_FLAG=${2:-""}

echo "================================================================================"
echo "DUAL EVALUATION LAUNCHER"
echo "================================================================================"
echo "GPUs: $GPU_IDS"
echo "Protocol: Question-level, Late Fusion (Mean Average)"
echo "Infrastructure: run_benchmarks_paper.py --mode evaluation --dual_eval"
echo "================================================================================"
echo ""

# Key experiments for dual evaluation
EXPERIMENTS=(
    "experiments/20260115_090230_benchpaper/gtransformer/assist2009/fold_1_537612"
    "experiments/20260116_101107_benchpaper_oraclecorrect_baseline_334772/gtransformer/assist2009/fold_1_886974"
    "experiments/20260116_120815_benchpaper_personalization_948799/gtransformer/assist2009/fold_1_116019"
    "experiments/20260115_133835_benchpaper_baseline_optimal/gtransformer/assist2009/fold_1_791412"
)

DESCRIPTIONS=(
    "090230: Grounded only (no probing)"
    "334772: Aligned Grounding (with probing)"
    "948799: Full (with personalization)"
    "133835: Ablated baseline (2/8, no grounding)"
)

# Get absolute path to project root (container path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Split GPUs into array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Found ${#EXPERIMENTS[@]} experiments to evaluate"
echo "Using $NUM_GPUS GPUs for parallel execution"
echo ""

# Launch evaluation for each experiment in background
PIDS=()
idx=0

for i in "${!EXPERIMENTS[@]}"; do
    EXP_PATH="${PROJECT_ROOT}/${EXPERIMENTS[$i]}"
    DESC="${DESCRIPTIONS[$i]}"
    
    # Check if experiment exists
    if [ ! -d "$EXP_PATH" ]; then
        echo "⚠️  SKIPPING: $DESC - folder not found"
        echo "    Expected: $EXP_PATH"
        continue
    fi
    
    # Assign GPU (round-robin)
    GPU_ID=${GPU_ARRAY[$((idx % NUM_GPUS))]}
    
    # Log file for this experiment
    EXP_LOG="${EXP_PATH}/dual_eval_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[$((idx+1))/${#EXPERIMENTS[@]}] Launching: $DESC"
    echo "    Path: $EXP_PATH"
    echo "    GPU: $GPU_ID"
    echo "    Log: $EXP_LOG"
    
    # Launch evaluation in background using run_benchmarks_paper.py
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$SCRIPT_DIR/run_benchmarks_paper.py" \
        --mode evaluation \
        --experiment_folder "$EXP_PATH" \
        --dual_eval \
        --gpus "$GPU_ID" \
        > "$EXP_LOG" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "    PID: $PID"
    echo ""
    
    idx=$((idx + 1))
    
    # Small delay to stagger launches
    sleep 2
done

echo "================================================================================"
echo "LAUNCHED ${#PIDS[@]} EVALUATION PROCESSES IN BACKGROUND"
echo "================================================================================"
echo ""
echo "Process IDs: ${PIDS[@]}"
echo ""
echo "Monitor progress with:"
echo "  tail -f $PROJECT_ROOT/saved_model/gtransformer_*/dual_eval_*.log"
echo ""
echo "Check running processes:"
echo "  ps -fp ${PIDS[@]}"
echo ""
echo "Wait for all to complete:"
echo "  wait ${PIDS[@]}"
echo ""
echo "Kill all processes:"
echo "  kill ${PIDS[@]}"
echo "================================================================================"

# Optionally wait for all processes to complete
if [ "$WAIT_FLAG" == "--wait" ]; then
    echo ""
    echo "Waiting for all evaluations to complete..."
    wait ${PIDS[@]}
    echo ""
    echo "================================================================================"
    echo "ALL EVALUATIONS COMPLETED"
    echo "================================================================================"
    
    # Collect and display results
    echo ""
    python3 << 'PYTHON_EOF'
import json
from pathlib import Path
import os

experiments = {
    '090230': 'saved_model/gtransformer_assist2009_qid_20260115_090230',
    '334772': 'saved_model/gtransformer_assist2009_qid_20260116_101107_gtransformer_assist2009_0_seed_42_d_model_64_n_heads_8_n_blocks_2_learning_rate_0.001_334772',
    '948799': 'saved_model/gtransformer_assist2009_qid_20260116_120815_gtransformer_assist2009_0_seed_42_d_model_64_n_heads_8_n_blocks_2_learning_rate_0.001_948799',
    '133835': 'saved_model/gtransformer_assist2009_qid_20260115_133835'
}

project_root = Path(os.environ.get('PROJECT_ROOT', '.'))

print('DUAL EVALUATION RESULTS')
print('=' * 80)
print(f"{'Exp ID':<10} {'p_sup AUC':<12} {'p_ref AUC':<12} {'Δ (sup-ref)':<12} {'Grounded'}")
print('-' * 80)

for exp_id, exp_path in experiments.items():
    full_path = project_root / exp_path
    results_file = full_path / 'eval_results.json'
    
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        
        sup_auc = data.get('testauc_supervised', data.get('testauc', 0))
        ref_auc = data.get('testauc_reference')
        gap = data.get('interpretability_gap')
        grounded = '✅' if data.get('grounded', False) else '❌'
        
        ref_str = f'{ref_auc:.4f}' if ref_auc else 'N/A'
        gap_str = f'{gap:.4f}' if gap else 'N/A'
        
        print(f'{exp_id:<10} {sup_auc:<12.4f} {ref_str:<12} {gap_str:<12} {grounded}')
    else:
        print(f'{exp_id:<10} Results file not found')

print('=' * 80)
PYTHON_EOF
fi
