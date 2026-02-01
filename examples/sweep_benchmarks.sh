#!/bin/bash
################################################################################
# Hyperparameter Sweep for run_benchmarks_paper.py
# 
# Description:
#   Runs multiple benchmark campaigns with different hyperparameter combinations
#   for gTransformer model. Each combination is executed as a separate campaign
#   with only fold 0 and a fixed seed for reproducible quick evaluation.
#
# Usage:
#   chmod +x examples/sweep_benchmarks.sh
#   ./examples/sweep_benchmarks.sh CAMPAIGN_NAME PHASE_MODE
#   nohup ./examples/sweep_benchmarks.sh campaign1 phase1 > sweep.log 2>&1 &
#   tail -f sweep.log  # Monitor progress
#
# Parameters:
#   CAMPAIGN_NAME (required): Name to identify this sweep campaign
#                            Used to group experiments and skip already completed ones
#   PHASE_MODE (required): Which phase(s) to run: phase1, phase2, or all
#                         - phase1: Quick evaluation (2 epochs for all configs)
#                         - phase2: Full training (200 epochs for top 3 per dataset)
#                         - all: Run both phases sequentially
#
# Customization:
#   Edit the arrays below to define your parameter search space:
#   - NUM_BLOCKS: Number of transformer blocks
#   - NUM_HEADS: Number of attention heads per block
#   - EMB_SIZES: Embedding dimension
#   - DATASETS: Dataset(s) to use (comma-separated for multiple)
#   - GPUS: GPU IDs to use (comma-separated)
#
# Output:
#   Each combination creates a separate experiment directory:
#   experiments/YYYYMMDD_HHMMSS_<campaign>_b<blocks>_h<heads>_e<emb>_<id>/
#
# Example:
#   # Phase 1 only (quick evaluation)
#   ./examples/sweep_benchmarks.sh sweep_init phase1
#   
#   # Phase 2 only (full training of top configs)
#   ./examples/sweep_benchmarks.sh sweep_init phase2
#   
#   # Both phases
#   ./examples/sweep_benchmarks.sh sweep_init all
#
################################################################################

# Check for required parameters
if [ $# -lt 2 ]; then
    echo "Error: Missing required parameters"
    echo ""
    echo "Usage: $0 CAMPAIGN_NAME PHASE_MODE"
    echo ""
    echo "Parameters:"
    echo "  CAMPAIGN_NAME: Name to identify this sweep campaign (e.g., sweep_init)"
    echo "  PHASE_MODE:    Which phase(s) to run: phase1, phase2, or all"
    echo ""
    echo "Examples:"
    echo "  $0 sweep_init phase1   # Quick evaluation only"
    echo "  $0 sweep_init phase2   # Full training only"
    echo "  $0 sweep_init all      # Both phases"
    echo ""
    exit 1
fi

# Campaign name (required)
CAMPAIGN_NAME="$1"

# Phase selection (required): phase1, phase2, or all
PHASE_MODE="$2"

# Validate PHASE_MODE
if [[ ! "$PHASE_MODE" =~ ^(phase1|phase2|all)$ ]]; then
    echo "Error: Invalid PHASE_MODE '$PHASE_MODE'"
    echo "Must be one of: phase1, phase2, all"
    echo ""
    echo "Usage: $0 CAMPAIGN_NAME PHASE_MODE"
    exit 1
fi

# Configuration
#DATASETS="assist2009 assist2015 algebra2005 bridge2algebra2006 nips_task34"
DATASETS="assist2015"
GPUS="1,2,3,4,5,6,7"  # Use 7 V100 GPUs (32GB each)
MODE="training"  # training, evaluation, or results

# Parameter search space: LR + Dropout sweep for assist2015 (ablation=all)
# Target: Beat p_sup=0.7078 (exp 589915) and AKT baseline 0.7081
# Sweep: lr=[2e-4, 3e-4] × dropout=[0.1, 0.15] × epochs=2 (quick test)
# Fixed params: blocks=4, heads=4, emb_dim=64, lambda_ref=0 (ablation=all)
# Total combinations: 2×2 = 4 configs
NUM_BLOCKS=(4)              # Fixed: 4 blocks (589915 baseline)
NUM_HEADS=(4)               # Fixed: 4 heads (589915 baseline)
EMB_SIZES=(64)              # Fixed: 64 emb_dim (589915 baseline)
LAMBDA_REF=(0)              # Fixed: 0 for ablation=all (black-box baseline)
DROPOUT=(0.1 0.15)          # Sweep: 0.1 (baseline), 0.15 (new)
LEARNING_RATES=(2e-4 3e-4)  # Sweep: 2e-4 (faster convergence), 3e-4 (aggressive)
ABLATION=(all)

# Refined search: 1×2×2×2×2×2 = 32 combinations (down from 108, 70% reduction)
# Sweep: 1×1×1×1×2×2 = 4 configs total (quick LR+dropout test)

# Optional: Limit total runs (comment out for full grid search)
MAX_RUNS=4 # per dataset (all 4 configs for this quick sweep)
RUN_COUNT=0

# Two-phase strategy: Quick evaluation → Select top configs → Full training
# Phase 1: Test ALL 4 combinations with 2 epochs (assist2015 only)
# Phase 2: Take top 1-2 configs and train to convergence (~70 epochs)
INITIAL_EPOCHS=2           # Quick evaluation epochs (all 4 combinations)
FULL_EPOCHS=70             # Full training epochs (top 1-2 from phase 1)

# Reproducibility settings for sweep (Phase 1)
SWEEP_FOLD=0               # Use only fold 0 for quick evaluation
SWEEP_SEED=42              # Fixed seed for reproducibility

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo "=========================================================================="
    echo "$1"
    echo "=========================================================================="
}

print_config() {
    echo "Total combinations: $1"
    echo "Datasets: $DATASETS"
    echo "GPUs: $GPUS"
    echo "Mode: $MODE"
    if [ -n "$MAX_RUNS" ]; then
        echo "Max runs: $MAX_RUNS (grid search limited)"
    else
        echo "Max runs: unlimited (full grid search)"
    fi
}

################################################################################
# Main Sweep Loop
################################################################################

print_header "HYPERPARAMETER SWEEP FOR gTransformer"

echo "Campaign: $CAMPAIGN_NAME"
echo "Phase mode: $PHASE_MODE"
echo ""

# Calculate total combinations
TOTAL_COMBOS=$((${#NUM_BLOCKS[@]} * ${#NUM_HEADS[@]} * ${#EMB_SIZES[@]} * ${#LAMBDA_REF[@]} * ${#DROPOUT[@]} * ${#LEARNING_RATES[@]}))
DATASET_ARRAY=($DATASETS)
NUM_DATASETS=${#DATASET_ARRAY[@]}
TOTAL_FULL_RUNS=$((MAX_RUNS * NUM_DATASETS))

echo "Total combinations to evaluate: $TOTAL_COMBOS"
echo "Datasets: ${DATASET_ARRAY[@]}"
echo "Top performers per dataset: $MAX_RUNS"
echo "Total full training runs: $TOTAL_FULL_RUNS (${MAX_RUNS} × ${NUM_DATASETS} datasets)"
echo "GPUs: $GPUS"
echo "Mode: $MODE"
echo ""
echo "* PHASE 1: Quick evaluation ($INITIAL_EPOCHS epochs for all $TOTAL_COMBOS configs on all datasets)"
echo "* PHASE 2: Full training ($FULL_EPOCHS epochs for top $MAX_RUNS configs per dataset)"

echo ""
echo "Starting sweep... (output logged to experiments/${CAMPAIGN_NAME}_*.log)"
echo ""

START_TIME=$(date +%s)

# Generate all combinations
ALL_COMBINATIONS=()
for blocks in "${NUM_BLOCKS[@]}"; do
  for heads in "${NUM_HEADS[@]}"; do
    for emb in "${EMB_SIZES[@]}"; do
      for lambda in "${LAMBDA_REF[@]}"; do
        for dropout in "${DROPOUT[@]}"; do
          for lr in "${LEARNING_RATES[@]}"; do
            ALL_COMBINATIONS+=("$blocks $heads $emb $lambda $dropout $lr")
          done
        done
      done
    done
  done
done
################################################################################
# PHASE 1: Quick Evaluation of All Combinations (IN PARALLEL)
################################################################################

if [ "$PHASE_MODE" != "phase2" ]; then

print_header "* PHASE 1: QUICK EVALUATION (${INITIAL_EPOCHS} epochs each) - PARALLEL"
print_header "* PHASE 1: QUICK EVALUATION (${INITIAL_EPOCHS} epochs each) - PARALLEL"

# Array to store PIDs and combo info
declare -a PHASE1_PIDS
declare -a PHASE1_COMBOS
declare -a PHASE1_TITLES

# GPU distribution: round-robin across available GPUs
GPU_ARRAY=(${GPUS//,/ })  # Convert comma-separated list to array
NUM_GPUS=${#GPU_ARRAY[@]}
MAX_JOBS_PER_GPU=4  # 7 GPUs × 4 jobs = 28 parallel (leaves ~2 CPU cores for others)
MAX_PARALLEL=$((NUM_GPUS * MAX_JOBS_PER_GPU))

echo "Distributing jobs across ${NUM_GPUS} GPUs: ${GPU_ARRAY[@]}"
echo "Max concurrent jobs: ${MAX_PARALLEL} (${MAX_JOBS_PER_GPU} per GPU)"
echo "Strategy: Each job processes 1 dataset (better parallelization)"
echo ""

RUN_COUNT=0
SKIPPED_COUNT=0
for combo in "${ALL_COMBINATIONS[@]}"; do
  # Parse combination
  read -r blocks heads emb lambda dropout lr <<< "$combo"
  
  # Format learning rate for filename
  LR_STR=$(echo "$lr" | awk '{printf "%.0e", $1}' | sed 's/e-0*/e/')
  
  # Create unique campaign title with campaign name
  TITLE="${CAMPAIGN_NAME}_b${blocks}_h${heads}_e${emb}_d${dropout}_lr${LR_STR}_lam${lambda}"
  
  # Check if this combination already completed ALL datasets (has experiment dirs with INITIAL_EPOCHS results)
  EXISTING_RUNS=$(find experiments -type d -name "*${TITLE}_*" 2>/dev/null)
  if [ ! -z "$EXISTING_RUNS" ]; then
    # Check if it actually completed INITIAL_EPOCHS across all datasets
    COMPLETED=$(find $EXISTING_RUNS -name "epoch_metrics.csv" -type f 2>/dev/null | xargs grep -l "^${INITIAL_EPOCHS}," 2>/dev/null | wc -l)
    # With fold 0 only: expect 1 fold per dataset × 5 datasets = 5 total folds
    if [ $COMPLETED -ge 5 ]; then
      SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
      echo "* Skipping combo ${SKIPPED_COUNT}: ${TITLE} (already completed with $COMPLETED folds)"
      continue
    fi
  fi
  
  # Launch separate job for each dataset
  for dataset in "${DATASET_ARRAY[@]}"; do
    RUN_COUNT=$((RUN_COUNT + 1))
    
    # Wait if we have too many jobs running (check actual processes, not jobs -r which doesn't work in nohup)
    while [ $(ps aux | grep "run_benchmarks_paper.py" | grep -v grep | wc -l) -ge $MAX_PARALLEL ]; do
      sleep 10
    done
    
    # Assign GPU in round-robin fashion
    GPU_IDX=$(( (RUN_COUNT - 1) % NUM_GPUS ))
    ASSIGNED_GPU=${GPU_ARRAY[$GPU_IDX]}
    
    echo "* Launching ${RUN_COUNT}/$((TOTAL_COMBOS * NUM_DATASETS - SKIPPED_COUNT * NUM_DATASETS)): ${dataset} | b${blocks}_h${heads}_e${emb}_lam${lambda}_d${dropout}_lr${lr} [GPU ${ASSIGNED_GPU}]"
    
    # Create log file with campaign name and dataset
    LOG_FILE="experiments/${CAMPAIGN_NAME}_${TITLE}_${dataset}.log"
    
    # Run quick evaluation ON SINGLE DATASET IN BACKGROUND with assigned GPU
    # Using only fold 0 with fixed seed for reproducibility
    python examples/run_benchmarks_paper.py \
      --mode $MODE \
      --model gtransformer \
      --dataset $dataset \
      --gpus $ASSIGNED_GPU \
      --short_title $TITLE \
      --fold $SWEEP_FOLD \
      --seed $SWEEP_SEED \
      --n_blocks $blocks \
      --num_attn_heads $heads \
      --emb_size $emb \
      --dropout $dropout \
      --learning_rate $lr \
      --lambda_ref $lambda \
      --epochs $INITIAL_EPOCHS \
      > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PHASE1_PIDS+=($PID)
    PHASE1_COMBOS+=("$combo")
    PHASE1_TITLES+=("$TITLE")
    
    echo "  Started in background (PID: $PID)"
    
    # Small delay to avoid overwhelming the system
    sleep 0.5
  done
done

echo ""
echo "All ${TOTAL_COMBOS} Phase 1 jobs launched. Waiting for completion..."
echo ""

# Wait for all Phase 1 jobs to complete
COMPLETED=0
TOTAL=${#PHASE1_PIDS[@]}
for pid in "${PHASE1_PIDS[@]}"; do
  wait $pid
  COMPLETED=$((COMPLETED + 1))
  echo "Progress: $COMPLETED/$TOTAL jobs completed"
done

echo ""
echo "All Phase 1 jobs completed! Collecting results..."
echo ""

# Array to store results per dataset: "dataset|combo|auc|title"
declare -A RESULTS_BY_DATASET

# Now collect results from all completed jobs
for i in "${!PHASE1_COMBOS[@]}"; do
  combo="${PHASE1_COMBOS[$i]}"
  TITLE="${PHASE1_TITLES[$i]}"
  
  # Find experiment folder
  EXPERIMENT_FOLDER=$(ls -td experiments/*${TITLE}* 2>/dev/null | head -1)
  
  if [ -n "$EXPERIMENT_FOLDER" ] && [ -d "$EXPERIMENT_FOLDER" ]; then
    # Extract AUC per dataset
    for dataset in "${DATASET_ARRAY[@]}"; do
      DATASET_AUC=0.0
      FOLD_COUNT=0
      
      # Average AUC across all folds for this dataset
      for fold_dir in "$EXPERIMENT_FOLDER"/gtransformer/${dataset}/fold_*; do
        if [ -f "$fold_dir/results.json" ]; then
          FOLD_AUC=$(python -c "import json; print(json.load(open('$fold_dir/results.json'))['test_auc'])" 2>/dev/null || echo "0.0")
          DATASET_AUC=$(python -c "print($DATASET_AUC + $FOLD_AUC)")
          FOLD_COUNT=$((FOLD_COUNT + 1))
        fi
      done
      
      if [ $FOLD_COUNT -gt 0 ]; then
        AVG_AUC=$(python -c "print($DATASET_AUC / $FOLD_COUNT)")
        # Store result for this dataset
        RESULTS_BY_DATASET["$dataset"]+="$combo|$AVG_AUC|$TITLE"$'\n'
      else
        RESULTS_BY_DATASET["$dataset"]+="$combo|0.0|$TITLE"$'\n'
      fi
    done
  else
    # No results found
    for dataset in "${DATASET_ARRAY[@]}"; do
      RESULTS_BY_DATASET["$dataset"]+="$combo|0.0|$TITLE (no results)"$'\n'
    done
  fi
done

fi  # End of PHASE_MODE != "phase2" check

################################################################################
# PHASE 2: Select Top Performers Per Dataset and Run Full Training
################################################################################

if [ "$PHASE_MODE" != "phase1" ]; then

print_header "* PHASE 2: SELECTING TOP $MAX_RUNS PERFORMERS PER DATASET"########

print_header "* PHASE 2: SELECTING TOP $MAX_RUNS PERFORMERS PER DATASET"

# For each dataset, rank and select top configs
ALL_FULL_TRAINING_JOBS=()

for dataset in "${DATASET_ARRAY[@]}"; do
  echo ""
  echo "========================================================================"
  echo "Dataset: $dataset"
  echo "========================================================================"
  
  # Get results for this dataset and sort by AUC
  DATASET_RESULTS="${RESULTS_BY_DATASET[$dataset]}"
  
  if [ -z "$DATASET_RESULTS" ]; then
    echo "WARNING  No results found for $dataset"
    continue
  fi
  
  # Sort and take top MAX_RUNS
  TOP_CONFIGS=($(echo "$DATASET_RESULTS" | sort -t'|' -k2 -rn | head -n $MAX_RUNS))
  
  echo "Top $MAX_RUNS configurations for $dataset:"
  echo "----------------------------------------------------------------------"
  RANK=1
  for result in "${TOP_CONFIGS[@]}"; do
    IFS='|' read -r combo auc title <<< "$result"
    echo "$RANK. AUC=$auc | $combo"
    ALL_FULL_TRAINING_JOBS+=("$dataset|$combo|$auc")
    RANK=$((RANK + 1))
  done
done

echo ""
print_header "* PHASE 2: FULL TRAINING (${FULL_EPOCHS} epochs)"
echo "Total jobs to launch: ${#ALL_FULL_TRAINING_JOBS[@]}"
echo "GPU distribution: round-robin across ${GPU_ARRAY[@]}"
echo "Max concurrent jobs: ${MAX_PARALLEL} (${MAX_JOBS_PER_GPU} per GPU)"
echo ""

RUN_COUNT=0
for job in "${ALL_FULL_TRAINING_JOBS[@]}"; do
  IFS='|' read -r dataset combo auc <<< "$job"
  read -r blocks heads emb lambda dropout lr <<< "$combo"
  
  RUN_COUNT=$((RUN_COUNT + 1))
  
  # Wait if we have too many jobs running (check actual processes, not jobs -r which doesn't work in nohup)
  while [ $(ps aux | grep "run_benchmarks_paper.py" | grep -v grep | wc -l) -ge $MAX_PARALLEL ]; do
    sleep 10
  done
  
  # Assign GPU in round-robin fashion for Phase 2
  GPU_IDX=$(( (RUN_COUNT - 1) % NUM_GPUS ))
  ASSIGNED_GPU=${GPU_ARRAY[$GPU_IDX]}
  
  # Format learning rate for filename
  LR_STR=$(echo "$lr" | awk '{printf "%.0e", $1}' | sed 's/e-0*/e/')
  
  # Create title for full training with campaign name
  FULL_TITLE="${CAMPAIGN_NAME}_full_${dataset}_b${blocks}_h${heads}_e${emb}_d${dropout}_lr${LR_STR}_lam${lambda}"
  
  echo ""
  echo "=========================================================================="
  echo "* FULL TRAINING ${RUN_COUNT}/${#ALL_FULL_TRAINING_JOBS[@]}: $dataset | b${blocks}_h${heads}_e${emb}_lam${lambda}_d${dropout}_lr${lr} [GPU ${ASSIGNED_GPU}]"
  echo "   Initial AUC: $auc"
  echo "=========================================================================="
  
  # Run full training in background (single dataset) with assigned GPU
  python examples/run_benchmarks_paper.py \
    --mode $MODE \
    --model gtransformer \
    --dataset $dataset \
    --gpus $ASSIGNED_GPU \
    --short_title $FULL_TITLE \
    --fold $SWEEP_FOLD \
    --seed $SWEEP_SEED \
    --n_blocks $blocks \
    --num_attn_heads $heads \
    --emb_size $emb \
    --dropout $dropout \
    --learning_rate $lr \
    --lambda_ref $lambda \
    --epochs $FULL_EPOCHS \
    > "experiments/${CAMPAIGN_NAME}_${FULL_TITLE}.log" 2>&1 &
  
  PID=$!
  echo "OK Started in background (PID: $PID)"
  
echo ""
echo "All full training jobs launched. Monitor with:"
echo "  ps aux | grep run_benchmarks_paper.py"
echo ""

fi  # End of PHASE_MODE != "phase1" check

################################################################################
# Summary
################################################################################

################################################################################
# Summary
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_header "SWEEP COMPLETE"

echo "Runs completed: $RUN_COUNT"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results location: experiments/"
echo ""
echo "Next steps:"
echo "  1. Evaluate models: ./examples/sweep_benchmarks.sh (set MODE='evaluation')"
echo "  2. Analyze results: python examples/analyze_sweep_results.py"
echo "  3. Generate plots: python examples/run_benchmarks_paper.py --mode results"
echo ""

print_header "To compare sweep results"

cat << 'EOF'
# Find best performing configuration by AUC
for dir in experiments/*/; do
  if [ -f "$dir/cv_results.json" ]; then
    echo "$dir: $(jq -r '.test_mean_auc // "N/A"' $dir/cv_results.json)"
  fi
done | sort -t: -k2 -rn | head -10

# Or use Python for detailed analysis
python -c "
import json
import glob
from pathlib import Path

results = []
for exp_dir in glob.glob('experiments/*/cv_results.json'):
    with open(exp_dir) as f:
        data = json.load(f)
        results.append({
            'experiment': Path(exp_dir).parent.name,
            'test_auc': data.get('test_mean_auc', 0),
            'test_acc': data.get('test_mean_acc', 0)
        })

# Sort by AUC
results.sort(key=lambda x: x['test_auc'], reverse=True)

print('\\nTop 5 Configurations by Test AUC:')
print('-' * 80)
for i, r in enumerate(results[:5], 1):
    print(f'{i}. {r[\"experiment\"]}')
    print(f'   AUC: {r[\"test_auc\"]:.4f}, ACC: {r[\"test_acc\"]:.4f}')
    print()
"
EOF

echo ""
