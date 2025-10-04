#!/bin/bash

# Direct GainAKT2 Benchmark - 10 Parameter Combinations
# Multi-GPU training with 3 epochs, results saved locally

echo "🚀 GainAKT2 Direct Multi-GPU Benchmark"
echo "GPUs: 0,1,2,3 | Epochs: 3 | Combinations: 10"
echo "============================================="

# Create results directory
mkdir -p benchmark_results
RESULTS_FILE="benchmark_results/gainakt2_benchmark_$(date +%Y%m%d_%H%M%S).txt"

echo "📊 Benchmark Results - $(date)" > $RESULTS_FILE
echo "=================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Parameter combinations (10 total)
combinations=(
    "128 1e-3 0.1 2 512"    # Combination 1: Small model, high LR
    "128 2e-4 0.1 2 512"    # Combination 2: Small model, medium LR  
    "128 5e-4 0.2 4 1024"   # Combination 3: Small model, medium LR, more blocks
    "256 1e-3 0.1 2 512"    # Combination 4: Large model, high LR
    "256 2e-4 0.1 4 1024"   # Combination 5: Large model, medium LR, more blocks
    "256 5e-4 0.2 2 512"    # Combination 6: Large model, low LR, more dropout
    "128 2e-4 0.2 4 1024"   # Combination 7: Small model, balanced params
    "256 1e-3 0.2 4 1024"   # Combination 8: Large model, high LR, more dropout
    "256 2e-4 0.2 4 512"    # Combination 9: Large model, balanced
    "128 5e-4 0.1 4 1024"   # Combination 10: Small model, low LR, more blocks
)

counter=1
best_auc="0.0"
best_combination=""

for combo in "${combinations[@]}"; do
    # Parse parameters
    IFS=' ' read -r d_model lr dropout blocks d_ff <<< "$combo"
    
    echo "🔥 Running Combination $counter/10"
    echo "   d_model=$d_model, lr=$lr, dropout=$dropout, blocks=$blocks, d_ff=$d_ff"
    
    # Log to results file
    echo "Combination $counter: d_model=$d_model, lr=$lr, dropout=$dropout, blocks=$blocks, d_ff=$d_ff" >> $RESULTS_FILE
    
    # Run training with multi-GPU
    output=$(cd /workspaces/pykt-toolkit/examples && \
        CUDA_VISIBLE_DEVICES=0,1,2,3 python wandb_gainakt2_train.py \
        --dataset_name=assist2015 \
        --use_wandb=0 \
        --d_model=$d_model \
        --learning_rate=$lr \
        --dropout=$dropout \
        --num_encoder_blocks=$blocks \
        --d_ff=$d_ff \
        --num_epochs=3 \
        --seed=42 \
        --fold=0 2>&1)
    
    # Extract metrics from output
    auc=$(echo "$output" | grep -oP "validauc: \K[0-9]+\.[0-9]+" | tail -1)
    acc=$(echo "$output" | grep -oP "validacc: \K[0-9]+\.[0-9]+" | tail -1)
    loss=$(echo "$output" | grep -oP "train loss: \K[0-9]+\.[0-9]+" | tail -1)
    
    if [ ! -z "$auc" ]; then
        echo "   📈 Results: AUC=$auc, ACC=$acc, Loss=$loss"
        
        # Log results
        echo "   Results: AUC=$auc, ACC=$acc, Loss=$loss" >> $RESULTS_FILE
        
        # Check if this is the best combination using Python
        is_better=$(python3 -c "print('yes' if float('$auc') > float('$best_auc') else 'no')")
        if [ "$is_better" = "yes" ]; then
            best_auc=$auc
            best_combination="Combination $counter: d_model=$d_model, lr=$lr, dropout=$dropout, blocks=$blocks, d_ff=$d_ff (AUC=$auc)"
        fi
    else
        echo "   ❌ Failed to extract results"
        echo "   Results: FAILED" >> $RESULTS_FILE
    fi
    
    echo "" >> $RESULTS_FILE
    echo ""
    
    counter=$((counter + 1))
    
    # Small delay between runs
    sleep 5
done

# Final summary
echo "🎯 BENCHMARK COMPLETE!" 
echo "========================"
echo "📊 Best Configuration: $best_combination"
echo ""
echo "📋 Full results saved to: $RESULTS_FILE"

# Add summary to results file
echo "" >> $RESULTS_FILE
echo "SUMMARY:" >> $RESULTS_FILE
echo "========" >> $RESULTS_FILE
echo "Best Configuration: $best_combination" >> $RESULTS_FILE
echo "Benchmark completed: $(date)" >> $RESULTS_FILE

echo ""
echo "🚀 Recommended next steps:"
echo "   1. Use the best configuration for full training"
echo "   2. Run with more epochs (50-200) for final model"
echo "   3. Consider testing on multiple folds for robustness"