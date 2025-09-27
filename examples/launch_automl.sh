#!/bin/bash
# 🚀 GainAKT2 AutoML Launcher Script
# 
# Quick launcher for different AutoML optimization scenarios
# Usage: ./launch_automl.sh [quick|standard|intensive|custom]

set -e

echo "🚀 GainAKT2 AutoML Hyperparameter Optimizer"
echo "=========================================="

# Parse command line arguments
MODE=${1:-standard}

case $MODE in
    "quick")
        echo "⚡ Quick Mode: 20 evaluations, 2 parallel jobs"
        python automl_gainakt2_optimizer.py \
            --max_evaluations 20 \
            --parallel_jobs 2 \
            --early_stopping_patience 5 \
            --target_auc 0.730
        ;;
        
    "standard")
        echo "🎯 Standard Mode: 50 evaluations, 2 parallel jobs" 
        python automl_gainakt2_optimizer.py \
            --max_evaluations 50 \
            --parallel_jobs 2 \
            --early_stopping_patience 10 \
            --target_auc 0.735
        ;;
        
    "intensive")
        echo "🔥 Intensive Mode: 100 evaluations, 3 parallel jobs"
        python automl_gainakt2_optimizer.py \
            --max_evaluations 100 \
            --parallel_jobs 3 \
            --early_stopping_patience 15 \
            --target_auc 0.740
        ;;
        
    "overnight")
        echo "🌙 Overnight Mode: 200 evaluations, 4 parallel jobs"
        python automl_gainakt2_optimizer.py \
            --max_evaluations 200 \
            --parallel_jobs 4 \
            --early_stopping_patience 25 \
            --target_auc 0.745
        ;;
        
    "custom")
        echo "🛠️  Custom Mode: Interactive configuration"
        echo ""
        read -p "Max evaluations (default: 50): " MAX_EVAL
        MAX_EVAL=${MAX_EVAL:-50}
        
        read -p "Parallel jobs (default: 2): " PARALLEL
        PARALLEL=${PARALLEL:-2}
        
        read -p "Early stopping patience (default: 10): " PATIENCE  
        PATIENCE=${PATIENCE:-10}
        
        read -p "Target AUC (default: 0.735): " TARGET
        TARGET=${TARGET:-0.735}
        
        echo "Running with: evaluations=$MAX_EVAL, parallel=$PARALLEL, patience=$PATIENCE, target=$TARGET"
        python automl_gainakt2_optimizer.py \
            --max_evaluations $MAX_EVAL \
            --parallel_jobs $PARALLEL \
            --early_stopping_patience $PATIENCE \
            --target_auc $TARGET
        ;;
        
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  quick     - 20 evaluations (fast testing)"
        echo "  standard  - 50 evaluations (balanced)"  
        echo "  intensive - 100 evaluations (thorough)"
        echo "  overnight - 200 evaluations (extensive)"
        echo "  custom    - Interactive configuration"
        echo ""
        echo "Usage: ./launch_automl.sh [mode]"
        exit 1
        ;;
esac

echo ""
echo "✅ AutoML optimization complete!"
echo "📊 Check the automl_results/ directory for detailed results"