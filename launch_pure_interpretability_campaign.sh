#!/bin/bash

# Pure Interpretability Campaign
# Based on Exp 636452 (Active Grounding) but with lambda_sup=0
# This experiment tests the impact of using ONLY interpretability losses (BKT reference + grounding)
# without any direct supervised prediction loss

echo "=========================================="
echo "Pure Interpretability Campaign"
echo "Based on Exp 636452 with lambda_sup=0"
echo "=========================================="
echo ""
echo "Loss Configuration:"
echo "  lambda_sup = 0.0        (NO supervised loss)"
echo "  lambda_ref = 0.5        (BKT reference loss)"
echo "  lambda_initmastery = 0.1 (L0 parameter loss)"
echo "  lambda_rate = 0.1       (T parameter loss)"
echo "  lambda_probe = 1.0      (Active probing loss)"
echo ""
echo "Research Question:"
echo "  Can a model trained PURELY on interpretability losses"
echo "  (BKT logic + grounding constraints) still achieve"
echo "  reasonable predictive performance?"
echo ""
echo "=========================================="
echo ""

# Configuration matching Exp 636452 exactly except lambda_sup
DATASET="assist2009"
MODEL="gtransformer"
ACTIVE_GROUNDING=1
LAMBDA_SUP=0.0
LAMBDA_PROBE=1.0
LAMBDA_REF=0.5
LAMBDA_INIT=0.1
LAMBDA_RATE=0.1
N_BLOCKS=2
N_HEADS=8
D_MODEL=64
D_FF=256
DROPOUT=0.1
LR=0.0001
SEED=3407

# Run 5-fold cross-validation campaign
for FOLD in {0..4}; do
    echo "--------------------------------------"
    echo "Launching Fold $FOLD..."
    echo "--------------------------------------"
    
    python3 examples/run_repro_experiment.py \
        --model_name ${MODEL} \
        --dataset ${DATASET} \
        --fold ${FOLD} \
        --short_title "pure_interpretability" \
        --active_grounding ${ACTIVE_GROUNDING} \
        --lambda_sup ${LAMBDA_SUP} \
        --lambda_probe ${LAMBDA_PROBE} \
        --lambda_ref ${LAMBDA_REF} \
        --lambda_initmastery ${LAMBDA_INIT} \
        --lambda_rate ${LAMBDA_RATE} \
        --n_blocks ${N_BLOCKS} \
        --n_heads ${N_HEADS} \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --dropout ${DROPOUT} \
        --learning_rate ${LR} \
        --seed ${SEED} \
        --num_gpus 1 &
    
    # Small delay to avoid race conditions
    sleep 5
done

echo ""
echo "=========================================="
echo "All 5 folds launched!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  watch -n 5 'nvidia-smi'"
echo "  tail -f experiments/*/fold_*/training.log"
echo ""
echo "Once complete, evaluate with:"
echo "  python3 examples/run_benchmarks_paper.py --mode evaluation --campaign <campaign_folder>"
echo "  python3 examples/run_benchmarks_paper.py --mode results --campaign <campaign_folder>"
echo ""
