#!/bin/bash

# GainAKT2 Benchmark Launch Script
# This script launches a WandB sweep for optimal parameter combinations
# Using 4 GPUs (0,1,2,3) and 3 epochs per combination

echo "🚀 Launching GainAKT2 Multi-GPU Benchmark Sweep"
echo "GPUs: 0,1,2,3 | Epochs: 3 | Combinations: ~48"
echo "============================================="

# Set multi-GPU environment
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Check if wandb config exists
if [ ! -f "../configs/wandb.json" ]; then
    echo "❌ Error: wandb.json not found in configs/"
    echo "Please configure your WandB API key first"
    exit 1
fi

# Extract API key from config
WANDB_API_KEY=$(python3 -c "
import json
with open('../configs/wandb.json') as f:
    config = json.load(f)
    print(config['api_key'])
")

if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ Error: Could not extract WandB API key"
    exit 1
fi

echo "✅ WandB API key configured"
echo "📊 Starting sweep initialization..."

# Create the sweep and capture sweep ID
SWEEP_OUTPUT=$(WANDB_API_KEY=$WANDB_API_KEY wandb sweep seedwandb/gainakt2_benchmark.yaml --project pykt-benchmark 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^/]+/[^/]+/[^ ]+' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Error: Could not extract sweep ID"
    echo "Output was: $SWEEP_OUTPUT"
    exit 1
fi

echo "✅ Sweep created: $SWEEP_ID"
echo "🎯 Starting benchmark agents..."

# Launch multiple agents for faster processing
for i in {1..4}; do
    echo "Starting agent $i..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_API_KEY=$WANDB_API_KEY wandb agent $SWEEP_ID &
    sleep 2
done

echo "🔥 All agents launched! Training in progress..."
echo "📈 Monitor progress at: https://wandb.ai/$(echo $SWEEP_ID | cut -d'/' -f1)/pykt-benchmark"
echo ""
echo "💡 To stop all agents: pkill -f 'wandb agent'"
echo "📊 Results will show optimal combinations of:"
echo "   - Learning rates: [1e-3, 2e-4, 5e-4]"
echo "   - Model dimensions: [128, 256]" 
echo "   - Encoder blocks: [2, 4]"
echo "   - Feed-forward dims: [512, 1024]"
echo "   - Dropout rates: [0.1, 0.2]"

wait