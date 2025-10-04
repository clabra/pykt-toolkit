#!/bin/bash

# GainAKT2 WandB Sweep Setup and Launch Script
# This script helps configure WandB and launch parameter optimization sweeps

echo "🔥 GainAKT2 WandB Sweep Setup & Launch"
echo "======================================"

# Function to setup WandB configuration
setup_wandb() {
    echo "📝 Setting up WandB configuration..."
    
    if [ ! -f "../configs/wandb.json" ]; then
        echo "❌ WandB config not found. Creating template..."
        cat > ../configs/wandb.json << EOF
{
    "uid": "your_wandb_username",
    "api_key": "your_wandb_api_key_here",
    "project": "gainakt2-optimization"
}
EOF
    fi
    
    echo "🌐 Please configure your WandB credentials:"
    echo "   1. Visit: https://wandb.ai/settings"
    echo "   2. Copy your API key"
    echo "   3. Edit: configs/wandb.json"
    echo ""
    
    read -p "Have you configured WandB credentials? (y/N): " configured
    if [[ ! "$configured" =~ ^[Yy]$ ]]; then
        echo "⚠️  Please configure WandB first, then run this script again."
        exit 1
    fi
}

# Function to validate WandB setup
validate_wandb() {
    echo "✅ Validating WandB setup..."
    
    # Extract API key
    API_KEY=$(python3 -c "
import json
try:
    with open('../configs/wandb.json') as f:
        config = json.load(f)
        print(config['api_key'])
except:
    print('ERROR')
")
    
    if [ "$API_KEY" = "ERROR" ] || [ "$API_KEY" = "your_wandb_api_key_here" ]; then
        echo "❌ Invalid WandB configuration"
        return 1
    fi
    
    export WANDB_API_KEY="$API_KEY"
    
    # Test WandB connection
    python3 -c "
import wandb
try:
    wandb.login(key='$API_KEY')
    print('✅ WandB connection successful')
except Exception as e:
    print(f'❌ WandB connection failed: {e}')
    exit(1)
"
    
    return $?
}

# Function to create and launch sweep
launch_sweep() {
    echo "🚀 Launching optimized parameter sweep..."
    
    # Create sweep
    echo "📊 Creating WandB sweep..."
    SWEEP_OUTPUT=$(WANDB_API_KEY="$API_KEY" wandb sweep seedwandb/gainakt2_optimized_sweep.yaml --project gainakt2-optimization 2>&1)
    echo "$SWEEP_OUTPUT"
    
    # Extract sweep ID
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^/]+/[^/]+/[^ ]+' | tail -1)
    
    if [ -z "$SWEEP_ID" ]; then
        echo "❌ Failed to create sweep"
        echo "Output: $SWEEP_OUTPUT"
        exit 1
    fi
    
    echo "✅ Sweep created: $SWEEP_ID"
    echo "🎯 Starting optimization agents with multi-GPU..."
    
    # Launch agents with GPU allocation
    for i in {1..2}; do
        echo "Starting agent $i with GPUs..."
        GPU_SET=$((i % 4)),$((((i+1) % 4)))  # Distribute GPUs
        
        CUDA_VISIBLE_DEVICES=$GPU_SET WANDB_API_KEY="$API_KEY" wandb agent $SWEEP_ID &
        echo "  Agent $i launched with GPUs: $GPU_SET"
        sleep 3
    done
    
    echo ""
    echo "🔥 Sweep optimization running!"
    echo "📊 Monitor at: https://wandb.ai/$(echo $SWEEP_ID | cut -d'/' -f1)/gainakt2-optimization"
    echo "💡 Stop agents: pkill -f 'wandb agent'"
    echo ""
    echo "🎯 Optimizing parameters:"
    echo "   • Model dimensions: [256, 384]"
    echo "   • Learning rates: 1e-4 to 5e-4 (log scale)"
    echo "   • Dropout: [0.15, 0.2, 0.25]"
    echo "   • Encoder blocks: [3, 4, 5]"
    echo "   • Feed-forward dims: [512, 768, 1024]"
    echo "   • All 5 folds for robust evaluation"
}

# Main execution
main() {
    # Check if WandB is installed
    if ! python3 -c "import wandb" 2>/dev/null; then
        echo "📦 Installing WandB..."
        pip install wandb
    fi
    
    # Setup WandB if needed
    setup_wandb
    
    # Validate configuration
    if ! validate_wandb; then
        echo "❌ WandB validation failed. Please check your configuration."
        exit 1
    fi
    
    # Launch sweep
    launch_sweep
    
    echo "✨ Setup complete! Your GainAKT2 parameter optimization is running."
}

# Parse command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "status")
        echo "📊 Checking sweep status..."
        python3 -c "
import wandb
api = wandb.Api()
runs = api.runs('gainakt2-optimization')
print(f'Active runs: {sum(1 for run in runs if run.state == \"running\")}')
print(f'Completed runs: {sum(1 for run in runs if run.state == \"finished\")}')
"
        ;;
    "stop")
        echo "🛑 Stopping all sweep agents..."
        pkill -f 'wandb agent'
        echo "✅ All agents stopped"
        ;;
    *)
        echo "Usage: $0 [setup|status|stop]"
        echo "  setup  - Configure and launch sweep (default)"
        echo "  status - Check sweep progress"
        echo "  stop   - Stop all running agents"
        ;;
esac