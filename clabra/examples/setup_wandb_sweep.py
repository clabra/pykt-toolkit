#!/usr/bin/env python3
"""
GainAKT2 WandB Sweep Configuration and Management Tool
"""

import json
import os
import sys
import subprocess
import getpass
from pathlib import Path

def setup_wandb_config():
    """Setup WandB configuration interactively"""
    config_path = Path("../configs/wandb.json")
    
    print("🔧 WandB Configuration Setup")
    print("=" * 30)
    
    # Check if config already exists
    if config_path.exists():
        with open(config_path) as f:
            existing_config = json.load(f)
        
        if existing_config.get("api_key", "") != "your_wandb_api_key_here":
            print("✅ WandB config already exists")
            return existing_config
    
    print("\n📝 Please provide your WandB credentials:")
    print("   1. Go to: https://wandb.ai/settings")
    print("   2. Copy your API key\n")
    
    # Get user input
    username = input("WandB Username: ").strip()
    api_key = getpass.getpass("WandB API Key: ").strip()
    project = input("Project Name (default: gainakt2-optimization): ").strip() or "gainakt2-optimization"
    
    # Create config
    config = {
        "uid": username,
        "api_key": api_key,
        "project": project
    }
    
    # Save config
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Configuration saved to {config_path}")
    return config

def test_wandb_connection(config):
    """Test WandB connection"""
    print("\n🔍 Testing WandB connection...")
    
    try:
        import wandb
        wandb.login(key=config["api_key"])
        print("✅ WandB connection successful!")
        return True
    except Exception as e:
        print(f"❌ WandB connection failed: {e}")
        return False

def create_sweep():
    """Create and return sweep configuration"""
    print("\n🚀 Creating optimized parameter sweep...")
    
    sweep_config = {
        "program": "wandb_gainakt2_train.py",
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "validauc"
        },
        "parameters": {
            "model_name": {"value": "gainakt2"},
            "dataset_name": {"value": "assist2015"},
            "emb_type": {"value": "qid"},
            "save_dir": {"value": "swept_models/gainakt2_optimized"},
            "seed": {"values": [42, 3407]},
            "fold": {"values": [0, 1, 2, 3, 4]},
            "num_epochs": {"value": 10},
            "use_wandb": {"value": 1},
            "add_uuid": {"value": 1},
            
            # Optimized ranges based on benchmark
            "d_model": {"values": [256, 384]},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 5e-4
            },
            "dropout": {"values": [0.15, 0.2, 0.25]},
            "num_encoder_blocks": {"values": [3, 4, 5]},
            "d_ff": {"values": [512, 768, 1024]},
            "n_heads": {"values": [8, 12]},
            "seq_len": {"value": 200}
        }
    }
    
    return sweep_config

def launch_sweep(config, sweep_config):
    """Launch WandB sweep"""
    try:
        import wandb
        
        # Initialize sweep
        os.environ["WANDB_API_KEY"] = config["api_key"]
        
        sweep_id = wandb.sweep(
            sweep_config, 
            project=config["project"]
        )
        
        print(f"✅ Sweep created: {sweep_id}")
        print(f"🌐 Monitor at: https://wandb.ai/{config['uid']}/{config['project']}")
        
        return sweep_id
        
    except Exception as e:
        print(f"❌ Failed to create sweep: {e}")
        return None

def start_agents(config, sweep_id, num_agents=2):
    """Start sweep agents with GPU allocation"""
    print(f"\n🎯 Starting {num_agents} optimization agents...")
    
    agents = []
    for i in range(num_agents):
        # Distribute GPUs across agents
        gpu_set = f"{i % 4},{(i+1) % 4}"
        
        cmd = [
            "bash", "-c",
            f"CUDA_VISIBLE_DEVICES={gpu_set} WANDB_API_KEY={config['api_key']} wandb agent {config['uid']}/{config['project']}/{sweep_id}"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        agents.append(process)
        
        print(f"  Agent {i+1} started with GPUs: {gpu_set}")
    
    print(f"\n🔥 {num_agents} agents launched!")
    print("💡 Stop with: pkill -f 'wandb agent'")
    
    return agents

def main():
    """Main execution"""
    print("🚀 GainAKT2 WandB Sweep Optimizer")
    print("=" * 35)
    
    # Install wandb if needed
    try:
        import wandb
    except ImportError:
        print("📦 Installing WandB...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb
    
    # Setup configuration
    config = setup_wandb_config()
    
    # Test connection
    if not test_wandb_connection(config):
        print("❌ Please check your WandB credentials and try again.")
        return
    
    # Create sweep
    sweep_config = create_sweep()
    sweep_id = launch_sweep(config, sweep_config)
    
    if not sweep_id:
        print("❌ Failed to create sweep")
        return
    
    # Ask to start agents
    start_agents_choice = input("\n🤖 Start optimization agents now? (Y/n): ").strip().lower()
    if start_agents_choice in ['', 'y', 'yes']:
        num_agents = int(input("Number of agents (default 2): ") or "2")
        start_agents(config, sweep_id, num_agents)
        
        print("\n✨ GainAKT2 parameter optimization is now running!")
        print("\n🎯 What's being optimized:")
        print("   • Model dimensions: [256, 384]")
        print("   • Learning rates: 1e-4 to 5e-4 (Bayesian)")
        print("   • Dropout: [0.15, 0.2, 0.25]")
        print("   • Encoder blocks: [3, 4, 5]")
        print("   • Feed-forward dims: [512, 768, 1024]")
        print("   • All 5 folds for robust results")
    else:
        print(f"\n📋 Sweep created but not started.")
        print(f"   Run agents with: wandb agent {config['uid']}/{config['project']}/{sweep_id}")

if __name__ == "__main__":
    main()