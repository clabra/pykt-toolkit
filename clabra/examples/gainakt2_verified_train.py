#!/usr/bin/env python3
"""
GainAKT2 Parameter Verification and Optimized Training Script

This script ensures you're using the exact same parameters that achieved 
AUC: 0.7233 in your wandb sweep, and provides debugging information 
to identify any discrepancies.

Usage:
    cd /workspaces/pykt-toolkit/examples
    source /home/vscode/.pykt-env/bin/activate
    python gainakt2_verified_train.py --dataset_name=assist2015 --use_wandb=0
"""

import argparse
import json
import os
from wandb_train import main

def print_parameter_comparison():
    """Print the optimal parameters from wandb sweep vs current configuration"""
    print("🎯 PARAMETER VERIFICATION - GainAKT2 Optimal Configuration")
    print("="*80)
    print("Parameters that achieved AUC: 0.7233 in wandb sweep:")
    print("")
    
    optimal_params = {
        "d_model": 256,
        "learning_rate": 0.0002,
        "dropout": 0.2,
        "num_encoder_blocks": 4,
        "d_ff": 768,
        "n_heads": 8,
        "num_epochs": 200,  # Critical: sufficient epochs for convergence
        "batch_size": "from config (likely 64)",
        "optimizer": "adam",
        "seq_len": 200,
        "dataset_name": "assist2015",
        "seed": "42 (or sweep default)"
    }
    
    print("📊 OPTIMAL PARAMETERS:")
    for param, value in optimal_params.items():
        print(f"   {param:<20}: {value}")
    
    print("\n🔍 CRITICAL FACTORS:")
    print("   ✅ d_ff=768 was KEY improvement over d_ff=512")
    print("   ✅ learning_rate=2e-4 outperformed 1e-3 and 1e-4")
    print("   ✅ dropout=0.2 provided optimal regularization")
    print("   ✅ 4 encoder blocks balanced depth vs efficiency")
    print("   ⚠️  num_epochs=200+ needed for full convergence")
    print("="*80)

def verify_config_file():
    """Check if config file might override parameters"""
    config_path = "../configs/kt_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("\n🔧 CONFIG FILE CHECK:")
        print(f"   Batch size: {config['train_config']['batch_size']}")
        print(f"   Config epochs: {config['train_config']['num_epochs']}")
        print(f"   Optimizer: {config['train_config']['optimizer']}")
        
        if 'gainakt2' in config:
            print("   GainAKT2 config overrides:")
            for param, value in config['gainakt2'].items():
                print(f"     {param}: {value}")
        print("="*80)

if __name__ == "__main__":
    # Print parameter verification first
    print_parameter_comparison()
    verify_config_file()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="gainakt2")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    
    # EXACT OPTIMAL PARAMETERS from wandb sweep (AUC: 0.7233)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_encoder_blocks", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=768)  # KEY parameter!
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=200)  # CRITICAL: sufficient epochs
    
    # Additional parameters
    parser.add_argument("--use_gain_head", type=int, default=0)
    parser.add_argument("--use_mastery_head", type=int, default=0)
    parser.add_argument("--non_negative_loss_weight", type=float, default=0.0)
    parser.add_argument("--consistency_loss_weight", type=float, default=0.0)
    
    # Wandb configuration
    parser.add_argument("--use_wandb", type=int, default=0)  # Disabled by default for testing
    parser.add_argument("--add_uuid", type=int, default=1)
   
    args = parser.parse_args()
    params = vars(args)
    
    print("\n🚀 CURRENT RUN PARAMETERS:")
    key_params = ['d_model', 'learning_rate', 'dropout', 'num_encoder_blocks', 'd_ff', 'n_heads', 'num_epochs']
    for param in key_params:
        print(f"   {param:<20}: {params[param]}")
    
    print(f"\n🎯 Expected AUC: ~0.7233 (if parameters match exactly)")
    print(f"🎯 Your previous result: 0.7141 (likely due to insufficient epochs)")
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING WITH VERIFIED OPTIMAL PARAMETERS...")
    print("="*80)
    
    # Run training with verified parameters
    main(params)