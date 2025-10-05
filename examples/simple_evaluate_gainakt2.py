#!/usr/bin/env python3

"""
Simple evaluation script for your trained GainAKT2 model
This bypasses some of the complex wandb_predict.py issues
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add the parent directory to path to import pykt
import sys
sys.path.append('..')

from pykt.models import load_model, evaluate
from pykt.datasets import init_test_datasets

def main():
    # Your model directory
    save_dir = "saved_model/assist2015_gainakt2_qid_saved_model_42_0_256_0.0002_8_4_768_0.2_200_200_0_0_0.0_0.0_0_1"
    
    print("🚀 Loading your trained GainAKT2 model...")
    print(f"Model directory: {save_dir}")
    
    # Load config
    with open(os.path.join(save_dir, "config.json")) as f:
        config = json.load(f)
    
    model_config = config["model_config"] 
    params = config["params"]
    data_config = config["data_config"]
    data_config["dataset_name"] = params["dataset_name"]  # Add missing dataset_name
    
    # Clean model config (remove training parameters)
    excluded_params = ['learning_rate', 'use_gain_head', 'use_mastery_head', 
                      'non_negative_loss_weight', 'consistency_loss_weight', 
                      'use_wandb', 'add_uuid', 'num_epochs']
    clean_model_config = {k: v for k, v in model_config.items() if k not in excluded_params}
    
    print(f"Model: {params['model_name']}")
    print(f"Dataset: {params['dataset_name']}")
    print(f"Model config: {clean_model_config}")
    
    # Load model
    model = load_model(params['model_name'], clean_model_config, data_config, params['emb_type'], save_dir)
    print("✅ Model loaded successfully!")
    
    # Initialize test datasets with no workers to avoid CUDA issues
    print("📊 Loading test datasets...")
    test_loader, test_window_loader, _, _ = init_test_datasets(data_config, params['model_name'], batch_size=64)
    
    # Override DataLoader to use no workers
    test_loader = DataLoader(test_loader.dataset, batch_size=64, shuffle=False, num_workers=0)
    test_window_loader = DataLoader(test_window_loader.dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print("🧮 Evaluating model on test set...")
    
    # Evaluate
    save_test_path = os.path.join(save_dir, "simple_test_predictions.txt")
    testauc, testacc = evaluate(model, test_loader, params['model_name'], save_test_path)
    
    print("🧮 Evaluating model on windowed test set...")
    save_test_window_path = os.path.join(save_dir, "simple_test_window_predictions.txt")  
    window_testauc, window_testacc = evaluate(model, test_window_loader, params['model_name'], save_test_window_path)
    
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS FOR YOUR TRAINED GAINAKT2 MODEL")
    print("="*60)
    print(f"Test AUC:        {testauc:.4f}")
    print(f"Test Accuracy:   {testacc:.4f}")
    print(f"Window AUC:      {window_testauc:.4f}")
    print(f"Window Accuracy: {window_testacc:.4f}")
    print("="*60)
    
    # Compare with baseline
    baseline_auc = 0.7242  # From the comments in your training script
    print(f"Baseline AUC:    {baseline_auc:.4f}")
    if testauc > baseline_auc:
        improvement = (testauc - baseline_auc) * 100
        print(f"🎉 IMPROVEMENT:   +{improvement:.2f} percentage points!")
    else:
        decline = (baseline_auc - testauc) * 100  
        print(f"📉 Below baseline: -{decline:.2f} percentage points")
    
    print(f"\n📁 Predictions saved to:")
    print(f"   - {save_test_path}")
    print(f"   - {save_test_window_path}")

if __name__ == "__main__":
    main()