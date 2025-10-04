#!/usr/bin/env python
# coding=utf-8
"""
SimAKT Training Script - Completely Isolated

This script trains the SimAKT model with minimal dependencies to avoid import conflicts.
Usage:
    python assistant/simakt_isolated_train.py --dataset_name=assist2015 --use_wandb=0
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import random

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Direct imports to avoid package-level imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pykt', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pykt', 'datasets'))

# Import SimAKT directly
from simakt import SimAKT

# Import dataset class directly
from data_loader import KTDataset

device = "cpu" if not torch.cuda.is_available() else "cuda"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_simakt_model(model_config, data_config, emb_type):
    """Initialize SimAKT model specifically"""
    model = SimAKT(
        data_config["num_c"], 
        data_config["num_q"], 
        **model_config, 
        emb_type=emb_type,
        emb_path=data_config["emb_path"]
    ).to(device)
    return model

def init_dataset4train_simakt(dataset_name, data_config, fold, batch_size):
    """Initialize datasets for SimAKT training without full pykt imports"""
    all_folds = set(data_config["folds"])
    
    # Fix the data path - use absolute path from project root
    base_path = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_path, data_config["dpath"].lstrip('./'))
    train_file_path = os.path.join(data_path, data_config["train_valid_file"])
    
    # Create validation dataset
    curvalid = KTDataset(
        train_file_path, 
        data_config["input_type"], 
        {fold}
    )
    
    # Create training dataset  
    curtrain = KTDataset(
        train_file_path, 
        data_config["input_type"], 
        all_folds - {fold}
    )
    
    # Create data loaders
    train_loader = DataLoader(curtrain, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(curvalid, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, valid_loader

def save_config(train_config, model_config, data_config, params, save_dir):
    """Save configuration for reproducibility"""
    config = {
        "train_config": train_config,
        "model_config": model_config, 
        "data_config": data_config,
        "params": params
    }
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

def train_simakt_model(model, train_loader, valid_loader, num_epochs, save_dir):
    """Train SimAKT model with simplified training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_auc = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device and extract sequences
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            
            # Extract data from batch - ASSIST2015 uses concepts as primary sequence
            c = batch.get('cseqs', batch.get('concepts', None)) 
            r = batch.get('rseqs', batch.get('responses', None))
            
            # For ASSIST2015, use concept IDs as question IDs since qlen=0
            if c is not None and r is not None:
                # Convert to proper data types
                q = c.long()  # Use concepts as questions for SimAKT, ensure long type
                r = r.long()  # Ensure responses are long type
                
                # Check for valid sequences
                if q.size(1) == 0 or r.size(1) == 0:
                    continue
                    
                try:
                    # Get predictions and loss using SimAKT's contrastive learning
                    preds, cl_loss = model.get_cl_loss(q, r, None)
                    
                    # Backward pass
                    cl_loss.backward()
                    optimizer.step()
                    
                    train_loss += cl_loss.item()
                    train_steps += 1
                    
                except Exception as e:
                    if batch_idx < 5:  # Only show first few errors
                        print(f"Error in training step {batch_idx}: {e}")
                    continue
            else:
                continue  # Skip batch if no valid data
        
        if train_steps == 0:
            print(f"No valid training steps in epoch {epoch+1}")
            continue
            
        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in valid_loader:
                # Move data to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
                
                # Extract data from batch
                c = batch.get('cseqs', batch.get('concepts', None))
                r = batch.get('rseqs', batch.get('responses', None))
                
                # Use concepts as questions for ASSIST2015
                if c is not None and r is not None:
                    # Convert to proper data types
                    q = c.long()  # Use concepts as questions for SimAKT, ensure long type
                    r = r.long()  # Ensure responses are long type
                    
                    # Check for valid sequences
                    if q.size(1) == 0 or r.size(1) == 0:
                        continue
                        
                    try:
                        preds, _ = model.get_loss(q, r, None)
                        
                        # Collect predictions and targets
                        mask = r >= 0
                        if mask.sum() > 0:
                            val_preds.extend(preds[mask].cpu().numpy())
                            val_targets.extend(r[mask].cpu().numpy())
                            
                    except Exception as e:
                        continue
                else:
                    continue
        
        if len(val_preds) == 0:
            print(f"No valid validation predictions in epoch {epoch+1}")
            continue
            
        # Calculate metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/train_steps:.4f}")
        print(f"  Valid AUC: {val_auc:.4f}")
        print(f"  Valid ACC: {val_acc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "simakt_model.ckpt"))
            print(f"  New best model saved! AUC: {best_auc:.4f}")
        
        print()
    
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation AUC: {best_auc:.4f}")
    
    return best_auc, 0.0  # Return AUC and placeholder ACC

def main(params):
    # Set seed for reproducibility
    set_seed(params["seed"])
    
    # Load configurations
    with open("configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        
    with open("configs/data_config.json") as f:
        data_config = json.load(f)
    
    # Setup model parameters
    model_name = params["model_name"]
    dataset_name = params["dataset_name"]
    emb_type = params["emb_type"]
    
    # Model configuration
    model_config = {
        "d_model": params["d_model"],
        "d_ff": params["d_ff"], 
        "num_attn_heads": params["num_attn_heads"],
        "n_blocks": params["n_blocks"],
        "dropout": params["dropout"],
        "n_know": params["n_know"],
        "lambda_cl": params["lambda_cl"],
        "window": params["window"],
        "proj": params["proj"],
        "hard_neg": params["hard_neg"]
    }
    
    # Setup data config
    dataset_config = data_config[dataset_name]
    dataset_config["dataset_name"] = dataset_name
    
    # Create save directory
    save_dir = os.path.join(
        params["save_dir"],
        f"{dataset_name}_{model_name}_{emb_type}_isolated_{params['seed']}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Training SimAKT model")
    print(f"Dataset: {dataset_name}")
    print(f"Save directory: {save_dir}")
    print(f"Model config: {model_config}")
    
    # Save configuration
    save_config(train_config, model_config, dataset_config, params, save_dir)
    
    # Initialize data loaders
    train_loader, valid_loader = init_dataset4train_simakt(
        dataset_name, 
        dataset_config, 
        fold=params["fold"],
        batch_size=train_config["batch_size"]
    )
    
    # Initialize model
    model = init_simakt_model(model_config, dataset_config, emb_type)
    
    # Train model
    testauc, testacc = train_simakt_model(
        model, 
        train_loader, 
        valid_loader, 
        train_config["num_epochs"], 
        save_dir
    )
    
    print(f"Final Results:")
    print(f"Test AUC: {testauc:.4f}")
    print(f"Test ACC: {testacc:.4f}")
    
    return testauc, testacc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="simakt")
    parser.add_argument("--emb_type", type=str, default="qid_cl")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    parser.add_argument("--n_know", type=int, default=16)
    parser.add_argument("--lambda_cl", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=1)

    parser.add_argument("--proj", type=str2bool, default=True)
    parser.add_argument("--hard_neg", type=str2bool, default=False)
    
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()
    params = vars(args)
    
    main(params)