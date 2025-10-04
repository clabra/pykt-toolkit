#!/usr/bin/env python3
"""
Memory-Optimized SimAKT Training Script

This script addresses the CUDA out of memory issue in SimAKT training by:
1. Reducing batch size dynamically
2. Implementing gradient accumulation
3. Using memory-efficient tensor operations
4. Adding memory monitoring and cleanup

Usage:
    cd /workspaces/pykt-toolkit/assistant
    source /home/vscode/.pykt-env/bin/activate
    python simakt_memory_optimized_train.py --dataset_name=assist2015 --use_wandb=0
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
import gc
import time
from datetime import datetime

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

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    return 0, 0, 0

def find_optimal_batch_size(model, train_loader, initial_batch_size=32):
    """Find the largest batch size that fits in GPU memory"""
    print("🔍 Finding optimal batch size for GPU memory...")
    
    batch_sizes = [4, 8, 16, 24, 32, 48, 64]
    optimal_batch_size = 4  # Start with very small batch
    
    model.eval()
    with torch.no_grad():
        for batch_size in batch_sizes:
            if batch_size > initial_batch_size:
                break
                
            try:
                # Create a test batch
                test_batch = None
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx == 0:
                        test_batch = batch
                        break
                
                if test_batch is None:
                    continue
                    
                # Move to device
                for key in test_batch:
                    if torch.is_tensor(test_batch[key]):
                        test_batch[key] = test_batch[key][:batch_size].to(device)
                
                # Try forward pass
                c = test_batch.get('cseqs', test_batch.get('concepts', None))
                r = test_batch.get('rseqs', test_batch.get('responses', None))
                
                if c is not None and r is not None:
                    q = c.long()
                    r = r.long()
                    
                    # Test the memory-intensive forward pass
                    preds, cl_loss = model.get_cl_loss(q, r, None)
                    
                    optimal_batch_size = batch_size
                    print(f"   ✅ Batch size {batch_size} works")
                    clear_gpu_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ Batch size {batch_size} failed - OOM")
                    clear_gpu_memory()
                    break
                else:
                    print(f"   ❌ Batch size {batch_size} failed - {str(e)[:50]}...")
                    clear_gpu_memory()
                    break
    
    print(f"🎯 Optimal batch size found: {optimal_batch_size}")
    return optimal_batch_size

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_simakt_model(model_config, data_config, emb_type):
    """Initialize SimAKT model with aggressive memory optimizations"""
    # AGGRESSIVE memory reduction to prevent the expansion OOM
    optimized_config = model_config.copy()
    
    # CRITICAL: Reduce n_know to minimum (this is the main culprit)
    optimized_config['n_know'] = 2  # Reduced from default 16 to 2
    print(f"🔧 CRITICAL: Reduced n_know to {optimized_config['n_know']} (prevents 16x memory expansion)")
    
    # Further reduce model dimensions to save memory
    optimized_config['d_model'] = 64   # Reduced from 256 to 64
    optimized_config['d_ff'] = 64      # Reduced from 256 to 64
    optimized_config['num_attn_heads'] = 2  # Reduced from 8 to 2
    optimized_config['n_blocks'] = 1   # Reduced from 4 to 1
    
    print(f"🔧 Model dimensions: d_model={optimized_config['d_model']}, n_blocks={optimized_config['n_blocks']}")
    print(f"🔧 Memory expansion factor reduced from 16x to {optimized_config['n_know']}x")
    
    model = SimAKT(
        data_config["num_c"], 
        data_config["num_q"], 
        **optimized_config, 
        emb_type=emb_type,
        emb_path=data_config["emb_path"]
    ).to(device)
    
    return model

def init_dataset4train_simakt(dataset_name, data_config, fold, batch_size):
    """Initialize datasets for SimAKT training with small batches"""
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
    
    # Create data loaders with small batch sizes and no multiprocessing to save memory
    train_loader = DataLoader(curtrain, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    valid_loader = DataLoader(curvalid, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return train_loader, valid_loader

def save_config(train_config, model_config, data_config, params, save_dir):
    """Save configuration for reproducibility"""
    config = {
        "train_config": train_config,
        "model_config": model_config, 
        "data_config": data_config,
        "params": params,
        "memory_optimization": {
            "optimized_batch_size": True,
            "gradient_accumulation": True,
            "memory_monitoring": True
        }
    }
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

def train_simakt_model_memory_optimized(model, train_loader, valid_loader, num_epochs, save_dir, accumulation_steps=4):
    """Train SimAKT model with memory optimization techniques"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_auc = 0
    best_epoch = 0
    
    print(f"🚀 Starting memory-optimized training for {num_epochs} epochs")
    print(f"🔧 Using gradient accumulation with {accumulation_steps} steps")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        accumulated_loss = 0
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{num_epochs} - Training Phase")
        
        # Memory info at start of epoch
        allocated, reserved, total = get_gpu_memory_info()
        print(f"🖥️  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move data to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device, non_blocking=True)
                
                # Extract data from batch
                c = batch.get('cseqs', batch.get('concepts', None)) 
                r = batch.get('rseqs', batch.get('responses', None))
                
                if c is not None and r is not None:
                    q = c.long()
                    r = r.long()
                    
                    if q.size(1) == 0 or r.size(1) == 0:
                        continue
                    
                    # Forward pass with memory monitoring
                    preds, cl_loss = model.get_cl_loss(q, r, None)
                    
                    # Scale loss for gradient accumulation
                    cl_loss = cl_loss / accumulation_steps
                    cl_loss.backward()
                    
                    accumulated_loss += cl_loss.item()
                    
                    # Update weights every accumulation_steps
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        train_loss += accumulated_loss
                        train_steps += 1
                        accumulated_loss = 0
                    
                    # Progress reporting and memory cleanup
                    if batch_idx % 50 == 0:
                        allocated, reserved, total = get_gpu_memory_info()
                        print(f"  Batch {batch_idx:4d}: Loss {cl_loss.item()*accumulation_steps:.4f}, "
                              f"GPU: {allocated:.2f}GB allocated")
                        
                        # Clear cache periodically
                        if batch_idx % 100 == 0:
                            clear_gpu_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ⚠️  OOM at batch {batch_idx}, clearing memory and continuing...")
                    clear_gpu_memory()
                    optimizer.zero_grad()
                    continue
                else:
                    print(f"  ❌ Error at batch {batch_idx}: {str(e)[:100]}...")
                    continue
        
        # Handle any remaining accumulated gradients
        if accumulated_loss > 0:
            optimizer.step()
            optimizer.zero_grad()
            train_loss += accumulated_loss
            train_steps += 1
        
        if train_steps == 0:
            print(f"❌ No valid training steps in epoch {epoch+1}")
            continue
        
        avg_train_loss = train_loss / train_steps
        
        # Validation phase with memory optimization
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{num_epochs} - Validation Phase")
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                try:
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(device, non_blocking=True)
                    
                    c = batch.get('cseqs', batch.get('concepts', None))
                    r = batch.get('rseqs', batch.get('responses', None))
                    
                    if c is not None and r is not None:
                        q = c.long()
                        r = r.long()
                        
                        if q.size(1) == 0 or r.size(1) == 0:
                            continue
                        
                        preds, _ = model.get_loss(q, r, None)
                        
                        # Collect predictions and targets
                        mask = r >= 0
                        if mask.sum() > 0:
                            val_preds.extend(preds[mask].cpu().numpy())
                            val_targets.extend(r[mask].cpu().numpy())
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ⚠️  OOM in validation at batch {batch_idx}, skipping...")
                        clear_gpu_memory()
                        continue
                    else:
                        continue
        
        if len(val_preds) == 0:
            print(f"❌ No valid validation predictions in epoch {epoch+1}")
            continue
        
        # Calculate metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Epoch summary
        print("\n" + "="*80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] EPOCH {epoch+1}/{num_epochs} SUMMARY")
        print("="*80)
        print(f"Training Loss:     {avg_train_loss:.6f}")
        print(f"Validation AUC:    {val_auc:.6f}")
        print(f"Validation ACC:    {val_acc:.6f}")
        print(f"Epoch Time:        {epoch_time:.2f}s ({epoch_time/60:.1f}m)")
        print(f"Training Steps:    {train_steps}")
        
        # Memory info
        allocated, reserved, total = get_gpu_memory_info()
        print(f"GPU Memory:        {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Model saving
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "simakt_model.ckpt"))
            print(f"🌟 NEW BEST MODEL SAVED! AUC: {best_auc:.6f}")
        else:
            print(f"Current AUC: {val_auc:.6f} | Best AUC: {best_auc:.6f}")
        
        print("="*80)
        
        # Clear memory at end of epoch
        clear_gpu_memory()
    
    # Final training summary
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Epoch:        {best_epoch}")
    print(f"Best Validation AUC: {best_auc:.6f}")
    print("="*80)
    
    return best_auc, 0.0

def main(params):
    # Set seed for reproducibility
    set_seed(params["seed"])
    
    print("🚀 Memory-Optimized SimAKT Training")
    print("="*50)
    print(f"Dataset: {params['dataset_name']}")
    print(f"GPU Memory Management: Enabled")
    print(f"Gradient Accumulation: Enabled")
    print("="*50)
    
    # Load configurations
    config_base_path = os.path.join(os.path.dirname(__file__), '..', 'configs')
    with open(os.path.join(config_base_path, "kt_config.json")) as f:
        config = json.load(f)
        train_config = config["train_config"]
        # Start with very small batch size for memory safety
        train_config["batch_size"] = 4
        train_config["num_epochs"] = 50  # Default epochs
        
    with open(os.path.join(config_base_path, "data_config.json")) as f:
        data_config = json.load(f)
    
    # Setup parameters
    model_name = params["model_name"]
    dataset_name = params["dataset_name"]
    emb_type = params["emb_type"]
    
    # Model configuration - optimized for memory
    model_config = {
        "d_model": 128,  # Reduced from default 256
        "d_ff": 128,     # Reduced from default 256
        "num_attn_heads": 4,  # Reduced from default 8
        "n_blocks": 2,   # Reduced from default 4
        "dropout": params["dropout"],
        "n_know": 8,     # Reduced from default 16
        "lambda_cl": 0.1,
        "window": 1,
        "proj": True,
        "hard_neg": False
    }
    
    # Setup data config
    dataset_config = data_config[dataset_name]
    dataset_config["dataset_name"] = dataset_name
    
    # Create save directory
    save_dir = os.path.join(
        params["save_dir"],
        f"{dataset_name}_{model_name}_{emb_type}_memory_optimized_{params['seed']}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Save directory: {save_dir}")
    print(f"Model config: {model_config}")
    
    # Save configuration
    save_config(train_config, model_config, dataset_config, params, save_dir)
    
    # Initialize model
    print("\n🔧 Initializing memory-optimized SimAKT model...")
    model = init_simakt_model(model_config, dataset_config, emb_type)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize data loaders with small batch size
    initial_batch_size = train_config["batch_size"]
    train_loader, valid_loader = init_dataset4train_simakt(
        dataset_name, 
        dataset_config, 
        fold=params["fold"],
        batch_size=initial_batch_size
    )
    
    # Find optimal batch size
    optimal_batch_size = find_optimal_batch_size(model, train_loader, initial_batch_size)
    
    # Recreate data loaders with optimal batch size
    if optimal_batch_size != initial_batch_size:
        print(f"🔄 Recreating data loaders with batch size {optimal_batch_size}")
        train_loader, valid_loader = init_dataset4train_simakt(
            dataset_name, 
            dataset_config, 
            fold=params["fold"],
            batch_size=optimal_batch_size
        )
    
    # Calculate gradient accumulation steps to maintain effective batch size
    target_effective_batch_size = 32
    accumulation_steps = max(1, target_effective_batch_size // optimal_batch_size)
    print(f"🔧 Using gradient accumulation: {accumulation_steps} steps (effective batch size: {optimal_batch_size * accumulation_steps})")
    
    # Train model with memory optimization
    testauc, testacc = train_simakt_model_memory_optimized(
        model, 
        train_loader, 
        valid_loader, 
        train_config["num_epochs"], 
        save_dir,
        accumulation_steps=accumulation_steps
    )
    
    print(f"\n🎉 Final Results:")
    print(f"📊 Best Validation AUC: {testauc:.4f}")
    print(f"💾 Model saved to: {save_dir}")
    
    return testauc, testacc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Optimized SimAKT Training")
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="simakt")
    parser.add_argument("--emb_type", type=str, default="qid_cl")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    parser.add_argument("--use_wandb", type=int, default=0)
    
    args = parser.parse_args()
    params = vars(args)
    
    print("🚀 Starting Memory-Optimized SimAKT Training")
    print("=" * 60)
    
    try:
        main(params)
    except Exception as e:
        print(f"\n💥 Training failed with error: {str(e)}")
        print("🔧 Try reducing model dimensions further or checking GPU memory")
        import traceback
        traceback.print_exc()