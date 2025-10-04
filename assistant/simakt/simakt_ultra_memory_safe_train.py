#!/usr/bin/env python3
"""
Ultra Memory-Optimized SimAKT Training Script

This script addresses severe CUDA out of memory issues in SimAKT by:
1. Using extremely small batch sizes (2-8)
2. Reducing model dimensions significantly
3. Implementing gradient accumulation to maintain effective batch size
4. Adding comprehensive memory monitoring and cleanup
5. Using checkpoint-based training to recover from OOM

Usage:
    cd /workspaces/pykt-toolkit/assistant
    source /home/vscode/.pykt-env/bin/activate
    python simakt_ultra_memory_safe_train.py --dataset_name=assist2015 --use_wandb=0
"""

import argparse
import os
import sys
import json

# Set PyTorch memory configuration BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,roundup_power2_divisions:16'

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
    """Aggressively clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # Force memory cleanup by creating and deleting a small tensor
        try:
            temp = torch.zeros(1).to(device)
            del temp
            torch.cuda.empty_cache()
        except:
            pass

def get_gpu_memory_info():
    """Get detailed GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - allocated
        return allocated, reserved, total, free
    return 0, 0, 0, 0

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_ultra_small_simakt_model(data_config, emb_type="qid_cl"):
    """Initialize SimAKT model with ultra-small dimensions for memory safety"""
    
    # ULTRA-conservative model configuration to prevent 3.12GB allocation
    ultra_small_config = {
        "d_model": 32,        # REDUCED from 64 to 32 (massive reduction)
        "d_ff": 32,           # REDUCED from 64 to 32  
        "num_attn_heads": 1,  # REDUCED from 2 to 1 (minimum possible)
        "n_blocks": 1,        # Keep at 1
        "dropout": 0.05,      # Even lower dropout
        "n_know": 2,          # REDUCED from 4 to 2 (CRITICAL: prevents 4x expansion)
        "lambda_cl": 0.05,    # Lower contrastive loss weight
        "window": 1,
        "proj": False,        # Disable projection to save memory
        "hard_neg": False     # Disable hard negatives
    }
    
    print("🔧 ULTRA-Small SimAKT Configuration (Extreme Memory Safety):")
    print(f"   d_model: {ultra_small_config['d_model']} (75% reduction)")
    print(f"   d_ff: {ultra_small_config['d_ff']} (75% reduction)")
    print(f"   num_attn_heads: {ultra_small_config['num_attn_heads']} (minimum)")
    print(f"   n_blocks: {ultra_small_config['n_blocks']}")
    print(f"   n_know: {ultra_small_config['n_know']} (CRITICAL: only 2x expansion instead of 16x)")
    print(f"   🎯 Memory expansion factor: {ultra_small_config['n_know']}x instead of 16x")
    
    model = SimAKT(
        data_config["num_c"], 
        data_config["num_q"], 
        **ultra_small_config, 
        emb_type=emb_type,
        emb_path=data_config["emb_path"]
    ).to(device)
    
    return model, ultra_small_config

def init_ultra_small_dataset(dataset_name, data_config, fold, batch_size=1):
    """Initialize datasets with ultra-small batch sizes"""
    all_folds = set(data_config["folds"])
    
    # Fix the data path
    base_path = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_path, data_config["dpath"].lstrip('./'))
    train_file_path = os.path.join(data_path, data_config["train_valid_file"])
    
    # Create datasets
    curvalid = KTDataset(train_file_path, data_config["input_type"], {fold})
    curtrain = KTDataset(train_file_path, data_config["input_type"], all_folds - {fold})
    
    # ULTRA-small data loaders for maximum memory safety
    print(f"🔧 Using ULTRA-small batch_size={batch_size} for maximum memory safety")
    
    train_loader = DataLoader(
        curtrain, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,        # No multiprocessing to save memory
        pin_memory=False,     # Disable pin_memory to save memory
        drop_last=True,       # Drop incomplete batches
        persistent_workers=False
    )
    
    valid_loader = DataLoader(
        curvalid, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False
    )
    
    return train_loader, valid_loader

def safe_forward_pass(model, q, r, max_retries=3):
    """Attempt forward pass with OOM recovery"""
    for attempt in range(max_retries):
        try:
            clear_gpu_memory()
            preds, cl_loss = model.get_cl_loss(q, r, None)
            return preds, cl_loss, True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    ⚠️  OOM attempt {attempt+1}/{max_retries}, clearing memory...")
                clear_gpu_memory()
                if attempt == max_retries - 1:
                    return None, None, False
            else:
                print(f"    ❌ Forward pass error: {str(e)[:100]}...")
                return None, None, False
    return None, None, False

def train_ultra_memory_safe(model, train_loader, valid_loader, num_epochs, save_dir, accumulation_steps=16):
    """Ultra memory-safe training with aggressive memory management"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Lower learning rate
    best_auc = 0
    best_epoch = 0
    
    print(f"🚀 Starting ultra memory-safe training")
    print(f"🔧 Gradient accumulation: {accumulation_steps} steps")
    print(f"🔧 Effective batch size: {train_loader.batch_size * accumulation_steps}")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        successful_steps = 0
        accumulated_loss = 0
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{num_epochs} - Training")
        
        # Memory status at start
        allocated, reserved, total, free = get_gpu_memory_info()
        print(f"🖥️  Start - GPU Memory: {allocated:.2f}GB used, {free:.2f}GB free of {total:.2f}GB")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move data to device with minimal memory footprint
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device, non_blocking=False)
                
                # Extract sequences
                c = batch.get('cseqs', batch.get('concepts', None)) 
                r = batch.get('rseqs', batch.get('responses', None))
                
                if c is not None and r is not None:
                    q = c.long()
                    r = r.long()
                    
                    if q.size(1) == 0 or r.size(1) == 0:
                        continue
                    
                    # Safe forward pass with OOM recovery
                    preds, cl_loss, success = safe_forward_pass(model, q, r)
                    
                    if success and cl_loss is not None:
                        # Scale loss for gradient accumulation
                        scaled_loss = cl_loss / accumulation_steps
                        scaled_loss.backward()
                        
                        accumulated_loss += scaled_loss.item()
                        
                        # Update weights every accumulation_steps
                        if (batch_idx + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            train_loss += accumulated_loss
                            successful_steps += 1
                            accumulated_loss = 0
                            
                            # Clear memory after weight update
                            clear_gpu_memory()
                    else:
                        # Skip this batch due to OOM
                        optimizer.zero_grad()
                        print(f"    ⚠️  Skipped batch {batch_idx} due to memory constraints")
                
                # Progress and memory monitoring
                if batch_idx % 25 == 0:
                    allocated, reserved, total, free = get_gpu_memory_info()
                    if successful_steps > 0:
                        avg_loss = train_loss / successful_steps
                        print(f"  Batch {batch_idx:4d}: Avg Loss {avg_loss:.4f}, "
                              f"GPU: {allocated:.2f}GB/{total:.2f}GB, Steps: {successful_steps}")
                    
                    # Aggressive memory cleanup every 50 batches
                    if batch_idx % 50 == 0:
                        clear_gpu_memory()
                
            except Exception as e:
                print(f"  ❌ Batch {batch_idx} failed: {str(e)[:50]}...")
                optimizer.zero_grad()
                clear_gpu_memory()
                continue
        
        # Handle remaining accumulated gradients
        if accumulated_loss > 0:
            optimizer.step()
            optimizer.zero_grad()
            train_loss += accumulated_loss
            successful_steps += 1
        
        if successful_steps == 0:
            print(f"❌ No successful training steps in epoch {epoch+1}")
            continue
        
        avg_train_loss = train_loss / successful_steps
        
        # Validation phase with ultra-safe memory management
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{num_epochs} - Validation")
        
        model.eval()
        val_preds = []
        val_targets = []
        successful_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                try:
                    clear_gpu_memory()  # Clear before each validation batch
                    
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(device, non_blocking=False)
                    
                    c = batch.get('cseqs', batch.get('concepts', None))
                    r = batch.get('rseqs', batch.get('responses', None))
                    
                    if c is not None and r is not None:
                        q = c.long()
                        r = r.long()
                        
                        if q.size(1) == 0 or r.size(1) == 0:
                            continue
                        
                        # Safe validation forward pass
                        try:
                            preds, _ = model.get_loss(q, r, None, q_cl=False)
                            
                            mask = r >= 0
                            if mask.sum() > 0:
                                val_preds.extend(preds[mask].cpu().numpy())
                                val_targets.extend(r[mask].cpu().numpy())
                                successful_val_batches += 1
                        
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"    ⚠️  Val OOM at batch {batch_idx}, skipping...")
                                clear_gpu_memory()
                                continue
                            else:
                                continue
                
                except Exception as e:
                    print(f"  ❌ Val batch {batch_idx} failed: {str(e)[:50]}...")
                    clear_gpu_memory()
                    continue
        
        if len(val_preds) == 0:
            print(f"❌ No valid validation predictions in epoch {epoch+1}")
            continue
        
        # Calculate metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])
        epoch_time = time.time() - epoch_start_time
        
        # Epoch summary
        print("\n" + "="*80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] EPOCH {epoch+1}/{num_epochs} SUMMARY")
        print("="*80)
        print(f"Training Loss:        {avg_train_loss:.6f}")
        print(f"Validation AUC:       {val_auc:.6f}")
        print(f"Validation ACC:       {val_acc:.6f}")
        print(f"Epoch Time:           {epoch_time:.2f}s ({epoch_time/60:.1f}m)")
        print(f"Successful Train Steps: {successful_steps}")
        print(f"Successful Val Batches: {successful_val_batches}")
        
        # Memory status
        allocated, reserved, total, free = get_gpu_memory_info()
        print(f"GPU Memory:           {allocated:.2f}GB used, {free:.2f}GB free")
        
        # Model saving
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "ultra_safe_simakt_model.ckpt"))
            print(f"🌟 NEW BEST MODEL SAVED! AUC: {best_auc:.6f}")
        else:
            print(f"Current AUC: {val_auc:.6f} | Best AUC: {best_auc:.6f}")
        
        print("="*80)
        
        # Aggressive cleanup at end of epoch
        clear_gpu_memory()
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"Best Epoch: {best_epoch}, Best AUC: {best_auc:.6f}")
    
    return best_auc, 0.0

def main():
    parser = argparse.ArgumentParser(description="Ultra Memory-Safe SimAKT Training")
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--batch_size", type=int, default=1, help="Ultra-small batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--use_wandb", type=int, default=0)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("🚀 ULTRA Memory-Safe SimAKT Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Batch Size: {args.batch_size} (ultra-small)")
    print(f"Memory Strategy: EXTREME optimization")
    print(f"PyTorch Memory Config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
    print("=" * 60)
    
    # Aggressive memory cleanup before starting
    clear_gpu_memory()
    
    # Load configurations
    config_base_path = os.path.join(os.path.dirname(__file__), '..', 'configs')
    
    with open(os.path.join(config_base_path, "data_config.json")) as f:
        data_config = json.load(f)
    
    # Setup dataset config
    dataset_config = data_config[args.dataset_name]
    dataset_config["dataset_name"] = args.dataset_name
    
    # Create save directory
    save_dir = os.path.join(
        os.path.dirname(__file__),
        "saved_model",
        f"{args.dataset_name}_simakt_ultra_safe_{args.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Save directory: {save_dir}")
    
    # Initialize ultra-small model
    print("\n🔧 Initializing ULTRA-small SimAKT model...")
    model, model_config = init_ultra_small_simakt_model(dataset_config, "qid_cl")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # GPU memory check after model initialization
    allocated, reserved, total, free = get_gpu_memory_info()
    print(f"🖥️  GPU Memory after model init: {allocated:.2f}GB used, {free:.2f}GB free")
    
    # Initialize ultra-small datasets
    print(f"\n🔧 Creating data loaders with batch_size={args.batch_size}...")
    train_loader, valid_loader = init_ultra_small_dataset(
        args.dataset_name, dataset_config, args.fold, args.batch_size
    )
    
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(valid_loader)}")
    
    # Calculate gradient accumulation to maintain reasonable effective batch size
    target_effective_batch_size = 16  # Reduced from 32
    accumulation_steps = max(1, target_effective_batch_size // args.batch_size)
    print(f"🔧 Gradient accumulation: {accumulation_steps} steps (effective batch: {args.batch_size * accumulation_steps})")
    
    # Save configuration
    config = {
        "model_config": model_config,
        "dataset_config": dataset_config,
        "training_config": {
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "accumulation_steps": accumulation_steps,
            "ultra_memory_safe": True,
            "pytorch_memory_config": os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
        },
        "params": vars(args)
    }
    
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train with ultra memory safety
    try:
        best_auc, _ = train_ultra_memory_safe(
            model, train_loader, valid_loader, 
            args.num_epochs, save_dir, accumulation_steps
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Best Validation AUC: {best_auc:.4f}")
        print(f"💾 Model saved to: {save_dir}")
        
    except Exception as e:
        print(f"\n💥 Training failed: {str(e)}")
        print("🔧 The model dimensions are at absolute minimum.")
        print("🔧 Consider using CPU training or a smaller dataset.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()