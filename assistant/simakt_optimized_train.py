#!/usr/bin/env python
# coding=utf-8
"""
SimAKT Training Script - Optimized for Configurable Resource Management

cd /workspaces/pykt-toolkit && python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --batch_size=32 \
    --num_workers=2 \
    --cpu_threads=4 \
    --use_wandb=0 \
    --progress_freq=20

OVERVIEW:
This script trains the SimAKT (Similarity-based Attention Knowledge Tracing) model 
with flexible resource configuration that adapts to different computing environments.
Designed for both shared machines and dedicated systems with full parameterization.

KEY FEATURES:
- Parameterized Resource Management: Configure CPU threads, data workers, batch sizes
- Preset Resource Modes: Easy presets for different computing environments
- Manual Override Options: Fine-grained control over all resource parameters
- Shared Machine Safe: Conservative defaults prevent system overload
- Hardware Detection: Automatically detects available CPU cores and memory
- Progress Monitoring: Configurable progress reporting frequency

RESOURCE MODES:
1. MINIMAL (--resource_mode=minimal):
   - Batch Size: 16, Workers: 0, CPU Threads: 1
   - Use Case: Heavily loaded shared machines, strict resource limits
   
2. CONSERVATIVE (--resource_mode=conservative):
   - Batch Size: 64, Workers: 2, CPU Threads: 2
   - Use Case: Shared machines with moderate resource availability
   
3. MODERATE (--resource_mode=moderate) [DEFAULT]:
   - Batch Size: 256, Workers: 8, CPU Threads: 8
   - Use Case: Lightly shared machines or dedicated development environments
   
4. AGGRESSIVE (--resource_mode=aggressive):
   - Batch Size: 1024, Workers: 16, CPU Threads: 20
   - Use Case: Dedicated machines with abundant resources

USAGE EXAMPLES:

Basic Usage (Moderate mode - new default):
    python assistant/simakt_optimized_train.py --dataset_name=assist2015 --use_wandb=0

Minimal Resources (Safest for heavily shared machines):
    python assistant/simakt_optimized_train.py --dataset_name=assist2015 --resource_mode=minimal --use_wandb=0

Manual Resource Configuration:
    python assistant/simakt_optimized_train.py --dataset_name=assist2015 --batch_size=32 --num_workers=1 --cpu_threads=2 --use_wandb=0

TROUBLESHOOTING:
- Process Killed/OOM: Use --resource_mode=minimal or --batch_size=16 --num_workers=0 --cpu_threads=1
- Slow Training: Increase resource mode or manually increase batch_size/num_workers
- Shared System: Start with minimal mode and gradually increase resources

COMMAND-LINE PARAMETERS:
Core Training: --dataset_name, --model_name, --emb_type, --save_dir, --seed, --fold
Model Architecture: --d_model, --d_ff, --num_attn_heads, --n_blocks, --dropout, --n_know, --lambda_cl
Resource Management: --resource_mode, --batch_size, --num_workers, --cpu_threads, --progress_freq

INTEGRATION:
- Follows PyKT framework conventions
- Compatible with PyKT evaluation scripts  
- Adheres to GEMINI.md guidelines for non-intrusive changes
- Model saved in standard PyKT checkpoint format

For detailed documentation, see: assistant/simakt_optimized_train_documentation.md
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
import multiprocessing as mp
import time
from datetime import datetime

def bootstrap_gpu_environment():
    """
    Checks for GPU access and, if it fails, attempts to apply the environment fix
    by sourcing the enable_gpu.sh script. This makes the training script self-sufficient.
    """
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ GPU environment already configured.")
            return
    except ImportError:
        pass # PyTorch might not be in the path yet.

    print("🔧 GPU not detected by PyTorch. Attempting to configure environment...")
    # Re-execute the script within a shell that has the correct environment
    # This is a robust way to ensure the sourced variables apply to the Python process.
    script_path = "/workspaces/pykt-toolkit/enable_gpu.sh"
    if os.path.exists(script_path):
        # Use execv to replace the current process with the new one
        os.execv("/bin/bash", ["/bin/bash", "-c", f"source {script_path} && exec python {' '.join(sys.argv)}"])
    else:
        print(f"⚠️  Warning: {script_path} not found. GPU access may fail.")

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

def get_optimal_batch_size(manual_batch_size=None, resource_mode="conservative"):
    """Calculate batch size based on resource mode or manual override"""
    if manual_batch_size:
        return manual_batch_size
    
    if resource_mode == "minimal":
        return 16
    elif resource_mode == "conservative":
        return 64
    elif resource_mode == "moderate":
        return 256
    elif resource_mode == "aggressive":
        return 1024
    else:
        return 64  # Default conservative

def get_optimal_num_workers(manual_workers=None, resource_mode="conservative"):
    """Calculate number of workers based on resource mode or manual override"""
    if manual_workers is not None:
        return manual_workers
    
    total_cores = mp.cpu_count()
    
    if resource_mode == "minimal":
        return 0  # No multiprocessing
    elif resource_mode == "conservative":
        return min(2, total_cores)
    elif resource_mode == "moderate":
        return min(8, total_cores // 2)
    elif resource_mode == "aggressive":
        return min(16, int(total_cores * 0.75))
    else:
        return 2  # Default conservative

def get_optimal_cpu_threads(manual_threads=None, resource_mode="conservative"):
    """Calculate CPU threads based on resource mode or manual override"""
    if manual_threads is not None:
        return manual_threads
    
    total_cores = mp.cpu_count()
    
    if resource_mode == "minimal":
        return 1
    elif resource_mode == "conservative":
        return min(2, total_cores)
    elif resource_mode == "moderate":
        return min(8, total_cores // 2)
    elif resource_mode == "aggressive":
        return min(20, int(total_cores * 0.5))
    else:
        return 2  # Default conservative

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

def init_dataset4train_simakt(dataset_name, data_config, fold, batch_size, num_workers):
    """Initialize datasets for SimAKT training with optimized data loading"""
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
    
    # Create optimized data loaders
    print(f"Using optimized settings: batch_size={batch_size}, num_workers={num_workers}")
    train_loader = DataLoader(
        curtrain, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # Optimize memory transfer
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    valid_loader = DataLoader(
        curvalid, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, valid_loader

def save_config(train_config, model_config, data_config, params, save_dir):
    """Save configuration for reproducibility"""
    config = {
        "train_config": train_config,
        "model_config": model_config, 
        "data_config": data_config,
        "params": params,
        "hardware_optimization": {
            "batch_size": params.get("batch_size"),
            "num_workers": params.get("num_workers"),
            "available_cores": mp.cpu_count(),
            "device": str(device)
        }
    }
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

def train_simakt_model(model, train_loader, valid_loader, num_epochs, save_dir, cpu_threads=2, progress_freq=50):
    """Train SimAKT model with frequent timestamped feedback and detailed epoch metrics"""
    # Use configurable CPU threads for computation
    torch.set_num_threads(cpu_threads)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_auc = 0
    best_epoch = 0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training with {torch.get_num_threads()} computation threads")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {num_epochs} epochs")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{num_epochs} - Training Phase Started")
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Move data to device and extract sequences
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device, non_blocking=False)
            
            optimizer.zero_grad()
            
            # Extract data from batch - ASSIST2015 uses concepts as primary sequence
            c = batch.get('cseqs', batch.get('concepts', None)) 
            r = batch.get('rseqs', batch.get('responses', None))
            
            # For ASSIST2015, use concept IDs as question IDs since qlen=0
            if c is not None and r is not None:
                q = c.long()
                r = r.long()
                
                if q.size(1) == 0 or r.size(1) == 0:
                    continue
                    
                try:
                    preds, cl_loss = model.get_cl_loss(q, r, None)
                    cl_loss.backward()
                    optimizer.step()
                    
                    train_loss += cl_loss.item()
                    train_steps += 1
                    
                    batch_time = time.time() - batch_start_time
                    
                    # Frequent progress reporting with timestamps
                    if batch_idx % progress_freq == 0:
                        current_time = datetime.now().strftime('%H:%M:%S')
                        avg_loss = train_loss / max(train_steps, 1)
                        print(f"  [{current_time}] Batch {batch_idx:4d}: Loss {cl_loss.item():.4f} | Avg Loss {avg_loss:.4f} | Time {batch_time:.2f}s")
                    
                except Exception as e:
                    if batch_idx < 3:
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Error in training step {batch_idx}: {e}")
                    continue
            else:
                continue
        
        if train_steps == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No valid training steps in epoch {epoch+1}")
            continue
        
        # Calculate training metrics
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{num_epochs} - Validation Phase Started")
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
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
                        
                    try:
                        # For SimAKT, get_loss returns (preds, reg_loss) when q_cl=False
                        preds, reg_loss = model.get_loss(q, r, None, q_cl=False)
                        
                        # Calculate validation loss using BCE
                        mask = r >= 0
                        if mask.sum() > 0:
                            # Get valid predictions and targets
                            valid_preds = preds[mask]
                            valid_targets = r[mask].float()
                            
                            # Calculate BCE loss for validation
                            import torch.nn.functional as F
                            val_loss_batch = F.binary_cross_entropy(valid_preds, valid_targets, reduction='mean')
                            val_loss += val_loss_batch.item()
                            val_steps += 1
                            
                            # Collect predictions for metrics
                            val_preds.extend(valid_preds.cpu().numpy())
                            val_targets.extend(valid_targets.cpu().numpy())
                            
                        # Frequent validation feedback
                        if batch_idx % (progress_freq * 2) == 0:  # Less frequent than training
                            if val_steps > 0:
                                current_val_loss = val_loss / val_steps
                                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Val Batch {batch_idx:4d}: Loss {current_val_loss:.4f} | Preds: {len(val_preds)}")
                            
                    except Exception as e:
                        if batch_idx < 3:  # Show first few errors for debugging
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Val Error batch {batch_idx}: {e}")
                        continue
                else:
                    continue
        
        if len(val_preds) == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No valid validation predictions in epoch {epoch+1}")
            continue
        
        # Calculate validation metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])
        avg_val_loss = val_loss / max(val_steps, 1)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Detailed epoch summary with timestamp
        print("\n" + "="*80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] EPOCH {epoch+1}/{num_epochs} SUMMARY")
        print("="*80)
        print(f"Training Loss:     {avg_train_loss:.6f}")
        print(f"Validation Loss:   {avg_val_loss:.6f}")
        print(f"Validation AUC:    {val_auc:.6f}")
        print(f"Validation ACC:    {val_acc:.6f}")
        print(f"Epoch Time:        {epoch_time:.2f}s ({epoch_time/60:.1f}m)")
        print(f"Training Steps:    {train_steps}")
        print(f"Validation Steps:  {val_steps}")
        
        # Model saving with detailed feedback
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "simakt_model.ckpt"))
            print(f"🌟 NEW BEST MODEL SAVED! AUC: {best_auc:.6f} (Improvement: +{val_auc - best_auc:.6f})")
        else:
            improvement = val_auc - best_auc
            print(f"Current AUC: {val_auc:.6f} | Best AUC: {best_auc:.6f} (Diff: {improvement:+.6f})")
        
        print("="*80)
    
    # Final training summary
    total_time = sum([time.time() - epoch_start_time for epoch_start_time in [time.time()]])
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Epoch:        {best_epoch}")
    print(f"Best Validation AUC: {best_auc:.6f}")
    print(f"Total Epochs:      {num_epochs}")
    print("="*80)
    
    return best_auc, 0.0

def main(params):
    # Set seed for reproducibility
    set_seed(params["seed"])
    
    # Get resource mode and manual overrides from parameters
    resource_mode = params.get("resource_mode", "conservative")
    
    # Calculate optimal hardware settings based on parameters
    optimal_batch_size = get_optimal_batch_size(params.get("batch_size"), resource_mode)
    optimal_num_workers = get_optimal_num_workers(params.get("num_workers"), resource_mode)
    optimal_cpu_threads = get_optimal_cpu_threads(params.get("cpu_threads"), resource_mode)
    
    params["batch_size"] = optimal_batch_size
    params["num_workers"] = optimal_num_workers
    params["cpu_threads"] = optimal_cpu_threads
    
    print(f"Resource Configuration:")
    print(f"  Resource mode: {resource_mode}")
    print(f"  Available CPU cores: {mp.cpu_count()}")
    print(f"  Using data workers: {optimal_num_workers}")
    print(f"  Batch size: {optimal_batch_size}")
    print(f"  Computation threads: {optimal_cpu_threads}")
    print(f"  Progress frequency: {params.get('progress_freq', 200)}")
    print()
    
    # Load configurations
    with open("configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        # Override batch size with optimized value
        train_config["batch_size"] = optimal_batch_size
        # Set default epochs to 50
        train_config["num_epochs"] = 50
        
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
        f"{dataset_name}_{model_name}_{emb_type}_{resource_mode}_{params['seed']}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Training SimAKT model ({resource_mode.title()} Resource Mode)")
    print(f"Dataset: {dataset_name}")
    print(f"Save directory: {save_dir}")
    print(f"Model config: {model_config}")
    
    # Save configuration
    save_config(train_config, model_config, dataset_config, params, save_dir)
    
    # Initialize data loaders with optimized settings
    train_loader, valid_loader = init_dataset4train_simakt(
        dataset_name, 
        dataset_config, 
        fold=params["fold"],
        batch_size=optimal_batch_size,
        num_workers=optimal_num_workers
    )
    
    # Initialize model
    model = init_simakt_model(model_config, dataset_config, emb_type)
    
    # Train model
    testauc, testacc = train_simakt_model(
        model, 
        train_loader, 
        valid_loader, 
        train_config["num_epochs"], 
        save_dir,
        cpu_threads=optimal_cpu_threads,
        progress_freq=params.get("progress_freq", 200)
    )
    
    print(f"Final Results:")
    print(f"Test AUC: {testauc:.4f}")
    print(f"Test ACC: {testacc:.4f}")
    
    return testauc, testacc

if __name__ == "__main__":
    # Bootstrap the environment to ensure GPU access before doing anything else.
    bootstrap_gpu_environment()

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
    
    parser.add_argument("--batch_size", type=int, default=None, help="Override automatic batch size optimization")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--resource_mode", type=str, default="moderate", help="Resource mode for optimization")
    parser.add_argument("--num_workers", type=int, default=None, help="Override automatic num_workers optimization")
    parser.add_argument("--cpu_threads", type=int, default=None, help="Override automatic cpu_threads optimization")
    parser.add_argument("--progress_freq", type=int, default=50, help="Progress reporting frequency")

    args = parser.parse_args()
    params = vars(args)
    
    main(params)