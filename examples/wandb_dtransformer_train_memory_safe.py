#!/usr/bin/env python3
"""
Memory-Safe DTransformer Training Script
Automatically falls back to CPU/RAM when CUDA memory is insufficient
"""

import argparse
import torch
import logging
import psutil
import os
from wandb_train import main

# Configure logging for memory management
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_system_resources():
    """Check available system resources and recommend settings"""
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"🖥️  GPU Memory: {gpu_memory_gb:.1f} GB")
        except:
            gpu_memory_gb = 0
            logging.warning("⚠️ Could not query GPU memory")
    
    logging.info(f"💾 Total RAM: {ram_gb:.1f} GB (Available: {available_ram_gb:.1f} GB)")
    logging.info(f"🔧 CPU Cores: {cpu_cores} physical, {logical_cores} logical")
    
    # Set optimal settings based on resources
    if available_ram_gb < 8:
        logging.warning("⚠️ Low RAM detected. Consider closing other applications.")
        torch.set_num_threads(max(1, cpu_cores // 2))
    else:
        torch.set_num_threads(cpu_cores)
    
    return {
        'ram_gb': ram_gb,
        'available_ram_gb': available_ram_gb,
        'cpu_cores': cpu_cores,
        'cuda_available': cuda_available
    }

def optimize_parameters_for_memory(args, use_cpu=False):
    """Optimize model parameters based on available memory"""
    if use_cpu:
        # CPU-optimized parameters - can handle larger models with more RAM
        logging.info("🔄 Optimizing parameters for CPU/RAM execution")
        
        # Increase model capacity since RAM is typically more abundant
        if args.d_model < 128:
            args.d_model = 128
            logging.info(f"📈 Increased d_model to {args.d_model} for CPU efficiency")
        
        if args.d_ff < 256:
            args.d_ff = 256
            logging.info(f"📈 Increased d_ff to {args.d_ff} for CPU efficiency")
            
        # Optimize threading for CPU
        args.num_workers = min(4, psutil.cpu_count(logical=False))
        
    else:
        # CUDA-optimized parameters - keep conservative for GPU memory
        logging.info("🚀 Optimizing parameters for CUDA execution")
        
        # Ensure parameters are memory-safe for GPU
        if args.d_model > 128:
            args.d_model = 128
            logging.info(f"📉 Reduced d_model to {args.d_model} for CUDA memory safety")
        
        if args.n_know > 8:
            args.n_know = 8
            logging.info(f"📉 Reduced n_know to {args.n_know} for CUDA memory safety")
    
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Safe DTransformer Training")
    
    # Dataset and model parameters
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dtransformer")
    parser.add_argument("--emb_type", type=str, default="qid_cl")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Memory-optimized model architecture
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--num_attn_heads", type=int, default=2)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--n_know", type=int, default=4)
    
    # DTransformer specific parameters
    parser.add_argument("--lambda_cl", type=float, default=0.05)  # Reduced for stability
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--proj", type=str2bool, default=False)  # Disabled to save memory
    parser.add_argument("--hard_neg", type=str2bool, default=False)
    
    # System parameters
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--force_cpu", type=str2bool, default=False, 
                      help="Force CPU execution even if CUDA is available")
    
    args = parser.parse_args()
    
    # Check system resources
    system_info = check_system_resources()
    
    # Force CPU if requested or if CUDA is not available
    if args.force_cpu or not system_info['cuda_available']:
        logging.info("💻 Using CPU/RAM for training")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        use_cpu = True
    else:
        logging.info("🚀 Attempting CUDA training with memory monitoring")
        use_cpu = False
    
    # Optimize parameters based on execution target
    args = optimize_parameters_for_memory(args, use_cpu)
    
    # Set optimal batch size based on memory type
    if use_cpu:
        # CPU can handle larger batches with abundant RAM
        batch_size = min(512, int(system_info['available_ram_gb'] * 32))
        torch.set_num_threads(system_info['cpu_cores'])
        logging.info(f"🔧 CPU batch size: {batch_size}, threads: {system_info['cpu_cores']}")
    else:
        # Conservative batch size for CUDA
        batch_size = 256
        logging.info(f"🚀 CUDA batch size: {batch_size}")
    
    # Log final configuration
    logging.info("🎯 Final DTransformer Configuration:")
    logging.info(f"   📊 Dataset: {args.dataset_name}")
    logging.info(f"   🧠 d_model: {args.d_model}, d_ff: {args.d_ff}")
    logging.info(f"   🔢 n_heads: {args.num_attn_heads}, n_blocks: {args.n_blocks}")
    logging.info(f"   🧮 n_know: {args.n_know}, lambda_cl: {args.lambda_cl}")
    logging.info(f"   💾 Execution: {'CPU/RAM' if use_cpu else 'CUDA/GPU'}")
    
    try:
        # Convert args to parameters dictionary
        params = vars(args)
        
        # Remove system-specific parameters that shouldn't go to the model
        model_params = {k: v for k, v in params.items() 
                       if k not in ['force_cpu', 'num_workers']}
        
        # Start training with memory monitoring
        logging.info("🚀 Starting DTransformer training with memory-safe execution...")
        main(model_params)
        
        logging.info("✅ DTransformer training completed successfully!")
        
    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"❌ CUDA out of memory: {str(e)}")
        if not use_cpu:
            logging.info("🔄 Automatically retrying with CPU/RAM...")
            # Retry with CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            args = optimize_parameters_for_memory(args, use_cpu=True)
            model_params = {k: v for k, v in vars(args).items() 
                           if k not in ['force_cpu', 'num_workers']}
            main(model_params)
        else:
            raise e
    
    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")
        raise e