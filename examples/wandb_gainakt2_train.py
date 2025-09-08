"""
Test this parameters combination to try to improve AUC: 
python wandb_gainakt2_train.py \
    --dataset_name=assist2015 \
    --use_wandb=0 \
    --learning_rate=2e-4 \
    --d_model=256 \
    --num_encoder_blocks=4 \
    --d_ff=1024 \
    --dropout=0.2 \
    --resume_from_checkpoint=1

OR to resume from a specific checkpoint:
python wandb_gainakt2_train.py \
    --dataset_name=assist2015 \
    --resume_checkpoint_path=saved_model/path_to_checkpoint
"""

import argparse
import os
import glob
from wandb_train import main

def find_latest_checkpoint(save_dir, model_name, dataset_name):
    """
    Find the latest checkpoint based on modification time.
    Returns the checkpoint directory path if found, None otherwise.
    """
    if not os.path.exists(save_dir):
        return None
    
    # Look for directories that match our model pattern
    pattern = os.path.join(save_dir, f"*{model_name}*{dataset_name}*")
    checkpoint_dirs = glob.glob(pattern)
    
    if not checkpoint_dirs:
        return None
    
    # Find directories that actually contain model checkpoints
    valid_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if os.path.isdir(checkpoint_dir):
            # Check if it contains a model checkpoint file
            model_files = glob.glob(os.path.join(checkpoint_dir, "*_model.ckpt"))
            config_files = glob.glob(os.path.join(checkpoint_dir, "config.json"))
            if model_files and config_files:
                valid_checkpoints.append(checkpoint_dir)
    
    if not valid_checkpoints:
        return None
    
    # Return the most recently modified checkpoint
    latest_checkpoint = max(valid_checkpoints, key=os.path.getmtime)
    return latest_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="gainakt2")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    
    # GainAKT2 specific parameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_encoder_blocks", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=200)
    
    # Wandb configuration
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    # Checkpoint resuming options
    parser.add_argument("--resume_from_checkpoint", type=int, default=0, 
                       help="1 to auto-resume from latest checkpoint, 0 to start fresh")
    parser.add_argument("--resume_checkpoint_path", type=str, default="", 
                       help="Specific checkpoint path to resume from")
   
    args = parser.parse_args()
    params = vars(args)
    
    # Handle checkpoint resuming
    if args.resume_from_checkpoint == 1 or args.resume_checkpoint_path:
        if args.resume_checkpoint_path:
            # Use specific checkpoint path
            checkpoint_dir = args.resume_checkpoint_path
            if not os.path.exists(checkpoint_dir):
                print(f"ERROR: Specified checkpoint path does not exist: {checkpoint_dir}")
                exit(1)
        else:
            # Auto-find latest checkpoint
            checkpoint_dir = find_latest_checkpoint(
                args.save_dir, args.model_name, args.dataset_name
            )
            
        if checkpoint_dir:
            print(f"Found checkpoint to resume from: {checkpoint_dir}")
            
            # Load the previous config to maintain consistency
            config_path = os.path.join(checkpoint_dir, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
                # Update params with saved model config, but keep new training params if specified
                saved_model_config = saved_config.get('model_config', {})
                saved_params = saved_config.get('params', {})
                
                # Keep the original training parameters (learning_rate, num_epochs, etc.)
                # but restore model architecture parameters from checkpoint
                model_arch_params = ['d_model', 'n_heads', 'num_encoder_blocks', 'd_ff', 'dropout', 'seq_len']
                for param in model_arch_params:
                    if param in saved_model_config:
                        params[param] = saved_model_config[param]
                    elif param in saved_params:
                        params[param] = saved_params[param]
                
                print(f"Restored model architecture parameters from checkpoint:")
                for param in model_arch_params:
                    if param in params:
                        print(f"  {param}: {params[param]}")
            
            # Set the checkpoint path for the training function
            params['resume_checkpoint_dir'] = checkpoint_dir
            
        else:
            print("No previous checkpoint found. Starting training from scratch.")
            if args.resume_from_checkpoint == 1:
                print("(Use --resume_checkpoint_path to specify a specific checkpoint)")
    
    print(f"Starting training with parameters: {params}")
    main(params)