"""
Shared helper functions for validation scripts.
"""

import os
import sys
import json
import torch
import pickle

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from pykt.models import init_model


def load_model_from_dir(exp_dir, device):
    """
    Helper to load a GTransformer model from an experiment directory.
    
    Returns:
        model: Loaded model
        dc: Data config for the dataset
        mc: Model config
        dpath: Path to dataset directory
        bkt_params: BKT parameters (if available)
    """
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    mc = config.get('model_config', config.get('train_config', config.get('params', {})))
    dataset_name = config.get('dataset_name', config.get('params', {}).get('dataset_name', 'assist2009'))
    
    # Robust defaults for older configs
    defaults = {
        'kq_same': 1, 'separate_qa': 0, 'l2_rasch': 0.0, 
        'pretrain_dim': 768, 'ablation': 'none', 'n_uid': 0,
        'num_attn_heads': 8, 'final_fc_dim': 512,
        'd_model': 64, 'n_blocks': 2, 'd_ff': 256, 'dropout': 0.1
    }
    for k, v in defaults.items():
        if k not in mc: 
            mc[k] = v
    
    # Force alignment between num_attn_heads and n_heads
    if 'n_heads' in mc:
        mc['num_attn_heads'] = mc['n_heads']
    elif 'num_attn_heads' in mc:
        mc['n_heads'] = mc['num_attn_heads']
        
    # Data config for model init (dimensions)
    data_config_path = os.path.join(PROJECT_ROOT, "configs/data_config.json")
    with open(data_config_path, 'r') as f:
        dc = json.load(f)
    
    dpath = dc[dataset_name]['dpath'].replace("../", "")
    dpath = os.path.join(PROJECT_ROOT, dpath)
    dc[dataset_name]['dpath'] = dpath

    model = init_model('gtransformer', mc, dc[dataset_name], mc.get('emb_type', 'qid'))
    
    # Checkpoint discovery
    checkpoint_path = None
    for root, _, files in os.walk(exp_dir):
        for f in files:
            if f.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, f)
                break
        if checkpoint_path: 
            break
        
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in {exp_dir}")

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.to(device)
    model.eval()

    # Load Theory Params
    bkt_params = None
    theory_path = os.path.join(dpath, "bkt_skill_params.pkl")
    if os.path.exists(theory_path):
        with open(theory_path, "rb") as f:
            bkt_params = pickle.load(f)
        model.load_theory_params(bkt_params)
    
    return model, dc[dataset_name], mc, dpath, bkt_params
