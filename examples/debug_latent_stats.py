
import torch
import torch.nn as nn
import os
import numpy as np
import json
import pandas as pd
from pykt.models import init_model
from pykt.datasets import init_dataset4train

def check_latent_stats(exp_dir):
    print(f"\n--- Checking Latent Stats: {exp_dir} ---")
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Minimal config for loading
    model_config = {
        'd_model': config['defaults']['d_model'],
        'd_ff': config['defaults']['d_ff'],
        'num_attn_heads': config['defaults']['n_heads'],
        'n_blocks': config['defaults']['n_blocks'],
        'dropout': 0,
        'final_fc_dim': config['defaults']['final_fc_dim'],
        'l2': 0,
        'n_uid': 2465 # Hardcode for now as we know it
    }
    
    data_config = {
        'dpath': '/workspaces/pykt-toolkit/data/assist2009',
        'num_c': 110,
        'train_valid_file': 'train_valid_sequences_bkt.csv'
    }
    
    model = init_model('idkt', model_config, data_config, 'qid')
    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pt'), map_location='cpu')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()
    
    # Get stats of student params
    vs = model.student_param.weight.data
    kc = model.student_gap_param.weight.data
    print(f"Student Velocity (vs): mean={vs.mean():.4f}, std={vs.std():.4f}, max={vs.max():.4f}, min={vs.min():.4f}")
    print(f"Student Gap (kc):      mean={kc.mean():.4f}, std={kc.std():.4f}, max={kc.max():.4f}, min={kc.min():.4f}")
    
    # Check axe weights
    ka = model.knowledge_axis_emb.weight.data
    va = model.velocity_axis_emb.weight.data
    print(f"Knowledge Axis:        mean={ka.mean():.4f}, std={ka.std():.4f}")
    print(f"Velocity Axis:         mean={va.mean():.4f}, std={va.std():.4f}")

check_latent_stats('experiments/20251223_193204_idkt_assist2009_baseline_742098')
check_latent_stats('experiments/20251223_215439_idkt_test_978367')
