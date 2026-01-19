#!/usr/bin/env python3
"""Debug script to inspect p_t values during inference"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.models import load_model
from pykt.datasets import init_dataset4train, init_test_datasets
from pykt.config import que_type_models
import os

# Load the quicktest model
exp_dir = 'experiments/20260119_075752_gtransformer_quicktest_timevarying_337220'
model_name = 'gtransformer'
dataset_name = 'assist2009'

# Find checkpoint
import glob
ckpt_files = glob.glob(f'{exp_dir}/**/qid_model.ckpt', recursive=True)
if not ckpt_files:
    print("No checkpoint found!")
    sys.exit(1)

ckpt_path = ckpt_files[0]
print(f"Loading model from: {ckpt_path}")

# Load model
device = torch.device('cpu')
checkpoint = torch.load(ckpt_path, map_location=device)
model_config = checkpoint['model_config']
model = load_model(model_name, model_config, checkpoint['net'])
model.to(device)
model.eval()

# Load test data
from pykt.datasets.data_loader import KTQueDataset
test_file = '/workspaces/pykt-toolkit/data/assist2009/test_question_sequences.csv'

print(f"\nLoading test data from: {test_file}")
test_dataset = KTQueDataset(test_file, input_type=['questions', 'concepts'], 
                             folds=[-1], qtest=True)

# Get one batch
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
batch = next(iter(test_loader))

print(f"\nBatch keys: {batch.keys()}")
print(f"Questions shape: {batch['questions'].shape}")
print(f"Concepts shape: {batch['concepts'].shape}")
print(f"Responses shape: {batch['responses'].shape}")

# Extract data
q_data = batch['concepts']  # Use concepts as q_data for qid mode
target = batch['responses']

print(f"\nFirst sequence:")
print(f"  Concepts (q_data): {q_data[0, :10].tolist()}")
print(f"  Responses (target): {target[0, :10].tolist()}")

# Run forward pass
with torch.no_grad():
    # Manually extract the grounded parameters
    from pykt.models.gtransformer import GTransformer
    
    # Get embeddings
    q_embed_data, qa_embed_data = model.base_emb(q_data, target)
    d_output = model.model(q_embed_data, qa_embed_data, None)
    z_context = torch.cat([d_output, q_embed_data], dim=-1)
    
    # Get axes and bases
    k_axis = model.knowledge_axis_emb(q_data)
    v_axis = model.velocity_axis_emb(q_data)
    l0_base = model.l0_base_emb(q_data).squeeze(-1)
    t_base = model.t_base_emb(q_data).squeeze(-1)
    
    # Compute logits
    l0_logits = l0_base + (z_context * k_axis).sum(dim=-1)
    t_logits = t_base + (z_context * v_axis).sum(dim=-1)
    
    # Get probabilities
    p_l0 = torch.sigmoid(l0_logits)
    p_t = torch.sigmoid(t_logits)
    
    print(f"\nFirst sequence p_t values:")
    print(f"  p_t: {p_t[0, :20].tolist()}")
    print(f"  Mean: {p_t[0].mean():.6f}")
    print(f"  Std: {p_t[0].std():.6f}")
    print(f"  Range: [{p_t[0].min():.6f}, {p_t[0].max():.6f}]")
    
    print(f"\nFirst sequence p_l0 values:")
    print(f"  p_l0: {p_l0[0, :20].tolist()}")
    print(f"  Mean: {p_l0[0].mean():.6f}")
    print(f"  Std: {p_l0[0].std():.6f}")
    print(f"  Range: [{p_l0[0].min():.6f}, {p_l0[0].max():.6f}]")
    
    # Check if p_t is constant
    if p_t[0].std() < 0.01:
        print(f"\n⚠️  WARNING: p_t is nearly constant (std < 0.01)!")
        print(f"    This will cause flat p_ref predictions.")
        
        # Debug why
        print(f"\n  Debugging why p_t is flat:")
        print(f"    t_base (first 10): {t_base[0, :10].tolist()}")
        print(f"    t_base std: {t_base[0].std():.6f}")
        
        # Check if z_context projection is varying
        proj = (z_context * v_axis).sum(dim=-1)
        print(f"    (z · v_axis) projection (first 10): {proj[0, :10].tolist()}")
        print(f"    projection std: {proj[0].std():.6f}")
