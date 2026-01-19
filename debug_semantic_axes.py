#!/usr/bin/env python3
"""
Debug why semantic axes produce flat p_t values.
Test: Same question (367) with different concepts (20,47,46,31).
Expected: Different v_axis for each concept → different p_t.
Actual: Need to verify.
"""

import torch
import numpy as np
import json
import os

# Checkpoint path
exp_dir = "experiments/20260119_075752_gtransformer_quicktest_timevarying_337220/gtransformer_assist2009_0_3407_64_8_2_0.0001"
ckpt_path = os.path.join(exp_dir, "qid_model.ckpt")

# Load model checkpoint
print(f"Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Get n_question from checkpoint shape
v_axis_weight = checkpoint['velocity_axis_emb.weight']  # Shape: (n_question, z_dim)
n_question = v_axis_weight.shape[0]
    
print(f"n_question: {n_question}")

# Extract velocity_axis_emb weights
print(f"velocity_axis_emb.weight shape: {v_axis_weight.shape}")

# Test concepts from our flat prediction example
# Question 367 repeated 4 times with concepts: 20, 47, 46, 31
test_concepts = [20, 47, 46, 31]

print("\nExtracting v_axis for test concepts:")
for c in test_concepts:
    v = v_axis_weight[c]
    print(f"  Concept {c}: v_axis norm = {v.norm().item():.6f}, mean = {v.mean().item():.6f}")

# Check if they're identical (which would be wrong)
v_20 = v_axis_weight[20]
v_47 = v_axis_weight[47]
v_46 = v_axis_weight[46]
v_31 = v_axis_weight[31]

print("\nPairwise cosine similarities:")
for i, c1 in enumerate(test_concepts):
    for j, c2 in enumerate(test_concepts):
        if j > i:
            v1 = v_axis_weight[c1]
            v2 = v_axis_weight[c2]
            cos_sim = (v1 @ v2) / (v1.norm() * v2.norm())
            print(f"  Concept {c1} vs {c2}: cos_sim = {cos_sim.item():.6f}")

# Check if the axes are near-zero (which could cause flatness)
print("\nNorm statistics:")
all_norms = v_axis_weight.norm(dim=1)
print(f"  Min norm: {all_norms.min().item():.6f}")
print(f"  Max norm: {all_norms.max().item():.6f}")
print(f"  Mean norm: {all_norms.mean().item():.6f}")
print(f"  Std norm: {all_norms.std().item():.6f}")

# Test prediction scenario
# If v_axis is near zero, then (z_context * v_axis).sum(dim=-1) ≈ 0
# → t_logits ≈ t_base (constant)
# → p_t ≈ sigmoid(t_base) (constant)

print("\nChecking t_base values:")
t_base_weight = checkpoint['t_base_emb.weight']
print(f"t_base_emb.weight shape: {t_base_weight.shape}")
for c in test_concepts:
    t_b = t_base_weight[c]
    print(f"  Concept {c}: t_base = {t_b.item():.6f}, sigmoid(t_base) = {torch.sigmoid(t_b).item():.6f}")

# HYPOTHESIS:
# If v_axis norms are very small, then even varying z_context won't change t_logits much
# → p_t stays nearly constant across sequence
# → BKT walk produces flat predictions

print("\n" + "="*60)
print("DIAGNOSIS:")
if all_norms.mean() < 0.1:
    print("❌ PROBLEM FOUND: velocity_axis_emb weights are near-zero!")
    print("   This causes p_t to be dominated by t_base (constant).")
    print("   Solution: Check weight initialization or training dynamics.")
elif torch.allclose(v_20, v_47, atol=1e-3) and torch.allclose(v_20, v_46, atol=1e-3):
    print("❌ PROBLEM FOUND: velocity_axis_emb weights are identical for different concepts!")
    print("   This shouldn't happen unless embeddings collapsed during training.")
else:
    print("✓ velocity_axis_emb weights look reasonable.")
    print("  Issue must be elsewhere (e.g., z_context not varying, or BKT walk itself).")
