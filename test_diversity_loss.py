#!/usr/bin/env python3
"""Test that the diversity loss is computed correctly."""

import torch
import torch.nn as nn

# Simulate the diversity loss computation
n_question = 124
z_dim = 128
batch_size = 16
seqlen = 50

# Create dummy embeddings (simulating collapsed axes)
knowledge_axis_emb = nn.Embedding(n_question + 1, z_dim)
velocity_axis_emb = nn.Embedding(n_question + 1, z_dim)

# Initialize to nearly identical vectors (simulating collapse)
nn.init.normal_(knowledge_axis_emb.weight, mean=1.0, std=0.02)
nn.init.normal_(velocity_axis_emb.weight, mean=1.0, std=0.02)

# Create dummy q_data
q_data = torch.randint(0, n_question, (batch_size, seqlen))

# Test diversity loss computation
unique_concepts = torch.unique(q_data)
print(f"Batch has {len(unique_concepts)} unique concepts")

if len(unique_concepts) > 1:
    # Get axes for unique concepts in this batch
    sampled_k_axes = knowledge_axis_emb(unique_concepts)  # [N_unique, z_dim]
    sampled_v_axes = velocity_axis_emb(unique_concepts)  # [N_unique, z_dim]
    
    print(f"sampled_k_axes shape: {sampled_k_axes.shape}")
    print(f"sampled_v_axes shape: {sampled_v_axes.shape}")
    
    # Normalize to unit vectors for cosine similarity
    k_normalized = sampled_k_axes / (sampled_k_axes.norm(dim=1, keepdim=True) + 1e-8)
    v_normalized = sampled_v_axes / (sampled_v_axes.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute pairwise cosine similarities
    k_sim_matrix = k_normalized @ k_normalized.t()  # [N_unique, N_unique]
    v_sim_matrix = v_normalized @ v_normalized.t()  # [N_unique, N_unique]
    
    print(f"k_sim_matrix shape: {k_sim_matrix.shape}")
    print(f"Sample k_sim_matrix diagonal: {k_sim_matrix.diag()[:5].tolist()}")
    print(f"Sample k_sim_matrix off-diagonal: {k_sim_matrix[0, 1:6].tolist()}")
    
    # Mask out diagonal
    mask = ~torch.eye(len(unique_concepts), dtype=torch.bool)
    
    # Mean absolute cosine similarity
    k_diversity_loss = k_sim_matrix[mask].abs().mean()
    v_diversity_loss = v_sim_matrix[mask].abs().mean()
    
    diversity_loss = 0.01 * (k_diversity_loss + v_diversity_loss)
    
    print(f"\nDiversity loss components:")
    print(f"  k_diversity_loss: {k_diversity_loss.item():.6f}")
    print(f"  v_diversity_loss: {v_diversity_loss.item():.6f}")
    print(f"  total diversity_loss: {diversity_loss.item():.6f}")
    
    # Check gradients
    diversity_loss.backward()
    
    print(f"\nGradient check:")
    print(f"  knowledge_axis_emb.weight.grad is not None: {knowledge_axis_emb.weight.grad is not None}")
    print(f"  velocity_axis_emb.weight.grad is not None: {velocity_axis_emb.weight.grad is not None}")
    
    if knowledge_axis_emb.weight.grad is not None:
        grad_norm = knowledge_axis_emb.weight.grad.norm().item()
        grad_nonzero = (knowledge_axis_emb.weight.grad.abs() > 1e-8).sum().item()
        print(f"  knowledge_axis grad norm: {grad_norm:.6f}")
        print(f"  knowledge_axis grad nonzero elements: {grad_nonzero}/{knowledge_axis_emb.weight.numel()}")
    
    print("\n✓ Diversity loss computation successful!")
    print("✓ Gradients flow through semantic axes!")
    
    # Expected behavior
    if k_diversity_loss > 0.9:
        print("\n⚠️  WARNING: Axes are highly similar (collapsed)")
        print("    This is expected with N(1.0, 0.02) initialization.")
        print("    During training, diversity loss will push them apart.")
    else:
        print("\n✓ Axes are reasonably diverse!")
