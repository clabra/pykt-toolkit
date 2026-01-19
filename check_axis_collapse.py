#!/usr/bin/env python3
"""Check if knowledge_axis_emb also collapsed."""

import torch

ckpt_path = "experiments/20260119_075752_gtransformer_quicktest_timevarying_337220/gtransformer_assist2009_0_3407_64_8_2_0.0001/qid_model.ckpt"
checkpoint = torch.load(ckpt_path, map_location='cpu')

k_axis_weight = checkpoint['knowledge_axis_emb.weight']
print(f"knowledge_axis_emb.weight shape: {k_axis_weight.shape}")

test_concepts = [20, 47, 46, 31]
print("\nPairwise cosine similarities (knowledge axis):")
for i, c1 in enumerate(test_concepts):
    for j, c2 in enumerate(test_concepts):
        if j > i:
            v1 = k_axis_weight[c1]
            v2 = k_axis_weight[c2]
            cos_sim = (v1 @ v2) / (v1.norm() * v2.norm())
            print(f"  Concept {c1} vs {c2}: cos_sim = {cos_sim.item():.6f}")

# Check all embeddings
all_k_cos_sims = []
for i in range(len(k_axis_weight)):
    for j in range(i+1, len(k_axis_weight)):
        v1 = k_axis_weight[i]
        v2 = k_axis_weight[j]
        cos_sim = (v1 @ v2) / (v1.norm() * v2.norm())
        all_k_cos_sims.append(cos_sim.item())

print(f"\nAll pairs statistics (knowledge_axis):")
print(f"  Min cos_sim: {min(all_k_cos_sims):.6f}")
print(f"  Max cos_sim: {max(all_k_cos_sims):.6f}")
print(f"  Mean cos_sim: {sum(all_k_cos_sims)/len(all_k_cos_sims):.6f}")

all_v_cos_sims = []
v_axis_weight = checkpoint['velocity_axis_emb.weight']
for i in range(len(v_axis_weight)):
    for j in range(i+1, len(v_axis_weight)):
        v1 = v_axis_weight[i]
        v2 = v_axis_weight[j]
        cos_sim = (v1 @ v2) / (v1.norm() * v2.norm())
        all_v_cos_sims.append(cos_sim.item())

print(f"\nAll pairs statistics (velocity_axis):")
print(f"  Min cos_sim: {min(all_v_cos_sims):.6f}")
print(f"  Max cos_sim: {max(all_v_cos_sims):.6f}")
print(f"  Mean cos_sim: {sum(all_v_cos_sims)/len(all_v_cos_sims):.6f}")

print("\n" + "="*60)
if sum(all_k_cos_sims)/len(all_k_cos_sims) > 0.95 and sum(all_v_cos_sims)/len(all_v_cos_sims) > 0.95:
    print("❌ CRITICAL: Both semantic axes collapsed!")
    print("   All concept embeddings are nearly identical.")
    print("   This defeats the purpose of semantic axis projection.")
    print("\nROOT CAUSE:")
    print("  Likely insufficient training or missing regularization.")
    print("  Need to check:")
    print("    1. Are semantic axes updated during training?")
    print("    2. Is there diversity loss or orthogonality constraint?")
    print("    3. Is learning rate too low for these parameters?")
else:
    print("✓ At least one semantic axis has diversity.")
