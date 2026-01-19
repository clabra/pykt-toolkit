# Semantic Axis Collapse Fix

## Problem Identified

After training the quick test model (Exp 337220, 10 epochs), discovered that **all p_ref predictions are completely flat** (identical values across sequences).

### Root Cause Analysis

1. **Symptom**: Same question repeated with different concepts produces identical p_ref predictions
   - Example: Question 367 with concepts [20, 47, 46, 31]
   - p_ref predictions: [0.2455, 0.2455, 0.2455, 0.2455] (std=0.0)
   - p_sup predictions: [0.1101, 0.1935, 0.1649, 0.3009] (std=0.078) ✓ Dynamic

2. **Initial hypothesis**: Question-level evaluation uses question IDs instead of concepts
   - **Disproven**: Verified evaluate_question passes concepts as q_data

3. **Second hypothesis**: Time-varying BKT implementation broken
   - **Disproven**: Verified time-varying code is correct

4. **Root cause discovered**: **Semantic axis embeddings collapsed during training**
   - Checked checkpoint weights:
     ```
     velocity_axis_emb cosine similarities:
       Concept 20 vs 47: 0.999116
       Concept 20 vs 46: 0.999214
       Concept 20 vs 31: 0.999182
       Mean across all pairs: 0.999441
     
     knowledge_axis_emb cosine similarities:
       Mean across all pairs: 0.999463
     ```
   - All 124 concept embeddings are nearly identical!

### Why This Happens

**Initialization**:
```python
self.knowledge_axis_emb = nn.Embedding(n_question + 1, z_dim)
self.velocity_axis_emb = nn.Embedding(n_question + 1, z_dim)
nn.init.normal_(self.knowledge_axis_emb.weight, mean=1.0, std=0.02)  # reset()
nn.init.normal_(self.velocity_axis_emb.weight, mean=1.0, std=0.02)
```

All axes initialized from N(1.0, 0.02²) → nearly identical vectors.

**Training signal**:
- Axes receive gradients through supervised and reference prediction losses
- But there's **no explicit diversity loss** to push them apart
- Without diversity pressure, they remain collapsed
- This defeats the semantic axis projection mechanism

**Effect on predictions**:
- Different concepts → nearly identical v_axis embeddings
- `t_logits = t_base + (z_context * v_axis).sum(dim=-1)`
- If v_axis constant → t_logits barely varies
- → p_t nearly constant across sequence
- → BKT walk produces flat reference predictions

## Fix Implemented

Added **diversity/orthogonality loss** to encourage semantic axes to learn distinct directions.

### Code Changes

**File**: `pykt/models/gtransformer.py`

**Location**: After computing l0_logits and t_logits (around line 277)

**Addition**:
```python
# Diversity Loss: Encourage semantic axes to be different across concepts
# This prevents all axes from collapsing to identical vectors
# Sample a subset of concepts from the current batch to compute diversity
unique_concepts = torch.unique(q_data)
if len(unique_concepts) > 1 and not qtest:  # Only during training
    # Get axes for unique concepts in this batch
    sampled_k_axes = self.knowledge_axis_emb(unique_concepts)  # [N_unique, z_dim]
    sampled_v_axes = self.velocity_axis_emb(unique_concepts)  # [N_unique, z_dim]
    
    # Normalize to unit vectors for cosine similarity
    k_normalized = sampled_k_axes / (sampled_k_axes.norm(dim=1, keepdim=True) + 1e-8)
    v_normalized = sampled_v_axes / (sampled_v_axes.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute pairwise cosine similarities (should be low for diversity)
    k_sim_matrix = k_normalized @ k_normalized.t()  # [N_unique, N_unique]
    v_sim_matrix = v_normalized @ v_normalized.t()  # [N_unique, N_unique]
    
    # Penalize high off-diagonal similarities (we want orthogonal axes)
    # Mask out diagonal (self-similarity = 1.0)
    mask = ~torch.eye(len(unique_concepts), dtype=torch.bool, device=q_data.device)
    
    # Mean absolute cosine similarity (want this near 0)
    k_diversity_loss = k_sim_matrix[mask].abs().mean()
    v_diversity_loss = v_sim_matrix[mask].abs().mean()
    
    diversity_loss = 0.01 * (k_diversity_loss + v_diversity_loss)  # Small weight
else:
    diversity_loss = torch.tensor(0.0, device=q_data.device)

# Add diversity loss to regularization
total_reg_loss = c_reg_loss + diversity_loss
```

**Loss integration**:
```python
# Before:
return outputs, c_reg_loss

# After:
return outputs, total_reg_loss
```

### How It Works

1. **Sample unique concepts** from current batch
2. **Normalize axes** to unit vectors (for cosine similarity)
3. **Compute pairwise similarities** (N×N matrix)
4. **Penalize high off-diagonal values** (we want low cosine sim = orthogonal)
5. **Add small weighted loss** (0.01 coefficient to avoid overwhelming main loss)
6. **Only during training** (not during qtest evaluation)

### Expected Outcome

After retraining with this fix:
- Semantic axes will learn diverse directions
- Different concepts will have different v_axis embeddings
- p_t will vary across sequences (not flat)
- p_ref predictions will become dynamic
- Interpretability gap should decrease

### Verification Plan

After retraining:
1. Run `debug_semantic_axes.py` to check cosine similarities (should be << 0.999)
2. Check p_ref prediction variance (std should be > 0)
3. Compare with p_sup variance (should be similar)
4. Verify p_ref AUC improves

## Historical Context

- **Baseline (Exp 533154, 200 epochs)**: p_ref AUC = 0.6756 (flat predictions due to static BKT)
- **Quick test (Exp 337220, 10 epochs)**: p_ref AUC = 0.6424 (flat predictions due to axis collapse)
- **Target**: p_ref dynamic with AUC ≥ 0.67 (matching or exceeding baseline)

## Next Steps

1. ✅ **Fix identified and implemented** (diversity loss added)
2. ⏳ **Rerun quick test** (10 epochs to verify fix works)
3. ⏳ **Full training** (200 epochs if quick test succeeds)
4. ⏳ **Regenerate all plots** with corrected model
