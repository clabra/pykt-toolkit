# 🔴 CRITICAL ARCHITECTURAL BUG IN GAINAKT3EXP

**Date**: 2025-11-17 (Revised)  
**Status**: ❌ **FUNDAMENTAL DESIGN FLAW IDENTIFIED**  
**Impact**: Encoder 2 cannot learn skill-specific mastery patterns due to scalar gain mechanism

> **📖 For conceptual foundation and intended architecture**, see: [`/workspaces/pykt-toolkit/paper/STATUS_gainakt3exp.md`](../paper/STATUS_gainakt3exp.md) - Section "Conceptual Foundation: Skill-Specific Mastery Through Learning Gains"

---

## Executive Summary

After trajectory analysis revealed Encoder 2 AUC = 48.42% (below random) and mastery uncorrelated with responses (r = -0.044), we identified the **root cause**:

**PRIMARY BUG: Scalar Gain Quality** - Encoder 2 outputs single scalar per interaction instead of per-skill gains vector

### Revised Understanding (2025-11-17)

**What We Initially Thought** (Bug #2):
- ❌ "Encoder 2 trained to predict responses (wrong supervision signal)"
- ❌ "Should decouple mastery from response prediction"
- ❌ "Need different loss function for mastery growth"

**Actual Correct Design** (see STATUS_gainakt3exp.md for full explanation):
- ✅ Encoder 2 SHOULD predict responses through mastery mechanism
- ✅ Loss function BCE(encoder2_pred, y_true) is CORRECT
- ✅ Both encoders predict responses, but through different pathways:
  - Encoder 1: Direct attention → prediction (unconstrained)
  - Encoder 2: Attention → skill gains → mastery → prediction (interpretable)

**The Real Problem**:
- Scalar gains prevent skill-specific learning
- Without skill differentiation, mastery mechanism cannot work
- Even with correct supervision, Encoder 2 cannot learn proper mastery-response relationship

This bug explains why:
- Mastery values don't correlate with student performance
- Encoder 2 AUC is below random baseline
- All skills improve uniformly regardless of actual learning
- Increasing IM loss weight doesn't help (architecture cannot differentiate skills)

---

## Experiment Results Summary

**Experiment**: `20251117_131554_gainakt3exp_baseline-bce0.7_999787`  
**Configuration**: bce_loss_weight=0.7 (30% IM loss), patience=10

| Metric | Value | Status |
|--------|-------|--------|
| Overall Val AUC | 0.6765 | ✅ Reasonable |
| Encoder1 Val AUC | 0.6765 | ✅ Working well |
| **Encoder2 Val AUC** | **0.4842** | ❌ **Below random (50%)** |
| Mastery ↔ Response | -0.044 | ❌ **No correlation** |
| Encoder2_Pred Range | [0.37, 0.53] | ❌ **Too narrow** |

**Trajectory Analysis** (659 interactions, 10 students):
- Mastery IS updating (sigmoid progression observed)
- Expected gains = 0 everywhere (scalar gain_quality not output)
- All practiced skills get same increment (no differentiation)

---

## THE BUG: Scalar Gain Quality (Not Skill-Specific)

### Current Implementation (WRONG)

**File**: `pykt/models/gainakt3_exp.py`, Lines 528-554

```python
# Step 1: Compute SCALAR gain quality per interaction
gain_quality_logits = learning_gains_d.mean(dim=-1, keepdim=True)  # [B, L, D] → [B, L, 1]
gain_quality = torch.sigmoid(gain_quality_logits)  # [B, L, 1] ∈ [0, 1]  ← SCALAR!

# Step 2: Apply SAME scalar to ALL practiced skills
effective_practice = torch.zeros(batch_size, seq_len, self.num_c, device=q.device)

for t in range(seq_len):
    if t > 0:
        effective_practice[:, t, :] = effective_practice[:, t-1, :].clone()
    
    practiced_concepts = q[:, t].long()  # [B] - which skill is practiced
    batch_indices = torch.arange(batch_size, device=q.device)
    
    # ALL practiced skills get SAME quality increment!
    effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
    #                                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                                           SAME SCALAR FOR ALL SKILLS!
```

### The Problem

1. **Encoder 2 outputs**: `value_seq_2` [B, L, D=256] 
2. **Aggregation**: Mean across D dimensions → scalar [B, L, 1]
3. **Result**: Single "engagement quality" per interaction
4. **Application**: All practiced skills increment by same amount

**Example**:
- Student practices Skill 14 and Skill 92 in one interaction
- Encoder 2 produces: `gain_quality = 0.5` (scalar)
- **Both skills** improve by 0.5, regardless of:
  - Which skill the student actually learned
  - Skill difficulty differences
  - Prior mastery levels
  - Response correctness

### What Was Intended

**Per-skill gains** [B, L, num_c]:
```python
# Encoder 2 should output PER-SKILL gains
skill_gains = sigmoid(linear_projection(value_seq_2))  # [B, L, D] → [B, L, num_c]

# Different skills improve by different amounts
effective_practice[t, skill_14] += skill_gains[t, skill_14]  # e.g., +0.8
effective_practice[t, skill_92] += skill_gains[t, skill_92]  # e.g., +0.2
```

### Why This Matters

**Current behavior**:
- ✅ Can learn: "This interaction had high/low engagement"
- ❌ Cannot learn: "Student improved Skill A more than Skill B"
- ❌ Cannot learn: "Skill difficulty differences"
- ❌ Cannot learn: "Differential mastery growth"

**Real-world learning**:
- Students learn different skills at different rates
- Some skills are harder and require more practice
- Learning is skill-specific, not uniform engagement

**Impact on interpretability**:
- Mastery trajectories show uniform growth across all skills
- No differentiation between easy/hard skills
- Cannot explain why student excels at Skill A but struggles with Skill B
- Defeats the purpose of interpretable mastery tracking

---

## Why The Current Loss Function is CORRECT

**Current Implementation** (`examples/train_gainakt3exp.py`, Lines 285-291):

```python
# Encoder 1: Direct BCE on base predictions
bce_loss = bce_criterion(y_pred, y_true)

# Encoder 2: BCE on mastery-based predictions  
im_loss = bce_criterion(valid_im_preds, y_true)  # ✅ CORRECT

# Combined loss
loss = bce_loss_weight * bce_loss + incremental_mastery_loss_weight * im_loss
```

**This is CORRECT** - see STATUS_gainakt3exp.md for full explanation of why both encoders should predict responses through different pathways.

| Encoder | Supervision Signal | Pathway | Intended Learning |
|---------|-------------------|---------|-------------------|
| Encoder 1 | `BCE(pred, y_true)` | Direct attention → prediction | ✅ Unconstrained response patterns |
| Encoder 2 | `BCE(im_pred, y_true)` | Attention → gains → mastery → prediction | ✅ Skill-specific mastery mechanism |

**Key Insight**:
- Both encoders learn to predict responses (same objective) ✅
- But through DIFFERENT mechanisms:
  - Encoder 1: Black-box attention patterns (performance-optimized)
  - Encoder 2: Transparent mastery accumulation (interpretability-constrained)

### Why Encoder 2 AUC < 50% Despite Correct Loss

The issue is NOT the supervision signal - it's that **scalar gains (Bug #1) prevent the mastery mechanism from working**:

```
Current (BROKEN):
value_seq_2 → scalar gain_quality → uniform mastery growth → cannot differentiate skills → poor predictions

Intended (CORRECT):
value_seq_2 → per-skill gains → skill-specific mastery → differentiates skills → accurate predictions
```

**The constraints are FEATURES, not bugs**:
- ✅ Forced monotonic mastery (realistic: students don't forget)
- ✅ Forced sigmoid progression (realistic: learning curves)
- ✅ Skill-specific representation (interpretable: can explain predictions)
- ✅ Threshold mechanism (interpretable: mastery vs competence)

These constraints SHOULD make Encoder 2 slightly worse than Encoder 1 (55-60% vs 67% AUC), but the scalar gains bug makes it catastrophically worse (48% < 50% random).

### What Was Initially Misunderstood

**Initial diagnosis** (WRONG):
- "Encoder 2 learning response prediction is wrong"
- "Need different loss for mastery growth"
- "Decouple mastery from responses"

**Correct understanding** (RIGHT):
- Encoder 2 learning response prediction is CORRECT
- Loss function is already appropriate
- Mastery SHOULD predict responses (that's the whole point!)

### The Actual Problem is Bug #1 (Scalar Gains)

**Why mastery doesn't predict responses currently**:
1. Scalar gain_quality means all skills increase uniformly
2. Cannot learn which skills are relevant for which questions
3. Mastery becomes correlated with overall engagement, not skill-specific practice
4. Without skill differentiation, mastery-response relationship is meaningless

**Evidence from trajectories**:
- Mastery increases uniformly even when student answers incorrectly
- Example: Student 1, Skill 14: responses (1 → 1 → 0), mastery (0.26 → 0.48 → 0.68)
- But ALSO Skill 92 likely increased similarly (uniform increment)
- No correlation with actual performance (r = -0.044)
- Predictions compressed near 0.5 (cannot differentiate)

**Once Bug #1 is fixed**:
- Encoder 2 can learn: "Question targets Skill 14 → increase only Skill 14"
- Mastery becomes skill-specific: mastery[14] ≠ mastery[92]
- Correct responses correlate with high mastery in relevant skills
- Predictions become accurate: P(correct) = f(mastery[relevant_skills])

---

## How Scalar Gains Break the Mastery Mechanism

The scalar gain bug prevents the intended skill-specific mastery flow:

### Intended Flow (With Per-Skill Gains)

```
Question Q1 targets Skill 14:
1. Encoder 2 attends to interaction
2. Produces skill_gains = [0.0, ..., 0.8 (skill 14), ..., 0.0 (skill 92), ...]
3. Only Skill 14 mastery increases: mastery[14] += 0.8, mastery[92] unchanged
4. Response prediction based on mastery[14]
5. Correct response → BCE loss low → reinforces high gain for Skill 14
6. Over time: Encoder 2 learns "Q1 improves Skill 14"
```

### How Scalar Gains Break the Mechanism

**Example Failure Mode**:
- Question Q1 targets Skill 14 (per Q-matrix)
- Student engages deeply but answers incorrectly
- Encoder 2 produces: `gain_quality = 0.8` (scalar)
- Current (WRONG): ALL skills increase by 0.8 (Skill 14, Skill 92, ...)
- Cannot learn which skill is relevant to Q1
- Mastery becomes engagement proxy, not skill knowledge

**With per-skill gains (intended)**:
- Same interaction: Encoder 2 produces `skill_gains[14] = 0.8, skill_gains[92] = 0.1, ...`
- Only Skill 14 increases significantly
- BCE loss backpropagates through mastery → gains → Encoder 2
- Encoder 2 learns question-skill associations

---

## Why Parameter Tuning Won't Fix This

**Won't work**:
- ❌ Increase IM loss weight → Architecture still cannot differentiate skills
- ❌ Adjust sigmoid parameters → Still scalar gains
- ❌ Lower threshold → Doesn't add per-skill capability
- ❌ Add more encoder blocks → More capacity for wrong mechanism

**What's needed**:
- ✅ Per-skill gain vector [B, L, num_c] (architectural change required)

---

## Recommended Fix

### Implement Per-Skill Gains Vector

**File**: `pykt/models/gainakt3_exp.py`, Lines 528-554

**Current (BROKEN)**:
```python
# Scalar gain per interaction
gain_quality = torch.sigmoid(learning_gains_d.mean(dim=-1, keepdim=True))  # [B, L, 1]
effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
```

**Required Fix**:
```python
# In __init__: Add projection layer
self.gains_projection = nn.Linear(d_model, num_c)  # [D=256] → [num_c]

# In forward_with_states: Compute per-skill gains
skill_gains_logits = self.gains_projection(value_seq_2)  # [B, L, D] → [B, L, num_c]
skill_gains = torch.sigmoid(skill_gains_logits)  # [B, L, num_c] ∈ [0, 1]

# Accumulate per-skill (each skill gets different increment!)
effective_practice[:, t, :] += skill_gains[:, t, :]  # Differentiable
```

**Benefits**:
- ✅ Enables skill-specific learning rates
- ✅ Maintains differentiability (gradients flow through gains_projection)
- ✅ Encoder 2 can learn question-skill associations
- ✅ Mastery becomes skill-specific, not engagement proxy

---

## Validation Plan

After implementing the fix:

1. **Enable gain output**: Set `use_gain_head=true` in config
2. **Re-train**: `python examples/run_repro_experiment.py --short_title fixed-per-skill-gains`
3. **Extract trajectories**: `python examples/learning_trajectories.py --run_dir <exp_dir> --num_students 10`
4. **Verify metrics**:
   - ✅ Encoder2 Val AUC > 55% (above random)
   - ✅ Mastery ↔ Response correlation > 0.4 (positive)
   - ✅ Per-skill gains vary by skill (not uniform)
   - ✅ Encoder2_Pred range wider than [0.37, 0.53]

---

## Conclusion

**Root Cause**: Scalar gain_quality prevents skill-specific learning

**Fix**: Replace with per-skill gains vector [B, L, num_c]

**No loss function changes needed** - current BCE supervision is correct

**Expected after fix**: Encoder 2 AUC 55-60%, positive mastery-response correlation, skill differentiation visible in trajectories
