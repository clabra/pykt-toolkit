# Quick Test Rerun Plan - With Diversity Loss Fix

## Problem Summary

**Discovery**: After 10-epoch quick test (Exp 337220), all p_ref predictions were completely flat.

**Root cause**: Semantic axis embeddings (`knowledge_axis_emb` and `velocity_axis_emb`) collapsed during training - all 124 concept embeddings became nearly identical (cosine similarity > 0.999).

**Why it happens**: 
- Initialized from N(1.0, 0.02²) → nearly identical vectors
- No explicit diversity loss → axes remain collapsed
- Collapsed axes → constant projections → flat p_t → flat p_ref predictions

## Fix Implemented

**File modified**: `pykt/models/gtransformer.py`

**Changes**:
1. Added diversity/orthogonality loss after semantic axis projections
2. Penalizes high pairwise cosine similarities between concept axes
3. Loss weight: 0.01 (small to avoid overwhelming main losses)
4. Only computed during training (not qtest)

**Verification**: 
- ✓ Syntax check passed
- ✓ Diversity loss computation tested
- ✓ Gradients flow correctly

## Next Steps

### Option A: Quick Rerun (10 epochs) - RECOMMENDED

**Why**: Verify fix works before committing to 200-epoch run

**Command**:
```bash
# Same config as previous quick test but with diversity loss fix
wandb agent concha-labs/pykt-assistant/337220
```

**Expected outcomes if fix works**:
- Semantic axes cosine similarity < 0.95 (diverse)
- p_ref predictions have variance (std > 0)
- p_ref AUC ≥ 0.64 (at least matching broken quick test)

**Duration**: ~1 hour

**Decision point**: If successful → proceed to Option B (full training)

### Option B: Full Training (200 epochs)

**When**: After Option A succeeds

**Command**:
```bash
# Use same config as baseline Exp 533154 but with both fixes:
# 1. Time-varying BKT
# 2. Diversity loss
wandb agent concha-labs/pykt-assistant/NEW_SWEEP_ID
```

**Expected outcomes**:
- p_ref AUC > 0.67 (exceeding baseline 0.6756)
- Reduced interpretability gap
- Dynamic prediction envelopes in all plots

**Duration**: ~20 hours

## Validation Checklist

After retraining:

1. **Semantic Axes**:
   ```bash
   python3 debug_semantic_axes.py
   ```
   - [ ] Mean cosine similarity < 0.95 (was 0.9994)
   - [ ] Axes have distinct directions

2. **Predictions**:
   ```bash
   python3 analyze_flat_predictions.py
   ```
   - [ ] p_ref predictions have std > 0 (was 0.0)
   - [ ] p_ref variance similar to p_sup

3. **Metrics**:
   - [ ] p_ref AUC ≥ 0.67 (baseline was 0.6756)
   - [ ] Interpretability gap ≤ 0.08 (baseline was 0.1107)

4. **Plots**:
   ```bash
   python3 assistant/generate_prediction_envelope_gallery.py --experiment_dir <new_exp_dir>
   ```
   - [ ] All 4 plots show dynamic envelopes (not flat)
   - [ ] p_ref envelope follows p_sup reasonably

## Files Modified

1. **pykt/models/gtransformer.py**:
   - Added diversity loss computation (lines ~277-307)
   - Integrated into regularization loss (line ~361)

2. **Documentation**:
   - `SEMANTIC_AXIS_COLLAPSE_FIX.md`: Detailed analysis and fix
   - `NEXT_STEPS_DIVERSITY_FIX.md`: This file

3. **Diagnostic tools**:
   - `debug_semantic_axes.py`: Check axis collapse
   - `check_axis_collapse.py`: Full analysis of all axes
   - `test_diversity_loss.py`: Unit test for diversity loss
   - `analyze_flat_predictions.py`: Check prediction variance

## Rollback Plan

If diversity loss causes training instability:

1. Reduce diversity loss weight from 0.01 to 0.001
2. Or change from mean to max cosine similarity penalty
3. Or use different normalization (L2 penalty instead of cosine)

## Historical Context

- **Baseline (Exp 533154)**: Static BKT, p_ref AUC 0.6756, flat predictions
- **Quick test (Exp 337220)**: Time-varying BKT, p_ref AUC 0.6424, flat predictions (axis collapse)
- **Next (Option A)**: Time-varying BKT + diversity loss, 10 epochs, expect dynamic predictions
- **Goal (Option B)**: Time-varying BKT + diversity loss, 200 epochs, p_ref AUC > 0.67
