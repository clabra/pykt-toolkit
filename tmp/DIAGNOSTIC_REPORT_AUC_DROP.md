# Diagnostic Report: Encoder 1 AUC Drop Investigation

**Date**: November 19, 2025  
**Issue**: Encoder 1 validation AUC dropped from 0.7232 to 0.6645 (~5.8 points)  
**Status**: 🔴 ROOT CAUSE NOT IDENTIFIED - REVERTING TO LAST KNOWN GOOD COMMIT

---

## Executive Summary

After extensive investigation (10+ experiments, multiple code comparisons, and systematic parameter analysis), we have been unable to identify the root cause of the Encoder 1 AUC drop that occurred after commit b59d7e41. Despite reverting specific changes, adjusting parameters, and testing multiple hypotheses, the performance degradation persists. The decision has been made to revert to commit 2acdc355 (the last known good state) and restart development from that baseline.

---

## Timeline of Events

### Good State (Commit 2acdc355 - Nov 16, 02:28 UTC)
- **Experiment**: 714616 (dual-loss)
- **Date**: Nov 16, 2025, 12:51 UTC
- **Validation AUC**: 0.7232 ✅
- **Test AUC**: 0.7183
- **Encoder 2 AUC**: 0.490 (random baseline, as expected)
- **Configuration**: Clean dual-encoder with weighted loss (BCE=0.9, IM=0.1)

### Breaking Change (Commit b59d7e41 - Nov 16, 15:06 UTC)
- **Commit Message**: "feat: dual-encoder version with cleaned-up parameters"
- **Changes**: Deprecated 27 parameters, major cleanup of training script
- **Files Modified**: 43 files, +9258 lines, -2123 lines
- **Key Changes**:
  - Massive simplification of `train_gainakt3exp.py` (2081 lines → 745 lines)
  - Parameter deprecation in `configs/parameter_default.json`
  - Cleanup of constraint losses and semantic losses

### Failed State (All Experiments After b59d7e41)
- **Experiments**: 281602, 143029, 869310, and many others
- **Date Range**: Nov 16-19, 2025
- **Validation AUC**: 0.665-0.690 ❌
- **Performance Drop**: ~5-6 AUC points
- **Encoder 2 Behavior**: Learning (AUC ~0.59) instead of random

---

## Investigation Attempts

### 1. IM Loss Target Hypothesis (Nov 19, 2025)
**Hypothesis**: IM loss was using wrong target (responses_shifted instead of responses)

**Action Taken**:
- Reverted IM loss to use `responses` instead of `responses_shifted`
- Experiment 869310 launched with fix

**Result**: ❌ FAILED
- Val AUC: 0.6645 (WORSE than before!)
- BCE Loss: 0.553
- IM Loss: 0.605
- Encoder2 AUC: 0.592 (still learning)

**Conclusion**: IM loss target was NOT the root cause

### 2. Sigmoid Double Application Hypothesis (Nov 19, 2025)
**Hypothesis**: Predictions being sigmoid-ed twice (in model and evaluation)

**Investigation**:
- Checked baseline code: Uses `logits` correctly (lines 360, 445)
- Checked current code: Also uses `logits` correctly (line 360)
- Compared forward() return values

**Result**: ✅ NO ISSUE FOUND
- Both baseline and current code handle sigmoid consistently
- Model returns both `logits` and `predictions` (sigmoid applied)
- Training uses `logits` for BCE loss
- Evaluation uses `predictions` (already sigmoid)

**Conclusion**: No double sigmoid issue exists

### 3. num_students Parameter Hypothesis (Nov 19, 2025)
**Hypothesis**: Mismatch between num_students parameter (3055) and dataset (12220)

**Investigation**:
- Dataset metadata: 12220 students (max_id=12219)
- Baseline parameter: 3055
- Current parameter: 3055
- Usage analysis: `gamma_student[student_ids]` indexing when use_student_speed=True

**Findings**:
- ✅ Baseline had `use_student_speed=False` (no indexing, safe mismatch)
- ✅ Current code also has `use_student_speed=False` (safe)
- ✅ Added validation logic to prevent future issues
- ✅ Documented in Appendix A of STATUS_gainakt3exp.md

**Result**: ✅ NO ISSUE - Working as intended
- Mismatch is safe because gamma_student.mean() used as fallback
- Parameter value matches baseline exactly

**Conclusion**: num_students is NOT the root cause

### 4. assist2009 Parameters Hypothesis (Nov 19, 2025)
**Hypothesis**: Model config might be using assist2009 parameters instead of assist2015

**Investigation**:
- Checked baseline config: `dataset: "assist2015"` ✅
- Checked current training script: No hardcoded assist2009 references ✅
- Searched codebase: Only found in old sweep documentation

**Result**: ✅ NO ISSUE FOUND

**Conclusion**: Dataset configuration is correct

### 5. use_gain_head Parameter Hypothesis (Nov 19, 2025)
**Hypothesis**: use_gain_head might be defaulting to True in model __init__

**Investigation**:
- Checked model __init__ signature (lines 92-106)
- Searched for self.use_gain_head usage

**Findings**:
- ✅ Parameter does NOT exist in current __init__ signature
- ✅ Only one commented-out reference found (line 584)
- ✅ Properly deprecated and removed from model

**Result**: ✅ NO ISSUE FOUND

**Conclusion**: use_gain_head is properly deprecated

### 6. V3+ Features Hypothesis (Nov 18, 2025)
**Hypothesis**: V3+ features (variance loss, contrastive loss, asymmetric init) degrading performance

**Action Taken**:
- Disabled all V3+ features:
  - variance_loss_weight: 0.0
  - skill_contrastive_loss_weight: 0.0
  - beta_spread_regularization_weight: 0.0
  - gains_projection_bias_std: 0.0
  - gains_projection_orthogonal: false

**Result**: ❌ NO IMPROVEMENT
- Still showing AUC ~0.69 (not 0.72)

**Conclusion**: V3+ features not the primary cause

### 7. Code Comparison: baseline vs current (Nov 19, 2025)
**Action**: Line-by-line comparison of critical sections

**Files Compared**:
- `examples/train_gainakt3exp.py`
- `pykt/models/gainakt3_exp.py`
- `configs/parameter_default.json`

**Findings**:
- ✅ Forward pass logic identical
- ✅ Loss computation identical
- ✅ Evaluation logic identical
- ✅ Sigmoid handling identical
- ❓ UNKNOWN: Something subtle changed that we cannot identify

---

## Code Changes in Problematic Commit (b59d7e41)

### Major File Changes

#### 1. examples/train_gainakt3exp.py
- **Before**: 2125 lines (comprehensive with all features)
- **After**: 745 lines (simplified, deprecated features removed)
- **Change**: -1380 lines (-65%)

**Key Deprecations**:
- Removed all constraint losses (non_negative, monotonicity, mastery_performance, etc.)
- Removed all semantic losses (alignment, retention, lag_gain)
- Removed warmup schedules for constraint losses
- Removed consistency rebalancing logic
- Removed variance floor adaptation
- Removed cosine performance schedule

#### 2. configs/parameter_default.json
- **Parameters Deprecated**: 27 parameters set to 0.0 or false
- **New Structure**: Cleaner organization, fewer legacy parameters

**Deprecated Parameters**:
```json
"enhanced_constraints": false,
"non_negative_loss_weight": 0.0,
"monotonicity_loss_weight": 0.0,
"mastery_performance_loss_weight": 0.0,
"gain_performance_loss_weight": 0.0,
"sparsity_loss_weight": 0.0,
"consistency_loss_weight": 0.0,
"enable_alignment_loss": false,
"alignment_weight": 0.0,
... (20+ more)
```

#### 3. pykt/models/gainakt3_exp.py
- **Changes**: 18 line modifications
- **Focus**: Removed deprecated parameter handling
- **Impact**: Model initialization slightly simplified

### Experiment Results Comparison

| Metric | Baseline (714616) | Current (869310) | Delta |
|--------|------------------|------------------|-------|
| Val AUC | 0.7232 | 0.6645 | -0.0587 (-8.1%) |
| BCE Loss | 0.511 | 0.553 | +0.042 (+8.2%) |
| IM Loss | 0.631 | 0.605 | -0.026 (-4.1%) |
| Enc2 AUC | 0.490 | 0.592 | +0.102 (+20.8%) |

**Key Observations**:
1. **Encoder 1**: Significant AUC drop
2. **Encoder 2**: Now learning (was random in baseline)
3. **BCE Loss**: Increased (worse)
4. **IM Loss**: Decreased (better)

**Interpretation**:
- Changes that enabled Encoder 2 to learn may have disrupted Encoder 1
- Possible trade-off introduced between the two encoders
- Some architectural coupling not visible in code review

---

## Theories About Root Cause

### Theory 1: Architectural Coupling (Most Likely)
**Evidence**:
- Encoder 2 was random (AUC=0.49) in baseline, now learning (AUC=0.59)
- Encoder 1 performance degraded simultaneously
- Both encoders should be independent but seem coupled

**Possible Mechanisms**:
- Shared batch statistics (BatchNorm effects)
- Gradient flow interaction during backward pass
- Loss landscape change affecting both paths
- DataParallel wrapper interactions

### Theory 2: Training Loop Subtle Bug (Possible)
**Evidence**:
- Code looks identical but behavior differs
- Something in the training loop flow changed

**Possible Issues**:
- Order of operations in loss computation
- Gradient accumulation differences
- Optimizer state handling
- AMP (automatic mixed precision) interactions

### Theory 3: Parameter Interaction Effects (Possible)
**Evidence**:
- 27 parameters deprecated simultaneously
- Some might have had non-zero effects despite being "deprecated"

**Risk**:
- Removing all at once might have masked which one(s) mattered
- Should have deprecated incrementally

### Theory 4: Numerical Precision Issue (Low Probability)
**Evidence**:
- Loss values slightly different
- Could indicate numerical instability

**Possible Causes**:
- Different operation ordering affecting floating point
- Batch size effects on numerical stability
- GPU kernel selection differences

---

## Failed Restoration Attempts Summary

| Attempt | Hypothesis | Action | Result | Conclusion |
|---------|-----------|--------|--------|------------|
| 1 | IM loss target | Revert to `responses` | AUC=0.665 ❌ | Not the cause |
| 2 | Double sigmoid | Code review | No issue ✅ | Not the cause |
| 3 | num_students | Validation logic | Working correctly ✅ | Not the cause |
| 4 | assist2009 params | Config check | No issue ✅ | Not the cause |
| 5 | use_gain_head | Signature check | Properly deprecated ✅ | Not the cause |
| 6 | V3+ features | Disable all | No improvement ❌ | Not primary cause |
| 7 | Code comparison | Line-by-line | Identical logic ✅ | Unknown cause |

---

## Experiments Log

### Baseline (Good Performance)
- **714616** (Nov 16, 12:51): AUC=0.7232, Commit=2acdc355 ✅

### After Cleanup (Degraded Performance)
- **281602** (Nov 16+): AUC=0.681 ❌
- **143029** (Nov 16+): AUC=0.690 ❌
- **869310** (Nov 19): AUC=0.665 ❌ (IM loss fix attempt)

### Investigation Experiments
- Multiple parameter tuning experiments (Nov 16-19)
- V3, V3+, V3++ differentiation experiments (all failed to restore baseline)
- V4, V5 semantic grounding experiments (unrelated to restoration)

---

## Data Analysis

### Dataset Statistics (assist2015, fold 0)
- Training sequences: 12,220 students, max_id=12219
- Validation sequences: 3,055 students, max_id=3054
- Skills: 100 (num_c)
- Sequence length: 200 (max)

### Parameter Values Comparison

| Parameter | Baseline (714616) | Current | Match |
|-----------|------------------|---------|-------|
| num_students | 3055 | 3055 | ✅ |
| use_student_speed | false | false | ✅ |
| use_skill_difficulty | false | false | ✅ |
| bce_loss_weight | 0.9 | 0.9 | ✅ |
| learning_rate | 0.000174 | 0.000174 | ✅ |
| batch_size | 64 | 64 | ✅ |
| d_model | 256 | 256 | ✅ |
| n_heads | 4 | 4 | ✅ |
| num_encoder_blocks | 4 | 4 | ✅ |
| mastery_threshold_init | 0.85 | 0.85 | ✅ |
| beta_skill_init | 2.5 | 2.5 | ✅ |
| m_sat_init | 0.7 | 0.7 | ✅ |

**All checked parameters match exactly** ✅

---

## Decision: Revert to Last Known Good Commit

### Rationale

1. **Exhaustive Investigation**: 7 different hypotheses tested, none successful
2. **Multiple Restoration Attempts**: 3+ experiments trying to restore AUC, all failed
3. **Time Investment**: 3+ days of investigation without progress
4. **Risk Assessment**: Continuing on broken baseline increases tech debt
5. **Clean Slate**: Starting fresh from known-good state is lower risk

### Revert Plan

**Target Commit**: 2acdc355 (Nov 16, 02:28 UTC)
- Last commit with AUC=0.72+
- Clean dual-encoder architecture
- Well-documented state

**Approach**:
1. Document all findings in this report
2. Update STATUS_gainakt3exp.md with concise summary
3. Commit current documentation state
4. Create backup branch of current HEAD
5. Hard reset to 2acdc355
6. Resume development with lessons learned

### Lessons Learned

1. **Incremental Changes**: Never deprecate 27 parameters simultaneously
2. **Continuous Testing**: Should have caught regression immediately
3. **Baseline Validation**: Always validate against known-good experiment
4. **Change Isolation**: One change at a time for critical code paths
5. **Documentation First**: This report should have been created at commit time

### Next Steps After Revert

1. **Validate Baseline**: Reproduce experiment 714616 to confirm AUC=0.72
2. **Incremental Deprecation**: Remove parameters one at a time
3. **Continuous Monitoring**: Run validation after each change
4. **Automated Testing**: Add AUC regression tests to CI/CD
5. **Parameter Tracking**: Document which deprecated parameters actually matter

---

## Technical Debt Created

1. **Unknown Root Cause**: Bug remains unidentified, could recur
2. **Lost Work**: Some good changes in b59d7e41 will need to be reapplied
3. **Investigation Time**: 3+ days spent without resolution
4. **Confidence Loss**: Uncertainty about code stability

---

## Appendices

### Appendix A: Commit Diff Statistics

```
Files changed: 43
Insertions: +9258 lines
Deletions: -2123 lines
Net change: +7135 lines
```

**Major changes**:
- examples/train_gainakt3exp.py: -1380 lines
- examples/train_gainakt3exp.py.old: +2125 lines (backup)
- configs/parameter_default.json: 303 lines modified
- paper/STATUS_gainakt3exp.md: +362 lines

### Appendix B: Key Code Sections Reviewed

1. Model forward pass (pykt/models/gainakt3_exp.py)
2. Training loop (examples/train_gainakt3exp.py)
3. Loss computation (both files)
4. Evaluation logic (examples/eval_gainakt3exp.py)
5. Parameter defaults (configs/parameter_default.json)

### Appendix C: Environment Information

- Python: 3.8
- PyTorch: 1.x (CUDA enabled)
- Hardware: 8 GPUs available, 5 used
- Container: Dev container (Ubuntu 20.04.6 LTS)

---

## Conclusion

Despite extensive investigation and multiple restoration attempts, the root cause of the Encoder 1 AUC drop remains unidentified. The decision to revert to commit 2acdc355 is based on practical considerations: we have a known-good baseline, exhaustive testing has not revealed the issue, and continuing development on a degraded baseline increases risk.

The revert will allow us to:
1. Start from a verified good state
2. Make incremental changes with proper validation
3. Apply lessons learned about change management
4. Potentially identify the issue through careful incremental deprecation

This investigation has been thoroughly documented to inform future development and serve as a reference if similar issues occur.

---

**Report Authors**: AI Development Team  
**Date**: November 19, 2025  
**Status**: INVESTIGATION COMPLETE - REVERT DECISION FINAL
