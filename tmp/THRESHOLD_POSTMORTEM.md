# Learnable Threshold Analysis: Post-Mortem

**Date:** November 10, 2025  
**Experiment:** threshold_v3_seed42 (stopped at epoch 8)  
**Status:** ❌ **FAILED - Approach Abandoned**

---

## Executive Summary

The learnable threshold mechanism was implemented to address the "saturation effect" - the hypothesis that a threshold exists above which skills are considered "learned". After full implementation and experimental validation, **the approach failed to improve mastery-performance correlations**.

**Key Finding:** The root cause is NOT the absence of a threshold, but rather that **mastery values from the projection head don't correlate with actual student performance**. Adding a threshold on top of meaningless mastery values cannot fix this fundamental issue.

---

## Implementation Summary

### What Was Implemented

✅ **Complete threshold architecture:**
1. **Learnable threshold parameter** (`threshold_raw`): nn.Parameter initialized at 0.0 (sigmoid → 0.5)
2. **Soft thresholding**: `sigmoid((mastery - threshold) / temperature)` for differentiability
3. **Q-matrix logic**: `compute_skill_readiness()` with conjunctive AND for multi-skill questions
4. **Prediction integration**: Modified prediction_head to receive skill_readiness (+1 dimension: 1536→1537)
5. **Threshold consistency loss**: BCE(skill_readiness, responses) with weight=0.5
6. **Configuration**: Added 3 parameters to parameter_default.json with correct MD5

### Bugs Fixed

1. ❌ **Bug 1:** Training v1 used OLD code before threshold implementation
2. ❌ **Bug 2:** `create_exp_model()` missing threshold parameters in constructor
3. ✅ **All bugs fixed:** v3 training confirmed threshold_raw in checkpoint

---

## Experimental Results

### Training Configuration
- **Model:** GainAKT2Exp with causal mastery + learnable alpha + learnable threshold
- **Dataset:** ASSIST2015, fold 0
- **Seed:** 42
- **Epochs:** 8 (stopped early due to poor results)
- **Threshold parameters:**
  - `use_learnable_threshold`: True
  - `threshold_temperature`: 0.1
  - `threshold_loss_weight`: 0.5

### Results

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| **Threshold value** | 0.3-0.7 | **0.5248** | ⚠️ Barely moved |
| **Mastery correlation** | > 0.5 | **0.0195** | ❌ Critically low |
| **Gain correlation** | > 0.3 | **0.0355** | ❌ Very low |
| **Valid AUC** | ~0.72 | **0.7252** | ✅ Maintained |

### Correlation Evolution

```
Epoch | Mastery Corr | Gain Corr | Threshold
------|--------------|-----------|----------
  1   |    0.0192    |  0.1143   |  0.5083
  2   |    0.0296    |  0.0613   |  0.5248
  3   |    0.0324    |  0.0450   |  0.5248
  4   |    0.0275    |  0.0330   |   -
  5   |    0.0261    |  0.0397   |   -
  6   |    0.0233    |  0.0297   |   -
  7   |    0.0206    |  0.0233   |   -
  8   |    0.0195    |  0.0355   |   -
```

**Observation:** Correlations remain near-zero throughout training, showing no improvement trend.

---

## Diagnostic Analysis

### Verified Implementation Details

✅ **Threshold parameter exists:** `module.threshold_raw` present in checkpoint  
✅ **Prediction head modified:** Input dimension 1537 (1536 + 1 for skill_readiness)  
✅ **Forward pass working:** `skill_readiness` successfully computed and returned  
✅ **Methods implemented:** `apply_soft_threshold()` and `compute_skill_readiness()` present  
✅ **Threshold loss computed:** BCE loss included in interpretability_loss  

### Root Cause Identification

**The fundamental problem is NOT the absence of a threshold. The problem is:**

1. **Mastery values don't reflect actual skill mastery**
   - Correlation with performance: **0.0195** (essentially random)
   - Expected: > 0.5 for meaningful mastery representation
   - Even without threshold, mastery should predict performance

2. **Mastery-performance loss is insufficient**
   - Current weight: 0.8
   - Loss encourages correlation but doesn't enforce it strongly enough
   - Other losses (BCE, consistency, sparsity, etc.) dominate optimization

3. **Threshold cannot fix bad inputs**
   - Threshold operates on mastery values: `threshold(mastery) → skill_readiness`
   - If mastery is garbage (corr=0.02), threshold output is also garbage
   - **Garbage in, garbage out**

### Why Threshold Learned Nothing

The threshold barely moved from initialization (0.5 → 0.52) because:

1. **No gradient signal:** If mastery doesn't predict performance, threshold has no useful gradient
2. **Weak threshold loss:** Weight=0.5 too small compared to main BCE loss
3. **Conflicting objectives:** Model can predict well (AUC=0.725) by ignoring mastery entirely

---

## Why This Approach Failed

### Conceptual Flaw

The threshold hypothesis was:
> "There exists a threshold θ such that mastery > θ means skill is learned, and learned skills predict correct responses."

**Reality:** This assumes mastery values are meaningful to begin with. They're not.

### Architecture Issues

1. **Projection heads are "post-hoc"**
   - Mastery/gain projected from encoder hidden states
   - Not intrinsically part of knowledge representation
   - Model can ignore them for main task (prediction)

2. **Weak supervision**
   - Auxiliary losses (mastery-performance, gain-performance) too weak
   - Main BCE loss dominates, model learns shortcut: ignore mastery, predict from context

3. **No forced dependency**
   - Prediction head receives: `[context, value, skill_emb, skill_readiness]`
   - Model can learn to ignore skill_readiness dimension entirely
   - No architectural constraint forces using mastery information

---

## Lessons Learned

### What We Learned

1. **Post-hoc interpretability is hard**
   - Adding projection heads after encoder doesn't guarantee meaningful representations
   - Auxiliary losses alone are insufficient for supervision

2. **Threshold is not the solution**
   - The saturation effect observed in analysis (optimal threshold=0.1) was a symptom
   - Root cause: mastery values themselves are poorly calibrated

3. **Correlation metrics reveal ground truth**
   - Mastery-performance correlation is the key diagnostic
   - If corr < 0.1, interpretability mechanisms will fail

### What NOT to Do

❌ Don't add complexity (threshold) to fix symptoms  
❌ Don't assume auxiliary losses alone provide strong enough supervision  
❌ Don't expect post-hoc projections to magically become interpretable  
❌ Don't ignore low correlations hoping they'll improve later  

---

## Recommendations

### Short Term: Abandon Threshold Approach

**Do NOT continue with learnable threshold** because:
- It doesn't address the root cause
- Adds complexity without benefit
- Wastes compute resources on failed experiments

### Medium Term: Fix Mastery Projection

**Focus on making mastery values meaningful:**

1. **Stronger mastery-performance loss:**
   - Increase weight from 0.8 to 2.0 or higher
   - Use ranking loss instead of BCE to enforce ordering
   - Add explicit correlation loss: `-corr(mastery, performance)`

2. **Architectural changes:**
   - Force prediction to depend on mastery (e.g., gating mechanism)
   - Use mastery as attention weights (intrinsic, not post-hoc)
   - Pre-train mastery head with supervised signal

3. **Alternative objectives:**
   - Predict NEXT response from CURRENT mastery (causal prediction)
   - Use contrastive learning: similar mastery → similar performance
   - Multi-task: predict both response AND mastery change

### Long Term: Intrinsic Interpretability

**Consider fundamental architecture redesign:**

1. **Attention-as-mastery:** Use attention weights directly as mastery indicators
2. **Structured latent space:** Force hidden states to have interpretable structure
3. **Causal models:** Use structural causal models (SCM) for provable interpretability
4. **Hybrid approaches:** Combine neural nets with symbolic knowledge structures

---

## Files and Artifacts

### Code
- `pykt/models/gainakt2_exp.py`: Lines 97-130 (threshold initialization), 260-310 (threshold methods), 450-475 (forward integration), 630-650 (threshold loss)
- `examples/train_gainakt2exp.py`: Lines 494-498 (threshold config), 1661-1666 (argparse)
- `examples/eval_gainakt2exp.py`: Lines 30-70 (threshold evaluation)
- `configs/parameter_default.json`: Lines 32-34, 115-117 (threshold defaults)

### Experiments
- `examples/experiments/20251110_041636_gainakt2exp_threshold_v3_seed42_370193/`
  - `model_best.pth`: Checkpoint with threshold (epoch 3, AUC=0.7252)
  - `metrics_epoch.csv`: 8 epochs of training metrics
  - `config.json`: Full configuration

### Documentation
- `tmp/LEARNABLE_THRESHOLD_IMPLEMENTATION.md`: 600+ line technical spec
- `tmp/LEARNABLE_THRESHOLD_QUICKSTART.md`: Quick-start guide
- `tmp/threshold_analysis/`: Analysis showing lack of natural threshold
- `tmp/quick_threshold_diagnostic.py`: Diagnostic script
- `paper/STATUS_gainakt2exp.md`: Updated architecture diagram (purple components)

### Commits
- `a11fe6a`: feat: Add learnable threshold architecture (Feature 8)
- `b4d3434`: fix: Add missing threshold parameters to create_exp_model factory

---

## Conclusion

The learnable threshold approach **failed** because it attempted to solve the wrong problem. The issue is not that we lack a threshold mechanism, but that **mastery values from the projection head are not meaningful indicators of skill mastery**.

**Moving forward:** Focus efforts on fixing the root cause (poor mastery quality) rather than adding Band-Aid solutions like thresholds. This requires stronger supervision, architectural changes, or a fundamental rethinking of how we achieve interpretability in knowledge tracing models.

---

**Status:** ABANDONED  
**Next Steps:** Document findings in STATUS_gainakt2exp.md, move forward with alternative approaches
