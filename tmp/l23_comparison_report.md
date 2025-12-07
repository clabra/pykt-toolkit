# Comparison: l_23 Loss Modification Impact

**Date:** December 7, 2025

## Experiments Compared

### Baseline: Experiment 290365 (with averaging in l_23)
- **Path:** `/workspaces/pykt-toolkit/experiments/20251207_015621_ikt3_irtbaseline_290365`
- **l_23 Implementation:** `MSE(mean(θ_t_learned, dim=1), θ_IRT)` - averages over sequence length
- **Problem:** Hides temporal scale mismatch by collapsing [B, L] → [B]

### New: Experiment 161656 (direct comparison in l_23)
- **Path:** `/workspaces/pykt-toolkit/experiments/20251207_201037_ikt3_l_23_161656`
- **l_23 Implementation:** `MSE(θ_t_learned, expand(θ_IRT))` - direct comparison at every timestep
- **Fix:** Penalizes scale mismatch at every timestep without temporal information loss

## Configuration
Both experiments use identical hyperparameters:
- Dataset: assist2015, fold 0, seed 42
- Architecture: d_model=256, n_heads=4, num_encoder_blocks=8, d_ff=1536
- Training: 30 epochs, batch_size=64, lr=0.0001, warmup_epochs=50
- Loss weights: lambda_target=0.5, c_stability_reg=0.01

---

## Key Findings

### 1. Prediction Performance (Minimal Change)
| Metric | Baseline (290365) | New (161656) | Δ |
|--------|-------------------|--------------|---|
| **AUC** | 0.7185 | 0.7182 | -0.0004 (-0.05%) |
| **Accuracy** | 0.7466 | 0.7473 | +0.0006 (+0.08%) |

**Conclusion:** Prediction performance essentially unchanged - the fix doesn't harm model capability.

---

### 2. Alignment Losses (Critical Improvements)

#### l_21 (Performance Alignment: σ(θ_learned - β_learned) vs σ(θ_IRT - β_IRT))
| Experiment | Value | Change |
|------------|-------|--------|
| Baseline | 9.5794 | - |
| **New** | **4.5743** | **-52.3%** ✓ |

**Analysis:** Massive improvement! The model's predictions now align much better with IRT reference predictions. This suggests the scale fix helped the model learn a more coherent internal representation.

#### l_22 (Difficulty Alignment: β_learned vs β_IRT)
| Experiment | Value | Change |
|------------|-------|--------|
| Baseline | 0.0048 | - |
| **New** | **0.0327** | **+579%** ✗ |

**Analysis:** Worsened by 6.8x. This is surprising - β alignment got worse. Possible explanation: the model is now prioritizing θ alignment (l_23) over β alignment.

#### l_23 (Ability Alignment: θ_learned vs θ_IRT)
| Experiment | Value | Change | Threshold | Pass |
|------------|-------|--------|-----------|------|
| Baseline | 0.0185 | - | 0.15 | ✓ |
| **New** | **0.2229** | **+1105%** | 0.15 | **✗** |

**Analysis:** CRITICAL FINDING - l_23 increased 12x! The baseline was artificially low because averaging hid the true scale mismatch. The new value (0.2229) reveals the actual problem: **θ_t_learned is still not properly scaled to match θ_IRT**.

---

### 3. Scale Analysis (Root Cause Revealed)

#### θ (Student Ability) Statistics
| Metric | Baseline (290365) | New (161656) | Reference (IRT) | Target |
|--------|-------------------|--------------|-----------------|--------|
| **Mean** | 28.22 | **-0.18** | ~0 | 0 |
| **Std** | 33.89 | **0.14** | ~2.5 | 2.5 |
| **Range** | [-13.37, 64.67] | **[-0.53, 0.13]** | [-5, +5] | [-5, +5] |

**Analysis:**
- **Baseline:** θ was inflated by 13.4x (std=33.89 vs 2.53 target)
- **New:** θ is now **deflated by 17.9x** (std=0.14 vs 2.5 target)!
- **Conclusion:** The model overcorrected. Previously it learned θ values that were too large; now it learns values that are too small.

#### β (Skill Difficulty) Statistics
| Metric | Baseline (290365) | New (161656) | Reference (IRT) |
|--------|-------------------|--------------|-----------------|
| Mean | - (not tracked) | -4.96 | -5.76 |
| Std | - (not tracked) | 3.78 | 3.99 |
| Range | - (not tracked) | [-13.09, 4.88] | [-13.22, 5.01] |

**Analysis:** β values are well-scaled in the new experiment (std=3.78 vs 3.99 reference).

---

### 4. Mastery Predictions

#### Mastery Statistics (σ(θ - β))
| Metric | Baseline (290365) | New (161656) |
|--------|-------------------|--------------|
| Mean | 0.683 | **0.857** |
| Std | 0.430 | **0.242** |
| Range | [2.1e-08, 1.0] | [0.0053, 1.0] |

**Analysis:** New model predicts higher mastery with less variance. This is because θ values are compressed (std=0.14), making θ-β differences smaller and pushing σ(θ-β) closer to 0.5.

#### Correlation with IRT Reference
| Metric | Baseline (290365) | New (161656) | Δ |
|--------|-------------------|--------------|---|
| **Pearson** | 0.309 | **0.071** | -77% ✗ |
| **Spearman** | 0.494 | **0.100** | -80% ✗ |

**Analysis:** MAJOR REGRESSION! Correlation with IRT reference dropped by ~80%. This confirms the scale mismatch is worse in the new experiment.

---

### 5. Training Dynamics

#### Loss Evolution (Epoch 30)
| Component | Baseline | New | Change |
|-----------|----------|-----|--------|
| Total Loss | 2.9594 | 1.3659 | -53.9% |
| l_bce | 0.0909 | 0.0929 | +2.2% |
| l_stability | 0.0382 | 0.5385 | +1310% |
| l_align_total | 9.6514 | 4.3183 | -55.3% |
| l_21 | 9.6283 | 4.1054 | -57.4% |
| l_22 | 0.0382 | 0.5385 | +1310% |
| l_23 | 0.0231 | 0.2129 | +822% |

**Key Observations:**
1. Total loss much lower in new experiment (1.37 vs 2.96) - model thinks it's doing better
2. l_21 dramatically improved (4.11 vs 9.63) - predictions align better with IRT
3. l_22 exploded (0.54 vs 0.038) - β alignment collapsed
4. l_23 increased 9x (0.21 vs 0.023) - θ scale mismatch worse

#### Lambda Warm-up
Both experiments completed 30 epochs with lambda ramping from 0.01 → 0.30 (linear warm-up over 50 epochs).

---

## Root Cause Analysis

### Why did θ scale get worse?

**Hypothesis:** The model is optimizing multiple competing objectives:

1. **l_bce (BCE Loss):** Encourages correct predictions - doesn't care about scale
2. **l_21 (Performance Alignment):** Wants σ(θ-β) ≈ σ(θ_IRT-β_IRT) - cares about **relative** scale
3. **l_22 (Difficulty Alignment):** Wants β ≈ β_IRT - locks β scale
4. **l_23 (Ability Alignment):** Wants θ ≈ θ_IRT - locks θ scale

**What happened:**
- **Baseline:** l_23 used averaging, so it was blind to per-timestep scale. Model could inflate θ and β proportionally (keeping σ(θ-β) reasonable) without penalty.
- **New:** l_23 now sees per-timestep scale mismatch. To minimize l_23, model **deflated θ** (std=0.14 vs target 2.5).

**Why deflation instead of correct scale?**
The model found a local minimum where:
- Small θ values → small l_23 MSE loss
- β remains well-scaled (l_22 keeps it anchored)
- σ(θ-β) still makes predictions (l_bce satisfied)
- l_21 improved because predictions are more consistent

---

## Conclusions

### What Worked
✓ **l_23 fix exposed the real problem:** The baseline's low l_23 (0.0185) was hiding scale mismatch
✓ **l_21 improved dramatically:** Model predictions now align better with IRT reference (-52%)
✓ **Prediction performance maintained:** AUC/Accuracy unchanged
✓ **β scale is good:** Difficulty parameters are properly scaled

### What Failed
✗ **θ scale is worse:** Model deflated θ by 17.9x (std=0.14 vs target 2.5)
✗ **Correlation collapsed:** Pearson correlation dropped from 0.31 → 0.07 (-77%)
✗ **l_23 exploded:** Increased from 0.0185 → 0.2229 (+1105%)
✗ **l_22 worsened:** β alignment degraded 6.8x (though absolute values still reasonable)

---

## Recommendations

### 1. Add Scale Regularization to θ
The model needs explicit guidance to learn θ with correct scale:

```python
# Target: θ_std ≈ 2.5 (matching IRT reference)
l_theta_scale = (theta_std - target_std)**2
loss_total += lambda_scale * l_theta_scale
```

### 2. Normalize l_23 by Sequence Length
Current l_23 sums errors across L timesteps, making long sequences dominate:

```python
# Before: MSE over [B, L] → penalizes long sequences more
l_23 = F.mse_loss(theta_t_learned, theta_irt_expanded)

# After: Mean over L, then MSE over B
l_23 = F.mse_loss(theta_t_learned.mean(dim=1), theta_irt)
```
Wait - this is what we removed! The issue is subtle: we need **per-timestep** penalty but **normalized magnitude**.

### 3. Use Standardized Loss (Better Solution)
Instead of raw MSE, use standardized differences:

```python
# Compute z-scores relative to IRT distribution
theta_z = (theta_t_learned - theta_irt_mean) / theta_irt_std
target_z = (theta_irt_expanded - theta_irt_mean) / theta_irt_std
l_23 = F.mse_loss(theta_z, target_z)
```

### 4. Dynamic IRT Targets (Already Implemented!)
The script `compute_irt_dynamic_targets.py` is running to generate time-varying θ_IRT[t]. This will:
- Provide natural [B, L] shape match (no expansion needed)
- Show realistic θ trajectories (increasing with practice)
- Give the model better supervision signal

### 5. Increase λ Weight for l_23
Current lambda_target=0.5 might be too low. Try:
- lambda_target=0.7 or 0.8
- Adjust warmup schedule to reach target sooner

---

## Next Steps

1. **Wait for dynamic IRT generation to complete** - this may solve the scale issue naturally
2. **Test with standardized l_23 loss** - prevents scale collapse
3. **Add explicit θ_std regularization** - enforce correct variance
4. **Analyze gradient magnitudes** - check if l_23 gradients are too small/large
5. **Try different loss balancing** - increase lambda_target or add separate scale loss

---

## Technical Details

### Scale Comparison Table
| Parameter | Baseline (290365) | New (161656) | IRT Reference | Ratio (New/Ref) |
|-----------|-------------------|--------------|---------------|-----------------|
| θ_std | 33.89 | **0.14** | 2.53 | **0.055** (17.9x too small) |
| β_std | - | 3.78 | 3.99 | 0.95 (good) |
| θ_mean | 28.22 | -0.18 | ~0 | good |
| β_mean | - | -4.96 | -5.76 | good (86% of target) |

### Loss Weight Evolution
Both experiments use identical warm-up:
- Epochs 1-50: lambda = epoch / 50  (linear 0→1)
- Epoch 30: lambda = 0.30
- Final: lambda_target = 0.5

### Correlation Diagnostic
| Experiment | Pearson (θ) | Spearman (θ) | Status |
|------------|-------------|--------------|--------|
| Baseline | 0.309 | 0.494 | Poor |
| New | 0.071 | 0.100 | **Critical** |
| Target | >0.85 | >0.85 | Required |

---

## Summary
The l_23 fix successfully **exposed** the scale problem (hidden by averaging) and **improved l_21** (performance alignment), but the model responded by **collapsing θ scale** instead of learning correct values. The fix revealed we need additional constraints to prevent scale collapse. Dynamic IRT targets + scale regularization are the path forward.
