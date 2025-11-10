# Gains Ablation Study: Testing Loss Function Necessity

**Date:** November 10, 2025  
**Context:** GainAKT2Exp Architecture - Recursive Mastery Accumulation  
**Branch:** v0.0.18-gainakt2exp-gainsablation

## Executive Summary

This document details an ablation study to test whether the consistency loss and gain-performance loss are necessary given the architectural constraint that mastery evolves as a deterministic accumulation of learning gains. Empirical evidence from Experiment 1 shows that gain correlation improved 15.2% (0.0409 → 0.0471) when only mastery supervision was strengthened (mastery_weight: 0.8 → 1.5), while gain_weight remained unchanged at 0.8. This suggests the recursive accumulation constraint enables gradient flow from mastery supervision to gain learning, potentially making explicit gain supervision redundant.

## Theoretical Foundation

### 1. Recursive Accumulation Constraint

The GainAKT2Exp architecture enforces a deterministic relationship between mastery and gains in the forward pass:

```python
# Lines 145, 162 in pykt/models/gainakt2_exp.py
accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
```

**Mathematical formulation:**
$$\text{mastery}_{t+1} = \text{mastery}_t + \alpha \cdot \text{ReLU}(\text{gain}_t), \quad \alpha = 0.1$$

**Key property:** Each mastery state is the cumulative sum of all prior learning gains:
$$\text{mastery}_t^{(c)} = \text{mastery}_0^{(c)} + \alpha \sum_{i=1}^{t} \text{ReLU}(\text{gain}_i^{(c)})$$

### 2. Gradient Flow Analysis

Since the recursive accumulation is a differentiable operation, gradients from mastery supervision should propagate backward to gain predictions through the chain rule:

$$\frac{\partial \mathcal{L}_{\text{mastery-perf}}}{\partial \text{gain}_t} = \frac{\partial \mathcal{L}_{\text{mastery-perf}}}{\partial \text{mastery}_{t+k}} \cdot \frac{\partial \text{mastery}_{t+k}}{\partial \text{gain}_t} = \frac{\partial \mathcal{L}_{\text{mastery-perf}}}{\partial \text{mastery}_{t+k}} \cdot \alpha$$

**Implication:** Strong mastery supervision should implicitly guide gain learning through the architectural constraint, potentially making explicit gain supervision redundant.

### 3. Empirical Evidence for Gradient Spillover

**Experiment 1 Results:**

| Metric | Baseline (λ_mastery=0.8) | Treatment (λ_mastery=1.5) | Improvement | Gain Weight Changed? |
|--------|---------------------------|---------------------------|-------------|----------------------|
| Test Mastery Corr | 0.0985 | 0.1069 | **+8.5%** | ❌ No |
| Test Gain Corr | 0.0409 | 0.0471 | **+15.2%** | ❌ No |
| gain_performance_loss_weight | 0.8 | 0.8 | **0%** | ❌ No |

**Critical observation:** Gain correlation improved 15.2% despite:
- gain_performance_loss_weight remaining at 0.8 (unchanged)
- No architectural changes to gain prediction mechanism
- Only mastery supervision was strengthened

**Interpretation:** The improvement in gain correlation must have come from increased gradient flow through the recursive accumulation constraint, validating the theoretical analysis.

## Current Loss Function Analysis

### Loss 1: Consistency Loss (Weight: 0.3)

**Purpose:** Penalizes deviation between mastery deltas and scaled gains

**Implementation (lines 303-308 in gainakt2_exp.py):**
```python
mastery_delta = projected_mastery[:, 1:, :] - projected_mastery[:, :-1, :]
scaled_gains = projected_gains[:, 1:, :] * 0.1
consistency_residual = torch.abs(mastery_delta - scaled_gains)
consistency_loss = consistency_residual.mean()
total_loss += self.consistency_loss_weight * consistency_loss
```

**Analysis:**
- **What it checks:** |mastery_{t+1} - mastery_t - α·gain_t|
- **Redundancy issue:** Forward pass already enforces mastery_{t+1} = mastery_t + α·gain_t deterministically
- **Problem:** Creates "trust but verify" mechanism for architectural constraint
- **Potential harm:** May create conflicting gradients or unnecessary computational overhead

**Verdict:** **LIKELY REDUNDANT** - penalizes deviation from relationship already enforced in forward pass

### Loss 2: Gain-Performance Loss (Weight: 0.8)

**Purpose:** Forces correct responses to have higher learning gains than incorrect responses

**Implementation (lines 285-293 in gainakt2_exp.py):**
```python
correct_gains = relevant_gains[(responses == 1).flatten()]
incorrect_gains = relevant_gains[(responses == 0).flatten()]
gain_performance_loss = torch.clamp(incorrect_gains.mean() - correct_gains.mean() + 0.1, min=0)
total_loss += self.gain_performance_loss_weight * gain_performance_loss
```

**Analysis:**
- **What it enforces:** correct_gains > incorrect_gains (hinge loss with 0.1 margin)
- **Rationale:** Learning gains should be higher when students answer correctly
- **Spillover evidence:** 15.2% gain improvement without changing this weight
- **Counterargument:** Gains capture temporal dynamics (learning rate), mastery captures state (cumulative) - orthogonal signals?

**Verdict:** **NECESSITY UNCLEAR** - empirical evidence suggests mastery supervision might be sufficient, but gains represent temporal information that may benefit from direct supervision

## Research Questions

**RQ1:** Is consistency loss necessary, or does the architectural constraint make it redundant?

**RQ2:** Is gain-performance loss necessary, or does mastery supervision provide sufficient gradient signal through recursive accumulation?

**RQ3:** If both losses are removed, does the model maintain interpretability quality (mastery-performance correlation) and predictive performance (AUC)?

## Ablation Study Design

### Experimental Configurations

We test three configurations to isolate the contribution of each loss function:

| Config | consistency_weight | gain_perf_weight | mastery_weight | Purpose |
|--------|-------------------|------------------|----------------|---------|
| **A: Current** | 0.3 | 0.8 | 1.5 | Baseline (full supervision) |
| **B: Simplified** | 0.0 | 0.0 | 1.5 | Test hypothesis (mastery only) |
| **C: Hybrid** | 0.0 | 0.8 | 1.5 | Conservative (remove consistency only) |

### Hypotheses

**H1 (Consistency Loss Redundant):**
- **Prediction:** Config C (no consistency) ≈ Config A (current)
- **Rationale:** Architectural constraint already enforces relationship
- **Test:** Compare mastery_corr, gain_corr, AUC between A and C
- **Success criterion:** |Δ AUC| < 0.005, |Δ mastery_corr| < 3%

**H2 (Gain Loss Redundant):**
- **Prediction:** Config B (simplified) ≈ Config A (current)
- **Rationale:** Mastery supervision provides sufficient gradient flow to gains
- **Test:** Compare all metrics between A and B
- **Success criterion:** |Δ AUC| < 0.005, |Δ mastery_corr| < 3%, |Δ gain_corr| < 10%

**H3 (Both Losses Redundant):**
- **Prediction:** Config B (simplified) ≈ Config C (hybrid) ≈ Config A (current)
- **Rationale:** Complete architecture simplification without performance loss
- **Test:** Three-way comparison across all metrics
- **Success criterion:** All pairwise differences within tolerance

### Expected Outcomes and Interpretations

#### Scenario 1: Complete Validation (Both Losses Redundant)
**Result:** B ≈ C ≈ A (all metrics within tolerance)

**Interpretation:**
- Consistency loss: Redundant (confirmed)
- Gain-performance loss: Redundant (confirmed)
- **Action:** Simplify model from 5 interpretability losses to 3
- **Benefit:** Fewer hyperparameters, clearer optimization landscape, faster training

**Model simplification:**
- Remove consistency_loss_weight parameter
- Remove gain_performance_loss_weight parameter
- Update documentation explaining architectural reasoning

#### Scenario 2: Partial Validation (Only Consistency Redundant)
**Result:** C ≈ A (hybrid matches current), but B < A (simplified degrades)

**Interpretation:**
- Consistency loss: Redundant (confirmed)
- Gain-performance loss: Necessary (disconfirmed hypothesis)
- **Action:** Remove consistency loss only, keep gain-performance loss
- **Benefit:** Partial simplification (5 losses → 4)

**Reasoning for gain loss necessity:**
- Gains capture temporal dynamics (learning rate) that are orthogonal to cumulative mastery
- Direct supervision on temporal patterns may be needed despite gradient spillover
- Hinge loss margin (0.1) may enforce distributional properties not captured by accumulation

#### Scenario 3: Both Losses Necessary (Hypothesis Rejected)
**Result:** B << A and C < A (both simplified configs degrade significantly)

**Interpretation:**
- Consistency loss: Necessary (hypothesis rejected)
- Gain-performance loss: Necessary (hypothesis rejected)
- **Action:** Keep current architecture unchanged
- **Insight:** Redundancy only apparent, explicit supervision critical despite architectural constraints

**Possible explanations:**
- Consistency loss may provide regularization beyond constraint enforcement
- Gain-performance loss may handle edge cases not covered by mastery supervision
- Combined supervision may create beneficial gradient dynamics beyond simple chain rule

#### Scenario 4: Mixed Results (Metric-Dependent Trade-offs)
**Result:** Some metrics improve in simplified configs, others degrade

**Example:** B has higher mastery_corr but lower AUC than A

**Interpretation:**
- Trade-offs exist between different objectives
- May need to tune λ_mastery to compensate for removed losses
- **Action:** Further experimentation with mastery_weight values in simplified architecture

## Detailed Execution Steps

### Step 1: Environment Verification

**Commands:**
```bash
# Activate virtual environment
source /home/vscode/.pykt-env/bin/activate

# Verify we're on ablation branch
git branch --show-current

# Check GPU availability
nvidia-smi
```

**Expected:** Clean environment with 8 GPUs available, on branch v0.0.18-gainakt2exp-gainsablation

### Step 2: Baseline Configuration (Config A: Current)

**Purpose:** Re-establish baseline with all losses enabled

**Command:**
```bash
cd /workspaces/pykt-toolkit

python3 examples/train_gainakt2exp.py \
  --dataset_name assist2015 \
  --fold 0 \
  --seed 42 \
  --epochs 12 \
  --mastery_performance_loss_weight 1.5 \
  --gain_performance_loss_weight 0.8 \
  --consistency_loss_weight 0.3 \
  --monotonicity_loss_weight 0.1 \
  --sparsity_loss_weight 0.2 \
  --non_negativity_loss_weight 0.0 \
  --short_title ablation_current \
  --use_wandb 0 \
  --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
```

**Key parameters:**
- mastery_weight: 1.5 (proven optimal from Experiment 1)
- gain_weight: 0.8 (current baseline)
- consistency_weight: 0.3 (current baseline)
- All other losses at standard values

**Expected runtime:** ~30 minutes

**Output location:** `examples/experiments/YYYYMMDD_HHMMSS_gainakt2exp_ablation_current_EXPID/`

### Step 3: Simplified Configuration (Config B: Mastery Only)

**Purpose:** Test hypothesis that mastery supervision alone is sufficient

**Command:**
```bash
python3 examples/train_gainakt2exp.py \
  --dataset_name assist2015 \
  --fold 0 \
  --seed 42 \
  --epochs 12 \
  --mastery_performance_loss_weight 1.5 \
  --gain_performance_loss_weight 0.0 \
  --consistency_loss_weight 0.0 \
  --monotonicity_loss_weight 0.1 \
  --sparsity_loss_weight 0.2 \
  --non_negativity_loss_weight 0.0 \
  --short_title ablation_simplified \
  --use_wandb 0 \
  --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
```

**Key changes:**
- gain_performance_loss_weight: 0.8 → **0.0** (removed)
- consistency_loss_weight: 0.3 → **0.0** (removed)
- mastery_weight: 1.5 (unchanged, provides gradient flow)

**Critical test:** Does mastery supervision at λ=1.5 provide sufficient gradient signal to learn gains via recursive accumulation?

**Expected runtime:** ~30 minutes

**Output location:** `examples/experiments/YYYYMMDD_HHMMSS_gainakt2exp_ablation_simplified_EXPID/`

### Step 4: Hybrid Configuration (Config C: Conservative)

**Purpose:** Isolate consistency loss contribution while keeping gain supervision

**Command:**
```bash
python3 examples/train_gainakt2exp.py \
  --dataset_name assist2015 \
  --fold 0 \
  --seed 42 \
  --epochs 12 \
  --mastery_performance_loss_weight 1.5 \
  --gain_performance_loss_weight 0.8 \
  --consistency_loss_weight 0.0 \
  --monotonicity_loss_weight 0.1 \
  --sparsity_loss_weight 0.2 \
  --non_negativity_loss_weight 0.0 \
  --short_title ablation_hybrid \
  --use_wandb 0 \
  --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
```

**Key changes:**
- consistency_loss_weight: 0.3 → **0.0** (removed)
- gain_performance_loss_weight: 0.8 (kept)
- mastery_weight: 1.5 (unchanged)

**Purpose:** If C ≈ A but B < A, this isolates gain loss as necessary while confirming consistency loss is redundant

**Expected runtime:** ~30 minutes

**Output location:** `examples/experiments/YYYYMMDD_HHMMSS_gainakt2exp_ablation_hybrid_EXPID/`

### Step 5: Results Collection

**For each experiment, extract:**

1. **Test metrics from eval_results.json:**
   - test_auc
   - test_acc
   - test_mastery_correlation
   - test_gain_correlation
   - test_correlation_students (verify n=3177)

2. **Training dynamics from repro_results.json:**
   - Epoch-by-epoch mastery_correlation
   - Epoch-by-epoch gain_correlation
   - Training stability (variance across epochs)

3. **Constraint violations from repro_results.json:**
   - monotonicity_violation_rate
   - performance_alignment_issues
   - non_negativity_violation_rate

**Extraction commands:**
```bash
# Config A: Current
cat examples/experiments/EXPDIR_A/eval_results.json | python3 -m json.tool | grep -E "test_auc|test_acc|test_mastery_correlation|test_gain_correlation"

# Config B: Simplified
cat examples/experiments/EXPDIR_B/eval_results.json | python3 -m json.tool | grep -E "test_auc|test_acc|test_mastery_correlation|test_gain_correlation"

# Config C: Hybrid
cat examples/experiments/EXPDIR_C/eval_results.json | python3 -m json.tool | grep -E "test_auc|test_acc|test_mastery_correlation|test_gain_correlation"
```

### Step 6: Statistical Analysis

**Comparison table template:**

| Metric | A: Current | B: Simplified | C: Hybrid | B vs A (Δ) | C vs A (Δ) | Status |
|--------|-----------|--------------|-----------|------------|------------|--------|
| test_auc | ? | ? | ? | ? | ? | ? |
| test_acc | ? | ? | ? | ? | ? | ? |
| test_mastery_corr | ? | ? | ? | ? | ? | ? |
| test_gain_corr | ? | ? | ? | ? | ? | ? |
| n_students | 3177 | 3177 | 3177 | Equal | Equal | ✅ |

**Statistical tests:**

1. **AUC comparison:** DeLong's test (if available) or simple difference
   - Threshold: |Δ AUC| < 0.005 indicates stable performance
   - Decision: If B or C within threshold → corresponding hypothesis supported

2. **Correlation comparison:** Fisher's z-transformation
   - For each config pair (B vs A, C vs A)
   - SE = √(2/(n-3)) = √(2/3174) = 0.0251
   - Threshold: |Δ mastery_corr| < 3% and |Δ gain_corr| < 10%

3. **Effect size interpretation:**
   - Relative change: (r₂ - r₁)/r₁ × 100%
   - Practical significance if |Δ| > 5% even if p > 0.05

### Step 7: Decision Matrix

**Based on results, apply this decision tree:**

```
IF (B ≈ A) AND (C ≈ A):
    → Both losses redundant
    → Remove consistency_weight and gain_weight from model
    → Update documentation
    → Simplify from 5 to 3 interpretability losses
    
ELIF (C ≈ A) AND (B < A):
    → Consistency loss redundant, gain loss necessary
    → Remove consistency_weight only
    → Keep gain_weight
    → Simplify from 5 to 4 interpretability losses
    
ELIF (B ≈ A) AND (C < A):
    → Gain loss redundant, consistency loss necessary (unexpected!)
    → Investigate why consistency loss matters despite architectural constraint
    → Keep consistency_weight, consider removing gain_weight
    
ELSE:
    → Both losses necessary
    → Keep current architecture
    → Document that redundancy is only apparent
    → Explain why both supervision signals are needed
```

## Metrics and Success Criteria

### Primary Metrics (Must satisfy for hypothesis support)

1. **Predictive Performance:**
   - test_auc: |Δ| < 0.005 (0.5%)
   - test_acc: |Δ| < 0.005 (0.5%)

2. **Interpretability Quality:**
   - test_mastery_correlation: |Δ| < 3%
   - test_gain_correlation: |Δ| < 10% (more lenient due to weaker baseline signal)

3. **Sample Size Validity:**
   - All configs must use n=3177 students (equal to Experiment 1)

### Secondary Metrics (Informative but not decisive)

1. **Training Stability:**
   - Variance of mastery_correlation across epochs
   - Variance of gain_correlation across epochs
   - If simplified configs show higher variance → suggests losses provide regularization

2. **Constraint Violations:**
   - monotonicity_violation_rate (should remain 0%)
   - performance_alignment (should remain healthy)
   - non_negativity_violation_rate (should remain 0%)

3. **Training Efficiency:**
   - Time per epoch
   - If simplified configs train faster → additional benefit

### Tolerance Thresholds

**Strict (Primary objectives):**
- AUC: ±0.005 (±0.5%)
- Accuracy: ±0.005 (±0.5%)
- Mastery correlation: ±3%

**Lenient (Secondary objectives):**
- Gain correlation: ±10% (baseline signal is weak ~0.04-0.06)
- Constraint violations: Must remain 0% (non-negotiable)

**Rationale for thresholds:**
- AUC variations <0.5% are typical noise in knowledge tracing evaluations
- Mastery correlation 3% threshold based on Experiment 1 improvement magnitude (8.5%)
- Gain correlation 10% threshold acknowledges weaker baseline signal strength

## Expected Timeline

- **Config A (Current):** 30 minutes training + 5 minutes evaluation = 35 min
- **Config B (Simplified):** 30 minutes training + 5 minutes evaluation = 35 min
- **Config C (Hybrid):** 30 minutes training + 5 minutes evaluation = 35 min
- **Analysis:** 15 minutes (extraction, statistical tests, comparison tables)

**Total estimated time:** ~2 hours

**Parallelization opportunity:** If we have 8 GPUs available, could run 2-3 experiments in parallel, reducing total time to ~45 minutes + analysis

## Documentation Updates (Post-Ablation)

### If Hypothesis Confirmed (Losses Redundant):

1. **Update paper/draft.md:**
   - Add subsection after Experiment 1: "### Ablation Study: Loss Function Necessity"
   - Include results table, statistical analysis, interpretation
   - Explain gradient flow theory validated by empirical results

2. **Update paper/STATUS_gainakt2exp.md:**
   - Add note to architecture diagram explaining simplified loss structure
   - Document removed losses and theoretical justification

3. **Update pykt/models/gainakt2_exp.py:**
   - Add docstring explaining why consistency and/or gain losses were removed
   - Reference ablation study results
   - Preserve backward compatibility (allow weights=0 but remove from default config)

4. **Update configs/parameter_default.json:**
   - Remove redundant loss weight parameters
   - Apply Parameter Evolution Protocol (update MD5, increment version)
   - Document parameter removal with ablation study reference

5. **Create paper/ablation_study_results.md:**
   - Comprehensive report with all metrics
   - Statistical analysis details
   - Theoretical explanation of findings
   - Implications for future architecture design

### If Hypothesis Rejected (Losses Necessary):

1. **Update paper/draft.md:**
   - Add subsection documenting ablation attempt
   - Explain why explicit supervision is necessary despite architectural constraints
   - Discuss potential reasons (regularization, edge cases, gradient dynamics)

2. **Create tmp/ablation_negative_results.md:**
   - Document unexpected findings
   - Analyze why gradient spillover alone is insufficient
   - Hypothesize about interaction effects between loss functions

## Risk Mitigation

### Risk 1: Experiments Fail to Complete
**Mitigation:** Use `--epochs 12` (same as Experiment 1) to ensure reasonable runtime. Monitor GPU memory usage.

### Risk 2: Results are Ambiguous (Near Threshold)
**Mitigation:** If differences are within 1-2% of threshold, run additional replicates with different seeds (43, 44) to assess variance.

### Risk 3: Unexpected Degradation in Simplified Config
**Mitigation:** This is informative! Document why losses are needed despite theoretical redundancy. Investigate:
- Are gains learning at all in Config B? (check training dynamics)
- Are constraint violations occurring? (check violation rates)
- Is training unstable? (check loss curves, correlation variance)

## Success Indicators

**Clear Success (Hypothesis Validated):**
- B ≈ A: test_auc within 0.003, mastery_corr within 2%, gain_corr within 8%
- C ≈ A: test_auc within 0.003, mastery_corr within 2%
- No constraint violations in any config

**Moderate Success (Partial Validation):**
- C ≈ A but B < A: Consistency loss confirmed redundant, gain loss needed
- Still valuable simplification (5 → 4 losses)

**Informative Failure (Hypothesis Rejected):**
- B << A and C < A: Both losses necessary despite architectural redundancy
- Gain insight into why explicit supervision needed
- Document unexpected interaction effects

**Any outcome is valuable** for understanding the architecture and publishing rigorous results.

## References

- **Experiment 1:** paper/draft.md, section "### Experiment 1 Results"
- **Architecture:** paper/STATUS_gainakt2exp.md, Mermaid diagram
- **Model code:** pykt/models/gainakt2_exp.py, lines 145, 162 (accumulation), 285-293 (gain loss), 303-308 (consistency loss)
- **Reproducibility:** examples/reproducibility.md, Parameter Evolution Protocol
- **Statistical methods:** paper/draft.md, appendix "## Statistical Tests"

---

**Document Status:** Ready for execution  
**Next Action:** Execute Step 1 (Environment Verification) followed by Steps 2-4 (training configurations)
