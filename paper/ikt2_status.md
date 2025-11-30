# iKT2 Model Status Report

**Date:** November 30, 2025  
**Model:** iKT2 (Interpretable Knowledge Tracing with Dual Heads)  
**Status:** Under Development - Critical Issues Identified

## Executive Summary

The iKT2 model shows promising predictive performance (AUC ≈ 0.714) but has encountered **critical interpretability degradation** in recent experiments. Only one experiment (947580) has achieved the target interpretability level (I = 0.766, "GOOD" rating). All subsequent attempts to reproduce these results have failed, with interpretability scores dropping 19-26% below baseline.

**Current Status:** 🔴 **NOT READY FOR PUBLICATION** - Reproducibility crisis identified.

---

## Timeline of Development

### November 29, 2025 - Baseline Establishment

**Experiment 581572** (baseline, λ_reg=0.1)
- First successful training run
- AUC: 0.7138, Interpretability: 0.694 (MODERATE)
- Established initial proof of concept

**Experiment 947580** (lambdareg-001, λ_reg=0.01) ⭐ **BEST RESULT**
- Reduced regularization to λ_reg=0.01
- **AUC: 0.7135, Interpretability: 0.766 (GOOD)**
- IRT Fidelity: 0.948, BKT: 0.506, Head Agreement: 0.844
- **Target baseline for reproducibility**

### November 29, 2025 - Recovery Attempts

**Experiment 346878** (lambdareg001-recovered, λ_reg=0.01)
- Attempted to reproduce 947580 results
- AUC: 0.7135 (✓ matched), Interpretability: 0.692 (-9.6%)
- IRT Fidelity dropped to 0.938 (-1.1%)

**Experiment 103929** (lambdareg001betazeros-recovered, λ_reg=0.01)
- Training stopped early at epoch 13 due to patience limit
- AUC: 0.7148 (✓ good), Interpretability: 0.640 (-16.5%)
- **Root cause identified:** Insufficient training time, β parameters didn't converge
- IRT Fidelity collapsed to 0.590 (-37.8%)

**Experiment 760279** (test, λ_reg=0.1)
- Test run with higher regularization
- AUC: 0.7138, Interpretability: 0.665 (-13.2%)

**Experiment 194542** (recovered-baseline, λ_reg=0.01)
- Major failure: AUC dropped to 0.6956 (-2.5%)
- Interpretability: 0.591 (-22.9%)
- Likely code regression or training issue

**Experiment 691006** (lambdareg005-recovered, λ_reg=0.05)
- Tested mid-range regularization
- AUC: 0.6957, Interpretability: 0.585 (-23.6%)
- Similar failure pattern to 194542

### November 30, 2025 - Code Changes & Investigation

**Major Code Change:** Commented out IRT initialization to fix "lazy learning" problem
- Hypothesis: Direct IRT initialization (β_init = β_IRT) prevents learning from data
- Solution: Initialize β=0.0, rely on weak regularization (λ_reg=0.01) for guidance
- Expected: Should improve or maintain interpretability

**Experiment 630671** (baseline-recovered, λ_reg=0.01)
- First experiment after code change
- AUC: 0.7148 (✓ excellent), Interpretability: 0.570 (-25.5%)
- **Worst interpretability yet despite good performance**
- IRT Fidelity: 0.529 (-44.2%)

**Experiment 212954** (test, λ_reg=0.01)
- AUC: 0.7148 (✓ excellent), Interpretability: 0.617 (-19.5%)
- IRT Fidelity catastrophic: 0.557 (-41.2%)
- BKT slightly worse: 0.488 (-3.5%)
- **Paradox:** Performance improved but interpretability collapsed

**Experiment 114419** (lambdareg01, λ_reg=0.1)
- Higher regularization test
- AUC: 0.7148, Interpretability: 0.594 (-22.4%)
- Higher regularization didn't help

**Experiment 343599** (lambdareg005, λ_reg=0.05)
- Mid-range regularization
- AUC: 0.7146, Interpretability: 0.604 (-21.2%)
- No improvement over other recent experiments

---

## Codebase Evolution Timeline

### Commit 3a35fda (Nov 28, 2025) - Initial iKT2 Implementation
**Title:** "feat: iKT2 model, an IRT-based interpretable KT with two-phase training - test AUC 0.7148"

**Major Changes:**
- Created `pykt/models/ikt2.py`: Dual-head transformer with IRT mastery estimation
- Created `examples/train_ikt2.py`: Two-phase training script with auto-switch
- Created `examples/eval_ikt2.py`: Evaluation script with interpretability metrics
- Updated `examples/mastery_states.py`: Added θ/β/M_IRT extraction support
- Updated `examples/run_repro_experiment.py`: Auto-launch mastery analysis

**Architecture Features:**
- Single encoder with dual prediction heads (performance + IRT mastery)
- Ability encoder: extracts θ_i(t) from knowledge state h[t]
- Learnable skill difficulty embeddings: β_k (initialized at 0.0)
- IRT mastery formula: M_IRT = σ(θ - β) per Rasch 1PL model

**Loss Function:**
- Phase 1: L = L_BCE + λ_reg × L_reg (warmup, 12 epochs)
- Phase 2: L = L_BCE + λ_align × L_align + λ_reg × L_reg (alignment, auto-switch)
- L_reg = MSE(β_learned, β_IRT) for scale preservation
- L_align = MSE(p_correct, M_IRT) for psychometric consistency

**Experiments Included:**
- 326769 (test): Initial proof-of-concept, AUC=0.7175
- 663970 (test): Replication, AUC=0.7175
- 400293 (test/baseline): First successful run, AUC=0.7148, λ_reg=0.1

**Initial Results:**
- Test AUC: 0.7148 (competitive performance)
- Test Accuracy: 74.6%
- β parameters: mean=-0.23, std=0.06, range=[-0.25, 0.24] ← **SCALE PROBLEM**
- Rasch scale: [-3.30, 1.09] (10x wider than learned β)

**Key Issue Identified:** β parameters compressed to wrong scale despite regularization

---

### Commit b3ff180 (Nov 29, 2025) - Baseline Documentation
**Title:** "doc: establish iKT2 baseline experiment (20251128_232738_ikt2_baseline_400293) - test AUC: 0.7148, IRT correlation: 0.8304"

**Changes:**
- Documented baseline configuration and training dynamics
- Added detailed analysis of two-phase training effectiveness
- Established reference metrics for future comparisons

**Experiments Included:**
- 400293 (baseline): Renamed from "test" to "baseline" status

**Key Findings:**
- Phase 1 (12 epochs): Val AUC 0.687→0.723, IRT corr -0.022→0.193
- Phase 2 (5 epochs): Val AUC 0.722→0.718, IRT corr 0.798→0.830
- Phase 2 Impact: +63.8pp IRT correlation for -0.54% AUC (118:1 tradeoff)
- IRT correlation: 0.8304 (STRONG, ≥0.80 threshold achieved)

**Status:** Established as baseline but scale problem remains (β wrong magnitude)

---

### Commit a70e95c (Nov 29, 2025) - IRT Static Ability Issue
**Title:** "doc: issue with IRT assumption of static ability"

**Changes:**
- Documented IRT assumption limitations
- Clarified θ_t is global scalar, not per-skill
- Explained how θ_t captures skill-specific performance via h[t] context

**Key Insights:**
- py-irt calibration treats students as having static ability (no temporal learning)
- θ_t adapts dynamically based on recent practice history in h[t]
- Model "remembers" skill context without explicit θ reset
- Example: Student 1038 shows skill 2 (θ=-0.009) vs skill 75 (θ=0.842)

**Files Modified:**
- `paper/ikt_architecture_approach.md`: Added θ_t behavior explanation
- Moved 596 lines from `assistant/ikt_irt_mastery_approach.md`

**No New Experiments**

---

### Commit fe9e086 (Nov 29, 2025) - IRT Initialization for Scale Fix
**Title:** "feat: solving the issue of difficulties β being out of scale by initializing them with IRT pre-calculated values created a lazy learning problem we try to solve changing λ_reg to 0.01 - test auc 0.713543"

**Major Changes:**
- Added IRT initialization: `model.load_irt_difficulties(beta_irt_device)` in `train_ikt2.py`
- Changed default λ_reg from 0.1 to 0.01 (weak regularization hypothesis)
- Updated interpretability metrics computation
- Modified `pykt/models/ikt2.py` to accept IRT initialization

**Rationale:**
- **Scale Problem Solution:** Initialize β=β_IRT to match Rasch scale [-3.30, 1.09]
- **Lazy Learning Concern:** Strong initialization might prevent learning from data
- **Mitigation Strategy:** Reduce λ_reg to 0.01 to allow more learning flexibility

**Experiments Included:**
- 581572 (baseline, λ_reg=0.1): With IRT init, AUC=0.7138, **I=0.694** (MODERATE)
- 947580 (lambdareg-001, λ_reg=0.01): With IRT init, AUC=0.7135, **I=0.766** (GOOD) ⭐
- 393607 (irt-finalability, λ_reg=0.1): Testing final ability extraction, AUC=0.7147

**Key Results (Experiment 947580):**
- β on correct scale: [-2.64, 0.12] (matches Rasch range)
- MSE: 0.109 (small magnitude errors, vs 3.806 without init)
- **IRT Fidelity: 0.948** (excellent task coherence)
- **BKT Correlation: 0.506** (moderate progression validity)
- **Head Agreement: 0.844** (excellent consistency)
- **Composite I: 0.766** ← **ONLY "GOOD" RATING ACHIEVED**

**Validation of Approach:**
- BKT improved 80% (0.303→0.546) - learning confirmed, not frozen
- IRT fidelity 0.948 (high but not 0.993 "frozen" level)
- β parameters shifted 8.5% from initialization (learning happening)
- MSE increased 1111% from initialization (adaptation to data)

**Commit Message Insights:**
- "High fidelity (r=0.948) can result from learning + quality IRT calibration"
- "Rank-order preserved because learned values naturally align with IRT scale"
- Distinguishes "frozen" (r=0.993, no learning) from "grounded" (r=0.948, learning)

**Files Modified (22 files):**
- Core: `examples/train_ikt2.py`, `pykt/models/ikt2.py`
- Configs: `configs/parameter_default.json`
- Evaluation: `examples/eval_ikt2.py`, `examples/compute_*.py`
- Documentation: `paper/*.md`, `examples/reproducibility.md`
- Experiments: 581572, 947580, 393607 (and renamed 400293)

---

### Uncommitted Changes (Nov 30, 2025) - IRT Initialization Removal
**Title:** [Not committed] "Remove IRT initialization to fix lazy learning hypothesis"

**Major Changes:**
- **Commented out IRT initialization** in `train_ikt2.py` (lines 533-547):
  ```python
  # COMMENTED OUT: Direct IRT initialization causes "lazy learning" problem
  # - Reduces model's incentive to learn from data
  # - Drops BKT correlation by ~32-40% (e.g., 0.506 → 0.345)
  # - Weak regularization (λ_reg=0.01) is sufficient for scale preservation
  # See experiments: 947580 (no init, I=0.766) vs 346878 (with init, I=0.692)
  ```
- Added `lambda_interpretability` parameter (unused, for backward compatibility)
- Commented out composite checkpoint selection approach
- Reverted to AUC-only checkpoint selection

**Rationale (Hypothesis):**
- IRT initialization prevents model from learning meaningful β parameters
- Causes 32-40% BKT correlation drops
- Weak regularization alone should preserve scale

**Experiments Included (Post-Change):**
- 346878 (lambdareg001-recovered, λ_reg=0.01): AUC=0.7135, I=0.692 (-9.6%)
- 103929 (lambdareg001betazeros-recovered, λ_reg=0.01): AUC=0.7148, I=0.640 (-16.5%) [early stop epoch 13]
- 760279 (test, λ_reg=0.1): AUC=0.7138, I=0.665 (-13.2%)
- 194542 (recovered-baseline, λ_reg=0.01): AUC=0.6956, I=0.591 (-22.9%) [major failure]
- 691006 (lambdareg005-recovered, λ_reg=0.05): AUC=0.6957, I=0.585 (-23.6%)
- 630671 (baseline-recovered, λ_reg=0.01): AUC=0.7148, **I=0.570 (-25.5%)** [worst interpretability]
- 212954 (test, λ_reg=0.01): AUC=0.7148, I=0.617 (-19.5%)
- 114419 (lambdareg01, λ_reg=0.1): AUC=0.7148, I=0.594 (-22.4%)
- 343599 (lambdareg005, λ_reg=0.05): AUC=0.7146, I=0.604 (-21.2%)

**Actual Results:**
- **HYPOTHESIS CONTRADICTED:** All experiments show worse interpretability
- IRT Fidelity collapsed: 0.948→0.53-0.64 (41-44% degradation)
- Composite I dropped: 0.766→0.57-0.64 (19-26% degradation)
- Performance stable/improved: AUC maintained at 0.714-0.715

**Critical Finding:**
The "fix" made things worse. Removing IRT initialization:
- ✗ Did NOT improve BKT correlation (stayed at 0.36-0.49)
- ✗ DESTROYED IRT fidelity (0.948→0.53-0.64)
- ✗ COLLAPSED composite interpretability (GOOD→POOR rating)
- ✓ Maintained/improved predictive performance (AUC stable)

**Status:** **NOT COMMITTED** - Changes staged but reproducibility crisis identified before commit

---

## Experiment Comparison Table

### Complete Results (Experiments with Full Metrics)

| Exp ID | Title | λ_reg | AUC | ΔAUC | I | ΔI | IRT Fid | ΔIRT | BKT | ΔBKT | Head | ΔHead |
|--------|-------|-------|-----|------|---|----|---------| -----|-----|------|------|-------|
| **947580** | **lambdareg-001** | **0.01** | **0.7135** | **-0.0013** | **0.7658** | **±0.0000** | **0.9478** | **-0.0456** | **0.5056** | **±0.0000** | **0.8439** | **±0.0000** |
| 581572 | baseline | 0.1 | 0.7138 | -0.0011 | 0.6943 | -0.0715 | 0.9933 | +0.0000 | 0.2859 | -0.2197 | 0.8036 | -0.0404 |
| 346878 | lambdareg001-recover | 0.01 | 0.7135 | -0.0013 | 0.6923 | -0.0735 | 0.9377 | -0.0556 | 0.3447 | -0.1610 | 0.7945 | -0.0495 |
| 760279 | test | 0.1 | 0.7138 | -0.0011 | 0.6650 | -0.1008 | 0.9317 | -0.0617 | 0.2792 | -0.2264 | 0.7841 | -0.0599 |
| 103929 | lambdareg001betazero | 0.01 | 0.7148 | +0.0000 | 0.6395 | -0.1262 | 0.5897 | -0.4037 | 0.5053 | -0.0003 | 0.8237 | -0.0203 |
| 400293 | test | 0.1 | 0.7148 | -0.0001 | 0.6012 | -0.1646 | 0.5739 | -0.4195 | 0.4707 | -0.0350 | 0.7591 | -0.0849 |
| 343599 | lambdareg005 | 0.05 | 0.7146 | -0.0002 | 0.6038 | -0.1620 | 0.6250 | -0.3683 | 0.4083 | -0.0973 | 0.7781 | -0.0659 |
| 212954 | test | 0.01 | 0.7148 | +0.0000 | 0.6168 | -0.1490 | 0.5573 | -0.4360 | 0.4879 | -0.0177 | 0.8052 | -0.0387 |
| 114419 | lambdareg01 | 0.1 | 0.7148 | -0.0001 | 0.5940 | -0.1718 | 0.6392 | -0.3541 | 0.3598 | -0.1459 | 0.7831 | -0.0609 |
| 194542 | recovered-baseline | 0.01 | 0.6956 | -0.0192 | 0.5907 | -0.1750 | 0.6842 | -0.3092 | 0.3302 | -0.1755 | 0.7579 | -0.0861 |
| 691006 | lambdareg005-recover | 0.05 | 0.6957 | -0.0191 | 0.5855 | -0.1803 | 0.6212 | -0.3721 | 0.4020 | -0.1036 | 0.7332 | -0.1107 |
| 630671 | baseline-recovered | 0.01 | 0.7148 | +0.0000 | 0.5703 | -0.1955 | 0.5288 | -0.4646 | 0.4177 | -0.0880 | 0.7645 | -0.0794 |

**Legend:**
- **Bold** = Best experiment (947580)
- AUC = Test set Area Under Curve (predictive performance)
- I = Composite Interpretability Score: (IRT_Fidelity + BKT_Correlation + Head_Agreement) / 3
- IRT Fid = IRT Difficulty Fidelity (correlation with Rasch model)
- BKT = BKT Correlation (learning progression validity)
- Head = Head Agreement (prediction consistency between dual heads)
- Δ columns = Difference from best value (positive is better)

### Interpretability Rating Scale
- **GOOD:** I ≥ 0.75 (Target for publication)
- **MODERATE:** 0.65 ≤ I < 0.75 (Needs improvement)
- **POOR:** I < 0.65 (Not acceptable for publication)

### Understanding Interpretability Measurement

The **Composite Interpretability Score (I)** is calculated as the average of three distinct correlation-based metrics:

$$I = \frac{\text{IRT Fidelity} + \text{BKT Correlation} + \text{Head Agreement}}{3}$$

This metric ranges from -1 to 1 (theoretically), but typically 0.5-1.0 in practice for functional models.

#### Component Metrics Explained:

**1. IRT Fidelity (Task Coherence)**
- **Measures:** Correlation between learned skill difficulty parameters (β) and ground-truth IRT/Rasch difficulties
- **Range:** -1 to 1 (Pearson correlation coefficient)
- **Interpretation:**
  - **High values (>0.9):** Model learns difficulty parameters that align with established psychometric models
  - **Moderate values (0.6-0.9):** Partial alignment, some divergence from expected difficulty ordering
  - **Low values (<0.6):** Model's difficulty estimates don't match expected task difficulty hierarchy
- **Best experimental values:** 0.993 (exp 581572), 0.948 (exp 947580)
- **Recent degradation:** 0.53-0.64 (drops of 41-44% from baseline)
- **Significance:** Primary driver of interpretability degradation

**2. BKT Correlation (Progression Validity)**
- **Measures:** Correlation between model's knowledge state estimates and Bayesian Knowledge Tracing predictions
- **Range:** -1 to 1 (Pearson correlation coefficient)
- **Interpretation:**
  - **High values (>0.5):** Model captures realistic learning progressions over time
  - **Moderate values (0.3-0.5):** Partial alignment with expected learning curves
  - **Low values (<0.3):** Model's knowledge evolution doesn't match BKT learning dynamics
- **Best experimental value:** 0.506 (exp 947580)
- **Recent performance:** 0.36-0.49 (relatively stable, minor 3-18% drops)
- **Significance:** Secondary indicator, more stable across experiments

**3. Head Agreement (Prediction Consistency)**
- **Measures:** Correlation between predictions from the two heads in the dual-head architecture (knowledge state head vs. performance prediction head)
- **Range:** -1 to 1 (Pearson correlation coefficient)
- **Interpretation:**
  - **High values (>0.8):** Both heads agree on student knowledge level, consistent internal representation
  - **Moderate values (0.6-0.8):** Partial agreement, some divergence in how heads interpret knowledge
  - **Low values (<0.6):** Heads give contradictory signals about student state
- **Best experimental value:** 0.844 (exp 947580)
- **Recent performance:** 0.76-0.81 (minor 4-10% drops)
- **Significance:** Indicates internal model consistency

#### Why "Interpretability Degradation"?

The term refers to the substantial drop from the best result to recent experiments:

- **Baseline (Exp 947580):** I = 0.766 (GOOD rating) ← Target for reproducibility
- **Recent experiments:** I = 0.570-0.640 (POOR rating)
- **Degradation magnitude:** -19% to -26% drop in composite score

**Root cause:** The **IRT Fidelity component** collapsed from 0.948 to 0.53-0.64, a 41-44% degradation. This indicates the model's learned difficulty parameters (β) no longer align with psychometric ground truth, making the model's internal reasoning opaque and potentially unreliable for educational applications.

#### Interpretability vs. Performance Trade-off

Notably, predictive performance (AUC) has remained stable or even improved slightly (0.713-0.715) while interpretability degraded. This creates a paradox:
- **Good news:** The model still predicts student performance accurately
- **Bad news:** We don't understand *why* it makes those predictions
- **Implication:** Without interpretability, the model is a "black box" unsuitable for educational decision-making

---

## Critical Analysis

### 1. Best Experiment: 947580

**Experiment ID:** 947580  
**Title:** lambdareg-001  
**Configuration:**
- λ_reg = 0.01 (weak regularization)
- Seed: 42
- 30 epochs training
- IRT initialization: Likely active (based on code timeline)

**Results:**
- **Performance:** AUC = 0.7135 (excellent)
- **Interpretability:** I = 0.7658 (**GOOD** rating)
  - IRT Fidelity: 0.9478 (excellent task coherence)
  - BKT Correlation: 0.5056 (moderate progression validity)
  - Head Agreement: 0.8439 (excellent prediction consistency)

**Significance:** This is the **only experiment** that achieved "GOOD" interpretability rating. All paper claims and publication readiness depend on reproducing these results.

### 2. Parameter Effects (Preliminary Observations)

⚠️ **CAUTION:** Code changes between experiments make these comparisons uncertain.

**Lambda_reg (Regularization Strength):**
- **λ_reg = 0.01** (weak): Best interpretability when working (947580: I=0.766)
- **λ_reg = 0.05** (medium): Poor results (343599: I=0.604, 691006: I=0.585)
- **λ_reg = 0.1** (strong): Mixed results (581572: I=0.694, but recent runs: I=0.594-0.665)

**Observation:** Lower regularization (0.01) appears optimal, but reproducibility issues prevent definitive conclusions.

**Training Duration:**
- Experiment 103929 stopped early (epoch 13) → Poor interpretability (I=0.640)
- Longer training appears necessary for β parameter convergence

**IRT Initialization:**
- **Hypothesis:** Removing IRT initialization fixes "lazy learning"
- **Reality:** Post-removal experiments (630671, 212954) have WORSE interpretability
- **Conclusion:** Hypothesis contradicted by evidence; may need IRT initialization

### 3. Code Evolution Impact

**November 29 → November 30 Code Change:**
- Commented out IRT initialization (β_init = β_IRT)
- Rationale: Direct initialization causes 32-40% BKT correlation drops
- **Actual Result:** Interpretability degraded 19-26% after change
- **Paradox:** The "fix" made things worse

**Possible Explanations:**
1. IRT initialization was actually beneficial, not harmful
2. β=0.0 initialization prevents proper learning trajectory
3. Other code changes introduced unintended side effects
4. Experiment 947580 success was due to specific conditions not yet understood

### 4. Reproducibility Crisis

**Status:** 🔴 **CRITICAL**

**Evidence:**
- 11/12 experiments failed to reproduce baseline interpretability
- Only experiment 947580 achieved "GOOD" rating (I ≥ 0.75)
- Recent experiments consistently achieve I ≈ 0.57-0.64 (POOR rating)
- Interpretability drops of 19-26% from baseline

**Impact on Publication:**
- Cannot claim reliable interpretability without reproducibility
- Single successful run insufficient for peer review
- Risk of cherry-picking accusations
- Need minimum 3 independent runs with similar results

**Root Causes (Hypotheses):**
1. **Early stopping:** Training terminating before β parameters converge
2. **Code changes:** IRT initialization removal had opposite effect than intended
3. **Hyperparameter sensitivity:** Model highly sensitive to specific training conditions
4. **Random seed effects:** Success may be seed-dependent (all experiments use seed=42)
5. **Unknown factors:** Critical variables not yet identified or controlled

---

## Publishing Readiness Assessment

### Peer Review Checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Novel architecture | ✓ Pass | Dual-head transformer with skill difficulty parameters |
| Competitive performance | ✓ Pass | AUC ≈ 0.714, comparable to state-of-the-art |
| Interpretability claims | ✗ **FAIL** | Cannot reproduce baseline interpretability |
| Statistical significance | ✗ **FAIL** | Single successful run, no variance estimates |
| Reproducibility | ✗ **FAIL** | 11/12 experiments failed to replicate baseline |
| Ablation studies | ⚠ Partial | Parameter sweep attempted, inconclusive |
| Code availability | ✓ Pass | Full implementation available |
| Clear methodology | ⚠ Partial | Training procedure documented but unstable |

**Overall:** 🔴 **2.5 / 8 criteria passed** - NOT READY FOR PUBLICATION

### Blocking Issues

1. **Reproducibility Crisis (CRITICAL)**
   - Only 1 successful run out of 12 experiments
   - 91.7% failure rate unacceptable for publication
   - No error bars, confidence intervals, or variance estimates

2. **Interpretability Degradation (CRITICAL)**
   - Recent experiments: I = 0.57-0.64 (POOR rating)
   - Target baseline: I = 0.766 (GOOD rating)
   - 19-26% degradation from target

3. **IRT Fidelity Collapse (CRITICAL)**
   - Recent IRT fidelity: 0.53-0.64 (poor task coherence)
   - Baseline: 0.948 (excellent task coherence)
   - 41-44% degradation

4. **Code Instability**
   - Changes intended to improve interpretability made it worse
   - Unclear which code version produced baseline results
   - Cannot identify exact conditions for success

### Timeline Estimate to Publication Readiness

**Optimistic (3-4 weeks):**
- Identify root cause of reproducibility failure (1 week)
- Implement fix and validate with 5 independent runs (1 week)
- Complete ablation studies with stable codebase (1 week)
- Write/revise paper with reproducible results (1 week)

**Realistic (6-8 weeks):**
- Debug training procedure and β parameter learning (2 weeks)
- Systematic hyperparameter optimization (2 weeks)
- Multi-seed validation (3+ runs per configuration) (2 weeks)
- Statistical analysis and paper revision (2 weeks)

**Pessimistic (3-6 months):**
- Fundamental architectural issue requiring redesign (4-8 weeks)
- Re-implementation and validation from scratch (4-8 weeks)
- Complete experimental pipeline rebuild (4-8 weeks)

**Current Estimate:** **Realistic scenario (6-8 weeks)** - assuming root cause is identifiable and fixable.

---

## Recommendations

### Immediate Actions (Priority 1 - This Week)

1. **Root Cause Analysis**
   - Compare exact code versions between experiment 947580 and recent failures
   - Use git diff to identify ALL changes, not just IRT initialization
   - Check if experiment 947580 actually used IRT initialization or not
   - Verify training logs for any differences in procedure

2. **Reproduce Experiment 947580 Exactly**
   - Restore code to exact state from Nov 29 (commit before IRT initialization removal)
   - Run 3 independent replications with seeds 42, 21, 63
   - If reproducible → confirms code change broke interpretability
   - If not reproducible → indicates deeper instability

3. **Training Duration Analysis**
   - Check training logs for all experiments
   - Identify if early stopping is systematic problem
   - Consider increasing patience parameter from 4 to 10
   - Monitor β parameter evolution epoch-by-epoch

4. **Emergency Recovery Plan**
   - If IRT initialization was necessary, restore it
   - Test hypothesis: Initialize β=β_IRT but use very weak regularization (λ_reg=0.001)
   - Goal: Balance initialization benefit with learning flexibility

### Short-term Actions (Priority 2 - Next 2 Weeks)

5. **Systematic Parameter Sweep**
   - Once stable baseline achieved, test:
     - λ_reg: [0.001, 0.005, 0.01, 0.05, 0.1]
     - patience: [4, 6, 8, 10]
     - learning_rate: [0.0001, 0.0005]
   - Run each configuration 3 times (seeds: 42, 21, 63)
   - Calculate mean ± std for all metrics

6. **Multi-Seed Validation**
   - Best configuration: minimum 5 runs with different seeds
   - Report mean AUC, interpretability with 95% confidence intervals
   - Verify interpretability consistently achieves "GOOD" rating (I ≥ 0.75)

7. **Ablation Studies**
   - Effect of IRT initialization: with vs. without
   - Effect of dual heads: dual vs. single prediction head
   - Effect of β parameters: learnable vs. fixed
   - Each ablation: 3 runs for statistical validity

8. **Monitoring Infrastructure**
   - Add epoch-level β parameter tracking
   - Log learning curves for interpretability metrics
   - Implement early warning system for interpretability degradation
   - Save checkpoints every 5 epochs for forensic analysis

### Medium-term Actions (Priority 3 - Next 4 Weeks)

9. **Code Stabilization**
   - Create frozen "paper" branch with validated code
   - Version control for parameter_default.json changes
   - Automated reproducibility tests before any code changes
   - Document exact environment (package versions, CUDA, etc.)

10. **Statistical Rigor**
    - Hypothesis testing: compare iKT2 vs. baseline models
    - Effect size calculations for interpretability improvements
    - Bonferroni correction for multiple comparisons
    - Power analysis to determine required sample size

11. **Baseline Comparisons**
    - Reproduce results from comparison models (DKT, SAKT, etc.)
    - Ensure fair comparison (same dataset, folds, hyperparameter search budget)
    - Statistical tests: paired t-tests or Wilcoxon signed-rank

12. **Paper Draft Preparation**
    - Write methods section with complete reproducibility details
    - Create figures: learning curves, ablation results, interpretability analysis
    - Draft results section with placeholder for final numbers
    - Prepare supplementary materials: hyperparameter details, additional experiments

### Long-term Strategic Recommendations

13. **Publication Venue Selection**
    - **If reproducibility achieved (I ≥ 0.75):**
      - Target: EDM (Educational Data Mining), AIED, or LAK conferences
      - Emphasize interpretability + competitive performance
      - Position as practical tool for educators
    
    - **If only moderate interpretability (0.65 ≤ I < 0.75):**
      - Target: Workshop papers or late-breaking results
      - Focus on promising approach needing refinement
      - Request feedback from community
    
    - **If reproducibility fails (I < 0.65):**
      - Do NOT submit to peer review
      - Risk of rejection and negative reviews
      - Consider pivot to technical report or arXiv preprint

14. **Alternative Interpretation of Results**
    - If interpretability remains unstable, consider:
      - Reframing paper as "performance-interpretability tradeoff" analysis
      - Position as cautionary tale about interpretability challenges
      - Focus on architectural contributions, acknowledge interpretability limitations
      - Compare multiple training strategies with different tradeoff points

15. **Collaboration and External Validation**
    - Share code with collaborators for independent reproduction attempts
    - Consider open-sourcing before publication for community validation
    - Engage with knowledge tracing research community for feedback
    - Potential co-authors with expertise in interpretable ML

---

## Summary

### Current Status

**Model Performance:** ✓ Excellent (AUC ≈ 0.714, competitive with state-of-the-art)

**Model Interpretability:** ✗ CRITICAL FAILURE
- Target: I = 0.766 (GOOD rating)
- Current: I = 0.57-0.64 (POOR rating)
- Degradation: -19% to -26% from baseline

**Reproducibility:** ✗ CRITICAL FAILURE
- Success rate: 1/12 experiments (8.3%)
- Failure rate: 11/12 experiments (91.7%)
- Statistical significance: Not established

**Publication Readiness:** 🔴 **NOT READY**
- Blocking issues: Reproducibility, interpretability instability
- Estimated time to publication: 6-8 weeks (realistic)
- Risk level: HIGH (may require architectural changes)

### Best Experiment

**Experiment 947580** remains the ONLY successful demonstration of the iKT2 model's interpretability claims:
- AUC: 0.7135 (competitive performance)
- Interpretability: 0.7658 (**GOOD** rating)
- Configuration: λ_reg = 0.01, seed = 42

**All publication claims depend on reproducing this single result.**

### Key Insights

1. **Lambda_reg = 0.01** appears optimal (when working)
2. **Early stopping** may prevent β parameter convergence
3. **IRT initialization removal** may have broken interpretability (contrary to hypothesis)
4. **Code changes** between experiments prevent definitive parameter effect conclusions
5. **Single seed (42)** used in all experiments - need multi-seed validation

### Critical Path to Publication

```
Week 1-2: Root cause identification & baseline reproduction (3 runs)
    ↓
Week 3-4: Parameter optimization & multi-seed validation (15-20 runs)
    ↓
Week 5-6: Ablation studies & statistical analysis (12-15 runs)
    ↓
Week 7-8: Paper revision & submission preparation
    ↓
Submit to EDM 2026 or AIED 2026
```

**Success criteria:**
- Minimum 3 independent reproductions of I ≥ 0.75
- Mean interpretability across 5 seeds: I ≥ 0.75 ± 0.05
- AUC maintained at ≥ 0.71
- Complete ablation studies showing component contributions

### Recommended Next Action

**THIS WEEK:** Execute git diff between Nov 29 (pre-IRT-removal) and Nov 30 (post-removal) to identify ALL code changes. Restore pre-removal state and attempt exact reproduction of experiment 947580 with 3 different seeds. This single action will determine if the reproducibility crisis is due to the IRT initialization change or a deeper issue.

**Status:** ⏸ **DEVELOPMENT PAUSED** until reproducibility crisis resolved.

---

## Appendix: Experiment Details

### Experiments Without Complete Metrics

The following experiments completed training but lack full interpretability metrics (excluded from main analysis):
- 326769, 663970, 393607, 390188, 496497, 471833, 342440, 353313, 379219, 592022, 877700, 897903, 524632, 538460

### Data Files

- Experiment configs: `/workspaces/pykt-toolkit/experiments/*/config.json`
- Test results: `/workspaces/pykt-toolkit/experiments/*/test_results.json`
- Interpretability metrics: `irt_correlation_test.json`, `bkt_validation_final.json`, `head_agreement_test.json`
- Full dataset: `data/assist2015/`

### Code Repository

- Branch: `v0.0.25-iKT`
- Main training script: `examples/train_ikt2.py`
- Model implementation: `pykt/models/ikt2.py`
- Evaluation scripts: `examples/eval_ikt2.py`, `examples/compute_*.py`

---

**Document Version:** 1.0  
**Last Updated:** November 30, 2025  
**Next Review:** After reproducibility investigation (estimated Dec 7, 2025)
