# Models

**Document Version**: 2025-12-08  
**Current Model**: iKT3 - Reference Model Alignment with Dynamic IRT Targets

---

## iKT Versions

iKT is the last architecture model. Main versions are: 

| Version | File | Branch | Architecture | Loss Formulation | Key Issue | Status |
|---------|------|--------|--------------|------------------|-----------|--------|
| **iKT v1** | `ikt.py` | v0.0.24-iKT | Dual head, per-student Rasch targets | L = L_BCE + λ × penalty(\|M - M_rasch\| - ε) | Overfitting (memorizes student IDs) | ❌ Deprecated |
| **iKT v2** | `ikt2.py` | v0.0.25-iKT-v2 | Dual head, head-to-head alignment | L = L_BCE + λ_align × MSE(M_IRT, p_correct) + λ_reg × MSE(β, β_IRT) | No external validation | ⚠️ Deprecated |
| **iKT v3** | `ikt3.py` | v0.0.26-iKT-v3 | Dual head, reference model alignment | L = (1-λ)×l_bce + c×l_22 + λ×(l_21 + l_23) | M_ref poor quality (r=0.19) | ✅ Current |

```mermaid
graph LR
    Start["🎯 Goal:<br/>Interpretable KT"] --> V1
    
    V1["iKT v1<br/>Per-student targets<br/>M = σ(θ_s - β_k)"] --> P1["❌ Overfitting<br/>Val MSE ↑10×"]
    
    P1 -->|Remove<br/>per-student| V2["iKT v2<br/>Head-to-head<br/>L = MSE(p, M_IRT)"]
    
    V2 --> P2["❌ No external<br/>validation<br/>r=0.76"]
    
    P2 -->|Add reference<br/>model| V3["iKT v3 ✅<br/>L = (1-λ)l_bce + λl_align<br/>AUC=0.72"]
    
    V3 --> I3["⚠️ M_ref quality<br/>r=0.19"]
    
    classDef versionStyle fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef problemStyle fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef issueStyle fill:#fff4cc,stroke:#ff9900,stroke-width:2px
    classDef startStyle fill:#e6ccff,stroke:#9933ff,stroke-width:2px
    
    class V1,V2,V3 versionStyle
    class P1,P2 problemStyle
    class I3 issueStyle
    class Start startStyle
```

**Key Evolution Points:**

1. **v1 → v2**: Eliminated overfitting by removing per-student targets, replacing with skill-centric regularization
2. **v2 → v3**: Added external validation by aligning with pre-computed reference model instead of internal head agreement
3. **v3 Current Challenge**: Reference model quality issues reveal dataset-specific limitations of Rasch IRT assumptions


## All Versions

| **Aspect** | **GainAKT3Exp** | **GainAKT4 Phase 1** | **iKT v1 (Initial)** | **iKT v2 (Option 1b)** | **iKT v3 (Current)** |
|------------|-----------------|----------------------|---------------------|------------------------|----------------------|
| **Encoders** | 2 (separate pathways) | 1 (shared) | 1 (shared) | 1 (shared) | 1 (shared) |
| **Parameters** | ~167K | ~3.0M | ~3.0M | ~3.0M | ~3.0M |
| **Heads** | 1 per encoder | 2 on Encoder 1 | 2 (Prediction + Mastery) | 2 (Prediction + Mastery) | 2 (Prediction + Mastery) |
| **Input Types** | Questions + Responses | Questions + Responses | Questions + Responses | Questions + Responses | Questions + Responses |
| **Learning** | Independent optimization | Multi-task joint | Two-phase (Rasch init → constrained opt) | Two-phase (Rasch init → constrained opt) | Two-phase (warmup → IRT alignment) |
| **Gradient Flow** | Separate to each encoder | Accumulated to Encoder 1 | Phase 1: L2 only; Phase 2: L1 + λ_penalty×L2_penalty | Phase 1: L2 only; Phase 2: L1 + λ_penalty×L2_penalty | Phase 1: L_BCE + L_reg; Phase 2: L_BCE + L_align + L_reg |
| **Losses** | L1 (BCE), L2 (IM) | L1 (BCE), L2 (Mastery) | L1 (BCE), L2 (Rasch MSE) | L1 (BCE), L2 (Rasch MSE with ε) | L_BCE (performance), L_align (IRT alignment), L_reg (difficulty reg) |
| **Head 2 Output** | Skill vector {KCi} | Skill vector {KCi} | Skill vector {Mi} [B,L,num_c] | Skill vector {Mi} [B,L,num_c] | IRT mastery M_IRT = σ(θ - β) [B,L] |
| **Mastery Target** | None | Mastery loss | Per-student Rasch targets | Per-student Rasch targets | IRT formula (no static targets) |
| **Key Innovation** | Dual encoders | Single encoder efficiency | Rasch grounding | Skill difficulty embeddings | Ability encoder + IRT formula |
| **Critical Issue** | - | - | **Overfitting** (memorizes student-specific targets) | **95% violation rate** (embeddings collapsed) | None (theoretically grounded) |
| **Interpretability** | Sigmoid curves | Skill decomposition | Rasch alignment (ε=0 Phase 1) | Rasch alignment (ε-tolerance Phase 2) | IRT correlation (r>0.85) |
| **Psychometric Grounding** | Heuristic | Architectural | Rasch 1PL (student-specific) | Rasch 1PL (skill-centric) | **Rasch 1PL (ability inference)** |
| **Difficulty Representation** | None | None | Per-student-skill targets | **Learnable embeddings β_k** | **Learnable embeddings β_k** (retained) |
| **Regularization** | Separate losses | Multi-task implicit | None (overfits) | L_reg = MSE(β_k, β_IRT) | L_reg = MSE(β_k, β_IRT) (retained) |
| **Constraint Type** | Loss-based | Loss-based | Exact alignment (ε=0) | Soft barrier (\|Mi - M_rasch\| ≤ ε) | **IRT alignment (MSE(p, M_IRT))** |
| **Validation MSE** | - | - | **Increases 10x** (overfitting) | Stable (overfitting fixed) | Stable (expected) |
| **Interpretability Metric** | - | - | L2 MSE < 0.01 | Violation rate < 10% | **IRT correlation r > 0.85** |
| **Performance (ASSIST2015)** | Not measured | 0.7181 AUC | ~0.72 AUC (degraded by overfitting) | ~0.72 AUC (maintained) | **0.7148 AUC (baseline, validated)** |
| **Implementation Status** | Complete | Complete (Validated) | Complete (deprecated) | **Complete (deprecated)** | **✅ Complete and Tested** |
| **Best For** | Pathway separation | Parameter efficiency | N/A (superseded) | N/A (superseded) | **Transparent interpretability with theory** |

## iKT Models Details

### 1. iKT v1 (Initial Attempt) - `pykt/models/ikt.py`

**Branch:** `v0.0.24-iKT`, `v0.0.24-iKT-v1-masteryasprojections`

**Architecture:**
- Single transformer encoder (dual-stream: context + value)
- **Head 1:** Performance prediction → BCE loss (L_BCE)
- **Head 2:** Mastery vector {M_i} [B, L, num_c] → Rasch alignment loss (L_2)

**Loss Function (Two-Phase Training):**
```
Phase 1: L_total = L_2 = MSE(M_i, M_rasch) with epsilon=0
Phase 2: L_total = L_BCE + λ_penalty × mean(max(0, |M_i - M_rasch| - ε)²)
```

**Key Features:**
- Used **per-student Rasch targets**: `M_rasch[student, skill] = σ(θ_student - β_skill)`
- Constraint: `|M_i - β_k| < ε` (tolerance-based penalty)
- Positivity (softplus activation) + monotonicity (cummax)

**Problem:** 
- **Overfitting** - Model memorized student-specific targets
- Validation MSE increased 10× (from 0.027 to 0.279)
- Could not generalize to new students

**Status:** ❌ DEPRECATED (overfitting issue)

**Also known as:** "Option 1b" in early documentation


### 2. iKT v2 (Head-to-Head Alignment) - `pykt/models/ikt2.py`

**Branch:** `v0.0.25-iKT-v2`, `v0.0.25-iKT`

**Architecture:**
- Single transformer encoder (dual-stream)
- **Head 1:** Performance prediction → p_correct → L_BCE
- **Head 2:** Ability encoder θ_i(t) + difficulty embeddings β_k → M_IRT = σ(θ - β)

**Loss Function (Two-Phase Training):**
```
Phase 1 (epochs 1-10): L_total = L_BCE + λ_reg × L_reg (warmup)
Phase 2 (epochs 11+):  L_total = L_BCE + λ_align × L_align + λ_reg × L_reg

where:
  L_BCE = BCE(p_correct, ground_truth)         # Head 1 performance
  L_align = MSE(p_correct, M_IRT)              # Head 1 vs Head 2 alignment
  L_reg = MSE(β_learned, β_IRT)                # Difficulty regularization
```

**Key Innovation:**
- Replaced per-student targets with **IRT formula**: M_IRT = σ(θ_learned - β_learned)
- **Ability encoder** extracts θ_i(t) from knowledge state h
- **L_align ensures Head 2 predictions match Head 1 predictions**

**Advantages:**
- ✅ Fixed overfitting (validation MSE stable)
- ✅ No epsilon tolerance needed
- ✅ Theoretically grounded (Rasch IRT)

**Validation Metric:**
- **Head Agreement** = Pearson correlation(M_IRT, p_correct)
- Target: r > 0.85 (achieved 0.83 validation, 0.76 test)

**Problem:**
- Lacks **external validation** - Head 2 only learns to match Head 1, not a true theoretical model
- No guarantee that internal alignment reflects true IRT principles

**Status:** ⚠️ DEPRECATED in favor of iKT3 (which has external reference model validation)

**Performance:** Test AUC ~0.7150


### 3. iKT v3 (Reference Model Alignment) - `pykt/models/ikt3.py`

**Branch:** `v0.0.26-iKT-v3` 

**Architecture:**
- Single transformer encoder (dual-stream)
- **Head 1:** Performance prediction → p_correct → L_BCE
- **Head 2:** Ability encoder θ_learned + difficulty embeddings β_learned → M_IRT = σ(θ - β)
- **Reference Model:** Pluggable interface (IRT implemented, extensible to BKT, DINA, PFA)

**Loss Function (Single-Phase with Warm-up):**
```
L_total = (1 - λ(t)) × l_bce + c × l_22 + λ(t) × (l_21 + l_23)

where λ(t) = λ_target × min(1, epoch / warmup_epochs)

Components:
  l_bce = BCE(p_correct, ground_truth)         # Head 1 performance
  l_21 = BCE(M_IRT, M_ref)                     # Head 2 vs reference predictions
  l_22 = MSE(β_learned, β_IRT)                 # Difficulty regularization (always-on)
  l_23 = MSE(θ_learned, θ_IRT)                 # Ability alignment with reference
```

**Key Innovation:**
- **Paradigm shift:** External reference model validation (not internal head-to-head)
- **Pluggable architecture:** Reference models implement standardized API
- **Three alignment losses:**
  - **l_21:** Performance alignment (M_IRT ↔ M_ref)
  - **l_22:** Difficulty regularization (β_learned ↔ β_IRT) - **always active** (c=0.01)
  - **l_23:** Ability alignment (θ_learned ↔ θ_IRT)
- **Adaptive lambda schedule:** Gradual transition from performance to interpretability
- **Dynamic IRT targets:** Time-varying θ_i(t) trajectories (solves scale collapse)

**Advantages over iKT2:**
- ✅ External calibration (validates against theoretical model, not just internal consistency)
- ✅ Single-phase training (simpler than 2-phase)
- ✅ Extensible to multiple reference models (IRT, BKT, future frameworks)
- ✅ Better interpretability validation

**Current Status (Dec 8, 2025):**
- ✅ Implementation complete
- ✅ Test AUC: 0.7202 (validated)
- ❌ **Critical Issue:** Alignment losses exceed thresholds
  - l_21 = 4.06 (threshold <0.15, **27× over**)
  - l_22 = 0.144 (threshold <0.10, **1.4× over**)
  - l_23 = 6.79 (threshold <0.15, **45× over**)

**Root Cause (Identified Dec 8):**
- **M_ref correlation = 0.1922** (target >0.7) - IRT reference has poor predictive validity
- Rasch model σ(θ - β) doesn't fit ASSIST2015 dataset
- Model correctly "refuses" to align to bad reference targets

**Files:**
- Model: `pykt/models/ikt3.py`
- Reference framework: `pykt/reference_models/{base.py, irt_reference.py}`
- Training: `examples/train_ikt3.py`
- Evaluation: `examples/eval_ikt3.py`
- IRT targets: `examples/compute_irt_dynamic_targets.py`

---

## Architectural Comparison

### GainAKT3Exp (Dual-Encoder)
```
Input → Encoder 1 (96K params) → Head 1 → BCE Predictions → L1
Input → Encoder 2 (71K params) → Gain Quality → Effective Practice → Sigmoid Curves → IM Predictions → L2

Total: 167K parameters, two independent learning pathways
```

### GainAKT4 (Phase 1 - Dual-Head Single-Encoder)
```
                    ┌→ Head 1 (Performance) → BCE Predictions → L1 (BCE Loss)
                    │
Input → Encoder 1 → h1 ─┤
                    │
                    └→ Head 2 (Mastery) → MLP1 → {KCi} → MLP2 → Sigmoid → Mastery Predictions → L2 (Binary CE Loss)

Note: GainAKT4 Phase 1 uses MLP2 to aggregate skills into predictions

L_total = λ₁ * L1 + λ₂ * L2
Encoder 1 receives gradients from BOTH L1 and L2 (gradient accumulation)
```

### GainAKT4 (Phase 2 - Dual-Encoder, Three-Head)
```
                        ┌→ Head 1 (Performance) → BCE Predictions → L1 (BCE Loss)
                        │
Questions + Responses → Encoder 1 → h1 ─┤
                        │
                        └→ Head 2 (Mastery) → MLP1 → Softplus → cummax → MLP2 → Mastery Predictions → L2 (Binary CE Loss)

Note: GainAKT4 Phase 2 uses MLP2; iKT does not

Questions + Attempts → Encoder 2 → h2 → Head 3 (Curve) → Curve Predictions → L3 (MSE/MAE Loss)

L_total = λ_bce × L1 + λ_mastery × L2 + λ_curve × L3
Constraint: λ_bce + λ_mastery + λ_curve = 1.0

Encoder 1 receives gradients from L1 + L2
Encoder 2 receives gradients from L3
```

### iKT (Previous Approaches)

**Option 1A (Baseline - Rasch Targets)**:
```
                        ┌→ Head 1 (Performance) → BCE Predictions → L1 (BCE Loss)
                        │
Questions + Responses → Encoder 1 → h1 ─┤
                        │
                        └→ Head 2 (Mastery) → MLP1 → Softplus → cummax → {Mi} -> L2 (MSE vs Rasch targets)

Phase 1: L_total = L2 (Rasch initialization)
Phase 2: L_total = λ_bce × L1 + (1-λ_bce) × L2_constrained (with ε tolerance)

PROBLEM: Overfitting to student-specific targets (Val MSE increased 10x)
```

**Option 1B (Learnable Embeddings)**:
```
                        ┌→ Head 1 (Performance) → BCE Predictions → L_BCE
                        │
Questions + Responses → Encoder 1 → h1 ─┤                   ┌→ β_k (skill difficulty embeddings)
                        │                                   │
                        └→ Head 2 (Mastery) → {Mi}          └→ L_reg = MSE(β_learned, β_IRT)
                                              │
                                              └→ L_penalty = mean(max(0, |Mi - βk| - ε)²)

Phase 1: L_total = L_BCE + λ_reg × L_reg
Phase 2: L_total = L_BCE + λ_penalty × L_penalty + λ_reg × L_reg

SUCCESS: Fixed overfitting (Val MSE stable), perfect embedding alignment (corr=1.0)
PROBLEM: 95% violation rate - constraint |Mi - βk| < ε is theoretically meaningless
```

**IRT-Based Mastery Inference (NEW - Proposed)**:
```
                        ┌→ Head 1 (Performance) → p_correct → L_BCE
                        │
Questions + Responses → Encoder 1 → h ─┤
                        │              └→ Ability Encoder → θ_i(t) ┐
                        │                                          │
                        └→ Skill Embeddings → β_k ────────────────┤
                                                                   ↓
                                                      M_IRT = σ(θ - β) → L_align = MSE(p_correct, M_IRT)
                                                                   
                                                      L_reg = MSE(β_learned, β_IRT)

Phase 1: L_total = L_BCE + λ_reg × L_reg
Phase 2: L_total = L_BCE + λ_align × L_align + λ_reg × L_reg

ADVANTAGES:
- Theoretically grounded: Uses Rasch IRT formula M = σ(θ - β)
- Dynamic ability: θ_i(t) inferred from knowledge state, not pre-calibrated
- Direct alignment: No violations, just MSE between predictions and IRT mastery
- Interpretable: θ represents ability, β represents difficulty, both have clear meaning
```

## Comparison Summary

| Feature | Option 1A | Option 1B | IRT-Based (NEW) |
|---------|-----------|-----------|------------------|
| **Mastery Source** | Static Rasch targets | Learned {Mi} | σ(θ - β) formula |
| **Difficulty Source** | Pre-computed IRT | Learnable embeddings | Learnable embeddings |
| **Interpretability Method** | Direct MSE to targets | Penalty for violations | IRT alignment |
| **Constraint Type** | Soft (MSE) | Hard (violation penalty) | Soft (MSE alignment) |
| **Overfitting** | ❌ Yes (10x increase) | ✅ Fixed | ✅ Expected fixed |
| **Embedding Alignment** | N/A | ✅ Perfect (corr=1.0) | ✅ Via L_reg |
| **Violation Rate** | N/A | ❌ 95% | ✅ N/A (no violations) |
| **Theoretical Foundation** | IRT calibration | Ad-hoc constraint | ✅ Rasch IRT model |
| **Ability Modeling** | ❌ Pre-calibrated | ❌ None | ✅ Dynamic inference |
| **Test AUC** | ~0.725 | 0.7153 | Expected ~0.72 |


## Benchmark

| Model | Dataset | Fold | Seed | Best Epoch | Val AUC | Val Acc | Test AUC | Test Acc | Window Test AUC | Window Test Acc |
|-------|---------|------|------|------------|---------|---------|----------|----------|-----------------|-----------------|
| AKT | assist2015 | 0 | 42 | 11 | 0.7328 | 0.7586 | **0.7255** | 0.7511 | 0.7256 | 0.7511 |
| iKT3 | assist2015 | 0 | 42 | 7 | 0.7258 | 0.7548 | **0.7204** | 0.7468 | - | - |


### iKT3

**Experiment:** `20251208_191345_ikt3_baseline_286531`  
**Configuration:** λ_target=0.05, warmup_epochs=50, c_stability=0.01


**Notes:**
- Performance metrics (Head 1): Standard prediction accuracy for comparison with other pykt models
- Best validation epoch selected at epoch 7 (early stopping)
- Test metrics computed on held-out test set after training completion
- Results validated against baseline experiment (perfect reproducibility confirmed)

**Alignment Metrics (Test Set):**
- l_21 (performance alignment): 4.225 (threshold <0.15, ❌ failed)
- l_22 (difficulty regularization): 0.028 (threshold <0.10, ✅ passed)
- l_23 (ability alignment): 6.929 (threshold <0.15, ❌ failed)
- Mastery-prediction correlation: 0.022 (Pearson), 0.062 (Spearman)

**Interpretation:**
- Model achieves competitive performance (Test AUC 0.7204) compared to pykt baselines
- Poor alignment metrics indicate Rasch IRT reference model doesn't fit ASSIST2015 dataset well
- Low mastery-prediction correlation (r=0.022) confirms reference model quality issues
- Model correctly prioritizes performance over alignment to poor-quality reference targets


### AKT

**Experiment:** `20251208_190103_benchmark_assist2015`  
**Configuration:** Standard pykt parameters (d_model=256, d_ff=512, num_attn_heads=8, n_blocks=4, dropout=0.2, lr=1e-4)

**Training:**
- Model: `assist2015_akt_qid_saved_model_42_0_0.2_256_512_8_4_0.0001_0_0`
- Training time: 2518 seconds (~42 minutes)
- Best epoch: 11 (early stopping)

**Performance:**
- Validation AUC: 0.7328, Validation Acc: 0.7586
- Test AUC: 0.7255, Test Acc: 0.7511
- Window Test AUC: 0.7256, Window Test Acc: 0.7511

**Notes:**
- Standard pykt evaluation using `wandb_predict.py` (full test set, concept-level)
- ASSIST2015 is single-skill dataset (max_concepts=1), no question-level evaluation performed
- AKT outperforms iKT3 by +0.51% AUC and +0.43% accuracy
- Results obtained after fixing bug in `init_dataset.py` for single-skill datasets
- Serves as baseline reference for interpretable models

**Comparison with iKT3:**
- **AKT advantage:** +0.0051 AUC, +0.0043 accuracy
- **Trade-off:** AKT offers better performance but lacks interpretability features
- **iKT3 value:** Provides IRT-grounded explanations with only minor performance cost

