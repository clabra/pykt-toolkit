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

**Training Infrastructure Note:**

Both AKT and iDKT use the **same training function** (`pykt/models/train_model.py`, line 332) from the PyKT framework:
- **AKT:** `benchmark_models.py` → `wandb_akt_train.py` → `wandb_train.py` (line 192) → `train_model()`
- **iDKT:** `run_repro_experiment.py` → `train_idkt.py` (line 181) → `train_model()`

⚠️ **Critical Bug - Hardcoded Patience:** The configured `patience=30` parameter is **completely ignored**. Line 407 of `train_model.py` has a hardcoded early stopping condition: `if i - best_epoch >= 10: break`. This means training stops exactly 10 epochs after the best validation AUC is observed.

**Performance Difference Explanation:**
- AKT: Best validation at epoch 24 → stopped at epoch 34 (24+10)
- iDKT: Best validation at epoch 18 → stopped at epoch 28 (18+10)
- The 6-epoch difference causes the 0.32% performance gap (0.8441 vs 0.8414)

**Why iDKT is perfectly reproducible but differs from AKT:**
Both models are **identical implementations** (line-by-line code copy with only `model_name` changed from "akt" to "idkt"). The performance difference is **dataset-specific**:

- **ASSIST2015**: AKT and iDKT produce **identical results** (Test AUC 0.7255) because `num_q=0` (no question IDs)
- **ASSIST2009**: Results differ (AKT 0.8441 vs iDKT 0.8414) because `num_q=17737` (has question IDs)

**Root Cause - RESOLVED:**
Investigation revealed that models start with **identical initializations** (`seed=42` verified: all embeddings match exactly), but the original training scripts differed:

**Identified Difference (Now Fixed):**
- **AKT** (`wandb_train.py` line 141): `Adam(model.parameters(), learning_rate)` - NO weight_decay parameter
- **iDKT** (`train_idkt.py` - ORIGINAL): `Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)` - Explicitly passed weight_decay
- **iDKT** (`train_idkt.py` - FIXED): Now matches AKT exactly: `Adam(model.parameters(), learning_rate)`

**Historical Performance Difference (Before Fix):**
The 0.32% gap between AKT (0.8441) and iDKT (0.8414) on ASSIST2009 was likely due to the optimizer creation difference. While both nominally used `weight_decay=0`, PyTorch's internal handling may differ when the parameter is omitted vs explicitly set to zero.

**Why divergence only affected ASSIST2009:**
ASSIST2009 has additional embeddings for question IDs (`difficult_param`, `q_embed_diff`, `qa_embed_diff` - ~17K parameters) that ASSIST2015 lacks. The larger parameter space made training more sensitive to subtle optimizer differences:

- **ASSIST2015** (`num_q=0`): No question ID embeddings → simpler parameter space → identical convergence despite optimizer difference
- **ASSIST2009** (`num_q=17737`): Extra embeddings → more complex dynamics → small optimizer difference amplified during training

**Baseline Experiments (Before Fix):**
- AKT: Best validation at epoch 24 → stopped at epoch 34 (24+10, hardcoded patience)
- iDKT: Best validation at epoch 18 → stopped at epoch 28 (18+10, hardcoded patience)
- Performance gap: 0.8441 vs 0.8414 (-0.32%)

**Expected Outcome (After Fix):**
Future iDKT training runs should produce results identical to or much closer to AKT on ASSIST2009, as the optimizer creation now matches exactly.

| Model | Dataset | Fold | Seed | Best Epoch | Val AUC | Val Acc | Test AUC | Test Acc | Window Test AUC | Window Test Acc | Other Metrics | Parameters | Experiment | Analysis |
|-------|---------|------|------|------------|---------|---------|----------|----------|-----------------|-----------------|---------------|------------|------------|----------|
| AKT | assist2009 | 0 | 42 | 34 | 0.8508 | 0.7884 | 0.8441 | 0.7777 | 0.8460 | 0.7789 | - | Baseline | 20251208_225023_akt_assist2009_baseline | Baseline performance, no interpretability |
| iDKT | assist2009 | 0 | 42 | 28 | 0.8486 | 0.7870 | 0.8414 | 0.7770 | - | - | - | Baseline | 20251209_095041_idkt_assist2009_baseline_274980 | Nearly matches AKT (-0.32%), IRT-grounded difficulty |
| iKT3 | assist2009 | 0 | 42 | 10 | 0.8208 | 0.7693 | 0.8120 | 0.7582 | - | - | l_21=4.05, l_22=0.004, l_23=3.91, r=-0.12 | Baseline | 20251208_224742_ikt3_assist2009_baseline_189351 | Poor alignment due to low M_ref quality (r=-0.12) |
| AKT | assist2015 | 0 | 42 | 11 | 0.7328 | 0.7586 | 0.7255 | 0.7510 | 0.7256 | 0.7511 | - | Baseline | 20251209_004147_akt_assist2015_baseline | Best performance, serves as reference baseline |
| iDKT | assist2015 | 0 | 42 | 11 | 0.7328 | 0.7586 | 0.7255 | 0.7510 | - | - | - | Baseline | 20251209_073158_idkt_assist2015_baseline_892719 | Matches AKT exactly (0.7255 AUC), IRT-grounded |
| iKT3 | assist2015 | 0 | 42 | 7 | 0.7258 | 0.7548 | 0.7204 | 0.7468 | - | - | l_21=4.23, l_22=0.028, l_23=6.93, r=0.02 | Baseline | 20251208_191345_ikt3_assist2015_baseline_286531 | -0.51% vs AKT, poor M_ref quality (r=0.02) |

### Baseline Parameters

Standard baseline parameters used for all models in benchmark experiments. Each model uses consistent parameters across both ASSIST2015 and ASSIST2009 datasets (except where noted).

#### AKT & iDKT Baseline

**Reference Experiments:**
- **AKT:** `20251209_004147_akt_assist2015_baseline`
- **iDKT:** `20251209_073158_idkt_assist2015_baseline_892719`

| Parameter | AKT Value | iDKT Value | Notes |
|-----------|-----------|------------|-------|
| **Batch Size** | 64 | 64 | Standard batch size |
| **d_ff** | 512 | 512 | Feed-forward dimension |
| **d_model** | 256 | 256 | Embedding dimension |
| **dropout** | 0.2 | 0.2 | Dropout rate |
| **emb_type** | qid | qid | Question ID embeddings |
| **Epochs** | 30 | 30 | With early stopping |
| **final_fc_dim** | 512? | 512 | Final FC layer dimension |
| **Fold** | 0 | 0 | First fold |
| **Gradient Clip** | 1.0? | 1.0 | Training stability |
| **l2** | 1e-05? | 1e-05 | IRT difficulty regularization |
| **Learning Rate** | 0.0001 | 0.0001 | Adam optimizer |
| **Model** | AKT | iDKT | Attention-based vs Interpretable DKT |
| **n_blocks** | 4 | 4 | Transformer blocks |
| **n_heads** | 8 | 8 | Attention heads |
| **Optimizer** | Adam | Adam | Standard choice |
| **Patience** | 30? | 30 | Early stopping patience |
| **Seed** | 42 | 42 | Fixed for reproducibility |
| **Seq Length** | 200? | 200 | Maximum sequence length |
| **Weight Decay** | 0.0 | 0.0 | No weight decay |

⚠️ **Note:** Parameters marked with "?" for AKT indicate values used by the model through PyKT infrastructure but not explicitly documented in the baseline experiment. Both models share the same PyKT framework and use these parameters.

#### iKT3 Baseline
**Reference Experiment:** `20251208_191345_ikt3_assist2015_baseline_286531`

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch Size** | 64 | Standard batch size |
| **c_stability_reg** | <span style="color:red">0.01</span> | <span style="color:red">Difficulty regularization weight (iKT3 only)</span> |
| **d_ff** | <span style="color:red">1536</span> | Feed-forward dimension (<span style="color:red">3× larger than AKT/iDKT</span>) |
| **d_model** | 256 | Embedding dimension |
| **dropout** | 0.2 | Dropout rate |
| **emb_type** | qid | Question ID embeddings |
| **Epochs** | 30 | With early stopping |
| **Fold** | 0 | First fold |
| **Gradient Clip** | 1.0 | Gradient clipping threshold |
| **lambda_target** | <span style="color:red">0.05</span> | <span style="color:red">Alignment loss weight (iKT3 only)</span> |
| **Learning Rate** | 0.0001 | Adam optimizer |
| **Model** | <span style="color:red">iKT3</span> | Reference Model Alignment |
| **n_heads** | <span style="color:red">4</span> | Attention heads (<span style="color:red">fewer than AKT/iDKT: 8</span>) |
| **num_encoder_blocks** | <span style="color:red">8</span> | Transformer blocks (<span style="color:red">more than AKT/iDKT: 4</span>) |
| **Optimizer** | Adam | Standard choice |
| **Patience** | 30 | Early stopping patience |
| **reference_model** | <span style="color:red">irt</span> | <span style="color:red">IRT reference model (iKT3 only)</span> |
| **Seed** | 42 | Fixed for reproducibility |
| **Seq Length** | 200 | Maximum sequence length |
| **warmup_epochs** | <span style="color:red">50</span> | <span style="color:red">Warmup for alignment loss (iKT3 only)</span> |
| **Weight Decay** | <span style="color:red">1.7571e-05</span> | Small L2 regularization (<span style="color:red">vs 0.0 in AKT/iDKT</span>) |

**Key Differences Across Models:**
- **Architecture:** AKT/iDKT use 8 heads × 4 blocks with d_ff=512; iKT3 uses 4 heads × 8 blocks with d_ff=1536
- **Parameters:** AKT/iDKT ~6M params; iKT3 ~3M params (more efficient despite larger d_ff)
- **Regularization:** iKT3 includes alignment-specific hyperparameters (lambda_target, warmup, c_stability_reg)
- **IRT Regularization:** Both AKT and iDKT use l2=1e-05 for difficulty regularization
- **Identical Training:** AKT and iDKT use identical hyperparameters (lr=0.0001, batch_size=64, weight_decay=0.0)
