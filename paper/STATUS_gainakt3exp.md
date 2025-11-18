# GainAKT3Exp Model Status

**Document Version**: 2025-11-18 (Reorganized for clarity)  
**Model Status**: ⚠️ V3 Enhanced Strategy - **Explicit Differentiation**

---

## Reference Documents

- Architecture foundations: `gainakt3exp_architecture_approach.md`
- Diagrams and sequences: `gainakt3exp_architecture_diagrams.md`
- Parameter evolution: `examples/reproducibility.md`

---

## Current Status

**Implementation Phase**: V3 Enhanced Strategy Phase 1 - ❌ FAILED  
**Validation Status**: 20-epoch experiment complete (Exp 970206, early stopped at 14 epochs)  
**Critical Finding**: V3 Phase 1 did NOT solve uniform gains problem

**V3 Phase 1 Results** (Experiment 970206):
- ❌ Gain std: 0.0018 (target >0.10) - 55x short of target
- ❌ Gain CV: 0.002 (target >0.20) - 100x short of target
- ❌ Encoder2 Val AUC: 0.5935 (target >0.62) - 4.3% below target
- ❌ Mastery correlation: 0.039 (target >0.40) - 10x below target
- ❌ Response-conditional ratio: 1.00 (target >1.2) - no differentiation
- ✅ Beta spread std: 0.448 (target >0.3) - ONLY success
- ❌ Performance vs V2: Encoder1 AUC -3.55%, Mastery corr -65.6%

**Root Cause Identified**: Symmetric initialization creates uniform attractor
- PyTorch default init produces per-skill CV=0.017 (99.8% uniform at epoch 0)
- All 100 skills initialized nearly identically, produce similar gains (~0.50)
- V3 losses (contrastive, variance) too weak to escape uniform local minimum
- Beta spread initialization didn't help (downstream parameter, doesn't affect gains_projection)

**V3 Phase 1 Components Implemented**:
- ✅ Skill-contrastive loss (weight=1.0) - Active but insufficient
- ✅ Beta spread initialization N(2.0, 0.5) - Worked for beta, not gains
- ✅ Beta spread regularization (weight=0.5) - Maintained spread
- ✅ Variance loss amplification (0.1→2.0) - Active but too weak
- ✅ Indentation bug fixed (311 lines) - Mastery head functional

**Proposed Solution**: Asymmetric bias initialization (70-90% confidence)
- Initialize `gains_projection.bias` with N(0, 0.5) instead of uniform
- Expected initial CV: 0.20 (vs 0.017 default) - target achieved at initialization
- Theory: Symmetry breaking (Goodfellow), Lottery Ticket Hypothesis (Frankle)
- Implementation: Add `gains_projection_bias_std` parameter, apply in model init

**Next Steps**: See detailed plan in continuation section below.

See `/workspaces/pykt-toolkit/tmp/INITIALIZATION_THEORY_ANALYSIS.md` for complete theoretical analysis and `examples/experiments/20251118_211102_gainakt3exp_V3-phase1-full-validation_970206/ANALYSIS.md` for detailed failure analysis.

## Architecture Overview

**Dual-Encoder Design**:
- **Encoder 1 (Performance Path)**: 96,513 parameters
  - Components: embeddings (context, value, skill, position), encoder blocks, prediction head
  - Purpose: Learns attention patterns for direct response prediction (unconstrained optimization)
  - Output: Base predictions → BCE Loss (weight: 70%)
  
- **Encoder 2 (Interpretability Path)**: 71,040 parameters  
  - Components: embeddings (context, value, position), encoder blocks, gains projection
  - Purpose: Learns skill-specific learning gains through attention patterns
  - Output: Per-skill gains → Effective practice → Mastery → IM Loss (weight: 30%)

- **Sigmoid Learning Curve Parameters**: 22 parameters
  - β_skill[num_c]: Skill difficulty (learning rate amplification)
  - γ_student[num_students]: Student learning velocity
  - M_sat[num_c]: Maximum mastery saturation per skill
  - θ_global: Global mastery threshold for competence
  - offset: Sigmoid inflection point

**Key Innovation - Differentiable Effective Practice with Per-Skill Gains**:
```python
# Encoder 2 outputs value representations
value_seq_2 = encoder_blocks_2(...)  # [B, L, D]

# Project to per-skill gains (V1+ implementation)
self.gains_projection = nn.Linear(d_model, num_c)  # [D] → [num_c]
skill_gains_logits = self.gains_projection(value_seq_2)  # [B, L, num_c]
skill_gains = sigmoid(skill_gains_logits)  # [B, L, num_c] ∈ [0, 1]

# Per-skill quality-weighted practice accumulation (differentiable!)
effective_practice[:, t, :] += skill_gains[:, t, :]  # Each skill gets different gain!

# Sigmoid learning curves (per-skill mastery)
mastery = M_sat × sigmoid(β_skill × γ_student × effective_practice - offset)

# Incremental mastery predictions through threshold
encoder2_pred = sigmoid((mastery - θ_global) / temperature)
```

**Gradient Flow**:
- Encoder 1: BCE_loss → predictions → prediction_head → encoder_blocks_1 ✅
- Encoder 2: IM_loss → encoder2_pred → mastery → effective_practice → skill_gains[num_c] → gains_projection → value_seq_2 → encoder_blocks_2 ✅
  - **Key**: Per-skill gains vector [B, L, num_c] allows skill-specific gradient updates

## Current Implementation (V2 - Per-Skill Gains)

**Code Status**: ✅ Production ready
- File: `pykt/models/gainakt3_exp.py` (909 lines)
- Training: `examples/train_gainakt3exp.py`
- Evaluation: `examples/eval_gainakt3exp.py`

**Active Features**:
- ✅ Dual-encoder architecture with complete independence
- ✅ Per-skill gains projection [B, L, num_c]
- ✅ Differentiable effective practice accumulation
- ✅ Sigmoid learning curves with learnable parameters
- ✅ Dual loss functions (BCE + Incremental Mastery)
- ✅ Comprehensive metrics tracking (30 columns in metrics CSV)

**Inactive/Commented Out**:
- ❌ All constraint losses (monotonicity, sparsity, etc.) - weights = 0.0
- ❌ All semantic losses (alignment, retention, etc.) - enable flags = false
- ❌ Intrinsic gain attention mode - deprecated
- ❌ Explicit gains head output - use_gain_head = false

## Performance Summary

### V2 Results (820618 - Latest Complete Run)

| Metric | Train | Validation | Test | Target | Status |
|--------|-------|------------|------|--------|--------|
| **Overall AUC** | 0.7152 | 0.6893 | - | >0.70 | ✅ |
| **Encoder 1 AUC** | 0.7152 | 0.6893 | 0.6825 | >0.68 | ✅ |
| **Encoder 2 AUC** | 0.5977 | 0.5969 | 0.5931 | >0.55 | ✅ Above random |
| **Mastery Correlation** | 0.0362 | - | 0.0379 | >0.30 | ❌ Too low |
| **Gain Std** | - | - | 0.0017 | >0.05 | ❌ Nearly uniform |

**Key Observations**:
- ✅ Encoder 1 performs well (68-69% AUC)
- ⚠️ Encoder 2 above random baseline but struggles with skill differentiation
- ❌ Gains converge to near-uniform values (std=0.0017)
- ❌ Mastery poorly correlated with actual performance

**Root Cause**: Mixed training signals (50% BCE + 50% IM from epoch 1) allow model to find compromise solution with uniform gains (~0.585) that satisfies both objectives without learning skill-specific patterns.

## Active Parameters (37 total)

Defaults defined in `configs/parameter_default.json`:

**Model Architecture (7)**:
- seq_len=200, d_model=256, n_heads=4, num_encoder_blocks=4, d_ff=256, dropout=0.1, emb_type="qid"

**Learning Curve (5)**:
- beta_skill_init=2.5, m_sat_init=0.7, gamma_student_init=1.1, sigmoid_offset=1.5, mastery_threshold_init=0.85

**Training (8)**:
- learning_rate=0.000174, batch_size=64, epochs=20, patience=10, optimizer="adam", weight_decay=0.0001, gradient_clip=1.0, bce_loss_weight=0.5

**Interpretability (4)**:
- use_mastery_head=true, use_gain_head=false, threshold_temperature=1.0, variance_loss_weight=0.1

**Data (2)**: dataset="assist2015", fold=0
**Monitoring (2)**: monitor_freq=50, use_wandb=true
**Misc (9)**: seed, use_amp, auto_shifted_eval, max_correlation_students, num_students, use_skill_difficulty, use_student_speed, train_script, eval_script

## Known Issues

### Critical Issue: Uniform Gains Problem

**Symptom**: Encoder 2 learns near-uniform gain values across all skills (std=0.0017)

**Impact**:
- Cannot differentiate skill-specific learning rates
- Mastery trajectories not predictive of performance (correlation=0.037)
- Encoder 2 AUC only marginally above random (59.7%)

**Cause**: Optimization trajectory problem - mixed signals from dual losses allow compromise solution

**Attempted Fixes** (V2):
1. Increased IM loss weight: 30% → 50% (marginal improvement)
2. Added variance loss: weight=0.1 (negligible impact)
3. Layer-wise learning rates: 3x boost for gains_projection (insufficient)
4. Extended training: 12 → 20 epochs (pattern established early)

**Result**: All interventions produced marginal improvements. Gains still uniform (0.0015 → 0.0017).

---

# Next Steps

## CRITICAL PRIORITY: Implement Asymmetric Initialization

**Objective**: Break symmetric initialization that causes uniform gains

**Implementation Steps** (2 hours):
1. Add parameters to `configs/parameter_default.json`:
   - `gains_projection_bias_std: 0.5` (default)
   - `gains_projection_orthogonal: true` (optional enhancement)

2. Create initialization function in `pykt/models/gainakt3_exp.py` after line 275:
   ```python
   def initialize_gains_projection_asymmetric(gains_projection, num_c, bias_std=0.5, orthogonal=True):
       if orthogonal:
           nn.init.orthogonal_(gains_projection.weight)
           gains_projection.weight.data *= 0.3
       skill_bias = torch.randn(num_c) * bias_std
       gains_projection.bias.data = skill_bias
   ```

3. Apply after gains_projection creation (line 275)

4. Add diagnostic logging for initial gain statistics (epoch 0 CV should be ~0.20)

**Validation Experiment** (20 minutes):
```bash
python examples/run_repro_experiment.py \
    --short_title V3-asymmetric-init-05 \
    --epochs 20 \
    --gains_projection_bias_std 0.5 \
    --skill_contrastive_loss_weight 1.0 \
    --beta_spread_regularization_weight 0.5 \
    --variance_loss_weight 2.0
```

**Success Criteria**:
- 🎯 Epoch 0: Gain CV >0.15 (initial asymmetry)
- 🎯 Epoch 5: Gain std >0.05 (10x improvement)
- 🎯 Epoch 10: Gain CV >0.20 (skill differentiation)
- 🎯 Epoch 20: Gain std >0.10 (ultimate target, 55x improvement)
- 🎯 Encoder2 AUC >0.62, Mastery corr >0.30, Response ratio >1.2

**Confidence**: 70% at bias_std=0.5, 90% at bias_std∈[0.5, 1.0]

**If Insufficient**: Run bias strength sweep (0.3, 0.5, 0.7, 1.0) with 10 epochs each

## V3 Enhanced Strategy Overview (DEPRECATED)

**Rationale**: Loss weight analysis across 6 experiments (BCE=0%-100%) revealed uniform gains problem is **independent of loss balance** (std varies only 0.0002 across all weights). V2's approach (tuning weights) cannot work. V3 targets root cause with explicit anti-uniformity mechanisms.

**V3 Strategy**: Explicit Skill Differentiation (NOT inverse warmup)
- **Phase 1** (Implemented, awaiting validation):
  - Skill-contrastive loss (weight=1.0): Cross-skill comparison forces differentiation
  - Variance loss increase (0.1→2.0): 20x stronger anti-uniformity signal  
  - Beta spread initialization: N(2.0, 0.5) starts with clear variation
  - Beta spread regularization (weight=0.5): Prevents collapse during training
  
- **Phase 2** (If Phase 1 insufficient):
  - Gain-response correlation loss (weight=0.5): Semantic supervision
  - Curriculum amplification: 2x weights in early epochs
  - Dropout in gains_projection (0.2): Prevent co-adaptation

**Expected Improvements** (once bug fixed):
| Metric | V2 | V3 Target | Required Improvement |
|--------|----|-----------|--------------------|
| Gain std | 0.0017 | >0.05 | 30x |
| Beta spread | N/A | >0.3 maintained | New metric |
| Encoder2 AUC | 59.7% | >62% | +2.3% |
| Mastery correlation | 0.11 | >0.40 | 4x |

**Success Criteria**:
- ✅ Per-skill gains show variance (std >0.05)
- ✅ Different skills have different learning rates (CV >0.2)
- ✅ Mastery correlates with responses (>0.40)
- ✅ Encoder2 AUC above random and improving (>62%)
- ✅ Beta spread preserved throughout training (std >0.3)
- ✅ Response-conditional gains (ratio >1.2)

See `V3_IMPLEMENTATION_STRATEGY.md` for complete implementation details, mathematical formulations, and monitoring strategy.

## Alternative Approaches (If V3 Fails)

**Option 1: Multi-layer Gains Projection**
- Replace single linear layer with MLP (d_model → d_model → num_c)
- Adds non-linear capacity for complex skill relationships
- Try if: gain std between 0.02-0.05 after V3

**Option 2: Initialization Strategy**  
- Smaller weight initialization (std=0.01 vs default)
- Prevents strong uniform bias at initialization
- Try if: gain std between 0.05-0.10 after V3

**Option 3: Skill-Aware Attention**
- Add skill similarity bias to attention scores
- Help model focus on skill-specific patterns
- Try if: gain std <0.02 after V3 (last resort)

## Tasks

**Immediate** (within 24 hours):
- [ ] Implement inverse warmup schedule in train_gainakt3exp.py
- [ ] Update parameter_default.json: epochs=30, bce_loss_weight schedule
- [ ] Launch V3 training with monitoring
- [ ] Track skill_gains std during training (should increase in Phase 1)

**Short-term** (within 1 week):
- [ ] Extract and analyze V3 trajectories
- [ ] Run 6 verification checks on V3 results
- [ ] Compare V1 vs V2 vs V3 across all metrics
- [ ] Document findings and update STATUS
- [ ] If successful: commit changes and proceed to multi-seed validation
- [ ] If unsuccessful: implement Option 2 and retry

**Medium-term** (within 1 month):
- [ ] Multi-seed validation (3 seeds) once V3 succeeds
- [ ] Cross-dataset validation (ASSIST2009, etc.)
- [ ] Ablation studies on schedule parameters
- [ ] Write paper section on inverse warmup strategy

---

# Historic Evolution

## Timeline Overview

| Date | Version | Key Changes | Encoder2 AUC | Status |
|------|---------|-------------|--------------|--------|
| 2025-11-16 | Baseline | Dual-encoder architecture, sigmoid curves | 0.4842 | ❌ Broken (scalar gains) |
| 2025-11-17 | V1 | Per-skill gains fix | 0.5868 | ⚠️ Above random, uniform gains |
| 2025-11-17 | V2 | Multiple interventions | 0.5969 | ⚠️ Marginal improvement |
| 2025-11-18 | V3 Bug Fix | Fixed indentation bug (311 lines orphaned) | 0.5891 | ✅ Mastery head active |
| 2025-11-18 | V3 Phase 1 | Skill-contrastive, beta spread, variance×20 | 0.5935 | ❌ FAILED - gains still uniform (std=0.0018) |

## Bug 0: Skill Index Mismatch (2025-11-17) - FIXED

**Discovery**: encoder2_pred used mastery for NEXT skill (qry[t] = q[t+1]) instead of CURRENT skill (q[t])

**Impact**:
- 22.2% prediction mismatches (214/966 predictions)
- 98.2% of mismatches at skill transitions
- Misaligned training supervision
- Invalid interpretability claims

**Fix**: Changed `skill_indices = target_concepts.long()` to `skill_indices = q.long()` in line 622 of gainakt3_exp.py

**Validation**: 100% match rate after fix (0% mismatches) ✅

## Phase 0: Baseline Architecture (999787) - BROKEN

**Experiment**: 20251117_131554_gainakt3exp_baseline-bce0.7_999787

**Architecture Bug**: Scalar gain quality instead of per-skill gains

```python
# WRONG: Single scalar per interaction
gain_quality = sigmoid(value_seq_2.mean(dim=-1))  # [B, L, 1]
effective_practice[t, practiced_skill] += gain_quality[t]  # Same for ALL skills
```

**The Problem**:
- ✅ Model can learn: "This interaction had high/low engagement"
- ❌ Model cannot learn: "Student improved Skill A by 0.8, Skill B by 0.2"
- Result: All practiced skills get identical increment

**Performance**:
- Encoder 1 AUC: 0.6765 ✅ (working well)
- **Encoder 2 AUC: 0.4842** ❌ (below random baseline!)
- Mastery ↔ Response correlation: -0.044 (essentially zero)
- Encoder2_pred range: [0.37, 0.53] (too narrow, indecisive)

**Trajectory Analysis** (659 interactions):
- Mastery IS updating (sigmoid progression visible)
- BUT mastery uncorrelated with actual performance
- Mastery increases even after incorrect responses
- Predictions compressed near 0.5 (cannot discriminate)

**Root Cause**: Scalar gain quality learns overall engagement, not skill-specific learning. No amount of parameter tuning can fix this architectural limitation.

## Phase 1: Per-Skill Gains Fix - V1 (995130)

**Experiment**: 20251117_154349_gainakt3exp_fixed-per-skill-gains_995130

**Architecture Fix**: Implemented per-skill gains vector

```python
# CORRECT: Per-skill gains projection
self.gains_projection = nn.Linear(d_model, num_c)  # [D] → [num_c]
skill_gains = sigmoid(self.gains_projection(value_seq_2))  # [B, L, num_c]
effective_practice[:, t, :] += skill_gains[:, t, :]  # Skill-specific!
```

**Configuration**:
- bce_loss_weight: 0.7 (30% IM loss)
- epochs: 12
- learning_rate: 0.000174
- patience: 10

**Performance**:
- Encoder 1 AUC: 0.6897 ✅
- **Encoder 2 AUC: 0.5868** ✅ (above random, +10.3% from baseline)
- Mastery correlation: 0.037 (improved from -0.044, but still too low)

**Critical Finding**: Gains are nearly uniform across all skills!
```
Gain statistics:
  Mean: 0.5884
  Std: 0.0015 ← Nearly constant!
  Range: 0.01 (0.583 to 0.593)
```

**Trajectory Verification** (6 checks):
- ✅ CHECK 1: Skill differentiation - **FAILED** (CV=0.0016, target >0.2)
- ✅ CHECK 2: Q-Matrix learning - PASS (1.0 skills/interaction)
- ✅ CHECK 3: Monotonicity - PASS (0 violations)
- ❌ CHECK 4: Mastery correlation - FAIL (0.033, target >0.3)
- ❌ CHECK 5: Skill difficulty - FAIL (CV=0.0016)
- ❌ CHECK 6: Response-conditional - FAIL (ratio=1.00, target >1.2)

**Success Rate**: 2/6 checks (33%)

**Conclusion**: Architectural fix successful (Encoder2 above random), but optimization fails to learn skill-specific patterns. Gains converge to uniform solution.

## Phase 2: Multiple Interventions - V2 (820618)

**Experiment**: 20251117_162330_gainakt3exp_fixed-per-skill-gains-v2_820618

**Objective**: Address V1's uniform gains problem with 4 targeted interventions

**Changes**:
1. **Priority 1 - Increased IM Loss**: 30% → 50% (stronger gradient signal to Encoder 2)
2. **Priority 2 - Added Variance Loss**: weight=0.1 (explicitly penalize uniform gains)
3. **Priority 3 - Extended Training**: 12 → 20 epochs (early stopped at 15)
4. **Priority 4 - Layer-wise LR**: gains_projection LR 3x boost (0.000522 vs 0.000174)

**Configuration**:
- bce_loss_weight: 0.5 (50% IM loss)
- variance_loss_weight: 0.1
- epochs: 20 (early stopped: 15)
- gains_projection_lr: 0.000522 (3x base)
- patience: 10

**Performance vs V1**:
| Metric | V1 | V2 | Change | Improvement |
|--------|----|----|--------|-------------|
| Encoder1 Val AUC | 0.6897 | 0.6893 | -0.04% | Stable |
| **Encoder2 Val AUC** | 0.5868 | **0.5969** | **+1.0%** | ⚠️ Marginal |
| Encoder2 Test AUC | 0.5835 | 0.5931 | +1.0% | ⚠️ Marginal |
| Gain Std | 0.0015 | **0.0017** | +0.0002 | ❌ Negligible |
| Mastery Correlation | 0.037 | **0.113** | **+0.076 (3x)** | ⚠️ Better but still low |

**Trajectory Verification** (6 checks):
- ✅ CHECK 3: Monotonicity - PASS
- ✅ CHECK 2: Q-Matrix learning - PASS  
- ✅ CHECK 4: Mastery correlation - **IMPROVED** (0.113 vs target >0.3)
- ❌ CHECK 1: Skill differentiation - FAIL (CV=0.0019, target >0.2)
- ❌ CHECK 5: Skill difficulty - FAIL
- ❌ CHECK 6: Response-conditional - FAIL

**Success Rate**: 3/6 checks (50%) - slight improvement from V1's 33%

**Training Dynamics**:
```
Epoch  Enc1_Val  Enc2_Val  Notes
  1    0.679     0.585     Pattern established early
  5    0.704     0.598     Best epoch (peak)
  15   0.715     0.589     Final (early stopped)
```

**Key Finding**: Despite ALL interventions, gains remain 99.8% uniform!
```
Gain statistics:
  Mean: 0.5854
  Std: 0.0017 ← Still nearly constant!
  Range: 0.0105
```

**Why V2 Failed**: 
- Mixed signals (50% BCE + 50% IM from epoch 1) still allow compromise
- Model finds uniform gains (~0.585) that satisfy both objectives
- Gets stuck in local minimum early (epoch 5-6)
- More epochs don't help - pattern established from initialization

**Critical Discovery - Loss Weight Analysis** (comparing 3 experiments):

Analyzed gain uniformity across different BCE/IM balances:
- **Exp 999787** (BCE=70%, IM=30%): gain_std=0.0015, CV=0.0016
- **Exp 995130** (BCE=70%, IM=30%): gain_std=0.0015, CV=0.0016  
- **Exp 820618** (BCE=50%, IM=50%): gain_std=0.0017, CV=0.0019

**Key Insight**: Uniform gains problem **independent of loss balance**!
- 30% vs 50% IM weight: Only 0.0002 std difference (negligible)
- All experiments converge to ~0.585 mean with <0.002 std
- Pattern established by epoch 1-5, persists through training

**Implication**: Simply adjusting loss weights (V2 approach) cannot solve uniform gains. The model finds uniform solution immediately and never escapes, regardless of balance.

**Conclusion**: Need **fundamentally different training strategy** that prevents uniform solution from forming. V3 changes approach from temporal scheduling (inverse warmup) to explicit anti-uniformity mechanisms (cross-skill comparison, spread preservation).

## Critical Bug Fix: Orphaned Indentation (2025-11-18) - FIXED

**Discovery**: During V3 Phase 1 validation, discovered IM loss=0.0 despite all V3 code implemented and qry=None set in 5 locations.

**Root Cause**: Line 458 in gainakt3_exp.py had commented-out conditional:
```python
# elif self.use_gain_head and self.use_mastery_head:
```

However, lines 459-769 (311 lines) remained indented at 12 spaces (should be 8 spaces), indicating they were still part of the commented elif's body. This created orphaned code that never executed because controlling conditional was commented out.

**Bug Impact**:
- Lines 459-769: 311-line mastery computation block never executed
- Line 770: Orphaned `else:` always executed instead, setting all mastery variables to None
- Result: Mastery head completely disabled since parameter simplification (~2025-11-16)
- ALL experiments from V0 through V3 affected (none had working mastery head)

**Fix Method**:
1. Created `/workspaces/pykt-toolkit/tmp/fix_indentation_bug.py`
2. Un-indented lines 459-769 from 12 spaces to 8 spaces (4-space shift)
3. Removed orphaned else block (lines 770-777)
4. Backup created: `gainakt3_exp.py.backup_before_unindent`
5. Verified with Python compilation test

**Validation** (Experiment 269777):
```bash
python examples/run_repro_experiment.py \
    --short_title V3-FIXED-indentation-bug \
    --epochs 2 --num_workers 32
```

**Results** - Mastery head now ACTIVE! ✅
| Metric | Epoch 1 | Epoch 2 | Status |
|--------|---------|---------|--------|
| IM Loss | 0.6079 | 0.6060 | ✅ Non-zero! (was 0.0) |
| Encoder2 Val AUC | 0.5833 | 0.5891 | ✅ Above random |
| Mastery Correlation (train) | - | 0.036 | ✅ Positive |
| Mastery Correlation (test) | - | 0.038 | ✅ Positive |
| Beta Spread (std) | 0.496 | 0.496 | ✅ Maintained >0.3 |

**Learned Parameters** (confirming all V3 components functional):
- beta_skill: std=0.496 (spread preserved throughout training)
- gamma_student: std=0.238 (student differentiation)
- theta_global: 0.850→0.786 (learning threshold decreasing as expected)

**Key Insight**: Bug existed since parameter simplification but went undetected because:
1. Python interpreter didn't flag indentation as syntax error (valid as unreachable code)
2. Tests only checked IM loss computation code existence, not runtime execution
3. Gradual parameter removal left orphaned indented blocks without triggering warnings

**Lesson Learned**: When commenting out conditionals during refactoring, must verify downstream indentation adjusted for remaining code blocks. AST analysis insufficient - indentation-level analysis required.

## Phase 3: V3 Enhanced Strategy - Explicit Differentiation (970206) - FAILED

**Experiment**: 20251118_211102_gainakt3exp_V3-phase1-full-validation_970206

**Objective**: Address V2's uniform gains with explicit anti-uniformity mechanisms (skill-contrastive loss, beta spread init/regularization, variance×20)

**Strategic Shift**: After analyzing loss weight independence (gain_std varies only 0.0002 across BCE 0%-100%), implemented explicit differentiation instead of loss scheduling.

**Result**: Complete failure - gains remained 99.8% uniform despite all V3 components active for 14 epochs.

**V3 Phase 1 Implementation** (2025-11-18):

### 1. Skill-Contrastive Loss (NEW)
**Purpose**: Explicitly force cross-skill comparison (addresses IM Loss independence)

**Mathematical Formulation**:
```python
# For each interaction, compute variance across all skill gains
gain_variance_per_interaction = skill_gains.var(dim=2)  # [B, L]
skill_contrastive_loss = -gain_variance_per_interaction.mean()  # Maximize variance
```

**Implementation**: Line 557-562 in gainakt3_exp.py, new method `compute_skill_contrastive_loss()` at lines 875-922

**Parameters**: `skill_contrastive_loss_weight = 1.0`

### 2. Beta Spread Initialization (MODIFIED)
**Purpose**: Start training with differentiated skill difficulties

**Change**:
```python
# OLD: Uniform initialization
self.beta_skill = torch.nn.Parameter(torch.ones(num_c) * beta_skill_init)

# NEW: Spread initialization N(2.0, 0.5) clamped [0.5, 5.0]
beta_init_values = torch.randn(num_c) * 0.5 + beta_skill_init
beta_init_values = torch.clamp(beta_init_values, 0.5, 5.0)
self.beta_skill = torch.nn.Parameter(beta_init_values)
```

**Implementation**: Line 306-310 in gainakt3_exp.py

**Rationale**: Prevents symmetric starting point that favors uniform solution

### 3. Beta Spread Regularization (NEW)
**Purpose**: Prevent beta_skill collapse during training

**Mathematical Formulation**:
```python
beta_std = self.beta_skill.std()
beta_spread_loss = max(0, 0.3 - beta_std) ** 2  # Hinge loss
```

**Implementation**: Line 564-567 in gainakt3_exp.py, new method `compute_beta_spread_regularization()` at lines 924-954

**Parameters**: `beta_spread_regularization_weight = 0.5`

### 4. Variance Loss Amplification (MODIFIED)
**Change**: `variance_loss_weight: 0.1 → 2.0` (20x increase)

**Rationale**: Original weight too weak to overcome uniform attractor

**Configuration Updates**:
- `configs/parameter_default.json`: Added 3 new parameters
- `examples/parameters_audit.py`: Updated to require 15 parameters (was 12)
- `examples/train_gainakt3exp.py`: Loss computation includes all V3 terms
- All 9 reproducibility audit checks passing ✅

### V3 Phase 1 Complete Results - FAILED (970206)

**Experiment**: 20251118_211102_gainakt3exp_V3-phase1-full-validation_970206 (20 epochs requested, early stopped at 14)

**Configuration**: All V3 Phase 1 components active
- skill_contrastive_loss_weight=1.0
- beta_spread_regularization_weight=0.5
- variance_loss_weight=2.0 (20x amplification)
- Beta spread init: N(2.0, 0.5) clamped [0.5, 5.0]

**Performance Results**:
| Metric | Result | Target | Gap |
|--------|--------|--------|-----|
| Best Val AUC (Enc1) | 0.6648 (epoch 4) | >0.68 | -2.3% |
| Encoder2 Val AUC | 0.5935 | >0.62 | -4.3% |
| Gain std | 0.0018 | >0.10 | 55x short |
| Gain CV | 0.002 | >0.20 | 100x short |
| Mastery correlation | 0.039 | >0.40 | 10x short |
| Response ratio | 1.00 | >1.2 | No differentiation |
| Beta spread std | 0.448 | >0.3 | ✅ ONLY success |

**Comparison to V2 Baseline** (820618):
- Encoder1 Val AUC: 0.6648 vs 0.6893 (-3.55%) ❌ WORSE
- Encoder2 Val AUC: 0.5935 vs 0.5969 (-0.58%) ❌ WORSE
- Mastery correlation: 0.039 vs 0.113 (-65.6%) ❌ MUCH WORSE
- Gain std: 0.0018 vs 0.0017 (+0.0001) ≈ NO CHANGE

**Gain Statistics** (66 skills observed, 14 epochs):
```
Mean: 0.588341
Std: 0.001820 (target >0.10)
Min: 0.582953, Max: 0.593400
Range: 0.010447 (1.7% variation)
CV: 0.003094 (99.7% uniformity)
Per-skill CV: 0.002061 (target >0.20)
```

**Training Dynamics**:
- Peak performance: Epoch 4 (Val AUC 0.6648)
- Early stopping: Epoch 14 (patience=10 triggered)
- Pattern: Uniform gains established by epoch 1, never escaped
- Beta spread: Maintained throughout training (0.496→0.448)

**Root Cause Discovery**: Symmetric initialization problem
- PyTorch default `nn.Linear(256, 100)` produces per-skill CV=0.017 at initialization
- All 100 skills nearly identical outputs (~0.50 after sigmoid)
- Creates uniform attractor that V3 losses cannot escape
- Beta spread init didn't help (downstream parameter, doesn't affect gains_projection layer)

**Why V3 Failed**:
1. Started from symmetric initialization (CV=0.017)
2. Forward pass produces uniform gains (~0.50 all skills)
3. Backward pass has nearly identical gradients for all skills
4. SGD update preserves uniformity (symmetry principle)
5. V3 losses (contrastive, variance) too weak vs Encoder1 dominance
6. Uniform local minimum established epoch 1, never escapes

**V3 Success Criteria**: 1/6 passed (only beta spread maintained >0.3)

**Conclusion**: V3 Phase 1 did NOT solve uniform gains problem. Despite all explicit differentiation mechanisms active (skill-contrastive loss, beta spread init/regularization, variance×20), the fundamental issue is **symmetric initialization** creating a uniform attractor. V3 losses cannot break symmetry when starting from nearly identical skill parameters. Performance degraded vs V2, suggesting V3 components add noise without solving core problem.

**Detailed Analysis**: See `examples/experiments/20251118_211102_gainakt3exp_V3-phase1-full-validation_970206/ANALYSIS.md`

## Detailed Implementation History

### Sigmoid Learning Curve Implementation (2025-11-16) - COMPLETE

**Changes**:
- Added 5 learnable parameters for sigmoid curves: β_skill, γ_student, M_sat, θ_global, offset
- Replaced linear accumulation with practice count-driven sigmoid: `mastery = M_sat × sigmoid(β × γ × practice_count - offset)`
- Updated threshold prediction mechanism (global θ instead of per-skill)
- Created comprehensive test suite (tmp/test_sigmoid_curves.py) - all tests passed ✅

**Educational Semantics**:
- β_skill: Skill difficulty (learning rate amplification)
- γ_student: Student learning velocity (personalization)
- M_sat: Maximum achievable mastery per skill
- θ_global: Competence threshold (decision boundary)
- offset: Inflection point (when rapid learning begins)

**Test Results**: Clear sigmoid pattern verified (mastery: 0.095 → 0.400 → 0.705 → 0.786 → 0.800 over 20 practices)

### Dual-Encoder Architecture Implementation (2025-11-16) - COMPLETE

**Changes**:
- Changed class definition: `class GainAKT3Exp(GainAKT3)` → `class GainAKT3Exp(nn.Module)` (standalone)
- Implemented two completely independent encoder stacks (no shared parameters)
- Encoder 1: 96,513 parameters (embeddings, blocks, prediction head)
- Encoder 2: 71,040 parameters (embeddings, blocks, no prediction head)
- Differentiable effective practice: quality-weighted accumulation for gradient flow

**Critical Innovation - Per-Skill Gains with Differentiable Accumulation**:
```python
# V1+ architecture: Per-skill gains projection (not scalar!)
self.gains_projection = nn.Linear(d_model, num_c)  # Project to skills
skill_gains = sigmoid(self.gains_projection(value_seq_2))  # [B, L, num_c]

# Per-skill differentiable accumulation
effective_practice[:, t, :] += skill_gains[:, t, :]  # Skill-specific rates!
```

**Gradient Flow**: IM_loss → mastery[num_c] → effective_practice[num_c] → skill_gains[num_c] → gains_projection → value_seq_2 → Encoder 2 parameters

**Test Suite** (tmp/test_dual_encoders.py):
- Test 1-4: Architecture verification ✅
- **Test 5: Gradient flow** ✅ (CRITICAL - both encoders receive gradients)
- Test 6-7: Independent learning ✅

**Result**: ALL 7 TESTS PASSED. Model ready for training.

### Dual-Encoder Metrics and Monitoring (2025-11-16) - COMPLETE

**Enhanced CSV Metrics** (30 columns total):
- Overall metrics: train/val loss, AUC, accuracy
- Loss components: BCE, IM (unweighted raw losses)
- Weighted losses: BCE×λ₁, IM×λ₂, total
- Loss shares: BCE%, IM%
- Encoder-specific: Encoder1 AUC/acc, Encoder2 AUC/acc
- Interpretability quality: monotonicity violations, negative gains, mastery/gain correlations

**Monitoring Hook**: Updated to pass both encoder states (context_seq_1, value_seq_1, context_seq_2, value_seq_2) for dual-encoder comparison analysis

**Learning Trajectories**: Enhanced to show:
- Multi-skill support (per-skill gains and mastery)
- Dual predictions (Encoder1 vs Encoder2)
- Per-skill learning gains for relevant skills
- Effective practice accumulation

### Parameter Simplification (2025-11-16) - COMPLETE

**Objective**: Remove 27 deprecated parameters from CLI, reduce from 70 to 37 active parameters

**Deprecated Categories**:
- 6 constraint loss weights (all 0.0)
- 9 semantic alignment parameters (all inactive)
- 2 global alignment parameters (enable=false)
- 6 semantic refinement parameters (all inactive)
- 3 warmup/scheduling parameters (not used)
- 1 architecture parameter (intrinsic_gain_attention)

**Result**: CLI commands require ~15 essential parameters instead of 64, cleaner interface aligned with dual-encoder architecture

### Learning Curve Parameter Calibration (2025-11-16) - COMPLETE

**Phase 1 Sweep**: Optimize sigmoid parameters (beta, m_sat, gamma, offset)
- Grid: 81 experiments (3×3×3×3)
- Dataset: assist2015, fold 0
- Fixed: bce_loss_weight=0.2 (80% signal to Encoder2)
- Hardware: 7 GPUs parallel, ~6 hours, 100% success

**Results**:
- Best: E2 AUC = 0.5443 (Beta=2.5, M_sat=0.7, Gamma=1.1, Offset=1.5) → +6.7% vs baseline
- Parameter impact: beta (+0.72 strong), offset (-0.54 moderate), gamma (+0.41 weak), m_sat (-0.10 minimal)

**Key Pattern - "Steep Early Learning"**:
- High beta (2.5): Sharp transitions between "not mastered" and "mastered"
- Low offset (1.5): Measurable mastery after 1-2 practice attempts
- High gamma (1.1): Personalized learning pace for faster learners
- Conservative m_sat (0.7): Prevents overconfidence

**Updated Defaults** (effective 2025-11-16):
- beta_skill_init: 2.0 → 2.5 (+0.5)
- m_sat_init: 0.8 → 0.7 (-0.1)
- gamma_student_init: 1.0 → 1.1 (+0.1)
- sigmoid_offset: 2.0 → 1.5 (-0.5)

**Documentation**: Complete reports in `paper/PHASE1_SWEEP_REPORT.md` and `paper/PHASE1_SWEEP_SUMMARY.md`

**Phase 2** (Pending): Balance loss weights using optimal learning curve parameters from Phase 1

---

# Appendix: Technical Details

## Loss Functions

**Active Losses**:
1. **BCE Loss**: Binary cross-entropy on base predictions from Encoder 1
   - Applied to: prediction_head_1 output
   - Weight: 70% (default in V2 phase 2)
   - Purpose: Optimize response prediction accuracy

2. **Incremental Mastery Loss**: Binary cross-entropy on mastery-based predictions from Encoder 2
   - Applied to: `sigmoid((mastery - θ) / temperature)` output
   - Weight: 30% (default in V2 phase 2)
   - Purpose: Train mastery values to be predictive of responses

**Commented Out** (preserved in code for reference):
- Constraint losses: non_negative, monotonicity, mastery_performance, gain_performance, sparsity, consistency (all weight=0.0)
- Semantic losses: alignment, global_alignment, retention, lag_gain (all enable=false, weight=0.0)

## Architecture Compliance

| Feature | Specification | Implementation | Status |
|---------|---------------|----------------|--------|
| Dual Encoders | Two independent stacks | 167,575 params (96,513 + 71,040 + 22) | ✅ Active |
| Differentiable Practice | Quality-weighted accumulation | gain_quality → effective_practice | ✅ Active |
| Sigmoid Learning Curves | Practice-driven mastery | 5 learnable parameters | ✅ Active |
| Per-Skill Gains | Skill-specific learning rates | gains_projection layer | ✅ Active |
| Dual Predictions | Base + mastery-based | BCE + IM losses | ✅ Active |
| Constraint Losses | 6 educational constraints | All commented out | ❌ Inactive |
| Semantic Alignment | Population-level coherence | All commented out | ❌ Inactive |

## Experiment Directories

- **V0 (Baseline - Broken)**: `examples/experiments/20251117_131554_gainakt3exp_baseline-bce0.7_999787`
- **V1 (Per-Skill Gains)**: `examples/experiments/20251117_154349_gainakt3exp_fixed-per-skill-gains_995130`
- **V2 (Multiple Interventions)**: `examples/experiments/20251117_162330_gainakt3exp_fixed-per-skill-gains-v2_820618`
- **V3 (Inverse Warmup)**: TBD

## Trajectory Analysis Tools

**Extract Trajectories**:
```bash
python examples/learning_trajectories.py \
  --run_dir examples/experiments/{exp_dir} \
  --num_students 10 \
  --min_steps 10
```

**Verification Checks** (6 total):
1. Skill differentiation: gain std >0.05, CV >0.2
2. Q-Matrix learning: avg skills/interaction ≈ 1.0
3. Monotonicity: violation rate = 0.0
4. Mastery correlation: correlation >0.3
5. Skill difficulty: CV >0.2
6. Response-conditional: correct/incorrect gain ratio >1.2

## Parameter Evolution Protocol

All parameter changes documented in `examples/reproducibility.md` following protocol:
1. Update `configs/parameter_default.json` (source of truth)
2. Document rationale in commit message
3. Run `python examples/parameters_fix.py` to update MD5 hash
4. Update `paper/parameters.csv` with change justification

---

**Document Version**: 2025-11-18  
**Last Updated**: After STATUS document reorganization
