# GainAKT3Exp Model Status

**Document Version**: Updated 2025-11-17 (Critical Bug Fix - encoder2_pred skill indexing)  
**Model Version**: GainAKT3Exp - Dual-encoder transformer with Sigmoid Learning Curve Mastery  
**Status**: ⚠️ **CRITICAL BUGS IDENTIFIED** - Two architectural flaws prevent Encoder 2 from learning skill-specific patterns (2025-11-17)

---

## Reference Documents

For a description of the architecture foundations, see `gainakt3exp_architecture_approach.md`.

## Architecture Summary - Current State

**What's ACTIVE** (actually executed):
- ✅ **Dual-Encoder Architecture**: TWO completely independent encoder stacks (167,575 parameters total)
  - **Encoder 1 (Performance Path)**: 96,513 parameters
    - Components: context_embedding_1, value_embedding_1, skill_embedding_1, pos_embedding_1, encoder_blocks_1, prediction_head_1
    - Purpose: Learns attention patterns for response prediction
    - Output: Base Predictions → BCE Loss (weight ≈ 1.0)
  - **Encoder 2 (Interpretability Path)**: 71,040 parameters
    - Components: context_embedding_2, value_embedding_2, pos_embedding_2, encoder_blocks_2
    - Purpose: Learns attention patterns for learning gains detection
    - Output: Value representations → Gain Quality → Effective Practice → Mastery
  - **Independent Parameters**: No shared layers or representations between encoders
  - **Test Verified**: Both encoders receive gradients during backpropagation ✓
- ✅ **Differentiable Effective Practice**: Quality-weighted practice accumulation (CRITICAL for Encoder 2 gradient flow)
  - Gain quality computation: `gain_quality = sigmoid(value_seq_2.mean(dim=-1))` from Encoder 2
  - Accumulation: `effective_practice[t] = effective_practice[t-1] + gain_quality[t]`
  - Enables gradient flow: IM_loss → mastery → effective_practice → gain_quality → value_seq_2 → Encoder 2
  - Replaces non-differentiable practice counting with learnable quality weighting
- ✅ **Sigmoid Learning Curve Mastery**: Mastery evolves via sigmoid curves driven by effective practice
  - Formula: `mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × effective_practice[i,s,t] - offset)`
  - Learnable parameters: β_skill[s] (skill difficulty), γ_student[i] (learning velocity), M_sat[s] (saturation level), θ_global (threshold), offset (inflection point)
  - Automatic three-phase learning: Initial (warm-up) → Growth (rapid learning) → Saturation (consolidation)
- ✅ **Dual-Prediction Architecture**: TWO independent prediction outputs:
  - **Base Predictions**: From Encoder 1 → prediction head [context_1, value_1, skill_1] → MLP → sigmoid
  - **Incremental Mastery Predictions**: From Encoder 2 → sigmoid curves via threshold mechanism `sigmoid((mastery - θ_global) / temperature)`
- ✅ **Temperature Parameter**: Config-based (threshold_temperature=1.0, hybrid approach) - controls prediction sharpness
- ✅ **Mastery Output**: `projected_mastery` (sigmoid learning curves per skill) included in model output dictionary
- ✅ **Base Predictions Output**: `predictions` from Encoder 1 prediction head in model output dictionary
- ✅ **Incremental Mastery Predictions Output**: `incremental_mastery_predictions` from Encoder 2 via threshold mechanism in output dictionary
- ✅ **Dual Loss Functions (DUAL-ENCODER ARCHITECTURE 2025-11-16)**:
  - **BCE Loss**: Binary cross-entropy on **base predictions** (from Encoder 1) vs ground truth (primary loss, weight ≈ 1.0)
  - **Incremental Mastery Loss**: Binary cross-entropy on **incremental mastery predictions** (from Encoder 2) vs ground truth (interpretability loss, weight=0.1)
  - **Independent Gradient Flow**: Gradients flow independently through Encoder 1 (via BCE) and Encoder 2 (via IM loss)
- ❌ **Constraint Losses**: ALL COMMENTED OUT - `compute_interpretability_loss()` returns 0.0 (all weights=0.0):
  - Monotonicity Loss (weight=0.0) - ❌ COMMENTED OUT
  - Mastery-Performance Loss (weight=0.0) - ❌ COMMENTED OUT
  - Gain-Performance Loss (weight=0.0) - ❌ COMMENTED OUT
  - Sparsity Loss (weight=0.0) - ❌ COMMENTED OUT
  - Consistency Loss (weight=0.0) - ❌ COMMENTED OUT
  - Non-Negativity Loss (weight=0.0) - ❌ COMMENTED OUT
- ❌ **Semantic Losses**: ALL COMMENTED OUT (all enable flags=false, all weights=0.0):
  - Alignment Loss (weight=0.0) - ❌ COMMENTED OUT
  - Global Alignment (enable=false) - ❌ COMMENTED OUT
  - Residual Alignment (enable=false) - ❌ COMMENTED OUT
  - Retention Loss (weight=0.0) - ❌ COMMENTED OUT
  - Lag Gain Loss (weight=0.0) - ❌ COMMENTED OUT

**What's SUPPRESSED** (computed but not included in output):
- ⚠️ **Gains Head Output**: `projected_gains` computed internally for mastery but **NOT included** in output dictionary (controlled by `use_gain_head=false`)
- ⚠️ **Gains D-dimensional Output**: `projected_gains_d` not included in output (requires both heads enabled)

**What's INACTIVE** (code commented out):
- ❌ **Attention-Derived Gains**: Intrinsic gain attention mode completely commented out

**Code Location**: The dual-encoder architecture implementation is in `gainakt3_exp.py` (909 lines):

**Import Changes** (lines 51-53):
```python
import torch.nn as nn
import torch.nn.functional as F
from .gainakt3 import EncoderBlock  # Import only encoder block, not full class
```

**Class Definition** (line 56):
```python
# OLD: class GainAKT3Exp(GainAKT3):
# NEW: class GainAKT3Exp(nn.Module):  # Standalone implementation
```

**Encoder 1 Initialization** (lines 182-213):
```python
# Encoder 1 (Performance Path) - 96,513 parameters
self.context_embedding_1 = nn.Embedding(num_c * 2, d_model)
self.value_embedding_1 = nn.Embedding(num_c * 2, d_model)
self.skill_embedding_1 = nn.Embedding(num_c, d_model)
self.pos_embedding_1 = nn.Embedding(seq_len, d_model)

self.encoder_blocks_1 = nn.ModuleList([
    EncoderBlock(d_model, n_heads, d_ff, dropout, 
                intrinsic_gain_attention=False, num_skills=None)
    for _ in range(num_encoder_blocks)
])

self.prediction_head_1 = nn.Sequential(
    nn.Linear(d_model * 3, d_ff),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_ff, 1)
)
```

**Encoder 2 Initialization** (lines 215-247):
```python
# Encoder 2 (Interpretability Path) - 71,040 parameters
self.context_embedding_2 = nn.Embedding(num_c * 2, d_model)
self.value_embedding_2 = nn.Embedding(num_c * 2, d_model)
self.pos_embedding_2 = nn.Embedding(seq_len, d_model)

self.encoder_blocks_2 = nn.ModuleList([
    EncoderBlock(d_model, n_heads, d_ff, dropout,
                intrinsic_gain_attention=False, num_skills=None)
    for _ in range(num_encoder_blocks)
])
# No prediction head - outputs used for mastery computation
```

**Encoder 1 Forward Pass** (lines 356-377):
```python
# Pass through Encoder 1 (learns response patterns)
context_seq_1 = self.context_embedding_1(interaction_tokens) + self.pos_embedding_1(positions)
value_seq_1 = self.value_embedding_1(interaction_tokens) + self.pos_embedding_1(positions)

for block in self.encoder_blocks_1:
    context_seq_1, value_seq_1 = block(context_seq_1, value_seq_1, mask)

# Generate base predictions
target_concept_emb_1 = self.skill_embedding_1(target_concepts)
concatenated_1 = torch.cat([context_seq_1, value_seq_1, target_concept_emb_1], dim=-1)
logits = self.prediction_head_1(concatenated_1).squeeze(-1)
predictions = torch.sigmoid(logits)
```

**Encoder 2 Forward Pass** (lines 379-398):
```python
# Pass through Encoder 2 (learns learning gains patterns)
context_seq_2 = self.context_embedding_2(interaction_tokens) + self.pos_embedding_2(positions)
value_seq_2 = self.value_embedding_2(interaction_tokens) + self.pos_embedding_2(positions)

for block in self.encoder_blocks_2:
    context_seq_2, value_seq_2 = block(context_seq_2, value_seq_2, mask)

# value_seq_2 represents learning gains for mastery computation
```

**Differentiable Effective Practice** (lines 500-540 - CRITICAL FIX):
```python
# Compute gain quality from Encoder 2 outputs (differentiable!)
learning_gains_d = torch.relu(value_seq_2)
gain_quality_logits = learning_gains_d.mean(dim=-1, keepdim=True)
gain_quality = torch.sigmoid(gain_quality_logits)  # [B, L, 1]

# Accumulate quality-weighted effective practice (differentiable through Encoder 2!)
effective_practice = torch.zeros(batch_size, seq_len, num_c, device=q.device)
for t in range(seq_len):
    if t > 0:
        effective_practice[:, t, :] = effective_practice[:, t-1, :].clone()
    
    practiced_concepts = q[:, t].long()
    batch_indices = torch.arange(batch_size, device=q.device)
    
    # Differentiable increment! Gradients: gain_quality → value_seq_2 → Encoder 2
    effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
```

**Sigmoid Learning Curve** (lines 564-566):
```python
# Use effective_practice (differentiable through Encoder 2!)
sigmoid_input = (beta_expanded * gamma_expanded * effective_practice) - self.offset
projected_mastery = M_sat_expanded * torch.sigmoid(sigmoid_input)
practice_count = torch.zeros(batch_size, seq_len, num_c, device=q.device)
for t in range(seq_len):
    if t > 0:
        practice_count[:, t, :] = practice_count[:, t-1, :].clone()
    practiced_concepts = q[:, t].long()
    batch_indices = torch.arange(batch_size, device=q.device)
    practice_count[batch_indices, t, practiced_concepts] += 1

# Step 2: Compute sigmoid learning curve mastery
# Handle gamma_student (fixed vs dynamic per-batch)
if self.has_fixed_student_params:
    gamma = self.gamma_student.mean().unsqueeze(0).expand(batch_size)
else:
    gamma = torch.ones(batch_size, device=q.device)

# Expand dimensions for broadcasting
beta_expanded = self.beta_skill.unsqueeze(0).unsqueeze(0)  # [1, 1, num_c]
gamma_expanded = gamma.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
M_sat_expanded = self.M_sat.unsqueeze(0).unsqueeze(0)  # [1, 1, num_c]

# Compute sigmoid input: β_skill × γ_student × practice_count - offset
sigmoid_input = (beta_expanded * gamma_expanded * practice_count) - self.offset

# Compute mastery: M_sat × sigmoid(...)
projected_mastery = M_sat_expanded * torch.sigmoid(sigmoid_input)  # [batch, seq_len, num_c]
projected_mastery = torch.clamp(projected_mastery, min=0.0, max=1.0)
```

**Global Threshold Prediction** (lines 500-530):
```python
# Get the skill ID for each timestep
skill_indices = target_concepts.long()  # [B, L]

# Gather mastery for the actual skills being tested
batch_indices = torch.arange(batch_size, device=q.device).unsqueeze(1).expand(-1, seq_len)
time_indices = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
skill_mastery = projected_mastery[batch_indices, time_indices, skill_indices]  # [B, L]

# Use global threshold (clamped to [0,1] for stability)
theta_clamped = torch.clamp(self.theta_global, 0.0, 1.0)

# Compute incremental mastery predictions (differentiable via sigmoid)
incremental_mastery_predictions = torch.sigmoid((skill_mastery - theta_clamped) / self.threshold_temperature)
```

Output control (lines 479-490):
```python
output = {
    'predictions': predictions,  # Base predictions from prediction head
    'incremental_mastery_predictions': incremental_mastery_predictions  # Threshold predictions
}
if projected_gains is not None and self.use_gain_head:  # Currently false
    output['projected_gains'] = projected_gains  # Suppressed
```

Loss computation (lines 498-516):
```python
# Standard BCE loss on base predictions
# (computed in training script)

# New: Incremental mastery loss on threshold predictions
if incremental_mastery_predictions is not None:
    incremental_mastery_loss = F.binary_cross_entropy(
        incremental_mastery_predictions, r.float(), reduction='mean'
    )
```

**Result**: The model produces TWO independent predictions and computes TWO losses. Base predictions train the standard prediction head for performance, while incremental mastery predictions (derived from sigmoid learning curves) provide interpretability-driven supervision on mastery evolution. The sigmoid curves automatically capture three learning phases (warm-up, growth, saturation) with learnable skill difficulty (β_skill), student learning velocity (γ_student), and saturation levels (M_sat). Temperature parameter (config-based, default=1.0) controls prediction sharpness.




---

## Dual-Encoder Metrics and Monitoring (2025-11-16)

**Recent Enhancement**: Comprehensive metrics tracking and monitoring infrastructure updated to fully support dual-encoder architecture with separate performance tracking for both encoder pathways.

### 1. Enhanced CSV Metrics Output

**Files Updated**:
- `examples/train_gainakt3exp.py` - Training metrics collection
- `examples/eval_gainakt3exp.py` - Evaluation metrics collection
- `configs/parameter_default.json` - Added `bce_loss_weight` to interpretability type group

**Metrics Tracked** (saved to `metrics_epoch.csv` during training):

**Overall Performance Metrics** (Combined):
- `train_loss`, `train_auc`, `train_acc` - Overall training performance
- `val_loss`, `val_auc`, `val_acc` - Overall validation performance

**Loss Components** (Unweighted raw losses):
- `train_bce_loss`, `val_bce_loss` - BCE loss from Encoder 1 base predictions
- `train_im_loss`, `val_im_loss` - Incremental mastery loss from Encoder 2

**Weighted Losses** (As used in optimization):
- `train_weighted_bce`, `val_weighted_bce` - BCE loss × λ₁ (lambda weight)
- `train_weighted_im`, `val_weighted_im` - IM loss × λ₂ (1 - lambda weight)
- `train_total_weighted`, `val_total_weighted` - Sum of weighted losses

**Loss Contribution Analysis**:
- `train_bce_share`, `val_bce_share` - Percentage of total loss from BCE
- `train_im_share`, `val_im_share` - Percentage of total loss from IM

**Encoder 1 (Performance Path) Metrics**:
- `train_encoder1_auc`, `val_encoder1_auc` - AUC using only Encoder 1 base predictions
- `train_encoder1_acc`, `val_encoder1_acc` - Accuracy using only Encoder 1 base predictions

**Encoder 2 (Interpretability Path) Metrics**:
- `train_encoder2_auc`, `val_encoder2_auc` - AUC using only Encoder 2 incremental mastery predictions
- `train_encoder2_acc`, `val_encoder2_acc` - Accuracy using only Encoder 2 incremental mastery predictions

**Interpretability Quality Metrics**:
- `monotonicity_violation_rate` - Rate of mastery decreases (should be 0.0)
- `negative_gain_rate` - Rate of negative learning gains (should be 0.0)
- `bounds_violation_rate` - Rate of mastery outside [0,1] bounds (should be 0.0)
- `mastery_correlation` - Correlation between mastery levels and student performance
- `gain_correlation` - Correlation between learning gains and student performance

**Usage**: After training, analyze CSV to compare encoder performance:
```bash
# View metrics for specific experiment
cat examples/experiments/{experiment_dir}/metrics_epoch.csv | column -t -s,

# Quick comparison of encoder AUCs
awk -F',' 'NR>1 {print $1, $18, $20, $22, $24}' metrics_epoch.csv | column -t
# Output: epoch train_enc1_auc val_enc1_auc train_enc2_auc val_enc2_auc
```

### 2. Updated Monitoring Hook

**Files Updated**:
- `pykt/models/gainakt3_exp.py` - Monitor call updated to pass both encoder outputs
- `examples/interpretability_monitor.py` - Monitor class updated to handle dual-encoder states

**Monitoring Enhancements**:
- **Encoder 1 States**: `context_seq_1`, `value_seq_1` (performance path representations)
- **Encoder 2 States**: `context_seq_2`, `value_seq_2` (interpretability path representations)
- **Dual-Encoder Comparison**: Monitor can now track and compare attention patterns from both encoders
- **Frequency Control**: `monitor_freq` parameter (default: 50 batches) controls monitoring frequency

**Monitor Call** (in `gainakt3_exp.py`):
```python
self.interpretability_monitor(
    batch_idx=batch_idx,
    # Encoder 1 outputs (performance path)
    context_seq_1=context_seq_1,
    value_seq_1=value_seq_1,
    # Encoder 2 outputs (interpretability path)
    context_seq_2=context_seq_2,
    value_seq_2=value_seq_2,
    # Interpretability projections
    projected_mastery=projected_mastery,
    projected_gains=projected_gains,
    predictions=predictions,
    questions=q,
    responses=r
)
```

**Usage**: Monitor logs provide real-time insights during training:
- Statistics computed every N batches (configurable via `monitor_freq`)
- Separate tracking for performance vs interpretability encoder states
- Enables debugging of encoder-specific learning dynamics

### 3. Enhanced Learning Trajectories

**File Updated**: `examples/learning_trajectories.py`

**New Features**:

**Multi-Skill Support**:
- Generic handling of questions with multiple skills (not just single-skill ASSIST2015)
- Per-skill gains and mastery tracking
- Skill aggregation for questions targeting multiple concepts

**Dual-Encoder Predictions**:
- **True Response**: Ground truth (0/1)
- **Encoder 1 Prediction**: Base prediction from performance path
- **Encoder 2 Prediction**: Incremental mastery prediction from interpretability path
- **Comparison**: Side-by-side view of how both encoders perform

**Per-Skill Learning Gains**:
- Shows learning gains for relevant skills only (those targeted by the question)
- Distinguishes between gains for practiced vs non-practiced skills
- Tracks effective practice accumulation (quality-weighted)

**Enhanced Output Format**:
```
STUDENT #1
Global Index: 6 | Total Interactions: 10 | Unique Skills: 3 | Accuracy: 80.0%
================================================================================
Step | Skills      | True | Enc1_Pred | Enc2_Pred | Gains           | Mastery
-----|-------------|------|-----------|-----------|-----------------|----------------
  1  | 24          |  1   |   0.841   |   0.523   | {24: 0.0234}    | {24: 0.0864}
  2  | 24          |  1   |   0.901   |   0.678   | {24: 0.0312}    | {24: 0.1409}
  3  | 24,27       |  0   |   0.541   |   0.445   | {24: 0.0156,    | {24: 0.2208,
     |             |      |           |           |  27: 0.0189}    |  27: 0.0858}
```

**Key Enhancements**:
- Per-skill gains and mastery (handles multi-skill questions)
- Three prediction columns: true, Encoder 1, Encoder 2
- Clear visibility into dual-encoder behavior differences
- Gain quality tracking from Encoder 2's effective practice mechanism

**Usage**:
```bash
# Basic trajectory analysis
python examples/learning_trajectories.py \
  --run_dir examples/experiments/{experiment_dir} \
  --num_students 10 \
  --min_steps 10

# Focus on longer sequences for detailed progression
python examples/learning_trajectories.py \
  --run_dir examples/experiments/{experiment_dir} \
  --num_students 5 \
  --min_steps 50
```

### How to Leverage Dual-Encoder Metrics

**1. Encoder Comparison Analysis**:
```python
import pandas as pd

# Load metrics
df = pd.read_csv('experiments/{exp_dir}/metrics_epoch.csv')

# Compare encoder performance over epochs
print("Encoder 1 (Performance) vs Encoder 2 (Interpretability):")
print(df[['epoch', 'val_encoder1_auc', 'val_encoder2_auc', 
          'val_encoder1_acc', 'val_encoder2_acc']])

# Check if interpretability encoder maintains reasonable performance
print(f"\nEncoder 2 final AUC: {df['val_encoder2_auc'].iloc[-1]:.4f}")
print(f"Performance gap: {df['val_encoder1_auc'].iloc[-1] - df['val_encoder2_auc'].iloc[-1]:.4f}")
```

**2. Loss Balance Analysis**:
```python
# Verify loss weighting is working as intended
print("Loss contribution over time:")
print(df[['epoch', 'train_bce_share', 'train_im_share']])

# Expected: BCE share ≈ 0.90, IM share ≈ 0.10 (with default λ₁=0.9)
```

**3. Interpretability Quality Validation**:
```python
# Check interpretability constraints
print("\nInterpretability Quality Metrics:")
print(df[['epoch', 'monotonicity_violation_rate', 'negative_gain_rate',
          'mastery_correlation', 'gain_correlation']].tail())

# Ideal: violation rates near 0.0, correlations > 0.3
```

**4. Encoder-Specific Training Dynamics**:
- Monitor CSV during training to detect:
  - **Encoder 1**: Should quickly optimize for prediction accuracy (high AUC)
  - **Encoder 2**: May start lower but should maintain reasonable AUC while learning interpretable patterns
  - **Loss balance**: Verify BCE dominates (90%) while IM provides interpretability constraint (10%)

**5. Post-Training Analysis**:
```bash
# Compare final performance
tail -1 metrics_epoch.csv | awk -F',' '{
  printf "Final Results:\n"
  printf "  Overall Val AUC: %.4f\n", $6
  printf "  Encoder 1 Val AUC: %.4f\n", $20
  printf "  Encoder 2 Val AUC: %.4f\n", $22
  printf "  Mastery Correlation: %.4f\n", $29
}'
```

### Parameter Configuration

**Key Parameter** (in `configs/parameter_default.json`):
```json
{
  "defaults": {
    "bce_loss_weight": 0.7,  // λ₁ - BCE loss weight (updated 2025-11-17 post-bug-fix)
    // incremental_mastery_loss_weight computed as 1 - λ₁
    "patience": 10,          // Increased for more stable training
    "monitor_freq": 50       // Monitor every 50 batches
  },
  "types": {
    "interpretability": [
      "bce_loss_weight",     // Added to interpretability type group
      "use_mastery_head",
      // ... other interpretability params
    ]
  }
}
```

**Lambda Weight Interpretation** (Updated 2025-11-17):
- `bce_loss_weight = 0.7` → BCE contributes 70% to total loss (performance optimization)
- `incremental_mastery_loss_weight = 0.3` → IM contributes 30% to total loss (interpretability constraint)
- **Rationale**: After encoder2_pred bug fix, increased IM loss signal from 10% to 30% for stronger gradient flow to Encoder 2
- Total Loss: `λ₁ × BCE + (1-λ₁) × IM` where λ₁ + (1-λ₁) = 1.0

**Monitoring Frequency**:
- `monitor_freq = 50` → Statistics logged every 50 batches
- Lower values → More frequent logging (slower training)
- Higher values → Less frequent logging (faster training, less granular tracking)

### Validation and Debugging

**Check Metric Collection**:
```bash
# Verify CSV has all expected columns (30 total)
head -1 metrics_epoch.csv | tr ',' '\n' | nl

# Expected columns:
#  1. epoch
#  2-4. train_loss, train_auc, train_acc
#  5-7. val_loss, val_auc, val_acc
#  8-11. train_bce_loss, train_im_loss, val_bce_loss, val_im_loss
#  12-17. weighted losses and totals
#  18-21. loss shares
#  22-25. encoder1 metrics
#  26-29. encoder2 metrics
#  30-34. interpretability quality metrics
```

**Monitor Log Inspection**:
```bash
# Check monitoring is active (should see periodic statistics)
grep "Monitor" examples/experiments/{exp_dir}/training.log | head -20

# Expected output every monitor_freq batches:
# [Monitor] Batch 50: context_seq_1 mean=0.123, value_seq_1 mean=0.456
# [Monitor] Batch 50: context_seq_2 mean=0.234, value_seq_2 mean=0.567
# [Monitor] Batch 50: mastery mean=0.345, gains mean=0.089
```

**Trajectory Validation**:
```bash
# Run on small sample to verify dual predictions working
python examples/learning_trajectories.py \
  --run_dir examples/experiments/{exp_dir} \
  --num_students 2 \
  --min_steps 5 | head -50

# Expected: Should see both Enc1_Pred and Enc2_Pred columns with different values
```

---

## Overall Architecture Compliance

**⚠️ ARCHITECTURE SIMPLIFICATION (2025-11-16)**: 27 parameters deprecated and removed from CLI. See `tmp/DEPRECATED_PARAMETERS_2025-11-16.md` for details.

| **Feature**                | **Diagram Specification**                                      | **Implementation Details**                                                                 | **Status**          |
|----------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------|
| **Skill Embedding Table**  | Separate embedding for target skills in prediction            | `concept_embedding` used in `[h, v, s]` concatenation                                     | ✅ Activated         |
| **Dynamic Value Stream**   | Dual context/value sequences, separate norms, Q/K from context, V from value | Dual embeddings + separate `norm1_ctx/val`, `norm2_ctx` + correct attention               | ✅ Activated         |
| **Ground Truth Integration** | Used in loss calculation + monitoring hooks                  | Integrated in BCE loss + `set_monitor()` + periodic execution                           | ✅ Activated         |
| **Projection Heads**       | Mastery via sigmoid learning curves, Gains from attention Values | Sigmoid learning curve with learnable parameters (β_skill, γ_student, M_sat, θ_global, offset) | ✅ Mastery head active, Gain head inactive |
| **Dual-Loss Architecture**  | BCE + Incremental Mastery Loss                              | BCE (base predictions, weight=0.9) + IM Loss (threshold predictions, weight=0.1)          | ✅ Activated         |
| **Constraint Losses**      | 6 losses (NonNeg, Monotonicity, Mastery-Perf, Gain-Perf, Sparsity, Consistency) | All weights=0.0, code commented out in train/eval scripts                                | ❌ Deactivated       |
| **Semantic Alignment**     | Local alignment correlation loss                             | All alignment code commented out, enable_alignment_loss=false                             | ❌ Deactivated       |
| **Global Alignment**       | Population-level mastery coherence                           | Code commented out, enable_global_alignment_pass=false                                    | ❌ Deactivated       |
| **Semantic Refinement**    | Retention + Lag gain losses                                  | All refinement code commented out, enable flags=false, weights=0.0                        | ❌ Deactivated       |
| **Monitoring**             | Real-time interpretability analysis, configurable frequency   | `interpretability_monitor` hook + `monitor_frequency` + DataParallel safety              | ✅ Activated         |
| **Intrinsic Gain Attention** | Alternative parameter-efficient mode                        | `--intrinsic_gain_attention` flag deprecated, argparse commented out                      | ❌ Deprecated        |
| **Sigmoid Learning Curves** | Practice count-driven mastery evolution                      | Fully implemented with β_skill, γ_student, M_sat learnable parameters                    | ✅ Activated         |

---

## Parameters

**⚠️ PARAMETER DEPRECATION (2025-11-16)**: 27 parameters deprecated and removed from CLI arguments. Active parameters reduced from 70 to 37. See `tmp/DEPRECATED_PARAMETERS_2025-11-16.md` for complete deprecation details and rationale.

The complete list of **active parameters** is in `configs/parameter_default.json` (defaults section). Deprecated parameters are documented in the separate "deprecated" section for reference only.

### Model Instantiation

Models are created via `create_exp_model(config)` (`gainakt3_exp.py` line 435), which requires all parameters explicitly in the config dictionary.

### Active Parameters (37 total)

**Runtime Configuration (13 parameters)**:
- `seed`, `epochs`, `batch_size`, `learning_rate`, `weight_decay`
- `optimizer`, `gradient_clip`, `patience`, `monitor_freq`
- `use_amp`, `use_wandb`, `auto_shifted_eval`, `max_correlation_students`

**Model Architecture (7 parameters)**:
- `seq_len`, `d_model`, `n_heads`, `num_encoder_blocks`, `d_ff`, `dropout`, `emb_type`

**Interpretability Features (8 parameters)**:
- `use_mastery_head`, `use_gain_head`, `use_skill_difficulty`, `use_student_speed`
- `bce_loss_weight`, `mastery_threshold_init`, `threshold_temperature`, `num_students`

**Data Configuration (2 parameters)**:
- `dataset`, `fold`

**Launcher Parameters (2 parameters)**:
- `train_script`, `eval_script`

**Legacy Flags (5 parameters)** - Kept for backward compatibility:
- `enhanced_constraints` (always false)
- Deprecated loss weights (all 0.0): `non_negative_loss_weight`, `monotonicity_loss_weight`, `mastery_performance_loss_weight`, `gain_performance_loss_weight`, `sparsity_loss_weight`, `consistency_loss_weight`
- Deprecated alignment flags (all false): `enable_alignment_loss`, `enable_global_alignment_pass`, `enable_retention_loss`, `enable_lag_gain_loss`

### Deprecated Parameters (27 total, commented out in argparse)

**Constraint Losses (6)**: All weights=0.0, code commented out  
**Semantic Alignment (9)**: enable_alignment_loss=false, all related parameters inactive  
**Global Alignment (2)**: enable_global_alignment_pass=false  
**Semantic Refinement (16)**: All enable flags=false, weights=0.0  
**Warmup/Scheduling (3)**: Not used (constraints/alignment disabled)  
**Deprecated Architecture (1)**: `intrinsic_gain_attention`

See `tmp/DEPRECATED_PARAMETERS_2025-11-16.md` for complete list with deprecation rationale and dates.

### Zero Defaults Policy

The factory function raises errors if required parameters are missing, ensuring no hidden defaults. All defaults are defined in `configs/parameter_default.json`, which is loaded by `run_repro_experiment.py` and can be overridden via CLI.

**Command Simplification**: Training commands now require ~15 essential parameters instead of 64, making the interface much cleaner and aligned with the dual-encoder architecture. 

---

### Learning Curve Parameter Calibration

**Objective**: Optimize the four sigmoid learning curve parameters (beta_skill_init, m_sat_init, gamma_student_init, sigmoid_offset) that control Encoder 2's mastery accumulation formula.

**Two-Phase Strategy**:

**Phase 1: Optimize Learning Curve Parameters** ✅ **COMPLETE (2025-11-16)**

**Experimental Design**:
- **Fixed**: bce_loss_weight = 0.2 (80% signal to Encoder 2 via IM loss)
- **Grid**: 81 experiments (3×3×3×3 combinations)
  - beta_skill_init: [1.5, 2.0, 2.5] - Learning rate amplification
  - m_sat_init: [0.7, 0.8, 0.9] - Maximum mastery saturation
  - gamma_student_init: [0.9, 1.0, 1.1] - Student learning velocity
  - sigmoid_offset: [1.5, 2.0, 2.5] - Sigmoid inflection point
- **Dataset**: assist2015, fold 0, 6 epochs per experiment
- **Hardware**: 7 GPUs parallel, ~6 hours total, 100% success rate
- **Artifacts**: All experiments in `examples/experiments/20251116_174547_gainakt3exp_sweep_861908/`

**Results Summary** (E2 AUC statistics):
- **Best**: 0.5443 (Beta=2.5, M_sat=0.7, Gamma=1.1, Offset=1.5) → **+6.7% vs baseline 0.51**
- Mean: 0.5223, Median: 0.5219, Std: 0.0080
- Range: 0.5075 to 0.5443 (span: 0.0368)

**Parameter Impact** (correlation with E2 AUC):
1. **beta_skill_init**: +0.72 ⭐⭐⭐ STRONG (most important - steeper curves = better interpretability)
2. **sigmoid_offset**: -0.54 ⭐⭐ MODERATE (earlier inflection = faster mastery emergence)
3. **gamma_student_init**: +0.41 ⭐ WEAK-MODERATE (personalization helps tracking)
4. **m_sat_init**: -0.10 ○ MINIMAL (saturation level less critical)

**Key Finding - "Steep Early Learning" Pattern**:
The optimal configuration (Beta=2.5, Offset=1.5) creates interpretable mastery trajectories where:
- Students show measurable mastery after just 1-2 practice attempts (low offset)
- Sharp transitions between "not mastered" and "mastered" states (high beta)
- Personalized learning pace for faster learners (gamma=1.1)
- Conservative mastery ceiling prevents overconfidence (m_sat=0.7)

**Validation**: Strategy validated - high IM loss weight (0.8) successfully optimized Encoder 2 parameters without degrading Encoder 1 performance (stable at 0.6860 across all configs).

**Updated Defaults** (effective 2025-11-16):
- `beta_skill_init`: 2.0 → **2.5** (+0.5)
- `m_sat_init`: 0.8 → **0.7** (-0.1)
- `gamma_student_init`: 1.0 → **1.1** (+0.1)
- `sigmoid_offset`: 2.0 → **1.5** (-0.5)

**Documentation**: See `paper/PHASE1_SWEEP_REPORT.md` for complete analysis and `paper/PHASE1_SWEEP_SUMMARY.md` for quick reference.

**Phase 2: Balance Loss Weights** ⏳ **PENDING**
- **Fixed**: Optimal learning curve parameters from Phase 1 (Beta=2.5, M_sat=0.7, Gamma=1.1, Offset=1.5)
- **Sweep**: bce_loss_weight in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- **Metric**: Combined score (Encoder 1 AUC + α × Encoder 2 AUC, where α balances performance vs interpretability)
- **Rationale**: Find optimal loss balance that maximizes both prediction accuracy and interpretability using the validated learning curve parameters.




---

## Loss Functions

**⚠️ ARCHITECTURE SIMPLIFICATION (2025-11-15)**: 
**ACTIVE LOSSES**: BCE Loss + Incremental Mastery Loss ONLY
**COMMENTED OUT**: All Constraint Losses + All Semantic Module Losses (weights set to 0.0, code preserved)

This section documents ALL loss functions (active and commented out) for reference and potential future restoration.

---

**Original Total Loss** = BCE Loss + Constraint Losses + Semantic Module Losses

**Simplified Total Loss (Current)** = BCE Loss + Incremental Mastery Loss

### Loss Parameters

| Category | Name | Parameter Name | Current Value | Description | Status |
|----------|------|----------------|---------------|-------------|--------|
| **Main** | BCE Loss | - | - | Binary cross-entropy for base response prediction | ✅ ACTIVE |
| **Main** | Incremental Mastery Loss | `incremental_mastery_loss_weight` | 0.1 | BCE on threshold-based mastery predictions | ✅ ACTIVE |
| **Constraint** | Non-Negative Gains | `non_negative_loss_weight` | 0.0 | Penalizes negative learning gains | ❌ COMMENTED OUT |
| **Constraint** | Monotonicity | `monotonicity_loss_weight` | 0.0 | Enforces non-decreasing mastery over time | ❌ COMMENTED OUT |
| **Constraint** | Mastery-Performance | `mastery_performance_loss_weight` | 0.0 | Penalizes low mastery on correct, high on incorrect | ❌ COMMENTED OUT |
| **Constraint** | Gain-Performance | `gain_performance_loss_weight` | 0.0 | Enforces higher gains for correct responses | ❌ COMMENTED OUT |
| **Constraint** | Sparsity | `sparsity_loss_weight` | 0.0 | Penalizes gains on non-relevant skills | ❌ COMMENTED OUT |
| **Constraint** | Consistency | `consistency_loss_weight` | 0.0 | Aligns mastery changes with scaled gains | ❌ COMMENTED OUT |
| **Semantic** | Alignment (Local) | `alignment_weight` | 0.0 | Maximizes correlation between mastery/gains and performance | ❌ COMMENTED OUT |
| **Semantic** | Global Alignment | `enable_global_alignment_pass` | false | Population-level mastery coherence regularization | ❌ COMMENTED OUT |
| **Semantic** | Residual Alignment | `use_residual_alignment` | false | Alignment on variance unexplained by global signal | ❌ COMMENTED OUT |
| **Semantic** | Retention | `retention_weight` | 0.0 | Prevents post-peak mastery decay | ❌ COMMENTED OUT |
| **Semantic** | Lag Gain | `lag_gain_weight` | 0.0 | Introduces temporal structure to gains (lag-1,2,3) | ❌ COMMENTED OUT |
| **Schedule** | Constraint Warmup | `warmup_constraint_epochs` | 8 | Epochs to ramp constraint losses from 0 to full | ⚠️ NOT USED (constraints disabled) |
| **Schedule** | Alignment Warmup | `alignment_warmup_epochs` | 8 | Epochs to ramp alignment loss from 0 to full | ⚠️ NOT USED (alignment disabled) |
| **Schedule** | Alignment Share Cap | `alignment_share_cap` | 0.08 | Maximum proportion of total loss from alignment | ⚠️ NOT USED (alignment disabled) |

### Active Losses (SIMPLIFIED ARCHITECTURE)

#### BCE Loss (✅ ACTIVE)

**Binary Cross-Entropy (BCE) Loss**: Primary loss function for base response correctness prediction.
- Applied to: **Base predictions** from standard prediction head
- Implementation: `torch.nn.BCEWithLogitsLoss()` applied to logits
- Purpose: Optimizes the model's primary task of predicting student responses
- Weight: Implicit weight = 1.0 (not a configurable parameter)

#### Incremental Mastery Loss (✅ ACTIVE, NEW in Simplified Architecture)

**Incremental Mastery Loss** (`incremental_mastery_loss_weight = 0.1`): Secondary loss function for threshold-based mastery predictions.

**Purpose**: Provides an interpretability-driven learning signal through the dual-prediction mechanism:
- Applied to: **Incremental mastery predictions** from threshold mechanism
- Formula: `sigmoid((skill_mastery - skill_threshold) / temperature)`
- Implementation: Binary cross-entropy between threshold predictions and ground truth responses
- Weight: 0.1 (contributes ~5-15% of total loss)

**Rationale**: This loss creates a **second independent prediction branch** that:
1. Forces mastery values to be semantically meaningful (high mastery → correct predictions)
2. Makes learnable thresholds adapt to represent "sufficient mastery for success"
3. Provides gradient signal to mastery accumulation mechanism
4. Improves interpretability without relying on complex constraint losses

**Code Location**: 
- Model computation: `pykt/models/gainakt3_exp.py` lines 511-519
- Training integration: `examples/train_gainakt3exp.py` lines 811, 965, 1133-1137, 1157

**Expected Behavior**:
- Loss share: ~5-15% of total loss (weight=0.1, but actual contribution varies by epoch)
- Should be non-zero in all training epochs (if 0.0, indicates bug in training loop)
- Gradients flow through: mastery values → threshold mechanism → incremental predictions

---

### Commented Out Losses (Code Preserved for Future Restoration)

The following losses are **COMMENTED OUT** in the simplified architecture (all weights=0.0 or disabled). Code is preserved in comments for potential future restoration. See commit 4849b72 for simplification details.

#### Constraint Losses (❌ ALL COMMENTED OUT)

**⚠️ STATUS**: All constraint losses are COMMENTED OUT (weights=0.0). The `compute_interpretability_loss()` method in `pykt/models/gainakt3_exp.py` (lines 572-649) now returns `torch.tensor(0.0)`. Original code preserved in comments for potential future restoration.

**Original Purpose** (when active): Constraint losses enforced structural validity and educational plausibility of the projected mastery and gain trajectories. They operated at the **interaction level**, penalizing specific violations of educational expectations. Unlike semantic module losses that shaped overall trajectory correlations, constraint losses acted as **hard regularizers** preventing degenerate or nonsensical states.

**Semantic Constraints** (original design):
1. **Non-negative gains**: Learning gains are always positive (>=0)
2. **Monotonic mastery**: Mastery can only increase or stay constant over time
3. **Consistency**: Mastery increments consistent with learning gains
4. **Sparsity**: Gains only on relevant skills per Q-Matrix

---

**Non-Negative Gains** (`non_negative_loss_weight = 0.0`, ❌ COMMENTED OUT): Originally penalized negative learning gains by computing `clamp(-projected_gains, min=0).mean()`. Code preserved but disabled as gains are naturally non-negative due to model architecture.

**Monotonicity** (`monotonicity_loss_weight = 0.0`, ❌ COMMENTED OUT): Originally enforced non-decreasing mastery over time by penalizing `clamp(mastery[t] - mastery[t+1], min=0).mean()`. Code preserved but disabled in simplified architecture.

**Mastery-Performance Alignment** (`mastery_performance_loss_weight = 0.0`, ❌ COMMENTED OUT): Originally penalized interaction-level mismatches between mastery and performance with hinge-style constraints. Code preserved but disabled - this functionality is now partially covered by Incremental Mastery Loss.

**Gain-Performance Alignment** (`gain_performance_loss_weight = 0.0`, ❌ COMMENTED OUT): Originally enforced that correct responses yield higher gains than incorrect responses via hinge loss. Code preserved but disabled.

**Sparsity** (`sparsity_loss_weight = 0.2`): Penalizes non-zero gains for skills not directly involved in the current interaction via `abs(non_relevant_gains).mean()`. Encourages skill-specific learning (gains concentrated on the question's target skill) rather than diffuse updates across all skills, improving interpretability and alignment with skill-specific educational theories. **Skill Mask Computation:** Uses Q-matrix structure via `skill_masks.scatter_(2, questions.unsqueeze(-1), 1)` to identify relevant skills—correctly implements sparsity constraint based on problem-skill mappings.

**Consistency** (`consistency_loss_weight = 0.3`): Enforces temporal coherence between mastery changes and scaled gains via `|mastery_delta - scaled_gains * 0.1|.mean()`. Ensures that mastery increments align with the projected gain magnitudes, preventing the model from producing contradictory mastery and gain trajectories (e.g., large gains with flat mastery, or mastery jumps with zero gains).

All constraint losses are subject to warm-up scheduling (`warmup_constraint_epochs = 8`), gradually ramping from zero to full weight to allow the model to establish baseline representations before enforcing strict constraints. Violation rates are monitored and logged; current optimal configuration achieves **zero violations** across all constraints.

### Semantic Module Losses

**⚠️ STATUS**: All semantic module losses are **COMMENTED OUT** in the simplified architecture (2025-11-15). All semantic module enable flags are set to `false` and weights are `0.0` in the current configuration. Code implementing these losses is preserved in `pykt/models/gainakt3_exp.py` but not active.

**Original Purpose** (when active): Semantic module losses enabled alignment, global alignment, retention, and lag objectives to restore strong semantic interpretability. When active, these losses produced mastery and gain correlations surpassing prior breakthrough levels with stable trajectories. The current simplified architecture focuses on BCE + Incremental Mastery Loss only, with semantic losses preserved for future architectural exploration.

**❌ COMMENTED OUT - Alignment Loss (Local)** (`alignment_weight = 0.0`): When active, this loss encourages the model's projected mastery estimates to align with actual student performance on individual interactions. It penalizes low mastery when students answer correctly and high mastery when they answer incorrectly. This local constraint shapes mastery trajectories to be performance-consistent at the interaction level, accelerating the emergence of educationally meaningful correlations.

**❌ COMMENTED OUT - Global Alignment Pass** (`enable_global_alignment_pass = false`): When active, computes population-level mastery statistics (mean/variance across students) and uses them to regularize individual mastery trajectories toward global coherence patterns. This cross-student alignment improves mastery correlation stability by reducing inter-student variance and reinforcing common learning progressions.

**❌ COMMENTED OUT - Residual Alignment** (`use_residual_alignment = false`): When active, applied after global alignment to capture unexplained variance. By removing the global signal component, residual alignment clarifies incremental mastery improvements specific to individual learning contexts, yielding sharper and more interpretable correlation patterns.

**❌ COMMENTED OUT - Retention Loss** (`retention_weight = 0.0`): When active, prevents post-peak decay of mastery trajectories by penalizing decreases in mastery levels after they reach local maxima. This ensures that once students demonstrate mastery, the model maintains elevated mastery estimates rather than allowing degradation, supporting higher final correlation retention ratios.

**❌ COMMENTED OUT - Lag Gain Loss** (`lag_gain_weight = 0.0`): When active, introduces temporal structure to learning gains by encouraging gains at timestep t to correlate with gains at previous timesteps (lag-1, lag-2, lag-3). This creates a coherent temporal narrative where gains emerge systematically rather than randomly, enhancing gain correlation interpretability and capturing causal learning progression patterns.

#### Alignment Schedule Parameters

**⚠️ STATUS**: These parameters are **NOT USED** in the simplified architecture since all semantic losses are commented out. Documentation preserved for future reference.

**Original Functionality** (when semantic losses are active): The semantic module losses, particularly alignment loss, use scheduling mechanisms to balance interpretability emergence with predictive performance:

**⚠️ NOT USED - Warm-up Scheduling** (`alignment_warmup_epochs = 8`): When active, alignment loss is gradually ramped from zero to full weight over the first 8 epochs, allowing the model to establish baseline representations before enforcing strict alignment constraints. This prevents early optimization conflicts where the model hasn't yet learned discriminative features, which could cause training instability or degrade predictive performance.

**⚠️ NOT USED - Share Cap** (`alignment_share_cap = 0.08`): When active, limits the maximum proportion of total loss contributed by alignment to 8%. This prevents alignment from dominating the optimization objective, which could sacrifice predictive accuracy (BCE loss) for interpretability. The cap acts as a soft constraint ensuring that performance remains competitive while still benefiting from alignment-driven trajectory shaping.

**Historical Rationale:** Early experiments showed that uncapped alignment loss could improve mastery correlation by 15-20% but degrade AUC by 2-3%. The combination of warm-up + share cap enabled a balanced regime where interpretability improved (mastery correlation: 0.095±0.018) while maintaining competitive AUC (0.720±0.001). The 8-epoch warm-up aligned with constraint warm-up (`warmup_constraint_epochs = 8`), creating a coordinated two-phase training strategy: (1) Phase 1 (epochs 1-8): representation learning with gradual constraint introduction, (2) Phase 2 (epochs 9-12): full multi-objective optimization with alignment capped at 8% of total loss.

**Implementation Note:** The share cap was dynamically computed per batch as `min(alignment_loss * alignment_weight, total_loss * alignment_share_cap)`, ensuring the constraint applied regardless of batch-level loss magnitude fluctuations. This provided stable training dynamics across different dataset characteristics and batch compositions.


---

## Implementation History

### Dual-Encoder Architecture Implementation (2025-11-16)

**Status**: ✅ **COMPLETED AND TESTED**

**Objective**: Implement two completely independent encoder stacks to separate performance optimization (Encoder 1) from interpretability learning (Encoder 2).

**Changes Made**:

1. **Model Architecture** (`pykt/models/gainakt3_exp.py`, 909 lines):
   - **Changed class definition** (line 56): From `class GainAKT3Exp(GainAKT3)` to `class GainAKT3Exp(nn.Module)`
     - No longer inherits from GainAKT3 base class
     - Standalone implementation with complete control over architecture
   
   - **Import changes** (lines 51-53): Import only `EncoderBlock` from gainakt3, not full class
   
   - **Encoder 1 (Performance Path)** (lines 182-213): 96,513 parameters
     - Independent embeddings: context_embedding_1, value_embedding_1, skill_embedding_1, pos_embedding_1
     - Independent encoder blocks: encoder_blocks_1 (ModuleList)
     - Prediction head: prediction_head_1 (MLP: concat → Linear → ReLU → Dropout → Linear)
     - Purpose: Learn attention patterns for response prediction
   
   - **Encoder 2 (Interpretability Path)** (lines 215-247): 71,040 parameters
     - Independent embeddings: context_embedding_2, value_embedding_2, pos_embedding_2
     - Independent encoder blocks: encoder_blocks_2 (ModuleList)
     - No prediction head (outputs used for mastery computation)
     - Purpose: Learn attention patterns for learning gains detection
   
   - **Forward pass updates**:
     - Lines 356-377: Encoder 1 forward (tokenization → embeddings → encoder blocks → prediction head)
     - Lines 379-398: Encoder 2 forward (tokenization → embeddings → encoder blocks → value outputs)
   
   - **Differentiable Effective Practice** (lines 500-540 - CRITICAL FIX):
     - Problem: Original implementation used discrete practice counting (non-differentiable)
       - `practice_count = count_interactions(q)` had no gradients
       - Prevented Encoder 2 from learning via backpropagation
     - Solution: Quality-weighted practice accumulation
       ```python
       # Encoder 2 learns gain quality (differentiable!)
       gain_quality = sigmoid(value_seq_2.mean(dim=-1))  # From Encoder 2
       
       # Quality-weighted accumulation (differentiable!)
       effective_practice[t] = effective_practice[t-1] + gain_quality[t]
       ```
     - Gradient flow: IM_loss → mastery → effective_practice → gain_quality → value_seq_2 → Encoder 2 parameters
   
   - **Sigmoid Learning Curve** (lines 564-566): Uses effective_practice instead of practice_count
     ```python
     sigmoid_input = (beta_expanded * gamma_expanded * effective_practice) - self.offset
     projected_mastery = M_sat_expanded * torch.sigmoid(sigmoid_input)
     ```

2. **Test Script** (`tmp/test_dual_encoders.py`, 263 lines):
   - Created comprehensive test suite with 7 test scenarios
   - Test 1: Model creation
   - Test 2: Independent encoder parameters verification
   - Test 3: Parameter counts (96,513 + 71,040 + 22 = 167,575)
   - Test 4: Forward pass execution and output shapes
   - Test 5: **Gradient flow verification** (CRITICAL TEST)
   - Test 6: Independent learning during training
   - Test 7: Encoder output differences
   - **Result**: ALL 7 TESTS PASSED ✓

**Test Results**:
```
================================================================================
ALL TESTS PASSED! ✓
================================================================================

1. Creating dual-encoder model... ✓
2. Verifying independent encoder parameters... ✓
   - Encoder 1 (Performance Path) components present ✓
   - Encoder 2 (Interpretability Path) components present ✓
   - Encoders have independent parameters (not shared) ✓

3. Analyzing parameter counts... ✓
   - Encoder 1 (Performance): 96,513 parameters
   - Encoder 2 (Interpretability): 71,040 parameters
   - Sigmoid curve parameters: 22 parameters
   - Total model parameters: 167,575 parameters

4. Testing forward pass... ✓
   - Base predictions shape: torch.Size([8, 15]) ✓
   - Incremental mastery predictions shape: torch.Size([8, 15]) ✓
   - Projected mastery shape: torch.Size([8, 15, 10]) ✓

5. Testing gradient flow through both encoders... ✓
   - Gradients flow through Encoder 1 (Performance Path) ✓
   - Gradients flow through Encoder 2 (Interpretability Path) ✓  ← CRITICAL FIX VERIFIED
   - Gradients flow through sigmoid learning curve parameters ✓

6. Verifying encoders learn independently... ✓
   - Both encoders update independently during training ✓

7. Verifying encoder outputs differ... ✓
   - Mean absolute difference between predictions: 0.492383 ✓
   - Encoder outputs are different (independent learning) ✓

The model is ready for training with dual-encoder architecture!
```

**Key Insights**:
- Differentiable effective practice mechanism is critical for Encoder 2 gradient flow
- Test confirmed gradients flow through both encoders independently
- Both encoders learn different patterns (output diff: 0.492)
- Ready for production training experiments

---

## Sigmoid Learning Curve Implementation (2025-11-16)

**Status**: ✅ **COMPLETED AND TESTED** (Now integrated with dual-encoder architecture)

**Changes Made**:

1. **Model Architecture** (`pykt/models/gainakt3_exp.py`):
   - Added 5 new learnable parameters for sigmoid learning curves:
     - `beta_skill[num_c]`: Skill difficulty (controls curve steepness), init=1.0
     - `gamma_student[num_students]`: Student learning velocity (optional), init=1.0
     - `M_sat[num_c]`: Saturation mastery level (max achievable), init=0.8
     - `theta_global`: Global threshold (scalar), init=0.85
     - `offset`: Sigmoid inflection point (scalar), init=3.0
   - Replaced linear accumulation with practice count-driven sigmoid curves
   - Updated threshold prediction mechanism (global θ_global instead of per-skill)
   - Updated factory function with new parameters

2. **Practice Count Tracking**:
   - Implemented per-student-skill interaction counting
   - Formula: `practice_count[i,s,t] = Σ(k=1 to t) 𝟙[question[k] targets skill s]`

3. **Sigmoid Learning Curve**:
   - Formula: `mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × practice_count[i,s,t] - offset)`
   - Automatic three-phase learning: Initial (warm-up) → Growth (rapid) → Saturation (consolidation)

4. **Configuration**:
   - `configs/parameter_default.json`: Verified parameters present
   - `paper/parameters.csv`: Added threshold_temperature documentation
   - Temperature parameter: Config-based (hybrid approach), default=1.0

5. **Testing**:
   - Created comprehensive test script: `tmp/test_sigmoid_curves.py`
   - All tests pass:
     - ✅ Model instantiation and parameter shapes
     - ✅ Forward pass execution
     - ✅ Sigmoid curve dynamics (three learning phases verified)
     - ✅ Monotonicity (mastery never decreases)
     - ✅ Boundedness (mastery ∈ [0, M_sat])
     - ✅ Threshold-based predictions
     - ✅ Loss computation (non-negative, finite)

**Test Results** (sample trajectory for 20 repeated practices of skill 0):
```
Timestep:  0      2      4      6      8     10     12     14     16     18
Mastery:   0.095  0.400  0.705  0.786  0.798  0.800  0.800  0.800  0.800  0.800
```
Clear sigmoid pattern: rapid growth → saturation at M_sat=0.8

**Educational Semantics**:
- **β_skill**: Controls how quickly a skill can be learned (skill difficulty)
- **γ_student**: Controls how fast a student learns (individual ability)
- **M_sat**: Controls how masterable a skill is (complexity ceiling)
- **θ_global**: Defines what mastery level indicates competence (decision boundary)
- **offset**: Controls when rapid learning begins (inflection point)

**Files Modified**:
- `pykt/models/gainakt3_exp.py` (808 lines): Core implementation
- `paper/STATUS_gainakt3exp.md`: Documentation updates
- `configs/parameter_default.json`: Parameter verification
- `paper/parameters.csv`: Added threshold_temperature

**Files Created**:
- `tmp/test_sigmoid_curves.py` (223 lines): Comprehensive test suite
- `tmp/SIGMOID_IMPLEMENTATION_REPORT.md`: Detailed implementation report

**Next Steps**:
1. Train model with sigmoid learning curves on ASSIST2015 dataset
2. Evaluate performance (AUC, accuracy, mastery correlations)
3. Compare with baseline models and previous GainAKT versions
4. Analyze learned parameter values (β_skill, M_sat, etc.)
5. Validate interpretability improvements


---

## Summary

**Model Status**: ✅ **DUAL-ENCODER IMPLEMENTATION COMPLETE** (2025-11-16)

The GainAKT3Exp model with dual-encoder architecture is fully implemented and tested. The model uses two completely independent encoder stacks to separate performance optimization from interpretability learning. A differentiable effective practice mechanism enables gradient flow through Encoder 2, allowing it to learn which interactions provide high-quality learning opportunities. All core components are functional:

**Architecture**:
- ✅ Dual-encoder design: 167,575 total parameters
  - Encoder 1 (Performance Path): 96,513 parameters → Base Predictions → BCE Loss
  - Encoder 2 (Interpretability Path): 71,040 parameters → Gain Quality → Mastery → IM Loss
  - Sigmoid curve parameters: 22 parameters (β_skill, γ_student, M_sat, θ_global, offset)
- ✅ Complete parameter independence (no shared layers between encoders)
- ✅ Independent gradient flow verified (test passed ✓)

**Differentiable Effective Practice** (Critical Innovation):
- ✅ Encoder 2 learns gain quality: `gain_quality = sigmoid(value_seq_2.mean(dim=-1))`
- ✅ Quality-weighted accumulation: `effective_practice[t] = effective_practice[t-1] + gain_quality[t]`
- ✅ Gradient flow: IM_loss → mastery → effective_practice → gain_quality → value_seq_2 → Encoder 2
- ✅ Replaces non-differentiable practice counting with learnable quality weighting

**Sigmoid Learning Curves**:
- ✅ Formula: `mastery = M_sat × sigmoid(β_skill × γ_student × effective_practice - offset)`
- ✅ Automatic three-phase learning: Initial (warm-up) → Growth (rapid learning) → Saturation (consolidation)
- ✅ 5 learnable parameters with clear educational semantics

**Dual-Prediction Architecture**:
- ✅ Base predictions from Encoder 1 (performance-optimized)
- ✅ Incremental mastery predictions from Encoder 2 (interpretability-optimized)
- ✅ Dual loss functions (BCE + incremental mastery loss)

**Test Validation** (tmp/test_dual_encoders.py - ALL PASSED ✓):
- ✅ Model creation successful
- ✅ Independent encoder parameters verified (no sharing)
- ✅ Parameter counts correct (Encoder 1: 96,513, Encoder 2: 71,040, Total: 167,575)
- ✅ Forward pass produces correct output shapes
- ✅ **Gradients flow through BOTH encoders** (critical test)
- ✅ Both encoders update independently during training
- ✅ Encoder outputs differ (mean abs diff: 0.492)

**Ready for Training**: The model is ready to be trained on knowledge tracing datasets (ASSIST2015, etc.) to evaluate performance and interpretability improvements compared to baseline models.

**Key Innovations**: 
1. **Dual-Encoder Architecture**: Complete separation of performance and interpretability pathways with independent attention mechanisms
2. **Differentiable Effective Practice**: Quality-weighted accumulation enables Encoder 2 to learn which interactions provide learning opportunities
3. **Educational Interpretability**: All parameters have clear educational meanings (skill difficulty, student velocity, gain quality, etc.)

---

**Previous GainAKT3Exp Results** (before sigmoid curves, for reference):
| Split      | AUC       | Accuracy  | Mastery Correlation | Gain Correlation | Correlation Students | Timestamp                  |
|------------|-----------|-----------|---------------------|------------------|----------------------|---------------------------|
| Training   | 0.7242    | 0.7510    | 0.0260              | 0.0257           | 3334                 | 2025-11-14T18:20:58.091915 |
| Validation | 0.7139    | 0.7512    | N/A                 | N/A              | N/A                  | 2025-11-14T18:20:58.091915 |
| Test       | 0.7095    | 0.7452    | 0.0221              | 0.0216           | 3177                 | 2025-11-14T18:20:58.091915 |

*Note: These results are from the linear accumulation version. New results with sigmoid learning curves pending.*


## Sweeps

# Performance mode (bce=0.9)
{
  "beta_skill_init": 2.0,
  "m_sat_init": 0.8,
  "gamma_student_init": 1.0,
  "sigmoid_offset": 2.0,
  "bce_loss_weight": 0.9
}

# Interpretability mode (bce=0.3) 
{
  "beta_skill_init": 2.5,
  "m_sat_init": 0.7,
  "gamma_student_init": 1.1,
  "sigmoid_offset": 1.5,
  "bce_loss_weight": 0.3
}

```
Option A: Sweep Confirms 714616's Params
If experiment #41 achieves ~0.7183:
# Update defaults to 714616's configuration
# Document as "performance mode" defaults
# Keep Phase 1+2 as "interpretability mode"


Option B: Sweep Finds Better Params
If another configuration exceeds 0.7183:

# Update to the best-found configuration
# Document improvement over 714616
# Update paper with new findings

Option C: Results Are Inconclusive
If we can't match 714616:

# Investigate other factors
# Run targeted follow-up experiments
# Document the mystery for future work

What I'll Do After Sweep Completes:
Extract experiment #41's Test AUC
Compare with 714616's 0.7183
Identify best overall configuration
Propose default updates with data justification
Run parameters_fix.py to update MD5
Create commit with sweep results

```

---

## Architectural Evolution

### Phase 0: Baseline (Broken Architecture)

**Experiment**: 20251117_131554_gainakt3exp_baseline-bce0.7_999787

**Architectural Bug**: Scalar gain quality instead of per-skill gains

**Implementation** (WRONG):
```python
# Step 1: Compute SCALAR gain quality per interaction
gain_quality_logits = learning_gains_d.mean(dim=-1, keepdim=True)  # [B, L, D] → [B, L, 1]
gain_quality = torch.sigmoid(gain_quality_logits)  # [B, L, 1] ∈ [0, 1]  ← SCALAR!

# Step 2: Apply SAME scalar to ALL practiced skills
effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
#                                                           SAME SCALAR FOR ALL SKILLS!
```

**The Problem**:
- ✅ Can learn: "This interaction had high/low engagement quality"
- ❌ Cannot learn: "Student improved Skill A by 0.8, Skill B by 0.2"
- Result: All practiced skills increment by SAME scalar value
- Impact: No skill differentiation, uniform mastery growth

**Results**:
| Metric | Value | Status |
|--------|-------|--------|
| Encoder1 Val AUC | 0.6765 | ✅ Working well |
| **Encoder2 Val AUC** | **0.4842** | ❌ **Below random (48.42%)** |
| Mastery ↔ Response | -0.044 | ❌ **No correlation** |
| Encoder2_Pred Range | [0.37, 0.53] | ❌ **Too narrow** |

### Phase 1: Per-Skill Gains Fix (V1)

**Experiment**: 20251117_154349_gainakt3exp_fixed-per-skill-gains_995130

**Architectural Fix**: Implemented per-skill gains vector

**Implementation** (CORRECT):
```python
# Add skill-specific projection in __init__
self.gains_projection = nn.Linear(d_model, num_c)  # [D=256] → [num_c]

# In forward_with_states, compute per-skill gains
skill_gains_logits = self.gains_projection(value_seq_2)  # [B, L, D] → [B, L, num_c]
skill_gains = torch.sigmoid(skill_gains_logits)  # [B, L, num_c] ∈ [0, 1]

# Accumulate skill-specific gains (each skill gets different increment!)
effective_practice[:, t, :] += skill_gains[:, t, :]  # Per-skill, differentiable
```

**Benefits**:
- ✅ Encoder 2 can learn different learning rates per skill
- ✅ Maintains differentiability (gradients flow through gains_projection)
- ✅ Enables skill difficulty learning
- ✅ Realistic skill-specific mastery trajectories

**Results**:
| Metric | V0 (Broken) | V1 (Fixed) | Change | Status |
|--------|-------------|------------|--------|--------|
| Encoder2 Val AUC | 0.4842 | **0.5868** | **+10.3%** | ✅ **IMPROVED** |
| Encoder2 Test AUC | ~0.48 | **0.5835** | **+10.3%** | ✅ Above random |
| Mastery Correlation | -0.044 | **0.037** | **+0.081** | ⚠️ Still too low |

**Trajectory Analysis** (659 interactions, 10 students):

**❌ CHECK 1: Skill Differentiation - FAILED**
```
Gain statistics:
  Min: 0.5834
  Max: 0.5934
  Mean: 0.5884
  Std: 0.0015 ← Nearly constant!
  Range: 0.01 (0.583 to 0.593)
```

**Critical Finding**: Gains are nearly uniform across all skills. The model converged to a near-constant gain value (~0.588) rather than learning skill-specific patterns.

**Other Checks**:
- ✅ CHECK 2: Q-Matrix Learning (1.0 skills/interaction) - PASS
- ✅ CHECK 3: Monotonicity (0 violations) - PASS
- ❌ CHECK 4: Mastery Correlation (0.033, target >0.3) - FAIL
- ❌ CHECK 5: Skill Difficulty (CV=0.0016, target >0.2) - FAIL
- ❌ CHECK 6: Response-Conditional (ratio=1.00, target >1.2) - FAIL

**Success Rate**: 2/6 trajectory checks (33%)

### Phase 2: V2 Enhancements (Multiple Interventions)

**Experiment**: 20251117_162330_gainakt3exp_fixed-per-skill-gains-v2_820618

**Objective**: Address V1's uniform gains problem with 4 targeted interventions

**Changes Made**:

**Priority 1: Increase IM Loss Weight**
- Changed `bce_loss_weight` 0.7 → 0.5
- Makes IM loss 50% instead of 30%
- **Rationale**: Stronger gradient signal for Encoder2

**Priority 2: Add Variance Loss**
- Added `variance_loss_weight = 0.1`
- Formula: `-skill_gains.var(dim=-1).mean()` (maximize variance)
- **Rationale**: Explicitly penalize uniform gains

**Priority 3: Increase Training Epochs**
- Changed `epochs` 12 → 20 (early stopped at 15)
- **Rationale**: More time to escape uniform solution

**Priority 4: Layer-wise Learning Rates**
- `gains_projection` LR: 0.000174 → 0.000522 (3x boost)
- All other parameters: 0.000174 (baseline)
- **Rationale**: Stronger gradient for projection layer

**Results**:

| Metric | V1 | V2 | Change | Status |
|--------|----|----|--------|--------|
| **Encoder2 Val AUC** | 0.5868 | **0.5969** | **+1.0%** | ⚠️ Marginal |
| **Encoder2 Test AUC** | 0.5835 | **0.5931** | **+1.0%** | ⚠️ Marginal |
| **Gain Std** | 0.0015 | **0.0017** | +0.0002 | ❌ Negligible |
| **Gain CV** | 0.0016 | **0.0019** | +0.0003 | ❌ Negligible |
| **Mastery Correlation** | 0.037 | **0.113** | **+0.076 (3x)** | ⚠️ Better but still << 0.3 |
| **Trajectory Checks** | 2/6 (33%) | 3/6 (50%) | +1 check | ⚠️ Slight improvement |

**Trajectory Analysis** (659 interactions, 10 students):

```
Gain statistics:
  Min: 0.5802
  Max: 0.5907
  Mean: 0.5854
  Std: 0.0017 ← Still nearly constant!
  Range: 0.0105
```

**Critical Finding**: Despite ALL V2 interventions, gains remain 99.8% uniform. The model still converges to the same degenerate solution.

**Training Dynamics**:
- Best epoch: 5 (early in training)
- Early stopped at 15 (patience=10)
- Encoder2 AUC peaked at 0.597 (epoch 6), then declined
- Pattern: Gains converged to uniform quickly, more epochs didn't help

**Success Rate**: 3/6 trajectory checks (50%) - slight improvement but insufficient

### Why V2 Failed: Root Cause Analysis

**Hypothesis 1: Insufficient Gradient Pressure** (Most Likely)
- Even with 50% IM weight (up from 30%), gains remain uniform
- Variance loss weight (0.1) too weak to overcome uniform solution
- Layer-wise LR boost (3x) can't escape initialization

**Evidence**:
- Gains uniform from epoch 1 (std=0.0015 at convergence)
- Encoder2 AUC stable 58.5-59% throughout training (not degrading)
- Pattern persists across train/val/test (it's the learned solution, not memorization)

**Conclusion**: The gradient signal through IM loss + variance loss is STILL too weak. The model finds a local minimum at uniform gains that satisfies both losses "well enough" without differentiating skills.

**Hypothesis 2: Conflicting Objectives**
- BCE loss (50%) pushes for direct prediction accuracy
- IM loss (50%) pushes for mastery-based prediction
- Model balances by using uniform gains (~0.6) which:
  - Creates *some* mastery progression (satisfies IM loss)
  - Doesn't interfere with Encoder1 (satisfies BCE loss)

**Conclusion**: Mixed signals from start allow model to find compromise solution.

**Why Parameter Tuning Won't Fix This**:

❌ **Won't work**:
- Increase IM loss weight to 60-70% → Still allows compromise
- Adjust variance loss weight to 0.5 → May conflict with BCE+IM
- Lower threshold → Doesn't add differentiation capability
- Add more encoder capacity → More capacity for wrong mechanism

✅ **Required fix**:
- **V3 Inverse Warmup**: Force 100% IM early (no compromise possible)
- This is an optimization trajectory fix, not a hyperparameter fix

---



## 🔴 BUG 0 (2025-11-17) - Skill Index Mismatch  

**Issue**: encoder2_pred Skill Index Mismatch  
**Severity**: CRITICAL - Affects training supervision and interpretability claims  
**Status**: ✅ FIXED in code, ⏳ RE-TRAINING REQUIRED

### Summary

Investigation revealed that `incremental_mastery_predictions` (encoder2_pred) was using mastery for the **NEXT** skill (`qry[t] = q[t+1]`) instead of the **CURRENT** skill (`q[t]`), causing:
- **22.2% prediction mismatches** (214 out of 966 predictions)
- **98.2% of mismatches at skill changes** (213 out of 217 skill transitions)
- Misaligned training supervision (predicting skill B with skill A labels)
- Invalid interpretability claims (encoder2_pred didn't represent current skill)

### Fix Applied

**File**: `pykt/models/gainakt3_exp.py`, Line 622

**Before** (WRONG):
```python
skill_indices = target_concepts.long()  # Uses qry[t] = q[t+1]
```

**After** (CORRECT):
```python
skill_indices = q.long()  # Uses q[t] (current skill)
```

### Validation Results

- **Before fix**: 77.8% match rate, 22.2% mismatches
- **After fix**: 100.0% match rate, 0.0% mismatches ✅
- **Investigation files**: `tmp/INVESTIGATION_INDEX.md`, `tmp/encoder2_pred_root_cause_confirmed.md`

### Impact

1. **Old experiments** (trained before 2025-11-17): encoder2_pred values are for NEXT skill, interpretability claims INVALID
2. **New experiments** (trained after fix): encoder2_pred values are correct, interpretability claims VALID
3. **Performance**: Fix may improve AUC (incremental_mastery_loss now uses correct supervision)
4. **Reproducibility**: All old checkpoints affected consistently, relative comparisons still valid

### Current Status (2025-11-17)

**Code Status**: ✅ PRODUCTION READY
- Critical bug fix applied and validated (encoder2_pred using correct skill index)
- Dual-encoder architecture fully implemented (167,575 parameters)
- Differentiable effective practice enables Encoder 2 gradient flow
- All tests passing (gradient flow, architecture validation)

**Recent Training Runs**:
- `20251117_131554_gainakt3exp_baseline-bce0.7_999787` - ✅ COMPLETED (latest baseline with adjusted parameters)
- `20251117_120903_gainakt3exp_baseline-bce0.7_621422` - Initial post-fix baseline
- `20251117_020600_gainakt3exp_bugfix_909644` - Bug fix validation run

**Configuration Adjustments Post-Bug-Fix**:
- `bce_loss_weight`: 0.9 → 0.7 (increased IM loss from 10% to 30%)
- `patience`: 10 (increased from 4 for more stable training convergence)
- Rationale: Bug fix revealed Encoder 2 needs stronger gradient signal for effective learning

**Parameter Defaults** (configs/parameter_default.json):
- Learning curve: beta=2.0, m_sat=0.8, gamma=1.0, offset=2.0
- Threshold: mastery_threshold=0.85, temperature=1.0
- Architecture: d_model=256, n_heads=4, num_encoder_blocks=4
- Training: lr=0.000174, epochs=12, batch_size=64

### Experiment Results: 20251117_131554_gainakt3exp_baseline-bce0.7_999787

**Configuration**: Post-bug-fix baseline with bce_loss_weight=0.7 (30% IM loss), patience=10

**Final Results** (Epoch 12):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overall Val AUC** | 0.6765 | Reasonable performance (67.65%) |
| **Overall Val Acc** | 0.7402 | Good accuracy (74.02%) |
| **Encoder1 Val AUC** | 0.6765 | ✅ Performance path working well |
| **Encoder1 Val Acc** | 0.7402 | ✅ Matches overall metrics |
| **Encoder2 Val AUC** | 0.4842 | ❌ **CRITICAL: Below random baseline (48.42%)** |
| **Encoder2 Val Acc** | 0.7402 | Matches overall (but AUC problematic) |
| Val BCE Loss | 0.5366 | Reasonable |
| Val IM Loss | 0.6044 | Higher than BCE loss |

**Training Progression** (Encoder 2 AUC):

```
Epoch  Val_AUC  Enc1_Val_AUC  Enc2_Val_AUC  Notes
  1    0.6752    0.6752        0.4668       Encoder2 starts below random
  2    0.6861    0.6861        0.4704       Slight improvement
  3    0.6897    0.6897        0.4722       Gradual increase
  4    0.6887    0.6887        0.4736       
  5    0.6894    0.6894        0.4751       
  6    0.6879    0.6879        0.4763       
  7    0.6871    0.6871        0.4774       
  8    0.6844    0.6844        0.4788       
  9    0.6814    0.6814        0.4800       
 10    0.6800    0.6800        0.4813       
 11    0.6777    0.6777        0.4827       
 12    0.6765    0.6765        0.4842       Final: still below 50%
```

**Key Observations**:

1. **Encoder 1 (Performance Path)**: ✅ Working Correctly
   - Validation AUC: 67.65% (reasonable for knowledge tracing)
   - Matches overall model performance
   - Demonstrates performance optimization pathway is functional

2. **Encoder 2 (Interpretability Path)**: ❌ Critical Performance Issue
   - **AUC below 50% throughout training** (random baseline is 50%)
   - Mastery-based predictions perform worse than random guessing
   - Shows gradual improvement (46.68% → 48.42%) but never reaches random baseline
   - **Interpretation**: Mastery values are not meaningfully correlated with student responses

3. **Both Encoders - Same Accuracy**: 
   - Both achieve 74.02% accuracy despite vastly different AUC values
   - Suggests Encoder 2 is making confident predictions on class-imbalanced data
   - Accuracy misleading when dataset has imbalanced positive/negative responses

4. **Loss Component Analysis**:
   - IM Loss (0.6044) higher than BCE Loss (0.5366)
   - Despite 30% weight on IM loss, Encoder 2 not learning predictive patterns
   - Suggests issue is not gradient strength but signal informativeness

### Diagnosis: Why Encoder 2 AUC < 50%

**Learning Trajectories Analysis** (Completed: 2025-11-17 13:33):

Analyzed `learning_trajectories.csv` with 10 students from experiment. Key findings:

1. **✅ Mastery Values ARE Updating**: Show clear sigmoid progression
   - Example Student 1, Skill 14: 0.261 → 0.479 → 0.678 (3 interactions)
   - Example Student 1, Skill 92: 0.251 → 0.459 → 0.652 → 0.767 → 0.819 (5 interactions)
   - Demonstrates sigmoid learning curve mechanism is working

2. **❌ Expected Gains Are ALL ZERO**: Critical discovery!
   - Column `expected_gain` shows 0.000000 for all 1000+ trajectory rows
   - Root cause: `use_gain_head=false` in config
   - Model computes gains internally but doesn't output them for trajectory analysis
   - Gains are implicit in effective_practice accumulation, not explicit

3. **Encoder 2 Architecture Uses Implicit Gains**:
   ```
   value_seq_2 (Encoder 2 output, D-dim) 
     → gain_quality = sigmoid(mean(value_seq_2))  [scalar per timestep]
     → effective_practice[t] = effective_practice[t-1] + gain_quality[t]
     → mastery[t] = M_sat × sigmoid(β × γ × effective_practice[t] - offset)
     → encoder2_pred = sigmoid((mastery - θ) / temp)
   ```
   
4. **Mastery Updates Through Effective Practice**:
   - NOT through explicit per-skill gains like gain_quality × skill_one_hot
   - INSTEAD: quality-weighted practice count accumulates
   - gain_quality is a **scalar engagement measure** (0-1), not per-skill gain vector
   - All practiced skills at timestep t get same gain_quality increment

**Root Cause Identified** - TWO CRITICAL ARCHITECTURAL BUGS: `Scalar Gain Quality` and `Insufficient Differentiation in Skill-Specific Gain Learning`.

---

## 🔴 BUG #1: Scalar Gain Quality (Not Per-Skill Gains)

**File**: `pykt/models/gainakt3_exp.py`, Lines 528-554

```python
# WRONG: Aggregates to scalar per interaction
gain_quality = torch.sigmoid(learning_gains_d.mean(dim=-1, keepdim=True))  # [B, L, 1]

# WRONG: Same scalar applied to ALL practiced skills
effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
```

**The Problem**:
- ✅ Model can learn: "This interaction had high/low engagement quality"
- ❌ Model cannot learn: "Skill A improved by 0.8, Skill B by 0.2"
- Result: All practiced skills increment by SAME scalar value
- Impact: No skill differentiation, uniform mastery growth
- Reality: Real learning is skill-specific with different rates

---

## 🔴 BUG #2: Insufficient Differentiation in Skill-Specific Gain Learning

**File**: `examples/train_gainakt3exp.py`, Lines 285-291

```python
# Encoder 1: Direct response prediction (CORRECT)
bce_loss = bce_criterion(y_pred, y_true)

# Encoder 2: Response prediction through mastery (CORRECT IN PRINCIPLE)
im_loss = bce_criterion(valid_im_preds, y_true)  # Same y_true labels

# Both encoders trained on SAME objective
loss = bce_loss_weight * bce_loss + incremental_mastery_loss_weight * im_loss
```

**The Problem - Clarified**:

The supervision signal (BCE on responses) is **CORRECT** - Encoder 2 SHOULD predict responses through mastery. The issue is that **Bug #1 (scalar gains) prevents Encoder 2 from learning skill-specific patterns**, so even with correct supervision, it cannot differentiate between skills.

**Why Both Encoders Learning Response Prediction is Actually Correct**:
- ✅ Encoder 1: Direct response prediction (unconstrained, performance-optimized)
- ✅ Encoder 2: Response prediction **through skill-specific mastery** (constrained, interpretable)
- ✅ Both predict responses, but through DIFFERENT mechanisms:
  - Encoder 1: Attention patterns → Direct prediction
  - Encoder 2: Attention patterns → Skill gains → Mastery → Prediction

**The Real Issue (Consequence of Bug #1)**:
- Current: Scalar gain_quality means all skills get same increment
- Result: Encoder 2 cannot learn "Skill A improved 0.8, Skill B improved 0.2"
- Impact: Without skill differentiation, mastery values are meaningless
- Outcome: Encoder 2 becomes a poor response predictor (AUC 48.42%)

**Why This Causes AUC < 50%**:
- Encoder 1: Unconstrained predictor → Good AUC (67.65%)
- Encoder 2: Tries to predict through mastery BUT:
  - Scalar gains → uniform mastery growth → no skill differentiation
  - Monotonic mastery + threshold mechanism → predictions compressed near 0.5
  - Cannot learn: "High mastery in relevant skills → correct response"
  - Can only learn: "High engagement → correct response" (wrong correlation)
- Result: Engagement anti-correlates with correctness → AUC < 50%

**Refined Understanding**:

Bug #2 is not really about the loss function - **BCE(encoder2_pred, y_true) is CORRECT**. The bug is that **without per-skill gains (Bug #1), Encoder 2 cannot learn the skill-specific mastery mechanism needed to predict responses accurately**.

Once Bug #1 is fixed:
- Encoder 2 can learn: "This interaction improved Skill A by 0.8, Skill B by 0.2"
- Mastery differentiates: mastery[A] = 0.9, mastery[B] = 0.5
- Response prediction becomes accurate: P(correct) = f(mastery[relevant_skills])
- AUC should improve to 55-60% (lower than Encoder 1 due to constraints, but above random)

**Why AUC < 50% Makes Sense Now** (Trajectory Analysis Results):

**Key Statistics from 659 interactions**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Encoder2_Pred Mean | 0.4773 | Predictions centered below 0.5 |
| Encoder2_Pred Range | [0.37, 0.53] | **Very narrow range (0.16)** |
| Mastery Mean | 0.6604 | High average mastery |
| Mastery Range | [0.22, 0.89] | Good variance (0.67) |
| Mastery ↔ Encoder2_Pred | **1.0000** | Perfect linear relationship (by design) |
| Mastery ↔ Response | **-0.0443** | ❌ **Mastery NOT predictive of performance** |
| Encoder2_Pred ↔ Response | **-0.0447** | ❌ **Predictions NOT correlated with truth** |
| Encoder2 Match Rate | 48.25% | **Below random baseline** |
| Threshold θ | 0.7529 | High threshold |
| Mastery > θ | 51.7% | Just over half exceed threshold |
| Dataset: Correct (r=1) | 69.8% | **Class imbalance** |
| Dataset: Incorrect (r=0) | 30.2% | Fewer failures |

**Root Cause Confirmed**:

1. **Mastery Has NO Correlation with Actual Performance** (-0.044):
   - The sigmoid learning curve produces mastery values that are **orthogonal to student performance**
   - Encoder 2 learns mastery patterns unrelated to correctness
   - Effective practice accumulation doesn't capture skill acquisition

2. **Encoder2_Pred Range Too Narrow** (0.37-0.53, std=0.054):
   - All predictions compressed near 0.5 (indecisive)
   - Cannot discriminate between high/low performance students
   - Temperature mechanism not providing sufficient spread

3. **Negative Correlation Trend** (-0.044):
   - Slight negative correlation means higher mastery → slightly LOWER performance
   - This is the opposite of educational logic!
   - Suggests Encoder 2 learning inverse of true skill progression

4. **Class Imbalance + Narrow Predictions**:
   - 69.8% correct responses (dataset bias)
   - Encoder2 predicts ~48% (below 50%) on average
   - With narrow range, cannot adjust to class distribution
   - Result: Misses majority class → low accuracy → AUC < 50%

5. **Threshold Miscalibration**:
   - θ = 0.7529 is high (only 51.7% exceed it)
   - But mastery mean = 0.66 (below threshold for many)
   - With (mastery - θ) / temp formula, predictions compressed near 0.5
   - Need lower θ (0.5-0.6) or higher temperature (2-3) for better spread

**Fundamental Issue**: The current architecture learns mastery trajectories that are **independent of actual student performance**. No amount of parameter tuning will fix this - the model needs skill-specific gains to capture meaningful learning patterns.

**How The Bugs Interact**:

The core issue is **Bug #1 (scalar gains)** prevents the intended architecture from working, even though the supervision signal is correct:

1. **Bug #1 (Scalar gains)**: Encoder 2 can only learn uniform engagement, not skill-specific learning
2. **Consequence**: Without skill differentiation, mastery values become meaningless
3. **Result**: Even with correct supervision (BCE on responses), Encoder 2 cannot learn proper mastery-response relationship
4. **Failure Mode**: Encoder 2 learns "engagement ≈ mastery" instead of "skill-specific practice → skill-specific mastery"

**Example Failure Mode**:
- Question targeting Skill 14: Student engages deeply (gain_quality = 0.8)
- Current (WRONG): Both Skill 14 AND Skill 92 increase by 0.8 (uniform increment)
- Intended (CORRECT): Only Skill 14 increases by 0.8, Skill 92 unchanged
- Impact: Cannot learn which skills are relevant for which questions
- Result: Mastery becomes correlated with overall engagement, not skill-specific practice

**Why Parameter Tuning Cannot Fix This**:

❌ **Won't work**:
- Increase IM loss weight to 50%+ → Doesn't add skill differentiation capability
- Adjust sigmoid parameters → Still scalar gains (no per-skill learning)
- Lower threshold → Doesn't fix uniform mastery growth
- Add more encoder capacity → More capacity for wrong mechanism

✅ **Required fix**:
- Implement per-skill gain vectors [B, L, num_c] (Bug #1)
- This enables: Question → Relevant Skills → Skill-Specific Gains → Skill-Specific Mastery → Response
- No change needed to loss function (BCE is correct for mastery-based response prediction)

### Recommended Fixes

**Primary Bug Identified**: 
1. **Scalar gains** prevent skill-specific learning (Bug #1) - **CRITICAL**
2. **Insufficient differentiation** (Bug #2) is a consequence of Bug #1, not separate issue

**Refined Understanding**:
- The current loss function (BCE on mastery-based predictions) is **CORRECT**
- The issue is that scalar gains prevent Encoder 2 from learning skill-specific patterns
- Once Bug #1 is fixed, the existing supervision should work properly

See detailed analysis in: `tmp/ARCHITECTURAL_BUGS_ANALYSIS.md`

**Required Fix**:

**FIX #1: Implement Per-Skill Gains Vector** (Priority: CRITICAL)

**Current** (Lines 528-554 in `pykt/models/gainakt3_exp.py`):
```python
# WRONG: Scalar per interaction
gain_quality = torch.sigmoid(learning_gains_d.mean(dim=-1, keepdim=True))  # [B, L, 1]
effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
```

**Required Fix**:
```python
# Add skill-specific projection in __init__
self.gains_projection = nn.Linear(d_model, num_c)  # [D=256] → [num_c]

# In forward_with_states, compute per-skill gains
skill_gains_logits = self.gains_projection(value_seq_2)  # [B, L, D] → [B, L, num_c]
skill_gains = torch.sigmoid(skill_gains_logits)  # [B, L, num_c] ∈ [0, 1]

# Accumulate skill-specific gains (each skill gets different increment!)
effective_practice[:, t, :] += skill_gains[:, t, :]  # Per-skill, differentiable
```

**Benefits**:
- ✅ Encoder 2 learns different learning rates per skill
- ✅ Maintains differentiability (gradients flow through gains_projection)
- ✅ Enables skill difficulty learning (β_skill parameters)
- ✅ Realistic skill-specific mastery trajectories
- ✅ Enables proper question → relevant skills → skill-specific gains flow
- ✅ Mastery-based response predictions become accurate

**Why This Fix Alone is Sufficient**:

Once per-skill gains are implemented, the existing loss function `BCE(encoder2_pred, y_true)` will naturally train Encoder 2 to:
1. Learn which skills are relevant for each question (from attention patterns)
2. Estimate appropriate learning gains for those specific skills
3. Build mastery values that correlate with response probability
4. Predict responses through mastery mechanism

No additional loss changes needed - the supervision signal is already correct.

**OPTIONAL ENHANCEMENT: Auxiliary Loss for Skill Differentiation** (If Needed)

If after implementing Fix #1 (per-skill gains), Encoder 2 still struggles to learn skill differentiation, consider adding an auxiliary loss to encourage variance in gains:

**Current Loss** (Lines 285-291 in `examples/train_gainakt3exp.py`):
```python
# Main loss: Mastery-based response prediction (CORRECT)
im_loss = bce_criterion(valid_im_preds, y_true)
```

**Optional Enhancement - Skill Differentiation Regularization**:
```python
# IM loss supervises MASTERY GROWTH based on response correctness
if use_mastery_head and 'projected_mastery' in outputs:
    mastery = outputs['projected_mastery']  # [B, L, num_c]
    
    # Compute mastery change per timestep
    mastery_delta = torch.zeros_like(mastery)
    mastery_delta[:, 1:, :] = mastery[:, 1:, :] - mastery[:, :-1, :]
    
    # Extract gains for practiced skills
    practiced_skills = questions.long()
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
    practiced_gains = mastery_delta[batch_indices, torch.arange(seq_len), practiced_skills]
    
    # Differential loss: correct responses should have HIGHER gains than incorrect
    correct_mask = (responses_shifted == 1) & valid_mask
    incorrect_mask = (responses_shifted == 0) & valid_mask
    
    if correct_mask.any() and incorrect_mask.any():
        gain_correct = practiced_gains[correct_mask].mean()
        gain_incorrect = practiced_gains[incorrect_mask].mean()
        
        # Margin-based ranking loss
        margin = 0.1
        im_loss = torch.relu(margin + gain_incorrect - gain_correct)
    else:
        im_loss = torch.tensor(0.0, device=questions.device)
```

**Required Fix - Option B (Correlation-Based Loss)**:
```python
# Simpler: Maximize correlation between mastery and performance
if use_mastery_head and 'projected_mastery' in outputs:
    mastery = outputs['projected_mastery']
    
    # Extract mastery for practiced skills
    practiced_skills = questions.long()
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
    practiced_mastery = mastery[batch_indices, torch.arange(seq_len), practiced_skills]
    
    # Pearson correlation
    valid_mastery = practiced_mastery[valid_mask]
    valid_responses = responses_shifted[valid_mask].float()
    
    mastery_centered = valid_mastery - valid_mastery.mean()
    responses_centered = valid_responses - valid_responses.mean()
    
    correlation = (mastery_centered * responses_centered).sum() / \
                  (torch.sqrt((mastery_centered**2).sum()) * 
                   torch.sqrt((responses_centered**2).sum()) + 1e-8)
    
    im_loss = 1.0 - correlation  # Minimize to maximize correlation
```

**Benefits**:
- ✅ Encoder 2 learns mastery patterns, NOT response prediction
- ✅ Restores encoder independence (different objectives)
- ✅ Mastery becomes meaningful learning trajectory
- ✅ Enables true interpretability

**Success Criteria for Next Iteration**:
- ✅ Encoder2 Val AUC > 55% (above random baseline)
- ✅ Gain values show variance (not flat or zero)
- ✅ Skill-specific gains differentiate learning rates
- ✅ encoder2_match rate > 60% in trajectory analysis
- ✅ Mastery correlates with responses (correlation > 0.4)

### Updated Task List

- [x] ✅ Re-train model with fixed code (Completed: 2025-11-17)
- [x] ✅ Analyze experiment results and metrics (Completed: 2025-11-17)
- [x] ✅ Document critical Encoder 2 performance issue (Completed: 2025-11-17)
- [x] ✅ Extract and analyze learning trajectories (Completed: 2025-11-17 13:33)
- [x] ✅ Diagnose mastery correlation issues (Completed: 2025-11-17)
  - **Finding**: Mastery ↔ Response correlation = -0.044 (NOT predictive)
  - **Root Cause**: Scalar gain_quality learns engagement, not skill-specific learning
  - **Conclusion**: Architecture needs skill-specific gains mechanism
- [ ] **[PRIORITY] Implement Skill-Specific Gains Architecture**:
  - Modify Encoder 2 to output per-skill gains [B, L, num_c] instead of scalar
  - Update effective_practice accumulation to use skill-specific gains
  - Maintain differentiability through Encoder 2
- [ ] Re-train with skill-specific gains and analyze mastery correlation
- [ ] If correlation improves, run parameter sweep for optimal IM loss weight
- [ ] Update paper with corrected interpretability results (once Encoder 2 fixed)

---

## 🔴 BUG 3 - ENCODER 2 NOT LEARNING MEANINGFUL PATTERNS (2025-11-17)

**Experiment**: `20251117_131554_gainakt3exp_baseline-bce0.7_999787`  

### The Problem

**Encoder 2 AUC = 48.42%** (below 50% random baseline) throughout 12 epochs of training.

### Root Cause Analysis

After comprehensive trajectory analysis (659 interactions, 10 students), we identified the fundamental architectural issue:

1. **Mastery Uncorrelated with Performance**: 
   - Correlation = -0.044 (essentially zero, slightly negative)
   - Mastery values don't predict student responses
   
2. **Scalar Gain Quality (Not Skill-Specific)**:
   - Encoder 2 outputs `gain_quality` ∈ [0, 1] (scalar per interaction)
   - All practiced skills get same quality increment
   - Cannot learn skill-specific learning rates
   
3. **Narrow Prediction Range**:
   - Encoder2_pred ∈ [0.37, 0.53] (std=0.054)
   - Cannot discriminate between students
   - Predictions compressed near 0.5 (indecisive)

4. **High Threshold Compression**:
   - θ = 0.7529, mastery mean = 0.66
   - Formula: `pred = sigmoid((mastery - θ) / temp)`
   - Results in predictions near 0.5 for most interactions

### Why Mastery Updates But Doesn't Predict

**Mastery IS updating** (e.g., 0.26 → 0.48 → 0.68 for skill 14), showing sigmoid progression works mechanically. However:

- ✅ Sigmoid curve computation: **WORKING**
- ✅ Effective practice accumulation: **WORKING**
- ❌ Skill-specific learning capture: **NOT WORKING**
- ❌ Correlation with performance: **NOT WORKING**

The model learns: *"Student engaged with content for X time"*  
Not: *"Student improved at skill A by amount X"*

### Required Architecture Change

**Current** (Broken):
```python
gain_quality = sigmoid(mean(value_seq_2))  # [B, L, 1] scalar
effective_practice[t, all_skills] += gain_quality[t]  # uniform increment
```

**Needed** (Fixed):
```python
skill_gains = sigmoid(linear(value_seq_2))  # [B, L, num_c] per-skill
effective_practice[t, skill] += skill_gains[t, skill]  # skill-specific increment
```

### Example Trajectories Showing the Issue

**Student 1, Skill 14** (3 interactions):
- Mastery: 0.26 → 0.48 → 0.68 (good sigmoid growth)
- Responses: 1 (correct) → 1 (correct) → 0 (incorrect)
- **Problem**: Mastery increases despite incorrect response!

**Student 2, Skill 7** (7 interactions):
- Mastery: 0.28 → 0.52 → 0.72 → 0.83 → 0.87 → 0.88 → 0.89
- Responses: 1 → 0 → 0 → 1 → 1 → 1 → 1
- Encoder2_pred: 0.38 → 0.44 → 0.49 → 0.52 → 0.53 → 0.53 → 0.53
- **Problem**: Predictions plateau at 0.53 (indecisive) despite high mastery

### Conclusion

**Parameter tuning will NOT fix this issue**. The architecture fundamentally cannot learn skill-specific patterns with scalar gain_quality. We need to implement per-skill gains to enable meaningful mastery learning.

**Next Steps**: 

1. **Implement Both Fixes** (See detailed code in `tmp/ARCHITECTURAL_BUGS_ANALYSIS.md`):
   - Fix #1: Replace scalar gain_quality with per-skill gains vector [B, L, num_c]
   - Fix #2: Replace response-prediction im_loss with mastery-growth-based loss
   
2. **Re-train and Validate**:
   ```bash
   # After fixes, re-run experiment
   python examples/run_repro_experiment.py --short_title fixed-architecture
   
   # Extract trajectories with gains
   python examples/learning_trajectories.py --run_dir <exp_dir> --num_students 10
   
   # Verify fixes worked:
   # - expected_gain column should have varying non-zero values
   # - Different skills should show different learning rates
   # - Mastery ↔ Response correlation should be positive (> 0.4)
   # - Encoder2 AUC should be above random (> 55%)
   ```

3. **Success Criteria**:
   - ✅ Per-skill gains vary by skill (not uniform)
   - ✅ Mastery ↔ Response correlation > 0.4 (was -0.044)
   - ✅ Encoder2 AUC > 55% (was 48.42%)
   - ✅ Skill differentiation visible in trajectories
   - ✅ Encoders learn independently (different objectives)

**Critical Understanding**: The current architecture **cannot be fixed by parameter tuning alone**. Both bugs are fundamental design flaws requiring code changes to the model architecture and training loop.


### Bug #1: Scalar Gain Quality (FIXED in V1)

**Discovery Date**: 2025-11-17  
**Severity**: CRITICAL - Encoder 2 cannot learn skill-specific mastery patterns

**The Bug**:
```python
# WRONG: Aggregates to scalar per interaction
gain_quality = torch.sigmoid(learning_gains_d.mean(dim=-1, keepdim=True))  # [B, L, 1]

# WRONG: Same scalar applied to ALL practiced skills
effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
```

**The Fix**:
```python
# CORRECT: Per-skill gains projection
self.gains_projection = nn.Linear(d_model, num_c)
skill_gains = torch.sigmoid(self.gains_projection(value_seq_2))  # [B, L, num_c]

# CORRECT: Skill-specific accumulation
effective_practice[:, t, :] += skill_gains[:, t, :]
```

**Impact**:
- ✅ Encoder2 AUC: 48% → 59% (above random)
- ❌ But gains still uniform (secondary issue)

### Bug #2: Loss Function Confusion (CLARIFIED - Not a Bug)

**Initial Misunderstanding**: "Encoder 2 trained to predict responses (wrong supervision signal)"

**Actual Correct Design**:
- ✅ Encoder 2 SHOULD predict responses through mastery mechanism
- ✅ Loss function BCE(encoder2_pred, y_true) is CORRECT
- ✅ Both encoders predict responses, but through different pathways:
  - Encoder 1: Direct attention → prediction (unconstrained)
  - Encoder 2: Attention → skill gains → mastery → prediction (interpretable)

**The Real Problem**: Without skill differentiation (Bug #1), mastery mechanism cannot work properly even with correct supervision.

---

## V1: Per-Skill Gains Fix (995130)

**Configuration**:
- bce_loss_weight: 0.7 (30% IM loss)
- epochs: 12
- Learning rate: 0.000174
- patience: 10

**Performance Metrics**:
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Encoder1 Val AUC | 0.6897 | ✅ Reasonable (69%) |
| Encoder2 Val AUC | 0.5868 | ⚠️ Above random but low (59%) |
| Test Encoder2 AUC | 0.5835 | ⚠️ Consistent with validation |
| Train Mastery Corr | 0.0350 | ❌ Very low (<< 0.3 target) |
| Test Mastery Corr | 0.0372 | ❌ Very low |

**Trajectory Analysis**:
- 659 interactions analyzed
- 10 students, 66 skills
- **Gain std**: 0.0015 (target >0.05) - 97% away from target
- **Mastery correlation**: 0.0331 (target >0.3) - 89% away from target

### V2: Multiple Interventions (820618)

**Configuration**:
- bce_loss_weight: 0.5 (50% IM loss) ← Changed from 0.7
- variance_loss_weight: 0.1 ← New
- epochs: 20 (early stopped at 15)
- Learning rate: 0.000174 (base), 0.000522 (gains_projection, 3x) ← Changed
- patience: 10

**Performance Metrics**:
| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| Encoder1 Val AUC | 0.6897 | 0.6893 | -0.0004 (≈ same) |
| Encoder2 Val AUC | 0.5868 | 0.5969 | **+1.0%** |
| Test Encoder2 AUC | 0.5835 | 0.5931 | **+1.0%** |
| Train Mastery Corr | 0.0350 | 0.0362 | +0.0012 (minimal) |
| Test Mastery Corr | 0.0372 | 0.0379 | +0.0007 (minimal) |

**Trajectory Analysis**:
- **Gain std**: 0.0017 (vs V1's 0.0015) - negligible improvement
- **Mastery correlation**: 0.1128 (vs V1's 0.0331) - **3x better** but still too low
- **Trajectory checks**: 3/6 passed (vs V1's 2/6)

**Training Progression** (V2):
```
Epoch  Enc1_Val  Enc2_Val  Notes
  1    0.679     0.585     Encoder2 starts
  5    0.704     0.598     Best epoch
  15   0.715     0.589     Final (early stopped)
```

**Key Observation**: Encoder2 peaked early (epoch 5-6), pattern established quickly.

### Comparison: V1 vs V2

**What Improved**:
1. ✅ Encoder2 AUC: +1.7% (58.7% → 59.7%)
2. ✅ Mastery correlation: 3x improvement (0.03 → 0.11)
3. ✅ Training stability: No gradient issues

**What Didn't Change**:
1. ❌ Skill differentiation: Gains still uniform (std: 0.0015 → 0.0017)
2. ❌ Response-conditional learning: Ratio still 1.00
3. ❌ Skill difficulty patterns: CV still ~0.002
4. ❌ Overall success criteria: 4/11 (36%) - same as V1

**Conclusion**: V2 improvements are MARGINAL. All interventions (50% IM, variance loss, layer-wise LR, 20 epochs) produced only minor gains. The core problem persists: gains converge to uniform solution.

---

## The Journey: From Bug Discovery to V2

**Initial Discovery (2025-11-17)**: Baseline experiment revealed Encoder 2 AUC = 48.42% (below random). Investigation identified **critical architectural bug**: scalar gain quality mechanism prevented skill-specific learning.

**V1 Fix (995130)**: Implemented per-skill gains architecture `[B, L, num_c]` instead of scalar `[B, L, 1]`. Result: Encoder 2 AUC improved 48%→59% ✅, but gains remained nearly uniform (std=0.0015) ❌.

**V2 Enhancement (820618)**: Implemented 4 priorities to encourage skill differentiation:
1. IM loss weight: 30%→50%
2. Variance loss: weight=0.1 (maximize gain variance)
3. Training epochs: 12→20
4. Layer-wise LR: gains_projection 3x boost

**V2 Results**: Marginal improvements only:
- Encoder2 AUC: 59.7% (+1.0% from V1)
- Mastery correlation: 0.113 (3x better than V1's 0.037, but still << 0.3 target)
- **Critical failure**: Gains still uniform (std=0.0017, negligible change from V1's 0.0015)

**Root Cause**: Even with all V2 interventions, the model converges to same degenerate solution (uniform gains ~0.585). The problem is **optimization trajectory**, not hyperparameters.

### Current Status

**What Works**:
- ✅ Per-skill gains architecture (architectural fix verified)
- ✅ Encoder2 above random baseline (59%)
- ✅ Training stability (no NaN, no explosions)
- ✅ Monotonicity preserved (mastery never decreases)

**What Fails**:
- ❌ Skill differentiation (std=0.0017, target >0.05) - 97% away from target
- ❌ Mastery correlation (0.11, target >0.3) - 63% away from target
- ❌ Response-conditional learning (ratio=1.00, target >1.2)

**Success Rate**: 4/11 verification checks passed (36%) - same as V1

### Recommendation: V3 Inverse Warmup

**Why V2 Failed**: Constant 50% IM weight is still a mixed signal. Model can compromise with uniform gains that satisfy both BCE (50%) and IM (50%) objectives.

**V3 Solution**: Eliminate compromise option in early training:
- **Phase 1 (epochs 1-15)**: 100% IM, 0% BCE → Force skill differentiation (no escape route)
- **Phase 2 (epochs 16-30)**: 70% BCE, 30% IM → Optimize performance (maintain patterns)

**Expected V3 Improvements**:
- Gain std: 0.0017 → **>0.10** (60x improvement)
- Mastery correlation: 0.11 → **>0.40** (4x improvement)  
- Encoder2 AUC: 59.7% → **>62%** (maintain above random)
- Success criteria: **8+/11 checks** (>70%)

---

## V3 Strategy: Inverse Warmup

### Why V3 Should Work

**V2 Showed**: IM weight matters (3x correlation improvement with 50% IM vs 30% IM)

**But**: 50% IM is still a compromise. Model can balance with uniform gains that satisfy:
- BCE loss: Encoder1 handles prediction
- IM loss: Uniform gains create *some* mastery progression

**V3 Eliminates Compromise**: 100% IM early = NO escape route = MUST differentiate skills

### Implementation

**Two-Phase Training Schedule**:

```python
def get_loss_weights(epoch, total_epochs=30):
    """
    Inverse warmup: Strong IM signal early → weak IM signal late
    
    Phase 1 (first 50% of epochs): Pure interpretability learning
        - IM weight = 1.0 (100%)
        - BCE weight = 0.0 (Encoder1 inactive)
        - Forces skill-specific gain learning
    
    Phase 2 (last 50% of epochs): Performance optimization
        - IM weight = 0.3 (30%)
        - BCE weight = 0.7 (Encoder1 dominant)
        - Maintains interpretability while optimizing accuracy
    """
    phase1_end = total_epochs // 2  # First 50% of epochs
    
    if epoch <= phase1_end:
        return {'bce_weight': 0.0, 'im_weight': 1.0}
    else:
        return {'bce_weight': 0.7, 'im_weight': 0.3}
```

**Example for 30 epochs**:
- Epochs 1-15: `bce=0.0, im=1.0` (pure interpretability - force skill learning)
- Epochs 16-30: `bce=0.7, im=0.3` (standard dual-encoder - optimize performance)

### Expected Benefits

**Phase 1 (Epochs 1-15)**:
- ✅ Forces skill differentiation early (no BCE escape route)
- ✅ Encoder2 MUST learn which skills are relevant for each question
- ✅ No compromise possible - gains must differentiate to reduce IM loss
- ✅ Builds skill-specific mastery patterns

**Phase 2 (Epochs 16-30)**:
- ✅ Encoder1 optimizes final performance
- ✅ Encoder2 maintains learned skill patterns (30% IM regularization)
- ✅ Best of both worlds: interpretability + performance

**Expected V3 Improvements**:
| Metric | V2 | V3 Target | Improvement Needed |
|--------|----|-----------|--------------------|
| Gain std | 0.0017 | **>0.10** | **60x** |
| Mastery correlation | 0.11 | **>0.40** | **4x** |
| Encoder2 AUC | 59.7% | **>62%** | +2.3% |
| Trajectory checks | 3/6 (50%) | **5/6 (83%)** | +2 checks |

### Why This Beats V2

**V2 Problem**: Mixed signals from start allow compromise
- 50% BCE + 50% IM from epoch 1
- Model finds uniform gains (~0.585) that satisfy both
- Gets stuck in this local minimum

**V3 Solution**: No compromise in Phase 1
- 100% IM, 0% BCE in epochs 1-15
- Model CANNOT use Encoder1 to handle prediction
- MUST differentiate skills to reduce loss
- Forces escape from uniform solution

**Then**: Maintain patterns in Phase 2
- 30% IM keeps learned skill differentiation
- 70% BCE optimizes final performance
- Best final AUC without losing interpretability

---

## Alternative Approaches

If V3 inverse warmup fails to achieve target metrics (gain std >0.1, mastery corr >0.4), consider these architectural modifications:

### Option 1: Multi-layer Gains Projection

**Current** (V1/V2): Single linear layer
```python
self.gains_projection = nn.Linear(d_model, num_c)
```

**Alternative**: Multi-layer with non-linearity
```python
self.gains_projection = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_model, num_c)
)
```

**Rationale**: More capacity to learn skill-specific patterns. The single linear layer may not have enough representational power to capture complex skill-response relationships.

**Expected Impact**:
- ✅ Better skill differentiation (more non-linear capacity)
- ✅ Higher gain variance
- ⚠️ May require more training data to avoid overfitting

### Option 2: Initialization Strategy

**Current**: Default PyTorch initialization
```python
# Default: nn.Linear uses Kaiming uniform initialization
```

**Alternative**: Smaller initialization variance
```python
nn.init.normal_(self.gains_projection.weight, mean=0.0, std=0.01)
nn.init.zeros_(self.gains_projection.bias)
```

**Rationale**: Prevent strong uniform bias at initialization. If initialized near uniform output (~0.5), gradient may be too weak to escape.

**Expected Impact**:
- ✅ Different initialization trajectory
- ✅ May escape uniform solution more easily
- ⚠️ May require more epochs to converge

### Option 3: Skill-Aware Attention Mechanism

**Current**: Standard multi-head attention
```python
attention_output = MultiHeadAttention(Q, K, V)
```

**Alternative**: Add skill-aware attention bias
```python
# Compute skill similarity matrix
skill_sim = compute_skill_similarity(skill_embeddings)  # [num_c, num_c]

# Add to attention scores
attention_scores = Q @ K.T + skill_sim[skill_idx_i, skill_idx_j]
attention_weights = softmax(attention_scores)
```

**Rationale**: Help model focus on skill-specific patterns by biasing attention toward related skills.

**Expected Impact**:
- ✅ Better skill-specific learning
- ✅ Improved gain differentiation
- ⚠️ Adds complexity and training time

### When to Try Alternatives

**Decision Tree**:
```
Run V3 inverse warmup (30 epochs)
  ├─ If gain std >0.10 and mastery corr >0.40: ✅ SUCCESS - No changes needed
  ├─ If gain std 0.05-0.10: Try Option 2 (initialization) with V3
  ├─ If gain std 0.02-0.05: Try Option 1 (multi-layer) with V3
  └─ If gain std <0.02: Try Option 3 (skill-aware attention) with V3
```

**Priority Order**:
1. **First**: V3 inverse warmup (most promising, minimal changes)
2. **Second**: V3 + Option 2 (simple, low risk)
3. **Third**: V3 + Option 1 (moderate complexity)
4. **Last**: V3 + Option 3 (high complexity, use only if others fail)

---

## Conclusions and Recommendations

### Key Findings

**Architectural Bug Fixed** ✅:
- Scalar gain quality → per-skill gains vector
- Encoder2 AUC improved 48% → 59% (above random)
- Gradient flow verified working

**V2 Interventions Insufficient** ❌:
- 50% IM loss: Marginal improvement (+1% AUC)
- Variance loss (0.1): Negligible impact on differentiation
- Layer-wise LR (3x): Helps but not enough
- 20 epochs: Pattern established early, more time didn't help

**Root Cause Identified**:
- Mixed signals (50% BCE + 50% IM) allow compromise
- Model finds uniform gains (~0.585) that satisfy both objectives
- Gets stuck in local minimum early (epoch 5-6)
- No amount of hyperparameter tuning will fix optimization trajectory

### Immediate Recommendations

**1. Implement V3 Inverse Warmup** (HIGH PRIORITY):
- Two-phase training: 100% IM early (epochs 1-15) → 70% BCE late (epochs 16-30)
- Expected results: gain std >0.10, mastery corr >0.40, Encoder2 AUC >62%
- Timeline: 2-3 hours implementation + 40-60 min training
- Risk: Low (well-justified theoretically, builds on V2 findings)

**2. Validate Success Criteria**:
- Extract trajectories after V3 training
- Run 6 verification checks (same as V1/V2 analysis)
- Target: 5+/6 checks passed (83%)
- Key metrics: gain std, mastery correlation, Encoder2 AUC

**3. If V3 Succeeds** (gain std >0.10):
- ✅ Commit all V2 + V3 changes to repository
- ✅ Document success in STATUS_gainakt3exp.md
- ✅ Write paper section on inverse warmup strategy
- ✅ Run multi-seed validation (3 seeds)
- ✅ Test on other datasets (ASSIST2009, etc.)

**4. If V3 Fails** (gain std <0.05):
- Try Option 2: Smaller initialization variance + V3 schedule
- Then Option 1: Multi-layer gains projection + V3 schedule
- Last resort: Option 3: Skill-aware attention + V3 schedule
- Document all attempts for future reference

### Long-term Strategy

**Research Questions to Explore**:
1. What is the optimal loss weight schedule? (V3 uses 100%→30%, try other curves)
2. Can we pre-train Encoder2 on IM loss only, then fine-tune with dual loss?
3. How does curriculum learning (easy skills first) affect differentiation?
4. Can we use Q-matrix pre-initialization to guide gains_projection?

**Ablation Studies**:
- V3 vs V3+Option1 vs V3+Option2 vs V3+Option3
- Phase 1 length: 40% vs 50% vs 60% of total epochs
- Phase 2 IM weight: 20% vs 30% vs 40%

**Cross-Dataset Validation**:
- Once V3 succeeds on ASSIST2015, test on:
  - ASSIST2009 (10 folds)
  - ASSISTments Challenge 2012
  - Bridge to Algebra 2006
  - Algebra 2005

### Documentation Updates Needed

**After V3 Training**:
1. Update this STATUS document with V3 results
2. Update V2_RESULTS_ANALYSIS.md with comparison table
3. Create V3_RESULTS_ANALYSIS.md if needed
4. Update paper/STATUS_gainakt3exp.md with unified findings
5. Archive tmp/ analysis files (ARCHITECTURAL_BUGS_ANALYSIS.md, FIX_VERIFICATION_REPORT.md, V2_RESULTS_ANALYSIS.md)

**For Paper**:
1. Write "Inverse Warmup Strategy" section
2. Compare V1, V2, V3 with ablation studies
3. Discuss optimization trajectory vs hyperparameter tuning
4. Highlight interpretability-performance trade-off

---

## Next Steps

**Immediate** (within 24 hours):
- [ ] Implement V3 inverse warmup loss schedule in train_gainakt3exp.py
- [ ] Update parameter_default.json with epochs=30
- [ ] Launch V3 training: `python examples/run_repro_experiment.py --short_title fixed-per-skill-gains-v3-inverse-warmup`
- [ ] Monitor training for skill_gains std in debug logs (should increase in Phase 1)

**Short-term** (within 1 week):
- [ ] Analyze V3 results (trajectories, verification checks)
- [ ] Compare V1 vs V2 vs V3 across all metrics
- [ ] If V3 succeeds: commit changes, write paper section
- [ ] If V3 fails: implement Option 2 (initialization) + V3

**Medium-term** (within 1 month):
- [ ] Multi-seed validation (3 seeds)
- [ ] Cross-dataset validation (ASSIST2009, etc.)
- [ ] Ablation studies (schedule variations)
- [ ] Paper draft with complete results

---

**Experiment Directories**:
- V0 (Broken): `/workspaces/pykt-toolkit/examples/experiments/20251117_131554_gainakt3exp_baseline-bce0.7_999787`
- V1 (Fixed): `/workspaces/pykt-toolkit/examples/experiments/20251117_154349_gainakt3exp_fixed-per-skill-gains_995130`
- V2 (Enhanced): `/workspaces/pykt-toolkit/examples/experiments/20251117_162330_gainakt3exp_fixed-per-skill-gains-v2_820618`
- V3 (Pending): TBD

**Document Version**: 2025-11-17  
**Last Updated**: After V2 analysis completion


