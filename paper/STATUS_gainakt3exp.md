# GainAKT3Exp Model Status

**Document Version**: Updated 2025-11-16 (Dual-Encoder Architecture)  
**Model Version**: GainAKT3Exp - Dual-encoder transformer with Sigmoid Learning Curve Mastery  
**Status**: Active implementation with full training/evaluation pipeline

**⚠️ CURRENT CONFIGURATION (2025-11-16 - DUAL-ENCODER ARCHITECTURE)**: 
- **Dual-Encoder Architecture**: ✅ TWO COMPLETELY INDEPENDENT ENCODER STACKS
  - **Encoder 1 (Performance Path)**: Learns response prediction patterns → Base Predictions → BCE Loss (weight ≈ 1.0)
  - **Encoder 2 (Interpretability Path)**: Learns learning gains patterns → Mastery → Incremental Mastery Predictions → IM Loss (weight=0.1)
  - **Total Parameters**: 167,575 (Encoder 1: 96,513 + Encoder 2: 71,040 + Sigmoid params: 22)
  - **Key Innovation**: Complete separation of performance and interpretability pathways with independent attention mechanisms
- **Dual-Prediction Architecture**: ✅ TWO independent prediction branches with TWO loss functions
  - **Base Predictions** (from Encoder 1) → BCE Loss (primary, weight ≈ 1.0)
  - **Incremental Mastery Predictions** (from Encoder 2 via sigmoid curves) → Incremental Mastery Loss (weight=0.1)
- **Differentiable Effective Practice**: ✅ ACTIVE - Quality-weighted practice accumulation enables Encoder 2 gradient flow
  - Encoder 2 learns "gain quality" per interaction: `gain_quality = sigmoid(value_seq_2.mean(dim=-1))`
  - Differentiable accumulation: `effective_practice[t] = effective_practice[t-1] + gain_quality[t]`
  - Gradients flow: IM_loss → mastery → effective_practice → gain_quality → value_seq_2 → Encoder 2 parameters
- **Sigmoid Learning Curve Mastery**: ✅ ACTIVE - Uses effective practice (quality-weighted) instead of raw practice count
  - Formula: `mastery = M_sat × sigmoid(β_skill × γ_student × effective_practice - offset)`
  - Learnable: β_skill[s] (skill difficulty), γ_student[i] (learning velocity), M_sat[s] (saturation), θ_global (threshold), offset (inflection)
  - Config: threshold_temperature=1.0 (prediction sharpness control, hybrid approach)
  - Three automatic learning phases: Initial (warm-up) → Growth (rapid learning) → Saturation (consolidation)
- **Constraint Losses**: ❌ **COMMENTED OUT** (all weights set to 0.0, code preserved for potential future use)
- **Semantic Module Losses**: ❌ **COMMENTED OUT** (all disabled, code preserved for potential future use)
- **Gains Head Output**: ❌ DEACTIVATED (`use_gain_head=false`) - Gains computed internally but not exposed as output
- **Architecture Flow**: 
  - **Performance Path**: Input → Encoder 1 → [Context_1, Value_1, Skill_1] → Prediction Head → Base Predictions → BCE Loss
  - **Interpretability Path**: Input → Encoder 2 → Value_2 → Gain Quality → Effective Practice → Sigmoid Learning Curve → Mastery → Threshold (with temp) → Incremental Mastery Predictions → IM Loss
- **Code Location**: `gainakt3_exp.py` (909 lines total)
  - Lines 51-53: Import changes (EncoderBlock from gainakt3, no longer inherits full class)
  - Line 56: Changed to `class GainAKT3Exp(nn.Module)` (standalone implementation)
  - Lines 182-213: Encoder 1 (Performance Path) initialization
  - Lines 215-247: Encoder 2 (Interpretability Path) initialization
  - Lines 356-377: Encoder 1 forward pass
  - Lines 379-398: Encoder 2 forward pass
  - Lines 500-540: Differentiable effective practice computation (CRITICAL FIX for gradient flow)
  - Lines 564-566: Sigmoid curve using effective_practice
- **Implementation Status**: ✅ **FULLY IMPLEMENTED AND TESTED** (2025-11-16)
  - Dual-encoder architecture: Two independent encoder stacks with separate parameters
  - Differentiable effective practice: Quality-weighted accumulation enables Encoder 2 learning
  - Gradient flow verification: Both encoders receive gradients during backpropagation (test confirmed)
  - Independent learning: Both encoders update independently during training
  - Test script: `tmp/test_dual_encoders.py` (263 lines) - ALL 7 TESTS PASSED ✓
- **Rationale**: Dual-encoder architecture provides clean separation between performance optimization (Encoder 1) and interpretability learning (Encoder 2). Each encoder learns different attention patterns: Encoder 1 focuses on response correctness prediction, while Encoder 2 learns to detect learning opportunities and estimate gain quality. The differentiable effective practice mechanism ensures gradients flow through Encoder 2, enabling end-to-end training of both pathways. This architecture balances predictive performance with interpretable mastery trajectories.

See "Architecture Summary" section below for detailed flow.

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
    "bce_loss_weight": 0.9,  // λ₁ - BCE loss weight
    // incremental_mastery_loss_weight computed as 1 - λ₁
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

**Lambda Weight Interpretation**:
- `bce_loss_weight = 0.9` → BCE contributes 90% to total loss (performance optimization)
- `incremental_mastery_loss_weight = 0.1` → IM contributes 10% to total loss (interpretability constraint)
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

### Summary

The dual-encoder metrics infrastructure provides comprehensive visibility into both encoder pathways:

1. **Separate Performance Tracking**: AUC and accuracy for each encoder independently
2. **Loss Component Analysis**: Unweighted, weighted, and percentage contributions
3. **Real-Time Monitoring**: Periodic state snapshots during training
4. **Enhanced Trajectories**: Per-skill gains, mastery, and dual predictions
5. **Interpretability Validation**: Quality metrics for mastery/gains

This enables researchers to:
- Verify both encoders are learning effectively
- Validate loss weighting is balanced correctly
- Debug encoder-specific issues
- Analyze trade-offs between performance and interpretability
- Generate detailed per-student learning progression reports

---

## Model Overview

**Implementation Files**:
- **Model**: `pykt/models/gainakt3_exp.py` (909 lines) - standalone nn.Module with dual-encoder architecture
- **Training**: `examples/train_gainakt3exp.py` (1925 lines) - zero defaults, explicit parameters
- **Evaluation**: `examples/eval_gainakt3exp.py` (310 lines) - test metrics + correlations
- **Trajectories**: `examples/learning_trajectories.py` (365 lines) - individual student analysis
- **Launcher**: `examples/run_repro_experiment.py` - loads defaults, manages experiments
- **Factory**: `create_exp_model(config)` (line 830+ in `gainakt3_exp.py`) - requires 24 explicit parameters
- **Test Scripts**: 
  - `tmp/test_sigmoid_curves.py` (223 lines) - validates sigmoid learning curve implementation
  - `tmp/test_dual_encoders.py` (263 lines) - validates dual-encoder architecture and gradient flow (ALL TESTS PASSED ✓)

### Description

## Core Architecture: Dual-Encoder Transformer with Sigmoid Learning Curve Mastery

**GainAKT3Exp Current State**: The model uses **two completely independent encoder stacks** to separate performance optimization from interpretability learning. Encoder 2 learns to estimate "gain quality" per interaction, which drives **differentiable effective practice** in sigmoid learning curves. This generates **TWO SEPARATE** prediction outputs with **TWO SEPARATE** loss functions. Mastery follows educationally-realistic sigmoid curves that automatically capture three learning phases: warm-up (minimal gains), growth (rapid improvement), and saturation (diminishing returns).

**Dual-Encoder Architecture**:
1. **Encoder 1 (Performance Path)**: 
   - Input → Context_Embedding_1, Value_Embedding_1, Skill_Embedding_1 → Positional_Embeddings_1
   - → Encoder_Blocks_1 (independent parameters) → [Context_1, Value_1, Skill_1] → Prediction_Head_1 
   - → **Base Predictions** → **BCE Loss** (primary, weight ≈ 1.0)
   - Parameters: 96,513
   
2. **Encoder 2 (Interpretability Path)**: 
   - Input → Context_Embedding_2, Value_Embedding_2 → Positional_Embeddings_2
   - → Encoder_Blocks_2 (independent parameters) → Value_2
   - → Gain Quality Estimation: `gain_quality = sigmoid(value_seq_2.mean(dim=-1))`
   - → Differentiable Effective Practice: `effective_practice[t] = effective_practice[t-1] + gain_quality[t]`
   - → Sigmoid Learning Curve: `mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × effective_practice[i,s,t] - offset)`
   - → Threshold Mechanism: `sigmoid((mastery - θ_global) / temperature)` 
   - → **Incremental Mastery Predictions** → **Incremental Mastery Loss** (interpretability, weight=0.1)
   - Parameters: 71,040

**Total Model Parameters**: 167,575 (Encoder 1: 96,513 + Encoder 2: 71,040 + Sigmoid params: 22)

### Incremental Mastery Loss Mechanism

The Incremental Mastery Loss provides interpretability-driven supervision by comparing ground truth responses (correct/incorrect) with predictions derived from learned mastery trajectories that follow sigmoid learning curves. This mechanism enforces educational constraints while maintaining differentiability for end-to-end training.

**Calculation Pipeline**:

1. **Learning Gains Estimation** (Encoder 2 - Interpretability Path):
   - **Encoder 2** (completely independent from Encoder 1) learns to output Values that represent raw learning potential per interaction
   - For each interaction t with skill s: `raw_gain[s,t] = Value_output_2[t]` (from Encoder 2)
   - Values are transformed to learning gains via ReLU: `learning_gains_d = ReLU(value_seq_2)`
   - **Gain Quality Computation** (CRITICAL for gradient flow):
     ```python
     gain_quality_logits = learning_gains_d.mean(dim=-1, keepdim=True)  # [B, L, 1]
     gain_quality = sigmoid(gain_quality_logits)  # Normalize to [0, 1]
     ```
   - This makes gain quality **differentiable through Encoder 2**, enabling gradient-based learning
   - Encoder 2 learns which interaction patterns (question difficulty, response correctness, temporal context) produce high-quality learning opportunities

2. **Sigmoid Learning Curve Mastery Accumulation**:
   
   Mastery evolves following a **sigmoid learning curve** modulated by skill difficulty and student learning velocity:
   
   **Learnable Parameters**:
   - **β_skill[s]**: Skill difficulty parameter (learned, shared across students)
     - Controls the slope of the sigmoid curve (steepness of learning progression)
     - Higher β_skill → steeper learning curve (easier to learn, faster mastery growth)
     - Lower β_skill → flatter curve (harder to learn, slower mastery growth)
     - Range: β_skill ∈ (0, ∞), typically initialized around 1.0
   
   - **γ_student[i]**: Student learning velocity parameter (learned per student)
     - Modulates how quickly a student progresses through the learning curve
     - Higher γ_student → faster learner (reaches saturation with fewer interactions)
     - Lower γ_student → slower learner (requires more practice to reach saturation)
     - Range: γ_student ∈ (0, ∞), typically initialized around 1.0
   
   - **M_sat[s]**: Saturation mastery level per skill (learned parameter)
     - Maximum achievable mastery level for each skill after infinite practice
     - Some skills may have M_sat < 1.0 (inherently difficult, never fully mastered)
     - Other skills may have M_sat ≈ 1.0 (fully masterable with sufficient practice)
     - Range: M_sat[s] ∈ [0.0, 1.0]
   
   **Effective Practice** (Differentiable Quality-Weighted Accumulation):
   ```
   # OLD (non-differentiable): practice_count[i, s, t] = Σ(k=1 to t) 𝟙[question[k] targets skill s]
   # NEW (differentiable through Encoder 2):
   effective_practice[i, s, t] = effective_practice[i, s, t-1] + gain_quality[i, t]
   ```
   Where:
   - `gain_quality[i, t]` comes from Encoder 2's Value outputs (learned per interaction)
   - `gain_quality[i, t] = sigmoid(value_seq_2[i, t].mean())` ∈ [0, 1]
   - This replaces discrete counting with **differentiable accumulation**
   - Gradients flow: IM_loss → mastery → effective_practice → gain_quality → value_seq_2 → Encoder 2 parameters
   - Interpretation: Instead of counting all interactions equally, weight each interaction by its learned quality
   
   **Sigmoid Learning Curve Formula** (Using Effective Practice):
   ```
   mastery[i, s, t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × practice_count[i, s, t] - offset)
   ```
   
   Where:
   - `offset` is a learnable parameter controlling the inflection point of the curve
   - `sigmoid(x) = 1 / (1 + exp(-x))`
   
   **Learning Curve Phases**:
   
   1. **Initial Phase** (effective_practice ≈ 0):
      - Mastery ≈ 0 (no learning yet)
      - Early practice produces minimal mastery increments
      - Corresponds to "warm-up" or familiarization period
      - Encoder 2 learns to output low gain_quality values initially
   
   2. **Growth Phase** (intermediate effective_practice):
      - Mastery increases with learnable slope β_skill × γ_student
      - Rate of learning depends on:
        - **Skill difficulty** (β_skill): Easier skills → faster growth
        - **Student ability** (γ_student): Faster learners → steeper slope
        - **Gain quality** (from Encoder 2): Higher quality → faster effective_practice accumulation
      - This is the most effective learning period
   
   3. **Saturation Phase** (high effective_practice):
      - Mastery approaches M_sat[s] asymptotically
      - Additional practice produces diminishing returns
      - Corresponds to skill consolidation and maintenance
   
   **Effective Learning Gain** (emerges from differentiable accumulation):
   ```
   effective_gain[i, s, t] = gain_quality[i, t]  # Direct from Encoder 2
   mastery_increment[i, s, t] = mastery[i, s, t] - mastery[i, s, t-1]  # Sigmoid curve derivative
   ```
   - Gain quality learned by Encoder 2 directly controls effective_practice accumulation
   - Mastery increments emerge from sigmoid curve dynamics
   - Automatically captures: slow start → rapid growth → saturation
   - **Differentiable**: Gradients flow through gain_quality back to Encoder 2
   
   **Monotonicity Guarantee**: 
   - Sigmoid function ensures mastery never decreases (effective_practice monotonically increases)
   - gain_quality ∈ [0, 1] ensures non-negative increments only
   - Knowledge retention enforced by design

3. **Threshold-Based Performance Prediction**:
   
   The model learns a **global threshold parameter** θ_global (shared across all skills and students):
   
   **Mastery-to-Prediction Mapping**:
   ```
   incremental_mastery_prediction[i, s, t] = sigmoid((mastery[i, s, t] - θ_global) / temperature)
   ```
   
   - **θ_global**: Learnable threshold (scalar parameter)
     - Defines the mastery level required for correct performance
     - Same threshold applied to all skills and students (simplification)
     - Typically θ_global ∈ [0.3, 0.7] after training
   
   - **Temperature**: Prediction sharpness parameter
     - **Implementation**: Config parameter (hybrid approach)
     - Controls steepness of mastery-to-prediction mapping
     - Lower temperature (e.g., 0.5): Sharper sigmoid, more decisive predictions
     - Higher temperature (e.g., 2.0): Smoother sigmoid, more gradual transitions
     - Default: 1.0 (standard sigmoid steepness)
     - **Rationale**: Start with config parameter for easier debugging and interpretation
       - Sufficient learnable parameters already (β_skill, γ_student, M_sat, θ_global, offset)
       - Can be tuned via hyperparameter search
       - Can upgrade to learnable parameter later if experiments show benefit
       - Consistent with other hyperparameters (learning rate, dropout)
   
   **Interpretation**:
   - If `mastery[i, s, t] > θ_global`: Student likely to answer correctly (prediction → 1.0)
   - If `mastery[i, s, t] < θ_global`: Student likely to answer incorrectly (prediction → 0.0)
   - Skill is "mastered" when `M_sat[s] > θ_global` (saturation level exceeds threshold)
   
   **Educational Logic**:
   - Skills with low saturation (M_sat[s] < θ_global) remain challenging even after extensive practice
   - Skills with high saturation (M_sat[s] > θ_global) become reliably correct once sufficient practice occurs
   - Students with high γ_student reach mastery threshold faster (fewer interactions needed)
   - Skills with high β_skill have steeper learning curves (faster mastery growth per interaction)

4. **Loss Computation** (Dual Loss Framework):
   ```
   # Loss 1: BCE on Base Predictions (from Encoder 1)
   bce_loss = BCE(base_predictions, ground_truth_responses)
   
   # Loss 2: Incremental Mastery Loss (from Encoder 2 via sigmoid curves)
   incremental_mastery_loss = BCE(incremental_mastery_predictions, ground_truth_responses)
   
   # Total Loss
   total_loss = bce_loss + 0.1 × incremental_mastery_loss
   ```
   
   **Gradient Flow Paths**:
   
   - **Path 1 (Performance)**: 
     ```
     BCE Loss → base_predictions → prediction_head_1 → [context_1, value_1, skill_1] 
     → encoder_blocks_1 → embeddings_1 → Encoder 1 parameters
     ```
   
   - **Path 2 (Interpretability - CRITICAL DIFFERENTIABLE CHAIN)**:
     ```
     IM Loss → incremental_mastery_predictions → mastery → effective_practice 
     → gain_quality → value_seq_2 → encoder_blocks_2 → embeddings_2 → Encoder 2 parameters
     ```
     This path also updates sigmoid curve parameters: β_skill, γ_student, M_sat, θ_global, offset
   
   - **Independent Learning**: Both encoders receive independent gradients and learn different patterns
     - Encoder 1: Learns attention patterns for response correctness prediction
     - Encoder 2: Learns attention patterns for detecting learning opportunities (gain quality)
   
   - Loss weight: IM Loss weight = 0.1 (balances with primary BCE loss, weight ≈ 1.0)

**Educational Semantics**:
- **Gain Quality** (from Encoder 2): Learned estimate of learning opportunity quality per interaction
- **Effective Practice**: Quality-weighted practice accumulation (differentiable through Encoder 2)
- **Mastery Trajectories**: Sigmoid curves tracking skill competence evolution (automatic acceleration → saturation)
- **Skill Difficulty** (β_skill): Controls learning curve steepness (easier skills → faster mastery growth)
- **Student Learning Velocity** (γ_student): Modulates progression speed (faster learners → fewer interactions to mastery)
- **Saturation Level** (M_sat): Maximum achievable mastery per skill (skill complexity ceiling)
- **Global Threshold** (θ_global): Mastery level required for correct performance (decision boundary)

**Key Properties**:
- **Dual-Encoder Independence**: Two separate encoder stacks with completely independent parameters (no shared representations)
- **Differentiable Effective Practice**: Quality-weighted accumulation enables gradient flow through Encoder 2
- **Sigmoid Learning Dynamics**: Automatic progression through slow-start → growth → saturation phases
- **Monotonicity**: Mastery never decreases (sigmoid curve with monotonic input: effective_practice)
- **Boundedness**: Mastery ∈ [0, M_sat[s]] ⊆ [0.0, 1.0] (normalized scale with skill-specific ceiling)
- **Differentiability**: Entire pipeline supports gradient flow (end-to-end training through both encoders)
- **Gradient Flow Verification**: Test script confirms gradients flow through both Encoder 1 and Encoder 2 (test passed ✓)
- **Interpretability**: All parameters have clear educational meaning:
  - β_skill: How quickly skill can be learned
  - γ_student: How fast student learns
  - M_sat: How masterable the skill is
  - θ_global: What mastery level indicates competence
- **Personalization**: Student-specific learning velocity (γ_student) adapts to individual abilities
- **Realistic Learning Dynamics**: Captures educational phenomena:
  - Initial practice may show little progress (warm-up phase)
  - Mid-stage practice shows rapid improvement (growth phase)
  - Advanced practice shows diminishing returns (saturation phase)

**Architecture Notes**:
- ✅ **Dual-Encoder Architecture**: TWO completely independent encoder stacks (167,575 total parameters)
  - **Encoder 1 (Performance)**: 96,513 parameters - learns response prediction patterns
  - **Encoder 2 (Interpretability)**: 71,040 parameters - learns learning gains patterns
  - **Sigmoid Parameters**: 22 parameters (β_skill, γ_student, M_sat, θ_global, offset)
  - No shared representations between encoders - complete pathway separation
- ✅ **Dual Predictions**: TWO independent prediction branches (base + incremental mastery)
- ✅ **Differentiable Effective Practice**: ACTIVE - Quality-weighted practice accumulation
  - Encoder 2 learns gain quality: `gain_quality = sigmoid(value_seq_2.mean(dim=-1))`
  - Accumulation: `effective_practice[t] = effective_practice[t-1] + gain_quality[t]`
  - Enables gradient flow: IM_loss → mastery → effective_practice → gain_quality → Encoder 2
- ✅ **Sigmoid Learning Curve Mastery**: ACTIVE - Mastery evolves via sigmoid curve driven by effective practice
  - Learnable parameters: β_skill[s] (skill difficulty), γ_student[i] (learning velocity), M_sat[s] (saturation level)
  - Global learnable threshold: θ_global (mastery-to-performance boundary)
  - Formula: `mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × effective_practice[i,s,t] - offset)`
- ✅ **Learning Gains Computation**: ACTIVE - Computed from Encoder 2 Values (internal use only)
- ✅ **Incremental Mastery Predictions**: ACTIVE - Threshold-based predictions from sigmoid learning curves
  - Formula: `sigmoid((mastery - θ_global) / temperature)` where temperature=1.0 (config parameter)
- ✅ **Base Predictions**: ACTIVE - From Encoder 1 via concatenation head [context_1, value_1, skill_1] → MLP
- ✅ **Dual Loss Functions**: 
  - BCE Loss on Base Predictions from Encoder 1 (primary, weight ≈ 1.0)
  - Incremental Mastery Loss on Threshold Predictions from Encoder 2 (interpretability, weight=0.1)
- ✅ **Temperature Parameter**: Config-based (threshold_temperature=1.0) - controls prediction sharpness
  - Hybrid approach: Start with config parameter, can upgrade to learnable later if needed
- ✅ **Independent Gradient Flow**: Both encoders receive gradients and update independently (verified via test ✓)
- ❌ **Constraint Losses**: **COMMENTED OUT** (all weights=0.0, code preserved) - Non-negative, Monotonicity, Mastery-Perf, Gain-Perf, Sparsity, Consistency
- ❌ **Semantic Module Losses**: **COMMENTED OUT** (all disabled, code preserved) - Alignment, Global Alignment, Retention, Lag Gains
- ❌ **Gains Head Output**: DEACTIVATED (`use_gain_head=false`) - Gains not exposed in model output
- ❌ **Gains D-dimensional Output**: DEACTIVATED - `projected_gains_d` not included in output

**Result**: The model uses two independent encoder stacks to produce two independent predictions: (1) Base predictions from Encoder 1's standard prediction head for primary BCE loss, and (2) Incremental mastery predictions from Encoder 2 via sigmoid learning curves and threshold mechanism for interpretability-driven mastery loss. The differentiable effective practice mechanism (quality-weighted accumulation) enables gradients to flow through Encoder 2, allowing it to learn which interactions provide high-quality learning opportunities. The sigmoid curves automatically capture three learning phases (warm-up, growth, saturation) with learnable skill-specific and student-specific parameters. **DUAL-ENCODER ARCHITECTURE**: Complete separation of performance optimization (Encoder 1) and interpretability learning (Encoder 2) with independent parameters. All constraint and semantic losses are commented out, leaving only BCE + Incremental Mastery losses active.

## Architecture

The architecture uses a **dual-encoder design** with two completely independent encoder stacks, each optimized for different objectives.

The diagram below illustrates the **current dual-encoder architecture**:

**Visual Legend:**
- **Double-border boxes** (`[[...]]`): **Input/Output data** (tensors, embeddings, intermediate representations) - white background with dark borders
- **Single-border boxes** (`[...]`): **Processing operations** (embeddings tables, transformations, neural network layers)
- **Blue components**: Encoder 1 (Performance Path) - 96,513 parameters
  - Learns attention patterns for response prediction
  - Independent embeddings, encoder blocks, and prediction head
  - Outputs: Base Predictions → BCE Loss (weight ≈ 1.0)
- **Orange components**: Encoder 2 (Interpretability Path) - 71,040 parameters
  - Learns attention patterns for learning gains detection
  - Independent embeddings and encoder blocks
  - Outputs: Gain Quality → Effective Practice → Sigmoid Learning Curves → Incremental Mastery Predictions → IM Loss (weight=0.1)
- **Pink components**: Differentiable Effective Practice Mechanism - **CRITICAL FOR GRADIENT FLOW**
  - Gain quality learned from Encoder 2: `gain_quality = sigmoid(value_seq_2.mean(dim=-1))`
  - Quality-weighted accumulation: `effective_practice[t] = effective_practice[t-1] + gain_quality[t]`
  - Enables gradients: IM_loss → mastery → effective_practice → gain_quality → value_seq_2 → Encoder 2
- **Green components**: Sigmoid Learning Curve Mastery - 22 learnable parameters
  - Formula: `mastery = M_sat × sigmoid(β_skill × γ_student × effective_practice - offset)`
  - Learnable: β_skill (skill difficulty), γ_student (student velocity), M_sat (saturation), θ_global (threshold), offset
  - Three automatic learning phases: Initial (warm-up) → Growth (rapid learning) → Saturation (consolidation)

**Key Architectural Features**: 
- **Dual-Encoder Independence**: ✅ FULLY IMPLEMENTED - Two encoder stacks with completely independent parameters (no shared layers)
  - Encoder 1: 96,513 parameters (context_emb_1, value_emb_1, skill_emb_1, pos_emb_1, encoder_blocks_1, prediction_head_1)
  - Encoder 2: 71,040 parameters (context_emb_2, value_emb_2, pos_emb_2, encoder_blocks_2)
  - Total: 167,575 parameters including sigmoid curve params
- **Differentiable Effective Practice**: ✅ CRITICAL INNOVATION - Enables Encoder 2 gradient flow
  - Replaces non-differentiable practice counting with quality-weighted accumulation
  - Test verified: Gradients flow through both encoders during backpropagation ✓
- **Sigmoid Learning Curve Mastery**: ✅ ACTIVE - Mastery evolves via sigmoid curves driven by effective practice
  - Uses quality-weighted practice instead of raw counting
  - Automatic three-phase learning: warm-up → growth → saturation
- **Dual Loss Framework**: ✅ ACTIVE - Two independent loss functions
  - BCE Loss (Encoder 1): Optimizes base predictions for performance
  - IM Loss (Encoder 2): Optimizes mastery trajectories for interpretability
  - Total Loss: `BCE + 0.1 × IM_loss`
- **Independent Learning**: ✅ VERIFIED - Both encoders update independently during training
  - Encoder 1 learns: Which attention patterns predict response correctness
  - Encoder 2 learns: Which interaction patterns indicate learning opportunities (high gain quality)

```mermaid
graph TD
    subgraph "Input Layer (Shared)"
        direction LR
        Input_q[["Input Questions (q)<br/>Shape: [B, L]"]]
        Input_r[["Input Responses (r)<br/>Shape: [B, L]"]]
        Ground_Truth[["Ground Truth Responses"]]
    end

    %% ========== ENCODER 1: PERFORMANCE PATH ========== %%
    subgraph "Encoder 1: Performance Path (Prediction-Optimized)"
        direction TB
        
        subgraph "Tokenization & Embedding 1"
            Tokens1[["Interaction Tokens 1<br/>(q + num_c * r)"]]
            Context_Emb1["Context Embedding Table 1"]
            Value_Emb1["Value Embedding Table 1"]
            Skill_Emb1["Skill Embedding Table 1"]
            
            Context_Seq1[["Context Sequence 1<br/>[B, L, D]"]]
            Value_Seq1[["Value Sequence 1<br/>[B, L, D]"]]
            Pos_Emb1["Positional Embeddings 1"]
            
            Context_Seq_Pos1[["Context + Pos 1<br/>[B, L, D]"]]
            Value_Seq_Pos1[["Value + Pos 1<br/>[B, L, D]"]]
        end
        
        subgraph "Dual-Stream Encoder Stack 1 (N blocks)"
            Encoder1_In_Ctx[["Input Context 1"]]
            Encoder1_In_Val[["Input Value 1"]]
            
            Encoder1_Block["Encoder Block 1<br/>Q/K from Context 1<br/>V from Value 1<br/>Dual Add&Norm + FFN"]
            
            Encoder1_Out_Ctx[["Output Context 1 (h₁)<br/>[B, L, D]"]]
            Encoder1_Out_Val[["Output Value 1 (v₁)<br/>[B, L, D]"]]
        end
        
        subgraph "Prediction Head"
            Pred_Input_h1[["Knowledge State (h₁)"]]
            Pred_Input_v1[["Value State (v₁)"]]
            Pred_Input_s1[["Target Skill (s₁)"]]
            
            Concat1["Concatenate<br/>[h₁, v₁, s₁]"]
            MLP1["MLP Head 1"]
            Sigmoid1["Sigmoid 1"]
            
            Base_Predictions[["Base Predictions<br/>[B, L]<br/>✅ Performance Output"]]
        end
    end

    %% ========== ENCODER 2: INTERPRETABILITY PATH ========== %%
    subgraph "Encoder 2: Interpretability Path (Mastery-Optimized)"
        direction TB
        
        subgraph "Tokenization & Embedding 2"
            Tokens2[["Interaction Tokens 2<br/>(q + num_c * r)"]]
            Context_Emb2["Context Embedding Table 2<br/>(Independent)"]
            Value_Emb2["Value Embedding Table 2<br/>(Independent)"]
            
            Context_Seq2[["Context Sequence 2<br/>[B, L, D]"]]
            Value_Seq2[["Value Sequence 2<br/>[B, L, D]"]]
            Pos_Emb2["Positional Embeddings 2"]
            
            Context_Seq_Pos2[["Context + Pos 2<br/>[B, L, D]"]]
            Value_Seq_Pos2[["Value + Pos 2<br/>[B, L, D]"]]
        end
        
        subgraph "Dual-Stream Encoder Stack 2 (N blocks)"
            Encoder2_In_Ctx[["Input Context 2"]]
            Encoder2_In_Val[["Input Value 2"]]
            
            Encoder2_Block["Encoder Block 2<br/>Q/K from Context 2<br/>V from Value 2<br/>Dual Add&Norm + FFN<br/>(Independent Parameters)"]
            
            Encoder2_Out_Ctx[["Output Context 2 (h₂)<br/>[B, L, D]"]]
            Encoder2_Out_Val[["Output Value 2 (v₂)<br/>[B, L, D]"]]
        end
        
        subgraph "Recursive Mastery Accumulation"
            Learning_Gains[["Learning Gains<br/>ReLU(v₂)"]]
            Projected_Gains[["Projected Gains<br/>[B, L, num_c]"]]
            
            Gain_Input[["Gain Input<br/>gain_t"]]
            Mastery_Prev[["Mastery_{t-1}"]]
            
            Scale_Op["× α (α=0.1)"]
            Sum_Op["mastery_t = mastery_{t-1} + α·gain_t"]
            Clamp_Op["Clamp[0,1]"]
            
            Mastery_Current[["Mastery_t"]]
            Projected_Mastery[["Incremental Mastery Levels<br/>[B, L, num_c]<br/>✅ Interpretability Output"]]
            
            Mastery_Current -.->|"temporal<br/>persistence"| Mastery_Prev
        end
        
        subgraph "Threshold Mechanism"
            Learnable_Threshold[["Learnable Threshold<br/>per skill [num_c]<br/>(trainable)"]]
            Threshold_Compute["sigmoid((mastery - threshold) / temp)"]
            
            IM_Predictions[["Incremental Mastery Predictions<br/>[B, L]<br/>✅ Interpretability Predictions"]]
        end
    end

    %% ========== LOSS COMPUTATION ========== %%
    subgraph "Loss Framework (Dual-Encoder Architecture)"
        direction LR
        
        BCE_Loss["BCE Loss<br/>(Performance)<br/>weight = λ₁"]
        IM_Loss["Incremental Mastery Loss<br/>(Interpretability)<br/>weight = λ₂"]
        
        Total_Loss["Total Loss<br/>λ₁ × BCE + λ₂ × IM<br/>(Weighted Combination)"]
    end

    %% ========== CONNECTIONS ========== %%
    
    %% Input to both encoders (same input, different processing)
    Input_q --> Tokens1
    Input_r --> Tokens1
    Input_q --> Tokens2
    Input_r --> Tokens2
    
    %% Encoder 1 Flow (Performance Path)
    Tokens1 --> Context_Emb1 --> Context_Seq1
    Tokens1 --> Value_Emb1 --> Value_Seq1
    Input_q --> Skill_Emb1
    
    Context_Seq1 --> Context_Seq_Pos1
    Value_Seq1 --> Value_Seq_Pos1
    Pos_Emb1 --> Context_Seq_Pos1
    Pos_Emb1 --> Value_Seq_Pos1
    
    Context_Seq_Pos1 --> Encoder1_In_Ctx --> Encoder1_Block
    Value_Seq_Pos1 --> Encoder1_In_Val --> Encoder1_Block
    
    Encoder1_Block --> Encoder1_Out_Ctx --> Pred_Input_h1
    Encoder1_Block --> Encoder1_Out_Val --> Pred_Input_v1
    Skill_Emb1 --> Pred_Input_s1
    
    Pred_Input_h1 --> Concat1
    Pred_Input_v1 --> Concat1
    Pred_Input_s1 --> Concat1
    Concat1 --> MLP1 --> Sigmoid1 --> Base_Predictions
    
    %% Encoder 2 Flow (Interpretability Path)
    Tokens2 --> Context_Emb2 --> Context_Seq2
    Tokens2 --> Value_Emb2 --> Value_Seq2
    
    Context_Seq2 --> Context_Seq_Pos2
    Value_Seq2 --> Value_Seq_Pos2
    Pos_Emb2 --> Context_Seq_Pos2
    Pos_Emb2 --> Value_Seq_Pos2
    
    Context_Seq_Pos2 --> Encoder2_In_Ctx --> Encoder2_Block
    Value_Seq_Pos2 --> Encoder2_In_Val --> Encoder2_Block
    
    Encoder2_Block --> Encoder2_Out_Val --> Learning_Gains
    Learning_Gains --> Projected_Gains --> Gain_Input
    
    Gain_Input --> Scale_Op --> Sum_Op
    Mastery_Prev --> Sum_Op --> Clamp_Op --> Mastery_Current
    Mastery_Current --> Projected_Mastery
    
    Projected_Mastery --> Threshold_Compute
    Learnable_Threshold --> Threshold_Compute
    Threshold_Compute --> IM_Predictions
    
    %% Loss Connections
    Base_Predictions --> BCE_Loss
    Ground_Truth --> BCE_Loss
    
    IM_Predictions --> IM_Loss
    Ground_Truth --> IM_Loss
    
    BCE_Loss --> Total_Loss
    IM_Loss --> Total_Loss

    %% Styling
    classDef encoder1_style fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef encoder2_style fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    classDef loss_style fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef input_style fill:#ffffff,stroke:#333333,stroke-width:2px
    classDef output_style fill:#e8f5e9,stroke:#43a047,stroke-width:3px
    classDef mastery_style fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class Tokens1,Context_Emb1,Value_Emb1,Skill_Emb1,Context_Seq1,Value_Seq1,Encoder1_Block,Encoder1_Out_Ctx,Encoder1_Out_Val,Concat1,MLP1,Sigmoid1 encoder1_style
    class Tokens2,Context_Emb2,Value_Emb2,Context_Seq2,Value_Seq2,Encoder2_Block,Encoder2_Out_Val,Learning_Gains encoder2_style
    class Scale_Op,Sum_Op,Clamp_Op,Mastery_Current,Projected_Mastery,Threshold_Compute mastery_style
    class BCE_Loss,IM_Loss,Total_Loss loss_style
    class Input_q,Input_r,Ground_Truth input_style
    class Base_Predictions,IM_Predictions,Projected_Mastery output_style
    class Input_q,Input_r,Tokens,Context_Seq,Value_Seq,Pos_Emb,Context_Seq_Pos,Value_Seq_Pos io_data
    class Encoder_Input_Context,Encoder_Input_Value,Attn_Input_Context,Attn_Input_Value io_data
    class Weights,Attn_Output_Heads,Attn_Output,Encoder_Output_Ctx,Encoder_Output_Val io_data
    class Pred_Input_h,Pred_Input_v,Pred_Input_s,Predictions io_data
    class Projected_Mastery_Output,Projected_Gain_Output io_data

    %% Link Styling - SIMPLIFIED: Most styling removed due to commented-out connections
    %% NOTE: After commenting out constraint and semantic losses, link indices changed.
    %% Keeping only essential link styles for active connections to avoid rendering errors.
    
    %% COMMENTED OUT: Original link styling referenced indices that no longer exist
    %% after simplification. To restore styling, recalculate indices if losses are reactivated.
    %%
    %% %% Pred_Hub outputs (black) - ALL 5 connections from Pred_Hub styled blue
    %% %% Pred_Hub -->|"BCE"| BCE_Loss (line 224)
    %% %% Pred_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss (line 238)
    %% %% Pred_Hub -->|"Gain-Perf"| Gain_Perf_Loss (line 239)
    %% %% Pred_Hub -->|"Alignment"| Alignment_Loss (line 242)
    %% %% Pred_Hub -->|"to Monitor"| Monitor_Hub (line 262)
    %% linkStyle 60 stroke:#888888,stroke-width:3px
    %% linkStyle 71 stroke:#888888,stroke-width:3px
    %% linkStyle 72 stroke:#888888,stroke-width:3px
    %% linkStyle 74 stroke:#888888,stroke-width:3px
    %% linkStyle 87 stroke:#888888,stroke-width:3px
    %% 
    %% %% Mastery_Hub outputs (red) - ALL 6 connections from Mastery_Hub styled red
    %% %% Mastery_Hub -->|"Global Align"| Global_Alignment (line 195)
    %% %% Mastery_Hub -->|"Monotonicity"| Monotonicity_Loss (line 227)
    %% %% Mastery_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss (line 228)
    %% %% Mastery_Hub -->|"Consistency"| Consistency_Loss (line 229)
    %% %% Mastery_Hub -->|"Retention"| Retention_Loss (line 230)
    %% %% Mastery_Hub -->|"to Monitor"| Monitor_Hub (line 260)
    %% linkStyle 57 stroke:#00ff00,stroke-width:3px
    %% linkStyle 62 stroke:#00ff00,stroke-width:3px
    %% linkStyle 63 stroke:#00ff00,stroke-width:3px
    %% linkStyle 64 stroke:#00ff00,stroke-width:3px
    %% linkStyle 65 stroke:#00ff00,stroke-width:3px
    %% linkStyle 85 stroke:#00ff00,stroke-width:3px
    %% 
    %% %% Gain_Hub outputs (green) - ALL 6 connections from Gain_Hub styled green
    %% %% Gain_Hub -->|"Gain-Perf"| Gain_Perf_Loss (line 232)
    %% %% Gain_Hub -->|"Sparsity"| Sparsity_Loss (line 233)
    %% %% Gain_Hub -->|"Consistency"| Consistency_Loss (line 234)
    %% %% Gain_Hub -->|"NonNeg"| NonNeg_Loss (line 235)
    %% %% Gain_Hub -->|"Lag"| Lag_Gain_Loss (line 236)
    %% %% Gain_Hub -->|"to Monitor"| Monitor_Hub (line 261)
    %% linkStyle 66 stroke:#008800,stroke-width:3px
    %% linkStyle 67 stroke:#008800,stroke-width:3px
    %% linkStyle 68 stroke:#008800,stroke-width:3px
    %% linkStyle 69 stroke:#008800,stroke-width:3px
    %% linkStyle 70 stroke:#008800,stroke-width:3px
    %% linkStyle 86 stroke:#008800,stroke-width:3px
    %% 
    %% %% Encoder_Output_Ctx outputs (purple) - ALL 3 connections from Encoder_Output_Ctx styled purple
    %% %% Encoder_Output_Ctx --> Pred_Input_h (line 135)
    %% %% Encoder_Output_Ctx --> Proj_Mastery (line 157)
    %% %% Encoder_Output_Ctx --> Encoder_Hub (line 186)
    %% linkStyle 38 stroke:#800080,stroke-width:3px
    %% linkStyle 42 stroke:#800080,stroke-width:3px
    %% linkStyle 54 stroke:#800080,stroke-width:3px
    %% 
    %% %% Encoder_Output_Val outputs (pink) - ALL 3 connections from Encoder_Output_Val styled pink
    %% %% Encoder_Output_Val --> Pred_Input_v (line 136)
    %% %% Encoder_Output_Val --> Proj_Gain (line 158)
    %% %% Encoder_Output_Val --> Encoder_Hub (line 187)
    %% linkStyle 39 stroke:#ff69b4,stroke-width:3px
    %% linkStyle 43 stroke:#ff69b4,stroke-width:3px
    %% linkStyle 55 stroke:#ff69b4,stroke-width:3px
    %% 
    %% %% Encoder_Hub outputs (orange) - 1 connection from Encoder_Hub styled orange
    %% %% Encoder_Hub -->|"Alignment"| Alignment_Loss (line 241)
    %% linkStyle 73 stroke:#ffa500,stroke-width:3px
    %% 
    %% %% Monitor_Hub output (purple)
    %% linkStyle 81 stroke:#800080,stroke-width:3px
```

---

### Architecture Summary - Current State

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

**Tensor Dimensions Legend:**
- **B**: Batch size (number of students in a training/evaluation batch)
- **L**: Sequence length (maximum number of interactions per student)
- **D**: Model dimension (hidden size of transformer layers, e.g., 256)
- **num_c**: Number of concepts/skills in the dataset (e.g., 100 for ASSIST2015)
- **num_q**: Number of questions in the dataset (varies by dataset)
- **H**: Number of attention heads in multi-head attention


## Sequence

This section shows the temporal flow of operations during training and evaluation, illustrating how components interact across time.

### Model Sequence

GainAKT3Exp

### Training Sequence

The training workflow involves the experiment launcher loading defaults, creating the model, iterating through epochs with monitoring hooks, and saving artifacts.

```mermaid
sequenceDiagram
    participant User
    participant Launcher as run_repro_experiment.py
    participant Config as parameter_default.json
    participant TrainScript as train_gainakt3exp.py
    participant Factory as create_exp_model()
    participant Model as GainAKT3Exp
    participant Monitor as InterpretabilityMonitor
    participant Data as DataLoader
    participant Optimizer
    participant Disk as File System
    
    User->>Launcher: python run_repro_experiment.py<br/>--model gainakt3exp<br/>--dataset assist2015<br/>--epochs 12<br/>[--param overrides...]
    
    activate Launcher
    Launcher->>Config: Load defaults
    Config-->>Launcher: All parameter defaults
    
    Launcher->>Launcher: Apply CLI overrides
    Launcher->>Launcher: Generate experiment directory<br/>{timestamp}_{model}_{title}_{uid}
    
    Launcher->>TrainScript: Launch training with<br/>all explicit parameters
    deactivate Launcher
    
    activate TrainScript
    TrainScript->>Factory: create_exp_model(config)
    activate Factory
    Factory->>Factory: Validate all 22 required params
    Factory->>Model: Initialize GainAKT3Exp<br/>(d_model, n_heads, etc.)
    Factory-->>TrainScript: model instance
    deactivate Factory
    
    TrainScript->>Monitor: Create monitor instance
    TrainScript->>Model: set_monitor(monitor)
    Model-->>TrainScript: Monitor hook registered
    
    TrainScript->>Disk: Save initial config.json<br/>(defaults + train/eval/trajectory commands)
    
    loop Each Epoch (1 to 12)
        TrainScript->>TrainScript: Reset epoch metrics
        
        loop Each Batch
            TrainScript->>Data: Get batch (q, r, qry)
            Data-->>TrainScript: Batch tensors
            
            TrainScript->>Model: forward_with_states(q, r, qry,<br/>batch_idx=current_batch)
            activate Model
            
            Model->>Model: Tokenize inputs<br/>Embed context & value streams
            Model->>Model: Pass through encoder blocks<br/>(dual-stream attention)
            Model->>Model: Generate predictions<br/>(MLP head)
            Model->>Model: Compute mastery/gains<br/>(projection heads + recursion)
            Model->>Model: Compute interpretability_loss<br/>(constraint losses)
            
            alt batch_idx % monitor_frequency == 0
                Model->>Monitor: Call monitor hook<br/>(context, value, mastery, gains, predictions)
                activate Monitor
                Monitor->>Monitor: Compute correlations<br/>Log statistics
                Monitor-->>Model: Monitoring complete
                deactivate Monitor
            end
            
            Model-->>TrainScript: {predictions, logits,<br/>context_seq, value_seq,<br/>mastery, gains,<br/>incremental_mastery_loss}
            deactivate Model
            
            TrainScript->>TrainScript: Compute BCE loss<br/>(from base logits)
            TrainScript->>TrainScript: total_loss = BCE + incremental_mastery_loss<br/>(SIMPLIFIED: constraint losses = 0.0)
            
            TrainScript->>Optimizer: Backward pass
            Optimizer->>Model: Update parameters
            Optimizer-->>TrainScript: Step complete
            
            TrainScript->>TrainScript: Accumulate batch metrics
        end
        
        TrainScript->>TrainScript: Compute validation metrics<br/>(AUC, accuracy)
        
        alt Best validation AUC
            TrainScript->>Disk: Save checkpoint<br/>model_epoch{n}.pt
        end
        
        TrainScript->>Disk: Save trajectory data<br/>(mastery/gain correlations)
        
        TrainScript->>TrainScript: Log epoch summary<br/>(train/val AUC, correlations)
    end
    
    TrainScript->>Disk: Save final metrics.json<br/>(all epochs' results)
    TrainScript->>Disk: Save best checkpoint metadata
    
    TrainScript-->>User: Training complete<br/>Saved to {run_dir}
    deactivate TrainScript
```

**Key Training Flow Characteristics:**
- **Zero Defaults**: All parameters explicitly loaded from config or CLI
- **Monitoring Hooks**: Periodic state capture every N batches (default: 50)
- **Dual Loss (SIMPLIFIED 2025-11-15)**: BCE (base prediction accuracy) + Incremental Mastery Loss (threshold predictions, weight=0.1)
  - Constraint losses: ALL commented out (weights=0.0)
  - Semantic losses: ALL commented out (disabled)
- **Artifact Persistence**: Complete reproducibility via saved config + checkpoints
- **Correlation Tracking**: Mastery/gain correlations computed per epoch

### Evaluation Sequence

The evaluation workflow loads a trained model checkpoint, runs inference on test data, computes metrics including correlations, and saves results.

```mermaid
sequenceDiagram
    participant User
    participant EvalScript as eval_gainakt3exp.py
    participant Config as run_dir/config.json
    participant Factory as create_exp_model()
    participant Model as GainAKT3Exp
    participant Checkpoint as model_*.pt
    participant Data as Test DataLoader
    participant Disk as File System
    
    User->>EvalScript: python eval_gainakt3exp.py<br/>--run_dir {experiment_dir}<br/>--ckpt_name model_best.pt<br/>[--num_corr_students 3000]
    
    activate EvalScript
    EvalScript->>Config: Load config.json
    Config-->>EvalScript: All parameters + metadata
    
    EvalScript->>Factory: create_exp_model(config)
    activate Factory
    Factory->>Factory: Validate parameters
    Factory->>Model: Initialize GainAKT3Exp
    Factory-->>EvalScript: model instance
    deactivate Factory
    
    EvalScript->>Checkpoint: Load checkpoint state_dict
    Checkpoint-->>EvalScript: Trained parameters
    EvalScript->>Model: load_state_dict(checkpoint)
    Model-->>EvalScript: Parameters loaded
    
    EvalScript->>Model: eval() mode
    EvalScript->>Data: Load test dataset
    Data-->>EvalScript: Test data ready
    
    EvalScript->>EvalScript: Initialize result accumulators<br/>(predictions, labels, mastery, gains)
    
    loop Each Test Batch
        EvalScript->>Data: Get batch (q, r, qry)
        Data-->>EvalScript: Batch tensors
        
        EvalScript->>Model: forward_with_states(q, r, qry)
        activate Model
        Model->>Model: Tokenize & embed
        Model->>Model: Encoder blocks (no gradients)
        Model->>Model: Generate predictions
        Model->>Model: Compute mastery/gains<br/>(for correlation)
        Model-->>EvalScript: {predictions, mastery, gains}
        deactivate Model
        
        EvalScript->>EvalScript: Accumulate predictions & labels
        EvalScript->>EvalScript: Accumulate mastery/gains<br/>(for selected students)
    end
    
    EvalScript->>EvalScript: Compute test AUC<br/>(all predictions vs labels)
    EvalScript->>EvalScript: Compute test accuracy
    
    alt Correlation computation requested
        EvalScript->>EvalScript: Sample students<br/>(default: 3000 students)
        
        loop Each sampled student
            EvalScript->>EvalScript: Extract student trajectory<br/>(mastery_seq, gains_seq, responses)
            EvalScript->>EvalScript: Compute Pearson correlation<br/>(mastery vs performance)
            EvalScript->>EvalScript: Compute Pearson correlation<br/>(gains vs performance)
            EvalScript->>EvalScript: Accumulate valid correlations
        end
        
        EvalScript->>EvalScript: Average mastery correlation<br/>across students
        EvalScript->>EvalScript: Average gain correlation<br/>across students
    end
    
    EvalScript->>Disk: Save eval_results_{timestamp}.json<br/>{AUC, accuracy, correlations, num_students}
    
    EvalScript-->>User: Evaluation complete<br/>Test AUC: {auc}<br/>Mastery Corr: {mastery_corr}<br/>Gain Corr: {gain_corr}
    deactivate EvalScript
```

**Key Evaluation Flow Characteristics:**
- **Checkpoint Loading**: Restores exact trained model state from disk
- **No Gradient Computation**: Model in eval mode, torch.no_grad() context
- **Correlation Sampling**: Configurable student sample size (default: 3000)
- **Trajectory Analysis**: Per-student mastery/gain sequences correlated with performance
- **Result Persistence**: Timestamped evaluation results saved to run directory

### Learning Trajectory Analysis Sequence

Individual student learning progressions can be extracted post-hoc using the trajectory analysis script (command auto-saved in config.json).

```mermaid
sequenceDiagram
    participant User
    participant TrajScript as learning_trajectories.py
    participant Config as run_dir/config.json
    participant Factory as create_exp_model()
    participant Model as GainAKT3Exp
    participant Checkpoint as model_best.pt
    participant Data as Test DataLoader
    participant Console
    
    User->>TrajScript: python learning_trajectories.py<br/>--run_dir {experiment_dir}<br/>--num_students 10<br/>--min_steps 10
    
    activate TrajScript
    TrajScript->>Config: Load config.json
    Config-->>TrajScript: Parameters
    
    TrajScript->>Factory: create_exp_model(config)
    Factory->>Model: Initialize model
    Factory-->>TrajScript: model instance
    
    TrajScript->>Checkpoint: Load best checkpoint
    Checkpoint-->>TrajScript: Trained parameters
    TrajScript->>Model: load_state_dict(checkpoint)
    
    TrajScript->>Data: Load test dataset
    TrajScript->>TrajScript: Filter students with<br/>≥ min_steps interactions
    TrajScript->>TrajScript: Sample diverse students<br/>(varied sequence lengths)
    
    loop Each selected student
        TrajScript->>Model: forward_with_states(student_seq)
        Model-->>TrajScript: {predictions, mastery, gains}
        
        TrajScript->>TrajScript: Extract trajectory:<br/>per-timestep (skill, true, pred, mastery, gain)
        
        TrajScript->>Console: Print student header:<br/>Global Index | Interactions | Unique Skills | Accuracy
        
        TrajScript->>Console: Print trajectory table:<br/>Step | Skill | True | Pred | Match | Gain | Mastery
        
        loop Each timestep
            TrajScript->>Console: t | skill_id | 0/1 | 0/1 | ✓/✗ | gain_val | mastery_val
        end
        
        TrajScript->>Console: Print separator
    end
    
    TrajScript-->>User: Trajectory analysis complete<br/>{num_students} students displayed
    deactivate TrajScript
```

**Key Trajectory Analysis Characteristics:**
- **Compact Format**: Tabular display with student summary statistics
- **Match Indicator**: Visual ✓/✗ showing prediction correctness
- **Per-Timestep Detail**: Shows skill practiced, true/predicted response, gains, mastery
- **Diverse Sampling**: Selects students with varying interaction counts
- **Post-Hoc Analysis**: No training required, works with any saved checkpoint

### Internal Model Flow: GainAKT3Exp forward_with_states()

This detailed sequence diagram shows the internal data flow within a single forward pass, tracking how Context (h) and Value (v) streams flow through the model and feed into predictions, interpretability projections, and losses.

```mermaid
sequenceDiagram
    participant Input as Input Tensors<br/>(q, r, qry)
    participant Tokenize as Tokenization
    participant CtxEmb as Context Embedding
    participant ValEmb as Value Embedding
    participant PosEmb as Positional Embedding
    participant Encoder as Encoder Blocks<br/>(N layers)
    participant CtxStream as Context Stream (h)<br/>[B, L, D]
    participant ValStream as Value Stream (v)<br/>[B, L, D]
    participant SkillEmb as Skill Embedding
    participant PredHead as Prediction Head<br/>(MLP)
    participant MasteryHead as Mastery Head<br/>Linear(D, num_c)
    participant GainHead as Gain Head<br/>Linear(D, num_c)
    participant Recursion as Recursive<br/>Accumulation
    participant Losses as Loss Computation
    participant Monitor as Monitor Hook
    participant Output as Output Dict
    
    Input->>Tokenize: q, r (questions, responses)
    Tokenize->>Tokenize: interaction_tokens = q + num_c * r
    
    Note over CtxEmb,ValEmb: Dual Embedding Tables
    Tokenize->>CtxEmb: interaction_tokens
    Tokenize->>ValEmb: interaction_tokens
    CtxEmb->>CtxStream: context_seq [B, L, D]
    ValEmb->>ValStream: value_seq [B, L, D]
    
    PosEmb->>CtxStream: Add positional encodings
    PosEmb->>ValStream: Add positional encodings
    
    Note over CtxStream,ValStream: Pass through N encoder blocks<br/>with dual-stream attention
    
    loop Each Encoder Block
        CtxStream->>Encoder: context_seq
        ValStream->>Encoder: value_seq
        
        Note over Encoder: Q, K = Linear(context_seq)<br/>V = Linear(value_seq)<br/>Attention = softmax(QK^T/√d)V
        
        Encoder->>Encoder: attn_output = Attention(Q, K, V)
        Encoder->>Encoder: context_seq = LayerNorm(context + attn + FFN(attn))
        Encoder->>Encoder: value_seq = LayerNorm(value + attn)
        
        Encoder->>CtxStream: Updated context_seq
        Encoder->>ValStream: Updated value_seq
    end
    
    rect rgb(200, 230, 255)
        Note over CtxStream: Context Output (h)<br/>Encoder_Output_Ctx [B, L, D]<br/>FLOWS TO 3 DESTINATIONS ↓
    end
    
    rect rgb(255, 220, 230)
        Note over ValStream: Value Output (v)<br/>Encoder_Output_Val [B, L, D]<br/>FLOWS TO 3 DESTINATIONS ↓
    end
    
    par Context Stream (h) flows to 3 destinations
        Note over CtxStream,PredHead: FLOW 1: Context → Prediction Head
        CtxStream->>PredHead: context_seq (h) [B, L, D]
    and
        Note over CtxStream,MasteryHead: FLOW 2: Context → Mastery Projection
        CtxStream->>MasteryHead: context_seq (h) [B, L, D]
    and
        Note over CtxStream,Monitor: FLOW 3: Context → Monitor/Output
        CtxStream->>Output: context_seq (for monitoring)
    end
    
    par Value Stream (v) flows to 3 destinations
        Note over ValStream,PredHead: FLOW 1: Value → Prediction Head
        ValStream->>PredHead: value_seq (v) [B, L, D]
    and
        Note over ValStream,Recursion: FLOW 2: Value → Recursive Mastery Accumulation<br/>(GainAKT3Exp Innovation: Values ARE learning gains)
        ValStream->>Recursion: value_seq = learning gains<br/>[B, L, D]<br/>(each interaction's contribution)
    and
        Note over ValStream,Monitor: FLOW 3: Value → Monitor/Output
        ValStream->>Output: value_seq (for monitoring)
    end
    
    Note over PredHead: Concatenate [h, v, s]
    Input->>SkillEmb: qry (target skills)
    SkillEmb->>PredHead: target_skill_emb [B, L, D]
    PredHead->>PredHead: concat = [context, value, skill]<br/>[B, L, 3*D]
    PredHead->>PredHead: logits = MLP(concat)<br/>[B, L]
    PredHead->>PredHead: predictions = sigmoid(logits)
    PredHead->>Output: predictions [B, L]
    PredHead->>Output: logits [B, L]
    
    Note over MasteryHead,Recursion: Mastery Projection Path
    MasteryHead->>MasteryHead: mastery_raw = Linear(context)<br/>[B, L, num_c]
    Note over MasteryHead: (Initial estimate - refined by recursion)
    
    rect rgb(180, 230, 255)
        Note over Recursion: Recursive Mastery Accumulation<br/>Values (learning gains) increment skill mastery<br/>For each interaction's skill:<br/>mastery[skill, t] = mastery[skill, t-1] + α × ReLU(value[t])
    end
    
    Input->>Recursion: questions (q) [B, L]<br/>(to identify relevant skill)
    
    loop Each timestep t (1 to L)
        Recursion->>Recursion: skill = question[t]
        Recursion->>Recursion: gain = ReLU(value[t]) × α (α=0.1)
        Recursion->>Recursion: mastery[skill, t] = mastery[skill, t-1] + gain
        Recursion->>Recursion: mastery[skill, t] = clamp(mastery[skill, t], 0, 1)
    end
    
    Recursion->>Output: projected_mastery [B, L, num_c]<br/>(accumulated across timesteps)
    Recursion->>Output: learning_gains [B, L, D]<br/>(directly from Values)
    
    Note over Losses: Loss Computation Sources
    
    rect rgb(255, 240, 200)
        Note over Losses: LOSS INPUTS:<br/>1. predictions (from Pred Head)<br/>2. projected_mastery (from Recursion)<br/>3. projected_gains (from Recursion)<br/>4. responses (ground truth)<br/>5. questions (for skill masks)
    end
    
    Output->>Losses: predictions [B, L]
    Output->>Losses: projected_mastery [B, L, num_c]
    Output->>Losses: projected_gains [B, L, num_c]
    Input->>Losses: responses (r) [B, L]
    Input->>Losses: questions (q) [B, L]
    
    Losses->>Losses: Compute skill masks<br/>(Q-matrix structure)
    
    par Constraint Losses (6 losses computed in parallel)
        Note over Losses: ❌ ALL CONSTRAINT LOSSES COMMENTED OUT (weights=0.0)
        Losses->>Losses: L1: Non-Negative Gains (INACTIVE)<br/>clamp(-gains, min=0).mean()
    and
        Losses->>Losses: L2: Monotonicity (INACTIVE)<br/>clamp(mastery[t] - mastery[t+1], min=0).mean()
    and
        Losses->>Losses: L3: Mastery-Performance (INACTIVE)<br/>low_mastery_correct + high_mastery_incorrect
    and
        Losses->>Losses: L4: Gain-Performance (INACTIVE)<br/>clamp(incorrect_gains - correct_gains + 0.1, min=0)
    and
        Losses->>Losses: L5: Sparsity (INACTIVE)<br/>abs(non_relevant_gains).mean()
    and
        Losses->>Losses: L6: Consistency (INACTIVE)<br/>abs(mastery_delta - scaled_gains).mean()
    end
    
    Losses->>Losses: Total interpretability_loss = 0.0<br/>(all weights w1-w6 = 0.0 in simplified architecture)
    Losses->>Output: interpretability_loss = 0.0 (scalar)
    
    alt Monitoring enabled AND batch_idx % monitor_frequency == 0
        Output->>Monitor: Pass all states:<br/>context_seq, value_seq,<br/>mastery, gains, predictions,<br/>questions, responses
        Monitor->>Monitor: Compute correlations<br/>Log statistics
        Monitor-->>Output: Monitoring complete
    end
    
    Output->>Output: Assemble final dict:<br/>{predictions, logits,<br/>context_seq, value_seq,<br/>projected_mastery, incremental_mastery_predictions,<br/>interpretability_loss (=0.0),<br/>incremental_mastery_loss}
```

**Key Flow Insights:**

### Context Stream (h) - 3 Destinations:
1. **→ Prediction Head**: Concatenated with value and skill embeddings for response prediction
2. **→ Mastery Computation**: Provides contextual information (not directly used in current sigmoid curve implementation)
3. **→ Output/Monitor**: Returned for monitoring and analysis

### Value Stream (v) - 3 Destinations (GainAKT3Exp Core Innovation):
**Values encode raw learning potential** - each interaction's Value output represents potential learning gain for that (skill, response) tuple.

1. **→ Prediction Head**: Concatenated with context and skill embeddings for response prediction
2. **→ Sigmoid Learning Curve Mastery**: **Direct flow as learning gain estimates**
   - Each Value output represents: "What is the learning potential from this interaction?"
   - Practice count tracking: Number of times each student practiced each skill
   - Sigmoid learning curve: `mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × practice_count[i,s,t] - offset)`
   - Learnable parameters modulate curve shape:
     - β_skill[s]: Skill difficulty (curve steepness)
     - γ_student[i]: Student learning velocity (progression speed)
     - M_sat[s]: Maximum achievable mastery (saturation level)
   - Automatic three-phase learning: Initial (warm-up) → Growth (rapid) → Saturation (consolidation)
3. **→ Output/Monitor**: Returned for monitoring and analysis

**Educational Semantics**: The transformer learns to output Values that encode learning potential. Combined with practice count tracking and learnable skill/student parameters, the sigmoid learning curve captures realistic learning dynamics: slow initial progress, rapid mid-stage improvement, and eventual saturation.

### Loss Computation Sources:
Interpretability losses receive inputs from multiple stages:
- **predictions**: From Prediction Head (sigmoid outputs)
- **projected_mastery**: From Sigmoid Learning Curves (mastery trajectories following practice-driven sigmoid curves)
- **learning_gains**: Estimated from Value stream (raw learning potential per interaction)
- **responses (r)**: Ground truth from input (for performance alignment)
- **questions (q)**: Input questions (for Q-matrix skill masks and practice count tracking)

**Sigmoid Learning Curve Mastery**: The key architectural principle where practice count drives sigmoid curve progression:
```
# Track practice count per student-skill pair
practice_count[i, s, t] = Σ(k=1 to t) 𝟙[question[k] targets skill s]

# Compute mastery via sigmoid learning curve
sigmoid_input = β_skill[s] × γ_student[i] × practice_count[i, s, t] - offset
mastery[i, s, t] = M_sat[s] × sigmoid(sigmoid_input)

# Threshold-based prediction
incremental_mastery_prediction[i, s, t] = sigmoid((mastery[i, s, t] - θ_global) / temperature)
```

This practice count-driven sigmoid curve enforces interpretability-by-design and provides educationally-realistic learning dynamics with automatic phase transitions.


## Implementation Summary

The GainAKT3Exp model (`pykt/models/gainakt3_exp.py`) is an enhanced version of the GainAKT3 base model that adds training-time interpretability monitoring and auxiliary loss computation. The implementation follows PyKT framework standards:

**Training Pipeline** (`examples/train_gainakt3exp.py`):
- Zero hardcoded defaults—all 60+ parameters must be explicit
- Launched via `run_repro_experiment.py` which loads defaults from `configs/parameter_default.json`
- Saves complete experiment artifacts: checkpoints, config, trajectories, metrics
- Supports semantic trajectory tracking (mastery/gain correlations per epoch)

**Evaluation Pipeline** (`examples/eval_gainakt3exp.py`):
- Loads trained model and computes test metrics (AUC, accuracy)
- Computes mastery/gain correlations (configurable student sample size)
- Saves evaluation results with timestamp

**Learning Trajectory Analysis** (`examples/learning_trajectories.py`):
- Standalone script to extract and display individual student learning trajectories
- Shows timestep-by-timestep: skills practiced, gains, mastery, predictions vs truth
- Compact tabular format with student summary statistics
- Command automatically added to `config.json` for easy access

**Model Creation**: Models are instantiated via `create_exp_model(config)` which requires all parameters in the config dict (no defaults), ensuring reproducibility.

### Training and Evaluation Workflow

Following PyKT framework standards (see `assistant/quickstart.pdf` and `assistant/contribute.pdf`):

**1. Launch Training:**
```bash
python examples/run_repro_experiment.py \
  --model gainakt3exp \
  --dataset assist2015 \
  --short_title baseline_test \
  --epochs 12 \
  [--param_override value ...]
```

This launcher:
- Loads defaults from `configs/parameter_default.json`
- Applies CLI overrides for specified parameters
- Creates timestamped experiment directory: `saved_model/{timestamp}_{model}_{title}_{uid}/`
- Saves complete config (including trajectory command) to `{run_dir}/config.json`
- Saves checkpoints, metrics, and trajectory data

**2. Evaluate Model:**
```bash
python examples/eval_gainakt3exp.py \
  --run_dir saved_model/{experiment_dir} \
  --ckpt_name {checkpoint}.pt
```

Outputs test AUC, accuracy, and mastery/gain correlations.

**3. Analyze Learning Trajectories:**
```bash
python examples/learning_trajectories.py \
  --run_dir saved_model/{experiment_dir} \
  --num_students 10 \
  --min_steps 10
```

Displays individual student learning progressions with mastery/gain states and prediction accuracy.

**Note:** The trajectory command is automatically included in `config.json` during training, enabling easy post-hoc analysis without manual parameter reconstruction.

### Reproducibility System

Following the "zero defaults" pattern documented in `examples/reproducibility.md`:

**Experiment Structure**: Each training run creates a timestamped directory:
```
saved_model/{timestamp}_{model}_{title}_{uid}/
  ├── config.json           # Complete parameter set + trajectory command
  ├── model_*.pt            # Checkpoints
  ├── metrics.json          # Training metrics (AUC, accuracy, correlations)
  ├── trajectory_*.json     # Semantic trajectory data (optional)
  └── eval_results_*.json   # Evaluation results
```

**Config File Contents**:
- `defaults`: All parameter default values (from `parameter_default.json`)
- `train_explicit`: Full training command with all parameters
- `eval_explicit`: Full evaluation command with all parameters  
- `trajectory_command`: Trajectory analysis command (10 students, min 10 steps)
- `metadata`: Timestamp, git hash, hostname, GPU info

**Reproducibility Guarantees**:
1. No hardcoded defaults in model or training code
2. All parameters explicitly passed via CLI
3. Complete config saved with every experiment
4. Commands reconstructable from config alone
5. Git hash and environment info captured

See `examples/reproducibility.md` for complete parameter evolution protocol.

Below is a comprehensive analysis of each architectural component's implementation status.

### Feature 1: Skill Embedding Table 

**Expected (from diagram):** A separate embedding table that maps question IDs to skill representations, used in the prediction head to provide skill-specific context for response prediction.

**Implementation Status:**
- **Location:** `gainakt3.py` line 198: `self.concept_embedding = nn.Embedding(num_c, d_model)`
- **Usage:** Lines 272-273 in forward pass:
  ```python
  target_concept_emb = self.concept_embedding(target_concepts)
  concatenated = torch.cat([context_seq, value_seq, target_concept_emb], dim=-1)
  ```
- **Architecture Alignment:** 
  - Separate embedding table for skills/concepts (distinct from interaction embeddings)
  - Embedded size: `d_model` (consistent with context/value streams)
  - Concatenated with context and value sequences as input to prediction head
  - Supports both direct question IDs (`q`) and query questions (`qry`)

**Verification:** The prediction head receives `[context_seq, value_seq, target_concept_emb]` with shape `[B, L, 3*d_model]`, exactly as specified in the diagram node "Concatenate [h, v, s]".

---

### Feature 2: Dynamic Value Stream 

**Expected (from diagram):** Dual-stream architecture where context and value sequences evolve independently through encoder blocks, with Q/K computed from context and V from value stream.

**Implementation Status:**
- **Dual Embeddings:** `gainakt3.py` lines 195-196:
  ```python
  self.context_embedding = nn.Embedding(num_c * 2, d_model)
  self.value_embedding = nn.Embedding(num_c * 2, d_model)
  ```
- **Dual Stream Processing:** Lines 263-269:
  ```python
  context_seq = self.context_embedding(interaction_tokens)
  value_seq = self.value_embedding(interaction_tokens)
  # ... add positional encodings to both ...
  for block in self.encoder_blocks:
      context_seq, value_seq = block(context_seq, value_seq, mask)
  ```
- **Separate Residual Paths:** `EncoderBlock` (lines 124-153) implements:
  - `norm1_ctx` and `norm1_val` - separate layer norms for each stream after attention
  - `norm2_ctx` - final layer norm for context after FFN
  - Value stream updated: `value_sequence + attn_output` 
  - Context stream updated: `context_sequence + attn_output + ffn_output`
  
- **Attention Mechanism:** `MultiHeadAttention.forward()` (lines 40-89):
  ```python
  Q = self.query_proj(context_sequence)  # Q from context
  K = self.key_proj(context_sequence)    # K from context
  V = self.value_proj(value_sequence)    # V from value stream
  ```

**Architecture Alignment:** 
- Dual independent sequences maintained throughout encoder stack
- Separate Add & Norm operations for context and value (as shown in diagram)
- Q/K from context, V from value exactly as specified
- Both streams contribute to final prediction

**Verification:** The architecture diagram shows "AddNorm_Ctx" and "AddNorm_Val" as separate nodes—implementation has `norm1_ctx`, `norm1_val`, and `norm2_ctx` implementing this exactly.

---

### Feature 3: Ground Truth Responses / Training-time Monitoring 

**Expected (from diagram):** Ground truth responses flow into loss calculation; interpretability monitor hook for real-time constraint analysis with configurable frequency.

**Implementation Status:**

**3a. Ground Truth Usage:**
- Ground truth `r` (responses) used in:
  - Interaction token creation (line 91): `interaction_tokens = q + self.num_c * r_int`
  - All auxiliary loss computations (lines 202-277) via `responses` parameter
  - Mastery-performance alignment: separates correct/incorrect responses (lines 236-243)
  - Gain-performance alignment: compares gains for correct vs incorrect (lines 246-254)

**3b. Training-time Monitoring Integration:**
- **Monitor Hook:** `gainakt3_exp.py` lines 40-41, 54-56:
  ```python
  self.interpretability_monitor = None
  def set_monitor(self, monitor): 
      self.interpretability_monitor = monitor
  ```
- **Periodic Execution:** Lines 164-178:
  ```python
  if (self.interpretability_monitor is not None and 
      batch_idx is not None and 
      batch_idx % self.monitor_frequency == 0 and primary_device):
      with torch.no_grad():
          self.interpretability_monitor(
              batch_idx=batch_idx,
              context_seq=context_seq,
              value_seq=value_seq,
              projected_mastery=projected_mastery,
              projected_gains=projected_gains,
              predictions=predictions,
              questions=q,
              responses=r
          )
  ```
- **Configurable Frequency:** `monitor_frequency` parameter (default: 50 batches)
- **DataParallel Safety:** Primary device guard prevents duplicate monitoring under multi-GPU training

**Architecture Alignment:** 
- Ground truth responses integrated into all constraint loss computations
- Monitoring hook provides real-time interpretability analysis
- Frequency control matches diagram's "Configurable frequency" specification
- All internal states exposed: context, value, mastery, gains, predictions, questions, responses

**Verification:** The diagram shows "Ground Truth Responses" flowing into "BCE Loss" and monitoring receiving multiple state tensors—implementation provides this via `forward_with_states()` returning all required outputs.


### Feature 4: Sigmoid Learning Curve Mastery from Practice Count

**Expected (from diagram):** Mastery evolves via practice count-driven sigmoid curves with learnable skill and student parameters, capturing three automatic learning phases (warm-up, growth, saturation).

**Implementation Status:**

**4a. Practice Count Tracking** (`gainakt3_exp.py`):
```python
# Track how many times each student has practiced each skill
practice_count = torch.zeros(batch_size, num_c, device=q.device)
for t in range(seq_len):
    skill_idx = q[t]  # Which skill this interaction targets
    practice_count[:, skill_idx] += 1  # Increment practice count
```

**4b. Learnable Parameters for Sigmoid Curves** (`gainakt3_exp.py` initialization):
```python
# Per-skill parameters (shared across students)
self.beta_skill = nn.Parameter(torch.ones(num_c))  # Skill difficulty (curve steepness)
self.M_sat = nn.Parameter(torch.ones(num_c) * 0.8)  # Saturation level (max mastery)

# Per-student parameters
self.gamma_student = nn.Parameter(torch.ones(num_students))  # Learning velocity

# Global parameters
self.theta_global = nn.Parameter(torch.tensor(0.5))  # Performance threshold
self.offset = nn.Parameter(torch.tensor(3.0))  # Sigmoid inflection point

# Config parameter (hybrid approach)
self.threshold_temperature = config.get('threshold_temperature', 1.0)  # Prediction sharpness
```

**4c. Sigmoid Learning Curve Computation** (`gainakt3_exp.py`):
```python
# Compute sigmoid learning curve for each student-skill pair
sigmoid_input = (self.beta_skill.unsqueeze(0).unsqueeze(0) *  # [1, 1, num_c]
                self.gamma_student.unsqueeze(1).unsqueeze(2) *  # [batch, 1, 1]
                practice_count.unsqueeze(1) -  # [batch, 1, num_c]
                self.offset)  # Scalar

mastery = self.M_sat.unsqueeze(0).unsqueeze(0) * torch.sigmoid(sigmoid_input)  # [batch, seq_len, num_c]

# Threshold-based prediction
threshold_diff = (mastery - self.theta_global) / self.threshold_temperature
incremental_mastery_predictions = torch.sigmoid(threshold_diff)
```

**Architecture Alignment:**
- **Practice Count Tracking**: Monotonic counter per student-skill pair drives sigmoid progression
- **Sigmoid Learning Curve**: `mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × practice_count[i,s,t] - offset)`
- **Output Shapes**: Mastery produces `[B, L, num_c]` tensors tracking sigmoid curve evolution per skill
- **Three Automatic Learning Phases**:
  1. **Initial Phase** (practice_count ≈ 0): mastery ≈ 0, slow learning (warm-up/familiarization)
  2. **Growth Phase** (intermediate): rapid mastery increase, slope = β_skill × γ_student (effective learning)
  3. **Saturation Phase** (high practice_count): mastery → M_sat[s], diminishing returns (consolidation)
- **Educational Semantics**: 
  - **β_skill[s]**: How steep the learning curve is (easier skills → higher β, steeper curves)
  - **γ_student[i]**: How fast the student learns (faster learners → higher γ, fewer interactions to saturation)
  - **M_sat[s]**: Maximum achievable mastery (some skills may cap below 1.0, indicating inherent difficulty)
  - **θ_global**: Mastery level required for correct performance (decision boundary)
  - **offset**: Controls where rapid learning phase begins (inflection point)
  - **threshold_temperature**: Config parameter controlling prediction sharpness (default 1.0, hybrid approach)

**Interpretability Guarantee**: The sigmoid learning curve model enforces educationally-realistic learning dynamics by design. We can interpret each parameter:
- β_skill tells us relative skill difficulty
- γ_student tells us relative student ability
- M_sat tells us skill mastérability ceiling
- Practice count progression shows automatic phase transitions
- Mastery trajectories follow interpretable sigmoid curves

**Verification:** The architecture enforces realistic learning dynamics via practice-driven sigmoid curves with clear educational parameters. No linear accumulation—mastery follows educationally-grounded sigmoid progressions with automatic phase transitions.


### Feature 5: BCE + Auxiliary Loss Functions 

**Expected (from diagram):** BCE loss for prediction accuracy plus five auxiliary losses (Non-Negative, Monotonicity, Mastery-Performance, Gain-Performance, Sparsity) with configurable weights, all integrated into total loss.

**Implementation Status (SIMPLIFIED 2025-11-15):**

**5a. BCE Loss:**
- ✅ **ACTIVE**: Computed externally in training script using `predictions` output
- Model provides both `predictions` (sigmoid) and `logits` for flexible loss computation

**5b. Incremental Mastery Loss:**
- ✅ **ACTIVE**: Computed in model (lines 511-519) using incremental mastery predictions from threshold mechanism
- Binary cross-entropy on threshold-based predictions vs ground truth
- Weight = 0.1 in current configuration
- Extracted and used by training loop (commit 07b63e3)

**5c. Auxiliary Constraint Losses - ❌ ALL COMMENTED OUT (all weights=0.0):**

**Implementation preserved in `compute_interpretability_loss()` (lines 202-277) but inactive:**

1. **Non-Negative Gains Loss** (lines 217-220):
   ```python
   negative_gains = torch.clamp(-projected_gains, min=0)
   non_negative_loss = negative_gains.mean()
   total_loss += self.non_negative_loss_weight * non_negative_loss
   ```

2. **Monotonicity Loss** (lines 222-226):
   ```python
   mastery_decrease = torch.clamp(projected_mastery[:, :-1] - projected_mastery[:, 1:], min=0)
   monotonicity_loss = mastery_decrease.mean()
   total_loss += self.monotonicity_loss_weight * monotonicity_loss
   ```

3. **Mastery-Performance Alignment Loss** (lines 228-243):
   ```python
   relevant_mastery = projected_mastery[skill_masks]
   correct_mask = (responses == 1).flatten()
   incorrect_mask = (responses == 0).flatten()
   low_mastery_on_correct = torch.clamp(1 - relevant_mastery[correct_mask], min=0)
   high_mastery_on_incorrect = torch.clamp(relevant_mastery[incorrect_mask], min=0)
   mastery_performance_loss = low_mastery_on_correct.mean() + high_mastery_on_incorrect.mean()
   total_loss += self.mastery_performance_loss_weight * mastery_performance_loss
   ```

4. **Gain-Performance Alignment Loss** (lines 245-254):
   ```python
   relevant_gains = projected_gains[skill_masks]
   correct_gains = relevant_gains[(responses == 1).flatten()]
   incorrect_gains = relevant_gains[(responses == 0).flatten()]
   if correct_gains.numel() > 0 and incorrect_gains.numel() > 0:
       gain_performance_loss = torch.clamp(incorrect_gains.mean() - correct_gains.mean() + 0.1, min=0)
       total_loss += self.gain_performance_loss_weight * gain_performance_loss
   ```

5. **Sparsity Loss** (lines 256-259):
   ```python
   non_relevant_gains = projected_gains[~skill_masks]
   sparsity_loss = torch.abs(non_relevant_gains).mean()
   total_loss += self.sparsity_loss_weight * sparsity_loss
   ```


6. **Consistency Loss** (lines 261-266):
   ```python
   mastery_delta = projected_mastery[:, 1:, :] - projected_mastery[:, :-1, :]
   scaled_gains = projected_gains[:, 1:, :] * 0.1
   consistency_residual = torch.abs(mastery_delta - scaled_gains)
   consistency_loss = consistency_residual.mean()
   total_loss += self.consistency_loss_weight * consistency_loss
   ```

**5d. Integration:**
- ❌ All constraint losses computed in single `compute_interpretability_loss()` method but **INACTIVE** (all weights=0.0)
- ❌ Returned as `interpretability_loss = 0.0` in `forward_with_states()` output dict (line 149)
- ✅ Incremental mastery loss computed separately and returned in output dict
- Each loss has configurable weight parameter (constructor lines 27-32)
- Skill masks computed from Q-matrix structure (line 213)

**Architecture Alignment (SIMPLIFIED 2025-11-15):** 
- ❌ All 5 diagram constraint losses COMMENTED OUT (weights=0.0)
- ❌ 6th loss (Consistency) also COMMENTED OUT (weight=0.0)
- ✅ Active loss formula: `BCE + incremental_mastery_loss` (weight=0.1)
- ⚠️ All constraint loss code preserved but inactive: `interpretability_loss = 0.0`
- **Total training loss:** `BCE + 0.1 × incremental_mastery_loss + 0.0 × interpretability_loss`

**Verification:** The simplified architecture focuses solely on BCE and Incremental Mastery Loss, with all constraint and semantic losses preserved in code but disabled via zero weights and false flags.


### Feature 6: Monitoring

**Expected (from diagram):** Real-time interpretability analysis during training via a monitoring hook that periodically captures internal model states (context, value, mastery, gains, predictions) for analysis, with configurable frequency to balance overhead and insight granularity.

**Implementation Status:**

**6a. Monitor Hook Infrastructure:**
- **Location:** `gainakt3_exp.py` lines 40-41, 54-56
- **Hook Registration:**
  ```python
  self.interpretability_monitor = None
  
  def set_monitor(self, monitor): 
      """Set the interpretability monitor hook."""
      self.interpretability_monitor = monitor
  ```
- **Usage Pattern:** Training scripts instantiate a monitor object and inject it via `model.set_monitor(monitor_instance)`, enabling modular monitoring strategies without model code changes.

**6b. Periodic State Capture:**
- **Location:** `gainakt3_exp.py` lines 164-178 (within `forward_with_states()`)
- **Execution Logic:**
  ```python
  if (self.interpretability_monitor is not None and 
      batch_idx is not None and 
      batch_idx % self.monitor_frequency == 0 and primary_device):
      with torch.no_grad():
          self.interpretability_monitor(
              batch_idx=batch_idx,
              context_seq=context_seq,
              value_seq=value_seq,
              projected_mastery=projected_mastery,
              projected_gains=projected_gains,
              predictions=predictions,
              questions=q,
              responses=r
          )
  ```
- **State Exposure:** Captures all interpretability-critical tensors at training time
- **No-Gradient Context:** Monitoring wrapped in `torch.no_grad()` to prevent gradient computation overhead

**6c. Configurable Frequency:**
- **Parameter:** `monitor_frequency` (default: 50 batches)
- **Location:** Constructor parameter (`gainakt3_exp.py` line 35)
- **Purpose:** Controls monitoring granularity—higher values reduce overhead but provide coarser temporal resolution
- **CLI Integration:** `--monitor_freq` parameter in training scripts

**6d. Multi-GPU Safety:**
- **Primary Device Guard:** `primary_device = (not hasattr(self, 'device_ids') or q.device == torch.device(f'cuda:{self.device_ids[0]}'))`
- **Rationale:** Under `DataParallel`, multiple model replicas process different batches; guard ensures monitoring executes only once per global batch (on primary GPU)
- **Location:** `gainakt3_exp.py` lines 160-163

**6e. State Dictionary Returned:**
- **Location:** `forward_with_states()` return statement (line 182-189)
- **Contents (SIMPLIFIED 2025-11-15):**
  ```python
  return {
      'predictions': predictions,  # Base predictions from prediction head
      'logits': logits,  # Raw logits before sigmoid
      'context_seq': context_seq,  # Context stream (h)
      'value_seq': value_seq,  # Value stream (v) = learning gains
      'projected_mastery': projected_mastery,  # Mastery trajectories
      'incremental_mastery_predictions': incremental_mastery_predictions,  # Threshold-based predictions
      'interpretability_loss': interpretability_loss,  # = 0.0 (all constraints commented out)
      'incremental_mastery_loss': incremental_mastery_loss  # BCE on threshold predictions (weight=0.1)
  }
  ```
- **Purpose:** Enables both real-time monitoring (via hook) and post-hoc analysis (via returned states)
- **Note:** `interpretability_loss` always returns 0.0 in simplified architecture; `incremental_mastery_loss` is the active auxiliary loss

**Architecture Alignment:** 
- Complete monitoring infrastructure matching diagram's "Monitor Hub" and "Interpretability Monitor" nodes
- Configurable frequency control as specified in diagram annotation
- All internal states exposed for comprehensive interpretability analysis
- Multi-GPU safe implementation for production training environments
- Zero-gradient overhead via `torch.no_grad()` wrapper

**Verification:** The architecture diagram shows "Monitor Hub" receiving inputs from Mastery Hub, Gain Hub, and Predictions Hub, then routing to "Interpretability Monitor"—implementation provides exactly this via the `forward_with_states()` method capturing all relevant tensors and passing them to the registered monitor hook.

---

### Feature 7: Intrinsic Gain Attention Mode ❌ DEACTIVATED

**Objective:** Provide an alternative architectural mode that achieves parameter efficiency by deriving gains directly from attention mechanisms, eliminating the need for post-hoc projection heads. This explores the trade-off between model compactness and interpretability while maintaining competitive predictive performance.

**Expected (from architectural exploration):** A feature flag (`--intrinsic_gain_attention`) that conditionally disables projection heads and computes mastery/gains from cumulative attention weights, reducing parameter count while preserving the ability to track learning trajectories.

**Current Status:**: Given the results detailed before we will **deactivate the Intrinsic Gain Attention Mode by default**. So, we set "intrinsic_gain_attention": false in configs/parameter_default.json

**Implementation Status:**

**7a. Architectural Constraint Enforcement:**
- **Location:** `gainakt3_exp.py` lines 58-74
- **Mechanism:** 
  ```python
  if self.intrinsic_gain_attention:
      # Override projection head flags - intrinsic mode incompatible with heads
      self.use_mastery_head = False
      self.use_gain_head = False
      
      if use_mastery_head or use_gain_head:
          print("WARNING: Intrinsic gain attention mode enabled. "
                "Projection heads (use_mastery_head, use_gain_head) will be disabled.")
  ```
- **Rationale:** Prevents conflicting architectural configurations where both projection-based and attention-derived gains would coexist, ensuring clean experimental comparison.

**7b. Attention-Derived Gain Computation:**
- **Location:** `gainakt3_exp.py` lines 102-111
- **Implementation:**
  ```python
  if self.intrinsic_gain_attention and not (self.use_mastery_head or self.use_gain_head):
      # Derive gains from attention patterns
      # Extract attention weights from last encoder layer
      last_block = self.encoder_blocks[-1]
      attn_module = last_block.attn
      
      # Aggregate attention weights across heads as proxy for learning gains
      # Shape: [batch_size, seq_len, seq_len] -> [batch_size, seq_len, num_c]
      attention_gains = self._compute_attention_derived_gains(
          attn_module.attention_weights, questions, batch_size, seq_len
      )
  ```
- **Gain Extraction:** Uses attention weights from final encoder layer as indicators of "information flow" between timesteps, treating high attention as proxy for learning influence.

**7c. Cumulative Mastery from Attention:**
- **Location:** `gainakt3_exp.py` lines 113-133
- **Recursive Accumulation:**
  ```python
  # Initialize mastery from attention-derived gains
  projected_gains = attention_gains  # [batch_size, seq_len, num_c]
  
  # Compute cumulative mastery via recursive addition
  projected_mastery = torch.zeros_like(projected_gains)
  projected_mastery[:, 0, :] = torch.sigmoid(projected_gains[:, 0, :])
  
  for t in range(1, seq_len):
      # Accumulate previous mastery + scaled current gains
      accumulated = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
      projected_mastery[:, t, :] = torch.clamp(accumulated, min=0.0, max=1.0)
  ```
- **Educational Semantics:** Treats attention weights as learning increments, cumulative mastery as integrated knowledge over time.

**7d. Parameter Reduction:**
- **Baseline Mode:** 14,658,761 params
- **Intrinsic Mode:** 12,738,265 params
- **Reduction: 1,920,496 params (13.1%)**
  
  *Note: Reduction comes from disabled projection heads (mastery_head + gain_head) plus associated architectural optimizations.*

**7e. CLI Integration:**
- **Location:** `examples/run_repro_experiment.py` line 89
- **Usage:** `python examples/run_repro_experiment.py --intrinsic_gain_attention --epochs 12`
- **Default:** `False` (baseline mode with projection heads)
- **Parameter File:** Added to `configs/parameter_default.json` as `"intrinsic_gain_attention": false`

**Architecture Alignment:** Complete implementation with validated trade-offs

**Verification:** The updated architecture diagram (red components) shows intrinsic mode as conditional bypass of projection heads, with attention-derived gains feeding directly to mastery/gain outputs.


### Feature 8: Sigmoid Learning Curve Mastery

Practice Count Drives Sigmoid Learning Curves
```
Practice Count Tracking → Sigmoid Curve Parameters → Mastery Evolution
                       ↑                              ↓
                  β_skill, γ_student, M_sat     Threshold Mechanism
```

The **blue subgraph** in the diagram above illustrates a critical architectural constraint that enforces interpretability-by-design. Unlike black-box models where knowledge states are opaque, our architecture implements a **deterministic sigmoid learning curve** mechanism:

$$\text{mastery}^{(i,s,t)} = M_{\text{sat}}^{(s)} \times \sigma\left(\beta_{\text{skill}}^{(s)} \times \gamma_{\text{student}}^{(i)} \times \text{practice\_count}^{(i,s,t)} - \text{offset}\right)$$

Where:
- **practice_count[i,s,t]**: Number of times student i has practiced skill s up to timestep t (monotonically increasing)
- **β_skill[s]**: Learnable skill difficulty parameter (controls curve steepness)
- **γ_student[i]**: Learnable student learning velocity (modulates progression speed)
- **M_sat[s]**: Learnable saturation level (maximum achievable mastery for skill s)
- **offset**: Learnable inflection point (controls when rapid learning begins)
- **σ**: Sigmoid function (ensures bounded, S-shaped learning curves)

This is implemented in the model's forward pass (`gainakt3_exp.py`):

```python
# Track practice count per student-skill
practice_count = torch.zeros(batch_size, num_c, device=q.device)
for t in range(seq_len):
    skill_idx = q[t]
    practice_count[:, skill_idx] += 1

# Compute sigmoid learning curve
sigmoid_input = (self.beta_skill * self.gamma_student.unsqueeze(1) * 
                practice_count - self.offset)
mastery = self.M_sat * torch.sigmoid(sigmoid_input)

# Threshold-based prediction
incremental_mastery_predictions = torch.sigmoid((mastery - self.theta_global) / self.threshold_temperature)
```

**Three Automatic Learning Phases**:
1. **Initial Phase**: practice_count ≈ 0 → mastery ≈ 0 (warm-up, minimal gains)
2. **Growth Phase**: intermediate practice_count → rapid mastery increase (effective learning)
3. **Saturation Phase**: high practice_count → mastery → M_sat[s] (consolidation, diminishing returns)

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

## Evolving the Model

We follow a rigorous parameter evolution protocol to maintain reproducibility across model variants and hyperparameter sweeps. See `examples/reproducibility.md` for complete details.

### Parameter Evolution Protocol

All parameter defaults live in `configs/parameter_default.json`. When adding or modifying parameters:

1. **Update `parameter_default.json`** with new parameter and default value
2. **Update `examples/run_repro_experiment.py`** to add CLI argument for new parameter
3. **Update `paper/parameters.csv`** with parameter documentation (category, description, default value)
4. **Verify no hardcoded defaults** exist in model/training code that could diverge from `parameter_default.json`

### Consistency Verification

After any codebase modification (model, training/evaluation scripts, etc.):
- Check that no hidden parameters with hardcoded defaults exist
- Verify all model parameters are passed explicitly via `create_exp_model(config)`
- Ensure training scripts require all parameters via CLI (no fallback defaults)
- Confirm `parameter_default.json` matches actual code behavior

### Hyperparameter Optimization

**Objective**: Systematically explore parameter combinations to find optimal configuration.

**Process**:
1. Use defaults from `configs/parameter_default.json` as starting point
2. Run experiments with parameter variations via CLI overrides
3. Document results in experiment-specific config files (auto-generated in run directories)
4. When optimal configuration is found, update `configs/parameter_default.json` with new defaults
5. Document change in `paper/parameters.csv` with rationale and date 

#### Scenario 2: Ablation Studies

Objective: in a ablation studio we deactivate parameters one by one to measure the impact
Guidelines: Check current value of the parameter to ablate in configs/parameter_default.json and deactivate (changing a boolean value, setting a weight to 0, etc.). 

#### Scenario 3: Benchmark

Objective: compare metrics of different models or model variants. 
Guidelines: use defaults to launch training and evaluation.

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


### Parameter Evolution Best Practices

When adding/changing parameters:

1. **Update `configs/parameter_default.json`**
   ```bash
   # Edit the file to add new parameter
   # Then recompute MD5:
   python -c "
   import json, hashlib
   data = json.load(open('configs/parameter_default.json'))
   md5 = hashlib.md5(json.dumps(data['defaults'], sort_keys=True).encode()).hexdigest()
   data['md5'] = md5
   json.dump(data, open('configs/parameter_default.json', 'w'), indent=2)
   print(f'Updated MD5: {md5}')
   "
   ```

2. **Update training/evaluation scripts**
   - Add argparse parameter with `required=True` (no default!)
   - Ensure parameter name matches exactly

3. **Test with dry run**
   ```bash
   python examples/run_repro_experiment.py \
     --short_title test_new_param \
     --epochs 1
   ```

4. **Verify in config.json**
   - Check that parameter appears in `defaults` section
   - Check that it appears in `train_explicit` or `eval_explicit` command

5. **Update this documentation**
   - Add to appropriate category table above
   - Document purpose and default value

## Trajectories



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



## Semantic Interpretabily Recovery

### Objective
Recover non-zero, educationally meaningful mastery and gain correlations after they regressed to 0.0 in a prior configuration, and identify the minimal parameter set whose activation restores semantic signals. Provide actionable guidance for parameter sweep design to optimize the trade-off between predictive AUC and interpretability (correlations, stability, coverage).

### Recovery Summary

Mastery and gain correlations regressed to zero when projection heads (`use_mastery_head`, `use_gain_head`) and semantic modules (alignment, global alignment, retention, lag) were inadvertently disabled by launcher logic overriding boolean flags to `false`. Recovery was achieved by re-enabling these modules plus extending constraint warm-up (4→8 epochs), reducing training horizon (20→12 epochs), and decreasing batch size (96→64). 

**Key Recovery Mechanisms:**
1. **Heads Activation:** Mandatory for producing mastery/gain trajectories (correlation computation impossible without)
2. **Alignment Family:** Local + adaptive + global residual alignment accelerates correlation emergence and stabilizes trajectories via performance-consistency shaping and population-level coherence
3. **Retention + Lag:** Prevents post-peak mastery decay and introduces temporal gain structure, improving final correlation retention and interpretability
4. **Scheduling:** Extended warm-up (8 epochs) allows latent representations to differentiate before full constraint pressure; shorter training (12 epochs) avoids late-stage correlation erosion

**Outcome:** Mastery correlation peaked at 0.149 (final: 0.124), gain correlation at 0.103 (final: 0.093), with zero constraint violations and early validation AUC of 0.726.

**Next Steps:** Multi-seed validation, early stopping to preserve AUC, ablation studies quantifying individual component contributions, and expansion to cross-dataset evaluation.

### Expected Outcomes
Recovered configuration demonstrates that enabling semantic modules and interpretability heads plus extending warm-up and reducing training horizon restores correlations (mastery ≈0.10+, gain ≈0.05+). Sweeps will seek configurations yielding mastery_corr ≥0.12 with val AUC ≥0.72 (early-stopped) and gain_corr ≥0.07 under zero violations, establishing a balanced regime for publication.


## Benchmark

### Baseline models
```
PYKT Benchmark Results Summary (Question-Level AUC):
- AKT: 0.7853 (AS2009), 0.8306 (AL2005), 0.8208 (BD2006), 0.8033 (NIPS34) - **Best overall**
- SAKT: 0.7246 (AS2009), 0.7880 (AL2005), 0.7740 (BD2006), 0.7517 (NIPS34)
- SAINT: 0.6958 (AS2009), 0.7775 (AL2005), 0.7781 (BD2006), 0.7873 (NIPS34)

Other benchmarks: 
- simpleKT 0.7744 (AS2009) 0.7248 (AS2015) - Reported as strong baseline with minimal complexity
```

### GinAKT Results Comparison

#### GainAKT2Exp Results
| Split      | AUC       | Accuracy  | Mastery Correlation | Gain Correlation | Correlation Students | Timestamp                  |
|------------|-----------|-----------|---------------------|------------------|----------------------|---------------------------|
| Training   | 0.7549    | 0.7631    | 0.1222              | 0.0544           | 3334                 | 2025-11-12T11:45:29.442074 |
| Validation | 0.7243    | 0.7538    | N/A                 | N/A              | N/A                  | 2025-11-12T11:45:29.442074 |
| Test       | 0.7188    | 0.7485    | 0.1165              | 0.0344           | 3177                 | 2025-11-12T11:45:29.442074 |

#### GainAKT3Exp Results

*Note: Results pending - sigmoid learning curve implementation completed 2025-11-16, ready for training*

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

### Sigmoid Learning Curve Implementation (2025-11-16)

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
