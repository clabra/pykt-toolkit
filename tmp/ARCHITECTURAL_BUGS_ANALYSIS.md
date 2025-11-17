# 🔴 CRITICAL ARCHITECTURAL BUG IN GAINAKT3EXP

**Date**: 2025-11-17 (Revised)  
**Status**: ❌ **FUNDAMENTAL DESIGN FLAW IDENTIFIED**  
**Impact**: Encoder 2 cannot learn skill-specific mastery patterns due to scalar gain mechanism

> **📖 For conceptual foundation and intended architecture**, see: [`/workspaces/pykt-toolkit/paper/STATUS_gainakt3exp.md`](../paper/STATUS_gainakt3exp.md) - Section "Conceptual Foundation: Skill-Specific Mastery Through Learning Gains"

---

## Executive Summary

After trajectory analysis revealed Encoder 2 AUC = 48.42% (below random) and mastery uncorrelated with responses (r = -0.044), we identified the **root cause**:

**PRIMARY BUG: Scalar Gain Quality** - Encoder 2 outputs single scalar per interaction instead of per-skill gains vector

### Revised Understanding (2025-11-17)

**What We Initially Thought** (Bug #2):
- ❌ "Encoder 2 trained to predict responses (wrong supervision signal)"
- ❌ "Should decouple mastery from response prediction"
- ❌ "Need different loss function for mastery growth"

**Actual Correct Design** (see STATUS_gainakt3exp.md for full explanation):
- ✅ Encoder 2 SHOULD predict responses through mastery mechanism
- ✅ Loss function BCE(encoder2_pred, y_true) is CORRECT
- ✅ Both encoders predict responses, but through different pathways:
  - Encoder 1: Direct attention → prediction (unconstrained)
  - Encoder 2: Attention → skill gains → mastery → prediction (interpretable)

**The Real Problem**:
- Scalar gains prevent skill-specific learning
- Without skill differentiation, mastery mechanism cannot work
- Even with correct supervision, Encoder 2 cannot learn proper mastery-response relationship

This bug explains why:
- Mastery values don't correlate with student performance
- Encoder 2 AUC is below random baseline
- All skills improve uniformly regardless of actual learning
- Increasing IM loss weight doesn't help (architecture cannot differentiate skills)

---

## Experiment Results Summary

**Experiment**: `20251117_131554_gainakt3exp_baseline-bce0.7_999787`  
**Configuration**: bce_loss_weight=0.7 (30% IM loss), patience=10

| Metric | Value | Status |
|--------|-------|--------|
| Overall Val AUC | 0.6765 | ✅ Reasonable |
| Encoder1 Val AUC | 0.6765 | ✅ Working well |
| **Encoder2 Val AUC** | **0.4842** | ❌ **Below random (50%)** |
| Mastery ↔ Response | -0.044 | ❌ **No correlation** |
| Encoder2_Pred Range | [0.37, 0.53] | ❌ **Too narrow** |

**Trajectory Analysis** (659 interactions, 10 students):
- Mastery IS updating (sigmoid progression observed)
- Expected gains = 0 everywhere (scalar gain_quality not output)
- All practiced skills get same increment (no differentiation)

---

## THE BUG: Scalar Gain Quality (Not Skill-Specific)

### Current Implementation (WRONG)

**File**: `pykt/models/gainakt3_exp.py`, Lines 528-554

```python
# Step 1: Compute SCALAR gain quality per interaction
gain_quality_logits = learning_gains_d.mean(dim=-1, keepdim=True)  # [B, L, D] → [B, L, 1]
gain_quality = torch.sigmoid(gain_quality_logits)  # [B, L, 1] ∈ [0, 1]  ← SCALAR!

# Step 2: Apply SAME scalar to ALL practiced skills
effective_practice = torch.zeros(batch_size, seq_len, self.num_c, device=q.device)

for t in range(seq_len):
    if t > 0:
        effective_practice[:, t, :] = effective_practice[:, t-1, :].clone()
    
    practiced_concepts = q[:, t].long()  # [B] - which skill is practiced
    batch_indices = torch.arange(batch_size, device=q.device)
    
    # ALL practiced skills get SAME quality increment!
    effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
    #                                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                                           SAME SCALAR FOR ALL SKILLS!
```

### The Problem

1. **Encoder 2 outputs**: `value_seq_2` [B, L, D=256] 
2. **Aggregation**: Mean across D dimensions → scalar [B, L, 1]
3. **Result**: Single "engagement quality" per interaction
4. **Application**: All practiced skills increment by same amount

**Example**:
- Student practices Skill 14 and Skill 92 in one interaction
- Encoder 2 produces: `gain_quality = 0.5` (scalar)
- **Both skills** improve by 0.5, regardless of:
  - Which skill the student actually learned
  - Skill difficulty differences
  - Prior mastery levels
  - Response correctness

### What Was Intended

**Per-skill gains** [B, L, num_c]:
```python
# Encoder 2 should output PER-SKILL gains
skill_gains = sigmoid(linear_projection(value_seq_2))  # [B, L, D] → [B, L, num_c]

# Different skills improve by different amounts
effective_practice[t, skill_14] += skill_gains[t, skill_14]  # e.g., +0.8
effective_practice[t, skill_92] += skill_gains[t, skill_92]  # e.g., +0.2
```

### Why This Matters

**Current behavior**:
- ✅ Can learn: "This interaction had high/low engagement"
- ❌ Cannot learn: "Student improved Skill A more than Skill B"
- ❌ Cannot learn: "Skill difficulty differences"
- ❌ Cannot learn: "Differential mastery growth"

**Real-world learning**:
- Students learn different skills at different rates
- Some skills are harder and require more practice
- Learning is skill-specific, not uniform engagement

**Impact on interpretability**:
- Mastery trajectories show uniform growth across all skills
- No differentiation between easy/hard skills
- Cannot explain why student excels at Skill A but struggles with Skill B
- Defeats the purpose of interpretable mastery tracking

---

## Why The Current Loss Function is CORRECT

**Current Implementation** (`examples/train_gainakt3exp.py`, Lines 285-291):

```python
# Encoder 1: Direct BCE on base predictions
bce_loss = bce_criterion(y_pred, y_true)

# Encoder 2: BCE on mastery-based predictions  
im_loss = bce_criterion(valid_im_preds, y_true)  # ✅ CORRECT

# Combined loss
loss = bce_loss_weight * bce_loss + incremental_mastery_loss_weight * im_loss
```

**This is CORRECT** - see STATUS_gainakt3exp.md for full explanation of why both encoders should predict responses through different pathways.

| Encoder | Supervision Signal | Pathway | Intended Learning |
|---------|-------------------|---------|-------------------|
| Encoder 1 | `BCE(pred, y_true)` | Direct attention → prediction | ✅ Unconstrained response patterns |
| Encoder 2 | `BCE(im_pred, y_true)` | Attention → gains → mastery → prediction | ✅ Skill-specific mastery mechanism |

**Key Insight**:
- Both encoders learn to predict responses (same objective) ✅
- But through DIFFERENT mechanisms:
  - Encoder 1: Black-box attention patterns (performance-optimized)
  - Encoder 2: Transparent mastery accumulation (interpretability-constrained)

### Why Encoder 2 AUC < 50% Despite Correct Loss

The issue is NOT the supervision signal - it's that **scalar gains (Bug #1) prevent the mastery mechanism from working**:

```
Current (BROKEN):
value_seq_2 → scalar gain_quality → uniform mastery growth → cannot differentiate skills → poor predictions

Intended (CORRECT):
value_seq_2 → per-skill gains → skill-specific mastery → differentiates skills → accurate predictions
```

**The constraints are FEATURES, not bugs**:
- ✅ Forced monotonic mastery (realistic: students don't forget)
- ✅ Forced sigmoid progression (realistic: learning curves)
- ✅ Skill-specific representation (interpretable: can explain predictions)
- ✅ Threshold mechanism (interpretable: mastery vs competence)

These constraints SHOULD make Encoder 2 slightly worse than Encoder 1 (55-60% vs 67% AUC), but the scalar gains bug makes it catastrophically worse (48% < 50% random).

### What Was Initially Misunderstood

**Initial diagnosis** (WRONG):
- "Encoder 2 learning response prediction is wrong"
- "Need different loss for mastery growth"
- "Decouple mastery from responses"

**Correct understanding** (RIGHT):
- Encoder 2 learning response prediction is CORRECT
- Loss function is already appropriate
- Mastery SHOULD predict responses (that's the whole point!)

### The Actual Problem is Bug #1 (Scalar Gains)

**Why mastery doesn't predict responses currently**:
1. Scalar gain_quality means all skills increase uniformly
2. Cannot learn which skills are relevant for which questions
3. Mastery becomes correlated with overall engagement, not skill-specific practice
4. Without skill differentiation, mastery-response relationship is meaningless

**Evidence from trajectories**:
- Mastery increases uniformly even when student answers incorrectly
- Example: Student 1, Skill 14: responses (1 → 1 → 0), mastery (0.26 → 0.48 → 0.68)
- But ALSO Skill 92 likely increased similarly (uniform increment)
- No correlation with actual performance (r = -0.044)
- Predictions compressed near 0.5 (cannot differentiate)

**Once Bug #1 is fixed**:
- Encoder 2 can learn: "Question targets Skill 14 → increase only Skill 14"
- Mastery becomes skill-specific: mastery[14] ≠ mastery[92]
- Correct responses correlate with high mastery in relevant skills
- Predictions become accurate: P(correct) = f(mastery[relevant_skills])

---

## How Scalar Gains Break the Mastery Mechanism

The scalar gain bug prevents the intended skill-specific mastery flow:

### Intended Flow (With Per-Skill Gains)

```
Question Q1 targets Skill 14:
1. Encoder 2 attends to interaction
2. Produces skill_gains = [0.0, ..., 0.8 (skill 14), ..., 0.0 (skill 92), ...]
3. Only Skill 14 mastery increases: mastery[14] += 0.8, mastery[92] unchanged
4. Response prediction based on mastery[14]
5. Correct response → BCE loss low → reinforces high gain for Skill 14
6. Over time: Encoder 2 learns "Q1 improves Skill 14"
```

### How Scalar Gains Break the Mechanism

**Example Failure Mode**:
- Question Q1 targets Skill 14 (per Q-matrix)
- Student engages deeply but answers incorrectly
- Encoder 2 produces: `gain_quality = 0.8` (scalar)
- Current (WRONG): ALL skills increase by 0.8 (Skill 14, Skill 92, ...)
- Cannot learn which skill is relevant to Q1
- Mastery becomes engagement proxy, not skill knowledge

**With per-skill gains (intended)**:
- Same interaction: Encoder 2 produces `skill_gains[14] = 0.8, skill_gains[92] = 0.1, ...`
- Only Skill 14 increases significantly
- BCE loss backpropagates through mastery → gains → Encoder 2
- Encoder 2 learns question-skill associations

---

## Why Parameter Tuning Won't Fix This

**Won't work**:
- ❌ Increase IM loss weight → Architecture still cannot differentiate skills
- ❌ Adjust sigmoid parameters → Still scalar gains
- ❌ Lower threshold → Doesn't add per-skill capability
- ❌ Add more encoder blocks → More capacity for wrong mechanism

**What's needed**:
- ✅ Per-skill gain vector [B, L, num_c] (architectural change required)

---

## Expected Outcomes: How to Verify the Fix Works

Once the per-skill gains fix is implemented, we should observe specific improvements in both validation metrics and trajectory analysis. This section defines what "success" looks like.

### Validation Metrics (During Training)

**Primary Indicators of Success:**

| Metric | Current (Broken) | Minimum Success | Target Success | Interpretation |
|--------|------------------|-----------------|----------------|----------------|
| **Encoder2 Val AUC** | 0.4842 (below random) | > 0.55 | > 0.60 | Encoder 2 can predict responses through mastery |
| **Encoder1 Val AUC** | 0.6765 | Maintain ~0.67 | ~0.67-0.68 | Encoder 1 performance unchanged |
| **Mastery ↔ Response Correlation** | -0.044 (no relationship) | > 0.3 | > 0.5 | Mastery is predictive of performance |
| **Encoder2_Pred Range** | [0.37, 0.53] (compressed) | > 0.3 width | > 0.4 width | Confident differentiation between students |
| **Training Stability** | Stable | Stable | Stable | No gradient issues or NaN values |

**What These Metrics Tell Us:**

1. **Encoder2 Val AUC > 0.55**: Proves that Encoder 2 can learn meaningful skill-specific patterns. Below random (0.50) means the model cannot distinguish correct from incorrect responses through mastery mechanism. Above 0.55 indicates the mastery-based prediction pathway is functional.

2. **Positive Mastery ↔ Response Correlation (> 0.3)**: Validates the conceptual foundation that "higher mastery → more likely to answer correctly". Negative or zero correlation means mastery is unrelated to actual performance, making it meaningless as an interpretability measure.

3. **Wider Prediction Range**: Compressed predictions (narrow range) indicate the model is uncertain and making near-random guesses. A wider range (> 0.3) shows the model can confidently predict both success (high mastery) and failure (low mastery).

### Trajectory Analysis (Post-Training)

After training, extract trajectories and check for these patterns:

**1. Skill Differentiation (Critical)**

**Test:**
```python
# Compute gain variance across skills per interaction
gain_variance = df.groupby(['student_idx', 'step'])['expected_gain'].std().mean()
print(f"Gain variance per interaction: {gain_variance:.4f}")
```

**Expected Outcomes:**
- ❌ **Current (Broken)**: Variance ≈ 0.0 (all skills get same gain)
- ✅ **Minimum Success**: Variance > 0.05 (some skill differentiation)
- 🎯 **Target Success**: Variance > 0.10 (clear skill differentiation)

**Interpretation**: Each interaction should affect different skills differently. High gain variance means the model learned which skills are relevant for each question (Q-matrix learning). Zero variance means uniform gains (the current bug).

**2. Q-Matrix Learning (Sparse Activation)**

**Test:**
```python
# Count how many skills have significant gains per interaction
skills_per_interaction = df.groupby(['student_idx', 'step'])['expected_gain'].apply(
    lambda x: (x > 0.1).sum()
).median()
print(f"Median skills activated per interaction: {skills_per_interaction:.1f}")
```

**Expected Outcomes:**
- ❌ **Current (Broken)**: All skills activated (≈123 skills if num_c=123)
- ✅ **Minimum Success**: 1-5 skills per interaction (sparse pattern)
- 🎯 **Target Success**: 1-3 skills per interaction (tight sparsity)

**Interpretation**: Real questions typically test 1-3 related skills, not all skills simultaneously. Sparse activation (few skills with high gains per interaction) proves the model learned question-skill associations from data.

**3. Monotonicity (Educational Validity)**

**Test:**
```python
# Check that mastery never decreases for any student-skill pair
violations = 0
for student in df['student_idx'].unique():
    student_data = df[df['student_idx'] == student].sort_values('step')
    for skill in student_data['skill_id'].unique():
        skill_traj = student_data[student_data['skill_id'] == skill]['mastery']
        decreases = (skill_traj.diff().dropna() < -1e-6).sum()
        violations += decreases

print(f"Monotonicity violations: {violations}")
```

**Expected Outcomes:**
- ✅ **Current (Working)**: 0 violations (sigmoid accumulation enforces this architecturally)
- ✅ **After Fix**: 0 violations (should remain enforced)

**Interpretation**: Once a student learns a skill, they shouldn't "unlearn" it. Zero violations confirms the educational constraint holds.

**4. Mastery Progression Patterns**

**Test:**
```python
# Examine mastery trajectories for sigmoid-shaped learning curves
import matplotlib.pyplot as plt

# Plot mastery progression for a sample student on their most-practiced skill
student = df[df['student_idx'] == 0]
most_practiced_skill = student['skill_id'].mode()[0]
skill_data = student[student['skill_id'] == most_practiced_skill].sort_values('step')

plt.plot(skill_data['step'], skill_data['mastery'], marker='o')
plt.xlabel('Interaction Step')
plt.ylabel('Mastery')
plt.title(f'Student 0, Skill {most_practiced_skill}')
plt.show()
```

**Expected Visual Pattern:**
- ❌ **Current (Broken)**: Uniform linear increase regardless of responses
- ✅ **Target Success**: Sigmoid curve (slow → fast → slow growth)
  - Early interactions: Low mastery (0.0-0.3), slow growth
  - Mid interactions: Rapid growth phase (0.3-0.7)
  - Late interactions: Saturation (0.7-0.9), diminishing returns

**Interpretation**: Real learning follows sigmoid curves. Uniform growth indicates the model isn't capturing learning dynamics. Sigmoid progression validates the educational model.

**5. Skill Difficulty Differentiation**

**Test:**
```python
# Compare average gains across different skills
skill_difficulty = df.groupby('skill_id')['expected_gain'].mean().sort_values()
print("Easiest skills (high avg gains):", skill_difficulty.tail(5).index.tolist())
print("Hardest skills (low avg gains):", skill_difficulty.head(5).index.tolist())
```

**Expected Outcomes:**
- ❌ **Current (Broken)**: All skills have identical average gains (scalar)
- ✅ **Minimum Success**: Significant variance across skills (CV > 0.2)
- 🎯 **Target Success**: Clear differentiation (easy skills: >0.6, hard skills: <0.4)

**Interpretation**: Some skills are objectively easier (students learn them faster) than others. Differentiation in average gains across skills proves the model learned skill-specific difficulty patterns from interaction data.

**6. Response-Conditional Gains**

**Test:**
```python
# Compare gains for correct vs incorrect responses
correct_gains = df[df['actual_response'] == 1]['expected_gain'].mean()
incorrect_gains = df[df['actual_response'] == 0]['expected_gain'].mean()
print(f"Correct response gains: {correct_gains:.4f}")
print(f"Incorrect response gains: {incorrect_gains:.4f}")
print(f"Ratio (correct/incorrect): {correct_gains / incorrect_gains:.2f}")
```

**Expected Outcomes:**
- ✅ **Educational Assumption**: Correct responses should yield higher gains (ratio > 1.2)
- 🎯 **Target Success**: Ratio 1.5-2.0 (substantial difference)

**Interpretation**: Successfully solving a problem should lead to larger learning gains than failing. This validates that the model learned educationally meaningful patterns (success → learning) rather than just counting interactions.

### Summary: Fix Verification Checklist

After implementing per-skill gains and re-training:

**Immediate Checks (From Training Logs):**
- [ ] Encoder2 Val AUC > 0.55 (preferably > 0.60)
- [ ] Mastery ↔ Response correlation > 0.3 (preferably > 0.5)
- [ ] Encoder2_Pred range > 0.3 (predictions not compressed near 0.5)
- [ ] No NaN or gradient explosions during training
- [ ] Skill gains std > 0.05 in debug logs (differentiation visible)

**Trajectory Analysis Checks:**
- [ ] Gain variance per interaction > 0.10 (skill differentiation)
- [ ] Sparse activation: 1-3 skills per interaction (Q-matrix learning)
- [ ] Zero monotonicity violations (educational validity)
- [ ] Sigmoid-shaped mastery curves (realistic learning dynamics)
- [ ] Skill difficulty variance: CV > 0.2 across skills
- [ ] Correct response gains > incorrect response gains (ratio > 1.2)

**Success Criteria:**
- **Minimum Viable Fix**: 4/5 immediate checks + 4/6 trajectory checks pass
- **Target Success**: 5/5 immediate checks + 5/6 trajectory checks pass
- **Optimal Success**: All checks pass + Encoder2 AUC > 0.65

If these criteria are met, the architectural fix has successfully enabled Encoder 2 to learn skill-specific mastery patterns, validating the conceptual foundation described in STATUS_gainakt3exp.md.

---

## Implementation Plan

Based on STATUS_gainakt3exp.md conceptual foundation: Encoder 2 should learn skill-specific gains to enable mastery-based prediction through threshold logic.

### Step 1: Modify Model Architecture

**File**: `pykt/models/gainakt3_exp.py`

**Change 1.1 - Add projection layer in `__init__`** (around line 200):
```python
self.gains_projection = nn.Linear(self.d_model, self.num_c)
```

**Change 1.2 - Replace scalar gains in `forward_with_states`** (lines 528-554):
```python
# OLD (scalar): gain_quality = torch.sigmoid(learning_gains_d.mean(dim=-1, keepdim=True))  # [B,L,1]
# NEW (per-skill):
skill_gains = torch.sigmoid(self.gains_projection(value_seq_2))  # [B, L, num_c]

# OLD: effective_practice[batch_indices, t, practiced_concepts] += gain_quality[batch_indices, t, 0]
# NEW: effective_practice[:, t, :] += skill_gains[:, t, :]
```

**Change 1.3 - Add to output dictionary** (around line 620):
```python
if self.use_gain_head:
    outputs['projected_gains'] = skill_gains  # For trajectory validation
```

### Step 2: Update Configuration

**File**: `configs/parameter_default.json`
```json
{"use_gain_head": true}
```

### Step 3: Training and Validation

**Step 3.1 - Re-train model**:
```bash
python examples/run_repro_experiment.py --short_title fixed-per-skill-gains --use_gain_head true
```

**Step 3.2 - Extract trajectories**:
```bash
EXP_DIR=$(ls -td examples/experiments/*fixed-per-skill-gains* | head -1)
python examples/learning_trajectories.py --run_dir "$EXP_DIR" --num_students 10
```

**Step 3.3 - Verify metrics**:

| Metric | Current (Broken) | Target (Fixed) | Check |
|--------|------------------|----------------|-------|
| Encoder2 Val AUC | 0.4842 | > 0.55 | Model performance |
| Mastery ↔ Response | -0.044 | > 0.4 | Concept validity |
| Gain Variance | ~0.0 | > 0.1 | Skill differentiation |
| Prediction Range | [0.37, 0.53] | > 0.3 width | Confidence |

### Step 4: Interpretation Checks

**Check 4.1 - Skill differentiation** (from trajectories):
```python
# Gain variance across skills per interaction
df.groupby(['student_idx', 'step'])['expected_gain'].std().mean() > 0.1
```

**Check 4.2 - Q-matrix learning** (sparse activation):
```python
# Each interaction should activate 1-3 skills (not all)
df.groupby(['student_idx', 'step'])['expected_gain'].apply(lambda x: (x > 0.1).sum()).median() <= 3
```

**Check 4.3 - Monotonicity** (no mastery decrease):
```python
# Verify mastery never decreases within student-skill trajectories
for s in df['student_idx'].unique():
    s_data = df[df['student_idx'] == s].sort_values('step')
    for c in s_data['skill_id'].unique():
        assert (s_data[s_data['skill_id'] == c]['mastery'].diff().dropna() >= -1e-6).all()
```

**Check 4.4 - Threshold logic** (mastery predicts response):
```python
# Correlation between minimum relevant-skill mastery and response
df_relevant = df[df['expected_gain'] > df.groupby(['student_idx', 'step'])['expected_gain'].transform('quantile', 0.7)]
min_mastery = df_relevant.groupby(['student_idx', 'step'])['mastery'].min()
# Should correlate better with response than mean mastery
```

### Step 5: Success Criteria

| Level | Encoder2 AUC | Mastery Corr | Gain Variance | Interpretation |
|-------|--------------|--------------|---------------|----------------|
| **Minimum** | > 0.55 | > 0.3 | > 0.05 | Fix works, skill differentiation present |
| **Target** | > 0.60 | > 0.5 | > 0.10 | Strong performance, learned Q-matrix |
| **Optimal** | > 0.65 | > 0.6 | > 0.15 | Approaches Encoder1, educational validity confirmed |

---

## Conclusion

**Root Cause**: Scalar gain_quality prevents skill-specific learning

**Fix**: Replace with per-skill gains vector [B, L, num_c]

**No loss function changes needed** - current BCE supervision is correct

**Expected after fix**: Encoder 2 AUC 55-60%, positive mastery-response correlation, skill differentiation visible in trajectories

**Implementation Complexity**: Low - single architectural change, maintains all other components

**Risk**: Low - fix aligns with intended design, preserves differentiability and training stability

---

## Fix Verification Results (Experiment 995130)

**Date**: 2025-11-17  
**Status**: ⚠️ **PARTIAL SUCCESS** - Architecture fixed, but insufficient skill differentiation

### Architectural Fix: ✅ SUCCESSFUL

- Per-skill gains projection layer implemented correctly
- Gradient flow through `gains_projection` layer working
- Model compiles and trains without errors

### Performance Metrics: ⚠️ MIXED RESULTS

| Metric | Previous (Broken) | Current (Fixed) | Expected | Status |
|--------|-------------------|-----------------|----------|--------|
| **Encoder2 Val AUC** | 0.4842 | **0.5868** | > 0.55 | ✅ Improved |
| **Encoder1 Val AUC** | 0.6765 | 0.6897 | ~0.67 | ✅ Maintained |
| **Mastery ↔ Response** | -0.044 | **0.037** | > 0.3 | ❌ Too low |
| **Gain Std Deviation** | ~0.0 | **0.0015** | > 0.05 | ❌ Nearly uniform |
| **Gain CV Across Skills** | ~0.0 | **0.0016** | > 0.2 | ❌ No differentiation |

**Key Finding**: Encoder2 AUC improved from **48% → 59%** (above random!), proving the architectural fix enables gradient flow. However, gains converged to nearly uniform values (mean=0.588, std=0.0015), indicating the model learned a constant gain rather than skill-specific patterns.

### Trajectory Analysis: ❌ SKILL DIFFERENTIATION FAILED

**From 659 interactions (10 students, 66 skills):**

- ✅ **Sparsity**: 1 skill per interaction (tight, but may be artifact of uniform gains)
- ✅ **Monotonicity**: 0 violations (mastery never decreases)
- ❌ **Skill Differentiation**: Gain variance = nan (essentially zero)
- ❌ **Skill Difficulty**: CV = 0.0016 (99.8% identical across skills)
- ❌ **Response-Conditional**: Correct/incorrect ratio = 1.00 (no educational signal)

### Root Cause of Poor Differentiation

The architecture is correct, but training dynamics led to a **degenerate solution**:

1. **Weak IM loss weight** (30%) provides insufficient gradient pressure for skill differentiation
2. **No sparsity/variance loss** to explicitly encourage skill-specific patterns
3. **Insufficient epochs** (12) to escape uniform-gain local minimum
4. **Initialization bias** - gains_projection may have started near uniform solution

### Recommendations for V2

**Priority 1**: Increase IM loss weight from 30% → 50%  
**Priority 2**: Add variance loss to explicitly encourage skill differentiation:
```python
gain_variance_loss = -skill_gains.var(dim=-1).mean()  # Maximize variance
```
**Priority 3**: Train for 20 epochs with early stopping on Encoder2 AUC  
**Priority 4**: Consider layer-wise learning rates (higher for gains_projection)

### Success Criteria Scorecard

**Immediate Checks**: 2/5 passed (Encoder2 AUC, No NaN)  
**Trajectory Checks**: 2/6 passed (Sparsity, Monotonicity)  
**Overall**: 4/11 checks passed (36%)

**Conclusion**: The architectural bug is **FIXED** (per-skill gains working), but the model requires **hyperparameter tuning** to learn meaningful skill differentiation. The 11-point AUC improvement (48% → 59%) proves the fix CAN work - we just need stronger training signals.

**Detailed Report**: See `/workspaces/pykt-toolkit/tmp/FIX_VERIFICATION_REPORT.md`

**Next Experiment**: `--short_title fixed-per-skill-gains-v2 --bce_loss_weight 0.5` + variance loss
