# GainAKT3Exp Architecture 

## Reference Documents

- `gainakt_architecture_approach.md` - Transformer Encoder-only architecture foundations
- `gainakt3_architecture_approach.md` - GainAKT3Exp model variant (this document)
- `model.md` - Link to current working model version
- `gainakt3exp_architecture_diagrams.md` - Architecture diagrams and sequences
- `paper/STATUS_gainakt3exp.md` - Detailed status, experiments, and bugs

Other documents: 
- `gainakt2_architecture_approach.md` - GainAKT2Exp model variant. It is deprecated in favour of GainAKT3Exp but the doc is referenced here in case it's neccesary to have an overview of the historic evolution of proposed models.   
---

## Overview

GainAKT3Exp is a **dual-encoder Transformer architecture** for interpretable knowledge tracing. It separates performance optimization (Encoder 1) from interpretability learning (Encoder 2) using completely independent encoder stacks with 167,575 total parameters.

**Key Innovation**: Encoder 2 learns "gain quality" per interaction through differentiable effective practice, enabling gradient flow for interpretability learning while Encoder 1 optimizes prediction accuracy.

---

## Conceptual Foundation: Skill-Specific Mastery Through Learning Gains

### Core Hypothesis

Our dual-encoder architecture is based on the hypothesis that **student mastery evolves through skill-specific learning gains** that can be modeled via sigmoid-shaped learning curves. The key principles are:

#### 1. Questions Target Specific Skills (Q-Matrix)

- Each question/item is designed to develop one or more skills
- Q-matrix defines relationships: Q[question_id] → {skill_1, skill_2, ..., skill_k}
- We call these "relevant skills" for a given question
- Example: In ASSIST2015 (single-skill dataset), each question targets exactly one skill

#### 2. Encoder 2 Learns to Predict Skill-Specific Learning Gains

**Critical Understanding**: Encoder 2 does NOT learn to predict responses directly. Instead:

**Encoder 2's Primary Objective**: Learn patterns from interaction data that quantify **how much each interaction contributes to increase the mastery level** of each skill relevant to the question (according to Q-matrix).

**What Encoder 2 Learns**:
- For each interaction with question Q at step t
- Estimate learning gains Δm[skill, t] for each relevant skill
- Quantify: "To what extent does this interaction improve mastery of Skill A, Skill B, etc.?"

**Training Signal for Encoder 2**:
- Encoder 2 weights are trained to predict gains such that:
  - Predicted gains → effective_practice accumulation → mastery trajectories
  - Mastery trajectories → threshold-based predictions → response predictions
  - Loss = BCE(mastery_based_predictions, actual_responses)
- Backpropagation path: `BCE_loss → ∂L/∂prediction → ∂L/∂mastery → ∂L/∂gains → ∂L/∂Encoder2_weights`
- Encoder 2 learns gain patterns that make mastery predictive of student performance

**Patterns Encoder 2 Should Learn**:
- Interaction quality (engagement, time spent, effort)
- Response correctness (correct responses → higher learning gains)
- Skill difficulty (harder skills require more practice for same gain)
- Prior mastery (diminishing returns as mastery approaches saturation)
- Temporal effects (spacing, recency, forgetting patterns)

#### 3. Practice Generates Skill-Specific Learning Gains

- When a student interacts with a question at step t, they practice the **relevant skills** (per Q-matrix)
- Each relevant skill experiences a **learning gain** Δm[skill, t]
- Gains are skill-specific: Δm[skill_A, t] ≠ Δm[skill_B, t] (NOT uniform!)
- Gains depend on:
  - Interaction quality ← **Learned by Encoder 2**
  - Skill difficulty (β_skill parameter)
  - Prior mastery level (sigmoid saturation effect)
  - Response correctness

#### 4. Mastery Accumulates Through Sigmoid Learning Curves (Monotonically)

- Skill mastery m[skill, t] increases **monotonically** with practice
- Growth follows sigmoid shape: slow initial growth → rapid learning → saturation
- Formula: m[skill, t] = M_sat × sigmoid(β × γ × effective_practice[skill, t] - offset)
- effective_practice[skill, t] = Σ(learning gains for that skill up to time t)
- **Monotonicity Guarantee**: Mastery never decreases (once learned, skills stay learned)

#### 5. Mastery Predicts Response Probability (Threshold Logic)

**Key Principle**: Response correctness depends on whether **ALL relevant skills are mastered**.

**Mastery Threshold (θ)**:
- A skill is considered "mastered" if `mastery[skill] ≥ θ`
- Typical threshold: θ = 0.80-0.85 (80-85% mastery required for competence)
- **Monotonicity**: Once mastered, always mastered (mastery only increases)

**Multi-Skill Logic** (For questions requiring multiple skills):
- Question Q requires skills {skill_A, skill_B, skill_C} (from Q-matrix)
- **Student succeeds** IF: ALL relevant skills are mastered
  - `mastery[skill_A] ≥ θ AND mastery[skill_B] ≥ θ AND mastery[skill_C] ≥ θ`
- **Student fails** IF: ANY relevant skill is not mastered
  - ∃ skill_i ∈ Q[question] such that `mastery[skill_i] < θ`

**Prediction Formulas**:
- Single-skill: `P(correct) = sigmoid((mastery[skill] - θ) / temp)`
- Multi-skill (conjunctive): `P(correct) = sigmoid((min(mastery[relevant_skills]) - θ) / temp)`
  - Takes minimum mastery (weakest skill determines success)
  - Implements AND logic: all skills must be mastered
- Alternative (compensatory): `P(correct) = sigmoid((mean(mastery[relevant_skills]) - θ) / temp)`
  - Allows partial compensation across skills

**Educational Interpretation**:
- ✅ "Student answered correctly because ALL relevant skills above threshold"
- ✅ "Student failed because Skill B not mastered (mastery=0.65 < θ=0.85)"
- ✅ "After 3 interactions practicing Skill B, mastery reached 0.87 → now succeeds"
- ✅ "Skill A mastered early (0.90), but Skill B weak (0.55) → fails multi-skill questions"

### Dual-Encoder Design Rationale

**Encoder 1 (Performance Path)**:
- Learns **direct response prediction** through unconstrained attention
- Optimizes for maximum AUC/accuracy
- No interpretability constraints
- Loss: BCE(base_predictions, responses)
- Black-box: Can learn any patterns that predict responses

**Encoder 2 (Interpretability Path)**:
- Learns **learning gains** that predict mastery, which in turn predicts responses
- Primary task: Estimate Δm[skill, t] for each relevant skill per interaction
- Secondary task: Mastery-based response prediction (through threshold logic)
- Loss: BCE(mastery_based_predictions, responses)
- Constraint: Predictions must come from mastery values derived from learned gains
- Interpretable: Every step from gains → mastery → prediction is transparent

**What Encoder 2 Weights Learn**:
- NOT: Direct response patterns (that's Encoder 1's job)
- YES: Gain patterns that make mastery trajectories predictive of responses
- Example: "Correct response + high engagement → large gain (0.8) for relevant skills"
- Example: "Incorrect response + low engagement → small gain (0.1)"
- Example: "Easy skills (low β) → larger gains per practice, hard skills (high β) → smaller gains"

### Why This Approach Enables Interpretability

The key innovation is that **Encoder 2 predicts responses THROUGH an interpretable mastery mechanism**, not directly:

```
Encoder 1 Path (Performance):
Input (Q, R) → Attention → Direct Prediction → BCE(prediction, response)
               ↓
          Black-box patterns

Encoder 2 Path (Interpretability):
Input (Q, R) → Attention → Learning Gains [per-skill] → Effective Practice → 
               ↓           ↓
          Learns gains   Skill-specific accumulation
          
→ Sigmoid Mastery → Threshold Check (≥θ?) → Prediction → BCE(prediction, response)
  ↓                 ↓                        ↓
  Monotonic        All skills mastered?     Same supervision
  trajectories     (AND logic)              as Encoder 1
```

This means:
- ✅ **Learning Gains**: "This interaction improved Skill A by 0.3, Skill B by 0.1"
- ✅ **Mastery Trajectories**: "Skill A: 0.45 → 0.68 → 0.85 (now mastered!)"
- ✅ **Threshold Logic**: "Skill A mastered (0.85 ≥ θ), Skill B not yet (0.65 < θ)"
- ✅ **Explainable Predictions**: "Failed because Skill B not mastered (0.65 < 0.85)"
- ✅ **Skill Differentiation**: Can see which skills student masters faster/slower
- ✅ **Monotonicity**: Once mastered (t=15), skill stays mastered for all t > 15

---

## Implementation Overview

**Implementation Files**:
- **Model**: `pykt/models/gainakt3_exp.py` (1014 lines) - standalone nn.Module with dual-encoder architecture
- **Training**: `examples/train_gainakt3exp.py` - zero defaults, explicit parameters
- **Evaluation**: `examples/eval_gainakt3exp.py` - test metrics + correlations
- **Trajectories**: `examples/learning_trajectories.py` - individual student analysis with dual-encoder predictions
- **Launcher**: `examples/run_repro_experiment.py` - loads defaults, manages experiments
- **Factory**: `create_exp_model(config)` - requires 24 explicit parameters
- **Test Scripts**: 
  - `tmp/test_sigmoid_curves.py` - validates sigmoid learning curve implementation
  - `tmp/test_dual_encoders.py` - validates dual-encoder architecture and gradient flow (ALL TESTS PASSED ✓)

---

## Dual-Encoder Architecture

**Current State**: The model uses **two completely independent encoder stacks** separating performance optimization from interpretability learning.

### Encoder 1 (Performance Path) - 96,513 parameters
- Input → Context_Embedding_1, Value_Embedding_1, Skill_Embedding_1 → Positional_Embeddings_1
- → Encoder_Blocks_1 (independent parameters) → [Context_1, Value_1, Skill_1] → Prediction_Head_1 
- → **Base Predictions** → **BCE Loss** (primary, weight ≈ 1.0)

### Encoder 2 (Interpretability Path) - 71,040 parameters
- Input → Context_Embedding_2, Value_Embedding_2 → Positional_Embeddings_2
- → Encoder_Blocks_2 (independent parameters) → Value_2
- → Gain Quality Estimation: `gain_quality = sigmoid(value_seq_2.mean(dim=-1))`
- → Differentiable Effective Practice: `effective_practice[t] = effective_practice[t-1] + gain_quality[t]`
- → Sigmoid Learning Curve: `mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × effective_practice[i,s,t] - offset)`
- → Threshold Mechanism: `sigmoid((mastery - θ_global) / temperature)` 
- → **Incremental Mastery Predictions** → **Incremental Mastery Loss** (interpretability, weight=0.1)

**Total Model Parameters**: 167,575

---

## Incremental Mastery Loss Mechanism

Provides interpretability-driven supervision by comparing ground truth responses with predictions derived from learned mastery trajectories following sigmoid learning curves.

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
- ✅ **Per-Skill Gains Output**: ACTIVE - Always computed via `gains_projection` layer and included in model outputs
- ✅ **Mastery Trajectories**: ACTIVE - Always computed (core feature, not optional)
- ❌ **Constraint Losses**: **COMMENTED OUT** (all weights=0.0, code preserved) - Non-negative, Monotonicity, Mastery-Perf, Gain-Perf, Sparsity, Consistency
- ❌ **Semantic Module Losses**: **COMMENTED OUT** (all disabled, code preserved) - Alignment, Global Alignment, Retention, Lag Gains

**Result**: Complete separation of performance optimization (Encoder 1) and interpretability learning (Encoder 2) with independent parameters. Differentiable effective practice mechanism enables gradients to flow through Encoder 2. Sigmoid curves automatically capture three learning phases (warm-up, growth, saturation). All constraint and semantic losses are commented out, leaving only BCE + Incremental Mastery losses active.

---

## Architecture Diagrams and Sequences

For detailed visual representations, see `paper/ARCH_gainakt3exp.md`:
- **Architecture Diagram**: Visual representation of dual-encoder design with color-coded components
- **Training Sequence**: Workflow from launcher to saved artifacts
- **Evaluation Sequence**: Test metrics and correlation computation
- **Learning Trajectory Analysis**: Individual student progression extraction
- **Internal Model Flow**: Data flow within forward_with_states()

**Key Flow Insights**:
- **Context Stream (h)**: → Prediction Head, Mastery Computation, Output/Monitor
- **Value Stream (v)**: → Prediction Head, Sigmoid Learning Curves (as gain estimates), Output/Monitor
- **Educational Semantics**: Values encode learning potential; sigmoid curves capture realistic learning dynamics

---



## Feature Implementation Status

Comprehensive analysis of each architectural component:

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

**Verification:** Monitor hub receives inputs from mastery/gain/predictions hubs via `forward_with_states()` method capturing all relevant tensors.

---

## Learning Trajectories

### Overview

Our proposal is based on the concept of *learning trajectories* that capture the evolution of student knowledge states over time. Each trajectory shows how mastery levels and learning gains progress as students practice skills, providing interpretability to the model while offering direct evidence of whether the model learns educationally meaningful patterns.

### Extraction Command

```bash
python examples/learning_trajectories.py \
    --run_dir examples/experiments/<experiment_folder> \
    --num_students 10 \
    --min_steps 10
```

**Output**: `learning_trajectories.csv` file with per-interaction data for selected students.

### Column Definitions

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `student_idx` | Student index in batch | Identifies individual learner |
| `step` | Interaction sequence number | Temporal ordering (1-indexed) |
| `skill_id` | Concept/question ID | Which skill is being practiced |
| `actual_response` | True student response (0/1) | Ground truth performance |
| `encoder1_pred` | Base prediction probability | Performance-optimized prediction |
| `encoder2_pred` | Mastery-based prediction | Interpretability-optimized prediction |
| `encoder1_match` | Prediction correctness (0/1) | Encoder 1 accuracy indicator |
| `encoder2_match` | Prediction correctness (0/1) | Encoder 2 accuracy indicator |
| `mastery` | Skill mastery level [0-1] | Knowledge state estimate |
| `expected_gain` | Predicted learning increment | How much learning occurred |
| `theta_global` | Global mastery threshold | Decision boundary for competence |
| `threshold_temp` | Temperature parameter | Prediction sharpness control |

### Key Metrics from Trajectories

1. **Mastery Progression**: Should increase monotonically with practice (sigmoid curve pattern)
2. **Encoder2 Accuracy**: `encoder2_match` rate indicates whether mastery predicts responses
3. **Gain-Response Correlation**: Correct responses should correlate with higher gains
4. **Learning Phases**: Early interactions (low mastery) → growth phase → saturation (mastery ≈ M_sat)

### Using Trajectories for Parameter Calibration

**Learning Curve Parameters** (from `configs/parameter_default.json`):

| Parameter | Current | Purpose | Calibration Signal |
|-----------|---------|---------|-------------------|
| `beta_skill_init` | 2.0 | Controls learning rate steepness | If mastery saturates too slowly → increase beta |
| `m_sat_init` | 0.8 | Maximum achievable mastery | If mastery plateaus below expected level → adjust M_sat |
| `gamma_student_init` | 1.0 | Student learning velocity | If some students learn faster → increase gamma |
| `sigmoid_offset` | 2.0 | Inflection point (when learning accelerates) | If mastery starts too late → decrease offset |
| `mastery_threshold_init` | 0.85 | Decision boundary for competence | If encoder2_match is low → adjust threshold |
| `threshold_temperature` | 1.0 | Prediction sharpness | If predictions too confident/uncertain → adjust temp |

**Diagnostic Patterns**:

- **Low encoder2_match (< 50%)**: Mastery values don't predict responses → Check if learning curves are learning (non-zero gradients) or adjust `bce_loss_weight` to increase IM loss signal
- **Flat mastery trajectories**: No learning progression → Increase `beta_skill_init` or decrease `sigmoid_offset` for earlier learning
- **Mastery overshoots**: Values exceed expected levels → Reduce `m_sat_init` or increase `sigmoid_offset`
- **All students identical mastery**: `gamma_student_init` not differentiating → Check if student IDs are provided or enable student-specific parameters

**Iterative Calibration**:
1. Extract trajectories from current experiment
2. Identify which phase (warm-up/growth/saturation) has issues
3. Adjust corresponding parameter(s)
4. Re-train and compare trajectories
5. Repeat until mastery correlates with performance (encoder2_match > 70%)

## Current Model Status Summary

**Dual-Encoder Features**:
- ✅ **TWO COMPLETELY INDEPENDENT ENCODER STACKS** (167,575 total parameters)
  - Encoder 1 (Performance): 96,513 params → Base Predictions → BCE Loss
  - Encoder 2 (Interpretability): 71,040 params → Mastery → IM Loss  
  - Sigmoid parameters: 22 params
- ✅ **Dual Predictions**: Base (from Encoder 1) + Incremental Mastery (from Encoder 2)
- ✅ **Differentiable Effective Practice**: Quality-weighted accumulation enables Encoder 2 gradient flow
- ✅ **Sigmoid Learning Curves**: Three automatic phases (warm-up → growth → saturation)
- ✅ **Per-Skill Gains**: Always computed and included in outputs (core feature)
- ✅ **Mastery Trajectories**: Always computed (core feature)
- ❌ **Constraint Losses**: COMMENTED OUT (weights=0.0, code preserved)
- ❌ **Semantic Module Losses**: COMMENTED OUT (all disabled, code preserved)

**Implementation Status**: ✅ **FULLY IMPLEMENTED AND TESTED** (2025-11-16)
- Test script: `tmp/test_dual_encoders.py` - ALL 7 TESTS PASSED ✓
- Gradient flow verified through both encoders
- Independent learning confirmed

**Code Location**: `pykt/models/gainakt3_exp.py` (1014 lines)
- Key sections: See `paper/STATUS_gainakt3exp.md` for detailed line references

**Rationale**: Clean separation between performance optimization and interpretability learning. Differentiable effective practice ensures end-to-end training of both pathways.

---

## Training and Evaluation Workflow

Following PyKT framework standards (see `assistant/quickstart.pdf` and `assistant/contribute.pdf`).

### Pipeline Overview

**Training** (`examples/train_gainakt3exp.py`):
- Zero hardcoded defaults—all parameters must be explicit
- Launched via `run_repro_experiment.py` which loads defaults from `configs/parameter_default.json`
- Saves complete experiment artifacts: checkpoints, config, trajectories, metrics
- Supports semantic trajectory tracking (mastery/gain correlations per epoch)

**Evaluation** (`examples/eval_gainakt3exp.py`):
- Loads trained model and computes test metrics (AUC, accuracy)
- Computes mastery/gain correlations (configurable student sample size)
- Saves evaluation results with timestamp

**Learning Trajectory Analysis** (`examples/learning_trajectories.py`):
- Extracts and displays individual student learning progressions
- Shows per-timestep: skills practiced, gains, mastery, predictions vs truth
- Compact tabular format with student summary statistics
- Command automatically added to `config.json` for easy access

**Model Creation**: Models instantiated via `create_exp_model(config)` requiring all parameters in config dict (no defaults), ensuring reproducibility.

### Usage Commands

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

**Note:** Trajectory command automatically included in `config.json` during training.

---

## Reproducibility System

Following "zero defaults" pattern documented in `examples/reproducibility.md`.

### Experiment Structure

Each training run creates timestamped directory:
```
saved_model/{timestamp}_{model}_{title}_{uid}/
  ├── config.json           # Complete parameter set + trajectory command
  ├── model_*.pt            # Checkpoints
  ├── metrics.json          # Training metrics (AUC, accuracy, correlations)
  ├── trajectory_*.json     # Semantic trajectory data (optional)
  └── eval_results_*.json   # Evaluation results
```

### Config File Contents
- `defaults`: All parameter default values (from `parameter_default.json`)
- `train_explicit`: Full training command with all parameters
- `eval_explicit`: Full evaluation command with all parameters  
- `trajectory_command`: Trajectory analysis command (10 students, min 10 steps)
- `metadata`: Timestamp, git hash, hostname, GPU info

### Reproducibility Guarantees
1. No hardcoded defaults in model or training code
2. All parameters explicitly passed via CLI
3. Complete config saved with every experiment
4. Commands reconstructable from config alone
5. Git hash and environment info captured

See `examples/reproducibility.md` for complete parameter evolution protocol.

---
## Parameter Evolution Protocol

Follow rigorous protocol to maintain reproducibility. See `examples/reproducibility.md` for complete details.

### When Adding/Modifying Parameters

1. **Update `configs/parameter_default.json`** with new parameter and default value
2. **Update `examples/run_repro_experiment.py`** to add CLI argument (with `required=True`, no default)
3. **Update `paper/parameters.csv`** with documentation (category, description, default)
4. **Verify no hardcoded defaults** exist in model/training code
5. **Recompute MD5 hash** for parameter_default.json

### Consistency Verification

After codebase modifications:
- Check no hidden parameters with hardcoded defaults exist
- Verify all model parameters passed explicitly via `create_exp_model(config)`
- Ensure training scripts require all parameters via CLI
- Confirm `parameter_default.json` matches actual code behavior

### Experimental Scenarios

**Hyperparameter Optimization**:
1. Use defaults from `configs/parameter_default.json` as starting point
2. Run experiments with CLI overrides
3. Document results in auto-generated config files
4. Update defaults with optimal configuration
5. Document change in `paper/parameters.csv` with rationale and date

**Ablation Studies**:
- Deactivate parameters one by one to measure impact
- Change boolean values, set weights to 0, etc.

**Benchmarking**:
- Compare metrics of different models/variants
- Use defaults to launch training and evaluation

