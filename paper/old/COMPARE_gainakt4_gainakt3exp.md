# Why GainAKT4 Underperforms Compared to GainAKT3Exp

## Performance Gap

**Test AUC Results:**
- GainAKT3Exp: **0.7183** (best)
- GainAKT4 (λ=0.9): **0.7069** (-1.6%)
- GainAKT4 (λ=1.0): **0.7063** (-1.7%)

## Critical Architectural Differences

### 1. **ENCODER ARCHITECTURE: Single vs Dual**

#### GainAKT3Exp (DUAL-ENCODER):
```
Input → Encoder 1 (Performance Path) → predictions → BCE Loss (weight ≈ 0.9)
     └→ Encoder 2 (Mastery Path) → sigmoid curves → IM predictions → IM Loss (weight ≈ 0.1)
```
- **TWO completely independent encoder stacks**
- **Separate embedding tables** for each encoder
- **Different attention patterns** learned by each encoder
- Encoder 1: Learns response correctness patterns
- Encoder 2: Learns learning gains patterns
- **Double the parameters** for richer representations

#### GainAKT4 (SINGLE-ENCODER):
```
Input → Encoder 1 → h1 ──┬→ Head 1 (Performance) → BCE predictions → BCE Loss
                         └→ Head 2 (Mastery) → skill vector → mastery predictions → Mastery Loss
```
- **ONE shared encoder** for both tasks
- **Same embeddings** for both heads
- **Same attention patterns** must serve both objectives
- Single representation bottleneck
- **Half the encoder parameters** compared to GainAKT3Exp

### 2. **MASTERY COMPUTATION: Sigmoid Learning Curves vs Simple MLP**

#### GainAKT3Exp (Educationally-Grounded):
```python
# Sigmoid learning curves with per-skill and per-student parameters
# mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × practice_count[i,s,t] - offset)

# Learnable parameters:
self.beta_skill = nn.Parameter(torch.ones(num_c))        # Skill difficulty
self.M_sat = nn.Parameter(torch.ones(num_c) * 0.8)      # Saturation level
self.gamma_student = nn.Parameter(torch.ones(num_students))  # Learning velocity
self.theta_global = nn.Parameter(torch.tensor(0.85))     # Performance threshold
self.offset = nn.Parameter(torch.tensor(3.0))            # Inflection point

# Practice-count driven, automatic learning phases
# Intrinsically monotonic, bounded, and interpretable
```

**Benefits:**
- Educationally grounded (learning theory)
- Automatic monotonicity (practice → higher mastery)
- Bounded outputs (0 to M_sat)
- Per-skill difficulty modeling
- Per-student learning rate modeling
- Rich parameterization (5 learnable parameter types)

#### GainAKT4 (Simplified MLP):
```python
# Two-stage MLP with enforced monotonicity
self.mlp1 = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_ff, num_c),
    nn.Softplus()  # Ensures positivity
)

# Enforced monotonicity via cummax (post-hoc)
kc_vector_mono = torch.cummax(kc_vector, dim=1)[0]

self.mlp2 = nn.Sequential(
    nn.Linear(num_c, num_c // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(num_c // 2, 1)
)
```

**Limitations:**
- No educational grounding
- Monotonicity enforced artificially (cummax)
- No explicit skill difficulty modeling
- No per-student learning rate
- Simple feed-forward (less expressive)
- Must learn everything from data

### 3. **PREDICTION MECHANISM**

#### GainAKT3Exp:
```python
# Encoder 1 predictions (main task)
predictions = sigmoid(prediction_head_1([context1, value1, skill1]))

# Encoder 2 incremental mastery predictions (auxiliary task)
# Derived from sigmoid learning curves
incremental_mastery_predictions = sigmoid((skill_mastery - θ_global) / temperature)

# Both contribute to final performance
```
- **Two independent prediction pathways**
- Primary path optimized purely for accuracy
- Secondary path provides regularization through educational constraints
- Weighted combination benefits from both

#### GainAKT4:
```python
# Single encoder, two heads competing for representation
bce_predictions = sigmoid(prediction_head([h1, v1, skill_emb]))
mastery_predictions = sigmoid(mlp2(kc_vector))

# Both heads pull the encoder in different directions
```
- **Single representation must serve both tasks**
- Heads compete for encoder capacity
- Gradient conflicts during backpropagation
- No independent optimization per task

### 4. **LOSS COMPUTATION**

#### GainAKT3Exp:
```python
# Base BCE loss (Encoder 1)
base_bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

# Incremental mastery loss (Encoder 2)
incremental_mastery_loss = F.binary_cross_entropy(
    incremental_mastery_predictions, targets
)

# Total loss with independent encoder optimization
total_loss = bce_weight * base_bce_loss + (1-bce_weight) * incremental_mastery_loss
```
- Each encoder can optimize independently
- No gradient conflicts
- IM loss provides soft regularization
- BCE loss dominates (weight ≈ 0.9)

#### GainAKT4:
```python
# Both losses computed from same encoder
bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
mastery_loss = F.binary_cross_entropy_with_logits(mastery_logits, targets)

total_loss = lambda_bce * bce_loss + (1-lambda_bce) * mastery_loss
```
- **Gradient conflicts**: both heads pull encoder differently
- When λ_bce=1.0, mastery head doesn't learn (AUC=0.5)
- When λ_bce=0.9, mastery head barely learns (AUC=0.65)
- Shared representation bottleneck limits both tasks

## Parameter Count Comparison

### GainAKT3Exp:
- Encoder 1: embeddings (3x) + transformer blocks + prediction head
- Encoder 2: embeddings (2x) + transformer blocks
- Sigmoid parameters: β_skill, M_sat, γ_student, θ_global, offset
- **Total: ~2× encoder parameters + rich mastery modeling**

### GainAKT4:
- Single encoder: embeddings (3x) + transformer blocks
- Head 1: prediction MLP
- Head 2: MLP1 + MLP2 (mastery estimation)
- **Total: ~1× encoder parameters + simple MLPs**

**Parameter efficiency paradox**: GainAKT3Exp has MORE parameters but achieves BETTER performance because:
- Each encoder specializes without conflicts
- Educationally-grounded mastery modeling
- No representation bottleneck

## Root Causes of Underperformance

### 1. **Representation Bottleneck**
- Single encoder must encode information for BOTH performance prediction AND mastery estimation
- These tasks have different optimal representations
- Performance: needs to capture immediate response patterns
- Mastery: needs to capture long-term learning trajectories
- One encoder cannot optimally serve both

### 2. **Gradient Conflicts**
- BCE loss wants encoder to focus on prediction accuracy
- Mastery loss wants encoder to learn interpretable trajectories
- Gradients from both heads compete during backprop
- Result: suboptimal for both objectives

### 3. **Simplified Mastery Modeling**
- MLP lacks educational grounding
- No explicit skill difficulty modeling
- No per-student learning rate
- Cummax enforces monotonicity post-hoc (not learned)
- Less expressive than sigmoid learning curves

### 4. **Missing Specialization**
- GainAKT3Exp: Encoder 1 learns "What will the student answer?"
- GainAKT3Exp: Encoder 2 learns "How much did the student learn?"
- GainAKT4: Single encoder tries to learn both simultaneously
- Result: neither objective achieved optimally

## Recommendations to Improve GainAKT4

### Option 1: Add Second Encoder (becomes GainAKT3Exp-like)
- Restore dual-encoder architecture
- Separate embeddings for each task
- Independent optimization

### Option 2: Improve Mastery Head
- Replace MLP with sigmoid learning curves
- Add learnable skill difficulty parameters
- Add per-student learning rate modeling
- Make monotonicity intrinsic, not enforced

### Option 3: Better Multi-Task Learning
- Use gradient normalization (e.g., GradNorm)
- Dynamic loss weighting based on task difficulty
- Task-specific layer normalization
- Separate batch normalization statistics

### Option 4: Increase Model Capacity
- Deeper encoder (more blocks)
- Wider encoder (larger d_model)
- More attention heads
- Attempt to compensate for single-encoder bottleneck

## Conclusion

**GainAKT4's underperformance is architectural, not a tuning issue:**

1. **Single-encoder bottleneck** prevents optimal task-specific learning
2. **Gradient conflicts** from multi-task learning degrade both objectives  
3. **Simplified mastery modeling** lacks educational grounding
4. **Missing specialization** - one encoder can't optimally serve both tasks

**GainAKT3Exp's superior performance comes from:**

1. **Dual-encoder specialization** - each encoder optimizes for its task
2. **No gradient conflicts** - independent parameter sets
3. **Educationally-grounded mastery** - sigmoid learning curves with rich parameterization
4. **Task-appropriate representations** - no forced bottleneck

**Verdict**: The architectural simplification in GainAKT4 sacrificed the key innovations that made GainAKT3Exp effective. To match GainAKT3Exp's performance, GainAKT4 would need to adopt similar complexity (dual encoders, sigmoid curves), at which point it would essentially be reimplementing GainAKT3Exp.
