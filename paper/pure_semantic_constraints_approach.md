# Pure Semantic Constraints for Mastery Growth: Regularization Approach

## Core Hypothesis

**"Semantic constraints on mastery dynamics should act as regularization, improving both interpretability AND performance."**

Rationale:
- Mastery growing monotonically ∈ [0,1] is pedagogically correct
- Enforcing this shouldn't hurt Head 1; it should help by making the encoder learn better representations
- The encoder benefits from learning features that satisfy both:
  1. Predict next-step correctness (Head 1)
  2. Exhibit pedagogically sound mastery growth (Head 2 semantic constraints)

---

## Solutions Categorized by Nature

### Category A: **Pure Semantic Constraints** (Regularization) ⭐

These enforce mathematical/pedagogical properties without introducing task-specific heuristics.

#### A1: **Monotonicity Constraint** (Already Implemented)
```python
kc_vector_mono = torch.cummax(kc_vector, dim=1)[0]
```
- ✅ Pure semantic: mastery never decreases
- ✅ No heuristics, just mathematical property
- ❌ **Problem**: Allows flat sequences (no growth enforcement)

#### A2: **Boundedness Constraint** 
```python
kc_vector = torch.clamp(kc_vector, 0.0, 1.0)
```
- ✅ Pure semantic: mastery is a probability/proportion
- ✅ No heuristics
- ✅ Already works well

#### A3: **Smoothness Constraint** (Loss-based regularization)
```python
def compute_smoothness_loss(kc_vector):
    """
    Penalize abrupt changes in mastery levels.
    Encourages gradual, pedagogically realistic growth.
    """
    # Second-order differences: discourage sharp turns
    second_diff = kc_vector[:, 2:] - 2*kc_vector[:, 1:-1] + kc_vector[:, :-2]
    smoothness_loss = second_diff.pow(2).mean()
    return smoothness_loss

# Add to total loss
L_smooth = compute_smoothness_loss(kc_vector)
total_loss = lambda_bce * L1 + lambda_mastery * L2 + 0.01 * L_smooth
```

**Properties:**
- ✅ Pure semantic: learning is gradual, not chaotic
- ✅ Regularization: encourages well-behaved representations
- ✅ No task-specific heuristics
- ✅ Should **help** encoder learn smoother knowledge states

#### A4: **Non-Decreasing Increment Constraint** (Architectural)
```python
# Instead of predicting absolute values, predict logarithmic increments
# This guarantees positive growth

# In __init__
self.mlp1_log_increment = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_ff, num_c),
    # No activation - can be negative (log space)
)

# In forward()
if self.lambda_mastery > 0:
    # Predict log-increments
    log_increments = self.mlp1_log_increment(h1)  # [B, L, num_c], unbounded
    
    # Exponentiate to get positive increments (always > 0)
    increments = torch.exp(log_increments) * 0.01  # Scale to reasonable range
    # Note: exp() guarantees positivity, so increments ∈ (0, ∞)
    
    # Cumulative sum for monotonicity
    initial = torch.zeros(batch_size, 1, num_c, device=q.device)
    kc_vector = initial + torch.cumsum(increments, dim=1)
    kc_vector = torch.clamp(kc_vector, 0.0, 1.0)  # Bound to [0,1]
    
    mastery_logits = self.mlp2(kc_vector).squeeze(-1)
    mastery_predictions = torch.sigmoid(mastery_logits)
```

**Properties:**
- ✅ **Guarantees non-zero growth** (exp always > 0)
- ✅ Pure semantic: monotonic increase is architectural
- ✅ No heuristics about correct/incorrect responses
- ✅ Network learns **rate of growth** (log-increments)
- ✅ Should regularize encoder without hurting Head 1

---

### Category B: **Contrastive Learning** (Semantic Regularization) ⭐⭐

#### B1: **Temporal Contrastive Loss**
```python
def compute_temporal_contrastive_loss(kc_vector, temperature=0.07):
    """
    Enforce that mastery states are temporally ordered:
    - States closer in time should be more similar
    - States far apart in time should be distinct
    
    This is a SEMANTIC constraint: learning progresses gradually.
    """
    B, L, C = kc_vector.shape
    
    # Normalize skill vectors
    kc_norm = F.normalize(kc_vector, p=2, dim=-1)  # [B, L, C]
    
    # Compute pairwise similarities within each sequence
    # kc_norm[b, i] · kc_norm[b, j] = similarity between timestep i and j for student b
    loss = 0.0
    
    for b in range(B):
        # Similarity matrix for student b: [L, L]
        sim_matrix = torch.matmul(kc_norm[b], kc_norm[b].t()) / temperature
        
        # For each timestep t, positive = t+1, negatives = all others
        for t in range(L - 1):
            # Positive: next timestep
            positive = sim_matrix[t, t+1]
            
            # Negatives: all timesteps except t and t+1
            negatives_mask = torch.ones(L, dtype=torch.bool, device=q.device)
            negatives_mask[t] = False
            negatives_mask[t+1] = False
            negatives = sim_matrix[t, negatives_mask]
            
            # Contrastive loss: maximize positive, minimize negatives
            # exp(positive) / (exp(positive) + sum(exp(negatives)))
            numerator = torch.exp(positive)
            denominator = numerator + torch.sum(torch.exp(negatives))
            loss += -torch.log(numerator / denominator)
    
    return loss / (B * (L - 1))

# Add to total loss
L_contrast = compute_temporal_contrastive_loss(kc_vector)
total_loss = lambda_bce * L1 + lambda_mastery * L2 + 0.05 * L_contrast
```

**Properties:**
- ✅ **Pure semantic**: learning is gradual (nearby states similar)
- ✅ **Forces temporal structure**: prevents flat sequences
- ✅ **Regularization**: encoder learns to represent temporal progression
- ✅ **No heuristics**: doesn't assume anything about correct/incorrect
- ✅ **Should help Head 1**: temporal structure benefits next-step prediction

#### B2: **Skill-Specific Contrastive Loss**
```python
def compute_skill_contrastive_loss(kc_vector, responses):
    """
    Within each skill, encourage:
    - Mastery states after correct responses to be distinct from incorrect
    - Temporal progression within correct/incorrect groups
    
    Semantic: correct responses → higher mastery growth
    """
    B, L, C = kc_vector.shape
    loss = 0.0
    
    for skill_idx in range(C):
        # Extract mastery trajectory for this skill: [B, L]
        skill_mastery = kc_vector[:, :, skill_idx]
        
        # Flatten across batch: [B*L]
        skill_flat = skill_mastery.reshape(-1)
        responses_flat = responses.reshape(-1)
        
        # Separate indices: correct vs incorrect
        correct_idx = (responses_flat == 1).nonzero(as_tuple=True)[0]
        incorrect_idx = (responses_flat == 0).nonzero(as_tuple=True)[0]
        
        if len(correct_idx) > 1 and len(incorrect_idx) > 1:
            # Correct states should be more similar to each other
            correct_states = skill_flat[correct_idx]
            incorrect_states = skill_flat[incorrect_idx]
            
            # Mean mastery: correct should be > incorrect
            mean_correct = correct_states.mean()
            mean_incorrect = incorrect_states.mean()
            
            # Contrastive loss: encourage separation
            margin = 0.1  # Correct should be at least 0.1 higher
            separation_loss = F.relu(margin - (mean_correct - mean_incorrect))
            loss += separation_loss
    
    return loss / C

# Add to total loss
L_skill_contrast = compute_skill_contrastive_loss(kc_vector, r)
total_loss = lambda_bce * L1 + lambda_mastery * L2 + 0.05 * L_skill_contrast
```

**Properties:**
- ✅ **Semantic**: correct responses → higher mastery (pedagogically sound)
- ✅ **Soft constraint**: uses margin loss, not hard rules
- ✅ **Per-skill learning**: respects skill independence
- ✅ **Should help Head 1**: separating correct/incorrect is useful for prediction

---

### Category C: **Heuristic-Based** (Not Pure Semantic) ❌

These introduce task-specific rules rather than pure semantic constraints.

#### C1: Rule-based increments (0.15 for correct, 0.03 for incorrect)
- ❌ Not pure semantic: specific to response correctness
- ❌ Hardcoded magnitudes
- ❌ Less "learned", more "engineered"

#### C2: Post-processing blending
- ❌ Not pure semantic: artificial mixing
- ❌ Reduces gradient flow

---

## Recommended Pure Semantic Approach ⭐⭐⭐

**Combine A4 (Log-Increment Architecture) + B1 (Temporal Contrastive) + A3 (Smoothness)**

```python
class GainAKT4(nn.Module):
    def __init__(self, ...):
        # ... existing code ...
        
        # CHANGE: MLP1 predicts log-increments (architectural guarantee of growth)
        self.mlp1_log_increment = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_c)
            # No final activation - log space
        )
        
        # Learnable scale for increments
        self.increment_scale = nn.Parameter(torch.tensor(-2.0))  # log(0.01) ≈ -4.6
    
    def forward(self, q, r, qry=None):
        # ... existing encoder code produces h1 ...
        
        if self.lambda_mastery > 0:
            # Predict log-increments
            log_increments = self.mlp1_log_increment(h1)  # [B, L, num_c]
            
            # Convert to positive increments
            # exp(log_increments + scale) gives increments ∈ (0, ∞)
            increments = torch.exp(log_increments + self.increment_scale)
            # With scale=-2, typical increments ≈ 0.01-0.2 range
            
            # Cumulative sum: guaranteed monotonic growth
            initial = torch.zeros(batch_size, 1, self.num_c, device=q.device)
            kc_vector = initial + torch.cumsum(increments, dim=1)
            
            # Clamp to [0, 1]: semantic bound
            kc_vector = torch.clamp(kc_vector, 0.0, 1.0)
            
            # Mastery prediction
            mastery_logits = self.mlp2(kc_vector).squeeze(-1)
            mastery_predictions = torch.sigmoid(mastery_logits)
            
            return {
                'bce_predictions': bce_predictions,
                'mastery_predictions': mastery_predictions,
                'skill_vector': kc_vector,
                'increments': increments,  # For loss computation
                'logits': logits,
                'mastery_logits': mastery_logits
            }
    
    def compute_loss(self, output, targets, responses):
        # Standard losses
        bce_loss = F.binary_cross_entropy_with_logits(output['logits'], targets)
        
        if output['mastery_logits'] is not None:
            mastery_loss = F.binary_cross_entropy_with_logits(
                output['mastery_logits'], targets
            )
            
            # SEMANTIC CONSTRAINT 1: Smoothness (gradual learning)
            kc_vector = output['skill_vector']
            second_diff = kc_vector[:, 2:] - 2*kc_vector[:, 1:-1] + kc_vector[:, :-2]
            smoothness_loss = second_diff.pow(2).mean()
            
            # SEMANTIC CONSTRAINT 2: Temporal contrastive (progressive learning)
            temporal_contrast_loss = self.compute_temporal_contrastive_loss(
                kc_vector, temperature=0.07
            )
            
            # SEMANTIC CONSTRAINT 3: Skill separation (correct vs incorrect)
            skill_contrast_loss = self.compute_skill_contrastive_loss(
                kc_vector, responses
            )
        else:
            mastery_loss = 0.0
            smoothness_loss = 0.0
            temporal_contrast_loss = 0.0
            skill_contrast_loss = 0.0
        
        # Weighted combination (all semantic constraints act as regularizers)
        total_loss = (self.lambda_bce * bce_loss + 
                     self.lambda_mastery * mastery_loss +
                     0.01 * smoothness_loss +           # Regularizer 1
                     0.05 * temporal_contrast_loss +    # Regularizer 2
                     0.05 * skill_contrast_loss)        # Regularizer 3
        
        return total_loss, {
            'bce_loss': bce_loss.item(),
            'mastery_loss': mastery_loss.item() if mastery_loss != 0.0 else 0.0,
            'smoothness_loss': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else 0.0,
            'temporal_contrast_loss': temporal_contrast_loss.item() if isinstance(temporal_contrast_loss, torch.Tensor) else 0.0,
            'skill_contrast_loss': skill_contrast_loss.item() if isinstance(skill_contrast_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
    
    def compute_temporal_contrastive_loss(self, kc_vector, temperature=0.07):
        """Enforce temporal progression (pure semantic constraint)."""
        B, L, C = kc_vector.shape
        
        # Normalize
        kc_norm = F.normalize(kc_vector, p=2, dim=-1)
        
        loss = 0.0
        for b in range(B):
            sim_matrix = torch.matmul(kc_norm[b], kc_norm[b].t()) / temperature
            
            for t in range(L - 1):
                positive = sim_matrix[t, t+1]
                negatives_mask = torch.ones(L, dtype=torch.bool, device=kc_vector.device)
                negatives_mask[t] = False
                negatives_mask[t+1] = False
                negatives = sim_matrix[t, negatives_mask]
                
                numerator = torch.exp(positive)
                denominator = numerator + torch.sum(torch.exp(negatives))
                loss += -torch.log(numerator / (denominator + 1e-8))
        
        return loss / (B * (L - 1)) if B * (L - 1) > 0 else torch.tensor(0.0, device=kc_vector.device)
    
    def compute_skill_contrastive_loss(self, kc_vector, responses):
        """Separate mastery states by response correctness (semantic constraint)."""
        B, L, C = kc_vector.shape
        loss = 0.0
        valid_skills = 0
        
        for skill_idx in range(C):
            skill_mastery = kc_vector[:, :, skill_idx].reshape(-1)
            responses_flat = responses.reshape(-1)
            
            correct_idx = (responses_flat == 1).nonzero(as_tuple=True)[0]
            incorrect_idx = (responses_flat == 0).nonzero(as_tuple=True)[0]
            
            if len(correct_idx) > 0 and len(incorrect_idx) > 0:
                mean_correct = skill_mastery[correct_idx].mean()
                mean_incorrect = skill_mastery[incorrect_idx].mean()
                
                # Margin loss: correct should be higher
                margin = 0.1
                separation = F.relu(margin - (mean_correct - mean_incorrect))
                loss += separation
                valid_skills += 1
        
        return loss / valid_skills if valid_skills > 0 else torch.tensor(0.0, device=kc_vector.device)
```

---

## Why This Approach Aligns with Your Philosophy

1. **Log-increment architecture**: Pure mathematical guarantee (exp > 0)
2. **Smoothness loss**: Pedagogical constraint (learning is gradual)
3. **Temporal contrastive**: Semantic constraint (progression structure)
4. **Skill separation**: Pedagogical constraint (correct → higher mastery)

**All constraints are regularizers:**
- Help encoder learn better representations
- Should **improve** Head 1 performance (better features)
- No hardcoded task-specific rules
- Fully differentiable and learnable

**Expected outcome:**
- Encoder learns representations that:
  - Predict next-step correctness well (Head 1)
  - Exhibit pedagogically sound mastery dynamics (Head 2 + semantic constraints)
- Semantic constraints act as **curriculum** for the encoder

---

## Answer to Your Question

**"What solutions match with this pure semantic constraints approach?"**
- ✅ A4: Log-increment architecture (mathematical guarantee)
- ✅ A3: Smoothness loss (pedagogical regularity)
- ✅ B1: Temporal contrastive loss (progression structure)
- ✅ B2: Skill contrastive loss (response-mastery relationship)

**"What about contrastive learning in particular?"**
- ✅ **Yes, contrastive learning is a pure semantic constraint!**
- It enforces temporal structure without heuristics
- It should **help** Head 1 by making the encoder learn better temporal representations
- It's a regularizer that benefits both heads

**Should NOT use:**
- ❌ Hardcoded increment magnitudes (0.15 for correct, etc.)
- ❌ Post-processing blends with detach()
- ❌ Any approach that reduces gradient flow

---

## Implementation Priority

**Phase 1: Architectural guarantee** (must-have)
- Implement log-increment MLP1 (A4)
- This alone solves the flat mastery problem

**Phase 2: Add semantic regularizers** (recommended)
- Add temporal contrastive loss (B1)
- Add smoothness loss (A3)
- Add skill contrastive loss (B2)

**Phase 3: Tune weights** (experimental)
- Start with small weights (0.01-0.05)
- Monitor Head 1 performance - it should improve or stay stable
- If Head 1 improves → your hypothesis is validated!

Would you like me to implement Phase 1 + Phase 2?
