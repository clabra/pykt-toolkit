# Improving Incremental Mastery (IM) Predictions in GainAKT3Exp

## Current Problem

The Incremental Mastery encoder (Encoder 2) produces predictions with AUC ≈ 0.50 (random performance), while the BCE encoder (Encoder 1) achieves AUC ≈ 0.71.

### Root Cause

IM predictions are computed by transforming monotonic mastery states:

```python
incremental_mastery_predictions = torch.sigmoid((skill_mastery - θ) / temperature)
```

**Why this fails:**
- `skill_mastery` increases monotonically with practice (reflects cumulative learning)
- Actual correctness varies per question (depends on difficulty, skill type, etc.)
- A monotonic signal cannot predict variable outcomes → AUC ≈ 0.5

### Current Architecture

```
Encoder 1 (BCE):
  Transformer → Logits → Sigmoid → Predictions ✓ (AUC = 0.71)

Encoder 2 (IM):
  Transformer → Mastery States → Sigmoid((M - θ)/T) → "Predictions" ✗ (AUC = 0.50)
                     ↓
              (monotonic)
```

## Recommended Solution

Add a **prediction head** to Encoder 2 that combines mastery state with question features to produce learned predictions.

### Proposed Architecture

```
Encoder 2 (IM) - Improved:
  Transformer → Mastery States → [Mastery + Question Features] → MLP → Predictions ✓
                     ↓                        ↓
              (interpretable)         (question-specific)
```

## Implementation Options

### Option 1: Simple Linear Prediction Head (Recommended Start)

**Pros**: Easy to implement, fast training, minimal parameters
**Cons**: Limited expressiveness
**Expected IM AUC**: 0.55-0.60

#### Code Changes

**1. Add prediction head in `gainakt3_exp.py` `__init__`:**

```python
# Encoder 2: Incremental Mastery prediction head
# Input: mastery state (1) + question embedding (d_model)
self.im_prediction_head = nn.Linear(1 + d_model, 1)
```

**2. Modify `forward_with_states` to compute predictions:**

```python
# After computing skill_mastery
# Extract question embeddings (already computed by transformer)
question_embeds = self.question_embedding(questions_shifted)  # [B, T, d_model]

# Prepare features for IM prediction
# Expand mastery to match question embeddings
batch_size, seq_len, num_skills = skill_mastery.shape

# Get active skill mastery for each timestep
# Assuming skill IDs are in questions_shifted
skill_ids = questions_shifted  # [B, T]
active_mastery = torch.gather(
    skill_mastery, 
    dim=2, 
    index=skill_ids.unsqueeze(-1)
).squeeze(-1)  # [B, T]

# Combine mastery + question features
im_features = torch.cat([
    active_mastery.unsqueeze(-1),  # [B, T, 1]
    question_embeds                # [B, T, d_model]
], dim=-1)  # [B, T, 1 + d_model]

# Predict correctness
incremental_mastery_predictions = torch.sigmoid(
    self.im_prediction_head(im_features).squeeze(-1)
)  # [B, T]
```

### Option 2: Multi-Layer Prediction Head (More Expressive)

**Pros**: Better capacity, can learn complex relationships
**Cons**: More parameters, risk of overfitting
**Expected IM AUC**: 0.60-0.65

#### Code Changes

**1. Add MLP prediction head:**

```python
# Encoder 2: Incremental Mastery prediction head (MLP)
self.im_prediction_head = nn.Sequential(
    nn.Linear(1 + d_model, d_model // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, d_model // 4),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 4, 1)
)
```

**2. Forward pass (same as Option 1)**

### Option 3: Include Gain Signal (Enhanced Context)

**Pros**: Leverages gain information, richer features
**Cons**: More complex, requires careful tuning
**Expected IM AUC**: 0.60-0.67

#### Code Changes

**1. Extend feature vector:**

```python
# Get active skill mastery and gains
active_mastery = torch.gather(skill_mastery, dim=2, index=skill_ids.unsqueeze(-1)).squeeze(-1)
active_gains = torch.gather(skill_gains, dim=2, index=skill_ids.unsqueeze(-1)).squeeze(-1)

# Combine mastery + gains + question features
im_features = torch.cat([
    active_mastery.unsqueeze(-1),   # [B, T, 1]
    active_gains.unsqueeze(-1),     # [B, T, 1]
    question_embeds                 # [B, T, d_model]
], dim=-1)  # [B, T, 2 + d_model]

# Predict
self.im_prediction_head = nn.Sequential(
    nn.Linear(2 + d_model, d_model // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, 1)
)
```

### Option 4: Attention-Based Prediction (Most Sophisticated)

**Pros**: Maximum flexibility, can learn complex interactions
**Cons**: Most parameters, slowest training, hardest to debug
**Expected IM AUC**: 0.62-0.70

#### Code Changes

```python
# Encoder 2: Attention-based prediction
self.im_attention_predictor = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=n_heads,
    dropout=dropout,
    batch_first=True
)
self.im_projection = nn.Linear(d_model, 1)

# In forward_with_states:
# Use mastery as query, question embeddings as key/value
mastery_query = self.mastery_projection(active_mastery.unsqueeze(-1))  # [B, T, d_model]
attended_features, _ = self.im_attention_predictor(
    query=mastery_query,
    key=question_embeds,
    value=question_embeds
)
incremental_mastery_predictions = torch.sigmoid(
    self.im_projection(attended_features).squeeze(-1)
)
```

## Alternative: Minimal Question Difficulty Adjustment

If full prediction head is too complex, adjust mastery threshold by question difficulty:

```python
# Extract difficulty from question embedding norm or learned parameter
question_difficulty = self.difficulty_embedding(questions_shifted)  # [B, T, 1]

# Adjust threshold dynamically per question
adjusted_threshold = theta_clamped + question_difficulty * self.difficulty_scale

# Predict with question-aware threshold
incremental_mastery_predictions = torch.sigmoid(
    (active_mastery - adjusted_threshold) / self.threshold_temperature
)
```

**Expected IM AUC**: 0.52-0.58 (modest improvement)

## Configuration Parameters

Add to `configs/parameter_default.json`:

```json
{
  "im_prediction_type": "linear",  // Options: "linear", "mlp", "attention", "difficulty_adjusted"
  "im_prediction_hidden_dim": 128,  // For MLP option
  "im_include_gains": false,        // Whether to include gain signal in features
  "difficulty_scale": 0.5           // For difficulty adjustment option
}
```

## Training Considerations

### Loss Function (Unchanged)

The incremental mastery loss already uses BCE, so no changes needed:

```python
# Encoder 1: BCE loss
bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

# Encoder 2: Incremental mastery loss (now with learned predictions)
im_loss = F.binary_cross_entropy(incremental_mastery_predictions, targets)

# Combined loss
total_loss = bce_loss_weight * bce_loss + im_loss_weight * im_loss
```

### Monitoring

Track separate AUC metrics (already implemented):
- `bce_auc`: Encoder 1 performance
- `im_auc`: Encoder 2 performance (should improve from 0.50 to 0.55-0.65)
- `global_auc`: Weighted combination

### Expected Results

| Metric | Before | After (Linear) | After (MLP) |
|--------|--------|----------------|-------------|
| bce_auc | 0.71 | 0.71 | 0.71 |
| im_auc | 0.50 | 0.57 | 0.63 |
| global_auc | 0.71 | 0.72 | 0.73 |

### Interpretability

**Maintained**: Mastery states still exist and evolve interpretably. The prediction head only affects how mastery is used for prediction, not how it's computed or updated.

## Implementation Roadmap

### Phase 1: Simple Linear Head (1-2 days)
1. Add linear prediction head to `gainakt3_exp.py`
2. Modify forward pass to use question features
3. Update tests
4. Train on assist2015 dataset
5. Evaluate IM AUC improvement

### Phase 2: Enhanced Features (2-3 days)
1. Add gain signal to features (if Phase 1 successful)
2. Experiment with MLP depth
3. Tune hyperparameters
4. Compare performance across datasets

### Phase 3: Advanced Options (Optional, 3-5 days)
1. Implement attention-based predictor
2. Add learned question difficulty embeddings
3. Ablation studies
4. Paper results comparison

## Testing Strategy

### Unit Tests
```python
def test_im_prediction_head():
    """Test that IM predictions use question features"""
    model = GainAKT3Exp(...)
    
    # Same mastery, different questions → different predictions
    mastery = torch.tensor([0.5, 0.5])
    q1 = torch.tensor([10])  # Easy question
    q2 = torch.tensor([20])  # Hard question
    
    pred1 = model.predict_im(mastery[0], q1)
    pred2 = model.predict_im(mastery[1], q2)
    
    assert pred1 != pred2, "Predictions should differ for different questions"
```

### Integration Tests
```python
def test_im_auc_improvement():
    """Verify IM AUC is better than random"""
    model = train_model(...)
    auc_metrics = evaluate(model, val_data)
    
    assert auc_metrics['im_auc'] > 0.52, "IM AUC should exceed random chance"
```

## Migration Path

### Backward Compatibility

Add feature flag to maintain old behavior:

```python
# In __init__:
self.use_learned_im_predictions = config.get('use_learned_im_predictions', False)

# In forward_with_states:
if self.use_learned_im_predictions:
    # NEW: Use prediction head
    im_predictions = self.im_prediction_head(im_features)
else:
    # OLD: Direct transformation (backward compatible)
    im_predictions = torch.sigmoid((skill_mastery - theta) / temperature)
```

### Experiment Naming

Use experiment suffix to distinguish:
- `gainakt3exp_baseline`: Original architecture (IM AUC ≈ 0.50)
- `gainakt3exp_im-linear`: Linear prediction head
- `gainakt3exp_im-mlp`: MLP prediction head
- `gainakt3exp_im-attention`: Attention-based prediction

## Documentation Updates

### Paper Impact

**Section to Update**: "4.2 Dual-Encoder Architecture"

**Current Text**:
> Encoder 2 predicts correctness by transforming mastery states: P(correct) = σ((M_t - θ) / τ)

**Revised Text**:
> Encoder 2 predicts correctness using a learned mapping from mastery states and question features: P(correct) = f_θ(M_t, q_t), where f_θ is a neural prediction head that combines interpretable mastery representations with question-specific context.

**New Subsection**: "4.2.3 Question-Aware Prediction"
> While mastery states evolve monotonically to reflect cumulative learning, per-question predictions must account for question difficulty and characteristics. We introduce a prediction head that maps (mastery, question_features) → prediction probability, enabling Encoder 2 to contribute meaningfully to prediction performance while maintaining interpretable internal states.

### Code Comments

Add detailed comments explaining the design choice:

```python
# ARCHITECTURAL NOTE (2025-11-20):
# IM predictions use a learned head rather than direct mastery transformation.
# Rationale: Mastery is monotonic (increases with practice) but correctness
# varies per question (depends on difficulty). The prediction head learns
# the relationship between mastery level and success probability for
# different question types, enabling better AUC while preserving interpretability.
```

## References

- IRT (Item Response Theory): Models difficulty-adjusted predictions
- DKVMN: Uses memory states + question features for predictions
- DTransformer: Question-aware attention for knowledge tracing

## Conclusion

**Recommended Action**: Implement **Option 1 (Linear Prediction Head)** as the starting point. This provides:
- ✅ Meaningful IM AUC improvement (0.50 → 0.57+)
- ✅ Maintained interpretability (mastery states unchanged)
- ✅ Minimal complexity (single linear layer)
- ✅ Fast training and debugging

If results are promising, proceed to Option 2 (MLP) or Option 3 (include gains) for further improvements.
