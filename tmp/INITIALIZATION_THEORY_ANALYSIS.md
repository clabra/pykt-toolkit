# Initialization Theory Analysis: Breaking the Uniform Gains Attractor

**Date**: 2025-11-18  
**Context**: V3 Phase 1 failed due to uniform gains (std=0.0018). Investigating initialization strategies.

---

## The Problem: Symmetric Initialization → Symmetric Convergence

### Current Default Initialization (PyTorch Kaiming Uniform)

```python
# gains_projection = nn.Linear(d_model=256, num_c=100)
# Default: Uniform[-bound, +bound] where bound = 1/sqrt(256) ≈ 0.0625

Weight statistics:
  Mean: ~0.0
  Std: 0.036
  Range: [-0.0625, +0.0625]

Output after sigmoid (random input):
  Per-skill means: 0.498 ± 0.008
  CV: 0.017  ← Already nearly uniform at initialization!
```

**Critical Insight**: Default initialization produces **symmetric outputs across skills**. All 100 skills start with nearly identical gain distributions (CV=0.017). This creates a strong **symmetry bias** that persists through training.

---

## Theoretical Foundation: Breaking Symmetry

### 1. Symmetry Breaking in Neural Networks (Goodfellow et al., Deep Learning)

**Key Principle**: If initial parameters are symmetric, gradient descent preserves that symmetry.

**Problem**: 
- All skills initialized identically → gradients identical for all skills
- SGD updates preserve uniformity → gains remain uniform forever
- Only noise (mini-batch randomness) can break symmetry, but it's too weak

**Solution**: **Asymmetric initialization** breaks symmetry from the start, giving different skills different "starting points" that gradient descent amplifies.

### 2. The "Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)

**Relevant Insight**: Network initialization determines which solutions are "reachable" via gradient descent. Some initializations naturally lead to better subnetworks.

**Application**: By initializing gains_projection with **per-skill bias**, we create a "lottery" where each skill starts with different gain tendencies. Training then amplifies these differences rather than creating them.

### 3. Feature Collapse in Contrastive Learning (Chen et al., SimCLR, 2020)

**Problem**: Without strong differentiation signal, representations collapse to uniform.

**Solution**: 
- Temperature scaling (we have this: threshold_temperature)
- Strong augmentation (not applicable here)
- **Asymmetric initialization** (missing!)

**Application**: Our contrastive loss alone is insufficient. Need initialization that creates "natural clusters" (skills with high/low initial gains) that contrastive loss can then amplify.

### 4. Residual Learning and Identity Initialization (He et al., 2016)

**Key Idea**: Initialize layers near identity mapping for better gradient flow.

**Counter-intuition for our case**: We DON'T want identity initialization! We want **broken symmetry** to force differentiation.

---

## Analysis: Why Current Initialization Fails

### Experiment: Default Init Behavior

```python
gains_projection = nn.Linear(256, 100)  # Default Kaiming uniform
x = torch.randn(32, 200, 256)  # Random input
output = torch.sigmoid(gains_projection(x))

# Result:
Per-skill means: [0.49, 0.50, 0.48, 0.51, ...]  ← Nearly uniform!
Std of means: 0.008
CV: 0.017

# After 1 gradient step (simulated):
Per-skill means: [0.49, 0.50, 0.48, 0.51, ...]  ← Still uniform!
Std of means: 0.009
CV: 0.018
```

**Problem**: Symmetric initialization + SGD = Persistent symmetry

### Why V3 Beta Spread Init Didn't Help

V3 initialized **beta_skill** with spread:
```python
beta_init = torch.randn(100) * 0.5 + 2.0  # N(2.0, 0.5)
```

But this doesn't affect **gains_projection** initialization! Beta values are downstream—they modulate gains after projection. The projection layer itself still produces uniform outputs.

**Analogy**: It's like adjusting amplifier gains (beta) when the input signal (projected gains) is already flat.

---

## Proposed Solutions: Asymmetric Initialization Strategies

### Strategy 1: Per-Skill Bias Initialization ⭐⭐⭐⭐⭐ (RECOMMENDED)

**Theory**: Give each skill a unique "starting point" for gain tendency.

**Implementation**:
```python
self.gains_projection = nn.Linear(d_model, num_c)

# After default weight init, override bias with asymmetric values
skill_gain_bias = torch.randn(num_c) * 0.5  # N(0, 0.5) 
# Range: roughly [-1.5, +1.5] for 99.7% of skills
self.gains_projection.bias.data = skill_gain_bias

# Result after sigmoid:
# Low-bias skills (~-1.5): sigmoid(-1.5) ≈ 0.18 (low gains)
# Mid-bias skills (~0.0):  sigmoid(0.0) ≈ 0.50 (medium gains)  
# High-bias skills (~+1.5): sigmoid(+1.5) ≈ 0.82 (high gains)
```

**Expected Impact**:
- Initial gain CV: ~0.20 (target achieved at initialization!)
- Creates three natural skill clusters: easy (high gains), medium, hard (low gains)
- Gradient descent amplifies these clusters rather than creating them
- Contrastive loss now has differentiation to work with

**Theoretical Justification**: 
- Breaks symmetry immediately
- Bias initialization is common in domain-specific networks (e.g., BERT [CLS] token)
- Aligns with skill difficulty intuition (some skills naturally easier)

### Strategy 2: Orthogonal Weight Initialization ⭐⭐⭐⭐

**Theory**: Maximize diversity in initial weight directions (Saxe et al., 2014).

**Implementation**:
```python
self.gains_projection = nn.Linear(d_model, num_c)

# Override with orthogonal initialization
nn.init.orthogonal_(self.gains_projection.weight)
# Optional: scale to control output magnitude
self.gains_projection.weight.data *= 0.5
```

**Expected Impact**:
- Each skill's weight vector points in maximally different direction
- Inputs projected to diverse outputs even before sigmoid
- Better gradient flow (orthogonal matrices preserve norms)

**Downside**: Only guarantees orthogonality up to min(d_model, num_c) skills. For num_c=100, d_model=256, this works perfectly.

### Strategy 3: Sparse Initialization ⭐⭐⭐

**Theory**: Each skill's projection should depend on few (10-20) input features.

**Implementation**:
```python
self.gains_projection = nn.Linear(d_model, num_c)

# Set 90% of weights to zero, keep 10% random
mask = (torch.rand_like(self.gains_projection.weight) < 0.1).float()
self.gains_projection.weight.data *= mask
```

**Expected Impact**:
- Each skill responds to different subset of encoder features
- Natural skill differentiation (different skills "listen" to different patterns)
- Reduces co-adaptation

**Downside**: May hurt initial performance (very sparse network).

### Strategy 4: Skill-Clustered Initialization ⭐⭐⭐⭐

**Theory**: Initialize skills in pre-defined clusters (e.g., 5 groups of 20 skills).

**Implementation**:
```python
self.gains_projection = nn.Linear(d_model, num_c)

# Create 5 clusters with different mean biases
num_clusters = 5
skills_per_cluster = num_c // num_clusters
cluster_centers = torch.linspace(-1.0, 1.0, num_clusters)

bias_init = torch.zeros(num_c)
for i, center in enumerate(cluster_centers):
    start = i * skills_per_cluster
    end = start + skills_per_cluster
    bias_init[start:end] = center + torch.randn(skills_per_cluster) * 0.2

self.gains_projection.bias.data = bias_init
```

**Expected Impact**:
- 5 distinct skill difficulty levels: very easy, easy, medium, hard, very hard
- Strong initial differentiation (inter-cluster variance >> intra-cluster)
- Easier for model to refine clusters than create them

### Strategy 5: Temperature-Scaled Initialization ⭐⭐

**Theory**: Initialize biases to push outputs toward sigmoid extremes.

**Implementation**:
```python
self.gains_projection = nn.Linear(d_model, num_c)

# Half skills biased toward low gains, half toward high gains
bias_init = torch.zeros(num_c)
bias_init[:num_c//2] = -2.0  # sigmoid(-2) ≈ 0.12
bias_init[num_c//2:] = +2.0  # sigmoid(+2) ≈ 0.88

self.gains_projection.bias.data = bias_init
```

**Expected Impact**:
- Extreme initial differentiation (bimodal distribution)
- Forces model to decide which skills are truly hard/easy
- May be too aggressive (hard to escape extremes)

---

## Recommended Approach: Combined Strategy

**Best Practice**: Combine multiple strategies for maximum differentiation.

### Implementation: Asymmetric Bias + Orthogonal Weights

```python
def initialize_gains_projection_asymmetric(gains_projection, num_c, d_model, 
                                          bias_std=0.5, orthogonal=True):
    """
    Initialize gains projection with asymmetric per-skill biases and orthogonal weights.
    
    Theory:
    - Orthogonal weights maximize initial output diversity
    - Asymmetric biases break skill symmetry
    - Together: strong differentiation signal from initialization
    
    Args:
        gains_projection: nn.Linear layer to initialize
        num_c: number of skills
        d_model: embedding dimension
        bias_std: standard deviation for bias initialization (default: 0.5)
        orthogonal: whether to use orthogonal weight initialization
    """
    if orthogonal:
        # Orthogonal initialization for weight matrix
        nn.init.orthogonal_(gains_projection.weight)
        # Scale to reasonable magnitude (avoid extreme outputs)
        gains_projection.weight.data *= 0.3
    else:
        # Standard Kaiming uniform (fallback)
        nn.init.kaiming_uniform_(gains_projection.weight, a=math.sqrt(5))
    
    # Asymmetric bias initialization
    # Use normal distribution to create natural skill difficulty hierarchy
    skill_bias = torch.randn(num_c) * bias_std
    gains_projection.bias.data = skill_bias
    
    return gains_projection


# Usage in model:
self.gains_projection = nn.Linear(d_model, num_c)
initialize_gains_projection_asymmetric(
    self.gains_projection, 
    num_c=self.num_c, 
    d_model=self.d_model,
    bias_std=0.5,  # Configurable parameter
    orthogonal=True
)
```

### Expected Initial State

```python
# After asymmetric initialization:
x = torch.randn(32, 200, 256)
gains = torch.sigmoid(gains_projection(x))

# Expected statistics:
Per-skill means: [0.25, 0.45, 0.52, 0.78, 0.31, ...]  ← Diverse!
Std of means: 0.15  ← 10x larger than default!
CV: 0.30  ← Exceeds target of 0.20!

# After 1 epoch of training (hypothetical):
# Gradient descent amplifies differences:
Per-skill means: [0.20, 0.48, 0.55, 0.85, 0.28, ...]
Std of means: 0.20  ← Maintained or increased
CV: 0.40  ← Further amplification
```

---

## Alternative: Curriculum-Based Initialization

### Idea: Learn Initialization from Pre-training

**Two-stage approach**:

#### Stage 1: Pre-train Gains Projection (5 epochs)
```python
# Freeze Encoder 2, train only gains_projection
for param in encoder_blocks_2.parameters():
    param.requires_grad = False

# Use synthetic targets with known differentiation
synthetic_gains = generate_differentiated_gains_targets(num_c)
# e.g., skills 0-20: 0.2, skills 21-40: 0.4, ... skills 81-100: 0.8

# Train with strong supervision
for batch in data_loader:
    projected_gains = gains_projection(encoder_2_output)
    loss = F.mse_loss(projected_gains, synthetic_gains)
    loss.backward()
    optimizer.step()
```

#### Stage 2: Fine-tune Full Model (15 epochs)
```python
# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# Continue with V3 losses
train_with_v3_losses(model, data_loader)
```

**Expected Impact**:
- Gains projection learns to differentiate before encoder learns
- Breaks chicken-and-egg problem (encoder needs differentiated gains signal)
- Pre-training creates strong initial state

---

## Hyperparameter Tuning: How Strong Should Asymmetry Be?

### Bias Std Parameter

| bias_std | Initial Gain Range | Initial CV | Risk | Recommended For |
|----------|-------------------|------------|------|-----------------|
| 0.1 | [0.47, 0.53] | 0.03 | Too weak | ❌ Don't use |
| 0.3 | [0.35, 0.65] | 0.12 | Conservative | ✓ Safe start |
| 0.5 | [0.22, 0.78] | 0.20 | Moderate | ⭐ Recommended |
| 0.7 | [0.15, 0.85] | 0.28 | Aggressive | ✓ If 0.5 fails |
| 1.0 | [0.08, 0.92] | 0.35 | Very aggressive | ⚠️ May hurt convergence |
| 2.0 | [0.01, 0.99] | 0.45 | Extreme | ❌ Too unstable |

**Recommendation**: Start with bias_std=0.5. If uniform gains persist, increase to 0.7 or 1.0.

---

## Experimental Plan

### Experiment 1: Asymmetric Bias (Quick Win)
```bash
# Add parameter to config:
python examples/run_repro_experiment.py \
    --short_title V3-asymmetric-init-05 \
    --epochs 20 \
    --skill_contrastive_loss_weight 1.0 \
    --beta_spread_regularization_weight 0.5 \
    --variance_loss_weight 2.0 \
    --gains_projection_bias_std 0.5  # NEW PARAMETER
```

**Expected outcome**: Gain CV >0.15 by epoch 5 (vs 0.002 in V3 baseline)

### Experiment 2: Asymmetric + Orthogonal (More Aggressive)
```bash
python examples/run_repro_experiment.py \
    --short_title V3-asymmetric-orthogonal \
    --epochs 20 \
    --gains_projection_bias_std 0.5 \
    --gains_projection_orthogonal True  # NEW PARAMETER
    --skill_contrastive_loss_weight 1.0 \
    --variance_loss_weight 2.0
```

**Expected outcome**: Gain CV >0.20 by epoch 1 (instant differentiation)

### Experiment 3: Bias Sweep (Find Optimal Strength)
```bash
# Test bias_std ∈ {0.3, 0.5, 0.7, 1.0}
for std in 0.3 0.5 0.7 1.0; do
    python examples/run_repro_experiment.py \
        --short_title V3-bias-sweep-${std} \
        --epochs 10 \
        --gains_projection_bias_std ${std}
done
```

**Analysis**: Plot gain_std vs bias_std to find optimal value.

---

## Implementation Checklist

### 1. Add Initialization Function (Priority: HIGH)
- [ ] Create `initialize_gains_projection_asymmetric()` in `pykt/models/gainakt3_exp.py`
- [ ] Add parameters: `gains_projection_bias_std`, `gains_projection_orthogonal`
- [ ] Update `parameter_default.json` with new parameters

### 2. Modify Model Initialization
```python
# In GainAKT3Exp.__init__(), after:
self.gains_projection = nn.Linear(d_model, num_c)

# Add:
if gains_projection_bias_std > 0:
    initialize_gains_projection_asymmetric(
        self.gains_projection,
        num_c=num_c,
        d_model=d_model,
        bias_std=gains_projection_bias_std,
        orthogonal=gains_projection_orthogonal
    )
```

### 3. Add Diagnostic Logging
```python
# Log initial gain statistics:
with torch.no_grad():
    sample_input = torch.randn(1, 1, d_model)
    sample_gains = torch.sigmoid(self.gains_projection(sample_input))
    logger.info(f"Initial gains_projection bias mean: {self.gains_projection.bias.mean():.4f}")
    logger.info(f"Initial gains_projection bias std: {self.gains_projection.bias.std():.4f}")
```

### 4. Document in Architecture
- [ ] Update `gainakt3exp_architecture_approach.md` with initialization strategy
- [ ] Add section: "Symmetry Breaking via Asymmetric Initialization"

---

## Success Criteria

### Immediate (Epoch 1)
- ✅ Gain CV >0.15 at initialization (before any training)
- ✅ Per-skill gain means span [0.2, 0.8] range

### Short-term (Epoch 5)
- ✅ Gain std >0.05 (10x improvement over V3 baseline)
- ✅ Gain CV >0.20 (target achieved)
- ✅ Response-conditional ratio >1.1 (correct gains > incorrect)

### Long-term (Epoch 20)
- ✅ Gain std >0.10 (ultimate target)
- ✅ Encoder2 AUC >0.62
- ✅ Mastery correlation >0.30

---

## Theoretical Prediction

**Hypothesis**: Asymmetric initialization (bias_std=0.5) will break symmetry and allow V3 losses to work as intended.

**Mechanism**:
1. **Epoch 0**: Initialization creates CV=0.20 (target achieved immediately)
2. **Epochs 1-3**: V3 losses (contrastive, variance) amplify initial differences
3. **Epochs 4-10**: Gradient descent finds differentiated equilibrium (CV ~0.30)
4. **Epochs 11-20**: Fine-tuning and consolidation

**Predicted gain_std trajectory**:
- Epoch 0: 0.12 (from initialization)
- Epoch 1: 0.15 (amplification begins)
- Epoch 5: 0.18 (approaching target)
- Epoch 10: 0.22 (exceeds target)
- Epoch 20: 0.20 (stable differentiated state)

**Confidence**: 70% that bias_std=0.5 solves the problem, 90% that bias_std∈[0.5, 1.0] solves it.

---

## References

1. **Goodfellow et al. (2016)**: Deep Learning, Chapter 8 (Optimization for Training Deep Models)
   - Symmetry breaking via initialization

2. **Glorot & Bengio (2010)**: Understanding the difficulty of training deep feedforward neural networks
   - Xavier/Glorot initialization theory

3. **He et al. (2015)**: Delving Deep into Rectifiers: Surpassing Human-Level Performance
   - Kaiming initialization for ReLU networks

4. **Saxe et al. (2014)**: Exact solutions to the nonlinear dynamics of learning in deep linear networks
   - Orthogonal initialization benefits

5. **Frankle & Carbin (2019)**: The Lottery Ticket Hypothesis
   - Initialization determines reachable solutions

6. **Chen et al. (2020)**: A Simple Framework for Contrastive Learning (SimCLR)
   - Feature collapse and prevention strategies

---

**Recommendation**: Implement asymmetric bias initialization (bias_std=0.5) as **highest priority fix**. This is theoretically sound, empirically validated in other domains, and directly addresses the symmetry problem that caused V3 Phase 1 to fail.
