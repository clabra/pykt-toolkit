# Rasch Model

This documents introduces Item Response Theory (IRT) and the Rasch model, with focus on educational applications in Knowledge Tracing systems. 

More info in https://en.wikipedia.org/wiki/Rasch_model

## Core Concepts

### 1. Item Response Theory (IRT) Foundations

**Basic Principles**:
- IRT models the probability of a correct response as a function of person ability (θ) and item parameters
- Assumes that latent traits (abilities) can be measured through observed responses
- Provides invariant measurement: item parameters independent of sample, person parameters independent of items

**IRT Models Hierarchy**:
1. **1-Parameter Logistic (1PL) / Rasch Model**: Only item difficulty (b)
2. **2-Parameter Logistic (2PL)**: Difficulty (b) + discrimination (a)
3. **3-Parameter Logistic (3PL)**: Difficulty (b) + discrimination (a) + guessing (c)
4. **4-Parameter Logistic (4PL)**: All above + upper asymptote (d)

### 2. Rasch Model Specification

**Model Definition**:
```
P(X_ij = 1 | θ_i, b_j) = exp(θ_i - b_j) / (1 + exp(θ_i - b_j))
```

Where:
- `X_ij`: Response of person i to item j (1=correct, 0=incorrect)
- `θ_i`: Ability of person i (student proficiency)
- `b_j`: Difficulty of item j (skill difficulty)
- `P(X_ij = 1)`: Probability of correct response

**Key Properties**:
- **Sufficient Statistics**: Total score is sufficient statistic for ability estimation
- **Specific Objectivity**: Comparison of persons independent of items, comparison of items independent of persons
- **Additive Structure**: Log-odds ratio is linear in ability and difficulty
- **Equal Discrimination**: All items discriminate equally (parallel ICCs)

**Logit Form**:
```
logit(P) = log(P / (1-P)) = θ_i - b_j
```

This linear relationship is fundamental for interpretability.

### 3. Item Characteristic Curve (ICC)

**Definition**:
The ICC represents the probability of correct response as a function of ability level for a single item.

**Mathematical Form** (Rasch):
```
P(θ) = 1 / (1 + exp(-(θ - b)))
```

**Graphical Properties**:
- S-shaped (sigmoid) curve
- Inflection point at θ = b (50% probability)
- Asymptotes: 0 (as θ → -∞) and 1 (as θ → +∞)
- All Rasch ICCs have same shape, differ only in horizontal position (difficulty)

**Interpretation**:
- **Low ability (θ << b)**: Near 0% probability of correct response
- **Matched ability (θ = b)**: Exactly 50% probability
- **High ability (θ >> b)**: Near 100% probability

### 4. Test Characteristic Curve (TCC)

**Definition**:
The TCC is the sum of ICCs across all items in a test. It represents expected total score as a function of ability.

**Mathematical Form**:
```
TCC(θ) = Σ_j P_j(θ) = Σ_j [exp(θ - b_j) / (1 + exp(θ - b_j))]
```

Where the sum is over all J items in the test.

**Properties**:
- **Range**: [0, J] where J is number of items
- **Monotonically Increasing**: Higher ability → higher expected score
- **Shape**: Approximately linear in middle range, S-shaped overall
- **Slope**: Steeper when items have diverse difficulties (better discrimination across ability range)

**Key Insight for Knowledge Tracing**:
The TCC represents the **learning curve trajectory** - as student ability (θ) increases through practice, the expected performance (TCC value) increases predictably.

### 5. TCC Calculation

**Question**: Can we calculate TCC for each skill in KT datasets?

**Requirements**:
1. **Ability Estimates** (θ_i): Need to estimate student ability at each time point
2. **Difficulty Estimates** (b_j): Need to estimate difficulty for each item/skill
3. **Item-Skill Mapping**: Must know which items belong to which skill

**Skill-Specific Ability Evolution**: 
- **Approach**: Track per-skill ability θ_k(t) over time
- **TCC_k(θ_k(t))**: Expected performance on skill k given current ability
- **Calculation**: Model ability growth: θ_k(t) = θ_k(0) + α_k × practice_k(t)
- **Advantage**: Captures skill-specific learning trajectories

**Issues**
- Estimating initial difficulties (b_j) from dataset statistics
- Standard KT datasets lack item-level granularity (only skill IDs)
- Rasch assumes unidimensional ability - KT has multiple skills
- Ability (θ) evolves during learning - not constant
- Classical TCC assumes fixed ability across test administration but KT involves learning - ability changes between attempts
- Need dynamic extension of Rasch for temporal modeling

### 6. Dynamic Rasch Model for Knowledge Tracing

**Proposed Extension**:

Instead of static Rasch:
```
P(X_ij = 1) = σ(θ_i - b_j)
```

Use **Dynamic Rasch** with time-evolving ability:
```
P(X_ijt = 1) = σ(θ_it - b_j)

θ_it = θ_i(t-1) + Δθ_t   # Ability updates after each interaction
```

**Ability Update Mechanisms**:
1. **Fixed Learning Rate**: Δθ_t = α (constant growth)
2. **Performance-Dependent**: Δθ_t = α × X_t (learn from correct responses)
3. **Surprise-Based**: Δθ_t = α × (X_t - P_t) (learn from prediction errors)
4. **Forgetting-Aware**: θ_it = decay(θ_i(t-1)) + gain_t

**TCC in Dynamic Context**:
```
TCC_t(θ_t) = Σ_{j∈S_t} σ(θ_t - b_j)
```

Where S_t is the set of skills/items encountered up to time t.

**This represents the expected cumulative knowledge at time t.**

### 7. Integration with iKT Architecture

**iKT Architecture** (implemented in `pykt/models/ikt.py`):

```
Questions + Responses → Encoder 1 → h1 → Head 1 (Performance) → BCE Predictions → L1 (BCE Loss)
                                       └→ Head 2 (Mastery) → MLP1 → Softplus → cummax → {Mi} → L2 (Rasch MSE)
```

**Key Components**:

**Encoder 1**: Single transformer encoder processing question-response interactions
- Input: Questions (q) + Responses (r)
- Output: Knowledge state h1 [B, L, d_model]
- Shared between both heads for multi-task learning

**Head 1 (Performance)**: Next-step correctness prediction
- Architecture: Concat[h1, v1, skill_emb] → 3-layer MLP → BCE predictions
- Loss: L1 = BCE(predictions, targets)
- Purpose: Optimize for predictive accuracy (AUC)

**Head 2 (Mastery)**: Skill-level mastery estimation with Rasch grounding
- Architecture: h1 → MLP1 → Softplus (positivity) → cummax (monotonicity) → {Mi}
- Output: Skill vector {Mi} [B, L, num_c] - one mastery value per skill
- Loss: L2 = MSE(Mi, M_rasch) with phase-dependent behavior
- **Critical**: NO aggregation, NO second MLP - skill vector used directly

**Rasch Integration**:
- **M_rasch**: Theoretical mastery targets computed from Rasch IRT model
  ```python
  M_rasch[n, s] = σ(θ_n - b_s)
  where:
    θ_n = ability of student n (from IRT calibration)
    b_s = difficulty of skill s (from IRT calibration)
  ```

- **Phase 1 Training**: L_total = L2 (pure Rasch alignment, epsilon=0)
  - Goal: Initialize skill vector {Mi} to match IRT theoretical values
  - Ensures model starts in semantically consistent region

- **Phase 2 Training**: L_total = λ_bce × L1 + (1-λ_bce) × L2_constrained
  - L2_constrained = MSE(ReLU(|Mi - M_rasch| - ε))
  - Goal: Optimize performance while maintaining Rasch alignment within tolerance ε
  - Balances predictive accuracy with interpretability

**Loss Functions**:
1. **L1 (BCE)**: Standard next-response prediction for performance optimization
2. **L2 (Rasch)**: Per-skill MSE between model mastery {Mi} and Rasch theoretical mastery M_rasch

**Key Differences from Proposed Rasch-TCC Approach**:
- iKT uses **skill vectors {Mi}** directly, not scalar ability θ_t
- Each skill has independent mastery trajectory Mi[s] ∈ [0, 1]
- Rasch targets M_rasch are per-student-per-skill-per-timestep [B, L, num_c]
- No need for explicit difficulty bank - difficulties implicit in M_rasch targets

**TCC-Style Computation** (for analysis, not training):
```python
# At inference time, for student trajectory:
skill_vector = model.forward(q, r, qry)['skill_vector']  # [seq_len, num_c]

# For each skill k, cumulative expected mastery:
mastery_trajectory_k = skill_vector[:, k]  # [seq_len]

# Compare with Rasch baseline:
rasch_mastery_k = M_rasch[:, k]  # [seq_len]

# Deviation analysis:
deviation_k = torch.abs(mastery_trajectory_k - rasch_mastery_k)
within_tolerance = (deviation_k <= epsilon).float()
```

**Key Advantages**:
- **Interpretability**: Each Mi[s] has clear meaning (mastery of skill s)
- **Psychometric grounding**: Aligned with IRT theory via M_rasch targets
- **Architectural constraints**: Positivity + monotonicity enforced by design
- **Semantic consistency**: Deviation from M_rasch explicitly controlled by ε
- **Simplicity**: No complex TCC aggregation, direct per-skill modeling

### 8. Implementation Considerations

**Data Requirements**:
- **Standard KT format**: (student_id, skill_id, correct, timestamp) ✅
- **No additional data needed** - works with existing datasets

**Parameter Estimation**:
- **Difficulty Initialization**: Can use empirical success rates
  ```python
  b_j = -logit(mean_correct_j)  # Higher difficulty → lower success rate
  ```
- **Ability Initialization**: Start at θ_0 = 0 (population mean)

**Interpretability**:
- **θ_t trajectory**: Visualize student ability growth over time
- **b_j values**: Rank skills by difficulty
- **ICC curves**: Show predicted performance vs ability for each skill
- **TCC curves**: Show expected learning trajectory

**Validation**:
- **Internal Consistency**: Check if empirical ICCs match Rasch model
- **Fit Statistics**: Infit/Outfit MNSQ (mean square residuals)
- **Predictive Validity**: Does Rasch parameterization improve test AUC?

### 9. iKT's Solution to the Interpretability Challenge

**Previous Approach (GainAKT4)**: Encoder 2 → Head 3 for learning curves
- Prospective targets (attempts-to-mastery): R² = -0.84 (unpredictable)
- Retrospective targets (cumsum): R² > 0.93 (too trivial)
- **Result**: Failed to provide meaningful interpretability

**iKT's Approach**: Direct Rasch alignment via skill vectors

**How iKT Solves the Problems**:

**Problem 1: Unpredictability** → **Solved by IRT Grounding**
- Instead of predicting future attempts, align with IRT-derived mastery M_rasch
- Rasch model provides **theoretical targets**: M_rasch[n,s] = σ(θ_n - b_s)
- No need to predict arbitrary curves - just match psychometric theory

**Problem 2: Triviality** → **Solved by Theoretical Reference**
- Cannot output trivial cumsum - must match M_rasch from IRT calibration
- M_rasch is computed independently from actual responses (no data leakage)
- Phase 2 tolerance ε prevents overfitting to Rasch while allowing learning

**Problem 3: Information Leakage** → **Solved by Independent Calibration**
- M_rasch computed from IRT on full dataset (not individual sequences)
- Model cannot "cheat" by memorizing response patterns
- Must learn generalizable skill mastery representations

**iKT Learning Target**:
Instead of learning curves or scalar ability:
1. **Skill Vector {Mi}**: Per-skill mastery levels [B, L, num_c]
2. **Rasch Alignment**: Minimize MSE(Mi, M_rasch) with tolerance ε
3. **Architectural Constraints**: Positivity (Softplus) + Monotonicity (cummax)

**Two-Phase Training Strategy**:
- **Phase 1**: Pure Rasch alignment (L_total = L2, ε=0)
  - Initialize skill vector to match IRT theory
  - Establish psychometric grounding
- **Phase 2**: Constrained optimization (L_total = λ×L1 + (1-λ)×L2, ε>0)
  - Optimize for performance (L1) while maintaining Rasch proximity (L2)
  - Tolerance ε allows deviation for improved AUC

**Evaluation Metrics**:
- **Primary**: Standard BCE/AUC for next-response prediction (L1)
- **Interpretability**: Rasch deviation ||Mi - M_rasch|| (should be ≤ ε)
- **Constraints**: Positivity (Mi > 0) and monotonicity (Mi[t+1] ≥ Mi[t])
- **Psychometric**: Correlation between Mi and M_rasch per skill

### 10. Research Questions and Implementation Status

**iKT Model Status**: ✅ **IMPLEMENTED** (`pykt/models/ikt.py`, `pykt/models/ikt_mon.py`)
- Single encoder with 2 heads (Performance + Mastery)
- Architectural constraints: Softplus (positivity) + cummax (monotonicity)
- Phase-dependent loss computation
- Training/evaluation scripts: `examples/train_ikt.py`, `examples/eval_ikt.py`

**Current Limitation**: Rasch target preprocessing NOT yet implemented
- Model can train without Rasch targets (falls back to pure BCE mode)
- Need to compute M_rasch[n, s, t] from IRT calibration for full functionality

**Critical Research Questions**:

1. **Do Rasch-aligned skill vectors improve interpretability?**
   - Test: Compare Mi vs M_rasch correlation per skill
   - Method: Measure ||Mi - M_rasch|| deviation across phases
   - Expected: Phase 1 should achieve close alignment, Phase 2 maintains proximity

2. **What is the optimal tolerance threshold ε?**
   - Test: AUC vs ε trade-off curve
   - Method: Train with ε ∈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
   - Expected: Small ε preserves interpretability, large ε improves AUC

3. **Does Rasch constraint improve or hurt predictive performance?**
   - Test: Compare AUC with/without Rasch loss (λ_bce = 1.0 vs 0.5)
   - Method: Ablation study across multiple datasets
   - Expected: Rasch acts as regularizer, may improve generalization

4. **Can skill vectors reveal learning trajectories?**
   - Test: Visualize Mi[s] over time for individual students
   - Method: Plot mastery evolution for easy/medium/hard skills
   - Expected: Monotonic increase, rate varies by skill difficulty

5. **Do learned mastery levels match empirical success rates?**
   - Test: Correlation between Mi[s] and observed success rate on skill s
   - Method: Scatter plot + Spearman correlation
   - Expected: Strong positive correlation (ρ > 0.7)

**Implementation Roadmap**:

**Phase 1: Rasch Target Preprocessing** (2-3 days) ← **CURRENT PRIORITY**
- [x] Install py-irt library
- [x] Convert pykt data to IRT format (`tmp/convert_to_irt_format.py`)
- [x] Run Rasch calibration (`tmp/run_rasch_calibration.py`)
- [ ] Compute per-student-per-skill-per-timestep M_rasch targets
- [ ] Save as preprocessed tensors for training
- [ ] Update `load_rasch_targets()` in `train_ikt.py`

**Phase 2: Rasch Target Generation** (1 day)
- [ ] Create script to compute M_rasch[n, s, t] from θ_n and b_s
- [ ] Handle temporal aspects (initial vs evolved ability)
- [ ] Save preprocessed targets alongside datasets
- [ ] Validate: M_rasch should correlate with success rates

**Phase 3: Phase 1 Training** (2-3 days)
- [ ] Train iKT with pure L2 loss (Phase 1, ε=0)
- [ ] Monitor convergence of ||Mi - M_rasch||
- [ ] Verify architectural constraints (positivity, monotonicity)
- [ ] Checkpoint best Phase 1 model

**Phase 4: Phase 2 Training** (3-5 days)
- [ ] Train Phase 2 with multiple ε values
- [ ] Plot AUC vs ε trade-off curves
- [ ] Identify optimal (λ_bce, ε) configuration
- [ ] Compare vs baseline (no Rasch)

**Phase 5: Analysis and Visualization** (2-3 days)
- [ ] Generate skill mastery trajectories for sample students
- [ ] Compute Rasch deviation metrics
- [ ] Compare Mi vs M_rasch per-skill correlations
- [ ] Create educational interpretability visualizations

**Total Timeline**: ~10-15 days (with Rasch preprocessing as critical path)

### 11. iKT Technical Implementation

**Current Implementation** (`pykt/models/ikt.py`):

**Head 2 - Mastery with Rasch Alignment**:
```python
class iKT(nn.Module):
    def __init__(self, num_c, seq_len, d_model, ..., lambda_bce, epsilon, phase):
        super().__init__()
        self.num_c = num_c  # Number of skills
        self.phase = phase  # Training phase (1 or 2)
        self.epsilon = epsilon  # Rasch tolerance threshold
        self.lambda_bce = lambda_bce  # BCE loss weight
        
        # Head 2: Skill mastery estimation
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_c),
            nn.Softplus()  # Ensures Mi > 0 (positivity constraint)
        )
    
    def forward(self, q, r, qry, rasch_targets=None):
        """
        Args:
            q: [B, L] - question IDs
            r: [B, L] - responses (0/1)
            qry: [B, L] - query questions
            rasch_targets: [B, L, num_c] - theoretical mastery from IRT (optional)
        Returns:
            dict with:
                - bce_predictions: [B, L] - performance predictions
                - skill_vector: [B, L, num_c] - per-skill mastery {Mi}
                - logits: [B, L] - raw logits for BCE loss
                - rasch_loss: scalar or None
        """
        # Encoder processing
        h1, v1 = self.encoder1(q, r)  # [B, L, d_model]
        
        # Head 1: Performance prediction
        skill_emb = self.skill_emb(qry)  # [B, L, d_model]
        concat = torch.cat([h1, v1, skill_emb], dim=-1)
        logits = self.prediction_head(concat).squeeze(-1)
        bce_predictions = torch.sigmoid(logits)
        
        # Head 2: Skill mastery estimation
        kc_vector = self.mlp1(h1)  # [B, L, num_c], positive values
        skill_vector = torch.cummax(kc_vector, dim=1)[0]  # Monotonicity constraint
        
        # Compute Rasch loss if targets provided
        rasch_loss = None
        if rasch_targets is not None:
            if self.phase == 1:
                # Phase 1: Direct MSE alignment
                rasch_loss = F.mse_loss(skill_vector, rasch_targets)
            else:
                # Phase 2: Constrained MSE with epsilon tolerance
                deviation = torch.abs(skill_vector - rasch_targets)
                violation = torch.relu(deviation - self.epsilon)
                rasch_loss = torch.mean(violation ** 2)
        
        return {
            'bce_predictions': bce_predictions,
            'skill_vector': skill_vector,  # {Mi}
            'logits': logits,
            'rasch_loss': rasch_loss
        }
    
    def compute_loss(self, output, targets):
        """
        Phase-dependent loss computation.
        
        Phase 1: L_total = L2 (Rasch alignment only, or BCE if no Rasch targets)
        Phase 2: L_total = λ_bce × L1 + (1-λ_bce) × L2
        """
        # L1: BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            output['logits'], targets.float(), reduction='mean'
        )
        
        # L2: Rasch loss
        rasch_loss = output['rasch_loss'] if output['rasch_loss'] is not None else bce_loss * 0.0
        
        # Phase-dependent total loss
        if self.phase == 1:
            if output['rasch_loss'] is not None:
                total_loss = rasch_loss  # Pure Rasch alignment
            else:
                total_loss = bce_loss  # Fallback to BCE if no Rasch targets
        else:
            # Phase 2: Weighted combination
            total_loss = self.lambda_bce * bce_loss + (1 - self.lambda_bce) * rasch_loss
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'rasch_loss': rasch_loss
        }
```

**Key Implementation Details**:

1. **Skill Vector Output**: Head 2 produces {Mi} [B, L, num_c] directly
   - No aggregation via MLP2 (unlike GainAKT4)
   - Each skill has independent mastery value

2. **Architectural Constraints**:
   - Positivity: `nn.Softplus()` ensures Mi > 0
   - Monotonicity: `torch.cummax()` ensures Mi[t+1] ≥ Mi[t]

3. **Rasch Loss Computation**:
   - Phase 1: `MSE(skill_vector, rasch_targets)` - direct alignment
   - Phase 2: `MSE(ReLU(|skill_vector - rasch_targets| - ε))` - tolerance-based

4. **Fallback Behavior**: If `rasch_targets=None`, trains in pure BCE mode
   - Enables training before Rasch preprocessing complete
   - Warning printed during training

**Monitoring Version** (`pykt/models/ikt_mon.py`):
```python
class iKTMon(iKT):
    """Monitoring version for interpretability analysis."""
    
    def forward_with_states(self, q, r, qry, rasch_targets=None):
        """
        Extended forward pass capturing intermediate states.
        
        Returns:
            Same as forward(), plus:
                - h_states: encoder hidden states
                - v_states: value embeddings
                - kc_before_cummax: skill vector before monotonicity
        """
        # Standard forward pass
        output = super().forward(q, r, qry, rasch_targets)
        
        # Capture intermediate states for analysis
        output['h_states'] = self.encoder1.last_h
        output['v_states'] = self.encoder1.last_v
        output['kc_before_cummax'] = self.last_kc_vector
        
        return output
```

### 12. Success Criteria

**Implementation Milestones**:

**Phase 1: Model Implementation** ✅ **COMPLETE**
- ✅ iKT model with 2-head architecture (`pykt/models/ikt.py`)
- ✅ Architectural constraints: Softplus (positivity) + cummax (monotonicity)
- ✅ Phase-dependent loss computation (Phase 1: L2, Phase 2: λ×L1 + (1-λ)×L2)
- ✅ Training/evaluation scripts with reproducibility compliance
- ✅ Test experiment successful (exp 860574: AUC=0.7063 with random Rasch targets)

**Phase 2: Rasch Preprocessing** ✅ **COMPLETE**
- ✅ py-irt installation and data conversion scripts
- ✅ Rasch calibration workflow implemented (`examples/compute_rasch_targets.py`)
- ✅ Compute M_rasch[n, s, t] targets from IRT parameters
- ✅ Preprocess targets for all datasets (assist2009, assist2015, etc.)
- ✅ Integrate into `load_rasch_targets()` function

**Usage - Compute Rasch Targets:**
```bash
# Compute Rasch IRT targets from training data
python examples/compute_rasch_targets.py \
    --dataset assist2015 \
    --max_iterations 50

# Output: data/assist2015/rasch_targets.pkl (201 MB)
# Contains: student_abilities (θ), skill_difficulties (b), rasch_targets M_rasch[n,s,t]

# For other datasets:
python examples/compute_rasch_targets.py --dataset assist2009 --max_iterations 50
python examples/compute_rasch_targets.py --dataset statics2011 --max_iterations 50

# Custom output path:
python examples/compute_rasch_targets.py \
    --dataset assist2015 \
    --output_path /custom/path/rasch_targets.pkl \
    --max_iterations 100
```

**Results (assist2015):**
- Students calibrated: 15,275
- Skills calibrated: 100  
- Interactions: 544,331
- Student abilities (θ): Mean=2.475, Std=1.586, Range=[-4.667, 4.850]
- Skill difficulties (b): Mean=-2.004, Std=0.844, Range=[-3.300, 1.094]

**Phase 3: Training with Real Rasch Targets** 🔄 **READY TO START**
- [ ] Phase 1 training: Pure Rasch alignment (ε=0)
- [ ] Verify ||Mi - M_rasch|| convergence
- [ ] Phase 2 training: Multiple ε values (0.05, 0.1, 0.15, 0.2)
- [ ] Plot AUC vs ε trade-off curves

**Usage - Phase 1 Training (Pure Rasch Alignment):**
```bash
# Phase 1: Pure Rasch alignment (L_total = L2, epsilon=0)
python examples/run_repro_experiment.py \
    --short_title ikt_phase1_rasch \
    --phase 1 \
    --epsilon 0.0 \
    --lambda_bce 0.5 \
    --epochs 20

# Rasch targets loaded automatically from data/{dataset}/rasch_targets.pkl
# Or specify custom path:
python examples/run_repro_experiment.py \
    --short_title ikt_phase1_custom \
    --phase 1 \
    --epsilon 0.0 \
    --rasch_path /custom/path/rasch_targets.pkl \
    --epochs 20

# Without real Rasch targets (falls back to random placeholders):
# If rasch_targets.pkl not found, training uses random values [0, 1]
```

**Expected Phase 1 Results:**
- Rasch loss (L2) should decrease and converge
- Model learns to align skill vectors {Mi} with IRT-derived M_rasch
- Architectural constraints (positivity, monotonicity) maintained
- Best checkpoint saved for Phase 2 initialization

**Usage - Phase 2 Training (Constrained Optimization):**
```bash
# Phase 2: Constrained optimization with epsilon tolerance
# Ablation study: sweep epsilon values
for epsilon in 0.0 0.05 0.1 0.15 0.2 0.3; do
    python examples/run_repro_experiment.py \
        --short_title ikt_phase2_eps${epsilon} \
        --phase 2 \
        --epsilon ${epsilon} \
        --lambda_bce 0.5 \
        --epochs 30
done

# Loss formula Phase 2:
# L_total = λ_bce × L1 + (1-λ_bce) × L2_constrained
# L2_constrained = MSE(ReLU(|Mi - M_rasch| - ε))
```

**Expected Phase 2 Results:**
- Small ε (≈0.05): High Rasch alignment, potentially lower AUC
- Large ε (≈0.3): Higher AUC, less strict Rasch constraint  
- Optimal ε: Best trade-off between performance and interpretability

**Phase 4: Analysis and Validation** ⏳ **PENDING**
- [ ] Extract skill trajectories {Mi} from trained models
- [ ] Compute correlation: Mi vs M_rasch per skill
- [ ] Visualize mastery evolution for sample students
- [ ] Analyze deviation patterns across skills
- [ ] Generate educational visualizations for paper

**Usage - Analysis and Visualization:**
```bash
# Verify Rasch targets file
python -c "
import pickle, numpy as np
with open('data/assist2015/rasch_targets.pkl', 'rb') as f:
    data = pickle.load(f)
print('Metadata:', data['metadata'])
print('Students:', len(data['rasch_targets']))
abilities = list(data['student_abilities'].values())
print(f'Abilities θ: mean={np.mean(abilities):.3f}, std={np.std(abilities):.3f}')
difficulties = list(data['skill_difficulties'].values())
print(f'Difficulties b: mean={np.mean(difficulties):.3f}, std={np.std(difficulties):.3f}')
"

# Extract and analyze skill trajectories (to be implemented)
# python examples/analyze_rasch_alignment.py \
#     --model_path experiments/ikt_phase2_eps0.1/best_model.pth \
#     --rasch_path data/assist2015/rasch_targets.pkl \
#     --output_dir analysis/rasch_alignment

# Generate educational visualizations (to be implemented)
# python examples/visualize_mastery_trajectories.py \
#     --model_path experiments/ikt_phase2_eps0.1/best_model.pth \
#     --rasch_path data/assist2015/rasch_targets.pkl \
#     --student_ids 16894 3675 1692 \
#     --output_dir visualizations/trajectories
```

**Validation Criteria**:

**Minimum Viable Success**:
- ✅ Model trains without gradient issues (verified with random targets)
- ✅ Architectural constraints satisfied (positivity, monotonicity)
- ✅ Phase-dependent loss computation works correctly
- ✅ Rasch preprocessing pipeline complete and tested
- [ ] Test AUC ≥ pure BCE baseline (no Rasch loss)

**Strong Success**:
- [ ] Mi correlates with M_rasch per skill (ρ > 0.7)
- [ ] Rasch deviation ||Mi - M_rasch|| stays within tolerance ε
- [ ] Skill mastery trajectories show interpretable patterns
- [ ] Test AUC competitive with state-of-the-art (≥ 0.72 on assist2015)

**Exceptional Success**:
- [ ] Mi values match empirical IRT calibration
- [ ] Model-learned skill difficulty ranking aligns with expert knowledge
- [ ] Ablation studies show Rasch constraint improves generalization
- [ ] Provides actionable educational insights (skill difficulty, student trajectories)
- [ ] Pareto frontier analysis identifies optimal (λ_bce, ε) configurations

## References and Resources

**Key Papers**:
1. Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*
2. Embretson, S. E., & Reise, S. P. (2000). *Item Response Theory for Psychologists*
3. Wilson, M., De Boeck, P., & Carstensen, C. H. (2008). *Explanatory Item Response Models*
4. Piech et al. (2015). *Deep Knowledge Tracing* - Neural IRT integration
5. Yeung (2019). *Deep-IRT* - Neural network IRT models

**Relevant Existing Models in pykt**:
- `IEKT` (pykt/models/iekt.py): Uses item embeddings, could inspire difficulty initialization
- `LPKT` (pykt/models/lpkt.py): Learns skill-specific parameters
- `QIKT` (pykt/models/qikt.py): Question-level modeling

**Python Libraries**:
- `py-irt`: Python IRT calibration for validation
- `mirt` (R package): Comprehensive IRT toolkit for reference

### Python Packages for Rasch Calibration

**Available Options**:

1. **`py-irt`** (Recommended for our use case)
   - **Pros**: 
     - Pure Python, easy pip install
     - Supports 1PL (Rasch), 2PL, 3PL models
     - Marginal Maximum Likelihood (MML) estimation
     - Simple API, handles large datasets
     - Active maintenance, good documentation
   - **Cons**:
     - Limited visualization tools (need custom plots)
     - Basic fit statistics (no extensive diagnostics)
   - **Data Format**: CSV with (user_id, item_id, response)
   - **Installation**: `pip install py-irt`
   - **Usage**:
     ```python
     from pyirt import irt
     item_param, user_param = irt(src_fp='data.csv', 
                                    theta_bnds=[-4,4],
                                    num_theta=11, 
                                    mode='mle',
                                    is_2pl=False)  # False = Rasch/1PL
     ```
   - **Verdict**: ✅ **Best choice for production use**

2. **`pyrasch`**
   - **Pros**:
     - Specifically designed for Rasch models
     - Joint Maximum Likelihood Estimation (JMLE)
     - Built-in fit statistics (infit, outfit)
   - **Cons**:
     - Less maintained (last update 2018)
     - Smaller community
     - Slower on large datasets
   - **Data Format**: Binary matrix (students × items)
   - **Installation**: `pip install pyrasch`
   - **Verdict**: ⚠️ Consider only if need specialized Rasch diagnostics

3. **`mirt` (via rpy2)**
   - **Pros**:
     - Gold standard in psychometrics (R package)
     - Comprehensive diagnostics and visualization
     - Highly validated, widely used in research
   - **Cons**:
     - Requires R installation + rpy2
     - Complex setup, harder to debug
     - Overkill for our neural network use case
   - **Installation**: `pip install rpy2` + `install.packages("mirt")` in R
   - **Verdict**: ❌ Not worth the overhead for our purposes

4. **`girth`**
   - **Pros**:
     - Fast, modern Python implementation
     - Multiple estimation methods (MML, JMLE)
     - Good for large-scale IRT
   - **Cons**:
     - Newer package (less battle-tested)
     - Documentation could be better
   - **Data Format**: NumPy array (students × items)
   - **Installation**: `pip install girth`
   - **Verdict**: ⚠️ Alternative to py-irt if speed critical

5. **Custom Implementation (from scratch)**
   - **Pros**:
     - Full control over algorithm
     - Can optimize for KT-specific structure
     - Educational value (understand the math)
   - **Cons**:
     - Time-consuming (1-2 weeks to implement + test)
     - Risk of bugs in optimization
     - Need to validate against established methods
   - **Verdict**: ❌ Not worth it unless specific requirements

**Recommendation: Use `py-irt`**

**Rationale**:
- ✅ **Easy integration**: Simple CSV format, minimal data prep
- ✅ **Production-ready**: Stable, well-tested, good performance
- ✅ **Sufficient features**: Covers all our needs (Rasch calibration, parameter estimates)
- ✅ **Time-efficient**: Can get results in < 1 day vs weeks for custom code
- ✅ **Validation**: Industry-standard method, results comparable to JMLE/MML

### Data Preparation Effort Analysis

**What ASSISTments data looks like** (after pykt preprocessing):
```python
# In /data/assist2015/assist2015_train.pkl
{
    'qseqs': [[123, 45, 67, ...], ...],      # skill sequences
    'cseqs': [[1, 0, 1, ...], ...],          # correct sequences  
    'uids': [student_1, student_2, ...]
}
```

**What `py-irt` needs**:
```csv
user_id,item_id,correct
student_1,123,1
student_1,45,0
student_1,67,1
student_2,123,1
...
```

**Conversion Code** (≈30 lines):
```python
import pickle
import pandas as pd

def convert_pykt_to_irt_format(pkl_path, output_csv):
    """Convert pykt format to py-irt CSV format."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    rows = []
    for uid, qseq, cseq in zip(data['uids'], data['qseqs'], data['cseqs']):
        for skill_id, correct in zip(qseq, cseq):
            rows.append({
                'user_id': uid,
                'item_id': skill_id,
                'correct': correct
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Converted {len(rows)} interactions to {output_csv}")

# Usage
convert_pykt_to_irt_format(
    '/workspaces/pykt-toolkit/data/assist2015/assist2015_train.pkl',
    '/tmp/assist2015_irt.csv'
)
```

**Effort**: ✅ **~30 minutes** (trivial conversion)

### Cost-Benefit Analysis

**Option A: Use `py-irt` Package**

**Time Investment**:
- Data conversion script: 30 minutes
- Install py-irt: 2 minutes  
- Run calibration: 5-10 minutes (ASSISTments scale)
- Parse results & validate: 1 hour
- **Total: ~2-3 hours**

**Benefits**:
- ✅ Industry-standard IRT parameters (validated algorithm)
- ✅ Can compare neural model against established baseline
- ✅ Provides ground truth for difficulty initialization
- ✅ Enables psychometric validation of learned parameters
- ✅ Quick iteration (re-run calibration after data changes)

**Risks**:
- ⚠️ May need to handle edge cases (students with <3 responses, etc.)
- ⚠️ Package might not handle sparse data gracefully

**Option B: Custom Rasch Implementation**

**Time Investment**:
- Implement JMLE/MML algorithm: 3-5 days
- Add numerical optimization: 1-2 days
- Debug convergence issues: 2-3 days
- Validate against known results: 1-2 days
- **Total: ~1-2 weeks**

**Benefits**:
- ✅ Full control over algorithm
- ✅ Can optimize for sequence structure
- ✅ Educational (deep understanding of IRT)

**Risks**:
- ❌ High risk of bugs in optimization
- ❌ Difficult to validate (no ground truth)
- ❌ Time sink (opportunity cost)
- ❌ Maintenance burden

**Option C: Hybrid Approach**

**Strategy**: Use `py-irt` for baseline, custom code for neural integration

**Time Investment**:
- py-irt baseline: 2-3 hours
- Custom TCC computation: 4-6 hours
- Visualization tools: 4-6 hours
- **Total: ~10-15 hours (1-2 days)**

**Benefits**:
- ✅ Best of both worlds
- ✅ Validated baseline from py-irt
- ✅ Custom tools optimized for KT sequences
- ✅ Can visualize temporal dynamics (py-irt is static)

**Risks**:
- None significant

### Recommendation: Hybrid Approach

**Phase 1: Baseline Calibration** (Use `py-irt`)
```python
# Step 1: Convert data (30 min)
python tmp/convert_to_irt_format.py --dataset assist2015

# Step 2: Run py-irt (10 min)
from pyirt import irt
theta, b, c = irt('tmp/assist2015_irt.csv', 
                  is_2pl=False,  # Rasch model
                  theta_bnds=[-4, 4])

# Step 3: Save parameters (5 min)
import json
with open('tmp/rasch_baseline_assist2015.json', 'w') as f:
    json.dump({'theta': theta, 'b': b}, f)
```

**Phase 2: Custom TCC Tools** (Build on top of py-irt results)
```python
# Step 4: Load baseline parameters
with open('tmp/rasch_baseline_assist2015.json', 'r') as f:
    params = json.load(f)

# Step 5: Compute temporal TCCs (our custom code)
def compute_dynamic_tcc(student_sequence, b_dict, theta_init):
    """Custom code for temporal TCC with learning."""
    # This handles the temporal aspect that py-irt doesn't model
    pass

# Step 6: Visualizations (our custom code)
def plot_learning_trajectories(students, b_dict):
    """Custom plots specific to KT context."""
    pass
```

**Why This Works**:
1. **py-irt** handles the hard part: parameter estimation (proven algorithm)
2. **Custom code** handles the easy part: TCC computation and visualization (simple formulas)
3. **Validation**: Can verify custom TCC against py-irt baseline for static case
4. **Efficiency**: Don't reinvent the wheel for parameter estimation

### Practical Implementation Plan

**Day 1: Baseline Calibration**
- [ ] Write data conversion script (30 min)
- [ ] Install and test py-irt on small sample (30 min)
- [ ] Run calibration on full ASSISTments datasets (1 hour)
- [ ] Validate results (check parameter ranges, correlations) (1 hour)
- [ ] **Deliverable**: `rasch_baseline_assist2015.json`, `rasch_baseline_assist2009.json`

**Day 2: Custom TCC Tools**
- [ ] Implement dynamic ability update functions (2 hours)
- [ ] Implement TCC computation for sequences (2 hours)
- [ ] Create visualization functions (2 hours)
- [ ] Test on sample students (1 hour)
- [ ] **Deliverable**: `tmp/rasch_tcc_tools.py` module

**Day 3: Integration with iKT**
- [ ] Initialize difficulty bank with py-irt parameters (1 hour)
- [ ] Implement Rasch layer using baseline b_j values (2 hours)
- [ ] Add TCC computation to evaluation (2 hours)
- [ ] Run test training to verify gradient flow (2 hours)
- [ ] **Deliverable**: Working iKT-Rasch model

**Total Timeline**: 3 days (vs 2-3 weeks for full custom implementation)

### Code Examples

**Complete Data Conversion**:
```python
# tmp/convert_to_irt_format.py
import pickle
import pandas as pd
import argparse

def convert_dataset(dataset_name):
    """Convert pykt pickle to py-irt CSV format."""
    pkl_path = f'data/{dataset_name}/{dataset_name}_train.pkl'
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Flatten sequences into (user, item, response) tuples
    irt_data = []
    for uid, qseq, cseq in zip(data['uids'], data['qseqs'], data['cseqs']):
        for pos, (skill, correct) in enumerate(zip(qseq, cseq)):
            irt_data.append({
                'user_id': uid,
                'item_id': skill,
                'correct': correct,
                'position': pos  # Keep temporal info for later analysis
            })
    
    df = pd.DataFrame(irt_data)
    
    # Save for py-irt (needs only user_id, item_id, correct)
    output_csv = f'tmp/{dataset_name}_irt.csv'
    df[['user_id', 'item_id', 'correct']].to_csv(output_csv, index=False)
    
    # Save full version for our custom tools
    df.to_pickle(f'tmp/{dataset_name}_irt_full.pkl')
    
    print(f"✓ Converted {len(df)} interactions")
    print(f"✓ {df['user_id'].nunique()} students")
    print(f"✓ {df['item_id'].nunique()} skills")
    print(f"✓ Saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='assist2015')
    args = parser.parse_args()
    convert_dataset(args.dataset)
```

**Run py-irt Calibration**:
```python
# tmp/run_rasch_calibration.py
from pyirt import irt
import json
import argparse

def calibrate_rasch(dataset_name):
    """Run Rasch calibration using py-irt."""
    csv_path = f'tmp/{dataset_name}_irt.csv'
    
    print(f"Running Rasch calibration on {csv_path}...")
    
    # Run 1PL (Rasch) model
    # theta_bnds: ability range
    # num_theta: discretization for numerical integration
    user_param, item_param = irt(
        src_fp=csv_path,
        theta_bnds=[-4, 4],
        num_theta=21,  # More points = more accurate
        mode='mle',    # Maximum likelihood estimation
        is_2pl=False   # False = 1PL (Rasch model)
    )
    
    # Convert to serializable format
    theta_dict = {k: float(v) for k, v in user_param.items()}
    b_dict = {k: float(v) for k, v in item_param.items()}
    
    # Save results
    output_json = f'tmp/rasch_baseline_{dataset_name}.json'
    with open(output_json, 'w') as f:
        json.dump({
            'theta': theta_dict,  # {user_id: ability}
            'b': b_dict,          # {skill_id: difficulty}
            'dataset': dataset_name,
            'model': 'Rasch (1PL)'
        }, f, indent=2)
    
    print(f"✓ Calibration complete")
    print(f"✓ {len(theta_dict)} student abilities estimated")
    print(f"✓ {len(b_dict)} skill difficulties estimated")
    print(f"✓ Saved to {output_json}")
    
    # Print summary statistics
    import numpy as np
    theta_vals = list(theta_dict.values())
    b_vals = list(b_dict.values())
    
    print(f"\nAbility (θ) statistics:")
    print(f"  Mean: {np.mean(theta_vals):.3f}")
    print(f"  Std:  {np.std(theta_vals):.3f}")
    print(f"  Range: [{np.min(theta_vals):.3f}, {np.max(theta_vals):.3f}]")
    
    print(f"\nDifficulty (b) statistics:")
    print(f"  Mean: {np.mean(b_vals):.3f}")
    print(f"  Std:  {np.std(b_vals):.3f}")
    print(f"  Range: [{np.min(b_vals):.3f}, {np.max(b_vals):.3f}]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='assist2015')
    args = parser.parse_args()
    calibrate_rasch(args.dataset)
```

**Verdict**: ✅ **Use py-irt for calibration, custom code for TCC/visualization**

This approach is:
- **Fast**: 2-3 hours vs 1-2 weeks
- **Reliable**: Validated algorithm vs potential bugs
- **Flexible**: Can extend with custom temporal modeling
- **Practical**: Minimal data prep, immediate results

## Guidelines for Agent Interaction

When consulting this agent about IRT/Rasch integration:

1. **Be Specific**: Clarify whether asking about theory, implementation, or validation
2. **Provide Context**: Mention dataset characteristics (num_skills, sequence lengths, etc.)
3. **State Objectives**: Prioritize interpretability vs performance vs computational efficiency
4. **Request Validation**: Ask for statistical tests to verify Rasch assumptions hold

This agent will provide psychometrically sound guidance while respecting deep learning engineering constraints.

---

## 13. Computing Rasch Curves for ASSISTments Datasets

### Dataset Characteristics

**ASSISTments 2009-2010**:
- **Structure**: Student interactions with math problems
- **Features**: `user_id`, `skill_id` (problem skill tag), `correct` (0/1), `timestamp`
- **Skills**: ~110-120 unique skills
- **Students**: ~4,000 students
- **Interactions**: ~325,000 total attempts
- **Typical Sequence Length**: 20-200 interactions per student

**ASSISTments 2015**:
- **Structure**: Similar to 2009, more recent data
- **Features**: `user_id`, `skill_id`, `correct`, `timestamp`
- **Skills**: ~100 unique skills
- **Students**: ~19,000 students
- **Interactions**: ~700,000 total attempts
- **Typical Sequence Length**: 10-100 interactions per student

### Feasibility Analysis

**✅ Can We Calculate Rasch Curves?**

**Yes**, both datasets contain the minimum required information:

1. **Student Sequences** ✅: Each student has ordered interaction history
2. **Skill Labels** ✅: Each interaction tagged with skill ID
3. **Binary Outcomes** ✅: Correct (1) or incorrect (0) responses
4. **Sufficient Sample Size** ✅: Thousands of students, hundreds of thousands of interactions

**What We Can Compute**:

#### A. Empirical Rasch Calibration (Baseline)
- Use classical IRT methods to estimate θ and b from observed data
- Provides ground truth for validating neural Rasch model
- Tools: `py-irt` library, MLE or joint maximum likelihood estimation (JMLE)

#### B. Item Characteristic Curves (ICCs)
- For each skill k, plot P(correct | θ) vs θ
- Shows how skill difficulty relates to student ability
- Can identify skills that don't fit Rasch model (discrimination varies)

#### C. Test Characteristic Curves (TCCs)
- For each student, plot expected cumulative score vs ability trajectory
- Shows learning progression over time
- Can compare predicted (from model) vs empirical (from data) TCCs

### Implementation Approach

#### Step 1: Empirical Rasch Calibration

**Purpose**: Establish baseline IRT parameters from data

**Method**:
```python
import numpy as np
from pyirt import irt

def calibrate_rasch_baseline(data):
    """
    Calibrate Rasch model on ASSISTments data.
    
    Args:
        data: DataFrame with columns ['user_id', 'skill_id', 'correct']
    Returns:
        theta_dict: {user_id: ability estimate}
        b_dict: {skill_id: difficulty estimate}
    """
    # Prepare data for py-irt (requires tuples)
    irt_data = [
        (row['user_id'], row['skill_id'], row['correct'])
        for _, row in data.iterrows()
    ]
    
    # Fit 1PL (Rasch) model
    # This uses marginal maximum likelihood (MML)
    src_fp = '/tmp/irt_data.csv'
    with open(src_fp, 'w') as f:
        for user, skill, correct in irt_data:
            f.write(f"{user},{skill},{correct}\n")
    
    # Run calibration
    theta, b, c = irt(src_fp, num_theta=1, is_2pl=False)
    
    return theta, b
```

**Output**:
- `theta[user_id]`: Ability estimate for each student (scalar, typically in range [-3, 3])
- `b[skill_id]`: Difficulty estimate for each skill (scalar, same scale as theta)

**Validation**:
- **Model Fit**: Check infit/outfit statistics (should be 0.5-1.5 for good fit)
- **Separation**: Ability and difficulty should show reasonable spread
- **Reliability**: Person reliability > 0.7, item reliability > 0.9 (desired)

#### Step 2: Compute Skill-Level ICCs

**Purpose**: Visualize difficulty of each skill

**Method**:
```python
import matplotlib.pyplot as plt

def plot_skill_icc(skill_id, b_skill, theta_range=(-3, 3), num_points=100):
    """
    Plot ICC for a single skill.
    
    Args:
        skill_id: Skill identifier
        b_skill: Difficulty parameter for this skill
        theta_range: Range of abilities to plot
        num_points: Number of points for smooth curve
    """
    theta_vals = np.linspace(theta_range[0], theta_range[1], num_points)
    
    # Rasch model: P(correct | θ, b) = σ(θ - b)
    probs = 1 / (1 + np.exp(-(theta_vals - b_skill)))
    
    plt.figure(figsize=(8, 6))
    plt.plot(theta_vals, probs, linewidth=2)
    plt.axvline(b_skill, color='red', linestyle='--', 
                label=f'Difficulty b={b_skill:.2f}')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Student Ability (θ)', fontsize=12)
    plt.ylabel('P(Correct Response)', fontsize=12)
    plt.title(f'Item Characteristic Curve - Skill {skill_id}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt

# Plot multiple skills for comparison
def plot_multiple_iccs(skill_ids, b_dict):
    """Compare ICCs for multiple skills."""
    theta_vals = np.linspace(-3, 3, 100)
    
    plt.figure(figsize=(10, 6))
    for skill_id in skill_ids:
        b = b_dict[skill_id]
        probs = 1 / (1 + np.exp(-(theta_vals - b)))
        plt.plot(theta_vals, probs, label=f'Skill {skill_id} (b={b:.2f})')
    
    plt.xlabel('Student Ability (θ)', fontsize=12)
    plt.ylabel('P(Correct Response)', fontsize=12)
    plt.title('Item Characteristic Curves - Multiple Skills', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt
```

**Interpretation**:
- **Easy skills** (b < -1): ICC shifts left, even low-ability students succeed
- **Hard skills** (b > 1): ICC shifts right, only high-ability students succeed
- **Medium skills** (b ≈ 0): ICC centered, 50% success at population mean ability

#### Step 3: Compute Student-Level TCCs

**Purpose**: Visualize learning trajectory for individual students

**Method**:
```python
def compute_student_tcc(student_data, theta_trajectory, b_dict):
    """
    Compute TCC for a single student's interaction sequence.
    
    Args:
        student_data: DataFrame with columns ['skill_id', 'correct', 'position']
                     sorted by timestamp
        theta_trajectory: Array of ability estimates at each position
                         (either constant or time-varying)
        b_dict: Dictionary mapping skill_id to difficulty
    Returns:
        tcc_expected: Expected cumulative score at each position
        tcc_observed: Observed cumulative score at each position
    """
    num_attempts = len(student_data)
    
    # Option 1: Constant ability (classical Rasch)
    if isinstance(theta_trajectory, float):
        theta_trajectory = np.full(num_attempts, theta_trajectory)
    
    # Option 2: Time-varying ability (dynamic Rasch)
    # theta_trajectory is array of length num_attempts
    
    tcc_expected = []
    tcc_observed = []
    
    cumsum_expected = 0.0
    cumsum_observed = 0
    
    for i, row in student_data.iterrows():
        skill_id = row['skill_id']
        correct = row['correct']
        theta_t = theta_trajectory[i]
        b = b_dict[skill_id]
        
        # ICC: probability of correct at this timestep
        p_correct = 1 / (1 + np.exp(-(theta_t - b)))
        
        # Update cumulative sums
        cumsum_expected += p_correct
        cumsum_observed += correct
        
        tcc_expected.append(cumsum_expected)
        tcc_observed.append(cumsum_observed)
    
    return np.array(tcc_expected), np.array(tcc_observed)

def plot_student_tcc(student_id, tcc_expected, tcc_observed):
    """Plot expected vs observed TCC for a student."""
    positions = np.arange(1, len(tcc_expected) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(positions, tcc_expected, label='Expected (TCC)', 
             linewidth=2, color='blue')
    plt.plot(positions, tcc_observed, label='Observed (Actual)', 
             linewidth=2, color='orange', alpha=0.7)
    plt.xlabel('Interaction Position (t)', fontsize=12)
    plt.ylabel('Cumulative Correct Responses', fontsize=12)
    plt.title(f'Test Characteristic Curve - Student {student_id}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt
```

**Interpretation**:
- **Good fit**: Observed TCC closely follows expected TCC
- **Underperformance**: Observed below expected (student struggling)
- **Overperformance**: Observed above expected (student excelling)
- **Slope**: Steeper slope indicates faster learning

#### Step 4: Dynamic Ability Estimation

**Purpose**: Model time-varying ability θ(t) instead of constant θ

**Method Options**:

**A. Simple Learning Rate Model**:
```python
def estimate_dynamic_ability(student_data, theta_0=0.0, alpha=0.1):
    """
    Estimate time-varying ability with fixed learning rate.
    
    Args:
        student_data: DataFrame with 'correct' column
        theta_0: Initial ability
        alpha: Learning rate (ability gain per correct response)
    Returns:
        theta_trajectory: Array of abilities at each timestep
    """
    theta_trajectory = [theta_0]
    theta_current = theta_0
    
    for _, row in student_data.iterrows():
        correct = row['correct']
        # Update ability based on performance
        theta_current += alpha * correct
        theta_trajectory.append(theta_current)
    
    return np.array(theta_trajectory[:-1])  # Exclude final state
```

**B. Performance-Based Update**:
```python
def estimate_dynamic_ability_performance(student_data, b_dict, 
                                         theta_0=0.0, alpha=0.2):
    """
    Update ability based on performance surprise (prediction error).
    
    θ(t+1) = θ(t) + α * (actual - predicted)
    """
    theta_trajectory = [theta_0]
    theta_current = theta_0
    
    for _, row in student_data.iterrows():
        skill_id = row['skill_id']
        correct = row['correct']
        b = b_dict[skill_id]
        
        # Predict probability
        p_correct = 1 / (1 + np.exp(-(theta_current - b)))
        
        # Prediction error
        error = correct - p_correct
        
        # Update ability (learn more from surprises)
        theta_current += alpha * error
        theta_trajectory.append(theta_current)
    
    return np.array(theta_trajectory[:-1])
```

**C. Exponential Moving Average**:
```python
def estimate_dynamic_ability_ema(student_data, b_dict, 
                                 theta_0=0.0, beta=0.9):
    """
    Smooth ability updates using exponential moving average.
    
    θ(t) = β * θ(t-1) + (1-β) * observed_performance
    """
    theta_trajectory = [theta_0]
    theta_current = theta_0
    
    for _, row in student_data.iterrows():
        skill_id = row['skill_id']
        correct = row['correct']
        b = b_dict[skill_id]
        
        # Infer ability from this observation
        # If correct: θ ≈ b, if incorrect: θ ≈ b - 2
        inferred_theta = b if correct else b - 2
        
        # Smooth update
        theta_current = beta * theta_current + (1 - beta) * inferred_theta
        theta_trajectory.append(theta_current)
    
    return np.array(theta_trajectory[:-1])
```

#### Step 5: Model Validation Metrics

**Purpose**: Quantify how well Rasch model fits ASSISTments data

**Metrics**:

**A. TCC Prediction Error**:
```python
def compute_tcc_metrics(tcc_expected, tcc_observed):
    """
    Compare predicted vs observed TCCs.
    
    Returns:
        mae: Mean absolute error
        rmse: Root mean squared error
        r2: R-squared (coefficient of determination)
    """
    mae = np.mean(np.abs(tcc_expected - tcc_observed))
    rmse = np.sqrt(np.mean((tcc_expected - tcc_observed)**2))
    
    # R-squared
    ss_total = np.sum((tcc_observed - np.mean(tcc_observed))**2)
    ss_residual = np.sum((tcc_observed - tcc_expected)**2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}
```

**B. Skill Difficulty Correlation**:
```python
def validate_difficulty_estimates(b_dict, empirical_success_rates):
    """
    Check if learned difficulties match empirical data.
    
    Expected: Higher b (difficulty) → Lower success rate
    """
    from scipy.stats import spearmanr
    
    skills = list(b_dict.keys())
    b_values = [b_dict[s] for s in skills]
    success_rates = [empirical_success_rates[s] for s in skills]
    
    # Should have strong negative correlation
    corr, p_value = spearmanr(b_values, success_rates)
    
    print(f"Difficulty vs Success Rate: ρ = {corr:.3f} (p={p_value:.3e})")
    print(f"Expected: ρ < -0.7 (higher difficulty → lower success)")
    
    return corr
```

**C. ICC Model Fit**:
```python
def assess_icc_fit(data, theta_dict, b_dict, num_bins=10):
    """
    Check if empirical ICCs match Rasch predictions.
    
    Method: Bin students by ability, compute observed vs expected success.
    """
    # Add theta to data
    data['theta'] = data['user_id'].map(theta_dict)
    data['b'] = data['skill_id'].map(b_dict)
    data['expected_p'] = 1 / (1 + np.exp(-(data['theta'] - data['b'])))
    
    # Bin by ability
    data['theta_bin'] = pd.cut(data['theta'], bins=num_bins, labels=False)
    
    # Compare observed vs expected per bin
    fit_metrics = []
    for bin_id in range(num_bins):
        bin_data = data[data['theta_bin'] == bin_id]
        observed_p = bin_data['correct'].mean()
        expected_p = bin_data['expected_p'].mean()
        
        fit_metrics.append({
            'bin': bin_id,
            'observed': observed_p,
            'expected': expected_p,
            'error': abs(observed_p - expected_p)
        })
    
    fit_df = pd.DataFrame(fit_metrics)
    mae = fit_df['error'].mean()
    
    print(f"ICC Fit MAE: {mae:.4f}")
    print(f"Expected: MAE < 0.05 for good fit")
    
    return fit_df
```

### Expected Results for ASSISTments Datasets

**Skill Difficulty Range** (based on typical educational data):
- **Easy skills** (b = -2 to -1): Basic arithmetic, simple concepts
- **Medium skills** (b = -0.5 to 0.5): Standard curriculum topics
- **Hard skills** (b = 1 to 2): Advanced problem-solving, multi-step

**Student Ability Range**:
- **Low performers** (θ = -2 to -1): Struggling students
- **Average performers** (θ = -0.5 to 0.5): Typical students
- **High performers** (θ = 1 to 2): Advanced students

**TCC Characteristics**:
- **Shape**: S-shaped curve (sigmoid-like cumulative)
- **Slope**: Varies by student (faster learners have steeper slope)
- **Ceiling**: Approaches total number of attempts
- **Learning Signal**: Should show upward trend if learning occurs

### Practical Implementation Steps

**Step-by-Step Guide**:

1. **Load ASSISTments Data**:
   ```bash
   # Data should already be processed in /data/assist2015 and /data/assist2009
   ```

2. **Run Empirical Calibration**:
   ```python
   # Script: tmp/rasch_calibration_baseline.py
   python tmp/rasch_calibration_baseline.py --dataset assist2015
   ```
   Output: `theta_estimates.json`, `difficulty_estimates.json`

3. **Compute and Plot ICCs**:
   ```python
   # Script: tmp/plot_skill_iccs.py
   python tmp/plot_skill_iccs.py --dataset assist2015 --num_skills 10
   ```
   Output: `skill_iccs.png` (visual comparison of skill difficulties)

4. **Compute and Plot TCCs**:
   ```python
   # Script: tmp/plot_student_tccs.py
   python tmp/plot_student_tccs.py --dataset assist2015 --num_students 5
   ```
   Output: `student_tccs.png` (learning trajectories)

5. **Validate Model Fit**:
   ```python
   # Script: tmp/validate_rasch_fit.py
   python tmp/validate_rasch_fit.py --dataset assist2015
   ```
   Output: Fit statistics, correlation metrics, validation report

**Timeline**: 2-3 days to implement and run all validation scripts

### Integration with iKT

Once we have empirical Rasch parameters:

1. **Initialize Difficulty Bank**: Use calibrated b_j values
2. **Validate Neural Model**: Compare learned θ_t with empirical estimates
3. **Benchmark TCCs**: Compare neural TCC predictions with empirical curves
4. **Interpretability**: Use Rasch parameters for educational insights

**Key Advantage**: Empirical calibration provides ground truth for validating that the neural model learns meaningful IRT parameters, not arbitrary values.

---

## 14. Implementation Plan: Hybrid Approach for iKT-Rasch

### Overview

**Strategy**: Use `py-irt` for empirical Rasch calibration (baseline), then build custom tools for temporal modeling and neural integration.

**Timeline**: 3 days for complete implementation and validation

**Key Principle**: Don't reinvent the wheel for parameter estimation (use proven py-irt), focus our effort on temporal dynamics and neural architecture integration.

---

### Phase 1: Empirical Rasch Baseline (Day 1)

**Objective**: Establish ground truth IRT parameters using classical Rasch calibration

**Duration**: ~3-4 hours

#### Task 1.1: Install py-irt Package (5 min)

```bash
cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate
pip install py-irt
```

**Validation**: Run `python -c "from pyirt import irt; print('✓ py-irt installed')"` to verify installation.

#### Task 1.2: Create Data Conversion Script (30 min)

**File**: `tmp/convert_to_irt_format.py`

**Purpose**: Convert pykt pickle format to py-irt CSV format

**Implementation**:
```python
#!/usr/bin/env python
"""
Convert pykt dataset format to py-irt CSV format.

Usage:
    python tmp/convert_to_irt_format.py --dataset assist2015
    python tmp/convert_to_irt_format.py --dataset assist2009
"""

import pickle
import pandas as pd
import argparse
import os

def convert_dataset(dataset_name):
    """Convert pykt pickle to py-irt CSV format."""
    # Path to pykt preprocessed data
    pkl_path = f'data/{dataset_name}/{dataset_name}_train.pkl'
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Dataset not found: {pkl_path}")
    
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Converting to IRT format...")
    # Flatten sequences into (user, item, response) tuples
    irt_data = []
    for uid, qseq, cseq in zip(data['uids'], data['qseqs'], data['cseqs']):
        for pos, (skill, correct) in enumerate(zip(qseq, cseq)):
            irt_data.append({
                'user_id': uid,
                'item_id': int(skill),
                'correct': int(correct),
                'position': pos  # Keep temporal info for later analysis
            })
    
    df = pd.DataFrame(irt_data)
    
    # Create tmp directory if not exists
    os.makedirs('tmp', exist_ok=True)
    
    # Save for py-irt (needs only user_id, item_id, correct)
    output_csv = f'tmp/{dataset_name}_irt.csv'
    df[['user_id', 'item_id', 'correct']].to_csv(output_csv, index=False)
    
    # Save full version for our custom tools (includes position)
    output_pkl = f'tmp/{dataset_name}_irt_full.pkl'
    df.to_pickle(output_pkl)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Total interactions: {len(df):,}")
    print(f"  Unique students: {df['user_id'].nunique():,}")
    print(f"  Unique skills: {df['item_id'].nunique()}")
    print(f"  Mean interactions per student: {len(df) / df['user_id'].nunique():.1f}")
    print(f"\n✓ Saved to:")
    print(f"  - {output_csv} (for py-irt)")
    print(f"  - {output_pkl} (for custom tools)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert pykt data to py-irt format')
    parser.add_argument('--dataset', default='assist2015', 
                        choices=['assist2015', 'assist2009'],
                        help='Dataset to convert')
    args = parser.parse_args()
    
    convert_dataset(args.dataset)
```

**Run**:
```bash
python tmp/convert_to_irt_format.py --dataset assist2015
python tmp/convert_to_irt_format.py --dataset assist2009
```

**Expected Output**:
- `tmp/assist2015_irt.csv` (~700K rows)
- `tmp/assist2009_irt.csv` (~325K rows)
- `tmp/assist2015_irt_full.pkl` (includes position)
- `tmp/assist2009_irt_full.pkl` (includes position)

#### Task 1.3: Create Rasch Calibration Script (30 min)

**File**: `tmp/run_rasch_calibration.py`

**Purpose**: Run py-irt to estimate ability (θ) and difficulty (b) parameters

**Implementation**:
```python
#!/usr/bin/env python
"""
Run Rasch (1PL) calibration using py-irt.

Usage:
    python tmp/run_rasch_calibration.py --dataset assist2015
    python tmp/run_rasch_calibration.py --dataset assist2009
"""

from pyirt import irt
import json
import argparse
import os
import numpy as np

def calibrate_rasch(dataset_name):
    """Run Rasch calibration using py-irt."""
    csv_path = f'tmp/{dataset_name}_irt.csv'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}\n"
                              f"Run convert_to_irt_format.py first.")
    
    print(f"Running Rasch (1PL) calibration on {csv_path}...")
    print("This may take 5-10 minutes for large datasets...\n")
    
    # Run 1PL (Rasch) model
    # theta_bnds: ability range (-4 to 4 is standard)
    # num_theta: discretization for numerical integration (more = slower but more accurate)
    # mode: 'mle' = maximum likelihood estimation
    # is_2pl: False = 1PL (Rasch model with equal discrimination)
    user_param, item_param = irt(
        src_fp=csv_path,
        theta_bnds=[-4, 4],
        num_theta=21,  # Good balance between speed and accuracy
        mode='mle',    
        is_2pl=False   # Rasch model (1PL)
    )
    
    # Convert to serializable format
    theta_dict = {str(k): float(v) for k, v in user_param.items()}
    b_dict = {int(k): float(v) for k, v in item_param.items()}
    
    # Save results
    output_json = f'tmp/rasch_baseline_{dataset_name}.json'
    with open(output_json, 'w') as f:
        json.dump({
            'theta': theta_dict,  # {user_id: ability}
            'b': b_dict,          # {skill_id: difficulty}
            'dataset': dataset_name,
            'model': 'Rasch (1PL)',
            'theta_range': [-4, 4],
            'num_theta_points': 21
        }, f, indent=2)
    
    print(f"✓ Calibration complete!")
    print(f"  Students with ability estimates: {len(theta_dict):,}")
    print(f"  Skills with difficulty estimates: {len(b_dict)}")
    print(f"  Saved to: {output_json}\n")
    
    # Print summary statistics
    theta_vals = list(theta_dict.values())
    b_vals = list(b_dict.values())
    
    print("=" * 60)
    print("ABILITY (θ) STATISTICS")
    print("=" * 60)
    print(f"  Mean:   {np.mean(theta_vals):7.3f}")
    print(f"  Std:    {np.std(theta_vals):7.3f}")
    print(f"  Min:    {np.min(theta_vals):7.3f}")
    print(f"  Q1:     {np.percentile(theta_vals, 25):7.3f}")
    print(f"  Median: {np.median(theta_vals):7.3f}")
    print(f"  Q3:     {np.percentile(theta_vals, 75):7.3f}")
    print(f"  Max:    {np.max(theta_vals):7.3f}")
    
    print("\n" + "=" * 60)
    print("DIFFICULTY (b) STATISTICS")
    print("=" * 60)
    print(f"  Mean:   {np.mean(b_vals):7.3f}")
    print(f"  Std:    {np.std(b_vals):7.3f}")
    print(f"  Min:    {np.min(b_vals):7.3f}")
    print(f"  Q1:     {np.percentile(b_vals, 25):7.3f}")
    print(f"  Median: {np.median(b_vals):7.3f}")
    print(f"  Q3:     {np.percentile(b_vals, 75):7.3f}")
    print(f"  Max:    {np.max(b_vals):7.3f}")
    
    # Identify easy, medium, hard skills
    print("\n" + "=" * 60)
    print("SKILL DIFFICULTY CATEGORIES")
    print("=" * 60)
    easy = [k for k, v in b_dict.items() if v < -0.5]
    medium = [k for k, v in b_dict.items() if -0.5 <= v <= 0.5]
    hard = [k for k, v in b_dict.items() if v > 0.5]
    
    print(f"  Easy skills (b < -0.5):   {len(easy):3d} ({100*len(easy)/len(b_dict):.1f}%)")
    print(f"  Medium skills (|b| ≤ 0.5): {len(medium):3d} ({100*len(medium)/len(b_dict):.1f}%)")
    print(f"  Hard skills (b > 0.5):    {len(hard):3d} ({100*len(hard)/len(b_dict):.1f}%)")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Rasch calibration')
    parser.add_argument('--dataset', default='assist2015',
                        choices=['assist2015', 'assist2009'],
                        help='Dataset to calibrate')
    args = parser.parse_args()
    
    calibrate_rasch(args.dataset)
```

**Run**:
```bash
python tmp/run_rasch_calibration.py --dataset assist2015
python tmp/run_rasch_calibration.py --dataset assist2009
```

**Expected Output**:
- `tmp/rasch_baseline_assist2015.json`
- `tmp/rasch_baseline_assist2009.json`
- Console output with ability/difficulty statistics

#### Task 1.4: Validate Baseline Parameters (1 hour)

**File**: `tmp/validate_rasch_baseline.py`

**Purpose**: Check if calibrated parameters are reasonable

**Implementation**:
```python
#!/usr/bin/env python
"""
Validate Rasch baseline calibration results.

Usage:
    python tmp/validate_rasch_baseline.py --dataset assist2015
"""

import json
import pickle
import pandas as pd
import numpy as np
import argparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def validate_baseline(dataset_name):
    """Validate Rasch calibration results."""
    
    # Load calibration results
    calib_path = f'tmp/rasch_baseline_{dataset_name}.json'
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    theta_dict = {k: v for k, v in calib['theta'].items()}
    b_dict = {int(k): v for k, v in calib['b'].items()}
    
    # Load original data for validation
    data_path = f'tmp/{dataset_name}_irt_full.pkl'
    df = pd.read_pickle(data_path)
    
    print("=" * 70)
    print(f"VALIDATION REPORT: {dataset_name.upper()}")
    print("=" * 70)
    
    # Test 1: Difficulty vs Success Rate Correlation
    print("\n[Test 1] Difficulty vs Empirical Success Rate")
    print("-" * 70)
    
    skill_stats = df.groupby('item_id')['correct'].agg(['mean', 'count']).reset_index()
    skill_stats.columns = ['skill_id', 'success_rate', 'count']
    skill_stats['b'] = skill_stats['skill_id'].map(b_dict)
    skill_stats = skill_stats.dropna()
    
    corr, p_value = spearmanr(skill_stats['b'], skill_stats['success_rate'])
    
    print(f"  Spearman correlation: ρ = {corr:.4f} (p = {p_value:.2e})")
    print(f"  Expected: Strong negative correlation (ρ < -0.7)")
    print(f"  Result: {'✓ PASS' if corr < -0.7 else '✗ FAIL'}")
    
    # Test 2: Ability vs Performance Correlation
    print("\n[Test 2] Ability vs Overall Performance")
    print("-" * 70)
    
    user_stats = df.groupby('user_id')['correct'].agg(['mean', 'count']).reset_index()
    user_stats.columns = ['user_id', 'success_rate', 'count']
    user_stats['theta'] = user_stats['user_id'].map(theta_dict)
    user_stats = user_stats.dropna()
    
    corr2, p_value2 = spearmanr(user_stats['theta'], user_stats['success_rate'])
    
    print(f"  Spearman correlation: ρ = {corr2:.4f} (p = {p_value2:.2e})")
    print(f"  Expected: Strong positive correlation (ρ > 0.7)")
    print(f"  Result: {'✓ PASS' if corr2 > 0.7 else '✗ FAIL'}")
    
    # Test 3: Parameter Distributions
    print("\n[Test 3] Parameter Distribution Checks")
    print("-" * 70)
    
    theta_vals = list(theta_dict.values())
    b_vals = list(b_dict.values())
    
    theta_std = np.std(theta_vals)
    b_std = np.std(b_vals)
    
    print(f"  Ability spread (std): {theta_std:.3f}")
    print(f"    Expected: > 0.5 (sufficient discrimination)")
    print(f"    Result: {'✓ PASS' if theta_std > 0.5 else '✗ FAIL'}")
    
    print(f"  Difficulty spread (std): {b_std:.3f}")
    print(f"    Expected: > 0.5 (diverse item difficulties)")
    print(f"    Result: {'✓ PASS' if b_std > 0.5 else '✗ FAIL'}")
    
    # Test 4: Prediction Accuracy
    print("\n[Test 4] Rasch Model Prediction Accuracy")
    print("-" * 70)
    
    df_sample = df.copy()
    df_sample['theta'] = df_sample['user_id'].map(theta_dict)
    df_sample['b'] = df_sample['item_id'].map(b_dict)
    df_sample = df_sample.dropna()
    
    # Rasch prediction
    df_sample['p_rasch'] = 1 / (1 + np.exp(-(df_sample['theta'] - df_sample['b'])))
    
    # Binarize predictions at 0.5 threshold
    df_sample['pred'] = (df_sample['p_rasch'] > 0.5).astype(int)
    
    accuracy = (df_sample['pred'] == df_sample['correct']).mean()
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(df_sample['correct'], df_sample['p_rasch'])
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Expected: AUC > 0.65 (better than random)")
    print(f"  Result: {'✓ PASS' if auc > 0.65 else '✗ FAIL'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = (corr < -0.7 and corr2 > 0.7 and theta_std > 0.5 and 
                b_std > 0.5 and auc > 0.65)
    print(f"  Overall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 70)
    
    return {
        'difficulty_correlation': corr,
        'ability_correlation': corr2,
        'theta_std': theta_std,
        'b_std': b_std,
        'accuracy': accuracy,
        'auc': auc,
        'all_pass': all_pass
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Rasch calibration')
    parser.add_argument('--dataset', default='assist2015',
                        choices=['assist2015', 'assist2009'],
                        help='Dataset to validate')
    args = parser.parse_args()
    
    results = validate_baseline(args.dataset)
```

**Run**:
```bash
python tmp/validate_rasch_baseline.py --dataset assist2015
python tmp/validate_rasch_baseline.py --dataset assist2009
```

**Deliverables (Phase 1)**:
- ✅ `tmp/rasch_baseline_assist2015.json` - θ and b parameters
- ✅ `tmp/rasch_baseline_assist2009.json` - θ and b parameters
- ✅ Validation report confirming parameters are reasonable

---

### Phase 2: Custom TCC Tools (Day 2)

**Objective**: Build temporal modeling tools on top of py-irt baseline

**Duration**: ~6-7 hours

#### Task 2.1: Create Dynamic Ability Module (2 hours)

**File**: `tmp/rasch_dynamic_ability.py`

**Purpose**: Implement time-varying ability estimation

**Implementation**:
```python
"""
Dynamic ability estimation for temporal Rasch modeling.

Provides multiple methods for modeling ability evolution θ(t).
"""

import numpy as np
import torch

def estimate_ability_simple(responses, theta_0=0.0, alpha=0.1):
    """
    Simple learning rate model: θ(t) = θ(t-1) + α * correct(t)
    
    Args:
        responses: Array of correct/incorrect [0, 1] at each timestep
        theta_0: Initial ability
        alpha: Learning rate
    Returns:
        theta_trajectory: Array of abilities [theta_0, theta_1, ...]
    """
    theta = [theta_0]
    current = theta_0
    
    for correct in responses:
        current = current + alpha * correct
        theta.append(current)
    
    return np.array(theta[:-1])  # Exclude final state (not used for prediction)


def estimate_ability_surprise(responses, skill_ids, b_dict, theta_0=0.0, alpha=0.2):
    """
    Surprise-based learning: θ(t) = θ(t-1) + α * (actual - predicted)
    
    Learn from prediction errors (larger update when surprised).
    
    Args:
        responses: Array of correct/incorrect [0, 1]
        skill_ids: Array of skill IDs
        b_dict: Dictionary {skill_id: difficulty}
        theta_0: Initial ability
        alpha: Learning rate
    Returns:
        theta_trajectory: Array of abilities
    """
    theta = [theta_0]
    current = theta_0
    
    for correct, skill_id in zip(responses, skill_ids):
        b = b_dict.get(skill_id, 0.0)
        
        # Predict probability using current ability
        p_correct = 1 / (1 + np.exp(-(current - b)))
        
        # Prediction error
        error = correct - p_correct
        
        # Update ability based on surprise
        current = current + alpha * error
        theta.append(current)
    
    return np.array(theta[:-1])


def estimate_ability_ema(responses, skill_ids, b_dict, theta_0=0.0, beta=0.9):
    """
    Exponential moving average: θ(t) = β*θ(t-1) + (1-β)*inferred_theta
    
    Smooth ability updates by averaging past and current evidence.
    
    Args:
        responses: Array of correct/incorrect [0, 1]
        skill_ids: Array of skill IDs
        b_dict: Dictionary {skill_id: difficulty}
        theta_0: Initial ability
        beta: Smoothing factor (higher = more weight to history)
    Returns:
        theta_trajectory: Array of abilities
    """
    theta = [theta_0]
    current = theta_0
    
    for correct, skill_id in zip(responses, skill_ids):
        b = b_dict.get(skill_id, 0.0)
        
        # Infer ability from this observation
        # If correct: ability likely around b or higher
        # If incorrect: ability likely below b
        inferred = b + (1.0 if correct else -1.0)
        
        # Smooth update
        current = beta * current + (1 - beta) * inferred
        theta.append(current)
    
    return np.array(theta[:-1])


def estimate_ability_forgetting(responses, skill_ids, b_dict, theta_0=0.0, 
                                alpha=0.2, decay=0.99):
    """
    Forgetting-aware: θ(t) = decay * θ(t-1) + α * learning_signal
    
    Models both learning (gain) and forgetting (decay).
    
    Args:
        responses: Array of correct/incorrect [0, 1]
        skill_ids: Array of skill IDs
        b_dict: Dictionary {skill_id: difficulty}
        theta_0: Initial ability
        alpha: Learning rate
        decay: Forgetting rate (< 1.0 for decay)
    Returns:
        theta_trajectory: Array of abilities
    """
    theta = [theta_0]
    current = theta_0
    
    for correct, skill_id in zip(responses, skill_ids):
        b = b_dict.get(skill_id, 0.0)
        
        # Predict probability
        p_correct = 1 / (1 + np.exp(-(current - b)))
        
        # Learning signal (prediction error)
        error = correct - p_correct
        
        # Update with decay
        current = decay * current + alpha * error
        theta.append(current)
    
    return np.array(theta[:-1])


# Batch processing for neural network integration
class DynamicAbilityEstimator:
    """
    Batch-compatible dynamic ability estimator for PyTorch.
    """
    
    def __init__(self, method='surprise', **kwargs):
        self.method = method
        self.kwargs = kwargs
    
    def estimate_batch(self, responses, skill_ids, b_dict):
        """
        Estimate abilities for a batch of sequences.
        
        Args:
            responses: [batch_size, seq_len] tensor
            skill_ids: [batch_size, seq_len] tensor
            b_dict: Dictionary {skill_id: difficulty}
        Returns:
            theta_batch: [batch_size, seq_len] tensor of abilities
        """
        batch_size, seq_len = responses.shape
        theta_batch = []
        
        for i in range(batch_size):
            resp = responses[i].cpu().numpy()
            skills = skill_ids[i].cpu().numpy()
            
            if self.method == 'simple':
                theta = estimate_ability_simple(resp, **self.kwargs)
            elif self.method == 'surprise':
                theta = estimate_ability_surprise(resp, skills, b_dict, **self.kwargs)
            elif self.method == 'ema':
                theta = estimate_ability_ema(resp, skills, b_dict, **self.kwargs)
            elif self.method == 'forgetting':
                theta = estimate_ability_forgetting(resp, skills, b_dict, **self.kwargs)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            theta_batch.append(theta)
        
        return torch.tensor(np.array(theta_batch), dtype=torch.float32)
```

**Test**: Create `tmp/test_dynamic_ability.py` to verify functions work correctly.

#### Task 2.2: Create TCC Computation Module (2 hours)

**File**: `tmp/rasch_tcc_computation.py`

**Purpose**: Compute Test Characteristic Curves

**Implementation**:
```python
"""
Test Characteristic Curve (TCC) computation for Rasch model.

TCC represents expected cumulative score as function of ability trajectory.
"""

import numpy as np
import torch

def compute_tcc_single(theta_trajectory, skill_sequence, b_dict):
    """
    Compute TCC for a single student sequence.
    
    Args:
        theta_trajectory: [seq_len] - ability at each timestep
        skill_sequence: [seq_len] - skill IDs attempted
        b_dict: {skill_id: difficulty}
    Returns:
        tcc_expected: [seq_len] - expected cumulative correct
        icc_values: [seq_len] - probability at each timestep
    """
    seq_len = len(theta_trajectory)
    icc_values = np.zeros(seq_len)
    
    for t in range(seq_len):
        theta_t = theta_trajectory[t]
        skill_t = skill_sequence[t]
        b_t = b_dict.get(skill_t, 0.0)
        
        # Rasch ICC: P(correct | θ, b) = σ(θ - b)
        icc_values[t] = 1 / (1 + np.exp(-(theta_t - b_t)))
    
    # TCC = cumulative sum of ICCs
    tcc_expected = np.cumsum(icc_values)
    
    return tcc_expected, icc_values


def compute_tcc_observed(responses):
    """
    Compute observed cumulative score.
    
    Args:
        responses: [seq_len] - actual correct/incorrect
    Returns:
        tcc_observed: [seq_len] - observed cumulative correct
    """
    return np.cumsum(responses)


def compute_tcc_batch(theta_batch, skill_batch, b_dict):
    """
    Compute TCCs for a batch of sequences (PyTorch compatible).
    
    Args:
        theta_batch: [batch_size, seq_len] tensor
        skill_batch: [batch_size, seq_len] tensor  
        b_dict: {skill_id: difficulty}
    Returns:
        tcc_batch: [batch_size, seq_len] tensor
        icc_batch: [batch_size, seq_len] tensor
    """
    batch_size, seq_len = theta_batch.shape
    
    # Create difficulty tensor
    b_tensor = torch.zeros_like(skill_batch, dtype=torch.float32)
    for skill_id, diff in b_dict.items():
        b_tensor[skill_batch == skill_id] = diff
    
    # Compute ICC: σ(θ - b)
    logits = theta_batch - b_tensor
    icc_batch = torch.sigmoid(logits)
    
    # TCC: cumulative sum
    tcc_batch = torch.cumsum(icc_batch, dim=1)
    
    return tcc_batch, icc_batch


def evaluate_tcc_fit(tcc_expected, tcc_observed):
    """
    Evaluate how well expected TCC matches observed.
    
    Args:
        tcc_expected: Expected cumulative scores
        tcc_observed: Observed cumulative scores
    Returns:
        metrics: Dictionary with MAE, RMSE, R²
    """
    mae = np.mean(np.abs(tcc_expected - tcc_observed))
    rmse = np.sqrt(np.mean((tcc_expected - tcc_observed) ** 2))
    
    # R-squared
    ss_total = np.sum((tcc_observed - np.mean(tcc_observed)) ** 2)
    ss_residual = np.sum((tcc_observed - tcc_expected) ** 2)
    r2 = 1 - (ss_residual / (ss_total + 1e-8))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
```

#### Task 2.3: Create Visualization Module (2 hours)

**File**: `tmp/rasch_visualization.py`

**Purpose**: Plot ICCs, TCCs, and ability trajectories

**Implementation**: (See Section 13 for plot functions, consolidate into this module)

#### Task 2.4: Integration Tests (1 hour)

**File**: `tmp/test_rasch_tools.py`

**Purpose**: Verify all custom tools work end-to-end

**Run**: Test on sample students from assist2015

**Deliverables (Phase 2)**:
- ✅ `tmp/rasch_dynamic_ability.py` - Dynamic θ(t) estimation
- ✅ `tmp/rasch_tcc_computation.py` - TCC calculation
- ✅ `tmp/rasch_visualization.py` - Plotting tools
- ✅ Verified integration tests pass

---

### Phase 3: iKT-Rasch Integration (Day 3)

**Objective**: Integrate Rasch layer into iKT architecture

**Duration**: ~7-8 hours

#### Task 3.1: Create Rasch Layer Module (2 hours)

**File**: `pykt/models/rasch_layer.py`

**Purpose**: PyTorch module for Rasch-constrained predictions

**Implementation**: (Use specification from Section 11)

#### Task 3.2: Modify iKT Model (2 hours)

**File**: `pykt/models/gainakt4_rasch.py` (copy from gainakt4.py)

**Changes**:
1. Add RaschLayer to architecture
2. Replace Encoder 2 output head with Rasch head
3. Initialize difficulty bank with py-irt baseline parameters
4. Add L3_rasch loss computation

#### Task 3.3: Update Training Script (2 hours)

**File**: `examples/train_gainakt4_rasch.py`

**Changes**:
1. Load rasch_baseline_*.json parameters
2. Pass difficulty initialization to model
3. Update loss computation for L3_rasch
4. Add TCC metrics to logging

#### Task 3.4: Update Evaluation Script (1 hour)

**File**: `examples/eval_gainakt4_rasch.py`

**Changes**:
1. Compute and save TCC predictions
2. Compare with empirical TCCs
3. Correlation analysis between learned and empirical parameters

#### Task 3.5: Initial Training Run (1 hour)

**Test Training**:
```bash
cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate

SKIP_PARAMETER_AUDIT=1 python examples/train_gainakt4_rasch.py \
    --dataset assist2015 \
    --epochs 3 \
    --use_gpu 0,1,2,3,4
```

**Verify**:
- Gradient flow works
- Loss decreases
- θ trajectories show learning
- b parameters don't collapse

**Deliverables (Phase 3)**:
- ✅ `pykt/models/rasch_layer.py` - Rasch layer module
- ✅ `pykt/models/gainakt4_rasch.py` - Modified architecture
- ✅ `examples/train_gainakt4_rasch.py` - Training script
- ✅ `examples/eval_gainakt4_rasch.py` - Evaluation script
- ✅ Initial training run successful

---

### Success Criteria

**Phase 1 Complete When**:
- ✅ py-irt installed and working
- ✅ Both datasets converted to IRT format
- ✅ Rasch calibration completes without errors
- ✅ Validation tests show reasonable parameters (correlations > 0.7, AUC > 0.65)

**Phase 2 Complete When**:
- ✅ Dynamic ability functions tested on sample sequences
- ✅ TCC computation matches hand calculations
- ✅ Visualizations render correctly
- ✅ Integration tests pass

**Phase 3 Complete When**:
- ✅ Rasch layer accepts ability input and skill IDs
- ✅ iKT-Rasch model initializes with empirical difficulties
- ✅ Training runs for 3 epochs without errors
- ✅ Losses (L1, L2, L3) all decrease
- ✅ Learned parameters show reasonable ranges

**Final Validation** (After Phase 3):
- Compare learned b_j vs empirical b_j (correlation > 0.5)
- Compare learned θ_t vs empirical θ_i (qualitative agreement)
- Test AUC ≥ baseline iKT (no performance regression)
- TCC visualizations show interpretable learning curves

---

### Appendix A: Inverse Rasch Function - From Probability to Ability

**Question**: Given a target probability `y` (e.g., mastery threshold), what ability `x` (θ) is required?

#### Mathematical Solution

**Forward Rasch Model**:
```
P(correct | θ, b) = 1 / (1 + exp(-(θ - b)))
```

Given: P = y (target probability), b (difficulty)
Find: θ (required ability)

**Derivation**:
```
y = 1 / (1 + exp(-(θ - b)))

# Rearrange
1 + exp(-(θ - b)) = 1/y

exp(-(θ - b)) = (1/y) - 1 = (1 - y)/y

# Take natural log of both sides
-(θ - b) = ln((1 - y)/y)

θ - b = -ln((1 - y)/y)

θ = b - ln((1 - y)/y)

# Using logit function
θ = b + logit(y)

where logit(y) = ln(y / (1-y))
```

**Inverse Rasch Formula**:
```python
θ = b + ln(y / (1 - y))
```

Or equivalently:
```python
θ = b + logit(y)
```

#### Implementation

**Python Function**:
```python
import numpy as np

def inverse_rasch(probability, difficulty):
    """
    Calculate required ability to achieve target probability.
    
    Args:
        probability: Target success probability (0 < p < 1)
        difficulty: Item difficulty (b parameter)
    Returns:
        ability: Required ability (θ) to achieve target probability
    
    Example:
        # To have 50% chance on skill with b=1.0
        theta = inverse_rasch(0.5, 1.0)  # Returns 1.0
        
        # To have 80% chance on skill with b=1.0
        theta = inverse_rasch(0.8, 1.0)  # Returns ~2.39
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be in (0, 1)")
    
    # θ = b + logit(p)
    logit_p = np.log(probability / (1 - probability))
    theta = difficulty + logit_p
    
    return theta


def logit(p):
    """Logit function: ln(p / (1-p))"""
    return np.log(p / (1 - p))


def sigmoid(x):
    """Sigmoid function: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))
```

#### Calculating Mastery Thresholds from ASSISTments Data

**Objective**: Determine what ability level constitutes "mastery" for each skill

**Approach Options**:

**Option 1: Fixed Probability Threshold (Conventional)**

Use a standard mastery threshold (e.g., 80% success probability):

```python
import json

# Load Rasch baseline parameters
with open('tmp/rasch_baseline_assist2015.json', 'r') as f:
    calib = json.load(f)

b_dict = {int(k): v for k, v in calib['b'].items()}

# Define mastery threshold
MASTERY_PROB = 0.80  # 80% success probability

# Calculate mastery ability for each skill
mastery_levels = {}
for skill_id, difficulty in b_dict.items():
    theta_mastery = inverse_rasch(MASTERY_PROB, difficulty)
    mastery_levels[skill_id] = theta_mastery

# Example output
print(f"Skill 10 (b={b_dict[10]:.2f}): Mastery at θ = {mastery_levels[10]:.2f}")
print(f"Skill 25 (b={b_dict[25]:.2f}): Mastery at θ = {mastery_levels[25]:.2f}")
```

**Interpretation**:
- **Easy skill** (b = -1.0): Mastery at θ = -1.0 + logit(0.8) ≈ 0.39
- **Hard skill** (b = 1.5): Mastery at θ = 1.5 + logit(0.8) ≈ 2.89

**Key Insight**: Harder skills require higher ability to reach same mastery probability.

**Option 2: Empirical Mastery Threshold (Data-Driven)**

Calculate mastery threshold from observed student performance:

**Data Source**: The `y` probability value comes from the `correct` column in ASSISTments datasets.

**ASSISTments Dataset Columns** (after pykt preprocessing):
- `qseqs`: Skill ID sequences (which skill was practiced)
- `cseqs`: Correctness sequences (0=incorrect, 1=correct) ← **This is y**
- `uids`: Student IDs

**How to Calculate y (Mastery Probability)**:

For each skill, calculate the **success rate** (proportion of correct responses):
```
y_skill = (number of correct attempts) / (total attempts)
```

**Important Distinction**:

❌ **y_skill alone is NOT sufficient to infer ability (θ)**

The Rasch model requires **both** parameters:
```
P(correct) = σ(θ - b)

Therefore: θ = b + logit(P)
```

**To infer ability, you need**:
1. **y_skill** (observed success rate on skill) ← From `correct` column
2. **b_skill** (difficulty of skill) ← From Rasch calibration (py-irt)

**Then**: `θ = b_skill + logit(y_skill)`

**Example**:
```python
# Student has 75% success rate on Skill 10
y_skill = 0.75  # From data: mean of 'correct' column

# Skill 10 has difficulty from Rasch calibration
b_skill = 0.5   # From rasch_baseline_assist2015.json

# Infer student's ability
import numpy as np
theta = b_skill + np.log(y_skill / (1 - y_skill))
# theta = 0.5 + np.log(0.75/0.25) = 0.5 + 1.099 = 1.599
```

**Why you need both**:
- **High y_skill** could mean: high ability OR easy skill
- **Low y_skill** could mean: low ability OR hard skill
- **Only with b_skill** can you disentangle ability from difficulty

**Aggregation Options**:
1. **Per-skill globally**: All students' performance on skill k
2. **Per-student per-skill**: Individual student's success rate on skill k
3. **Top performers**: Success rate of students at a given percentile

```python
import pickle
import pandas as pd
import numpy as np

def calculate_empirical_mastery_thresholds(dataset_name, percentile=75):
    """
    Calculate empirical mastery thresholds per skill.
    
    Strategy: Define mastery as achieving performance at top percentile
    of students who attempted that skill.
    
    Args:
        dataset_name: 'assist2015' or 'assist2009'
        percentile: Percentile to use for mastery (e.g., 75 = top quartile)
    Returns:
        mastery_dict: {skill_id: (probability, required_theta)}
    """
    # Load data (converted format with user_id, item_id, correct columns)
    data_path = f'tmp/{dataset_name}_irt_full.pkl'
    df = pd.read_pickle(data_path)
    
    # Columns in df:
    # - user_id: student identifier
    # - item_id: skill identifier  
    # - correct: 0 or 1 (this is our y value for each attempt)
    # - position: temporal position in student's sequence
    
    # Load Rasch parameters
    calib_path = f'tmp/rasch_baseline_{dataset_name}.json'
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    theta_dict = {k: v for k, v in calib['theta'].items()}
    b_dict = {int(k): v for k, v in calib['b'].items()}
    
    # Calculate per-student success rate on each skill
    df['theta'] = df['user_id'].map(theta_dict)
    
    mastery_dict = {}
    
    for skill_id in df['item_id'].unique():
        skill_data = df[df['item_id'] == skill_id].copy()
        
        # Calculate success rate per student on this skill
        student_perf = skill_data.groupby('user_id')['correct'].mean()
        
        # Define mastery probability as percentile threshold
        mastery_prob = np.percentile(student_perf, percentile)
        mastery_prob = np.clip(mastery_prob, 0.01, 0.99)  # Avoid 0/1
        
        # Calculate required ability
        b = b_dict.get(int(skill_id), 0.0)
        theta_mastery = inverse_rasch(mastery_prob, b)
        
        mastery_dict[int(skill_id)] = {
            'mastery_probability': mastery_prob,
            'difficulty': b,
            'required_ability': theta_mastery,
            'num_students': len(student_perf)
        }
    
    return mastery_dict


# Usage
mastery = calculate_empirical_mastery_thresholds('assist2015', percentile=75)

# Save results
with open('tmp/mastery_thresholds_assist2015.json', 'w') as f:
    json.dump(mastery, f, indent=2)

# Print sample
for skill_id in list(mastery.keys())[:5]:
    info = mastery[skill_id]
    print(f"Skill {skill_id}:")
    print(f"  Difficulty (b): {info['difficulty']:.3f}")
    print(f"  Mastery threshold: {info['mastery_probability']:.1%} success")
    print(f"  Required ability (θ): {info['required_ability']:.3f}")
    print()
```

**Option 3: Adaptive Mastery Threshold (Performance-Based)**

Different threshold for different skills based on their importance/difficulty:

```python
def calculate_adaptive_mastery_thresholds(b_dict, base_prob=0.80, 
                                          scale_factor=0.1):
    """
    Adaptive mastery: easier skills require higher mastery probability.
    
    Logic: Easy skills should be fully mastered (90%+), 
           hard skills can be considered mastered at lower levels (70%+).
    
    Args:
        b_dict: {skill_id: difficulty}
        base_prob: Base mastery probability (at b=0)
        scale_factor: How much to adjust per unit difficulty
    Returns:
        mastery_dict: {skill_id: (probability, required_theta)}
    """
    mastery_dict = {}
    
    for skill_id, difficulty in b_dict.items():
        # Adjust mastery probability based on difficulty
        # Easier skills (b < 0): higher probability required
        # Harder skills (b > 0): lower probability acceptable
        mastery_prob = base_prob - scale_factor * difficulty
        mastery_prob = np.clip(mastery_prob, 0.6, 0.95)  # Keep in reasonable range
        
        # Calculate required ability
        theta_mastery = inverse_rasch(mastery_prob, difficulty)
        
        mastery_dict[skill_id] = {
            'mastery_probability': mastery_prob,
            'difficulty': difficulty,
            'required_ability': theta_mastery
        }
    
    return mastery_dict


# Example usage
adaptive_mastery = calculate_adaptive_mastery_thresholds(b_dict)

# Results:
# Easy skill (b=-1.0): Requires 90% success → θ ≈ 1.20
# Medium skill (b=0.0): Requires 80% success → θ ≈ 1.39  
# Hard skill (b=1.5): Requires 65% success → θ ≈ 2.12
```

#### Practical Application in iKT-Rasch

**Use Case 1: Mastery Classification (Head 2)**

```python
# In iKT-Rasch model
class MasteryHead(nn.Module):
    def __init__(self, num_skills):
        super().__init__()
        # Learnable mastery thresholds per skill
        self.mastery_threshold = nn.Parameter(torch.ones(num_skills) * 1.386)
        # 1.386 = logit(0.8) for 80% mastery
        
    def forward(self, theta, skill_ids, difficulty_bank):
        """
        Args:
            theta: [batch, seq_len, 1] - estimated ability
            skill_ids: [batch, seq_len] - skill IDs
            difficulty_bank: Embedding(num_skills, 1) - difficulties
        Returns:
            mastery_probs: [batch, seq_len] - P(mastered)
        """
        b = difficulty_bank(skill_ids)  # [batch, seq_len, 1]
        threshold = self.mastery_threshold[skill_ids]  # [batch, seq_len]
        
        # Required ability for mastery: θ_mastery = b + threshold
        theta_mastery = b.squeeze(-1) + threshold  # [batch, seq_len]
        
        # Is current ability above mastery threshold?
        mastery_probs = torch.sigmoid(theta.squeeze(-1) - theta_mastery)
        
        return mastery_probs
```

**Use Case 2: Progress Tracking**

```python
def track_mastery_progress(student_data, b_dict, mastery_thresholds):
    """
    Track student's progress toward mastery on each skill.
    
    Args:
        student_data: DataFrame with columns ['skill_id', 'correct', 'position']
        b_dict: {skill_id: difficulty}
        mastery_thresholds: {skill_id: theta_mastery}
    Returns:
        progress_df: DataFrame with mastery progress per skill
    """
    from rasch_dynamic_ability import estimate_ability_surprise
    
    # Estimate student ability trajectory
    responses = student_data['correct'].values
    skill_ids = student_data['skill_id'].values
    theta_trajectory = estimate_ability_surprise(responses, skill_ids, b_dict)
    
    # Track mastery status per skill
    progress = []
    for skill_id in student_data['skill_id'].unique():
        skill_mask = student_data['skill_id'] == skill_id
        skill_positions = student_data[skill_mask].index
        
        theta_mastery = mastery_thresholds[skill_id]
        
        # Find when student achieved mastery (if at all)
        for pos in skill_positions:
            theta_current = theta_trajectory[pos]
            mastered = (theta_current >= theta_mastery)
            
            progress.append({
                'skill_id': skill_id,
                'position': pos,
                'theta_current': theta_current,
                'theta_mastery': theta_mastery,
                'gap': theta_mastery - theta_current,
                'mastered': mastered
            })
    
    return pd.DataFrame(progress)
```

#### Visualization: Mastery Curves

```python
import matplotlib.pyplot as plt

def plot_mastery_curves(b_dict, mastery_prob=0.80):
    """
    Visualize mastery thresholds across skills.
    
    Shows how required ability varies with skill difficulty.
    """
    skills = sorted(b_dict.keys())
    difficulties = [b_dict[s] for s in skills]
    mastery_abilities = [inverse_rasch(mastery_prob, b) for b in difficulties]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Difficulty vs Mastery Ability
    axes[0].scatter(difficulties, mastery_abilities, alpha=0.6)
    axes[0].plot(difficulties, difficulties, 'r--', 
                 label='θ = b (50% probability)')
    axes[0].set_xlabel('Skill Difficulty (b)', fontsize=12)
    axes[0].set_ylabel('Required Ability for Mastery (θ)', fontsize=12)
    axes[0].set_title(f'Mastery Threshold ({mastery_prob:.0%} success)', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Right: Distribution of Mastery Gaps
    gaps = [m - d for m, d in zip(mastery_abilities, difficulties)]
    axes[1].hist(gaps, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(gaps), color='red', linestyle='--',
                   label=f'Mean gap: {np.mean(gaps):.2f}')
    axes[1].set_xlabel('Mastery Gap (θ_mastery - b)', fontsize=12)
    axes[1].set_ylabel('Number of Skills', fontsize=12)
    axes[1].set_title('Distribution of Ability Gaps', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# Usage
fig = plot_mastery_curves(b_dict, mastery_prob=0.80)
plt.savefig('tmp/mastery_thresholds.png', dpi=150)
plt.close()
```

#### Complete Script: Calculate Mastery Thresholds

**File**: `tmp/calculate_mastery_thresholds.py`

```python
#!/usr/bin/env python
"""
Calculate mastery thresholds for ASSISTments datasets.

Usage:
    python tmp/calculate_mastery_thresholds.py --dataset assist2015 --method fixed
    python tmp/calculate_mastery_thresholds.py --dataset assist2015 --method empirical
    python tmp/calculate_mastery_thresholds.py --dataset assist2015 --method adaptive
"""

import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def inverse_rasch(probability, difficulty):
    """Calculate required ability for target probability."""
    logit_p = np.log(probability / (1 - probability))
    return difficulty + logit_p

def calculate_fixed_thresholds(b_dict, mastery_prob=0.80):
    """Fixed probability threshold (e.g., 80% for all skills)."""
    return {
        skill_id: {
            'mastery_probability': mastery_prob,
            'difficulty': difficulty,
            'required_ability': inverse_rasch(mastery_prob, difficulty),
            'method': 'fixed'
        }
        for skill_id, difficulty in b_dict.items()
    }

def calculate_empirical_thresholds(dataset_name, percentile=75):
    """Data-driven thresholds from empirical performance."""
    # Load data
    df = pd.read_pickle(f'tmp/{dataset_name}_irt_full.pkl')
    
    with open(f'tmp/rasch_baseline_{dataset_name}.json', 'r') as f:
        calib = json.load(f)
    
    theta_dict = calib['theta']
    b_dict = {int(k): v for k, v in calib['b'].items()}
    
    df['theta'] = df['user_id'].map(theta_dict)
    
    mastery_dict = {}
    for skill_id in df['item_id'].unique():
        skill_data = df[df['item_id'] == skill_id]
        student_perf = skill_data.groupby('user_id')['correct'].mean()
        
        mastery_prob = np.percentile(student_perf, percentile)
        mastery_prob = np.clip(mastery_prob, 0.01, 0.99)
        
        b = b_dict.get(int(skill_id), 0.0)
        
        mastery_dict[int(skill_id)] = {
            'mastery_probability': mastery_prob,
            'difficulty': b,
            'required_ability': inverse_rasch(mastery_prob, b),
            'method': 'empirical',
            'percentile': percentile
        }
    
    return mastery_dict

def calculate_adaptive_thresholds(b_dict, base_prob=0.80, scale_factor=0.1):
    """Adaptive thresholds: easier skills require higher mastery."""
    mastery_dict = {}
    for skill_id, difficulty in b_dict.items():
        mastery_prob = base_prob - scale_factor * difficulty
        mastery_prob = np.clip(mastery_prob, 0.6, 0.95)
        
        mastery_dict[skill_id] = {
            'mastery_probability': mastery_prob,
            'difficulty': difficulty,
            'required_ability': inverse_rasch(mastery_prob, difficulty),
            'method': 'adaptive',
            'base_prob': base_prob,
            'scale_factor': scale_factor
        }
    
    return mastery_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='assist2015')
    parser.add_argument('--method', default='fixed', 
                       choices=['fixed', 'empirical', 'adaptive'])
    parser.add_argument('--mastery_prob', type=float, default=0.80)
    parser.add_argument('--percentile', type=int, default=75)
    args = parser.parse_args()
    
    # Load Rasch baseline
    with open(f'tmp/rasch_baseline_{args.dataset}.json', 'r') as f:
        calib = json.load(f)
    
    b_dict = {int(k): v for k, v in calib['b'].items()}
    
    # Calculate mastery thresholds
    if args.method == 'fixed':
        mastery = calculate_fixed_thresholds(b_dict, args.mastery_prob)
    elif args.method == 'empirical':
        mastery = calculate_empirical_thresholds(args.dataset, args.percentile)
    elif args.method == 'adaptive':
        mastery = calculate_adaptive_thresholds(b_dict)
    
    # Save results
    output_file = f'tmp/mastery_thresholds_{args.dataset}_{args.method}.json'
    with open(output_file, 'w') as f:
        json.dump(mastery, f, indent=2)
    
    print(f"✓ Calculated mastery thresholds using '{args.method}' method")
    print(f"✓ Saved to {output_file}")
    
    # Print summary
    abilities = [m['required_ability'] for m in mastery.values()]
    probs = [m['mastery_probability'] for m in mastery.values()]
    
    print(f"\nSummary Statistics:")
    print(f"  Mastery probability: {np.mean(probs):.3f} ± {np.std(probs):.3f}")
    print(f"  Required ability: {np.mean(abilities):.3f} ± {np.std(abilities):.3f}")
    print(f"  Range: [{np.min(abilities):.2f}, {np.max(abilities):.2f}]")

if __name__ == '__main__':
    main()
```

**Summary**: 

The inverse Rasch function `θ = b + logit(y)` allows you to calculate the required ability level to achieve a target probability `y`. For ASSISTments datasets, you can:

1. **Fixed threshold** (e.g., 80% for all skills)
2. **Empirical threshold** (data-driven from top performers)
3. **Adaptive threshold** (varies by difficulty)

This enables mastery classification and progress tracking in the iKT-Rasch model.

---

### Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Empirical Baseline | 3-4 hours | rasch_baseline_*.json files, validation report |
| Phase 2: Custom TCC Tools | 6-7 hours | dynamic_ability, tcc_computation, visualization modules |
| Phase 3: Neural Integration | 7-8 hours | iKT-Rasch model, training/eval scripts, test run |
| **Total** | **16-19 hours** | **Complete Rasch-based KT system** |

Spread over 3 days with buffer time: ~6-7 hours per day.

---

### Next Steps After Implementation

1. **Full Training Run**: Train iKT-Rasch for 200 epochs on assist2015
2. **Ablation Studies**: Compare λ_curve values (0.0, 0.1, 0.2, 0.3)
3. **Parameter Analysis**: Deep dive into learned θ and b distributions
4. **Educational Validation**: Consult domain experts on skill difficulty rankings
5. **Paper Writing**: Document approach, results, and educational insights
