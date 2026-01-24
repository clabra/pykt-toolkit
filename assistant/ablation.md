# GTransformer Ablation Strategy

This document outlines the ablation strategy for the GTransformer model, specifically focusing on how the system reverts to a baseline state and how individual Neuro-Symbolic features are toggled.


## Quick Reference Table

| Ablation Mode | `ablation` | `active_grounding` | `lambda_sup` | `lambda_ref` | `lambda_probe` | `lambda_initmastery` | `lambda_rate` | `personalization` | Components Active | Expected AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Neural (baseline)** | `"all"` | 0 | 1.0 | 0 | 0 | 0 | 0 | false | Transformer + Supervised head only | 0.7825 |
| **Full Grounding** | `"none"` | 1 | *as-is* | *as-is* | *as-is* | 0 | 0 | false | All (axes, bases, probes, BKT) | 0.7790-0.7812 |
| **No Reference Pipeline** | `"reference"` | 0 | 1.0 | 0 | 0 | 0 | 0 | false | Transformer + Supervised head (semantic axes unused) | ~0.7825 |
| **No Probe Loss** | `"probe"` | 1 | *as-is* | *as-is* | 0 | 0 | 0 | false | All except probe training | ~0.7800-0.7810 |
| **No Personalization** | `"personalization"` | 1 | *as-is* | *as-is* | *as-is* | 0 | 0 | false | All except student-specific params | ~0.7780-0.7800 |

**Notes**:
- **Parameter Precedence**: For ablation-related parameters (lambda_*, personalization, active_grounding), the precedence is: **ablation mode > command-line > config files**
- **`ablation='none'`**: Special mode that uses lambda_* and personalization parameters *as-is* from command-line or config files, but **enforces `active_grounding=1`** (required for BKT reference evaluation)
- **Other ablation modes**: Override all parameters shown in the table regardless of command-line or config file values
- **`lambda_sup`**: Always 1.0 (supervised loss weight)
- **`lambda_ref`**: Weight for BKT reference prediction loss (0 = disabled)
- **`lambda_initmastery`**: Weight for L0 parameter grounding loss (0 = disabled)
- **`lambda_rate`**: Weight for T parameter grounding loss (0 = disabled)
- **`lambda_probe`**: Weight for probe alignment loss (0 = disabled)
- **`personalization`**: Boolean flag to enable/disable student-specific parameters (true = enabled, false = disabled)
- **`active_grounding`**: Enables BKT target loading and p_ref evaluation (0 = disabled, 1 = enabled)
  - Set to `0` when `ablation="all"` or `ablation="reference"` (no grounding components)
  - Set to `1` when probe heads are trained (enables p_ref metrics)
- Expected AUC values are estimates based on ASSIST2009 5-fold CV; actual performance may vary 



## 1. The Baseline Codebase (`--ablation all`)

The **Ablation Baseline** refers to the state of the model where it is functionally equivalent to a standard Context-Aware Attentive Knowledge Tracing (**baseline**) architecture. This serves as the predictive performance floor.

When `ablation` is set to `"all"`, the following deactivations occur:

*   **Architecture Components (Skip Initialization)**:
    *   **Grounded Embeddings** (`self.ablation != "all"` check at line 68):
        - Semantic axes (`knowledge_axis_emb`, `velocity_axis_emb`) are NOT created
        - BKT base embeddings (`l0_base_emb`, `t_base_emb`) are NOT created
        - Population parameters (`bkt_guess`, `bkt_slip`, `bkt_l0_pop`, `bkt_t_pop`) are NOT registered
        - Student-specific parameters (`student_param`, `student_gap_param`) are NOT created
        - Probe heads (`probe_l0`, `probe_t`) are NOT created
    *   Only standard baseline components remain: question embeddings, transformer blocks, supervised output head

*   **Forward Pass (Early Exit at line 260)**:
    *   Model computes only supervised predictions: `preds = sigmoid(out(z_context))`
    *   Returns minimal outputs: `{'predictions': preds}`
    *   Skips all grounded parameter computation (p_L0, p_T via semantic axes)
    *   Skips probe predictions (p_l0_probe, p_t_probe)
    *   Skips BKT reference output (no differentiable BKT walk)
    *   Skips diversity loss computation

*   **Loss Function Simplification**:
    *   Training uses only standard **Supervised Loss** (BCE on predictions)
    *   Multi-component grounding losses are excluded:
        - λ_ref (reference BKT predictions)
        - λ_initmastery (L0 parameter grounding)
        - λ_rate (T parameter grounding)
        - λ_probe (probe alignment)

## 2. Full Grounding Mode (`--ablation none`)

When `ablation` is set to `"none"` (or any value != `"all"`), the model operates in **full theory-guided mode**:

### Architecture Components (All Enabled)

1. **Semantic Axes** (Step 3: Textured Grounding):
   - `knowledge_axis_emb`: Directions in latent space for "More Knowledgeable" (z_dim per skill)
   - `velocity_axis_emb`: Directions in latent space for "Faster Learner" (z_dim per skill)
   - Initialized with normal distribution (mean=0, std=0.02)

2. **BKT Bases** (Grounding Points):
   - `l0_base_emb`: Initial mastery base logits per skill (scalar)
   - `t_base_emb`: Learning rate base logits per skill (scalar)
   - Initialized from pre-computed BKT parameters (`bkt_skill_params.pkl`)
   - Uses "textured grounding": N(logit, 0.05) to survive LayerNorm

3. **Population Parameters** (Fixed Buffers):
   - `bkt_guess`: Guess rates per skill (default 0.2)
   - `bkt_slip`: Slip rates per skill (default 0.1)
   - `bkt_l0_pop`: Population-level initial mastery (from BKT)
   - `bkt_t_pop`: Population-level learning rate (from BKT)
   - Used in BKT reference output computation

4. **Student Individualization** (if personalization=true):
   - `student_param`: Learning velocity scalar per student
   - `student_gap_param`: Knowledge gap scalar per student
   - Adds personalized bias to dynamic parameter estimates

5. **Probe Heads** (Active Grounding):
   - `probe_l0`: Linear(z_dim, 1) - extracts L0 from context
   - `probe_t`: Linear(z_dim, 1) - extracts T from context
   - Enforces global linear interpretability

### Forward Pass (Full Pipeline)

1. **Standard Processing**: Transformer encoding → z_context
2. **Supervised Prediction**: Standard baseline output head
3. **Grounded Parameters** (lines 267-328):
   ```python
   # Get concept-specific axes and bases
   k_axis = knowledge_axis_emb(q_data)  # [BS, seqlen, z_dim]
   v_axis = velocity_axis_emb(q_data)   # [BS, seqlen, z_dim]
   l0_base = l0_base_emb(q_data)        # [BS, seqlen]
   t_base = t_base_emb(q_data)          # [BS, seqlen]
   
   # Projection: Base + (z · Axis)
   l0_logits = l0_base + (z_context * k_axis).sum(-1)
   t_logits = t_base + (z_context * v_axis).sum(-1)
   
   # Add student bias if personalized
   if personalization:
       l0_logits += student_gap_param(uid_data)
       t_logits += student_param(uid_data)
   
   p_l0 = sigmoid(l0_logits)
   p_t = sigmoid(t_logits)
   ```

4. **Probe Predictions** (lines 333-338):
   ```python
   probe_l0_logits = probe_l0(z_context)
   probe_t_logits = probe_t(z_context)
   p_l0_probe = sigmoid(probe_l0_logits)
   p_t_probe = sigmoid(probe_t_logits)
   ```

5. **BKT Reference Output** (lines 344+):
   - Uses p_l0, p_t with fixed guess/slip rates
   - Implements differentiable BKT walk through history
   - Produces interpretable predictions via classical BKT logic

6. **Diversity Loss** (lines 284-313):
   - Penalizes high cosine similarity between semantic axes of different skills
   - Encourages orthogonality: prevents axis collapse
   - Weight: 0.1 * (k_diversity_loss + v_diversity_loss)

### Output Dictionary (Full Mode)

```python
outputs = {
    'predictions': preds,           # Supervised head (neural)
    'p_l0': p_l0,                  # Grounded initial mastery
    'p_t': p_t,                    # Grounded learning rate
    'p_l0_probe': p_l0_probe,      # Probe-extracted L0
    'p_t_probe': p_t_probe,        # Probe-extracted T
    'reference_preds': ref_preds   # BKT logic predictions
}
```

## 3. Parameter Control Summary

| Scenario | Parameter Setting | Components Active | Loss Terms | Expected Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Full Model** | `ablation="none"` | All (axes, bases, probes, BKT) | λ_sup + λ_ref + λ_init + λ_rate + λ_probe + diversity | Best interpretability (0.7790-0.7812 AUC on AS2009) |
| **All Features Ablated** | `ablation="all"` | Only standard baseline components | λ_sup only | baseline parity (0.7825 AUC on AS2009) |

**Note**: The original document mentioned `ablation="regularization"` mode, but this is **not implemented** in the current codebase. Only `"all"` and `"none"` (or any other value) are supported.

## 4. Implementation Details

### Initialization Logic (pykt/models/gtransformer.py, lines 18-120)

```python
class GTransformer(nn.Module):
    def __init__(self, ...):
        self.ablation = ablation  # From config
        
        # Standard components (always created)
        self.q_embed = nn.Embedding(...)
        self.model = Architecture(...)  # Transformer blocks
        self.out = nn.Sequential(...)   # Supervised head
        
        # Grounded components (only if ablation != "all")
        if self.ablation != "all":
            # Semantic axes
            self.knowledge_axis_emb = nn.Embedding(n_question+1, z_dim)
            self.velocity_axis_emb = nn.Embedding(n_question+1, z_dim)
            
            # BKT bases
            self.l0_base_emb = nn.Embedding(n_question+1, 1)
            self.t_base_emb = nn.Embedding(n_question+1, 1)
            
            # Population parameters (buffers)
            self.register_buffer('bkt_guess', torch.ones(n_question+1) * 0.2)
            self.register_buffer('bkt_slip', torch.ones(n_question+1) * 0.1)
            self.register_buffer('bkt_l0_pop', torch.ones(n_question+1) * 0.5)
            self.register_buffer('bkt_t_pop', torch.ones(n_question+1) * 0.1)
            
            # Student parameters (if personalized)
            if self.personalization:
                self.student_param = nn.Embedding(n_uid+1, 1)
                self.student_gap_param = nn.Embedding(n_uid+1, 1)
            
            # Probe heads
            self.probe_l0 = nn.Linear(z_dim, 1)
            self.probe_t = nn.Linear(z_dim, 1)
```

### Theory Parameter Loading (examples/wandb_gtransformer_train.py, lines 123-137)

```python
# Load BKT parameters only if grounding is enabled
if model.ablation != "all":
    bkt_path = os.path.join(dpath, "bkt_skill_params.pkl")
    if os.path.exists(bkt_path):
        with open(bkt_path, "rb") as f:
            bkt_params = pickle.load(f)
        model.load_theory_params(bkt_params)  # Initialize l0_base_emb, t_base_emb
```

### Forward Pass Logic (pykt/models/gtransformer.py, lines 250-366)

```python
def forward(self, c_data, r_data, ...):
    # Standard processing
    z_context = concat([transformer_output, q_embed])
    preds = sigmoid(out(z_context))
    
    # Early exit for ablation mode
    if self.ablation == "all":
        return {'predictions': preds}, reg_loss
    
    # Grounded parameter computation
    k_axis = self.knowledge_axis_emb(q_data)
    v_axis = self.velocity_axis_emb(q_data)
    l0_base = self.l0_base_emb(q_data)
    t_base = self.t_base_emb(q_data)
    
    l0_logits = l0_base + (z_context * k_axis).sum(-1)
    t_logits = t_base + (z_context * v_axis).sum(-1)
    
    # Personalization
    if self.personalization:
        l0_logits += self.student_gap_param(uid_data)
        t_logits += self.student_param(uid_data)
    
    p_l0 = sigmoid(l0_logits)
    p_t = sigmoid(t_logits)
    
    # Probe predictions
    p_l0_probe = sigmoid(self.probe_l0(z_context))
    p_t_probe = sigmoid(self.probe_t(z_context))
    
    # BKT reference output
    ref_preds = self._bkt_ref_output(q_data, target, p_l0, p_t)
    
    # Diversity loss
    diversity_loss = compute_diversity_loss(...)
    
    return {
        'predictions': preds,
        'p_l0': p_l0, 'p_t': p_t,
        'p_l0_probe': p_l0_probe, 'p_t_probe': p_t_probe,
        'reference_preds': ref_preds
    }, reg_loss + diversity_loss
```

## 5. Operational Requirements

1. **Parity Verification**: Any significant change to the GTransformer architecture must be verified by running `--ablation all` on `assist2009`. The resulting Mean AUC must match the baseline benchmark (**0.7825 ± 0.0017**) to ensure the foundation remains sound.

2. **Audit Compliance**: All parameters must be explicitly passed to the training script to satisfy reproducibility audits, including:
   - `--ablation {all|none}`
   - `--lambda_ref`, `--lambda_initmastery`, `--lambda_rate`, `--lambda_probe`
   - Even in ablation mode, these parameters are required (but ignored)

3. **BKT Parameter Files**: When `ablation != "all"`, the model requires:
   - `data/{dataset}/bkt_skill_params.pkl` containing:
     - `params`: Dict mapping skill_id → {prior, learns, guess, slip}
     - `global`: Fallback global parameters
   - Generated by pre-training pyBKT on the dataset

4. **Performance Benchmarks** (5-fold CV on ASSIST2009):
   - `ablation="all"`: 0.7825 ± 0.0017 AUC (pure neural baseline)
   - `ablation="none"` (minimalist): 0.7790 ± 0.0015 AUC, p_ref=0.6756 ± 0.0028
   - `ablation="none"` (with diversity): 0.7812 ± 0.0012 AUC, p_ref=0.6727 ± 0.0002
   - Cost of interpretability: -0.35% to +0.03% (architecture dependent)

## 6. Granular Ablation Modes (Fine-Grained Control)

Beyond the binary `all` (pure neural) vs `none` (full grounding) modes, the GTransformer supports **granular ablation** for systematic component analysis. These modes enable ablation studies to measure the contribution of specific architectural components independently.

### 6.1 Ablation Mode: `"reference"`

**What It Removes**:
1. **Theory Base (Population Priors)** - Section 4 of architecture
   - BKT base embeddings: `l0_base_emb`, `t_base_emb`
   - Population parameter buffers: `bkt_l0_pop`, `bkt_t_pop`
2. **Contextual Projection (Linear Probes)** - Section 4 of architecture
   - Probe heads: `probe_l0`, `probe_t`
3. **Final Parameters (Sigmoid Activation)** - Section 4 of architecture
   - Grounded parameter computation: `p_L0 = σ(logit_base_L0 + ΔL0)`
   - No output of `p_l0`, `p_t` in forward pass
4. **Reference Output (BKT Logic)** - Section 5 of architecture
   - BKT reference predictions: `ŷ_ref = BKT(p_L0, p_T)`
   - No `reference_preds` in output dictionary

**What It Keeps**:
- Semantic axes: `knowledge_axis_emb`, `velocity_axis_emb` (for downstream analysis)
- Student personalization: `student_param`, `student_gap_param`
- Supervised output head: standard baseline predictions

**Required Parameter Settings** (in `configs/parameter_default.json`):
```json
{
  "ablation": "reference",
  "lambda_ref": 0,
  "lambda_initmastery": 0,
  "lambda_rate": 0,
  "lambda_probe": 0,
  "active_grounding": 0
}
```

**Expected Behavior**:
- Model reverts to standard baseline with semantic axis embeddings (unused in forward pass)
- Training uses only supervised loss (BCE on predictions)
- Output: `{'predictions': preds}` (minimal output)
- Performance: Expected to match `ablation="all"` baseline (0.7825 AUC)

**Use Case**:
- Measure contribution of reference prediction pipeline
- Test if BKT logic wrapper provides regularization benefit
- Isolate impact of grounded parameter computation

---

### 6.2 Ablation Mode: `"probe"`

**What It Removes**:
1. **Probe Loss Function** - Section 5 of architecture
   - ℒ_probe term in total loss: `MSE(p̂_L₀, μ_L₀) + MSE(p̂_T, μ_T)`
   - No alignment loss between probe predictions and BKT priors

**What It Keeps**:
- **All architecture components** (semantic axes, BKT bases, probe heads, student params)
- **Probe predictions** (for visualization/analysis): `p_l0_probe`, `p_t_probe`
- **Grounded parameters**: `p_l0`, `p_t` (via semantic axis projection)
- **BKT reference output**: `reference_preds`
- **All other loss terms**: λ_sup, λ_ref, λ_initmastery, λ_rate

**Required Parameter Settings**:
```json
{
  "ablation": "probe",
  "lambda_probe": 0,
  "lambda_sup": 1.0,
  "lambda_ref": 1.0,
  "lambda_initmastery": 10.0,
  "lambda_rate": 10.0,
  "active_grounding": 1
}
```

**Expected Behavior**:
- Model trains with full grounding except probe alignment loss
- Probe heads remain active (for analysis) but untrained
- Output: Full output dictionary with all predictions
- Performance: Expected slight drop from full model (probes provide active grounding signal)

**Use Case**:
- Measure contribution of probe alignment loss vs passive grounding
- Test if semantic axes + BKT bases sufficient without probe supervision
- Analyze probe predictions without training them (diagnostic tool)

---

### 6.3 Ablation Mode: `"personalization"`

**What It Removes**:
1. **Student-Specific Parameters** - Personalization layer
   - Student embeddings: `student_param`, `student_gap_param`
   - Student-level bias terms in grounded parameter computation
   - Controlled by `personalization=false` flag

**What It Keeps**:
- **All grounding components**: semantic axes, BKT bases, probe heads
- **Population-level grounding**: Theory base from BKT priors
- **Contextual projection**: Probe-based parameter extraction
- **BKT reference output**: Uses population-level parameters only
- **All loss terms**: λ_sup, λ_ref, λ_initmastery, λ_rate, λ_probe

**Required Parameter Settings**:
```json
{
  "ablation": "personalization",
  "personalization": false,
  "lambda_sup": 1.0,
  "lambda_ref": 1.0,
  "lambda_initmastery": 10.0,
  "lambda_rate": 10.0,
  "lambda_probe": 1.0,
  "active_grounding": 1
}
```

**Expected Behavior**:
- Model uses population-level grounding only (no student-specific biases)
- Grounded parameters computed from context + semantic axes (no student offset)
- All students share same theoretical base (BKT priors)
- Performance: Expected to drop modestly (personalization provides individualization)

**Use Case**:
- Measure contribution of student-level personalization vs population-level grounding
- Test if theory-guided grounding sufficient without personalization
- Analyze impact of student heterogeneity on performance

---

### 6.4 Configuration Summary Table

| Ablation Mode | Components Removed | Parameters to Set | Expected AUC | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `"all"` | All grounding (full ablation) | active_grounding=0 | 0.7825 | Pure neural baseline (baseline parity) |
| `"none"` | Nothing (full grounding) | All λ > 0, active_grounding=1 | 0.7790-0.7812 | Best interpretability |
| `"reference"` | Theory base, probes, final params, BKT output | λ_ref=0, λ_init=0, λ_rate=0, λ_probe=0, active_grounding=0 | ~0.7825 | Measure reference pipeline contribution |
| `"probe"` | Probe loss only | λ_probe=0, active_grounding=1 | ~0.7800-0.7810 | Measure active grounding contribution |
| `"personalization"` | Student parameters | personalization=false, active_grounding=1 | ~0.7780-0.7800 | Measure personalization contribution |

**Note**: Expected AUC values are estimates based on architecture analysis. Actual performance requires empirical validation through 5-fold CV on ASSIST2009.

---

### 6.5 Implementation Guidance

**Code Changes Required** (in `pykt/models/gtransformer.py`):

1. **Update ablation parsing logic** (line 38):
   ```python
   # Current: self.ablation = ablation (string)
   # New: Parse as set of ablation targets
   if ablation == "all":
       self.ablation_set = {"all"}
   elif ablation == "none":
       self.ablation_set = set()
   else:
       # Parse comma-separated values: "reference,probe" -> {"reference", "probe"}
       self.ablation_set = set(ablation.split(','))
   ```

2. **Update component creation logic** (lines 68-100):
   ```python
   # Current: if self.ablation != "all":
   # New: Fine-grained checks
   
   # Skip ALL grounding if "all" in ablation_set
   if "all" in self.ablation_set:
       # Create only standard baseline components
       pass
   else:
       # Create semantic axes (unless "reference" ablated)
       if "reference" not in self.ablation_set:
           self.knowledge_axis_emb = nn.Embedding(n_question+1, z_dim)
           self.velocity_axis_emb = nn.Embedding(n_question+1, z_dim)
           self.l0_base_emb = nn.Embedding(n_question+1, 1)
           self.t_base_emb = nn.Embedding(n_question+1, 1)
           self.probe_l0 = nn.Linear(z_dim, 1)
           self.probe_t = nn.Linear(z_dim, 1)
       
       # Create student parameters (unless "personalization" ablated)
       if "personalization" not in self.ablation_set and self.personalization:
           self.student_param = nn.Embedding(n_uid+1, 1)
           self.student_gap_param = nn.Embedding(n_uid+1, 1)
   ```

3. **Update forward pass logic** (lines 260-366):
   ```python
   def forward(self, ...):
       # Standard processing
       z_context = concat([transformer_output, q_embed])
       preds = sigmoid(out(z_context))
       
       # Early exit for full ablation
       if "all" in self.ablation_set:
           return {'predictions': preds}, reg_loss
       
       # Conditional grounded parameter computation
       if "reference" not in self.ablation_set:
           # Compute p_l0, p_t via semantic axes
           k_axis = self.knowledge_axis_emb(q_data)
           # ... (existing logic)
           
           # Add student bias if personalization enabled
           if "personalization" not in self.ablation_set and self.personalization:
               l0_logits += self.student_gap_param(uid_data)
               t_logits += self.student_param(uid_data)
           
           # Probe predictions
           p_l0_probe = sigmoid(self.probe_l0(z_context))
           p_t_probe = sigmoid(self.probe_t(z_context))
           
           # BKT reference output
           ref_preds = self._bkt_ref_output(...)
       
       # Build output dictionary
       outputs = {'predictions': preds}
       if "reference" not in self.ablation_set:
           outputs.update({
               'p_l0': p_l0, 'p_t': p_t,
               'p_l0_probe': p_l0_probe, 'p_t_probe': p_t_probe,
               'reference_preds': ref_preds
           })
       
       return outputs, reg_loss + diversity_loss
   ```

4. **Update loss computation** (in training script):
   ```python
   # Conditional loss terms based on ablation_set
   loss = lambda_sup * loss_sup
   
   if "reference" not in model.ablation_set:
       loss += lambda_ref * loss_ref
       loss += lambda_initmastery * loss_l0
       loss += lambda_rate * loss_t
   
   if "probe" not in model.ablation_set:
       loss += lambda_probe * loss_probe
   ```

---

### 6.6 Validation Protocol

To validate granular ablation modes:

1. **Run 5-fold CV on ASSIST2009** for each mode
2. **Compare performance metrics**:
   - AUC (predictive performance)
   - p_ref (reference prediction correlation)
   - Parameter distributions (p_L0, p_T ranges)
3. **Verify component isolation**:
   - `ablation="reference"`: Ensure no BKT output in logs
   - `ablation="probe"`: Verify λ_probe=0 in loss breakdown
   - `ablation="personalization"`: Check all students use same base parameters
4. **Document contribution**:
   - ΔAUC = AUC(full) - AUC(ablation mode)
   - Attribute performance to removed component

**Expected Research Findings**:
- Reference pipeline contribution: ~0-0.5% AUC (regularization effect)
- Probe loss contribution: ~0.5-1.0% AUC (active grounding signal)
- Personalization contribution: ~1.0-2.0% AUC (student heterogeneity)

---

## 7. Ablation Control Center (Parameter Validation)

The **Ablation Control Center** ensures that all ablation modes are applied consistently with clear parameter precedence rules and transparency.

### 7.1 Design Principles

1. **Flexible 'none' mode**: `ablation='none'` uses lambda_* and personalization parameters as configured (command-line > config files > defaults), but **enforces `active_grounding=1`** to enable BKT reference evaluation
2. **Strict override modes**: Other ablation modes (`all`, `reference`, `probe`, `personalization`) override all parameters shown in Quick Reference Table to enforce experimental conditions
3. **Transparency**: All final parameter values are printed before training and documented in `config.json`
4. **Early Feedback**: Parameter values are shown before model initialization begins

**Parameter Precedence (for ablation-related parameters only):**
- **ablation='none'**: command-line > config files > parameter_default.json (no overrides)
- **Other ablation modes**: **ablation mode** (from command-line/config) **> command-line > config files**
  - The ablation mode setting determines which parameters get overridden
  - Once an ablation mode is set, it takes absolute precedence for its controlled parameters
  - Example: `--ablation all --lambda_ref 0.5` → lambda_ref will be set to 0 (ablation mode wins)

### 7.2 Implementation Location

**Function**: `validate_and_apply_ablation_config(model_config, source="config")`

**File**: `pykt/models/init_model.py`

**Rationale**:
- Centralized location already responsible for model initialization
- Has access to both `model_config` and `data_config`
- Called before model creation, enabling early feedback
- Used by all training and evaluation scripts

### 7.3 Quick Reference Table

| Ablation Mode | active_grounding | lambda_sup | lambda_ref | lambda_probe | lambda_initmastery | lambda_rate | personalization | Behavior |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **none** | *as-is* | *as-is* | *as-is* | *as-is* | *as-is* | *as-is* | *as-is* | Use configured values (no overrides) |
| **all** | 0 | 1.0 | 0 | 0 | 0 | 0 | False | Pure Neural (baseline) |
| **reference** | 0 | 1.0 | 0 | 0 | 0 | 0 | False | No Reference Pipeline |
| **probe** | 1 | *as-is* | *as-is* | 0 | 0 | 0 | False | No Probe Loss |
| **personalization** | 1 | *as-is* | *as-is* | *as-is* | 0 | 0 | False | No Personalization |

*Note: "as-is" means parameter uses value from command-line, config file, or parameter_default.json (in precedence order). `active_grounding` is enforced for all modes (no *as-is* option).*

### 7.4 Function Implementation

```python
def validate_and_apply_ablation_config(model_config, source="config"):
    """
    Ablation Control Center: Validates and applies ablation mode parameter settings.
    
    Behavior:
    - ablation='none': Uses parameters as-is from command/config (no overrides)
    - ablation='all', 'reference', 'probe', 'personalization': Overrides specific parameters
    
    Parameter precedence: command-line > config files > ablation defaults
    
    Args:
        model_config (dict): Model configuration dictionary (modified in-place)
        source (str): Source of config ("config" or "command_line") for info messages
    
    Returns:
        dict: Validated and updated model_config
    """
    
    # Ablation mode parameter requirements (from Quick Reference Table)
    ABLATION_CONFIGS = {
        "all": {
            "lambda_sup": 1.0,
            "lambda_ref": 0,
            "lambda_probe": 0,
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "active_grounding": 0,
            "description": "Pure Neural (baseline) - all grounding ablated"
        },
        "none": {
            # Special mode: use lambda_* and personalization as-is, but enforce active_grounding
            "active_grounding": 1,
            "description": "No Ablation - use parameters as configured (enforces active_grounding=1)"
        },
        "reference": {
            "lambda_sup": 1.0,
            "lambda_ref": 0,
            "lambda_probe": 0,
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "active_grounding": 0,
            "description": "No Reference Pipeline - BKT grounding ablated"
        },
        "probe": {
            "lambda_probe": 0,
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "active_grounding": 1,
            "description": "No Probe Loss - probe training ablated"
        },
        "personalization": {
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "active_grounding": 1,
            "description": "No Personalization - student-specific parameters ablated"
        }
    }
    
    # Get ablation mode (default to "none" if not specified)
    ablation = model_config.get("ablation", "none")
    
    # Validate ablation mode
    if ablation not in ABLATION_CONFIGS:
        valid_modes = ", ".join(ABLATION_CONFIGS.keys())
        raise ValueError(
            f"Invalid ablation mode '{ablation}'. "
            f"Valid modes are: {valid_modes}"
        )
    
    # Get required configuration for this ablation mode
    required_config = ABLATION_CONFIGS[ablation]
    ablation_desc = required_config["description"]
    
    # Print header
    print(f"\n{'='*70}")
    print(f"ABLATION CONTROL CENTER")
    print(f"{'='*70}")
    print(f"Ablation mode: '{ablation}' ({ablation_desc})")
    
    # Special handling for 'none' mode - no parameter overrides
    if ablation == "none":
        print("Using parameters as configured (command-line > config files):")
        print()
        # Just display current values, don't override anything
        controlled_params = [
            "lambda_sup", "lambda_ref", "lambda_probe", 
            "lambda_initmastery", "lambda_rate", "personalization", "active_grounding"
        ]
        for param in controlled_params:
            value = model_config.get(param, "not set")
            print(f"  {param:20s} = {value}")
        print(f"{'='*70}\n")
        return model_config
    
    # For other ablation modes: enforce required parameter values
    print("Applying configuration:")
    
    # Check each parameter that should be controlled by ablation
    controlled_params = [
        "lambda_sup", "lambda_ref", "lambda_probe", 
        "lambda_initmastery", "lambda_rate", "personalization", "active_grounding"
    ]
    
    for param in controlled_params:
        required_value = required_config[param]
        old_value = model_config.get(param, "not set")
        model_config[param] = required_value
        
        # Display status
        if old_value == "not set":
            status = "✓ set"
        elif old_value != required_value:
            # Type normalization for comparison
            if isinstance(required_value, bool):
                old_cmp = bool(old_value) if not isinstance(old_value, bool) else old_value
            else:
                old_cmp = float(old_value) if old_value != "not set" else None
                
            if isinstance(required_value, bool):
                req_cmp = required_value
            else:
                req_cmp = float(required_value)
            
            if old_cmp != req_cmp:
                status = f"OVERRIDE: {old_value} → {required_value}"
            else:
                status = f"✓ {required_value}"
        else:
            status = f"✓ {required_value}"
        
        print(f"  {param:20s} = {required_value:5} {status}")
    
    print(f"{'='*70}\n")
    
    return model_config
```

### 7.4 Integration Points

**Call the Ablation Control Center in these locations:**

#### 7.4.1 Training Scripts

In `examples/wandb_gtransformer_train.py` (after config loading, before `init_model()`):

```python
def main(params):
    # ... existing code to load configurations ...
    
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        model_config = copy.deepcopy(params)
        # ... existing model_config processing ...
    
    # === ABLATION CONTROL CENTER ===
    # Validate and apply ablation configuration before model initialization
    from pykt.models.init_model import validate_and_apply_ablation_config
    try:
        model_config = validate_and_apply_ablation_config(
            model_config, 
            source="command_line"
        )
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    # === END ABLATION CONTROL CENTER ===
    
    # Save configuration (now includes validated ablation params)
    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    
    # Continue with model initialization
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    # ... rest of code ...
```

#### 7.4.2 Config File Loading

When loading from `configs/parameter_default.json`:

```python
# After loading config from file
with open("configs/parameter_default.json") as f:
    params = json.load(f)

# Apply ablation control
from pykt.models.init_model import validate_and_apply_ablation_config
params = validate_and_apply_ablation_config(params, source="parameter_default.json")
```

#### 7.4.3 Evaluation Scripts

Before loading model for evaluation:

```python
# Load config
with open(config_path) as f:
    saved_config = json.load(f)
    model_config = saved_config['model_config']

# Validate ablation configuration
from pykt.models.init_model import validate_and_apply_ablation_config
model_config = validate_and_apply_ablation_config(model_config, source="saved_config")

# Load model
model = init_model(...)
```

### 7.5 Example Execution Flows

#### 7.5.1 ablation='none' - Use Parameters As-Is

**Scenario**: User wants full control over individual lambda parameters.

**Command:**
```bash
python wandb_gtransformer_train.py --ablation none --lambda_ref 0.5 --lambda_probe 0.0
```

**Output:**
```
======================================================================
ABLATION CONTROL CENTER
======================================================================
Ablation mode: 'none' (No Ablation - use parameters as configured (enforces active_grounding=1))
Using parameters as configured (enforces active_grounding=1):

  active_grounding     = 1  (enforced: not set → 1)
  lambda_sup           = 1.0
  lambda_ref           = 0.5
  lambda_probe         = 0.0
  lambda_initmastery   = 0.1
  lambda_rate          = 0.1
  personalization      = False
======================================================================
```

**Result**: Lambda and personalization parameters are used as specified. `active_grounding` is enforced to 1.

---

#### 7.5.2 ablation='all' - Ablation Mode Overrides Command-Line

**Scenario**: User wants pure neural baseline, ablation mode overrides any conflicting command-line parameters.

**Command:**
```bash
python wandb_gtransformer_train.py --ablation all --lambda_ref 0.5 --lambda_probe 1.0
```

**Output:**
```
======================================================================
ABLATION CONTROL CENTER
======================================================================
Ablation mode: 'all' (Pure Neural (baseline) - all grounding ablated)
Applying configuration:

  lambda_sup           = 1.0    ✓ 1.0
  lambda_ref           = 0      OVERRIDE: 0.5 → 0
  lambda_probe         = 0      OVERRIDE: 1.0 → 0
  lambda_initmastery   = 0      ✓ set
  lambda_rate          = 0      ✓ set
  personalization      = False  ✓ set
  active_grounding     = 0      ✓ set
======================================================================
```

**Result**: Ablation mode wins. User's `--lambda_ref 0.5` is overridden to 0. This demonstrates **ablation mode > command-line**.

---

#### 7.5.3 ablation='probe' - Selective Override

**Scenario**: User wants to ablate probe loss while keeping other grounding components.

**Command:**
```bash
python wandb_gtransformer_train.py --ablation probe --lambda_probe 1.0
```

**Output:**
```
======================================================================
ABLATION CONTROL CENTER
======================================================================
Ablation mode: 'probe' (No Probe Loss - probe training ablated)
Applying configuration:

  lambda_sup           = 1.0    ✓ 1.0
  lambda_ref           = 1.0    ✓ 1.0
  lambda_probe         = 0      OVERRIDE: 1.0 → 0
  lambda_initmastery   = 0      ✓ 0
  lambda_rate          = 0      ✓ 0
  personalization      = False  ✓ False
  active_grounding     = 1      ✓ 1
======================================================================
```

**Result**: Ablation mode overrides `lambda_probe` to 0, even though user specified 1.0.

---

#### 7.5.4 Config File vs Command-Line (ablation='none')

**Scenario**: Config file has `lambda_ref=1.0`, command-line specifies `--lambda_ref 0.5`.

**Config file (parameter_default.json):**
```json
{
  "lambda_ref": 1.0,
  "lambda_probe": 1.0
}
```

**Command:**
```bash
python wandb_gtransformer_train.py --ablation none --lambda_ref 0.5
```

**Output:**
```
======================================================================
ABLATION CONTROL CENTER
======================================================================
Ablation mode: 'none' (No Ablation - use parameters as configured (enforces active_grounding=1))
Using parameters as configured (enforces active_grounding=1):

  active_grounding     = 1       ← enforced
  lambda_sup           = 1.0
  lambda_ref           = 0.5     ← from command-line
  lambda_probe         = 1.0     ← from config file
  ...
======================================================================
```

**Result**: Command-line value (0.5) takes precedence over config file (1.0) for lambda parameters. `active_grounding` is enforced to 1. This demonstrates **command-line > config files** when ablation='none', except for `active_grounding` which is always enforced.

---

#### 7.5.5 Precedence Summary Example

**Full precedence chain demonstration:**

| Source | ablation | lambda_ref value | Who Wins? |
|:---|:---|:---|:---|
| Config file | `"none"` | 1.0 | Config file (1.0) |
| Config file + Command | `"none"` | 1.0 (config) + 0.5 (cmd) | Command-line (0.5) |
| Config file + Command | `"all"` | 1.0 (config) + 0.5 (cmd) | **Ablation mode (0)** |
| Config file + Command | `"probe"` | 0.5 (config) + 0.8 (cmd) | Command-line (0.8) - no override for probe mode |

**Key Insight**: 
- For `ablation='none'`: Standard precedence (command > config > defaults)
- For other ablation modes: **Ablation mode wins** for its controlled parameters

### 7.6 Benefits

1. **Consistency**: Guarantees all experiments use correct parameter combinations
2. **Error Prevention**: Stops execution before wasting compute on invalid configurations
3. **Transparency**: Clear output shows exactly what parameters are being used
4. **Reproducibility**: All final parameters are explicitly saved in `config.json`
5. **Documentation**: Config files serve as complete experiment documentation
6. **User-Friendly**: Clear error messages guide users to fix configuration issues

### 7.7 Testing Strategy

To validate the Ablation Control Center:

```python
# Test cases to implement
def test_ablation_control_center():
    # Test 1: Valid configuration passes
    config = {"ablation": "none"}
    result = validate_and_apply_ablation_config(config)
    assert result["lambda_ref"] == 1.0
    
    # Test 2: Conflict raises ValueError
    config = {"ablation": "probe", "lambda_probe": 1.0}
    with pytest.raises(ValueError) as exc_info:
        validate_and_apply_ablation_config(config)
    assert "ABLATION CONFLICT" in str(exc_info.value)
    
    # Test 3: Invalid ablation mode raises ValueError
    config = {"ablation": "invalid_mode"}
    with pytest.raises(ValueError) as exc_info:
        validate_and_apply_ablation_config(config)
    assert "Invalid ablation mode" in str(exc_info.value)
    
    # Test 4: Parameters are correctly set
    config = {"ablation": "reference"}
    result = validate_and_apply_ablation_config(config)
    assert result["lambda_ref"] == 0
    assert result["lambda_probe"] == 0
    assert result["personalization"] == False
```

---

## 8. Reproducibility Guidelines: Training, Evaluation, and Results

### 8.1 Training Command Documentation

**Requirement**: All training runs must save the complete command used to launch training with ALL parameters explicitly set.

**Implementation** (`examples/run_repro_experiment.py`):

When training is launched via `run_repro_experiment.py`, the system:

1. **Builds Explicit Training Command**: Constructs the full command with ALL parameters from `parameter_default.json` explicitly passed as flags (zero defaults philosophy)

2. **Saves to config.json**: Stores the complete training command in the experiment folder's `config.json`:
   ```json
   {
     "experiment": {
       "id": "123456",
       "created": "2026-01-24T10:30:00",
       "description": "Training gtransformer on assist2009 fold 0 - benchpaper"
     },
     "commands": {
       "train_explicit": "python examples/wandb_gtransformer_train.py --model gtransformer --dataset assist2009 --fold 0 --ablation none --lambda_sup 1.0 --lambda_ref 1.0 --lambda_probe 1.0 ...",
       "eval_explicit": "python examples/wandb_gtransformer_predict.py --save_dir /path/to/experiment --bz 64 --use_wandb 0 --fusion_type late_fusion --prediction_type supervised --dual_eval 0 --ablation none --personalization False --lambda_sup 1.0 ..."
     },
     "train_config": { /* All resolved training parameters */ },
     "params": { /* Alias for compatibility */ },
     "_documentation": {
       "purpose": "Complete reproducibility record for experiment",
       "train_command_info": "The train_explicit command contains ALL parameters used for training with explicit values (zero defaults)",
       "eval_command_info": "The eval_explicit command contains ALL parameters needed for evaluation, synchronized with training parameters",
       "parameter_precedence": "Command line > ablation mode > config file defaults",
       "ablation_control": "Ablation mode automatically sets lambda_* and personalization parameters"
     }
   }
   ```

3. **Training Script Also Saves Config**: `wandb_gtransformer_train.py` saves its own `config.json` in the checkpoint directory:
   ```json
   {
     "train_config": { /* Training hyperparameters */ },
     "model_config": { /* Model architecture parameters */ },
     "data_config": { /* Dataset configuration */ },
     "params": { /* Full parameter dictionary */ },
     "commands": {
       "train_explicit": "python examples/wandb_gtransformer_train.py ...",
       "launched_at": "2026-01-24T10:30:00"
     },
     "_documentation": {
       "purpose": "Configuration checkpoint for model training",
       "train_command_info": "The train_explicit command shows the exact command used to launch this training run",
       "note": "For evaluation, use the config.json in the parent experiment folder which contains the complete eval_explicit command"
     }
   }
   ```

### 8.2 Evaluation Command Synchronization

**Requirement**: Evaluation must use parameters that are properly synchronized with training parameters to ensure consistency.

**Implementation**:

1. **Evaluation Command Built from Training Parameters**: When `run_repro_experiment.py` creates the training config, it simultaneously builds the evaluation command using the SAME parameter set:

   ```python
   # In run_repro_experiment.py
   train_command_explicit = build_explicit_train_command(train_script, training_params, experiment_dir)
   eval_command_explicit = build_explicit_eval_command(eval_script, experiment_dir, training_params)
   ```

2. **Gtransformer-Specific Synchronization**: For gtransformer, the evaluation command includes all ablation-related parameters:
   ```python
   # Evaluation command includes:
   --ablation none
   --personalization False  # Synchronized with training
   --lambda_sup 1.0
   --lambda_ref 1.0
   --lambda_probe 1.0
   --lambda_initmastery 0
   --lambda_rate 0
   --n_uid 0  # Derived from personalization flag
   --active_grounding 1
   ```

3. **Parameters NOT Duplicated in Evaluation**:
   - Architecture parameters (d_model, n_heads, etc.) are read from saved model checkpoint
   - Training hyperparameters (learning_rate, optimizer, etc.) are not needed for evaluation
   - Only evaluation-specific parameters are passed:
     - `--save_dir`: Where to load model and save results
     - `--bz`: Batch size for evaluation
     - `--use_wandb`: Whether to log to W&B
     - `--fusion_type`: How to combine predictions (late_fusion, early_fusion, etc.)
     - `--prediction_type`: Which prediction to use (supervised, reference, etc.)
     - `--dual_eval`: Whether to evaluate both supervised and reference predictions
     - Ablation parameters (for proper model loading and interpretation)

### 8.3 Running Benchmarks with `run_benchmarks_paper.py`

**Script Purpose**: Orchestrate training, evaluation, and results collection for benchmark experiments.

**Usage**:

#### 8.3.1 Training Mode

Launch training for multiple models and datasets:

```bash
# Train all models on all datasets with 5-fold CV
python examples/run_benchmarks_paper.py --mode training

# Train specific model on specific dataset
python examples/run_benchmarks_paper.py --mode training --model gtransformer --dataset assist2009

# Train with specific parameters (override defaults)
python examples/run_benchmarks_paper.py --mode training --model gtransformer --dataset assist2009 --ablation none --epochs 200
```

**What Happens**:
1. Creates grouped experiment folder: `experiments/YYYYMMDD_HHMMSS_benchpaper_[UNIQUE_ID]/`
2. For each fold, creates subfolder: `fold_[X]_[ID]/`
3. Calls `run_repro_experiment.py` to execute training
4. Saves complete `config.json` with `train_explicit` and `eval_explicit` commands
5. Logs all output to `execution_history.log` in each fold directory

**Verification**:
```bash
# Check that config.json contains explicit commands
cat experiments/[experiment_folder]/fold_0_[ID]/config.json | jq '.commands'

# Should show:
# {
#   "train_explicit": "python examples/wandb_gtransformer_train.py --model gtransformer ...",
#   "eval_explicit": "python examples/wandb_gtransformer_predict.py --save_dir ... --ablation none ..."
# }
```

#### 8.3.2 Evaluation Mode

Run evaluation on trained experiments:

```bash
# Evaluate all experiments
python examples/run_benchmarks_paper.py --mode evaluation

# Evaluate specific dataset
python examples/run_benchmarks_paper.py --mode evaluation --dataset assist2009

# Evaluate specific experiment folder
python examples/run_benchmarks_paper.py --mode evaluation --experiment_folder experiments/[folder]/fold_0_[ID]

# Evaluate with dual evaluation (supervised + reference predictions)
python examples/run_benchmarks_paper.py --mode evaluation --dual_eval
```

**What Happens**:
1. Finds experiment folder(s) matching the criteria
2. Reads `config.json` from experiment folder
3. Extracts `commands.eval_explicit` command
4. **Logs the explicit commands for audit trail**:
   ```
   ================================================================================
   EXPERIMENT: fold_0_123456
   ================================================================================
   Training command: python examples/wandb_gtransformer_train.py --model gtransformer ...
   Evaluation command: python examples/wandb_gtransformer_predict.py --save_dir ... --ablation none ...
   ================================================================================
   ```
5. Executes the evaluation command
6. Saves results to `eval_results.json` in experiment folder

**Verification**:
```bash
# Check that evaluation used the correct command from config.json
cat experiments/[experiment_folder]/fold_0_[ID]/eval_results.json

# Verify parameters match training config
jq '.train_config.ablation, .train_config.lambda_ref' experiments/[folder]/fold_0_[ID]/config.json
```

#### 8.3.3 Results Mode

Collect and summarize results across experiments:

```bash
# Show results for all models and datasets
python examples/run_benchmarks_paper.py --mode results

# Show results for specific dataset
python examples/run_benchmarks_paper.py --mode results --dataset assist2009

# Show results for specific model
python examples/run_benchmarks_paper.py --mode results --model gtransformer
```

**What Happens**:
1. Scans experiment folders for completed training/evaluation
2. Reads `eval_results.json` from each fold
3. Computes statistics (mean, std) across folds
4. Displays summary table:
   ```
   Model        | Dataset         | Status     | AUC            | ACC            | Fold Details
   -------------|-----------------|------------|----------------|----------------|--------------
   gtransformer | assist2009      | Complete   | 0.7812 ± 0.001 | 0.7301 ± 0.002 | 5/5 folds
   ```

### 8.4 Audit Compliance Checklist

For each experiment, verify the following are saved in `config.json`:

✅ **Experiment Metadata**:
- `experiment.id`: 6-digit unique identifier
- `experiment.created`: Timestamp of creation
- `experiment.description`: Human-readable description

✅ **Complete Commands**:
- `commands.train_explicit`: Full training command with ALL parameters
- `commands.eval_explicit`: Full evaluation command with synchronized parameters

✅ **Parameter Documentation**:
- `train_config`: All resolved training parameters (after applying defaults and overrides)
- `params`: Alias for compatibility with evaluation scripts
- `defaults`: Original default parameters from `parameter_default.json`

✅ **Ablation Parameters** (for gtransformer):
- `ablation`: Ablation mode (all, none, reference, probe, personalization)
- `lambda_sup`, `lambda_ref`, `lambda_probe`, `lambda_initmastery`, `lambda_rate`: All explicitly set
- `personalization`: Boolean flag (synchronized with n_uid)

✅ **Documentation Section**:
- `_documentation.purpose`: Explains the config file's role
- `_documentation.train_command_info`: Explains train_explicit content
- `_documentation.eval_command_info`: Explains eval_explicit content
- `_documentation.parameter_precedence`: Documents parameter precedence rules
- `_documentation.ablation_control`: Documents ablation control center behavior

### 8.5 Parameter Synchronization Rules

**Rule 1: Ablation Mode Precedence**
- The `ablation` parameter takes absolute precedence over individual lambda parameters
- Ablation Control Center enforces this and raises errors on conflicts
- Example: `--ablation all` sets `lambda_ref=0`, `lambda_probe=0` regardless of explicit flags

**Rule 2: Training-to-Evaluation Synchronization**
- Evaluation command is built from the SAME `training_params` dictionary as training command
- Ensures ablation parameters, personalization flags, etc. are identical between training and evaluation
- Example: If training used `--ablation none --personalization False`, evaluation uses the same

**Rule 3: Checkpoint Configuration vs Experiment Configuration**
- Experiment folder `config.json`: Contains `train_explicit` and `eval_explicit` commands (authoritative)
- Checkpoint folder `config.json`: Contains model architecture and training hyperparameters (for model loading)
- For reproduction, always use experiment folder config as source of truth

**Rule 4: Config.json as Audit Trail**
- All parameters must be explicitly documented in config.json (zero defaults)
- Commands must be reproducible: copy-paste should work
- Documentation section explains what each command does

### 8.6 Common Verification Steps

**After Training**:
```bash
# 1. Check config.json was created
ls experiments/[folder]/fold_0_[ID]/config.json

# 2. Verify explicit commands are present
jq '.commands | keys' experiments/[folder]/fold_0_[ID]/config.json
# Should show: ["eval_explicit", "train_explicit", ...]

# 3. Check ablation parameters are synchronized
jq '.train_config | {ablation, lambda_sup, lambda_ref, lambda_probe, personalization}' \
   experiments/[folder]/fold_0_[ID]/config.json

# 4. Verify training command has all parameters
jq -r '.commands.train_explicit' experiments/[folder]/fold_0_[ID]/config.json | wc -w
# Should be a long command with many flags
```

**Before Evaluation**:
```bash
# 1. Check that eval_explicit command exists
jq -r '.commands.eval_explicit' experiments/[folder]/fold_0_[ID]/config.json

# 2. Verify evaluation command includes ablation parameters
jq -r '.commands.eval_explicit' experiments/[folder]/fold_0_[ID]/config.json | grep "ablation"

# 3. Confirm model checkpoint exists
ls experiments/[folder]/fold_0_[ID]/[params_str]/qid_model.ckpt
```

**After Evaluation**:
```bash
# 1. Check results were saved
ls experiments/[folder]/fold_0_[ID]/eval_results.json

# 2. Verify results contain expected metrics
jq 'keys' experiments/[folder]/fold_0_[ID]/eval_results.json
# Should show: ["auc", "acc", "predictions", ...]
```

---

## 9. Parameter Transparency and Auditing

### 9.1 Training Parameter Dump

When training starts, all parameters that will be used are displayed:

```
======================================================================
TRAINING PARAMETERS DUMP (Final Configuration)
======================================================================

model_config (architecture parameters):
  ablation                  = none
  active_grounding          = 1
  d_ff                      = 256
  d_model                   = 256
  ...

train_config (training hyperparameters):
  batch_size                = 24
  learning_rate             = 0.001
  num_epochs                = 200
  ...

data_config (dataset parameters):
  dataset_name              = assist2009
  num_skills                = 123
  sequence_length           = 200
  ...
======================================================================
```

**Location**: `examples/wandb_gtransformer_train.py` (lines ~126-143)

**Purpose**:
- Shows all resolved parameter values after ablation control center processing
- Enables verification that parameters match expected configuration
- Provides audit trail for reproducibility
- Alphabetically sorted for easy lookup

### 9.2 Evaluation Parameter Dump

When evaluation starts, parameters are shown in two stages:

**Stage 1: Command-Line Parameters**
```
======================================================================
EVALUATION PARAMETERS DUMP (Command-Line Arguments)
======================================================================
  ablation                  = none
  bz                        = 256
  fusion_type               = early_fusion,late_fusion
  save_dir                  = experiments/.../fold_0_123456
  ...
======================================================================
```

**Stage 2: Loaded Training Configuration**
```
======================================================================
LOADED TRAINING CONFIGURATION (from config.json)
======================================================================

model_config (architecture from training):
  ablation                  = none
  active_grounding          = 1
  d_ff                      = 256
  ...

train_config (hyperparameters from training):
  batch_size                = 24
  learning_rate             = 0.001
  ...
======================================================================
```

**Location**: `examples/wandb_gtransformer_predict.py` (lines ~26-58)

**Purpose**:
- Shows evaluation command-line arguments
- Shows configuration loaded from training's config.json
- Enables verification that evaluation uses same architecture as training
- Helps debug parameter mismatches between training and evaluation

### 9.3 Benchmark Runner Command Display

When running benchmarks via `run_benchmarks_paper.py`, full commands are displayed:

```
================================================================================
EXPERIMENT: fold_0_848274
================================================================================
Training command (used to train this model):
  python examples/wandb_gtransformer_train.py --model gtransformer --ablation none --lambda_ref 1.0 --lambda_probe 1.0 ...

Evaluation command (will be executed now):
  python examples/wandb_gtransformer_predict.py --save_dir experiments/.../fold_0_848274 --ablation none ...

Configuration loaded from: experiments/.../fold_0_848274/config.json
================================================================================
```

**Location**: `examples/run_benchmarks_paper.py` (lines ~287-302)

**Purpose**:
- Shows full training command without truncation
- Shows full evaluation command that will be executed
- Links to config.json for parameter verification
- Creates audit trail for reproducibility

### 9.4 Best Practices

1. **Always check parameter dumps** before starting long training runs
2. **Verify ablation mode** is correctly reflected in parameter values
3. **Compare training vs evaluation** parameter dumps to ensure consistency
4. **Save terminal output** from benchmark runs for audit trail
5. **Check config.json** matches the explicit commands shown

---

## 10. Current Status (v0.0.32-gtransformer-probe))

**Implemented Features**:
- ✅ Full ablation control via `--ablation {all|none|reference|probe|personalization}`
- ✅ **Ablation Control Center** in `pykt/models/init_model.py`
- ✅ **Parameter precedence system**: ablation mode > command-line > config files
- ✅ **Passthrough mode** (`ablation='none'`): uses parameters as-is
- ✅ **Override feedback**: shows "OVERRIDE: old → new" for transparency
- ✅ **Parameter dumping** in training (comprehensive multi-section dump)
- ✅ **Parameter dumping** in evaluation (command-line + loaded config)
- ✅ **Full command display** in benchmark runner (no truncation)
- ✅ Semantic axis projection for grounded parameters
- ✅ Probe heads for active grounding
- ✅ BKT reference output with differentiable walk
- ✅ Diversity loss for axis orthogonality
- ✅ Student personalization support (via personalization flag)
- ✅ Theory-guided initialization from BKT parameters
- ✅ Explicit command documentation in config.json (training + evaluation)
- ✅ Training-to-evaluation parameter synchronization
- ✅ Audit trail logging in run_benchmarks_paper.py
- ✅ Unit tests for Ablation Control Center (4/4 passing)

**Pending Implementation**:
- ⏸️ Ablation set parsing logic (comma-separated values for multi-ablation)
- ⏸️ Fine-grained component creation conditionals (selective initialization)
- ⏸️ Conditional forward pass outputs (skip unused computations)
- ⏸️ Conditional loss computation in training script (optimize based on ablation)

**Removed/Deprecated**:
- ❌ `ablation="regularization"` mode (not implemented)

**Active Research Direction**:
- Testing probe-based grounding (v1.0) vs PCA-based grounding (v2.0)
- Branch `v0.0.31-gtransformer-pca` explores alternative three-term decomposition
- Current branch (`v0.0.32-gtransformer-probe`) focuses on probe-based active grounding
- **Next**: Implement Ablation Control Center and granular ablation modes

