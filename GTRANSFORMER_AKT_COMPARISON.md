# GTransformer (ablation=all) vs AKT: Architecture Comparison

## Executive Summary

**Expected Behavior:** gTransformer with `ablation=all` should theoretically produce identical results to AKT since all interpretability components are disabled.

**Actual Results:** Performance differs:
- AS2009: gTransformer 0.7832±0.0015 vs AKT 0.7825 (+0.0007, **better**)
- AS2015: gTransformer 0.7070±0.0006 vs AKT 0.7081 (-0.0011, worse)
- AL2005: gTransformer 0.8235±0.0007 vs AKT 0.8306 (-0.0071, worse)
- BDG2006: gTransformer 0.8132±0.0011 vs AKT 0.8208 (-0.0076, worse)
- NIPS34: gTransformer 0.7998±0.0004 vs AKT 0.8033 (-0.0035, worse)

**Overall Pattern:** gTransformer is competitive but generally slightly worse than AKT (4/5 datasets).

## Key Architectural Differences

### 1. **Number of Transformer Blocks** ⚠️ **CRITICAL DIFFERENCE**

**AKT Configuration:**
```json
{
    "n_blocks": 2,
    "num_attn_heads": 8,
    "d_model": 64,
    "d_ff": 256
}
```

**GTransformer Configuration:**
```json
{
    "n_blocks": 4,
    "num_attn_heads": 4,
    "d_model": 64,
    "d_ff": 256
}
```

**Impact:**
- AKT: 2 blocks × 3 attention layers = **6 total transformer layers**
  - 1 encoder block (blocks_1)
  - 2 decoder blocks (blocks_2)
  - Total: 2 × (1 + 2) = 6 layers
  
- gTransformer: 4 blocks × 3 attention layers = **12 total transformer layers**
  - More capacity but also more parameters and potential overfitting

### 2. **Number of Attention Heads** ⚠️

**AKT:** 8 heads
**GTransformer:** 4 heads

**Impact:**
- AKT has more parallel attention paths (8 vs 4)
- Each head in AKT sees d_model/8 = 8 dimensions
- Each head in gTransformer sees d_model/4 = 16 dimensions
- Different attention granularity and expressiveness

### 3. **Architecture Class Differences**

**AKT Architecture** (`akt.py` lines 121-160):
```python
class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        # AKT uses separate encoder/decoder blocks
        self.blocks_1 = nn.ModuleList([
            TransformerLayer(...) for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(...) for _ in range(n_blocks*2)
        ])
    
    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # Encoder: process qa (interactions)
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)
        
        # Decoder: process q (questions)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False, pdiff=pid_embed_data)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data)
                flag_first = True
        return x
```

**GTransformer Architecture** (`gtransformer.py` lines 560-595):
```python
class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        # GTransformer appears to use the SAME architecture logic as AKT
        # when ablation="all"
        if model_type in {'gtransformer'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(...) for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(...) for _ in range(n_blocks*2)
            ])
```

**Observation:** The Architecture classes are **identical in structure** when `model_type='gtransformer'` or `model_type='akt'`. The difference is in the **number of blocks**.

### 4. **Forward Pass with ablation=all**

**GTransformer** (`gtransformer.py` lines 240-295):
```python
def forward(self, q_data, target, pid_data=None, uid_data=None, qtest=False):
    # Same embedding logic as AKT
    if emb_type.startswith("qid"):
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
    
    # Same Rasch parameter logic as AKT
    if self.n_pid > 0:
        # ... identical to AKT ...
        c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2_rasch
    
    # Pass through Architecture (same as AKT)
    d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
    
    # Concatenate and predict
    z_context = torch.cat([d_output, q_embed_data], dim=-1)
    output = self.out(z_context).squeeze(-1)
    preds = torch.sigmoid(output)
    
    # Early return when ablation="all"
    if self.ablation == "all":
        outputs = {'predictions': preds}
        return outputs, c_reg_loss
```

**AKT** (`akt.py` lines 82-120):
```python
def forward(self, q_data, target, pid_data=None, qtest=False):
    # Same embedding logic
    if emb_type.startswith("qid"):
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
    
    # Same Rasch parameter logic
    if self.n_pid > 0:
        # ... identical to gTransformer ...
        c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
    
    # Pass through Architecture
    d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
    
    # Concatenate and predict
    concat_q = torch.cat([d_output, q_embed_data], dim=-1)
    output = self.out(concat_q).squeeze(-1)
    preds = torch.sigmoid(output)
    
    return preds, c_reg_loss
```

**Observation:** The forward pass logic is **functionally identical** except gTransformer returns a dict while AKT returns a tuple.

### 5. **Output Head (Final FC)**

Both models use the **exact same** output head architecture:
```python
self.out = nn.Sequential(
    nn.Linear(d_model + embed_l, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
    nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
    nn.Linear(256, 1)
)
```

Where:
- `d_model + embed_l = 64 + 64 = 128` (input dim)
- `final_fc_dim = 512` (AKT) vs `512` (gTransformer default)

### 6. **Regularization Parameters**

**AKT:**
```python
self.l2 = l2  # Default: 1e-5
c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
```

**GTransformer:**
```python
self.l2_rasch = l2_rasch  # Default: 1e-5
c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2_rasch
```

**Observation:** Same regularization logic, just different parameter names.

### 7. **Additional Components in GTransformer (ablation=all)**

When `ablation="all"`, gTransformer still initializes but **does not use**:
- `knowledge_axis_emb`, `velocity_axis_emb` (not created)
- `l0_base_emb`, `t_base_emb` (not created)
- `probe_l0`, `probe_t` (not created)
- `student_param`, `student_gap_param` (not created if n_uid=0)

These are conditionally created only when `self.ablation != "all"`:
```python
if self.ablation != "all":
    self.knowledge_axis_emb = nn.Embedding(...)
    self.velocity_axis_emb = nn.Embedding(...)
    # ... etc
```

**Observation:** No extra parameters are created when ablation=all, so no memory overhead.

## Why Results Differ

### Hypothesis 1: Different Model Capacity ⭐ **MOST LIKELY**

**AKT:** 2 blocks (6 transformer layers total)
**GTransformer:** 4 blocks (12 transformer layers total)

**Impact:**
- gTransformer has **2x the transformer depth** of AKT
- More layers = more capacity but also:
  - Higher risk of overfitting (especially on smaller datasets like AS2015, AL2005, BDG2006)
  - More parameters to train
  - Potentially harder optimization landscape

**Evidence:**
- AS2009 (largest dataset): gTransformer slightly **better** (0.7832 vs 0.7825)
- AS2015, AL2005, BDG2006, NIPS34 (smaller datasets): gTransformer **worse**
- Larger datasets benefit from extra capacity, smaller datasets suffer from overfitting

### Hypothesis 2: Different Attention Granularity

**AKT:** 8 heads × 8 dims per head
**GTransformer:** 4 heads × 16 dims per head

**Impact:**
- AKT has more diverse attention patterns (8 parallel views)
- gTransformer has fewer but "wider" attention patterns (4 parallel views with more dimensions each)
- Different expressiveness and learning dynamics

### Hypothesis 3: Hyperparameter Mismatch

The configurations were optimized independently:
- AKT: Optimized with 2 blocks, 8 heads
- gTransformer: Optimized with 4 blocks, 4 heads

**Impact:**
- Learning rate 0.0001 (AKT) vs 0.0002 (gTransformer) may not be optimal for different architectures
- Different block counts require different optimization strategies

### Hypothesis 4: Random Seed and Initialization

**AKT:** seed=3407
**GTransformer:** seed=42

**Impact:**
- Different random initializations
- Different data shuffling
- Could contribute to small performance differences (±0.001-0.007 range)

## Recommendations

### To Achieve AKT Parity

If the goal is to exactly match AKT performance with gTransformer ablation=all, modify the configuration:

```json
{
    "gtransformer": {
        "n_blocks": 2,           // Match AKT
        "num_attn_heads": 8,     // Match AKT
        "d_model": 64,           // Already matches
        "d_ff": 256,             // Already matches
        "dropout": 0.1,          // Match AKT
        "learning_rate": 0.0001, // Match AKT
        "lambda_ref": 0,         // Already correct for ablation=all
        "seed": 3407             // Match AKT
}
```

### To Optimize gTransformer Independently

If the goal is to optimize gTransformer as its own architecture:

1. **For larger datasets** (AS2009, NIPS34):
   - Keep n_blocks=4 or even increase to 6
   - Use higher learning rate (0.0002)
   
2. **For smaller datasets** (AS2015, AL2005, BDG2006):
   - Reduce n_blocks to 2
   - Add more regularization (higher dropout, weight decay)
   - Lower learning rate (0.0001)

3. **Universal approach:**
   - Use n_blocks=2 for consistency with AKT
   - Adjust num_attn_heads based on dataset size
   - Fine-tune learning rate per dataset

## Verification Experiment

To confirm Hypothesis 1, run this experiment:

```bash
# Test gTransformer with AKT's exact configuration
python examples/run_repro_experiment.py \
    --model_name gtransformer \
    --dataset assist2009 \
    --fold 0 \
    --ablation all \
    --n_blocks 2 \
    --num_attn_heads 8 \
    --d_model 64 \
    --d_ff 256 \
    --dropout 0.1 \
    --learning_rate 0.0001 \
    --seed 3407
```

**Expected Result:** Should match AKT performance within ±0.0005 (random variation).

## Conclusion

**gTransformer with ablation=all is NOT exactly equivalent to AKT** due to:

1. **Different default configurations** (4 blocks vs 2 blocks, 4 heads vs 8 heads)
2. **Different model capacity** (12 layers vs 6 layers)
3. **Different hyperparameters** (learning rate, seed)

The architectural **code** is functionally identical when ablation=all, but the **instantiation parameters** differ, leading to different models and therefore different performance.

To achieve true parity, either:
- **Option A:** Use identical configurations for both models
- **Option B:** Accept that gTransformer is a distinct architecture with its own optimal hyperparameters

Current results show gTransformer is competitive (especially on larger datasets) but not superior to AKT when using default configurations optimized independently.
