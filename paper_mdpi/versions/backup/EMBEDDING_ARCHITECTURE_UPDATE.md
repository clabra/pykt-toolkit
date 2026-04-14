# Embedding Architecture Update - D2 Diagram Corrections

## Summary

The D2 diagram `arch_probe_v1.d2` was missing critical embedding components from the actual GTransformer implementation. This document summarizes the updates made to accurately reflect the codebase.

---

## Missing Components (Now Added)

### 1. **Theory Bases** (BKT Prior Embeddings)
**File**: `pykt/models/gtransformer.py` lines 84-85

```python
self.l0_base_emb = nn.Embedding(self.n_question + 1, 1)  # L0_skill (Prior Base)
self.t_base_emb = nn.Embedding(self.n_question + 1, 1)   # T_skill (Velocity Base)
```

**Initialization**: Lines 196-199
- Textured logits: `N(σ⁻¹(BKT_prior), 0.05)` for L₀
- Textured logits: `N(σ⁻¹(BKT_learns), 0.05)` for T
- NOT constant vectors - distribution ensures non-zero variance for LayerNorm survival

**Shape**: `[n_question+1, 1]` → scalar logits per skill

---

### 2. **Semantic Axes** (Relational Directions)
**File**: `pykt/models/gtransformer.py` lines 77-80

```python
self.knowledge_axis_emb = nn.Embedding(self.n_question + 1, z_dim)  # Knowledge Direction
self.velocity_axis_emb = nn.Embedding(self.n_question + 1, z_dim)   # Velocity Direction
nn.init.normal_(self.knowledge_axis_emb.weight, mean=0.0, std=0.02)
nn.init.normal_(self.velocity_axis_emb.weight, mean=0.0, std=0.02)
```

**Re-initialization**: Lines 209-212 (after BKT loading)
- Orthogonal initialization to prevent axis collapse
- Diversity loss (lines 288-310) maintains orthogonality during training

**Shape**: `[n_question+1, z_dim]` where `z_dim = d_model + embed_l`

**Usage** (lines 273-279):
```python
k_axis = self.knowledge_axis_emb(q_data)  # BS, seqlen, z_dim
v_axis = self.velocity_axis_emb(q_data)   # BS, seqlen, z_dim

l0_logits = l0_base + (z_context * k_axis).sum(dim=-1)  # Dot product projection
t_logits = t_base + (z_context * v_axis).sum(dim=-1)
```

---

### 3. **Student Parameters** (Personalization)
**File**: `pykt/models/gtransformer.py` lines 95-96

```python
if self.personalization:  # Derived from n_uid > 0
    self.student_param = nn.Embedding(self.n_uid + 1, 1)      # Learning velocity bias (v_s)
    self.student_gap_param = nn.Embedding(self.n_uid + 1, 1)  # Knowledge gap bias (k_c)
```

**Usage** (lines 314-326):
```python
s_gap = self.student_gap_param(uid_seq).squeeze(-1)  # BS, seqlen
s_vel = self.student_param(uid_seq).squeeze(-1)

l0_logits = l0_logits + s_gap  # Add student-specific bias
t_logits = t_logits + s_vel
```

**Shape**: `[n_uid+1, 1]` → scalar bias per student

---

### 4. **BKT Population Buffers** (Reference Generation)
**File**: `pykt/models/gtransformer.py` lines 88-91

```python
self.register_buffer('bkt_guess', torch.ones(n_question + 1) * 0.2)
self.register_buffer('bkt_slip', torch.ones(n_question + 1) * 0.1)
self.register_buffer('bkt_l0_pop', torch.ones(n_question + 1) * 0.5)
self.register_buffer('bkt_t_pop', torch.ones(n_question + 1) * 0.1)
```

**Purpose**: Frozen population-level BKT parameters for Reference Output generation
- `bkt_guess`, `bkt_slip`: Noise parameters for BKT mastery walk
- `bkt_l0_pop`, `bkt_t_pop`: Population defaults (not used in grounded parameter generation)

**Shape**: `[n_question+1]` → scalar per skill (registered buffers, not trainable)

---

### 5. **Probe Architecture** (Active Grounding)
**File**: `pykt/models/gtransformer.py` lines 100-101

```python
self.probe_l0 = nn.Linear(z_dim, 1)  # Direct linear probe for L₀
self.probe_t = nn.Linear(z_dim, 1)   # Direct linear probe for T
```

**Usage** (lines 332-336):
```python
probe_l0_logits = self.probe_l0(z_context).squeeze(-1)  # BS, seqlen
probe_t_logits = self.probe_t(z_context).squeeze(-1)

p_l0_probe = torch.sigmoid(probe_l0_logits)
p_t_probe = torch.sigmoid(probe_t_logits)
```

**Purpose**: Separate from grounded parameters - used for:
1. Validation diagnostics (parameter recovery)
2. Active grounding loss (alignment with BKT targets)

---

## Key Architecture Corrections in D2

### Section 2: Embeddings Layer

**Before** (Old):
- Only showed standard embeddings (history + task)
- No mention of theory-guided components

**After** (Updated):
- **Standard Track (Blue)**: History + Task embeddings (unchanged)
- **Interpretable Track (Green)**:
  - Theory Bases: `L₀_base[q]`, `T_base[q]` with textured initialization
  - Semantic Axes: `K_axis[q]`, `V_axis[q]` with orthogonal initialization
  - Student Parameters: `k_gap[uid]`, `v_student[uid]` (if personalization enabled)
  - BKT Buffers: Frozen population parameters for reference output

---

### Section 4: Grounded Parameters

**Before** (Old):
```
Theory Base: σ⁻¹(μ_L₀[q])
Contextual Projection: W_L₀ · z_t + b_L₀  (linear probe)
Final: σ(ℓ_L₀ + Δ_L₀)
```

**After** (Updated):
```
1. Theory Base: ℓ_L₀[q] = L₀_base[q]  (from embedding)
2. Semantic Projection: Δ_L₀ = z_t · K_axis[q]  (dot product, NOT linear layer)
3. Student Bias: β_L₀ = k_gap[uid]  (if personalization)
Final: p_L₀ = σ(ℓ_L₀ + Δ_L₀ + β_L₀)  (three-term sum)
```

**Critical Difference**: 
- OLD: Used generic linear probes `W·z + b`
- NEW: Uses **semantic axis projection** `z·axis` with **per-skill axes**
- This is the core innovation - axes are skill-specific learned directions, not global weight matrices

---

### Section 5: Outputs

**Updated**:
- Clarified that **Diagnostic Probes** (`p̂_L₀`, `p̂_T`) are separate from grounded parameters
- Probes use simple `Linear(z_dim, 1)` for validation/alignment
- Grounded parameters use the 3-component composition (base + projection + bias)

---

## Data Flow Connections

**New Connections Added**:
1. `BKT → Theory Bases`: Load priors during initialization
2. `BKT → BKT Buffers`: Load population parameters (G, S, L₀_pop, T_pop)
3. `Semantic Axes → Semantic Projection`: Provide K_axis, V_axis for projection
4. `Student Parameters → Personal Bias`: Add student-specific scalars
5. `BKT Buffers → Reference Output`: Supply G, S for BKT walk
6. Color-coded green for interpretable track, blue for standard track

---

## Parameter Counts

For typical configuration (`n_question=100`, `d_model=64`, `embed_l=64`, `n_uid=500`):

**Standard Track** (Blue):
- `q_embed`: 100 × 64 = 6,400
- `qa_embed`: 201 × 64 = 12,864
- `q_embed_diff`: 101 × 64 = 6,464
- `qa_embed_diff`: 2 × 64 = 128
- `difficult_param`: 101 × 1 = 101

**Interpretable Track** (Green):
- `l0_base_emb`: 101 × 1 = 101
- `t_base_emb`: 101 × 1 = 101
- `knowledge_axis_emb`: 101 × 128 = 12,928
- `velocity_axis_emb`: 101 × 128 = 12,928
- `student_gap_param`: 501 × 1 = 501 (if personalization)
- `student_param`: 501 × 1 = 501 (if personalization)
- `probe_l0`: 128 × 1 + 1 = 129
- `probe_t`: 128  1 + 1 = 129
- **Total Green Track**: ~27,318 parameters

**Buffers** (non-trainable):
- `bkt_guess`, `bkt_slip`, `bkt_l0_pop`, `bkt_t_pop`: 4 × 101 = 404

---

## Verification

Generated diagram: `paper/latex/d2/arch_probe_v1.svg`

```bash
cd /workspaces/pykt-toolkit/paper/latex/d2
d2 --theme=200 arch_probe_v1.d2 arch_probe_v1.svg
# success: successfully compiled in 1.118s
```

All embeddings now accurately documented and connected in the architecture diagram.
