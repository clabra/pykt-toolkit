# GTransformer v2.0: Three-Term Decomposition with PCA Cluster Grounding

## Overview

GTransformer v2.0 introduces a novel parameter decomposition that separates student performance into three interpretable components grounded by Principal Component Analysis (PCA):

1. **Population-level parameters** (μ): Skill-specific baselines from BKT (fixed)
2. **Per-student trait parameters** (δ): **Computed dynamically from interaction history** (context-based, no memorization)
3. **Per-skill residuals** (ε): Interaction-specific deviations

This decomposition enables:
- **No student memorization**: Traits computed from context, not stored embeddings
- **Generalizes to new students**: Works with any interaction history
- **Interpretability**: Student traits aligned with principal components of variance
- **Reproducibility**: Deterministic PCA (unlike stochastic t-SNE)
- **Mathematical grounding**: PCA components have clear statistical meaning

## Key Design Decision: Context-Based Traits (No Memorization)

**CRITICAL**: Unlike traditional personalization approaches that use static student embeddings (`nn.Embedding`), we compute student traits **dynamically from the Transformer's context vector**. This avoids:

❌ **Student memorization** (overfitting to student IDs)
❌ **Cold-start problems** (can't handle new students)
❌ **Scalability issues** (memory grows with students)

✅ **Generalization** (learns from behavior patterns)
✅ **Works for new students** (no retraining needed)
✅ **Scalable** (no per-student parameters)

## Mathematical Formulation

### Parameter Decomposition

For student `i` and skill `q` at time `t`:

```
p_L0[i,q,t] = σ(μ_L0[q] + δ_L0[i,t] + ε_L0[i,q,t])
p_T[i,q,t]  = σ(μ_T[q]  + δ_T[i,t]  + ε_T[i,q,t])
```

Where:
- **μ[q] ∈ ℝ¹**: Population-level logit for skill q (from BKT, fixed)
- **δ[i,t] ∈ ℝ¹**: Per-student trait offset (**computed from z_context**)
- **ε[i,q,t] ∈ ℝ¹**: Per-interaction residual (computed from z_context)
- **σ**: Sigmoid function to convert logits to probabilities

### Student Trait Computation (From Context)

```
# Aggregate interaction history
z_agg[i,t] = mean(z_context[i, 1:t])  # Mean pooling over sequence

# Project to 2D trait space
traits[i,t] = W_traits · z_agg[i,t] + b_traits  # [δ_L0, δ_T]

Where:
- z_context ∈ ℝ^(seqlen × z_dim): Transformer context vectors
- W_traits ∈ ℝ^(2 × z_dim): Learnable projection matrix
- traits ∈ ℝ²: 2D trait vector
```

## Architecture Components

### 1. Population-Level Parameters (Fixed from BKT)

```python
class GTransformerV2(nn.Module):
    def __init__(self, n_question, d_model, ...):
        super().__init__()
        
        # Fixed population-level parameters (from BKT)
        self.register_buffer('mu_L0', torch.zeros(n_question + 1))
        self.register_buffer('mu_T', torch.zeros(n_question + 1))
        
        # PCA reference (from BKT analysis, for grounding loss)
        # Note: n_uid only needed for PCA reference, not for model parameters
        self.register_buffer('pca_reference', torch.zeros(n_uid + 1, 2))
        self.register_buffer('pca_components', torch.eye(2))  # [2, 2]
        self.register_buffer('pca_mean', torch.zeros(2))
        
    def load_population_params(self, bkt_skill_params):
        """Load fixed population-level parameters from BKT"""
        params_dict = bkt_skill_params.get('params', {})
        
        mu_L0 = torch.zeros(self.n_question + 1)
        mu_T = torch.zeros(self.n_question + 1)
        
        for q_idx in range(self.n_question + 1):
            s_params = params_dict.get(q_idx, params_dict.get(str(q_idx), {}))
            
            # Convert probabilities to logits
            l0_p = s_params.get('prior', 0.5)
            t_p = s_params.get('learns', 0.1)
            
            mu_L0[q_idx] = self.prob_to_logit(l0_p)
            mu_T[q_idx] = self.prob_to_logit(t_p)
        
        self.mu_L0.copy_(mu_L0)
        self.mu_T.copy_(mu_T)
    
    def prob_to_logit(self, p, eps=1e-6):
        """Convert probability to logit"""
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))
```

### 2. Student Trait Encoder (Context-Based, No Embeddings!)

```python
class StudentTraitEncoder(nn.Module):
    """Compute student traits from interaction history (no memorization)"""
    
    def __init__(self, z_dim, trait_dim=2):
        super().__init__()
        # Project aggregated context to trait space
        self.trait_proj = nn.Linear(z_dim, trait_dim)
        nn.init.normal_(self.trait_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.trait_proj.bias)
        
    def forward(self, z_context):
        """
        Compute student traits from context
        
        Args:
            z_context: [BS, seqlen, z_dim] - Transformer context vectors
        
        Returns:
            delta: [BS, 2] - student traits [δ_L0, δ_T]
        """
        # Aggregate across sequence (mean pooling)
        z_agg = z_context.mean(dim=1)  # [BS, z_dim]
        
        # Project to 2D trait space
        delta = self.trait_proj(z_agg)  # [BS, 2]
        
        return delta


class GTransformerV2(nn.Module):
    def __init__(self, ...):
        # ... (previous code)
        
        # Student trait encoder (context-based)
        z_dim = 2 * d_model  # [d_output || q_embed]
        self.student_trait_encoder = StudentTraitEncoder(z_dim, trait_dim=2)
```

### 3. Per-Skill Residual Projections (From Context)

```python
class GTransformerV2(nn.Module):
    def __init__(self, ...):
        # ... (previous code)
        
        # Project z_context to skill-specific residuals
        z_dim = 2 * d_model
        self.residual_L0_proj = nn.Linear(z_dim, 1)
        self.residual_T_proj = nn.Linear(z_dim, 1)
        
        # Initialize with small weights
        nn.init.normal_(self.residual_L0_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.residual_T_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residual_L0_proj.bias)
        nn.init.zeros_(self.residual_T_proj.bias)
```

## Forward Pass

```python
def forward(self, q_data, target, uid_data=None, pid_data=None, qtest=False):
    """
    Forward pass with three-term decomposition
    
    Args:
        q_data: [BS, seqlen] skill IDs
        target: [BS, seqlen] correctness
        uid_data: [BS] student IDs (ONLY for PCA loss, not used in forward)
        pid_data: [BS, seqlen] problem IDs (optional)
    
    Returns:
        outputs: dict with predictions and parameters
        reg_loss: regularization loss
    """
    BS, seqlen = q_data.size()
    
    # ========================================
    # Encoder-Decoder: Get z_context
    # ========================================
    q_embed_data, qa_embed_data = self.base_emb(q_data, target)
    
    if self.n_pid > 0:
        q_embed_diff_data = self.q_embed_diff(q_data)
        pid_embed_data = self.difficult_param(pid_data)
        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
        
        qa_embed_diff_data = self.qa_embed_diff(target)
        qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
        c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2_rasch
    else:
        c_reg_loss = 0.
    
    # Transformer encoder-decoder
    d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
    
    # Context vector: [BS, seqlen, z_dim]
    z_context = torch.cat([d_output, q_embed_data], dim=-1)
    
    # ========================================
    # Component 1: Population-Level (μ)
    # ========================================
    mu_L0_q = self.mu_L0[q_data]  # [BS, seqlen]
    mu_T_q = self.mu_T[q_data]    # [BS, seqlen]
    
    # ========================================
    # Component 2: Per-Student Traits (δ) - FROM CONTEXT!
    # ========================================
    student_traits_2d = self.student_trait_encoder(z_context)  # [BS, 2]
    
    # Expand to sequence length
    delta_L0 = student_traits_2d[:, 0].unsqueeze(1).expand(-1, seqlen)  # [BS, seqlen]
    delta_T = student_traits_2d[:, 1].unsqueeze(1).expand(-1, seqlen)   # [BS, seqlen]
    
    # ========================================
    # Component 3: Per-Skill Residuals (ε)
    # ========================================
    epsilon_L0 = self.residual_L0_proj(z_context).squeeze(-1)  # [BS, seqlen]
    epsilon_T = self.residual_T_proj(z_context).squeeze(-1)    # [BS, seqlen]
    
    # ========================================
    # Combine: Three-Term Decomposition
    # ========================================
    L0_logits = mu_L0_q + delta_L0 + epsilon_L0
    T_logits = mu_T_q + delta_T + epsilon_T
    
    # Convert to probabilities
    p_L0 = torch.sigmoid(L0_logits)
    p_T = torch.sigmoid(T_logits)
    
    # ========================================
    # Standard supervised prediction
    # ========================================
    output = self.out(z_context).squeeze(-1)
    preds = torch.sigmoid(output)
    
    # ========================================
    # BKT Reference Output
    # ========================================
    ref_preds = self._bkt_ref_output(q_data, target, p_L0, p_T)
    
    outputs = {
        'predictions': preds,
        'p_L0': p_L0,
        'p_T': p_T,
        'reference_preds': ref_preds,
        'student_traits': student_traits_2d,  # [BS, 2] - for PCA loss
        'epsilon_L0': epsilon_L0,
        'epsilon_T': epsilon_T,
    }
    
    if not qtest:
        return outputs, c_reg_loss
    else:
        return outputs, c_reg_loss, z_context
```

## Deriving Student-Specific Parameters from BKT

Before we can compute the PCA reference, we need to understand how to derive **student-specific** `p_L0_bkt` and `p_T_bkt` values from BKT's forward inference.

### BKT Background

Standard BKT has **4 parameters per skill** (population-level):
- **P(L0)**: Prior knowledge (probability student knows skill initially)
- **P(T)**: Learning rate (probability of learning on each practice)
- **P(G)**: Guess rate (probability of correct answer when don't know)
- **P(S)**: Slip rate (probability of incorrect answer when do know)

### Key Insight

While BKT learns **population-level parameters**, during **forward inference** it tracks **per-student mastery** `P(L_t)` at each timestep. We can use this mastery trajectory to derive **student-specific effective parameters**.

### Step 1: Train BKT (Get Population Parameters)

```python
from pykt.models import BKT

# Train BKT on all student data
bkt_model = BKT(n_skills=n_question)
bkt_model.fit(train_data)

# Result: Per-skill population parameters
skill_params = bkt_model.get_skill_params()
# {
#     'skill_1': {'prior': 0.3, 'learns': 0.1, 'guess': 0.2, 'slip': 0.1},
#     'skill_2': {'prior': 0.5, 'learns': 0.15, 'guess': 0.2, 'slip': 0.1},
#     ...
# }
```

### Step 2: Forward Inference (Track Per-Student Mastery)

For each student's interaction sequence, BKT computes `P(L_t)` - probability of mastery at time `t`:

```python
def bkt_forward_inference(interactions, skill_params):
    """
    Run BKT forward inference to track mastery over time
    
    Args:
        interactions: List of (correct/incorrect) for one student-skill pair
        skill_params: Dict with {'prior', 'learns', 'guess', 'slip'}
    
    Returns:
        mastery_trajectory: List of P(L_t) values
    """
    P_L0 = skill_params['prior']
    P_T = skill_params['learns']
    P_G = skill_params['guess']
    P_S = skill_params['slip']
    
    mastery_trajectory = [P_L0]
    P_L_t = P_L0
    
    for correct in interactions:
        # Bayesian update based on observation
        if correct == 1:
            # Correct answer: update belief
            numerator = P_L_t * (1 - P_S)
            denominator = P_L_t * (1 - P_S) + (1 - P_L_t) * P_G
            P_L_t = numerator / denominator
        else:
            # Incorrect answer: update belief
            numerator = P_L_t * P_S
            denominator = P_L_t * P_S + (1 - P_L_t) * (1 - P_G)
            P_L_t = numerator / denominator
        
        # Apply learning (student might have learned from this interaction)
        P_L_t = P_L_t + (1 - P_L_t) * P_T
        
        mastery_trajectory.append(P_L_t)
    
    return mastery_trajectory
```

### Step 3: Derive Student-Specific Parameters

From the mastery trajectory, extract student-specific effective parameters:

```python
def derive_student_params(mastery_trajectory, skill_params):
    """
    Derive student-specific P(L0) and P(T) from mastery trajectory
    
    Args:
        mastery_trajectory: List of P(L_t) from forward inference
        skill_params: Population parameters
    
    Returns:
        p_L0_student: Student's effective prior knowledge
        p_T_student: Student's effective learning rate
    """
    # P(L0): Use initial mastery
    p_L0_student = mastery_trajectory[0]
    
    # P(T): Estimate from mastery growth
    if len(mastery_trajectory) > 1:
        # Method 1: Average learning gain per interaction
        learning_gains = []
        for t in range(len(mastery_trajectory) - 1):
            P_L_prev = mastery_trajectory[t]
            P_L_next = mastery_trajectory[t + 1]
            
            # Solve for P(T) from: P(L_next) = P(L_prev) + (1 - P(L_prev)) * P(T)
            if P_L_prev < 1.0:
                p_T_implied = (P_L_next - P_L_prev) / (1 - P_L_prev)
                # Clip to [0, 1] (can be negative due to Bayesian updates)
                p_T_implied = max(0.0, min(1.0, p_T_implied))
                learning_gains.append(p_T_implied)
        
        if learning_gains:
            p_T_student = np.mean(learning_gains)
        else:
            p_T_student = skill_params['learns']  # Fallback to population
    else:
        # Only one interaction: use population parameter
        p_T_student = skill_params['learns']
    
    return p_L0_student, p_T_student
```

### Step 4: Generate BKT Predictions DataFrame

Process all students and skills to create the predictions DataFrame:

```python
def generate_bkt_predictions(data, bkt_model):
    """
    Generate per-student, per-skill BKT predictions
    
    Args:
        data: DataFrame with columns [uid, skill, correct, ...]
        bkt_model: Trained BKT model
    
    Returns:
        bkt_predictions_df: DataFrame with [uid, skill, p_L0_bkt, p_T_bkt]
    """
    skill_params = bkt_model.get_skill_params()
    predictions = []
    
    # Group by student and skill
    for (uid, skill), group in data.groupby(['uid', 'skill']):
        # Get interaction sequence for this student-skill pair
        interactions = group['correct'].values
        
        if len(interactions) > 0:
            # Get population parameters for this skill
            params = skill_params.get(skill, skill_params.get(str(skill), {}))
            
            # Run forward inference
            mastery_trajectory = bkt_forward_inference(interactions, params)
            
            # Derive student-specific parameters
            p_L0_bkt, p_T_bkt = derive_student_params(mastery_trajectory, params)
            
            predictions.append({
                'uid': uid,
                'skill': skill,
                'p_L0_bkt': p_L0_bkt,
                'p_T_bkt': p_T_bkt,
                'n_interactions': len(interactions)
            })
    
    return pd.DataFrame(predictions)
```

### Complete Example

```python
# Example: Student 1, Skill 1
# Interactions: [1, 1, 0, 1, 1] (1=correct, 0=incorrect)
# Population params: P(L0)=0.3, P(T)=0.1, P(G)=0.2, P(S)=0.1

interactions = [1, 1, 0, 1, 1]
params = {'prior': 0.3, 'learns': 0.1, 'guess': 0.2, 'slip': 0.1}

# Forward inference:
P_L_0 = 0.30  # Initial

# t=1 (correct):
# Update: P_L_1 = 0.30 * 0.9 / (0.30 * 0.9 + 0.70 * 0.2) = 0.66
# Learn:  P_L_1 = 0.66 + (1 - 0.66) * 0.1 = 0.69

# t=2 (correct):
# Update: P_L_2 = 0.69 * 0.9 / (0.69 * 0.9 + 0.31 * 0.2) = 0.91
# Learn:  P_L_2 = 0.91 + (1 - 0.91) * 0.1 = 0.92

# t=3 (incorrect):
# Update: P_L_3 = 0.92 * 0.1 / (0.92 * 0.1 + 0.08 * 0.8) = 0.59
# Learn:  P_L_3 = 0.59 + (1 - 0.59) * 0.1 = 0.63

# t=4 (correct):
# Update: P_L_4 = 0.63 * 0.9 / (0.63 * 0.9 + 0.37 * 0.2) = 0.88
# Learn:  P_L_4 = 0.88 + (1 - 0.88) * 0.1 = 0.89

# Mastery trajectory: [0.30, 0.69, 0.92, 0.63, 0.89]

# Derive student parameters:
p_L0_bkt = 0.30  # Initial mastery

# Estimate P(T):
# t=0→1: (0.69 - 0.30) / (1 - 0.30) = 0.56
# t=1→2: (0.92 - 0.69) / (1 - 0.69) = 0.74
# t=2→3: (0.63 - 0.92) / (1 - 0.92) = -3.6 (negative due to incorrect answer, clip to 0)
# t=3→4: (0.89 - 0.63) / (1 - 0.63) = 0.70
# Average: (0.56 + 0.74 + 0 + 0.70) / 4 = 0.50

p_T_bkt = 0.50  # Student learns much faster than population (0.1)!

# Result for this student-skill pair:
# p_L0_bkt = 0.30 (same as population)
# p_T_bkt = 0.50 (5x faster than population!)
```

### Integration with pykt-toolkit

In practice, use the existing BKT implementation:

```python
from pykt.models import train_model
from pykt.datasets import load_data

# 1. Load data
train_data, valid_data, test_data = load_data('assist2009')

# 2. Train BKT
bkt_config = {
    'model_name': 'bkt',
    'n_question': n_question,
    # ... other config
}
bkt_model = train_model(bkt_config, train_data, valid_data)

# 3. Generate predictions
bkt_predictions_df = generate_bkt_predictions(train_data, bkt_model)

# 4. Save for PCA computation
bkt_predictions_df.to_csv('bkt_student_predictions.csv', index=False)
```

### Why This Matters

These student-specific `p_L0_bkt` and `p_T_bkt` values capture:
- **Individual differences**: Students with same skill show different prior knowledge and learning rates
- **Natural clustering**: Students naturally group into clusters (high/low L0 × fast/slow learners)
- **Grounding signal**: Provides theory-based targets for gTransformer to learn from

## PCA Reference Computation

Now that we have `bkt_predictions_df` with student-specific parameters, we can compute the PCA reference:

```python
def compute_pca_reference(bkt_predictions_df, bkt_skill_params, n_uid, n_question):
    """
    Compute PCA reference from BKT predictions
    
    This provides a target for the PCA alignment loss, ensuring that
    learned student traits align with BKT-derived cluster structure.
    
    Args:
        bkt_predictions_df: DataFrame with [uid, skill, p_L0_bkt, p_T_bkt]
        bkt_skill_params: BKT population parameters
        n_uid: Number of unique students
        n_question: Number of unique skills
    
    Returns:
        pca_reference: [n_uid, 2] PCA coordinates
        pca_components: [2, 2] principal components
        pca_mean: [2] mean of deviations
        cluster_labels: [n_uid] cluster assignments (0-3)
        explained_variance: [2] variance explained by each PC
    """
    from sklearn.decomposition import PCA
    import numpy as np
    
    # Extract population-level parameters
    mu_L0 = np.zeros(n_question + 1)
    mu_T = np.zeros(n_question + 1)
    
    params_dict = bkt_skill_params.get('params', {})
    for q_idx in range(n_question + 1):
        s_params = params_dict.get(q_idx, params_dict.get(str(q_idx), {}))
        mu_L0[q_idx] = s_params.get('prior', 0.5)
        mu_T[q_idx] = s_params.get('learns', 0.1)
    
    # Compute per-student average deviations from BKT
    delta_L0_ref = np.zeros(n_uid + 1)
    delta_T_ref = np.zeros(n_uid + 1)
    
    for uid in range(1, n_uid + 1):
        student_data = bkt_predictions_df[bkt_predictions_df['uid'] == uid]
        
        if len(student_data) > 0:
            skills = student_data['skill'].values
            p_L0_student = student_data['p_L0_bkt'].values
            p_T_student = student_data['p_T_bkt'].values
            
            # Average deviation from population mean
            delta_L0_ref[uid] = np.mean(p_L0_student - mu_L0[skills])
            delta_T_ref[uid] = np.mean(p_T_student - mu_T[skills])
    
    # Stack into [n_uid-1, 2] array
    student_deviations = np.stack([delta_L0_ref[1:], delta_T_ref[1:]], axis=1)
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(student_deviations)
    
    # Extract PCA parameters
    pca_components = pca.components_
    pca_mean = pca.mean_
    explained_variance = pca.explained_variance_ratio_
    
    print(f"\nPCA Analysis:")
    print(f"  PC1 explains {explained_variance[0]*100:.1f}% of variance")
    print(f"  PC2 explains {explained_variance[1]*100:.1f}% of variance")
    
    # Add back uid=0 (padding)
    pca_reference = np.vstack([np.zeros((1, 2)), pca_coords])
    
    # Assign cluster labels (4 quadrants)
    median_pc1 = np.median(pca_coords[:, 0])
    median_pc2 = np.median(pca_coords[:, 1])
    
    cluster_labels = np.zeros(n_uid + 1, dtype=int)
    for uid in range(1, n_uid + 1):
        pc1, pc2 = pca_coords[uid-1]
        cluster_labels[uid] = (0 if pc1 < median_pc1 else 2) + (0 if pc2 < median_pc2 else 1)
    
    return (
        torch.FloatTensor(pca_reference),
        torch.FloatTensor(pca_components),
        torch.FloatTensor(pca_mean),
        torch.LongTensor(cluster_labels),
        explained_variance
    )
```

## Loss Functions

### PCA Alignment Loss (Modified Option 2: Balanced Approach)

We use a **balanced combination** of direct alignment and structure preservation to achieve:
- ✅ **Interpretable axes** (aligned with PCA components)
- ✅ **Good supervised AUC** (flexible for predictions)
- ✅ **Clear visualization** (consistent cluster structure)

```python
def pca_alignment_loss_balanced(student_traits_batch, pca_reference, uid_batch,
                                 alpha=0.2, beta=0.8):
    """
    Balanced PCA alignment loss: weak direct alignment + strong structure preservation
    
    This approach provides interpretable axes (PC1 ≈ ability, PC2 ≈ learning rate)
    while maintaining flexibility for the model to optimize predictions.
    
    Design rationale:
    - alpha=0.2: Weak alignment keeps axes interpretable without over-constraining
    - beta=0.8: Strong structure preservation allows flexibility for good AUC
    
    Args:
        student_traits_batch: [BS, 2] - traits computed from z_context
        pca_reference: [n_uid, 2] - PCA coordinates from BKT
        uid_batch: [BS] - student IDs (for reference lookup)
        alpha: weight for direct alignment (0.1-0.3 recommended)
        beta: weight for structure preservation (0.7-0.9 recommended)
    
    Returns:
        L_pca: scalar loss
        loss_dict: dict with individual components for monitoring
    """
    # Get reference traits for students in batch
    pca_ref_batch = pca_reference[uid_batch]  # [BS, 2]
    
    # Component 1: Weak direct alignment (for interpretable axes)
    # This ensures PC1 ≈ ability, PC2 ≈ learning rate
    L_mse = F.mse_loss(student_traits_batch, pca_ref_batch)
    
    # Component 2: Strong structure preservation (for prediction flexibility)
    # This preserves relative distances between students (cluster structure)
    D_learned = torch.cdist(student_traits_batch, student_traits_batch, p=2)  # [BS, BS]
    D_ref = torch.cdist(pca_ref_batch, pca_ref_batch, p=2)  # [BS, BS]
    L_pairwise = F.mse_loss(D_learned, D_ref)
    
    # Balanced combination
    L_pca = alpha * L_mse + beta * L_pairwise
    
    # Return components for monitoring
    loss_dict = {
        'L_pca_mse': L_mse.item(),
        'L_pca_pairwise': L_pairwise.item(),
    }
    
    return L_pca, loss_dict
```

### Integration into Model

Add the PCA loss computation to your model:

```python
class GTransformerV2(nn.Module):
    def __init__(self, ...):
        # ... (previous code)
        
        # Store PCA reference and components
        self.register_buffer('pca_reference', torch.zeros(n_uid + 1, 2))
        self.register_buffer('pca_components', torch.eye(2))
        self.register_buffer('pca_mean', torch.zeros(2))
        
    def compute_pca_loss(self, student_traits_batch, uid_batch, 
                         alpha=0.2, beta=0.8):
        """
        Compute PCA alignment loss with balanced objectives
        
        Args:
            student_traits_batch: [BS, 2] - computed from context
            uid_batch: [BS] - student IDs
            alpha: weight for direct alignment (default: 0.2)
            beta: weight for structure preservation (default: 0.8)
        
        Returns:
            L_pca: scalar loss
            loss_dict: dict with components
        """
        if uid_batch is None or len(uid_batch) == 0:
            return torch.tensor(0.0, device=student_traits_batch.device), {}
        
        return pca_alignment_loss_balanced(
            student_traits_batch,
            self.pca_reference,
            uid_batch,
            alpha=alpha,
            beta=beta
        )
```

### Residual Regularization

Prevent residuals from dominating the parameter decomposition:

```python
def residual_regularization_loss(epsilon_L0, epsilon_T):
    """
    Regularize residuals to prevent overfitting
    
    Args:
        epsilon_L0: [BS, seqlen] - L0 residuals
        epsilon_T: [BS, seqlen] - T residuals
    
    Returns:
        L_residual: scalar loss
    """
    L_residual = epsilon_L0.pow(2).mean() + epsilon_T.pow(2).mean()
    return L_residual
```

### Total Multi-Objective Loss

Combine all loss components with appropriate weights:

```python
def compute_total_loss(outputs, targets, uid_batch, model, config):
    """
    Compute total multi-objective loss
    
    Loss components (in priority order):
    1. L_sup: Supervised prediction (HIGHEST PRIORITY - optimize AUC)
    2. L_ref: BKT reference alignment (moderate priority)
    3. L_pca: PCA cluster coherence (low priority - grounding only)
    4. L_residual: Residual regularization (very low priority)
    
    Args:
        outputs: dict from model forward pass
        targets: ground truth labels [BS, seqlen]
        uid_batch: student IDs [BS]
        model: GTransformerV2 instance
        config: loss weight configuration
    
    Returns:
        total_loss: scalar
        loss_dict: dict with individual loss components
    """
    # 1. Supervised prediction loss (MAIN OBJECTIVE)
    L_sup = F.binary_cross_entropy(
        outputs['predictions'], 
        targets.float()
    )
    
    # 2. Reference output loss (BKT logic wrapper)
    L_ref = F.binary_cross_entropy(
        outputs['reference_preds'],
        targets.float()
    )
    
    # 3. PCA cluster coherence loss (BALANCED GROUNDING)
    if uid_batch is not None:
        L_pca, pca_loss_dict = model.compute_pca_loss(
            outputs['student_traits'],
            uid_batch,
            alpha=config.get('pca_alpha', 0.2),
            beta=config.get('pca_beta', 0.8)
        )
    else:
        L_pca = torch.tensor(0.0, device=outputs['predictions'].device)
        pca_loss_dict = {}
    
    # 4. Residual regularization
    L_residual = residual_regularization_loss(
        outputs['epsilon_L0'],
        outputs['epsilon_T']
    )
    
    # Combine with weights (prioritize supervised AUC)
    total_loss = (
        config['lambda_sup'] * L_sup +
        config['lambda_ref'] * L_ref +
        config['lambda_pca'] * L_pca +
        config['lambda_residual'] * L_residual
    )
    
    # Detailed loss dictionary for monitoring
    loss_dict = {
        'loss': total_loss.item(),
        'L_sup': L_sup.item(),
        'L_ref': L_ref.item(),
        'L_pca': L_pca.item() if isinstance(L_pca, torch.Tensor) else L_pca,
        'L_residual': L_residual.item(),
        **pca_loss_dict  # Include L_pca_mse and L_pca_pairwise
    }
    
    return total_loss, loss_dict
```

### Alternative: Option 5 (Hybrid with Cluster Compactness)

**When to use**: Only if Modified Option 2 produces clusters that are too loose or overlapping.

```python
def pca_alignment_loss_hybrid(student_traits_batch, pca_reference, 
                               cluster_labels, uid_batch,
                               alpha=0.5, beta=0.3, gamma=0.2):
    """
    Hybrid loss with cluster compactness (Option 5)
    
    Use this ONLY if Modified Option 2 doesn't produce tight enough clusters.
    Warning: May reduce AUC due to over-constraint.
    
    Args:
        student_traits_batch: [BS, 2]
        pca_reference: [n_uid, 2]
        cluster_labels: [n_uid] - cluster assignments (0-3)
        uid_batch: [BS]
        alpha: direct alignment weight
        beta: structure preservation weight
        gamma: cluster compactness weight
    
    Returns:
        L_pca: scalar loss
        loss_dict: dict with components
    """
    pca_ref_batch = pca_reference[uid_batch]
    labels_batch = cluster_labels[uid_batch]
    
    # Component 1: Direct alignment
    L_mse = F.mse_loss(student_traits_batch, pca_ref_batch)
    
    # Component 2: Structure preservation
    D_learned = torch.cdist(student_traits_batch, student_traits_batch, p=2)
    D_ref = torch.cdist(pca_ref_batch, pca_ref_batch, p=2)
    L_pairwise = F.mse_loss(D_learned, D_ref)
    
    # Component 3: Cluster compactness
    L_compact = torch.tensor(0.0, device=student_traits_batch.device)
    for cluster_id in range(4):
        mask = labels_batch == cluster_id
        if mask.sum() > 1:
            cluster_traits = student_traits_batch[mask]
            center = cluster_traits.mean(dim=0)
            L_compact += ((cluster_traits - center) ** 2).mean()
    L_compact = L_compact / 4
    
    # Combined loss
    L_pca = alpha * L_mse + beta * L_pairwise + gamma * L_compact
    
    loss_dict = {
        'L_pca_mse': L_mse.item(),
        'L_pca_pairwise': L_pairwise.item(),
        'L_pca_compact': L_compact.item(),
    }
    
    return L_pca, loss_dict
```

**When to try Option 5**:
1. After training with Modified Option 2, visualize student clusters
2. If clusters are too loose or overlapping significantly
3. If interpretability is more important than maximizing AUC
4. Be prepared to tune 3 hyperparameters (α, β, γ) instead of 2

**Recommended transition**:
```python
# Start with Modified Option 2
config = {'pca_alpha': 0.2, 'pca_beta': 0.8}

# If clusters are loose, try Option 5
config = {'pca_alpha': 0.5, 'pca_beta': 0.3, 'pca_gamma': 0.2}
```

## Training Configuration

### Recommended Hyperparameters (Modified Option 2)

```python
config = {
    # Model architecture
    'd_model': 64,
    'n_blocks': 2,
    'num_attn_heads': 8,
    'd_ff': 256,
    'dropout': 0.1,
    
    # Loss weights (PRIORITIZE SUPERVISED AUC)
    'lambda_sup': 1.0,        # Supervised prediction (HIGHEST)
    'lambda_ref': 0.3,        # BKT reference alignment (moderate)
    'lambda_pca': 0.1,        # PCA cluster coherence (LOW - grounding only)
    'lambda_residual': 0.01,  # Residual regularization (very low)
    
    # PCA loss balance (INTERPRETABILITY + FLEXIBILITY)
    'pca_alpha': 0.2,         # Weak alignment (interpretable axes)
    'pca_beta': 0.8,          # Strong structure (prediction flexibility)
    
    # Training
    'learning_rate': 0.0001,
    'batch_size': 64,
    'max_epochs': 100,
}
```

### Tuning Guide

**If AUC is too low:**
```python
# Reduce PCA constraint
config['lambda_pca'] = 0.05  # Was 0.1
config['pca_alpha'] = 0.1    # Was 0.2
```

**If axes are not interpretable:**
```python
# Increase direct alignment
config['pca_alpha'] = 0.3    # Was 0.2
# Keep beta high
config['pca_beta'] = 0.8
```

**If clusters are too loose:**
```python
# Increase PCA weight
config['lambda_pca'] = 0.15  # Was 0.1
# Or try Option 5 (see above)

```

## Benefits of Context-Based Traits

### 1. No Memorization
- ✅ Traits computed from behavior, not student IDs
- ✅ Model learns patterns, not identities
- ✅ No overfitting to specific students

### 2. Generalization
- ✅ Works with new students (no retraining)
- ✅ Transfers across datasets
- ✅ Learns from interaction patterns

### 3. Scalability
- ✅ No per-student parameters
- ✅ Memory doesn't grow with students
- ✅ Constant model size

### 4. Interpretability
- ✅ Can still visualize in 2D PCA space
- ✅ Traits reflect actual behavior
- ✅ PCA grounding ensures meaningful clusters

## Training Configuration

```python
config = {
    # Model architecture
    'd_model': 64,
    'n_blocks': 2,
    'num_attn_heads': 8,
    'd_ff': 256,
    'dropout': 0.1,
    
    # Loss weights
    'lambda_sup': 1.0,        # Supervised prediction
    'lambda_ref': 0.5,        # BKT reference alignment
    'lambda_pca': 0.1,        # PCA cluster coherence
    'lambda_residual': 0.01,  # Residual regularization
    
    # Training
    'learning_rate': 0.0001,
    'batch_size': 64,
    'max_epochs': 100,
}
```

## Comparison: Context-Based vs Static Embeddings

| Aspect | Static Embeddings | **Context-Based (Ours)** |
|--------|-------------------|--------------------------|
| **Memorization** | ❌ Memorizes student IDs | ✅ Learns from behavior |
| **New students** | ❌ Can't handle | ✅ Works immediately |
| **Scalability** | ❌ Grows with students | ✅ Constant size |
| **Overfitting** | ❌ High risk | ✅ Lower risk |
| **Interpretability** | ⚠️ Static traits | ✅ Dynamic, context-aware |
| **Generalization** | ❌ Poor | ✅ Good |

## References

- Principal Component Analysis: Pearson (1901), Hotelling (1933)
- Bayesian Knowledge Tracing: Corbett & Anderson (1994)
- Deep Knowledge Tracing: Piech et al. (2015)
- Interpretable ML: Rudin (2019)

## PCA-Based Loss Function: Mathematical Formulation

Using a combination of MSE and Pairwise Distance. MSE keeps the axes aligned (Rotation invariance fix), while Pairwise Distance allows the model to find "better" clusters that theory might miss, as long as the relative topology of the student population is preserved.

### High-Level Formula

The PCA alignment loss combines two objectives:

```
L_pca = α · L_mse + β · L_pairwise

where:
  α = 0.2  (weak alignment weight)
  β = 0.8  (strong structure weight)
```

### Component 1: Direct Alignment (L_mse)

Aligns learned traits with PCA reference coordinates:

```
L_mse = MSE(δ_learned, δ_ref)

      = (1/BS) · Σ(i=1 to BS) ||δ_learned[i] - δ_ref[i]||²

      = (1/BS) · Σ(i=1 to BS) [(δ_L0_learned[i] - δ_L0_ref[i])² + 
                                (δ_T_learned[i] - δ_T_ref[i])²]

where:
  δ_learned[i] ∈ ℝ² : Learned student traits (computed from z_context)
  δ_ref[i] ∈ ℝ²    : PCA reference traits (from BKT analysis)
  BS               : Batch size
```

**Purpose**: Ensures axes remain interpretable (PC1 ≈ ability, PC2 ≈ learning rate)

### Component 2: Structure Preservation (L_pairwise)

Preserves relative distances between students:

```
L_pairwise = MSE(D_learned, D_ref)

           = (1/BS²) · Σ(i=1 to BS) Σ(j=1 to BS) (D_learned[i,j] - D_ref[i,j])²

where:
  D_learned[i,j] = ||δ_learned[i] - δ_learned[j]||₂  : Euclidean distance in learned space
  D_ref[i,j]     = ||δ_ref[i] - δ_ref[j]||₂          : Euclidean distance in reference space
```

**Expanded distance computation**:
```
D_learned[i,j] = √[(δ_L0_learned[i] - δ_L0_learned[j])² + 
                   (δ_T_learned[i] - δ_T_learned[j])²]

D_ref[i,j] = √[(δ_L0_ref[i] - δ_L0_ref[j])² + 
               (δ_T_ref[i] - δ_T_ref[j])²]
```

**Purpose**: Maintains cluster structure while allowing flexibility for prediction optimization

### Complete Formula

```
L_pca = 0.2 · (1/BS) · Σ(i=1 to BS) ||δ_learned[i] - δ_ref[i]||²
      + 0.8 · (1/BS²) · Σ(i=1 to BS) Σ(j=1 to BS) (||δ_learned[i] - δ_learned[j]||₂ - 
                                                     ||δ_ref[i] - δ_ref[j]||₂)²
```

### Concrete Example

Consider a batch of 3 students:

```python
# Learned traits (from model)
δ_learned = [
    [0.5, 0.3],   # Student 1: [δ_L0, δ_T]
    [-0.2, 0.8],  # Student 2
    [0.1, -0.4]   # Student 3
]

# PCA reference (from BKT)
δ_ref = [
    [0.6, 0.2],   # Student 1 reference
    [-0.3, 0.7],  # Student 2 reference
    [0.2, -0.3]   # Student 3 reference
]
```

**Step 1: Compute L_mse**
```
# Squared differences for each student
diff_1 = (0.5 - 0.6)² + (0.3 - 0.2)² = 0.01 + 0.01 = 0.02
diff_2 = (-0.2 - (-0.3))² + (0.8 - 0.7)² = 0.01 + 0.01 = 0.02
diff_3 = (0.1 - 0.2)² + (-0.4 - (-0.3))² = 0.01 + 0.01 = 0.02

L_mse = (0.02 + 0.02 + 0.02) / 3 = 0.02
```

**Step 2: Compute L_pairwise**
```
# Distance matrices
D_learned[1,2] = √[(0.5-(-0.2))² + (0.3-0.8)²] = √[0.49 + 0.25] = 0.86
D_learned[1,3] = √[(0.5-0.1)² + (0.3-(-0.4))²] = √[0.16 + 0.49] = 0.81
D_learned[2,3] = √[(-0.2-0.1)² + (0.8-(-0.4))²] = √[0.09 + 1.44] = 1.24

D_ref[1,2] = √[(0.6-(-0.3))² + (0.2-0.7)²] = √[0.81 + 0.25] = 1.03
D_ref[1,3] = √[(0.6-0.2)² + (0.2-(-0.3))²] = √[0.16 + 0.25] = 0.64
D_ref[2,3] = √[(-0.3-0.2)² + (0.7-(-0.3))²] = √[0.25 + 1.00] = 1.12

# Squared differences (symmetric, so count each pair once)
diff_12 = (0.86 - 1.03)² = 0.029
diff_13 = (0.81 - 0.64)² = 0.029
diff_23 = (1.24 - 1.12)² = 0.014

# Average over all pairs (including diagonal zeros)
L_pairwise = (0 + 0.029 + 0.029 + 0.029 + 0 + 0.014 + 0.029 + 0.014 + 0) / 9
           = 0.144 / 9 = 0.016
```

**Step 3: Combine**
```
L_pca = 0.2 × 0.02 + 0.8 × 0.016
      = 0.004 + 0.0128
      = 0.0168
```

### PyTorch Implementation

```python
def pca_alignment_loss_balanced(student_traits_batch, pca_reference, uid_batch,
                                 alpha=0.2, beta=0.8):
    """
    Balanced PCA alignment loss
    
    Args:
        student_traits_batch: [BS, 2] - learned traits
        pca_reference: [n_uid, 2] - PCA reference
        uid_batch: [BS] - student IDs
        alpha: weight for L_mse (default: 0.2)
        beta: weight for L_pairwise (default: 0.8)
    
    Returns:
        L_pca: scalar loss
        loss_dict: dict with components
    """
    # Get reference for batch
    pca_ref_batch = pca_reference[uid_batch]  # [BS, 2]
    
    # Component 1: Direct MSE
    L_mse = F.mse_loss(student_traits_batch, pca_ref_batch)
    # Equivalent to: ((student_traits_batch - pca_ref_batch) ** 2).mean()
    
    # Component 2: Pairwise distance MSE
    D_learned = torch.cdist(student_traits_batch, student_traits_batch, p=2)  # [BS, BS]
    D_ref = torch.cdist(pca_ref_batch, pca_ref_batch, p=2)  # [BS, BS]
    L_pairwise = F.mse_loss(D_learned, D_ref)
    # Equivalent to: ((D_learned - D_ref) ** 2).mean()
    
    # Balanced combination
    L_pca = alpha * L_mse + beta * L_pairwise
    
    return L_pca, {
        'L_pca_mse': L_mse.item(),
        'L_pca_pairwise': L_pairwise.item()
    }
```

### Intuition

**L_mse (20% weight)**:
- **What it does**: Pulls learned traits toward PCA coordinates
- **Effect**: Keeps axes aligned (PC1 ≈ ability, PC2 ≈ learning rate)
- **Why weak (α=0.2)**: Provides guidance without over-constraining
- **Allows**: Model flexibility to optimize for AUC

**L_pairwise (80% weight)**:
- **What it does**: Preserves relative distances between students
- **Effect**: Maintains cluster structure without forcing exact positions
- **Why strong (β=0.8)**: Primary objective is structure preservation
- **Allows**: Rotation/translation of learned space while preserving clusters

**Combined Effect**:
- ✅ Axes stay interpretable (from L_mse)
- ✅ Model has flexibility to optimize AUC (from L_pairwise dominance)
- ✅ Cluster structure is preserved (from L_pairwise)
- ✅ Best balance between interpretability and performance

### Computational Complexity

- **L_mse**: O(BS) - linear in batch size
- **L_pairwise**: O(BS²) - quadratic in batch size
- **Total**: O(BS²) - dominated by pairwise distance computation

**Optimization tip**: For large batches (BS > 128), consider:
1. Using smaller batches for PCA loss computation
2. Computing L_pca on a subset of the batch
3. Using approximate distance methods


## Suggestions for Refinement

### 1. Upgraded Architecture (v2.1 Attention Pooling)
The current approach uses `delta = proj(mean(z))` to aggregate history into student traits. It makes the "Student Traits" ($\delta$) much more robust to interaction noise. If a student guesses correctly on a hard question (an "outlier"), the Trait Aggregator can learn to discount that interaction, whereas the Simple Mean would be forced to shift the entire student profile toward higher mastery, creating "flicker" in the diagnostics.

#### Evolutionary Roadmap:
*   **GTransformer v2.0 (Arithmetic Mean)**:
    *   `aggregation = mean(z)`
    *   Pedagogy: All interactions are equally important for trait estimation.
*   **GTransformer v2.1 (Intelligence Pooling Head)**:
    *   `aggregation = AttentionPool(z)` using a learnable **Trait Query** ($Q_{trait}$).
    *   Pedagogy: Specific "temporal signatures" (e.g., mastery shifts) are prioritized for diagnostic grounding.

#### Technical Implementation Details:
1.  **Trait Query ($Q_{\text{trait}}$)**: Initialize a learnable parameter `self.trait_query = nn.Parameter(torch.randn(1, 1, z_dim))`.
2.  **Specialized Attention Head**: For a batch of student histories $H \in \mathbb{R}^{BS \times T \times z\_dim}$:
    *   Compute attention weights: $\alpha = \text{Softmax}\left(\frac{Q_{\text{trait}} \cdot (W_K H)^\top}{\sqrt{d}}\right)$
    *   $\alpha \in \mathbb{R}^{BS \times 1 \times T}$ represents the "diagnostic relevance" of each timestep.
3.  **Weighted Aggregation**: The final trait vector $\mathbf{z}_{\text{student}} = \sum_{t=1}^T \alpha_t (W_V z_t)$.
4.  **Benefits**: This "summarizes" the interaction sequence by prioritizing high-information moments over interaction noise, producing a cleaner, more stable input for the PCA grounding loss.

### 2. The "Parsimony Ratio" Rule for $\epsilon$
To maintain the **Grounded-to-Heuristic Ratio** and prevent "Interpretability Leakage," the weight of the Skill Residuals ($\lambda_{residual}$) must be mathematically anchored to the PCA grounding weight ($\lambda_{pca}$). If $\epsilon$ is too unconstrained, the model may "cheat" by pushing all information into the residual term to maximize supervised AUC, effectively ignoring the grounded traits ($\delta$).

*   **Heuristic**: Keep $\lambda_{residual}$ at approximately **10% of $\lambda_{pca}$**.
*   **Formula**: $\lambda_{residual} = k \cdot \lambda_{pca} \quad \text{where } k \approx 0.1$
*   **Contribution Hierarchy**: In the composition $p = \sigma(\mu + \delta + \epsilon)$, the influence should follow: **Fixed Theory ($\mu$)** $\rightarrow$ **Grounded Identity ($\delta$)** $\rightarrow$ **Contextual Nuance ($\epsilon$)**.
*   **Warning**: If $\lambda_{residual} \ge \lambda_{pca}$, the model will likely collapse the 2D trait space into a single point and use $\epsilon$ for all personalization, reverting gTransformer into a black box.

### 3. Interpretation of the Axes (Calibration Phase)
By grounding the latent space to PCA-derived coordinates, we expect the resulting axes to capture stable pedagogical constructs. We recommend a **Calibration Phase** after training to empirically validate these semantics:

*   **PC1 (General Proficiency)**: This axis typically captures the student's initial mastery level. Validation: `Corr(PC1, GroundTruth_Initial_Correctness)`.
*   **PC2 (Learning Momentum)**: This axis often captures the effective learning rate or the student's response to interventions. Validation: `Corr(PC2, GroundTruth_Learning_Gain)`.
*   **Verification**: High correlations provide terminal proof that the multi-objective loss successfully forced the Transformer's latent representation to align with meaningful educational theory.

### 4. Design Rationale: Gradient Integrity
A critical design choice is placing the **Trait Aggregator** (Attention Pooling) *outside* the core Transformer blocks. This maximizes gradient efficiency while protecting the "canonical" attention heads.

#### 1. Additive vs. Disruptive Architecture
The Trait Aggregator acts as a **"Consumer Model"**. It consumes the temporal outputs $\{z_t\}_{1:T}$ of the Transformer without interfering with the internal self-attention or cross-attention scores that perform the sequence modeling. This ensures the model's primary predictive power remains intact and mathematically grounded in the DKT paradigm.

#### 2. Gradients as a "Guided Regularizer"
During the backwards pass, gradients flow from the diagnostic loss $\mathcal{L}_{pca} \rightarrow$ Trait Aggregator $\rightarrow$ Transformer Outputs. 
*   **The Effect**: This does not "break" the canonical heads; rather, it informs them. It encourages the Transformer to develop hidden representations that are not only accurate for prediction ($\mathcal{L}_{sup}$) but also mathematically "summarizable" for diagnosis.
*   **Stability**: The simultaneous optimization of $\mathcal{L}_{sup}$ and $\mathcal{L}_{pca}$ creates a stable Multi-Task Learning (MTL) environment where the Transformer finds a representation that satisfies both requirements: performance and interpretability.

#### 3. Gradient Magnitude Control (Low-Pass Filtering)
The $\lambda_{pca}$ coefficient ($0.1$) acts as a gradient low-pass filter. By ensuring the grounding signal is an order of magnitude smaller than the supervised signal, we prevent the "Tail from wagging the dog." The canonical attention heads remain primary driven by the sequence modeling task, with the PCA grounding actings as a secondary, structural bias.
