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

Ablation studies: 

In order to be able to do ablation studies, we'll use "ablation parameters" to control the application of the 3 terms μ, δ, and ε and analyze to what extent each one is responsible for the model's performance.

Modular PCA-based Loss Function: 

The calculation of L_pca should be implemented as an independent function in such a way that, in the future, we can easily modify it to test different loss functions types.

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

### 1. The "Parsimony Ratio" Rule for $\epsilon$
To maintain the **Grounded-to-Heuristic Ratio** and prevent "Interpretability Leakage," the weight of the Skill Residuals ($\lambda_{residual}$) must be mathematically anchored to the PCA grounding weight ($\lambda_{pca}$). If $\epsilon$ is too unconstrained, the model may "cheat" by pushing all information into the residual term to maximize supervised AUC, effectively ignoring the grounded traits ($\delta$).

*   **Heuristic**: Keep $\lambda_{residual}$ at approximately **10% of $\lambda_{pca}$**.
*   **Formula**: $\lambda_{residual} = k \cdot \lambda_{pca} \quad \text{where } k \approx 0.1$
*   **Contribution Hierarchy**: In the composition $p = \sigma(\mu + \delta + \epsilon)$, the influence should follow: **Fixed Theory ($\mu$)** $\rightarrow$ **Grounded Identity ($\delta$)** $\rightarrow$ **Contextual Nuance ($\epsilon$)**.
*   **Warning**: If $\lambda_{residual} \ge \lambda_{pca}$, the model will likely collapse the 2D trait space into a single point and use $\epsilon$ for all personalization, reverting gTransformer into a black box.

### 2. Interpretation of the Axes (Calibration Phase)
By grounding the latent space to PCA-derived coordinates, we expect the resulting axes to capture stable pedagogical constructs. We recommend a **Calibration Phase** after training to empirically validate these semantics:

*   **PC1 (General Proficiency)**: This axis typically captures the student's initial mastery level. Validation: `Corr(PC1, GroundTruth_Initial_Correctness)`.
*   **PC2 (Learning Momentum)**: This axis often captures the effective learning rate or the student's response to interventions. Validation: `Corr(PC2, GroundTruth_Learning_Gain)`.
*   **Verification**: High correlations provide terminal proof that the multi-objective loss successfully forced the Transformer's latent representation to align with meaningful educational theory.

### 3. Design Rationale: Gradient Integrity
A critical design choice is placing the **Trait Aggregator** (Attention Pooling) *outside* the core Transformer blocks. This maximizes gradient efficiency while protecting the "canonical" attention heads.

#### 1. Additive vs. Disruptive Architecture
The Trait Aggregator acts as a **"Consumer Model"**. It consumes the temporal outputs $\{z_t\}_{1:T}$ of the Transformer without interfering with the internal self-attention or cross-attention scores that perform the sequence modeling. This ensures the model's primary predictive power remains intact and mathematically grounded in the DKT paradigm.

#### 2. Gradients as a "Guided Regularizer"
During the backwards pass, gradients flow from the diagnostic loss $\mathcal{L}_{pca} \rightarrow$ Trait Aggregator $\rightarrow$ Transformer Outputs. 
*   **The Effect**: This does not "break" the canonical heads; rather, it informs them. It encourages the Transformer to develop hidden representations that are not only accurate for prediction ($\mathcal{L}_{sup}$) but also mathematically "summarizable" for diagnosis.
*   **Stability**: The simultaneous optimization of $\mathcal{L}_{sup}$ and $\mathcal{L}_{pca}$ creates a stable Multi-Task Learning (MTL) environment where the Transformer finds a representation that satisfies both requirements: performance and interpretability.

#### 3. Gradient Magnitude Control (Low-Pass Filtering)
The $\lambda_{pca}$ coefficient ($0.1$) acts as a gradient low-pass filter. By ensuring the grounding signal is an order of magnitude smaller than the supervised signal, we prevent the "Tail from wagging the dog." The canonical attention heads remain primary driven by the sequence modeling task, with the PCA grounding actings as a secondary, structural bias.

## Architectural Comparison: Current vs v2.0

### Current Implementation (arch_sections.d2)

**Section 4: Grounded Parameters (Two-Term Composition)**
```
TheoryBase (p_L0_base[q], p_T_base[q])  ← Learnable embeddings
    +
ContextualProjection (z · k_axis[q], z · v_axis[q])  ← Skill-specific axes
    ↓
FinalParams (p_L0 = σ(Base + Context), p_T = σ(Base + Context))
```

**Key characteristics:**
- Theory base uses **learnable embeddings** initialized from BKT
- Contextual component uses **skill-specific relational axes** (knowledge_axis, velocity_axis)
- **Two-term additive** composition: Base + Context
- Probe architecture for active grounding (L_probe loss)
- Optional student embeddings for personalization

### v2.0 Implementation (arch_pca.d2)

**Section 4: Grounded Parameters (Three-Term Decomposition)**
```
PopulationLevel (mu_L0[q], mu_T[q])  ← Fixed from BKT
    +
StudentTraits (delta = proj(mean(z)))  ← Per-student, context-based
    +
SkillResiduals (epsilon = proj(z))  ← Per-interaction
    ↓
FinalParams (p_L0 = σ(μ + δ + ε), p_T = σ(μ + δ + ε))
```

**Key characteristics:**
- Population level uses **fixed buffers** (not learnable)
- Student traits computed from **aggregated history** (mean pooling)
- Skill residuals from **current context** (per-timestep)
- **Three-term additive** composition: μ + δ + ε
- PCA grounding for student traits (L_pca loss)
- Parsimony regularization for residuals (L_residual loss)
- **No student embeddings** (context-based only)

### Section 5: Loss Function Changes

**Current:**
```
L_total = λ_sup·L_sup + λ_ref·L_ref + λ_probe·L_probe
```

**v2.0:**
```
L_total = λ_sup·L_sup + λ_ref·L_ref + λ_pca·L_pca + λ_residual·L_residual
```

**Changes:**
- Remove: `L_probe` (active grounding via probes)
- Add: `L_pca` (cluster coherence: 0.2·L_mse + 0.8·L_pairwise)
- Add: `L_residual` (parsimony regularization for ε)

### Output Heads Comparison

**Current outputs:**
```python
{
    'predictions': preds,           # Supervised MLP
    'p_l0': p_l0,                  # Grounded parameters
    'p_t': p_t,
    'p_l0_probe': p_l0_probe,      # Probe outputs (REMOVE)
    'p_t_probe': p_t_probe,        # Probe outputs (REMOVE)
    'reference_preds': ref_preds   # BKT logic wrapper
}
```

**v2.0 outputs:**
```python
{
    'predictions': preds,           # Supervised MLP
    'p_l0': p_l0,                  # Grounded parameters
    'p_t': p_t,
    'reference_preds': ref_preds,  # BKT logic wrapper
    'student_traits': traits,      # [BS, 2] for L_pca (NEW)
    'epsilon_l0': epsilon_l0,      # [BS, seqlen] for L_residual (NEW)
    'epsilon_t': epsilon_t,        # [BS, seqlen] for L_residual (NEW)
    'delta_l0': delta_l0,          # [BS, seqlen] for analysis (NEW)
    'delta_t': delta_t             # [BS, seqlen] for analysis (NEW)
}
```

### Key Architectural Transformations

| Aspect | Current | v2.0 | Rationale |
|--------|---------|------|-----------|
| **Population params** | Learnable embeddings | Fixed buffers | BKT provides ground truth, not initial guess |
| **Student personalization** | Skill-specific axes + optional embeddings | Context-based traits (δ) | No memorization, generalizes to new students |
| **Interaction context** | Implicit in axis projection | Explicit residuals (ε) | Clear separation of stable traits vs. noise |
| **Grounding mechanism** | Probe loss (L_probe) | PCA cluster loss (L_pca) | Aligns trait space to pedagogical dimensions |
| **Interpretability** | Axes hard to interpret | 2D PCA space (Proficiency, Momentum) | Clear pedagogical meaning |
| **Complexity** | 4 components (base, axes, probes, optional embeddings) | 3 components (μ, δ, ε) | Simpler, cleaner decomposition |

### Input Data Requirements

**IMPORTANT**: v2.0 does **NOT** require changes to the interaction-level input data format.

#### Current Input Data (Unchanged)

The model forward pass receives the same inputs as before:

```python
def forward(self, q_data, target, pid_data=None, uid_data=None, qtest=False):
    # q_data: [BS, seqlen] - Question/skill IDs
    # target: [BS, seqlen] - Response correctness (0/1)
    # pid_data: [BS, seqlen] - Problem IDs (optional, for Rasch)
    # uid_data: [BS] or [BS, seqlen] - Student IDs
    # qtest: bool - Test mode flag
```

**No changes needed** to:
- Interaction sequences (q_data, target)
- Problem IDs (pid_data)
- Student IDs (uid_data)
- Data loading pipelines
- Preprocessing scripts

#### New Requirement: PCA Reference File (Student-Level)

The **only** new data requirement is a **student-level** PCA reference file, loaded **once** at model initialization (not per-interaction).

**File format**: JSON file with student PCA coordinates

```json
{
  "student_coords": {
    "0": [0.234, -0.156],     # uid -> [PC1, PC2]
    "1": [-0.421, 0.089],
    "2": [0.156, 0.234],
    ...
  },
  "pca_components": [[...], [...]],  # Optional: PCA transformation matrix
  "pca_mean": [0.5, 0.1],            # Optional: PCA center
  "explained_variance_ratio": [0.68, 0.32]  # Optional: variance explained
}
```

**Generation**: Created by `examples/generate_pca_reference.py` (Step 8) from BKT forward inference results.

**Loading**: Called once during model initialization:

```python
# In training script initialization
model = GTransformer(...)
model.load_theory_params(bkt_skill_params)  # Existing

# NEW: Load PCA reference
with open(pca_reference_file, 'r') as f:
    pca_data = json.load(f)
model.load_pca_reference(pca_data)  # NEW method
```

**Storage**: Stored in model buffer `self.pca_reference[uid]` for fast lookup during training.

#### Why No Interaction-Level Changes?

1. **Population parameters (μ)**: Already loaded from BKT skill params (existing mechanism)
2. **Student traits (δ)**: Computed dynamically from `z_context` (no input needed)
3. **Skill residuals (ε)**: Computed dynamically from `z_context` (no input needed)
4. **PCA targets**: Student-level (not interaction-level), loaded once at init

#### Data Flow Comparison

**Current (no changes):**
```
Interaction Data (q, r, pid, uid)
    ↓
Embeddings
    ↓
Transformer (produces z_context)
    ↓
Parameter Computation (uses z_context + skill-specific axes)
```

**v2.0 (same input, different computation):**
```
Interaction Data (q, r, pid, uid)  ← SAME INPUT
    ↓
Embeddings
    ↓
Transformer (produces z_context)
    ↓
Parameter Computation:
  - μ: Lookup in bkt_l0_pop[q], bkt_t_pop[q]  ← From BKT (existing)
  - δ: proj(mean(z_context))  ← Computed from z_context
  - ε: proj(z_context)  ← Computed from z_context
```

#### Summary

| Data Type | Current | v2.0 | Change Required? |
|-----------|---------|------|------------------|
| Interaction sequences | (q, r, pid, uid) | (q, r, pid, uid) | ❌ No |
| BKT skill params | Loaded at init | Loaded at init | ❌ No |
| PCA reference | N/A | Loaded at init | ✅ Yes (new file) |
| Data preprocessing | Existing pipeline | Existing pipeline | ❌ No |
| Batch format | [BS, seqlen] | [BS, seqlen] | ❌ No |

**Bottom line**: The only new requirement is generating and loading the PCA reference file (student-level, one-time). All interaction-level data remains unchanged.

## Prerequisites: Data Preparation Workflow

Before implementing v2.0, you must prepare the PCA reference data. This is a **one-time preprocessing step** per dataset.

### Required Workflow

```
1. Train BKT Model (existing)
   ↓
2. Generate BKT Forward Inference (existing)
   ↓
3. Generate PCA Reference (NEW - Step 8)
   ↓
4. Implement v2.0 Model (Steps 0-7, 9-10)
   ↓
5. Train v2.0 with PCA Grounding
```

### Step-by-Step Prerequisites

#### Prerequisite 1: BKT Training (Existing)

Train BKT model on your dataset to obtain skill-level parameters:

```bash
# Example for assist2009
python examples/train_bkt.py \
  --dataset assist2009 \
  --fold 0
```

**Output**: `data/assist2009/bkt_skill_params.pkl`

#### Prerequisite 2: BKT Forward Inference (Existing)

Run BKT forward inference to get student-level parameters (p_L0, p_T per student per skill):

```bash
# Example for assist2009
python examples/bkt_forward_inference.py \
  --dataset assist2009 \
  --fold 0 \
  --bkt_params data/assist2009/bkt_skill_params.pkl
```

**Output**: `data/assist2009/bkt_forward.pkl`

**Format**: 
```python
{
  original_uid_1: {
    skill_1: {'p_L0': 0.45, 'p_T': 0.12, ...},
    skill_2: {'p_L0': 0.67, 'p_T': 0.08, ...},
    ...
  },
  original_uid_2: {...},
  ...
}
```

#### Prerequisite 3: Generate PCA Reference (NEW - Step 8)

Generate PCA reference coordinates from BKT forward inference:

```bash
# Example for assist2009
python examples/generate_pca_reference.py \
  --bkt_forward data/assist2009/bkt_forward.pkl \
  --dataset_dir data/assist2009 \
  --output data/assist2009/pca_reference.json
```

**Output**: `data/assist2009/pca_reference.json`

**Format**:
```json
{
  "student_coords": {
    "0": [0.234, -0.156],  // Model index -> [PC1, PC2]
    "1": [-0.421, 0.089],
    ...
  },
  "metadata": {
    "n_students": 1234,
    "dataset": "assist2009"
  }
}
```

**Critical**: This file contains the **ground truth targets** for L_pca loss. The model will learn student traits (δ) and the PCA loss will align them to these coordinates.

### How L_pca Uses PCA Reference

During training, for each batch:

1. **Model computes student traits**: `δ = proj(mean(z_context))` → `[BS, 2]`
2. **Lookup PCA targets**: `Z_pca[uid_batch]` → `[BS, 2]` (from pca_reference.json)
3. **Compute L_pca**: 
   ```python
   L_mse = MSE(δ, Z_pca)  # Direct alignment
   L_pairwise = MSE(dist(δ), dist(Z_pca))  # Topology preservation
   L_pca = 0.2 * L_mse + 0.8 * L_pairwise
   ```

### Verification Checklist

Before training v2.0, verify:

- [ ] BKT model trained: `bkt_skill_params.pkl` exists
- [ ] BKT forward inference complete: `bkt_forward.pkl` exists
- [ ] PCA reference generated: `pca_reference.json` exists
- [ ] PCA reference has correct format (student_coords with model indices)
- [ ] Number of students in PCA reference matches dataset
- [ ] `n_uid` parameter in config matches max index in PCA reference + 1

### Per-Dataset Requirements

**IMPORTANT**: This workflow must be completed **for each dataset** you want to train on:

| Dataset | BKT Params | BKT Forward | PCA Reference |
|---------|------------|-------------|---------------|
| assist2009 | `data/assist2009/bkt_skill_params.pkl` | `data/assist2009/bkt_forward.pkl` | `data/assist2009/pca_reference.json` |
| assist2015 | `data/assist2015/bkt_skill_params.pkl` | `data/assist2015/bkt_forward.pkl` | `data/assist2015/pca_reference.json` |
| bridge2algebra2006 | `data/bridge2algebra2006/bkt_skill_params.pkl` | `data/bridge2algebra2006/bkt_forward.pkl` | `data/bridge2algebra2006/pca_reference.json` |

## Implementation Steps

### Step 0: Remove Legacy Components from Current Implementation

Before implementing v2.0, we need to remove components from the current gtransformer.py that are incompatible with the new three-term decomposition architecture.

#### 0.1: Remove Probe Architecture (Active Grounding)

**Location**: Lines 95-98, 334-340

**Components to remove:**
```python
# In __init__:
self.probe_l0 = nn.Linear(z_dim, 1)
self.probe_t = nn.Linear(z_dim, 1)

# In forward:
probe_l0_logits = self.probe_l0(z_context).squeeze(-1)
probe_t_logits = self.probe_t(z_context).squeeze(-1)
p_l0_probe = torch.sigmoid(probe_l0_logits)
p_t_probe = torch.sigmoid(probe_t_logits)
```

**Rationale**: v2.0 uses PCA grounding instead of probe-based active grounding. The probe architecture was designed to align learned parameters with BKT estimates, but v2.0 achieves this through the PCA cluster coherence loss.

**Also remove from outputs dictionary:**
```python
# Remove these keys:
'p_l0_probe': p_l0_probe,
'p_t_probe': p_t_probe,
```

**Remove from loss computation** (in training script):
```python
# Remove:
loss_probe_l0 = F.binary_cross_entropy(outputs['p_l0_probe'], outputs['p_l0'].detach())
loss_probe_t = F.binary_cross_entropy(outputs['p_t_probe'], outputs['p_t'].detach())
loss_probe = loss_probe_l0 + loss_probe_t
```

#### 0.2: Remove Relational Axes (Contextual Projection)

**Location**: Lines 71-77, 270-280

**Components to remove:**
```python
# In __init__:
self.knowledge_axis_emb = nn.Embedding(self.n_question + 1, z_dim)
self.velocity_axis_emb = nn.Embedding(self.n_question + 1, z_dim)
nn.init.normal_(self.knowledge_axis_emb.weight, mean=0.0, std=0.02)
nn.init.normal_(self.velocity_axis_emb.weight, mean=0.0, std=0.02)

# In forward:
k_axis = self.knowledge_axis_emb(q_data)  # BS, seqlen, z_dim
v_axis = self.velocity_axis_emb(q_data)   # BS, seqlen, z_dim
l0_logits = l0_base + (z_context * k_axis).sum(dim=-1)
t_logits = t_base + (z_context * v_axis).sum(dim=-1)
```

**Rationale**: v2.0 uses a simpler decomposition where contextual information comes from:
- Student traits (δ): Aggregated from history via mean pooling
- Skill residuals (ε): Direct projection from current context

The relational axes approach was too complex and didn't provide clear interpretability.

#### 0.3: Remove Theoretical Bases Embeddings

**Location**: Lines 79-82, 273-274

**Components to remove:**
```python
# In __init__:
self.l0_base_emb = nn.Embedding(self.n_question + 1, 1)
self.t_base_emb = nn.Embedding(self.n_question + 1, 1)

# In forward:
l0_base = self.l0_base_emb(q_data).squeeze(-1)
t_base = self.t_base_emb(q_data).squeeze(-1)
```

**Rationale**: v2.0 uses fixed population-level parameters (μ) loaded from BKT via buffers, not learnable embeddings. The buffers `bkt_l0_pop` and `bkt_t_pop` (lines 88-89) will be retained and used directly.

#### 0.4: Remove Student-Specific Embeddings (if personalization disabled)

**Location**: Lines 91-93, 316-329

**Components to conditionally remove:**
```python
# In __init__ (only if n_uid > 0):
self.student_param = nn.Embedding(self.n_uid + 1, 1)
self.student_gap_param = nn.Embedding(self.n_uid + 1, 1)

# In forward (only if n_uid > 0):
if self.n_uid > 0 and uid_data is not None:
    uid_seq = uid_data.unsqueeze(1).expand(-1, q_data.size(1))
    s_gap = self.student_gap_param(uid_seq).squeeze(-1)
    s_vel = self.student_param(uid_seq).squeeze(-1)
    l0_logits = l0_logits + s_gap
    t_logits = t_logits + s_vel
```

**Rationale**: v2.0 computes student traits dynamically from interaction history (context-based), not from static embeddings. This is a core design principle: no student memorization.

**Exception**: Keep this code if we want to support a hybrid mode where both context-based traits (δ) and static embeddings coexist. For pure v2.0, remove it.

#### 0.5: Remove Diversity Loss

**Location**: Lines 282-312

**Components to remove:**
```python
# Diversity Loss: Encourage semantic axes to be different across concepts
if self.ablation != "all":
    unique_concepts = torch.unique(q_data)
    if len(unique_concepts) > 1 and not qtest:
        sampled_k_axes = self.knowledge_axis_emb(unique_concepts)
        sampled_v_axes = self.velocity_axis_emb(unique_concepts)
        k_normalized = sampled_k_axes / (sampled_k_axes.norm(dim=1, keepdim=True) + 1e-8)
        v_normalized = sampled_v_axes / (sampled_v_axes.norm(dim=1, keepdim=True) + 1e-8)
        k_sim_matrix = k_normalized @ k_normalized.t()
        v_sim_matrix = v_normalized @ v_normalized.t()
        mask = ~torch.eye(len(unique_concepts), dtype=torch.bool, device=q_data.device)
        k_diversity_loss = k_sim_matrix[mask].abs().mean()
        v_diversity_loss = v_sim_matrix[mask].abs().mean()
        diversity_loss = 0.1 * (k_diversity_loss + v_diversity_loss)
    else:
        diversity_loss = torch.tensor(0.0, device=q_data.device)
else:
    diversity_loss = torch.tensor(0.0, device=q_data.device)

# In return:
total_reg_loss = c_reg_loss + diversity_loss
```

**Rationale**: Diversity loss was specific to the relational axes architecture. v2.0 doesn't use skill-specific axes, so this loss is no longer needed.

#### 0.6: Remove Duplicate Buffer Registrations

**Location**: Lines 109-110

**Components to remove:**
```python
# Duplicate registration (already done at lines 88-89)
self.register_buffer('bkt_l0_pop', torch.ones(n_question + 1) * 0.5)
self.register_buffer('bkt_t_pop', torch.ones(n_question + 1) * 0.1)
```

**Rationale**: These buffers are already registered at lines 88-89. The duplicate registration at lines 109-110 should be removed.

#### 0.7: Update Parameter Loading Method

**Location**: Lines 193-196

**Components to modify:**
```python
# OLD (loads into learnable embeddings):
if hasattr(self, 'l0_base_emb'):
    self.l0_base_emb.weight[q_idx].normal_(mean=l0_logit, std=0.05)
if hasattr(self, 't_base_emb'):
    self.t_base_emb.weight[q_idx].normal_(mean=t_logit, std=0.05)

# NEW (loads into fixed buffers):
# Just update the buffers directly (already done at lines 199-200)
self.bkt_l0_pop[q_idx] = l0_p
self.bkt_t_pop[q_idx] = t_p
```

**Rationale**: v2.0 uses fixed population parameters, not learnable embeddings with textured initialization.

#### 0.8: Remove Axes Initialization

**Location**: Lines 204-209

**Components to remove:**
```python
# Initialize axes with orthogonal vectors
if hasattr(self, 'knowledge_axis_emb'):
    nn.init.orthogonal_(self.knowledge_axis_emb.weight)
if hasattr(self, 'velocity_axis_emb'):
    nn.init.orthogonal_(self.velocity_axis_emb.weight)
```

**Rationale**: No axes in v2.0 architecture.

### Summary of Removals

| Component | Lines | Reason |
|-----------|-------|--------|
| Probe architecture | 95-98, 334-340 | Replaced by PCA grounding |
| Relational axes | 71-77, 270-280 | Replaced by simple projections |
| Theoretical base embeddings | 79-82, 273-274 | Replaced by fixed buffers |
| Student embeddings | 91-93, 316-329 | Replaced by context-based traits |
| Diversity loss | 282-312 | No longer needed |
| Duplicate buffers | 109-110 | Already registered |
| Axes initialization | 204-209 | No axes in v2.0 |

**Total lines removed**: ~150 lines
**Architecture simplification**: From 4-component system to clean 3-term decomposition

#### 1.1: Add Parameters to `configs/parameter_default.json`

**Location**: `configs/parameter_default.json` - `defaults` section

Add the following v2.0 parameters:

```json
{
  "defaults": {
    ...existing parameters...,
    "lambda_pca": 0.1,
    "lambda_residual": 0.01,
    "use_population": true,
    "use_traits": true,
    "use_residuals": true,
    "pca_alpha": 0.2,
    "pca_beta": 0.8,
    "personalization": false,
    "n_uid": 4163
  }
}
```

**IMPORTANT Notes:**

1. **`personalization: false`**: Disables student-specific embeddings (student_param, student_gap_param)
   - v2.0 uses context-based traits (δ), not memorized embeddings
   - This is a core design principle: no student memorization

2. **`n_uid: 4163`**: Number of students in the dataset (example for assist2009)
   - **Required** for PCA reference buffer size: `self.pca_reference = torch.zeros(n_uid + 1, 2)`
   - Must match the number of students in your dataset
   - Set per-dataset (assist2009: 4163, assist2015: 19840, etc.)
   - Get from `data/{dataset}/keyid2idx.json` → `len(users)`

3. **Distinction**:
   - `n_uid > 0`: Enables PCA reference buffer (REQUIRED for v2.0)
   - `personalization = false`: Disables student embeddings (REQUIRED for v2.0)
   - These are independent: we need the buffer but not the embeddings

**Location**: `configs/parameter_default.json` - `types.model_config` section

Add parameters to the model_config type list:

```json
{
  "types": {
    "model_config": [
      ...existing parameters...,
      "lambda_pca",
      "lambda_residual",
      "use_population",
      "use_traits",
      "use_residuals",
      "pca_alpha",
      "pca_beta",
      "n_uid"
    ]
  }
}
```

**After modification, update MD5 hash:**

```bash
python examples/parameters_audit.py --fix-md5
```

#### 1.2: Update Model `__init__` Method

**Location**: `pykt/models/gtransformer.py` - `__init__` method (after line 89)

**IMPORTANT**: Use `kwargs['key']` (fail-fast) instead of `kwargs.get('key', default)` (silent fallback)

```python
# PCA Grounding Infrastructure (v2.0)
# Store PCA-derived student trait references for grounding loss
# Shape: [n_uid + 1, 2] where 2 = [PC1: General Proficiency, PC2: Learning Momentum]
self.register_buffer('pca_reference', torch.zeros(n_uid + 1, 2))
self.pca_reference_loaded = False  # Flag to track if PCA data is available

# Loss weights for v2.0 (NO DEFAULTS - must be in parameter_default.json)
self.lambda_pca = kwargs['lambda_pca']  # PCA cluster coherence
self.lambda_residual = kwargs['lambda_residual']  # Parsimony for residuals

# Ablation controls for three-term decomposition (NO DEFAULTS)
self.use_population = kwargs['use_population']  # μ term
self.use_traits = kwargs['use_traits']  # δ term  
self.use_residuals = kwargs['use_residuals']  # ε term

# PCA loss component weights (NO DEFAULTS)
self.pca_alpha = kwargs['pca_alpha']  # MSE weight
self.pca_beta = kwargs['pca_beta']   # Pairwise distance weight
```

**Location**: After line 98 (probe architecture)

```python
# Student Trait Encoder (v2.0)
# Projects aggregated context to 2D trait space: δ = [δ_L0, δ_T]
self.student_trait_encoder = nn.Linear(z_dim, 2)
nn.init.xavier_uniform_(self.student_trait_encoder.weight)
nn.init.zeros_(self.student_trait_encoder.bias)

# Skill Residual Projectors (v2.0)
# Per-interaction residuals: ε_L0, ε_T
self.residual_l0_proj = nn.Linear(z_dim, 1)
self.residual_t_proj = nn.Linear(z_dim, 1)
nn.init.xavier_uniform_(self.residual_l0_proj.weight)
nn.init.xavier_uniform_(self.residual_t_proj.weight)
nn.init.zeros_(self.residual_l0_proj.bias)
nn.init.zeros_(self.residual_t_proj.bias)
```

### Step 2: Add PCA Reference Loading Method

**Location**: After `load_theory_params` method (after line 211)

```python
def load_pca_reference(self, pca_data):
    """
    Load PCA-derived student trait references for grounding loss.
    
    Args:
        pca_data: dict with keys:
            - 'student_coords': dict mapping uid -> [PC1, PC2] coordinates
            - 'pca_components': PCA transformation matrix (optional)
            - 'pca_mean': PCA center point (optional)
    """
    if self.ablation == "all" or not self.use_traits:
        print("  [GTransformer] PCA grounding disabled (ablation mode)")
        return
    
    if pca_data is None:
        print("  [GTransformer] No PCA data provided, skipping PCA reference loading")
        return
    
    student_coords = pca_data.get('student_coords', {})
    
    if not student_coords:
        print("  [GTransformer] Empty PCA student coordinates, skipping")
        return
    
    with torch.no_grad():
        for uid, coords in student_coords.items():
            if isinstance(uid, str):
                uid = int(uid)
            if uid < self.pca_reference.size(0):
                self.pca_reference[uid] = torch.tensor(coords, dtype=torch.float32)
    
    self.pca_reference_loaded = True
    print(f"  [GTransformer] PCA reference loaded for {len(student_coords)} students")
    print(f"  [GTransformer] PC1 range: [{self.pca_reference[:, 0].min():.3f}, {self.pca_reference[:, 0].max():.3f}]")
    print(f"  [GTransformer] PC2 range: [{self.pca_reference[:, 1].min():.3f}, {self.pca_reference[:, 1].max():.3f}]")
```

### Step 3: Modify Forward Pass for Three-Term Decomposition

**Location**: Replace lines 267-332 (current grounded parameter computation)

```python
# ============================================================================
# V2.0: THREE-TERM DECOMPOSITION WITH PCA GROUNDING
# ============================================================================

# Step 1: Compute Student Traits (δ) from Aggregated Context
# Aggregate context across sequence using simple mean (unbiased estimator)
# z_context: [BS, seqlen, z_dim]
z_agg = z_context.mean(dim=1)  # [BS, z_dim]

# Project to 2D trait space: [δ_L0, δ_T]
student_traits = self.student_trait_encoder(z_agg)  # [BS, 2]

# Extract individual trait components
delta_l0 = student_traits[:, 0:1].expand(-1, seqlen)  # [BS, seqlen]
delta_t = student_traits[:, 1:2].expand(-1, seqlen)   # [BS, seqlen]

# Step 2: Compute Skill Residuals (ε) from Per-Timestep Context
epsilon_l0 = self.residual_l0_proj(z_context).squeeze(-1)  # [BS, seqlen]
epsilon_t = self.residual_t_proj(z_context).squeeze(-1)    # [BS, seqlen]

# Step 3: Get Population-Level Parameters (μ) from BKT
mu_l0 = self.bkt_l0_pop[q_data.long()]  # [BS, seqlen]
mu_t = self.bkt_t_pop[q_data.long()]    # [BS, seqlen]

# Convert to logits for additive composition
def prob_to_logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))

mu_l0_logit = prob_to_logit(mu_l0)
mu_t_logit = prob_to_logit(mu_t)

# Step 4: Three-Term Additive Composition (with ablation controls)
l0_logits = torch.zeros_like(mu_l0_logit)
t_logits = torch.zeros_like(mu_t_logit)

if self.use_population:
    l0_logits = l0_logits + mu_l0_logit
    t_logits = t_logits + mu_t_logit

if self.use_traits:
    l0_logits = l0_logits + delta_l0
    t_logits = t_logits + delta_t

if self.use_residuals:
    l0_logits = l0_logits + epsilon_l0
    t_logits = t_logits + epsilon_t

# Convert back to probabilities
p_l0 = torch.sigmoid(l0_logits)  # [BS, seqlen]
p_t = torch.sigmoid(t_logits)    # [BS, seqlen]
```

### Step 4: Add PCA Grounding Loss Function

**Location**: After `_bkt_ref_output` method (after line 495)

```python
def compute_pca_loss(self, student_traits, uid_data, alpha=0.2, beta=0.8):
    """
    Compute PCA cluster coherence loss (L_pca).
    
    Combines two objectives:
    1. L_mse: Direct MSE to PCA coordinates (keeps axes aligned)
    2. L_pairwise: Preserves relative distances (allows rotation/translation)
    
    Args:
        student_traits: [BS, 2] learned trait coordinates
        uid_data: [BS] or [BS, seqlen] student IDs
        alpha: Weight for L_mse (default: 0.2)
        beta: Weight for L_pairwise (default: 0.8)
    
    Returns:
        L_pca: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    if not self.pca_reference_loaded or not self.use_traits:
        return torch.tensor(0.0, device=student_traits.device), {}
    
    # Get student IDs (handle both 1D and 2D uid_data)
    if uid_data.dim() == 2:
        uid_batch = uid_data[:, 0]  # Take first timestep (constant per sequence)
    else:
        uid_batch = uid_data
    
    # Get PCA reference coordinates for this batch
    pca_ref_batch = self.pca_reference[uid_batch.long()]  # [BS, 2]
    
    # Component 1: Direct MSE (axis alignment)
    L_mse = F.mse_loss(student_traits, pca_ref_batch)
    
    # Component 2: Pairwise distance preservation (topology preservation)
    # Compute pairwise distances in learned space
    D_learned = torch.cdist(student_traits, student_traits, p=2)  # [BS, BS]
    
    # Compute pairwise distances in reference space
    D_ref = torch.cdist(pca_ref_batch, pca_ref_batch, p=2)  # [BS, BS]
    
    # MSE between distance matrices
    L_pairwise = F.mse_loss(D_learned, D_ref)
    
    # Balanced combination
    L_pca = alpha * L_mse + beta * L_pairwise
    
    return L_pca, {
        'L_pca_mse': L_mse.item(),
        'L_pca_pairwise': L_pairwise.item(),
        'L_pca_total': L_pca.item()
    }
```

### Step 5: Add Residual Parsimony Loss

**Location**: After `compute_pca_loss` method

```python
def compute_residual_loss(self, epsilon_l0, epsilon_t):
    """
    Compute parsimony regularization for skill residuals.
    Encourages the model to use δ (traits) as primary personalization,
    keeping ε (residuals) small for interaction-specific noise only.
    
    Args:
        epsilon_l0: [BS, seqlen] L0 residuals
        epsilon_t: [BS, seqlen] T residuals
    
    Returns:
        L_residual: L2 norm of residuals
    """
    if not self.use_residuals:
        return torch.tensor(0.0, device=epsilon_l0.device)
    
    # L2 regularization on residuals
    L_residual = (epsilon_l0 ** 2).mean() + (epsilon_t ** 2).mean()
    
    return L_residual
```

### Step 6: Update Forward Pass to Return New Outputs

**Location**: Modify outputs dictionary (around line 351)

```python
# Collect all outputs in a structured dictionary
outputs = {
    'predictions': preds,
    'p_l0': p_l0,
    'p_t': p_t,
    'p_l0_probe': p_l0_probe,
    'p_t_probe': p_t_probe,
    'reference_preds': ref_preds,
    # V2.0 additions
    'student_traits': student_traits,  # [BS, 2] for PCA loss
    'epsilon_l0': epsilon_l0,          # [BS, seqlen] for residual loss
    'epsilon_t': epsilon_t,            # [BS, seqlen] for residual loss
    'delta_l0': delta_l0,              # [BS, seqlen] for analysis
    'delta_t': delta_t,                # [BS, seqlen] for analysis
}
```

### Step 7: Update Training Script to Compute New Losses

**Location**: Training script (e.g., `examples/wandb_gtransformer_train.py`)

```python
# In training loop, after model forward pass:
outputs, reg_loss = model(q, r, qshft, rshft, m, sm, q_data, pid_data, uid_data)

# Compute multi-objective loss
loss_sup = criterion(outputs['predictions'], target)

# PCA grounding loss (v2.0)
loss_pca, pca_metrics = model.compute_pca_loss(
    outputs['student_traits'], 
    uid_data,
    alpha=0.2,  # MSE weight
    beta=0.8    # Pairwise distance weight
)

# Residual parsimony loss (v2.0)
loss_residual = model.compute_residual_loss(
    outputs['epsilon_l0'],
    outputs['epsilon_t']
)

# Reference fidelity loss (existing)
loss_ref = F.binary_cross_entropy(
    outputs['reference_preds'],
    target,
    reduction='mean'
)

# Probe grounding loss (existing)
loss_probe_l0 = F.binary_cross_entropy(outputs['p_l0_probe'], outputs['p_l0'].detach())
loss_probe_t = F.binary_cross_entropy(outputs['p_t_probe'], outputs['p_t'].detach())
loss_probe = loss_probe_l0 + loss_probe_t

# Total loss
total_loss = (
    model.lambda_sup * loss_sup +
    model.lambda_ref * loss_ref +
    model.lambda_pca * loss_pca +
    model.lambda_residual * loss_residual +
    model.lambda_probe * loss_probe +
    reg_loss
)

# Log individual components
wandb.log({
    'loss/supervised': loss_sup.item(),
    'loss/reference': loss_ref.item(),
    'loss/pca': loss_pca.item(),
    'loss/pca_mse': pca_metrics.get('L_pca_mse', 0),
    'loss/pca_pairwise': pca_metrics.get('L_pca_pairwise', 0),
    'loss/residual': loss_residual.item(),
    'loss/probe': loss_probe.item(),
    'loss/total': total_loss.item(),
})
```

### Step 8: Create PCA Reference Generation Script

**Location**: New file `examples/generate_pca_reference.py`

**CRITICAL**: Must use dataset-specific `keyid2idx.json` mapping to convert original student IDs to zero-based indices.

```python
import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle
import json
import os

def generate_pca_reference(bkt_forward_file, dataset_dir, output_file):
    """
    Generate PCA reference coordinates from BKT forward inference.
    
    IMPORTANT: Uses keyid2idx.json mapping to convert original student IDs
    to zero-based indices used internally by the model.
    
    Args:
        bkt_forward_file: Path to BKT forward inference results (pickle)
        dataset_dir: Path to dataset directory (e.g., 'data/assist2009')
        output_file: Path to save PCA reference data (JSON)
    """
    # Load dataset-specific student ID mapping
    # This bidirectional mapping converts between original IDs and model indices
    keyid2idx_file = os.path.join(dataset_dir, 'keyid2idx.json')
    if not os.path.exists(keyid2idx_file):
        raise FileNotFoundError(
            f"keyid2idx.json not found at {keyid2idx_file}. "
            f"This file is required to map student IDs to model indices."
        )
    
    with open(keyid2idx_file, 'r') as f:
        keyid2idx = json.load(f)
    
    # Extract student ID mapping (original_id -> model_index)
    # keyid2idx structure: {"questions": {...}, "users": {...}}
    user_mapping = keyid2idx.get('users', {})
    if not user_mapping:
        raise ValueError(
            f"No 'users' mapping found in {keyid2idx_file}. "
            f"Cannot map student IDs to model indices."
        )
    
    print(f"Loaded student ID mapping: {len(user_mapping)} students")
    
    # Load BKT forward inference (p_L0, p_T per student per skill)
    with open(bkt_forward_file, 'rb') as f:
        bkt_data = pickle.load(f)
    
    print(f"Loaded BKT forward inference: {len(bkt_data)} students")
    
    # Extract student-level aggregates
    # Aggregate across skills: mean(p_L0), mean(p_T) per student
    student_features = {}
    original_to_index = {}  # Track mapping for validation
    
    for original_uid, skill_params in bkt_data.items():
        # Convert original UID to string for lookup
        uid_str = str(original_uid)
        
        # Get model index from mapping
        if uid_str not in user_mapping:
            print(f"Warning: Student {uid_str} not in keyid2idx mapping, skipping")
            continue
        
        model_idx = user_mapping[uid_str]
        original_to_index[original_uid] = model_idx
        
        # Aggregate BKT parameters across skills
        l0_values = [params['p_L0'] for params in skill_params.values()]
        t_values = [params['p_T'] for params in skill_params.values()]
        
        student_features[model_idx] = [
            np.mean(l0_values),  # Average initial mastery
            np.mean(t_values)    # Average learning rate
        ]
    
    print(f"Mapped {len(student_features)} students to model indices")
    
    # Convert to matrix (sorted by model index for reproducibility)
    model_indices = sorted(student_features.keys())
    X = np.array([student_features[idx] for idx in model_indices])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"  PC1 (Initial Mastery) range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"  PC2 (Learning Rate) range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create student_coords mapping (model_index -> [PC1, PC2])
    # CRITICAL: Keys are model indices (0-based), not original student IDs
    student_coords = {
        str(model_idx): X_pca[i].tolist() 
        for i, model_idx in enumerate(model_indices)
    }
    
    # Save PCA reference
    pca_data = {
        'student_coords': student_coords,
        'pca_components': pca.components_.tolist(),
        'pca_mean': pca.mean_.tolist(),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'metadata': {
            'n_students': len(student_coords),
            'dataset': os.path.basename(dataset_dir),
            'bkt_source': os.path.basename(bkt_forward_file),
            'keyid2idx_source': os.path.basename(keyid2idx_file)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    print(f"\n✅ PCA reference generated successfully")
    print(f"  Students: {len(student_coords)}")
    print(f"  PC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  PC2 explained variance: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  Saved to: {output_file}")
    
    # Validation: Check index range
    max_idx = max(int(k) for k in student_coords.keys())
    print(f"\n📊 Index validation:")
    print(f"  Model index range: [0, {max_idx}]")
    print(f"  Expected n_uid parameter: {max_idx + 1}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate PCA reference from BKT forward inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PCA reference for assist2009
  python examples/generate_pca_reference.py \\
    --bkt_forward data/assist2009/bkt_forward.pkl \\
    --dataset_dir data/assist2009 \\
    --output data/assist2009/pca_reference.json
  
  # Generate for assist2015
  python examples/generate_pca_reference.py \\
    --bkt_forward data/assist2015/bkt_forward.pkl \\
    --dataset_dir data/assist2015 \\
    --output data/assist2015/pca_reference.json

Note: This script MUST be run separately for each dataset because
      each dataset has its own keyid2idx.json mapping.
        """
    )
    
    parser.add_argument('--bkt_forward', required=True,
                        help='Path to BKT forward inference pickle file')
    parser.add_argument('--dataset_dir', required=True,
                        help='Path to dataset directory (must contain keyid2idx.json)')
    parser.add_argument('--output', required=True,
                        help='Path to save PCA reference JSON file')
    
    args = parser.parse_args()
    
    generate_pca_reference(args.bkt_forward, args.dataset_dir, args.output)
```

**Usage per dataset:**

```bash
# assist2009
python examples/generate_pca_reference.py \
  --bkt_forward data/assist2009/bkt_forward.pkl \
  --dataset_dir data/assist2009 \
  --output data/assist2009/pca_reference.json

# assist2015
python examples/generate_pca_reference.py \
  --bkt_forward data/assist2015/bkt_forward.pkl \
  --dataset_dir data/assist2015 \
  --output data/assist2015/pca_reference.json

# bridge2algebra2006
python examples/generate_pca_reference.py \
  --bkt_forward data/bridge2algebra2006/bkt_forward.pkl \
  --dataset_dir data/bridge2algebra2006 \
  --output data/bridge2algebra2006/pca_reference.json
```

**Key points:**
1. **Per-dataset requirement**: Each dataset has its own `keyid2idx.json` mapping
2. **ID mapping**: Original student IDs → Model indices (0-based)
3. **Validation**: Script prints expected `n_uid` parameter for verification
4. **Metadata**: Output includes dataset name and source files for provenance

    student_coords = {uid: X_pca[i].tolist() for i, uid in enumerate(uids)}
    
    # Save PCA reference
    pca_data = {
        'student_coords': student_coords,
        'pca_components': pca.components_.tolist(),
        'pca_mean': pca.mean_.tolist(),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    print(f"PCA reference generated for {len(student_coords)} students")
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_pca_reference.py <bkt_forward_file> <output_file>")
        sys.exit(1)
    
    generate_pca_reference(sys.argv[1], sys.argv[2])
```

### Step 9: Update Model Initialization in Training Script

**Location**: Training script initialization

```python
# After model creation
model = GTransformer(...)

# Load BKT population parameters (existing)
model.load_theory_params(bkt_skill_params)

# Load PCA reference (v2.0)
if os.path.exists(pca_reference_file):
    with open(pca_reference_file, 'r') as f:
        pca_data = json.load(f)
    model.load_pca_reference(pca_data)
else:
    print(f"Warning: PCA reference file not found: {pca_reference_file}")
```

### Step 10: Add Ablation Study Support

**Location**: Configuration file (`configs/parameter_default.json`)

```json
{
  "lambda_pca": 0.1,
  "lambda_residual": 0.01,
  "use_population": true,
  "use_traits": true,
  "use_residuals": true,
  "pca_alpha": 0.2,
  "pca_beta": 0.8
}
```

### Summary of Changes

1. **New buffers**: `pca_reference` for student trait targets
2. **New modules**: `student_trait_encoder`, `residual_l0_proj`, `residual_t_proj`
3. **New methods**: `load_pca_reference()`, `compute_pca_loss()`, `compute_residual_loss()`
4. **Modified forward**: Three-term decomposition with ablation controls
5. **New outputs**: `student_traits`, `epsilon_l0`, `epsilon_t`, `delta_l0`, `delta_t`
6. **New losses**: `L_pca` (cluster coherence), `L_residual` (parsimony)
7. **New script**: `generate_pca_reference.py` for preprocessing
8. **Updated training**: Multi-objective loss with PCA grounding

## Metrics

### Experimental Validation: Exp 743149 (v2.0 PCA Grounding)

**Configuration:**
- Model: GTransformer v2.0 with PCA-grounded three-term decomposition
- Dataset: Assist2009 (5-fold CV)
- Seed: 3407
- PCA parameters: α=0.2, β=0.8, λ_pca=0.1, λ_residual=0.01
- λ_ref=0.5, λ_initmastery=0.1, λ_rate=0.1

### Per-Fold Results

| Fold | p_sup (AUC) | p_ref (AUC) | Gap |
|------|-------------|-------------|-----|
| 0 | 0.7765 | 0.6679 | 0.1086 |
| 1 | 0.7760 | 0.6680 | 0.1080 |
| 2 | 0.7780 | 0.6680 | 0.1100 |
| 3 | 0.7745 | 0.6678 | 0.1066 |
| 4 | 0.7764 | 0.6675 | 0.1089 |
| **Mean** | **0.7763 ± 0.0012** | **0.6678 ± 0.0002** | **0.1084** |

### Comparison with v1.0 Baseline (Exp 801184)

| Metric | v2.0 (Exp 743149) | v1.0 Baseline (Exp 801184) | Delta |
|--------|-------------------|---------------------------|-------|
| **p_sup** | 0.7763 ± 0.0012 | 0.7812 ± 0.0011 | **-0.0049** (-0.63%) |
| **p_ref** | 0.6678 ± 0.0002 | 0.6727 ± 0.0001 | **-0.0049** (-0.73%) |
| **Gap** | 0.1084 | 0.1085 | -0.0001 |

### Analysis

**Key Findings:**
1. **Performance regression**: v2.0 shows -0.63% p_sup and -0.73% p_ref vs v1.0
2. **Gap unchanged**: Interpretability gap remains ~0.108 (no improvement)
3. **Low variance**: Both p_sup and p_ref have very tight confidence intervals

**Root Cause Investigation:**
- Cluster bug fix had no effect (student-level splitting + permutation-invariant β loss)
- Regression likely due to: three-term decomposition overhead, PCA grounding constraints, or hyperparameter sensitivity

### Suggestions for Improvement

#### **Parameters**

**1. `lambda_residual = 0` (HIGH PRIORITY)**
- **Hypothesis**: Removing residual penalty lets ε capture more signal
- **Current**: L_residual = mean(|ε_L0|) + mean(|ε_T|) pushes residuals toward 0
- **With 0**: Model can use residuals freely → may improve p_sup
- **Risk**: May reduce interpretability (δ traits become less meaningful)
- **Status**: Experiment `exp_residual_0` launched (campaign 567618)

**2. Lower `lambda_ref` (0.2 or 0.1)**
- **Hypothesis**: Strong BKT alignment pressure may constrain supervised head
- **Current**: λ_ref = 0.5 (relatively strong)
- **Try**: 0.2 or 0.1 to give more freedom to supervised predictions
- **Expected**: Higher p_sup, possibly lower p_ref

**3. Adjust PCA loss ratio (`pca_alpha` / `pca_beta`)**
- **Current**: α=0.2, β=0.8 (distances dominate)
- **Try α=0.5, β=0.5**: Balance direct matching vs structure preservation
- **Try α=0.0, β=1.0**: Pure distance preservation (fully permutation-invariant)
- **Rationale**: β-dominated loss is already permutation-invariant; may not need α

**4. Reduce `lambda_pca` (0.05 or 0)**
- **Hypothesis**: PCA grounding may over-constrain trait learning
- **Current**: λ_pca = 0.1 (10% of total loss)
- **Try 0.05**: Lighter grounding
- **Try 0**: Disable PCA grounding → ablation baseline
- **Expected**: Higher p_sup if PCA constraints are too restrictive

#### **Possible Causes of -0.55% Gap**

**1. Information Bottleneck in 2D Trait Space**
- **V1.0**: Probe heads extract BKT params directly from full $z$ context ($128D \rightarrow 1D$).
- **V2.0**: Student traits compressed to 2D space first ($128D \rightarrow 2D \rightarrow 1D$).
- **Diagnostic**: Check if 2D trait space loses predictive information.
    - Plot explained variance of student-level PCA.
    - Compare trait dimensionality: try 4D, 8D trait space.

**2. Aggregation Loss in Student Traits**
- **V2.0** computes $\delta$ from $mean(z)$ across all student interactions.
- This averaging may "wash out" fine-grained temporal patterns or recent behavioral shifts.
- **Diagnostic**:
    - Compare per-interaction variance before/after aggregation.
    - Test alternative aggregation: max pooling, attention-weighted mean.

**3. Interaction-Level vs Student-Level Grounding**
- **V1.0**: Probe loss applied per interaction (fine-grained supervision).
- **V2.0**: PCA loss applied per student (coarse-grained supervision).
- **Diagnostic**:
    - Add per-interaction PCA loss variant.
    - Compare gradient magnitudes: probe loss vs PCA loss.

**4. PCA Grounding Signal Quality**
- PCA clusters may be "noisier" targets than the direct BKT oracle.
- PCA is derived from BKT params (indirect), whereas probes use BKT params directly (direct).
- **Diagnostic**:
    - Measure PCA reconstruction error.
    - Compare BKT param variance within vs between clusters.

**5. Three-Term Decomposition Constraints**
- Forcing $p = \sigma(\mu + \delta + \epsilon)$ may be structurally too restrictive.
- **V1.0** allows flexible composition: $p = \sigma(base + context \cdot axis)$.
- **Diagnostic**:
    - Ablate three-term: test $p = \sigma(\mu + \delta)$ only.
    - Test multiplicative fusion: $p = \mu \cdot (1 + \delta + \epsilon)$.

**6. Residual Suppression**
- $\lambda_{residual}$ creates pressure to minimize $\epsilon$, which might capture vital prediction signals.
- **Diagnostic**: 
    - Already tested $\lambda_{residual}=0$ (marginal improvement).
    - Try negative regularization: encourage larger residuals for edge cases.
    - Analyze $\epsilon$ magnitude distribution across sequence length.

#### Recommended Next Experiments

- **Priority 1: Trait Dimensionality.** Increase trait space from 2D to 4D/8D to reduce the bottleneck.
- **Priority 2: Per-Interaction PCA Loss.** Move from student-level to sequence-level grounding to increase supervision density.
- **Priority 3: Hybrid Approach.** Test if combining direct probe loss with PCA grounding provides better stability.

> **Summary**: The 0.55% gap likely stems from information loss in the 2D trait bottleneck combined with coarser supervision granularity (student-level vs interaction-level).

