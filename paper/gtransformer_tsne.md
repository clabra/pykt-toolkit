# GTransformer v2.0: Three-Term Decomposition with t-SNE Cluster Grounding

## Overview

GTransformer v2.0 introduces a novel parameter decomposition that separates student performance into three interpretable components:

1. **Population-level parameters** (μ): Skill-specific baselines from BKT
2. **Per-student trait parameters** (δ): Student-level offsets representing cluster membership
3. **Per-skill residuals** (ε): Interaction-specific deviations from the student's typical behavior

This decomposition enables:
- **Interpretability**: Student traits can be visualized in 2D space (δ_L0, δ_T)
- **Cluster-based learning**: Students with similar traits share learned patterns
- **Personalization**: Skill-specific residuals capture individual variance
- **Grounding**: t-SNE loss ensures learned clusters match BKT-derived structure

## Mathematical Formulation

### Parameter Decomposition

For student `i` and skill `q` at time `t`:

```
p_L0[i,q,t] = σ(μ_L0[q] + δ_L0[i] + ε_L0[i,q,t])
p_T[i,q,t]  = σ(μ_T[q]  + δ_T[i]  + ε_T[i,q,t])
```

Where:
- **μ[q] ∈ ℝ¹**: Population-level logit for skill q (from BKT, fixed)
- **δ[i] ∈ ℝ¹**: Per-student trait offset (learned, cluster-level)
- **ε[i,q,t] ∈ ℝ¹**: Per-interaction residual (learned from context z_t)
- **σ**: Sigmoid function to convert logits to probabilities

### Student Trait Space

Each student is represented by a 2D trait vector:

```
traits[i] = [δ_L0[i], δ_T[i]] ∈ ℝ²
```

This creates a 2D embedding space where students cluster according to:
- **δ_L0**: Average prior knowledge offset (Low/High initial mastery)
- **δ_T**: Average learning velocity offset (Slow/Fast learner)

## Architecture Components

### 1. Population-Level Parameters (Fixed)

Extracted from BKT reference model and kept constant:

```python
# During initialization (from BKT)
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
    
    # Register as non-trainable buffers
    self.register_buffer('mu_L0', mu_L0)
    self.register_buffer('mu_T', mu_T)

def prob_to_logit(self, p, eps=1e-6):
    """Convert probability to logit"""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))
```

### 2. Per-Student Trait Embeddings (Learned with t-SNE Grounding)

```python
class GTransformerV2(nn.Module):
    def __init__(self, n_question, n_uid, d_model, ...):
        super().__init__()
        
        # Student trait embeddings: [n_uid, 2]
        # Dimension 0: δ_L0 (prior knowledge offset)
        # Dimension 1: δ_T (learning velocity offset)
        self.student_traits = nn.Embedding(n_uid + 1, 2)
        nn.init.normal_(self.student_traits.weight, mean=0.0, std=0.1)
        
        # t-SNE reference (computed from BKT, fixed)
        self.register_buffer('tsne_reference', torch.zeros(n_uid + 1, 2))
```

### 3. Per-Skill Residual Projections (From Context)

```python
class GTransformerV2(nn.Module):
    def __init__(self, ...):
        # ... (previous code)
        
        # Project z_context to skill-specific residuals
        z_dim = 2 * d_model  # [d_output || q_embed]
        self.residual_L0_proj = nn.Linear(z_dim, 1)
        self.residual_T_proj = nn.Linear(z_dim, 1)
        
        # Initialize with small weights to start near population mean
        nn.init.normal_(self.residual_L0_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.residual_T_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residual_L0_proj.bias)
        nn.init.zeros_(self.residual_T_proj.bias)
```

## Forward Pass

```python
def forward(self, q_data, target, uid_data, pid_data=None, qtest=False):
    """
    Args:
        q_data: [BS, seqlen] skill IDs
        target: [BS, seqlen] correctness
        uid_data: [BS] student IDs
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
        # Add Rasch difficulty embeddings
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
    # Component 2: Per-Student Traits (δ)
    # ========================================
    # Get 2D trait vector for each student
    student_traits_2d = self.student_traits(uid_data)  # [BS, 2]
    
    # Extract individual trait components
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
    L0_logits = mu_L0_q + delta_L0 + epsilon_L0  # [BS, seqlen]
    T_logits = mu_T_q + delta_T + epsilon_T      # [BS, seqlen]
    
    # Convert to probabilities
    p_L0 = torch.sigmoid(L0_logits)  # [BS, seqlen]
    p_T = torch.sigmoid(T_logits)    # [BS, seqlen]
    
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
        'student_traits': student_traits_2d,  # For t-SNE loss
        'epsilon_L0': epsilon_L0,  # For residual regularization
        'epsilon_T': epsilon_T,
    }
    
    if not qtest:
        return outputs, c_reg_loss
    else:
        return outputs, c_reg_loss, z_context
```

## t-SNE Reference Computation

Compute reference t-SNE clusters from BKT baseline:

```python
def compute_tsne_reference(bkt_predictions_df, bkt_skill_params, n_uid, n_question):
    """
    Compute t-SNE reference from BKT predictions
    
    Args:
        bkt_predictions_df: DataFrame with columns [uid, skill, p_L0_bkt, p_T_bkt]
        bkt_skill_params: BKT population parameters
        n_uid: Number of unique students
        n_question: Number of unique skills
    
    Returns:
        tsne_reference: [n_uid, 2] t-SNE coordinates
        cluster_labels: [n_uid] cluster assignments (0-3)
    """
    from sklearn.manifold import TSNE
    import numpy as np
    
    # Extract population-level parameters
    mu_L0 = np.zeros(n_question + 1)
    mu_T = np.zeros(n_question + 1)
    
    params_dict = bkt_skill_params.get('params', {})
    for q_idx in range(n_question + 1):
        s_params = params_dict.get(q_idx, params_dict.get(str(q_idx), {}))
        mu_L0[q_idx] = s_params.get('prior', 0.5)
        mu_T[q_idx] = s_params.get('learns', 0.1)
    
    # Compute per-student average deviations
    delta_L0_ref = np.zeros(n_uid + 1)
    delta_T_ref = np.zeros(n_uid + 1)
    
    for uid in range(1, n_uid + 1):
        student_data = bkt_predictions_df[bkt_predictions_df['uid'] == uid]
        
        if len(student_data) > 0:
            # Average deviation from population mean
            skills = student_data['skill'].values
            p_L0_student = student_data['p_L0_bkt'].values
            p_T_student = student_data['p_T_bkt'].values
            
            # Compute deviations in probability space
            delta_L0_ref[uid] = np.mean(p_L0_student - mu_L0[skills])
            delta_T_ref[uid] = np.mean(p_T_student - mu_T[skills])
    
    # Stack into [n_uid, 2] array
    student_deviations = np.stack([delta_L0_ref, delta_T_ref], axis=1)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(student_deviations[1:])  # Skip uid=0
    
    # Add back uid=0 (padding)
    tsne_reference = np.vstack([np.zeros((1, 2)), tsne_coords])
    
    # Assign cluster labels (4 quadrants)
    # Based on median split of delta_L0 and delta_T
    median_L0 = np.median(delta_L0_ref[1:])
    median_T = np.median(delta_T_ref[1:])
    
    cluster_labels = np.zeros(n_uid + 1, dtype=int)
    for uid in range(1, n_uid + 1):
        low_L0 = delta_L0_ref[uid] < median_L0
        low_T = delta_T_ref[uid] < median_T
        
        if low_L0 and low_T:
            cluster_labels[uid] = 0  # Low L0, Low T
        elif low_L0 and not low_T:
            cluster_labels[uid] = 1  # Low L0, High T
        elif not low_L0 and low_T:
            cluster_labels[uid] = 2  # High L0, Low T
        else:
            cluster_labels[uid] = 3  # High L0, High T
    
    return torch.FloatTensor(tsne_reference), torch.LongTensor(cluster_labels)
```

## Loss Functions

### 1. t-SNE Alignment Loss

Ensures learned student traits match BKT-derived cluster structure:

```python
def tsne_alignment_loss(student_traits, tsne_reference, uid_batch):
    """
    Align learned student traits with reference t-SNE clusters
    
    Args:
        student_traits: [n_uid, 2] learned trait embeddings
        tsne_reference: [n_uid, 2] reference t-SNE coordinates from BKT
        uid_batch: [BS] student IDs in current batch
    
    Returns:
        L_tsne: scalar loss
    """
    # Get traits for students in batch
    traits_batch = student_traits[uid_batch]  # [BS, 2]
    z_ref_batch = tsne_reference[uid_batch]   # [BS, 2]
    
    # Option A: Direct MSE in t-SNE space
    # L_tsne = F.mse_loss(traits_batch, z_ref_batch)
    
    # Option B: Pairwise distance preservation (better for cluster structure)
    # Preserve relative distances between students
    D_learned = torch.cdist(traits_batch, traits_batch, p=2)  # [BS, BS]
    D_ref = torch.cdist(z_ref_batch, z_ref_batch, p=2)        # [BS, BS]
    
    L_tsne = F.mse_loss(D_learned, D_ref)
    
    return L_tsne
```

### 2. Cluster Assignment Loss (Alternative)

For discrete cluster assignments:

```python
def cluster_assignment_loss(student_traits, cluster_labels, uid_batch):
    """
    Ensure students are assigned to correct clusters
    
    Args:
        student_traits: [n_uid, 2] learned trait embeddings
        cluster_labels: [n_uid] cluster ID for each student (0-3)
        uid_batch: [BS] student IDs in current batch
    
    Returns:
        L_cluster: scalar loss
    """
    # Compute cluster centroids from learned traits
    n_clusters = 4
    centroids = []
    
    for k in range(n_clusters):
        mask = (cluster_labels == k)
        if mask.sum() > 0:
            centroids.append(student_traits[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros(2, device=student_traits.device))
    
    centroids = torch.stack(centroids)  # [4, 2]
    
    # For each student, compute distance to all centroids
    traits_batch = student_traits[uid_batch]  # [BS, 2]
    labels_batch = cluster_labels[uid_batch]  # [BS]
    
    # Negative distances as logits (closer = higher logit)
    distances = torch.cdist(traits_batch, centroids, p=2)  # [BS, 4]
    logits = -distances
    
    # Cross-entropy loss
    L_cluster = F.cross_entropy(logits, labels_batch)
    
    return L_cluster
```

### 3. Residual Regularization

Prevent residuals from dominating:

```python
def residual_regularization_loss(epsilon_L0, epsilon_T):
    """
    Regularize residuals to prevent overfitting
    
    Args:
        epsilon_L0: [BS, seqlen] L0 residuals
        epsilon_T: [BS, seqlen] T residuals
    
    Returns:
        L_residual: scalar loss
    """
    L_residual = epsilon_L0.pow(2).mean() + epsilon_T.pow(2).mean()
    return L_residual
```

### 4. Total Multi-Objective Loss

```python
def compute_total_loss(outputs, targets, uid_batch, model, config):
    """
    Compute total multi-objective loss
    
    Args:
        outputs: dict from model forward pass
        targets: ground truth labels
        uid_batch: student IDs
        model: GTransformerV2 instance
        config: loss weight configuration
    
    Returns:
        total_loss: scalar
        loss_dict: dict with individual loss components
    """
    # 1. Supervised prediction loss
    L_sup = F.binary_cross_entropy(
        outputs['predictions'], 
        targets.float()
    )
    
    # 2. Reference output loss (BKT logic wrapper)
    L_ref = F.binary_cross_entropy(
        outputs['reference_preds'],
        targets.float()
    )
    
    # 3. t-SNE cluster coherence loss
    L_tsne = tsne_alignment_loss(
        model.student_traits.weight,
        model.tsne_reference,
        uid_batch
    )
    
    # 4. Residual regularization
    L_residual = residual_regularization_loss(
        outputs['epsilon_L0'],
        outputs['epsilon_T']
    )
    
    # Combine with weights
    total_loss = (
        config['lambda_sup'] * L_sup +
        config['lambda_ref'] * L_ref +
        config['lambda_tsne'] * L_tsne +
        config['lambda_residual'] * L_residual
    )
    
    loss_dict = {
        'loss': total_loss.item(),
        'L_sup': L_sup.item(),
        'L_ref': L_ref.item(),
        'L_tsne': L_tsne.item(),
        'L_residual': L_residual.item(),
    }
    
    return total_loss, loss_dict
```

## Training Configuration

Recommended hyperparameters:

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
    'lambda_tsne': 0.1,       # t-SNE cluster coherence
    'lambda_residual': 0.01,  # Residual regularization
    
    # Training
    'learning_rate': 0.0001,
    'batch_size': 64,
    'max_epochs': 100,
}
```

## Visualization and Interpretation

### Visualize Student Clusters

```python
def visualize_student_clusters(model, cluster_labels, save_path='student_clusters.png'):
    """
    Visualize learned student trait embeddings
    
    Args:
        model: Trained GTransformerV2
        cluster_labels: [n_uid] cluster assignments
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    # Extract learned traits
    traits = model.student_traits.weight.detach().cpu().numpy()  # [n_uid, 2]
    tsne_ref = model.tsne_reference.detach().cpu().numpy()       # [n_uid, 2]
    labels = cluster_labels.cpu().numpy()
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Learned traits
    ax = axes[0]
    scatter = ax.scatter(
        traits[1:, 0], traits[1:, 1],
        c=labels[1:], cmap='viridis',
        alpha=0.6, s=50
    )
    ax.set_xlabel('δ_L0 (Prior Knowledge Offset)')
    ax.set_ylabel('δ_T (Learning Velocity Offset)')
    ax.set_title('Learned Student Traits')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Plot 2: t-SNE reference
    ax = axes[1]
    scatter = ax.scatter(
        tsne_ref[1:, 0], tsne_ref[1:, 1],
        c=labels[1:], cmap='viridis',
        alpha=0.6, s=50
    )
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Reference (from BKT)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved cluster visualization to {save_path}")
```

### Analyze Cluster Characteristics

```python
def analyze_clusters(model, bkt_predictions_df, cluster_labels):
    """
    Analyze characteristics of each cluster
    
    Args:
        model: Trained GTransformerV2
        bkt_predictions_df: BKT predictions with [uid, skill, p_L0_bkt, p_T_bkt]
        cluster_labels: [n_uid] cluster assignments
    """
    traits = model.student_traits.weight.detach().cpu().numpy()
    labels = cluster_labels.cpu().numpy()
    
    cluster_names = [
        'Low L0, Low T (Struggling)',
        'Low L0, High T (Fast Learners)',
        'High L0, Low T (Knowledgeable but Slow)',
        'High L0, High T (Advanced)',
    ]
    
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)
    
    for k in range(4):
        mask = (labels == k)
        n_students = mask.sum()
        
        if n_students == 0:
            continue
        
        # Compute cluster statistics
        cluster_traits = traits[mask]
        mean_delta_L0 = cluster_traits[:, 0].mean()
        mean_delta_T = cluster_traits[:, 1].mean()
        std_delta_L0 = cluster_traits[:, 0].std()
        std_delta_T = cluster_traits[:, 1].std()
        
        print(f"\nCluster {k}: {cluster_names[k]}")
        print(f"  Students: {n_students}")
        print(f"  δ_L0: {mean_delta_L0:.3f} ± {std_delta_L0:.3f}")
        print(f"  δ_T:  {mean_delta_T:.3f} ± {std_delta_T:.3f}")
```

## Benefits of v2.0 Architecture

### 1. Interpretability
- **2D visualization**: Student traits can be plotted and interpreted
- **Cluster membership**: Clear grouping of students by learning characteristics
- **Decomposed variance**: Separate population, cluster, and individual effects

### 2. Generalization
- **Shared patterns**: Students in same cluster share learned attention patterns
- **Cold-start**: New students can be assigned to clusters based on initial interactions
- **Transfer learning**: Cluster-level knowledge transfers across skills

### 3. Personalization
- **Individual residuals**: Capture skill-specific deviations from cluster mean
- **Context-aware**: Residuals adapt based on interaction history (z_context)
- **Balanced**: Prevents overfitting through residual regularization

### 4. Grounding
- **Theory-guided**: Population parameters from validated BKT model
- **Cluster coherence**: t-SNE loss ensures meaningful cluster structure
- **Pedagogical alignment**: Clusters correspond to interpretable learning profiles

## Comparison with v1.0

| Aspect | v1.0 (Axis Projection) | v2.0 (Three-Term Decomposition) |
|--------|------------------------|----------------------------------|
| **Parameters** | Base + Context | Population + Traits + Residuals |
| **Grounding** | Linear probes (L_probe) | t-SNE clusters (L_tsne) |
| **Interpretability** | Implicit in axes | Explicit 2D trait space |
| **Clustering** | Not explicit | 4 quadrants (Low/High L0 × T) |
| **Personalization** | Per-skill axes | Per-student traits + residuals |
| **Visualization** | Difficult | Direct 2D scatter plot |
| **Cold-start** | Requires full context | Can assign to cluster |

## Future Extensions

### 1. Hierarchical Clustering
- Use more than 4 clusters (e.g., k-means with k=8)
- Hierarchical structure (coarse → fine-grained)

### 2. Dynamic Traits
- Allow student traits to evolve over time
- Use RNN/LSTM to model trait trajectories

### 3. Multi-Task Learning
- Share cluster structure across multiple datasets
- Transfer cluster definitions to new domains

### 4. Causal Interpretation
- Use traits to recommend interventions
- Predict effect of moving student between clusters

## References

- Bayesian Knowledge Tracing (BKT): Corbett & Anderson (1994)
- t-SNE: van der Maaten & Hinton (2008)
- Deep Knowledge Tracing: Piech et al. (2015)
- Interpretable ML: Rudin (2019)
