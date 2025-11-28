"""
iKT: Interpretable Knowledge Tracing with Rasch-Grounded Mastery Estimation

Architecture:
- Single Encoder: Questions + Responses (binary)
  → Head 1 (Performance): BCE Loss (L1)
  → Head 2 (Mastery): Outputs skill vector {Mi} [B, L, num_c] with Rasch Loss (L2)

Two-Phase Training:
- Phase 1: Psychometric Grounding - L_total = L2 = MSE(Mi, M_rasch) with epsilon=0
- Phase 2: Constrained Optimization - L_total = L1 + λ_penalty × mean(max(0, |Mi - M_rasch| - ε)²)

Phase 2 Strategy:
- Primary objective: Optimize L1 (BCE) for maximum AUC
- Constraint enforcement: Large penalty (λ_penalty) on deviations beyond tolerance (ε)
- No penalty when |Mi - M_rasch| ≤ ε (soft barrier approach)

Key Features:
- Positivity: Softplus activation ensures Mi > 0
- Monotonicity: cummax ensures non-decreasing mastery over time
- Interpretability: Direct comparison to IRT-based theoretical mastery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with dual-stream input (context for Q/K, value for V).
    """
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_sequence, value_sequence, mask=None):
        """
        Args:
            context_sequence: [B, L, d_model] - for Q and K
            value_sequence: [B, L, d_model] - for V
            mask: [L, L] - causal mask
        
        Returns:
            [B, L, d_model]
        """
        batch_size, seq_len = context_sequence.size(0), context_sequence.size(1)

        # Project and reshape for multi-head attention
        Q = self.query_proj(context_sequence).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(context_sequence).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(value_sequence).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [B, n_heads, L, d_k]

        # Scaled dot-product attention with float32 for numerical stability (matches GainAKT3)
        orig_dtype = Q.dtype
        scores = torch.matmul(Q.to(torch.float32), K.transpose(-2, -1).to(torch.float32)) / (self.d_k ** 0.5)
        
        if mask is not None:
            # Use dtype-dependent masking value to prevent overflow
            if orig_dtype == torch.float16:
                neg_fill = -1e4
            elif orig_dtype == torch.bfloat16:
                neg_fill = -1e30
            else:
                neg_fill = float('-inf')
            scores = scores.masked_fill(mask == 0, neg_fill)
        
        # Cast back to original dtype
        scores = scores.to(orig_dtype)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, n_heads, L, d_k]
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.output_proj(attn_output)
        
        return output


class EncoderBlock(nn.Module):
    """
    Transformer encoder block with dual-stream processing.
    Context and Value streams have separate residual connections and layer norms.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        self.attention = MultiHeadAttention(n_heads, d_model, dropout)
        
        # Separate layer norms for dual streams
        self.norm1_ctx = nn.LayerNorm(d_model)
        self.norm1_val = nn.LayerNorm(d_model)
        self.norm2_ctx = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_seq, value_seq, mask=None):
        """
        Args:
            context_seq: [B, L, d_model]
            value_seq: [B, L, d_model]
            mask: [L, L]
        
        Returns:
            tuple: (context_seq, value_seq) both [B, L, d_model]
        """
        # Multi-head attention with dual streams
        attn_output = self.attention(context_seq, value_seq, mask)
        
        # Separate residual for each stream (NO dropout on attention path - matches GainAKT3Exp)
        context_seq = self.norm1_ctx(context_seq + attn_output)
        value_seq = self.norm1_val(value_seq + attn_output)
        
        # Feed-forward only on context stream (dropout on FFN path - matches GainAKT3Exp)
        ffn_output = self.ffn(context_seq)
        context_seq = self.norm2_ctx(context_seq + self.dropout(ffn_output))
        
        return context_seq, value_seq


class iKT(nn.Module):
    """
    iKT: Interpretable Knowledge Tracing with Single-Encoder, Two-Head Architecture
    
    Architecture:
        Questions + Responses → Encoder → h ──┬→ Head 1 (Performance) → BCE Predictions → L1
                                              └→ Head 2 (Mastery) → {Mi} [B, L, num_c] → L2 (Rasch)
    
    Two-phase training:
        Phase 1: L_total = L2 (epsilon=0, direct Rasch alignment)
        Phase 2: L_total = λ_bce * L1 + (1 - λ_bce) * L2 (epsilon>0, tolerance-based)
    
    Constraint: λ_bce + λ_mastery = 1.0 (λ_mastery = 1 - λ_bce)
    """
    
    def __init__(self, num_c, seq_len, d_model, n_heads, num_encoder_blocks,
                 d_ff, dropout, emb_type, lambda_penalty, 
                 epsilon, phase):
        super().__init__()
        
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.emb_type = emb_type
        self.epsilon = epsilon  # Tolerance threshold for Phase 2
        self.phase = phase  # 1 or 2
        
        # Penalty weight for Phase 2 constraint enforcement
        self.lambda_penalty = lambda_penalty
        
        # Validate lambda_penalty is positive
        assert lambda_penalty > 0, f"lambda_penalty must be positive, got {lambda_penalty}"
        
        # === SINGLE ENCODER: Performance & Mastery Pathway ===
        # Embeddings for binary responses (0/1)
        self.context_embedding = nn.Embedding(num_c * 2, d_model)  # q + num_c * r
        self.value_embedding = nn.Embedding(num_c * 2, d_model)
        self.skill_embedding = nn.Embedding(num_c, d_model)  # For Head 1
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_blocks)
        ])
        
        # Skill difficulty embeddings (Option 1b)
        # Learnable parameters regularized toward IRT-calibrated values
        self.skill_difficulty_emb = nn.Embedding(num_c, 1)
        # Initialize to neutral value (0.0 difficulty -> 0.5 mastery probability)
        # Will be overridden with IRT values during training initialization
        nn.init.constant_(self.skill_difficulty_emb.weight, 0.0)
        
        # Head 1: Performance Prediction (BCE)
        # Deeper architecture matching AKT for better capacity
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),  # [h, v, skill_emb]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 256),  # Additional layer (matches AKT)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Head 2: Mastery Estimation
        # MLP1: h → {Mi} skill vector with Softplus for positivity
        # Output: [B, L, num_c] - direct skill mastery vector (no MLP2 aggregation)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_c),
            nn.Softplus()  # Ensures Mi > 0
        )
    
    def forward(self, q, r, qry=None):
        """
        Args:
            q: [B, L] - question IDs
            r: [B, L] - responses (0 or 1)
            qry: [B, L] - query question IDs (optional, defaults to q)
        
        Returns:
            dict with keys:
                - 'bce_predictions': [B, L] from Head 1
                - 'skill_vector': [B, L, num_c] - interpretable {Mi}
                - 'beta_targets': [B, L, num_c] - skill-only mastery targets from embeddings
                - 'logits': [B, L] - BCE logits for loss computation
        """
        batch_size, seq_len = q.size()
        device = q.device
        
        if qry is None:
            qry = q
        
        # Create interaction tokens: q + num_c * r
        r_int = r.long()
        interaction_tokens = q + self.num_c * r_int
        
        # Create positional indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        
        # === ENCODER: Tokenization & Embedding ===
        context_seq = self.context_embedding(interaction_tokens) + self.pos_embedding(positions)
        value_seq = self.value_embedding(interaction_tokens) + self.pos_embedding(positions)
        # context_seq, value_seq: [B, L, d_model]
        
        # === ENCODER: Dual-Stream Encoder Stack ===
        for encoder_block in self.encoder_blocks:
            context_seq, value_seq = encoder_block(context_seq, value_seq, mask)
        
        h = context_seq  # Knowledge state [B, L, d_model]
        v = value_seq    # Value state [B, L, d_model]
        
        # === HEAD 1: Performance Prediction ===
        skill_emb = self.skill_embedding(qry)  # [B, L, d_model]
        concat = torch.cat([h, v, skill_emb], dim=-1)  # [B, L, 3*d_model]
        logits = self.prediction_head(concat).squeeze(-1)  # [B, L]
        bce_predictions = torch.sigmoid(logits)
        
        # === HEAD 2: Mastery Estimation ===
        # Always compute mastery head - this is the core feature of iKT
        # Step 1: MLP1 → Skill Vector {Mi} with positivity
        skill_vector = self.mlp1(h)  # [B, L, num_c], guaranteed positive by Softplus
        
        # Step 2: Enforce monotonicity (cumulative max across time)
        skill_vector = torch.cummax(skill_vector, dim=1)[0]  # [B, L, num_c]
        
        # Step 3: Compute skill-only mastery targets from difficulty embeddings
        # Extract skill difficulties: [K]
        beta_skills = self.skill_difficulty_emb.weight.squeeze(-1)  # [num_c]
        
        # Compute mastery targets: M_k = sigmoid(-beta_k)
        # Higher difficulty (positive beta) -> lower mastery probability
        mastery_targets_1d = torch.sigmoid(-beta_skills)  # [num_c]
        
        # Broadcast to batch dimensions: [1, 1, num_c] -> [B, L, num_c]
        beta_targets = mastery_targets_1d.unsqueeze(0).unsqueeze(0)  # [1, 1, num_c]
        beta_targets = beta_targets.expand(batch_size, seq_len, -1)  # [B, L, num_c]
        
        return {
            'bce_predictions': bce_predictions,
            'skill_vector': skill_vector,  # Interpretable {Mi} [B, L, num_c]
            'beta_targets': beta_targets,  # Skill-only targets [B, L, num_c]
            'logits': logits
        }
    
    def compute_loss(self, output, targets, beta_irt=None, lambda_reg=0.1):
        """
        Compute loss with phase-dependent behavior and skill regularization.
        
        Phase 1: L_total = L1 + λ_reg × L_reg
        Phase 2: L_total = L1 + λ_penalty × L2_penalty + λ_reg × L_reg
                 where L2_penalty = mean(max(0, |Mi - sigma(-beta)| - ε)²)
                       L_reg = MSE(beta_learned, beta_irt)
        
        Args:
            output: dict from forward()
            targets: [B, L] - ground truth responses (0 or 1) for L1
            beta_irt: [K] - IRT-calibrated skill difficulties (optional)
            lambda_reg: regularization strength for skill embeddings
        
        Returns:
            dict with keys:
                - 'total_loss': phase-dependent weighted loss
                - 'bce_loss': L1 (performance prediction)
                - 'penalty_loss': L2_penalty (constraint violation penalty)
                - 'reg_loss': L_reg (skill difficulty regularization)
        """
        device = output['logits'].device
        
        # L1: BCE Loss (Head 1 - Performance Prediction)
        bce_loss = F.binary_cross_entropy_with_logits(
            output['logits'],
            targets.float(),
            reduction='mean'
        )
        
        # L_reg: Skill Difficulty Regularization Loss
        if beta_irt is not None:
            beta_learned = self.skill_difficulty_emb.weight.squeeze(-1)  # [K]
            reg_loss = F.mse_loss(beta_learned, beta_irt, reduction='mean')
        else:
            # No IRT targets - no regularization
            reg_loss = torch.tensor(0.0, device=device)
        
        # L2_penalty: Interpretability Constraint (Phase 2 only)
        if self.phase == 2:
            skill_vector = output['skill_vector']  # [B, L, K]
            beta_targets = output['beta_targets']  # [B, L, K]
            
            # Compute violations: max(0, |Mi - sigma(-beta)| - epsilon)
            deviation = torch.abs(skill_vector - beta_targets)
            violation = torch.relu(deviation - self.epsilon)
            penalty_loss = torch.mean(violation ** 2)
        else:
            penalty_loss = torch.tensor(0.0, device=device)
        
        # Phase-dependent total loss
        if self.phase == 1:
            # Phase 1: Predictive learning + skill regularization
            total_loss = bce_loss + lambda_reg * reg_loss
        else:
            # Phase 2: Performance + interpretability + regularization
            total_loss = bce_loss + self.lambda_penalty * penalty_loss + lambda_reg * reg_loss
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'penalty_loss': penalty_loss,
            'reg_loss': reg_loss
        }


def create_model(config):
    """
    Factory function to create iKT model.
    
    All parameters are REQUIRED and must be present in config.
    No hardcoded defaults per reproducibility guidelines.
    Defaults should come from configs/parameter_default.json.
    
    Required config keys:
        - num_c: number of skills
        - seq_len: sequence length
        - d_model: model dimension
        - n_heads: number of attention heads
        - num_encoder_blocks: number of encoder blocks
        - d_ff: feed-forward dimension
        - dropout: dropout rate
        - emb_type: embedding type
        - lambda_penalty: penalty coefficient for Phase 2 constraint
                         Recommended range: [10.0, 1000.0]
        - epsilon: tolerance threshold for Phase 2
                   Recommended range: [0.05, 0.15]
        - phase: training phase 1 or 2
    """
    # Fail-fast: require all parameters (no .get() with defaults)
    required_keys = ['num_c', 'seq_len', 'd_model', 'n_heads', 'num_encoder_blocks',
                     'd_ff', 'dropout', 'emb_type', 'lambda_penalty', 'epsilon', 'phase']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Required config parameter '{key}' not found. "
                          f"All defaults must be specified in parameter_default.json")
    
    return iKT(
        num_c=config['num_c'],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_blocks=config['num_encoder_blocks'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        emb_type=config['emb_type'],
        lambda_penalty=config['lambda_penalty'],
        epsilon=config['epsilon'],
        phase=config['phase']
    )
