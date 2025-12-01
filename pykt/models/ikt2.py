"""
iKT2: Interpretable Knowledge Tracing with IRT-Based Mastery Inference

Architecture:
- Single Encoder: Questions + Responses (binary)
  → Head 1 (Performance): BCE Loss (L_BCE)
  → Head 2 (IRT Mastery): Ability encoder + IRT formula → M_IRT = σ(θ - β)

Two-Phase Training:
- Phase 1: Warmup - L_total = L_BCE + λ_reg × L_reg (epochs 1-10)
- Phase 2: IRT Alignment - L_total = L_BCE + λ_align × L_align + λ_reg × L_reg (epochs 11+)

Key Innovation:
- Ability encoder extracts θ_i(t) from knowledge state h
- IRT formula M_IRT = σ(θ_i(t) - β_k) computes mastery probability
- L_align ensures predictions align with IRT expectations

Advantages over iKT (Option 1b):
- Theoretically grounded (Rasch IRT model)
- No epsilon tolerance needed
- Simpler hyperparameters (only λ_align)
- Better interpretability (IRT correlation instead of violation rate)
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

        # Scaled dot-product attention with float32 for numerical stability
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
        
        # Separate layer norms for dual streams (symmetric)
        self.norm1_ctx = nn.LayerNorm(d_model)
        self.norm1_val = nn.LayerNorm(d_model)
        self.norm2_ctx = nn.LayerNorm(d_model)
        self.norm2_val = nn.LayerNorm(d_model)  # Added for symmetric value stream
        
        # Feed-forward networks (one per stream for symmetric processing)
        self.ffn_ctx = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_val = nn.Sequential(
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
        
        # Separate residual for each stream (NO dropout on attention path)
        context_seq = self.norm1_ctx(context_seq + attn_output)
        value_seq = self.norm1_val(value_seq + attn_output)
        
        # Feed-forward on BOTH streams (symmetric processing)
        ffn_output_ctx = self.ffn_ctx(context_seq)
        context_seq = self.norm2_ctx(context_seq + self.dropout(ffn_output_ctx))
        
        ffn_output_val = self.ffn_val(value_seq)
        value_seq = self.norm2_val(value_seq + self.dropout(ffn_output_val))
        
        return context_seq, value_seq


class iKT2(nn.Module):
    """
    iKT2: Interpretable Knowledge Tracing with IRT-Based Mastery Inference
    
    Architecture:
        Questions + Responses → Encoder → h ──┬→ Head 1 (Performance) → p_correct → L_BCE
                                              └→ Head 2 (IRT Mastery) → θ, β → M_IRT → L_align
    
    Two-phase training:
        Phase 1 (epochs 1-10): L_total = L_BCE + λ_reg × L_reg (warmup)
        Phase 2 (epochs 11+): L_total = L_BCE + λ_align × L_align + λ_reg × L_reg (IRT alignment)
    
    Key Innovation:
        - Ability encoder extracts θ_i(t) from knowledge state h
        - IRT formula M_IRT = σ(θ_i(t) - β_k) computes mastery
        - L_align = MSE(p_correct, M_IRT) ensures IRT consistency
    """
    
    def __init__(self, num_c, seq_len, d_model, n_heads, num_encoder_blocks,
                 d_ff, dropout, emb_type, lambda_align, lambda_reg, phase):
        """
        Initialize iKT2 model.
        
        All parameters REQUIRED (no defaults) per reproducibility guidelines.
        Defaults must come from configs/parameter_default.json.
        """
        super().__init__()
        
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.emb_type = emb_type
        self.phase = phase  # 1 or 2
        
        # Hyperparameters (explicit, no defaults)
        self.lambda_align = lambda_align  # IRT alignment weight (Phase 2)
        self.lambda_reg = lambda_reg      # Difficulty regularization weight (both phases)
        
        # Validate positive weights
        assert lambda_align > 0, f"lambda_align must be positive, got {lambda_align}"
        assert lambda_reg > 0, f"lambda_reg must be positive, got {lambda_reg}"
        
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
        
        # Skill difficulty embeddings (retained from Option 1b)
        # Learnable parameters regularized toward IRT-calibrated values
        self.skill_difficulty_emb = nn.Embedding(num_c, 1)
        # Initialize to neutral value (0.0 difficulty) - will be overridden by load_irt_difficulties()
        nn.init.constant_(self.skill_difficulty_emb.weight, 0.0)
        
        # === HEAD 1: Performance Prediction ===
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),  # [h, v, skill_emb]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 256),  # Additional layer for depth
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # === HEAD 2: IRT-Based Mastery Inference (NEW) ===
        # Ability encoder: extracts scalar student ability θ_i(t) from knowledge state h
        self.ability_encoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)  # Output: scalar ability θ_i(t)
        )
    
    def load_irt_difficulties(self, beta_irt):
        """
        Initialize skill difficulty embeddings from IRT-calibrated values.
        
        This fixes scale drift: instead of starting from 0.0 and relying on weak
        regularization to reach IRT scale (mean=-0.6, std=0.4), we initialize
        directly from IRT calibration. Model can still fine-tune based on data.
        
        Args:
            beta_irt: [num_c] tensor - IRT-calibrated skill difficulties
        
        Returns:
            None (modifies self.skill_difficulty_emb.weight in-place)
        """
        with torch.no_grad():
            self.skill_difficulty_emb.weight.copy_(beta_irt.view(-1, 1))
        print(f"✓ Initialized skill difficulties from IRT calibration")
        print(f"  β_init: mean={beta_irt.mean().item():.4f}, std={beta_irt.std().item():.4f}")
    
    def forward(self, q, r, qry=None):
        """
        Args:
            q: [B, L] - question IDs
            r: [B, L] - responses (0 or 1)
            qry: [B, L] - query question IDs (optional, defaults to q)
        
        Returns:
            dict with keys:
                - 'bce_predictions': [B, L] - performance predictions from Head 1
                - 'theta_t': [B, L] - student ability at each timestep
                - 'beta_k': [B, L] - skill difficulty for each question
                - 'mastery_irt': [B, L] - IRT-based mastery M = σ(θ - β)
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
        
        # === HEAD 2: IRT-Based Mastery Inference ===
        # Step 1: Infer student ability from knowledge state
        theta_t = self.ability_encoder(h).squeeze(-1)  # [B, L] - scalar ability per timestep
        
        # Step 2: Extract skill difficulties for questions being answered
        beta_k = self.skill_difficulty_emb(qry).squeeze(-1)  # [B, L] - difficulty per question
        
        # Step 3: Compute IRT mastery probability using Rasch formula
        mastery_irt = torch.sigmoid(theta_t - beta_k)  # [B, L] - M_IRT = σ(θ - β)
        
        return {
            'bce_predictions': bce_predictions,
            'performance_logits': logits,  # Performance logits
            'theta_t': theta_t,          # Student ability
            'beta_k': beta_k,            # Skill difficulty
            'mastery_irt': mastery_irt,  # IRT-based mastery
            'p_correct': torch.sigmoid(logits),  # Predicted correctness probability
            'logits': logits  # Legacy compatibility
        }
    
    def compute_loss(self, output, targets, beta_irt, lambda_reg):
        """
        Compute phase-dependent loss with IRT alignment.
        
        Phase 1 (Warmup): L_total = L_BCE + λ_reg × L_reg
        Phase 2 (IRT Alignment): L_total = L_BCE + λ_align × L_align + λ_reg × L_reg
        
        Args:
            output: dict from forward()
            targets: [B, L] - ground truth responses (0 or 1)
            beta_irt: [K] - IRT-calibrated skill difficulties (required, can be None tensor)
            lambda_reg: Regularization weight (required, explicit)
        
        Returns:
            dict with keys:
                - 'total_loss': phase-dependent weighted loss
                - 'bce_loss': L_BCE (performance prediction)
                - 'alignment_loss': L_align (IRT alignment, Phase 2 only)
                - 'reg_loss': L_reg (skill difficulty regularization)
        
        Note:
            All parameters REQUIRED. No defaults per reproducibility guidelines.
            Code will fail if parameters not explicitly provided.
        """
        device = output['logits'].device
        
        # L_BCE: Binary Cross-Entropy Loss (Head 1 - Performance Prediction)
        bce_loss = F.binary_cross_entropy_with_logits(
            output['logits'],
            targets.float(),
            reduction='mean'
        )
        
        # L_reg: Skill Difficulty Regularization Loss (Both Phases)
        if beta_irt is not None:
            beta_learned = self.skill_difficulty_emb.weight.squeeze(-1)  # [K]
            reg_loss = F.mse_loss(beta_learned, beta_irt, reduction='mean')
        else:
            # No IRT targets - no regularization
            reg_loss = torch.tensor(0.0, device=device)
        
        # L_align: IRT Alignment Loss (Phase 2 Only)
        if self.phase == 2:
            # Ensure predictions align with IRT mastery expectations
            alignment_loss = F.mse_loss(
                output['bce_predictions'],
                output['mastery_irt'],
                reduction='mean'
            )
        else:
            alignment_loss = torch.tensor(0.0, device=device)
        
        # Phase-dependent total loss
        if self.phase == 1:
            # Phase 1: Warmup - performance learning + embedding regularization
            total_loss = bce_loss + lambda_reg * reg_loss
        else:
            # Phase 2: IRT Alignment - performance + IRT consistency + regularization
            total_loss = bce_loss + self.lambda_align * alignment_loss + lambda_reg * reg_loss
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'align_loss': alignment_loss,  # IRT alignment loss
            'alignment_loss': alignment_loss,  # Legacy compatibility
            'reg_loss': reg_loss
        }


def create_model(config):
    """
    Factory function to create iKT2 model.
    
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
        - lambda_align: IRT alignment weight (Phase 2)
                       Recommended: 1.0
        - lambda_reg: difficulty regularization weight (both phases)
                     Recommended: 0.1
        - phase: training phase 1 or 2
    """
    # Fail-fast: require all parameters (no .get() with defaults)
    required_keys = ['num_c', 'seq_len', 'd_model', 'n_heads', 'num_encoder_blocks',
                     'd_ff', 'dropout', 'emb_type', 'lambda_align', 'lambda_reg', 'phase']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Required config parameter '{key}' not found. "
                          f"All defaults must be specified in parameter_default.json")
    
    return iKT2(
        num_c=config['num_c'],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_blocks=config['num_encoder_blocks'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        emb_type=config['emb_type'],
        lambda_align=config['lambda_align'],
        lambda_reg=config['lambda_reg'],
        phase=config['phase']
    )
