"""
iKT3: Interpretable Knowledge Tracing with Alignment-Based Interpretability

Architecture:
- Single Encoder: Transformer processing (q,r) interaction sequences
- Single Output Head: Ability encoder θ_i(t) = MLP(h)
- Fixed β_IRT: Pre-computed skill difficulties (not learnable)
- IRT Formula: m_pred = σ(θ_i(t) - β_k)

Two-Phase Training:
- Phase 1: Performance learning - L_total = L_per (BCE)
- Phase 2: Alignment - L_total = (1-λ_int)×L_per + λ_int×L_ali (both BCE)

Key Simplifications from iKT2:
- No learnable skill difficulties (fixes scale drift)
- No L_reg (fixed β eliminates need)
- Single head instead of dual heads
- Cleaner gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
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

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, d_model] - input sequence
            mask: [L, L] - causal mask
        
        Returns:
            [B, L, d_model]
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # Project and reshape for multi-head attention
        Q = self.query_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
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
    """Transformer encoder block with self-attention and feed-forward network."""
    
    def __init__(self, n_heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, d_model]
            mask: [L, L] - causal mask
        
        Returns:
            [B, L, d_model]
        """
        # Multi-head attention with residual connection
        attn_out = self.attention(x, mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class ScaledAbilityEncoder(nn.Module):
    """
    Ability encoder with automatic scale control based on β_IRT statistics.
    Ensures θ and β have compatible scales for IRT formula σ(θ - β).
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1, beta_irt_stats=None, target_ratio=0.4):
        """
        Args:
            d_model: Hidden dimension from encoder
            d_ff: Feedforward dimension
            dropout: Dropout rate
            beta_irt_stats: Dict with 'mean' and 'std' of β_IRT (from calibration)
            target_ratio: Target θ/β scale ratio (default 0.4, valid range 0.3-0.5)
        """
        super().__init__()
        # Ability encoder receives [h, skill_emb] concatenated
        self.ability_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),  # Changed from d_model to d_model * 2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )
        
        # Initialize scale based on β_IRT statistics
        if beta_irt_stats is not None:
            beta_std = beta_irt_stats['std']
            target_theta_std = target_ratio * beta_std  # e.g., 0.4 × 0.8 = 0.32
            initial_scale = target_theta_std
        else:
            # Fallback: assume typical β std ≈ 0.8
            initial_scale = target_ratio * 0.8
        
        # Learnable scale parameter (can adjust during training within reasonable range)
        self.scale = nn.Parameter(torch.tensor([initial_scale]))
        
        # Store target statistics for monitoring
        self.beta_mean = beta_irt_stats['mean'] if beta_irt_stats else -2.0
        self.beta_std = beta_irt_stats['std'] if beta_irt_stats else 0.8
        self.target_ratio = target_ratio
        
    def forward(self, h, skill_emb):
        """
        Args:
            h: [B, L, d_model] - encoder hidden states
            skill_emb: [B, L, d_model] - skill embeddings for query skills
        
        Returns:
            theta_t: [B, L] - scaled student abilities
        """
        # Concatenate knowledge state with skill embeddings
        concat = torch.cat([h, skill_emb], dim=-1)  # [B, L, d_model * 2]
        
        # Extract raw ability
        theta_raw = self.ability_encoder(concat).squeeze(-1)  # [B, L]
        
        # Apply learned scale
        # If scale = 0.32 (for β std=0.8, ratio=0.4):
        #   - θ will have std ≈ 0.32 (assuming normalized θ_raw)
        #   - θ/β ratio ≈ 0.32/0.8 = 0.4 ✓
        theta_scaled = self.scale * theta_raw
        
        return theta_scaled
    
    def get_scale_info(self, theta_t):
        """Diagnostic: Check if θ/β ratio is in valid range."""
        with torch.no_grad():
            theta_std = theta_t.std().item()
            current_ratio = theta_std / self.beta_std
            
            in_range = 0.3 <= current_ratio <= 0.5
            
            return {
                'theta_std': theta_std,
                'beta_std': self.beta_std,
                'theta_beta_ratio': current_ratio,
                'target_ratio': self.target_ratio,
                'scale_parameter': self.scale.item(),
                'valid': in_range,
            }


class iKT3(nn.Module):
    """
    iKT3: Interpretable Knowledge Tracing with Alignment-Based Interpretability.
    
    Simplified architecture with single output head and fixed skill difficulties.
    """
    
    def __init__(self, num_c, d_model=256, n_heads=4, num_encoder_blocks=8, 
                 d_ff=1536, dropout=0.2, seq_len=200, beta_irt=None, target_ratio=0.4):
        """
        Args:
            num_c: Number of skills (concepts)
            d_model: Model dimension
            n_heads: Number of attention heads
            num_encoder_blocks: Number of transformer encoder layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            seq_len: Maximum sequence length
            beta_irt: Pre-computed IRT skill difficulties [num_c]
            target_ratio: Target θ/β scale ratio (0.3-0.5 valid range)
        """
        super().__init__()
        self.num_c = num_c
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.d_ff = d_ff
        self.dropout = dropout
        self.seq_len = seq_len
        
        # Embedding layers
        # Token space: [0, 2*num_c) where token = q*2 + r
        self.context_embedding = nn.Embedding(num_c * 2, d_model)  # (q,r) pairs
        self.skill_embedding = nn.Embedding(num_c, d_model)  # Skill embeddings for ability encoder
        self.positional_embedding = nn.Embedding(seq_len, d_model)
        
        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_heads, d_model, d_ff, dropout)
            for _ in range(num_encoder_blocks)
        ])
        
        # Compute β_IRT statistics for scale initialization
        beta_irt_stats = None
        if beta_irt is not None:
            beta_irt_stats = {
                'mean': beta_irt.mean().item(),
                'std': beta_irt.std().item(),
            }
        
        # Single output head: Ability encoder with smart scale initialization
        self.ability_encoder = ScaledAbilityEncoder(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            beta_irt_stats=beta_irt_stats,
            target_ratio=target_ratio
        )
        
        # Store fixed β_IRT as buffer (not trainable)
        if beta_irt is not None:
            self.register_buffer('beta_irt', beta_irt)
        else:
            # Fallback: initialize with zeros (will be loaded later)
            self.register_buffer('beta_irt', torch.zeros(num_c))
        
        # Create causal mask once during initialization
        self.register_buffer('causal_mask', self._create_causal_mask(seq_len))
        
        # Initialize weights
        self._init_weights()
    
    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions."""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def load_irt_difficulties(self, beta_irt):
        """
        Load pre-computed IRT skill difficulties.
        
        Args:
            beta_irt: Tensor of shape [num_c] with skill difficulties
        """
        with torch.no_grad():
            self.beta_irt.copy_(beta_irt)
        print(f"✓ Loaded IRT skill difficulties: mean={beta_irt.mean():.4f}, std={beta_irt.std():.4f}")
    
    def forward(self, q, r, qry=None):
        """
        Forward pass for iKT3 model with proper autoregressive prediction.
        
        Args:
            q: Questions [B, L] - skill IDs (current questions for context)
            r: Responses [B, L] - correctness (0/1) (current responses for context)
            qry: Query skills [B, L] - next questions to predict responses for (shifted)
            
        Returns:
            dict with predictions and interpretability components
        
        Note:
            The model uses q[0:t] and r[0:t] to build context, then predicts
            the response to qry[t]. This ensures no information leakage.
            In training, qry should be shft_cseqs (shifted questions).
        """
        # qry must be provided for proper autoregressive prediction
        # It should be the shifted questions (next question at each timestep)
        if qry is None:
            qry = q  # Fallback for compatibility, but this breaks autoregressive property
        
        batch_size, seq_len = q.size()
        
        # 1. Embedding layer: encode (q,r) pairs + positional encoding
        # IMPORTANT: We encode the CURRENT (q,r) pairs for context
        # These represent past interactions available when predicting qry
        tokens = q * 2 + r  # [B, L] - convert (q,r) to token IDs
        context_emb = self.context_embedding(tokens)  # [B, L, d_model]
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0)  # [1, L]
        pos_emb = self.positional_embedding(positions)  # [1, L, d_model]
        h = context_emb + pos_emb  # [B, L, d_model]
        
        # 2. Transformer encoder: process through N layers
        mask = self.causal_mask[:seq_len, :seq_len]  # [L, L]
        for encoder_block in self.encoder_blocks:
            h = encoder_block(h, mask=mask)  # [B, L, d_model]
        
        # 3. Get skill embeddings for query skills (next questions to predict)
        skill_emb = self.skill_embedding(qry)  # [B, L, d_model]
        
        # 4. Ability encoder: extract student ability from context + skill context
        theta_t = self.ability_encoder(h, skill_emb)  # [B, L]
        
        # 5. Lookup skill difficulties from pre-computed IRT values
        beta_k = self.beta_irt[qry]  # [B, L] - fixed, not learnable
        
        # 6. Compute mastery probability using IRT-like formula
        m_pred = torch.sigmoid(theta_t - beta_k)  # [B, L]
        
        # 7. Generate response predictions (same as mastery for performance loss)
        p_correct = m_pred  # [B, L] - continuous predictions for BCE loss
        
        return {
            'p_correct': p_correct,     # Performance predictions (for L_per)
            'm_pred': m_pred,           # Mastery probability (for L_ali)
            'theta_t': theta_t,         # Student ability (interpretability)
            'beta_k': beta_k,           # Skill difficulty (interpretability)
        }
    
    def compute_loss(self, outputs, targets, p_ref=None, phase=1, lambda_int=0.0, mask=None):
        """
        Compute loss based on training phase.
        
        Args:
            outputs: Model outputs dict with 'p_correct', 'm_pred', etc.
            targets: True responses [B, L] in {0, 1}
            p_ref: Reference IRT predictions [B, L] in [0, 1] (required for both phases)
            phase: Training phase (1 or 2)
            lambda_int: Alignment weight λ_int for Phase 2
            mask: Optional mask for valid positions [B, L]
            
        Phase 1: L_total = L_ali (align m_pred with IRT reference)
        Phase 2: L_total = L_per + λ_int×L_ali (optimize performance + maintain alignment)
            
        Returns:
            dict with loss components and total loss
        """
        # Extract predictions from model outputs
        p_correct = outputs['p_correct']  # [B, L]
        m_pred = outputs['m_pred']        # [B, L]
        
        # Validate inputs
        assert torch.all((p_correct >= 0) & (p_correct <= 1)), "p_correct must be in [0, 1]"
        assert torch.all((targets == 0) | (targets == 1)), "targets must be in {0, 1}"
        
        # Convert targets to float for BCE computation
        targets_float = targets.float()  # [B, L]
        
        # Compute performance loss (always computed)
        loss_per = F.binary_cross_entropy(p_correct, targets_float, reduction='none')  # [B, L]
        
        # Apply mask if provided
        if mask is not None:
            loss_per = loss_per * mask
            L_per = loss_per.sum() / mask.sum()
        else:
            L_per = loss_per.mean()
        
        # Phase-dependent loss computation
        if phase == 1:
            # Phase 1: Pure alignment learning with IRT reference
            assert p_ref is not None, "p_ref required for Phase 1"
            assert torch.all((p_ref >= 0) & (p_ref <= 1)), "p_ref must be in [0, 1]"
            
            # Compute alignment loss (BCE equivalent to KL divergence)
            loss_ali = F.binary_cross_entropy(m_pred, p_ref, reduction='none')  # [B, L]
            
            # Apply mask if provided
            if mask is not None:
                loss_ali = loss_ali * mask
                L_ali = loss_ali.sum() / mask.sum()
            else:
                L_ali = loss_ali.mean()
            
            # Phase 1: Use only alignment loss
            L_total = L_ali
            
        elif phase == 2:
            # Phase 2: Combine performance + alignment
            assert p_ref is not None, "p_ref required for Phase 2"
            assert torch.all((p_ref >= 0) & (p_ref <= 1)), "p_ref must be in [0, 1]"
            
            # Compute alignment loss (BCE equivalent to KL divergence)
            loss_ali = F.binary_cross_entropy(m_pred, p_ref, reduction='none')  # [B, L]
            
            # Apply mask if provided
            if mask is not None:
                loss_ali = loss_ali * mask
                L_ali = loss_ali.sum() / mask.sum()
            else:
                L_ali = loss_ali.mean()
            
            # Weighted combination: L_per + λ_int * L_ali
            L_total = L_per + lambda_int * L_ali
            
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1 or 2.")
        
        return {
            'total_loss': L_total,
            'per_loss': L_per,
            'alignment_loss': L_ali,
            'phase': phase,
            'lambda_int': lambda_int,
        }
    
    def monitor_scale_health(self, theta_t, beta_k, epoch):
        """
        Monitor θ/β scale ratio during training.
        Should be called every few epochs to detect scale drift.
        
        Args:
            theta_t: Student abilities [B, L]
            beta_k: Skill difficulties [B, L]
            epoch: Current epoch number
            
        Returns:
            dict with scale health metrics
        """
        with torch.no_grad():
            theta_mean = theta_t.mean().item()
            theta_std = theta_t.std().item()
            beta_mean = beta_k.mean().item()
            beta_std = beta_k.std().item()
            
            ratio = theta_std / (beta_std + 1e-8)
            
            scale_param = self.ability_encoder.scale.item()
            
            print(f"\n[Epoch {epoch}] Scale Health Check:")
            print(f"  θ: mean={theta_mean:.3f}, std={theta_std:.3f}")
            print(f"  β: mean={beta_mean:.3f}, std={beta_std:.3f}")
            print(f"  θ/β ratio: {ratio:.3f} (target: 0.3-0.5)")
            print(f"  Learned scale parameter: {scale_param:.3f}")
            
            # Health status
            if ratio < 0.2:
                print("  ⚠️  WARNING: θ scale too small (ratio < 0.2)")
            elif ratio > 0.6:
                print("  ⚠️  WARNING: θ scale too large (ratio > 0.6)")
            elif 0.3 <= ratio <= 0.5:
                print("  ✅ Scale ratio healthy")
            else:
                print("  ⚡ Scale ratio acceptable but suboptimal")
            
            return {
                'theta_mean': theta_mean,
                'theta_std': theta_std,
                'beta_mean': beta_mean,
                'beta_std': beta_std,
                'ratio': ratio,
                'scale_parameter': scale_param,
                'healthy': 0.3 <= ratio <= 0.5,
            }
