"""
GainAKT4: Dual-Encoder, Three-Head Knowledge Tracing Model

Architecture:
- Encoder 1 (Performance & Mastery Pathway): Questions + Responses (binary)
  → Head 1 (Performance): BCE Loss (L1)
  → Head 2 (Mastery): Mastery Loss (L2)
  
- Encoder 2 (Curve Learning Pathway): Questions + Attempts (integer)
  → Head 3 (Curve): MSE/MAE Loss (L3)

Multi-task learning: L_total = λ_bce * L1 + λ_mastery * L2 + λ_curve * L3
Constraint: λ_bce + λ_mastery + λ_curve = 1.0
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


class GainAKT4(nn.Module):
    """
    GainAKT4: Dual-Encoder, Three-Head Architecture
    
    Architecture:
        Questions + Responses → Encoder 1 → h1 ──┬→ Head 1 (Performance) → BCE Predictions → L1
                                                 └→ Head 2 (Mastery) → {KCi} → Mastery Predictions → L2
        
        Questions + Attempts → Encoder 2 → h2 ──→ Head 3 (Curve) → Curve Predictions → L3
    
    Multi-task loss: L_total = λ_bce * L1 + λ_mastery * L2 + λ_curve * L3
    Constraint: λ_bce + λ_mastery + λ_curve = 1.0
    """
    
    def __init__(self, num_c, seq_len, d_model=256, n_heads=4, num_encoder_blocks=4,
                 d_ff=512, dropout=0.2, emb_type='qid', lambda_bce=0.7, lambda_mastery=None, 
                 lambda_curve=0.0, max_attempts=10):
        super().__init__()
        
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.emb_type = emb_type
        self.max_attempts = max_attempts
        
        # Lambda weights with automatic computation for backward compatibility
        # If lambda_mastery is None, compute it as 1.0 - lambda_bce - lambda_curve
        if lambda_mastery is None:
            lambda_mastery = 1.0 - lambda_bce - lambda_curve
        
        self.lambda_bce = lambda_bce
        self.lambda_mastery = lambda_mastery
        self.lambda_curve = lambda_curve
        
        # Validate constraint: sum must equal 1.0
        lambda_sum = lambda_bce + lambda_mastery + lambda_curve
        assert abs(lambda_sum - 1.0) < 1e-6, f"Lambda weights must sum to 1.0, got {lambda_sum} (λ_bce={lambda_bce}, λ_mastery={lambda_mastery}, λ_curve={lambda_curve})"
        
        # === ENCODER 1: Performance & Mastery Pathway ===
        # Embeddings for binary responses (0/1)
        self.context_embedding = nn.Embedding(num_c * 2, d_model)  # q + num_c * r
        self.value_embedding = nn.Embedding(num_c * 2, d_model)
        self.skill_embedding = nn.Embedding(num_c, d_model)  # For Head 1
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Encoder 1 blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_blocks)
        ])
        
        # === ENCODER 2: Curve Learning Pathway ===
        # Embeddings for integer attempts (0 to max_attempts)
        self.context_embedding2 = nn.Embedding(num_c * (max_attempts + 1), d_model)
        self.value_embedding2 = nn.Embedding(num_c * (max_attempts + 1), d_model)
        self.skill_embedding2 = nn.Embedding(num_c, d_model)  # Separate from Encoder 1
        # Share positional embedding (optional: could create separate pos_embedding2)
        
        # Encoder 2 blocks (same structure as Encoder 1)
        self.encoder_blocks2 = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_blocks)
        ])
        
        # Head 1: Performance Prediction (BCE)
        # Deeper architecture matching AKT for better capacity
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),  # [h1, v1, skill_emb]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 256),  # Additional layer (matches AKT)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Head 2: Mastery Estimation
        # MLP1: h1 → {KCi} with Softplus for positivity
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_c),
            nn.Softplus()  # Ensures KCi > 0
        )
        
        # MLP2: {KCi} → mastery prediction
        self.mlp2 = nn.Sequential(
            nn.Linear(num_c, num_c // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_c // 2, 1)
        )
        
        # === HEAD 3: Curve Prediction (Integer Regression) ===
        # Same structure as Head 1 (3-layer MLP)
        self.curve_head = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),  # [h2, v2, skill_emb2]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Integer regression (no sigmoid)
        )
    
    def forward(self, q, r, qry=None, attempts=None):
        """
        Args:
            q: [B, L] - question IDs (shared by both encoders)
            r: [B, L] - responses (0 or 1) for Encoder 1
            qry: [B, L] - query question IDs (optional, defaults to q)
            attempts: [B, L] - integer attempts-to-mastery for Encoder 2 (optional)
        
        Returns:
            dict with keys:
                - 'bce_predictions': [B, L] from Head 1
                - 'mastery_predictions': [B, L] from Head 2 (None if λ_mastery=0)
                - 'skill_vector': [B, L, num_c] - interpretable {KCi} (None if λ_mastery=0)
                - 'curve_predictions': [B, L] from Head 3 (None if λ_curve=0)
                - 'logits': [B, L] - BCE logits for loss computation
                - 'mastery_logits': [B, L] - Mastery logits (None if λ_mastery=0)
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
        
        # === ENCODER 1: Tokenization & Embedding ===
        context_seq = self.context_embedding(interaction_tokens) + self.pos_embedding(positions)
        value_seq = self.value_embedding(interaction_tokens) + self.pos_embedding(positions)
        # context_seq, value_seq: [B, L, d_model]
        
        # === ENCODER 1: Dual-Stream Encoder Stack ===
        for encoder_block in self.encoder_blocks:
            context_seq, value_seq = encoder_block(context_seq, value_seq, mask)
        
        h1 = context_seq  # Knowledge state [B, L, d_model]
        v1 = value_seq    # Value state [B, L, d_model]
        
        # === HEAD 1: Performance Prediction ===
        skill_emb = self.skill_embedding(qry)  # [B, L, d_model]
        concat = torch.cat([h1, v1, skill_emb], dim=-1)  # [B, L, 3*d_model]
        logits = self.prediction_head(concat).squeeze(-1)  # [B, L]
        bce_predictions = torch.sigmoid(logits)
        
        # === HEAD 2: Mastery Estimation ===
        # Conditional computation: skip if λ_mastery = 0 (no gradient flow)
        if self.lambda_mastery > 0:
            # Step 1: MLP1 → Skill Vector {KCi} with positivity
            kc_vector = self.mlp1(h1)  # [B, L, num_c], guaranteed positive by Softplus
            
            # Step 1.5: Enforce monotonicity (cumulative max across time)
            kc_vector_mono = torch.cummax(kc_vector, dim=1)[0]  # [B, L, num_c]
            
            # Step 2: MLP2 → Mastery prediction
            mastery_logits = self.mlp2(kc_vector_mono).squeeze(-1)  # [B, L]
            mastery_predictions = torch.sigmoid(mastery_logits)
        else:
            # Skip mastery head computation when λ_mastery=0 (pure BCE mode)
            kc_vector = None
            kc_vector_mono = None
            mastery_logits = None
            mastery_predictions = None
        
        # === ENCODER 2 + HEAD 3: Curve Prediction ===
        # Conditional computation: skip if λ_curve = 0 or attempts not provided
        if self.lambda_curve > 0 and attempts is not None:
            # Clip attempts to valid range [0, max_attempts]
            attempts_clipped = torch.clamp(attempts.long(), 0, self.max_attempts)
            
            # Create interaction tokens: q + num_c * attempts
            interaction_tokens2 = q + self.num_c * attempts_clipped
            
            # === ENCODER 2: Tokenization & Embedding ===
            context_seq2 = self.context_embedding2(interaction_tokens2) + self.pos_embedding(positions)
            value_seq2 = self.value_embedding2(interaction_tokens2) + self.pos_embedding(positions)
            # context_seq2, value_seq2: [B, L, d_model]
            
            # === ENCODER 2: Dual-Stream Encoder Stack ===
            for encoder_block in self.encoder_blocks2:
                context_seq2, value_seq2 = encoder_block(context_seq2, value_seq2, mask)
            
            h2 = context_seq2  # Curve representation [B, L, d_model]
            v2 = value_seq2    # Value state [B, L, d_model]
            
            # === HEAD 3: Curve Prediction ===
            skill_emb2 = self.skill_embedding2(qry)  # [B, L, d_model]
            concat2 = torch.cat([h2, v2, skill_emb2], dim=-1)  # [B, L, 3*d_model]
            curve_predictions = self.curve_head(concat2).squeeze(-1)  # [B, L]
        else:
            # Skip Encoder 2 + Head 3 when λ_curve=0 or no attempts data
            curve_predictions = None
        
        return {
            'bce_predictions': bce_predictions,
            'mastery_predictions': mastery_predictions,
            'skill_vector': kc_vector_mono,  # Interpretable {KCi}
            'curve_predictions': curve_predictions,  # Integer regression predictions
            'logits': logits,
            'mastery_logits': mastery_logits
        }
    
    def compute_loss(self, output, targets, attempts_targets=None):
        """
        Compute multi-task loss with three components.
        
        Args:
            output: dict from forward()
            targets: [B, L] - ground truth responses (0 or 1) for L1 and L2
            attempts_targets: [B, L] - ground truth attempts-to-mastery (integers) for L3
        
        Returns:
            dict with keys:
                - 'total_loss': weighted sum of all losses
                - 'bce_loss': L1 (performance prediction)
                - 'mastery_loss': L2 (mastery estimation)
                - 'curve_loss': L3 (curve prediction)
        """
        device = output['logits'].device
        
        # L1: BCE Loss (Head 1 - Performance Prediction)
        bce_loss = F.binary_cross_entropy_with_logits(
            output['logits'],
            targets.float(),
            reduction='mean'
        )
        
        # L2: Mastery Loss (Head 2 - Mastery Estimation)
        if output['mastery_logits'] is not None:
            mastery_loss = F.binary_cross_entropy_with_logits(
                output['mastery_logits'],
                targets.float(),
                reduction='mean'
            )
        else:
            mastery_loss = torch.tensor(0.0, device=device)
        
        # L3: Curve Loss (Head 3 - Curve Prediction)
        if output['curve_predictions'] is not None and attempts_targets is not None:
            # MSE loss for integer regression
            curve_loss = F.mse_loss(
                output['curve_predictions'],
                attempts_targets.float(),
                reduction='mean'
            )
        else:
            curve_loss = torch.tensor(0.0, device=device)
        
        # Total loss: L_total = λ_bce * L1 + λ_mastery * L2 + λ_curve * L3
        total_loss = (
            self.lambda_bce * bce_loss + 
            self.lambda_mastery * mastery_loss + 
            self.lambda_curve * curve_loss
        )
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'mastery_loss': mastery_loss,
            'curve_loss': curve_loss
        }


def create_model(config):
    """
    Factory function to create GainAKT4 model.
    
    Required config keys:
        - num_c: number of skills
        - seq_len: sequence length
        - d_model: model dimension (default: 256)
        - n_heads: number of attention heads (default: 4)
        - num_encoder_blocks: number of encoder blocks (default: 4)
        - d_ff: feed-forward dimension (default: 512)
        - dropout: dropout rate (default: 0.2)
        - emb_type: embedding type (default: 'qid')
        - lambda_bce: BCE loss weight (default: 0.7)
        - lambda_mastery: Mastery loss weight (default: 0.2)
        - lambda_curve: Curve loss weight (default: 0.1)
        - max_attempts: Maximum attempts for curve learning (default: 10)
    
    Note: lambda_bce + lambda_mastery + lambda_curve must equal 1.0
    """
    return GainAKT4(
        num_c=config['num_c'],
        seq_len=config['seq_len'],
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        num_encoder_blocks=config.get('num_encoder_blocks', 4),
        d_ff=config.get('d_ff', 512),
        dropout=config.get('dropout', 0.2),
        emb_type=config.get('emb_type', 'qid'),
        lambda_bce=config.get('lambda_bce', 0.7),
        lambda_mastery=config.get('lambda_mastery', 0.2),
        lambda_curve=config.get('lambda_curve', 0.1),
        max_attempts=config.get('max_attempts', 10)
    )
