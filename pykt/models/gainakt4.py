"""
GainAKT4: Dual-Head Single-Encoder Knowledge Tracing Model

A simplified architecture with one encoder and two prediction heads:
- Head 1 (Performance): Next-step correctness prediction → BCE Loss
- Head 2 (Mastery): Skill-level mastery estimation → Mastery Loss

Multi-task learning with gradient accumulation to a single shared encoder.
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
    GainAKT4: Dual-Head Single-Encoder Architecture
    
    Architecture:
        Input → Encoder 1 → h1 ──┬→ Head 1 (Performance) → BCE Predictions → L1
                                 └→ Head 2 (Mastery) → {KCi} → Mastery Predictions → L2
    
    Multi-task loss: L_total = λ₁ * L1 + λ₂ * L2
    Constraint: λ₁ + λ₂ = 1.0 (only λ₁ is configurable, λ₂ = 1 - λ₁)
    """
    
    def __init__(self, num_c, seq_len, d_model=256, n_heads=4, num_encoder_blocks=4,
                 d_ff=512, dropout=0.2, emb_type='qid', lambda_bce=0.9):
        super().__init__()
        
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.emb_type = emb_type
        self.lambda_bce = lambda_bce
        self.lambda_mastery = 1.0 - lambda_bce  # Constraint: λ₁ + λ₂ = 1
        
        # Embeddings
        self.context_embedding = nn.Embedding(num_c * 2, d_model)  # q + num_c * r
        self.value_embedding = nn.Embedding(num_c * 2, d_model)
        self.skill_embedding = nn.Embedding(num_c, d_model)  # For prediction head
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
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
    
    def forward(self, q, r, qry=None):
        """
        Args:
            q: [B, L] - question IDs
            r: [B, L] - responses (0 or 1)
            qry: [B, L] - query question IDs (optional, defaults to q)
        
        Returns:
            dict with keys:
                - 'bce_predictions': [B, L] from Head 1
                - 'mastery_predictions': [B, L] from Head 2 (None if λ_mastery=0)
                - 'skill_vector': [B, L, num_c] - interpretable {KCi} (None if λ_mastery=0)
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
        
        return {
            'bce_predictions': bce_predictions,
            'mastery_predictions': mastery_predictions,
            'skill_vector': kc_vector_mono,  # Interpretable {KCi}
            'logits': logits,
            'mastery_logits': mastery_logits
        }
    
    def compute_loss(self, output, targets, lambda_bce=None):
        """
        Compute multi-task loss with constraint λ₁ + λ₂ = 1.
        
        Args:
            output: dict from forward()
            targets: [B, L] - ground truth (0 or 1)
            lambda_bce: BCE loss weight (overrides self.lambda_bce if provided)
                       lambda_mastery is automatically computed as 1.0 - lambda_bce
        
        Returns:
            dict with keys:
                - 'total_loss': weighted sum
                - 'bce_loss': L1
                - 'mastery_loss': L2
        """
        lambda_bce = lambda_bce if lambda_bce is not None else self.lambda_bce
        lambda_mastery = 1.0 - lambda_bce  # Enforce constraint: λ₁ + λ₂ = 1
        
        # L1: BCE Loss (Head 1)
        bce_loss = F.binary_cross_entropy_with_logits(
            output['logits'],
            targets.float(),
            reduction='mean'
        )
        
        # Multi-task loss: weighted combination
        if output['mastery_logits'] is not None:
            # L2: Mastery Loss (Head 2)
            mastery_loss = F.binary_cross_entropy_with_logits(
                output['mastery_logits'],
                targets.float(),
                reduction='mean'
            )
            # Total loss: L_total = λ₁ * L1 + λ₂ * L2
            total_loss = lambda_bce * bce_loss + lambda_mastery * mastery_loss
        else:
            # Pure BCE mode (λ_mastery=0, Head 2 skipped)
            mastery_loss = torch.tensor(0.0, device=bce_loss.device)
            total_loss = bce_loss
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'mastery_loss': mastery_loss
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
        - lambda_bce: BCE loss weight (default: 0.9)
    
    Note: lambda_mastery is automatically computed as 1.0 - lambda_bce
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
        lambda_bce=config.get('lambda_bce', 0.9)
    )
