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
                 d_ff=512, dropout=0.2, emb_type='qid', lambda_bce=0.9,
                 lambda_temporal_contrast=0.0, temporal_contrast_temperature=0.07,
                 lambda_smoothness=0.0, lambda_skill_contrast=0.0, 
                 skill_contrast_margin=0.1, use_log_increment=True, 
                 increment_scale_init=-2.0, kc_emb_init_mean=0.1):
        super().__init__()
        
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.emb_type = emb_type
        self.lambda_bce = lambda_bce
        self.lambda_mastery = 1.0 - lambda_bce  # Constraint: λ₁ + λ₂ = 1
        
        # Phase 2: Semantic constraint parameters
        self.lambda_temporal_contrast = lambda_temporal_contrast
        self.temporal_contrast_temperature = temporal_contrast_temperature
        self.lambda_smoothness = lambda_smoothness
        self.lambda_skill_contrast = lambda_skill_contrast
        self.skill_contrast_margin = skill_contrast_margin
        self.use_log_increment = use_log_increment
        self.kc_emb_init_mean = kc_emb_init_mean
        
        # Embeddings
        self.context_embedding = nn.Embedding(num_c * 2, d_model)  # q + num_c * r
        self.value_embedding = nn.Embedding(num_c * 2, d_model)
        self.skill_embedding = nn.Embedding(num_c, d_model)  # For prediction head
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Initialize KC embeddings with random values around kc_emb_init_mean
        with torch.no_grad():
            nn.init.normal_(self.skill_embedding.weight, mean=self.kc_emb_init_mean, std=0.02)
        
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
        
        # Head 2: Mastery Estimation (Phase 1: A4 - Log-Increment Architecture)
        # MLP1: h1 → log-increments (architectural guarantee of growth)
        self.mlp1_log_increment = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_c)
            # No final activation - predict log-increments in (-∞, +∞)
        )
        
        # Learnable scale parameter for increment magnitude
        # Initialized to increment_scale_init → exp(increment_scale_init) typical increment
        self.increment_scale = nn.Parameter(torch.tensor(increment_scale_init))
        
        # MLP2: [h1, v1, skill_specific_KC] → mastery prediction
        # Similar to Head 1 but includes skill-specific KC value for interpretability
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_ff),  # [h1, v1, KC_skill] concatenated
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
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
        
        # === HEAD 2: Mastery Estimation (Phase 1: A4 - Log-Increment Architecture) ===
        # Conditional computation: skip if λ_mastery = 0 (no gradient flow)
        if self.lambda_mastery > 0:
            # Step 1: Predict log-increments (guaranteed positive growth via exp)
            log_increments = self.mlp1_log_increment(h1)  # [B, L, num_c], unbounded
            
            # Step 2: Convert to positive increments via exp
            # exp(log_increments + scale) guarantees increments > 0
            increments = torch.exp(log_increments + self.increment_scale)  # [B, L, num_c], always > 0
            # With scale=-2.0, typical increments ≈ 0.01-0.2 range
            
            # Step 3: Cumulative sum for monotonic growth (guaranteed by cumsum)
            # Initial state: zeros (mastery starts at ≈0)
            initial = torch.zeros(batch_size, 1, self.num_c, device=device)
            kc_vector = torch.cat([initial, increments], dim=1)  # [B, L+1, num_c]
            kc_vector = torch.cumsum(kc_vector, dim=1)[:, 1:, :]  # [B, L, num_c], monotonic by construction
            
            # Step 4: Clamp to [0, 1] (semantic boundedness constraint)
            kc_vector = torch.clamp(kc_vector, 0.0, 1.0)  # [B, L, num_c]
            
            # Step 5: Extract skill-specific KC values and combine with context
            # For each position, use the KC value of the queried skill
            # kc_vector: [B, L, num_c], qry: [B, L] → extract kc_vector[b, t, qry[b, t]]
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)  # [B, L]
            time_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, L]
            skill_specific_kc = kc_vector[batch_indices, time_indices, qry]  # [B, L]
            
            # MLP2: [h1, v1, skill_specific_KC] → mastery prediction
            # Concatenate context (h1, v1) with skill-specific KC value
            mastery_input = torch.cat([h1, v1, skill_specific_kc.unsqueeze(-1)], dim=-1)  # [B, L, 2*d_model + 1]
            mastery_logits = self.mlp2(mastery_input).squeeze(-1)  # [B, L]
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
            'skill_vector': kc_vector,  # Interpretable {KCi} (monotonic via cumsum)
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
            
            # Phase 2: Semantic constraint losses
            temporal_contrast_loss = torch.tensor(0.0, device=bce_loss.device)
            smoothness_loss = torch.tensor(0.0, device=bce_loss.device)
            skill_contrast_loss = torch.tensor(0.0, device=bce_loss.device)
            
            if self.lambda_temporal_contrast > 0 and output['skill_vector'] is not None:
                temporal_contrast_loss = self.compute_temporal_contrastive_loss(
                    output['skill_vector']
                )
            
            if self.lambda_smoothness > 0 and output['skill_vector'] is not None:
                smoothness_loss = self.compute_smoothness_loss(
                    output['skill_vector']
                )
            
            if self.lambda_skill_contrast > 0 and output['skill_vector'] is not None:
                skill_contrast_loss = self.compute_skill_contrastive_loss(
                    output['skill_vector'],
                    targets
                )
            
            # Total loss: L_total = λ_bce * L1 + λ_mastery * L2 + semantic constraints
            total_loss = (lambda_bce * bce_loss + 
                         lambda_mastery * mastery_loss +
                         self.lambda_temporal_contrast * temporal_contrast_loss +
                         self.lambda_smoothness * smoothness_loss +
                         self.lambda_skill_contrast * skill_contrast_loss)
        else:
            # Pure BCE mode (λ_mastery=0, Head 2 skipped)
            mastery_loss = torch.tensor(0.0, device=bce_loss.device)
            temporal_contrast_loss = torch.tensor(0.0, device=bce_loss.device)
            smoothness_loss = torch.tensor(0.0, device=bce_loss.device)
            skill_contrast_loss = torch.tensor(0.0, device=bce_loss.device)
            total_loss = bce_loss
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'mastery_loss': mastery_loss,
            'temporal_contrast_loss': temporal_contrast_loss,
            'smoothness_loss': smoothness_loss,
            'skill_contrast_loss': skill_contrast_loss
        }
    
    def compute_temporal_contrastive_loss(self, kc_vector):
        """
        Phase 2-B1: Temporal Contrastive Loss
        
        Enforce temporal progression: states closer in time should be more similar
        than states far apart. This is a pure semantic constraint that encourages
        gradual learning progression.
        
        Args:
            kc_vector: [B, L, num_c] - skill mastery vectors
        
        Returns:
            loss: scalar tensor
        """
        B, L, C = kc_vector.shape
        
        if L <= 1:
            return torch.tensor(0.0, device=kc_vector.device)
        
        # Normalize skill vectors for cosine similarity
        kc_norm = F.normalize(kc_vector, p=2, dim=-1)  # [B, L, C]
        
        loss = 0.0
        count = 0
        
        for b in range(B):
            # Similarity matrix for student b: [L, L]
            sim_matrix = torch.matmul(kc_norm[b], kc_norm[b].t()) / self.temporal_contrast_temperature
            
            # For each timestep t, positive = t+1, negatives = all others
            for t in range(L - 1):
                # Positive: next timestep (should be similar)
                positive = sim_matrix[t, t+1]
                
                # Negatives: all timesteps except t and t+1
                negatives_mask = torch.ones(L, dtype=torch.bool, device=kc_vector.device)
                negatives_mask[t] = False
                negatives_mask[t+1] = False
                negatives = sim_matrix[t, negatives_mask]
                
                if negatives.numel() > 0:
                    # InfoNCE loss: maximize similarity to positive, minimize to negatives
                    numerator = torch.exp(positive)
                    denominator = numerator + torch.sum(torch.exp(negatives))
                    loss += -torch.log(numerator / (denominator + 1e-8))
                    count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, device=kc_vector.device)
    
    def compute_smoothness_loss(self, kc_vector):
        """
        Phase 2-A3: Smoothness Loss
        
        Penalize abrupt changes in mastery levels. Encourages gradual,
        pedagogically realistic growth.
        
        Args:
            kc_vector: [B, L, num_c] - skill mastery vectors
        
        Returns:
            loss: scalar tensor
        """
        if kc_vector.size(1) <= 2:
            return torch.tensor(0.0, device=kc_vector.device)
        
        # Second-order differences: discourage sharp turns
        # d²KC/dt² ≈ KC[t+2] - 2*KC[t+1] + KC[t]
        second_diff = kc_vector[:, 2:] - 2*kc_vector[:, 1:-1] + kc_vector[:, :-2]
        smoothness_loss = second_diff.pow(2).mean()
        
        return smoothness_loss
    
    def compute_skill_contrastive_loss(self, kc_vector, responses):
        """
        Phase 2-B2: Skill-Specific Contrastive Loss
        
        Encourage separation: mastery after correct responses should be
        higher than mastery after incorrect responses. This is a semantic
        constraint that correct responses lead to higher mastery.
        
        Args:
            kc_vector: [B, L, num_c] - skill mastery vectors
            responses: [B, L] - ground truth responses (0 or 1)
        
        Returns:
            loss: scalar tensor
        """
        B, L, C = kc_vector.shape
        loss = 0.0
        valid_skills = 0
        
        for skill_idx in range(C):
            # Extract mastery trajectory for this skill: [B, L]
            skill_mastery = kc_vector[:, :, skill_idx].reshape(-1)  # [B*L]
            responses_flat = responses.reshape(-1)  # [B*L]
            
            # Separate indices: correct vs incorrect
            correct_idx = (responses_flat == 1).nonzero(as_tuple=True)[0]
            incorrect_idx = (responses_flat == 0).nonzero(as_tuple=True)[0]
            
            if len(correct_idx) > 0 and len(incorrect_idx) > 0:
                mean_correct = skill_mastery[correct_idx].mean()
                mean_incorrect = skill_mastery[incorrect_idx].mean()
                
                # Margin loss: correct should be at least margin higher
                separation = F.relu(self.skill_contrast_margin - (mean_correct - mean_incorrect))
                loss += separation
                valid_skills += 1
        
        return loss / valid_skills if valid_skills > 0 else torch.tensor(0.0, device=kc_vector.device)


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
        lambda_bce=config.get('lambda_bce', 0.9),
        lambda_temporal_contrast=config.get('lambda_temporal_contrast', 0.0),
        temporal_contrast_temperature=config.get('temporal_contrast_temperature', 0.07),
        lambda_smoothness=config.get('lambda_smoothness', 0.0),
        lambda_skill_contrast=config.get('lambda_skill_contrast', 0.0),
        skill_contrast_margin=config.get('skill_contrast_margin', 0.1),
        use_log_increment=config.get('use_log_increment', True),
        increment_scale_init=config.get('increment_scale_init', -2.0)
    )
