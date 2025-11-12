"""
GainAKT2: Gain-based Attention Knowledge Tracing Model

This model uses a Transformer-based architecture with a two-stream attention mechanism.
It separates context information (for Q and K) from value information (for V) 
to better capture learning gains and knowledge state evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Efficient Multi-Head Attention with two input streams for context and values.

    This implementation computes all heads in parallel for efficiency. Q and K are
    derived from the `context_sequence`, while V is derived from the `value_sequence`.
    
    Supports two modes:
    - Legacy: Values are d_model dimensional latent representations
    - Intrinsic: Values are num_skills dimensional gains (h_t = Σ α g)

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        dropout (float): Dropout probability.
        intrinsic_gain_attention (bool): If True, values are skill-space gains.
        num_skills (int): Number of skills (required if intrinsic_gain_attention=True).
    """
    
    def __init__(self, n_heads, d_model, dropout=0.1, intrinsic_gain_attention=False, 
                 num_skills=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.intrinsic_gain_attention = intrinsic_gain_attention
        self.num_skills = num_skills
        
        if intrinsic_gain_attention:
            assert num_skills is not None, "num_skills required for intrinsic gain attention"
            self.d_k = d_model // n_heads
            # Pad num_skills to be divisible by n_heads for uniform head distribution
            self.num_skills_padded = ((num_skills + n_heads - 1) // n_heads) * n_heads
            self.d_g = self.num_skills_padded // n_heads  # Gain dimension per head
            # Create gain_to_context as a proper submodule (for DataParallel compatibility)
            # Use actual num_skills (not padded) for projection
            self.gain_to_context = nn.Linear(num_skills, d_model)
        else:
            self.d_k = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        
        if intrinsic_gain_attention:
            # Value projection maps num_skills to num_skills (identity or refinement)
            self.value_proj = nn.Linear(num_skills, num_skills)
        else:
            # Legacy: d_model to d_model
            self.value_proj = nn.Linear(d_model, d_model)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_sequence, value_sequence, mask=None):
        """
        Forward pass for the multi-head attention.

        Args:
            context_sequence (torch.Tensor): The sequence for queries and keys.
                                            Shape: [batch_size, seq_len, d_model]
            value_sequence (torch.Tensor): The sequence for values.
                                           Shape: [batch_size, seq_len, d_model] (legacy)
                                           or [batch_size, seq_len, num_skills] (intrinsic)
            mask (torch.Tensor, optional): A boolean mask to prevent attention to certain positions.
                                           Shape: [seq_len, seq_len]

        Returns:
            torch.Tensor: The output of the attention mechanism.
                          Shape: [batch_size, seq_len, d_model]
        """
        batch_size = context_sequence.size(0)

        # 1) Project and reshape Q, K for all heads in parallel
        Q = self.query_proj(context_sequence).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(context_sequence).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        if self.intrinsic_gain_attention:
            # Intrinsic mode: V has shape [B, L, num_skills]
            # Project values
            V_proj = self.value_proj(value_sequence)  # [B, L, num_skills]
            
            # Pad to num_skills_padded for uniform head distribution
            if self.num_skills_padded > self.num_skills:
                padding = torch.zeros(batch_size, value_sequence.size(1), 
                                    self.num_skills_padded - self.num_skills,
                                    device=value_sequence.device, dtype=V_proj.dtype)
                V_proj = torch.cat([V_proj, padding], dim=-1)
            
            # Reshape to [B, n_heads, L, d_g]
            V = V_proj.view(batch_size, -1, self.n_heads, self.d_g).transpose(1, 2)
        else:
            # Legacy mode: V has shape [B, L, d_model]
            V = self.value_proj(value_sequence).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2) Compute attention scores
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        # Compute attention scores; keep computation in float32 for stability under AMP
        # then cast back to original dtype (usually fp16) after masking.
        orig_dtype = Q.dtype
        scores = torch.matmul(Q.to(torch.float32), K.transpose(-2, -1).to(torch.float32)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # The mask shape [seq_len, seq_len] is broadcast to the scores tensor.
            # Use a dtype-dependent large negative value that will not overflow in fp16.
            if orig_dtype == torch.float16:
                neg_fill = -1e4  # within fp16 range (~-65504)
            elif orig_dtype == torch.bfloat16:
                neg_fill = -1e30  # bfloat16 min ~ -3.39e38, safe large negative
            else:
                neg_fill = -1e9
            scores = scores.masked_fill(mask, neg_fill)
        # Cast back to original dtype if needed
        scores = scores.to(orig_dtype)
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # 3) Apply attention to V
        if self.intrinsic_gain_attention:
            # Aggregate gains: head_gains shape [B, n_heads, L, d_g]
            head_gains = torch.matmul(attention_weights, V)
            # Concatenate heads to get padded skill space: [B, L, num_skills_padded]
            aggregated_gains_padded = head_gains.transpose(1, 2).contiguous().view(batch_size, -1, self.num_skills_padded)
            # Remove padding to get actual skill space: [B, L, num_skills]
            aggregated_gains = aggregated_gains_padded[..., :self.num_skills]
            # Store for monitoring pathway
            self.last_aggregated_gains = aggregated_gains
            # Project gains to d_model for compatibility
            output = self.gain_to_context(aggregated_gains)
        else:
            # Legacy: output shape [B, n_heads, L, d_k]
            output = torch.matmul(attention_weights, V)
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            self.last_aggregated_gains = None
        
        # 4) Final linear projection
        projected_output = self.output_proj(output)
        return self.dropout(projected_output)

class EncoderBlock(nn.Module):
    """
    A single Transformer Encoder block for the GainAKT2 model.

    This block consists of a two-stream Multi-Head Attention layer followed by a 
    Position-wise Feed-Forward Network. Layer normalization and residual connections
    are applied after each sub-layer. This version updates both context and value streams.

    Args:
        d_model (int): The dimensionality of the model.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the feed-forward network.
        dropout (float): The dropout rate.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, intrinsic_gain_attention=False,
                 num_skills=None):
        super().__init__()
        self.intrinsic_gain_attention = intrinsic_gain_attention
        self.attn = MultiHeadAttention(n_heads, d_model, dropout, intrinsic_gain_attention, 
                                       num_skills)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1_ctx = nn.LayerNorm(d_model)
        if not intrinsic_gain_attention:
            # Only need value normalization in legacy mode
            self.norm1_val = nn.LayerNorm(d_model)
        self.norm2_ctx = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_sequence, value_sequence, mask=None):
        """
        Forward pass for the encoder block.

        Args:
            context_sequence (torch.Tensor): The input sequence for context.
                                            Shape: [batch_size, seq_len, d_model]
            value_sequence (torch.Tensor): The input sequence for values.
                                           Shape: [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): The attention mask.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The output context sequence. Shape: [batch_size, seq_len, d_model]
                - The output value sequence. Shape: [batch_size, seq_len, d_model]
        """
        # Attention sub-layer
        attn_output = self.attn(context_sequence, value_sequence, mask)

        if self.intrinsic_gain_attention:
            # Intrinsic mode: value sequence stays as raw gains (no residual update)
            new_value_sequence = value_sequence
            # Only context gets updated with attention output
            x = self.norm1_ctx(context_sequence + attn_output)
        else:
            # Legacy mode: both streams get updated
            new_value_sequence = self.norm1_val(value_sequence + attn_output)
            x = self.norm1_ctx(context_sequence + attn_output)

        # Feed-forward sub-layer
        ffn_output = self.ffn(x)
        # Second residual connection for context
        new_context_sequence = self.norm2_ctx(x + self.dropout(ffn_output))

        return new_context_sequence, new_value_sequence

class GainAKT2(nn.Module):
    """
    GainAKT2 model for Knowledge Tracing using a gain-based attention mechanism.
    
    This model implements a Transformer-based architecture that separates context
    and value streams in the attention computation to better model learning gains.
    The architecture is composed of three main parts:
    1. Embedding Layer: Creates separate embeddings for context, value (gains), and concepts.
    2. Encoder Stack: A stack of EncoderBlocks that process the context and value streams.
    3. Prediction Head: A feed-forward network that predicts response correctness.
    """
    
    def __init__(self, num_c, seq_len=200, d_model=128, n_heads=8, num_encoder_blocks=2, 
                 d_ff=256, dropout=0.1, emb_type="qid", emb_path="", pretrain_dim=768,
                 use_mastery_head=False, use_gain_head=False, intrinsic_gain_attention=False,
                 non_negative_loss_weight=0.0, 
                 monotonicity_loss_weight=0.0, mastery_performance_loss_weight=0.0,
                 gain_performance_loss_weight=0.0, sparsity_loss_weight=0.0, consistency_loss_weight=0.0,
                 use_skill_difficulty=False, use_student_speed=False, num_students=None):
        super().__init__()
        self.model_name = "gainakt2"
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.dropout = dropout
        self.emb_type = emb_type
        self.intrinsic_gain_attention = intrinsic_gain_attention
        self.use_mastery_head = use_mastery_head and not intrinsic_gain_attention  # Disable in intrinsic mode
        self.use_gain_head = use_gain_head and not intrinsic_gain_attention  # Disable in intrinsic mode
        self.use_skill_difficulty = use_skill_difficulty
        self.use_student_speed = use_student_speed
        self.num_students = num_students
        self.non_negative_loss_weight = non_negative_loss_weight
        self.monotonicity_loss_weight = monotonicity_loss_weight
        self.mastery_performance_loss_weight = mastery_performance_loss_weight
        self.gain_performance_loss_weight = gain_performance_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        
        # The pykt framework uses emb_type to distinguish embedding types.
        # We focus on the "qid" type, which is standard for this project.
        if emb_type.startswith("qid"):
            # Interaction embeddings for context (Q,K) and value (V) streams
            # The size is num_c * 2 to account for both concepts and responses (correct/incorrect)
            self.context_embedding = nn.Embedding(num_c * 2, d_model)
            
            if intrinsic_gain_attention:
                # Intrinsic mode: Values represent per-skill gains (num_c dimensional)
                self.value_embedding = nn.Embedding(num_c * 2, num_c)
                self.gain_activation = nn.Softplus()  # Ensures g_i >= 0
            else:
                # Legacy mode: Opaque latent values
                self.value_embedding = nn.Embedding(num_c * 2, d_model)
            
            # A separate embedding for the target concept in the prediction head
            self.concept_embedding = nn.Embedding(num_c, d_model)
        
        # Positional encoding for the sequence
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        # Stack of encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout, intrinsic_gain_attention,
                        num_c if intrinsic_gain_attention else None)
            for _ in range(num_encoder_blocks)
        ])
        
        # Final prediction head
        if intrinsic_gain_attention:
            # Intrinsic mode: [h_t, concept_embedding] where h_t already from Σ α g
            self.prediction_head = nn.Sequential(
                nn.Linear(d_model * 2, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, 1)
            )
        else:
            # Legacy mode: [context_seq, value_seq, concept_embedding]
            self.prediction_head = nn.Sequential(
                nn.Linear(d_model * 3, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, 1)
            )

        # Optional projection heads for interpretability
        if self.use_mastery_head:
            self.mastery_head = nn.Linear(self.d_model, self.num_c)
        if self.use_gain_head:
            self.gain_head = nn.Linear(self.d_model, self.num_c)
        
        # Skill difficulty parameters (Phase 1: Architectural Improvements - DEPRECATED)
        if self.use_skill_difficulty:
            # Learnable per-skill difficulty scale: modulates embedding magnitude
            # Initialized to 1.0 (neutral), constrained to [0.5, 2.0] range
            # Scale > 1.0 = harder skills (larger embeddings → more attention)
            # Scale < 1.0 = easier skills (smaller embeddings → less attention)
            self.skill_difficulty_scale = nn.Parameter(torch.ones(num_c))
            print(f"[GainAKT2] Skill difficulty (embedding modulation) enabled: +{num_c} parameters")
        
        # Student learning speed parameters (Phase 2: Architectural Improvements)
        if self.use_student_speed:
            assert num_students is not None, "num_students required for student_speed feature"
            # Per-student learning speed embedding (16-dim rich representation)
            # Applied across all 200 sequence positions → stronger gradient signal
            # Captures individual differences in learning rate/aptitude
            self.student_speed_embedding = nn.Embedding(num_students, 16)
            # Xavier init for student embeddings
            nn.init.xavier_uniform_(self.student_speed_embedding.weight)
            print(f"[GainAKT2] Student learning speed enabled: +{num_students * 16} parameters ({num_students} students)")
            
            # Update prediction head input dimension
            if intrinsic_gain_attention:
                # Intrinsic mode: [h_t, concept_embedding, student_speed] = d_model*2 + 16
                self.prediction_head = nn.Sequential(
                    nn.Linear(d_model * 2 + 16, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, 1)
                )
            else:
                # Legacy mode: [context_seq, value_seq, concept_embedding, student_speed] = d_model*3 + 16
                self.prediction_head = nn.Sequential(
                    nn.Linear(d_model * 3 + 16, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, 1)
                )

    def forward(self, q, r, qry=None, qtest=False, student_ids=None):
        """
        Forward pass for the GainAKT2 model, following PyKT conventions.
        
        Args:
            q (torch.Tensor): A tensor of question/concept IDs.
                              Shape: [batch_size, seq_len]
            r (torch.Tensor): A tensor of student responses (0 for incorrect, 1 for correct).
                              Shape: [batch_size, seq_len]
            qry (torch.Tensor, optional): A tensor of query questions for prediction.
                                          If None, `q` is used as the target.
                                          Shape: [batch_size, seq_len]
            qtest (bool): If True, the output dictionary will include the final context sequence.
            student_ids (torch.Tensor, optional): Student IDs for each sequence in batch.
                                                  Shape: [batch_size] (scalar per sequence)
                                                  Required if use_student_speed=True.
        
        Returns:
            dict: A dictionary containing the model's outputs, including:
                - 'predictions': Response probabilities. Shape: [batch_size, seq_len]
                - 'encoded_seq': Final context sequence (if qtest=True).
                - 'projected_mastery': Projected mastery vectors (if use_mastery_head=True).
                - 'projected_gains': Projected gain vectors (if use_gain_head=True).
        """
        batch_size, seq_len = q.size()
        
        # Create interaction tokens by combining question and response IDs.
        interaction_tokens = q + self.num_c * r
        
        # Determine the target concepts for prediction.
        if qry is None:
            target_concepts = q
        else:
            target_concepts = qry
        
        # Create a causal attention mask to prevent attending to future positions.
        mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device), diagonal=1).bool()

        # 1. Get embeddings for the two streams (context and value)
        context_seq = self.context_embedding(interaction_tokens)
        value_seq = self.value_embedding(interaction_tokens)
        
        if self.intrinsic_gain_attention:
            # Apply non-negativity activation to gains
            value_seq = self.gain_activation(value_seq)

        # 2. Add positional encodings to both streams
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        context_seq += pos_emb
        
        if not self.intrinsic_gain_attention:
            # Legacy mode: add positional encoding to value stream
            value_seq += pos_emb
        
        # 3. Pass sequences through the stack of encoder blocks
        for block in self.encoder_blocks:
            context_seq, value_seq = block(context_seq, value_seq, mask)
        
        # 4. Prepare inputs for the prediction head
        target_concept_emb = self.concept_embedding(target_concepts)
        
        # Apply skill difficulty modulation (if enabled)
        if self.use_skill_difficulty:
            # Get difficulty scale for each target concept [batch_size, seq_len]
            difficulty_scale = torch.gather(
                self.skill_difficulty_scale.unsqueeze(0).expand(batch_size, -1),
                1, target_concepts
            )  # [batch_size, seq_len]
            # Expand to match embedding dimension and apply constrained scaling
            # Scale > 1.0 = harder skills (amplified embeddings)
            # Scale < 1.0 = easier skills (dampened embeddings)
            difficulty_scale = torch.clamp(difficulty_scale, 0.5, 2.0).unsqueeze(-1)  # [batch_size, seq_len, 1]
            target_concept_emb = target_concept_emb * difficulty_scale  # [batch_size, seq_len, d_model]
        
        # Add student learning speed embedding (if enabled)
        if self.use_student_speed:
            assert student_ids is not None, "student_ids required when use_student_speed=True"
            # Get student embedding [batch_size, 16]
            student_emb = self.student_speed_embedding(student_ids)
            # Expand to match sequence length [batch_size, seq_len, 16]
            student_emb = student_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        if self.intrinsic_gain_attention:
            # Intrinsic mode: [h_t, concept_embedding] or [h_t, concept_embedding, student_speed]
            # context_seq already represents h_t from Σ α g
            if self.use_student_speed:
                concatenated = torch.cat([context_seq, target_concept_emb, student_emb], dim=-1)
            else:
                concatenated = torch.cat([context_seq, target_concept_emb], dim=-1)
        else:
            # Legacy mode: [context_seq, value_seq, concept_embedding] or [..., student_speed]
            if self.use_student_speed:
                concatenated = torch.cat([context_seq, value_seq, target_concept_emb, student_emb], dim=-1)
            else:
                concatenated = torch.cat([context_seq, value_seq, target_concept_emb], dim=-1)
        
        # 5. Generate predictions
        logits = self.prediction_head(concatenated).squeeze(-1)
        predictions = torch.sigmoid(logits)
        
        # 6. Prepare output dictionary
        output = {'predictions': predictions}
        if qtest:
            output['encoded_seq'] = context_seq

        # 7. Compute projected mastery and gains if heads are enabled
        if self.use_mastery_head:
            projected_mastery = self.mastery_head(context_seq)
            output['projected_mastery'] = projected_mastery
        
        if self.use_gain_head:
            projected_gains = self.gain_head(value_seq)
            output['projected_gains'] = projected_gains
            
        return output    
    def get_aggregated_gains(self):
        """
        Retrieve aggregated gains from the last encoder block's attention mechanism.
        Only applicable in intrinsic_gain_attention mode.
        
        Returns:
            torch.Tensor: Aggregated gains [B, L, num_skills] or None if not in intrinsic mode
        """
        if not self.intrinsic_gain_attention:
            return None
        
        # Get the last encoder block's attention layer
        last_block = self.encoder_blocks[-1]
        if hasattr(last_block.attn, 'last_aggregated_gains'):
            return last_block.attn.last_aggregated_gains
        return None
