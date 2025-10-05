"""
GainAKT2 Enhanced: Advanced Knowledge Tracing with Multi-Scale Attention
Target: AUC ~0.8+ through architectural innovations and training enhancements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiScaleAttention(nn.Module):
    """Multi-scale attention to capture both short-term and long-term dependencies"""
    
    def __init__(self, d_model, n_heads, scales=[1, 2, 4]):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.head_dim = d_model // n_heads
        
        # Use manual attention implementation instead of nn.MultiheadAttention to avoid mask issues
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Scale fusion network
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, d_model = query.size()
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale > 1:
                # For larger scales, use simple attention without masks
                q_scaled = F.avg_pool1d(
                    query.transpose(1, 2), kernel_size=scale, stride=scale, padding=0
                ).transpose(1, 2)
                k_scaled = F.avg_pool1d(
                    key.transpose(1, 2), kernel_size=scale, stride=scale, padding=0
                ).transpose(1, 2)
                v_scaled = F.avg_pool1d(
                    value.transpose(1, 2), kernel_size=scale, stride=scale, padding=0
                ).transpose(1, 2)
                
                # Manual attention computation
                q_proj = self.w_q(q_scaled).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
                k_proj = self.w_k(k_scaled).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
                v_proj = self.w_v(v_scaled).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
                
                scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_weights = F.softmax(scores, dim=-1)
                attn_out = torch.matmul(attn_weights, v_proj)
                
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
                attn_out = self.w_o(attn_out)
                
                # Upsample back to original resolution
                current_len = attn_out.size(1)
                if current_len != seq_len:
                    # Ensure correct dimensions for interpolation
                    attn_out_transposed = attn_out.transpose(1, 2)  # (batch, d_model, current_len)
                    attn_out = F.interpolate(
                        attn_out_transposed, size=seq_len, mode='linear', align_corners=False
                    ).transpose(1, 2)  # Back to (batch, seq_len, d_model)
                # If lengths match, no need to interpolate
            else:
                # For scale=1, use manual attention with mask
                q_proj = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k_proj = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v_proj = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                # Apply causal mask if provided
                if attn_mask is not None:
                    # Expand mask for all heads and batches
                    expanded_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                    scores = scores.masked_fill(expanded_mask, float('-inf'))
                
                attn_weights = F.softmax(scores, dim=-1)
                attn_out = torch.matmul(attn_weights, v_proj)
                
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
                attn_out = self.w_o(attn_out)
                
            scale_outputs.append(attn_out)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused)
        
        return output

class AdaptiveGatingMechanism(nn.Module):
    """Adaptive gating to balance context and value streams dynamically"""
    
    def __init__(self, d_model):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, context, value):
        # Compute adaptive gate based on both streams
        combined = torch.cat([context, value], dim=-1)
        gate = self.gate_network(combined)
        
        # Apply gating
        gated_context = context * gate
        gated_value = value * (1 - gate)
        
        return gated_context, gated_value

class AdvancedEncoderBlock(nn.Module):
    """Enhanced encoder with multi-scale attention and adaptive gating"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-scale attention for both streams
        self.context_attention = MultiScaleAttention(d_model, n_heads)
        self.value_attention = MultiScaleAttention(d_model, n_heads)
        
        # Adaptive gating mechanism
        self.adaptive_gate = AdaptiveGatingMechanism(d_model)
        
        # Cross-stream attention components (manual implementation)
        self.cross_w_q = nn.Linear(d_model, d_model)
        self.cross_w_k = nn.Linear(d_model, d_model)
        self.cross_w_v = nn.Linear(d_model, d_model)
        self.cross_w_o = nn.Linear(d_model, d_model)
        
        # Feed-forward networks
        self.ffn_context = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU instead of ReLU for better performance
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.ffn_value = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1_ctx = nn.LayerNorm(d_model)
        self.norm1_val = nn.LayerNorm(d_model)
        self.norm2_ctx = nn.LayerNorm(d_model)
        self.norm2_val = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _cross_attention(self, query, key, value, mask=None):
        """Manual cross-attention implementation"""
        batch_size, seq_len, d_model = query.size()
        
        # Project to Q, K, V
        q = self.cross_w_q(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.cross_w_k(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.cross_w_v(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(expanded_mask, float('-inf'))
        
        # Apply softmax and compute output
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_out = self.cross_w_o(attn_out)
        
        return attn_out
        
    def forward(self, context_sequence, value_sequence, mask=None):
        # Multi-scale self-attention
        ctx_attn = self.context_attention(context_sequence, context_sequence, context_sequence, mask)
        val_attn = self.value_attention(value_sequence, value_sequence, value_sequence, mask)
        
        # Residual connection and normalization
        context_sequence = self.norm1_ctx(context_sequence + self.dropout(ctx_attn))
        value_sequence = self.norm1_val(value_sequence + self.dropout(val_attn))
        
        # Cross-stream attention for information exchange
        ctx_cross = self._cross_attention(context_sequence, value_sequence, value_sequence, mask)
        val_cross = self._cross_attention(value_sequence, context_sequence, context_sequence, mask)
        
        context_sequence = self.norm_cross(context_sequence + self.dropout(ctx_cross))
        value_sequence = self.norm_cross(value_sequence + self.dropout(val_cross))
        
        # Adaptive gating
        context_sequence, value_sequence = self.adaptive_gate(context_sequence, value_sequence)
        
        # Feed-forward networks
        ctx_ffn = self.ffn_context(context_sequence)
        val_ffn = self.ffn_value(value_sequence)
        
        # Final residual connections
        new_context_sequence = self.norm2_ctx(context_sequence + self.dropout(ctx_ffn))
        new_value_sequence = self.norm2_val(value_sequence + self.dropout(val_ffn))
        
        return new_context_sequence, new_value_sequence

class KnowledgeStateTracker(nn.Module):
    """Explicit knowledge state tracking for interpretability and performance"""
    
    def __init__(self, num_concepts, d_model):
        super().__init__()
        self.num_concepts = num_concepts
        self.d_model = d_model
        
        # Knowledge state evolution network
        self.state_update = nn.GRU(d_model, d_model, batch_first=True)
        
        # Concept mastery prediction heads
        self.mastery_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_concepts),
            nn.Sigmoid()
        )
        
        # Learning gain prediction
        self.gain_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_concepts),
            nn.Tanh()  # Gains can be positive or negative
        )
        
    def forward(self, value_sequence, concepts):
        batch_size, seq_len, d_model = value_sequence.size()
        
        # Track knowledge state evolution
        knowledge_states, _ = self.state_update(value_sequence)
        
        # Predict concept mastery levels
        mastery_levels = self.mastery_head(knowledge_states)
        
        # Predict learning gains
        learning_gains = self.gain_head(knowledge_states)
        
        return knowledge_states, mastery_levels, learning_gains

class GainAKT2Enhanced(nn.Module):
    """Enhanced GainAKT2 with multi-scale attention and advanced features"""
    
    def __init__(self, num_c, seq_len=200, d_model=256, n_heads=8, num_encoder_blocks=4, 
                 d_ff=768, dropout=0.2, emb_type="qid", emb_path="", pretrain_dim=768,
                 use_knowledge_tracking=True, temperature=1.0):
        super().__init__()
        self.model_name = "gainakt2_enhanced"
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.dropout = dropout
        self.emb_type = emb_type
        self.use_knowledge_tracking = use_knowledge_tracking
        self.temperature = temperature  # For calibrated predictions
        
        # Enhanced embeddings with better initialization
        self.concept_embedding = nn.Embedding(num_c, d_model)
        self.context_embedding = nn.Embedding(num_c * 2, d_model)
        self.value_embedding = nn.Embedding(num_c * 2, d_model)
        
        # Positional encoding with learnable parameters
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        
        # Enhanced encoder stack
        self.encoder_blocks = nn.ModuleList([
            AdvancedEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_blocks)
        ])
        
        # Knowledge state tracker
        if use_knowledge_tracking:
            self.knowledge_tracker = KnowledgeStateTracker(num_c, d_model)
        
        # Enhanced prediction head with uncertainty estimation
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # 3x for context + value + concept
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Better weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, q, r, qry=None, qtest=False):
        batch_size, seq_len = q.size()
        
        # Create interaction tokens
        interaction_tokens = q + r * self.num_c
        
        # Enhanced embeddings
        context_seq = self.context_embedding(interaction_tokens)
        value_seq = self.value_embedding(interaction_tokens)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        context_seq = context_seq + pos_emb
        value_seq = value_seq + pos_emb
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process through enhanced encoder blocks
        for encoder_block in self.encoder_blocks:
            context_seq, value_seq = encoder_block(context_seq, value_seq, mask)
        
        # Knowledge state tracking
        knowledge_states = None
        mastery_levels = None
        learning_gains = None
        
        if self.use_knowledge_tracking and self.training:
            knowledge_states, mastery_levels, learning_gains = self.knowledge_tracker(value_seq, q)
        
        # Determine target concepts for prediction (same as original GainAKT2)
        if qry is None:
            target_concepts = q
        else:
            target_concepts = qry
        
        # Get target concept embeddings
        target_concept_emb = self.concept_embedding(target_concepts)
        
        # Concatenate features for prediction
        prediction_input = torch.cat([
            context_seq, 
            value_seq, 
            target_concept_emb
        ], dim=-1)
        
        # Make predictions with temperature scaling
        logits = self.prediction_head(prediction_input) / self.temperature
        predictions = torch.sigmoid(logits.squeeze(-1))
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(prediction_input)
        
        # Return in the same format as original GainAKT2
        output = {'predictions': predictions}
        if qtest:
            output['encoded_seq'] = context_seq
        if uncertainty is not None:
            output['uncertainty'] = uncertainty.squeeze(-1)
        if knowledge_states is not None:
            output['knowledge_states'] = knowledge_states
        if mastery_levels is not None:
            output['mastery_levels'] = mastery_levels
        if learning_gains is not None:
            output['learning_gains'] = learning_gains
            
        return output

# Enhanced loss function with multiple objectives
def enhanced_loss_function(outputs, targets, mask=None, alpha=1.0, beta=0.1, gamma=0.05):
    """
    Enhanced loss with uncertainty weighting and knowledge tracking objectives
    
    Args:
        outputs: Model outputs dict
        targets: Ground truth responses
        mask: Valid position mask
        alpha: Weight for main prediction loss
        beta: Weight for uncertainty regularization
        gamma: Weight for knowledge tracking losses
    """
    predictions = outputs['predictions']
    uncertainty = outputs['uncertainty'] if 'uncertainty' in outputs else None
    
    # Main prediction loss with uncertainty weighting
    if uncertainty is not None:
        # Heteroscedastic uncertainty weighting
        precision = 1.0 / (uncertainty + 1e-8)
        main_loss = precision * F.binary_cross_entropy(predictions, targets.float(), reduction='none')
        main_loss = main_loss + torch.log(uncertainty + 1e-8)  # Regularize uncertainty
    else:
        main_loss = F.binary_cross_entropy(predictions, targets.float(), reduction='none')
    
    if mask is not None:
        main_loss = main_loss * mask
        total_loss = alpha * main_loss.sum() / mask.sum()
    else:
        total_loss = alpha * main_loss.mean()
    
    # Uncertainty regularization
    if uncertainty is not None and beta > 0:
        uncertainty_reg = beta * uncertainty.mean()
        total_loss = total_loss + uncertainty_reg
    
    # Knowledge tracking losses
    if gamma > 0 and outputs.get('mastery_levels') is not None:
        # Encourage non-negative learning gains for correct responses
        learning_gains = outputs['learning_gains']
        if learning_gains is not None:
            gain_consistency = gamma * F.relu(-learning_gains * targets.unsqueeze(-1)).mean()
            total_loss = total_loss + gain_consistency
    
    return total_loss