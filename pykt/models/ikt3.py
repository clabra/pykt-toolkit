"""
iKT3: Interpretable Knowledge Tracing with External Reference Model Alignment

Architecture:
- Single Encoder: Questions + Responses (binary)
  → Head 1 (Performance): BCE Loss (L_BCE)
  → Head 2 (IRT Mastery): Ability encoder + IRT formula → M_IRT = σ(θ - β)

Single-Phase Training with Warm-up:
    L_total = (1 - λ(t)) × L_BCE + c × L_22 + λ(t) × (L_21 + L_23)
    
    where λ(t) = λ_target × min(1, epoch / warmup_epochs)

Key Innovation (vs iKT2):
- External validity: Aligns Head 2 predictions with reference theoretical model
- Pluggable reference models: IRT (current), BKT (future), DINA, PFA, etc.
- Three alignment losses (IRT example):
  - L_21: BCE(M_IRT, M_ref) - performance alignment
  - L_22: MSE(β_learned, β_IRT) - difficulty regularization (always-on)
  - L_23: MSE(θ_learned, θ_IRT) - ability alignment
  
Advantages over iKT2:
- External calibration (not just internal head consistency)
- Single-phase training (simpler than 2-phase)
- Extensible to multiple reference models via pluggable architecture
- Better interpretability through validated theoretical alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from pykt.reference_models import create_reference_model, ReferenceModel


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
        self.norm2_val = nn.LayerNorm(d_model)
        
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
        
        # Separate residual for each stream
        context_seq = self.norm1_ctx(context_seq + attn_output)
        value_seq = self.norm1_val(value_seq + attn_output)
        
        # Feed-forward on BOTH streams (symmetric processing)
        ffn_output_ctx = self.ffn_ctx(context_seq)
        context_seq = self.norm2_ctx(context_seq + self.dropout(ffn_output_ctx))
        
        ffn_output_val = self.ffn_val(value_seq)
        value_seq = self.norm2_val(value_seq + self.dropout(ffn_output_val))
        
        return context_seq, value_seq


class iKT3(nn.Module):
    """
    iKT3: Interpretable Knowledge Tracing with External Reference Model Alignment
    
    Architecture:
        Questions + Responses → Encoder → h ──┬→ Head 1 (Performance) → p_correct → L_BCE
                                              └→ Head 2 (Reference) → factors → L_align
    
    Single-phase training with warm-up:
        L_total = (1 - λ(t)) × L_BCE + c × L_stability + λ(t) × L_align
        where λ(t) = λ_target × min(1, epoch / warmup_epochs)
    
    Key Innovation:
        - Pluggable reference models (IRT, BKT, etc.) via dependency injection
        - External validity: Head 2 aligns with validated theoretical model
        - Dynamic loss computation: Reference model defines its own alignment losses
    """
    
    def __init__(self, num_c, seq_len, d_model, n_heads, num_encoder_blocks,
                 d_ff, dropout, emb_type, reference_model_type, lambda_target,
                 warmup_epochs, c_stability_reg):
        """
        Initialize iKT3 model.
        
        All parameters REQUIRED (no defaults) per reproducibility guidelines.
        Defaults must come from configs/parameter_default.json.
        
        Args:
            num_c: Number of skills/concepts
            seq_len: Maximum sequence length
            d_model: Model dimension
            n_heads: Number of attention heads
            num_encoder_blocks: Number of encoder blocks
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            emb_type: Embedding type ('qid')
            reference_model_type: Type of reference model ('irt', 'bkt', etc.)
            lambda_target: Target weight for alignment loss after warm-up
            warmup_epochs: Number of epochs for λ warm-up
            c_stability_reg: Always-on stability regularization weight
        """
        super().__init__()
        
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.emb_type = emb_type
        self.reference_model_type = reference_model_type
        
        # Hyperparameters (explicit, no defaults)
        self.lambda_target = lambda_target
        self.warmup_epochs = warmup_epochs
        self.c_stability_reg = c_stability_reg
        
        # Validate positive weights
        assert lambda_target > 0, f"lambda_target must be positive, got {lambda_target}"
        assert warmup_epochs > 0, f"warmup_epochs must be positive, got {warmup_epochs}"
        assert c_stability_reg >= 0, f"c_stability_reg must be non-negative, got {c_stability_reg}"
        
        # Reference model (pluggable architecture)
        self.reference_model = create_reference_model(reference_model_type, num_c)
        print(f"✓ Created reference model: {reference_model_type}")
        
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
        
        # === HEAD 1: Performance Prediction ===
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),  # [h, v, skill_emb]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # === HEAD 2: Reference Model-Specific Heads ===
        # Initialize reference model-specific components
        if reference_model_type == 'irt':
            self._init_irt_heads(d_ff, dropout)
        elif reference_model_type == 'bkt':
            self._init_bkt_heads(d_ff, dropout)
        else:
            raise ValueError(f"Unsupported reference model type: {reference_model_type}")
    
    def _init_irt_heads(self, d_ff, dropout):
        """Initialize IRT-specific prediction heads."""
        # Ability encoder: extracts scalar student ability θ_i(t) from knowledge state h
        self.ability_encoder = nn.Sequential(
            nn.Linear(self.d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)  # Output: scalar ability θ_i(t)
        )
        
        # Skill difficulty embeddings (learnable parameters)
        self.skill_difficulty_emb = nn.Embedding(self.num_c, 1)
        # Initialize to neutral value (0.0 difficulty)
        nn.init.constant_(self.skill_difficulty_emb.weight, 0.0)
        
        print("  ✓ Initialized IRT heads (ability encoder + difficulty embeddings)")
    
    def _init_bkt_heads(self, d_ff, dropout):
        """Initialize BKT-specific prediction heads (future implementation)."""
        # BKT requires:
        # - Knowledge state encoder: h → P(L_t) (mastery probability)
        # - Skill-specific parameters: {prior, learns, slips, guesses} per skill
        raise NotImplementedError("BKT reference model not yet implemented")
    
    def load_reference_targets(self, targets_path):
        """
        Load reference model targets from file.
        
        Args:
            targets_path: Path to reference targets file (.pkl)
        
        Returns:
            dict: Reference targets (structure depends on reference model type)
        """
        return self.reference_model.load_targets(targets_path)
    
    def initialize_from_reference(self, ref_targets):
        """
        Initialize model parameters from reference targets.
        
        For IRT: Initialize skill difficulty embeddings from β_IRT
        For BKT: Initialize skill parameters from {prior, learns, slips, guesses}
        
        Args:
            ref_targets: dict from load_reference_targets()
        """
        if self.reference_model_type == 'irt':
            if 'beta_irt' in ref_targets or 'skill_difficulties' in ref_targets:
                beta_key = 'beta_irt' if 'beta_irt' in ref_targets else 'skill_difficulties'
                beta_irt = ref_targets[beta_key]
                
                with torch.no_grad():
                    # Convert dict to tensor if needed
                    if isinstance(beta_irt, dict):
                        beta_tensor = torch.tensor([beta_irt.get(i, 0.0) for i in range(self.num_c)])
                    else:
                        beta_tensor = beta_irt
                    
                    self.skill_difficulty_emb.weight.copy_(beta_tensor.view(-1, 1))
                
                print(f"✓ Initialized skill difficulties from IRT calibration")
                print(f"  β_init: mean={beta_tensor.mean().item():.4f}, std={beta_tensor.std().item():.4f}")
            else:
                print("⚠ Warning: No beta_irt or skill_difficulties in ref_targets, using zero initialization")
        
        elif self.reference_model_type == 'bkt':
            # Future: Initialize BKT parameters
            raise NotImplementedError("BKT parameter initialization not yet implemented")
    
    def forward(self, q, r, qry=None):
        """
        Forward pass.
        
        Args:
            q: [B, L] - question IDs
            r: [B, L] - responses (0 or 1)
            qry: [B, L] - query question IDs (optional, defaults to q)
        
        Returns:
            dict: Model outputs (structure depends on reference model type)
                Common keys:
                    - 'bce_predictions': [B, L] - performance predictions from Head 1
                    - 'logits': [B, L] - BCE logits for loss computation
                IRT-specific keys:
                    - 'theta_t_learned': [B, L] - learned student ability at each timestep t
                    - 'beta_learned': [B, L] - learned skill difficulty (static)
                    - 'mastery_irt': [B, L] - IRT-based mastery M = σ(θ_t - β)
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
        
        # === HEAD 2: Reference Model-Specific Forward ===
        if self.reference_model_type == 'irt':
            ref_outputs = self._forward_irt(h, qry)
        elif self.reference_model_type == 'bkt':
            ref_outputs = self._forward_bkt(h, qry)
        else:
            raise ValueError(f"Unsupported reference model type: {self.reference_model_type}")
        
        # Combine outputs
        return {
            'bce_predictions': bce_predictions,
            'logits': logits,
            **ref_outputs
        }
    
    def _forward_irt(self, h, qry):
        """
        IRT-specific forward pass for Head 2.
        
        Args:
            h: [B, L, d_model] - knowledge state from encoder
            qry: [B, L] - query question IDs
        
        Returns:
            dict with keys:
                - 'theta_t_learned': [B, L] - learned student ability at each timestep t
                - 'beta_learned': [B, L] - learned skill difficulty (static)
                - 'mastery_irt': [B, L] - IRT mastery M = σ(θ_t - β)
        """
        # Infer student ability from knowledge state (time-varying)
        theta_t_learned = self.ability_encoder(h).squeeze(-1)  # [B, L]
        
        # Extract skill difficulties for questions being answered (static per skill)
        beta_learned = self.skill_difficulty_emb(qry).squeeze(-1)  # [B, L]
        
        # Compute IRT mastery probability using Rasch formula
        mastery_irt = torch.sigmoid(theta_t_learned - beta_learned)  # [B, L]
        
        return {
            'theta_t_learned': theta_t_learned,
            'beta_learned': beta_learned,
            'mastery_irt': mastery_irt,
            'questions': qry  # Include for alignment loss computation
        }
    
    def _forward_bkt(self, h, qry):
        """BKT-specific forward pass (future implementation)."""
        raise NotImplementedError("BKT forward pass not yet implemented")
    
    def compute_loss(self, output, targets, ref_targets, lambda_interp):
        """
        Compute loss with reference model alignment.
        
        Loss formula:
            L_total = (1 - λ) × L_BCE + c × L_stability + λ × L_align
        
        where:
            - L_BCE: Binary cross-entropy for performance prediction
            - L_stability: Always-on regularization (e.g., difficulty stability)
            - L_align: Alignment with reference model (warm-up scheduled)
            - λ = lambda_interp (from warm-up schedule)
            - c = c_stability_reg (constant weight)
        
        Args:
            output: dict from forward()
            targets: [B, L] - ground truth responses (0 or 1)
            ref_targets: dict from load_reference_targets()
            lambda_interp: Current λ value from warm-up schedule [0, lambda_target]
        
        Returns:
            dict with keys:
                - 'total_loss': weighted total loss
                - 'l_bce': binary cross-entropy loss
                - 'l_stability': always-on regularization loss
                - 'l_align_total': combined alignment loss
                + additional losses from reference model (e.g., l_21, l_22, l_23 for IRT)
        """
        device = output['logits'].device
        
        # L_BCE: Binary Cross-Entropy Loss (Head 1 - Performance Prediction)
        l_bce = F.binary_cross_entropy_with_logits(
            output['logits'],
            targets.float(),
            reduction='mean'
        )
        
        # Compute reference model-specific alignment losses
        # This returns a dict with all alignment losses (structure depends on reference model)
        alignment_losses = self.reference_model.compute_alignment_losses(
            model_outputs=output,
            targets=ref_targets,
            lambda_weights={'lambda_interp': lambda_interp}
        )
        
        # Extract stability loss (always-on) and total alignment loss (warm-up scheduled)
        # For IRT: l_stability = l_22 (difficulty regularization)
        #          l_align_total = l_21 + l_23 (performance + ability alignment)
        if self.reference_model_type == 'irt':
            l_stability = alignment_losses.get('l_22_difficulty', torch.tensor(0.0, device=device))
            l_align_total = alignment_losses.get('l_align_total', torch.tensor(0.0, device=device))
        else:
            # Generic fallback (should not happen if reference model is properly implemented)
            l_stability = torch.tensor(0.0, device=device)
            l_align_total = sum(alignment_losses.values())
        
        # Total loss with warm-up schedule
        total_loss = (
            (1.0 - lambda_interp) * l_bce +
            self.c_stability_reg * l_stability +
            lambda_interp * l_align_total
        )
        
        # Return comprehensive loss dict
        return {
            'total_loss': total_loss,
            'l_bce': l_bce,
            'l_stability': l_stability,
            'l_align_total': l_align_total,
            **alignment_losses  # Include all reference model-specific losses
        }


def create_model(config):
    """
    Factory function to create iKT3 model.
    
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
        - reference_model: reference model type ('irt', 'bkt', etc.)
        - lambda_target: target weight for alignment after warm-up (e.g., 0.5)
        - warmup_epochs: number of epochs for λ warm-up (e.g., 50)
        - c_stability_reg: always-on stability regularization weight (e.g., 0.01)
    """
    # Fail-fast: require all parameters (no .get() with defaults)
    required_keys = [
        'num_c', 'seq_len', 'd_model', 'n_heads', 'num_encoder_blocks',
        'd_ff', 'dropout', 'emb_type', 'reference_model', 'lambda_target',
        'warmup_epochs', 'c_stability_reg'
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(
                f"Required config parameter '{key}' not found. "
                f"All defaults must be specified in parameter_default.json"
            )
    
    return iKT3(
        num_c=config['num_c'],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_blocks=config['num_encoder_blocks'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        emb_type=config['emb_type'],
        reference_model_type=config['reference_model'],
        lambda_target=config['lambda_target'],
        warmup_epochs=config['warmup_epochs'],
        c_stability_reg=config['c_stability_reg']
    )
