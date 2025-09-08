import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .utils import pos_encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearningGainsLayer(nn.Module):
    """
    Core innovation: Computes learning gains from skill-response interactions.
    
    This layer takes interaction embeddings and decomposes them into skill and response
    components, then computes the learning gains induced by each interaction.
    
    Args:
        emb_size: Embedding dimension
        n_skills: Number of unique skills/concepts
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, emb_size: int, n_skills: int, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.n_skills = n_skills
        
        # Skill and response extractors from interaction embeddings
        self.skill_extractor = nn.Linear(emb_size, emb_size)
        self.response_extractor = nn.Linear(emb_size, emb_size)
        
        # Learning gains computation layers
        self.skill_response_interaction = nn.Linear(emb_size * 2, emb_size)
        self.gains_transform = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
            nn.Sigmoid()  # Ensure gains are in [0, 1] range
        )
        
        # Difficulty awareness (educational parameter)
        self.difficulty_layer = nn.Linear(emb_size, emb_size)
        
    def forward(self, interaction_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute learning gains from interaction embeddings.
        
        Args:
            interaction_emb: Interaction embeddings [batch_size, seq_len, emb_size]
            
        Returns:
            gains: Learning gains [batch_size, seq_len, emb_size]
        """
        # Extract skill and response components
        skill_comp = self.skill_extractor(interaction_emb)
        response_comp = self.response_extractor(interaction_emb)
        
        # Combine skill and response information
        combined = torch.cat([skill_comp, response_comp], dim=-1)
        interaction_context = self.skill_response_interaction(combined)
        
        # Apply difficulty awareness
        difficulty_adjusted = self.difficulty_layer(interaction_context)
        
        # Compute final learning gains
        gains = self.gains_transform(difficulty_adjusted + interaction_context)
        
        return gains

class GainSAKTAttention(nn.Module):
    """
    Modified attention mechanism where Values represent learning gains.
    
    This implements the core innovation of GainSAKT where:
    - Q, K come from context embeddings (what the interaction is about)
    - V comes from learning gains (the value/gain of that interaction)
    
    Args:
        emb_size: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, emb_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        
        # Standard Q, K projections from context
        self.W_Q = nn.Linear(emb_size, emb_size)
        self.W_K = nn.Linear(emb_size, emb_size)
        
        # INNOVATION: V projection from learning gains
        self.W_V = nn.Linear(emb_size, emb_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, 
                context_seq: torch.Tensor, 
                gains_seq: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing attention as learning gain aggregation.
        
        Args:
            context_seq: Context sequence for Q,K [batch_size, seq_len, emb_size]
            gains_seq: Learning gains sequence for V [batch_size, seq_len, emb_size]
            mask: Causal mask [seq_len, seq_len]
            
        Returns:
            knowledge_state: Aggregated knowledge state [batch_size, seq_len, emb_size]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, emb_size = context_seq.shape
        
        # Project to Q, K, V
        Q = self.W_Q(context_seq)  # What we're looking for (current context)
        K = self.W_K(context_seq)  # What we have (historical patterns)
        V = self.W_V(gains_seq)    # INNOVATION: Values are learning gains
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate learning gains (core innovation)
        knowledge_state = torch.matmul(attention_weights, V)
        
        # Reshape back
        knowledge_state = knowledge_state.transpose(1, 2).contiguous().view(
            batch_size, seq_len, emb_size)
        
        return knowledge_state, attention_weights

class GainSAKTEncoderBlock(nn.Module):
    """
    Single encoder block implementing the GainSAKT attention mechanism.
    
    This block follows the standard Transformer architecture but with the key innovation
    that attention operates on learning gains rather than standard embeddings.
    
    Args:
        emb_size: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        ff_hidden_mult: Hidden size multiplier for feed-forward network
    """
    
    def __init__(self, emb_size: int, num_heads: int, dropout: float = 0.1, ff_hidden_mult: int = 4):
        super().__init__()
        
        # Core attention mechanism with learning gains
        self.attention = GainSAKTAttention(emb_size, num_heads, dropout)
        
        # Standard Transformer components
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * ff_hidden_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * ff_hidden_mult, emb_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                context_seq: torch.Tensor, 
                gains_seq: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder block.
        
        Args:
            context_seq: Context embeddings [batch_size, seq_len, emb_size]
            gains_seq: Learning gains [batch_size, seq_len, emb_size]
            mask: Causal attention mask
            
        Returns:
            output: Contextualized knowledge state [batch_size, seq_len, emb_size]
            attention_weights: Attention weights for interpretability
        """
        # Self-attention with learning gains
        attn_out, attention_weights = self.attention(context_seq, gains_seq, mask)
        
        # Add & Norm (residual connection with context)
        norm1_out = self.norm1(context_seq + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(norm1_out)
        
        # Add & Norm
        output = self.norm2(norm1_out + ffn_out)
        
        return output, attention_weights

class GainSAKT(nn.Module):
    """
    Complete GainSAKT model implementing the learning gains attention architecture.
    
    This model introduces a novel approach to knowledge tracing by using attention
    mechanisms to aggregate learning gains rather than abstract embeddings.
    
    Architecture Components:
    1. Base embeddings for interactions and exercises 
    2. Learning gains computation layer
    3. Single encoder block with learning gains attention
    4. Prediction head for response probability
    
    Args:
        num_c: Number of unique skills/concepts
        seq_len: Maximum sequence length
        emb_size: Embedding dimension
        num_attn_heads: Number of attention heads
        dropout: Dropout rate
        num_en: Number of encoder blocks (default: 1 for GainSAKT)
        emb_type: Embedding type (default: "qid")
        emb_path: Path to pretrained embeddings
        pretrain_dim: Dimension of pretrained embeddings
    """
    
    def __init__(self,
                 num_c: int,
                 seq_len: int = 200,
                 emb_size: int = 128,
                 num_attn_heads: int = 8,
                 dropout: float = 0.1,
                 num_en: int = 1,
                 emb_type: str = "qid",
                 emb_path: str = "",
                 pretrain_dim: int = 768):
        super().__init__()
        
        self.model_name = "gainsakt"
        self.emb_type = emb_type
        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en
        
        # Base embeddings similar to SAKT
        if emb_type.startswith("qid"):
            # Interaction embeddings (question * response combinations)
            self.interaction_emb = nn.Embedding(num_c * 2, emb_size)
            # Exercise embeddings (questions only)
            self.exercise_emb = nn.Embedding(num_c, emb_size)
            
        # Positional embeddings
        self.position_emb = nn.Embedding(seq_len, emb_size)
        
        # Learning gains computation layer (core innovation)
        self.learning_gains_layer = LearningGainsLayer(emb_size, num_c, dropout)
        
        # Single encoder block with gains attention
        self.encoder_block = GainSAKTEncoderBlock(emb_size, num_attn_heads, dropout)
        
        # Prediction layers
        self.dropout_layer = nn.Dropout(dropout)
        self.pred = nn.Linear(emb_size, 1)
        
        # Interpretability storage
        self.last_attention_weights = None
        self.last_learning_gains = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask to prevent attending to future interactions."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool().to(next(self.parameters()).device)
    
    def base_emb(self, q, r, qry):
        """
        Create base embeddings similar to SAKT.
        
        Args:
            q: Question sequences
            r: Response sequences  
            qry: Query sequences (shifted questions)
            
        Returns:
            qshftemb: Query embeddings
            xemb: Interaction embeddings with positional encoding
        """
        # Create interaction tokens (q + num_c * r)
        x = q + self.num_c * r
        
        # Get embeddings
        qshftemb = self.exercise_emb(qry)  # Query embeddings
        xemb = self.interaction_emb(x)     # Interaction embeddings
        
        # Add positional encoding
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        
        return qshftemb, xemb
    
    def forward(self, q, r, qry, qtest=False):
        """
        Forward pass of GainSAKT model.
        
        Args:
            q: Question sequences [batch_size, seq_len]
            r: Response sequences [batch_size, seq_len]
            qry: Query sequences [batch_size, seq_len]
            qtest: Whether in test mode
            
        Returns:
            p: Prediction probabilities [batch_size, seq_len]
            (optional) xemb: Final embeddings for analysis
        """
        # Get base embeddings
        if self.emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
        
        # Compute learning gains (core innovation)
        learning_gains = self.learning_gains_layer(xemb)
        self.last_learning_gains = learning_gains.detach()
        
        # Create causal mask
        seq_len = xemb.shape[1]
        causal_mask = self._create_causal_mask(seq_len)
        
        # Apply encoder block with gains attention
        knowledge_state, attention_weights = self.encoder_block(
            context_seq=xemb,
            gains_seq=learning_gains,
            mask=causal_mask
        )
        
        # Store attention weights for interpretability
        self.last_attention_weights = attention_weights.detach()
        
        # Make predictions
        p = torch.sigmoid(self.pred(self.dropout_layer(knowledge_state))).squeeze(-1)
        
        if not qtest:
            return p
        else:
            return p, knowledge_state
    
    def get_learning_gains(self):
        """Extract interpretable learning gains for analysis."""
        if self.last_learning_gains is not None:
            return self.last_learning_gains.cpu().numpy()
        return None
    
    def get_attention_weights(self):
        """Extract attention weights for interpretability."""
        if self.last_attention_weights is not None:
            return self.last_attention_weights.cpu().numpy()
        return None
    
    def explain_prediction(self, batch_idx: int = 0, seq_idx: int = -1):
        """
        Generate causal explanation based on gain aggregation.
        
        Args:
            batch_idx: Batch index to explain
            seq_idx: Sequence position to explain
            
        Returns:
            dict: Explanation containing attention weights and learning gains
        """
        if self.last_attention_weights is None or self.last_learning_gains is None:
            return None
        
        attn = self.last_attention_weights[batch_idx, :, seq_idx, :].cpu().numpy()  # [num_heads, seq_len]
        gains = self.last_learning_gains[batch_idx, :, :].cpu().numpy()  # [seq_len, emb_size]
        
        # Aggregate across heads
        avg_attn = attn.mean(axis=0)  # [seq_len]
        
        return {
            'attention_weights': avg_attn,
            'learning_gains': gains,
            'explanation': [(float(weight), gain) for weight, gain in zip(avg_attn, gains)]
        }