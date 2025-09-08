"""
GainAKT2: Gain-based Attention Knowledge Tracing Model

This model uses a Transformer-based architecture with two-stream attention mechanism.
The model separates context information from value information to better capture
learning gains and knowledge state evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):
    """Single attention head with separate context and value streams."""
    
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.query_proj = nn.Linear(d_model, d_k)
        self.key_proj = nn.Linear(d_model, d_k)
        self.value_proj = nn.Linear(d_model, d_k)

    def forward(self, query_source, key_source, value_source, mask=None):
        Q = self.query_proj(query_source)
        K = self.key_proj(key_source)
        V = self.value_proj(value_source)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

class MultiHeadAttention(nn.Module):
    """Multi-head attention with two input streams for context and values."""
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([AttentionHead(d_model, self.d_k) for _ in range(n_heads)])
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_sequence, value_sequence, mask=None):
        # Q and K come from the context_sequence, V comes from the value_sequence
        head_outputs = [h(context_sequence, context_sequence, value_sequence, mask) for h in self.heads]
        
        # Concatenate heads
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Final linear projection
        projected_output = self.output_proj(concatenated)
        return self.dropout(projected_output)

class EncoderBlock(nn.Module):
    """Transformer encoder block with two-stream attention."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_sequence, value_sequence, mask=None):
        # Attention step
        attn_output = self.attn(context_sequence, value_sequence, mask)
        # Add & Norm (residual from context)
        x = self.norm1(context_sequence + attn_output)
        
        # FFN step
        ffn_output = self.ffn(x)
        # Add & Norm
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class GainAKT2(nn.Module):
    """
    GainAKT2 model for Knowledge Tracing using gain-based attention mechanism.
    
    This model implements a Transformer-based architecture that separates context
    and value streams in attention computation to better model learning gains.
    """
    
    def __init__(self, num_c, seq_len=200, d_model=128, n_heads=8, num_encoder_blocks=2, 
                 d_ff=256, dropout=0.1, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "gainakt2"
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.dropout = dropout
        self.emb_type = emb_type
        
        if emb_type.startswith("qid"):
            # Standard PyKT interaction embedding: num_c * 2 for (concept, response) pairs
            self.context_embedding = nn.Embedding(num_c * 2, d_model)
            self.value_embedding = nn.Embedding(num_c * 2, d_model)
            # Concept embedding for prediction head
            self.concept_embedding = nn.Embedding(num_c, d_model)
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        # Encoder Stack
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_blocks)
        ])
        
        # Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )

    def forward(self, q, r, qry=None, qtest=False):
        """
        Forward pass following PyKT conventions.
        
        Args:
            q: questions/concepts [batch_size, seq_len]
            r: responses [batch_size, seq_len] 
            qry: query questions for prediction [batch_size, seq_len]
            qtest: whether in test mode
        
        Returns:
            Prediction logits [batch_size, seq_len] or tuple if qtest=True
        """
        batch_size, seq_len = q.size()
        
        # Create interaction tokens: q + num_c * r (standard PyKT encoding)
        interaction_tokens = q + self.num_c * r
        
        # Use query questions or shifted questions for prediction
        if qry is None:
            target_concepts = q  # Use current concepts for prediction
        else:
            target_concepts = qry
        
        # Create a causal mask to prevent attending to future positions
        mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device), diagonal=1).bool()

        # Get the two streams from the embedding tables
        context_seq = self.context_embedding(interaction_tokens)
        value_seq = self.value_embedding(interaction_tokens)

        # Add positional encodings
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        context_seq += pos_emb
        value_seq += pos_emb
        
        # Pass through the encoder blocks
        encoded_seq = context_seq
        for block in self.encoder_blocks:
            encoded_seq = block(encoded_seq, value_seq, mask)
            
        # Get embeddings for the target concepts for prediction
        target_concept_emb = self.concept_embedding(target_concepts)
        
        # Concatenate the final knowledge state with the target concept embedding
        concatenated = torch.cat([encoded_seq, target_concept_emb], dim=-1)
        
        # Pass through the final prediction head
        logits = self.prediction_head(concatenated)
        predictions = torch.sigmoid(logits.squeeze(-1))
        
        if not qtest:
            return predictions
        else:
            return predictions, encoded_seq