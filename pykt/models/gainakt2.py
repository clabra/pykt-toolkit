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

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        dropout (float): Dropout probability.
    """
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
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
                                           Shape: [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): A boolean mask to prevent attention to certain positions.
                                           Shape: [seq_len, seq_len]

        Returns:
            torch.Tensor: The output of the attention mechanism.
                          Shape: [batch_size, seq_len, d_model]
        """
        batch_size = context_sequence.size(0)

        # 1) Project and reshape Q, K, V for all heads in parallel
        Q = self.query_proj(context_sequence).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(context_sequence).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(value_sequence).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2) Compute attention scores
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # The mask shape [seq_len, seq_len] is broadcast to the scores tensor.
            scores = scores.masked_fill(mask, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # 3) Apply attention to V and reshape back
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4) Final linear projection
        projected_output = self.output_proj(output)
        return self.dropout(projected_output)

class EncoderBlock(nn.Module):
    """
    A single Transformer Encoder block for the GainAKT2 model.

    This block consists of a two-stream Multi-Head Attention layer followed by a 
    Position-wise Feed-Forward Network. Layer normalization and residual connections
    are applied after each sub-layer.

    Args:
        d_model (int): The dimensionality of the model.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the feed-forward network.
        dropout (float): The dropout rate.
    """
    
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
        """
        Forward pass for the encoder block.

        Args:
            context_sequence (torch.Tensor): The input sequence for context.
                                            Shape: [batch_size, seq_len, d_model]
            value_sequence (torch.Tensor): The input sequence for values.
                                           Shape: [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): The attention mask.

        Returns:
            torch.Tensor: The output sequence from the encoder block.
                          Shape: [batch_size, seq_len, d_model]
        """
        # Attention sub-layer
        attn_output = self.attn(context_sequence, value_sequence, mask)
        # Residual connection and layer normalization
        x = self.norm1(context_sequence + attn_output)
        
        # Feed-forward sub-layer
        ffn_output = self.ffn(x)
        # Residual connection and layer normalization
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

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
        
        # The pykt framework uses emb_type to distinguish embedding types.
        # We focus on the "qid" type, which is standard for this project.
        if emb_type.startswith("qid"):
            # Interaction embeddings for context (Q,K) and value (V) streams
            # The size is num_c * 2 to account for both concepts and responses (correct/incorrect)
            self.context_embedding = nn.Embedding(num_c * 2, d_model)
            self.value_embedding = nn.Embedding(num_c * 2, d_model)
            # A separate embedding for the target concept in the prediction head
            self.concept_embedding = nn.Embedding(num_c, d_model)
        
        # Positional encoding for the sequence
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        # Stack of encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_blocks)
        ])
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_ff), # Takes concatenated [knowledge_state, concept_embedding]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )

    def forward(self, q, r, qry=None, qtest=False):
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
            qtest (bool): If True, returns a tuple (predictions, encoded_sequence).
                          Used for analysis and interpretability.
        
        Returns:
            torch.Tensor or tuple: 
            - If qtest is False, returns prediction probabilities.
              Shape: [batch_size, seq_len]
            - If qtest is True, returns a tuple (predictions, encoded_sequence).
        """
        batch_size, seq_len = q.size()
        
        # Create interaction tokens by combining question and response IDs.
        # This is a standard encoding technique in the pykt framework.
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

        # 2. Add positional encodings to both streams
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        context_seq += pos_emb
        value_seq += pos_emb
        
        # 3. Pass sequences through the stack of encoder blocks
        # The context sequence is updated at each block, while the value sequence remains the same.
        encoded_seq = context_seq
        for block in self.encoder_blocks:
            encoded_seq = block(encoded_seq, value_seq, mask)
            
        # 4. Prepare inputs for the prediction head
        # Get the embedding for the target concept
        target_concept_emb = self.concept_embedding(target_concepts)
        
        # Concatenate the final knowledge state with the target concept embedding
        concatenated = torch.cat([encoded_seq, target_concept_emb], dim=-1)
        
        # 5. Generate predictions
        logits = self.prediction_head(concatenated)
        predictions = torch.sigmoid(logits.squeeze(-1))
        
        if not qtest:
            return predictions
        else:
            return predictions, encoded_seq