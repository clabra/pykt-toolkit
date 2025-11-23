"""
GainAKT4Exp: GainAKT4 with Training-Time Interpretability Monitoring

Extends GainAKT4 with monitoring hooks for real-time interpretability analysis.
"""

import torch
import torch.nn as nn
from .gainakt4 import GainAKT4


class GainAKT4Exp(GainAKT4):
    """
    GainAKT4 with monitoring support for training-time interpretability analysis.
    
    Additional features:
    - Monitoring hook for periodic state capture
    - Configurable monitoring frequency
    - DataParallel-safe monitoring
    """
    
    def __init__(self, num_c, seq_len, d_model=256, n_heads=4, num_encoder_blocks=4,
                 d_ff=512, dropout=0.2, emb_type='qid', lambda_bce=0.9,
                 monitor_frequency=50):
        super().__init__(
            num_c=num_c,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_blocks=num_encoder_blocks,
            d_ff=d_ff,
            dropout=dropout,
            emb_type=emb_type,
            lambda_bce=lambda_bce
        )
        
        self.monitor_frequency = monitor_frequency
        self.monitor = None
        self.global_batch_counter = 0
    
    def set_monitor(self, monitor):
        """Register a monitoring callback."""
        self.monitor = monitor
    
    def forward_with_states(self, q, r, qry=None):
        """
        Forward pass with full state capture for monitoring.
        
        Returns:
            dict with all outputs plus intermediate states:
                - All keys from base forward()
                - 'h1': knowledge state [B, L, d_model]
                - 'v1': value state [B, L, d_model]
                - 'questions': q
                - 'responses': r
        """
        # Run base forward
        output = self.forward(q, r, qry)
        
        # Capture intermediate states (recompute efficiently)
        batch_size, seq_len = q.size()
        device = q.device
        
        if qry is None:
            qry = q
        
        # Recreate embeddings and encoder pass
        r_int = r.long()
        interaction_tokens = q + self.num_c * r_int
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        
        context_seq = self.context_embedding(interaction_tokens) + self.pos_embedding(positions)
        value_seq = self.value_embedding(interaction_tokens) + self.pos_embedding(positions)
        
        for encoder_block in self.encoder_blocks:
            context_seq, value_seq = encoder_block(context_seq, value_seq, mask)
        
        # Add states to output
        output['h1'] = context_seq
        output['v1'] = value_seq
        output['questions'] = q
        output['responses'] = r
        
        # Call monitor if enabled
        if self.monitor is not None and self.training:
            self.global_batch_counter += 1
            
            # Check if on primary device (DataParallel safety)
            primary_device = (not hasattr(self, 'device_ids') or 
                            q.device == torch.device(f'cuda:{self.device_ids[0]}'))
            
            should_monitor = (
                self.global_batch_counter % self.monitor_frequency == 0 and
                primary_device
            )
            
            if should_monitor:
                with torch.no_grad():
                    self.monitor(
                        batch_idx=self.global_batch_counter,
                        h1=output['h1'],
                        v1=output['v1'],
                        skill_vector=output['skill_vector'],
                        bce_predictions=output['bce_predictions'],
                        mastery_predictions=output['mastery_predictions'],
                        questions=q,
                        responses=r
                    )
        
        return output


def create_exp_model(config):
    """
    Factory function to create GainAKT4Exp model.
    
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
        - monitor_frequency: batches between monitoring (default: 50)
    
    Note: lambda_mastery is automatically computed as 1.0 - lambda_bce
    """
    return GainAKT4Exp(
        num_c=config['num_c'],
        seq_len=config['seq_len'],
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        num_encoder_blocks=config.get('num_encoder_blocks', 4),
        d_ff=config.get('d_ff', 512),
        dropout=config.get('dropout', 0.2),
        emb_type=config.get('emb_type', 'qid'),
        lambda_bce=config.get('lambda_bce', 0.9),
        monitor_frequency=config.get('monitor_frequency', 50)
    )
