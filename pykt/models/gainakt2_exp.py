"""
GainAKT2 model with training-time interpretability monitoring support.

This is an enhanced version of the GainAKT2 model that includes:
1. Additional forward method returning internal states for monitoring
2. Integration hooks for training-time interpretability analysis
3. Support for auxiliary loss functions based on interpretability constraints
"""

import torch
from .gainakt2 import GainAKT2


class GainAKT2Exp(GainAKT2):
    """
    Enhanced GainAKT2 model with training-time interpretability monitoring.
    
    Extends the base GainAKT2 model to provide:
    - Internal state access for interpretability analysis
    - Training-time monitoring hook integration
    - Auxiliary loss computation for interpretability constraints
    """
    
    def __init__(self, num_c, seq_len=200, d_model=128, n_heads=8, num_encoder_blocks=2, 
                 d_ff=256, dropout=0.1, emb_type="qid", emb_path="", pretrain_dim=768,
                 use_mastery_head=True, use_gain_head=True, non_negative_loss_weight=0.1,
                 monotonicity_loss_weight=0.1, mastery_performance_loss_weight=0.1,
                 gain_performance_loss_weight=0.1, sparsity_loss_weight=0.1,
                 consistency_loss_weight=0.1, monitor_frequency=50):
        """
        Initialize the monitored GainAKT2Exp model.
        
        Args:
            monitor_frequency (int): How often to compute interpretability metrics during training
            Other args: Same as base GainAKT2 model
        """
        # Force interpretability heads to be enabled for monitoring
        super().__init__(
            num_c=num_c, seq_len=seq_len, d_model=d_model, n_heads=n_heads,
            num_encoder_blocks=num_encoder_blocks, d_ff=d_ff, dropout=dropout,
            emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim,
            use_mastery_head=True, use_gain_head=True, 
            non_negative_loss_weight=non_negative_loss_weight,
            monotonicity_loss_weight=monotonicity_loss_weight,
            mastery_performance_loss_weight=mastery_performance_loss_weight,
            gain_performance_loss_weight=gain_performance_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            consistency_loss_weight=consistency_loss_weight
        )
        
        self.monitor_frequency = monitor_frequency
        self.interpretability_monitor = None
        self.step_count = 0
        
    def set_monitor(self, monitor):
        """Set the interpretability monitor hook."""
        self.interpretability_monitor = monitor
        
    def forward_with_states(self, q, r, qry=None, batch_idx=None):
        """
        Forward pass that returns internal states for interpretability monitoring.
        
        Args:
            q, r, qry: Same as base forward method
            batch_idx: Current batch index for monitoring frequency
        
        Returns:
            dict: Standard model outputs plus internal states:
                - 'predictions': Response probabilities
                - 'context_seq': Final context sequence
                - 'value_seq': Final value sequence  
                - 'projected_mastery': Mastery projections
                - 'projected_gains': Gain projections
                - 'interpretability_loss': Additional loss from constraints
        """
        batch_size, seq_len = q.size()
        
        # Create interaction tokens by combining question and response IDs
        # Convert responses to integers (0 or 1) for embedding lookup
        r_int = r.long()
        interaction_tokens = q + self.num_c * r_int
        
        # Determine the target concepts for prediction
        if qry is None:
            target_concepts = q
        else:
            target_concepts = qry
        
        # Create a causal attention mask
        mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device), diagonal=1).bool()

        # 1. Get embeddings for the two streams
        context_seq = self.context_embedding(interaction_tokens)
        value_seq = self.value_embedding(interaction_tokens)

        # 2. Add positional encodings
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        context_seq += pos_emb
        value_seq += pos_emb
        
        # 3. Pass sequences through encoder blocks
        for block in self.encoder_blocks:
            context_seq, value_seq = block(context_seq, value_seq, mask)
        
        # 4. Generate predictions
        target_concept_emb = self.concept_embedding(target_concepts)
        concatenated = torch.cat([context_seq, value_seq, target_concept_emb], dim=-1)
        logits = self.prediction_head(concatenated)
        predictions = torch.sigmoid(logits.squeeze(-1))
        
        # 5. Compute interpretability projections with proper constraints
        # Apply ReLU to gains to ensure non-negativity
        projected_gains_raw = self.gain_head(value_seq)  
        projected_gains = torch.relu(projected_gains_raw)
        
        # Compute cumulative mastery to ENFORCE monotonicity
        # Method: mastery[t] = mastery[t-1] + gains[t] (cumulative learning)
        projected_mastery_raw = self.mastery_head(context_seq)
        initial_mastery = torch.sigmoid(projected_mastery_raw)  # Base mastery estimates
        
        # Build cumulative mastery sequence (perfectly monotonic)
        batch_size, seq_len, num_c = initial_mastery.shape
        projected_mastery = torch.zeros_like(initial_mastery)
        
        # Initialize first timestep with base mastery
        projected_mastery[:, 0, :] = initial_mastery[:, 0, :]
        
        # Accumulate learning gains to ensure monotonicity
        for t in range(1, seq_len):
            # Mastery = previous mastery + current gains, capped at 1.0
            accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1  # Scale gains
            projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
        
        # 6. Prepare output with internal states
        output = {
            'predictions': predictions,
            'context_seq': context_seq,
            'value_seq': value_seq,
            'projected_mastery': projected_mastery,
            'projected_gains': projected_gains
        }
        
        # 7. Compute interpretability loss
        interpretability_loss = self.compute_interpretability_loss(
            projected_mastery, projected_gains, predictions, q, r
        )
        output['interpretability_loss'] = interpretability_loss
        
        # 8. Call interpretability monitor if enabled and at right frequency
        if (self.interpretability_monitor is not None and 
            batch_idx is not None and 
            batch_idx % self.monitor_frequency == 0):
            
            with torch.no_grad():
                self.interpretability_monitor(
                    batch_idx=batch_idx,
                    context_seq=context_seq,
                    value_seq=value_seq, 
                    projected_mastery=projected_mastery,
                    projected_gains=projected_gains,
                    predictions=predictions,
                    questions=q,
                    responses=r
                )
        
        return output
    
    def forward(self, q, r, qry=None, qtest=False):
        """
        Standard forward method maintaining compatibility with PyKT framework.
        """
        # For standard forward, just return basic outputs
        output = self.forward_with_states(q, r, qry)
        
        # Return only the standard outputs for PyKT compatibility
        result = {'predictions': output['predictions']}
        
        if qtest:
            result['encoded_seq'] = output['context_seq']
        if self.use_mastery_head:
            result['projected_mastery'] = output['projected_mastery']
        if self.use_gain_head:
            result['projected_gains'] = output['projected_gains']
            
        return result
    
    def compute_interpretability_loss(self, projected_mastery, projected_gains, predictions, questions, responses):
        """
        Compute auxiliary loss based on interpretability constraints.
        
        Args:
            projected_mastery: [batch_size, seq_len, num_c] - mastery projections
            projected_gains: [batch_size, seq_len, num_c] - gain projections  
            predictions: [batch_size, seq_len] - model predictions
            questions: [batch_size, seq_len] - question IDs
            responses: [batch_size, seq_len] - correct/incorrect responses
            
        Returns:
            torch.Tensor: Auxiliary loss value
        """
        total_loss = 0.0
        batch_size, seq_len, num_c = projected_mastery.shape

        # Create masks for relevant skills
        skill_masks = torch.zeros((batch_size, seq_len, num_c), device=questions.device).bool()
        skill_masks.scatter_(2, questions.unsqueeze(-1), 1)

        # 1. Non-negative learning gains
        if self.non_negative_loss_weight > 0:
            negative_gains = torch.clamp(-projected_gains, min=0)
            non_negative_loss = negative_gains.mean()
            total_loss += self.non_negative_loss_weight * non_negative_loss

        # 2. Monotonicity of mastery
        if self.monotonicity_loss_weight > 0 and seq_len > 1:
            mastery_decrease = torch.clamp(projected_mastery[:, :-1] - projected_mastery[:, 1:], min=0)
            monotonicity_loss = mastery_decrease.mean()
            total_loss += self.monotonicity_loss_weight * monotonicity_loss

        # 3. Mastery-performance correlation
        if self.mastery_performance_loss_weight > 0:
            relevant_mastery = projected_mastery[skill_masks]
            correct_mask = (responses == 1).flatten()
            incorrect_mask = (responses == 0).flatten()

            # Penalize low mastery for correct answers
            low_mastery_on_correct = torch.clamp(1 - relevant_mastery[correct_mask], min=0)
            # Penalize high mastery for incorrect answers
            high_mastery_on_incorrect = torch.clamp(relevant_mastery[incorrect_mask], min=0)

            mastery_performance_loss = low_mastery_on_correct.mean() + high_mastery_on_incorrect.mean()
            total_loss += self.mastery_performance_loss_weight * mastery_performance_loss

        # 4. Gain-performance correlation
        if self.gain_performance_loss_weight > 0:
            relevant_gains = projected_gains[skill_masks]
            correct_gains = relevant_gains[(responses == 1).flatten()]
            incorrect_gains = relevant_gains[(responses == 0).flatten()]

            if correct_gains.numel() > 0 and incorrect_gains.numel() > 0:
                # Hinge loss: incorrect gains should be smaller than correct gains
                gain_performance_loss = torch.clamp(incorrect_gains.mean() - correct_gains.mean() + 0.1, min=0) # 0.1 margin
                total_loss += self.gain_performance_loss_weight * gain_performance_loss

        # 5. Sparsity of gains
        if self.sparsity_loss_weight > 0:
            non_relevant_gains = projected_gains[~skill_masks]
            sparsity_loss = torch.abs(non_relevant_gains).mean()
            total_loss += self.sparsity_loss_weight * sparsity_loss

        # 6. Consistency between mastery increments and gains (architectural scaling factor 0.1)
        #    Penalize deviation between actual mastery change and scaled gains.
        if self.consistency_loss_weight > 0 and seq_len > 1:
            mastery_delta = projected_mastery[:, 1:, :] - projected_mastery[:, :-1, :]
            scaled_gains = projected_gains[:, 1:, :] * 0.1
            consistency_residual = torch.abs(mastery_delta - scaled_gains)
            consistency_loss = consistency_residual.mean()
            total_loss += self.consistency_loss_weight * consistency_loss

        return total_loss


def create_exp_model(config):
    """
    Factory function to create a GainAKT2Exp model from config.
    
    Args:
        config (dict): Model configuration parameters
        
    Returns:
        GainAKT2Exp: Configured model instance
    """
    return GainAKT2Exp(
        num_c=config.get('num_c', 100),
        seq_len=config.get('seq_len', 200), 
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        num_encoder_blocks=config.get('num_encoder_blocks', 4),
        d_ff=config.get('d_ff', 768),
        dropout=config.get('dropout', 0.2),
        emb_type=config.get('emb_type', 'qid'),
        non_negative_loss_weight=config.get('non_negative_loss_weight', 0.1),
        monotonicity_loss_weight=config.get('monotonicity_loss_weight', 0.1),
        mastery_performance_loss_weight=config.get('mastery_performance_loss_weight', 0.1),
        gain_performance_loss_weight=config.get('gain_performance_loss_weight', 0.1),
        sparsity_loss_weight=config.get('sparsity_loss_weight', 0.1),
        consistency_loss_weight=config.get('consistency_loss_weight', 0.1),
        monitor_frequency=config.get('monitor_frequency', 50)
    )