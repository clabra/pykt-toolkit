"""
GainAKT3 model with training-time interpretability monitoring support.

This is an enhanced version of the GainAKT3 model that includes:
1. Additional forward method returning internal states for monitoring
2. Integration hooks for training-time interpretability analysis
3. Support for auxiliary loss functions based on interpretability constraints
"""

import torch
import os
from .gainakt3 import GainAKT3


class GainAKT3Exp(GainAKT3):
    """
    Enhanced GainAKT3 model with training-time interpretability monitoring.
    
    Extends the base GainAKT3 model to provide:
    - Internal state access for interpretability analysis
    - Training-time monitoring hook integration
    - Auxiliary loss computation for interpretability constraints
    """
    
    def __init__(self, num_c, seq_len=200, d_model=128, n_heads=8, num_encoder_blocks=2, 
                 d_ff=256, dropout=0.1, emb_type="qid", emb_path="", pretrain_dim=768,
                 use_mastery_head=True, use_gain_head=True, intrinsic_gain_attention=False,
                 non_negative_loss_weight=0.1,
                 monotonicity_loss_weight=0.1, mastery_performance_loss_weight=0.1,
                 gain_performance_loss_weight=0.1, sparsity_loss_weight=0.1,
                 consistency_loss_weight=0.1, monitor_frequency=50, use_skill_difficulty=False,
                 use_student_speed=False, num_students=None, mastery_threshold_init=0.85,
                 threshold_temperature=1.0):
        """
        Initialize the monitored GainAKT3Exp model.
        
        Args:
            monitor_frequency (int): How often to compute interpretability metrics during training
            intrinsic_gain_attention (bool): If True, use intrinsic gain attention mode
            use_student_speed (bool): If True, add per-student learning speed embeddings
            num_students (int): Number of unique students (required if use_student_speed=True)
            mastery_threshold_init (float): Initial value for learnable mastery threshold (default: 0.85, normalized [0,1])
            threshold_temperature (float): Temperature for sigmoid threshold function (default: 1.0)
            Other args: Same as base GainAKT3 model
        """
        # Allow disabling heads for a pure predictive baseline; otherwise enable.
        super().__init__(
            num_c=num_c, seq_len=seq_len, d_model=d_model, n_heads=n_heads,
            num_encoder_blocks=num_encoder_blocks, d_ff=d_ff, dropout=dropout,
            emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim,
            use_mastery_head=use_mastery_head, use_gain_head=use_gain_head,
            intrinsic_gain_attention=intrinsic_gain_attention,
            non_negative_loss_weight=non_negative_loss_weight,
            monotonicity_loss_weight=monotonicity_loss_weight,
            mastery_performance_loss_weight=mastery_performance_loss_weight,
            gain_performance_loss_weight=gain_performance_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            use_skill_difficulty=use_skill_difficulty,
            use_student_speed=use_student_speed,
            num_students=num_students
        )
        
        self.monitor_frequency = monitor_frequency
        self.interpretability_monitor = None
        self.step_count = 0
        
        # GainAKT3Exp specific: Learnable mastery threshold per skill
        # Initialize around 0.85 (normalized [0,1] scale) representing mastery level for correct predictions
        self.mastery_threshold = torch.nn.Parameter(
            torch.clamp(torch.full((num_c,), mastery_threshold_init, dtype=torch.float32), 0.0, 1.0)
        )
        self.threshold_temperature = threshold_temperature
        
    def set_monitor(self, monitor):
        """Set the interpretability monitor hook."""
        self.interpretability_monitor = monitor
        
    def forward_with_states(self, q, r, qry=None, batch_idx=None, student_ids=None):
        """
        Forward pass that returns internal states for interpretability monitoring.
        
        Args:
            q, r, qry: Same as base forward method
            batch_idx: Current batch index for monitoring frequency
            student_ids: Student IDs [batch_size] (required if use_student_speed=True)
        
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
        
        if self.intrinsic_gain_attention:
            # Apply non-negativity activation to gains
            value_seq = self.gain_activation(value_seq)

        # 2. Add positional encodings
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        context_seq += pos_emb
        
        if not self.intrinsic_gain_attention:
            # Legacy mode: add positional encoding to value stream
            value_seq += pos_emb
        
        # 3. Pass sequences through encoder blocks
        for block in self.encoder_blocks:
            context_seq, value_seq = block(context_seq, value_seq, mask)
        
        # 4. Generate predictions
        target_concept_emb = self.concept_embedding(target_concepts)
        
        # Apply skill difficulty modulation (if enabled)
        if self.use_skill_difficulty:
            difficulty_scale = torch.gather(
                self.skill_difficulty_scale.unsqueeze(0).expand(batch_size, -1),
                1, target_concepts
            )
            difficulty_scale = torch.clamp(difficulty_scale, 0.5, 2.0).unsqueeze(-1)
            target_concept_emb = target_concept_emb * difficulty_scale
        
        # Add student learning speed embedding (if enabled)
        if self.use_student_speed:
            assert student_ids is not None, "student_ids required when use_student_speed=True"
            student_emb = self.student_speed_embedding(student_ids)
            student_emb = student_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        if self.intrinsic_gain_attention:
            # Intrinsic mode: [h_t, concept_embedding] or [h_t, concept_embedding, student_speed]
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
            
        logits = self.prediction_head(concatenated).squeeze(-1)
        # Defer sigmoid until evaluation to allow using BCEWithLogitsLoss safely under AMP
        predictions = torch.sigmoid(logits)
        
        # 5. Optionally compute interpretability projections
        if self.intrinsic_gain_attention:
            # Intrinsic mode: retrieve aggregated gains directly from attention mechanism
            aggregated_gains = self.get_aggregated_gains()
            
            if aggregated_gains is not None:
                # Apply non-negativity activation to aggregated gains
                projected_gains = torch.relu(aggregated_gains)  # Ensure non-negativity
                
                # Compute cumulative mastery from gains
                batch_size, seq_len, num_c = projected_gains.shape
                projected_mastery = torch.zeros_like(projected_gains)
                projected_mastery[:, 0, :] = torch.clamp(projected_gains[:, 0, :] * 0.1, min=0.0, max=1.0)
                for t in range(1, seq_len):
                    accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
                    projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
            else:
                projected_mastery = None
                projected_gains = None
        elif self.use_gain_head and self.use_mastery_head:
            # GainAKT3Exp mode: Values directly as gains (ReLU ensures non-negativity)
            # Gains are D-dimensional [B, L, D] from encoder values
            gains_d = torch.relu(value_seq)  # ReLU ensures positive gains
            
            # Project D-dimensional gains to per-skill gains [B, L, num_c]
            # Use gain_head to extract skill-specific learning gains from D-dimensional representation
            if hasattr(self, 'gain_head') and self.gain_head is not None:
                projected_gains_raw = torch.relu(self.gain_head(gains_d))  # [B, L, num_c]
            else:
                # Fallback: aggregate D-dimensional gains across dimensions
                # Then broadcast to num_c (each skill gets same aggregated gain)
                aggregated_gains = gains_d.mean(dim=-1, keepdim=True)  # [B, L, 1]
                projected_gains_raw = aggregated_gains.expand(-1, -1, self.num_c)  # [B, L, num_c]
            
            # Normalize gains to [0, 1] range using sigmoid
            # This ensures gains are bounded and interpretable
            projected_gains = torch.sigmoid(projected_gains_raw)  # [B, L, num_c] in [0, 1]
            
            # Initialize per-skill mastery at 0.0 (no knowledge)
            batch_size, seq_len, _ = projected_gains.shape
            projected_mastery = torch.zeros(batch_size, seq_len, self.num_c, device=q.device)
            
            # Accumulate gains with soft normalization to keep mastery in reasonable range
            # mastery[skill][t] = mastery[skill][t-1] + gains[skill][t]
            # Apply soft clipping using tanh to prevent unbounded growth while maintaining differentiability
            max_theoretical_mastery = 10.0  # Theoretical maximum after many interactions
            for t in range(1, seq_len):
                # Per-skill mastery accumulation: only update mastery for concepts being practiced
                # Copy previous mastery levels for all concepts
                projected_mastery[:, t, :] = projected_mastery[:, t-1, :].clone()
                # Only update the concept being practiced at this time step
                practiced_concepts = q[:, t].long()  # [B] - concepts being practiced at time t
                batch_indices = torch.arange(batch_size, device=q.device)
                # Add gains and apply normalization only for practiced concepts
                updated_mastery = projected_mastery[batch_indices, t-1, practiced_concepts] + projected_gains[batch_indices, t, practiced_concepts]
                projected_mastery[batch_indices, t, practiced_concepts] = max_theoretical_mastery * torch.tanh(updated_mastery / max_theoretical_mastery)
            
            # Final clipping to ensure mastery stays in [0, 1] for interpretability
            projected_mastery = torch.clamp(projected_mastery, min=0.0, max=1.0)
            
            # DEBUG: Log mastery accumulation statistics for first batch
            if batch_idx == 0:  # Log for first batch of each epoch
                mastery_range = projected_mastery.max() - projected_mastery.min()
                mastery_std = projected_mastery.std()
                practiced_mask = (q != 0).float()  # Non-zero concepts are practiced
                practiced_count = practiced_mask.sum(dim=1).mean()  # Average practiced concepts per sequence
                
                print("DEBUG GainAKT3Exp - Mastery accumulation stats:")
                print(f"  Mastery range: {mastery_range:.4f}, std: {mastery_std:.4f}")
                print(f"  Avg practiced concepts per sequence: {practiced_count:.1f}")
                print("  Sample mastery progression for first student, first 5 skills:")
                for skill in range(min(5, projected_mastery.shape[2])):
                    skill_mastery = projected_mastery[0, :, skill]
                    practiced_timesteps = (q[0] == skill).nonzero(as_tuple=True)[0]
                    if len(practiced_timesteps) > 0:
                        print(f"    Skill {skill}: {skill_mastery[practiced_timesteps].cpu().detach().numpy()}")
            
            # Generate predictions using learnable threshold
            # For each skill at each timestep: sigmoid((mastery - threshold) / temperature)
            # If mastery >= threshold, prediction approaches 1.0 (correct)
            # If mastery < threshold, prediction approaches 0.0 (incorrect)
            
            # Get the skill ID for each timestep
            skill_indices = target_concepts.long()  # [B, L]
            
            # Gather mastery for the actual skills being tested
            # projected_mastery: [B, L, num_c], we need [B, L] by selecting skill at each step
            batch_indices = torch.arange(batch_size, device=q.device).unsqueeze(1).expand(-1, seq_len)
            time_indices = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
            skill_mastery = projected_mastery[batch_indices, time_indices, skill_indices]  # [B, L]
            
            # Gather threshold for each skill and clamp to [0,1] during training/inference
            skill_threshold = torch.clamp(self.mastery_threshold[skill_indices], 0.0, 1.0)  # [B, L]
            
            # Compute threshold-based predictions (differentiable via sigmoid)
            threshold_predictions = torch.sigmoid((skill_mastery - skill_threshold) / self.threshold_temperature)
            
            # DEBUG: Log threshold and prediction stats for first batch
            if batch_idx == 0:  # Log for first batch of each epoch
                threshold_range = skill_threshold.max() - skill_threshold.min()
                threshold_std = skill_threshold.std()
                prediction_range = threshold_predictions.max() - threshold_predictions.min()
                prediction_std = threshold_predictions.std()
                
                print("DEBUG GainAKT3Exp - Threshold and prediction stats:")
                print(f"  Threshold range: {threshold_range:.4f}, std: {threshold_std:.4f}")
                print(f"  Prediction range: {prediction_range:.4f}, std: {prediction_std:.4f}")
                print(f"  Temperature: {self.threshold_temperature}")
                print(f"  Sample thresholds for first 5 skills: {self.mastery_threshold[:5].cpu().detach().numpy()}")
                print(f"  Sample predictions for first student: {threshold_predictions[0, :10].cpu().detach().numpy()}")
            
            # Override the base model predictions with threshold-based predictions
            predictions = threshold_predictions
        else:
            projected_mastery = None
            projected_gains = None

        # 6. Prepare output with internal states
        output = {
            'predictions': predictions,
            'logits': logits,
            'context_seq': context_seq,
            'value_seq': value_seq
        }
        if projected_mastery is not None:
            output['projected_mastery'] = projected_mastery
        if projected_gains is not None:
            output['projected_gains'] = projected_gains
        
        # Store D-dimensional gains for interpretability (if in GainAKT3Exp mode)
        if self.use_gain_head and self.use_mastery_head and value_seq is not None:
            output['projected_gains_d'] = torch.relu(value_seq)
        
        # 7. Compute interpretability loss
        if (projected_mastery is not None) and (projected_gains is not None):
            interpretability_loss = self.compute_interpretability_loss(
                projected_mastery, projected_gains, predictions, q, r
            )
        else:
            interpretability_loss = torch.tensor(0.0, device=q.device)
        output['interpretability_loss'] = interpretability_loss
        
        # Optional debug: log device placement once (controlled by env var)
        if batch_idx == 0 and bool(int(os.environ.get('PYKT_DEBUG_DP_DEVICES', '0'))):
            try:
                # Only emit from primary device (index 0) to avoid duplicate logs under DataParallel
                if hasattr(q, 'device') and (q.device.index is None or q.device.index == 0):
                    print(f"[DP-DEBUG] forward_with_states: q.device={q.device} context_seq.device={context_seq.device}")
            except Exception:
                pass

        # 8. Call interpretability monitor if enabled and at right frequency.
        # Guard to execute only on primary replica (device index 0) to prevent duplicate side-effects under DataParallel.
        primary_device = (hasattr(q, 'device') and (q.device.index is None or q.device.index == 0))
        if (self.interpretability_monitor is not None and 
            batch_idx is not None and 
            batch_idx % self.monitor_frequency == 0 and primary_device):
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
    
    def forward(self, q, r, qry=None, qtest=False, student_ids=None):
        """
        Standard forward method maintaining compatibility with PyKT framework.
        """
        # For standard forward, just return basic outputs
        output = self.forward_with_states(q, r, qry, student_ids=student_ids)
        
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
    Factory function to create a GainAKT3Exp model from config.
    
    All parameters must be present in config dict (no fallback defaults).
    Fails fast with clear KeyError if any parameter is missing.
    
    Args:
        config (dict): Model configuration parameters (all required)
        
    Returns:
        GainAKT3Exp: Configured model instance
        
    Raises:
        KeyError: If any required parameter is missing from config
    """
    try:
        return GainAKT3Exp(
            num_c=config['num_c'],
            seq_len=config['seq_len'], 
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_encoder_blocks=config['num_encoder_blocks'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            emb_type=config['emb_type'],
            use_mastery_head=config['use_mastery_head'],
            use_gain_head=config['use_gain_head'],
            intrinsic_gain_attention=config['intrinsic_gain_attention'],
            use_skill_difficulty=config['use_skill_difficulty'],
            use_student_speed=config['use_student_speed'],
            num_students=config['num_students'],  # Set dynamically from dataset, fallback handled in training script
            non_negative_loss_weight=config['non_negative_loss_weight'],
            monotonicity_loss_weight=config['monotonicity_loss_weight'],
            mastery_performance_loss_weight=config['mastery_performance_loss_weight'],
            gain_performance_loss_weight=config['gain_performance_loss_weight'],
            sparsity_loss_weight=config['sparsity_loss_weight'],
            consistency_loss_weight=config['consistency_loss_weight'],
            monitor_frequency=config['monitor_frequency']
        )
    except KeyError as e:
        raise ValueError(f"Missing required parameter in model config: {e}. "
                        f"All parameters must be explicitly provided (no defaults).") from e