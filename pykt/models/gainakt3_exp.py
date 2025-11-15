"""
GainAKT3 model with training-time interpretability monitoring support.

This is an enhanced version of the GainAKT3 model that includes:
1. Additional forward method returning internal states for monitoring
2. Integration hooks for training-time interpretability analysis
3. Support for auxiliary loss functions based on interpretability constraints

================================================================================
CORE ARCHITECTURAL PRINCIPLE: Values ARE Learning Gains
================================================================================

Key Conceptual Innovation:
--------------------------
In GainAKT3Exp, the Value stream from the transformer encoder DIRECTLY represents
learning gains. Each interaction's Value output quantifies: "How much did the 
student learn from this (skill, response) experience?"

Educational Semantics:
----------------------
- Value[t] = learning gain from interaction t
- Direct mapping: no intermediate projections obscure the relationship
- Interpretable by design: can inspect any Value and understand its meaning
- Recursive accumulation: mastery[skill, t] = mastery[skill, t-1] + α × ReLU(Value[t])

Architectural Simplification:
-----------------------------
OLD APPROACH (commented out):
  Value → ReLU → gain_head projection → per-skill gains → mastery accumulation

NEW APPROACH (current):
  Value → ReLU (non-negativity) → aggregate to scalar → mastery accumulation

Benefits:
---------
1. Maximal Interpretability: Values have direct educational meaning
2. Architectural Simplicity: Fewer layers, clearer data flow
3. Transparency: Can trace any learning gain to its source Value
4. Educational Validity: Aligns with learning science principles

The gain_head projection layer has been DEPRECATED and commented out throughout
the codebase. The use_gain_head parameter is kept for backward compatibility 
but does not instantiate a projection layer.

See paper/STATUS_gainakt3exp.md for detailed architectural documentation.
================================================================================
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
                 consistency_loss_weight=0.1, incremental_mastery_loss_weight=0.1,
                 monitor_frequency=50, use_skill_difficulty=False,
                 use_student_speed=False, num_students=None, mastery_threshold_init=0.85,
                 threshold_temperature=1.0):
        """
        Initialize the monitored GainAKT3Exp model.
        
        ⚠️  REPRODUCIBILITY WARNING ⚠️
        ═══════════════════════════════════════════════════════════════════════════
        The default parameter values in this constructor are provided ONLY for:
        - Backwards compatibility with existing code
        - Unit testing convenience
        - Quick prototyping in notebooks
        
        **DO NOT RELY ON THESE DEFAULTS IN PRODUCTION CODE**
        
        Production Usage:
        - ✅ ALWAYS use the factory function: create_exp_model(config)
        - ✅ ALL parameters must be explicit in config dict
        - ✅ Single source of truth: configs/parameter_default.json
        - ❌ NEVER instantiate directly: GainAKT3Exp(num_c=100)  # Uses hidden defaults!
        
        See examples/reproducibility.md for the "Zero Defaults" policy.
        ═══════════════════════════════════════════════════════════════════════════
        
        Args:
            num_c (int): Number of unique skills/concepts (REQUIRED, no default)
            seq_len (int): Maximum sequence length (default: 200, DO NOT RELY ON THIS)
            d_model (int): Model embedding dimension (default: 128, DO NOT RELY ON THIS)
            n_heads (int): Number of attention heads (default: 8, DO NOT RELY ON THIS)
            num_encoder_blocks (int): Number of transformer blocks (default: 2, DO NOT RELY ON THIS)
            d_ff (int): Feed-forward dimension (default: 256, DO NOT RELY ON THIS)
            dropout (float): Dropout rate (default: 0.1, DO NOT RELY ON THIS)
            emb_type (str): Embedding type "qid" or other (default: "qid", DO NOT RELY ON THIS)
            emb_path (str): Path to pretrained embeddings (default: "", DO NOT RELY ON THIS)
            pretrain_dim (int): Pretrained embedding dimension (default: 768, DO NOT RELY ON THIS)
            use_mastery_head (bool): Enable mastery projection head (default: True, DO NOT RELY ON THIS)
            use_gain_head (bool): Enable gain projection head (default: True, DO NOT RELY ON THIS)
            intrinsic_gain_attention (bool): Use intrinsic gain attention mode (default: False, DO NOT RELY ON THIS)
            non_negative_loss_weight (float): Weight for non-negative constraint (default: 0.1, DO NOT RELY ON THIS)
            monotonicity_loss_weight (float): Weight for monotonicity constraint (default: 0.1, DO NOT RELY ON THIS)
            mastery_performance_loss_weight (float): Weight for mastery-performance alignment (default: 0.1, DO NOT RELY ON THIS)
            gain_performance_loss_weight (float): Weight for gain-performance alignment (default: 0.1, DO NOT RELY ON THIS)
            sparsity_loss_weight (float): Weight for sparsity constraint (default: 0.1, DO NOT RELY ON THIS)
            consistency_loss_weight (float): Weight for consistency constraint (default: 0.1, DO NOT RELY ON THIS)
            monitor_frequency (int): How often to compute interpretability metrics (default: 50, DO NOT RELY ON THIS)
            use_skill_difficulty (bool): Enable per-skill difficulty embeddings (default: False, DO NOT RELY ON THIS)
            use_student_speed (bool): Enable per-student learning speed embeddings (default: False, DO NOT RELY ON THIS)
            num_students (int): Number of unique students (default: None, DO NOT RELY ON THIS)
            mastery_threshold_init (float): Initial learnable mastery threshold (default: 0.85, DO NOT RELY ON THIS)
            threshold_temperature (float): Temperature for sigmoid threshold function (default: 1.0, DO NOT RELY ON THIS)
        """
        # ═══════════════════════════════════════════════════════════════════════════
        # REPRODUCIBILITY ENFORCEMENT: Warn if strict mode enabled
        # ═══════════════════════════════════════════════════════════════════════════
        # In strict reproducibility mode, warn about potential use of constructor defaults
        # Set PYKT_STRICT_REPRODUCIBILITY=1 to enable this warning during development
        if bool(int(os.environ.get('PYKT_STRICT_REPRODUCIBILITY', '0'))):
            import warnings
            warnings.warn(
                "\n" + "="*80 + "\n"
                "⚠️  REPRODUCIBILITY WARNING: Direct model instantiation detected!\n"
                "="*80 + "\n"
                "You appear to be instantiating GainAKT3Exp directly instead of using\n"
                "the factory function create_exp_model(config).\n\n"
                "This violates the 'Zero Defaults' reproducibility policy.\n\n"
                "Correct usage:\n"
                "  from pykt.models.gainakt3_exp import create_exp_model\n"
                "  model = create_exp_model(config)  # All params explicit in config\n\n"
                "Incorrect usage:\n"
                "  model = GainAKT3Exp(num_c=100)  # Uses hidden defaults!\n\n"
                "To suppress this warning, set PYKT_STRICT_REPRODUCIBILITY=0\n"
                "See examples/reproducibility.md for details.\n"
                + "="*80,
                stacklevel=2
            )
        
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
        
        # Incremental mastery loss weight (for dual-prediction architecture)
        self.incremental_mastery_loss_weight = incremental_mastery_loss_weight
        
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
        
        # COMMENTED OUT: Intrinsic Gain Attention feature (DEPRECATED)
        # ════════════════════════════════════════════════════════════════════════
        # This alternative architecture extracted gains directly from attention weights
        # instead of using projection heads. Deactivated in favor of the current
        # "Values as Learning Gains" approach which is more interpretable.
        # ════════════════════════════════════════════════════════════════════════
        # if self.intrinsic_gain_attention:
        #     # Apply non-negativity activation to gains
        #     value_seq = self.gain_activation(value_seq)

        # 2. Add positional encodings
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        context_seq += pos_emb
        
        # COMMENTED OUT: Intrinsic Gain Attention conditional (DEPRECATED)
        # if not self.intrinsic_gain_attention:
        #     # Legacy mode: add positional encoding to value stream
        #     value_seq += pos_emb
        
        # Always add positional encoding to value stream (standard mode)
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
        
        # COMMENTED OUT: Intrinsic Gain Attention prediction mode (DEPRECATED)
        # ════════════════════════════════════════════════════════════════════════
        # Intrinsic mode used a different concatenation (without value_seq)
        # Current standard mode always uses [context, value, skill] concatenation
        # ════════════════════════════════════════════════════════════════════════
        # if self.intrinsic_gain_attention:
        #     # Intrinsic mode: [h_t, concept_embedding] or [h_t, concept_embedding, student_speed]
        #     if self.use_student_speed:
        #         concatenated = torch.cat([context_seq, target_concept_emb, student_emb], dim=-1)
        #     else:
        #         concatenated = torch.cat([context_seq, target_concept_emb], dim=-1)
        # else:
        #     # Legacy mode: [context_seq, value_seq, concept_embedding] or [..., student_speed]
        #     if self.use_student_speed:
        #         concatenated = torch.cat([context_seq, value_seq, target_concept_emb, student_emb], dim=-1)
        #     else:
        #         concatenated = torch.cat([context_seq, value_seq, target_concept_emb], dim=-1)
        
        # Standard mode: always use [context, value, skill] (and optionally student_speed)
        if self.use_student_speed:
            concatenated = torch.cat([context_seq, value_seq, target_concept_emb, student_emb], dim=-1)
        else:
            concatenated = torch.cat([context_seq, value_seq, target_concept_emb], dim=-1)
            
        logits = self.prediction_head(concatenated).squeeze(-1)
        # Defer sigmoid until evaluation to allow using BCEWithLogitsLoss safely under AMP
        predictions = torch.sigmoid(logits)
        
        # 5. Optionally compute interpretability projections
        
        # COMMENTED OUT: Intrinsic Gain Attention gains computation (DEPRECATED)
        # ════════════════════════════════════════════════════════════════════════
        # This alternative approach extracted gains directly from attention weights
        # via get_aggregated_gains() method from the encoder blocks.
        # Deactivated in favor of "Values as Learning Gains" (current standard).
        # ════════════════════════════════════════════════════════════════════════
        # if self.intrinsic_gain_attention:
        #     # Intrinsic mode: retrieve aggregated gains directly from attention mechanism
        #     aggregated_gains = self.get_aggregated_gains()
        #     
        #     if aggregated_gains is not None:
        #         # Apply non-negativity activation to aggregated gains
        #         projected_gains = torch.relu(aggregated_gains)  # Ensure non-negativity
        #         
        #         # Compute cumulative mastery from gains
        #         batch_size, seq_len, num_c = projected_gains.shape
        #         projected_mastery = torch.zeros_like(projected_gains)
        #         projected_mastery[:, 0, :] = torch.clamp(projected_gains[:, 0, :] * 0.1, min=0.0, max=1.0)
        #         for t in range(1, seq_len):
        #             accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
        #             projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
        #     else:
        #         projected_mastery = None
        #         projected_gains = None
        # elif self.use_gain_head and self.use_mastery_head:
        
        # 5. Optionally compute interpretability projections (controlled by parameters)
        if self.use_mastery_head:
            # ============================================================================
            # GainAKT3Exp CORE INNOVATION: Values ARE Learning Gains
            # ============================================================================
            # Conceptual Model: Each interaction's Value output directly represents how
            # much the student learned from that (skill, response) tuple.
            # 
            # Educational Meaning:
            # - Value output = learning gain for this interaction
            # - No intermediate projection needed (Values encode gains directly)
            # - ReLU ensures non-negative learning (no knowledge loss)
            # - Direct mapping provides maximal interpretability
            # ============================================================================
            
            # Values from encoder directly represent learning gains [B, L, D]
            # Each interaction's contribution to knowledge accumulation
            learning_gains_d = torch.relu(value_seq)  # Non-negative learning only
            
            # COMMENTED OUT: Old approach using gain_head projection
            # The projection layer obscured the direct Value→Gain relationship
            # # Project D-dimensional gains to per-skill gains [B, L, num_c]
            # # Use gain_head to extract skill-specific learning gains from D-dimensional representation
            # if hasattr(self, 'gain_head') and self.gain_head is not None:
            #     projected_gains_raw = torch.relu(self.gain_head(learning_gains_d))  # [B, L, num_c]
            # else:
            #     # Fallback: aggregate D-dimensional gains across dimensions
            #     # Then broadcast to num_c (each skill gets same aggregated gain)
            #     aggregated_gains = learning_gains_d.mean(dim=-1, keepdim=True)  # [B, L, 1]
            #     projected_gains_raw = aggregated_gains.expand(-1, -1, self.num_c)  # [B, L, num_c]
            # 
            # # Normalize gains to [0, 1] range using sigmoid
            # # This ensures gains are bounded and interpretable
            # projected_gains = torch.sigmoid(projected_gains_raw)  # [B, L, num_c] in [0, 1]
            
            # NEW APPROACH: Direct Value→Gain mapping
            # For each timestep, the learning gain applies to the skill being practiced
            # We aggregate the D-dimensional gain representation to a scalar per interaction
            aggregated_gains = learning_gains_d.mean(dim=-1, keepdim=True)  # [B, L, 1]
            
            # Normalize to [0, 1] range for bounded learning increments
            # Using sigmoid ensures smooth, bounded gains that don't explode during accumulation
            projected_gains = torch.sigmoid(aggregated_gains).expand(-1, -1, self.num_c)  # [B, L, num_c]
            
            # ============================================================================
            # Recursive Mastery Accumulation: Values → Learning Gains → Mastery Evolution
            # ============================================================================
            # Educational Model:
            # 1. Each interaction with a skill generates a learning gain (from Values)
            # 2. The skill's mastery level increases by this gain
            # 3. Formula: mastery[skill, t] = mastery[skill, t-1] + α × learning_gain[t]
            #    where α = 0.1 (scaling factor to bound increments)
            # 4. Mastery is clamped to [0, 1] (normalized competence scale)
            # 
            # Interpretability Guarantee:
            # - Can trace any skill's mastery evolution across interactions
            # - Can understand: "This interaction contributed X learning gain"
            # - Direct transparency: no hidden projections or transformations
            # ============================================================================
            
            # Initialize all skills at zero mastery (no prior knowledge)
            batch_size, seq_len, _ = projected_gains.shape
            projected_mastery = torch.zeros(batch_size, seq_len, self.num_c, device=q.device)
            
            # Recursive accumulation: accumulate learning gains into skill mastery
            # Only the skill being practiced at each timestep receives the learning gain
            alpha = 0.1  # Scaling factor: limits max gain per interaction to ~0.1
            max_theoretical_mastery = 10.0  # Theoretical maximum before tanh normalization
            
            for t in range(1, seq_len):
                # Copy previous timestep's mastery for all skills
                projected_mastery[:, t, :] = projected_mastery[:, t-1, :].clone()
                
                # Identify which skill is being practiced at timestep t
                practiced_concepts = q[:, t].long()  # [B] - skill index for each student
                batch_indices = torch.arange(batch_size, device=q.device)
                
                # Apply learning gain to the practiced skill's mastery
                # Learning gain comes directly from Values (via projected_gains)
                current_mastery = projected_mastery[batch_indices, t-1, practiced_concepts]
                learning_gain = projected_gains[batch_indices, t, practiced_concepts]
                updated_mastery = current_mastery + alpha * learning_gain
                
                # Apply soft normalization to prevent unbounded growth
                # tanh keeps values bounded while maintaining differentiability
                projected_mastery[batch_indices, t, practiced_concepts] = \
                    max_theoretical_mastery * torch.tanh(updated_mastery / max_theoretical_mastery)
            
            # Final normalization: ensure all mastery values are in [0, 1] for interpretability
            # This provides a normalized competence scale where:
            # - 0.0 = no mastery (beginner)
            # - 1.0 = full mastery (expert)
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
            
            # Compute incremental mastery predictions (differentiable via sigmoid)
            # These are separate from the base predictions and used for interpretability loss
            incremental_mastery_predictions = torch.sigmoid((skill_mastery - skill_threshold) / self.threshold_temperature)
            
            # DEBUG: Log threshold and prediction stats for first batch
            if batch_idx == 0:  # Log for first batch of each epoch
                threshold_range = skill_threshold.max() - skill_threshold.min()
                threshold_std = skill_threshold.std()
                im_pred_range = incremental_mastery_predictions.max() - incremental_mastery_predictions.min()
                im_pred_std = incremental_mastery_predictions.std()
                base_pred_range = predictions.max() - predictions.min()
                base_pred_std = predictions.std()
                
                print("DEBUG GainAKT3Exp - Threshold and dual prediction stats:")
                print(f"  Threshold range: {threshold_range:.4f}, std: {threshold_std:.4f}")
                print(f"  Temperature: {self.threshold_temperature}")
                print(f"  Sample thresholds for first 5 skills: {self.mastery_threshold[:5].cpu().detach().numpy()}")
                print(f"  Base Predictions - range: {base_pred_range:.4f}, std: {base_pred_std:.4f}")
                print(f"  Incremental Mastery Predictions - range: {im_pred_range:.4f}, std: {im_pred_std:.4f}")
                print(f"  Sample base predictions: {predictions[0, :10].cpu().detach().numpy()}")
                print(f"  Sample incremental predictions: {incremental_mastery_predictions[0, :10].cpu().detach().numpy()}")
            
            # Do NOT override base predictions - keep both for dual loss computation
        else:
            # Heads disabled - use base prediction mechanism only
            projected_mastery = None
            projected_gains = None
            incremental_mastery_predictions = None

        # 6. Prepare output with internal states
        output = {
            'predictions': predictions,  # Base predictions from prediction head
            'logits': logits,
            'context_seq': context_seq,
            'value_seq': value_seq
        }
        if projected_mastery is not None:
            output['projected_mastery'] = projected_mastery
        # Only output gains if gain_head is explicitly enabled
        if projected_gains is not None and self.use_gain_head:
            output['projected_gains'] = projected_gains
        # Include incremental mastery predictions if mastery head is enabled
        if incremental_mastery_predictions is not None:
            output['incremental_mastery_predictions'] = incremental_mastery_predictions
        
        # Store D-dimensional gains for interpretability (only if gain_head enabled)
        # Note: Gains are computed internally for mastery accumulation even when use_gain_head=False
        # but only exposed as output when use_gain_head=True
        if self.use_gain_head and self.use_mastery_head and value_seq is not None:
            output['projected_gains_d'] = torch.relu(value_seq)  # Values as learning gains
        
        # 7. Compute interpretability loss (constraint losses on mastery/gains)
        if (projected_mastery is not None) and (projected_gains is not None):
            interpretability_loss = self.compute_interpretability_loss(
                projected_mastery, projected_gains, predictions, q, r
            )
        else:
            interpretability_loss = torch.tensor(0.0, device=q.device)
        output['interpretability_loss'] = interpretability_loss
        
        # 8. Compute incremental mastery loss (separate prediction branch)
        if incremental_mastery_predictions is not None:
            # BCE loss between incremental mastery predictions and ground truth responses
            incremental_mastery_loss = torch.nn.functional.binary_cross_entropy(
                incremental_mastery_predictions, r.float(), reduction='mean'
            )
        else:
            incremental_mastery_loss = torch.tensor(0.0, device=q.device)
        output['incremental_mastery_loss'] = incremental_mastery_loss
        
        # Optional debug: log device placement once (controlled by env var)
        if batch_idx == 0 and bool(int(os.environ.get('PYKT_DEBUG_DP_DEVICES', '0'))):
            try:
                # Only emit from primary device (index 0) to avoid duplicate logs under DataParallel
                if hasattr(q, 'device') and (q.device.index is None or q.device.index == 0):
                    print(f"[DP-DEBUG] forward_with_states: q.device={q.device} context_seq.device={context_seq.device}")
            except Exception:
                pass

        # 9. Call interpretability monitor if enabled and at right frequency.
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
        
        # Include mastery and gain outputs if heads are enabled
        if self.use_mastery_head and 'projected_mastery' in output:
            result['projected_mastery'] = output['projected_mastery']
        # Note: use_gain_head flag kept for backward compatibility
        # Gains come directly from Values (no gain_head projection)
        if self.use_gain_head and 'projected_gains' in output:
            result['projected_gains'] = output['projected_gains']
            
        return result
    
    def compute_interpretability_loss(self, projected_mastery, projected_gains, predictions, questions, responses):
        """
        Compute auxiliary loss based on interpretability constraints.
        
        ⚠️ ARCHITECTURE SIMPLIFICATION (2025-11-15): ALL CONSTRAINT LOSSES COMMENTED OUT
        Only returning 0.0 to maintain interface compatibility.
        Code preserved below for potential future restoration.
        
        Args:
            projected_mastery: [batch_size, seq_len, num_c] - mastery projections
            projected_gains: [batch_size, seq_len, num_c] - gain projections  
            predictions: [batch_size, seq_len] - model predictions
            questions: [batch_size, seq_len] - question IDs
            responses: [batch_size, seq_len] - correct/incorrect responses
            
        Returns:
            torch.Tensor: Auxiliary loss value (currently 0.0 - all constraints commented out)
        """
        # SIMPLIFIED ARCHITECTURE: Return 0.0 (all constraint losses commented out)
        return torch.tensor(0.0, device=projected_mastery.device)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # COMMENTED OUT: All constraint losses (2025-11-15)
        # Code preserved for potential future restoration
        # ═══════════════════════════════════════════════════════════════════════════
        # total_loss = 0.0
        # batch_size, seq_len, num_c = projected_mastery.shape
        #
        # # Create masks for relevant skills
        # skill_masks = torch.zeros((batch_size, seq_len, num_c), device=questions.device).bool()
        # skill_masks.scatter_(2, questions.unsqueeze(-1), 1)
        #
        # # 1. Non-negative learning gains
        # if self.non_negative_loss_weight > 0:
        #     negative_gains = torch.clamp(-projected_gains, min=0)
        #     non_negative_loss = negative_gains.mean()
        #     total_loss += self.non_negative_loss_weight * non_negative_loss
        #
        # # 2. Monotonicity of mastery
        # if self.monotonicity_loss_weight > 0 and seq_len > 1:
        #     mastery_decrease = torch.clamp(projected_mastery[:, :-1] - projected_mastery[:, 1:], min=0)
        #     monotonicity_loss = mastery_decrease.mean()
        #     total_loss += self.monotonicity_loss_weight * monotonicity_loss
        #
        # # 3. Mastery-performance correlation
        # if self.mastery_performance_loss_weight > 0:
        #     relevant_mastery = projected_mastery[skill_masks]
        #     correct_mask = (responses == 1).flatten()
        #     incorrect_mask = (responses == 0).flatten()
        #
        #     # Penalize low mastery for correct answers
        #     low_mastery_on_correct = torch.clamp(1 - relevant_mastery[correct_mask], min=0)
        #     # Penalize high mastery for incorrect answers
        #     high_mastery_on_incorrect = torch.clamp(relevant_mastery[incorrect_mask], min=0)
        #
        #     mastery_performance_loss = low_mastery_on_correct.mean() + high_mastery_on_incorrect.mean()
        #     total_loss += self.mastery_performance_loss_weight * mastery_performance_loss
        #
        # # 4. Gain-performance correlation
        # if self.gain_performance_loss_weight > 0:
        #     relevant_gains = projected_gains[skill_masks]
        #     correct_gains = relevant_gains[(responses == 1).flatten()]
        #     incorrect_gains = relevant_gains[(responses == 0).flatten()]
        #
        #     if correct_gains.numel() > 0 and incorrect_gains.numel() > 0:
        #         # Hinge loss: incorrect gains should be smaller than correct gains
        #         gain_performance_loss = torch.clamp(incorrect_gains.mean() - correct_gains.mean() + 0.1, min=0) # 0.1 margin
        #         total_loss += self.gain_performance_loss_weight * gain_performance_loss
        #
        # # 5. Sparsity of gains
        # if self.sparsity_loss_weight > 0:
        #     non_relevant_gains = projected_gains[~skill_masks]
        #     sparsity_loss = torch.abs(non_relevant_gains).mean()
        #     total_loss += self.sparsity_loss_weight * sparsity_loss
        #
        # # 6. Consistency between mastery increments and gains (architectural scaling factor 0.1)
        # #    Penalize deviation between actual mastery change and scaled gains.
        # if self.consistency_loss_weight > 0 and seq_len > 1:
        #     mastery_delta = projected_mastery[:, 1:, :] - projected_mastery[:, :-1, :]
        #     scaled_gains = projected_gains[:, 1:, :] * 0.1
        #     consistency_residual = torch.abs(mastery_delta - scaled_gains)
        #     consistency_loss = consistency_residual.mean()
        #     total_loss += self.consistency_loss_weight * consistency_loss
        #
        # return total_loss


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
            incremental_mastery_loss_weight=config['incremental_mastery_loss_weight'],
            monitor_frequency=config['monitor_frequency']
        )
    except KeyError as e:
        raise ValueError(f"Missing required parameter in model config: {e}. "
                        f"All parameters must be explicitly provided (no defaults).") from e