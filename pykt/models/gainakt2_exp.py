"""
GainAKT2 model with training-time interpretability monitoring support.

This is an enhanced version of the GainAKT2 model that includes:
1. Additional forward method returning internal states for monitoring
2. Integration hooks for training-time interpretability analysis
3. Support for auxiliary loss functions based on interpretability constraints
"""

import torch
import os
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
                 use_mastery_head=True, use_gain_head=True, intrinsic_gain_attention=False,
                 non_negative_loss_weight=0.1,
                 monotonicity_loss_weight=0.1, mastery_performance_loss_weight=0.1,
                 gain_performance_loss_weight=0.1, sparsity_loss_weight=0.1,
                 consistency_loss_weight=0.1, monitor_frequency=50, 
                 alpha_learning_rate=1.0, use_causal_mastery=False, 
                 num_students=10000, use_learnable_alpha=False):
        """
        Initialize the monitored GainAKT2Exp model.
        
        Args:
            monitor_frequency (int): How often to compute interpretability metrics during training
            intrinsic_gain_attention (bool): If True, use intrinsic gain attention mode
            alpha_learning_rate (float): Learning rate parameter for sigmoid mastery curve (default: 1.0)
                                        [DEPRECATED if use_learnable_alpha=True]
            use_causal_mastery (bool): If True, use causal mastery architecture with skill-specific masking
            num_students (int): Total number of unique students in training set (for learnable alpha)
            use_learnable_alpha (bool): If True, use learnable IRT-inspired alpha parameters
            Other args: Same as base GainAKT2 model
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
            consistency_loss_weight=consistency_loss_weight
        )
        
        self.monitor_frequency = monitor_frequency
        self.interpretability_monitor = None
        self.step_count = 0
        
        # Causal mastery architecture parameters
        self.use_causal_mastery = use_causal_mastery
        self.alpha_learning_rate = alpha_learning_rate
        
        # Learnable alpha parameters (IRT-inspired)
        self.use_learnable_alpha = use_learnable_alpha
        self.num_students = num_students
        
        if self.use_learnable_alpha:
            # Per-skill learning steepness parameter (related to skill difficulty)
            # Lower α_skill → harder skill (shallow learning curve)
            # Higher α_skill → easier skill (steep learning curve)
            self.alpha_skill_raw = torch.nn.Parameter(torch.empty(num_c))
            torch.nn.init.uniform_(self.alpha_skill_raw, 0.0, 1.0)
            
            # Per-student learning speed parameter (related to student ability)
            # Lower α_student → slower learner
            # Higher α_student → faster learner
            self.alpha_student_raw = torch.nn.Parameter(torch.empty(num_students))
            torch.nn.init.uniform_(self.alpha_student_raw, 0.0, 1.0)
            
            # Default alpha for unseen students (test time)
            # Represents "average" student learning speed
            self.alpha_student_default_raw = torch.nn.Parameter(torch.tensor(0.5))
        
    def set_monitor(self, monitor):
        """Set the interpretability monitor hook."""
        self.interpretability_monitor = monitor
    
    def build_skill_causal_mask(self, questions, device):
        """
        Build a skill-specific causal mask for cumulative mastery computation.
        
        This mask enforces two constraints:
        1. Temporal causality: Only past interactions (i <= t) can contribute to mastery at time t
        2. Skill relevance: Only interactions involving skill k can contribute to mastery[k]
        
        Args:
            questions (torch.Tensor): Question IDs [batch_size, seq_len]
            device: Device to create the mask on
            
        Returns:
            torch.Tensor: Boolean mask of shape [batch_size, seq_len, seq_len, num_skills]
                         mask[b,t,i,k] = True if:
                         - interaction i occurred at or before time t (i <= t)
                         - interaction i involved skill k (questions[b,i] == k)
        """
        batch_size, seq_len = questions.shape
        num_c = self.num_c
        
        # 1. Temporal causality: i <= t (lower triangular mask including diagonal)
        # Shape: [seq_len, seq_len]
        temporal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        
        # 2. Skill relevance: questions[b,i] == k
        # Create one-hot encoding: skill_mask[b,i,k] = (questions[b,i] == k)
        # Shape: [batch_size, seq_len, num_skills]
        skill_mask = torch.zeros((batch_size, seq_len, num_c), device=device).bool()
        skill_mask.scatter_(2, questions.unsqueeze(-1), 1)
        
        # 3. Combine both constraints: [B, L, L, C]
        # temporal_mask[t,i]: can time t see time i?
        # skill_mask[b,i,k]: does interaction i involve skill k?
        # Broadcast: [1, L, L, 1] & [B, 1, L, C] -> [B, L, L, C]
        causal_mask = temporal_mask.unsqueeze(0).unsqueeze(-1) & skill_mask.unsqueeze(1)
        
        return causal_mask
    
    def compute_cumulative_gains(self, gains, questions, device):
        """
        Compute cumulative gains per skill using causal masking.
        
        For each skill k at time t, sum all gains from past interactions involving skill k:
        cumulative_gains[t,k] = Σ_{i≤t, q[i]=k} gains[i,k]
        
        Args:
            gains (torch.Tensor): Learning gains [batch_size, seq_len, num_skills]
            questions (torch.Tensor): Question IDs [batch_size, seq_len]
            device: Device for computation
            
        Returns:
            torch.Tensor: Cumulative gains [batch_size, seq_len, num_skills]
        """
        # Build skill-specific causal mask: [B, L, L, C]
        causal_mask = self.build_skill_causal_mask(questions, device)
        
        # Aggregate gains using einsum:
        # causal_mask[b,t,i,c]: can time t use gain from time i for skill c?
        # gains[b,i,c]: gain at time i for skill c
        # Sum over i: cumulative_gains[b,t,c] = Σ_i mask[b,t,i,c] * gains[b,i,c]
        cumulative_gains = torch.einsum('btic,bic->btc', causal_mask.float(), gains)
        
        return cumulative_gains
    
    def apply_learning_curve(self, cumulative_gains, alpha=None, student_ids=None, questions=None):
        """
        Apply sigmoid learning curve transformation to cumulative gains.
        
        If use_learnable_alpha=True and student_ids provided, uses personalized alpha.
        Otherwise, uses global alpha parameter.
        
        Transforms unbounded cumulative gains into bounded mastery [0,1] with S-shaped growth:
        mastery = sigmoid(alpha * cumulative_gains)
        
        Args:
            cumulative_gains (torch.Tensor): Cumulative gains [batch_size, seq_len, num_skills]
            alpha (float, optional): Learning rate parameter controlling curve steepness.
                                    If None, uses self.alpha_learning_rate (ignored if learnable alpha)
            student_ids (torch.Tensor, optional): Student indices [batch_size] for personalized alpha
            questions (torch.Tensor, optional): Question/skill indices [batch_size, seq_len]
                                    
        Returns:
            torch.Tensor: Mastery levels [batch_size, seq_len, num_skills] in range (0, 1)
        """
        if self.use_learnable_alpha and student_ids is not None:
            # Use personalized IRT-inspired alpha
            alpha_combined = self.compute_alpha_combined(student_ids, questions)
            mastery = torch.sigmoid(alpha_combined * cumulative_gains)
        else:
            # Use global alpha (backward compatibility)
            if alpha is None:
                alpha = self.alpha_learning_rate
            mastery = torch.sigmoid(alpha * cumulative_gains)
        
        return mastery
    
    def compute_alpha_combined(self, student_ids, questions):
        """
        Compute personalized alpha for each (student, skill) pair using IRT-inspired formulation.
        
        Combines per-skill difficulty and per-student learning speed:
        α[s,k] = softplus(α_skill_raw[k]) + softplus(α_student_raw[s])
        
        Where:
        - α_skill[k]: Skill-specific learning steepness (inversely related to difficulty)
        - α_student[s]: Student-specific learning speed (related to ability)
        - softplus ensures positivity: softplus(x) = log(1 + exp(x))
        
        Args:
            student_ids (torch.Tensor): Student indices [batch_size]
            questions (torch.Tensor): Question/skill indices [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Personalized alpha values [batch_size, seq_len, num_skills]
        """
        batch_size, seq_len = questions.shape
        
        # Apply softplus to ensure positivity
        alpha_skill = torch.nn.functional.softplus(self.alpha_skill_raw)  # [num_skills]
        alpha_student = torch.nn.functional.softplus(self.alpha_student_raw)  # [num_students]
        alpha_default = torch.nn.functional.softplus(self.alpha_student_default_raw)  # scalar
        
        # Handle unseen students (student_ids >= num_students)
        # This happens during test time with new students
        valid_mask = student_ids < self.num_students
        student_alphas = torch.where(
            valid_mask,
            alpha_student[student_ids],
            alpha_default.expand(batch_size)
        )
        
        # Expand dimensions for broadcasting
        # alpha_skill: [num_skills] → [1, 1, num_skills]
        # student_alphas: [batch_size] → [batch_size, 1, 1]
        alpha_skill_expanded = alpha_skill.unsqueeze(0).unsqueeze(0)
        alpha_student_expanded = student_alphas.unsqueeze(1).unsqueeze(2)
        
        # Combine: [batch_size, 1, 1] + [1, 1, num_c] → [batch_size, 1, num_c]
        # Then expand to [batch_size, seq_len, num_c] for all time steps
        alpha_combined = (alpha_student_expanded + alpha_skill_expanded).expand(batch_size, seq_len, self.num_c)
        
        return alpha_combined
        
    def forward_with_states(self, q, r, qry=None, batch_idx=None, student_ids=None):
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
        
        if self.intrinsic_gain_attention:
            # Intrinsic mode: [h_t, concept_embedding]
            concatenated = torch.cat([context_seq, target_concept_emb], dim=-1)
        else:
            # Legacy mode: [context_seq, value_seq, concept_embedding]
            concatenated = torch.cat([context_seq, value_seq, target_concept_emb], dim=-1)
            
        logits = self.prediction_head(concatenated).squeeze(-1)
        # Defer sigmoid until evaluation to allow using BCEWithLogitsLoss safely under AMP
        predictions = torch.sigmoid(logits)
        
        # 5. Optionally compute interpretability projections
        if self.use_causal_mastery and self.use_gain_head:
            # Causal Mastery Mode: Architecturally enforce cumulative learning principle
            # Step 1: Project gains with sigmoid activation (bounded [0,1])
            projected_gains_raw = self.gain_head(value_seq)
            projected_gains = torch.sigmoid(projected_gains_raw)  # Sigmoid for [0,1] bounds
            
            # Step 2: Compute cumulative gains using skill-specific causal masking
            cumulative_gains = self.compute_cumulative_gains(projected_gains, q, q.device)
            
            # Step 3: Apply sigmoid learning curve transformation (personalized if learnable alpha)
            projected_mastery = self.apply_learning_curve(
                cumulative_gains, 
                alpha=self.alpha_learning_rate,
                student_ids=student_ids,
                questions=q
            )
            
        elif self.intrinsic_gain_attention:
            # Intrinsic mode: retrieve aggregated gains directly from attention mechanism
            aggregated_gains = self.get_aggregated_gains()
            
            if aggregated_gains is not None:
                # Apply non-negativity activation to aggregated gains
                projected_gains = torch.relu(aggregated_gains)  # Ensure non-negativity
                
                # Compute cumulative mastery from gains (recursive approach)
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
            # Baseline (Recursive) Mode: use projection heads with recursive accumulation
            projected_gains_raw = self.gain_head(value_seq)  
            projected_gains = torch.relu(projected_gains_raw)  # enforce non-negativity

            projected_mastery_raw = self.mastery_head(context_seq)
            initial_mastery = torch.sigmoid(projected_mastery_raw)

            batch_size, seq_len, num_c = initial_mastery.shape
            projected_mastery = torch.zeros_like(initial_mastery)
            projected_mastery[:, 0, :] = initial_mastery[:, 0, :]
            for t in range(1, seq_len):
                accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
                projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
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
        
        Args:
            q: Question IDs
            r: Responses
            qry: Query (optional)
            qtest: Test mode flag
            student_ids: Student indices for learnable alpha (optional)
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
        use_mastery_head=config.get('use_mastery_head', True),
        use_gain_head=config.get('use_gain_head', True),
        intrinsic_gain_attention=config.get('intrinsic_gain_attention', False),
        use_causal_mastery=config.get('use_causal_mastery', False),
        alpha_learning_rate=config.get('alpha_learning_rate', 1.0),
        num_students=config.get('num_students', 10000),
        use_learnable_alpha=config.get('use_learnable_alpha', False),
        non_negative_loss_weight=config.get('non_negative_loss_weight', 0.1),
        monotonicity_loss_weight=config.get('monotonicity_loss_weight', 0.1),
        mastery_performance_loss_weight=config.get('mastery_performance_loss_weight', 0.1),
        gain_performance_loss_weight=config.get('gain_performance_loss_weight', 0.1),
        sparsity_loss_weight=config.get('sparsity_loss_weight', 0.1),
        consistency_loss_weight=config.get('consistency_loss_weight', 0.1),
        monitor_frequency=config.get('monitor_frequency', 50)
    )