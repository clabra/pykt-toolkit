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
import torch.nn as nn
import torch.nn.functional as F
import os
from .gainakt3 import EncoderBlock


class GainAKT3Exp(nn.Module):
    """
    Enhanced GainAKT3 model with DUAL-ENCODER architecture and interpretability monitoring.
    
    **DUAL-ENCODER ARCHITECTURE (2025-11-16)**:
    =============================================
    
    This model implements TWO completely independent encoder stacks:
    
    1. **Encoder 1 (Performance Path)**:
       - Independent embedding tables (context_emb1, value_emb1, skill_emb1)
       - Independent encoder blocks (encoder_blocks_1)
       - Attention mechanism learns Q, K, V for detecting **response patterns**
       - Optimized for prediction accuracy
       - Outputs → Base Predictions → BCE Loss (weight ≈ 1.0)
    
    2. **Encoder 2 (Interpretability Path)**:
       - Independent embedding tables (context_emb2, value_emb2)
       - Independent encoder blocks (encoder_blocks_2)
       - Attention mechanism learns Q, K, V for detecting **learning gains patterns**
       - Optimized for interpretable mastery trajectories
       - Outputs → Sigmoid Learning Curves → Incremental Mastery Predictions → IM Loss (weight = 0.1)
    
    **Key Features**:
    - Complete parameter independence between encoders
    - Each encoder learns different attention patterns
    - No shared representations between pathways
    - Clean separation of performance vs interpretability objectives
    
    **Benefits**:
    - Encoder 1: Pure prediction focus without interpretability constraints
    - Encoder 2: Pure interpretability focus without prediction pressure
    - Dual losses enable performance/interpretability trade-off tuning
    - More parameters allow richer representations for both objectives
    """
    
    def __init__(self, num_c, seq_len=200, d_model=128, n_heads=8, num_encoder_blocks=2, 
                 d_ff=256, dropout=0.1, emb_type="qid", emb_path="", pretrain_dim=768,
                 intrinsic_gain_attention=False,
                 non_negative_loss_weight=0.1,
                 monotonicity_loss_weight=0.1, mastery_performance_loss_weight=0.1,
                 gain_performance_loss_weight=0.1, sparsity_loss_weight=0.1,
                 consistency_loss_weight=0.1, incremental_mastery_loss_weight=0.1,
                 variance_loss_weight=0.1,
                 monitor_frequency=50, use_skill_difficulty=False,
                 use_student_speed=False, num_students=None, mastery_threshold_init=0.6,
                 threshold_temperature=1.5, beta_skill_init=2.0, m_sat_init=0.8, 
                 gamma_student_init=1.0, sigmoid_offset=2.0):
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
            mastery_threshold_init (float): Initial learnable mastery threshold (default: 0.6, DO NOT RELY ON THIS)
            threshold_temperature (float): Temperature for sigmoid threshold function (default: 1.5, DO NOT RELY ON THIS)
            beta_skill_init (float): Initial β_skill for learning rate amplification (default: 2.0, DO NOT RELY ON THIS)
            m_sat_init (float): Initial M_sat for mastery saturation level (default: 0.8, DO NOT RELY ON THIS)
            gamma_student_init (float): Initial γ_student for learning velocity (default: 1.0, DO NOT RELY ON THIS)
            sigmoid_offset (float): Sigmoid inflection point offset (default: 2.0, DO NOT RELY ON THIS)
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
        
        # ═══════════════════════════════════════════════════════════════════════════
        # DUAL-ENCODER ARCHITECTURE INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════════
        # Instead of inheriting from GainAKT3, we create TWO independent encoder stacks
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Store model configuration
        super().__init__()
        self.model_name = "gainakt3exp"
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.d_ff = d_ff
        self.dropout = dropout
        self.emb_type = emb_type
        self.intrinsic_gain_attention = intrinsic_gain_attention
        self.use_skill_difficulty = use_skill_difficulty
        self.use_student_speed = use_student_speed
        self.num_students = num_students
        
        # Loss weights (constraint losses currently commented out = 0.0)
        self.non_negative_loss_weight = non_negative_loss_weight
        self.monotonicity_loss_weight = monotonicity_loss_weight
        self.mastery_performance_loss_weight = mastery_performance_loss_weight
        self.gain_performance_loss_weight = gain_performance_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        
        # Monitoring
        self.monitor_frequency = monitor_frequency
        self.interpretability_monitor = None
        self.step_count = 0
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ENCODER 1: PERFORMANCE PATH (Response Patterns)
        # ═══════════════════════════════════════════════════════════════════════════
        # This encoder learns Q, K, V optimized for detecting response correctness patterns
        # Independent parameters ensure it focuses purely on prediction accuracy
        # ═══════════════════════════════════════════════════════════════════════════
        
        if emb_type.startswith("qid"):
            # Encoder 1 embeddings: context, value, skill (for prediction)
            self.context_embedding_1 = nn.Embedding(num_c * 2, d_model)
            self.value_embedding_1 = nn.Embedding(num_c * 2, d_model)
            self.skill_embedding_1 = nn.Embedding(num_c, d_model)  # For target concepts
            
        # Encoder 1 positional embeddings
        self.pos_embedding_1 = nn.Embedding(seq_len, d_model)
        
        # Encoder 1 transformer blocks (learns response patterns)
        self.encoder_blocks_1 = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout, 
                        intrinsic_gain_attention=False,  # Standard attention
                        num_skills=None)
            for _ in range(num_encoder_blocks)
        ])
        
        # Encoder 1 prediction head: [context1, value1, skill1] → prediction
        self.prediction_head_1 = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ENCODER 2: INTERPRETABILITY PATH (Learning Gains Patterns)
        # ═══════════════════════════════════════════════════════════════════════════
        # This encoder learns Q, K, V optimized for detecting learning gains patterns
        # Independent parameters ensure it focuses purely on mastery trajectories
        # ═══════════════════════════════════════════════════════════════════════════
        
        if emb_type.startswith("qid"):
            # Encoder 2 embeddings: context, value (for mastery)
            # No separate skill embedding needed for mastery computation
            self.context_embedding_2 = nn.Embedding(num_c * 2, d_model)
            self.value_embedding_2 = nn.Embedding(num_c * 2, d_model)
        
        # Encoder 2 positional embeddings
        self.pos_embedding_2 = nn.Embedding(seq_len, d_model)
        
        # Encoder 2 transformer blocks (learns learning gains patterns)
        self.encoder_blocks_2 = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout,
                        intrinsic_gain_attention=False,  # Standard attention
                        num_skills=None)
            for _ in range(num_encoder_blocks)
        ])
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PER-SKILL GAINS PROJECTION (FIX: 2025-11-17)
        # ═══════════════════════════════════════════════════════════════════════════
        # Project Encoder 2 value representations to per-skill gain estimates
        # This enables skill-specific learning: different skills can have different gains
        # per interaction, allowing the model to learn question-skill associations
        # ═══════════════════════════════════════════════════════════════════════════
        self.gains_projection = nn.Linear(d_model, num_c)
        
        # Encoder 2 does NOT have a prediction head
        # Its outputs feed directly into sigmoid learning curve computation
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SIGMOID LEARNING CURVE PARAMETERS (2025-11-16 Architecture Update)
        # ═══════════════════════════════════════════════════════════════════════════
        # Mastery evolves via practice count-driven sigmoid curves:
        # mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × practice_count[i,s,t] - offset)
        # 
        # Educational Interpretation:
        # - β_skill[s]: Skill difficulty (controls learning curve steepness)
        # - γ_student[i]: Student learning velocity (modulates progression speed)
        # - M_sat[s]: Saturation level (maximum achievable mastery per skill)
        # - θ_global: Global threshold (mastery level for correct performance)
        # - offset: Inflection point (controls when rapid learning begins)
        # 
        # Three Automatic Learning Phases:
        # 1. Initial Phase (practice_count ≈ 0): Mastery ≈ 0 (warm-up)
        # 2. Growth Phase (intermediate): Rapid mastery increase (effective learning)
        # 3. Saturation Phase (high practice_count): Mastery → M_sat[s] (consolidation)
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Per-skill learnable parameters (shared across students)
        # FIX (2025-11-16): Now configurable via beta_skill_init parameter
        # With β=1, effective_practice of 1-5 gives weak sigmoid response
        # With β=2, practice is amplified 2x, providing better dynamic range
        self.beta_skill = torch.nn.Parameter(torch.ones(num_c) * beta_skill_init)  # Skill difficulty (curve steepness)
        self.M_sat = torch.nn.Parameter(torch.ones(num_c) * m_sat_init)  # Saturation level (max mastery)
        
        # Per-student learnable parameters (if num_students provided)
        # Note: These parameters affect INTERPRETABILITY (mastery trajectories), NOT predictions
        # γ_student controls learning velocity: how fast each student's mastery grows with practice
        # Used in sigmoid learning curve: mastery = M_sat × sigmoid(β × γ × practice - offset)
        if num_students is not None and num_students > 0:
            self.gamma_student = torch.nn.Parameter(torch.ones(num_students) * gamma_student_init)  # Learning velocity
            self.has_fixed_student_params = True
        else:
            # Dynamic mode: will be handled per-batch
            self.gamma_student = None
            self.has_fixed_student_params = False
        
        # Global learnable parameters
        self.theta_global = torch.nn.Parameter(torch.tensor(mastery_threshold_init))  # Performance threshold
        # FIX (2025-11-16): Now configurable via sigmoid_offset parameter
        # Formula: mastery = M_sat × sigmoid(β × γ × practice - offset)
        # With β=2, offset=2, practice count of 1 → sigmoid_input = 2×1 - 2 = 0 → sigmoid = 0.5 → mastery = 0.4
        # With β=2, offset=2, practice count of 2 → sigmoid_input = 2×2 - 2 = 2 → sigmoid = 0.88 → mastery = 0.70
        # With β=2, offset=2, practice count of 5 → sigmoid_input = 2×5 - 2 = 8 → sigmoid = 1.0 → mastery = 0.8
        # This provides good dynamic range for typical practice counts (1-10)
        self.offset = torch.nn.Parameter(torch.tensor(sigmoid_offset))  # Sigmoid inflection point
        
        # Config parameter (hybrid approach - can upgrade to learnable later)
        self.threshold_temperature = threshold_temperature
        
        # Incremental mastery loss weight (for dual-prediction architecture)
        self.incremental_mastery_loss_weight = incremental_mastery_loss_weight
        
        # Variance loss weight (V2: encourage skill differentiation in gains)
        self.variance_loss_weight = variance_loss_weight
        
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

        # ═══════════════════════════════════════════════════════════════════════════
        # ENCODER 1: PERFORMANCE PATH (Response Patterns)
        # ═══════════════════════════════════════════════════════════════════════════
        # Process inputs through Encoder 1 which learns attention patterns for prediction
        # ═══════════════════════════════════════════════════════════════════════════
        
        # 1.1. Get embeddings for Encoder 1 (context and value streams)
        context_seq_1 = self.context_embedding_1(interaction_tokens)
        value_seq_1 = self.value_embedding_1(interaction_tokens)
        
        # 1.2. Add positional encodings for Encoder 1
        positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb_1 = self.pos_embedding_1(positions)
        context_seq_1 += pos_emb_1
        value_seq_1 += pos_emb_1
        
        # 1.3. Pass through Encoder 1 blocks (learns response patterns)
        for block in self.encoder_blocks_1:
            context_seq_1, value_seq_1 = block(context_seq_1, value_seq_1, mask)
        
        # 1.4. Generate base predictions from Encoder 1 outputs
        target_concept_emb_1 = self.skill_embedding_1(target_concepts)
        concatenated_1 = torch.cat([context_seq_1, value_seq_1, target_concept_emb_1], dim=-1)
        logits = self.prediction_head_1(concatenated_1).squeeze(-1)
        predictions = torch.sigmoid(logits)  # Base predictions for BCE loss
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ENCODER 2: INTERPRETABILITY PATH (Learning Gains Patterns)
        # ═══════════════════════════════════════════════════════════════════════════
        # Process inputs through Encoder 2 which learns attention patterns for mastery
        # ═══════════════════════════════════════════════════════════════════════════
        
        # 2.1. Get embeddings for Encoder 2 (context and value streams)
        context_seq_2 = self.context_embedding_2(interaction_tokens)
        value_seq_2 = self.value_embedding_2(interaction_tokens)
        
        # 2.2. Add positional encodings for Encoder 2
        pos_emb_2 = self.pos_embedding_2(positions)
        context_seq_2 += pos_emb_2
        value_seq_2 += pos_emb_2
        
        # 2.3. Pass through Encoder 2 blocks (learns learning gains patterns)
        for block in self.encoder_blocks_2:
            context_seq_2, value_seq_2 = block(context_seq_2, value_seq_2, mask)
        
        # 2.4. Use Encoder 2 outputs for sigmoid learning curves
        # Value stream from Encoder 2 represents learning gains for mastery computation
        # (will be used for sigmoid curves below)
        
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
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SIGMOID LEARNING CURVE COMPUTATION FROM ENCODER 2 OUTPUTS
        # ═══════════════════════════════════════════════════════════════════════════
        # Use Encoder 2 outputs (Interpretability Path) for mastery trajectories
        # ═══════════════════════════════════════════════════════════════════════════
        # GainAKT3Exp always computes sigmoid mastery trajectories (core feature)
        # ============================================================================
        # GainAKT3Exp CORE INNOVATION: Values from Encoder 2 ARE Learning Gains
            # ============================================================================
            # Conceptual Model: Encoder 2's Value output directly represents how
            # much the student learned from that (skill, response) tuple.
            # 
            # Encoder 2 learns Q, K, V specifically for detecting learning gains patterns
            # This encoder is optimized for interpretability (mastery trajectories)
            # 
            # Educational Meaning:
            # - Value output from Encoder 2 = learning gain for this interaction
            # - No intermediate projection needed (Values encode gains directly)
            # - ReLU ensures non-negative learning (no knowledge loss)
            # - Direct mapping provides maximal interpretability
            # ============================================================================
            
            # Values from Encoder 2 directly represent learning gains [B, L, D]
            # Each interaction's contribution to knowledge accumulation
            learning_gains_d = torch.relu(value_seq_2)  # Non-negative learning only
            
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
            # ═══════════════════════════════════════════════════════════════════════════
            # SIGMOID LEARNING CURVE MASTERY (2025-11-16 Architecture Update)
            # ═══════════════════════════════════════════════════════════════════════════
            # Educational Model: Practice count-driven sigmoid learning curves
            # 
            # Formula: mastery[i,s,t] = M_sat[s] × sigmoid(β_skill[s] × γ_student[i] × practice_count[i,s,t] - offset)
            # 
            # Where:
            # - practice_count[i,s,t] = number of times student i practiced skill s up to timestep t
            # - β_skill[s]: Skill difficulty (controls learning curve steepness)
            # - γ_student[i]: Student learning velocity (modulates progression speed)
            # - M_sat[s]: Saturation mastery level (maximum achievable per skill)
            # - offset: Inflection point (controls when rapid learning begins)
            # 
            # Three Automatic Learning Phases:
            # 1. Initial Phase (practice_count ≈ 0): Mastery ≈ 0 (warm-up/familiarization)
            # 2. Growth Phase (intermediate): Rapid mastery increase (effective learning)
            #    Slope = (β_skill × γ_student × M_sat) / 4 at inflection point
            # 3. Saturation Phase (high practice_count): Mastery → M_sat[s] (consolidation)
            # 
            # Interpretability Guarantee:
            # - Direct transparency: mastery depends ONLY on practice count (observable)
            # - Can predict future mastery: "After N more practices, mastery will be X"
            # - Can explain current mastery: "Student has practiced this skill N times"
            # - No hidden accumulation or opaque transformations
            # ═══════════════════════════════════════════════════════════════════════════
            
            batch_size, seq_len, _ = projected_gains.shape
            
            # Step 1: Compute per-skill gains from Encoder 2 (FIX: 2025-11-17)
            # ═══════════════════════════════════════════════════════════════════════════
            # OLD (BROKEN): Scalar gain_quality [B, L, 1] - same increment for ALL skills
            # NEW (FIXED): Per-skill gains [B, L, num_c] - different increment per skill
            # 
            # This enables Encoder 2 to learn:
            # - Which skills are relevant for each question (Q-matrix learning)
            # - How much each interaction improves each specific skill
            # - Skill-specific learning rates (some skills harder than others)
            # ═══════════════════════════════════════════════════════════════════════════
            # Project value_seq_2 to skill-space and normalize to [0, 1]
            skill_gains = torch.sigmoid(self.gains_projection(value_seq_2))  # [B, L, num_c]
            
            # V2 (2025-11-17): Compute variance loss to encourage skill differentiation
            # Maximize variance across skills per interaction to prevent uniform gains
            # variance_loss is negative (we want to minimize -variance = maximize variance)
            gain_variance_per_interaction = skill_gains.var(dim=-1, keepdim=False)  # [B, L]
            variance_loss = -gain_variance_per_interaction.mean()  # Scalar loss (negative = encourage high variance)
            
            # Step 2: Accumulate per-skill effective practice (quality-weighted)
            # Each skill accumulates its own gain at each timestep
            # This replaces the scalar gain_quality that applied uniformly to all skills
            # ═══════════════════════════════════════════════════════════════════════════
            effective_practice = torch.zeros(batch_size, seq_len, self.num_c, device=q.device)
            
            for t in range(seq_len):
                if t > 0:
                    # Carry forward previous effective practice for all skills
                    effective_practice[:, t, :] = effective_practice[:, t-1, :].clone()
                
                # Add per-skill gains - each skill gets its own increment
                # This is fully differentiable: gradients flow back through
                # skill_gains → gains_projection → value_seq_2 → Encoder 2
                effective_practice[:, t, :] += skill_gains[:, t, :]
            
            # Step 3: Compute sigmoid learning curve mastery using effective practice
            # Handle gamma_student (fixed vs dynamic per-batch)
            if self.has_fixed_student_params:
                if student_ids is not None:
                    # Use per-student gamma values indexed by student IDs
                    gamma = self.gamma_student[student_ids]  # [batch_size]
                else:
                    # Fallback: use mean gamma if student IDs not provided (backward compatibility)
                    gamma = self.gamma_student.mean().unsqueeze(0).expand(batch_size)  # [batch_size]
            else:
                # Dynamic mode: learn gamma per batch (not per-student identity)
                # Use uniform gamma=1.0 for all students in this batch
                gamma = torch.ones(batch_size, device=q.device)  # [batch_size]
            
            # Expand dimensions for broadcasting:
            # beta_skill: [num_c] → [1, 1, num_c]
            # gamma: [batch_size] → [batch_size, 1, 1]
            # M_sat: [num_c] → [1, 1, num_c]
            # offset: scalar
            # practice_count: [batch_size, seq_len, num_c]
            
            beta_expanded = self.beta_skill.unsqueeze(0).unsqueeze(0)  # [1, 1, num_c]
            gamma_expanded = gamma.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            M_sat_expanded = self.M_sat.unsqueeze(0).unsqueeze(0)  # [1, 1, num_c]
            
            # Compute sigmoid input: β_skill[s] × γ_student[i] × effective_practice[i,s,t] - offset
            # effective_practice is differentiable through Encoder 2!
            sigmoid_input = (beta_expanded * gamma_expanded * effective_practice) - self.offset
            
            # Compute mastery: M_sat[s] × sigmoid(...)
            projected_mastery = M_sat_expanded * torch.sigmoid(sigmoid_input)  # [batch_size, seq_len, num_c]
            
            # Clamp to [0, 1] for numerical stability (sigmoid output should already be in this range)
            projected_mastery = torch.clamp(projected_mastery, min=0.0, max=1.0)
            
            # DEBUG: Log mastery accumulation statistics for first batch
            if batch_idx == 0:  # Log for first batch of each epoch
                mastery_range = projected_mastery.max() - projected_mastery.min()
                mastery_std = projected_mastery.std()
                avg_effective_practice = effective_practice.sum() / (batch_size * seq_len * self.num_c)
                
                print("DEBUG GainAKT3Exp - Dual-Encoder Mastery Stats:")
                print(f"  Mastery range: {mastery_range:.4f}, std: {mastery_std:.4f}")
                print(f"  Avg effective practice: {avg_effective_practice:.4f}")
                print(f"  Avg per-skill gains (Encoder 2 output): {skill_gains.mean():.4f}")
                print(f"  Skill gains std (differentiation): {skill_gains.std():.4f}")
                print("  Sample mastery progression for first student, first 5 skills:")
                for skill in range(min(5, projected_mastery.shape[2])):
                    skill_mastery = projected_mastery[0, :, skill]
                    practiced_timesteps = (q[0] == skill).nonzero(as_tuple=True)[0]
                    if len(practiced_timesteps) > 0:
                        print(f"    Skill {skill}: {skill_mastery[practiced_timesteps].cpu().detach().numpy()}")
            
            # ═══════════════════════════════════════════════════════════════════════════
            # GLOBAL THRESHOLD PREDICTION (2025-11-16 Architecture Update)
            # ═══════════════════════════════════════════════════════════════════════════
            # Generate predictions using global learnable threshold θ_global
            # 
            # Formula: incremental_mastery_predictions = sigmoid((mastery - θ_global) / temperature)
            # 
            # If mastery >= θ_global: prediction approaches 1.0 (correct)
            # If mastery < θ_global: prediction approaches 0.0 (incorrect)
            # 
            # Global threshold simplifies the model:
            # - Single learnable parameter θ_global instead of per-skill thresholds
            # - Same performance criterion applies across all skills
            # - Reduces parameter count while maintaining interpretability
            # ═══════════════════════════════════════════════════════════════════════════
            
            # Get the skill ID for each timestep
            # FIX (2025-11-17): Use CURRENT skill (q) not NEXT skill (target_concepts)
            # Investigation showed 98.2% of prediction mismatches occurred at skill changes
            # because model was using mastery for q[t+1] instead of q[t]
            skill_indices = q.long()  # [B, L] - use current question's skill
            
            # Gather mastery for the actual skills being tested
            # projected_mastery: [B, L, num_c], we need [B, L] by selecting skill at each step
            batch_indices = torch.arange(batch_size, device=q.device).unsqueeze(1).expand(-1, seq_len)
            time_indices = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
            skill_mastery = projected_mastery[batch_indices, time_indices, skill_indices]  # [B, L]
            
            # Use global threshold (clamped to [0,1] for stability)
            theta_clamped = torch.clamp(self.theta_global, 0.0, 1.0)
            
            # Compute incremental mastery predictions (differentiable via sigmoid)
            # These are separate from the base predictions and used for interpretability loss
            incremental_mastery_predictions = torch.sigmoid((skill_mastery - theta_clamped) / self.threshold_temperature)
            
            # DEBUG: Comprehensive diagnostics for first batch of each epoch
            if batch_idx == 0:  # Log for first batch of each epoch
                import numpy as np
                
                # Basic statistics
                im_pred_range = incremental_mastery_predictions.max() - incremental_mastery_predictions.min()
                im_pred_std = incremental_mastery_predictions.std()
                base_pred_range = predictions.max() - predictions.min()
                base_pred_std = predictions.std()
                
                # Mastery value statistics (CRITICAL DIAGNOSTIC)
                mastery_min = skill_mastery.min().item()
                mastery_max = skill_mastery.max().item()
                mastery_mean = skill_mastery.mean().item()
                mastery_std = skill_mastery.std().item()
                
                # Mastery-performance correlation (CRITICAL DIAGNOSTIC)
                # Flatten and filter valid positions (use r which is the response tensor)
                valid_mask = (r >= 0) & (r <= 1)
                valid_mastery = skill_mastery[valid_mask].cpu().detach().numpy()
                valid_responses = r[valid_mask].cpu().detach().numpy()
                
                if len(valid_mastery) > 1:
                    mastery_perf_corr = np.corrcoef(valid_mastery, valid_responses)[0, 1]
                    # Also check correlation between IM predictions and responses
                    valid_im_preds = incremental_mastery_predictions[valid_mask].cpu().detach().numpy()
                    im_pred_perf_corr = np.corrcoef(valid_im_preds, valid_responses)[0, 1]
                else:
                    mastery_perf_corr = 0.0
                    im_pred_perf_corr = 0.0
                
                # Theta gradient check (is it learning?)
                theta_grad = self.theta_global.grad
                theta_has_grad = theta_grad is not None
                theta_grad_norm = theta_grad.abs().item() if theta_has_grad else 0.0
                
                print("\n" + "="*80)
                print("DEBUG GainAKT3Exp - COMPREHENSIVE DIAGNOSTICS (batch_idx=0)")
                print("="*80)
                
                print("\n[1] MASTERY VALUE DISTRIBUTION (CRITICAL):")
                print(f"  Min:  {mastery_min:.4f}")
                print(f"  Max:  {mastery_max:.4f}")
                print(f"  Mean: {mastery_mean:.4f}")
                print(f"  Std:  {mastery_std:.4f}")
                print(f"  ⚠️  Expected: min≈0, max≈1, std>0.2 for good variance")
                if mastery_std < 0.1:
                    print(f"  ❌ LOW VARIANCE! Mastery values too compressed (std={mastery_std:.4f})")
                if mastery_mean < 0.1 or mastery_mean > 0.9:
                    print(f"  ❌ EXTREME MEAN! Mastery values biased (mean={mastery_mean:.4f})")
                
                print("\n[2] MASTERY-PERFORMANCE CORRELATION (CRITICAL):")
                print(f"  Mastery ↔ Response:     {mastery_perf_corr:.4f}")
                print(f"  IM_Prediction ↔ Response: {im_pred_perf_corr:.4f}")
                print(f"  ⚠️  Expected: >0.4 for predictive mastery, >0.5 ideal")
                if abs(mastery_perf_corr) < 0.2:
                    print(f"  ❌ MASTERY NOT PREDICTIVE! Correlation too low ({mastery_perf_corr:.4f})")
                
                print("\n[3] THETA GRADIENT FLOW:")
                print(f"  θ_global value:    {theta_clamped:.4f}")
                print(f"  Has gradient:      {theta_has_grad}")
                print(f"  Gradient magnitude: {theta_grad_norm:.6f}")
                if not theta_has_grad:
                    print(f"  ❌ NO GRADIENT! Theta not learning")
                elif theta_grad_norm < 1e-6:
                    print(f"  ⚠️  VANISHING GRADIENT! ({theta_grad_norm:.6f})")
                
                print("\n[4] PREDICTION DISTRIBUTIONS:")
                print(f"  Base (Encoder1):   range={base_pred_range:.4f}, std={base_pred_std:.4f}")
                print(f"  IM (Encoder2):     range={im_pred_range:.4f}, std={im_pred_std:.4f}")
                print(f"  ⚠️  IM range should be >0.3 for useful predictions")
                if im_pred_range < 0.2:
                    print(f"  ❌ COMPRESSED PREDICTIONS! IM range too narrow ({im_pred_range:.4f})")
                
                print("\n[5] TEMPERATURE & THRESHOLD:")
                print(f"  Temperature:  {self.threshold_temperature}")
                print(f"  θ_global:     {theta_clamped:.4f}")
                print(f"  Formula: sigmoid((mastery - {theta_clamped:.4f}) / {self.threshold_temperature})")
                
                print("\n[6] SAMPLE VALUES (first student, first 10 steps):")
                print(f"  Mastery:      {skill_mastery[0, :10].cpu().detach().numpy()}")
                print(f"  Base preds:   {predictions[0, :10].cpu().detach().numpy()}")
                print(f"  IM preds:     {incremental_mastery_predictions[0, :10].cpu().detach().numpy()}")
                print(f"  True labels:  {r[0, :10].cpu().detach().numpy()}")
                
                print("\n[7] LEARNABLE PARAMETERS (first 5):")
                print(f"  β_skill:   {self.beta_skill[:5].cpu().detach().numpy()}")
                print(f"  M_sat:     {self.M_sat[:5].cpu().detach().numpy()}")
                if self.has_fixed_student_params:
                    print(f"  γ_student: {self.gamma_student[:5].cpu().detach().numpy()}")
                else:
                    print(f"  γ_student: Dynamic (batch mean={gamma.mean():.4f})")
                print(f"  offset:    {self.offset.item():.4f}")
                
                print("="*80 + "\n")
            
            # Do NOT override base predictions - keep both for dual loss computation
        else:
            # Heads disabled - use base prediction mechanism only
            projected_mastery = None
            projected_gains = None
            incremental_mastery_predictions = None
            skill_gains = None  # FIX (2025-11-17): Initialize for output dict
            variance_loss = None  # V2 (2025-11-17): Initialize for output dict

        # 6. Prepare output with internal states
        # Return Encoder 2 outputs for interpretability monitoring (context_seq_2, value_seq_2)
        # Base predictions come from Encoder 1
        output = {
            'predictions': predictions,  # Base predictions from Encoder 1 → BCE Loss
            'logits': logits,  # Logits from Encoder 1
            'context_seq': context_seq_2,  # Encoder 2 context (for monitoring)
            'value_seq': value_seq_2  # Encoder 2 value (learning gains)
        }
        if projected_mastery is not None:
            output['projected_mastery'] = projected_mastery
        # Only output gains if gain_head is explicitly enabled
        if projected_gains is not None:
            output['projected_gains'] = projected_gains
        # Include incremental mastery predictions (Encoder 2)
        if incremental_mastery_predictions is not None:
            output['incremental_mastery_predictions'] = incremental_mastery_predictions
        
        # FIX (2025-11-17): Add per-skill gains to output for trajectory validation
        # skill_gains [B, L, num_c] provides per-skill gain estimates for interpretability
        if skill_gains is not None:
            output['skill_gains'] = skill_gains  # [B, L, num_c] per-skill gains
        
        # V2 (2025-11-17): Add variance loss to output for training script to combine with other losses
        if variance_loss is not None:
            output['variance_loss'] = variance_loss  # Scalar loss encouraging skill differentiation
        
        # Store D-dimensional gains for interpretability
        # Gains are always computed (core feature of GainAKT3Exp)
        if value_seq_2 is not None:
            output['projected_gains_d'] = torch.relu(value_seq_2)  # Values as learning gains
        
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
                    print(f"[DP-DEBUG] forward_with_states: q.device={q.device} context_seq_2.device={context_seq_2.device}")
            except Exception:
                pass

        # 9. Call interpretability monitor if enabled and at right frequency.
        # Guard to execute only on primary replica (device index 0) to prevent duplicate side-effects under DataParallel.
        # DUAL-ENCODER MONITORING (2025-11-16): Monitor receives outputs from both encoders
        primary_device = (hasattr(q, 'device') and (q.device.index is None or q.device.index == 0))
        if (self.interpretability_monitor is not None and 
            batch_idx is not None and 
            batch_idx % self.monitor_frequency == 0 and primary_device):
            with torch.no_grad():
                self.interpretability_monitor(
                    batch_idx=batch_idx,
                    # Encoder 1 (Performance Path) outputs
                    context_seq_1=context_seq_1,
                    value_seq_1=value_seq_1,
                    base_predictions=predictions,  # From Encoder 1
                    # Encoder 2 (Interpretability Path) outputs
                    context_seq_2=context_seq_2,
                    value_seq_2=value_seq_2,  # Learning gains representation
                    projected_mastery=projected_mastery,
                    projected_gains=projected_gains,
                    incremental_mastery_predictions=incremental_mastery_predictions,  # From Encoder 2
                    # Common inputs
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
        
        # Include mastery and gain outputs (always computed in GainAKT3Exp)
        if 'projected_mastery' in output:
            result['projected_mastery'] = output['projected_mastery']
        if 'projected_gains' in output:
            result['projected_gains'] = output['projected_gains']
        
        # Include incremental mastery predictions (Encoder 2) for dual-encoder evaluation
        if 'incremental_mastery_predictions' in output:
            result['incremental_mastery_predictions'] = output['incremental_mastery_predictions']
            
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
    
    Required config parameters:
        - num_c (int): Number of unique skills/concepts
        - seq_len (int): Maximum sequence length
        - d_model (int): Model embedding dimension
        - n_heads (int): Number of attention heads
        - num_encoder_blocks (int): Number of transformer blocks
        - d_ff (int): Feed-forward dimension
        - dropout (float): Dropout rate
        - emb_type (str): Embedding type ("qid" or other)
        - use_mastery_head (bool): Enable mastery projection head
        - use_gain_head (bool): Enable gain projection head
        - intrinsic_gain_attention (bool): Use intrinsic gain attention mode
        - use_skill_difficulty (bool): Enable per-skill difficulty embeddings
        - use_student_speed (bool): Enable per-student learning speed embeddings
        - num_students (int): Number of unique students (or None for dynamic)
        - non_negative_loss_weight (float): Weight for non-negative constraint
        - monotonicity_loss_weight (float): Weight for monotonicity constraint
        - mastery_performance_loss_weight (float): Weight for mastery-performance alignment
        - gain_performance_loss_weight (float): Weight for gain-performance alignment
        - sparsity_loss_weight (float): Weight for sparsity constraint
        - consistency_loss_weight (float): Weight for consistency constraint
        - incremental_mastery_loss_weight (float): Weight for incremental mastery loss
        - monitor_frequency (int): How often to compute interpretability metrics
        - mastery_threshold_init (float): Initial value for global threshold θ_global
        - threshold_temperature (float): Temperature for sigmoid threshold function
        - beta_skill_init (float): Initial β_skill for learning rate amplification
        - m_sat_init (float): Initial M_sat for mastery saturation level
        - gamma_student_init (float): Initial γ_student for learning velocity
        - sigmoid_offset (float): Sigmoid inflection point offset
    
    Args:
        config (dict): Model configuration parameters (all required)
        
    Returns:
        GainAKT3Exp: Configured model instance with sigmoid learning curves
        
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
            monitor_frequency=config['monitor_frequency'],
            mastery_threshold_init=config['mastery_threshold_init'],
            threshold_temperature=config['threshold_temperature'],
            beta_skill_init=config['beta_skill_init'],
            m_sat_init=config['m_sat_init'],
            gamma_student_init=config['gamma_student_init'],
            sigmoid_offset=config['sigmoid_offset']
        )
    except KeyError as e:
        raise ValueError(f"Missing required parameter in model config: {e}. "
                        f"All parameters must be explicitly provided (no defaults).") from e