"""
IRT (Item Response Theory) Reference Model

Implements Rasch IRT model as reference for construct validity in iKT3.

The Rasch model: P(correct) = σ(θ - β)
- θ (theta): Student ability
- β (beta): Skill difficulty
- M = σ(θ - β): Mastery probability (IRT-based)

Alignment losses:
- l_21: Performance alignment BCE(M_IRT, M_ref)
- l_22: Difficulty alignment MSE(β_learned, β_IRT)
- l_23: Ability alignment MSE(θ_t_learned, θ_IRT)
"""

import os
import pickle
from typing import Dict, List
import torch
import torch.nn.functional as F

from .base import ReferenceModel


class IRTReferenceModel(ReferenceModel):
    """
    Rasch IRT as reference model for construct validity.
    
    Validates that iKT3's learned factors (θ, β) align with
    independent IRT calibration, establishing convergent validity.
    """
    
    def __init__(self, num_skills: int):
        """
        Initialize IRT reference model.
        
        Args:
            num_skills: Number of skills in the dataset
        """
        super().__init__("IRT", num_skills)
    
    def load_targets(self, targets_path: str) -> Dict[str, torch.Tensor]:
        """
        Load IRT targets with θ_IRT, β_IRT, M_ref.
        
        Supports both static and dynamic IRT targets:
        - Static: student_abilities dict {uid: scalar}
        - Dynamic: theta_trajectories dict {uid: [L]}
        
        Args:
            targets_path: Path to IRT targets file (.pkl)
        
        Returns:
            Dictionary with:
            - 'beta_irt': [num_skills] skill difficulties (always static)
            - 'theta_irt': dict {uid: θ_value or θ_trajectory} student abilities
            - 'm_ref': dict {uid: tensor[seq_len]} reference predictions
            - 'is_dynamic': bool - True if using time-varying theta
        
        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If required keys missing
        """
        if not os.path.exists(targets_path):
            raise FileNotFoundError(
                f"IRT targets file not found: {targets_path}\n"
                f"Generate with: python examples/compute_irt_extended_targets.py (static)\n"
                f"         or: python examples/compute_irt_dynamic_targets.py (dynamic)"
            )
        
        with open(targets_path, 'rb') as f:
            data = pickle.load(f)
        
        # Detect format: dynamic (theta_trajectories) vs static (student_abilities)
        is_dynamic = 'theta_trajectories' in data
        
        if is_dynamic:
            # Dynamic IRT targets
            required_keys = ['skill_difficulties', 'theta_trajectories', 'm_ref_trajectories']
            theta_key = 'theta_trajectories'
            m_ref_key = 'm_ref_trajectories'
            format_name = "Dynamic IRT (time-varying θ)"
        else:
            # Static IRT targets
            required_keys = ['skill_difficulties', 'student_abilities', 'reference_predictions']
            theta_key = 'student_abilities'
            m_ref_key = 'reference_predictions'
            format_name = "Static IRT (constant θ)"
        
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise KeyError(
                f"IRT targets file missing required keys: {missing_keys}\n"
                f"File: {targets_path}\n"
                f"Available keys: {list(data.keys())}"
            )
        
        # Convert skill difficulties to tensor
        skill_difficulties = data['skill_difficulties']
        beta_irt = torch.tensor(
            [skill_difficulties.get(k, 0.0) for k in range(self.num_skills)],
            dtype=torch.float32
        )
        
        theta_data = data[theta_key]
        m_ref_data = data[m_ref_key]
        
        print(f"✓ Loaded IRT targets from {targets_path}")
        print(f"  Format: {format_name}")
        print(f"  - {len(beta_irt)} skill difficulties (static)")
        print(f"  - {len(theta_data)} student ability {'trajectories' if is_dynamic else 'values'}")
        print(f"  - {len(m_ref_data)} reference prediction sequences")
        
        return {
            'beta_irt': beta_irt,
            'theta_irt': theta_data,
            'm_ref': m_ref_data,
            'is_dynamic': is_dynamic,
            'metadata': data.get('metadata', {})
        }
    
    def compute_alignment_losses(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        lambda_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute IRT alignment losses: l_21 (performance), l_22 (difficulty), l_23 (ability).
        
        Args:
            model_outputs: Must contain 'mastery_irt', 'beta_learned', 'theta_t_learned'
            targets: From load_targets() - 'beta_irt', 'theta_irt', 'm_ref'
            lambda_weights: Not used in loss computation (kept for interface compatibility)
        
        Returns:
            {
                'l_21_performance': BCE(M_IRT, M_ref),
                'l_22_difficulty': MSE(β_learned, β_IRT),
                'l_23_ability': MSE(θ_t_learned, θ_IRT),
                'l_align_total': l_21 + l_23  # For λ weighting
            }
        """
        device = next(iter(model_outputs.values())).device
        
        # l_21: Performance alignment (prediction consistency)
        # BCE between learned IRT predictions and reference IRT predictions
        if 'mastery_irt' in model_outputs and 'm_ref' in targets:
            l_21 = F.binary_cross_entropy(
                model_outputs['mastery_irt'],
                targets['m_ref'],
                reduction='mean'
            )
        else:
            l_21 = torch.tensor(0.0, device=device)
            print("⚠️  Warning: Cannot compute l_21 - missing mastery_irt or m_ref")
        
        # l_22: Difficulty regularization (stability constraint)
        # MSE between learned difficulties and IRT-calibrated difficulties
        # Extract ground truth difficulties for the specific skills in the batch
        if 'beta_learned' in model_outputs and 'beta_irt' in targets and 'questions' in model_outputs:
            beta_irt_full = targets['beta_irt'].to(device)  # [num_skills]
            questions = model_outputs['questions']  # [B, L] skill indices
            
            # Extract IRT difficulties for each question in the batch
            # beta_irt_batch[i, j] = beta_irt[questions[i, j]]
            beta_irt_batch = beta_irt_full[questions]  # [B, L]
            
            l_22 = F.mse_loss(
                model_outputs['beta_learned'],
                beta_irt_batch,
                reduction='mean'
            )
        else:
            l_22 = torch.tensor(0.0, device=device)
            if 'beta_learned' not in model_outputs or 'beta_irt' not in targets:
                print("⚠️  Warning: Cannot compute l_22 - missing beta_learned or beta_irt")
            if 'questions' not in model_outputs:
                print("⚠️  Warning: Cannot compute l_22 - missing questions for indexing")
        
        # l_23: Ability alignment (convergent validity)
        # MSE between learned abilities and IRT-calibrated abilities
        # Supports both static (expanded) and dynamic (trajectory) theta
        if 'theta_t_learned' in model_outputs and 'theta_irt' in targets:
            theta_t_learned = model_outputs['theta_t_learned']  # [B, L]
            theta_irt = targets['theta_irt']
            
            # Check if theta_irt is static [B] or dynamic [B, L]
            if theta_irt.dim() == 1:
                # Static IRT: expand to [B, L] by broadcasting
                batch_size, seq_len = theta_t_learned.shape
                theta_irt_expanded = theta_irt.unsqueeze(1).expand(batch_size, seq_len)  # [B] → [B, L]
                theta_comparison = theta_irt_expanded
            else:
                # Dynamic IRT: already [B, L], use directly
                theta_comparison = theta_irt  # [B, L]
            
            # Direct comparison at every timestep
            l_23 = F.mse_loss(
                theta_t_learned,    # [B, L] - model's time-varying ability
                theta_comparison,   # [B, L] - IRT reference (static expanded or dynamic)
                reduction='mean'
            )
        else:
            l_23 = torch.tensor(0.0, device=device)
            print("⚠️  Warning: Cannot compute l_23 - missing theta_t_learned or theta_irt")
        
        # Combined alignment loss for λ weighting
        # Note: l_22 is handled separately (always-on with weight c)
        l_align_total = l_21 + l_23
        
        return {
            'l_21_performance': l_21,
            'l_22_difficulty': l_22,
            'l_23_ability': l_23,
            'l_align_total': l_align_total
        }
    
    def get_loss_names(self) -> List[str]:
        """Return IRT-specific loss names."""
        return ['l_21_performance', 'l_22_difficulty', 'l_23_ability']
    
    def get_interpretable_factors(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract IRT interpretable factors: θ (ability), β (difficulty), M (mastery).
        
        Args:
            model_outputs: From iKT3 forward() pass
        
        Returns:
            {
                'theta': Student abilities [B, L] (time-varying),
                'beta': Skill difficulties [B, L] (static per skill),
                'mastery': Mastery probabilities [B, L]
            }
        """
        return {
            'theta': model_outputs.get('theta_t_learned'),
            'beta': model_outputs.get('beta_learned'),
            'mastery': model_outputs.get('mastery_irt')
        }
