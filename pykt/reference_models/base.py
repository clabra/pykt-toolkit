"""
Abstract Base Class for Reference Models

Defines the interface that all reference models must implement for
integration with the iKT3 model architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch


class ReferenceModel(ABC):
    """
    Abstract interface for theoretical reference models.
    
    Reference models provide:
    1. Pre-computed targets from theoretical calibration
    2. Alignment loss computation for construct validity
    3. Interpretable factor extraction for validation
    
    All concrete reference models (IRT, BKT, DINA, etc.) must inherit
    from this class and implement its abstract methods.
    """
    
    def __init__(self, model_name: str, num_skills: int):
        """
        Initialize reference model.
        
        Args:
            model_name: Human-readable name (e.g., "IRT", "BKT")
            num_skills: Number of skills in the dataset
        """
        self.model_name = model_name
        self.num_skills = num_skills
    
    @abstractmethod
    def load_targets(self, targets_path: str) -> Dict[str, torch.Tensor]:
        """
        Load pre-computed reference targets from file.
        
        Args:
            targets_path: Path to .pkl file containing reference targets
        
        Returns:
            Dictionary with reference-model-specific keys
            (e.g., for IRT: {'beta_irt', 'theta_irt', 'm_ref'})
        
        Raises:
            FileNotFoundError: If targets file doesn't exist
            KeyError: If required keys are missing from file
        """
        pass
    
    @abstractmethod
    def compute_alignment_losses(
        self, 
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        lambda_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute alignment losses between model outputs and reference targets.
        
        Args:
            model_outputs: Dictionary from iKT3 forward() pass
            targets: Dictionary from load_targets()
            lambda_weights: Dictionary with 'lambda_interp' and 'lambda_reg' keys
        
        Returns:
            Dictionary with loss components:
            - Individual losses (l_21, l_22, l_23, etc.)
            - 'l_align_total': Combined alignment loss for λ weighting
        
        Example:
            For IRT:
            {
                'l_21_performance': BCE(M_IRT, M_ref),
                'l_22_difficulty': MSE(β_learned, β_IRT),
                'l_23_ability': MSE(θ_learned, θ_IRT),
                'l_align_total': l_21 + l_23
            }
        """
        pass
    
    @abstractmethod
    def get_loss_names(self) -> List[str]:
        """
        Return list of loss component names for logging.
        
        Returns:
            List of strings (e.g., ['l_21_performance', 'l_22_difficulty', 'l_23_ability'])
        """
        pass
    
    @abstractmethod
    def get_interpretable_factors(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract interpretable factors from model outputs for validation.
        
        Args:
            model_outputs: Dictionary from iKT3 forward() pass
        
        Returns:
            Dictionary with interpretable factors
            (e.g., for IRT: {'theta', 'beta', 'mastery'})
        
        Used for:
        - Computing correlations with reference model factors
        - Visualizing learning trajectories
        - Generating interpretability reports
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', num_skills={self.num_skills})"
