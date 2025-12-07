"""
Reference Models Package

Provides abstract interface and concrete implementations for theoretical
reference models used in iKT3 for construct validity alignment.

Available reference models:
- IRT (Item Response Theory / Rasch model)
- BKT (Bayesian Knowledge Tracing) - Future

Usage:
    from pykt.reference_models import create_reference_model
    
    ref_model = create_reference_model('irt', num_skills=100)
    targets = ref_model.load_targets('path/to/targets.pkl')
    losses = ref_model.compute_alignment_losses(model_outputs, targets, lambda_weights)
"""

from .base import ReferenceModel
from .irt_reference import IRTReferenceModel

REFERENCE_MODELS = {
    'irt': IRTReferenceModel,
}

def create_reference_model(model_type: str, num_skills: int) -> ReferenceModel:
    """
    Factory function for creating reference models.
    
    Args:
        model_type: Type of reference model ('irt', 'bkt', etc.)
        num_skills: Number of skills in the dataset
    
    Returns:
        ReferenceModel instance
    
    Raises:
        ValueError: If model_type is not recognized
    
    Example:
        >>> ref_model = create_reference_model('irt', num_skills=100)
        >>> targets = ref_model.load_targets('data/assist2015/irt_targets.pkl')
    """
    if model_type not in REFERENCE_MODELS:
        raise ValueError(
            f"Unknown reference model: '{model_type}'. "
            f"Available models: {list(REFERENCE_MODELS.keys())}"
        )
    return REFERENCE_MODELS[model_type](num_skills)

__all__ = ['ReferenceModel', 'IRTReferenceModel', 'create_reference_model', 'REFERENCE_MODELS']
