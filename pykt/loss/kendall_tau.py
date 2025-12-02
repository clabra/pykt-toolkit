"""
Differentiable Kendall Tau correlation for ranking supervision.

This module provides a differentiable approximation of Kendall's Tau rank correlation
coefficient, enabling its use as a loss function for neural network training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableKendallTau(nn.Module):
    """
    Differentiable approximation of Kendall's Tau correlation coefficient.
    
    Kendall's Tau measures rank correlation between two sequences by counting
    concordant and discordant pairs. This implementation uses a smooth sigmoid
    approximation to make it differentiable.
    
    Mathematical formulation:
        τ = (C - D) / (C + D)
    where C = concordant pairs, D = discordant pairs
    
    Smooth approximation:
        agreement(i,j) = tanh((pred_i - pred_j)(target_i - target_j) / temperature)
        τ ≈ mean(agreement)
    
    Args:
        temperature (float): Temperature parameter for sigmoid smoothness.
            Lower values → sharper (closer to sign function)
            Higher values → smoother gradients
            Default: 0.1 (balanced)
        reduction (str): 'mean', 'sum', or 'none'
        eps (float): Small constant for numerical stability
        
    Input shapes:
        pred: [batch_size, seq_len, num_skills] or [batch_size, seq_len]
        target: same as pred
        mask: optional [batch_size, seq_len, num_skills] or [batch_size, seq_len]
              True/1 for valid entries, False/0 for invalid
              
    Output:
        Kendall Tau correlation in range [-1, 1]
        Higher values = better agreement
        
    Example:
        >>> tau_module = DifferentiableKendallTau(temperature=0.1)
        >>> pred = torch.randn(32, 50, 100)  # [batch, seq, skills]
        >>> target = torch.randn(32, 50, 100)
        >>> tau = tau_module(pred, target)
        >>> loss = 1.0 - tau  # Convert to loss (minimize)
    """
    
    def __init__(self, temperature=0.1, reduction='mean', eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, pred, target, mask=None):
        """
        Compute differentiable Kendall Tau correlation.
        
        Args:
            pred: Predicted values [batch, seq, skills] or [batch, seq]
            target: Target values, same shape as pred
            mask: Optional validity mask, same shape as pred
            
        Returns:
            Kendall Tau correlation (scalar if reduction != 'none')
        """
        # Handle NaN values by creating/updating mask
        valid_pred = ~torch.isnan(pred)
        valid_target = ~torch.isnan(target)
        valid_both = valid_pred & valid_target
        
        if mask is not None:
            valid_both = valid_both & mask.bool()
        
        # Replace NaN with zeros (will be masked out)
        pred_clean = torch.where(valid_both, pred, torch.zeros_like(pred))
        target_clean = torch.where(valid_both, target, torch.zeros_like(target))
        
        # Flatten to [num_valid_elements]
        pred_flat = pred_clean[valid_both]
        target_flat = target_clean[valid_both]
        
        if pred_flat.numel() < 2:
            # Not enough valid pairs for correlation
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Compute pairwise differences (upper triangle only for efficiency)
        # pred_diff[i,j] = pred[i] - pred[j] for i < j
        pred_diff = pred_flat.unsqueeze(0) - pred_flat.unsqueeze(1)
        target_diff = target_flat.unsqueeze(0) - target_flat.unsqueeze(1)
        
        # Agreement: positive if both diffs have same sign (concordant)
        #            negative if different signs (discordant)
        pairwise_agreement = pred_diff * target_diff
        
        # Smooth sign approximation: tanh(x/T) ≈ sign(x) as T→0
        agreement_score = torch.tanh(pairwise_agreement / self.temperature)
        
        # Use upper triangle only (avoid double counting and diagonal)
        n = pred_flat.size(0)
        triu_indices = torch.triu_indices(n, n, offset=1, device=pred.device)
        agreement_upper = agreement_score[triu_indices[0], triu_indices[1]]
        
        # Kendall Tau = average agreement over all pairs
        if self.reduction == 'mean':
            tau = agreement_upper.mean()
        elif self.reduction == 'sum':
            tau = agreement_upper.sum()
        else:  # 'none'
            tau = agreement_upper
            
        return tau
    
    def extra_repr(self):
        return f'temperature={self.temperature}, reduction={self.reduction}'


class KendallTauLoss(nn.Module):
    """
    Kendall Tau correlation as a loss function (minimizing 1 - τ).
    
    Wraps DifferentiableKendallTau to provide a loss interface with optional
    statistics tracking.
    
    Args:
        temperature (float): Temperature for sigmoid smoothness (default: 0.1)
        reduction (str): 'mean', 'sum', or 'none'
        track_stats (bool): If True, maintains running statistics
        
    Example:
        >>> loss_fn = KendallTauLoss(temperature=0.1)
        >>> pred_mastery = model.get_irt_mastery()  # [batch, seq, skills]
        >>> rasch_mastery = load_rasch_targets()    # [batch, seq, skills]
        >>> loss = loss_fn(pred_mastery, rasch_mastery)
        >>> loss.backward()
    """
    
    def __init__(self, temperature=0.1, reduction='mean', track_stats=False):
        super().__init__()
        self.kendall_tau = DifferentiableKendallTau(
            temperature=temperature,
            reduction=reduction
        )
        self.track_stats = track_stats
        
        if track_stats:
            self.register_buffer('running_tau', torch.tensor(0.0))
            self.register_buffer('num_updates', torch.tensor(0))
    
    def forward(self, pred, target, mask=None):
        """
        Compute Kendall Tau loss = 1 - τ (lower is better).
        
        Args:
            pred: Predicted values
            target: Target values  
            mask: Optional validity mask
            
        Returns:
            Loss value (0 = perfect correlation, 2 = perfect anti-correlation)
        """
        tau = self.kendall_tau(pred, target, mask)
        loss = 1.0 - tau
        
        # Track statistics if enabled
        if self.track_stats and self.training:
            with torch.no_grad():
                self.running_tau.mul_(self.num_updates).add_(tau)
                self.num_updates.add_(1)
                self.running_tau.div_(self.num_updates)
        
        return loss
    
    def get_average_tau(self):
        """Get running average of Kendall Tau (if tracking enabled)."""
        if not self.track_stats:
            raise RuntimeError("Statistics tracking not enabled")
        return self.running_tau.item()
    
    def reset_stats(self):
        """Reset running statistics."""
        if self.track_stats:
            self.running_tau.zero_()
            self.num_updates.zero_()
    
    def extra_repr(self):
        return f'temperature={self.kendall_tau.temperature}, track_stats={self.track_stats}'
