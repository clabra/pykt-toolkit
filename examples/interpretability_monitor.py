"""
Training-Time Interpretability Monitoring Hook

This hook calculates the 4 interpretability metrics during training
and logs them alongside performance metrics.
"""

import torch
import numpy as np
from scipy.stats import pearsonr
import logging

class InterpretabilityMonitor:
    """
    Hook to monitor interpretability metrics during training.
    
    Calculates and logs:
    1. Mastery-Performance Correlation
    2. Gain-Correctness Correlation  
    3. Non-Negative Gains Violation Rate
    4. Mastery Monotonicity Violation Rate
    """
    
    def __init__(self, model, log_frequency=50):
        self.model = model
        self.log_frequency = log_frequency
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, batch_idx, context_seq, value_seq, projected_mastery, 
                 projected_gains, predictions, questions, responses):
        """
        Called during training to compute interpretability metrics.
        
        Args:
            context_seq: [batch_size, seq_len, d_model] - knowledge states
            value_seq: [batch_size, seq_len, d_model] - learning gains  
            projected_mastery: [batch_size, seq_len, num_skills] - skill mastery
            projected_gains: [batch_size, seq_len, num_skills] - skill gains
            predictions: [batch_size, seq_len] - model predictions
            questions: [batch_size, seq_len] - question/skill IDs
            responses: [batch_size, seq_len] - correct/incorrect responses
        """
        self.step_count += 1
        
        if self.step_count % self.log_frequency != 0:
            return
            
        metrics = {}
        
        # 1. Mastery-Performance Correlation
        metrics['mastery_perf_corr'] = self._compute_mastery_performance_correlation(
            projected_mastery, predictions, questions)
        
        # 2. Gain-Correctness Correlation
        metrics['gain_correctness_corr'] = self._compute_gain_correctness_correlation(
            projected_gains, responses)
        
        # 3. Non-Negative Gains Violation Rate
        metrics['negative_gains_pct'] = self._compute_negative_gains_rate(projected_gains)
        
        # 4. Mastery Monotonicity Violation Rate  
        metrics['monotonicity_violations_pct'] = self._compute_monotonicity_violations(
            projected_mastery, responses, questions)
        
        # Log all metrics
        self._log_metrics(batch_idx, metrics)
        
    def _compute_mastery_performance_correlation(self, projected_mastery, predictions, questions):
        """Correlation between skill mastery and performance on that skill."""
        correlations = []
        batch_size, seq_len, num_skills = projected_mastery.shape
        
        for skill_id in range(num_skills):
            skill_masteries = []
            skill_predictions = []
            
            for b in range(batch_size):
                for t in range(seq_len):
                    if questions[b, t].item() == skill_id:
                        skill_masteries.append(projected_mastery[b, t, skill_id].item())
                        skill_predictions.append(predictions[b, t].item())
            
            if len(skill_masteries) > 1:
                corr, _ = pearsonr(skill_masteries, skill_predictions)
                if not np.isnan(corr):
                    correlations.append(corr)
                    
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_gain_correctness_correlation(self, projected_gains, responses):
        """Correlation between learning gain magnitude and response correctness."""
        gain_magnitudes = torch.norm(projected_gains, dim=-1).flatten().detach().cpu().numpy()
        responses_flat = responses.flatten().detach().cpu().numpy()
        
        if len(gain_magnitudes) > 1:
            corr, _ = pearsonr(gain_magnitudes, responses_flat)
            return corr if not np.isnan(corr) else 0.0
        return 0.0
    
    def _compute_negative_gains_rate(self, projected_gains):
        """Percentage of projected gains that are negative."""
        negative_gains = (projected_gains < 0).float()
        return negative_gains.mean().item() * 100
    
    def _compute_monotonicity_violations(self, projected_mastery, responses, questions):
        """Percentage of cases where mastery decreases after correct response."""
        violations = 0
        total_cases = 0
        
        batch_size, seq_len, num_skills = projected_mastery.shape
        
        for b in range(batch_size):
            for t in range(1, seq_len):
                skill_id = questions[b, t].item()
                prev_skill_id = questions[b, t-1].item()
                
                if skill_id == prev_skill_id:  # Same skill consecutive interactions
                    prev_mastery = projected_mastery[b, t-1, skill_id].item()
                    curr_mastery = projected_mastery[b, t, skill_id].item()
                    response = responses[b, t-1].item()
                    
                    # Violation: mastery decreases after correct response
                    # OR mastery increases after incorrect response  
                    if (response == 1 and curr_mastery < prev_mastery) or \
                       (response == 0 and curr_mastery > prev_mastery):
                        violations += 1
                    total_cases += 1
        
        return (violations / total_cases * 100) if total_cases > 0 else 0.0
    
    def _log_metrics(self, batch_idx, metrics):
        """Log interpretability metrics."""
        log_str = f"Step {self.step_count:6d} | "
        log_str += f"Mastery-Perf: {metrics['mastery_perf_corr']:+.3f} | "
        log_str += f"Gain-Corr: {metrics['gain_correctness_corr']:+.3f} | " 
        log_str += f"Neg-Gains: {metrics['negative_gains_pct']:5.1f}% | "
        log_str += f"Monoton-Viol: {metrics['monotonicity_violations_pct']:5.1f}%"
        
        self.logger.info(log_str)
        
        # Also log to wandb if available and initialized
        try:
            import wandb
            if wandb.run is not None:  # Check if wandb is initialized
                wandb.log({
                    'interpretability/mastery_performance_correlation': metrics['mastery_perf_corr'],
                    'interpretability/gain_correctness_correlation': metrics['gain_correctness_corr'], 
                    'interpretability/negative_gains_percentage': metrics['negative_gains_pct'],
                    'interpretability/monotonicity_violations_percentage': metrics['monotonicity_violations_pct'],
                    'step': self.step_count
                })
        except (ImportError, Exception):
            pass  # Silently skip wandb logging if not available or not initialized