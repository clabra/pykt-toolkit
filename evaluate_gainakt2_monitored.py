#!/usr/bin/env python3
"""
Evaluation script for trained GainAKT2Monitored model.

This script loads a saved model and evaluates it on the test set,
providing both performance metrics and interpretability analysis.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def load_trained_model(model_path, device):
    """Load a trained GainAKT2Monitored model from checkpoint."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint['model_config']
    logger.info(f"Model config: {model_config}")
    
    # Create model
    model = create_monitored_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info("Model loaded successfully!")
    logger.info(f"Training epoch: {checkpoint['epoch']}")
    logger.info(f"Best AUC: {checkpoint['best_auc']:.4f}")
    
    return model, model_config


def evaluate_model_comprehensive(model, dataloader, device, logger, dataset_name="test"):
    """Comprehensive evaluation of the model."""
    model.eval()
    
    # Metrics storage
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probs = []
    
    # Interpretability metrics storage
    all_mastery_values = []
    all_gain_values = []
    all_questions = []
    all_responses = []
    
    criterion = nn.BCELoss()
    
    logger.info(f"Evaluating on {dataset_name} set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Evaluating {dataset_name}')):
            # Move data to device (use concepts for assist2015)
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Forward pass with states to get interpretability info
            outputs = model.forward_with_states(q=questions, r=responses, qry=questions_shifted)
            predictions = outputs['predictions']
            projected_mastery = outputs['projected_mastery']
            projected_gains = outputs['projected_gains']
            
            # Apply mask
            valid_mask = mask.bool()
            valid_predictions = predictions[valid_mask]
            valid_targets = responses_shifted[valid_mask].float()
            
            # Compute loss
            loss = criterion(valid_predictions, valid_targets)
            total_loss += loss.item()
            
            # Collect predictions and targets
            valid_predictions_np = valid_predictions.cpu().numpy()
            valid_targets_np = valid_targets.cpu().numpy()
            
            all_predictions.extend(valid_predictions_np > 0.5)
            all_targets.extend(valid_targets_np)
            all_probs.extend(valid_predictions_np)
            
            # Collect interpretability data
            valid_questions = questions_shifted[valid_mask].cpu().numpy()
            valid_responses = responses_shifted[valid_mask].cpu().numpy()
            
            all_questions.extend(valid_questions)
            all_responses.extend(valid_responses)
            
            # Extract relevant mastery and gain values for each question
            batch_size, seq_len, num_concepts = projected_mastery.shape
            for b in range(batch_size):
                for t in range(seq_len):
                    if mask[b, t]:  # Only include valid positions
                        concept_id = questions_shifted[b, t].item()
                        if 0 <= concept_id < num_concepts:
                            mastery_val = projected_mastery[b, t, concept_id].item()
                            gain_val = projected_gains[b, t, concept_id].item()
                            all_mastery_values.append(mastery_val)
                            all_gain_values.append(gain_val)
    
    # Compute performance metrics
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_targets, all_probs)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    
    # Compute interpretability metrics
    interpretability_metrics = compute_interpretability_metrics(
        all_mastery_values, all_gain_values, all_questions, 
        all_responses, all_probs, logger
    )
    
    # Performance metrics
    performance_metrics = {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return performance_metrics, interpretability_metrics


def compute_interpretability_metrics(mastery_values, gain_values, questions, responses, predictions, logger):
    """Compute interpretability constraint metrics."""
    metrics = {}
    
    # Convert to numpy arrays
    mastery_values = np.array(mastery_values)
    gain_values = np.array(gain_values) 
    questions = np.array(questions)
    responses = np.array(responses)
    predictions = np.array(predictions)
    
    logger.info("Computing interpretability metrics...")
    
    # 1. Mastery-Performance Correlation
    if len(mastery_values) > 1:
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(mastery_values, predictions)
        metrics['mastery_performance_correlation'] = corr if not np.isnan(corr) else 0.0
        metrics['mastery_performance_p_value'] = p_value if not np.isnan(p_value) else 1.0
    else:
        metrics['mastery_performance_correlation'] = 0.0
        metrics['mastery_performance_p_value'] = 1.0
    
    # 2. Gain-Correctness Correlation
    if len(gain_values) > 1:
        corr, p_value = pearsonr(np.abs(gain_values), responses)
        metrics['gain_correctness_correlation'] = corr if not np.isnan(corr) else 0.0
        metrics['gain_correctness_p_value'] = p_value if not np.isnan(p_value) else 1.0
    else:
        metrics['gain_correctness_correlation'] = 0.0
        metrics['gain_correctness_p_value'] = 1.0
    
    # 3. Non-negative gains constraint
    negative_gains_pct = (gain_values < 0).mean() * 100
    metrics['negative_gains_percentage'] = negative_gains_pct
    
    # 4. Summary statistics
    metrics['mastery_mean'] = mastery_values.mean()
    metrics['mastery_std'] = mastery_values.std()
    metrics['gain_mean'] = gain_values.mean()
    metrics['gain_std'] = gain_values.std()
    metrics['gain_min'] = gain_values.min()
    metrics['gain_max'] = gain_values.max()
    
    return metrics


def print_evaluation_results(performance_metrics, interpretability_metrics, logger):
    """Print comprehensive evaluation results."""
    
    logger.info("="*70)
    logger.info("COMPREHENSIVE MODEL EVALUATION RESULTS")
    logger.info("="*70)
    
    # Performance metrics
    logger.info("📊 PERFORMANCE METRICS:")
    logger.info(f"  AUC:        {performance_metrics['auc']:.4f}")
    logger.info(f"  Accuracy:   {performance_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:  {performance_metrics['precision']:.4f}")
    logger.info(f"  Recall:     {performance_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:   {performance_metrics['f1_score']:.4f}")
    logger.info(f"  Loss:       {performance_metrics['loss']:.4f}")
    
    logger.info("")
    logger.info("🔍 INTERPRETABILITY METRICS:")
    
    # Correlations
    mastery_corr = interpretability_metrics['mastery_performance_correlation']
    gain_corr = interpretability_metrics['gain_correctness_correlation']
    
    logger.info(f"  Mastery-Performance Correlation: {mastery_corr:+.4f}")
    logger.info(f"  Gain-Correctness Correlation:    {gain_corr:+.4f}")
    
    # Constraint violations
    neg_gains = interpretability_metrics['negative_gains_percentage']
    logger.info(f"  Negative Gains Percentage:       {neg_gains:.2f}%")
    
    logger.info("")
    logger.info("📈 INTERPRETABILITY STATISTICS:")
    logger.info(f"  Mastery Values - Mean: {interpretability_metrics['mastery_mean']:.4f}, "
               f"Std: {interpretability_metrics['mastery_std']:.4f}")
    logger.info(f"  Gain Values - Mean: {interpretability_metrics['gain_mean']:.4f}, "
               f"Std: {interpretability_metrics['gain_std']:.4f}")
    logger.info(f"  Gain Range: [{interpretability_metrics['gain_min']:.4f}, "
               f"{interpretability_metrics['gain_max']:.4f}]")
    
    # Interpretability assessment
    logger.info("")
    logger.info("✅ INTERPRETABILITY ASSESSMENT:")
    
    if mastery_corr > 0.3:
        logger.info(f"  ✅ Mastery-Performance correlation is GOOD ({mastery_corr:.3f} > 0.3)")
    elif mastery_corr > 0.1:
        logger.info(f"  ⚠️  Mastery-Performance correlation is MODERATE ({mastery_corr:.3f})")
    else:
        logger.info(f"  ❌ Mastery-Performance correlation is LOW ({mastery_corr:.3f} < 0.1)")
    
    if gain_corr > 0.2:
        logger.info(f"  ✅ Gain-Correctness correlation is GOOD ({gain_corr:.3f} > 0.2)")
    elif gain_corr > 0.0:
        logger.info(f"  ⚠️  Gain-Correctness correlation is MODERATE ({gain_corr:.3f})")
    else:
        logger.info(f"  ❌ Gain-Correctness correlation is LOW/NEGATIVE ({gain_corr:.3f})")
    
    if neg_gains < 10.0:
        logger.info(f"  ✅ Negative gains percentage is LOW ({neg_gains:.1f}% < 10%)")
    elif neg_gains < 25.0:
        logger.info(f"  ⚠️  Negative gains percentage is MODERATE ({neg_gains:.1f}%)")
    else:
        logger.info(f"  ❌ Negative gains percentage is HIGH ({neg_gains:.1f}% > 25%)")
    
    logger.info("="*70)


def main():
    """Main evaluation function."""
    logger = setup_logging()
    
    # Model path
    model_path = "/workspaces/pykt-toolkit/saved_model/gainakt2_monitored_quick_auc_0.7245/model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, model_config = load_trained_model(model_path, device)
    
    # Dataset configuration
    dataset_name = "assist2015"
    model_name = "gainakt2"
    fold = 0
    batch_size = 32
    
    data_config = {
        "assist2015": {
            "dpath": "/workspaces/pykt-toolkit/data/assist2015",
            "num_q": 0,
            "num_c": 100,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": 3,
            "maxlen": 200,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1, 2, 3, 4],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv"
        }
    }
    
    # Load datasets
    logger.info("Loading datasets...")
    train_loader, valid_loader = init_dataset4train(
        dataset_name, model_name, data_config, fold, batch_size
    )
    
    # Evaluate on validation set
    logger.info("\\nEvaluating on VALIDATION set...")
    valid_performance, valid_interpretability = evaluate_model_comprehensive(
        model, valid_loader, device, logger, "validation"
    )
    
    print_evaluation_results(valid_performance, valid_interpretability, logger)
    
    # Save results
    results = {
        'model_path': model_path,
        'model_config': model_config,
        'validation_performance': valid_performance,
        'validation_interpretability': valid_interpretability
    }
    
    results_path = model_path.replace('model.pth', 'evaluation_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\\n💾 Evaluation results saved to: {results_path}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)