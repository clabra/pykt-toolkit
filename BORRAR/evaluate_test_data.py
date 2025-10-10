#!/usr/bin/env python3
"""
Test Dataset Evaluation for GainAKT2Monitored with Cumulative Mastery.
Evaluates the model performance on the actual held-out test dataset.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import time
from datetime import datetime
from tqdm import tqdm
import logging

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_test_datasets
from pykt.models.gainakt2_monitored import create_monitored_model


def evaluate_on_test_data(model_path: str, output_dir: str = "test_evaluation_results"):
    """Evaluate the cumulative mastery model on the held-out test dataset."""
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("🚀 STARTING TEST DATASET EVALUATION")
    logger.info("=" * 60)
    
    # Load test data
    dataset_name = "assist2015"
    data_config = {
        "dataset_name": dataset_name,
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
    
    # Load test datasets with reduced workers to avoid CUDA issues
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(
        data_config, "gainakt2", batch_size=32
    )
    
    # Reduce workers to avoid CUDA initialization issues
    if hasattr(test_loader, 'num_workers'):
        test_loader.num_workers = 0
    
    logger.info("✓ Test datasets loaded successfully")
    logger.info(f"📊 Test set batches: {len(test_loader)}")
    
    # Load cumulative mastery model
    logger.info(f"Loading cumulative mastery model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = create_monitored_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("✓ Cumulative mastery model loaded")
    
    # Evaluate on test data
    logger.info("🧪 Evaluating on held-out test dataset...")
    
    start_time = time.time()
    all_predictions = []
    all_targets = []
    consistency_perfect = 0
    consistency_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test Evaluation'):
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Get predictions
            outputs = model.forward_with_states(
                q=questions, r=responses, qry=questions_shifted
            )
            predictions = outputs['predictions']
            
            # Mask predictions and targets
            valid_mask = mask.bool()
            valid_predictions = predictions[valid_mask].cpu().numpy()
            valid_targets = responses_shifted[valid_mask].cpu().numpy()
            
            all_predictions.extend(valid_predictions.flatten())
            all_targets.extend(valid_targets.flatten())
            
            # Check consistency for each student in batch
            if 'mastery_states' in outputs:
                mastery_states = outputs['mastery_states']  # [batch, seq_len, num_concepts]
                
                for b in range(mastery_states.shape[0]):
                    student_mastery = mastery_states[b]  # [seq_len, num_concepts]
                    student_mask = mask[b].bool()
                    
                    if student_mask.sum() <= 1:  # Skip students with only one interaction
                        continue
                    
                    valid_mastery = student_mastery[student_mask]  # [valid_steps, num_concepts]
                    
                    # Check monotonicity for each concept
                    is_consistent = True
                    for concept in range(valid_mastery.shape[1]):
                        concept_progression = valid_mastery[:, concept]
                        
                        # Check if monotonically non-decreasing
                        if not torch.all(concept_progression[1:] >= concept_progression[:-1]):
                            is_consistent = False
                            break
                        
                        # Check if all values are in [0, 1]
                        if torch.any(concept_progression < 0) or torch.any(concept_progression > 1):
                            is_consistent = False
                            break
                    
                    consistency_total += 1
                    if is_consistent:
                        consistency_perfect += 1
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    try:
        auc = roc_auc_score(all_targets, all_predictions)
    except ValueError:
        auc = 0.0  # Handle case where only one class is present
    
    accuracy = accuracy_score(all_targets, (all_predictions > 0.5).astype(int))
    
    consistency_rate = consistency_perfect / consistency_total if consistency_total > 0 else 0.0
    
    # Grade performance
    def get_performance_grade(auc_score):
        if auc_score >= 0.70:
            return "A"
        elif auc_score >= 0.65:
            return "B"
        elif auc_score >= 0.60:
            return "C"
        elif auc_score >= 0.55:
            return "D"
        else:
            return "F"
    
    def get_consistency_grade(consistency_rate):
        if consistency_rate >= 0.95:
            return "A+"
        elif consistency_rate >= 0.90:
            return "A"
        elif consistency_rate >= 0.85:
            return "B"
        elif consistency_rate >= 0.80:
            return "C"
        else:
            return "F"
    
    performance_grade = get_performance_grade(auc)
    consistency_grade = get_consistency_grade(consistency_rate)
    
    # Overall verdict
    if consistency_rate >= 0.99 and auc >= 0.65:
        verdict = "EXCELLENT"
    elif consistency_rate >= 0.95 and auc >= 0.60:
        verdict = "GOOD"
    elif consistency_rate >= 0.90:
        verdict = "ACCEPTABLE"
    else:
        verdict = "NEEDS_IMPROVEMENT"
    
    # Create results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "evaluation_type": "TEST_DATASET",
        "performance_metrics": {
            "auc": float(auc),
            "accuracy": float(accuracy),
            "total_predictions": len(all_predictions),
            "evaluation_time": float(evaluation_time)
        },
        "consistency_metrics": {
            "perfect_consistency_rate": float(consistency_rate),
            "students_analyzed": consistency_total,
            "perfect_students": consistency_perfect,
            "violation_students": consistency_total - consistency_perfect
        },
        "assessment": {
            "performance_grade": performance_grade,
            "consistency_grade": consistency_grade,
            "overall_verdict": verdict
        },
        "dataset_info": {
            "dataset": dataset_name,
            "test_batches": len(test_loader),
            "model_type": "GainAKT2Monitored_Cumulative_Mastery"
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, f"test_evaluation_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log summary
    logger.info("\\n" + "=" * 60)
    logger.info("📊 TEST DATASET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"🎯 Test AUC: {auc:.4f} (Grade: {performance_grade})")
    logger.info(f"🎯 Test Accuracy: {accuracy:.4f}")
    logger.info(f"✅ Perfect Consistency: {consistency_rate*100:.1f}% ({consistency_perfect}/{consistency_total} students)")
    logger.info(f"🏆 Overall Assessment: {verdict}")
    logger.info(f"⏱️  Evaluation Time: {evaluation_time:.2f} seconds")
    logger.info(f"📄 Results saved: {results_file}")
    logger.info("=" * 60)
    
    print(f"\\n🎉 TEST EVALUATION COMPLETED!")
    print(f"📊 AUC: {auc:.4f}")
    print(f"✅ Consistency: {consistency_rate*100:.1f}%") 
    print(f"🏆 Verdict: {verdict}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate cumulative mastery model on test dataset')
    parser.add_argument('--model_path', type=str, 
                       default='saved_model/gainakt2_cumulative_mastery_quick_test/best_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default='test_evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    evaluate_on_test_data(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()