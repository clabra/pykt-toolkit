#!/usr/bin/env python3
"""
CPU-Safe Test Evaluation - Works around CUDA tensor serialization issues
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import time
from datetime import datetime
import logging

# Force CPU-only evaluation
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.models.gainakt2_monitored import create_monitored_model
from pykt.datasets.data_loader import KTDataset
from torch.utils.data import DataLoader


def evaluate_test_cpu_safe(model_path: str):
    """CPU-safe evaluation on test data by recreating data loaders."""
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"test_evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Force CPU device
    device = torch.device('cpu')
    
    logger.info("🚀 STARTING CPU-SAFE TEST EVALUATION")
    logger.info("=" * 60)
    logger.info("⚠️  CORRECTING DATA LEAKAGE: Using actual test data")
    logger.info("💻 Using CPU-only evaluation to avoid CUDA serialization issues")
    
    # Load model first
    logger.info(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = create_monitored_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info("✓ Model loaded on CPU")
    
    # Create test dataset directly (avoiding pickled CUDA tensors)
    test_file = "/workspaces/pykt-toolkit/data/assist2015/test_sequences.csv"
    logger.info(f"Loading test data from: {test_file}")
    
    try:
        # Create fresh test dataset (this will avoid CUDA serialization issues)
        test_dataset = KTDataset(test_file, ["concepts"], {-1})
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        logger.info(f"✓ Test dataset loaded: {len(test_dataset)} sequences, {len(test_loader)} batches")
    except Exception as e:
        logger.error(f"❌ Failed to load test data: {e}")
        return None
    
    # Evaluate
    logger.info("🧪 Evaluating on actual test dataset...")
    
    start_time = time.time()
    all_predictions = []
    all_targets = []
    consistency_perfect = 0
    consistency_total = 0
    processed_batches = 0
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                try:
                    # Move batch to CPU (should already be CPU)
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
                    
                    # Collect predictions
                    valid_mask = mask.bool()
                    valid_predictions = predictions[valid_mask].cpu().numpy()
                    valid_targets = responses_shifted[valid_mask].cpu().numpy()
                    
                    all_predictions.extend(valid_predictions.flatten())
                    all_targets.extend(valid_targets.flatten())
                    
                    # Check consistency
                    if 'mastery_states' in outputs:
                        mastery_states = outputs['mastery_states']
                        
                        for b in range(mastery_states.shape[0]):
                            student_mastery = mastery_states[b]
                            student_mask = mask[b].bool()
                            
                            if student_mask.sum() <= 1:
                                continue
                            
                            valid_mastery = student_mastery[student_mask]
                            
                            # Check monotonicity
                            is_consistent = True
                            for concept in range(valid_mastery.shape[1]):
                                concept_progression = valid_mastery[:, concept]
                                
                                # Check monotonic non-decreasing
                                if not torch.all(concept_progression[1:] >= concept_progression[:-1]):
                                    is_consistent = False
                                    break
                                
                                # Check bounds [0, 1]
                                if torch.any(concept_progression < 0) or torch.any(concept_progression > 1):
                                    is_consistent = False
                                    break
                            
                            consistency_total += 1
                            if is_consistent:
                                consistency_perfect += 1
                    
                    processed_batches += 1
                    if batch_idx % 20 == 0:
                        logger.info(f"Processed {processed_batches}/{len(test_loader)} batches...")
                        
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return None
    
    evaluation_time = time.time() - start_time
    
    if len(all_predictions) == 0:
        logger.error("❌ No predictions collected")
        return None
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    try:
        test_auc = roc_auc_score(all_targets, all_predictions)
    except ValueError as e:
        logger.error(f"AUC calculation failed: {e}")
        test_auc = 0.0
    
    test_accuracy = accuracy_score(all_targets, (all_predictions > 0.5).astype(int))
    consistency_rate = consistency_perfect / consistency_total if consistency_total > 0 else 0.0
    
    # Compare with validation performance
    validation_auc = 0.7210  # From training results
    auc_difference = test_auc - validation_auc
    
    # Assessment
    performance_grade = "A" if test_auc >= 0.70 else "B" if test_auc >= 0.65 else "C" if test_auc >= 0.60 else "D" if test_auc >= 0.55 else "F"
    consistency_grade = "A+" if consistency_rate >= 0.95 else "A" if consistency_rate >= 0.90 else "B" if consistency_rate >= 0.85 else "C" if consistency_rate >= 0.80 else "F"
    
    if consistency_rate >= 0.99 and test_auc >= 0.65:
        verdict = "EXCELLENT"
    elif consistency_rate >= 0.95 and test_auc >= 0.60:
        verdict = "GOOD"
    elif consistency_rate >= 0.90:
        verdict = "ACCEPTABLE"
    else:
        verdict = "NEEDS_IMPROVEMENT"
        
    # Generalization assessment
    if abs(auc_difference) < 0.01:
        generalization = "EXCELLENT - Minimal overfitting"
    elif abs(auc_difference) < 0.02:
        generalization = "GOOD - Normal generalization gap" 
    elif auc_difference < -0.02:
        generalization = "OVERFITTING - Test much lower than validation"
    else:
        generalization = "UNEXPECTED - Test higher than validation"
    
    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "evaluation_type": "ACTUAL_TEST_DATASET_CPU_SAFE",
        "data_leakage_corrected": True,
        "performance_metrics": {
            "test_auc": float(test_auc),
            "test_accuracy": float(test_accuracy),
            "validation_auc": validation_auc,
            "auc_difference": float(auc_difference),
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
            "overall_verdict": verdict,
            "generalization_assessment": generalization
        },
        "dataset_info": {
            "test_sequences": len(test_dataset),
            "test_batches": processed_batches,
            "device": "cpu"
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, f"corrected_test_evaluation_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results
    logger.info("\\n" + "=" * 60)
    logger.info("📊 CORRECTED TEST EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"🎯 TEST AUC: {test_auc:.4f} (Grade: {performance_grade})")
    logger.info(f"📊 VALIDATION AUC was: {validation_auc:.4f}")
    logger.info(f"📊 DIFFERENCE: {auc_difference:+.4f}")
    logger.info(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"✅ Perfect Consistency: {consistency_rate*100:.1f}% ({consistency_perfect}/{consistency_total} students)")
    logger.info(f"🔍 Generalization: {generalization}")
    logger.info(f"🏆 Overall Verdict: {verdict}")
    logger.info(f"⏱️  Evaluation Time: {evaluation_time:.2f} seconds")
    logger.info(f"📄 Results saved: {results_file}")
    logger.info("=" * 60)
    
    print("\\n🎉 CORRECTED TEST EVALUATION COMPLETED!")
    print(f"📊 Test AUC: {test_auc:.4f} (vs Validation: {validation_auc:.4f})")
    print(f"📊 Difference: {auc_difference:+.4f}")
    print(f"✅ Consistency: {consistency_rate*100:.1f}%")
    print(f"🔍 {generalization}")
    print(f"🏆 Verdict: {verdict}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                       default='saved_model/gainakt2_cumulative_mastery_quick_test/best_model.pth')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        exit(1)
    
    evaluate_test_cpu_safe(args.model_path)