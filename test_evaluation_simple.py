#!/usr/bin/env python3
"""
Simple Test Dataset Evaluation for GainAKT2Monitored with Cumulative Mastery.
Uses the same approach as quick_benchmark.py but loads the actual test dataset.
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import time
from datetime import datetime
from tqdm import tqdm
import logging

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model


def load_test_data_manually():
    """Load test data using pandas to avoid CUDA dataloader issues."""
    import pandas as pd
    import pickle
    
    # Load the test data directly
    test_file = "/workspaces/pykt-toolkit/data/assist2015/test_sequences.csv_-1.pkl"
    
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        return test_data
    else:
        print(f"❌ Test data file not found: {test_file}")
        return None


def evaluate_on_test_data_simple(model_path: str):
    """Simple evaluation on test data without complex dataloaders."""
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"test_evaluation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("🚀 STARTING SIMPLE TEST DATASET EVALUATION")
    logger.info("=" * 60)
    
    # Load model
    logger.info(f"Loading cumulative mastery model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = create_monitored_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("✓ Cumulative mastery model loaded")
    
    # Load test data manually
    test_data = load_test_data_manually()
    
    if test_data is None:
        logger.error("❌ Failed to load test data")
        return None
        
    logger.info(f"✓ Test data loaded with keys: {list(test_data.keys())}")
    
    # Check if we have the expected keys
    if 'cseqs' in test_data:
        num_sequences = len(test_data['cseqs'])
        logger.info(f"✓ Test data contains {num_sequences} sequences")
    else:
        logger.error(f"❌ Unexpected test data structure: {list(test_data.keys())}")
        return None
    
    # Evaluate
    logger.info("🧪 Evaluating on test dataset...")
    start_time = time.time()
    
    all_predictions = []
    all_targets = []
    consistency_perfect = 0
    consistency_total = 0
    
    # Process in smaller batches to avoid memory issues
    batch_size = 32
    num_sequences = len(test_data['cseqs'])
    
    with torch.no_grad():
        for i in tqdm(range(0, num_sequences, batch_size), desc='Test Evaluation'):
            end_idx = min(i + batch_size, num_sequences)
            
            # Create batch - handle both tensor and numpy data
            def to_tensor(data_slice):
                if torch.is_tensor(data_slice[0]):
                    # Already tensors, stack them
                    return torch.stack([t.cpu() for t in data_slice]).long().to(device)
                else:
                    # Convert numpy to tensor
                    return torch.from_numpy(np.array(data_slice)).long().to(device)
            
            # Handle different key names in test data
            cseqs_key = 'cseqs' if 'cseqs' in test_data else 'concepts'
            rseqs_key = 'rseqs' if 'rseqs' in test_data else 'responses'
            
            batch_cseqs = to_tensor(test_data[cseqs_key][i:end_idx])
            batch_rseqs = to_tensor(test_data[rseqs_key][i:end_idx])
            
            # For shifted sequences, they might be computed on the fly
            if 'shft_cseqs' in test_data:
                batch_shft_cseqs = to_tensor(test_data['shft_cseqs'][i:end_idx])
            else:
                # Create shifted sequences by padding and shifting
                batch_shft_cseqs = torch.cat([batch_cseqs[:, 1:], torch.zeros_like(batch_cseqs[:, :1])], dim=1)
            
            if 'shft_rseqs' in test_data:
                batch_shft_rseqs = to_tensor(test_data['shft_rseqs'][i:end_idx])
            else:
                batch_shft_rseqs = torch.cat([batch_rseqs[:, 1:], torch.zeros_like(batch_rseqs[:, :1])], dim=1)
            
            if 'masks' in test_data:
                batch_masks = to_tensor(test_data['masks'][i:end_idx])
            else:
                # Create masks based on non-zero entries
                batch_masks = (batch_cseqs != 0).long()
            
            try:
                # Get predictions
                outputs = model.forward_with_states(
                    q=batch_cseqs, r=batch_rseqs, qry=batch_shft_cseqs
                )
                predictions = outputs['predictions']
                
                # Mask predictions and targets
                valid_mask = batch_masks.bool()
                valid_predictions = predictions[valid_mask].cpu().numpy()
                valid_targets = batch_shft_rseqs[valid_mask].cpu().numpy()
                
                all_predictions.extend(valid_predictions.flatten())
                all_targets.extend(valid_targets.flatten())
                
                # Check consistency for each student in batch
                if 'mastery_states' in outputs:
                    mastery_states = outputs['mastery_states']
                    
                    for b in range(mastery_states.shape[0]):
                        student_mastery = mastery_states[b]
                        student_mask = batch_masks[b].bool()
                        
                        if student_mask.sum() <= 1:
                            continue
                        
                        valid_mastery = student_mastery[student_mask]
                        
                        # Check consistency
                        is_consistent = True
                        for concept in range(valid_mastery.shape[1]):
                            concept_progression = valid_mastery[:, concept]
                            
                            if not torch.all(concept_progression[1:] >= concept_progression[:-1]):
                                is_consistent = False
                                break
                            
                            if torch.any(concept_progression < 0) or torch.any(concept_progression > 1):
                                is_consistent = False
                                break
                        
                        consistency_total += 1
                        if is_consistent:
                            consistency_perfect += 1
                            
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {e}")
                continue
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    try:
        auc = roc_auc_score(all_targets, all_predictions)
    except ValueError:
        auc = 0.0
    
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
    
    # Create results
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
            "dataset": "assist2015",
            "test_sequences": num_sequences,
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
    
    print("\\n🎉 TEST EVALUATION COMPLETED!")
    print(f"📊 Test AUC: {auc:.4f}")
    print(f"✅ Consistency: {consistency_rate*100:.1f}%")
    print(f"🏆 Verdict: {verdict}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate cumulative mastery model on test dataset')
    parser.add_argument('--model_path', type=str,
                       default='saved_model/gainakt2_cumulative_mastery_quick_test/best_model.pth',
                       help='Path to the trained model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    evaluate_on_test_data_simple(args.model_path)


if __name__ == "__main__":
    main()