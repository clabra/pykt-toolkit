#!/usr/bin/env python3
"""
Quick Performance Benchmark for GainAKT2Monitored with Cumulative Mastery.
Runs a simple performance comparison between cumulative mastery and baseline models.
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

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model


def benchmark_model(model_path: str, output_dir: str = "benchmark_results"):
    """Run a quick performance benchmark on the cumulative mastery model."""
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("🚀 STARTING PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    # Load data
    dataset_name = "assist2015"
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
    
    train_loader, valid_loader = init_dataset4train(
        dataset_name, "gainakt2", data_config, 0, 32
    )
    
    logger.info("✓ Dataset loaded successfully")
    
    # Load cumulative mastery model
    logger.info(f"Loading cumulative mastery model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = create_monitored_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("✓ Cumulative mastery model loaded")
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    
    start_time = time.time()
    all_predictions = []
    all_targets = []
    consistency_perfect = 0
    consistency_total = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='Evaluation'):
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
            
            # Extract consistency info
            skill_mastery = outputs['projected_mastery']
            skill_gains = outputs['projected_gains']
            
            # Performance metrics
            valid_mask = mask.bool()
            valid_predictions = predictions[valid_mask]
            valid_targets = responses_shifted[valid_mask].float()
            
            all_predictions.extend(valid_predictions.cpu().numpy())
            all_targets.extend(valid_targets.cpu().numpy())
            
            # Quick consistency check
            batch_size = skill_mastery.size(0)
            for i in range(batch_size):
                student_mask = mask[i].bool()
                student_mastery = skill_mastery[i][student_mask]
                student_gains = skill_gains[i][student_mask]
                
                if student_mastery.size(0) < 2:
                    continue
                
                consistency_total += 1
                
                # Convert to numpy
                mastery_np = student_mastery.cpu().numpy()
                gains_np = student_gains.cpu().numpy()
                
                # Check violations
                monotonic = True
                seq_len = mastery_np.shape[0]
                mean_mastery = np.mean(mastery_np, axis=1)
                
                for t in range(1, seq_len):
                    if mean_mastery[t] < mean_mastery[t-1] - 1e-6:
                        monotonic = False
                        break
                
                bounds_ok = np.all((mastery_np >= -1e-6) & (mastery_np <= 1 + 1e-6))
                gains_ok = np.all(gains_np >= -1e-6)
                
                if monotonic and bounds_ok and gains_ok:
                    consistency_perfect += 1
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    auc = roc_auc_score(all_targets, all_predictions)
    accuracy = accuracy_score(all_targets, np.round(all_predictions))
    consistency_rate = consistency_perfect / max(consistency_total, 1)
    
    # Generate report
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'performance_metrics': {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'total_predictions': len(all_predictions),
            'evaluation_time': evaluation_time
        },
        'consistency_metrics': {
            'perfect_consistency_rate': float(consistency_rate),
            'students_analyzed': consistency_total,
            'perfect_students': consistency_perfect
        },
        'assessment': {
            'performance_grade': 'A' if auc > 0.72 else 'B' if auc > 0.68 else 'C',
            'consistency_grade': 'A+' if consistency_rate > 0.95 else 'A' if consistency_rate > 0.9 else 'B',
            'overall_verdict': 'EXCELLENT' if (auc > 0.7 and consistency_rate > 0.95) else 'GOOD' if (auc > 0.65 and consistency_rate > 0.8) else 'NEEDS_IMPROVEMENT'
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\\n" + "=" * 60)
    logger.info("📊 BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"🎯 Performance AUC: {auc:.4f} (Grade: {results['assessment']['performance_grade']})")
    logger.info(f"🎯 Accuracy: {accuracy:.4f}")
    logger.info(f"✅ Perfect Consistency: {consistency_rate:.1%} ({consistency_perfect}/{consistency_total} students)")
    logger.info(f"🏆 Overall Assessment: {results['assessment']['overall_verdict']}")
    logger.info(f"⏱️  Evaluation Time: {evaluation_time:.2f} seconds")
    logger.info(f"📄 Results saved: {results_file}")
    logger.info("=" * 60)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick performance benchmark')
    parser.add_argument('--model_path', type=str, 
                       default='saved_model/gainakt2_cumulative_mastery_test/model.pth',
                       help='Path to cumulative mastery model')
    parser.add_argument('--output_dir', type=str, 
                       default=f"quick_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("Available models:")
        for root, dirs, files in os.walk('saved_model'):
            for file in files:
                if file.endswith('.pth'):
                    print(f"  📦 {os.path.join(root, file)}")
        return
    
    results = benchmark_model(args.model_path, args.output_dir)
    
    # Final summary
    print("\\n🎉 BENCHMARK COMPLETED!")
    print(f"📊 AUC: {results['performance_metrics']['auc']:.4f}")
    print(f"✅ Consistency: {results['consistency_metrics']['perfect_consistency_rate']:.1%}")
    print(f"🏆 Verdict: {results['assessment']['overall_verdict']}")


if __name__ == "__main__":
    main()