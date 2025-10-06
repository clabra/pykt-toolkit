#!/usr/bin/env python3
"""
Validate consistency of the cumulative mastery model.
This tests the improved architecture that should have perfect monotonicity.
"""

import os
import sys
import torch
import numpy as np
import json
from collections import defaultdict
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_monitored import create_monitored_model


def validate_cumulative_mastery_consistency():
    """Validate the consistency of the cumulative mastery model."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("VALIDATING CUMULATIVE MASTERY MODEL CONSISTENCY")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load the test model with cumulative mastery
    model_path = "saved_model/gainakt2_cumulative_mastery_test/model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run test_monotonicity_fix.py first")
        return False
    
    logger.info(f"Loading cumulative mastery model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Create model
    model = create_monitored_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("✓ Cumulative mastery model loaded successfully")
    
    # Load dataset
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
    
    train_loader, valid_loader = init_dataset4train(
        dataset_name, model_name, data_config, fold, batch_size
    )
    
    # Validation results
    results = {
        'monotonicity_violations': [],
        'negative_gains': [],
        'mastery_bounds_violations': [],
        'mastery_performance_correlations': [],
        'gain_performance_correlations': []
    }
    
    num_students_to_check = 50  # Test on 50 students for speed
    logger.info(f"Validating consistency on {num_students_to_check} students...")
    
    student_count = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            if student_count >= num_students_to_check:
                break
                
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Get model outputs with states
            outputs = model.forward_with_states(
                q=questions, r=responses, qry=questions_shifted
            )
            
            skill_mastery = outputs['projected_mastery']
            skill_gains = outputs['projected_gains']
            
            batch_size_actual = questions.size(0)
            
            for i in range(batch_size_actual):
                if student_count >= num_students_to_check:
                    break
                    
                # Get valid sequence for this student
                student_mask = mask[i].bool()
                student_mastery = skill_mastery[i][student_mask]  # Shape: (seq_len, num_c)
                student_gains = skill_gains[i][student_mask]      # Shape: (seq_len, num_c)
                student_performance = responses_shifted[i][student_mask].float()
                
                seq_len = student_mastery.size(0)
                if seq_len < 2:  # Need at least 2 steps for monotonicity
                    continue
                
                # Convert to numpy for analysis
                mastery_np = student_mastery.cpu().numpy()  # (seq_len, num_c)
                gains_np = student_gains.cpu().numpy()      # (seq_len, num_c) 
                performance_np = student_performance.cpu().numpy()  # (seq_len,)
                
                # For analysis, aggregate across concepts (mean mastery/gains per timestep)
                mean_mastery = np.mean(mastery_np, axis=1)  # (seq_len,)
                mean_gains = np.mean(gains_np, axis=1)      # (seq_len,)
                
                # Check monotonicity violations (aggregated)
                monotonicity_violations = 0
                for t in range(1, seq_len):
                    if mean_mastery[t] < mean_mastery[t-1]:
                        monotonicity_violations += 1
                
                monotonicity_violation_rate = monotonicity_violations / (seq_len - 1)
                results['monotonicity_violations'].append(monotonicity_violation_rate)
                
                # Check negative gains (any negative gain across concepts)
                negative_gain_count = np.sum(gains_np < 0)
                total_gain_entries = gains_np.size
                negative_gain_rate = negative_gain_count / total_gain_entries
                results['negative_gains'].append(negative_gain_rate)
                
                # Check mastery bounds violations (any violation across concepts)
                bounds_violations = np.sum((mastery_np < 0) | (mastery_np > 1))
                total_mastery_entries = mastery_np.size
                bounds_violation_rate = bounds_violations / total_mastery_entries
                results['mastery_bounds_violations'].append(bounds_violation_rate)
                
                # Compute correlations if we have enough data points
                if seq_len >= 3:
                    mastery_perf_corr = np.corrcoef(mean_mastery, performance_np)[0, 1]
                    if not np.isnan(mastery_perf_corr):
                        results['mastery_performance_correlations'].append(mastery_perf_corr)
                    
                    gain_perf_corr = np.corrcoef(mean_gains, performance_np)[0, 1]
                    if not np.isnan(gain_perf_corr):
                        results['gain_performance_correlations'].append(gain_perf_corr)
                
                student_count += 1
                
                if student_count % 10 == 0:
                    logger.info(f"  Processed {student_count} students...")
    
    # Compute summary statistics
    summary = {}
    
    # Monotonicity violations
    monotonicity_viols = np.array(results['monotonicity_violations'])
    summary['monotonicity_violation_rate'] = np.mean(monotonicity_viols)
    summary['students_with_monotonicity_violations'] = np.sum(monotonicity_viols > 0) / len(monotonicity_viols)
    
    # Negative gains
    negative_gains = np.array(results['negative_gains'])
    summary['negative_gain_rate'] = np.mean(negative_gains)
    summary['students_with_negative_gains'] = np.sum(negative_gains > 0) / len(negative_gains)
    
    # Mastery bounds violations
    bounds_viols = np.array(results['mastery_bounds_violations'])
    summary['mastery_bounds_violation_rate'] = np.mean(bounds_viols)
    summary['students_with_bounds_violations'] = np.sum(bounds_viols > 0) / len(bounds_viols)
    
    # Correlations
    mastery_corrs = np.array(results['mastery_performance_correlations'])
    if len(mastery_corrs) > 0:
        summary['mean_mastery_performance_correlation'] = np.mean(mastery_corrs)
        summary['positive_mastery_correlations'] = np.sum(mastery_corrs > 0) / len(mastery_corrs)
    else:
        summary['mean_mastery_performance_correlation'] = 0.0
        summary['positive_mastery_correlations'] = 0.0
    
    gain_corrs = np.array(results['gain_performance_correlations'])
    if len(gain_corrs) > 0:
        summary['mean_gain_performance_correlation'] = np.mean(gain_corrs)
        summary['positive_gain_correlations'] = np.sum(gain_corrs > 0) / len(gain_corrs)
    else:
        summary['mean_gain_performance_correlation'] = 0.0
        summary['positive_gain_correlations'] = 0.0
    
    # Report results
    logger.info("\\n" + "="*80)
    logger.info("CUMULATIVE MASTERY MODEL CONSISTENCY VALIDATION RESULTS")
    logger.info("="*80)
    
    logger.info(f"Students analyzed: {student_count}")
    logger.info("\\n📈 MONOTONICITY REQUIREMENT:")
    logger.info(f"  Average monotonicity violation rate: {summary['monotonicity_violation_rate']:.1%}")
    logger.info(f"  Students with violations: {summary['students_with_monotonicity_violations']:.1%}")
    
    logger.info("\\n➕ NON-NEGATIVE GAINS REQUIREMENT:")
    logger.info(f"  Average negative gain rate: {summary['negative_gain_rate']:.1%}")
    logger.info(f"  Students with negative gains: {summary['students_with_negative_gains']:.1%}")
    
    logger.info("\\n📊 MASTERY BOUNDS REQUIREMENT:")
    logger.info(f"  Average bounds violation rate: {summary['mastery_bounds_violation_rate']:.1%}")
    logger.info(f"  Students with bounds violations: {summary['students_with_bounds_violations']:.1%}")
    
    logger.info("\\n🔗 PERFORMANCE CORRELATION REQUIREMENTS:")
    logger.info(f"  Mean mastery-performance correlation: {summary['mean_mastery_performance_correlation']:.3f}")
    logger.info(f"  Positive mastery correlations: {summary['positive_mastery_correlations']:.1%}")
    logger.info(f"  Mean gain-performance correlation: {summary['mean_gain_performance_correlation']:.3f}")
    logger.info(f"  Positive gain correlations: {summary['positive_gain_correlations']:.1%}")
    
    # Final assessment
    logger.info("\\n" + "="*80)
    logger.info("FINAL ASSESSMENT:")
    
    all_requirements_met = True
    
    if summary['monotonicity_violation_rate'] == 0.0:
        logger.info("✅ MONOTONICITY: PERFECT (0% violations)")
    else:
        logger.info(f"❌ MONOTONICITY: {summary['monotonicity_violation_rate']:.1%} violations")
        all_requirements_met = False
    
    if summary['negative_gain_rate'] == 0.0:
        logger.info("✅ NON-NEGATIVE GAINS: PERFECT (0% violations)")
    else:
        logger.info(f"❌ NON-NEGATIVE GAINS: {summary['negative_gain_rate']:.1%} violations")
        all_requirements_met = False
    
    if summary['mastery_bounds_violation_rate'] == 0.0:
        logger.info("✅ MASTERY BOUNDS: PERFECT (0% violations)")
    else:
        logger.info(f"❌ MASTERY BOUNDS: {summary['mastery_bounds_violation_rate']:.1%} violations")
        all_requirements_met = False
    
    if summary['mean_mastery_performance_correlation'] > 0.3:
        logger.info(f"✅ MASTERY-PERFORMANCE CORRELATION: STRONG ({summary['mean_mastery_performance_correlation']:.3f})")
    else:
        logger.info(f"⚠️  MASTERY-PERFORMANCE CORRELATION: WEAK ({summary['mean_mastery_performance_correlation']:.3f})")
    
    if summary['mean_gain_performance_correlation'] > 0.3:
        logger.info(f"✅ GAIN-PERFORMANCE CORRELATION: STRONG ({summary['mean_gain_performance_correlation']:.3f})")
    else:
        logger.info(f"⚠️  GAIN-PERFORMANCE CORRELATION: WEAK ({summary['mean_gain_performance_correlation']:.3f})")
    
    logger.info("="*80)
    
    if all_requirements_met:
        logger.info("🎉 ALL EDUCATIONAL CONSISTENCY REQUIREMENTS MET!")
        logger.info("✅ The cumulative mastery approach SUCCESSFULLY eliminated all violations!")
    else:
        logger.info("⚠️  Some requirements still have violations")
    
    # Save results
    summary['validation_timestamp'] = datetime.now().isoformat()
    summary['model_type'] = 'gainakt2_cumulative_mastery'
    summary['students_analyzed'] = student_count
    
    results_file = f"cumulative_mastery_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\\n📄 Detailed results saved to: {results_file}")
    
    return all_requirements_met


if __name__ == "__main__":
    success = validate_cumulative_mastery_consistency()
    
    if success:
        print("\\n🎯 VALIDATION SUCCESSFUL!")
        print("The cumulative mastery model meets all educational consistency requirements!")
    else:
        print("\\n⚠️  Validation incomplete - check results above")