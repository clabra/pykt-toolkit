#!/usr/bin/env python3
"""
Consistency Requirements Validator for GainAKT2Monitored

This script validates that the model outputs satisfy the 4 educational consistency requirements
from "Step 2: Check Consistency Requirements" in the architecture document.
"""

import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

import torch
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
from extract_knowledge_evolution import KnowledgeStateEvolutionExtractor
from pykt.datasets import init_dataset4train

class ConsistencyValidator:
    """Validates educational consistency requirements for interpretable KT models."""
    
    def __init__(self):
        self.violations = {
            'monotonicity': [],
            'non_negative_gains': [],
            'mastery_bounds': [],
            'mastery_performance_correlation': None,
            'gain_performance_correlation': None
        }
        
    def validate_student_evolution(self, evolution_data, verbose=True):
        """
        Validate a single student's evolution against all consistency requirements.
        
        Args:
            evolution_data (dict): Output from KnowledgeStateEvolutionExtractor
            verbose (bool): Print detailed violation information
            
        Returns:
            dict: Comprehensive validation results
        """
        mastery_evolution = evolution_data['mastery_evolution']  # [seq_len, num_skills]
        learning_gains = evolution_data['learning_gains']        # [seq_len, num_skills]
        questions = evolution_data['questions']
        responses = evolution_data['responses']
        predictions = evolution_data['predictions']
        
        seq_len, num_skills = mastery_evolution.shape
        
        results = {
            'student_id': evolution_data['student_id'],
            'violations': {
                'monotonicity_count': 0,
                'negative_gains_count': 0,
                'mastery_bounds_count': 0
            },
            'correlations': {},
            'summary': {}
        }
        
        # 1. Check Monotonicity of Mastery
        if verbose:
            print(f"\\n🔍 Checking Monotonicity for {evolution_data['student_id']}...")
            
        monotonicity_violations = []
        for skill_id in range(num_skills):
            for t in range(1, seq_len):
                current_mastery = mastery_evolution[t, skill_id]
                previous_mastery = mastery_evolution[t-1, skill_id]
                
                if current_mastery < previous_mastery:
                    violation = {
                        'skill_id': skill_id,
                        'time_step': t,
                        'previous_mastery': previous_mastery,
                        'current_mastery': current_mastery,
                        'decrease': previous_mastery - current_mastery
                    }
                    monotonicity_violations.append(violation)
        
        results['violations']['monotonicity_count'] = len(monotonicity_violations)
        
        if verbose and len(monotonicity_violations) > 0:
            print(f"  ❌ {len(monotonicity_violations)} monotonicity violations found")
            for violation in monotonicity_violations[:3]:  # Show first 3
                print(f"     Skill {violation['skill_id']} at t={violation['time_step']}: "
                      f"{violation['previous_mastery']:.3f} → {violation['current_mastery']:.3f} "
                      f"(decrease: {violation['decrease']:.3f})")
        elif verbose:
            print(f"  ✅ No monotonicity violations")
            
        # 2. Check Non-Negative Learning Gains
        if verbose:
            print(f"\\n🔍 Checking Non-Negative Gains...")
            
        negative_gains = []
        total_gains = 0
        for t in range(seq_len):
            for skill_id in range(num_skills):
                gain = learning_gains[t, skill_id]
                total_gains += 1
                if gain < 0:
                    negative_gains.append({
                        'skill_id': skill_id,
                        'time_step': t,
                        'gain': gain
                    })
        
        results['violations']['negative_gains_count'] = len(negative_gains)
        negative_gains_pct = (len(negative_gains) / total_gains) * 100
        
        if verbose:
            print(f"  📊 {len(negative_gains)} / {total_gains} gains are negative ({negative_gains_pct:.1f}%)")
            if len(negative_gains) > 0:
                print(f"  ❌ Violations found (should be 0%)")
                avg_negative = np.mean([v['gain'] for v in negative_gains])
                print(f"     Average negative gain: {avg_negative:.3f}")
            else:
                print(f"  ✅ All gains are non-negative")
        
        # 3. Check Mastery Bounds (should be 0 to 1)
        if verbose:
            print(f"\\n🔍 Checking Mastery Bounds...")
            
        mastery_bounds_violations = []
        for t in range(seq_len):
            for skill_id in range(num_skills):
                mastery = mastery_evolution[t, skill_id]
                if mastery < 0 or mastery > 1:
                    mastery_bounds_violations.append({
                        'skill_id': skill_id,
                        'time_step': t,
                        'mastery': mastery,
                        'violation_type': 'below_zero' if mastery < 0 else 'above_one'
                    })
        
        results['violations']['mastery_bounds_count'] = len(mastery_bounds_violations)
        
        if verbose:
            print(f"  📊 {len(mastery_bounds_violations)} mastery values outside [0,1] bounds")
            if len(mastery_bounds_violations) > 0:
                above_one = sum(1 for v in mastery_bounds_violations if v['violation_type'] == 'above_one')
                below_zero = sum(1 for v in mastery_bounds_violations if v['violation_type'] == 'below_zero')
                print(f"  ❌ {above_one} above 1.0, {below_zero} below 0.0")
                if above_one > 0:
                    max_mastery = max(v['mastery'] for v in mastery_bounds_violations if v['violation_type'] == 'above_one')
                    print(f"     Maximum mastery: {max_mastery:.3f}")
            else:
                print(f"  ✅ All mastery values in valid range")
        
        # 4. Check Mastery-Performance Correlation
        if verbose:
            print(f"\\n🔍 Checking Mastery-Performance Correlation...")
            
        practiced_skills = np.unique(questions)
        skill_correlations = []
        
        for skill_id in practiced_skills:
            # Get interactions for this skill
            skill_interactions = []
            skill_masteries = []
            skill_predictions = []
            
            for t in range(seq_len):
                if questions[t] == skill_id:
                    skill_interactions.append(t)
                    skill_masteries.append(mastery_evolution[t, skill_id])
                    skill_predictions.append(predictions[t])
            
            if len(skill_masteries) > 1:
                corr, p_value = pearsonr(skill_masteries, skill_predictions)
                if not np.isnan(corr):
                    skill_correlations.append({
                        'skill_id': skill_id,
                        'correlation': corr,
                        'p_value': p_value,
                        'interactions_count': len(skill_masteries)
                    })
        
        if skill_correlations:
            avg_correlation = np.mean([s['correlation'] for s in skill_correlations])
            results['correlations']['mastery_performance'] = {
                'average': avg_correlation,
                'per_skill': skill_correlations
            }
            
            if verbose:
                print(f"  📊 Average mastery-performance correlation: {avg_correlation:.3f}")
                if avg_correlation > 0.3:
                    print(f"  ✅ Strong positive correlation (good)")
                elif avg_correlation > 0.1:
                    print(f"  ⚠️  Weak positive correlation")
                else:
                    print(f"  ❌ Poor or negative correlation (bad)")
        
        # 5. Check Gain-Performance Correlation
        if verbose:
            print(f"\\n🔍 Checking Gain-Performance Correlation...")
            
        interaction_gains = []
        interaction_responses = []
        
        for t in range(seq_len):
            skill_id = questions[t]
            if skill_id < num_skills:
                gain = learning_gains[t, skill_id]
                response = responses[t]
                interaction_gains.append(gain)
                interaction_responses.append(response)
        
        if len(interaction_gains) > 1:
            gain_corr, gain_p = pearsonr(interaction_gains, interaction_responses)
            results['correlations']['gain_performance'] = {
                'correlation': gain_corr,
                'p_value': gain_p
            }
            
            if verbose:
                print(f"  📊 Gain-performance correlation: {gain_corr:.3f}")
                if gain_corr > 0.1:
                    print(f"  ✅ Positive correlation (correct answers → higher gains)")
                else:
                    print(f"  ❌ Poor or negative correlation")
        
        # Summary
        total_violations = (results['violations']['monotonicity_count'] + 
                          results['violations']['negative_gains_count'] +
                          results['violations']['mastery_bounds_count'])
        
        results['summary'] = {
            'total_violations': total_violations,
            'monotonicity_violation_rate': results['violations']['monotonicity_count'] / (seq_len * num_skills),
            'negative_gains_rate': negative_gains_pct / 100,
            'mastery_bounds_violation_rate': results['violations']['mastery_bounds_count'] / (seq_len * num_skills),
            'avg_mastery_performance_corr': results['correlations'].get('mastery_performance', {}).get('average', 0.0),
            'gain_performance_corr': results['correlations'].get('gain_performance', {}).get('correlation', 0.0)
        }
        
        if verbose:
            print(f"\\n📋 CONSISTENCY VALIDATION SUMMARY")
            print(f"   Total Violations: {total_violations}")
            print(f"   Monotonicity Rate: {results['summary']['monotonicity_violation_rate']:.1%}")
            print(f"   Negative Gains Rate: {results['summary']['negative_gains_rate']:.1%}")
            print(f"   Bounds Violations Rate: {results['summary']['mastery_bounds_violation_rate']:.1%}")
            print(f"   Mastery-Perf Correlation: {results['summary']['avg_mastery_performance_corr']:.3f}")
            print(f"   Gain-Perf Correlation: {results['summary']['gain_performance_corr']:.3f}")
        
        return results
    
    def batch_validate(self, num_students=10):
        """Validate consistency across multiple students."""
        
        print("="*60)
        print("BATCH CONSISTENCY VALIDATION")
        print("="*60)
        
        # Setup dataset and extractor (same as other scripts)
        dataset_name = "assist2015"
        model_name = "gainakt2"
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
            dataset_name, model_name, data_config, 0, 32
        )
        
        model_config = {
            'num_c': 100,
            'seq_len': 200,
            'd_model': 512,
            'n_heads': 4,
            'num_encoder_blocks': 4,
            'd_ff': 512,
            'dropout': 0.4,
            'emb_type': 'qid',
            'non_negative_loss_weight': 0.485828,
            'consistency_loss_weight': 0.173548,
            'monitor_frequency': 25
        }
        
        extractor = KnowledgeStateEvolutionExtractor(
            model_path="saved_model/gainakt2_enhanced_auc_0.7253/model.pth",
            model_config=model_config
        )
        
        # Validate multiple students
        all_results = []
        student_count = 0
        
        for batch in valid_loader:
            batch_size = batch['cseqs'].size(0)
            
            for i in range(min(batch_size, num_students - student_count)):
                student_id = f"validation_student_{student_count:03d}"
                questions = batch['cseqs'][i]
                responses = batch['rseqs'][i]
                mask = batch['masks'][i]
                
                # Only analyze students with reasonable interactions
                valid_length = mask.sum().item()
                if valid_length > 5:
                    questions = questions[:valid_length]
                    responses = responses[:valid_length]
                    
                    # Extract evolution
                    evolution_data = extractor.extract_student_journey(
                        questions, responses, student_id=student_id
                    )
                    
                    # Validate consistency
                    validation_results = self.validate_student_evolution(
                        evolution_data, verbose=False  # Quiet for batch processing
                    )
                    all_results.append(validation_results)
                    
                    student_count += 1
                    if student_count % 5 == 0:
                        print(f"Validated {student_count} students...")
                    
                    if student_count >= num_students:
                        break
            
            if student_count >= num_students:
                break
        
        # Generate batch summary
        self.generate_batch_summary(all_results)
        return all_results
    
    def generate_batch_summary(self, all_results):
        """Generate summary statistics across all validated students."""
        
        print(f"\\n" + "="*60)
        print("BATCH VALIDATION SUMMARY")
        print("="*60)
        
        # Aggregate statistics
        total_violations = [r['summary']['total_violations'] for r in all_results]
        monotonicity_rates = [r['summary']['monotonicity_violation_rate'] for r in all_results]
        negative_gains_rates = [r['summary']['negative_gains_rate'] for r in all_results]
        bounds_violation_rates = [r['summary']['mastery_bounds_violation_rate'] for r in all_results]
        mastery_corrs = [r['summary']['avg_mastery_performance_corr'] for r in all_results]
        gain_corrs = [r['summary']['gain_performance_corr'] for r in all_results]
        
        print(f"📊 Validation Results Across {len(all_results)} Students:")
        print(f"")
        print(f"1. 📉 MONOTONICITY VIOLATIONS")
        print(f"   Average Rate: {np.mean(monotonicity_rates):.1%}")
        print(f"   Range: {np.min(monotonicity_rates):.1%} - {np.max(monotonicity_rates):.1%}")
        print(f"   Target: 0% (mastery should never decrease)")
        
        print(f"")
        print(f"2. ➖ NEGATIVE GAINS VIOLATIONS") 
        print(f"   Average Rate: {np.mean(negative_gains_rates):.1%}")
        print(f"   Range: {np.min(negative_gains_rates):.1%} - {np.max(negative_gains_rates):.1%}")
        print(f"   Target: 0% (learning gains should be ≥ 0)")
        
        print(f"")
        print(f"3. 🎯 MASTERY BOUNDS VIOLATIONS")
        print(f"   Average Rate: {np.mean(bounds_violation_rates):.1%}")
        print(f"   Range: {np.min(bounds_violation_rates):.1%} - {np.max(bounds_violation_rates):.1%}")
        print(f"   Target: 0% (mastery should be in [0,1])")
        
        print(f"")
        print(f"4. 📈 MASTERY-PERFORMANCE CORRELATION")
        print(f"   Average: {np.mean(mastery_corrs):.3f}")
        print(f"   Range: {np.min(mastery_corrs):.3f} - {np.max(mastery_corrs):.3f}")
        print(f"   Target: > 0.3 (higher mastery should predict better performance)")
        
        print(f"")
        print(f"5. 🎓 GAIN-PERFORMANCE CORRELATION")
        print(f"   Average: {np.mean(gain_corrs):.3f}")
        print(f"   Range: {np.min(gain_corrs):.3f} - {np.max(gain_corrs):.3f}")
        print(f"   Target: > 0.1 (correct answers should yield higher gains)")
        
        # Overall assessment
        print(f"\\n🏆 OVERALL ASSESSMENT:")
        
        critical_issues = []
        if np.mean(monotonicity_rates) > 0.01:  # More than 1%
            critical_issues.append("High monotonicity violations")
        if np.mean(negative_gains_rates) > 0.01:  # More than 1%
            critical_issues.append("High negative gains rate")
        if np.mean(bounds_violation_rates) > 0.01:  # More than 1%
            critical_issues.append("High bounds violations")
        if np.mean(mastery_corrs) < 0.1:
            critical_issues.append("Poor mastery-performance correlation")
        if np.mean(gain_corrs) < 0.05:
            critical_issues.append("Poor gain-performance correlation")
        
        if len(critical_issues) == 0:
            print("   ✅ MODEL MEETS ALL CONSISTENCY REQUIREMENTS")
        else:
            print("   ❌ MODEL FAILS CONSISTENCY REQUIREMENTS:")
            for issue in critical_issues:
                print(f"      • {issue}")


def main():
    """Run consistency validation on the trained model."""
    
    validator = ConsistencyValidator()
    
    # Option 1: Single student detailed validation
    print("🔍 Running detailed validation on one student...")
    
    # Get one student for detailed analysis
    dataset_name = "assist2015"
    model_name = "gainakt2"
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
        dataset_name, model_name, data_config, 0, 32
    )
    
    # Get first valid student
    for batch in valid_loader:
        for i in range(batch['cseqs'].size(0)):
            questions = batch['cseqs'][i]
            responses = batch['rseqs'][i]
            mask = batch['masks'][i]
            
            valid_length = mask.sum().item()
            if valid_length > 5:
                questions = questions[:valid_length]
                responses = responses[:valid_length]
                break
        break
    
    # Extract evolution and validate
    model_config = {
        'num_c': 100,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 4,
        'num_encoder_blocks': 4,
        'd_ff': 512,
        'dropout': 0.4,
        'emb_type': 'qid',
        'non_negative_loss_weight': 0.485828,
        'consistency_loss_weight': 0.173548,
        'monitor_frequency': 25
    }
    
    extractor = KnowledgeStateEvolutionExtractor(
        model_path="saved_model/gainakt2_enhanced_auc_0.7253/model.pth",
        model_config=model_config
    )
    
    evolution_data = extractor.extract_student_journey(
        questions, responses, student_id="detailed_validation_student"
    )
    
    detailed_results = validator.validate_student_evolution(evolution_data, verbose=True)
    
    # Option 2: Batch validation across multiple students
    print("\\n" + "="*60)
    print("🔍 Running batch validation across multiple students...")
    batch_results = validator.batch_validate(num_students=20)
    
    return detailed_results, batch_results


if __name__ == "__main__":
    print("="*60)
    print("GAINAKT2 CONSISTENCY REQUIREMENTS VALIDATION")
    print("="*60)
    
    detailed, batch = main()
    
    print("\\n🎯 NEXT STEPS:")
    print("1. Address any critical violations found")
    print("2. Adjust auxiliary loss weights if needed")
    print("3. Re-train model with stronger consistency constraints")
    print("4. Validate interpretability claims with these metrics")