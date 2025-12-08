"""
Validate IRT Reference Targets

Investigates why IRT alignment is failing by analyzing:
1. M_ref correlation with actual student responses
2. IRT calibration quality (β_IRT vs empirical difficulty)
3. θ_IRT scale and distribution
4. Rasch model assumptions (unidimensionality, constant ability)

Usage:
    python examples/validate_irt_targets.py \\
        --dataset assist2015 \\
        --fold 0 \\
        --targets data/assist2015/irt_dynamic_targets_fold0.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pykt.datasets import init_dataset4train


def load_irt_targets(targets_path):
    """Load IRT targets file."""
    print(f"Loading IRT targets from: {targets_path}")
    with open(targets_path, 'rb') as f:
        data = pickle.load(f)
    
    # Detect format
    is_dynamic = 'theta_trajectories' in data
    
    if is_dynamic:
        beta_irt = data['skill_difficulties']
        theta_data = data['theta_trajectories']
        m_ref_data = data['m_ref_trajectories']
        format_name = "Dynamic IRT"
    else:
        beta_irt = data['skill_difficulties']
        theta_data = data['student_abilities']
        m_ref_data = data['reference_predictions']
        format_name = "Static IRT"
    
    print(f"Format: {format_name}")
    print(f"- Skills: {len(beta_irt)}")
    print(f"- Students: {len(theta_data)}")
    print(f"- M_ref sequences: {len(m_ref_data)}")
    print()
    
    return {
        'beta_irt': beta_irt,
        'theta_data': theta_data,
        'm_ref_data': m_ref_data,
        'is_dynamic': is_dynamic,
        'metadata': data.get('metadata', {})
    }


def analyze_m_ref_correlation(train_loader, valid_loader, irt_targets):
    """
    Analyze correlation between M_ref (IRT predictions) and actual responses.
    
    If correlation is poor, IRT formula doesn't fit the data.
    """
    print("=" * 80)
    print("ANALYSIS 1: M_ref Correlation with Actual Responses")
    print("=" * 80)
    
    m_ref_data = irt_targets['m_ref_data']
    is_dynamic = irt_targets['is_dynamic']
    
    # Collect predictions and actuals
    all_m_ref = []
    all_responses = []
    
    for split_name, loader in [('train', train_loader), ('valid', valid_loader)]:
        for batch in loader:
            uids = batch['uid']
            responses = batch['responses'].numpy()  # [B, L]
            
            for i, uid in enumerate(uids):
                uid_key = uid.item() if hasattr(uid, 'item') else int(uid)
                
                if uid_key in m_ref_data:
                    m_ref_seq = m_ref_data[uid_key]
                    response_seq = responses[i]
                    
                    # Get valid positions (not padding)
                    seq_len = min(len(m_ref_seq), len(response_seq))
                    
                    for t in range(seq_len):
                        if response_seq[t] >= 0:  # Valid response (not padding)
                            all_m_ref.append(m_ref_seq[t])
                            all_responses.append(response_seq[t])
    
    all_m_ref = np.array(all_m_ref)
    all_responses = np.array(all_responses)
    
    # Compute correlations
    pearson_corr, _ = pearsonr(all_m_ref, all_responses)
    spearman_corr, _ = spearmanr(all_m_ref, all_responses)
    
    # Compute calibration metrics
    mae = np.mean(np.abs(all_m_ref - all_responses))
    rmse = np.sqrt(np.mean((all_m_ref - all_responses) ** 2))
    
    # Compute AUC-like metric (predicted probability vs binary outcome)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_responses, all_m_ref)
    except:
        auc = np.nan
    
    print(f"Total interactions: {len(all_responses):,}")
    print()
    print("CORRELATION METRICS:")
    print(f"  Pearson correlation:  {pearson_corr:.4f}")
    print(f"  Spearman correlation: {spearman_corr:.4f}")
    print()
    print("CALIBRATION METRICS:")
    print(f"  MAE (Mean Absolute Error):  {mae:.4f}")
    print(f"  RMSE (Root Mean Squared):   {rmse:.4f}")
    print(f"  AUC (Predictive):           {auc:.4f}")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    if pearson_corr > 0.7:
        print("  ✅ EXCELLENT: M_ref highly correlates with responses")
        print("     IRT formula fits data well")
    elif pearson_corr > 0.5:
        print("  ⚠️  MODERATE: M_ref partially correlates with responses")
        print("     IRT captures some patterns but not all")
    else:
        print("  ❌ POOR: M_ref weakly correlates with responses")
        print("     IRT formula does NOT fit this dataset")
        print("     → Root cause of alignment failure")
    
    print()
    print(f"M_ref statistics:")
    print(f"  Mean: {np.mean(all_m_ref):.4f}")
    print(f"  Std:  {np.std(all_m_ref):.4f}")
    print(f"  Range: [{np.min(all_m_ref):.4f}, {np.max(all_m_ref):.4f}]")
    print()
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'mae': mae,
        'rmse': rmse,
        'auc': auc,
        'n_interactions': len(all_responses)
    }


def analyze_beta_calibration(train_loader, valid_loader, irt_targets):
    """
    Compare IRT-calibrated β with empirical difficulty.
    
    If they disagree, IRT calibration may be incorrect.
    """
    print("=" * 80)
    print("ANALYSIS 2: β_IRT Calibration Quality")
    print("=" * 80)
    
    beta_irt = irt_targets['beta_irt']
    
    # Compute empirical difficulty (success rate per skill)
    skill_correct = defaultdict(int)
    skill_total = defaultdict(int)
    
    for split_name, loader in [('train', train_loader), ('valid', valid_loader)]:
        for batch in loader:
            questions = batch['questions'].numpy()  # [B, L]
            responses = batch['responses'].numpy()  # [B, L]
            
            for i in range(len(questions)):
                for t in range(len(questions[i])):
                    q = questions[i, t]
                    r = responses[i, t]
                    
                    if r >= 0:  # Valid response
                        skill_total[q] += 1
                        skill_correct[q] += r
    
    # Compute empirical difficulty (higher = easier)
    # Convert to IRT scale (higher = harder) via logit
    empirical_difficulties = {}
    for skill in range(len(beta_irt)):
        if skill in skill_total and skill_total[skill] > 0:
            success_rate = skill_correct[skill] / skill_total[skill]
            # Clip to avoid log(0)
            success_rate = np.clip(success_rate, 0.01, 0.99)
            # logit transform: harder skills have negative logit (low success)
            empirical_difficulties[skill] = -np.log(success_rate / (1 - success_rate))
        else:
            empirical_difficulties[skill] = 0.0
    
    # Compare IRT vs empirical
    skills_with_data = [s for s in range(len(beta_irt)) if s in skill_total and skill_total[s] > 10]
    
    beta_irt_values = [beta_irt[s] for s in skills_with_data]
    empirical_values = [empirical_difficulties[s] for s in skills_with_data]
    
    if len(beta_irt_values) > 0:
        corr, _ = pearsonr(beta_irt_values, empirical_values)
        mae = np.mean(np.abs(np.array(beta_irt_values) - np.array(empirical_values)))
        
        print(f"Skills analyzed: {len(skills_with_data)} (with ≥10 observations)")
        print()
        print(f"β_IRT vs Empirical Difficulty:")
        print(f"  Correlation: {corr:.4f}")
        print(f"  MAE: {mae:.4f}")
        print()
        print(f"β_IRT statistics:")
        print(f"  Mean: {np.mean(beta_irt_values):.4f}")
        print(f"  Std:  {np.std(beta_irt_values):.4f}")
        print(f"  Range: [{np.min(beta_irt_values):.4f}, {np.max(beta_irt_values):.4f}]")
        print()
        print(f"Empirical difficulty statistics:")
        print(f"  Mean: {np.mean(empirical_values):.4f}")
        print(f"  Std:  {np.std(empirical_values):.4f}")
        print(f"  Range: [{np.min(empirical_values):.4f}, {np.max(empirical_values):.4f}]")
        print()
        
        # Interpretation
        print("INTERPRETATION:")
        if corr > 0.8:
            print("  ✅ EXCELLENT: β_IRT matches empirical difficulty well")
        elif corr > 0.5:
            print("  ⚠️  MODERATE: β_IRT partially matches empirical difficulty")
            print("     May need recalibration")
        else:
            print("  ❌ POOR: β_IRT does NOT match empirical difficulty")
            print("     IRT calibration likely incorrect")
        print()
        
        return {
            'correlation': corr,
            'mae': mae,
            'n_skills': len(skills_with_data)
        }
    else:
        print("⚠️  Not enough data to analyze β calibration")
        return {}


def analyze_theta_distribution(irt_targets):
    """
    Analyze θ_IRT distribution and scale.
    
    Check if abilities are well-distributed or collapsed.
    """
    print("=" * 80)
    print("ANALYSIS 3: θ_IRT Distribution")
    print("=" * 80)
    
    theta_data = irt_targets['theta_data']
    is_dynamic = irt_targets['is_dynamic']
    
    if is_dynamic:
        # Dynamic: analyze final θ values (last timestep)
        theta_final = []
        theta_all = []
        for uid, trajectory in theta_data.items():
            if len(trajectory) > 0:
                theta_final.append(trajectory[-1])  # Last timestep
                theta_all.extend(trajectory)  # All timesteps
        
        theta_final = np.array(theta_final)
        theta_all = np.array(theta_all)
        
        print("Dynamic θ trajectories:")
        print(f"  Students: {len(theta_data)}")
        print()
        print("Final θ (last timestep) statistics:")
        print(f"  Mean: {np.mean(theta_final):.4f}")
        print(f"  Std:  {np.std(theta_final):.4f}")
        print(f"  Range: [{np.min(theta_final):.4f}, {np.max(theta_final):.4f}]")
        print()
        print("All θ (all timesteps) statistics:")
        print(f"  Mean: {np.mean(theta_all):.4f}")
        print(f"  Std:  {np.std(theta_all):.4f}")
        print(f"  Range: [{np.min(theta_all):.4f}, {np.max(theta_all):.4f}]")
        print()
        
        # Check for scale collapse
        if np.std(theta_final) < 0.5:
            print("⚠️  WARNING: θ values collapsed (std < 0.5)")
            print("   Abilities not sufficiently differentiated")
        else:
            print("✅ θ values well-distributed (std ≥ 0.5)")
        
        return {
            'mean_final': np.mean(theta_final),
            'std_final': np.std(theta_final),
            'mean_all': np.mean(theta_all),
            'std_all': np.std(theta_all)
        }
    else:
        # Static: single θ per student
        theta_values = np.array([theta_data[uid] for uid in theta_data])
        
        print("Static θ values:")
        print(f"  Students: {len(theta_values)}")
        print()
        print("θ statistics:")
        print(f"  Mean: {np.mean(theta_values):.4f}")
        print(f"  Std:  {np.std(theta_values):.4f}")
        print(f"  Range: [{np.min(theta_values):.4f}, {np.max(theta_values):.4f}]")
        print()
        
        # Check for scale collapse
        if np.std(theta_values) < 0.5:
            print("⚠️  WARNING: θ values collapsed (std < 0.5)")
            print("   Abilities not sufficiently differentiated")
        else:
            print("✅ θ values well-distributed (std ≥ 0.5)")
        
        return {
            'mean': np.mean(theta_values),
            'std': np.std(theta_values)
        }


def analyze_rasch_assumptions(train_loader, valid_loader, irt_targets):
    """
    Test Rasch model assumptions:
    1. Unidimensionality: Single ability explains performance
    2. Constant ability: θ doesn't change during sequence (for static IRT)
    3. Local independence: Responses conditionally independent given θ, β
    """
    print("=" * 80)
    print("ANALYSIS 4: Rasch Model Assumptions")
    print("=" * 80)
    
    is_dynamic = irt_targets['is_dynamic']
    
    # Test 1: Check if performance correlates with sequence position
    # If students learn, Rasch assumption of constant ability violated
    print("Test 1: Constant Ability Assumption")
    print("-" * 40)
    
    position_performance = defaultdict(list)
    
    for batch in train_loader:
        responses = batch['responses'].numpy()  # [B, L]
        
        for i in range(len(responses)):
            for t in range(len(responses[i])):
                r = responses[i, t]
                if r >= 0:
                    position_performance[t].append(r)
    
    # Compute average performance at each position
    avg_performance = {}
    for pos, values in position_performance.items():
        if len(values) > 10:
            avg_performance[pos] = np.mean(values)
    
    if len(avg_performance) > 5:
        positions = sorted(avg_performance.keys())[:50]  # First 50 positions
        performance_values = [avg_performance[p] for p in positions]
        
        # Check if performance increases over time (learning effect)
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(positions, performance_values)
        
        print(f"  Performance vs Position slope: {slope:.6f}")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.4e}")
        print()
        
        if p_value < 0.05 and slope > 0.001:
            print("  ❌ VIOLATION: Significant learning effect detected")
            print("     Static IRT assumes constant ability - violated")
            if is_dynamic:
                print("     ✅ Using dynamic IRT (handles learning)")
            else:
                print("     → Consider switching to dynamic IRT")
        else:
            print("  ✅ PASS: No significant learning effect")
            print("     Constant ability assumption reasonable")
    
    print()
    
    # Test 2: Skill-specific performance variance
    print("Test 2: Skill Consistency")
    print("-" * 40)
    
    skill_performance = defaultdict(list)
    
    for batch in train_loader:
        questions = batch['questions'].numpy()  # [B, L]
        responses = batch['responses'].numpy()  # [B, L]
        
        for i in range(len(questions)):
            for t in range(len(questions[i])):
                q = questions[i, t]
                r = responses[i, t]
                if r >= 0:
                    skill_performance[q].append(r)
    
    # Compute variance in performance for each skill
    skill_variances = {}
    for skill, values in skill_performance.items():
        if len(values) > 20:
            skill_variances[skill] = np.var(values)
    
    if len(skill_variances) > 0:
        avg_variance = np.mean(list(skill_variances.values()))
        print(f"  Average performance variance per skill: {avg_variance:.4f}")
        print(f"  Skills analyzed: {len(skill_variances)}")
        print()
        
        if avg_variance > 0.2:
            print("  ✅ High variance: Skills differentiate students well")
        else:
            print("  ⚠️  Low variance: Limited differentiation")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Validate IRT Reference Targets')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, required=True, help='Cross-validation fold')
    parser.add_argument('--targets', type=str, required=True, help='Path to IRT targets .pkl file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("IRT REFERENCE TARGETS VALIDATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Targets: {args.targets}")
    print("=" * 80)
    print()
    
    # Load IRT targets
    irt_targets = load_irt_targets(args.targets)
    
    # Load dataset
    print("Loading dataset...")
    from pykt.config import que_type_models
    import json
    
    # Load data config and fix paths
    with open('configs/data_config.json') as f:
        data_config = json.load(f)
    
    # Fix relative path to absolute
    for dataset_key in data_config:
        if 'dpath' in data_config[dataset_key]:
            dpath = data_config[dataset_key]['dpath']
            if dpath.startswith('../data'):
                data_config[dataset_key]['dpath'] = dpath.replace('../data', 'data')
    
    # Use a simple model name for loading
    model_name = 'dkt'  # Generic model for dataset loading
    
    train_loader, valid_loader = init_dataset4train(
        args.dataset,
        model_name,
        data_config,
        args.fold,
        args.batch_size
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print()
    
    # Run analyses
    results = {}
    
    results['m_ref_correlation'] = analyze_m_ref_correlation(
        train_loader, valid_loader, irt_targets
    )
    
    results['beta_calibration'] = analyze_beta_calibration(
        train_loader, valid_loader, irt_targets
    )
    
    results['theta_distribution'] = analyze_theta_distribution(irt_targets)
    
    analyze_rasch_assumptions(train_loader, valid_loader, irt_targets)
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    m_ref_corr = results['m_ref_correlation']['pearson']
    m_ref_auc = results['m_ref_correlation']['auc']
    
    print(f"M_ref Correlation: {m_ref_corr:.4f}")
    print(f"M_ref AUC: {m_ref_auc:.4f}")
    print()
    
    if m_ref_corr > 0.7:
        print("✅ DIAGNOSIS: IRT targets are HIGH QUALITY")
        print("   Problem likely in model architecture or training")
        print()
        print("   Recommended next steps:")
        print("   1. Increase λ_target to 0.9 (prioritize alignment)")
        print("   2. Train for 100+ epochs")
        print("   3. Try learnable scaling: M = σ(α × (θ - β))")
    elif m_ref_corr > 0.5:
        print("⚠️  DIAGNOSIS: IRT targets are MODERATE QUALITY")
        print("   IRT captures some patterns but not all")
        print()
        print("   Recommended next steps:")
        print("   1. Try recalibrating IRT with more iterations")
        print("   2. Consider alternative reference model (BKT)")
        print("   3. Or switch to Path A (performance-first)")
    else:
        print("❌ DIAGNOSIS: IRT targets are POOR QUALITY")
        print("   IRT formula does NOT fit this dataset")
        print("   → This is the ROOT CAUSE of alignment failure")
        print()
        print("   Recommended next steps:")
        print("   1. Recalibrate IRT from scratch")
        print("   2. Try different IRT model (2PL, 3PL)")
        print("   3. Switch to different reference model (BKT)")
        print("   4. Or abandon alignment (Path A: λ=0.0)")
    
    print()


if __name__ == '__main__':
    main()
