#!/usr/bin/env python3
"""
Run BKT (Bayesian Knowledge Tracing) baseline for comparison with neural models.

This script supports two evaluation modes:

MODE 1 - SKILL-LEVEL EVALUATION:
    - Train on skill-level data
    - Validate on skill-level data
    - Test on skill-level data
    - BKT naturally predicts at skill level

MODE 2 - QUESTION-LEVEL EVALUATION:
    - Train on skill-level data (BKT learns P(L0), P(T), P(S), P(G) per skill)
    - Validate on skill-level data
    - Test on question-level data with late fusion (mean of skill predictions)
    - Uses pre-trained model without updating on test data to avoid leakage

Usage:
    # Skill-level evaluation (default)
    python examples/run_bkt_benchmark.py --dataset assist2009 --mode skill
    
    # Question-level evaluation with late fusion
    python examples/run_bkt_benchmark.py --dataset assist2009 --mode question
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import sys
from datetime import datetime
from sklearn import metrics

# Add pyBKT to path
try:
    from pyBKT.models import Model
except ImportError:
    print("Error: pyBKT not installed. Run: pip install pyBKT")
    sys.exit(1)

def prepare_bkt_data(df):
    """Convert pykt sequence format to pyBKT format (skill-level)."""
    records = []
    
    for idx, row in df.iterrows():
        uid = row['uid']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        selectmasks = [int(m) for m in row['selectmasks'].split(',') if m != '-1']
        
        for order_id, (skill, response, mask) in enumerate(zip(concepts, responses, selectmasks)):
            if mask == 1:
                records.append({
                    'user_id': uid,
                    'skill_name': skill,
                    'correct': response,
                    'order_id': order_id
                })
    
    return pd.DataFrame(records)

def prepare_question_data(df):
    """Convert question-level data for late fusion evaluation.
    Creates a unique qidx for each evaluation point.
    """
    records = []
    qidx_counter = 0
    
    for idx, row in df.iterrows():
        uid = row['uid']
        questions = [int(q) for q in row['questions'].split(',') if q != '-1']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        selectmasks = [int(m) for m in row['selectmasks'].split(',') if m != '-1']
        
        for order_id, (qid, skill, response, mask) in enumerate(zip(questions, concepts, responses, selectmasks)):
            if mask == 1:
                records.append({
                    'user_id': uid,
                    'question_id': qid,
                    'skill_name': skill,
                    'correct': response,
                    'order_id': order_id,
                    'qidx': qidx_counter,
                    'seq_idx': idx
                })
                qidx_counter += 1
    
    return pd.DataFrame(records)

def evaluate_skill_level(model, skill_df):
    """Evaluate BKT at skill level (standard BKT evaluation).
    
    Uses pyBKT's built-in evaluation which properly handles sequential predictions.
    """
    auc = model.evaluate(metric='auc', data=skill_df)
    rmse = model.evaluate(metric='rmse', data=skill_df)
    acc = model.evaluate(metric='accuracy', data=skill_df)
    
    return auc, acc, rmse

def predict_without_update(model, skill_df):
    """Get BKT predictions without updating the model's belief state.
    
    This uses the trained model parameters to predict for each skill WITHOUT
    updating the mastery probabilities based on observed responses.
    This prevents data leakage during test evaluation.
    
    For each skill, computes: P(correct) = P(L_0) * (1 - P(S)) + (1 - P(L_0)) * P(G)
    where P(L_0) is initial mastery, P(S) is slip, P(G) is guess.
    
    Returns:
        predictions: Array of predicted probabilities
    """
    predictions_list = []
    
    # Get model's learned parameters by skill
    skill_params = model.coef_  # Dictionary: {skill_name: {prior, learns, guesses, slips, forgets}}
    
    for idx, row in skill_df.iterrows():
        skill = str(row['skill_name'])  # Convert to string (pyBKT uses string keys)
        
        # Get the model's learned parameters for this skill
        if skill in skill_params:
            params = skill_params[skill]
            # Use initial mastery probability P(L_0)
            p_l0 = params['prior']
            # Slip and guess probabilities
            p_s = params['slips'][0] if hasattr(params['slips'], '__getitem__') else params['slips']
            p_g = params['guesses'][0] if hasattr(params['guesses'], '__getitem__') else params['guesses']
            
            # P(correct) = P(L_0) * (1 - P(S)) + (1 - P(L_0)) * P(G)
            p_correct = p_l0 * (1 - p_s) + (1 - p_l0) * p_g
            predictions_list.append(p_correct)
        else:
            # Unknown skill - use default prior (0.5)
            predictions_list.append(0.5)
    
    return np.array(predictions_list)

def evaluate_question_level_late_fusion(model, question_df):
    """Evaluate BKT at question level using late fusion WITHOUT updating on test data.
    
    Groups by qidx (unique evaluation point), then aggregates skills within each point.
    Uses pre-trained model parameters without updating beliefs on test responses.
    This prevents data leakage.
    """
    # Get predictions WITHOUT updating model state
    predictions = predict_without_update(model, question_df)
    
    # Add predictions to dataframe
    question_df = question_df.copy()
    question_df['prediction'] = predictions
    
    # Group by qidx (unique evaluation point), then aggregate skills
    question_groups = question_df.groupby('qidx')
    
    true_labels = []
    pred_scores = []
    
    for qidx, group in question_groups:
        # Late fusion - mean: average skill-level predictions
        avg_pred = group['prediction'].mean()
        # Average responses across skills (matches neural model)
        avg_response = int(group['correct'].mean())
        
        true_labels.append(avg_response)
        pred_scores.append(avg_pred)
    
    # Calculate metrics
    auc = metrics.roc_auc_score(true_labels, pred_scores)
    pred_labels = [1 if p >= 0.5 else 0 for p in pred_scores]
    acc = metrics.accuracy_score(true_labels, pred_labels)
    rmse = np.sqrt(metrics.mean_squared_error(true_labels, pred_scores))
    
    return auc, acc, rmse

def run_fold(dataset, fold, output_dir, mode='skill'):
    """Train and evaluate BKT on a single fold.
    
    Args:
        dataset: Dataset name (e.g., 'assist2009')
        fold: Fold number (0-4)
        output_dir: Output directory for results
        mode: 'skill' for skill-level evaluation, 'question' for question-level evaluation
    """
    print(f"\n{'='*80}")
    print(f"FOLD {fold} - MODE: {mode.upper()}")
    print(f"{'='*80}")
    
    data_dir = Path(f"data/{dataset}")
    fold_dir = Path(output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data (folds 0-4 except current fold)
    train_dfs = []
    for f in range(5):
        if f != fold:
            fold_file = data_dir / "train_valid_sequences.csv"
            if fold_file.exists():
                df = pd.read_csv(fold_file)
                # Filter by fold
                df_fold = df[df['fold'] == f]
                train_dfs.append(df_fold)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Training data: {len(train_df)} sequences")
    
    # Load validation data (same as neural models - from train_valid with current fold)
    valid_file = data_dir / "train_valid_sequences.csv"
    valid_df = pd.read_csv(valid_file)
    valid_df = valid_df[valid_df['fold'] == fold]
    print(f"Validation data: {len(valid_df)} sequences")
    
    # Load test data based on mode
    if mode == 'question':
        # Question-level evaluation: use test_question_sequences.csv
        test_file = data_dir / "test_question_sequences.csv"
        if not test_file.exists():
            print(f"Error: {test_file} not found for question-level evaluation")
            sys.exit(1)
        test_df = pd.read_csv(test_file)
        test_df = test_df[test_df['fold'] == -1]
        print(f"Test data (question-level): {len(test_df)} sequences")
    else:
        # Skill-level evaluation: use test_sequences.csv
        test_file = data_dir / "test_sequences.csv"
        if not test_file.exists():
            # Fallback to test_question_sequences.csv
            test_file = data_dir / "test_question_sequences.csv"
        test_df = pd.read_csv(test_file)
        test_df = test_df[test_df['fold'] == -1]
        print(f"Test data (skill-level): {len(test_df)} sequences")
    
    # Prepare BKT format
    print("\nPreparing data for BKT...")
    train_bkt = prepare_bkt_data(train_df)
    valid_bkt = prepare_bkt_data(valid_df)
    
    if mode == 'question':
        test_bkt = prepare_question_data(test_df)
        print(f"Training interactions (skill-level): {len(train_bkt)}")
        print(f"Validation interactions (skill-level): {len(valid_bkt)}")
        print(f"Test interactions (question-level): {len(test_bkt)}")
    else:
        test_bkt = prepare_bkt_data(test_df)
        print(f"Training interactions (skill-level): {len(train_bkt)}")
        print(f"Validation interactions (skill-level): {len(valid_bkt)}")
        print(f"Test interactions (skill-level): {len(test_bkt)}")
    
    # Train BKT model
    print("\nTraining BKT model...")
    model = Model(seed=42, num_fits=1, parallel=False)
    model.fit(data=train_bkt)
    
    # Evaluate on validation data (always skill-level)
    print("\nEvaluating on validation data (skill-level)...")
    valid_auc, valid_acc, valid_rmse = evaluate_skill_level(model, valid_bkt)
    
    print(f"\nValidation Results (skill-level):")
    print(f"  AUC: {valid_auc:.4f}")
    print(f"  RMSE: {valid_rmse:.4f}")
    print(f"  ACC: {valid_acc:.4f}")
    
    # Evaluate on held-out test data based on mode
    if mode == 'question':
        # Question-level evaluation with late fusion (no model updates)
        print("\nEvaluating on held-out test data (question-level, late fusion - mean, no updates)...")
        test_auc, test_acc, test_rmse = evaluate_question_level_late_fusion(model, test_bkt)
        eval_type = 'question_level_late_fusion_mean_no_update'
        
        print(f"\nTest Results (question-level, late fusion - mean):")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  ACC: {test_acc:.4f}")
    else:
        # Skill-level evaluation
        print("\nEvaluating on held-out test data (skill-level)...")
        test_auc, test_acc, test_rmse = evaluate_skill_level(model, test_bkt)
        eval_type = 'skill_level'
        
        print(f"\nTest Results (skill-level):")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  ACC: {test_acc:.4f}")
    
    # Save results
    results = {
        'fold': fold,
        'valid_auc': float(valid_auc),
        'valid_rmse': float(valid_rmse),
        'valid_acc': float(valid_acc),
        'test_auc': float(test_auc),
        'test_rmse': float(test_rmse),
        'test_acc': float(test_acc),
        'num_train': len(train_bkt),
        'num_valid': len(valid_bkt),
        'num_test': len(test_bkt),
        'evaluation_type': eval_type,
        'evaluation_mode': mode,
        'model': 'BKT',
        'library': 'pyBKT'
    }
    
    results_file = fold_dir / "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run BKT baseline benchmark')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., assist2009)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--mode', type=str, default='skill', choices=['skill', 'question'],
                        help='Evaluation mode: "skill" for skill-level, "question" for question-level with late fusion')
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/{timestamp}_bkt_{args.mode}_{args.dataset}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"BKT BASELINE BENCHMARK - {args.mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    
    # Run all folds
    all_results = []
    for fold in range(5):
        results = run_fold(args.dataset, fold, output_dir, mode=args.mode)
        all_results.append(results)
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS (5-FOLD CV) - {args.mode.upper()} MODE")
    print(f"{'='*80}")
    
    valid_auc_values = [r['valid_auc'] for r in all_results]
    valid_acc_values = [r['valid_acc'] for r in all_results]
    valid_rmse_values = [r['valid_rmse'] for r in all_results]
    
    test_auc_values = [r['test_auc'] for r in all_results]
    test_acc_values = [r['test_acc'] for r in all_results]
    test_rmse_values = [r['test_rmse'] for r in all_results]
    
    summary = {
        'model': 'BKT',
        'dataset': args.dataset,
        'evaluation_mode': args.mode,
        'valid_mean_auc': float(np.mean(valid_auc_values)),
        'valid_std_auc': float(np.std(valid_auc_values)),
        'valid_mean_acc': float(np.mean(valid_acc_values)),
        'valid_std_acc': float(np.std(valid_acc_values)),
        'valid_mean_rmse': float(np.mean(valid_rmse_values)),
        'valid_std_rmse': float(np.std(valid_rmse_values)),
        'test_mean_auc': float(np.mean(test_auc_values)),
        'test_std_auc': float(np.std(test_auc_values)),
        'test_mean_acc': float(np.mean(test_acc_values)),
        'test_std_acc': float(np.std(test_acc_values)),
        'test_mean_rmse': float(np.mean(test_rmse_values)),
        'test_std_rmse': float(np.std(test_rmse_values)),
        'folds': all_results
    }
    
    print(f"\nValidation (5-fold CV, skill-level):")
    print(f"  AUC:  {summary['valid_mean_auc']:.4f} ± {summary['valid_std_auc']:.4f}")
    print(f"  ACC:  {summary['valid_mean_acc']:.4f} ± {summary['valid_std_acc']:.4f}")
    print(f"  RMSE: {summary['valid_mean_rmse']:.4f} ± {summary['valid_std_rmse']:.4f}")
    
    eval_type = all_results[0].get('evaluation_type', 'unknown')
    print(f"\nHeld-out Test ({eval_type}):")
    print(f"  AUC:  {summary['test_mean_auc']:.4f} ± {summary['test_std_auc']:.4f}")
    print(f"  ACC:  {summary['test_mean_acc']:.4f} ± {summary['test_std_acc']:.4f}")
    print(f"  RMSE: {summary['test_mean_rmse']:.4f} ± {summary['test_std_rmse']:.4f}")
    
    # Save summary
    summary_file = output_dir / "cv_results.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == '__main__':
    main()
