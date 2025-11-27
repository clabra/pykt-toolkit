"""
Compute BKT (Bayesian Knowledge Tracing) targets for iKT model training.

This script uses pyBKT to:
1. Learn BKT parameters (P_L0, P_T, P_S, P_G) from training data
2. Run forward inference to compute P(learned) at each timestep
3. Save mastery targets for use in iKT L2 loss

BKT is designed for dynamic knowledge tracing, unlike Rasch which assumes static ability.
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
import sys

# Add pyBKT to path
try:
    from pyBKT.models import Model
except ImportError:
    print("Error: pyBKT not installed. Run: pip install pyBKT")
    sys.exit(1)


def prepare_bkt_data(df):
    """
    Convert pykt sequence format to pyBKT format.
    
    pyBKT expects: user_id, skill_name, correct, order_id
    """
    records = []
    
    for idx, row in df.iterrows():
        uid = row['uid']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        selectmasks = [int(m) for m in row['selectmasks'].split(',') if m != '-1']
        
        # Convert to row-per-interaction format
        for order_id, (skill, response, mask) in enumerate(zip(concepts, responses, selectmasks)):
            if mask == 1:
                records.append({
                    'user_id': uid,
                    'skill_name': skill,
                    'correct': response,
                    'order_id': order_id
                })
    
    bkt_df = pd.DataFrame(records)
    return bkt_df


def fit_bkt_model(bkt_df, num_skills):
    """
    Fit BKT model to training data using EM algorithm.
    """
    print(f"\n{'='*80}")
    print("FITTING BKT MODEL")
    print(f"{'='*80}")
    print(f"Training on {len(bkt_df)} interactions")
    print(f"Students: {bkt_df['user_id'].nunique()}")
    print(f"Skills: {num_skills}")
    
    # Initialize model
    model = Model(
        seed=42,
        num_fits=1,  # Number of EM restarts
        parallel=False
    )
    
    # Fit model (learns parameters via EM)
    print("\nFitting BKT parameters via Expectation-Maximization...")
    model.fit(data=bkt_df)
    
    print("✓ BKT model fitted successfully")
    
    # Extract learned parameters
    params_df = model.params()
    
    # Convert to dict format: {skill: {param: value}}
    params = {}
    skills = params_df.index.get_level_values('skill').unique()
    
    for skill in skills:
        skill_int = int(skill)
        params[skill_int] = {
            'prior': params_df.loc[(skill, 'prior', 'default'), 'value'],
            'learns': params_df.loc[(skill, 'learns', 'default'), 'value'],
            'slips': params_df.loc[(skill, 'slips', 'default'), 'value'],
            'guesses': params_df.loc[(skill, 'guesses', 'default'), 'value'],
        }
    
    print(f"\n{'='*80}")
    print("LEARNED BKT PARAMETERS (averaged across skills)")
    print(f"{'='*80}")
    
    # Parameters are per-skill
    p_l0_mean = np.mean([params[skill]['prior'] for skill in params])
    p_t_mean = np.mean([params[skill]['learns'] for skill in params])
    p_s_mean = np.mean([params[skill]['slips'] for skill in params])
    p_g_mean = np.mean([params[skill]['guesses'] for skill in params])
    
    print(f"P(L0) - Prior knowledge:     {p_l0_mean:.4f}")
    print(f"P(T)  - Learning rate:       {p_t_mean:.4f}")
    print(f"P(S)  - Slip probability:    {p_s_mean:.4f}")
    print(f"P(G)  - Guess probability:   {p_g_mean:.4f}")
    
    return model, params


def enforce_monotonicity_skill_wise(targets_dict):
    """
    Enforce monotonicity for each skill independently.
    Once P(L) increases for a skill, it cannot decrease in future timesteps.
    
    Args:
        targets_dict: dict {uid: torch.Tensor [seq_len, num_skills]}
    
    Returns:
        smoothed_dict: dict with same structure, monotonic per skill
    """
    smoothed = {}
    
    for uid, target_matrix in targets_dict.items():
        seq_len, num_skills = target_matrix.shape
        smoothed_matrix = target_matrix.clone()
        
        # For each skill column independently
        for skill_id in range(num_skills):
            # Forward pass: ensure P(L)[t] >= P(L)[t-1]
            for t in range(1, seq_len):
                if smoothed_matrix[t, skill_id] < smoothed_matrix[t-1, skill_id]:
                    smoothed_matrix[t, skill_id] = smoothed_matrix[t-1, skill_id]
        
        smoothed[uid] = smoothed_matrix
    
    return smoothed


def compute_bkt_targets(df, model, params, num_skills):
    """
    Run BKT forward algorithm to compute mastery targets for each student.
    
    Returns dict: {student_id: torch.Tensor[seq_len, num_skills]}
    """
    print(f"\n{'='*80}")
    print("COMPUTING BKT MASTERY TARGETS")
    print(f"{'='*80}")
    
    bkt_targets = {}
    
    for idx, row in df.iterrows():
        uid = row['uid']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        selectmasks = [int(m) for m in row['selectmasks'].split(',') if m != '-1']
        
        # Filter valid interactions
        valid_idx = [i for i in range(len(concepts)) if selectmasks[i] == 1]
        concepts = [concepts[i] for i in valid_idx]
        responses = [responses[i] for i in valid_idx]
        
        seq_len = len(concepts)
        target_tensor = torch.zeros(seq_len, num_skills, dtype=torch.float32)
        
        # Track P(learned) for each skill
        skill_mastery = {}  # {skill_id: P_L}
        
        # Run forward algorithm
        for t, (skill, response) in enumerate(zip(concepts, responses)):
            # Initialize skill on first encounter
            if skill not in skill_mastery:
                if skill in params:
                    skill_mastery[skill] = params[skill]['prior']
                else:
                    # Skill not seen in training, use average
                    skill_mastery[skill] = np.mean([params[s]['prior'] for s in params])
            
            # Get BKT parameters for this skill
            if skill in params:
                p_l = skill_mastery[skill]
                p_t = params[skill]['learns']
                p_s = params[skill]['slips']
                p_g = params[skill]['guesses']
            else:
                # Use average parameters for unseen skills
                p_l = skill_mastery[skill]
                p_t = np.mean([params[s]['learns'] for s in params])
                p_s = np.mean([params[s]['slips'] for s in params])
                p_g = np.mean([params[s]['guesses'] for s in params])
            
            # BKT Forward Algorithm
            # 1. Predict probability of correct response
            p_correct = p_l * (1 - p_s) + (1 - p_l) * p_g
            
            # Avoid division by zero
            p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)
            
            # 2. Update belief based on observation (Bayes rule)
            if response == 1:  # Correct
                p_l_updated = (p_l * (1 - p_s)) / p_correct
            else:  # Incorrect
                p_l_updated = (p_l * p_s) / (1 - p_correct)
            
            # 3. Apply learning transition
            p_l_new = p_l_updated + (1 - p_l_updated) * p_t
            
            # Clip to valid probability range
            p_l_new = np.clip(p_l_new, 0.0, 1.0)
            
            # Update skill mastery
            skill_mastery[skill] = p_l_new
            
            # Populate target tensor: use current mastery for all skills
            for s in range(num_skills):
                if s in skill_mastery:
                    target_tensor[t, s] = skill_mastery[s]
                else:
                    # Skills not yet practiced: use prior
                    if s in params:
                        target_tensor[t, s] = params[s]['prior']
                    else:
                        target_tensor[t, s] = np.mean([params[sk]['prior'] for sk in params])
        
        bkt_targets[uid] = target_tensor
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} students...")
    
    print(f"✓ Computed BKT targets for {len(bkt_targets)} students")
    
    return bkt_targets


def main():
    parser = argparse.ArgumentParser(description='Compute BKT targets for iKT training')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., assist2015)')
    parser.add_argument('--output_path', type=str, default=None, 
                       help='Output path for targets (default: data/{dataset}/bkt_targets.pkl)')
    
    args = parser.parse_args()
    
    # Paths
    data_path = f'data/{args.dataset}/train_valid_sequences.csv'
    if args.output_path is None:
        output_path = f'data/{args.dataset}/bkt_targets.pkl'
    else:
        output_path = args.output_path
    
    print(f"\n{'='*80}")
    print("BKT TARGET COMPUTATION")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    
    # Load data
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} student sequences")
    
    # Get number of skills
    all_skills = set()
    for concepts_str in df['concepts']:
        skills = [int(c) for c in concepts_str.split(',') if c != '-1']
        all_skills.update(skills)
    num_skills = len(all_skills)
    print(f"Number of unique skills: {num_skills}")
    
    # Prepare data for pyBKT
    print("\nPreparing data for BKT...")
    bkt_df = prepare_bkt_data(df)
    print(f"Prepared {len(bkt_df)} interactions from {bkt_df['user_id'].nunique()} students")
    
    # Fit BKT model
    model, params = fit_bkt_model(bkt_df, num_skills)
    
    # Compute targets
    bkt_targets = compute_bkt_targets(df, model, params, num_skills)
    
    # Apply monotonic smoothing
    print(f"\n{'='*80}")
    print("APPLYING MONOTONIC SMOOTHING")
    print(f"{'='*80}\n")
    bkt_targets_mono = enforce_monotonicity_skill_wise(bkt_targets)
    print(f"✓ Monotonic version created")
    
    # Save standard version
    print(f"\n{'='*80}")
    print("SAVING STANDARD VERSION")
    print(f"{'='*80}")
    
    result_standard = {
        'bkt_targets': bkt_targets,
        'bkt_params': params,
        'metadata': {
            'dataset': args.dataset,
            'num_students': len(bkt_targets),
            'num_skills': num_skills,
            'method': 'BKT (Bayesian Knowledge Tracing)',
            'model': 'pyBKT',
            'monotonic': False
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(result_standard, f)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ Standard BKT targets saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Save monotonic version
    output_path_mono = output_path.replace('.pkl', '_mono.pkl')
    
    print(f"\n{'='*80}")
    print("SAVING MONOTONIC VERSION")
    print(f"{'='*80}")
    
    result_mono = {
        'bkt_targets': bkt_targets_mono,
        'bkt_params': params,
        'metadata': {
            'dataset': args.dataset,
            'num_students': len(bkt_targets_mono),
            'num_skills': num_skills,
            'method': 'BKT (Bayesian Knowledge Tracing)',
            'model': 'pyBKT',
            'monotonic': True
        }
    }
    
    with open(output_path_mono, 'wb') as f:
        pickle.dump(result_mono, f)
    
    file_size_mb_mono = Path(output_path_mono).stat().st_size / (1024 * 1024)
    print(f"\n✓ Monotonic BKT targets saved to: {output_path_mono}")
    print(f"  File size: {file_size_mb_mono:.2f} MB")
    
    # Sample statistics
    sample_uid = list(bkt_targets.keys())[0]
    sample_target = bkt_targets[sample_uid]
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Sample target tensor shape: {sample_target.shape}")
    print(f"Sample mastery values (first timestep): {sample_target[0, :10]}")
    print(f"\nGenerated files:")
    print(f"  1. {output_path} (standard, monotonic=False)")
    print(f"  2. {output_path_mono} (smoothed, monotonic=True)")
    
    print(f"\n{'='*80}")
    print("BKT TARGET COMPUTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Use --rasch_path {output_path} --mastery_method bkt (standard)")
    print(f"Use --rasch_path {output_path_mono} --mastery_method bkt_mono (monotonic)")
    print(f"(Note: iKT code refers to targets as 'rasch_targets' but BKT targets work the same way)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
