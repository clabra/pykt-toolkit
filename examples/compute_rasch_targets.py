#!/usr/bin/env python
"""
Compute Rasch IRT targets for iKT model training.

This script uses py-irt library to calibrate Rasch model parameters (student abilities 
and skill difficulties) from training data, then computes mastery targets M_rasch[n,s,t] 
for each student-skill-timestep combination.

Usage:
    python examples/compute_rasch_targets.py --dataset assist2015
    python examples/compute_rasch_targets.py --dataset assist2009 --output_path data/assist2009/rasch_targets.pkl
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from py_irt.models.one_param_logistic import OneParamLog
from py_irt.training import IrtModelTrainer
from py_irt.dataset import Dataset
from sklearn.preprocessing import LabelEncoder


def load_dataset_sequences(dataset_name, data_dir="data"):
    """Load training sequences from CSV file."""
    dataset_path = os.path.join(data_dir, dataset_name, "train_valid_sequences.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} student sequences")
    
    return df


def prepare_irt_data(df, use_final_ability=False):
    """
    Convert sequence data to IRT format: (student_id, skill_id, response).
    
    For multi-skill datasets (assist2009), each question may have multiple skills.
    We create separate records for each skill.
    
    Args:
        df: DataFrame with student sequences
        use_final_ability: If True, only use the LAST interaction per student-skill pair
                          (for computing consolidated mastery levels)
    """
    irt_records = []
    
    for idx, row in df.iterrows():
        uid = row['uid']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        selectmasks = [int(m) for m in row['selectmasks'].split(',') if m != '-1']
        
        # Filter by selectmask (only actual attempts, not padding)
        for t, (skill, response, mask) in enumerate(zip(concepts, responses, selectmasks)):
            if mask == 1:  # Valid interaction
                irt_records.append({
                    'student_id': uid,
                    'skill_id': skill,
                    'response': response,
                    'timestep': t
                })
    
    irt_df = pd.DataFrame(irt_records)
    
    if use_final_ability:
        # Keep only the LAST interaction per student-skill pair
        # This represents consolidated mastery after all practice with that skill
        print(f"Filtering to final abilities: keeping only last interaction per student-skill...")
        irt_df = irt_df.sort_values('timestep').groupby(['student_id', 'skill_id']).tail(1).reset_index(drop=True)
        print(f"Reduced to {len(irt_df)} records (final interactions)")
    
    print(f"Prepared {len(irt_df)} IRT records from {df['uid'].nunique()} students")
    print(f"Number of unique skills: {irt_df['skill_id'].nunique()}")
    print(f"Response distribution: {irt_df['response'].value_counts().to_dict()}")
    
    return irt_df


def calibrate_rasch_model(irt_df, max_iterations=100, seed=42):
    """
    Calibrate Rasch model using py-irt library.
    
    Args:
        irt_df: DataFrame with IRT records
        max_iterations: Maximum EM iterations
        seed: Random seed for reproducibility
    
    Returns:
        student_abilities: dict mapping student_id -> ability (theta)
        skill_difficulties: dict mapping skill_id -> difficulty (b)
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("\n" + "="*80)
    print("RASCH MODEL CALIBRATION")
    print("="*80)
    print(f"Random seed: {seed}")
    
    # Encode student and skill IDs
    student_encoder = LabelEncoder()
    skill_encoder = LabelEncoder()
    
    irt_df['student_idx'] = student_encoder.fit_transform(irt_df['student_id'])
    irt_df['skill_idx'] = skill_encoder.fit_transform(irt_df['skill_id'])
    
    # Prepare data for py-irt (requires list of tuples)
    # Format: [(student_idx, item_idx, response), ...]
    irt_data = list(zip(
        irt_df['student_idx'].values,
        irt_df['skill_idx'].values,
        irt_df['response'].values
    ))
    
    print(f"Calibrating Rasch model on {len(irt_data)} interactions...")
    print(f"Students: {len(student_encoder.classes_)}, Skills: {len(skill_encoder.classes_)}")
    
    # Prepare dataset for py-irt
    # Create ordered sets and mappings
    from ordered_set import OrderedSet
    
    num_students = len(student_encoder.classes_)
    num_skills = len(skill_encoder.classes_)
    
    item_ids = OrderedSet([str(i) for i in range(num_skills)])
    subject_ids = OrderedSet([str(i) for i in range(num_students)])
    
    item_id_to_ix = {str(i): i for i in range(num_skills)}
    ix_to_item_id = {i: str(i) for i in range(num_skills)}
    subject_id_to_ix = {str(i): i for i in range(num_students)}
    ix_to_subject_id = {i: str(i) for i in range(num_students)}
    
    observation_subjects = []
    observation_items = []
    observations = []
    training_example = []
    
    for student_idx, skill_idx, response in irt_data:
        observation_subjects.append(student_idx)
        observation_items.append(skill_idx)
        observations.append(float(response))
        training_example.append(True)
    
    # Construct Dataset manually
    dataset = Dataset(
        item_ids=item_ids,
        subject_ids=subject_ids,
        item_id_to_ix=item_id_to_ix,
        ix_to_item_id=ix_to_item_id,
        subject_id_to_ix=subject_id_to_ix,
        ix_to_subject_id=ix_to_subject_id,
        observation_subjects=observation_subjects,
        observation_items=observation_items,
        observations=observations,
        training_example=training_example
    )
    
    # Initialize 1PL (Rasch) model
    model = OneParamLog(
        item_ids=[str(i) for i in range(num_skills)],
        subject_ids=[str(i) for i in range(num_students)],
        num_items=num_skills,
        num_subjects=num_students,
        initializer_kwargs={'method': 'default'},
        priors='vague'  # Use vague priors for Rasch model
    )
    
    # Train model using variational inference
    print(f"Training Rasch model ({max_iterations} epochs)...")
    try:
        # Prepare inputs for model.fit() - need to convert to tensors
        models = torch.tensor(observation_subjects, dtype=torch.long)  # Subject indices
        items = torch.tensor(observation_items, dtype=torch.long)      # Item indices
        responses = torch.tensor(observations, dtype=torch.float32)    # Response values
        
        # Train model
        model.fit(
            models=models,
            items=items,
            responses=responses,
            num_epochs=max_iterations
        )
        
        print("✓ Rasch calibration completed successfully")
        
        # Export parameters
        params = model.export()
        
        # Extract abilities and difficulties
        # params contains 'ability' (subject abilities θ) and 'diff' (item difficulties b)
        # These are returned as lists, convert to numpy arrays
        student_abilities_array = np.array(params['ability'])  # Shape: (num_students,)
        skill_difficulties_array = np.array(params['diff'])     # Shape: (num_skills,)
        
        # Map back to original IDs
        student_abilities = {
            student_encoder.classes_[i]: float(student_abilities_array[i])
            for i in range(len(student_encoder.classes_))
        }
        
        skill_difficulties = {
            skill_encoder.classes_[i]: float(skill_difficulties_array[i])
            for i in range(len(skill_encoder.classes_))
        }
        
        # Print statistics
        print(f"\nStudent Abilities (θ):")
        print(f"  Mean: {np.mean(student_abilities_array):.3f}")
        print(f"  Std:  {np.std(student_abilities_array):.3f}")
        print(f"  Min:  {np.min(student_abilities_array):.3f}")
        print(f"  Max:  {np.max(student_abilities_array):.3f}")
        
        print(f"\nSkill Difficulties (b):")
        print(f"  Mean: {np.mean(skill_difficulties_array):.3f}")
        print(f"  Std:  {np.std(skill_difficulties_array):.3f}")
        print(f"  Min:  {np.min(skill_difficulties_array):.3f}")
        print(f"  Max:  {np.max(skill_difficulties_array):.3f}")
        
        return student_abilities, skill_difficulties
        
    except Exception as e:
        print(f"✗ Rasch calibration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def enforce_monotonicity_skill_wise(targets_dict):
    """
    Enforce monotonicity for each skill independently.
    Once mastery increases for a skill, it cannot decrease in future timesteps.
    
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
            # Forward pass: ensure M[t] >= M[t-1]
            for t in range(1, seq_len):
                if smoothed_matrix[t, skill_id] < smoothed_matrix[t-1, skill_id]:
                    smoothed_matrix[t, skill_id] = smoothed_matrix[t-1, skill_id]
        
        smoothed[uid] = smoothed_matrix
    
    return smoothed


def compute_rasch_targets(df, student_abilities, skill_difficulties, num_skills, dynamic=True, learning_rate=0.1):
    """
    Compute Rasch mastery targets M_rasch[n,s,t] for each student-skill-timestep.
    
    Three modes:
    1. Static (dynamic=False): P(correct | θ, b) = σ(θ - b) with constant θ
    2. Dynamic incremental (dynamic=True, learning_rate>0): θₜ = θₜ₋₁ + α × error
    3. Dynamic cumulative (dynamic=True, learning_rate=0): Recalibrate θ at each timestep
       using all observations up to that point (snapshot approach)
    
    Args:
        df: DataFrame with student sequences
        student_abilities: dict mapping student_id -> initial ability θ₀
        skill_difficulties: dict mapping skill_id -> difficulty b
        num_skills: Total number of skills in the dataset
        dynamic: If True, model learning progression; if False, use static abilities
        learning_rate: α parameter (0 = cumulative recalibration, >0 = incremental updates)
    
    Returns:
        rasch_targets: dict mapping student_id -> tensor of shape (seq_len, num_skills)
    """
    print("\n" + "="*80)
    print("COMPUTING RASCH TARGETS")
    print("="*80)
    
    if not dynamic:
        mode = "Static (constant ability)"
    elif learning_rate == 0:
        mode = "Dynamic - Cumulative recalibration (snapshot approach)"
    else:
        mode = "Dynamic - Incremental updates"
    
    print(f"Mode: {mode}")
    if dynamic:
        print(f"Learning rate (α): {learning_rate}")
    
    rasch_targets = {}
    students_without_ability = []
    
    for idx, row in df.iterrows():
        uid = row['uid']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        selectmasks = [int(m) for m in row['selectmasks'].split(',') if m != '-1']
        
        # Get initial student ability (default to 0 if not calibrated)
        theta_initial = student_abilities.get(uid, 0.0)
        if uid not in student_abilities:
            students_without_ability.append(uid)
        
        # Initialize target tensor: (seq_len, num_skills)
        seq_len = len(concepts)
        target_tensor = torch.zeros(seq_len, num_skills, dtype=torch.float32)
        
        if not dynamic:
            # Static mode: constant ability for all timesteps
            for t, (skill_id, mask) in enumerate(zip(concepts, selectmasks)):
                if mask == 0:
                    continue
                b = skill_difficulties.get(skill_id, 0.0)
                mastery = torch.sigmoid(torch.tensor(theta_initial - b))
                target_tensor[t, skill_id] = mastery
        
        elif learning_rate == 0:
            # Cumulative recalibration: Skill-specific ability estimation
            # Each skill maintains its own ability θₛ based only on observations of that skill
            # θₛ is recalculated from scratch each time skill s is practiced, using ALL past observations of s
            
            skill_observations = {}  # Track observations per skill: {skill_id: [responses]}
            skill_abilities = {}     # Track current ability per skill: {skill_id: theta_s}
            
            for t, (skill_id, response, mask) in enumerate(zip(concepts, responses, selectmasks)):
                if mask == 0:
                    continue
                
                # Initialize skill-specific tracking on first encounter
                if skill_id not in skill_observations:
                    skill_observations[skill_id] = []
                    # Start with global initial ability as prior
                    skill_abilities[skill_id] = theta_initial
                
                # Add current observation for this skill
                skill_observations[skill_id].append(response)
                
                # Recalibrate ability for THIS skill using ALL observations of this skill
                obs_list = skill_observations[skill_id]
                n_obs = len(obs_list)
                n_correct = sum(obs_list)
                
                if n_obs > 0:
                    # Empirical success rate for this skill
                    success_rate = n_correct / n_obs
                    
                    # Get skill difficulty
                    b = skill_difficulties.get(skill_id, 0.0)
                    
                    # Bayesian update: blend prior (initial ability) with evidence (empirical rate)
                    # Prior precision (confidence in initial estimate)
                    prior_precision = 1.0
                    
                    # Evidence precision (confidence in empirical rate, grows with observations)
                    evidence_precision = n_obs
                    
                    # Posterior estimate of ability for this skill
                    # More observations → more weight on empirical success rate
                    total_precision = prior_precision + evidence_precision
                    
                    # Convert success rate back to ability scale via inverse sigmoid
                    # θ = log(p / (1 - p)) + b  where p is success probability
                    if success_rate > 0.999:
                        success_rate = 0.999
                    elif success_rate < 0.001:
                        success_rate = 0.001
                    
                    empirical_ability = np.log(success_rate / (1 - success_rate)) + b
                    
                    # Weighted average: prior + evidence
                    theta_skill = (prior_precision * theta_initial + evidence_precision * empirical_ability) / total_precision
                    
                    skill_abilities[skill_id] = theta_skill
                
                # Compute mastery for ALL skills at this timestep
                # Each skill uses its own ability estimate (or global initial if not yet practiced)
                for s in range(num_skills):
                    b_s = skill_difficulties.get(s, 0.0)
                    
                    if s in skill_abilities:
                        # Use skill-specific ability
                        theta_s = skill_abilities[s]
                    else:
                        # Not yet practiced: use global initial ability
                        theta_s = theta_initial
                    
                    mastery = 1.0 / (1.0 + np.exp(-(theta_s - b_s)))
                    target_tensor[t, s] = mastery
        
        else:
            # Incremental update mode (original dynamic)
            theta_t = theta_initial
            skill_first_encounter = {}
            
            for t, (skill_id, response, mask) in enumerate(zip(concepts, responses, selectmasks)):
                if mask == 0:
                    continue
                
                b = skill_difficulties.get(skill_id, 0.0)
                
                if skill_id not in skill_first_encounter:
                    skill_first_encounter[skill_id] = t
                    theta_skill = theta_initial
                else:
                    theta_skill = theta_t
                
                mastery = torch.sigmoid(torch.tensor(theta_skill - b))
                prediction_error = response - mastery.item()
                theta_t = theta_t + learning_rate * prediction_error
                
                target_tensor[t, skill_id] = mastery
        
        rasch_targets[uid] = target_tensor
    
    if students_without_ability:
        print(f"⚠️  Warning: {len(students_without_ability)} students not in calibration set (using θ=0)")
    
    print(f"✓ Computed Rasch targets for {len(rasch_targets)} students")
    
    # Print sample statistics
    sample_uid = list(rasch_targets.keys())[0]
    sample_target = rasch_targets[sample_uid]
    print(f"\nSample target tensor shape: {sample_target.shape}")
    print(f"Sample mastery values (first timestep): {sample_target[0, :10]}")
    
    return rasch_targets


def get_num_skills(dataset_name, data_dir="data"):
    """Infer number of skills from dataset."""
    # Check if keyid2idx.json exists (contains mapping)
    keyid_path = os.path.join(data_dir, dataset_name, "keyid2idx.json")
    if os.path.exists(keyid_path):
        import json
        with open(keyid_path, 'r') as f:
            keyid = json.load(f)
            if 'concepts' in keyid:
                return len(keyid['concepts'])
    
    # Fallback: scan CSV file for max skill ID
    csv_path = os.path.join(data_dir, dataset_name, "train_valid_sequences.csv")
    df = pd.read_csv(csv_path)
    max_skill = 0
    for concepts_str in df['concepts']:
        concepts = [int(c) for c in concepts_str.split(',') if c != '-1']
        if concepts:
            max_skill = max(max_skill, max(concepts))
    
    return max_skill + 1  # +1 because skills are 0-indexed


def save_rasch_targets(rasch_targets, student_abilities, skill_difficulties, 
                       output_path, metadata=None):
    """Save Rasch targets and calibration parameters to pickle file."""
    data = {
        'rasch_targets': rasch_targets,
        'student_abilities': student_abilities,
        'skill_difficulties': skill_difficulties,
        'metadata': metadata or {}
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Rasch targets saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Compute Rasch IRT targets for iKT training")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., assist2015, assist2009)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root directory for datasets')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for Rasch targets (default: data/{dataset}/rasch_targets.pkl)')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum iterations for EM algorithm')
    parser.add_argument('--dynamic', action='store_true',
                        help='Use dynamic IRT (model learning progression over time)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for dynamic ability updates (default: 0.1)')
    parser.add_argument('--use_final_ability', action='store_true',
                        help='Use only final interaction per student-skill for IRT calibration (consolidated mastery)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible calibration (default: 42)')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = os.path.join(args.data_dir, args.dataset, 'rasch_targets.pkl')
    
    print("="*80)
    print("RASCH TARGET COMPUTATION FOR iKT MODEL")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Use final ability: {args.use_final_ability}")
    print("="*80 + "\n")
    
    # Load dataset
    df = load_dataset_sequences(args.dataset, args.data_dir)
    
    # Prepare IRT data
    irt_df = prepare_irt_data(df, use_final_ability=args.use_final_ability)
    
    # Calibrate Rasch model
    student_abilities, skill_difficulties = calibrate_rasch_model(
        irt_df, 
        max_iterations=args.max_iterations,
        seed=args.seed
    )
    
    # Get number of skills
    num_skills = get_num_skills(args.dataset, args.data_dir)
    print(f"\nTotal number of skills in dataset: {num_skills}")
    
    # Compute Rasch targets
    rasch_targets = compute_rasch_targets(
        df, 
        student_abilities, 
        skill_difficulties,
        num_skills,
        dynamic=args.dynamic,
        learning_rate=args.learning_rate
    )
    
    # Save results
    metadata = {
        'dataset': args.dataset,
        'num_students': len(student_abilities),
        'num_skills': num_skills,
        'num_calibrated_skills': len(skill_difficulties),
        'max_iterations': args.max_iterations,
        'seed': args.seed,
        'dynamic': args.dynamic,
        'learning_rate': args.learning_rate if args.dynamic else None,
        'use_final_ability': args.use_final_ability,
        'calibration_method': 'final_ability_per_skill' if args.use_final_ability else 'averaged_ability'
    }
    
    save_rasch_targets(
        rasch_targets,
        student_abilities,
        skill_difficulties,
        args.output_path,
        metadata
    )
    
    output_path_mono = args.output_path.replace('.pkl', '_mono.pkl')
    
    print("\n" + "="*80)
    print("RASCH TARGET COMPUTATION COMPLETE")
    print("="*80)
    print(f"Use --rasch_path {args.output_path} --mastery_method irt (standard)")
    print(f"Use --rasch_path {output_path_mono} --mastery_method irt_mono (monotonic)")
    print("="*80)


if __name__ == '__main__':
    main()
