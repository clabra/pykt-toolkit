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


def prepare_irt_data(df):
    """
    Convert sequence data to IRT format: (student_id, skill_id, response).
    
    For multi-skill datasets (assist2009), each question may have multiple skills.
    We create separate records for each skill.
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
    print(f"Prepared {len(irt_df)} IRT records from {df['uid'].nunique()} students")
    print(f"Number of unique skills: {irt_df['skill_id'].nunique()}")
    print(f"Response distribution: {irt_df['response'].value_counts().to_dict()}")
    
    return irt_df


def calibrate_rasch_model(irt_df, max_iterations=100):
    """
    Calibrate Rasch model using py-irt library.
    
    Returns:
        student_abilities: dict mapping student_id -> ability (theta)
        skill_difficulties: dict mapping skill_id -> difficulty (b)
    """
    print("\n" + "="*80)
    print("RASCH MODEL CALIBRATION")
    print("="*80)
    
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


def compute_rasch_targets(df, student_abilities, skill_difficulties, num_skills):
    """
    Compute Rasch mastery targets M_rasch[n,s,t] for each student-skill-timestep.
    
    Rasch model: P(correct | θ, b) = σ(θ - b)
    We use this probability as the mastery target.
    
    Args:
        df: DataFrame with student sequences
        student_abilities: dict mapping student_id -> ability θ
        skill_difficulties: dict mapping skill_id -> difficulty b
        num_skills: Total number of skills in the dataset
    
    Returns:
        rasch_targets: dict mapping student_id -> tensor of shape (seq_len, num_skills)
    """
    print("\n" + "="*80)
    print("COMPUTING RASCH TARGETS")
    print("="*80)
    
    rasch_targets = {}
    students_without_ability = []
    
    for idx, row in df.iterrows():
        uid = row['uid']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        
        # Get student ability (default to 0 if not calibrated)
        theta = student_abilities.get(uid, 0.0)
        if uid not in student_abilities:
            students_without_ability.append(uid)
        
        # Initialize target tensor: (seq_len, num_skills)
        seq_len = len(concepts)
        target_tensor = torch.zeros(seq_len, num_skills, dtype=torch.float32)
        
        # Compute mastery for each timestep and skill
        for t, skill_id in enumerate(concepts):
            # Get skill difficulty (default to 0 if not calibrated)
            b = skill_difficulties.get(skill_id, 0.0)
            
            # Rasch probability: P(correct) = σ(θ - b)
            mastery = torch.sigmoid(torch.tensor(theta - b))
            
            # Set mastery for this skill at this timestep
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
    print("="*80 + "\n")
    
    # Load dataset
    df = load_dataset_sequences(args.dataset, args.data_dir)
    
    # Prepare IRT data
    irt_df = prepare_irt_data(df)
    
    # Calibrate Rasch model
    student_abilities, skill_difficulties = calibrate_rasch_model(
        irt_df, 
        max_iterations=args.max_iterations
    )
    
    # Get number of skills
    num_skills = get_num_skills(args.dataset, args.data_dir)
    print(f"\nTotal number of skills in dataset: {num_skills}")
    
    # Compute Rasch targets
    rasch_targets = compute_rasch_targets(
        df, 
        student_abilities, 
        skill_difficulties,
        num_skills
    )
    
    # Save results
    metadata = {
        'dataset': args.dataset,
        'num_students': len(student_abilities),
        'num_skills': num_skills,
        'num_calibrated_skills': len(skill_difficulties),
        'max_iterations': args.max_iterations
    }
    
    save_rasch_targets(
        rasch_targets,
        student_abilities,
        skill_difficulties,
        args.output_path,
        metadata
    )
    
    print("\n" + "="*80)
    print("RASCH TARGET COMPUTATION COMPLETE")
    print("="*80)
    print(f"Use --rasch_path {args.output_path} when training iKT model")
    print("="*80)


if __name__ == '__main__':
    main()
