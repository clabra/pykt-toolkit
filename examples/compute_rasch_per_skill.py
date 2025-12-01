#!/usr/bin/env python
"""
Compute skill-specific, time-varying Rasch calibration.

Option B: Cumulative per-skill approach
- Run one Rasch calibration per skill using all interactions with that skill
- Extract time-varying abilities θ_i,k(t) for each student based on their interaction sequence
- Each calibration treats skill-specific interactions as a static assessment
- Time variation comes from tracking student ability after each successive interaction

Output: rasch_per_skill_targets.pkl with:
  - skill_difficulties: {skill_id: β_k} - one per skill
  - student_abilities_per_skill: {student_id: {skill_id: [θ_1, θ_2, ..., θ_n]}}
    where θ_j is ability after j-th interaction with that skill
  - rasch_targets_per_skill: per-student-skill-time mastery matrix

Usage:
    python examples/compute_rasch_per_skill.py --dataset assist2015 --seed 42
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

import pandas as pd
import json


def load_dataset_sequences(dataset_name, data_dir='data'):
    """
    Load dataset from CSV file.
    
    Returns:
        all_sequences: list of dicts with 'qseqs' and 'rseqs' for each student
        num_skills: number of unique skills
    """
    # Load data config to get number of skills
    data_config_path = os.path.join('/workspaces/pykt-toolkit/configs', 'data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    if dataset_name not in data_config:
        raise ValueError(f"Dataset {dataset_name} not found in data_config.json")
    
    num_skills = data_config[dataset_name]['num_c']
    
    # Load sequences from CSV
    dataset_path = os.path.join(data_dir, dataset_name, "train_valid_sequences.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Extract sequences
    all_sequences = []
    for _, row in df.iterrows():
        # Parse concepts and responses from strings
        concepts_str = row['concepts']
        responses_str = row['responses']
        
        # Convert from string representation to arrays
        qseqs = np.array([int(x) for x in concepts_str.split(',')])
        rseqs = np.array([int(x) for x in responses_str.split(',')])
        
        all_sequences.append({'qseqs': qseqs, 'rseqs': rseqs})
    
    return all_sequences, num_skills


def extract_skill_interactions(all_sequences, skill_id):
    """
    Extract all interactions with a specific skill across all students.
    
    Args:
        all_sequences: list of dicts with 'qseqs' and 'rseqs'
        skill_id: Skill to extract
    
    Returns:
        skill_interactions: list of (student_idx, position_in_seq, correct)
        student_sequences: dict {student_idx: [(position, correct)]}
    """
    skill_interactions = []
    student_sequences = defaultdict(list)
    
    for student_idx, seq in enumerate(all_sequences):
        qseqs = seq['qseqs']
        rseqs = seq['rseqs']
        
        for pos, (q, r) in enumerate(zip(qseqs, rseqs)):
            if int(q) == skill_id:
                skill_interactions.append((student_idx, pos, int(r)))
                student_sequences[student_idx].append((pos, int(r)))
    
    return skill_interactions, dict(student_sequences)


def calibrate_rasch_per_skill(all_sequences, skill_id, max_iterations=300, seed=42):
    """
    Run Rasch calibration for a single skill using all interactions with that skill.
    
    Args:
        all_sequences: list of student sequences
        skill_id: Skill to calibrate
        max_iterations: EM iterations
        seed: Random seed
    
    Returns:
        student_abilities: dict {student_idx: ability θ}
        difficulty: scalar β for this skill
        student_sequences: dict {student_idx: [(position, correct)]}
    """
    # Extract interactions with this skill
    skill_interactions, student_sequences = extract_skill_interactions(all_sequences, skill_id)
    
    if len(skill_interactions) < 10:
        print(f"  ⚠ Skill {skill_id}: Only {len(skill_interactions)} interactions, skipping")
        return None, None, None
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Prepare IRT data
    students = torch.LongTensor([s[0] for s in skill_interactions])  # student_idx
    items = torch.zeros_like(students)  # Single skill, all items = 0
    responses = torch.FloatTensor([s[2] for s in skill_interactions])  # correct
    
    try:
        from py_irt.models.one_param_logistic import OneParamLog
        from ordered_set import OrderedSet
        
        # Prepare unique student indices that appear in this skill
        unique_students = sorted(set(s[0] for s in skill_interactions))
        num_students_in_skill = len(unique_students)
        
        # Create student ID mapping (global -> local)
        student_id_map = {student_idx: local_idx for local_idx, student_idx in enumerate(unique_students)}
        
        # Prepare observation lists
        observation_subjects = []
        observation_items = []
        observations = []
        
        for student_idx, pos, correct in skill_interactions:
            # Map student_idx to local index (0, 1, 2, ...)
            local_idx = student_id_map[student_idx]
            observation_subjects.append(local_idx)
            observation_items.append(0)  # Single item
            observations.append(float(correct))
        
        # Create model
        model = OneParamLog(
            item_ids=['0'],
            subject_ids=[str(i) for i in range(num_students_in_skill)],
            num_items=1,
            num_subjects=num_students_in_skill,
            initializer_kwargs={'method': 'default'},
            priors='vague'
        )
        
        # Train model
        models_tensor = torch.tensor(observation_subjects, dtype=torch.long)
        items_tensor = torch.tensor(observation_items, dtype=torch.long)
        responses_tensor = torch.tensor(observations, dtype=torch.float32)
        
        model.fit(
            models=models_tensor,
            items=items_tensor,
            responses=responses_tensor,
            num_epochs=max_iterations
        )
        
        # Extract parameters
        params = model.export()
        student_abilities_raw = np.array(params['ability'])
        difficulty = float(params['diff'][0])
        
        # Map abilities back to original student indices
        student_abilities = {
            unique_students[local_idx]: float(student_abilities_raw[local_idx])
            for local_idx in range(len(student_abilities_raw))
        }
        
        print(f"  ✓ Skill {skill_id}: {len(skill_interactions)} interactions, β={difficulty:.3f}")
        return student_abilities, difficulty, student_sequences
        
    except Exception as e:
        print(f"  ✗ Skill {skill_id} calibration failed: {e}")
        return None, None, None


def extract_time_varying_abilities(student_abilities, student_sequences):
    """
    Extract time-varying abilities for each student based on interaction order.
    
    For each student, we track their ability after each interaction with this skill.
    The Rasch model gives us final ability; we approximate progression by assuming
    gradual improvement from initial to final ability.
    
    Args:
        student_abilities: dict {student_idx: final_θ}
        student_sequences: dict {student_idx: [(position, correct)]}
    
    Returns:
        time_varying_abilities: dict {student_idx: [(position, θ_after_interaction)]}
    """
    time_varying_abilities = {}
    
    for student_idx, interactions in student_sequences.items():
        if student_idx not in student_abilities:
            continue
        
        final_ability = student_abilities[student_idx]
        total_interactions = len(interactions)
        
        # Linear interpolation from 0 to final_ability
        # Assumes student starts at neutral ability (0) and grows to final_ability
        progression = []
        for i, (position, correct) in enumerate(interactions):
            if total_interactions == 1:
                ability_at_this_point = final_ability
            else:
                progress_ratio = (i + 1) / total_interactions
                ability_at_this_point = progress_ratio * final_ability
            
            progression.append((position, ability_at_this_point))
        
        time_varying_abilities[student_idx] = progression
    
    return time_varying_abilities


def compute_rasch_targets_per_skill(all_sequences, student_abilities_per_skill, skill_difficulties, num_skills):
    """
    Compute per-student-skill-time mastery targets using skill-specific abilities.
    
    Args:
        all_sequences: list of student sequences with 'qseqs' and 'rseqs'
        student_abilities_per_skill: {skill_id: {student_idx: [(position, θ)]}}
        skill_difficulties: {skill_id: β}
        num_skills: total number of skills
    
    Returns:
        rasch_targets: list of tensors [seq_len, num_skills] (one per student)
    """
    rasch_targets = []
    
    for student_idx, seq in enumerate(all_sequences):
        qseqs = seq['qseqs']
        seq_len = len(qseqs)
        
        # Initialize with NaN for all positions and skills
        student_matrix = torch.full((seq_len, num_skills), float('nan'))
        
        # For each skill, check if we have abilities for this student
        for skill_id, skill_abilities in student_abilities_per_skill.items():
            if student_idx not in skill_abilities:
                continue
            
            if skill_id not in skill_difficulties:
                continue
            
            beta = skill_difficulties[skill_id]
            time_ability_pairs = skill_abilities[student_idx]
            
            # Fill in mastery values at each position where this skill appears
            for position, theta in time_ability_pairs:
                # Rasch formula: M = σ(θ - β)
                mastery = torch.sigmoid(torch.tensor(theta - beta)).item()
                
                # Set mastery at this position and all future positions
                # (mastery doesn't decrease over time)
                for t in range(int(position), seq_len):
                    # Only update if current value is NaN or less than new mastery
                    if torch.isnan(student_matrix[t, skill_id]) or student_matrix[t, skill_id] < mastery:
                        student_matrix[t, skill_id] = mastery
        
        rasch_targets.append(student_matrix)
    
    return rasch_targets


def main():
    parser = argparse.ArgumentParser(
        description="Compute skill-specific, time-varying Rasch calibration (Option B)"
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., assist2015, assist2009)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root directory for datasets')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path (default: data/{dataset}/rasch_per_skill_targets.pkl)')
    parser.add_argument('--max_iterations', type=int, default=300,
                        help='Maximum EM iterations per skill (default: 300)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = os.path.join(
            args.data_dir, args.dataset, 'rasch_per_skill_targets.pkl'
        )
    
    print("="*80)
    print("SKILL-SPECIFIC TIME-VARYING RASCH CALIBRATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Max iterations per skill: {args.max_iterations}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {args.output_path}")
    print("="*80 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    all_sequences, num_skills = load_dataset_sequences(args.dataset, args.data_dir)
    print(f"  Total students: {len(all_sequences)}")
    print(f"  Skills: {num_skills}")
    
    # Count total interactions
    total_interactions = sum(len(seq['qseqs']) for seq in all_sequences)
    print(f"  Total interactions: {total_interactions}")
    
    # Calibrate each skill
    print(f"\nCalibrating {num_skills} skills...")
    skill_difficulties = {}
    student_abilities_per_skill = defaultdict(dict)  # {skill_id: {student_idx: [(position, θ)]}}
    
    successful_calibrations = 0
    failed_calibrations = 0
    
    for skill_id in range(num_skills):
        print(f"[{skill_id+1}/{num_skills}] Skill {skill_id}... ", end='')
        
        abilities, difficulty, student_sequences = calibrate_rasch_per_skill(
            all_sequences, skill_id, args.max_iterations, args.seed
        )
        
        if abilities is None:
            failed_calibrations += 1
            print("")
            continue
        
        # Store difficulty
        skill_difficulties[skill_id] = difficulty
        
        # Extract time-varying abilities
        time_varying = extract_time_varying_abilities(abilities, student_sequences)
        
        # Store in the structure: {skill_id: {student_idx: [(position, θ)]}}
        for student_idx, time_ability_pairs in time_varying.items():
            student_abilities_per_skill[skill_id][student_idx] = time_ability_pairs
        
        successful_calibrations += 1
    
    print(f"\n{'='*80}")
    print(f"Calibration Summary:")
    print(f"  Successful: {successful_calibrations}/{num_skills}")
    print(f"  Failed: {failed_calibrations}/{num_skills}")
    print(f"{'='*80}\n")
    
    if successful_calibrations == 0:
        print("✗ No successful calibrations, aborting")
        return 1
    
    # Compute per-student-skill-time mastery targets
    print("Computing mastery targets...")
    rasch_targets = compute_rasch_targets_per_skill(
        all_sequences, student_abilities_per_skill, skill_difficulties, num_skills
    )
    print(f"  Generated targets for {len(rasch_targets)} students")
    
    # Statistics
    all_difficulties = np.array(list(skill_difficulties.values()))
    print(f"\nSkill Difficulties (β):")
    print(f"  Mean: {all_difficulties.mean():.3f}")
    print(f"  Std:  {all_difficulties.std():.3f}")
    print(f"  Min:  {all_difficulties.min():.3f}")
    print(f"  Max:  {all_difficulties.max():.3f}")
    
    # Count interactions per student-skill
    total_time_varying_entries = sum(
        sum(len(student_abilities) for student_abilities in skill_abilities.values())
        for skill_abilities in student_abilities_per_skill.values()
    )
    print(f"\nTime-Varying Abilities:")
    print(f"  Total student-skill-time entries: {total_time_varying_entries}")
    print(f"  Avg per student: {total_time_varying_entries / len(rasch_targets):.1f}")
    
    # Save results
    print(f"\nSaving results to {args.output_path}...")
    
    data = {
        'skill_difficulties': skill_difficulties,
        'student_abilities_per_skill': dict(student_abilities_per_skill),
        'rasch_targets': rasch_targets,
        'metadata': {
            'dataset': args.dataset,
            'num_students': len(rasch_targets),
            'num_skills': len(skill_difficulties),
            'calibration_method': 'per_skill_cumulative',
            'max_iterations': args.max_iterations,
            'seed': args.seed,
            'successful_calibrations': successful_calibrations,
            'failed_calibrations': failed_calibrations
        }
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved to {args.output_path}")
    print(f"  File size: {os.path.getsize(args.output_path) / 1024 / 1024:.2f} MB")
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
