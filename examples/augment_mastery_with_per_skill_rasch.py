#!/usr/bin/env python
"""
Augment mastery_test.csv with per-skill Rasch calibration values.

This script loads the per-skill Rasch calibration results and adds columns to
mastery_test.csv for comparison with existing global Rasch values (theta, beta).

New columns added:
- theta_skill: Skill-specific student ability θ_{i,k}(t) at this time step
- beta_skill: Skill-specific difficulty β_k (same as global per skill)
- m_rasch_skill: Skill-specific mastery M = σ(θ_{i,k}(t) - β_k)

Usage:
    python examples/augment_mastery_with_per_skill_rasch.py \
        --experiment_dir experiments/20251201_205818_ikt2_vsymmetric_baseline_601503 \
        --rasch_per_skill_path data/assist2015/rasch_per_skill_targets.pkl
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch


def load_per_skill_rasch(path):
    """Load per-skill Rasch calibration results."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_skill_specific_ability(student_abilities_per_skill, student_idx, skill_id, position):
    """
    Get skill-specific ability for a student at a given position.
    
    Args:
        student_abilities_per_skill: {skill_id: {student_idx: [(position, θ)]}}
        student_idx: Student index in dataset
        skill_id: Skill ID
        position: Time step position
    
    Returns:
        theta_skill: Ability at this position, or NaN if no data
    """
    if skill_id not in student_abilities_per_skill:
        return np.nan
    
    skill_abilities = student_abilities_per_skill[skill_id]
    
    if student_idx not in skill_abilities:
        return np.nan
    
    # Get time-varying abilities for this student-skill pair
    time_abilities = skill_abilities[student_idx]  # [(position, θ)]
    
    if not time_abilities:
        return np.nan
    
    # Find the ability at or before this position
    # Abilities are cumulative - use the most recent one
    theta = np.nan
    for pos, ability in time_abilities:
        if pos <= position:
            theta = ability
        else:
            break
    
    return theta


def compute_mastery_rasch_skill(theta_skill, beta_skill):
    """Compute skill-specific Rasch mastery: M = σ(θ - β)."""
    if np.isnan(theta_skill) or np.isnan(beta_skill):
        return np.nan
    return 1.0 / (1.0 + np.exp(-(theta_skill - beta_skill)))


def augment_mastery_csv(experiment_dir, rasch_per_skill_path, output_suffix='_augmented'):
    """
    Augment mastery_test.csv with per-skill Rasch values.
    
    Args:
        experiment_dir: Path to experiment directory containing mastery_test.csv
        rasch_per_skill_path: Path to rasch_per_skill_targets.pkl
        output_suffix: Suffix for output file (default: '_augmented')
    """
    # Load per-skill Rasch data
    print(f"Loading per-skill Rasch data from {rasch_per_skill_path}...")
    rasch_data = load_per_skill_rasch(rasch_per_skill_path)
    
    skill_difficulties = rasch_data['skill_difficulties']
    student_abilities_per_skill = rasch_data['student_abilities_per_skill']
    
    print(f"  Skills calibrated: {len(skill_difficulties)}")
    print(f"  Skills with student abilities: {len(student_abilities_per_skill)}")
    
    # Load mastery_test.csv
    mastery_csv_path = os.path.join(experiment_dir, 'mastery_test.csv')
    print(f"\nLoading mastery data from {mastery_csv_path}...")
    df = pd.read_csv(mastery_csv_path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Need to map student_id in CSV to student_idx used in calibration
    # The student_idx in rasch_per_skill corresponds to order in the sequences file
    
    print("\nLoading student ID mapping from dataset...")
    dataset_name = rasch_data['metadata']['dataset']
    
    # The mastery test students are actually from train_valid (fold validation splits)
    # Always use train_valid for the mapping since that's where calibration was done
    sequences_path = f"data/{dataset_name}/train_valid_sequences.csv"
    print(f"  Using TRAIN/VALID sequences for mapping (mastery test uses validation fold)")
    
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
    
    # Load dataset to get student ID mapping
    dataset_df = pd.read_csv(sequences_path)
    
    # Check if there's a student_id column or if we need to use row index
    if 'uid' in dataset_df.columns:
        # Map uid to index (0, 1, 2, ...)
        student_id_to_idx = {uid: idx for idx, uid in enumerate(dataset_df['uid'].values)}
        print(f"  Mapped {len(student_id_to_idx)} student IDs from 'uid' column")
        
        # Report how many mastery students are found
        mastery_student_ids = set(df['student_id'].unique())
        found_students = mastery_student_ids & set(student_id_to_idx.keys())
        print(f"  Mastery students found in mapping: {len(found_students)} / {len(mastery_student_ids)}")
        if len(found_students) > 0:
            print(f"    Found IDs: {sorted(list(found_students))[:10]}")
    else:
        # Assume row index is the student_idx
        print("  Warning: No 'uid' column found, using row indices")
        student_id_to_idx = {i: i for i in range(len(dataset_df))}
    
    print(f"  Total students in mapping: {len(student_id_to_idx)}")
    
    # Add new columns
    print("\nAugmenting mastery data with per-skill Rasch values...")
    theta_skill_values = []
    beta_skill_values = []
    m_rasch_skill_values = []
    
    for _, row in df.iterrows():
        student_id = row['student_id']
        skill_id = int(row['skill_id'])
        time_step = int(row['time_step'])
        
        # Map student_id to student_idx
        student_idx = student_id_to_idx.get(student_id, None)
        
        if student_idx is None:
            # Student not in training data
            theta_skill = np.nan
            beta_skill = np.nan
            m_rasch_skill = np.nan
        else:
            # Get skill-specific ability at this time step
            theta_skill = get_skill_specific_ability(
                student_abilities_per_skill, student_idx, skill_id, time_step
            )
            
            # Get skill difficulty
            beta_skill = skill_difficulties.get(skill_id, np.nan)
            
            # Compute skill-specific mastery
            m_rasch_skill = compute_mastery_rasch_skill(theta_skill, beta_skill)
        
        theta_skill_values.append(theta_skill)
        beta_skill_values.append(beta_skill)
        m_rasch_skill_values.append(m_rasch_skill)
    
    # Add columns to dataframe
    df['theta_skill'] = theta_skill_values
    df['beta_skill'] = beta_skill_values
    df['m_rasch_skill'] = m_rasch_skill_values
    
    # Compute additional comparison columns
    df['theta_diff'] = df['theta'] - df['theta_skill']  # Global vs skill-specific ability
    df['beta_diff'] = df['beta'] - df['beta_skill']     # Global vs skill-specific difficulty
    
    # Save augmented file
    output_path = mastery_csv_path.replace('.csv', f'{output_suffix}.csv')
    print(f"\nSaving augmented data to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"\nSkill-specific abilities (theta_skill):")
    print(f"  Valid values: {df['theta_skill'].notna().sum()} / {len(df)}")
    print(f"  Mean: {df['theta_skill'].mean():.3f}")
    print(f"  Std:  {df['theta_skill'].std():.3f}")
    print(f"  Range: [{df['theta_skill'].min():.3f}, {df['theta_skill'].max():.3f}]")
    
    print(f"\nSkill-specific difficulties (beta_skill):")
    print(f"  Valid values: {df['beta_skill'].notna().sum()} / {len(df)}")
    print(f"  Mean: {df['beta_skill'].mean():.3f}")
    print(f"  Std:  {df['beta_skill'].std():.3f}")
    print(f"  Range: [{df['beta_skill'].min():.3f}, {df['beta_skill'].max():.3f}]")
    
    print(f"\nSkill-specific mastery (m_rasch_skill):")
    print(f"  Valid values: {df['m_rasch_skill'].notna().sum()} / {len(df)}")
    print(f"  Mean: {df['m_rasch_skill'].mean():.3f}")
    print(f"  Std:  {df['m_rasch_skill'].std():.3f}")
    print(f"  Range: [{df['m_rasch_skill'].min():.3f}, {df['m_rasch_skill'].max():.3f}]")
    
    print(f"\nAbility difference (theta - theta_skill):")
    valid_diff = df['theta_diff'].notna()
    print(f"  Valid values: {valid_diff.sum()}")
    print(f"  Mean: {df.loc[valid_diff, 'theta_diff'].mean():.3f}")
    print(f"  Std:  {df.loc[valid_diff, 'theta_diff'].std():.3f}")
    
    print(f"\nDifficulty difference (beta - beta_skill):")
    valid_diff = df['beta_diff'].notna()
    print(f"  Valid values: {valid_diff.sum()}")
    print(f"  Mean: {df.loc[valid_diff, 'beta_diff'].mean():.3f}")
    print(f"  Std:  {df.loc[valid_diff, 'beta_diff'].std():.3f}")
    
    # Compare with mi_prev
    if 'mi_prev' in df.columns:
        print(f"\nComparison with mi_prev:")
        valid_both = df['m_rasch_skill'].notna() & df['mi_prev'].notna()
        if valid_both.sum() > 0:
            correlation = df.loc[valid_both, 'm_rasch_skill'].corr(df.loc[valid_both, 'mi_prev'])
            print(f"  Correlation(m_rasch_skill, mi_prev): {correlation:.3f}")
    
    print("\n" + "="*80)
    print(f"✓ Augmented data saved to {output_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Augment mastery_test.csv with per-skill Rasch values"
    )
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory containing mastery_test.csv')
    parser.add_argument('--rasch_per_skill_path', type=str, required=True,
                        help='Path to rasch_per_skill_targets.pkl')
    parser.add_argument('--output_suffix', type=str, default='_augmented',
                        help='Suffix for output file (default: _augmented)')
    
    args = parser.parse_args()
    
    augment_mastery_csv(args.experiment_dir, args.rasch_per_skill_path, args.output_suffix)


if __name__ == '__main__':
    main()
