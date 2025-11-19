#!/usr/bin/env python
"""
Compute skill difficulty from training data for semantic grounding.

Skill difficulty is measured as the average number of practice attempts
needed across all students to achieve mastery (correct response), normalized
by the global average.

Copyright (c) 2025 Concha Labra. All Rights Reserved.
"""

import os
import json
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path


def compute_skill_difficulty(data_path, output_path):
    """
    Compute skill difficulty from training data.
    
    Difficulty is measured as the total number of practice attempts students
    typically need with a skill across their entire learning sequence, not just
    until first success. This captures skills that require repeated practice.
    
    Args:
        data_path: Path to pickle file with training data
        output_path: Path to save skill_difficulty.json
        
    Returns:
        dict: skill_id -> difficulty_factor (relative to global avg)
    """
    # Load training data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract cseqs (concepts/skills) and rseqs (responses)
    cseqs = data['cseqs']  # [num_students, max_seq_len]
    rseqs = data['rseqs']  # [num_students, max_seq_len]
    num_students = data['num_students']
    
    print(f"Loaded training data from {data_path}")
    print(f"Total students: {num_students}")
    print(f"Sequence shape: {cseqs.shape}")
    
    # Track attempts needed to achieve mastery (3 correct in a row)
    # This captures both initial learning AND consistency
    skill_practice_counts = defaultdict(list)  # skill_id -> [attempts_to_mastery_per_student]
    
    for student_idx in range(len(cseqs)):
        if (student_idx + 1) % 1000 == 0:
            print(f"Processing student {student_idx + 1}/{num_students}...")
        
        skills = cseqs[student_idx].numpy()
        responses = rseqs[student_idx].numpy()
        
        # Track progress toward mastery for each skill (3 correct in a row)
        skill_attempts = defaultdict(int)  # Total attempts for each skill
        skill_consecutive_correct = defaultdict(int)  # Consecutive correct count
        skill_mastered = set()  # Skills that achieved 3 correct in a row
        
        for skill_id, response in zip(skills, responses):
            skill_id = int(skill_id)
            
            # Skip padding (skill_id might be 0 or negative)
            if skill_id <= 0:
                continue
            
            # Skip if already mastered (3 correct in a row achieved)
            if skill_id in skill_mastered:
                continue
            
            # Count this attempt
            skill_attempts[skill_id] += 1
            
            # Track consecutive correct responses
            if int(response) == 1:
                skill_consecutive_correct[skill_id] += 1
                # Check if mastery achieved (3 correct in a row)
                if skill_consecutive_correct[skill_id] >= 3:
                    skill_practice_counts[skill_id].append(skill_attempts[skill_id])
                    skill_mastered.add(skill_id)
            else:
                # Reset consecutive count on incorrect response
                skill_consecutive_correct[skill_id] = 0
    
    # Compute average practice needed per skill
    skill_avg_practice = {}
    for skill_id, counts in skill_practice_counts.items():
        skill_avg_practice[skill_id] = np.mean(counts)
    
    # Compute global average
    global_avg_practice = np.mean(list(skill_avg_practice.values()))
    
    # Normalize by global average to get difficulty factors
    skill_difficulty = {}
    for skill_id, avg_practice in skill_avg_practice.items():
        skill_difficulty[skill_id] = float(avg_practice / global_avg_practice)
    
    # Report statistics
    print(f"\n=== Skill Difficulty Statistics ===")
    print(f"Skills analyzed: {len(skill_difficulty)}")
    print(f"Global avg practice needed: {global_avg_practice:.2f}")
    print(f"Difficulty range: {min(skill_difficulty.values()):.3f} - {max(skill_difficulty.values()):.3f}")
    print(f"Difficulty mean: {np.mean(list(skill_difficulty.values())):.3f}")
    print(f"Difficulty std: {np.std(list(skill_difficulty.values())):.3f}")
    
    # Show hardest and easiest skills
    sorted_skills = sorted(skill_difficulty.items(), key=lambda x: x[1], reverse=True)
    print(f"\nHardest 5 skills (require most practice):")
    for skill_id, difficulty in sorted_skills[:5]:
        avg_practice = skill_avg_practice[skill_id]
        print(f"  Skill {skill_id}: difficulty={difficulty:.3f} (avg {avg_practice:.1f} attempts)")
    
    print(f"\nEasiest 5 skills (require least practice):")
    for skill_id, difficulty in sorted_skills[-5:]:
        avg_practice = skill_avg_practice[skill_id]
        print(f"  Skill {skill_id}: difficulty={difficulty:.3f} (avg {avg_practice:.1f} attempts)")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(skill_difficulty, f, indent=2)
    
    print(f"\nSaved skill difficulty to {output_path}")
    
    return skill_difficulty


if __name__ == '__main__':
    # ASSIST2015 dataset - use the fold 0 training data
    data_path = '/workspaces/pykt-toolkit/data/assist2015/train_valid_sequences.csv_0.pkl'
    output_path = '/workspaces/pykt-toolkit/data/assist2015/skill_difficulty.json'
    
    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print("Please ensure ASSIST2015 dataset is properly set up.")
        exit(1)
    
    skill_difficulty = compute_skill_difficulty(data_path, output_path)
    
    print("\n=== Computation Complete ===")
    print(f"Skill difficulty factors computed for {len(skill_difficulty)} skills")
    print(f"These factors will be used to scale learning gains:")
    print(f"  - Harder skills (difficulty > 1.0) get LOWER base gains")
    print(f"  - Easier skills (difficulty < 1.0) get HIGHER base gains")
    print(f"This introduces semantic differentiation grounded in observable learning patterns.")
