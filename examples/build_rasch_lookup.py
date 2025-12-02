"""
Build unified Rasch lookup structure for efficient training.

This script creates a tensor-based lookup structure that can be efficiently
accessed during training to retrieve Rasch-calibrated mastery values.

Usage:
    python build_rasch_lookup.py --dataset assist2015 --fold 0
    
Output:
    data/{dataset}/rasch_lookup_fold{fold}.pkl with structure:
    {
        'rasch_mastery': torch.Tensor,      # [num_students, max_seq_len, num_skills]
        'rasch_mask': torch.Tensor,         # [num_students, max_seq_len, num_skills]  
        'skill_difficulties': torch.Tensor, # [num_skills]
        'student_ids': list,                # Mapping of student index to student ID
        'metadata': dict                    # Calibration parameters and statistics
    }
"""

import argparse
import os
import pickle
import torch
import numpy as np
from collections import defaultdict


def build_rasch_lookup(dataset, fold, data_dir='data'):
    """
    Build efficient Rasch lookup structure from per-skill calibration.
    
    Args:
        dataset: Dataset name (e.g., 'assist2015')
        fold: Fold number
        data_dir: Root data directory
        
    Returns:
        Dictionary with tensors for efficient lookup
    """
    dataset_dir = os.path.join(data_dir, dataset)
    
    # Load per-skill Rasch calibration
    rasch_file = os.path.join(dataset_dir, f'rasch_per_skill_targets_fold{fold}.pkl')
    if not os.path.exists(rasch_file):
        raise FileNotFoundError(f"Rasch calibration not found: {rasch_file}")
    
    with open(rasch_file, 'rb') as f:
        rasch_data = pickle.load(f)
    
    print(f"Loaded Rasch calibration from {rasch_file}")
    print(f"Keys: {rasch_data.keys()}")
    
    # Extract student-sequence-skill structure
    rasch_targets = rasch_data['rasch_targets']  # dict[student_id] -> tensor[seq, skills]
    skill_difficulties = rasch_data.get('skill_difficulties', {})
    
    # Get dataset info
    num_students = len(rasch_targets)
    student_ids = sorted(rasch_targets.keys())
    
    # Determine dimensions
    max_seq_len = max(targets.shape[0] for targets in rasch_targets.values())
    num_skills = max(targets.shape[1] for targets in rasch_targets.values())
    
    print(f"\nDataset statistics:")
    print(f"  Students: {num_students}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Skills: {num_skills}")
    
    # Build tensors
    rasch_mastery = torch.full((num_students, max_seq_len, num_skills), float('nan'))
    rasch_mask = torch.zeros((num_students, max_seq_len, num_skills), dtype=torch.bool)
    
    # Fill tensors
    valid_entries = 0
    for student_idx, student_id in enumerate(student_ids):
        targets = rasch_targets[student_id]  # [seq_len, num_skills]
        seq_len = targets.shape[0]
        
        # Copy valid entries
        rasch_mastery[student_idx, :seq_len, :] = targets
        rasch_mask[student_idx, :seq_len, :] = ~torch.isnan(targets)
        
        valid_entries += rasch_mask[student_idx].sum().item()
    
    # Convert skill difficulties to tensor
    if skill_difficulties:
        skill_diff_tensor = torch.tensor([
            skill_difficulties.get(k, 0.0) for k in range(num_skills)
        ])
    else:
        skill_diff_tensor = torch.zeros(num_skills)
    
    coverage = valid_entries / (num_students * max_seq_len * num_skills) * 100
    print(f"\nCoverage statistics:")
    print(f"  Valid entries: {valid_entries:,}")
    print(f"  Total possible: {num_students * max_seq_len * num_skills:,}")
    print(f"  Coverage: {coverage:.2f}%")
    
    # Build lookup structure
    lookup = {
        'rasch_mastery': rasch_mastery,
        'rasch_mask': rasch_mask,
        'skill_difficulties': skill_diff_tensor,
        'student_ids': student_ids,
        'metadata': {
            'dataset': dataset,
            'fold': fold,
            'num_students': num_students,
            'max_seq_len': max_seq_len,
            'num_skills': num_skills,
            'valid_entries': int(valid_entries),
            'coverage_pct': float(coverage),
            'rasch_params': rasch_data.get('metadata', {})
        }
    }
    
    return lookup


def main():
    parser = argparse.ArgumentParser(description='Build Rasch lookup structure')
    parser.add_argument('--dataset', type=str, default='assist2015',
                       help='Dataset name')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root data directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: data/{dataset}/rasch_lookup_fold{fold}.pkl)')
    
    args = parser.parse_args()
    
    # Build lookup
    lookup = build_rasch_lookup(args.dataset, args.fold, args.data_dir)
    
    # Save
    if args.output is None:
        output_path = os.path.join(args.data_dir, args.dataset, 
                                   f'rasch_lookup_fold{args.fold}.pkl')
    else:
        output_path = args.output
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(lookup, f)
    
    print(f"\n✓ Saved Rasch lookup to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
