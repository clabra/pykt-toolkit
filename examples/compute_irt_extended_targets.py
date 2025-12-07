#!/usr/bin/env python3
"""
Generate Extended IRT Targets for iKT3 Training

Creates IRT reference targets with:
- β_IRT: Skill difficulties (from existing rasch calibration)
- θ_IRT: Student abilities (computed via IRT calibration)
- M_ref: Reference predictions σ(θ_IRT - β_IRT)

Usage:
    python examples/compute_irt_extended_targets.py \\
        --dataset assist2015 \\
        --fold 0 \\
        --rasch_path data/assist2015/rasch_test_iter300.pkl \\
        --output_path data/assist2015/irt_extended_targets_fold0.pkl
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train


def load_rasch_difficulties(rasch_path, num_skills):
    """
    Load skill difficulties from existing Rasch calibration.
    
    Args:
        rasch_path: Path to rasch_test_iter300.pkl
        num_skills: Number of skills
    
    Returns:
        Dictionary {skill_id: β_value}
    """
    if not os.path.exists(rasch_path):
        raise FileNotFoundError(f"Rasch file not found: {rasch_path}")
    
    with open(rasch_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'skill_difficulties' not in data:
        raise KeyError(
            f"'skill_difficulties' key not found in {rasch_path}\n"
            f"Available keys: {list(data.keys())}"
        )
    
    skill_difficulties = data['skill_difficulties']
    print(f"✓ Loaded {len(skill_difficulties)} skill difficulties from {rasch_path}")
    print(f"  Range: [{min(skill_difficulties.values()):.3f}, {max(skill_difficulties.values()):.3f}]")
    
    return skill_difficulties


def compute_student_abilities(train_loader, valid_loader, skill_difficulties, num_skills):
    """
    Compute student abilities via simple IRT calibration.
    
    For each student, estimate θ_i by averaging observed logits:
    θ_i = mean(logit(response_ij) + β_j) over all student's interactions
    
    Args:
        train_loader: Training data loader
        valid_loader: Validation data loader
        skill_difficulties: Dict {skill_id: β}
        num_skills: Number of skills
    
    Returns:
        Dictionary {uid: θ_value}
    """
    student_abilities = {}
    student_interaction_counts = {}
    
    # Convert difficulties to tensor for vectorized ops
    beta_tensor = torch.tensor([skill_difficulties.get(k, 0.0) for k in range(num_skills)])
    
    print("Computing student abilities from train+valid data...")
    
    for loader_name, loader in [('train', train_loader), ('valid', valid_loader)]:
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 100 == 0:
                print(f"  Processing {loader_name} batch {batch_idx}/{len(loader)}")
            
            questions = batch['cseqs']  # [B, L]
            responses = batch['rseqs']  # [B, L]
            masks = batch['masks']      # [B, L]
            
            batch_size, seq_len = questions.size()
            
            for i in range(batch_size):
                # Get student ID (if available, otherwise use batch index)
                if 'uids' in batch:
                    uid = batch['uids'][i]
                else:
                    uid = f'{loader_name}_{batch_idx}_{i}'
                
                # Get valid interactions
                valid_mask = masks[i] == 1
                q_valid = questions[i][valid_mask]
                r_valid = responses[i][valid_mask]
                
                if len(q_valid) == 0:
                    continue
                
                # Get difficulties for these questions
                beta_vals = beta_tensor[q_valid]
                
                # Estimate ability from observed responses
                # logit(p) ≈ θ - β  →  θ ≈ logit(p) + β
                # Use smoothed response: p = (r + 0.5) / 2 to avoid log(0)
                smoothed_responses = (r_valid.float() + 0.5) / 2.0
                logit_responses = torch.log(smoothed_responses / (1 - smoothed_responses))
                
                # Estimate θ for each interaction
                theta_estimates = logit_responses + beta_vals
                
                # Accumulate for student
                if uid not in student_abilities:
                    student_abilities[uid] = 0.0
                    student_interaction_counts[uid] = 0
                
                student_abilities[uid] += theta_estimates.sum().item()
                student_interaction_counts[uid] += len(theta_estimates)
    
    # Average abilities
    for uid in student_abilities:
        if student_interaction_counts[uid] > 0:
            student_abilities[uid] /= student_interaction_counts[uid]
    
    print(f"✓ Computed abilities for {len(student_abilities)} students")
    abilities_array = np.array(list(student_abilities.values()))
    print(f"  Mean: {abilities_array.mean():.3f}, Std: {abilities_array.std():.3f}")
    print(f"  Range: [{abilities_array.min():.3f}, {abilities_array.max():.3f}]")
    
    return student_abilities


def compute_reference_predictions(test_loader, skill_difficulties, student_abilities, num_skills):
    """
    Compute reference predictions M_ref = σ(θ_IRT - β_IRT) for test set.
    
    Args:
        test_loader: Test data loader
        skill_difficulties: Dict {skill_id: β}
        student_abilities: Dict {uid: θ}
        num_skills: Number of skills
    
    Returns:
        Dictionary {uid: torch.Tensor[seq_len]} with reference predictions
    """
    reference_predictions = {}
    beta_tensor = torch.tensor([skill_difficulties.get(k, 0.0) for k in range(num_skills)])
    
    # Default ability for unseen students (use mean of known students)
    mean_ability = np.mean(list(student_abilities.values()))
    
    print("Computing reference predictions for test set...")
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx % 50 == 0:
            print(f"  Processing test batch {batch_idx}/{len(test_loader)}")
        
        questions = batch['cseqs']  # [B, L]
        masks = batch['masks']      # [B, L]
        batch_size, seq_len = questions.size()
        
        for i in range(batch_size):
            uid = batch.get('uid', [f'test_{batch_idx}_{i}'])[i]
            
            # Get student ability (use mean if unseen)
            theta = student_abilities.get(uid, mean_ability)
            
            # Get valid interactions
            valid_mask = masks[i] == 1
            q_valid = questions[i][valid_mask]
            
            if len(q_valid) == 0:
                continue
            
            # Get difficulties
            beta_vals = beta_tensor[q_valid]
            
            # Compute M_ref = σ(θ - β)
            m_ref = torch.sigmoid(theta - beta_vals)
            
            reference_predictions[uid] = m_ref
    
    print(f"✓ Computed reference predictions for {len(reference_predictions)} test students")
    
    return reference_predictions


def main():
    parser = argparse.ArgumentParser(description='Generate extended IRT targets for iKT3')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (assist2015, assist2009, etc.)')
    parser.add_argument('--fold', type=int, required=True, help='Fold number (0-4)')
    parser.add_argument('--rasch_path', type=str, required=True, help='Path to existing rasch calibration file')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for extended targets')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING EXTENDED IRT TARGETS FOR iKT3")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Rasch input: {args.rasch_path}")
    print(f"Output: {args.output_path}")
    print()
    
    # Initialize dataset
    print("Loading dataset...")
    
    # Load data config
    project_root = '/workspaces/pykt-toolkit'
    with open(os.path.join(project_root, 'configs/data_config.json')) as f:
        data_config = json.load(f)
    
    # Fix relative paths
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name] and data_config[dataset_name]['dpath'].startswith('../'):
            data_config[dataset_name]['dpath'] = os.path.join(project_root, data_config[dataset_name]['dpath'][3:])
    
    train_loader, valid_loader = init_dataset4train(
        args.dataset,
        'ikt3',  # model_name
        data_config,
        args.fold,
        batch_size=32
    )
    
    # Get number of skills from data config
    num_skills = data_config[args.dataset]['num_c']
    print(f"✓ Loaded dataset with {num_skills} skills")
    print(f"  Train: {len(train_loader.dataset)} students")
    print(f"  Valid: {len(valid_loader.dataset)} students")
    print()
    
    # Step 1: Load skill difficulties
    print("Step 1: Loading skill difficulties...")
    skill_difficulties = load_rasch_difficulties(args.rasch_path, num_skills)
    print()
    
    # Step 2: Compute student abilities
    print("Step 2: Computing student abilities...")
    student_abilities = compute_student_abilities(
        train_loader, valid_loader, skill_difficulties, num_skills
    )
    print()
    
    # Step 3: Compute reference predictions for all students
    print("Step 3: Computing reference predictions M_ref = σ(θ - β)...")
    reference_predictions = {}
    beta_tensor = torch.tensor([skill_difficulties.get(k, 0.0) for k in range(num_skills)])
    mean_ability = np.mean(list(student_abilities.values()))
    
    # Process all data splits
    for loader_name, loader in [('train', train_loader), ('valid', valid_loader)]:
        print(f"  Processing {loader_name} predictions...")
        batch_count = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 100 == 0:
                print(f"    Batch {batch_idx}/{len(loader)}")
            
            questions = batch['cseqs']  # [B, L]
            masks = batch['masks']      # [B, L]
            batch_size, seq_len = questions.size()
            
            for i in range(batch_size):
                # Get student ID
                if 'uids' in batch:
                    uid = batch['uids'][i]
                    if isinstance(uid, torch.Tensor):
                        uid = uid.item()
                else:
                    uid = torch.tensor(batch_count)
                    batch_count += 1
                
                # Skip if already processed
                if uid in reference_predictions:
                    continue
                
                # Get student ability (use mean if unseen - shouldn't happen)
                theta = student_abilities.get(uid, mean_ability)
                
                # Get valid interactions
                valid_mask = masks[i] == 1
                q_valid = questions[i][valid_mask]
                
                if len(q_valid) == 0:
                    continue
                
                # Get difficulties for these questions
                beta_vals = beta_tensor[q_valid]
                
                # Compute M_ref = σ(θ - β) for this student's sequence
                m_ref = torch.sigmoid(torch.tensor(theta) - beta_vals)
                
                reference_predictions[uid] = m_ref.cpu()
        
        print(f"    Completed {loader_name}: {len(reference_predictions)} total students")
    
    print(f"✓ Computed reference predictions for {len(reference_predictions)} students")
    print()
    
    # Save extended targets
    print("Saving extended IRT targets...")
    output_data = {
        'skill_difficulties': skill_difficulties,
        'student_abilities': student_abilities,
        'reference_predictions': reference_predictions,
        'metadata': {
            'dataset': args.dataset,
            'fold': args.fold,
            'num_skills': num_skills,
            'num_students': len(student_abilities),
            'num_test_students': len(reference_predictions),
            'source_rasch_file': args.rasch_path,
            'method': 'IRT Extended Targets for iKT3'
        }
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    # Verify file size
    file_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"✓ Saved to {args.output_path} ({file_size_mb:.1f} MB)")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skill difficulties: {len(skill_difficulties)}")
    print(f"Student abilities: {len(student_abilities)}")
    print(f"Reference predictions: {len(reference_predictions)}")
    print(f"Output file: {args.output_path}")
    print()
    print("✅ Extended IRT targets ready for iKT3 training!")


if __name__ == '__main__':
    main()
