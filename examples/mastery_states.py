#!/usr/bin/env python3
"""
Mastery States Analyzer for iKT3 Knowledge Tracing Model

This script extracts IRT-based mastery states for each skill at each time step.
For iKT3, mastery states represent M_IRT = σ(θ - β) where θ is student ability
and β is skill difficulty, following the Rasch IRT model.

Usage:
    # Analyze test set (default: 20 students)
    python examples/mastery_states.py --run_dir <experiment_dir> --split test
    
    # Analyze with custom number of students
    python examples/mastery_states.py --run_dir <experiment_dir> --split test --num_students 50
    
    # Analyze training set
    python examples/mastery_states.py --run_dir <experiment_dir> --split train --num_students 20
    
    # Run command from config.json (automatically generated during training)
    # The command is stored in config.json under commands.mastery_states

The script generates:
    - mastery_states_{split}.csv: Complete mastery state trajectory for all students
      Columns: student_id, time_step, question_id, skill_id, response, mastery_state
      
    - mastery_states_summary_{split}.json: Aggregate statistics about mastery progression
      Contains: per-skill statistics (mean, std, count, range), progression samples
      
Output Format:
    mastery_states_{split}.csv:
        - student_id: Unique student identifier
        - time_step: Sequential position in student's learning trajectory (0-indexed)
        - question_id: Question attempted at this time step
        - skill_id: Skill/concept being assessed (for single-skill: skill_id == question_id)
        - response: Student's response (1=correct, 0=incorrect)
        - mastery_state: IRT mastery probability M = σ(θ - β) for this skill at this time step
          (continuous value between 0 and 1, represents probability of mastery)
    
    mastery_states_summary_{split}.json:
        - total_observations: Total number of (student, time_step, skill) tuples
        - num_concepts: Total number of skills/concepts in the dataset
        - skills_observed: Number of skills that appeared in the data
        - skill_statistics: Per-skill aggregate metrics and temporal progression samples

Notes:
    - For iKT3: mastery states are IRT-based mastery probabilities from Head 2
    - For single-skill datasets (like assist2015): one skill per question
    - Mastery values are derived from learned θ (ability) and β (difficulty)
    - Values represent probability that student has mastered the skill
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import csv
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.datasets.data_loader import KTDataset
from torch.utils.data import DataLoader
from pykt.models.ikt3 import iKT3


def load_model_and_config(run_dir, ckpt_name):
    """Load trained iKT3 model and configuration."""
    # Load config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    # Merge defaults with overrides (overrides take precedence)
    config = full_config['defaults'].copy()
    config.update(full_config.get('overrides', {}))
    
    # Setup data config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'data_config.json')
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    # Convert relative paths to absolute paths
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name] and data_config[dataset_name]['dpath'].startswith('../'):
            data_config[dataset_name]['dpath'] = os.path.join(project_root, data_config[dataset_name]['dpath'][3:])
    
    if config['dataset'] not in data_config:
        raise ValueError(f"Dataset '{config['dataset']}' not found in data_config.json. Available: {list(data_config.keys())}")
    
    num_c = data_config[config['dataset']]['num_c']
    
    # Initialize iKT3 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iKT3(
        num_c=num_c,
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_blocks=config['num_encoder_blocks'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        emb_type=config['emb_type'],
        reference_model_type=config.get('reference_model', 'irt'),
        lambda_target=config.get('lambda_target', 0.5),
        warmup_epochs=config.get('warmup_epochs', 50),
        c_stability_reg=config.get('c_stability_reg', 0.1)
    ).to(device)
    
    # Multi-GPU support: wrap model with DataParallel if multiple GPUs available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Load checkpoint
    checkpoint_path = os.path.join(run_dir, ckpt_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, data_config, device, num_c


def extract_mastery_states(model, data_loader, device, num_concepts, max_students=None):
    """
    Extract mastery states for each skill at each time step.
    
    Args:
        max_students: Maximum number of students to process (None for all)
    
    Returns:
        mastery_data: List of dicts with student_id, time_step, question_id, skills, responses, mastery_states
    """
    model.eval()
    mastery_data = []
    students_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Check if we've reached the student limit
            if max_students is not None and students_processed >= max_students:
                break
            
            questions = batch['cseqs'].to(device)  # [B, L]
            responses = batch['rseqs'].to(device)  # [B, L]
            questions_shifted = batch['shft_cseqs'].to(device)  # [B, L]
            mask = batch['masks'].to(device)  # [B, L]
            
            # Get student IDs if available
            if 'uids' in batch:
                student_ids = batch['uids'].cpu().numpy()
            else:
                # Generate sequential IDs if not available
                student_ids = np.arange(batch_idx * questions.shape[0], 
                                       (batch_idx + 1) * questions.shape[0])
            
            # Forward pass to get IRT mastery states
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            
            # For iKT3, use mastery_irt which is [B, L] - mastery probability per question
            # This represents M = σ(θ - β) for each question at each timestep
            if 'mastery_irt' not in outputs:
                print("\n" + "="*80)
                print("⚠️  MASTERY STATES NOT AVAILABLE")
                print("="*80)
                print("The model does not provide mastery_irt outputs.")
                print("="*80)
                return []
            
            # mastery_irt contains IRT mastery probability for each question at each time step
            # Shape: [B, L] - one mastery value per (student, timestep) for the question answered
            mastery_irt = outputs['mastery_irt'].cpu().numpy()  # [B, L]
            
            # Convert to numpy for processing
            questions_np = questions.cpu().numpy()
            responses_np = responses.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Process each student in the batch
            batch_size, seq_len = questions_np.shape
            
            for student_idx in range(batch_size):
                # Check student limit
                if max_students is not None and students_processed >= max_students:
                    break
                
                student_id = student_ids[student_idx]
                students_processed += 1
                
                # Process each time step
                for time_step in range(seq_len):
                    # Check if this is a valid step (not padding)
                    if mask_np[student_idx, time_step] == 0:
                        continue
                    
                    question_id = int(questions_np[student_idx, time_step])
                    response = int(responses_np[student_idx, time_step])
                    
                    # Get IRT mastery probability for this question at this time step
                    # For iKT3: mastery_irt is [B, L], one value per question answered
                    mastery_value = float(mastery_irt[student_idx, time_step])
                    
                    # For single-skill datasets: question_id corresponds to skill_id
                    # For iKT3, we track the mastery for the skill being assessed
                    skills_involved = [question_id]  # Skill ID = Question ID for single-skill datasets
                    
                    # Mastery value represents M_IRT = σ(θ - β) for this question
                    mastery_values = {
                        skill_id: mastery_value 
                        for skill_id in skills_involved if skill_id < num_concepts
                    }
                    
                    # Store the data
                    mastery_data.append({
                        'student_id': int(student_id),
                        'time_step': int(time_step),
                        'question_id': question_id,
                        'skills': skills_involved,
                        'response': response,
                        'mastery_states': mastery_values,
                        'batch_idx': batch_idx
                    })
    
    return mastery_data


def compute_mastery_statistics(mastery_data, num_concepts):
    """Compute aggregate statistics about mastery progression."""
    
    # Organize by skill
    skill_mastery = defaultdict(list)
    skill_progression = defaultdict(lambda: defaultdict(list))
    
    for entry in mastery_data:
        for skill_id, mastery_value in entry['mastery_states'].items():
            skill_mastery[skill_id].append(mastery_value)
            skill_progression[skill_id][entry['time_step']].append(mastery_value)
    
    # Compute statistics
    statistics = {
        'total_observations': len(mastery_data),
        'num_concepts': num_concepts,
        'skills_observed': len(skill_mastery),
        'skill_statistics': {}
    }
    
    for skill_id in skill_mastery:
        values = skill_mastery[skill_id]
        statistics['skill_statistics'][int(skill_id)] = {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
        
        # Add progression over time if available
        if skill_progression[skill_id]:
            time_steps = sorted(skill_progression[skill_id].keys())
            progression = [
                {
                    'time_step': int(t),
                    'mean_mastery': float(np.mean(skill_progression[skill_id][t])),
                    'count': len(skill_progression[skill_id][t])
                }
                for t in time_steps[:10]  # First 10 time steps as example
            ]
            statistics['skill_statistics'][int(skill_id)]['progression_sample'] = progression
    
    return statistics


def save_mastery_states_csv(mastery_data, output_path):
    """Save mastery states to CSV file."""
    
    # Flatten the data for CSV
    rows = []
    for entry in mastery_data:
        for skill_id, mastery_value in entry['mastery_states'].items():
            rows.append({
                'student_id': entry['student_id'],
                'time_step': entry['time_step'],
                'question_id': entry['question_id'],
                'skill_id': skill_id,
                'response': entry['response'],
                'mastery_state': mastery_value
            })
    
    # Write to CSV
    if rows:
        fieldnames = ['student_id', 'time_step', 'question_id', 'skill_id', 'response', 'mastery_state']
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Extract mastery states from trained model')
    parser.add_argument('--run_dir', type=str, required=True, 
                       help='Experiment directory containing model checkpoint')
    parser.add_argument('--ckpt_name', type=str, default='best_model.pt',
                       help='Checkpoint filename (default: best_model.pt)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                       help='Data split to analyze (default: test)')
    parser.add_argument('--num_students', type=int, default=20,
                       help='Number of students to process (default: 20)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MASTERY STATES ANALYZER")
    print("="*80)
    print(f"Run directory: {args.run_dir}")
    print(f"Checkpoint: {args.ckpt_name}")
    print(f"Data split: {args.split}")
    print("="*80)
    
    # Load model and config
    print("\n📊 Loading model and configuration...")
    model, config, data_config, device, num_concepts = load_model_and_config(
        args.run_dir, args.ckpt_name
    )
    print(f"✓ Model loaded successfully")
    print(f"✓ Number of concepts: {num_concepts}")
    
    # Load data
    print(f"\n📚 Loading {args.split} data...")
    
    if args.split in ['train', 'valid']:
        # Use train/valid loaders
        train_loader, valid_loader = init_dataset4train(
            config['dataset'], 'gainakt4', data_config, config['fold'], config['batch_size']
        )
        data_loader = train_loader if args.split == 'train' else valid_loader
    else:
        # Load test data
        test_cfg = data_config[config['dataset']]
        test_dataset = KTDataset(
            os.path.join(test_cfg['dpath'], test_cfg['test_file']),
            test_cfg['input_type'],
            {-1}
        )
        data_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=int(os.getenv('PYKT_NUM_WORKERS', '4')),
            pin_memory=True
        )
    
    print(f"✓ Data loaded: {len(data_loader)} batches")
    
    # Extract mastery states
    print(f"\n🔍 Extracting mastery states (max {args.num_students} students)...")
    mastery_data = extract_mastery_states(model, data_loader, device, num_concepts, max_students=args.num_students)
    
    # Check if mastery data is available
    if not mastery_data:
        print("\n" + "="*80)
        print("MASTERY STATES EXTRACTION SKIPPED")
        print("="*80)
        print("No mastery data available. This occurs when:")
        print("  - Model trained with λ_bce=1.0 (pure BCE mode, Head 2 disabled)")
        print("  - No students in dataset (empty data loader)")
        print("\nNo output files created.")
        print("="*80)
        return
    
    print(f"✓ Extracted {len(mastery_data)} observations from up to {args.num_students} students")
    
    # Compute statistics
    print("\n📈 Computing statistics...")
    statistics = compute_mastery_statistics(mastery_data, num_concepts)
    print(f"✓ Analyzed {statistics['skills_observed']} skills")
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save CSV
    csv_path = os.path.join(args.run_dir, f'mastery_states_{args.split}.csv')
    num_rows = save_mastery_states_csv(mastery_data, csv_path)
    print(f"✓ Saved {num_rows} rows to: {csv_path}")
    
    # Save statistics JSON
    stats_path = os.path.join(args.run_dir, f'mastery_states_summary_{args.split}.json')
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Saved summary to: {stats_path}")
    
    # Print sample statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total observations: {statistics['total_observations']:,}")
    print(f"Skills observed: {statistics['skills_observed']}")
    print(f"\nSample skill statistics (first 5 skills):")
    
    for skill_id in sorted(statistics['skill_statistics'].keys())[:5]:
        stats = statistics['skill_statistics'][skill_id]
        print(f"\n  Skill {skill_id}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean mastery: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    print("\n" + "="*80)
    print("✅ Mastery states analysis completed successfully")
    print("="*80)


if __name__ == '__main__':
    main()
