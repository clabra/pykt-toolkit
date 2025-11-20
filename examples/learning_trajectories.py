#!/usr/bin/env python3
"""
Learning Trajectories Analyzer for GainAKT3Exp

Extracts and displays detailed learning trajectories for individual students,
showing timestep-by-timestep evolution of mastery and gains for practiced skills.

Usage:
    python examples/learning_trajectories.py \
        --run_dir examples/experiments/20251115_164618_gainakt3exp_baseline_defaults_114045 \
        --num_trajectories 10 \
        --min_steps 10

Output:
    Prints detailed trajectories to console showing:
    - Student ID
    - Number of interactions
    - Per-step: skills practiced, gains, mastery states, actual performance
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/workspaces/pykt-toolkit')
from pykt.models.gainakt3_exp import create_exp_model
from pykt.datasets.data_loader import KTDataset


def load_model_and_config(run_dir):
    """Load trained model and configuration from experiment directory."""
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model configuration from defaults
    # Note: num_students will be detected from checkpoint later (gamma_student tensor size)
    defaults = config['defaults']
    model_config = {
        'seq_len': defaults['seq_len'],
        'd_model': defaults['d_model'],
        'n_heads': defaults['n_heads'],
        'num_encoder_blocks': defaults['num_encoder_blocks'],
        'd_ff': defaults['d_ff'],
        'dropout': defaults['dropout'],
        'emb_type': defaults['emb_type'],
        'use_mastery_head': defaults['use_mastery_head'],
        'intrinsic_gain_attention': defaults.get('intrinsic_gain_attention', False),
        'use_skill_difficulty': defaults.get('use_skill_difficulty', False),
        'use_student_speed': defaults.get('use_student_speed', False),
        'non_negative_loss_weight': defaults['non_negative_loss_weight'],
        'monotonicity_loss_weight': defaults['monotonicity_loss_weight'],
        'mastery_performance_loss_weight': defaults['mastery_performance_loss_weight'],
        'gain_performance_loss_weight': defaults['gain_performance_loss_weight'],
        'sparsity_loss_weight': defaults['sparsity_loss_weight'],
        'consistency_loss_weight': defaults['consistency_loss_weight'],
        'bce_loss_weight': defaults.get('bce_loss_weight', 0.9),
        'monitor_freq': defaults['monitor_freq'],
        'mastery_threshold_init': defaults['mastery_threshold_init'],
        'threshold_temperature': defaults['threshold_temperature']
    }
    
    dataset_name = defaults['dataset']
    fold = defaults['fold']
    
    return model_config, dataset_name, fold, config


def select_diverse_students(data_loader, num_trajectories=10, min_steps=10, max_students_to_check=500):
    """
    Select students with diverse trajectory lengths (trying to get both short and long).
    
    Returns list of (student_index, sequence_length) tuples.
    """
    candidates = []
    checked = 0
    
    for batch in data_loader:
        masks = batch['masks']
        B = masks.size(0)
        
        for i in range(B):
            if checked >= max_students_to_check:
                break
            
            seq_len = int(masks[i].sum().item())
            if seq_len >= min_steps:
                candidates.append((checked, seq_len))
            
            checked += 1
            
        if checked >= max_students_to_check:
            break
    
    if len(candidates) < num_trajectories:
        print(f"Warning: Only found {len(candidates)} students with >= {min_steps} steps")
        return candidates
    
    # Sort by sequence length
    candidates.sort(key=lambda x: x[1])
    
    # Select diverse range: take students distributed across the length range
    indices = np.linspace(0, len(candidates) - 1, num_trajectories).astype(int)
    selected = [candidates[i] for i in indices]
    
    return selected


def extract_trajectory(model, batch, student_idx_in_batch, device):
    """
    Extract detailed trajectory for a single student.
    
    Returns dict with:
        - steps: list of dicts, each containing:
            - timestep: int
            - skills_practiced: list of skill IDs
            - gains: dict {skill_id: gain_value}
            - mastery: dict {skill_id: mastery_value}
            - performance: 0 or 1 (actual response)
            - prediction: float (model's predicted probability)
    """
    q = batch['cseqs'].to(device)
    r = batch['rseqs'].to(device)
    qry = batch.get('shft_cseqs', q).to(device)
    responses = batch.get('shft_rseqs', r).to(device)
    mask = batch['masks'].to(device)
    
    # Get model outputs
    with torch.no_grad():
        model.eval()
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        outputs = core.forward_with_states(q=q, r=r, qry=qry)
    
    # Extract for specific student
    student_q = q[student_idx_in_batch]  # [seq_len]
    student_responses = responses[student_idx_in_batch]  # [seq_len]
    student_mask = mask[student_idx_in_batch]  # [seq_len]
    student_predictions = outputs['predictions'][student_idx_in_batch]  # [seq_len]
    
    # Get mastery and gains
    if 'projected_mastery' in outputs and 'projected_gains' in outputs:
        student_mastery = outputs['projected_mastery'][student_idx_in_batch]  # [seq_len, num_skills]
        student_gains = outputs['projected_gains'][student_idx_in_batch]  # [seq_len, num_skills]
    else:
        return None
    
    # Build trajectory
    trajectory = {'steps': []}
    valid_steps = student_mask.bool()
    num_steps = int(valid_steps.sum().item())
    
    for t in range(num_steps):
        skill_id = int(student_q[t].item())
        performance = int(student_responses[t].item())
        prediction = float(student_predictions[t].item())
        
        # Get mastery and gain for this skill at this timestep
        mastery_val = float(student_mastery[t, skill_id].item())
        gain_val = float(student_gains[t, skill_id].item())
        
        step_data = {
            'timestep': t + 1,
            'skills_practiced': [skill_id],  # In this dataset, typically 1 skill per interaction
            'gains': {skill_id: gain_val},
            'mastery': {skill_id: mastery_val},
            'performance': performance,
            'prediction': prediction
        }
        
        trajectory['steps'].append(step_data)
    
    return trajectory


def print_trajectory(student_idx, trajectory, global_idx):
    """Print trajectory in compact tabular format with student header."""
    if trajectory is None or not trajectory['steps']:
        print(f"\n{'='*120}")
        print(f"STUDENT #{student_idx}")
        print(f"{'='*120}")
        print(f"Global Index: {global_idx}")
        print(f"Status: NO VALID TRAJECTORY DATA")
        print(f"{'='*120}\n")
        return
    
    num_steps = len(trajectory['steps'])
    
    # Calculate statistics
    correct_count = sum(1 for step in trajectory['steps'] if step['performance'] == 1)
    accuracy = correct_count / num_steps if num_steps > 0 else 0.0
    unique_skills = len(set(step['skills_practiced'][0] for step in trajectory['steps']))
    
    # Student header with features
    print(f"\n{'='*120}")
    print(f"STUDENT #{student_idx}")
    print(f"{'='*120}")
    print(f"Global Index: {global_idx:>6} │ Total Interactions: {num_steps:>4} │ Unique Skills: {unique_skills:>3} │ Accuracy: {accuracy:>5.1%}")
    print(f"{'='*120}")
    
    # Column headers for this student's table
    print(f"{'Step':>4} │ {'Skill':>5} │ {'True':>4} │ {'Pred':>5} │ {'Match':>5} │ {'Gain':>6} │ {'Mastery':>7}")
    print(f"{'─'*4}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*7}")
    
    for step in trajectory['steps']:
        t = step['timestep']
        skill_id = step['skills_practiced'][0]
        true_ans = step['performance']
        pred = step['prediction']
        pred_ans = 1 if pred >= 0.5 else 0
        match = '✓' if true_ans == pred_ans else '✗'
        gain_val = step['gains'][skill_id]
        mastery_val = step['mastery'][skill_id]
        
        print(f"{t:4d} │ {skill_id:5d} │ {true_ans:4d} │ {pred:5.3f} │ {match:>5} │ {gain_val:6.4f} │ {mastery_val:7.4f}")
    
    print(f"{'='*120}\n")


def save_trajectories_to_csv(trajectories_data, csv_path):
    """
    Save trajectory data to CSV file.
    
    Each row represents one step in a student's learning trajectory.
    """
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'student_num', 'global_idx', 'step', 'skill_id', 
            'true_answer', 'prediction', 'correct', 
            'gain', 'mastery', 'total_steps', 'unique_skills'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for traj_data in trajectories_data:
            student_num = traj_data['student_num']
            global_idx = traj_data['global_idx']
            trajectory = traj_data['trajectory']
            
            if trajectory is None:
                continue
            
            steps = trajectory['steps']
            total_steps = len(steps)
            unique_skills = len(set(
                skill_id 
                for step in steps 
                for skill_id in step['skills_practiced']
            ))
            
            for step in steps:
                timestep = step['timestep']
                skill_id = step['skills_practiced'][0]  # Single skill per step
                true_answer = step['performance']
                prediction = step['prediction']
                correct = 1 if (true_answer == 1 and prediction >= 0.5) or (true_answer == 0 and prediction < 0.5) else 0
                gain = step['gains'][skill_id]
                mastery = step['mastery'][skill_id]
                
                writer.writerow({
                    'student_num': student_num,
                    'global_idx': global_idx,
                    'step': timestep,
                    'skill_id': skill_id,
                    'true_answer': true_answer,
                    'prediction': f'{prediction:.6f}',
                    'correct': correct,
                    'gain': f'{gain:.6f}',
                    'mastery': f'{mastery:.6f}',
                    'total_steps': total_steps,
                    'unique_skills': unique_skills
                })


def main():
    parser = argparse.ArgumentParser(
        description='Extract and display learning trajectories for individual students'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to experiment directory containing model_best.pth and config.json')
    parser.add_argument('--num_trajectories', type=int, default=10,
                        help='Number of students to analyze (default: 10)')
    parser.add_argument('--min_steps', type=int, default=10,
                        help='Minimum number of interaction steps required (default: 10)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for data loading (default: 1 for per-student processing)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV file path (default: trajectories_<timestamp>.csv in run_dir)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*120}")
    print(f"LEARNING TRAJECTORIES ANALYZER - GainAKT3Exp")
    print(f"{'='*120}")
    print(f"Experiment Directory: {args.run_dir}")
    print(f"Target Students: {args.num_trajectories} (with >= {args.min_steps} steps)")
    print(f"{'='*120}\n")
    
    # Check if directory exists
    if not os.path.exists(args.run_dir):
        print(f"ERROR: Experiment directory not found: {args.run_dir}")
        sys.exit(1)
    
    # Check for model checkpoint
    model_path = os.path.join(args.run_dir, 'model_best.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    print("Loading configuration and model...")
    model_config, dataset_name, fold, config = load_model_and_config(args.run_dir)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name} (fold {fold})...")
    
    # Data config (hardcoded for assist2015, can be extended)
    data_config = {
        dataset_name: {
            'dpath': f'/workspaces/pykt-toolkit/data/{dataset_name}',
            'num_q': 0,
            'num_c': 100,
            'input_type': ['concepts'],
            'max_concepts': 1,
            'min_seq_len': 3,
            'maxlen': model_config['seq_len'],
            'emb_path': '',
            'folds': [0,1,2,3,4],
            'train_valid_file': 'train_valid_sequences.csv',
            'test_file': 'test_sequences.csv',
            'test_window_file': 'test_window_sequences.csv'
        }
    }
    
    test_cfg = data_config[dataset_name]
    test_dataset = KTDataset(
        os.path.join(test_cfg['dpath'], test_cfg['test_file']),
        test_cfg['input_type'],
        {-1}
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model architecture...")
    num_skills = test_cfg['num_c']
    
    # Build complete config dict (copy all defaults and add num_c from dataset)
    defaults = config['defaults']
    complete_config = dict(defaults)  # Copy all defaults
    complete_config['num_c'] = num_skills  # Override with actual dataset skill count
    
    # Load checkpoint first to detect num_students from gamma_student tensor
    print(f"Loading checkpoint to detect configuration: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect actual num_students from checkpoint (gamma_student tensor size)
    if 'gamma_student' in state_dict:
        actual_num_students = state_dict['gamma_student'].shape[0]
        complete_config['num_students'] = actual_num_students
        print(f"Detected num_students from checkpoint: {actual_num_students}")
    elif 'module.gamma_student' in state_dict:
        actual_num_students = state_dict['module.gamma_student'].shape[0]
        complete_config['num_students'] = actual_num_students
        print(f"Detected num_students from checkpoint: {actual_num_students}")
    
    # Calculate incremental_mastery_loss_weight if not present
    if 'incremental_mastery_loss_weight' not in complete_config:
        bce_weight = complete_config.get('bce_loss_weight', 0.9)
        complete_config['incremental_mastery_loss_weight'] = 1.0 - bce_weight
        print(f"Calculated incremental_mastery_loss_weight: {complete_config['incremental_mastery_loss_weight']}")
    
    # Map monitor_freq to monitor_frequency if needed
    if 'monitor_freq' in complete_config and 'monitor_frequency' not in complete_config:
        complete_config['monitor_frequency'] = complete_config['monitor_freq']
    
    # Create model with complete configuration
    print("Creating model architecture...")
    model = create_exp_model(complete_config)
    
    # Load trained weights
    print(f"Loading model weights into architecture...")
    
    # Remove 'module.' prefix if present (from DataParallel training)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("\nSelecting diverse student sample...")
    selected_students = select_diverse_students(
        test_loader, 
        num_trajectories=args.num_trajectories,
        min_steps=args.min_steps
    )
    
    if not selected_students:
        print("ERROR: No students found meeting criteria")
        sys.exit(1)
    
    print(f"Selected {len(selected_students)} students:")
    for i, (global_idx, seq_len) in enumerate(selected_students, 1):
        print(f"  Student {i}: Global Index={global_idx}, Sequence Length={seq_len}")
    
    print("\n" + "="*120)
    print("EXTRACTING TRAJECTORIES...")
    print("="*120)
    
    # Extract and print trajectories
    trajectories_data = []
    current_global_idx = 0
    
    for student_num, (target_global_idx, expected_len) in enumerate(selected_students, 1):
        # Skip batches until we reach target student
        for batch in test_loader:
            B = batch['cseqs'].size(0)
            
            if current_global_idx <= target_global_idx < current_global_idx + B:
                # Found the batch containing this student
                student_idx_in_batch = target_global_idx - current_global_idx
                
                trajectory = extract_trajectory(model, batch, student_idx_in_batch, device)
                trajectories_data.append({
                    'student_num': student_num,
                    'global_idx': target_global_idx,
                    'trajectory': trajectory
                })
                
                print_trajectory(student_num, trajectory, target_global_idx)
                break
            
            current_global_idx += B
        
        # Reset for next student
        current_global_idx = 0
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\n" + "="*120)
    print("TRAJECTORY EXTRACTION COMPLETE")
    print("="*120)
    print(f"Analyzed {len(trajectories_data)} students from test set")
    print(f"Results show: Step | Skill | True answer (0/1) | Predicted probability | Match | Gain | Mastery")
    print("="*120 + "\n")
    
    # Save to CSV
    if args.output_csv is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(args.run_dir, f'trajectories_{timestamp}.csv')
    else:
        csv_path = args.output_csv
    
    print(f"Saving trajectories to CSV: {csv_path}")
    save_trajectories_to_csv(trajectories_data, csv_path)
    print(f"CSV file saved successfully\n")


if __name__ == '__main__':
    main()
