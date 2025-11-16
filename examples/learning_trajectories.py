#!/usr/bin/env python3
"""
Learning Trajectories Analyzer for GainAKT3Exp

Extracts and displays detailed learning trajectories for individual students,
showing timestep-by-timestep evolution of mastery and gains for practiced skills.

Usage:
    python examples/learning_trajectories.py \
        --run_dir examples/experiments/20251115_164618_gainakt3exp_baseline_defaults_114045 \
        --num_students 10 \
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
    defaults = config['defaults']
    model_config = {
        'seq_len': defaults['seq_len'],
        'd_model': defaults['d_model'],
        'n_heads': defaults['n_heads'],
        'num_encoder_blocks': defaults['num_encoder_blocks'],
        'd_ff': defaults['d_ff'],
        'dropout': defaults['dropout'],
        'emb_type': defaults['emb_type'],
        'num_students': defaults['num_students'],
        'non_negative_loss_weight': defaults['non_negative_loss_weight'],
        'monotonicity_loss_weight': defaults['monotonicity_loss_weight'],
        'mastery_performance_loss_weight': defaults['mastery_performance_loss_weight'],
        'gain_performance_loss_weight': defaults['gain_performance_loss_weight'],
        'sparsity_loss_weight': defaults['sparsity_loss_weight'],
        'consistency_loss_weight': defaults['consistency_loss_weight'],
        'use_mastery_head': defaults['use_mastery_head'],
        'use_gain_head': defaults['use_gain_head'],
        'mastery_threshold_init': defaults['mastery_threshold_init'],
        'threshold_temperature': defaults['threshold_temperature']
    }
    
    dataset_name = defaults['dataset']
    fold = defaults['fold']
    
    return model_config, dataset_name, fold, config


def select_diverse_students(data_loader, num_students=10, min_steps=10, max_students_to_check=500):
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
    
    if len(candidates) < num_students:
        print(f"Warning: Only found {len(candidates)} students with >= {min_steps} steps")
        return candidates
    
    # Sort by sequence length
    candidates.sort(key=lambda x: x[1])
    
    # Select diverse range: take students distributed across the length range
    indices = np.linspace(0, len(candidates) - 1, num_students).astype(int)
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
    
    # DUAL-ENCODER PREDICTIONS (2025-11-16): Get predictions from both encoders
    # Encoder 1: Base predictions (Performance Path) - used for BCE Loss
    student_predictions_encoder1 = outputs['predictions'][student_idx_in_batch]  # [seq_len]
    
    # Encoder 2: Incremental Mastery predictions (Interpretability Path) - used for IM Loss
    if 'incremental_mastery_predictions' in outputs:
        student_predictions_encoder2 = outputs['incremental_mastery_predictions'][student_idx_in_batch]  # [seq_len]
    else:
        student_predictions_encoder2 = None
    
    # Get mastery (required) and gains (optional)
    if 'projected_mastery' not in outputs:
        return None
    
    student_mastery = outputs['projected_mastery'][student_idx_in_batch]  # [seq_len, num_skills]
    
    # Gains might not be available if use_gain_head=False
    # Priority: projected_gains > projected_gains_d > compute from value_seq_2 > zeros
    if 'projected_gains' in outputs:
        student_gains = outputs['projected_gains'][student_idx_in_batch]  # [seq_len, num_skills]
    elif 'projected_gains_d' in outputs:
        # Fall back to D-dimensional gains if available
        student_gains = outputs['projected_gains_d'][student_idx_in_batch]  # [seq_len, num_skills]
    elif 'value_seq' in outputs:
        # Compute gains from value sequence (Encoder 2 outputs)
        # Value sequence represents learning gains in the interpretability path
        value_seq_student = outputs['value_seq'][student_idx_in_batch]  # [seq_len, d_model]
        # Apply ReLU to get non-negative gains and project to skill space
        # For simplicity, take mean across d_model dimension as skill-level gain
        student_gains = torch.relu(value_seq_student).mean(dim=-1, keepdim=True)  # [seq_len, 1]
        # Broadcast to num_skills if needed
        num_skills = student_mastery.shape[-1]
        student_gains = student_gains.expand(-1, num_skills)  # [seq_len, num_skills]
    else:
        # If no gains available, use zero gains (mastery-only mode)
        student_gains = torch.zeros_like(student_mastery)
    
    # Build trajectory
    trajectory = {'steps': []}
    valid_steps = student_mask.bool()
    num_steps = int(valid_steps.sum().item())
    
    for t in range(num_steps):
        # MULTI-SKILL SUPPORT: Handle questions that may involve multiple skills
        # For assist2015, typically 1 skill per question, but this supports the general case
        skill_ids = []
        
        # Extract skill ID(s) from questions - support both single skill and multi-skill cases
        skill_id_raw = student_q[t].item()
        if isinstance(skill_id_raw, (list, tuple)):
            skill_ids = [int(sid) for sid in skill_id_raw if sid >= 0]
        else:
            skill_id = int(skill_id_raw)
            if skill_id >= 0:  # Filter out padding (-1)
                skill_ids = [skill_id]
        
        if not skill_ids:  # Skip if no valid skills
            continue
        
        # True performance (0/1)
        performance = int(student_responses[t].item())
        
        # DUAL-ENCODER PREDICTIONS: Collect from both encoders
        prediction_encoder1 = float(student_predictions_encoder1[t].item())
        prediction_encoder2 = float(student_predictions_encoder2[t].item()) if student_predictions_encoder2 is not None else None
        
        # Collect gains and mastery for all skills involved in this interaction
        gains_dict = {}
        mastery_dict = {}
        
        for skill_id in skill_ids:
            if skill_id < student_mastery.shape[-1]:  # Check bounds
                mastery_val = float(student_mastery[t, skill_id].item())
                gain_val = float(student_gains[t, skill_id].item())
                gains_dict[skill_id] = gain_val
                mastery_dict[skill_id] = mastery_val
        
        step_data = {
            'timestep': t + 1,
            'skills_practiced': skill_ids,  # List of skills (generic multi-skill support)
            'gains': gains_dict,  # Dict mapping skill_id -> gain value
            'mastery': mastery_dict,  # Dict mapping skill_id -> mastery value
            'true_response': performance,  # True response (0 or 1)
            'prediction_encoder1': prediction_encoder1,  # Encoder 1: Base prediction
            'prediction_encoder2': prediction_encoder2  # Encoder 2: IM prediction (or None)
        }
        
        trajectory['steps'].append(step_data)
    
    return trajectory


def print_trajectory(student_idx, trajectory, global_idx):
    """
    Print trajectory in tabular format with dual-encoder predictions and multi-skill support.
    
    DUAL-ENCODER DISPLAY (2025-11-16):
    - Shows predictions from both Encoder 1 (Base) and Encoder 2 (IM)
    - Displays learning gains and mastery per skill
    - Supports multi-skill questions (shows all skills practiced in each step)
    """
    if trajectory is None or not trajectory['steps']:
        print(f"\n{'='*150}")
        print(f"STUDENT #{student_idx}")
        print(f"{'='*150}")
        print(f"Global Index: {global_idx}")
        print("Status: NO VALID TRAJECTORY DATA")
        print(f"{'='*150}\n")
        return
    
    num_steps = len(trajectory['steps'])
    
    # Calculate statistics
    correct_count = sum(1 for step in trajectory['steps'] if step['true_response'] == 1)
    accuracy = correct_count / num_steps if num_steps > 0 else 0.0
    all_skills = set()
    for step in trajectory['steps']:
        all_skills.update(step['skills_practiced'])
    unique_skills = len(all_skills)
    
    # Student header with features
    print(f"\n{'='*150}")
    print(f"STUDENT #{student_idx}")
    print(f"{'='*150}")
    print(f"Global Index: {global_idx:>6} │ Total Interactions: {num_steps:>4} │ Unique Skills: {unique_skills:>3} │ Accuracy: {accuracy:>5.1%}")
    print(f"{'='*150}")
    
    # Check if Encoder 2 predictions are available
    has_encoder2 = trajectory['steps'][0]['prediction_encoder2'] is not None
    
    # Column headers - adapt based on encoder availability
    if has_encoder2:
        print(f"{'Step':>4} │ {'Skill(s)':>8} │ {'True':>4} │ {'Enc1':>5} │ {'Enc2':>5} │ {'M1':>3} │ {'M2':>3} │ {'Gain':>7} │ {'Mastery':>7}")
        print(f"{'─'*4}─┼─{'─'*8}─┼─{'─'*4}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*3}─┼─{'─'*3}─┼─{'─'*7}─┼─{'─'*7}")
    else:
        print(f"{'Step':>4} │ {'Skill(s)':>8} │ {'True':>4} │ {'Pred':>5} │ {'Match':>5} │ {'Gain':>7} │ {'Mastery':>7}")
        print(f"{'─'*4}─┼─{'─'*8}─┼─{'─'*4}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*7}")
    
    for step in trajectory['steps']:
        t = step['timestep']
        skills = step['skills_practiced']
        true_ans = step['true_response']
        
        # For multi-skill: show primary skill ID (first one) in compact view
        # Full skill list shown in skills column
        primary_skill = skills[0] if skills else -1
        skills_str = ','.join(str(s) for s in skills) if len(skills) <= 3 else f"{skills[0]},+{len(skills)-1}"
        
        if has_encoder2:
            # DUAL-ENCODER MODE: Show predictions from both encoders
            pred_enc1 = step['prediction_encoder1']
            pred_enc2 = step['prediction_encoder2']
            
            # Binary predictions and matches
            pred_ans_enc1 = 1 if pred_enc1 >= 0.5 else 0
            pred_ans_enc2 = 1 if pred_enc2 >= 0.5 else 0
            match_enc1 = '✓' if true_ans == pred_ans_enc1 else '✗'
            match_enc2 = '✓' if true_ans == pred_ans_enc2 else '✗'
            
            # Get gain and mastery for primary skill
            gain_val = step['gains'].get(primary_skill, 0.0)
            mastery_val = step['mastery'].get(primary_skill, 0.0)
            
            print(f"{t:4d} │ {skills_str:>8} │ {true_ans:4d} │ {pred_enc1:5.3f} │ {pred_enc2:5.3f} │ {match_enc1:>3} │ {match_enc2:>3} │ {gain_val:7.4f} │ {mastery_val:7.4f}")
        else:
            # SINGLE PREDICTION MODE (legacy): Show Encoder 1 prediction only
            pred = step['prediction_encoder1']
            pred_ans = 1 if pred >= 0.5 else 0
            match = '✓' if true_ans == pred_ans else '✗'
            
            # Get gain and mastery for primary skill
            gain_val = step['gains'].get(primary_skill, 0.0)
            mastery_val = step['mastery'].get(primary_skill, 0.0)
            
            print(f"{t:4d} │ {skills_str:>8} │ {true_ans:4d} │ {pred:5.3f} │ {match:>5} │ {gain_val:7.4f} │ {mastery_val:7.4f}")
    
    print(f"{'='*150}\n")
    
    # Print legend if dual-encoder mode
    if has_encoder2:
        print("Legend: Enc1=Encoder1 (Base Predictions), Enc2=Encoder2 (IM Predictions), M1=Match1, M2=Match2")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and display learning trajectories for individual students'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to experiment directory containing model_best.pth and config.json')
    parser.add_argument('--num_students', type=int, default=10,
                        help='Number of students to analyze (default: 10)')
    parser.add_argument('--min_steps', type=int, default=10,
                        help='Minimum number of interaction steps required (default: 10)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for data loading (default: 1 for per-student processing)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*120}")
    print(f"LEARNING TRAJECTORIES ANALYZER - GainAKT3Exp")
    print(f"{'='*120}")
    print(f"Experiment Directory: {args.run_dir}")
    print(f"Target Students: {args.num_students} (with >= {args.min_steps} steps)")
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
    
    # Load checkpoint first to get the correct model config
    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Build complete config dict (copy all defaults and add num_c from dataset)
    defaults = config['defaults']
    complete_config = dict(defaults)  # Copy all defaults
    complete_config['num_c'] = num_skills  # Override with actual dataset skill count
    
    # If checkpoint has model_config, use num_students from it (critical for gamma_student size)
    if 'model_config' in checkpoint and 'num_students' in checkpoint['model_config']:
        complete_config['num_students'] = checkpoint['model_config']['num_students']
        print(f"Using num_students={checkpoint['model_config']['num_students']} from checkpoint")
    
    # Map monitor_freq to monitor_frequency if needed
    if 'monitor_freq' in complete_config and 'monitor_frequency' not in complete_config:
        complete_config['monitor_frequency'] = complete_config['monitor_freq']
    # Compute incremental_mastery_loss_weight from bce_loss_weight if needed
    if 'bce_loss_weight' in complete_config and 'incremental_mastery_loss_weight' not in complete_config:
        complete_config['incremental_mastery_loss_weight'] = 1.0 - complete_config['bce_loss_weight']
    
    model = create_exp_model(complete_config)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel training)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("\nSelecting diverse student sample...")
    selected_students = select_diverse_students(
        test_loader, 
        num_students=args.num_students,
        min_steps=args.min_steps
    )
    
    if not selected_students:
        print("No students found meeting the criteria (min_steps >= {})".format(args.min_steps))
        return
    
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


if __name__ == '__main__':
    main()
