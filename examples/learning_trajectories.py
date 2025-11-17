#!/usr/bin/env python3
"""
Learning Trajectories Analyzer for GainAKT3Exp - CSV Export Version

Extracts detailed learning trajectories for individual students and exports to CSV format
for easier analysis, filtering, and visualization.

Usage:
    python examples/learning_trajectories_csv.py \
        --run_dir examples/experiments/20251116_210414_gainakt3exp_baseline-before-warm_223545 \
        --num_students 10 \
        --min_steps 10 \
        --output trajectories.csv

Output:
    CSV file with columns:
    - student_idx: Student index in batch
    - global_idx: Global student ID in dataset
    - step: Step number (1-indexed)
    - skill_id: Skill/concept ID
    - actual_response: True response (0/1)
    - encoder1_pred: Encoder 1 prediction probability
    - encoder2_pred: Encoder 2 prediction probability
    - encoder1_match: Whether Encoder 1 prediction matches actual (0/1)
    - encoder2_match: Whether Encoder 2 prediction matches actual (0/1)
    - mastery: Mastery level for the skill [0-1]
    - expected_gain: Expected learning gain
    - mastery_threshold: Threshold used for binary classification (from config)
"""

import os
import sys
import json
import csv
import math
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
    
    # Build complete model config including all architectural and interpretability parameters
    model_config = {
        'seq_len': defaults['seq_len'],
        'd_model': defaults['d_model'],
        'n_heads': defaults['n_heads'],
        'num_encoder_blocks': defaults['num_encoder_blocks'],
        'd_ff': defaults['d_ff'],
        'dropout': defaults['dropout'],
        'emb_type': defaults['emb_type'],
        'use_mastery_head': defaults.get('use_mastery_head', True),
        'use_gain_head': defaults.get('use_gain_head', False),
        'use_skill_difficulty': defaults.get('use_skill_difficulty', False),
        'use_student_speed': defaults.get('use_student_speed', False),
        'mastery_threshold_init': defaults.get('mastery_threshold_init', 0.6),
        'threshold_temperature': defaults.get('threshold_temperature', 1.5),
        'beta_skill_init': defaults.get('beta_skill_init', 2.5),
        'm_sat_init': defaults.get('m_sat_init', 0.7),
        'gamma_student_init': defaults.get('gamma_student_init', 1.1),
        'sigmoid_offset': defaults.get('sigmoid_offset', 1.5),
    }
    
    dataset_name = defaults['dataset']
    fold = defaults['fold']
    
    return model_config, dataset_name, fold, config


def extract_student_trajectory(batch, batch_idx, student_idx_in_batch, outputs, device):
    """
    Extract trajectory for a single student from batch outputs.
    
    Returns dict with:
    - steps: List of dicts, each containing timestep data
    """
    # Get student data from batch
    q = batch['cseqs'].to(device)
    r = batch['rseqs'].to(device)
    responses = batch.get('shft_rseqs', r).to(device)
    mask = batch['masks'].to(device)
    
    student_q = q[student_idx_in_batch]  # [seq_len]
    student_responses = responses[student_idx_in_batch]  # [seq_len]
    student_mask = mask[student_idx_in_batch]  # [seq_len]
    
    # Encoder 1: Base performance predictions
    student_predictions_encoder1 = outputs['predictions'][student_idx_in_batch]  # [seq_len]
    
    # Encoder 2: Incremental Mastery predictions
    if 'incremental_mastery_predictions' in outputs:
        student_predictions_encoder2 = outputs['incremental_mastery_predictions'][student_idx_in_batch]
    else:
        student_predictions_encoder2 = None
    
    # Get mastery
    if 'projected_mastery' not in outputs:
        return None
    
    student_mastery = outputs['projected_mastery'][student_idx_in_batch]  # [seq_len, num_skills]
    
    # Get gains
    if 'projected_gains' in outputs:
        student_gains = outputs['projected_gains'][student_idx_in_batch]
    elif 'projected_gains_d' in outputs:
        student_gains = outputs['projected_gains_d'][student_idx_in_batch]
    else:
        student_gains = torch.zeros_like(student_mastery)
    
    # Build trajectory
    trajectory = {'steps': []}
    valid_steps = student_mask.bool()
    num_steps = int(valid_steps.sum().item())
    
    for t in range(num_steps):
        # Extract skill IDs
        skill_ids = []
        skill_id_raw = student_q[t].item()
        if isinstance(skill_id_raw, (list, tuple)):
            skill_ids = [int(sid) for sid in skill_id_raw if sid >= 0]
        else:
            skill_id = int(skill_id_raw)
            if skill_id >= 0:
                skill_ids = [skill_id]
        
        if not skill_ids:
            continue
        
        # True performance
        performance = int(student_responses[t].item())
        
        # Predictions
        prediction_encoder1 = float(student_predictions_encoder1[t].item())
        prediction_encoder2 = float(student_predictions_encoder2[t].item()) if student_predictions_encoder2 is not None else None
        
        # Collect gains and mastery for all skills
        gains_dict = {}
        mastery_dict = {}
        
        for skill_id in skill_ids:
            if skill_id < student_mastery.shape[-1]:
                mastery_val = float(student_mastery[t, skill_id].item())
                gain_val = float(student_gains[t, skill_id].item())
                gains_dict[skill_id] = gain_val
                mastery_dict[skill_id] = mastery_val
        
        step_data = {
            'timestep': t + 1,
            'skills_practiced': skill_ids,
            'gains': gains_dict,
            'mastery': mastery_dict,
            'true_response': performance,
            'prediction_encoder1': prediction_encoder1,
            'prediction_encoder2': prediction_encoder2
        }
        
        trajectory['steps'].append(step_data)
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description='Extract learning trajectories to CSV format'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--num_students', type=int, default=10,
                        help='Number of students to analyze (default: 10)')
    parser.add_argument('--min_steps', type=int, default=10,
                        help='Minimum interaction steps required (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (default: <run_dir>/learning_trajectories.csv)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for data loading (default: 1)')
    
    args = parser.parse_args()
    
    # Set output path - save in the experiment directory by default
    if args.output is None:
        args.output = os.path.join(args.run_dir, 'learning_trajectories.csv')
    
    print(f"\n{'='*120}")
    print(f"LEARNING TRAJECTORIES ANALYZER - CSV EXPORT")
    print(f"{'='*120}")
    print(f"Experiment Directory: {args.run_dir}")
    print(f"Target Students: {args.num_students} (with >= {args.min_steps} steps)")
    print(f"Output CSV: {args.output}")
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
    
    # Get mastery threshold from config (initial value for reference)
    mastery_threshold_init = model_config.get('mastery_threshold_init', 0.6)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name} (fold {fold})...")
    
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
    num_students_from_dataset = len(test_dataset)
    
    # Get num_students from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # gamma_student contains per-student learning rates
    num_students_from_checkpoint = checkpoint['model_state_dict']['gamma_student'].shape[0]
    print(f"Using num_students={num_students_from_checkpoint} from checkpoint")
    
    # Extract actual learned parameters from checkpoint
    # Note: These are the final learned values after training, not the initialization values
    # Training script saves these to learned_parameters_epoch.csv during training
    theta_global_actual = checkpoint['model_state_dict']['theta_global'].item()
    
    # Check if learned parameters were tracked during training
    params_csv_path = os.path.join(args.run_dir, 'learned_parameters_epoch.csv')
    if os.path.exists(params_csv_path):
        print(f"Found learned parameters tracking file: {params_csv_path}")
        import pandas as pd
        params_df = pd.read_csv(params_csv_path)
        final_epoch = params_df.iloc[-1]
        print(f"  Final epoch: {int(final_epoch['epoch'])}")
        print(f"  theta_global: {final_epoch['theta_global']:.6f} (init: {mastery_threshold_init:.6f})")
        print(f"  beta_skill: mean={final_epoch['beta_skill_mean']:.6f}, std={final_epoch['beta_skill_std']:.6f}")
        print(f"  M_sat: mean={final_epoch['M_sat_mean']:.6f}, std={final_epoch['M_sat_std']:.6f}")
        print(f"  gamma_student: mean={final_epoch['gamma_student_mean']:.6f}, std={final_epoch['gamma_student_std']:.6f}")
    else:
        print(f"Learned theta_global (mastery threshold): {theta_global_actual:.6f} (init was {mastery_threshold_init:.6f})")
        print(f"  Note: No learned_parameters_epoch.csv found. Using values from checkpoint.")
    
    # Get threshold temperature from config (this is a hyperparameter, not learned)
    threshold_temperature = model_config.get('threshold_temperature', 1.5)
    
    # Save learned parameters summary to experiment directory for reference
    learned_params = {
        'theta_global': float(theta_global_actual),
        'theta_global_init': float(mastery_threshold_init),
        'threshold_temperature': float(threshold_temperature),
        'beta_skill_mean': float(checkpoint['model_state_dict']['beta_skill'].mean().item()),
        'beta_skill_std': float(checkpoint['model_state_dict']['beta_skill'].std().item()),
        'M_sat_mean': float(checkpoint['model_state_dict']['M_sat'].mean().item()),
        'M_sat_std': float(checkpoint['model_state_dict']['M_sat'].std().item()),
        'gamma_student_mean': float(checkpoint['model_state_dict']['gamma_student'].mean().item()),
        'gamma_student_std': float(checkpoint['model_state_dict']['gamma_student'].std().item()),
    }
    
    learned_params_path = os.path.join(args.run_dir, 'learned_parameters.json')
    with open(learned_params_path, 'w') as f:
        json.dump(learned_params, f, indent=2)
    print(f"Saved learned parameters summary to: {learned_params_path}")
    
    # Check if model_config is in checkpoint (new format)
    if 'model_config' in checkpoint:
        print("Using model_config from checkpoint")
        complete_config = checkpoint['model_config']
    else:
        # Build from loaded config + dataset info
        defaults = config['defaults']
        complete_config = {
            'num_c': num_skills,
            'seq_len': defaults['seq_len'],
            'd_model': defaults['d_model'],
            'n_heads': defaults['n_heads'],
            'num_encoder_blocks': defaults['num_encoder_blocks'],
            'd_ff': defaults['d_ff'],
            'dropout': defaults['dropout'],
            'emb_type': defaults['emb_type'],
            'use_mastery_head': defaults.get('use_mastery_head', True),
            'use_gain_head': defaults.get('use_gain_head', False),
            'intrinsic_gain_attention': defaults.get('intrinsic_gain_attention', False),
            'use_skill_difficulty': defaults.get('use_skill_difficulty', False),
            'use_student_speed': defaults.get('use_student_speed', False),
            'num_students': num_students_from_checkpoint,
            'non_negative_loss_weight': defaults.get('non_negative_loss_weight', 0.0),
            'monotonicity_loss_weight': defaults.get('monotonicity_loss_weight', 0.0),
            'mastery_performance_loss_weight': defaults.get('mastery_performance_loss_weight', 0.0),
            'gain_performance_loss_weight': defaults.get('gain_performance_loss_weight', 0.0),
            'sparsity_loss_weight': defaults.get('sparsity_loss_weight', 0.0),
            'consistency_loss_weight': defaults.get('consistency_loss_weight', 0.0),
            'incremental_mastery_loss_weight': defaults.get('incremental_mastery_loss_weight', 1.0),
            'monitor_frequency': defaults.get('monitor_frequency', 10),
            'mastery_threshold_init': defaults.get('mastery_threshold_init', 0.6),
            'threshold_temperature': defaults.get('threshold_temperature', 1.5),
            'beta_skill_init': defaults.get('beta_skill_init', 2.5),
            'm_sat_init': defaults.get('m_sat_init', 0.7),
            'gamma_student_init': defaults.get('gamma_student_init', 1.1),
            'sigmoid_offset': defaults.get('sigmoid_offset', 1.5),
        }
    
    model = create_exp_model(complete_config)
    model = model.to(device)
    
    # Load weights
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Open CSV file
    print(f"\nWriting trajectories to: {args.output}")
    csv_file = open(args.output, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write header
    csv_writer.writerow([
        'student_idx',
        'global_idx',
        'step',
        'skill_id',
        'actual_response',
        'encoder1_pred',
        'encoder2_pred',
        'encoder1_binary',
        'encoder2_binary',
        'encoder1_match',
        'encoder2_match',
        'mastery',
        'expected_gain',
        'theta_global',  # Actual learned threshold
        'threshold_temp',  # Temperature parameter
        'encoder2_pred_expected',  # Computed from formula for verification
        'total_steps',
        'unique_skills',
        'accuracy'
    ])
    
    # Process batches
    students_collected = 0
    batch_idx = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if students_collected >= args.num_students:
                break
            
            # Move batch to device and extract sequences
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            qry = batch.get('shft_cseqs', q).to(device)
            
            # Forward pass with states
            outputs = model.forward_with_states(q=q, r=r, qry=qry)
            
            # Process each student in batch
            batch_size = batch['cseqs'].shape[0]
            for student_idx_in_batch in range(batch_size):
                if students_collected >= args.num_students:
                    break
                
                # Extract trajectory
                trajectory = extract_student_trajectory(batch, batch_idx, student_idx_in_batch, outputs, device)
                
                if trajectory is None or len(trajectory['steps']) < args.min_steps:
                    continue
                
                # Calculate statistics
                total_steps = len(trajectory['steps'])
                all_skills = set()
                correct_count = 0
                
                for step in trajectory['steps']:
                    all_skills.update(step['skills_practiced'])
                    if step['true_response'] == 1:
                        correct_count += 1
                
                unique_skills = len(all_skills)
                accuracy = correct_count / total_steps if total_steps > 0 else 0.0
                
                # Global index (approximate based on batch processing)
                global_idx = batch_idx * args.batch_size + student_idx_in_batch
                
                # Write rows for each step
                for step in trajectory['steps']:
                    timestep = step['timestep']
                    actual = step['true_response']
                    enc1_pred = step['prediction_encoder1']
                    enc2_pred = step['prediction_encoder2']
                    
                    # Binary predictions
                    enc1_binary = 1 if enc1_pred >= 0.5 else 0
                    enc2_binary = 1 if enc2_pred >= 0.5 else 0 if enc2_pred is not None else None
                    
                    # Matches
                    enc1_match = 1 if enc1_binary == actual else 0
                    enc2_match = 1 if enc2_binary == actual else 0 if enc2_binary is not None else None
                    
                    # For each skill in this step
                    for skill_id in step['skills_practiced']:
                        mastery = step['mastery'].get(skill_id, 0.0)
                        gain = step['gains'].get(skill_id, 0.0)
                        
                        # Compute expected encoder2_pred from formula for verification
                        if mastery is not None and theta_global_actual is not None:
                            logit = (mastery - theta_global_actual) / threshold_temperature
                            enc2_pred_expected = 1 / (1 + math.exp(-logit))
                        else:
                            enc2_pred_expected = None
                        
                        csv_writer.writerow([
                            students_collected + 1,
                            global_idx,
                            timestep,
                            skill_id,
                            actual,
                            f"{enc1_pred:.6f}",
                            f"{enc2_pred:.6f}" if enc2_pred is not None else '',
                            enc1_binary,
                            enc2_binary if enc2_binary is not None else '',
                            enc1_match,
                            enc2_match if enc2_match is not None else '',
                            f"{mastery:.6f}",
                            f"{gain:.6f}",
                            f"{theta_global_actual:.6f}",  # Actual learned threshold
                            f"{threshold_temperature:.2f}",  # Temperature parameter
                            f"{enc2_pred_expected:.6f}" if enc2_pred_expected is not None else '',  # Expected value
                            total_steps,
                            unique_skills,
                            f"{accuracy:.4f}"
                        ])
                
                students_collected += 1
                print(f"Collected student {students_collected}/{args.num_students} (Global ID: {global_idx}, Steps: {total_steps}, Skills: {unique_skills})")
            
            batch_idx += 1
    
    csv_file.close()
    
    print(f"\n{'='*120}")
    print(f"TRAJECTORY EXTRACTION COMPLETE")
    print(f"{'='*120}")
    print(f"Total students analyzed: {students_collected}")
    print(f"Output saved to: {args.output}")
    print(f"\nYou can now analyze the trajectories using:")
    print(f"  - pandas: df = pd.read_csv('{args.output}')")
    print(f"  - Filter by student: df[df['student_idx'] == 10]")
    print(f"  - Filter by skill: df[df['skill_id'] == 64]")
    print(f"  - Check specific case: df[(df['student_idx'] == 10) & (df['skill_id'] == 64)]")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    main()
