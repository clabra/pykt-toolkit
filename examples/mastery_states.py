#!/usr/bin/env python3
"""
Mastery States Analyzer for Knowledge Tracing Models

This script calculates mastery states for each skill practiced in each question
at each time step. Works with both single-skill and multi-skill questions.

Supported Models:
    - iKT: Extracts {Mi} skill vectors from Head 2 (mastery head)
    - GainAKT4: Extracts skill_vector (KC vector) from mastery representations

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
        - mastery_state: Model's estimated mastery level for this skill at this time step
          (continuous value from positivity constraint, monotonically increasing over time)
    
    mastery_states_summary_{split}.json:
        - total_observations: Total number of (student, time_step, skill) tuples
        - num_concepts: Total number of skills/concepts in the dataset
        - skills_observed: Number of skills that appeared in the data
        - skill_statistics: Per-skill aggregate metrics and temporal progression samples

Notes:
    - For single-skill datasets (like assist2015): one skill per question
    - For multi-skill datasets: multiple rows per question, one per skill
    - Mastery states are extracted from the model's skill_vector (KC vector)
    - Monotonicity constraint ensures mastery_state[t+1] >= mastery_state[t]
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
from pykt.models.gainakt4 import GainAKT4
from pykt.models.ikt import iKT
from pykt.models.ikt2 import iKT2


def load_rasch_targets(dataset_name, data_config):
    """
    Load pre-computed Rasch IRT targets for L2 loss input.
    
    Args:
        dataset_name: Name of dataset
        data_config: Dataset configuration dict
    
    Returns:
        dict: Rasch targets or {'mode': 'random'} if not available
    """
    import pickle
    
    # Get dataset path
    dataset_cfg = data_config.get(dataset_name, {})
    dataset_path = dataset_cfg.get('dpath', f'/workspaces/pykt-toolkit/data/{dataset_name}')
    rasch_path = os.path.join(dataset_path, 'rasch_targets.pkl')
    
    # Try to load from file
    if os.path.exists(rasch_path):
        print(f"✓ Loading Rasch targets from: {rasch_path}")
        try:
            with open(rasch_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate
            if 'rasch_targets' not in data:
                raise ValueError("Invalid Rasch file: missing 'rasch_targets' key")
            
            print(f"  Loaded targets for {len(data['rasch_targets'])} students")
            return data
            
        except Exception as e:
            print(f"✗ Failed to load Rasch targets: {e}")
    else:
        print(f"⚠️  Rasch targets not found at: {rasch_path}")
        print("  L2 (Rasch Loss) inputs will be empty in CSV")
    
    # Fallback: no Rasch targets
    return {'mode': 'random'}


def load_model_and_config(run_dir, ckpt_name):
    """Load trained model and configuration."""
    # Load config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    #config = full_config['defaults']
    # Merge defaults with overrides (overrides take precedence)
    config = full_config['defaults'].copy()
    config.update(full_config.get('overrides', {}))
    
    # Setup data config
    # Load data config from configs/data_config.json
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
    
    # Initialize model based on model type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config.get('model', 'gainakt4')
    
    if model_name == 'ikt':
        # iKT model
        model = iKT(
            num_c=num_c,
            seq_len=config['seq_len'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_encoder_blocks=config['num_encoder_blocks'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            emb_type=config['emb_type'],
            lambda_penalty=config['lambda_penalty'],
            epsilon=config['epsilon'],
            phase=config.get('phase')
        ).to(device)
    elif model_name == 'ikt2':
        # iKT2 model
        model = iKT2(
            num_c=num_c,
            seq_len=config['seq_len'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_encoder_blocks=config['num_encoder_blocks'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            emb_type=config['emb_type'],
            lambda_align=config.get('lambda_align', 1.0),
            phase=config.get('phase', 1)
        ).to(device)
    else:
        # GainAKT4 model (default)
        model = GainAKT4(
            num_c=num_c,
            seq_len=config['seq_len'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_encoder_blocks=config['num_encoder_blocks'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            emb_type=config['emb_type'],
            lambda_bce=config['lambda_bce']
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


def select_students_by_sequence_length(data_loader, num_students_per_bin=3):
    """
    Select students stratified by sequence length.
    
    Creates 5 bins with 20% of sequence length range each, then randomly
    selects num_students_per_bin from each bin.
    
    Args:
        data_loader: DataLoader with student data
        num_students_per_bin: Number of students to select per bin (default: 3)
    
    Returns:
        set: Selected student IDs
    """
    import numpy as np
    
    # First pass: collect sequence lengths for all students
    student_lengths = {}
    
    for batch_idx, batch in enumerate(data_loader):
        mask = batch['masks'].cpu().numpy()  # [B, L]
        
        if 'uids' in batch:
            uids = batch['uids'].cpu().numpy()
        else:
            uids = np.arange(batch_idx * mask.shape[0], (batch_idx + 1) * mask.shape[0])
        
        # Count valid timesteps per student
        for i, uid in enumerate(uids):
            seq_len = int(mask[i].sum())
            if seq_len > 0:
                student_lengths[int(uid)] = seq_len
    
    if not student_lengths:
        print("⚠️  No students found in dataset")
        return set()
    
    # Calculate bins
    lengths = np.array(list(student_lengths.values()))
    min_len = lengths.min()
    max_len = lengths.max()
    
    print(f"\n📊 Sequence length distribution:")
    print(f"   Min: {min_len}, Max: {max_len}")
    
    # Create 5 bins with 20% of range each
    bin_edges = np.linspace(min_len, max_len + 1, 6)  # 6 edges = 5 bins
    bins = [(bin_edges[i], bin_edges[i+1]) for i in range(5)]
    
    # Assign students to bins
    binned_students = {i: [] for i in range(5)}
    for uid, length in student_lengths.items():
        for bin_idx, (low, high) in enumerate(bins):
            if low <= length < high or (bin_idx == 4 and length == high):  # Last bin includes max
                binned_students[bin_idx].append(uid)
                break
    
    # Select students from each bin
    selected = set()
    print(f"\n🎯 Stratified sampling (target: {num_students_per_bin} students per bin):")
    
    for bin_idx in range(5):
        bin_low, bin_high = bins[bin_idx]
        available = binned_students[bin_idx]
        
        if available:
            # Randomly select up to num_students_per_bin
            num_select = min(num_students_per_bin, len(available))
            selected_from_bin = np.random.choice(available, size=num_select, replace=False)
            selected.update(selected_from_bin)
            
            print(f"   Bin {bin_idx+1} [{int(bin_low):3d}-{int(bin_high):3d}): "
                  f"{len(available):4d} available, {num_select} selected")
        else:
            print(f"   Bin {bin_idx+1} [{int(bin_low):3d}-{int(bin_high):3d}): "
                  f"   0 available, 0 selected")
    
    print(f"\n✓ Total selected: {len(selected)} students")
    return selected


def extract_mastery_states(model, data_loader, device, num_concepts, config, max_students=None):
    """
    Extract mastery states and loss inputs for each skill at each time step.
    
    Args:
        config: Model configuration dict (needed to detect model type)
        max_students: Maximum number of students to process (None for all)
                     Stratified by sequence length if provided
    
    Returns:
        mastery_data: List of dicts with student_id, time_step, question_id, skills, responses, 
                      and loss inputs (L1, L2, L3)
    """
    model.eval()
    mastery_data = []
    
    # Stratified sampling by sequence length if max_students specified
    selected_students = None
    if max_students is not None:
        num_students_per_bin = max(1, max_students // 5)  # Distribute across 5 bins
        selected_students = select_students_by_sequence_length(data_loader, num_students_per_bin)
    
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
            labels = batch['shft_rseqs'].to(device)  # [B, L] - targets for BCE loss
            
            # Get student IDs if available
            if 'uids' in batch:
                student_ids = batch['uids'].cpu().numpy()
                uids = batch['uids']
            else:
                # Generate sequential IDs if not available
                student_ids = np.arange(batch_idx * questions.shape[0], 
                                       (batch_idx + 1) * questions.shape[0])
                uids = None
            
            # Forward pass to get skill vectors (mastery states) and predictions
            # Option 1b: Model uses internal skill difficulty embeddings, no rasch_targets needed
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            
            # Detect model type from outputs
            model_name = config.get('model', 'gainakt4')
            
            if model_name == 'ikt2':
                # iKT2: Per-interaction IRT mastery, not per-skill vectors
                # Check if outputs are available
                if 'mastery_irt' not in outputs:
                    print("\n" + "="*80)
                    print("⚠️  MASTERY STATES NOT AVAILABLE")
                    print("="*80)
                    print("iKT2 model outputs not found.")
                    print("="*80)
                    return []
                
                # Extract iKT2 outputs
                # iKT2 returns per-interaction mastery M_IRT = σ(θ - β), not per-skill vectors
                mastery_irt = outputs['mastery_irt'].cpu().numpy()  # [B, L] - IRT mastery per interaction
                theta_t = outputs['theta_t'].cpu().numpy()  # [B, L] - student ability
                beta_k = outputs['beta_k'].cpu().numpy()  # [B, L] - skill difficulty
                bce_predictions = outputs['bce_predictions'].cpu().numpy()  # [B, L] - Head 1 output
                logits = outputs['logits'].cpu().numpy()  # [B, L] - Head 1 logits
                
                skill_vector = None  # iKT2 doesn't have per-skill vectors
            else:
                # GainAKT4/iKT: Per-skill mastery vectors
                # Check if mastery head is active (skill_vector will be None when λ=1.0)
                if outputs['skill_vector'] is None:
                    print("\n" + "="*80)
                    print("⚠️  MASTERY STATES NOT AVAILABLE")
                    print("="*80)
                    print("The model was trained with λ_bce=1.0 (pure BCE mode).")
                    print("Mastery head (Head 2) was not computed, so skill vectors are unavailable.")
                    print("Mastery states analysis requires λ_bce < 1.0 to activate Head 2.")
                    print("="*80)
                    return []
                
                # Extract all outputs for loss computation
                # L1 (BCE): bce_predictions vs labels
                # L2 (Rasch): skill_vector (Mi) vs rasch_batch (M_rasch)
                # L3 (Constraints): architectural constraints on skill_vector
                
                skill_vector = outputs['skill_vector'].cpu().numpy()  # [B, L, num_concepts] - Head 2 output (Mi)
                bce_predictions = outputs['bce_predictions'].cpu().numpy()  # [B, L] - Head 1 output
                logits = outputs['logits'].cpu().numpy()  # [B, L] - Head 1 logits (pre-sigmoid)
                
                # iKT2-specific outputs not available for other models
                mastery_irt = None
                theta_t = None
                beta_k = None
            
            # Convert to numpy for processing
            questions_np = questions.cpu().numpy()
            responses_np = responses.cpu().numpy()
            mask_np = mask.cpu().numpy()
            labels_np = labels.cpu().numpy()  # Target for L1 BCE loss
            
            # Beta targets for alignment comparison (skill-only targets from embeddings)
            if 'beta_targets' in outputs:
                beta_targets_np = outputs['beta_targets'].cpu().numpy()  # [B, L, num_concepts]
            else:
                beta_targets_np = None
            
            # Process each student in the batch
            batch_size, seq_len = questions_np.shape
            
            for student_idx in range(batch_size):
                student_id = student_ids[student_idx]
                
                # Check if this student is selected (stratified sampling)
                if selected_students is not None and student_id not in selected_students:
                    continue
                
                students_processed += 1
                
                # Process each time step
                for time_step in range(seq_len):
                    # Check if this is a valid step (not padding)
                    if mask_np[student_idx, time_step] == 0:
                        continue
                    
                    question_id = int(questions_np[student_idx, time_step])
                    response = int(responses_np[student_idx, time_step])
                    
                    # L1 (BCE Loss) inputs:
                    # - Prediction: bce_predictions[student_idx, time_step]
                    # - Target: labels_np[student_idx, time_step]
                    # - Logit: logits[student_idx, time_step] (pre-sigmoid)
                    bce_prediction = float(bce_predictions[student_idx, time_step])
                    bce_target = int(labels_np[student_idx, time_step])
                    bce_logit = float(logits[student_idx, time_step])
                    
                    # For single-skill: question_id corresponds to skill_id
                    skill_id = question_id  # Simplified for single-skill case
                    
                    if skill_id >= num_concepts:
                        continue  # Skip invalid skill IDs
                    
                    # Model-specific mastery extraction
                    if model_name == 'ikt2':
                        # iKT2: Per-interaction IRT mastery
                        mi_value = float(mastery_irt[student_idx, time_step])  # M_IRT = σ(θ - β)
                        theta_value = float(theta_t[student_idx, time_step])  # Student ability
                        beta_value = float(beta_k[student_idx, time_step])  # Skill difficulty
                        m_rasch_value = None  # iKT2 doesn't use static Rasch targets
                        mi_prev = None
                        if time_step > 0:
                            for prev_t in range(time_step - 1, -1, -1):
                                if mask_np[student_idx, prev_t] == 1:
                                    mi_prev = float(mastery_irt[student_idx, prev_t])
                                    break
                    else:
                        # GainAKT4/iKT: Per-skill mastery vectors
                        # Get mastery states for all skills at this time step (L2 loss input)
                        mastery_vector = skill_vector[student_idx, time_step, :]
                        
                        # L2 (Alignment) inputs:
                        # - Prediction: skill_vector (Mi) from Head 2
                        # - Target: beta_targets (skill-only targets from embeddings)
                        mi_value = float(mastery_vector[skill_id])  # Head 2 output for this skill
                        theta_value = None  # Not applicable for these models
                        beta_value = None  # Not applicable for these models
                        m_rasch_value = None
                        if beta_targets_np is not None:
                            m_rasch_value = float(beta_targets_np[student_idx, time_step, skill_id])
                        
                        # L3 (Architectural Constraints) - implicit in architecture:
                        # - Positivity: Mi > 0 (enforced by Softplus)
                        # - Monotonicity: Mi[t] ≤ Mi[t+1] (enforced by cummax)
                        # Check if this is the first occurrence of this skill for this student
                        mi_prev = None
                        if time_step > 0:
                            # Get previous mastery value for monotonicity check
                            mi_prev = float(mastery_vector[skill_id])  # This is post-cummax
                            # Find actual previous timestep with valid data
                            for prev_t in range(time_step - 1, -1, -1):
                                if mask_np[student_idx, prev_t] == 1:
                                    mi_prev = float(skill_vector[student_idx, prev_t, skill_id])
                                    break
                    
                    # Store comprehensive data for all loss inputs
                    data_entry = {
                        'student_id': int(student_id),
                        'time_step': int(time_step),
                        'question_id': question_id,
                        'skill_id': skill_id,
                        'response': response,
                        # L1 BCE Loss inputs
                        'bce_prediction': bce_prediction,
                        'bce_target': bce_target,
                        'bce_logit': bce_logit,
                        # L2 Rasch Loss inputs
                        'mi_value': mi_value,  # Head 2 output (skill vector or IRT mastery)
                        'm_rasch_value': m_rasch_value,  # Rasch IRT target (if available)
                        # L3 Constraint checks
                        'mi_prev': mi_prev,  # Previous mastery (for monotonicity)
                        'batch_idx': batch_idx
                    }
                    
                    # Add iKT2-specific fields
                    if model_name == 'ikt2':
                        data_entry['theta'] = theta_value  # Student ability
                        data_entry['beta'] = beta_value  # Skill difficulty
                    
                    mastery_data.append(data_entry)
    
    return mastery_data


def compute_mastery_statistics(mastery_data, num_concepts):
    """Compute aggregate statistics about mastery progression."""
    
    # Organize by skill
    skill_mastery = defaultdict(list)
    skill_progression = defaultdict(lambda: defaultdict(list))
    
    # New data structure has mi_value directly at top level
    for entry in mastery_data:
        skill_id = entry['skill_id']
        mi_value = entry['mi_value']
        
        skill_mastery[skill_id].append(mi_value)
        skill_progression[skill_id][entry['time_step']].append(mi_value)
    
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
    """
    Save mastery states and loss inputs to CSV file.
    
    Columns:
    - Basic: student_id, time_step, question_id, skill_id, response
    - L1 (BCE): bce_logit, bce_prediction, bce_target
    - L2 (Rasch): mi_value, m_rasch_value, rasch_deviation
    - L3 (Constraints): mi_prev, is_positive, is_monotonic
    """
    
    # Prepare rows with all loss inputs
    rows = []
    for entry in mastery_data:
        # Basic info
        student_id = entry['student_id']
        time_step = entry['time_step']
        question_id = entry['question_id']
        skill_id = entry['skill_id']
        response = entry['response']
        
        # L1 (BCE Loss) inputs
        bce_logit = entry['bce_logit']
        bce_prediction = entry['bce_prediction']
        bce_target = entry['bce_target']
        
        # L2 (Rasch Loss) inputs
        mi_value = entry['mi_value']
        m_rasch_value = entry.get('m_rasch_value')
        rasch_deviation = None
        if m_rasch_value is not None:
            rasch_deviation = abs(mi_value - m_rasch_value)
        
        # L3 (Constraint) checks
        mi_prev = entry.get('mi_prev')
        is_positive = mi_value > 0  # Softplus ensures this
        is_monotonic = True  # Default for first occurrence
        if mi_prev is not None:
            is_monotonic = mi_value >= mi_prev  # cummax ensures this
        
        row = {
            'student_id': student_id,
            'time_step': time_step,
            'question_id': question_id,
            'skill_id': skill_id,
            'response': response,
            # L1 inputs
            'bce_logit': f'{bce_logit:.6f}',
            'bce_prediction': f'{bce_prediction:.6f}',
            'bce_target': bce_target,
            # L2 inputs
            'mi_value': f'{mi_value:.6f}',
            'm_rasch_value': f'{m_rasch_value:.6f}' if m_rasch_value is not None else '',
            'rasch_deviation': f'{rasch_deviation:.6f}' if rasch_deviation is not None else '',
            # L3 constraints
            'mi_prev': f'{mi_prev:.6f}' if mi_prev is not None else '',
            'is_positive': is_positive,
            'is_monotonic': is_monotonic
        }
        
        # Add iKT2-specific fields if available
        if 'theta' in entry:
            row['theta'] = f"{entry['theta']:.6f}"
        if 'beta' in entry:
            row['beta'] = f"{entry['beta']:.6f}"
        
        rows.append(row)
    
    # Write to CSV
    if rows:
        # Base fieldnames
        fieldnames = [
            'student_id', 'time_step', 'question_id', 'skill_id', 'response',
            'bce_logit', 'bce_prediction', 'bce_target',
            'mi_value', 'm_rasch_value', 'rasch_deviation',
            'mi_prev', 'is_positive', 'is_monotonic'
        ]
        
        # Add iKT2-specific fields if present
        if rows and 'theta' in rows[0]:
            fieldnames.extend(['theta', 'beta'])
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Extract mastery states from trained model')
    parser.add_argument('--run_dir', type=str, required=True, 
                       help='Experiment directory containing model checkpoint')
    parser.add_argument('--ckpt_name', type=str, default='model_best.pth',
                       help='Checkpoint filename (default: model_best.pth)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                       help='Data split to analyze (default: test)')
    parser.add_argument('--num_students', type=int, default=15,
                       help='Target number of students (default: 15). Stratified by sequence length: 5 bins, ~3 students per bin')
    
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
    # Option 1b: Model uses internal skill difficulty embeddings, no rasch_targets needed
    print(f"\n🔍 Extracting mastery states (max {args.num_students} students)...")
    mastery_data = extract_mastery_states(model, data_loader, device, num_concepts, config,
                                          max_students=args.num_students)    # Check if mastery data is available
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
    csv_path = os.path.join(args.run_dir, f'mastery_{args.split}.csv')
    num_rows = save_mastery_states_csv(mastery_data, csv_path)
    print(f"✓ Saved {num_rows} rows to: {csv_path}")
    
    # Save statistics JSON
    stats_path = os.path.join(args.run_dir, f'mastery_{args.split}.json')
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
