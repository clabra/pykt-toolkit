#!/usr/bin/env python
"""
Generate PCA reference from student interaction data using BKT parameters.

This script:
1. Loads BKT skill-level parameters (prior, learns, slip, guess)
2. Runs BKT forward inference on each student's interaction history
3. Aggregates student-specific p_L0 (initial mastery) and p_T (learning rate) across skills
4. Applies PCA to derive 2D student trait coordinates
5. Saves PCA reference tensor for GTransformer v2.0

Author: GTransformer v2.0 Implementation
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle
import json
import os
import argparse
import pandas as pd


def run_bkt_forward_inference(student_data, bkt_params):
    """
    Run BKT forward inference for a single student.
    
    Returns dict: {skill_id: {'p_L0': initial_mastery, 'p_T': avg_learning_rate}}
    """
    skill_states = {}  # Track mastery state per skill
    skill_learning_events = {}  # Track learning events per skill
    
    concepts = [int(c) for c in student_data['concepts'].split(',') if c != '-1']
    responses = [int(r) for r in student_data['responses'].split(',') if r != '-1']
    selectmasks = [int(m) for m in student_data['selectmasks'].split(',') if m != '-1']
    
    # Filter valid interactions
    valid_idx = [i for i in range(len(concepts)) if selectmasks[i] == 1]
    concepts = [concepts[i] for i in valid_idx]
    responses = [responses[i] for i in valid_idx]
    
    # Get global params for fallback
    if 'global' in bkt_params and 'prior' in bkt_params['global']:
        global_prior = bkt_params['global']['prior']
        global_learns = bkt_params['global']['learns']
        global_slip = bkt_params['global'].get('slip', 0.1)
        global_guess = bkt_params['global'].get('guess', 0.2)
    else:
        # Calculate from skill params
        all_priors = [p['prior'] for p in bkt_params['params'].values() if 'prior' in p]
        all_learns = [p['learns'] for p in bkt_params['params'].values() if 'learns' in p]
        global_prior = np.mean(all_priors) if all_priors else 0.5
        global_learns = np.mean(all_learns) if all_learns else 0.1
        global_slip = 0.1
        global_guess = 0.2
    
    def get_param(skill_id, param_name, default):
        """Get parameter for skill, with fallback to global."""
        skill_key = str(skill_id) if str(skill_id) in bkt_params['params'] else skill_id
        if skill_key in bkt_params['params']:
            val = bkt_params['params'][skill_key].get(param_name, default)
            # Handle numpy arrays from pyBKT
            if isinstance(val, (np.ndarray, list)):
                return float(val[0]) if len(val) > 0 else default
            return float(val)
        return default
    
    # Process each interaction
    for skill, response in zip(concepts, responses):
        # Initialize skill on first encounter
        if skill not in skill_states:
            p_prior = get_param(skill, 'prior', global_prior)
            skill_states[skill] = {
                'p_L': p_prior,
                'p_L0': p_prior,  # Store initial state
                'learning_rates': []
            }
            skill_learning_events[skill] = []
        
        # Get BKT parameters
        p_l = skill_states[skill]['p_L']
        p_t = get_param(skill, 'learns', global_learns)
        p_s = get_param(skill, 'slip', global_slip)
        p_g = get_param(skill, 'guess', global_guess)
        
        # BKT Forward Algorithm
        # 1. Predict probability of correct response
        p_correct = p_l * (1 - p_s) + (1 - p_l) * p_g
        p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)
        
        # 2. Bayesian update based on observation
        if response == 1:  # Correct
            p_l_updated = (p_l * (1 - p_s)) / p_correct
        else:  # Incorrect
            p_l_updated = (p_l * p_s) / (1 - p_correct)
        
        # 3. Learning transition
        p_l_new = p_l_updated + (1 - p_l_updated) * p_t
        p_l_new = np.clip(p_l_new, 0.0, 1.0)
        
        # Track learning (change in mastery)
        learning_delta = p_l_new - p_l
        skill_states[skill]['learning_rates'].append(learning_delta)
        
        # Update state
        skill_states[skill]['p_L'] = p_l_new
    
    # Aggregate per-skill parameters
    skill_params_out = {}
    for skill, state in skill_states.items():
        # Average learning rate across interactions
        avg_learning = np.mean(state['learning_rates']) if state['learning_rates'] else 0.0
        skill_params_out[skill] = {
            'p_L0': state['p_L0'],
            'p_T': avg_learning
        }
    
    return skill_params_out


def generate_pca_reference(data_file, bkt_params_file, dataset_dir, output_file):
    """
    Generate PCA reference coordinates from student interaction data.
    
    Args:
        data_file: Path to train_valid_sequences.csv
        bkt_params_file: Path to BKT skill parameters
        dataset_dir: Path to dataset directory
        output_file: Path to save PCA reference tensor (.pt)
    """
    print(f"\n{'='*60}")
    print(f"GTransformer v2.0: PCA Reference Generation")
    print(f"{'='*60}\n")
    
    # Load student ID mapping
    keyid2idx_file = os.path.join(dataset_dir, 'keyid2idx.json')
    if not os.path.exists(keyid2idx_file):
        raise FileNotFoundError(
            f"keyid2idx.json not found at {keyid2idx_file}. "
            f"This file is required to map student IDs to model indices."
        )
    
    with open(keyid2idx_file, 'r') as f:
        keyid2idx = json.load(f)
    
    user_mapping = keyid2idx.get('uid', {})
    if not user_mapping:
        raise ValueError(f"No 'uid' mapping found in {keyid2idx_file}")
    
    print(f"📂 Loaded student ID mapping: {len(user_mapping)} students")
    print(f"   Source: {keyid2idx_file}")
    
    # Load BKT parameters
    with open(bkt_params_file, 'rb') as f:
        bkt_params = pickle.load(f)
    
    print(f"📂 Loaded BKT parameters")
    print(f"   Source: {bkt_params_file}")
    print(f"   Skills: {len(bkt_params.get('params', {}))}\n")
    
    # Load student interaction data
    df = pd.read_csv(data_file)
    print(f"📂 Loaded student data: {len(df)} students")
    print(f"   Source: {data_file}\n")
    
    # Run BKT forward inference for each student
    print(f"🔬 Running BKT forward inference...")
    student_features = {}
    skipped = 0
    
    for idx, row in df.iterrows():
        # The 'uid' column in CSV already contains model indices (0-based)
        model_idx = int(row['uid'])
        
        # Run BKT forward inference
        skill_params = run_bkt_forward_inference(row, bkt_params)
        
        if not skill_params:
            skipped += 1
            continue
        
        # Aggregate across skills: mean(p_L0), mean(p_T)
        l0_values = [params['p_L0'] for params in skill_params.values()]
        t_values = [params['p_T'] for params in skill_params.values()]
        
        student_features[model_idx] = [
            np.mean(l0_values),  # Average initial mastery
            np.mean(t_values)    # Average learning rate
        ]
        
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1}/{len(df)} students...")
    
    if skipped > 0:
        print(f"⚠️  Warning: Skipped {skipped} students with no valid interactions")
    
    print(f"✅ Processed {len(student_features)} students with valid data\n")
    
    # Convert to matrix
    model_indices = sorted(student_features.keys())
    X = np.array([student_features[idx] for idx in model_indices])
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"   Initial Mastery (L0) range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"   Learning Rate (T) range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]\n")
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"🔬 PCA fitting complete")
    print(f"   PC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"   PC2 explained variance: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.3f}\n")
    
    # Create reference tensor: [n_uid, 2]
    max_idx = max(model_indices)
    n_uid = max_idx + 1
    
    pca_reference = torch.zeros(n_uid, 2, dtype=torch.float32)
    
    # Fill in student coordinates
    for i, model_idx in enumerate(model_indices):
        pca_reference[model_idx, 0] = X_pca[i, 0]  # PC1
        pca_reference[model_idx, 1] = X_pca[i, 1]  # PC2
    
    # Save PCA reference tensor
    torch.save(pca_reference, output_file)
    
    print(f"✅ PCA reference generated successfully")
    print(f"   Shape: {pca_reference.shape}")
    print(f"   Students with coordinates: {len(model_indices)}")
    print(f"   Saved to: {output_file}\n")
    
    # Validation
    print(f"📊 Index validation:")
    print(f"   Model index range: [0, {max_idx}]")
    print(f"   Expected n_uid parameter: {n_uid}")
    print(f"   Tensor shape: {pca_reference.shape}")
    
    if torch.isnan(pca_reference).any():
        print(f"   ⚠️  WARNING: NaN values detected in PCA reference")
    if torch.isinf(pca_reference).any():
        print(f"   ⚠️  WARNING: Inf values detected in PCA reference")
    
    print(f"\n{'='*60}\n")
    
    return pca_reference, pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate PCA reference from student interaction data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PCA reference for assist2009
  python examples/generate_pca_reference_from_data.py \\
    --data_file data/assist2009/train_valid_sequences.csv \\
    --bkt_params data/assist2009/bkt_skill_params.pkl \\
    --dataset_dir data/assist2009 \\
    --output data/assist2009/pca_reference.pt
  
  # Generate for assist2015
  python examples/generate_pca_reference_from_data.py \\
    --data_file data/assist2015/train_valid_sequences.csv \\
    --bkt_params data/assist2015/bkt_skill_params.pkl \\
    --dataset_dir data/assist2015 \\
    --output data/assist2015/pca_reference.pt

Output:
  - .pt file with shape [n_uid, 2] containing student PCA coordinates
  - PC1: General Proficiency (initial mastery dimension)
  - PC2: Learning Momentum (learning rate dimension)

Requirements:
  - Student interaction data (train_valid_sequences.csv)
  - BKT skill parameters (bkt_skill_params.pkl)
  - Dataset-specific keyid2idx.json mapping
  - scikit-learn (for PCA)
"""
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to student interaction data (train_valid_sequences.csv)'
    )
    parser.add_argument(
        '--bkt_params',
        type=str,
        required=True,
        help='Path to BKT skill parameters (pkl file)'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to dataset directory containing keyid2idx.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save PCA reference tensor (.pt file)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_file):
        print(f"❌ Error: Data file not found: {args.data_file}")
        exit(1)
    
    if not os.path.exists(args.bkt_params):
        print(f"❌ Error: BKT parameters not found: {args.bkt_params}")
        exit(1)
    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate PCA reference
    try:
        pca_reference, pca_model = generate_pca_reference(
            data_file=args.data_file,
            bkt_params_file=args.bkt_params,
            dataset_dir=args.dataset_dir,
            output_file=args.output
        )
        print(f"✅ Success! PCA reference saved to {args.output}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
