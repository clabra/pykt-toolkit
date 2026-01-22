#!/usr/bin/env python
"""
Generate PCA reference from BKT forward inference for GTransformer v2.0.

This script aggregates BKT student-specific parameters (p_L0, p_T) across skills
and fits a 2-component PCA to derive interpretable student trait coordinates.

CRITICAL: Uses dataset-specific keyid2idx.json mapping to convert original student IDs
to zero-based indices used internally by the model.

Output format (.pt file):
    Tensor of shape [n_uid + 1, 2] where:
    - Index 0: padding/unknown student
    - Indices 1 to n_uid: student trait coordinates [PC1, PC2]
    - PC1: General Proficiency (captures average initial mastery)
    - PC2: Learning Momentum (captures average learning rate)

Author: GTransformer v2.0 Implementation
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle
import json
import os
import argparse


def generate_pca_reference(bkt_forward_file, dataset_dir, output_file):
    """
    Generate PCA reference coordinates from BKT forward inference.
    
    IMPORTANT: Uses keyid2idx.json mapping to convert original student IDs
    to zero-based indices used internally by the model.
    
    Args:
        bkt_forward_file: Path to BKT forward inference results (pickle)
        dataset_dir: Path to dataset directory (e.g., 'data/assist2009')
        output_file: Path to save PCA reference tensor (.pt)
    """
    print(f"\n{'='*60}")
    print(f"GTransformer v2.0: PCA Reference Generation")
    print(f"{'='*60}\n")
    
    # Load dataset-specific student ID mapping
    # This bidirectional mapping converts between original IDs and model indices
    keyid2idx_file = os.path.join(dataset_dir, 'keyid2idx.json')
    if not os.path.exists(keyid2idx_file):
        raise FileNotFoundError(
            f"keyid2idx.json not found at {keyid2idx_file}. "
            f"This file is required to map student IDs to model indices."
        )
    
    with open(keyid2idx_file, 'r') as f:
        keyid2idx = json.load(f)
    
    # Extract student ID mapping (original_id -> model_index)
    # keyid2idx structure: {"questions": {...}, "users": {...}}
    user_mapping = keyid2idx.get('users', {})
    if not user_mapping:
        raise ValueError(
            f"No 'users' mapping found in {keyid2idx_file}. "
            f"Cannot map student IDs to model indices."
        )
    
    print(f"📂 Loaded student ID mapping: {len(user_mapping)} students")
    print(f"   Source: {keyid2idx_file}")
    
    # Load BKT forward inference (p_L0, p_T per student per skill)
    with open(bkt_forward_file, 'rb') as f:
        bkt_data = pickle.load(f)
    
    print(f"📂 Loaded BKT forward inference: {len(bkt_data)} students")
    print(f"   Source: {bkt_forward_file}\n")
    
    # Extract student-level aggregates
    # Aggregate across skills: mean(p_L0), mean(p_T) per student
    student_features = {}
    original_to_index = {}  # Track mapping for validation
    skipped = 0
    
    for original_uid, skill_params in bkt_data.items():
        # Convert original UID to string for lookup
        uid_str = str(original_uid)
        
        # Get model index from mapping
        if uid_str not in user_mapping:
            skipped += 1
            continue
        
        model_idx = user_mapping[uid_str]
        original_to_index[original_uid] = model_idx
        
        # Aggregate BKT parameters across skills
        l0_values = [params['p_L0'] for params in skill_params.values()]
        t_values = [params['p_T'] for params in skill_params.values()]
        
        student_features[model_idx] = [
            np.mean(l0_values),  # Average initial mastery
            np.mean(t_values)    # Average learning rate
        ]
    
    if skipped > 0:
        print(f"⚠️  Warning: Skipped {skipped} students not in keyid2idx mapping")
    
    print(f"✅ Mapped {len(student_features)} students to model indices\n")
    
    # Convert to matrix (sorted by model index for reproducibility)
    model_indices = sorted(student_features.keys())
    X = np.array([student_features[idx] for idx in model_indices])
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"   PC1 (Initial Mastery) range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"   PC2 (Learning Rate) range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]\n")
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"🔬 PCA fitting complete")
    print(f"   PC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"   PC2 explained variance: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.3f}\n")
    
    # Create reference tensor: [n_uid + 1, 2]
    # Index 0: padding (zeros)
    # Indices 1 to n_uid: student PCA coordinates
    max_idx = max(model_indices)
    n_uid = max_idx + 1  # Account for zero-based indexing
    
    pca_reference = torch.zeros(n_uid, 2, dtype=torch.float32)
    
    # Fill in student coordinates (model_idx already in range [0, n_uid-1])
    for i, model_idx in enumerate(model_indices):
        pca_reference[model_idx, 0] = X_pca[i, 0]  # PC1
        pca_reference[model_idx, 1] = X_pca[i, 1]  # PC2
    
    # Save PCA reference tensor
    torch.save(pca_reference, output_file)
    
    print(f"✅ PCA reference generated successfully")
    print(f"   Shape: {pca_reference.shape}")
    print(f"   Students with coordinates: {len(model_indices)}")
    print(f"   Saved to: {output_file}\n")
    
    # Validation: Check index range
    print(f"📊 Index validation:")
    print(f"   Model index range: [0, {max_idx}]")
    print(f"   Expected n_uid parameter: {n_uid}")
    print(f"   Tensor shape: {pca_reference.shape}")
    
    # Additional validation: Check for NaN/Inf
    if torch.isnan(pca_reference).any():
        print(f"   ⚠️  WARNING: NaN values detected in PCA reference")
    if torch.isinf(pca_reference).any():
        print(f"   ⚠️  WARNING: Inf values detected in PCA reference")
    
    print(f"\n{'='*60}\n")
    
    return pca_reference, pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate PCA reference from BKT forward inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PCA reference for assist2009
  python examples/generate_pca_reference.py \\
    --bkt_forward data/assist2009/bkt_forward.pkl \\
    --dataset_dir data/assist2009 \\
    --output data/assist2009/pca_reference.pt
  
  # Generate for assist2015
  python examples/generate_pca_reference.py \\
    --bkt_forward data/assist2015/bkt_forward.pkl \\
    --dataset_dir data/assist2015 \\
    --output data/assist2015/pca_reference.pt

Output:
  - .pt file with shape [n_uid + 1, 2] containing student PCA coordinates
  - PC1: General Proficiency (initial mastery dimension)
  - PC2: Learning Momentum (learning rate dimension)

Requirements:
  - BKT forward inference results (bkt_forward.pkl)
  - Dataset-specific keyid2idx.json mapping
  - scikit-learn (for PCA)
"""
    )
    
    parser.add_argument(
        '--bkt_forward',
        type=str,
        required=True,
        help='Path to BKT forward inference results (pickle file)'
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
    if not os.path.exists(args.bkt_forward):
        print(f"❌ Error: BKT forward file not found: {args.bkt_forward}")
        exit(1)
    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate PCA reference
    try:
        pca_reference, pca_model = generate_pca_reference(
            bkt_forward_file=args.bkt_forward,
            dataset_dir=args.dataset_dir,
            output_file=args.output
        )
        print(f"✅ Success! PCA reference saved to {args.output}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
