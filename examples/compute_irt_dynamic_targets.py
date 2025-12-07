"""
Compute Dynamic IRT Targets with Time-Varying Student Abilities

This script generates IRT reference targets where:
- θ_IRT[student, t]: Time-varying ability (Bayesian MLE at each timestep)
- β_IRT[skill]: Static skill difficulty (pre-calibrated)
- M_ref[student, t]: Dynamic predictions σ(θ_t - β)

Approach:
- Option 2: Bayesian IRT - At each timestep t, estimate θ from observations [0:t]
- Option 1: Keep β Static - Skill difficulties are intrinsic properties

Usage:
    python examples/compute_irt_dynamic_targets.py \\
        --dataset assist2015 \\
        --fold 0 \\
        --rasch_targets data/assist2015/rasch_test_iter300.pkl \\
        --output data/assist2015/irt_dynamic_targets_fold0.pkl
"""

import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pykt.datasets import init_dataset4train


def optimize_theta_mle(responses, questions, beta_irt, n_iter=50, learning_rate=1.0):
    """
    Bayesian MLE estimation for θ given responses and fixed β.
    
    Uses Newton-Raphson optimization with safeguards to find θ that maximizes 
    likelihood of observed responses given skill difficulties.
    
    Args:
        responses: np.array [t+1] - responses up to timestep t
        questions: np.array [t+1] - question IDs up to timestep t
        beta_irt: np.array [num_skills] - static skill difficulties
        n_iter: Number of optimization iterations
        learning_rate: Step size for updates
    
    Returns:
        theta_mle: float - Maximum likelihood estimate of ability
    """
    # Initialize θ near mean of difficulties (better starting point)
    beta_seq = beta_irt[questions]
    theta = np.mean(beta_seq)  # Start near average difficulty
    
    # Bounds to prevent divergence
    theta_min, theta_max = -15.0, 15.0
    
    for iteration in range(n_iter):
        # Compute predicted probabilities: P(correct) = σ(θ - β)
        # Clip logits to prevent overflow
        logits = np.clip(theta - beta_seq, -20, 20)
        probs = 1.0 / (1.0 + np.exp(-logits))
        
        # Gradient of log-likelihood: ∂ℓ/∂θ = Σ(response - prob)
        grad = np.sum(responses - probs)
        
        # Hessian (second derivative): ∂²ℓ/∂θ² = -Σ(prob * (1 - prob))
        hessian = -np.sum(probs * (1.0 - probs))
        
        # Newton-Raphson update with safeguards
        if abs(hessian) > 1e-8:
            # Standard Newton-Raphson step
            delta = learning_rate * (grad / hessian)
            # Limit step size to prevent divergence
            delta = np.clip(delta, -2.0, 2.0)
            theta_new = theta - delta
        else:
            # Fallback to gradient ascent if hessian is too small
            delta = learning_rate * grad * 0.01
            delta = np.clip(delta, -0.5, 0.5)
            theta_new = theta + delta
        
        # Apply bounds
        theta_new = np.clip(theta_new, theta_min, theta_max)
        
        # Convergence check
        if abs(theta_new - theta) < 1e-5:
            theta = theta_new
            break
        
        theta = theta_new
    
    return theta


def compute_theta_trajectory_bayesian(responses, questions, beta_irt):
    """
    Compute time-varying θ using Bayesian IRT calibration.
    
    At each timestep t, estimate θ from all observations [0:t] using MLE.
    This gives a trajectory showing how our estimate of ability evolves
    as we gather more evidence about the student.
    
    Args:
        responses: np.array [L] - sequence of 0/1 responses
        questions: np.array [L] - sequence of skill IDs
        beta_irt: np.array [num_skills] - pre-calibrated difficulties
    
    Returns:
        theta_trajectory: np.array [L] - θ estimate at each timestep
    """
    L = len(responses)
    theta_trajectory = np.zeros(L)
    
    for t in range(L):
        # Data up to and including this timestep
        responses_so_far = responses[:t+1]
        questions_so_far = questions[:t+1]
        
        # Optimize θ to maximize likelihood of observed responses
        theta_t = optimize_theta_mle(
            responses_so_far,
            questions_so_far,
            beta_irt,
            n_iter=20,
            learning_rate=0.5
        )
        
        theta_trajectory[t] = theta_t
    
    return theta_trajectory


def compute_dynamic_irt_targets(train_loader, valid_loader, beta_irt, num_skills):
    """
    Compute dynamic IRT targets: time-varying θ trajectories with static β.
    
    Args:
        train_loader, valid_loader: DataLoaders for training and validation sets
        beta_irt: np.array [num_skills] - static skill difficulties
        num_skills: Total number of skills
    
    Returns:
        dict with:
            - theta_trajectories: {uid: np.array[L]} - dynamic abilities
            - beta_irt: np.array[num_skills] - static difficulties
            - m_ref_trajectories: {uid: np.array[L]} - dynamic predictions
            - metadata: Statistics
    """
    print("\n" + "="*80)
    print("COMPUTING DYNAMIC IRT TARGETS")
    print("="*80)
    print(f"Approach: Bayesian MLE (time-varying θ) + Static β")
    print(f"Number of skills: {num_skills}")
    
    theta_trajectories = {}
    m_ref_trajectories = {}
    
    all_theta_values = []
    total_sequences = 0
    
    # Process all data splits (only train and valid)
    for loader_name, loader in [('train', train_loader), ('valid', valid_loader)]:
        print(f"\n{loader_name.upper()} SET:")
        print(f"  Processing sequences...")
        
        batch_count = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"    Batch {batch_idx}/{len(loader)}")
            
            questions = batch['cseqs'].numpy()  # [B, L]
            responses = batch['rseqs'].numpy()  # [B, L]
            masks = batch['masks'].numpy()      # [B, L]
            batch_size, seq_len = questions.shape
            
            # Get student IDs if available
            if 'uids' in batch:
                uids = batch['uids'].numpy()
            else:
                uids = np.arange(batch_count, batch_count + batch_size)
                batch_count += batch_size
            
            for i in range(batch_size):
                uid = int(uids[i])
                
                # Skip if already processed
                if uid in theta_trajectories:
                    continue
                
                # Get valid interactions
                valid_mask = masks[i] == 1
                if not valid_mask.any():
                    continue
                
                q_valid = questions[i][valid_mask]
                r_valid = responses[i][valid_mask]
                L = len(q_valid)
                
                # Compute dynamic theta trajectory
                theta_traj = compute_theta_trajectory_bayesian(
                    r_valid,
                    q_valid,
                    beta_irt
                )
                
                # Compute reference predictions: M_ref[t] = σ(θ[t] - β[q[t]])
                beta_seq = beta_irt[q_valid]
                m_ref_traj = 1.0 / (1.0 + np.exp(-(theta_traj - beta_seq)))
                
                # Store
                theta_trajectories[uid] = theta_traj
                m_ref_trajectories[uid] = m_ref_traj
                all_theta_values.extend(theta_traj.tolist())
                total_sequences += 1
        
        print(f"  Completed: {total_sequences} total sequences processed")
    
    # Compute statistics
    all_theta_values = np.array(all_theta_values)
    theta_mean = float(np.mean(all_theta_values))
    theta_std = float(np.std(all_theta_values))
    theta_min = float(np.min(all_theta_values))
    theta_max = float(np.max(all_theta_values))
    
    beta_mean = float(np.mean(beta_irt))
    beta_std = float(np.std(beta_irt))
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"\nθ (Dynamic Student Abilities):")
    print(f"  Total timestep estimates: {len(all_theta_values)}")
    print(f"  Mean:  {theta_mean:.4f}")
    print(f"  Std:   {theta_std:.4f}")
    print(f"  Range: [{theta_min:.4f}, {theta_max:.4f}]")
    
    print(f"\nβ (Static Skill Difficulties):")
    print(f"  Number of skills: {num_skills}")
    print(f"  Mean:  {beta_mean:.4f}")
    print(f"  Std:   {beta_std:.4f}")
    print(f"  Range: [{np.min(beta_irt):.4f}, {np.max(beta_irt):.4f}]")
    
    print(f"\nθ/β Scale Ratio:")
    print(f"  θ_std / β_std = {theta_std / beta_std:.4f}")
    
    metadata = {
        'num_skills': num_skills,
        'num_sequences': total_sequences,
        'num_timesteps': len(all_theta_values),
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'theta_min': theta_min,
        'theta_max': theta_max,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'approach': 'bayesian_mle_dynamic_theta_static_beta'
    }
    
    return {
        'theta_trajectories': theta_trajectories,   # {uid: np.array[L]}
        'beta_irt': beta_irt,                       # np.array[num_skills]
        'm_ref_trajectories': m_ref_trajectories,   # {uid: np.array[L]}
        'metadata': metadata
    }


def main():
    parser = argparse.ArgumentParser(description='Compute dynamic IRT targets')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--rasch_targets', type=str, required=True,
                        help='Path to Rasch targets file with pre-calibrated β')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for dynamic IRT targets')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DYNAMIC IRT TARGET GENERATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Rasch targets: {args.rasch_targets}")
    print(f"Output: {args.output}")
    
    # Load pre-calibrated skill difficulties (static β)
    print("\nStep 1: Loading pre-calibrated skill difficulties...")
    with open(args.rasch_targets, 'rb') as f:
        rasch_data = pickle.load(f)
    
    skill_difficulties = rasch_data.get('skill_difficulties', rasch_data.get('rasch_b', {}))
    num_skills = len(skill_difficulties)
    
    # Convert to numpy array
    beta_irt = np.array([skill_difficulties.get(k, 0.0) for k in range(num_skills)])
    
    print(f"✓ Loaded {num_skills} skill difficulties")
    print(f"  β range: [{beta_irt.min():.4f}, {beta_irt.max():.4f}]")
    print(f"  β mean: {beta_irt.mean():.4f}, std: {beta_irt.std():.4f}")
    
    # Initialize datasets
    print("\nStep 2: Loading datasets...")
    
    # Load data config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_config_path = os.path.join(project_root, 'configs', 'data_config.json')
    with open(data_config_path, 'r') as f:
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
        batch_size=args.batch_size
    )
    
    # Get number of skills from data config
    num_skills_dataset = data_config[args.dataset]['num_c']
    
    assert num_skills == num_skills_dataset, \
        f"Skill count mismatch: Rasch has {num_skills}, dataset has {num_skills_dataset}"
    
    print(f"✓ Loaded dataloaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")
    
    # Compute dynamic IRT targets
    print("\nStep 3: Computing dynamic θ trajectories (Bayesian MLE)...")
    targets = compute_dynamic_irt_targets(
        train_loader,
        valid_loader,
        beta_irt,
        num_skills
    )
    
    # Convert to proper format for saving
    print("\nStep 4: Preparing output...")
    output_data = {
        'skill_difficulties': {k: float(beta_irt[k]) for k in range(num_skills)},
        'theta_trajectories': {uid: traj.tolist() for uid, traj in targets['theta_trajectories'].items()},
        'm_ref_trajectories': {uid: traj.tolist() for uid, traj in targets['m_ref_trajectories'].items()},
        'metadata': targets['metadata']
    }
    
    # Save
    print(f"\nStep 5: Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("\n" + "="*80)
    print("✓ DYNAMIC IRT TARGETS GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"Output file: {args.output}")
    print(f"Sequences: {targets['metadata']['num_sequences']}")
    print(f"Total timesteps: {targets['metadata']['num_timesteps']}")
    print(f"Approach: Bayesian MLE with time-varying θ and static β")


if __name__ == '__main__':
    main()
