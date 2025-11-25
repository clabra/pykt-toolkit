"""
Attempts-to-Mastery Preprocessing

Calculates how many attempts each student needs to master each skill.
Mastery is defined as achieving N consecutive correct responses on the skill (default: 3).

This module adds 'attempts' field to the dataset, which can be used by models
like GainAKT4 for learning curve prediction.
"""

import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def calculate_attempts_to_mastery(concepts, responses, mastery_threshold=3):
    """
    Calculate attempts-to-mastery for each interaction in a sequence.
    
    Args:
        concepts: Array of concept/skill IDs [L]
        responses: Array of responses (0 or 1) [L]
        mastery_threshold: Number of consecutive correct responses to achieve mastery (default: 3)
    
    Returns:
        attempts: Array of attempts-to-mastery [L]
            - Before mastery: counts down remaining attempts until mastery
            - At mastery: equals 1 (the final attempt that achieves mastery)
            - After mastery: equals 0
            - Never mastered: total attempts on that skill (capped at 10)
    
    Example:
        concepts =  [1, 1, 1, 1, 2, 2, 2, 2]
        responses = [0, 1, 1, 1, 0, 1, 1, 1]
        
        Skill 1: mastered at position 2 (after 3 consecutive correct at positions 1,2,3)
        - Position 0: Not on path to current mastery run → high value
        - Position 1: 2 more after this → 3
        - Position 2: 1 more after this → 2  
        - Position 3: achieves mastery → 1
        
        Skill 2: mastered at position 7
        - Position 4: Not on path → high value
        - Position 5: 2 more after this → 3
        - Position 6: 1 more after this → 2
        - Position 7: achieves mastery → 1
    """
    seq_len = len(concepts)
    attempts = np.zeros(seq_len, dtype=np.int32)
    
    # Track consecutive correct count for each skill
    skill_consecutive_correct = defaultdict(int)
    skill_total_attempts = defaultdict(int)
    
    # First pass: identify when each skill reaches mastery
    mastery_positions = {}  # skill_id -> position where mastery is achieved
    
    for pos in range(seq_len):
        skill = concepts[pos]
        response = responses[pos]
        
        # Skip padding (-1)
        if skill == -1:
            attempts[pos] = -1
            continue
        
        skill_total_attempts[skill] += 1
        
        if response == 1:
            skill_consecutive_correct[skill] += 1
            # Mastery achieved when threshold is reached for the first time
            if skill_consecutive_correct[skill] == mastery_threshold and skill not in mastery_positions:
                mastery_positions[skill] = pos
        else:
            skill_consecutive_correct[skill] = 0
    
    # Second pass: calculate attempts-to-mastery for each position
    skill_attempts_count = defaultdict(int)
    
    for pos in range(seq_len):
        skill = concepts[pos]
        
        if skill == -1:
            attempts[pos] = -1
            continue
        
        skill_attempts_count[skill] += 1
        
        if skill in mastery_positions:
            # Skill will eventually be mastered
            mastery_pos = mastery_positions[skill]
            mastery_attempt_number = sum(1 for p in range(mastery_pos + 1) if concepts[p] == skill)
            current_attempt = skill_attempts_count[skill]
            
            if pos < mastery_pos:
                # Before mastery: count down remaining attempts
                attempts[pos] = mastery_attempt_number - current_attempt + 1
            elif pos == mastery_pos:
                # At mastery position: attempt 1 (achieves mastery)
                attempts[pos] = 1
            else:
                # After mastery: skill is mastered
                attempts[pos] = 0
        else:
            # Skill never reaches mastery: all positions get max value (indicating far from mastery)
            attempts[pos] = 10
    
    return attempts


def add_attempts_to_dataset(dori, mastery_threshold=3):
    """
    Add attempts-to-mastery sequences to a dataset dictionary.
    
    Args:
        dori: Dataset dictionary with keys ['cseqs', 'rseqs', 'masks', ...]
        mastery_threshold: Number of consecutive correct responses for mastery
    
    Returns:
        dori: Updated dataset dictionary with 'atseqs' (attempts-to-mastery sequences)
    """
    if 'atseqs' in dori:
        print("Warning: 'atseqs' already exists in dataset, skipping preprocessing")
        return dori
    
    num_sequences = len(dori['cseqs'])
    attempts_sequences = []
    
    for i in range(num_sequences):
        concepts = dori['cseqs'][i]
        responses = dori['rseqs'][i]
        
        # Get numpy arrays from the source data
        if hasattr(concepts, 'cpu'):  # torch.Tensor
            concepts_np = concepts.cpu().numpy()
        elif not isinstance(concepts, np.ndarray):
            concepts_np = np.array(concepts)
        else:
            concepts_np = concepts
            
        if hasattr(responses, 'cpu'):  # torch.Tensor
            responses_np = responses.cpu().numpy()
        elif not isinstance(responses, np.ndarray):
            responses_np = np.array(responses)
        else:
            responses_np = responses
        
        # Calculate attempts (returns numpy array)
        attempts_np = calculate_attempts_to_mastery(concepts_np, responses_np, mastery_threshold)
        
        # Convert to same type as source data (tensor if source is tensor, array if source is array)
        if hasattr(concepts, 'dtype') and hasattr(concepts, 'device'):  # torch.Tensor
            attempts = torch.from_numpy(attempts_np).to(dtype=torch.int64, device=concepts.device)
        else:
            attempts = attempts_np
        
        attempts_sequences.append(attempts)
    
    dori['atseqs'] = attempts_sequences
    print(f"✓ Added attempts-to-mastery sequences for {num_sequences} students")
    
    return dori


def preprocess_dataset_file(input_path, output_path=None, mastery_threshold=3):
    """
    Preprocess a pickle file to add attempts-to-mastery.
    
    Args:
        input_path: Path to input .pkl file
        output_path: Path to output .pkl file (default: input_path with '_with_attempts' suffix)
        mastery_threshold: Number of consecutive correct responses for mastery
    """
    import pickle
    
    if output_path is None:
        output_path = input_path.replace('.pkl', '_with_attempts.pkl')
    
    # Load data
    print(f"Loading dataset from {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both single dict and list of dicts
    if isinstance(data, list):
        print(f"Processing {len(data)} dataset splits")
        for i, dori in enumerate(data):
            if isinstance(dori, dict):
                data[i] = add_attempts_to_dataset(dori, mastery_threshold)
    elif isinstance(data, dict):
        data = add_attempts_to_dataset(data, mastery_threshold)
    
    # Save
    print(f"Saving preprocessed dataset to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("✓ Preprocessing complete")
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python attempts_to_mastery.py <input_pkl_file> [output_pkl_file] [mastery_threshold]")
        print("Example: python attempts_to_mastery.py data/assist2015/train_valid_sequences.csv_0.pkl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    mastery_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    preprocess_dataset_file(input_path, output_path, mastery_threshold)
