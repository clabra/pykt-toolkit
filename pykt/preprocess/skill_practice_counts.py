"""
Per-Skill Practice Count - Learnable Curve Target

Instead of prospective "attempts to mastery" (unpredictable) or global cumsum (trivial),
we compute per-skill practice counts which are:
1. Retrospective (based on past)
2. Non-trivial (requires tracking individual skills)
3. Meaningful (actual practice intensity per skill)
"""

import numpy as np
import torch
from collections import defaultdict


def calculate_per_skill_practice(concepts, responses):
    """
    Calculate how many times each skill has been practiced up to each position.
    
    Args:
        concepts: [L] - skill IDs
        responses: [L] - responses (0/1), not used but kept for API compatibility
    
    Returns:
        practice_counts: [L] - for each position, how many times that skill was seen before (including current)
    
    Example:
        concepts = [1, 2, 1, 1, 2, 1]
        practice_counts = [1, 1, 2, 3, 2, 4]
        
        Position 0: skill 1, first time → 1
        Position 1: skill 2, first time → 1  
        Position 2: skill 1, second time → 2
        Position 3: skill 1, third time → 3
        Position 4: skill 2, second time → 2
        Position 5: skill 1, fourth time → 4
    """
    seq_len = len(concepts)
    practice_counts = np.zeros(seq_len, dtype=np.int32)
    skill_count = defaultdict(int)
    
    for pos in range(seq_len):
        skill = concepts[pos]
        
        if skill == -1:  # Padding
            practice_counts[pos] = -1
            continue
        
        skill_count[skill] += 1
        practice_counts[pos] = skill_count[skill]
    
    return practice_counts


def add_practice_counts_to_dataset(dori):
    """
    Add per-skill practice counts to dataset.
    
    Args:
        dori: Dataset dictionary with keys ['cseqs', 'rseqs', ...]
    
    Returns:
        dori: Updated with 'practice_counts' field
    """
    if 'practice_counts' in dori:
        print("Warning: practice_counts already exists")
        return dori
    
    num_sequences = len(dori['cseqs'])
    practice_sequences = []
    
    for i in range(num_sequences):
        concepts = dori['cseqs'][i]
        responses = dori['rseqs'][i]
        
        # Get numpy arrays
        if hasattr(concepts, 'cpu'):
            concepts_np = concepts.cpu().numpy()
        else:
            concepts_np = np.array(concepts)
        
        if hasattr(responses, 'cpu'):
            responses_np = responses.cpu().numpy()
        else:
            responses_np = np.array(responses)
        
        # Calculate practice counts
        practice_np = calculate_per_skill_practice(concepts_np, responses_np)
        
        # Convert to tensor if source is tensor
        if hasattr(concepts, 'dtype') and hasattr(concepts, 'device'):
            practice = torch.from_numpy(practice_np).to(dtype=torch.int64, device=concepts.device)
        else:
            practice = practice_np
        
        practice_sequences.append(practice)
    
    dori['practice_counts'] = practice_sequences
    print(f"✓ Added per-skill practice counts for {num_sequences} students")
    
    return dori
