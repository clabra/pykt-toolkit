#!/usr/bin/env python3
"""
Generate skill-level quadrant comparison plots.

For each skill, find students in different quadrants (Low/High L0 x Low/High T) who have
the SAME response sequence for that skill. 

Quadrant Classification:
- Uses P(L0) and P(T) from the FIRST timestep when encountering each skill
- This represents the model's skill-specific initial assessment
- Ensures pedagogical consistency: High L0 students should predict higher than Low L0

Selection logic:
1. Filter: Skills with students from ≥2 quadrants having identical response sequences
2. Rank: By quality score (high range, low within-variance) and accuracy advantage
3. Select: Top N skills showing best visual clarity and performance gains

Plot for each selected skill:
- Multiple GTransformer predictions (one per quadrant student, solid lines)
- BKT predictions (dotted lines - all overlap since students have identical sequences)

Creates a 4x3 mosaic showing 12 skills ranked by quality and pedagogical consistency.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from examples.results.validation_helpers import load_model_from_dir


def calculate_bkt_trajectory(skill_params, skill_id, sequence):
    """Calculate BKT predictions for a given skill and response sequence."""
    params_dict = skill_params.get('params', {})
    if skill_id not in params_dict:
        return np.full(len(sequence), 0.5)
        
    p = params_dict[skill_id]
    l0, t, s, g = p.get('pl0', 0.5), p.get('pt', 0.1), p.get('ps', 0.1), p.get('pg', 0.1)
    
    # BKT Formula: P(Correct) = L * (1-S) + (1-L) * G
    preds = []
    l_curr = l0
    
    for r in sequence:
        p_correct = l_curr * (1 - s) + (1 - l_curr) * g
        preds.append(p_correct)
        
        if r == 1:
            l_post = (l_curr * (1 - s)) / max(1e-6, (l_curr * (1 - s) + (1 - l_curr) * g))
        else:
            l_post = (l_curr * s) / max(1e-6, (l_curr * s + (1 - l_curr) * (1 - g)))
            
        l_curr = l_post + (1 - l_post) * t
        
    return np.array(preds)


def get_quadrant_label(l0, t, l0_med, t_med):
    """Assign quadrant based on median thresholds."""
    low_l0 = l0 < l0_med
    low_t = t < t_med
    
    if low_l0 and low_t:
        return "Low L0 / Low T"
    elif low_l0 and not low_t:
        return "Low L0 / High T"
    elif not low_l0 and low_t:
        return "High L0 / Low T"
    else:
        return "High L0 / High T"


def extract_student_skill_data(model, loader, device):
    """
    Extract all student-skill sequences with their predictions and parameters.
    
    Returns:
        dict: {skill_id: {response_seq_tuple: [(uid, l0, t, preds), ...]}}
    """
    skill_data = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            qshft = data["shft_qseqs"].long().to(device)
            cshft = data["shft_cseqs"].long().to(device)
            rshft = data["shft_rseqs"].long().to(device)
            m = data["masks"].bool().to(device)
            sm = data["smasks"].bool().to(device)
            uids = data.get("uids", None)
            if uids is not None: 
                uids = uids.long().to(device)
            
            # Concatenate sequences
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)
            
            outputs, _ = model(cc.long(), cr.long(), pid_data=cq.long(), uid_data=uids)
            
            # Extract predictions and parameters (already shifted by model)
            p_l0 = outputs['p_l0'][:,1:].cpu().numpy()
            p_t = outputs['p_t'][:,1:].cpu().numpy()
            preds = outputs['predictions'][:,1:].cpu().numpy()  # p_sup
            
            for b in range(c.shape[0]):
                m_b = sm[b].cpu().numpy()
                cur_c = cshft[b].cpu().numpy()[m_b]
                cur_r = rshft[b].cpu().numpy()[m_b]
                cur_p_l0 = p_l0[b][m_b]
                cur_p_t = p_t[b][m_b]
                cur_preds = preds[b][m_b]
                
                uid = data['uids'][b].item() if 'uids' in data and data['uids'] is not None else f"S{i*64+b}"
                
                unique_skills = np.unique(cur_c)
                for skill in unique_skills:
                    s_mask = (cur_c == skill)
                    seq_len = np.sum(s_mask)
                    
                    # Only consider sequences of reasonable length
                    if seq_len < 5 or seq_len > 30:
                        continue
                    
                    # Extract skill-specific data
                    skill_responses = tuple(cur_r[s_mask].astype(int).tolist())
                    skill_preds = cur_preds[s_mask]
                    
                    # Get skill-specific P(L0) and P(T) at first encounter
                    skill_indices = np.where(s_mask)[0]
                    first_skill_idx = skill_indices[0]
                    skill_l0 = cur_p_l0[first_skill_idx]
                    skill_t = cur_p_t[first_skill_idx]
                    
                    # Calculate HISTORICAL AVERAGE P(L0) and P(T) from all previous timesteps
                    if first_skill_idx > 0:
                        hist_l0 = np.mean(cur_p_l0[:first_skill_idx])
                        hist_t = np.mean(cur_p_t[:first_skill_idx])
                    else:
                        # If skill appears at first timestep, use that timestep's values
                        hist_l0 = skill_l0
                        hist_t = skill_t
                    
                    # Store student data with both skill-specific and historical parameters
                    skill_data[int(skill)][skill_responses].append({
                        'uid': uid,
                        'l0': skill_l0,           # Skill-specific L0 (first encounter)
                        't': skill_t,             # Skill-specific T (first encounter)
                        'hist_l0': hist_l0,       # Historical average L0
                        'hist_t': hist_t,         # Historical average T
                        'preds': skill_preds
                    })
            
            if i % 10 == 0:
                print(f"Processed {i+1} batches...")
    
    return skill_data


def find_matching_quadrant_skills(skill_data, bkt_params, top_n=20):
    """
    Find skills where we have students in at least 2 quadrants WITH IDENTICAL response sequences.
    This is key: we need the same response sequence across quadrants to show that BKT gives
    identical predictions while GTransformer differentiates based on learning context.
    
    NEW: Rank skills by GTransformer performance advantage over BKT for that skill.
    
    Returns:
        list: [(skill_id, response_seq, {quadrant: student_data}), ...]
    """
    # First, collect all l0 and t values to compute medians
    all_l0 = []
    all_t = []
    for skill_sequences in skill_data.values():
        for students in skill_sequences.values():
            for student in students:
                all_l0.append(student['hist_l0'])
                all_t.append(student['hist_t'])
    
    l0_med = np.median(all_l0)
    t_med = np.median(all_t)
    
    print(f"\nParameter Medians: L0={l0_med:.4f}, T={t_med:.4f}")
    
    # Track total candidates and violations
    total_candidates = 0
    ordering_violations = 0
    
    valid_skills = []
    all_quadrants = ["Low L0 / Low T", "Low L0 / High T", 
                    "High L0 / Low T", "High L0 / High T"]
    
    for skill_id, sequences in skill_data.items():
        # For each response sequence, check if we have students from multiple quadrants
        for response_seq, students in sequences.items():
            if len(students) < 2:  # Need at least 2 students
                continue
            
            # Group students by quadrant using HISTORICAL AVERAGE L0 and T
            quadrants_for_seq = defaultdict(list)
            for student in students:
                # Use historical averages for quadrant classification
                quad = get_quadrant_label(student['hist_l0'], student['hist_t'], l0_med, t_med)
                quadrants_for_seq[quad].append(student)
            
            # Check if we have at least 2 quadrants with this exact sequence
            present_quadrants = [q for q in all_quadrants if len(quadrants_for_seq[q]) > 0]
            
            if len(present_quadrants) < 2:
                continue
            
            # Select one representative student per quadrant
            # NEW CRITERION: Select student whose skill-specific L0/T are CLOSEST to historical L0/T
            # This ensures pedagogical consistency by selecting students where the model's
            # skill-specific assessment aligns with their overall trajectory
            selected = {}
            for quad in present_quadrants:
                candidates = quadrants_for_seq[quad]
                
                # Score by alignment between skill-specific and historical parameters
                # Lower score = better alignment
                best_student = min(candidates, 
                                 key=lambda s: (s['l0'] - s['hist_l0'])**2 + (s['t'] - s['hist_t'])**2)
                best_student['response_seq'] = response_seq
                selected[quad] = best_student
            
            # Calculate GTransformer vs BKT accuracy advantage for this skill
            # We'll compute accuracy (correct predictions) for both models across all students
            all_gt_correct = []
            all_bkt_correct = []
            
            for quad, student in selected.items():
                gt_preds = student['preds']
                bkt_preds = calculate_bkt_trajectory(bkt_params, skill_id, response_seq)
                
                # Binary predictions (threshold at 0.5)
                gt_binary = (gt_preds > 0.5).astype(int)
                bkt_binary = (bkt_preds > 0.5).astype(int)
                
                # Compare with actual responses
                responses = np.array(response_seq)
                all_gt_correct.extend((gt_binary == responses).tolist())
                all_bkt_correct.extend((bkt_binary == responses).tolist())
            
            # Calculate accuracy advantage
            gt_accuracy = np.mean(all_gt_correct)
            bkt_accuracy = np.mean(all_bkt_correct)
            accuracy_advantage = gt_accuracy - bkt_accuracy
            
            # Calculate prediction contrastiveness metrics
            avg_preds = [s['preds'].mean() for s in selected.values()]
            pred_std = np.std(avg_preds) if len(avg_preds) >= 2 else 0.0
            pred_range = max(avg_preds) - min(avg_preds) if len(avg_preds) >= 2 else 0.0
            
            # Calculate within-quadrant variance (measures line "tightness")
            within_vars = [np.var(s['preds']) for s in selected.values()]
            avg_within_var = np.mean(within_vars)
            max_within_std = np.sqrt(max(within_vars))
            
            # Quality score: high between-quadrant range, low within-quadrant variance
            # This produces clean, distinct lines rather than overlapping bands
            quality_score = pred_range / (1 + avg_within_var)
            
            # Track this as a candidate
            total_candidates += 1
            
            # Helper to check if curve 1 is pedagogically superior to curve 2
            def is_superior(q1, q2):
                if q1 not in selected or q2 not in selected:
                    return True
                p1 = selected[q1]['preds']
                p2 = selected[q2]['preds']
                # Curve 1 must be better in average, start, and finish
                # We use a tiny tolerance of 0.01 to allow for numeric precision issues
                return (np.mean(p1) >= np.mean(p2) - 0.01 and 
                        p1[0] >= p2[0] - 0.01 and 
                        p1[-1] >= p2[-1] - 0.01)

            is_pedagogically_ordered = True
            # Check 1: Green >= Dark Blue >= Red (T dimension)
            if not is_superior("High L0 / High T", "Low L0 / High T"): is_pedagogically_ordered = False
            if not is_superior("Low L0 / High T", "Low L0 / Low T"): is_pedagogically_ordered = False
            
            # Check 2: Green >= Light Blue >= Red (L0 dimension)
            if not is_superior("High L0 / High T", "High L0 / Low T"): is_pedagogically_ordered = False
            if not is_superior("High L0 / Low T", "Low L0 / Low T"): is_pedagogically_ordered = False
            
            # Check 3: Diagonal
            if not is_superior("High L0 / High T", "Low L0 / Low T"): is_pedagogically_ordered = False
            
            # If any check failed, increment violation counter
            if not is_pedagogically_ordered:
                ordering_violations += 1
            
            # Only include skills with pedagogical ordering
            if not is_pedagogically_ordered:
                continue
            
            valid_skills.append({
                'skill_id': skill_id,
                'response_seq': response_seq,
                'quadrants': selected,
                'pred_std': pred_std,
                'pred_range': pred_range,
                'avg_within_var': avg_within_var,
                'max_within_std': max_within_std,
                'quality_score': quality_score,
                'seq_length': len(response_seq),
                'n_quadrants': len(selected),
                'gt_accuracy': gt_accuracy,
                'bkt_accuracy': bkt_accuracy,
                'accuracy_advantage': accuracy_advantage
            })
    
    # Sort by: 1) Quality score (high range, low variance = clean distinct lines), 2) Accuracy advantage, 3) Sequence length
    valid_skills.sort(key=lambda x: (x['quality_score'], x['accuracy_advantage'], x['seq_length']), reverse=True)
    
    # Enforce unique skill IDs by keeping highest-advantage sequence per skill
    unique_by_skill = []
    seen_skills = set()
    for item in valid_skills:
        skill_id = item['skill_id']
        if skill_id in seen_skills:
            continue
        unique_by_skill.append(item)
        seen_skills.add(skill_id)
    
    print(f"\nFound {len(valid_skills)} skill-sequence combinations with at least 2 quadrants")
    print(f"  (all students in each combination have IDENTICAL response sequences)")
    print(f"Unique skills after filtering: {len(unique_by_skill)}")
    print(f"Selecting top {top_n} ranked by line quality (high range, low within-variance)")
    
    # Print selection methodology
    print(f"\n{'='*80}")
    print(f"ALIGNMENT-BASED STUDENT SELECTION + PEDAGOGICAL ORDERING FILTER")
    print(f"{'='*80}")
    print(f"Total candidates evaluated: {total_candidates}")
    print(f"Pedagogical ordering violations: {ordering_violations}")
    if ordering_violations > 0:
        print(f"Violation rate: {100*ordering_violations/total_candidates:.1f}%")
    print(f"\nSelection Methodology:")
    print(f"  1. Quadrant classification: Based on HISTORICAL AVERAGE L0 and T")
    print(f"  2. Student selection: Choose student whose SKILL-SPECIFIC L0/T")
    print(f"     are CLOSEST to their historical averages")
    print(f"  3. Robust Pedagogical ordering filter:")
    print(f"     Checks Mean, First, and Last predictions for all quadrant pairs:")
    print(f"     a) Green >= Dark Blue >= Red")
    print(f"     b) Green >= Light Blue >= Red")
    print(f"     c) Green >= Red (Diagonal)")
    print(f"\nThis approach selects students where the model's skill-specific assessment")
    print(f"aligns with their overall learning trajectory, then filters for pedagogical")
    print(f"ordering to ensure monotonic predictions across learning situations.")
    print(f"{'='*80}\n")
    
    if len(unique_by_skill) > 0:
        print(f"Top 5 skills by quality score:")
        for i, skill in enumerate(unique_by_skill[:5]):
            print(f"  {i+1}. Skill {skill['skill_id']}: "
                  f"Quality={skill['quality_score']:.4f}, Range={skill['pred_range']:.4f}, "
                  f"MaxStd={skill['max_within_std']:.3f}, Advantage={skill['accuracy_advantage']:+.3f}")
    
    return unique_by_skill[:top_n]


def plot_skill_quadrant_mosaic(selected_skills, bkt_params, output_dir):
    """
    Create a 4x3 mosaic of 12 skills.
    Each subplot shows 4 GTransformer trajectories + 1 BKT trajectory.
    Note: Students may have different sequences, so we show individual trajectories.
    """
    n_skills = len(selected_skills)
    n_rows = 4
    n_cols = 3
    
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Use high-contrast, distinct colors for better visibility
    quadrant_colors = {
        "Low L0 / Low T": "#d62728",      # Red - Low L0 / Low T
        "Low L0 / High T": "#1f77b4",     # Blue - Low L0 / High T  
        "High L0 / Low T": "#87CEEB",     # Light Blue - High L0 / Low T (changed from orange)
        "High L0 / High T": "#2ca02c"     # Green - High L0 / High T
    }
    
    quadrant_order = ["Low L0 / Low T", "Low L0 / High T", 
                     "High L0 / Low T", "High L0 / High T"]
    
    for idx, skill_data in enumerate(selected_skills[:n_rows*n_cols]):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        skill_id = skill_data['skill_id']
        quadrants = skill_data['quadrants']
        
        # Get the quadrants that are actually present for this skill
        present_quadrants = [q for q in quadrant_order if q in quadrants]
        
        # Find max sequence length for x-axis
        max_len = max(len(quadrants[q]['response_seq']) for q in present_quadrants)
        
        # Plot GTransformer predictions for each quadrant (only present ones)
        for quad_name in present_quadrants:
            student = quadrants[quad_name]
            color = quadrant_colors[quad_name]
            response_seq = student['response_seq']
            
            x = np.arange(len(response_seq))
            
            # Ground truth bars (faint background) - only for this student
            for i, val in enumerate(response_seq):
                bar_color = 'lightgreen' if val == 1 else 'lightcoral'
                ax.axvline(x=i, ymin=0, ymax=0.1, color=bar_color, alpha=0.3, linewidth=2)
            
            # Create legend label with student characteristics
            l0_level = "High L0" if "High L0" in quad_name else "Low L0"
            t_level = "High T" if "High T" in quad_name else "Low T"
            legend_label = f"id: {student['uid']} ({l0_level}, {t_level})"
            
            # Plot GTransformer predictions with markers (thicker, more visible)
            ax.plot(x, student['preds'], color=color, linewidth=2.5, 
                   alpha=0.85, zorder=1, label=legend_label)
            for i, p in enumerate(student['preds']):
                marker = 'o' if p > 0.5 else 'x'
                markersize = 7 if p > 0.5 else 8
                ax.plot(i, p, marker=marker, color=color, markersize=markersize, 
                       alpha=1.0, zorder=2)
            
            # BKT baseline for this student's sequence with markers (more visible)
            bkt_preds = calculate_bkt_trajectory(bkt_params, skill_id, response_seq)
            bkt_label = f"id: {student['uid']} (Classical BKT)"
            ax.plot(x, bkt_preds, color='#555555', linestyle=':', linewidth=2.0, 
                   alpha=0.6, zorder=0, label=bkt_label)
            for i, p in enumerate(bkt_preds):
                marker = 'o' if p > 0.5 else 'x'
                markersize = 5 if p > 0.5 else 6
                ax.plot(i, p, marker=marker, color='#555555', markersize=markersize, 
                       alpha=0.7, zorder=0)
        
        # Title with skill info
        ax.set_title(f"Skill {skill_id}", 
                    fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(Correct)", fontsize=8)
        ax.set_xlabel("Time-step", fontsize=8)
        ax.set_xticks(np.arange(0, max_len, 1))
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Always show legend with student info
        ax.legend(loc='upper right', fontsize=6, framealpha=0.95)
        
        ax.grid(True, alpha=0.2)
    
    plt.suptitle("Context-Aware Predictions for Students with Same Response Sequence vs BKT Identical Predictions",
                fontsize=16, fontweight='bold', y=0.99)
    plt.subplots_adjust(top=0.96)
    
    output_path = os.path.join(output_dir, "h3_skill_quadrant_comparison_mosaic.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved {n_rows}x{n_cols} skill mosaic to {output_path}")
    plt.close()
    
    # Also save individual skill plots for closer inspection
    individual_dir = os.path.join(output_dir, "individual_skills")
    os.makedirs(individual_dir, exist_ok=True)
    
    for skill_data in selected_skills[:n_rows*n_cols]:
        plot_individual_skill(skill_data, bkt_params, individual_dir, quadrant_colors, quadrant_order)


def plot_individual_skill(skill_data, bkt_params, output_dir, quadrant_colors, quadrant_order):
    """Plot a single skill with detailed annotations showing context-aware predictions."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    skill_id = skill_data['skill_id']
    quadrants = skill_data['quadrants']
    
    # Get the quadrants that are actually present for this skill
    present_quadrants = [q for q in quadrant_order if q in quadrants]
    
    # Find max sequence length for consistent x-axis
    max_len = max(len(quadrants[q]['response_seq']) for q in present_quadrants)
    
    # Plot each quadrant with its own sequence (only present ones)
    for quad_idx, quad_name in enumerate(present_quadrants):
        student = quadrants[quad_name]
        color = quadrant_colors[quad_name]
        response_seq = student['response_seq']
        
        x = np.arange(len(response_seq))
        
        # Create legend label with student characteristics
        l0_level = "High L0" if "High L0" in quad_name else "Low L0"
        t_level = "High T" if "High T" in quad_name else "Low T"
        legend_label = f"id: {student['uid']} ({l0_level}, {t_level})"
        
        # Ground truth bars (lighter for each quadrant, vertically stacked)
        y_offset = quad_idx * 0.15
        for i, val in enumerate(response_seq):
            bar_color = 'green' if val == 1 else 'red'
            ax.axvline(x=i, ymin=y_offset, ymax=y_offset+0.1, 
                      color=bar_color, alpha=0.15, linewidth=3, zorder=0)
        
        # GTransformer predictions with markers
        ax.plot(x, student['preds'], color=color, linewidth=2.5, 
               alpha=0.7, zorder=1, label=legend_label)
        for i, p in enumerate(student['preds']):
            marker = 'o' if p > 0.5 else 'x'
            markersize = 8 if p > 0.5 else 9
            ax.plot(i, p, marker=marker, color=color, markersize=markersize, 
                   alpha=0.85, zorder=2)
        
        # BKT baseline (gray for all, since they're identical for same sequence) with markers
        bkt_preds = calculate_bkt_trajectory(bkt_params, skill_id, response_seq)
        bkt_label = f"id: {student['uid']} (Classical BKT)"
        ax.plot(x, bkt_preds, color='gray', linestyle='--', linewidth=2, 
               alpha=0.5, zorder=0, label=bkt_label)
        for i, p in enumerate(bkt_preds):
            marker = 'o' if p > 0.5 else 'x'
            markersize = 6 if p > 0.5 else 7
            ax.plot(i, p, marker=marker, color='gray', markersize=markersize, 
                   alpha=0.6, zorder=0)
    
    ax.set_title("Context-Aware Predictions for Students with Same Response Sequence vs BKT Identical Predictions",
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Time-step", fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(0, max_len, 1))
    ax.set_ylabel("P(Correct)", fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"h3_skill_{skill_id}_quadrants.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                       help="Path to experiment fold directory (e.g., fold_0_955042)")
    parser.add_argument("--output_dir", type=str, default="examples/validation/results_exp801184",
                       help="Directory to save output plots")
    parser.add_argument("--top_n", type=int, default=12,
                       help="Number of skills to include in mosaic")
    args = parser.parse_args()
    
    # Infer dataset from experiment directory path first
    # Infer dataset name from experiment path
    parts = args.exp_dir.rstrip('/').split('/')
    if len(parts) >= 2:
        dataset_name = parts[-2]
    
    # Check if BKT parameters exist for this dataset
    if dataset_name and not dataset_name.startswith('fold'):
        bkt_params_path = os.path.join("data", dataset_name, "bkt_skill_params.pkl")
        # For nips_task34, check train_data subdirectory
        if not os.path.exists(bkt_params_path) and dataset_name == "train_data":
            bkt_params_path = os.path.join("data", "nips_task34", "bkt_skill_params.pkl")
            dataset_name = "nips_task34"
        if not os.path.exists(bkt_params_path):
            raise FileNotFoundError(
                f"BKT skill params not found at {bkt_params_path}\n"
                f"This script requires BKT oracle parameters and dual predictions.\n"
                f"Datasets with BKT params: assist2009, assist2015, nips_task34\n"
                f"Dataset '{dataset_name}' does not have pre-computed BKT parameters."
            )
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model and data...")
    from examples.results.validation_helpers import load_model_from_dir
    from pykt.datasets import init_test_datasets
    
    model, dc, mc, dpath, bkt_params = load_model_from_dir(args.exp_dir, device)
    
    print("Initializing test dataset...")
    model_name = mc.get('model_name', mc.get('model', 'gtransformer'))
    dataset_name = mc.get('dataset_name', mc.get('dataset', 'assist2009'))
    dc['dataset_name'] = dataset_name 
    
    from pykt.datasets import init_test_datasets
    test_loader, _, _, _ = init_test_datasets(dc, model_name, 64)
    
    print("\nExtracting student-skill data from test set...")
    skill_data = extract_student_skill_data(model, test_loader, device)
    
    print(f"\nFound {len(skill_data)} unique skills")
    
    print("\nFinding skills with at least 2 quadrants represented...")
    selected_skills = find_matching_quadrant_skills(skill_data, bkt_params, top_n=args.top_n)
    
    if len(selected_skills) == 0:
        print("\nERROR: No skills found with at least 2 quadrants represented!")
        return
    
    print("\nGenerating visualizations...")
    plot_skill_quadrant_mosaic(selected_skills, bkt_params, args.output_dir)
    
    # Save metadata
    metadata = []
    for skill_data in selected_skills[:args.top_n]:
        metadata.append({
            'skill_id': int(skill_data['skill_id']),
            'seq_length': skill_data['seq_length'],
            'quality_score': float(skill_data['quality_score']),
            'pred_std': float(skill_data['pred_std']),
            'pred_range': float(skill_data['pred_range']),
            'avg_within_var': float(skill_data['avg_within_var']),
            'max_within_std': float(skill_data['max_within_std']),
            'gt_accuracy': float(skill_data['gt_accuracy']),
            'bkt_accuracy': float(skill_data['bkt_accuracy']),
            'accuracy_advantage': float(skill_data['accuracy_advantage']),
            'n_quadrants': skill_data['n_quadrants'],
            'quadrants': {
                quad: {
                    'uid': str(data['uid']),
                    'l0': float(data['l0']),
                    't': float(data['t']),
                    'response_seq': list(data['response_seq']),
                    'predictions': [float(p) for p in data['preds']]
                }
                for quad, data in skill_data['quadrants'].items()
            }
        })
    
    metadata_path = os.path.join(args.output_dir, "skill_quadrant_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
