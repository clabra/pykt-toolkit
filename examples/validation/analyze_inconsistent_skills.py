#!/usr/bin/env python3
"""
Deep analysis of why certain skills show pedagogically inconsistent predictions.

Investigates:
1. Parameter classification accuracy (are students correctly classified into quadrants?)
2. Prediction-parameter alignment (do predictions follow parameter values?)
3. Temporal dynamics (how do parameters evolve during the sequence?)
"""

import json
import numpy as np
import sys

def analyze_skill(skill_data):
    """Analyze a single skill for inconsistencies."""
    skill_id = skill_data['skill_id']
    quadrants = skill_data['quadrants']
    
    print(f"\n{'='*80}")
    print(f"SKILL {skill_id} DEEP ANALYSIS")
    print(f"{'='*80}\n")
    
    # Extract data for each quadrant
    for quad_name, quad_data in quadrants.items():
        uid = quad_data['uid']
        l0 = quad_data['l0']
        t = quad_data['t']
        responses = quad_data['response_seq']
        preds = quad_data['predictions']
        
        print(f"{quad_name}:")
        print(f"  Student ID: {uid}")
        print(f"  Historical L0 (avg): {l0:.4f}")
        print(f"  Historical T (avg): {t:.4f}")
        print(f"  Response sequence: {responses}")
        print(f"  Predictions: {[f'{p:.3f}' for p in preds]}")
        print(f"  Avg prediction: {np.mean(preds):.3f}")
        print(f"  First prediction: {preds[0]:.3f}")
        print(f"  Last prediction: {preds[-1]:.3f}")
        print(f"  Trajectory slope: {preds[-1] - preds[0]:+.3f}")
        print()
    
    # Analyze the inconsistency
    print("INCONSISTENCY ANALYSIS:")
    
    # Check if High L0 students have lower predictions than Low L0
    quad_avgs = {q: np.mean(d['predictions']) for q, d in quadrants.items()}
    
    if "High L0 / Low T" in quad_avgs and "Low L0 / Low T" in quad_avgs:
        high_l0 = quad_avgs["High L0 / Low T"]
        low_l0 = quad_avgs["Low L0 / Low T"]
        if high_l0 < low_l0:
            print(f"  ✗ Low T group: High L0 ({high_l0:.3f}) < Low L0 ({low_l0:.3f})")
            print(f"    Difference: {low_l0 - high_l0:.3f} (should be negative)")
            
            # Check parameter differences
            high_l0_val = quadrants["High L0 / Low T"]['l0']
            low_l0_val = quadrants["Low L0 / Low T"]['l0']
            print(f"    L0 difference: {high_l0_val - low_l0_val:+.4f} (High - Low)")
            
            # Check first predictions (should reflect L0)
            high_first = quadrants["High L0 / Low T"]['predictions'][0]
            low_first = quadrants["Low L0 / Low T"]['predictions'][0]
            print(f"    First pred difference: {high_first - low_first:+.3f} (High - Low)")
            
    if "High L0 / High T" in quad_avgs and "Low L0 / High T" in quad_avgs:
        high_l0 = quad_avgs["High L0 / High T"]
        low_l0 = quad_avgs["Low L0 / High T"]
        if high_l0 < low_l0:
            print(f"  ✗ High T group: High L0 ({high_l0:.3f}) < Low L0 ({low_l0:.3f})")
            print(f"    Difference: {low_l0 - high_l0:.3f} (should be negative)")
            
            # Check parameter differences
            high_l0_val = quadrants["High L0 / High T"]['l0']
            low_l0_val = quadrants["Low L0 / High T"]['l0']
            print(f"    L0 difference: {high_l0_val - low_l0_val:+.4f} (High - Low)")
            
            # Check first predictions (should reflect L0)
            high_first = quadrants["High L0 / High T"]['predictions'][0]
            low_first = quadrants["Low L0 / High T"]['predictions'][0]
            print(f"    First pred difference: {high_first - low_first:+.3f} (High - Low)")
    
    print("\nPOSSIBLE CAUSES:")
    print("  1. Historical averaging may not reflect skill-specific L0")
    print("  2. Model may be using skill-specific context that contradicts historical params")
    print("  3. First-timestep predictions may be influenced by other factors")
    print("  4. Parameter extraction may have issues")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_inconsistent_skills.py <metadata_json_path> [skill_ids...]")
        print("Example: python analyze_inconsistent_skills.py metadata.json 12 0 5")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    target_skills = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else [12, 0, 5]
    
    with open(metadata_path, 'r') as f:
        all_skills = json.load(f)
    
    # Filter to target skills
    for skill_data in all_skills:
        if skill_data['skill_id'] in target_skills:
            analyze_skill(skill_data)
