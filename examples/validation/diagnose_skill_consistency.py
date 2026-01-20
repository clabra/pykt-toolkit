#!/usr/bin/env python3
"""
Diagnose pedagogical consistency of skill predictions.

Checks whether predictions follow expected ordering:
- High L0 should predict HIGHER than Low L0 (for same T level)
- High T should show faster learning (steeper trajectory) than Low T (for same L0 level)
"""

import json
import numpy as np
import sys

def check_pedagogical_consistency(metadata_path):
    """
    Check if predictions follow pedagogical expectations.
    
    Expected orderings:
    1. High L0 > Low L0 (for same T level)
    2. High T shows faster learning than Low T (for same L0 level)
    """
    with open(metadata_path, 'r') as f:
        skills = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"PEDAGOGICAL CONSISTENCY ANALYSIS")
    print(f"{'='*80}\n")
    
    consistent_skills = []
    inconsistent_skills = []
    
    for skill in skills:
        skill_id = skill['skill_id']
        quadrants = skill['quadrants']
        
        # Extract average predictions for each quadrant
        quad_avgs = {}
        for quad_name, quad_data in quadrants.items():
            quad_avgs[quad_name] = np.mean(quad_data['predictions'])
        
        # Check L0 ordering (High L0 should be > Low L0 for same T)
        l0_violations = []
        
        # Check Low T: High L0/Low T should be > Low L0/Low T
        if "High L0 / Low T" in quad_avgs and "Low L0 / Low T" in quad_avgs:
            high_l0_low_t = quad_avgs["High L0 / Low T"]
            low_l0_low_t = quad_avgs["Low L0 / Low T"]
            if high_l0_low_t < low_l0_low_t:
                l0_violations.append(f"Low T: High L0 ({high_l0_low_t:.3f}) < Low L0 ({low_l0_low_t:.3f})")
        
        # Check High T: High L0/High T should be > Low L0/High T
        if "High L0 / High T" in quad_avgs and "Low L0 / High T" in quad_avgs:
            high_l0_high_t = quad_avgs["High L0 / High T"]
            low_l0_high_t = quad_avgs["Low L0 / High T"]
            if high_l0_high_t < low_l0_high_t:
                l0_violations.append(f"High T: High L0 ({high_l0_high_t:.3f}) < Low L0 ({low_l0_high_t:.3f})")
        
        # Check T ordering (High T should be > Low T for same L0)
        t_violations = []
        
        # Check Low L0: Low L0/High T should be > Low L0/Low T
        if "Low L0 / High T" in quad_avgs and "Low L0 / Low T" in quad_avgs:
            low_l0_high_t = quad_avgs["Low L0 / High T"]
            low_l0_low_t = quad_avgs["Low L0 / Low T"]
            if low_l0_high_t < low_l0_low_t:
                t_violations.append(f"Low L0: High T ({low_l0_high_t:.3f}) < Low T ({low_l0_low_t:.3f})")
        
        # Check High L0: High L0/High T should be > High L0/Low T
        if "High L0 / High T" in quad_avgs and "High L0 / Low T" in quad_avgs:
            high_l0_high_t = quad_avgs["High L0 / High T"]
            high_l0_low_t = quad_avgs["High L0 / Low T"]
            if high_l0_high_t < high_l0_low_t:
                t_violations.append(f"High L0: High T ({high_l0_high_t:.3f}) < Low T ({high_l0_low_t:.3f})")
        
        # Check diagonal ordering
        diagonal_violations = []
        
        # High L0/High T should be >= Low L0/Low T (both dimensions favor high)
        if "High L0 / High T" in quad_avgs and "Low L0 / Low T" in quad_avgs:
            high_high = quad_avgs["High L0 / High T"]
            low_low = quad_avgs["Low L0 / Low T"]
            if high_high < low_low:
                diagonal_violations.append(f"Diagonal: High L0/High T ({high_high:.3f}) < Low L0/Low T ({low_low:.3f})")
        
        # Report
        all_violations = l0_violations + t_violations + diagonal_violations
        is_consistent = len(all_violations) == 0
        
        if is_consistent:
            consistent_skills.append(skill_id)
            print(f"✓ Skill {skill_id:3d}: CONSISTENT")
        else:
            inconsistent_skills.append({
                'skill_id': skill_id,
                'l0_violations': l0_violations,
                't_violations': t_violations,
                'diagonal_violations': diagonal_violations,
                'quadrant_avgs': quad_avgs,
                'quality_score': skill['quality_score'],
                'accuracy_advantage': skill['accuracy_advantage']
            })
            print(f"✗ Skill {skill_id:3d}: INCONSISTENT")
            for violation in all_violations:
                print(f"    - {violation}")
            print(f"    Quality: {skill['quality_score']:.4f}, Advantage: {skill['accuracy_advantage']:+.3f}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Consistent skills: {len(consistent_skills)}/{len(skills)}")
    print(f"Inconsistent skills: {len(inconsistent_skills)}/{len(skills)}")
    
    if inconsistent_skills:
        print(f"\n{'='*80}")
        print(f"INCONSISTENT SKILLS DETAILS")
        print(f"{'='*80}\n")
        
        for item in inconsistent_skills:
            print(f"\nSkill {item['skill_id']}:")
            print(f"  Quality Score: {item['quality_score']:.4f}")
            print(f"  Accuracy Advantage: {item['accuracy_advantage']:+.3f}")
            print(f"  Quadrant Averages:")
            for quad, avg in sorted(item['quadrant_avgs'].items()):
                print(f"    {quad:20s}: {avg:.3f}")
            if item['l0_violations']:
                print(f"  L0 Violations:")
                for violation in item['l0_violations']:
                    print(f"    - {violation}")
            if item['t_violations']:
                print(f"  T Violations:")
                for violation in item['t_violations']:
                    print(f"    - {violation}")
            if item['diagonal_violations']:
                print(f"  Diagonal Violations:")
                for violation in item['diagonal_violations']:
                    print(f"    - {violation}")
    
    return consistent_skills, inconsistent_skills


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_skill_consistency.py <metadata_json_path>")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    consistent, inconsistent = check_pedagogical_consistency(metadata_path)
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    print(f"Filter out inconsistent skills from the visualization.")
    print(f"Update the selection script to add a pedagogical consistency check.")
