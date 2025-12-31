import pandas as pd
import numpy as np
import os

def find_twins(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure uid is string for grouping
    df['student_id'] = df['student_id'].astype(str)
    
    # Group by student_id and skill_id and collect sequences
    print("Grouping sequences...")
    student_sequences = {}
    
    # We use a subset of the dataframe if it's too large, but 5MB should be fine
    concept_col = 'skill_id'

    for (uid, concept), group in df.groupby(['student_id', concept_col]):
        # Sequence must be long enough to be interesting
        if len(group) < 10:
            continue
        
        y_seq = tuple(group['y_true'].tolist())
        p_idkt_seq = group['p_idkt'].tolist()
        p_bkt_seq = group['p_bkt'].tolist()
        
        key = (concept, y_seq)
        if key not in student_sequences:
            student_sequences[key] = []
        
        student_sequences[key].append({
            'uid': uid,
            'p_idkt': p_idkt_seq,
            'p_bkt': p_bkt_seq,
            'len': len(group)
        })
    
    # Filter for twins (concept and response sequence are exactly the same)
    twins = {k: v for k, v in student_sequences.items() if len(v) > 1}
    
    print(f"Found {len(twins)} Twin Sequence types.")
    
    # Sort by sequence length to find the best examples
    sorted_twins = sorted(twins.items(), key=lambda x: len(x[0][1]), reverse=True)
    
    for (concept, y_seq), holders in sorted_twins:
        if len(holders) < 2: continue
        
        # Calculate stats for the first two holders
        uids = [h['uid'] for h in holders[:2]]
        global_accs = [df[df['student_id'] == uid]['y_true'].mean() for uid in uids]
        p_starts = [h['p_idkt'][0] for h in holders[:2]]
        
        # We want the one with higher global acc to also have higher p_idkt start
        # AND we want a significant difference in global acc
        if abs(global_accs[0] - global_accs[1]) < 0.2:
            continue
            
        is_consistent = (global_accs[0] > global_accs[1]) == (p_starts[0] > p_starts[1])
        
        if not is_consistent:
            continue

        max_idkt_diff = np.abs(np.array(holders[0]['p_idkt']) - np.array(holders[1]['p_idkt'])).max()
        if max_idkt_diff < 0.2:
            continue

        print(f"\n[INTUITIVE PAIR] Concept: {concept}, Len: {len(y_seq)}, Max Diff: {max_idkt_diff:.4f}")
        for i, h in enumerate(holders[:2]):
            print(f"  UID: {h['uid']} (Global Success: {global_accs[i]:.1%})")
            print(f"    p_idkt start/end: {h['p_idkt'][0]:.4f} -> {h['p_idkt'][-1]:.4f}")

if __name__ == "__main__":
    path = "experiments/20251230_224907_idkt_setS-pure_364494/traj_predictions.csv"
    find_twins(path)
