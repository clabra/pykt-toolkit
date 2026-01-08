import pandas as pd
import sys

def find_perfect_divergence(csv_path, skill_id):
    df = pd.read_csv(csv_path)
    df['student_id'] = df['student_id'].astype(str)
    
    skill_df = df[df['skill_id'] == skill_id]
    if skill_df.empty: return None
    
    sequences = skill_df.groupby('student_id')['y_true'].apply(list)
    idkt_preds = skill_df.groupby('student_id')['p_idkt'].apply(list)
    uids = sequences.index.tolist()
    
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            u1, u2 = uids[i], uids[j]
            s1, s2 = sequences[u1], sequences[u2]
            
            if len(s1) < 6 or s1 != s2: continue
            
            p1, p2 = idkt_preds[u1], idkt_preds[u2]
            
            # Identify who starts higher
            if p1[0] > p2[0]:
                high_start, low_start = p1, p2
                u_high, u_low = u1, u2
            else:
                high_start, low_start = p2, p1
                u_high, u_low = u2, u1
                
            # Requirement 1: Significant Start Gap (> 0.1)
            if high_start[0] - low_start[0] < 0.1: continue
            
            # Requirement 2: Cross-over (The one starting lower ends higher by > 0.05)
            if low_start[-1] > high_start[-1] + 0.05:
                # Find intersection
                ix = None
                for t in range(len(p1)-1):
                    if (p1[t] - p2[t]) * (p1[t+1] - p2[t+1]) < 0:
                        ix = t + 0.5
                        iy = (p1[t] + p2[t]) / 2
                        break
                return [u_high, u_low], s1, ix, iy
                
    return None

if __name__ == "__main__":
    csv = "experiments/20260107_210218_benchpaper/idkt/assist2009_S/fold_0_123082/traj_predictions.csv"
    
    found = False
    for s_id in range(123):
        res = find_perfect_divergence(csv, s_id)
        if res:
            twins, seq, ix, iy = res
            print(f"SELECTED_UIDS={twins[0]},{twins[1]} SELECTED_SKILL={s_id} INTERSECT=({ix}, {iy}) SEQ={seq}")
            found = True
            break
