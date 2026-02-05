
import os
import pandas as pd
import numpy as np
import pickle
import argparse
import json
from pathlib import Path

def load_bkt_params(dataset_path):
    possible_paths = [
        dataset_path / 'bkt_skill_params.pkl',
        dataset_path / 'bkt' / 'bkt_skill_params.pkl',
        dataset_path.parent / 'bkt_skill_params.pkl',  # Check parent directory
        dataset_path.parent / 'bkt' / 'bkt_skill_params.pkl'
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                return data['params']
    
    raise FileNotFoundError(f"BKT parameters not found in {dataset_path} or parent directory")

def generate_targets(df, bkt_params, max_seq_len=200):
    """
    Generate step-wise targets (L0, T) for each sequence in the dataframe.
    Includes Late Fusion (Mean) for multi-skill questions.
    """
    num_sequences = len(df)
    target_l0 = np.zeros((num_sequences, max_seq_len), dtype=np.float32)
    target_t = np.zeros((num_sequences, max_seq_len), dtype=np.float32)
    
    for idx, row in df.iterrows():
        # concepts/questions are strings
        q_list = [x for x in row['questions'].split(',') if x != '-1']
        c_list = [int(x) for x in row['concepts'].split(',') if x != '-1']
        
        # Use qidxs if available, else infer groups from consecutive question IDs
        if 'qidxs' in df.columns:
            q_ids = [int(x) for x in row['qidxs'].split(',') if x != '-1']
        else:
            # Infer groups from consecutive identical question IDs
            q_ids = []
            cur_qidx = 0
            if len(q_list) > 0:
                q_ids.append(0)
                for i in range(1, len(q_list)):
                    if q_list[i] != q_list[i-1]:
                        cur_qidx += 1
                    q_ids.append(cur_qidx)
        
        seq_len = min(len(c_list), max_seq_len)
        
        # Group by qidx
        groups = {}
        for t in range(seq_len):
            qid_step = q_ids[t]
            groups.setdefault(qid_step, []).append(t)
            
        # Process each question group
        for qid_step, steps in groups.items():
            skills = [c_list[t] for t in steps]
            l0_vals = []
            t_vals = []
            
            for s in skills:
                if s in bkt_params:
                    l0_vals.append(bkt_params[s]['prior'])
                    t_vals.append(bkt_params[s]['learns'])
                else:
                    # Use default BKT values for skills with no first-attempt data (all repeats)
                    # This happens when a skill only appears in repeat/review problems
                    l0_vals.append(0.5)  # Neutral prior
                    t_vals.append(0.1)   # Low learning rate
            
            # Late Fusion: Mean
            mean_l0 = np.mean(l0_vals)
            mean_t = np.mean(t_vals)
            
            # Distribute mean to all steps in the question
            for t in steps:
                target_l0[idx, t] = mean_l0
                target_t[idx, t] = mean_t
                
    return target_l0, target_t

def main():
    parser = argparse.ArgumentParser(description='Generate BKT soft labels for Active Grounding with Late Fusion')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Maximum sequence length')
    args = parser.parse_args()
    
    project_root = Path("/workspaces/pykt-toolkit")
    with open(project_root / 'configs' / 'data_config.json', 'r') as f:
        data_config = json.load(f)
    
    dataset_info = data_config[args.dataset]
    dpath = project_root / dataset_info['dpath'].replace('../', '')
    
    print(f"Dataset path: {dpath}")
    bkt_params = load_bkt_params(dpath)
    print(f"Loaded BKT parameters for {len(bkt_params)} skills.")

    # 1. Process train_valid_sequences.csv
    train_valid_path = dpath / 'train_valid_sequences.csv'
    if train_valid_path.exists():
        print(f"Processing {train_valid_path} (Aligned with 5-fold CV)...")
        df = pd.read_csv(train_valid_path)
        l0, t = generate_targets(df, bkt_params, args.max_seq_len)
        output_file = dpath / 'bkt_targets_train_valid.npz'
        np.savez(output_file, target_l0=l0, target_t=t)
        print(f"Saved targets to {output_file}")

    # 2. Process test_question_sequences.csv (Priority for Question-Level Alignment)
    test_path = dpath / 'test_question_sequences.csv'
    if not test_path.exists():
        test_path = dpath / 'test_sequences.csv'
        
    if test_path.exists():
        print(f"Processing {test_path} (Aligned with Benchmark Evaluation)...")
        df_test = pd.read_csv(test_path)
        l0_test, t_test = generate_targets(df_test, bkt_params, args.max_seq_len)
        output_file_test = dpath / 'bkt_targets_test.npz'
        np.savez(output_file_test, target_l0=l0_test, target_t=t_test)
        print(f"Saved test targets to {output_file_test}")

if __name__ == '__main__':
    main()
