
import os
import pandas as pd
import numpy as np
import pickle
import argparse
import json
from pathlib import Path

def load_keyid2idx(dataset_path):
    with open(dataset_path / 'keyid2idx.json', 'r') as f:
        return json.load(f)

def load_bkt_params(dataset_path):
    # Try different possible locations for parameters
    possible_paths = [
        dataset_path / 'bkt_skill_params.pkl',
        dataset_path / 'bkt' / 'bkt_skill_params.pkl'
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                return data['params']
    
    # Fallback to parameters.json if pkl not found
    json_path = dataset_path / 'bkt' / 'parameters.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            params_json = json.load(f)
            # Map names to IDs if possible, but pkl is preferred
            print(f"Warning: Loading from {json_path}. This might require name-to-ID mapping.")
            return params_json
            
    raise FileNotFoundError(f"BKT parameters not found in {dataset_path}")

def generate_targets(df, bkt_params, max_seq_len=200):
    """
    Generate step-wise targets (L0, T) for each sequence in the dataframe.
    """
    num_sequences = len(df)
    target_l0 = np.zeros((num_sequences, max_seq_len), dtype=np.float32)
    target_t = np.zeros((num_sequences, max_seq_len), dtype=np.float32)
    
    for idx, row in df.iterrows():
        # concepts are original skill IDs in the CSV
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        
        # Fill targets up to min(len(concepts), max_seq_len)
        seq_len = min(len(concepts), max_seq_len)
        for t in range(seq_len):
            skill_id = concepts[t]
            if skill_id in bkt_params:
                target_l0[idx, t] = bkt_params[skill_id]['prior']
                target_t[idx, t] = bkt_params[skill_id]['learns']
            else:
                # Default values if skill not found in BKT (should not happen if BKT was fit on same data)
                target_l0[idx, t] = 0.5
                target_t[idx, t] = 0.1
                
    return target_l0, target_t

def main():
    parser = argparse.ArgumentParser(description='Generate BKT soft labels for Active Grounding')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Maximum sequence length')
    args = parser.parse_args()
    
    # Load data config
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / 'configs' / 'data_config.json', 'r') as f:
        data_config = json.load(f)
    
    if args.dataset not in data_config:
        print(f"Error: dataset {args.dataset} not in data_config.json")
        return
        
    dataset_info = data_config[args.dataset]
    dpath = project_root / dataset_info['dpath'].replace('../', '')
    
    print(f"Dataset path: {dpath}")
    
    # Load BKT parameters
    try:
        bkt_params = load_bkt_params(dpath)
        print(f"Loaded BKT parameters for {len(bkt_params)} skills.")
    except Exception as e:
        print(f"Error loading BKT parameters: {e}")
        return

    # Process train/valid data
    train_valid_path = dpath / 'train_valid_sequences.csv'
    if train_valid_path.exists():
        print(f"Processing {train_valid_path}...")
        df = pd.read_csv(train_valid_path)
        l0, t = generate_targets(df, bkt_params, args.max_seq_len)
        
        output_file = dpath / 'bkt_targets_train_valid.npz'
        np.savez(output_file, target_l0=l0, target_t=t)
        print(f"Saved targets to {output_file}")
    else:
        print(f"Warning: {train_valid_path} not found.")

    # Also process test data if we want to evaluate probe MSE on test set
    test_path = dpath / 'test_sequences.csv'
    if test_path.exists():
        print(f"Processing {test_path}...")
        df_test = pd.read_csv(test_path)
        l0_test, t_test = generate_targets(df_test, bkt_params, args.max_seq_len)
        
        output_file_test = dpath / 'bkt_targets_test.npz'
        np.savez(output_file_test, target_l0=l0_test, target_t=t_test)
        print(f"Saved test targets to {output_file_test}")

if __name__ == '__main__':
    main()
