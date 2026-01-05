import pandas as pd
import os
import argparse
import json
from pathlib import Path

def get_s_uids(dpath, filename, max_len):
    """
    Identify UIDs that have a total history length <= max_len.
    Expected format: csv with 'concepts' column containing comma-separated IDs.
    """
    f_path = os.path.join(dpath, filename)
    if not os.path.exists(f_path):
        print(f"  Warning: Parent file {filename} not found in {dpath}")
        return set()
    
    df = pd.read_csv(f_path)
    # Count tokens in the concepts column
    if 'concepts' not in df.columns:
        print(f"  Error: 'concepts' column not found in {f_path}")
        return set()
        
    df['seq_len'] = df['concepts'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    s_uids = set(df[df['seq_len'] <= max_len]['uid'].astype(str))
    return s_uids

def filter_file(dpath, src_name, dst_name, s_uids):
    """Filter a sequence file to keep only Set S rows."""
    src_path = os.path.join(dpath, src_name)
    dst_path = os.path.join(dpath, dst_name)
    
    if not os.path.exists(src_path):
        return
        
    print(f"  Filtering {src_name} -> {dst_name}...")
    df = pd.read_csv(src_path)
    if 'uid' not in df.columns:
        print(f"    Skipping: No 'uid' column in {src_name}")
        return
        
    df['uid_str'] = df['uid'].astype(str)
    df_filtered = df[df['uid_str'].isin(s_uids)].drop(columns=['uid_str'])
    df_filtered.to_csv(dst_path, index=False)
    print(f"    Done: {len(df)} -> {len(df_filtered)} rows")

def main():
    parser = argparse.ArgumentParser(description='Create Set S (Short Learner) version of a dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., assist2009)')
    parser.add_argument('--max_len', type=int, default=200, help='Max sequence length for Set S (default: 200)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / 'configs' / 'data_config.json'
    
    with open(config_path, 'r') as f:
        data_config = json.load(f)
        
    if args.dataset not in data_config:
        print(f"Error: Dataset {args.dataset} not found in data_config.json")
        return
        
    dpath = data_config[args.dataset]['dpath']
    if dpath.startswith('../'):
        dpath = os.path.join(project_root, dpath[3:])
        
    print(f"Creating Set S for {args.dataset} (max_len={args.max_len}) in {dpath}")

    # 1. Identify S UIDs
    train_file = data_config[args.dataset].get('train_valid_original_file', 'train_valid.csv')
    test_file = data_config[args.dataset].get('test_original_file', 'test.csv')
    
    s_uids = get_s_uids(dpath, train_file, args.max_len)
    s_uids |= get_s_uids(dpath, test_file, args.max_len)
    
    if not s_uids:
        print("Error: No S-students found. Check dataset format.")
        return
        
    print(f"Found {len(s_uids)} total S-students in dataset.")

    # 2. Filter all relevant sequence variations
    # We look for common patterns used in PyKT
    train_seq = data_config[args.dataset].get('train_valid_file', 'train_valid_sequences.csv')
    test_seq = data_config[args.dataset].get('test_file', 'test_sequences.csv')
    test_win = data_config[args.dataset].get('test_window_file', 'test_window_sequences.csv')
    train_que = data_config[args.dataset].get('train_valid_original_file_quelevel', 'train_valid_quelevel.csv')
    test_que = data_config[args.dataset].get('test_original_file_quelevel', 'test_quelevel.csv')

    targets = [
        (train_seq, train_seq.replace('.csv', '_S.csv')),
        (test_seq, test_seq.replace('.csv', '_S.csv')),
        (train_seq.replace('.csv', '_bkt.csv'), train_seq.replace('.csv', '_S_bkt.csv')),
        (test_seq.replace('.csv', '_bkt.csv'), test_seq.replace('.csv', '_S_bkt.csv')),
        (test_win, test_win.replace('.csv', '_S.csv')),
        (train_que, train_que.replace('.csv', '_S.csv')),
        (test_que, test_que.replace('.csv', '_S.csv'))
    ]
    
    for src, dst in targets:
        filter_file(dpath, src, dst, s_uids)

    print("\nSet S creation complete.")
    print("Next step: Update configs/data_config.json to include the _S fields.")

if __name__ == '__main__':
    main()
