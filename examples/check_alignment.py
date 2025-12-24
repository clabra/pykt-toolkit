
import os
import sys
import pandas as pd
import torch
import numpy as np
import json

# Add project root to path
project_root = "/workspaces/pykt-toolkit"
sys.path.append(project_root)

from pykt.datasets.init_dataset import init_dataset4train
from pykt.models.init_model import init_model

def main():
    dataset = "assist2009"
    fold = 0
    batch_size = 64
    csv_path = "experiments/20251223_215439_idkt_test_978367/traj_predictions.csv"
    
    data_config_path = "configs/data_config.json"
    with open(data_config_path, "r") as f:
        data_config = json.load(f)
    
    # Standardize paths
    for dname in data_config:
        dpath = data_config[dname]['dpath']
        if dpath.startswith("../"):
            data_config[dname]['dpath'] = dpath.replace("../", "")

    # Ensure use of _bkt.csv
    orig_file = data_config[dataset]['train_valid_file']
    bkt_file = orig_file.replace('.csv', '_bkt.csv')
    data_config[dataset]['train_valid_file'] = bkt_file
    print(f"Using dataset file: {bkt_file}")

    _, valid_loader = init_dataset4train(dataset, 'idkt', data_config, fold, batch_size)
    
    # ID Mapping
    ds = valid_loader.dataset
    if hasattr(ds, 'dataset'): ds = ds.dataset
    uid_to_index = ds.dori['uid_to_index']
    idx_to_uid = {v: k for k, v in uid_to_index.items()}
    print(f"Loaded {len(idx_to_uid)} students from loader mapping.")

    # Load CSV
    df = pd.read_csv(csv_path)
    csv_uids = df['student_id'].unique()
    print(f"Loaded {len(csv_uids)} unique students from CSV.")

    # Get first student from loader
    it = iter(valid_loader)
    data = next(it)
    uid_idx = data["uids"][0].item()
    sample_uid = idx_to_uid.get(uid_idx, uid_idx)
    
    print(f"\n--- Checking first student from Loader ---")
    print(f"Loader Index: {uid_idx}, Mapped UID: {sample_uid}")
    
    print(f"\nLoader Seq (first 10 interactions):")
    cshft = data["shft_cseqs"][0].numpy()
    rshft = data["shft_rseqs"][0].numpy()
    sm = data["smasks"][0].numpy()
    valid_idx = np.where(sm == 1)[0]
    for t in valid_idx[:10]:
        print(f"  Step: {t}, Skill: {int(cshft[t])}, Result: {int(rshft[t])}")

    print(f"\n--- Checking CSV for UID: {sample_uid} ---")
    csv_seq = df[df['student_id'] == sample_uid]
    if len(csv_seq) > 0:
        print(f"CSV Seq (first 10):")
        # CSV steps in traj_predictions are continuous, let's just take head
        for i, row in csv_seq.head(10).reset_index().iterrows():
            print(f"  Step: {i}, Skill: {int(row['skill_id'])}, Result: {int(row['y_true'])}, p_idkt: {row['p_idkt']:.6f}")
    else:
        print("UID NOT FOUND in CSV!")

    print(f"\n--- Checking Raw BKT CSV for UID: {sample_uid} ---")
    # We can't easily read raw CSV here without complex parsing, but let's try to match the first skill
    # row = ...

if __name__ == "__main__":
    main()
