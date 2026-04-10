#!/usr/bin/env python3
"""
Extract Trajectory Data (Initial Mastery, Learn Rate) from GTransformer.
Generates traj_rate.csv and traj_initmastery.csv required for roster plots.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from examples.results.validation_helpers import load_model_from_dir
from pykt.datasets.init_dataset import init_test_datasets
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help="Path to experiment fold directory")
    parser.add_argument('--limit_students', type=int, default=500, help="Max students to extract")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.run_dir}...")
    
    try:
        model, dc, mc, dpath, bkt_params = load_model_from_dir(args.run_dir, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Load dataset
    # We use the train_valid_sequences.csv fold split for interpretability analysis
    dataset_name = dc.get('dataset_name', mc.get('dataset', 'assist2009'))
    fold = mc.get('fold', 0)
    
    # Standard pykt test datasets or custom extraction
    print(f"Extracting trajectories for {dataset_name} fold {fold}...")
    
    # Re-initialize dataset to get aligned UIDs
    test_file = os.path.join(dpath, dc.get('train_valid_file', 'train_valid_sequences.csv'))
    # Handle older naming if needed
    if not os.path.exists(test_file):
        test_file = os.path.join(dpath, 'train_valid_sequences.csv')
        
    ds = GTransformerDataset(test_file, dc["input_type"], {fold})
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    
    # Mappings
    uid_to_index = ds.dori['uid_to_index']
    idx_to_uid = {v: k for k, v in uid_to_index.items()}
    
    rate_records = []
    init_records = []
    
    student_count = 0
    with torch.no_grad():
        for data in loader:
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].bool().to(device)
            uids = data["uids"]
            
            # Forward pass
            # GTransformer expects (c, r, pid_data=q, uid_data=uids)
            # Note: handle shifted inputs if necessary, but GTransformer handles them internally or expects shifted
            # Based on gtransformer.py forward:
            outputs, _ = model(c, r, pid_data=q, uid_data=uids.to(device))
            
            p_l0 = outputs['p_l0'].cpu().numpy()
            p_t = outputs['p_t'].cpu().numpy()
            
            for b in range(uids.shape[0]):
                uid = idx_to_uid[uids[b].item()]
                mask = sm[b].cpu().numpy()
                
                cur_c = c[b].cpu().numpy()[mask]
                cur_l0 = p_l0[b][mask]
                cur_t = p_t[b][mask]
                
                for i in range(len(cur_c)):
                    rate_records.append({
                        'student_id': uid,
                        'interaction_idx': i,
                        'skill_id': int(cur_c[i]),
                        'ts': float(cur_t[i])
                    })
                    init_records.append({
                        'student_id': uid,
                        'interaction_idx': i,
                        'skill_id': int(cur_c[i]),
                        'lc': float(cur_l0[i])
                    })
                
                student_count += 1
                if student_count >= args.limit_students:
                    break
            
            if student_count >= args.limit_students:
                break

    # Save to CSV
    df_rate = pd.DataFrame(rate_records)
    df_init = pd.DataFrame(init_records)
    
    rate_path = os.path.join(args.run_dir, 'traj_rate.csv')
    init_path = os.path.join(args.run_dir, 'traj_initmastery.csv')
    
    df_rate.to_csv(rate_path, index=False)
    df_init.to_csv(init_path, index=False)
    
    print(f"Successfully saved trajectories to:")
    print(f"  - {rate_path}")
    print(f"  - {init_path}")

if __name__ == "__main__":
    main()
