#!/usr/bin/env python3
"""
Evaluation script for iDKT interpretability (alignment with BKT reference model).
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pickle

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.models import init_model
from pykt.datasets import init_dataset4train
from pykt.utils import set_seed

device = "cpu" if not torch.cuda.is_available() else "cuda"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate iDKT interpretability alignment")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Architecture
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--final_fc_dim", type=int, default=512)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--seq_len", type=int, default=200)
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    
    # Load data config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/data_config.json'))
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    # Add absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for ds in data_config:
        if 'dpath' in data_config[ds]:
            dpath = data_config[ds]['dpath']
            if dpath.startswith('../'):
                data_config[ds]['dpath'] = os.path.abspath(os.path.join(project_root, dpath.replace('../', '')))
            else:
                data_config[ds]['dpath'] = os.path.abspath(os.path.join(project_root, dpath))

    # Load BKT parameters
    bkt_params_path = os.path.join(data_config[args.dataset]['dpath'], 'bkt_skill_params.pkl')
    with open(bkt_params_path, 'rb') as f:
        bkt_skill_params = pickle.load(f)

    # Load Augmented Test Data
    # NOTE: We assume evaluate script normally uses test_file, but for interpretability we need BKT columns.
    # We'll use the train_valid_file_bkt.csv for now as a proxy if test_bkt.csv doesn't exist.
    # Actually, let's just use the validation loader from init_dataset4train which we know is augmented.
    
    dpath = data_config[args.dataset]['dpath']
    orig_file = data_config[args.dataset]['train_valid_file']
    data_config[args.dataset]['train_valid_file'] = orig_file.replace('.csv', '_bkt.csv')
    
    _, valid_loader = init_dataset4train(args.dataset, 'idkt', data_config, args.fold, args.batch_size)

    # Init Model
    model_config = {
        'd_model': args.d_model, 'd_ff': args.d_ff, 'num_attn_heads': args.n_heads,
        'n_blocks': args.n_blocks, 'dropout': args.dropout, 'final_fc_dim': args.final_fc_dim, 'l2': args.l2
    }
    model = init_model('idkt', model_config, data_config[args.dataset], args.emb_type)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    all_idkt_p = []
    all_bkt_p = []
    all_idkt_initmastery = []
    all_bkt_initmastery = []
    all_idkt_rate = []
    all_bkt_rate = []
    
    # Interaction-level records for heatmap
    interaction_records = []
    max_export_students = 1000 # Limit to avoid massive CSVs

    print("Running inference for interpretability alignment...")
    with torch.no_grad():
        for data in valid_loader:
            q, c, r = data["qseqs"].to(device), data["cseqs"].to(device), data["rseqs"].to(device)
            qshft, cshft, rshft = data["shft_qseqs"].to(device), data["shft_cseqs"].to(device), data["shft_rseqs"].to(device)
            sm = data["smasks"].to(device)
            uids = data["uids"]
            
            # Augmented ground truth
            bkt_p_batch = data["bkt_p_correct"].to(device)
            bkt_im_batch = data["bkt_mastery"].to(device)
            
            # Forward pass
            cc_full = torch.cat((c[:,0:1], cshft), dim=1)
            cr_full = torch.cat((r[:,0:1], rshft), dim=1)
            cq_full = torch.cat((q[:,0:1], qshft), dim=1)
            
            y, initmastery, rate, _ = model(cc_full.long(), cr_full.long(), cq_full.long())
            
            # Collect aggregated metrics
            all_idkt_p.extend(torch.masked_select(y[:, 1:], sm).cpu().numpy())
            all_bkt_p.extend(torch.masked_select(bkt_p_batch, sm).cpu().numpy())
            
            idkt_im_batch = initmastery[:, 1:]
            idkt_r_batch = rate[:, 1:]

            all_idkt_initmastery.extend(torch.masked_select(idkt_im_batch, sm).cpu().numpy())
            all_bkt_initmastery.extend(torch.masked_select(bkt_im_batch, sm).cpu().numpy())
            
            # Reference rates
            ref_rate_batch = torch.zeros_like(cshft).float().to(device)
            for b in range(cshft.shape[0]):
                for s in range(cshft.shape[1]):
                    skill_id = cshft[b, s].item()
                    if skill_id != -1:
                        ref_rate_batch[b, s] = bkt_skill_params['params'].get(skill_id, bkt_skill_params['global'])['learns']
            
            all_idkt_rate.extend(torch.masked_select(idkt_r_batch, sm).cpu().numpy())
            all_bkt_rate.extend(torch.masked_select(ref_rate_batch, sm).cpu().numpy())

            # Export interaction records for heatmap
            for b in range(uids.shape[0]):
                uid = uids[b].item()
                if uid >= max_export_students: continue
                
                # Get indices where mask is true for this student
                student_mask = sm[b]
                indices = torch.where(student_mask)[0]
                
                for idx in indices:
                    interaction_records.append({
                        'student_id': uid,
                        'skill_id': cshft[b, idx].item(),
                        'Mi': idkt_im_batch[b, idx].item(),
                        'M_rasch': bkt_im_batch[b, idx].item(), # Using BKT mastery as "M_rasch" for heatmap compatibility
                    })

    # Save interaction-level data for heatmap
    if interaction_records:
        df_mastery = pd.DataFrame(interaction_records)
        mastery_path = os.path.join(args.output_dir, "mastery_test.csv")
        df_mastery.to_csv(mastery_path, index=False)
        print(f"✓ Saved interaction data for heatmap: {mastery_path}")

    # Calculate Alignment Metrics
    results = {}
    
    def calc_metrics(name, pred, ref):
        mse = np.mean((pred - ref)**2)
        corr, _ = pearsonr(pred, ref)
        return {f"{name}_mse": float(mse), f"{name}_corr": float(corr)}

    results.update(calc_metrics("prediction", np.array(all_idkt_p), np.array(all_bkt_p)))
    results.update(calc_metrics("initmastery", np.array(all_idkt_initmastery), np.array(all_bkt_initmastery)))
    results.update(calc_metrics("learning_rate", np.array(all_idkt_rate), np.array(all_bkt_rate)))

    print("\nAlignment Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "interpretability_alignment.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
