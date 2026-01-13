#!/usr/bin/env python3
"""
Evaluate pyBKT on PyKT datasets using Question-Level Late Fusion (Mean) metrics.
This ensures a fair comparison between BKT and transformer models like GTransformer.

Usage:
    python examples/eval_bkt_question_level.py --dataset assist2009_bkt --fold 0
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import json
from sklearn import metrics
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.datasets import init_dataset4train
from pyBKT.models import Model, Roster

def main():
    parser = argparse.ArgumentParser(description='Question-level evaluation for BKT')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # Load data config
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)

    # Add absolute paths
    for ds in data_config:
        if 'dpath' in data_config[ds]:
            dpath = data_config[ds]['dpath']
            if dpath.startswith('../'):
                data_config[ds]['dpath'] = os.path.abspath(os.path.join(project_root, dpath.replace('../', '')))
            else:
                data_config[ds]['dpath'] = os.path.abspath(os.path.join(project_root, dpath))

    # Load BKT parameters
    dpath = data_config[args.dataset]['dpath']
    # Load keyid2idx to map indices back to original skill names
    with open(os.path.join(dpath, 'keyid2idx.json'), 'r') as f:
        keyid2idx = json.load(f)
    idx2skill = {v: k for k, v in keyid2idx['concepts'].items()}

    bkt_params_path = os.path.join(dpath, 'bkt_skill_params.pkl')
    if not os.path.exists(bkt_params_path):
        alt_path = os.path.join(os.path.dirname(dpath), 'bkt_skill_params.pkl')
        if os.path.exists(alt_path):
            bkt_params_path = alt_path
        else:
            print(f"Error: BKT parameters not found at {bkt_params_path}")
            sys.exit(1)

    with open(bkt_params_path, 'rb') as f:
        bkt_skill_params = pickle.load(f)

    # Initialize BKT model
    bkt_model = Model(seed=42)
    bkt_model.fit_model = {}
    for sid, p in bkt_skill_params['params'].items():
        skill_name = str(sid)
        prior, learns, guesses, slips = p['prior'], p['learns'], p['guesses'], p['slips']
        bkt_model.fit_model[skill_name] = {
            'prior': prior,
            'learns': np.array([learns]),
            'guesses': np.array([guesses]),
            'slips': np.array([slips]),
            'forgets': np.array([0.0]),
            'resource_names': {'default': 0},
            'gs_names': {'default': 0},
            'pi_0': np.array([[1 - prior], [prior]]),
            'As': np.array([[[1 - learns, learns], [0.0, 1.0]]]), 
            'gs': np.array([[[1 - guesses, guesses], [slips, 1 - slips]]])
        }

    # Load Test Set
    print(f"Loading dataset: {args.dataset}, fold: {args.fold}")
    _, valid_loader = init_dataset4train(args.dataset, 'gtransformer', data_config, args.fold, args.batch_size)

    # Collect predictions
    y_trues = []
    y_scores = []
    
    print("Evaluating BKT with Question-Level Late Fusion...")
    
    num_concepts = data_config[args.dataset]['num_c']
    # We need all original skills for the roster
    # Some original skills might not have been in the 'concepts' list if they were only part of composites
    # But FitzModel should have everything.
    unique_original_skills = list(bkt_model.fit_model.keys())
    
    rosters = {}
    
    with torch.no_grad():
        for data in valid_loader:
            c = data["cseqs"].numpy()
            r = data["rseqs"].numpy()
            cshft = data["shft_cseqs"].numpy()
            rshft = data["shft_rseqs"].numpy()
            sm = data["smasks"].numpy()
            uids = data["uids"].numpy()
            
            for b in range(c.shape[0]):
                uid = int(uids[b])
                if uid not in rosters:
                    rosters[uid] = Roster([uid], unique_original_skills, model=bkt_model)
                
                for t in range(c.shape[1]):
                    # 1. Update with observation c[t]
                    # c[t] is an index. We map to original skill(s).
                    idx = int(c[b, t])
                    if idx != -1:
                        skill_str = idx2skill.get(idx, "")
                        # Handle multi-concept questions (split by _)
                        original_skills = skill_str.split("_") if skill_str else []
                        curr_r = int(r[b, t])
                        for s_name in original_skills:
                            if s_name in bkt_model.fit_model:
                                rosters[uid].update_state(s_name, uid, curr_r)
                    
                    # 2. Predict for next step
                    if t < cshft.shape[1] and sm[b, t]:
                        idx_next = int(cshft[b, t])
                        skill_str_next = idx2skill.get(idx_next, "")
                        original_skills_next = skill_str_next.split("_") if skill_str_next else []
                        
                        probs = []
                        for s_name in original_skills_next:
                            if s_name in bkt_model.fit_model:
                                p_mastery = rosters[uid].get_mastery_prob(s_name, uid)
                                params = bkt_skill_params['params'].get(int(s_name) if s_name.isdigit() else s_name, bkt_skill_params['global'])
                                s_rate, g_rate = params['slips'], params['guesses']
                                p_correct = p_mastery * (1 - s_rate) + (1 - p_mastery) * g_rate
                                probs.append(p_correct)
                        
                        if probs:
                            # Question-level fusion: Mean of skill predictions
                            p_final = np.mean(probs)
                        else:
                            p_final = 0.5
                            
                        y_scores.append(p_final)
                        y_trues.append(int(rshft[b, t]))

    # Calculate Metrics
    ts = np.array(y_trues)
    ps = np.array(y_scores)
    auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
    prelabels = [1 if p >= 0.5 else 0 for p in ps]
    acc = metrics.accuracy_score(ts, prelabels)
    
    print(f"\nFinal BKT Evaluation Results (Fold {args.fold}):")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    
    # Save results
    save_dir = os.path.join(dpath, "bkt_eval")
    os.makedirs(save_dir, exist_ok=True)
    results = {"auc": float(auc), "acc": float(acc)}
    with open(os.path.join(save_dir, f"results_fold{args.fold}.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
