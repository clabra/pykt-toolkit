#!/usr/bin/env python3
"""
Evaluation script for iDKT interpretability (alignment with BKT reference model).

Note that data/[DATASET]/keyid2idx.json is a bidirectional mapping dictionary that converts between original dataset IDs and zero-based 
sequential indices used internally by the pykt framework
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

from pyBKT.models import Model, Roster
from pykt.models.idkt_roster import IDKTRoster

from pykt.models import init_model
from pykt.datasets import init_dataset4train
from pykt.utils import set_seed

device = "cpu" if not torch.cuda.is_available() else "cuda"

def bkt_step(p_l, y, t, s, g):
    """
    Perform a single BKT mastery update step.
    """
    # Prediction (probability of correct before interaction)
    p_c = p_l * (1 - s) + (1 - p_l) * g
    
    # Bayesian Update: P(L_t | y_t)
    if y == 1:
        p_l_updated = (p_l * (1 - s)) / max(p_c, 1e-10)
    else:
        p_l_updated = (p_l * s) / max(1 - p_c, 1e-10)
    
    # Transition to next state: P(L_{t+1})
    p_l_next = p_l_updated + (1 - p_l_updated) * t
    return np.clip(p_l_next, 0.0, 1.0)

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
    
    parser.add_argument("--roster_sampling_rate", type=int, default=10, help="Sample every N steps for roster export")
    parser.add_argument("--max_correlation_students", type=int, default=1000, help="Limit number of students for correlation export")
    parser.add_argument("--skip_roster", action="store_true", help="Skip expensive roster CSV export")
    
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
    
    # Build Index -> Raw UID Mapping
    ds = valid_loader.dataset
    if hasattr(ds, 'dataset'): ds = ds.dataset # Handle Subset
    uid_to_index = ds.dori['uid_to_index']
    idx_to_uid = {v: k for k, v in uid_to_index.items()}
    print(f"Loaded student ID mapping for {len(idx_to_uid)} students.")

    # Detect n_uid from checkpoint
    print(f"Loading checkpoint from {args.checkpoint} to detect n_uid...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    n_uid = 0
    if 'student_param.weight' in state_dict:
        n_uid = state_dict['student_param.weight'].shape[0] - 1
        print(f"Detected {n_uid} students in checkpoint.")

    # Init Model
    model_config = {
        'd_model': args.d_model, 'd_ff': args.d_ff, 'num_attn_heads': args.n_heads,
        'n_blocks': args.n_blocks, 'dropout': args.dropout, 'final_fc_dim': args.final_fc_dim, 
        'l2': args.l2, 'n_uid': n_uid
    }
    model = init_model('idkt', model_config, data_config[args.dataset], args.emb_type)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_idkt_p = []
    all_bkt_p = []
    all_idkt_initmastery = []
    all_bkt_initmastery = []
    all_idkt_rate = []
    all_bkt_rate = []
    
    # H2: Functional Substitutability
    all_idkt_induced_mastery = []
    all_bkt_reference_mastery = []
    induced_mastery_states = {} # (uid, skill_id) -> current_p_l
    
    # Interaction-level records for heatmap
    param_records = []     # Static Parameters (L0 vs L0)
    traj_records = []      # Dynamic Trajectories (Pred vs Pred)
    rate_records = []      # Dynamic Rates (RT vs RT)
    
    # Roster-style records (Wide table format)
    roster_bkt_records = []
    roster_idkt_records = []
    
    # Limit students to capture representative sample for trajectories
    max_export_students = args.max_correlation_students
    
    # Initialize Rosters
    num_skills = data_config[args.dataset]['num_c'] # Concepts are skills in pykt
    all_skills = list(range(num_skills))
    
    # Initialize BKT Model with saved params (internal structure needed for Roster.py)
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
            # As structure: [resource, [from_state, to_state]]
            # State 0: Unmastered, State 1: Mastered
            # learns is P(U -> M)
            'As': np.array([[[1 - learns, learns], [0.0, 1.0]]]), 
            # gs structure: [gs_class, [state, correctness]]
            # State 0 (U): P(Inc) = 1-g, P(Cor) = g
            # State 1 (M): P(Inc) = s,   P(Cor) = 1-s
            'gs': np.array([[[1 - guesses, guesses], [slips, 1 - slips]]])
        }
    
    # Pre-initialize Rosters with a fixed set of student IDs to avoid KeyError
    # We load the unique UIDs from the augmented CSV and pick the first N
    augmented_csv_path = os.path.join(dpath, orig_file.replace('.csv', '_bkt.csv'))
    df_aug = pd.read_csv(augmented_csv_path)
    all_uids_in_csv = sorted(df_aug['uid'].unique().tolist())
    export_uids = all_uids_in_csv[:max_export_students]
    export_uids_set = set(export_uids)
    
    print(f"Pre-initializing rosters for {len(export_uids)} students...")
    bkt_roster = Roster(export_uids, [str(s) for s in all_skills], model=bkt_model)
    idkt_roster = IDKTRoster(export_uids, all_skills, model, device=device)

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
            # Forward with uid_data
            y, idkt_im_batch, idkt_r_batch, _, concat_q, reg_losses_dict = model(cc_full.long(), cr_full.long(), cq_full.long(), uid_data=uids.to(device), qtest=True)
            
            # Collect aggregated metrics
            # y[:, 1:] corresponds to shifted concepts cshft
            all_idkt_p.extend(torch.masked_select(y[:, 1:], sm).cpu().numpy())
            all_bkt_p.extend(torch.masked_select(bkt_p_batch, sm).cpu().numpy())
            
            # Static BKT Priors for Initial Mastery alignment
            bkt_im_static_batch = torch.zeros_like(cshft).float().to(device)
            for b in range(cshft.shape[0]):
                for s in range(cshft.shape[1]):
                    skill_id = cshft[b, s].item()
                    if skill_id != -1:
                        bkt_im_static_batch[b, s] = bkt_skill_params['params'].get(skill_id, bkt_skill_params['global'])['prior']

            # Use shifted idkt results for masked select
            all_idkt_initmastery.extend(torch.masked_select(idkt_im_batch[:, 1:], sm).cpu().numpy())
            all_bkt_initmastery.extend(torch.masked_select(bkt_im_static_batch, sm).cpu().numpy())
            
            # Reference rates
            ref_rate_batch = torch.zeros_like(cshft).float().to(device)
            for b in range(cshft.shape[0]):
                for s in range(cshft.shape[1]):
                    skill_id = cshft[b, s].item()
                    if skill_id != -1:
                        ref_rate_batch[b, s] = bkt_skill_params['params'].get(skill_id, bkt_skill_params['global'])['learns']
            
            all_idkt_rate.extend(torch.masked_select(idkt_r_batch[:, 1:], sm).cpu().numpy())
            all_bkt_rate.extend(torch.masked_select(ref_rate_batch, sm).cpu().numpy())

            # H2: Calculate Induced Mastery Trajectories
            for b in range(uids.shape[0]):
                uid_idx = uids[b].item()
                raw_uid = idx_to_uid.get(uid_idx, uid_idx) # Get raw student ID (identity mapping now)
                student_mask = sm[b]
                indices = torch.where(student_mask)[0]
                
                for idx in indices:
                    skill_id = int(cshft[b, idx].item())
                    y_true = int(rshft[b, idx].item())
                    idkt_l0 = idkt_im_batch[b, 1+idx].item()
                    idkt_t = idkt_r_batch[b, 1+idx].item()

                    # Update rosters if student is tracked
                    if int(raw_uid) in export_uids_set:
                        # BKT skill params for S/G
                        params = bkt_skill_params['params'].get(skill_id, bkt_skill_params['global'])
                        s_c, g_c = params['slips'], params['guesses']
                        
                        # State tracking for induced trajectory
                        state_key = (raw_uid, skill_id)
                        if state_key not in induced_mastery_states:
                            # Initialize with iDKT's projected initial mastery
                            induced_mastery_states[state_key] = idkt_l0
                        
                        current_m = induced_mastery_states[state_key]
                        all_idkt_induced_mastery.append(current_m)
                        all_bkt_reference_mastery.append(bkt_im_batch[b, idx].item())
                        
                        # Update for next step using BKT formula but iDKT learning rate
                        induced_mastery_states[state_key] = bkt_step(current_m, y_true, idkt_t, s_c, g_c)

            # Export interaction records and Rosters
            for b in range(uids.shape[0]):
                uid_idx = uids[b].item()
                raw_uid = idx_to_uid.get(uid_idx, uid_idx)
                
                student_mask = sm[b]
                indices = torch.where(student_mask)[0]
                
                # 1. Interaction Records (FOR ALL STUDENTS)
                for idx in indices:
                    # Parameter Alignment
                    param_records.append({
                        'student_id': raw_uid,
                        'skill_id': cshft[b, idx].item(),
                        'idkt_im': idkt_im_batch[b, 1+idx].item(),
                        'bkt_im': bkt_im_static_batch[b, idx].item(),
                    })
                    # Trajectory Alignment
                    traj_records.append({
                        'student_id': raw_uid,
                        'skill_id': cshft[b, idx].item(),
                        'y_true': int(rshft[b, idx].item()),
                        'p_idkt': y[b, 1+idx].item(),
                        'y_idkt': 1 if y[b, 1+idx].item() > 0.5 else 0,
                        'p_bkt': bkt_p_batch[b, idx].item(),
                        'y_bkt': 1 if bkt_p_batch[b, idx].item() > 0.5 else 0,
                    })
                    # Rate Alignment
                    rate_records.append({
                        'student_id': raw_uid,
                        'skill_id': cshft[b, idx].item(),
                        'idkt_rate': idkt_r_batch[b, 1+idx].item(),
                        'bkt_rate': ref_rate_batch[b, idx].item(),
                    })

                # 2. Roster Export (ONLY FOR TRACKED STUDENTS)
                if int(raw_uid) in export_uids_set and not args.skip_roster:
                    sampling_rate = args.roster_sampling_rate
                    # Sequential updates for tracking state
                    for step_idx, idx in enumerate(indices):
                        skill_id = int(cshft[b, idx].item())
                        correct = int(rshft[b, idx].item())
                        
                        idkt_roster.update_state(skill_id, raw_uid, correct)
                        bkt_roster.update_state(str(skill_id), raw_uid, correct)
                        
                        if (step_idx + 1) % sampling_rate == 0 or (step_idx == len(indices) - 1):
                            bkt_mastery = {f"S{s}": bkt_roster.get_mastery_prob(str(s), raw_uid) for s in all_skills}
                            roster_bkt_records.append({
                                'student_id': raw_uid,
                                'step': step_idx + 1,
                                'skill_id': skill_id,
                                'correct': correct,
                                **bkt_mastery
                            })

                    idkt_matrix = idkt_roster.get_mastery_matrix(raw_uid)
                    if idkt_matrix is not None:
                        for step_idx, idx in enumerate(indices):
                            if (step_idx + 1) % sampling_rate == 0 or (step_idx == len(indices) - 1):
                                step_mastery = idkt_matrix[step_idx]
                                idkt_mastery_formatted = {f"S{s}": v for s, v in zip(all_skills, step_mastery)}
                                roster_idkt_records.append({
                                    'student_id': raw_uid,
                                    'step': step_idx + 1,
                                    'skill_id': int(cshft[b, idx].item()),
                                    'correct': int(rshft[b, idx].item()),
                                    **idkt_mastery_formatted
                                })

    # Save Record Files
    if param_records:
        df_param = pd.DataFrame(param_records)
        param_path = os.path.join(args.output_dir, "traj_initmastery.csv")
        df_param.to_csv(param_path, index=False)
        print(f"✓ Saved Initial Mastery Alignment: {param_path}")
        
    if traj_records:
        df_traj = pd.DataFrame(traj_records)
        traj_path = os.path.join(args.output_dir, "traj_predictions.csv")
        df_traj.to_csv(traj_path, index=False)
        print(f"✓ Saved Prediction Trajectory Alignment: {traj_path}")

    if rate_records:
        df_rate = pd.DataFrame(rate_records)
        rate_path = os.path.join(args.output_dir, "traj_rate.csv")
        df_rate.to_csv(rate_path, index=False)
        print(f"✓ Saved Learning Rate Alignment: {rate_path}")

    # Save Roster Files
    if roster_bkt_records:
        roster_bkt_path = os.path.join(args.output_dir, "roster_bkt.csv")
        pd.DataFrame(roster_bkt_records).to_csv(roster_bkt_path, index=False)
        print(f"✓ Saved BKT Roster: {roster_bkt_path}")
        
    if roster_idkt_records:
        roster_idkt_path = os.path.join(args.output_dir, "roster_idkt.csv")
        pd.DataFrame(roster_idkt_records).to_csv(roster_idkt_path, index=False)
        print(f"✓ Saved iDKT Roster: {roster_idkt_path}")

    # Calculate Alignment Metrics
    results = {}
    
    def calc_metrics(name, pred, ref):
        mse = np.mean((pred - ref)**2)
        corr, _ = pearsonr(pred, ref)
        return {f"{name}_mse": float(mse), f"{name}_corr": float(corr)}

    results.update(calc_metrics("prediction", np.array(all_idkt_p), np.array(all_bkt_p)))
    results.update(calc_metrics("initmastery", np.array(all_idkt_initmastery), np.array(all_bkt_initmastery)))
    results.update(calc_metrics("learning_rate", np.array(all_idkt_rate), np.array(all_bkt_rate)))

    # H2: Functional Substitutability (Induced vs Reference Mastery)
    print("Calculating Functional Substitutability (H2)...")
    res_h2 = calc_metrics("h2_functional", np.array(all_idkt_induced_mastery), np.array(all_bkt_reference_mastery))
    results["h2_functional_alignment"] = res_h2["h2_functional_corr"]
    
    # H3: Discriminant Validity (Distinctness of Latent Projections)
    print("Calculating Discriminant Validity (H3)...")
    h3_corr, _ = pearsonr(all_idkt_initmastery, all_idkt_rate)
    results["h3_discriminant_overlap"] = float(h3_corr)
    
    if n_uid > 0:
        # Correlation between raw latent scalars (v_s vs k_c)
        vs_weights = model.student_param.weight.detach().cpu().numpy()[1:n_uid+1]
        kc_weights = model.student_gap_param.weight.detach().cpu().numpy()[1:n_uid+1]
        results["h3_latent_overlap"], _ = pearsonr(vs_weights.flatten(), kc_weights.flatten())
        results["h3_latent_overlap"] = float(results["h3_latent_overlap"])

    print("\nAlignment Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "interpretability_alignment.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
