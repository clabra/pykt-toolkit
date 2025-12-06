#!/usr/bin/env python3
"""Minimal evaluation script for iKT2 model - auto-generated

CRITICAL ARCHITECTURAL FLAGS:
None - iKT2 has fixed dual-head architecture (prediction + IRT)
"""
import sys, os, torch, pickle, json
import numpy as np
from scipy.stats import kendalltau, spearmanr
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets.data_loader import KTDataset
from torch.utils.data import DataLoader
from pykt.models.ikt2 import iKT2
from examples.experiment_utils import compute_auc_acc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    project_root = '/workspaces/pykt-toolkit'
    with open(os.path.join(project_root, 'configs/data_config.json')) as f:
        data_config = json.load(f)
    
    dataset_config = data_config[args.dataset]
    num_c = dataset_config['num_c']
    
    # Handle relative paths
    dpath = dataset_config['dpath']
    if dpath.startswith('../'):
        dpath = os.path.join(project_root, dpath[3:])
    
    # Load test data
    test_file = os.path.join(dpath, dataset_config['test_file'])
    test_dataset = KTDataset(test_file, dataset_config['input_type'], {-1})
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = iKT2(
        num_c=num_c, seq_len=args.seq_len, d_model=256, n_heads=4,
        num_encoder_blocks=8, d_ff=1536, dropout=0.2, emb_type='qid',
        lambda_align=1.0, lambda_reg=0.1, phase='eval'
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            qry = batch['shft_cseqs'].to(device)
            labels = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            outputs = model(q, r, qry)
            
            preds = outputs['bce_predictions'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for i in range(len(preds)):
                valid = mask_np[i] == 1
                all_preds.extend(preds[i][valid])
                all_labels.extend(labels_np[i][valid])
    
    metrics = compute_auc_acc(np.array(all_labels), np.array(all_preds))
    
    # Compute L_ref correlation metrics if Rasch targets available
    kendall_corr = None
    spearman_corr = None
    try:
        rasch_file = os.path.join(dpath, 'rasch_targets.pkl')
        if os.path.exists(rasch_file):
            with open(rasch_file, 'rb') as f:
                rasch_data = pickle.load(f)
            
            rasch_lookup = rasch_data['rasch_targets']  # dict[student_id] -> tensor
            
            # Re-evaluate to extract IRT mastery
            model.eval()
            all_irt_mastery = []
            all_rasch_mastery = []
            
            with torch.no_grad():
                for batch in test_loader:
                    q = batch['cseqs'].to(device)
                    r = batch['rseqs'].to(device)
                    qry = batch['shft_cseqs'].to(device)
                    sid = batch['uids'].cpu().numpy()
                    mask = batch['masks'].to(device)
                    
                    outputs = model(q, r, qry)
                    irt_mastery = outputs['mastery_irt'].cpu()  # [B, L] - mastery per timestep
                    
                    # Extract correlations for students with Rasch data
                    for i in range(len(q)):
                        student_id = sid[i].item()
                        if student_id in rasch_lookup:
                            valid_indices = torch.where(mask[i] == 1)[0]
                            rasch_student = rasch_lookup[student_id]  # [L, num_skills] tensor
                            
                            for t_idx in valid_indices:
                                skill_id = q[i, t_idx].item()
                                model_mastery = irt_mastery[i, t_idx].item()
                                
                                # Rasch data is [timestep, skill]
                                if t_idx < rasch_student.shape[0] and skill_id < rasch_student.shape[1]:
                                    rasch_mastery = rasch_student[t_idx, skill_id].item()
                                    
                                    if not (np.isnan(model_mastery) or np.isnan(rasch_mastery)):
                                        all_irt_mastery.append(model_mastery)
                                        all_rasch_mastery.append(rasch_mastery)
            
            if len(all_irt_mastery) > 10:
                # Compute rank correlations
                kendall_corr, _ = kendalltau(all_irt_mastery, all_rasch_mastery)
                spearman_corr, _ = spearmanr(all_irt_mastery, all_rasch_mastery)
                metrics['ref_kendall'] = kendall_corr
                metrics['ref_spear'] = spearman_corr
                
                # Compute L_ref loss (MSE between model IRT and Rasch reference)
                all_irt_np = np.array(all_irt_mastery)
                all_rasch_np = np.array(all_rasch_mastery)
                lref_mse = np.mean((all_irt_np - all_rasch_np) ** 2)
                lref_mae = np.mean(np.abs(all_irt_np - all_rasch_np))
                metrics['lref_mse'] = lref_mse
                metrics['lref_mae'] = lref_mae
    except Exception as e:
        print(f"Warning: Could not compute L_ref correlations: {e}")
    
    print(f"\nTest Results:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  ACC: {metrics['acc']:.4f}")
    if kendall_corr is not None:
        print(f"  L_ref MSE: {metrics['lref_mse']:.4f}")
        print(f"  L_ref MAE: {metrics['lref_mae']:.4f}")
        print(f"  Ref Kendall: {kendall_corr:.4f}")
        print(f"  Ref Spearman: {spearman_corr:.4f}")
    
    # Save results
    if args.output_dir:
        # JSON summary
        result_file = os.path.join(args.output_dir, 'test_results.json')
        result_dict = {'test_auc': metrics['auc'], 'test_acc': metrics['acc']}
        if 'ref_kendall' in metrics:
            result_dict['lref_mse'] = metrics['lref_mse']
            result_dict['lref_mae'] = metrics['lref_mae']
            result_dict['ref_kendall'] = metrics['ref_kendall']
            result_dict['ref_spear'] = metrics['ref_spear']
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nSaved results to {result_file}")

        # CSV for reproducibility standard (metrics_test.csv)
        csv_path = os.path.join(args.output_dir, 'metrics_test.csv')
        try:
            import csv
            fieldnames = ['split', 'auc', 'acc']
            row_data = {'split': 'test', 'auc': f"{metrics['auc']:.6f}", 'acc': f"{metrics['acc']:.6f}"}
            if 'ref_kendall' in metrics:
                fieldnames.extend(['lref_mse', 'lref_mae', 'ref_kendall', 'ref_spear'])
                row_data['lref_mse'] = f"{metrics['lref_mse']:.6f}"
                row_data['lref_mae'] = f"{metrics['lref_mae']:.6f}"
                row_data['ref_kendall'] = f"{metrics['ref_kendall']:.6f}"
                row_data['ref_spear'] = f"{metrics['ref_spear']:.6f}"
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)
            print(f"Saved CSV metrics to {csv_path}")
        except Exception as e:
            print(f"Warning: could not write metrics_test.csv ({e})")

if __name__ == '__main__':
    main()
