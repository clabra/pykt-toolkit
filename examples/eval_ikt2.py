#!/usr/bin/env python3
"""Minimal evaluation script for iKT2 model - auto-generated"""
import sys, os, torch, pickle, json
import numpy as np
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
        lambda_align=1.0, lambda_reg=0.1
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
    print(f"\nTest Results:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  ACC: {metrics['acc']:.4f}")
    
    # Save results
    if args.output_dir:
        # JSON summary
        result_file = os.path.join(args.output_dir, 'test_results.json')
        with open(result_file, 'w') as f:
            json.dump({'test_auc': metrics['auc'], 'test_acc': metrics['acc']}, f, indent=2)
        print(f"\nSaved results to {result_file}")

        # CSV for reproducibility standard (metrics_test.csv)
        csv_path = os.path.join(args.output_dir, 'metrics_test.csv')
        try:
            import csv
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=['split', 'auc', 'acc'])
                writer.writeheader()
                writer.writerow({'split': 'test', 'auc': f"{metrics['auc']:.6f}", 'acc': f"{metrics['acc']:.6f}"})
            print(f"Saved CSV metrics to {csv_path}")
        except Exception as e:
            print(f"Warning: could not write metrics_test.csv ({e})")

if __name__ == '__main__':
    main()
