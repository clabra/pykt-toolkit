#!/usr/bin/env python3
"""
Evaluation script for GainAKT2Exp.

Loads a saved checkpoint (best_model.pth) from a training run directory and evaluates
on the validation and test splits using PyKT dataset initialization patterns.

Usage:
  python examples/eval_gainakt2exp.py \
      --run_dir saved_model/gainakt2exp_align_reproduce_cf4f4017 \
      --dataset assist2015 \
      --batch_size 96 \
      --fold 0 \
      --seq_len 200 \
      --d_model 512 --n_heads 8 --num_encoder_blocks 6 --d_ff 1024 --dropout 0.2 \
      --use_mastery_head --use_gain_head

Outputs a JSON file alongside the checkpoint directory:
  <run_dir>/eval_results.json
Containing: test_auc, test_acc, valid_auc, valid_acc and basic interpretability correlations.

We restrict evaluation to standard predictive metrics plus optional mastery/gain correlations
computed on the test set for up to max_students samples to keep runtime bounded.
"""
import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/workspaces/pykt-toolkit')
from pykt.models.gainakt2_exp import create_exp_model


def compute_correlations(model, data_loader, device, max_students=300):
    model.eval()
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    mastery_corrs = []
    gain_corrs = []
    checked = 0
    with torch.no_grad():
        for batch in data_loader:
            if checked >= max_students:
                break
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            qry = batch.get('shft_cseqs', q).to(device)
            out = core.forward_with_states(q=q, r=r, qry=qry)
            if 'projected_mastery' not in out or 'projected_gains' not in out:
                break
            pm = out['projected_mastery']  # [B, T, C]
            pg = out['projected_gains']    # [B, T, C]
            responses = batch.get('shft_rseqs', r).to(device)
            mask = batch['masks'].to(device)
            B = q.size(0)
            for i in range(B):
                if checked >= max_students:
                    break
                m = mask[i].bool()
                if m.sum() < 3:
                    continue
                mastery_seq = pm[i][m].mean(dim=1).cpu().numpy()
                gain_seq = pg[i][m].mean(dim=1).cpu().numpy()
                perf_seq = responses[i][m].float().cpu().numpy()
                if len(mastery_seq) >= 3:
                    try:
                        mc = np.corrcoef(mastery_seq, perf_seq)[0,1]
                        gc = np.corrcoef(gain_seq, perf_seq)[0,1]
                        if not np.isnan(mc):
                            mastery_corrs.append(mc)
                        if not np.isnan(gc):
                            gain_corrs.append(gc)
                    except Exception:
                        pass
                checked += 1
    mastery_mean = float(np.mean(mastery_corrs)) if mastery_corrs else 0.0
    gain_mean = float(np.mean(gain_corrs)) if gain_corrs else 0.0
    return mastery_mean, gain_mean, len(mastery_corrs)


def evaluate_predictions(model, data_loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in data_loader:
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            out = model(q, r)
            p = out['predictions']  # [B, T]
            mask = batch['masks'].to(device).bool()
            for i in range(q.size(0)):
                m = mask[i]
                pr = p[i][m].cpu().numpy()
                gt = r[i][m].float().cpu().numpy()
                preds.append(pr)
                trues.append(gt)
    if not preds:
        return 0.0, 0.0
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    try:
        auc = roc_auc_score(trues, preds)
    except ValueError:
        auc = 0.0
    acc = float(np.mean((preds >= 0.5) == (trues == 1)))
    return float(auc), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Directory with best_model.pth')
    parser.add_argument('--dataset', type=str, default='assist2015')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size (match training default: 96)')
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_encoder_blocks', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    # Heads enabled by default for interpretability parity with training; disable flags provided.
    parser.add_argument('--use_mastery_head', action='store_true', default=True, help='Enable mastery head (default: enabled)')
    parser.add_argument('--disable_mastery_head', action='store_true', help='Disable mastery head')
    parser.add_argument('--use_gain_head', action='store_true', default=True, help='Enable gain head (default: enabled)')
    parser.add_argument('--disable_gain_head', action='store_true', help='Disable gain head')
    # Constraint weights (match training optimal defaults); evaluation does not apply losses but identical config ensures structural parity.
    parser.add_argument('--non_negative_loss_weight', type=float, default=0.0)
    parser.add_argument('--monotonicity_loss_weight', type=float, default=0.1)
    parser.add_argument('--mastery_performance_loss_weight', type=float, default=0.8)
    parser.add_argument('--gain_performance_loss_weight', type=float, default=0.8)
    parser.add_argument('--sparsity_loss_weight', type=float, default=0.2)
    parser.add_argument('--consistency_loss_weight', type=float, default=0.3)
    parser.add_argument('--max_correlation_students', type=int, default=300, help='Max students sampled for mastery/gain correlation computation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    data_config = {
        args.dataset: {
            'dpath': f'/workspaces/pykt-toolkit/data/{args.dataset}',
            'num_q': 0,
            'num_c': 100,
            'input_type': ['concepts'],
            'max_concepts': 1,
            'min_seq_len': 3,
            'maxlen': args.seq_len,
            'emb_path': '',
            'folds': [0,1,2,3,4],  # injected folds for evaluation consistency
            'train_valid_file': 'train_valid_sequences.csv',
            'test_file': 'test_sequences.csv',
            'test_window_file': 'test_window_sequences.csv'
        }
    }

    # Resolve disable overrides
    if args.disable_mastery_head:
        args.use_mastery_head = False
    if args.disable_gain_head:
        args.use_gain_head = False

    model_config = {
        'num_c': data_config[args.dataset]['num_c'],
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'num_encoder_blocks': args.num_encoder_blocks,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'emb_type': 'qid',
        'use_mastery_head': args.use_mastery_head,
        'use_gain_head': args.use_gain_head,
        # Use provided weights for structural parity (interpretability losses not applied during evaluation forward path)
        'non_negative_loss_weight': args.non_negative_loss_weight,
        'monotonicity_loss_weight': args.monotonicity_loss_weight,
        'mastery_performance_loss_weight': args.mastery_performance_loss_weight,
        'gain_performance_loss_weight': args.gain_performance_loss_weight,
        'sparsity_loss_weight': args.sparsity_loss_weight,
        'consistency_loss_weight': args.consistency_loss_weight
    }

    device = torch.device(args.device)
    model = create_exp_model(model_config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Flexible checkpoint loading: support raw state_dict or training wrapper dict
    if isinstance(state, dict):
        # Extract candidate state_dict
        candidate = None
        if 'model_state_dict' in state:
            candidate = state['model_state_dict']
        elif 'state_dict' in state:
            candidate = state['state_dict']
        # If candidate still None, assume entire dict is a state_dict (may include non-weight metadata)
        if candidate is None:
            # Heuristic: if keys look like weights (contain '.') treat as state_dict
            if any(k.endswith('weight') or k.endswith('bias') for k in state.keys()):
                candidate = state
            else:
                raise RuntimeError(f"Unrecognized checkpoint format keys: {list(state.keys())[:15]}")
        # Strip 'module.' prefix if present (DataParallel)
        stripped_candidate = {}
        for k, v in candidate.items():
            new_k = k[7:] if k.startswith('module.') else k
            stripped_candidate[new_k] = v
        try:
            model.load_state_dict(stripped_candidate)
        except Exception as e:
            missing = [mk for mk in model.state_dict().keys() if mk not in stripped_candidate]
            extra = [ek for ek in stripped_candidate.keys() if ek not in model.state_dict()]
            raise RuntimeError(f"Failed to load checkpoint after prefix strip. Missing: {missing[:10]} Extra: {extra[:10]} Error: {e}")
    else:
        # Not a dict; assume it's a raw state_dict
        model.load_state_dict(state)

    # Wrap for multi-GPU if visible
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Initialize loaders (train_valid for validation); custom lightweight test loader
    from pykt.datasets import init_dataset4train
    train_loader, valid_loader = init_dataset4train(args.dataset, 'gainakt2exp', data_config, args.fold, args.batch_size)
    # Build test loader using KTDataset directly (standard path) to avoid dependence on init_test_datasets signature differences.
    from pykt.datasets.data_loader import KTDataset
    test_cfg = data_config[args.dataset]
    test_dataset = KTDataset(os.path.join(test_cfg['dpath'], test_cfg['test_file']), test_cfg['input_type'], {-1})
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(os.getenv('PYKT_NUM_WORKERS','32')), pin_memory=True)

    valid_auc, valid_acc = evaluate_predictions(model, valid_loader, device)
    test_auc, test_acc = evaluate_predictions(model, test_loader, device)
    mastery_corr, gain_corr, n_students = compute_correlations(model, test_loader, device, max_students=args.max_correlation_students)

    results = {
        'valid_auc': valid_auc,
        'valid_acc': valid_acc,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'test_mastery_correlation': mastery_corr,
        'test_gain_correlation': gain_corr,
        'test_correlation_students': n_students,
        'timestamp': datetime.utcnow().isoformat()
    }

    out_path = os.path.join(args.run_dir, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
