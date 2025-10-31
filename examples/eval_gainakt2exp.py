#!/usr/bin/env python3
"""
Evaluation script for GainAKT2Exp (UPDATED FOR NEXT-STEP METRICS).

CHANGE LOG (2025-10-30):
    - Previous version computed SAME-TIME metrics (predictions vs r_t) which are
        inflated because the interaction embedding includes r_t (label leakage).
    - This updated version now computes NEXT-STEP predictive metrics by aligning
        logits from forward_with_states(q, r, qry=shft_cseqs) with shifted targets
        shft_rseqs under the mask. The outputs valid_auc / test_auc now reflect
        unbiased next-step performance consistent with the training objective.
    - SAME-TIME evaluation has been removed from primary metric computation to
        avoid accidental leakage reporting. To quantify leakage, use the separate
        next-step diagnostic script (examples/eval_gainakt2exp_nextstep.py) or the
        integrated post-training helper (tmp/shifted_eval_helper.py).

RATIONALE:
    In GainAKT2Exp, interaction_tokens = q_t + num_c * r_t embeds the current
    response. Evaluating on r_t directly gives the model access to the label it
    is asked to predict. Next-step alignment (predict r_{t+1}) mirrors the
    training loss and produces defensible AUC/ACC.

PRIMARY OUTPUTS (next-step): valid_auc, valid_acc, test_auc, test_acc, plus
correlation metrics using shifted performance sequences.

Usage:
    python examples/eval_gainakt2exp.py \
            --run_dir <experiment_dir> --dataset assist2015 \
            --batch_size 96 --fold 0 --seq_len 200 \
            --d_model 512 --n_heads 8 --num_encoder_blocks 6 --d_ff 1024 --dropout 0.2 \
            --use_mastery_head --use_gain_head

Outputs a JSON file: <run_dir>/eval_results.json containing:
    valid_auc, valid_acc, test_auc, test_acc, mastery/gain correlations.
"""
import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from examples.experiment_utils import compute_auc_acc

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
    """Next-step evaluation: predict r_{t+1} using context up to t."""
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        for batch in data_loader:
            q = batch['cseqs'].to(device)          # current concepts (t)
            r = batch['rseqs'].to(device)          # current responses (t)
            q_shft = batch['shft_cseqs'].to(device)  # next concepts (t+1)
            r_shft = batch['shft_rseqs'].to(device)  # next responses (t+1)
            mask = batch['masks'].to(device).bool()  # valid interaction mask
            out = core.forward_with_states(q=q, r=r, qry=q_shft)
            logits = out.get('logits')
            if logits is None:
                # Fallback: invert predictions safely if logits missing
                preds_raw = out['predictions']
                eps = 1e-6
                logits = torch.log(preds_raw.clamp(eps, 1 - eps) / (1 - preds_raw.clamp(eps, 1 - eps)))
            for i in range(q.size(0)):
                m = mask[i]
                if m.sum() == 0:
                    continue
                pr = torch.sigmoid(logits[i][m]).detach().cpu().numpy()
                gt = r_shft[i][m].float().cpu().numpy()
                preds.append(pr)
                trues.append(gt)
    if not preds:
        return 0.0, 0.0
    import numpy as _np
    flat_preds = _np.concatenate(preds)
    flat_trues = _np.concatenate(trues)
    stats = compute_auc_acc(flat_trues, flat_preds)
    return stats['auc'], stats['acc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Experiment directory containing config.json and model_best.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--override', action='append', default=[], help='Optional key=value overrides for evaluation_defaults')
    parser.add_argument('--max_correlation_students', type=int, default=None, help='Override max students for correlation if desired')
    parser.add_argument('--dataset', type=str, help='Explicit dataset name (overrides config/eval defaults)')
    args = parser.parse_args()

    # Load experiment config.json
    cfg_path = os.path.join(args.run_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'config.json not found in run_dir: {cfg_path}')
    with open(cfg_path) as f:
        cfg = json.load(f)
    eval_defaults = cfg.get('evaluation_defaults') or cfg.get('evaluation_snapshot') or {}
    # Apply overrides from CLI
    for ov in args.override:
        if '=' not in ov:
            raise ValueError(f"Override must be key=value: {ov}")
        k, v = ov.split('=', 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ('true','false'):
            v_cast = v.lower() == 'true'
        else:
            try:
                v_cast = int(v) if v.isdigit() else float(v)
            except ValueError:
                v_cast = v
        eval_defaults[k] = v_cast

    dataset = args.dataset or cfg.get('data', {}).get('dataset', eval_defaults.get('dataset', 'assist2015'))
    fold = cfg.get('data', {}).get('fold', eval_defaults.get('fold', 0))
    batch_size = eval_defaults.get('batch_size', 96)
    seq_len = eval_defaults.get('seq_len', 200)
    d_model = eval_defaults.get('d_model', 512)
    n_heads = eval_defaults.get('n_heads', 8)
    num_encoder_blocks = eval_defaults.get('num_encoder_blocks', 6)
    d_ff = eval_defaults.get('d_ff', 1024)
    dropout = eval_defaults.get('dropout', 0.2)
    use_mastery_head = bool(cfg.get('interpretability', {}).get('use_mastery_head', eval_defaults.get('use_mastery_head', True)))
    use_gain_head = bool(cfg.get('interpretability', {}).get('use_gain_head', eval_defaults.get('use_gain_head', True)))
    non_negative_loss_weight = eval_defaults.get('non_negative_loss_weight', 0.0)
    monotonicity_loss_weight = eval_defaults.get('monotonicity_loss_weight', 0.1)
    mastery_performance_loss_weight = eval_defaults.get('mastery_performance_loss_weight', 0.8)
    gain_performance_loss_weight = eval_defaults.get('gain_performance_loss_weight', 0.8)
    sparsity_loss_weight = eval_defaults.get('sparsity_loss_weight', 0.2)
    consistency_loss_weight = eval_defaults.get('consistency_loss_weight', 0.3)
    max_corr_students = args.max_correlation_students or eval_defaults.get('max_correlation_students', 300)

    # Primary checkpoint name produced by training is model_best.pth; fallback to legacy best_model.pth
    primary_ckpt = os.path.join(args.run_dir, 'model_best.pth')
    fallback_ckpt = os.path.join(args.run_dir, 'best_model.pth')
    if os.path.exists(primary_ckpt):
        ckpt_path = primary_ckpt
    elif os.path.exists(fallback_ckpt):
        ckpt_path = fallback_ckpt
    else:
        raise FileNotFoundError(f'Checkpoint not found (searched {primary_ckpt} and {fallback_ckpt})')

    data_config = {
        dataset: {
            'dpath': f'/workspaces/pykt-toolkit/data/{dataset}',
            'num_q': 0,
            'num_c': 100,
            'input_type': ['concepts'],
            'max_concepts': 1,
            'min_seq_len': 3,
            'maxlen': seq_len,
            'emb_path': '',
            'folds': [0,1,2,3,4],  # injected folds for evaluation consistency
            'train_valid_file': 'train_valid_sequences.csv',
            'test_file': 'test_sequences.csv',
            'test_window_file': 'test_window_sequences.csv'
        }
    }

    # Resolve disable overrides
    # (Disable flags via config not needed; use_mastery_head/gain already resolved)

    model_config = {
        'num_c': data_config[dataset]['num_c'],
        'seq_len': seq_len,
        'd_model': d_model,
        'n_heads': n_heads,
        'num_encoder_blocks': num_encoder_blocks,
        'd_ff': d_ff,
        'dropout': dropout,
        'emb_type': 'qid',
        'use_mastery_head': use_mastery_head,
        'use_gain_head': use_gain_head,
        # Use provided weights for structural parity (interpretability losses not applied during evaluation forward path)
        'non_negative_loss_weight': non_negative_loss_weight,
        'monotonicity_loss_weight': monotonicity_loss_weight,
        'mastery_performance_loss_weight': mastery_performance_loss_weight,
        'gain_performance_loss_weight': gain_performance_loss_weight,
        'sparsity_loss_weight': sparsity_loss_weight,
        'consistency_loss_weight': consistency_loss_weight
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
    train_loader, valid_loader = init_dataset4train(dataset, 'gainakt2exp', data_config, fold, batch_size)
    # Build test loader using KTDataset directly (standard path) to avoid dependence on init_test_datasets signature differences.
    from pykt.datasets.data_loader import KTDataset
    test_cfg = data_config[dataset]
    test_dataset = KTDataset(os.path.join(test_cfg['dpath'], test_cfg['test_file']), test_cfg['input_type'], {-1})
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(os.getenv('PYKT_NUM_WORKERS','32')), pin_memory=True)

    valid_auc, valid_acc = evaluate_predictions(model, valid_loader, device)
    test_auc, test_acc = evaluate_predictions(model, test_loader, device)
    mastery_corr, gain_corr, n_students = compute_correlations(model, test_loader, device, max_students=max_corr_students)

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
