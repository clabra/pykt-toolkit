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
    parser.add_argument('--experiment_id', type=str, default=None, help='Explicit experiment id for logging (overrides id derived from run_dir/config).')
    # Architecture flags promoted for audit transparency
    parser.add_argument('--seq_len', type=int, help='Sequence length (must match training model_config)')
    parser.add_argument('--d_model', type=int, help='Model hidden dimension (must match)')
    parser.add_argument('--n_heads', type=int, help='Number of attention heads (must match)')
    parser.add_argument('--num_encoder_blocks', type=int, help='Number of transformer blocks (must match)')
    parser.add_argument('--d_ff', type=int, help='Feed-forward layer dimension (must match)')
    parser.add_argument('--dropout', type=float, help='Dropout probability (must match)')
    parser.add_argument('--emb_type', type=str, help='Embedding type (qid, tag, etc.) (must match)')
    # Performance / interpretability weight flags (for explicit validation; must match stored config if provided)
    parser.add_argument('--batch_size', type=int, help='Evaluation batch size (must match evaluation_defaults if provided)')
    parser.add_argument('--fold', type=int, help='Data split fold (must match)')
    parser.add_argument('--use_mastery_head', action='store_true', help='Expect mastery head enabled')
    parser.add_argument('--use_gain_head', action='store_true', help='Expect gain head enabled')
    parser.add_argument('--non_negative_loss_weight', type=float, help='Must match training config value')
    parser.add_argument('--monotonicity_loss_weight', type=float, help='Must match training config value')
    parser.add_argument('--mastery_performance_loss_weight', type=float, help='Must match training config value')
    parser.add_argument('--gain_performance_loss_weight', type=float, help='Must match training config value')
    parser.add_argument('--sparsity_loss_weight', type=float, help='Must match training config value')
    parser.add_argument('--consistency_loss_weight', type=float, help='Must match training config value')
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
    # Prefer model_config for architecture; fall back to evaluation_defaults only if model_config absent
    model_cfg = cfg.get('model_config')
    arch_source = 'model_config' if model_cfg is not None else 'evaluation_defaults'
    arch_dict = model_cfg if model_cfg is not None else eval_defaults
    mandatory_arch = ['seq_len','d_model','n_heads','num_encoder_blocks','d_ff','dropout','emb_type']
    missing_arch = [k for k in mandatory_arch if k not in arch_dict]
    if missing_arch:
        raise KeyError(f"Missing mandatory architecture keys ({missing_arch}) in {arch_source}; reproduction invalid.")
    # Resolved (baseline) architecture from stored config
    baseline_arch = {k: arch_dict[k] for k in mandatory_arch}
    # Apply CLI overrides ONLY for validation (do not mutate baseline).
    cli_arch = {}
    for k in mandatory_arch:
        v = getattr(args, k, None)
        if v is not None:
            cli_arch[k] = v
    # Validation guard: if user supplied any CLI arch flags, all supplied must match baseline_arch
    arch_drift = {}
    for k, v in cli_arch.items():
        if str(v) != str(baseline_arch[k]):
            arch_drift[k] = {'expected': baseline_arch[k], 'provided': v}
    if arch_drift:
        raise ValueError(f"Architecture drift detected for keys: {json.dumps(arch_drift)}. Evaluation aborted to preserve reproducibility.")
    # Final resolved architecture values
    seq_len = baseline_arch['seq_len']
    d_model = baseline_arch['d_model']
    n_heads = baseline_arch['n_heads']
    num_encoder_blocks = baseline_arch['num_encoder_blocks']
    d_ff = baseline_arch['d_ff']
    dropout = baseline_arch['dropout']
    emb_type = baseline_arch['emb_type']

    # Resolve baseline performance/interpretability weights from config before validation
    use_mastery_head = bool(cfg.get('interpretability', {}).get('use_mastery_head', eval_defaults.get('use_mastery_head', True)))
    use_gain_head = bool(cfg.get('interpretability', {}).get('use_gain_head', eval_defaults.get('use_gain_head', True)))
    non_negative_loss_weight = eval_defaults.get('non_negative_loss_weight', 0.0)
    monotonicity_loss_weight = eval_defaults.get('monotonicity_loss_weight', 0.1)
    mastery_performance_loss_weight = eval_defaults.get('mastery_performance_loss_weight', 0.8)
    gain_performance_loss_weight = eval_defaults.get('gain_performance_loss_weight', 0.8)
    sparsity_loss_weight = eval_defaults.get('sparsity_loss_weight', 0.2)
    consistency_loss_weight = eval_defaults.get('consistency_loss_weight', 0.3)
    # Validate performance/interpretability weights if provided via CLI (after baseline resolution)
    perf_keys = [
        'batch_size','fold','use_mastery_head','use_gain_head','non_negative_loss_weight','monotonicity_loss_weight',
        'mastery_performance_loss_weight','gain_performance_loss_weight','sparsity_loss_weight','consistency_loss_weight'
    ]
    baseline_perf = {
        'batch_size': eval_defaults.get('batch_size', 96),
        'fold': fold,
        'use_mastery_head': use_mastery_head,
        'use_gain_head': use_gain_head,
        'non_negative_loss_weight': non_negative_loss_weight,
        'monotonicity_loss_weight': monotonicity_loss_weight,
        'mastery_performance_loss_weight': mastery_performance_loss_weight,
        'gain_performance_loss_weight': gain_performance_loss_weight,
        'sparsity_loss_weight': sparsity_loss_weight,
        'consistency_loss_weight': consistency_loss_weight
    }
    perf_drift = {}
    for k in perf_keys:
        cli_val = getattr(args, k, None)
        # For store_true flags, argparse sets False when absent; treat absence as None to avoid false drift.
        if k in {'use_mastery_head','use_gain_head'}:
            # Determine if flag explicitly present in sys.argv
            flag_present = any(arg == f"--{k}" for arg in sys.argv[1:])
            if not flag_present:
                continue  # skip validation for unsupplied flag
        if cli_val is None:
            continue
        provided = cli_val
        expected = baseline_perf[k]
        if str(provided) != str(expected):
            perf_drift[k] = {'expected': expected, 'provided': provided}
    if perf_drift:
        raise ValueError(f"Performance/interpretability weight drift detected: {json.dumps(perf_drift)}")
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

    # Assemble model_config (interpretability weights retained for parity; not all used in eval forward path)
    model_config = {
        'num_c': data_config[dataset]['num_c'],
        'seq_len': seq_len,
        'd_model': d_model,
        'n_heads': n_heads,
        'num_encoder_blocks': num_encoder_blocks,
        'd_ff': d_ff,
        'dropout': dropout,
        'emb_type': emb_type,
        'use_mastery_head': use_mastery_head,
        'use_gain_head': use_gain_head,
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

    # Compute predictions for all splits
    train_auc, train_acc = evaluate_predictions(model, train_loader, device)
    valid_auc, valid_acc = evaluate_predictions(model, valid_loader, device)
    test_auc, test_acc = evaluate_predictions(model, test_loader, device)
    
    # Compute correlations for train and test (not validation - avoid overfitting to interpretability)
    train_mastery_corr, train_gain_corr, train_n_students = compute_correlations(model, train_loader, device, max_students=max_corr_students)
    test_mastery_corr, test_gain_corr, test_n_students = compute_correlations(model, test_loader, device, max_students=max_corr_students)

    experiment_id = args.experiment_id or cfg.get('experiment', {}).get('id') or os.path.basename(args.run_dir)
    
    # Build config_eval.json mirroring config.json structure
    config_eval = {
        'runtime': {
            'eval_command': ' '.join(sys.argv),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'device': args.device,
            'checkpoint_path': ckpt_path
        },
        'evaluation': {
            'batch_size': batch_size,
            'max_correlation_students': max_corr_students
        },
        'data': {
            'dataset': dataset,
            'fold': fold
        },
        'model_config': {
            'seq_len': seq_len,
            'd_model': d_model,
            'n_heads': n_heads,
            'num_encoder_blocks': num_encoder_blocks,
            'd_ff': d_ff,
            'dropout': dropout,
            'emb_type': emb_type,
            'use_mastery_head': use_mastery_head,
            'use_gain_head': use_gain_head,
            'non_negative_loss_weight': non_negative_loss_weight,
            'monotonicity_loss_weight': monotonicity_loss_weight,
            'mastery_performance_loss_weight': mastery_performance_loss_weight,
            'gain_performance_loss_weight': gain_performance_loss_weight,
            'sparsity_loss_weight': sparsity_loss_weight,
            'consistency_loss_weight': consistency_loss_weight
        },
        'experiment': {
            'id': experiment_id,
            'source_run_dir': args.run_dir,
            'model': 'gainakt2exp'
        },
        'architecture_validation': {
            'baseline_architecture': baseline_arch,
            'cli_overrides_provided': cli_arch,
            'drift_detected': bool(arch_drift),
            'drift_details': arch_drift if arch_drift else {}
        },
        'config_md5_eval': None  # Will compute if needed
    }
    
    results = {
        'experiment_id': experiment_id,
        'dataset': dataset,
        'fold': fold,
        'train_auc': train_auc,
        'train_acc': train_acc,
        'train_mastery_correlation': train_mastery_corr,
        'train_gain_correlation': train_gain_corr,
        'train_correlation_students': train_n_students,
        'valid_auc': valid_auc,
        'valid_acc': valid_acc,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'test_mastery_correlation': test_mastery_corr,
        'test_gain_correlation': test_gain_corr,
        'test_correlation_students': test_n_students,
        'timestamp': datetime.utcnow().isoformat()
    }

    # Save eval_results.json (legacy output)
    out_path = os.path.join(args.run_dir, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save config_eval.json
    config_eval_path = os.path.join(args.run_dir, 'config_eval.json')
    with open(config_eval_path, 'w') as f:
        json.dump(config_eval, f, indent=2)
    
    # Save metrics_epoch_eval.csv (evaluation metrics with correlations)
    import csv
    metrics_csv_path = os.path.join(args.run_dir, 'metrics_epoch_eval.csv')
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'split', 'auc', 'accuracy', 
            'mastery_correlation', 'gain_correlation', 
            'correlation_students', 'timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Write training metrics (with correlations)
        writer.writerow({
            'split': 'training',
            'auc': train_auc,
            'accuracy': train_acc,
            'mastery_correlation': train_mastery_corr,
            'gain_correlation': train_gain_corr,
            'correlation_students': train_n_students,
            'timestamp': datetime.utcnow().isoformat()
        })
        # Write validation metrics (AUC/accuracy only, correlations set to N/A to avoid overfitting)
        writer.writerow({
            'split': 'validation',
            'auc': valid_auc,
            'accuracy': valid_acc,
            'mastery_correlation': 'N/A',
            'gain_correlation': 'N/A',
            'correlation_students': 'N/A',
            'timestamp': datetime.utcnow().isoformat()
        })
        # Write test/evaluation metrics (with correlations - primary evaluation results)
        writer.writerow({
            'split': 'test',
            'auc': test_auc,
            'accuracy': test_acc,
            'mastery_correlation': test_mastery_corr,
            'gain_correlation': test_gain_corr,
            'correlation_students': test_n_students,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    print(json.dumps(results, indent=2))
    print(f"\n✅ Saved config_eval.json to {config_eval_path}")
    print(f"✅ Saved metrics_epoch_eval.csv to {metrics_csv_path}")

if __name__ == '__main__':
    main()
