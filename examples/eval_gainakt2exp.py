#!/usr/bin/env python3
"""
Evaluation script for GainAKT2Exp - EXPLICIT PARAMETERS (matches train script design).

WARNING: DO NOT CALL DIRECTLY - Use run_repro_experiment_simple.py
This script requires ALL parameters explicitly - zero hardcoded defaults.
"""
import os
import sys
import json
import argparse
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/workspaces/pykt-toolkit')
from pykt.models.gainakt2_exp import create_exp_model
from pykt.datasets import init_dataset4train
from pykt.datasets.data_loader import KTDataset
from examples.experiment_utils import compute_auc_acc


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
            pm = out['projected_mastery']
            pg = out['projected_gains']
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
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        for batch in data_loader:
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            q_shft = batch['shft_cseqs'].to(device)
            r_shft = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device).bool()
            out = core.forward_with_states(q=q, r=r, qry=q_shft)
            logits = out.get('logits')
            if logits is None:
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
    flat_preds = np.concatenate(preds)
    flat_trues = np.concatenate(trues)
    stats = compute_auc_acc(flat_trues, flat_preds)
    return stats['auc'], stats['acc']


def main():
    parser = argparse.ArgumentParser(description='GainAKT2Exp Evaluation - All parameters explicit')
    
    # Required parameters
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--num_encoder_blocks', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--emb_type', type=str, required=True)
    # CRITICAL ARCHITECTURAL FLAGS: These determine model structure and MUST match training
    # Note: action='store_true' means flag absent=False, present=True
    # Launcher (run_repro_experiment.py) explicitly passes these based on config
    # If evaluating manually, MUST specify these flags correctly to match training architecture
    parser.add_argument('--use_mastery_head', action='store_true',
                        help='Enable mastery projection head (REQUIRED for correct model loading)')
    parser.add_argument('--use_gain_head', action='store_true',
                        help='Enable gain projection head (REQUIRED for correct model loading)')
    parser.add_argument('--intrinsic_gain_attention', action='store_true',
                        help='Use intrinsic gain attention mode (changes architecture)')
    parser.add_argument('--use_skill_difficulty', action='store_true',
                        help='Enable learnable per-skill difficulty parameters (Phase 1)')
    parser.add_argument('--use_student_speed', action='store_true',
                        help='Enable learnable per-student learning speed embeddings (Phase 2)')
    parser.add_argument('--num_students', type=int, required=True,
                        help='Number of unique students in dataset (required for student_speed)')
    parser.add_argument('--non_negative_loss_weight', type=float, required=True)
    parser.add_argument('--monotonicity_loss_weight', type=float, required=True)
    parser.add_argument('--mastery_performance_loss_weight', type=float, required=True)
    parser.add_argument('--gain_performance_loss_weight', type=float, required=True)
    parser.add_argument('--sparsity_loss_weight', type=float, required=True)
    parser.add_argument('--consistency_loss_weight', type=float, required=True)
    parser.add_argument('--max_correlation_students', type=int, required=True)
    parser.add_argument('--monitor_freq', type=int, required=True,
                        help='How often to compute interpretability metrics during training')
    
    # Optional
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--experiment_id', type=str)
    
    args = parser.parse_args()
    
    # ARCHITECTURAL CONSTRAINT: Intrinsic gain attention and projection heads are mutually exclusive
    # Intrinsic mode uses attention-derived gains; projection heads would be unused (wasting ~2M parameters)
    if args.intrinsic_gain_attention:
        if args.use_mastery_head or args.use_gain_head:
            print("=" * 100)
            print("⚠️  WARNING: ARCHITECTURAL PARAMETER CONFLICT DETECTED")
            print("=" * 100)
            print("intrinsic_gain_attention=True is INCOMPATIBLE with projection heads")
            print("")
            print("  Intrinsic mode uses attention-derived gains directly from the model.")
            print("  Projection heads (use_mastery_head, use_gain_head) are NOT used in this mode.")
            print("  Enabling them wastes ~2M parameters without any benefit.")
            print("")
            print("AUTOMATIC CORRECTION APPLIED:")
            if args.use_mastery_head:
                print("  • use_mastery_head: True → False")
            if args.use_gain_head:
                print("  • use_gain_head: True → False")
            print("")
            print("Model will be loaded in pure intrinsic mode (attention-derived gains only).")
            print("Expected parameters: ~12.7M (vs ~14.7M with unused projection heads)")
            print("=" * 100)
            args.use_mastery_head = False
            args.use_gain_head = False
    
    # Find checkpoint
    primary_ckpt = os.path.join(args.run_dir, 'model_best.pth')
    fallback_ckpt = os.path.join(args.run_dir, 'best_model.pth')
    if os.path.exists(primary_ckpt):
        ckpt_path = primary_ckpt
    elif os.path.exists(fallback_ckpt):
        ckpt_path = fallback_ckpt
    else:
        raise FileNotFoundError(f'Checkpoint not found in {args.run_dir}')
    
    # Data config
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
            'folds': [0,1,2,3,4],
            'train_valid_file': 'train_valid_sequences.csv',
            'test_file': 'test_sequences.csv',
            'test_window_file': 'test_window_sequences.csv'
        }
    }
    
    # Model config
    model_config = {
        'num_c': data_config[args.dataset]['num_c'],
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'num_encoder_blocks': args.num_encoder_blocks,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'emb_type': args.emb_type,
        'use_mastery_head': args.use_mastery_head,
        'use_gain_head': args.use_gain_head,
        'intrinsic_gain_attention': args.intrinsic_gain_attention,
        'use_skill_difficulty': args.use_skill_difficulty,
        'use_student_speed': args.use_student_speed,
        'num_students': args.num_students,
        'non_negative_loss_weight': args.non_negative_loss_weight,
        'monotonicity_loss_weight': args.monotonicity_loss_weight,
        'mastery_performance_loss_weight': args.mastery_performance_loss_weight,
        'gain_performance_loss_weight': args.gain_performance_loss_weight,
        'sparsity_loss_weight': args.sparsity_loss_weight,
        'consistency_loss_weight': args.consistency_loss_weight,
        'monitor_frequency': args.monitor_freq  # Map monitor_freq -> monitor_frequency for model
    }
    
    # Create and load model
    device = torch.device(args.device)
    model = create_exp_model(model_config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    
    # Load checkpoint
    if isinstance(state, dict):
        candidate = state.get('model_state_dict') or state.get('state_dict') or state
        stripped = {k[7:] if k.startswith('module.') else k: v for k, v in candidate.items()}
        model.load_state_dict(stripped)
    else:
        model.load_state_dict(state)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Initialize loaders
    train_loader, valid_loader = init_dataset4train(args.dataset, 'gainakt2exp', data_config, args.fold, args.batch_size)
    test_cfg = data_config[args.dataset]
    test_dataset = KTDataset(os.path.join(test_cfg['dpath'], test_cfg['test_file']), test_cfg['input_type'], {-1})
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(os.getenv('PYKT_NUM_WORKERS','32')), pin_memory=True)
    
    # Evaluate
    train_auc, train_acc = evaluate_predictions(model, train_loader, device)
    train_mastery_corr, train_gain_corr, train_n = compute_correlations(model, train_loader, device, args.max_correlation_students)
    valid_auc, valid_acc = evaluate_predictions(model, valid_loader, device)
    test_auc, test_acc = evaluate_predictions(model, test_loader, device)
    test_mastery_corr, test_gain_corr, test_n = compute_correlations(model, test_loader, device, args.max_correlation_students)
    
    experiment_id = args.experiment_id or os.path.basename(args.run_dir)
    
    results = {
        'experiment_id': experiment_id,
        'dataset': args.dataset,
        'fold': args.fold,
        'train_auc': float(train_auc),
        'train_acc': float(train_acc),
        'train_mastery_correlation': float(train_mastery_corr),
        'train_gain_correlation': float(train_gain_corr),
        'train_correlation_students': int(train_n),
        'valid_auc': float(valid_auc),
        'valid_acc': float(valid_acc),
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'test_mastery_correlation': float(test_mastery_corr),
        'test_gain_correlation': float(test_gain_corr),
        'test_correlation_students': int(test_n),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    # Save results
    with open(os.path.join(args.run_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    config_eval = {
        'runtime': {'eval_command': ' '.join(sys.argv), 'timestamp': datetime.utcnow().isoformat() + 'Z', 'device': args.device, 'checkpoint_path': ckpt_path},
        'evaluation': {'batch_size': args.batch_size, 'max_correlation_students': args.max_correlation_students},
        'data': {'dataset': args.dataset, 'fold': args.fold},
        'model_config': model_config,
        'experiment': {'id': experiment_id, 'source_run_dir': args.run_dir, 'model': 'gainakt2exp'}
    }
    with open(os.path.join(args.run_dir, 'config_eval.json'), 'w') as f:
        json.dump(config_eval, f, indent=2)
    
    # Save CSV
    with open(os.path.join(args.run_dir, 'metrics_epoch_eval.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['split', 'auc', 'accuracy', 'mastery_correlation', 'gain_correlation', 'correlation_students', 'timestamp'])
        writer.writeheader()
        ts = datetime.utcnow().isoformat()
        writer.writerow({'split': 'training', 'auc': train_auc, 'accuracy': train_acc, 'mastery_correlation': train_mastery_corr, 'gain_correlation': train_gain_corr, 'correlation_students': train_n, 'timestamp': ts})
        writer.writerow({'split': 'validation', 'auc': valid_auc, 'accuracy': valid_acc, 'mastery_correlation': 'N/A', 'gain_correlation': 'N/A', 'correlation_students': 'N/A', 'timestamp': ts})
        writer.writerow({'split': 'test', 'auc': test_auc, 'accuracy': test_acc, 'mastery_correlation': test_mastery_corr, 'gain_correlation': test_gain_corr, 'correlation_students': test_n, 'timestamp': ts})
    
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
