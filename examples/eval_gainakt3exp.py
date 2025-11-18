#!/usr/bin/env python3
"""
Evaluation script for GainAKT3Exp - EXPLICIT PARAMETERS (matches train script design).

WARNING: DO NOT CALL DIRECTLY - Use run_repro_experiment_simple.py
This script requires ALL parameters explicitly - zero hardcoded defaults.

CRITICAL ARCHITECTURAL FLAGS:
GainAKT3Exp (2025-11-18): No architectural flags - all features always enabled
  - Mastery trajectories: Always computed (removed use_mastery_head parameter)
  - Per-skill gains: Always computed (removed use_gain_head parameter)
  - Intrinsic gain attention: Removed in favor of dual-encoder architecture
  
This is different from GainAKT2Exp which had optional mastery_head and gain_head.
GainAKT3Exp treats mastery and gains as core features, not optional outputs.
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
from pykt.models.gainakt3_exp import create_exp_model
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
            # CRITICAL FIX (2025-11-18): Pass qry=None to enable mastery head computation
            out = core.forward_with_states(q=q, r=r, qry=None)
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
    """
    Evaluate model predictions from both encoders.
    Returns dict with metrics for combined, encoder1, and encoder2 predictions.
    """
    model.eval()
    preds = []
    trues = []
    preds_encoder1 = []  # Encoder 1: Base predictions
    preds_encoder2 = []  # Encoder 2: Incremental mastery predictions
    with torch.no_grad():
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        for batch in data_loader:
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            q_shft = batch['shft_cseqs'].to(device)
            r_shft = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device).bool()
            # CRITICAL FIX (2025-11-18): Pass qry=None to enable mastery head computation
            out = core.forward_with_states(q=q, r=r, qry=None)
            logits = out.get('logits')
            if logits is None:
                preds_raw = out['predictions']
                eps = 1e-6
                logits = torch.log(preds_raw.clamp(eps, 1 - eps) / (1 - preds_raw.clamp(eps, 1 - eps)))
            for i in range(q.size(0)):
                m = mask[i]
                if m.sum() == 0:
                    continue
                # Encoder 1: Base predictions from logits
                pr_enc1 = torch.sigmoid(logits[i][m]).detach().cpu().numpy()
                gt = r_shft[i][m].float().cpu().numpy()
                
                # Encoder 2: Incremental mastery predictions (if available)
                if 'incremental_mastery_predictions' in out:
                    im_preds = out['incremental_mastery_predictions']
                    pr_enc2 = im_preds[i][m].detach().cpu().numpy()
                    preds_encoder2.append(pr_enc2)
                else:
                    preds_encoder2.append(np.zeros_like(pr_enc1))
                
                preds.append(pr_enc1)  # Legacy: use encoder1 as combined
                preds_encoder1.append(pr_enc1)
                trues.append(gt)
    
    if not preds:
        return {'auc': 0.0, 'accuracy': 0.0, 'encoder1_auc': 0.0, 'encoder1_acc': 0.0, 
                'encoder2_auc': 0.0, 'encoder2_acc': 0.0}
    
    flat_preds = np.concatenate(preds)
    flat_trues = np.concatenate(trues)
    flat_preds_enc1 = np.concatenate(preds_encoder1)
    flat_preds_enc2 = np.concatenate(preds_encoder2)
    
    # Combined stats (legacy, using encoder1)
    stats = compute_auc_acc(flat_trues, flat_preds)
    
    # Encoder 1 stats
    stats_enc1 = compute_auc_acc(flat_trues, flat_preds_enc1)
    
    # Encoder 2 stats (if available)
    if np.any(flat_preds_enc2):
        stats_enc2 = compute_auc_acc(flat_trues, flat_preds_enc2)
    else:
        stats_enc2 = {'auc': 0.0, 'acc': 0.0}
    
    return {
        'auc': stats['auc'],
        'accuracy': stats['acc'],
        'encoder1_auc': stats_enc1['auc'],
        'encoder1_acc': stats_enc1['acc'],
        'encoder2_auc': stats_enc2['auc'],
        'encoder2_acc': stats_enc2['acc']
    }


def main():
    parser = argparse.ArgumentParser(description='GainAKT3Exp Evaluation - All parameters explicit')
    
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
    # DEPRECATED (2025-11-16): Intrinsic gain attention mode removed in GainAKT3Exp
    # parser.add_argument('--intrinsic_gain_attention', action='store_true',
    #                     help='Use intrinsic gain attention mode (changes architecture)')
    parser.add_argument('--use_skill_difficulty', action='store_true',
                        help='Enable learnable per-skill difficulty parameters (Phase 1)')
    parser.add_argument('--use_student_speed', action='store_true',
                        help='Enable learnable per-student learning speed embeddings (Phase 2)')
    parser.add_argument('--num_students', type=int, required=True,
                        help='Number of unique students in dataset (required for student_speed)')
    
    # =====================================================================
    # DEPRECATED PARAMETERS (2025-11-16)
    # These constraint loss parameters are no longer used in GainAKT3Exp.
    # Kept here for reference only - DO NOT use in new experiments.
    # =====================================================================
    # parser.add_argument('--non_negative_loss_weight', type=float, required=True)
    # parser.add_argument('--monotonicity_loss_weight', type=float, required=True)
    # parser.add_argument('--mastery_performance_loss_weight', type=float, required=True)
    # parser.add_argument('--gain_performance_loss_weight', type=float, required=True)
    # parser.add_argument('--sparsity_loss_weight', type=float, required=True)
    # parser.add_argument('--consistency_loss_weight', type=float, required=True)
    # =====================================================================
    # END DEPRECATED PARAMETERS
    # =====================================================================
    
    # Active loss parameters (GainAKT3Exp dual-encoder architecture)
    parser.add_argument('--bce_loss_weight', type=float, required=True, 
                        help='Weight for BCE loss (lambda1). Incremental mastery loss weight = 1 - lambda1')
    parser.add_argument('--mastery_threshold_init', type=float, required=True,
                        help='Initial mastery threshold (θ_global)')
    parser.add_argument('--threshold_temperature', type=float, required=True,
                        help='Temperature for sigmoid threshold functions')
    parser.add_argument('--beta_skill_init', type=float, required=True,
                        help='Initial beta_skill for learning rate amplification')
    parser.add_argument('--m_sat_init', type=float, required=True,
                        help='Initial M_sat for mastery saturation level')
    parser.add_argument('--gamma_student_init', type=float, required=True,
                        help='Initial gamma_student for learning velocity')
    parser.add_argument('--sigmoid_offset', type=float, required=True,
                        help='Sigmoid inflection point offset')
    parser.add_argument('--max_correlation_students', type=int, required=True)
    parser.add_argument('--monitor_freq', type=int, required=True,
                        help='How often to compute interpretability metrics during training')
    
    # Optional
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--experiment_id', type=str)
    
    args = parser.parse_args()
    
    # Removed deprecated intrinsic gain attention validation (2025-11-18)
    
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
        'intrinsic_gain_attention': False,  # DEPRECATED (2025-11-16)
        'use_skill_difficulty': args.use_skill_difficulty,
        'use_student_speed': args.use_student_speed,
        'num_students': args.num_students,
        # DEPRECATED (2025-11-16): Constraint loss weights removed in dual-encoder architecture
        'non_negative_loss_weight': 0.0,
        'monotonicity_loss_weight': 0.0,
        'mastery_performance_loss_weight': 0.0,
        'gain_performance_loss_weight': 0.0,
        'sparsity_loss_weight': 0.0,
        'consistency_loss_weight': 0.0,
        'bce_loss_weight': args.bce_loss_weight,
        'incremental_mastery_loss_weight': 1.0 - args.bce_loss_weight,  # Lambda2 = 1 - Lambda1
        'mastery_threshold_init': args.mastery_threshold_init,
        'threshold_temperature': args.threshold_temperature,
        'beta_skill_init': args.beta_skill_init,
        'm_sat_init': args.m_sat_init,
        'gamma_student_init': args.gamma_student_init,
        'sigmoid_offset': args.sigmoid_offset,
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
    train_loader, valid_loader = init_dataset4train(args.dataset, 'gainakt3exp', data_config, args.fold, args.batch_size)
    test_cfg = data_config[args.dataset]
    test_dataset = KTDataset(os.path.join(test_cfg['dpath'], test_cfg['test_file']), test_cfg['input_type'], {-1})
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(os.getenv('PYKT_NUM_WORKERS','32')), pin_memory=True)
    
    # Evaluate (dual-encoder metrics)
    train_metrics = evaluate_predictions(model, train_loader, device)
    train_mastery_corr, train_gain_corr, train_n = compute_correlations(model, train_loader, device, args.max_correlation_students)
    valid_metrics = evaluate_predictions(model, valid_loader, device)
    test_metrics = evaluate_predictions(model, test_loader, device)
    test_mastery_corr, test_gain_corr, test_n = compute_correlations(model, test_loader, device, args.max_correlation_students)
    
    experiment_id = args.experiment_id or os.path.basename(args.run_dir)
    
    results = {
        'experiment_id': experiment_id,
        'dataset': args.dataset,
        'fold': args.fold,
        # Combined metrics (legacy, using encoder1)
        'train_auc': float(train_metrics['auc']),
        'train_acc': float(train_metrics['accuracy']),
        'valid_auc': float(valid_metrics['auc']),
        'valid_acc': float(valid_metrics['accuracy']),
        'test_auc': float(test_metrics['auc']),
        'test_acc': float(test_metrics['accuracy']),
        # Encoder 1 (Performance Path) metrics
        'train_encoder1_auc': float(train_metrics['encoder1_auc']),
        'train_encoder1_acc': float(train_metrics['encoder1_acc']),
        'valid_encoder1_auc': float(valid_metrics['encoder1_auc']),
        'valid_encoder1_acc': float(valid_metrics['encoder1_acc']),
        'test_encoder1_auc': float(test_metrics['encoder1_auc']),
        'test_encoder1_acc': float(test_metrics['encoder1_acc']),
        # Encoder 2 (Interpretability Path) metrics
        'train_encoder2_auc': float(train_metrics['encoder2_auc']),
        'train_encoder2_acc': float(train_metrics['encoder2_acc']),
        'valid_encoder2_auc': float(valid_metrics['encoder2_auc']),
        'valid_encoder2_acc': float(valid_metrics['encoder2_acc']),
        'test_encoder2_auc': float(test_metrics['encoder2_auc']),
        'test_encoder2_acc': float(test_metrics['encoder2_acc']),
        # Interpretability correlations
        'train_mastery_correlation': float(train_mastery_corr),
        'train_gain_correlation': float(train_gain_corr),
        'train_correlation_students': int(train_n),
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
        'experiment': {'id': experiment_id, 'source_run_dir': args.run_dir, 'model': 'gainakt3exp'}
    }
    with open(os.path.join(args.run_dir, 'config_eval.json'), 'w') as f:
        json.dump(config_eval, f, indent=2)
    
    # Save CSV with dual-encoder metrics
    with open(os.path.join(args.run_dir, 'metrics_epoch_eval.csv'), 'w', newline='') as csvfile:
        # REMOVED (2025-11-16): mastery_correlation, gain_correlation, correlation_students
        # Reason: Loss components and encoder AUCs provide better correlation measures
        fieldnames = ['split', 'auc', 'accuracy', 'encoder1_auc', 'encoder1_acc', 'encoder2_auc', 'encoder2_acc',
                      'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        ts = datetime.utcnow().isoformat()
        writer.writerow({
            'split': 'training',
            'auc': train_metrics['auc'],
            'accuracy': train_metrics['accuracy'],
            'encoder1_auc': train_metrics['encoder1_auc'],
            'encoder1_acc': train_metrics['encoder1_acc'],
            'encoder2_auc': train_metrics['encoder2_auc'],
            'encoder2_acc': train_metrics['encoder2_acc'],
            # REMOVED (2025-11-16): Correlation metrics
            # 'mastery_correlation': train_mastery_corr,
            # 'gain_correlation': train_gain_corr,
            # 'correlation_students': train_n,
            'timestamp': ts
        })
        writer.writerow({
            'split': 'validation',
            'auc': valid_metrics['auc'],
            'accuracy': valid_metrics['accuracy'],
            'encoder1_auc': valid_metrics['encoder1_auc'],
            'encoder1_acc': valid_metrics['encoder1_acc'],
            'encoder2_auc': valid_metrics['encoder2_auc'],
            'encoder2_acc': valid_metrics['encoder2_acc'],
            # REMOVED (2025-11-16): Correlation metrics
            # 'mastery_correlation': 'N/A',
            # 'gain_correlation': 'N/A',
            # 'correlation_students': 'N/A',
            'timestamp': ts
        })
        writer.writerow({
            'split': 'test',
            'auc': test_metrics['auc'],
            'accuracy': test_metrics['accuracy'],
            'encoder1_auc': test_metrics['encoder1_auc'],
            'encoder1_acc': test_metrics['encoder1_acc'],
            'encoder2_auc': test_metrics['encoder2_auc'],
            'encoder2_acc': test_metrics['encoder2_acc'],
            # REMOVED (2025-11-16): Correlation metrics
            # 'mastery_correlation': test_mastery_corr,
            # 'gain_correlation': test_gain_corr,
            # 'correlation_students': test_n,
            'timestamp': ts
        })
    
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
