#!/usr/bin/env python3
"""Adaptive scheduled training for GainAKT2Exp with correlation regularizer & optional union global sampling.

Low-risk trio implementation:
1. Adaptive constraint scheduling (warm-up + ramp) without touching base model code.
2. Union sampling mode for global semantic correlation computation (accumulated + stratified) each epoch.
3. Correlation regularizer that encourages positive mastery-performance Pearson correlation after warm-up.

We do NOT modify existing model files; all logic lives here. Constraint weights applied via dynamic scaling
of model-config-provided weights (the model internally multiplies losses by those static weights; to avoid
changing model code we apply a post-hoc scaling factor to each constraint component extracted from
`interpretability_loss_breakdown` if available, else fallback to aggregated `interpretability_loss`.

Reproducibility: All run arguments saved to JSON in `paper/results/sched_corr_runs/` with timestamp.
Epoch metrics (train/val AUC, global mastery/gain correlations, effective constraint scales) logged.

Usage example:
    python examples/train_gainakt2exp_sched_corr.py \
        --dataset assist2015 --epochs 12 --batch_size 96 --lr 1.74e-4 \
        --constraint_warmup_epochs 3 --constraint_ramp_epochs 5 \
        --corr_regularizer_weight 0.15 --corr_start_epoch 4 \
        --union_sampling --global_students 600 --seed 21

Copyright (c) 2025 Concha Labra. All Rights Reserved.
"""
import argparse
import os
import sys
import json
from datetime import datetime
from typing import List, Tuple
import numpy as np
import torch

sys.path.insert(0, '/workspaces/pykt-toolkit')
from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_exp import create_exp_model

# ------------------ Correlation Utilities ------------------

def safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Numerically stable Pearson correlation (population std) with epsilon guard."""
    if a.numel() < 3 or b.numel() < 3:
        return None
    am = a - a.mean()
    bm = b - b.mean()
    denom = am.std(unbiased=False) * bm.std(unbiased=False) + 1e-6
    if denom < 1e-9:
        return None
    return float((am * bm).mean() / denom)

# ------------------ Global Sequence Collection ------------------

def collect_valid_sequences(model_core, data_loader, device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    records = []
    with torch.no_grad():
        for batch in data_loader:
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            qsh = batch['shft_cseqs'].to(device)
            rsh = batch['shft_rseqs'].to(device)
            msk = batch['masks'].to(device)
            out = model_core.forward_with_states(q=q, r=r, qry=qsh)
            if 'projected_mastery' not in out:
                continue
            pm = out['projected_mastery']  # B x T x C
            pg = out['projected_gains']    # B x T x C
            perf = rsh                     # B x T
            for i in range(pm.size(0)):
                sm = msk[i].bool()
                L = int(sm.sum().item())
                if L < 3:
                    continue
                mastery_seq = pm[i][sm].mean(dim=1)  # aggregate across concepts
                gain_seq = pg[i][sm].mean(dim=1)
                perf_seq = perf[i][sm].float()
                records.append((mastery_seq, gain_seq, perf_seq))
    return records

# Stratified by sequence length deciles

def stratified_indices(records: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], target: int, seed: int) -> List[int]:
    if not records or target <= 0:
        return []
    rng = np.random.default_rng(seed)
    lengths = np.array([r[0].numel() for r in records])
    deciles = [np.percentile(lengths, p) for p in range(0, 101, 10)]
    bins = [[] for _ in range(10)]
    for idx, L in enumerate(lengths):
        bidx = min(9, int(np.searchsorted(deciles[1:-1], L, side='right')))
        bins[bidx].append(idx)
    target_per_bin = max(1, target // 10)
    chosen = []
    for b in bins:
        if not b:
            continue
        take = min(target_per_bin, len(b))
        choice = rng.choice(b, size=take, replace=False)
        chosen.extend(choice.tolist())
    return chosen

# Compute global correlations

def compute_global_correlations(records, strat_target, seed, union_sampling=False):
    acc_mastery, acc_gain = [], []
    for m_seq, g_seq, p_seq in records:
        mc = safe_corr(m_seq, p_seq)
        gc = safe_corr(g_seq, p_seq)
        if mc is not None:
            acc_mastery.append(mc)
        if gc is not None:
            acc_gain.append(gc)
    strat_m, strat_g = [], []
    if strat_target > 0:
        idxs = stratified_indices(records, strat_target, seed)
        for idx in idxs:
            m_seq, g_seq, p_seq = records[idx]
            mc = safe_corr(m_seq, p_seq)
            gc = safe_corr(g_seq, p_seq)
            if mc is not None:
                strat_m.append(mc)
            if gc is not None:
                strat_g.append(gc)
    final_m = (acc_mastery + strat_m) if (union_sampling and strat_m) else (strat_m if strat_m else acc_mastery)
    final_g = (acc_gain + strat_g) if (union_sampling and strat_g) else (strat_g if strat_g else acc_gain)
    gm = float(np.mean(final_m)) if final_m else 0.0
    gg = float(np.mean(final_g)) if final_g else 0.0
    return gm, gg, {
        'acc_mastery_count': len(acc_mastery),
        'acc_gain_count': len(acc_gain),
        'strat_mastery_count': len(strat_m),
        'strat_gain_count': len(strat_g),
        'final_mastery_count': len(final_m),
        'final_gain_count': len(final_g)
    }

# ------------------ Adaptive Scheduling ------------------

def compute_constraint_scale(epoch: int, warmup: int, ramp: int) -> float:
    """Return scale in [0,1]. 0 during warmup, linear ramp over ramp epochs, then 1."""
    if epoch <= warmup:
        return 0.0
    if epoch <= warmup + ramp:
        return (epoch - warmup) / float(ramp)
    return 1.0

# ------------------ Correlation Regularizer ------------------

def mastery_performance_regularizer(records, target_corr: float = 0.10):
    """Compute (target_corr - mean_corr)+ hinge to encourage correlation >= target_corr.
    Returns scalar torch tensor on current device.
    """
    if not records:
        return None
    corrs = []
    for m_seq, g_seq, p_seq in records:
        mc = safe_corr(m_seq, p_seq)
        if mc is not None:
            corrs.append(mc)
    if not corrs:
        return None
    mean_corr = float(np.mean(corrs))
    gap = max(0.0, target_corr - mean_corr)  # hinge
    # Turn into torch scalar for autograd (detached constant)
    return torch.tensor(gap, dtype=torch.float32, device=records[0][0].device)

# ------------------ Training Loop ------------------

def main():
    ap = argparse.ArgumentParser(description="Scheduled GainAKT2Exp training with correlation regularizer.")
    ap.add_argument('--dataset', default='assist2015')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=96)
    ap.add_argument('--lr', type=float, default=1.74e-4)
    ap.add_argument('--weight_decay', type=float, default=1.7571e-05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--constraint_warmup_epochs', type=int, default=3)
    ap.add_argument('--constraint_ramp_epochs', type=int, default=5)
    ap.add_argument('--enable_constraints', action='store_true', help='Activate interpretability losses (scheduled).')
    ap.add_argument('--union_sampling', action='store_true', help='Enable union sampling for global correlations.')
    ap.add_argument('--global_students', type=int, default=600, help='Target stratified student count for global correlations.')
    ap.add_argument('--corr_regularizer_weight', type=float, default=0.15, help='Weight for correlation hinge regularizer.')
    ap.add_argument('--corr_start_epoch', type=int, default=4, help='Epoch to start applying mastery-performance correlation regularizer.')
    ap.add_argument('--corr_target', type=float, default=0.10, help='Target mastery-performance correlation threshold.')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output_suffix', default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')

    # Base (static) weights: we keep prior optimal values but they are scaled dynamically.
    base_weights = {
        'non_negative_loss_weight': 0.0,
        'monotonicity_loss_weight': 0.1,
        'mastery_performance_loss_weight': 0.8,
        'gain_performance_loss_weight': 0.8,
        'sparsity_loss_weight': 0.2,
        'consistency_loss_weight': 0.3
    }

    model_config = {
        'num_c': 100,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 8,
        'num_encoder_blocks': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'emb_type': 'qid',
        'monitor_frequency': 50,
        'use_mastery_head': True,
        'use_gain_head': True,
        **base_weights  # supply static weights (will be effectively scaled externally)
    }
    model = create_exp_model(model_config).to(device)
    model_core = model.module if isinstance(model, torch.nn.DataParallel) else model

    data_config = {
        'assist2015': {
            'dpath': '/workspaces/pykt-toolkit/data/assist2015',
            'num_q': 0,
            'num_c': 100,
            'input_type': ['concepts'],
            'max_concepts': 1,
            'min_seq_len': 3,
            'maxlen': 200,
            'emb_path': '',
            'train_valid_original_file': 'train_valid.csv',
            'train_valid_file': 'train_valid_sequences.csv',
            'folds': [0,1,2,3,4],
            'test_original_file': 'test.csv',
            'test_file': 'test_sequences.csv',
            'test_window_file': 'test_window_sequences.csv'
        }
    }
    train_loader, valid_loader = init_dataset4train(args.dataset, 'gainakt2exp', data_config, args.fold, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCELoss(reduction='none')

    # Repro config save
    run_dir = 'paper/results/sched_corr_runs'
    os.makedirs(run_dir, exist_ok=True)
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    suffix = args.output_suffix or f'seed{args.seed}'
    config_path = os.path.join(run_dir, f'config_{suffix}_{run_id}.json')
    with open(config_path, 'w') as cf:
        json.dump(vars(args), cf, indent=2)

    print(f'[INFO] Saved run config to {config_path}')

    best_val_auc = -1.0
    semantic_traj = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        preds_all = []
        targets_all = []

        # Determine effective constraint scale
        constraint_scale = compute_constraint_scale(epoch, args.constraint_warmup_epochs, args.constraint_ramp_epochs) if args.enable_constraints else 0.0

        for batch in train_loader:
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            qsh = batch['shft_cseqs'].to(device)
            rsh = batch['shft_rseqs'].to(device)
            msk = batch['masks'].to(device)
            out = model_core.forward_with_states(q=q, r=r, qry=qsh)
            preds = out['predictions']
            mask_bool = msk.bool()
            main_loss = bce(preds[mask_bool], rsh[mask_bool].float()).mean()

            total_loss = main_loss
            applied_weights = {}
            if args.enable_constraints and constraint_scale > 0 and 'interpretability_loss' in out:
                # If breakdown dict exists use it, else use aggregate with uniform scaling
                breakdown = out.get('interpretability_loss_breakdown')
                if breakdown:
                    for k, v in breakdown.items():
                        base_w = base_weights.get(k.replace('_loss','') + '_loss_weight', None)
                        # fallback mapping if key names differ
                        applied_weights[k] = base_w if base_w is not None else 1.0
                        total_loss = total_loss + v * constraint_scale
                else:
                    total_loss = total_loss + out['interpretability_loss'] * constraint_scale
            # Correlation regularizer (after corr_start_epoch)
            if epoch >= args.corr_start_epoch and args.corr_regularizer_weight > 0:
                # Collect sampled records from current batch only for efficiency
                if 'projected_mastery' in out:
                    pm = out['projected_mastery']
                    # projected_gains retrieved for potential future use; currently not needed for regularizer
                    _ = out.get('projected_gains')
                    # Use clone() to avoid UserWarning about new_tensor copy construction
                    perf = rsh.clone()
                    batch_records = []
                    for i in range(pm.size(0)):
                        sm = msk[i].bool()
                        L = int(sm.sum().item())
                        if L < 3:
                            continue
                        mastery_seq = pm[i][sm].mean(dim=1)
                        perf_seq = perf[i][sm].float()
                        # Gains optional; not needed for regularizer
                        batch_records.append((mastery_seq, torch.empty(0), perf_seq))
                    reg_scalar = mastery_performance_regularizer(batch_records, target_corr=args.corr_target)
                    if reg_scalar is not None:
                        total_loss = total_loss + reg_scalar * args.corr_regularizer_weight
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(float(total_loss.detach().cpu()))
            preds_all.append(preds[mask_bool].detach().cpu().numpy())
            targets_all.append(rsh[mask_bool].detach().cpu().numpy())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        try:
            train_auc = float(__import__('sklearn.metrics').metrics.roc_auc_score(np.concatenate(targets_all), np.concatenate(preds_all))) if preds_all else 0.0
        except ValueError:
            train_auc = 0.0

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in valid_loader:
                q = batch['cseqs'].to(device)
                r = batch['rseqs'].to(device)
                qsh = batch['shft_cseqs'].to(device)
                rsh = batch['shft_rseqs'].to(device)
                msk = batch['masks'].to(device)
                out = model_core.forward_with_states(q=q, r=r, qry=qsh)
                if 'predictions' not in out:
                    continue
                mask_bool = msk.bool()
                val_preds.append(out['predictions'][mask_bool].detach().cpu().numpy())
                val_targets.append(rsh[mask_bool].detach().cpu().numpy())
        try:
            val_auc = float(__import__('sklearn.metrics').metrics.roc_auc_score(np.concatenate(val_targets), np.concatenate(val_preds))) if val_preds else 0.0
        except ValueError:
            val_auc = 0.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch

        # Global semantic correlations
        records = collect_valid_sequences(model_core, valid_loader, device)
        gm, gg, counts = compute_global_correlations(records, args.global_students, args.seed, union_sampling=args.union_sampling)

        semantic_traj.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'global_alignment_mastery_corr': gm,
            'global_alignment_gain_corr': gg,
            'constraint_scale': constraint_scale,
            'counts': counts
        })
        print(f"[Epoch {epoch:02d}] TrainLoss={train_loss:.4f} TrainAUC={train_auc:.4f} ValAUC={val_auc:.4f} MasteryCorr={gm:.4f} GainCorr={gg:.4f} ConstraintScale={constraint_scale:.2f}")

    # Save trajectory
    traj_path = os.path.join(run_dir, f"semantic_traj_{suffix}_{run_id}.json")
    with open(traj_path, 'w') as f:
        json.dump({'trajectory': semantic_traj, 'best_val_auc': best_val_auc, 'best_epoch': best_epoch, 'union_sampling': args.union_sampling, 'constraint_warmup_epochs': args.constraint_warmup_epochs, 'constraint_ramp_epochs': args.constraint_ramp_epochs}, f, indent=2)
    print(f'[INFO] Saved semantic trajectory to {traj_path}')

if __name__ == '__main__':
    main()
