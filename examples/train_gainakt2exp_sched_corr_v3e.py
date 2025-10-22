#!/usr/bin/env python3
"""Scheduled GainAKT2Exp Training v3e (Enhanced Variance Preservation)

Additions over v3d:
1. Hard variance health gate: No interpretability or correlation losses until variance_health_streak >= gate_streak and VarMean >= variance_floor.
2. Adaptive penalty escalation: When in recovery (low variance) linear and square weights multiply by escalation factors.
3. Spread regularizer: Encourages distributional spread (mean absolute deviation) when variance falls below floor.
4. Correlation delay: Requires constraint_scale >= corr_min_scale AND variance healthy.
5. Batch variance diagnostics: Log min/median/max batch variance per epoch and mean gap.
6. Early emergency noise injection: If batch variance below tiny threshold, inject stronger noise and skip correlation for that batch.

Objective: Prevent catastrophic variance collapse prior to scale ≥0.5 and enable sustainable mastery correlation promotion.

Usage example:
  python examples/train_gainakt2exp_sched_corr_v3e.py \
    --dataset assist2015 --epochs 12 --batch_size 96 --lr 1.74e-4 \
    --enable_constraints --constraint_warmup_epochs 3 --constraint_ramp_epochs 8 \
    --variance_floor 0.002 --var_linear_weight 5.0 --var_square_weight 1.0 \
    --escalation_linear_factor 3.0 --escalation_square_factor 5.0 \
    --spread_weight 0.15 --health_gate_streak 2 --corr_min_scale 0.3 \
    --performance_delay_scale 0.35 --mono_delay_scale 0.2 \
    --corr_regularizer_weight 0.30 --corr_weight_low 0.10 --corr_ramp_scale 0.35 \
    --initial_corr_target 0.06 --corr_targets 0.08 0.09 0.10 \
    --corr_start_epoch 5 --target_margin 0.01 --corr_promote_epochs 2 \
    --restoration_noise_std 0.02 --emergency_noise_std 0.05 --variance_gate_factor 0.5 \
    --recovery_patience 2 --collapse_patience 3 \
    --sparsity_ramp_start 0.45 --sparsity_ramp_end 0.85 \
    --union_sampling --global_students 600 --seed 21 --output_suffix v3e_seed21

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

def safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 3 or b.numel() < 3:
        return None
    am = a - a.mean()
    bm = b - b.mean()
    denom = am.std(unbiased=False) * bm.std(unbiased=False) + 1e-6
    if denom < 1e-9:
        return None
    return float((am * bm).mean() / denom)

def collect_valid_sequences(model_core, data_loader, device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rec = []
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
            pm = out['projected_mastery']
            pg = out['projected_gains']
            perf = rsh
            for i in range(pm.size(0)):
                sm = msk[i].bool()
                L = int(sm.sum().item())
                if L < 3:
                    continue
                m_seq = pm[i][sm].mean(dim=1)
                g_seq = pg[i][sm].mean(dim=1)
                p_seq = perf[i][sm].float()
                rec.append((m_seq, g_seq, p_seq))
    return rec

def stratified_indices(records, target, seed):
    if not records or target <= 0:
        return []
    rng = np.random.default_rng(seed)
    lengths = np.array([r[0].numel() for r in records])
    deciles = [np.percentile(lengths, p) for p in range(0,101,10)]
    bins = [[] for _ in range(10)]
    for idx,L in enumerate(lengths):
        b = min(9, int(np.searchsorted(deciles[1:-1], L, side='right')))
        bins[b].append(idx)
    take_per = max(1, target//10)
    chosen = []
    for b in bins:
        if not b:
            continue
        take = min(take_per, len(b))
        chosen.extend(rng.choice(b, size=take, replace=False).tolist())
    return chosen

def compute_global_correlations(records, strat_target, seed, union_sampling=False):
    acc_m, acc_g = [], []
    for m_seq, g_seq, p_seq in records:
        mc = safe_corr(m_seq, p_seq)
        gc = safe_corr(g_seq, p_seq)
        if mc is not None:
            acc_m.append(mc)
        if gc is not None:
            acc_g.append(gc)
    strat_m, strat_g = [], []
    if strat_target>0:
        idxs = stratified_indices(records, strat_target, seed)
        for idx in idxs:
            m_seq,g_seq,p_seq = records[idx]
            mc = safe_corr(m_seq, p_seq)
            gc = safe_corr(g_seq, p_seq)
            if mc is not None:
                strat_m.append(mc)
            if gc is not None:
                strat_g.append(gc)
    final_m = (acc_m+strat_m) if (union_sampling and strat_m) else (strat_m if strat_m else acc_m)
    final_g = (acc_g+strat_g) if (union_sampling and strat_g) else (strat_g if strat_g else acc_g)
    return (float(np.mean(final_m)) if final_m else 0.0,
            float(np.mean(final_g)) if final_g else 0.0,
            {'acc_mastery_count': len(acc_m), 'acc_gain_count': len(acc_g), 'strat_mastery_count': len(strat_m), 'strat_gain_count': len(strat_g), 'final_mastery_count': len(final_m), 'final_gain_count': len(final_g)})

def mastery_corr_regularizer(batch_records, target_corr: float):
    if not batch_records:
        return None
    cs = []
    for m_seq, p_seq in batch_records:
        mc = safe_corr(m_seq, p_seq)
        if mc is not None:
            cs.append(mc)
    if not cs:
        return None
    mean_corr = float(np.mean(cs))
    gap = max(0.0, target_corr - mean_corr)
    return torch.tensor(gap, dtype=torch.float32, device=batch_records[0][0].device)

def compute_constraint_scale(epoch, warmup, ramp):
    if epoch <= warmup:
        return 0.0
    if epoch <= warmup + ramp:
        return (epoch - warmup) / float(ramp)
    return 1.0

def compute_sparsity_subscale(scale, start, end):
    if scale <= start:
        return 0.0
    if scale >= end:
        return 1.0
    return (scale - start) / (end - start)

def correlation_weight_schedule(scale, low, high, ramp_scale):
    if scale <= ramp_scale:
        if ramp_scale <= 0:
            return high
        return low + (high - low) * (scale / ramp_scale)
    return high

def mean_absolute_deviation(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()).abs().mean()

def main():
    ap = argparse.ArgumentParser(description="GainAKT2Exp Scheduled v3e")
    ap.add_argument('--dataset', default='assist2015')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=96)
    ap.add_argument('--lr', type=float, default=1.74e-4)
    ap.add_argument('--weight_decay', type=float, default=1.7571e-05)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--constraint_warmup_epochs', type=int, default=3)
    ap.add_argument('--constraint_ramp_epochs', type=int, default=8)
    ap.add_argument('--enable_constraints', action='store_true')
    ap.add_argument('--performance_delay_scale', type=float, default=0.35)
    ap.add_argument('--mono_delay_scale', type=float, default=0.2)
    ap.add_argument('--staging_scale_threshold', type=float, default=0.5)
    ap.add_argument('--variance_floor', type=float, default=0.002)
    ap.add_argument('--variance_gate_factor', type=float, default=0.5)
    ap.add_argument('--var_linear_weight', type=float, default=5.0)
    ap.add_argument('--var_square_weight', type=float, default=1.0)
    ap.add_argument('--escalation_linear_factor', type=float, default=3.0)
    ap.add_argument('--escalation_square_factor', type=float, default=5.0)
    ap.add_argument('--spread_weight', type=float, default=0.15)
    ap.add_argument('--health_gate_streak', type=int, default=2)
    ap.add_argument('--corr_min_scale', type=float, default=0.3)
    ap.add_argument('--restoration_noise_std', type=float, default=0.02)
    ap.add_argument('--emergency_noise_std', type=float, default=0.05)
    ap.add_argument('--recovery_patience', type=int, default=2)
    ap.add_argument('--collapse_patience', type=int, default=3)
    ap.add_argument('--sparsity_ramp_start', type=float, default=0.45)
    ap.add_argument('--sparsity_ramp_end', type=float, default=0.85)
    ap.add_argument('--corr_regularizer_weight', type=float, default=0.30)
    ap.add_argument('--corr_weight_low', type=float, default=0.10)
    ap.add_argument('--corr_ramp_scale', type=float, default=0.35)
    ap.add_argument('--initial_corr_target', type=float, default=0.06)
    ap.add_argument('--corr_targets', type=float, nargs='+', default=[0.08,0.09,0.10])
    ap.add_argument('--target_margin', type=float, default=0.01)
    ap.add_argument('--corr_promote_epochs', type=int, default=2)
    ap.add_argument('--corr_start_epoch', type=int, default=5)
    ap.add_argument('--union_sampling', action='store_true')
    ap.add_argument('--global_students', type=int, default=600)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output_suffix', default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) else 'cpu')

    base_weights = {
        'non_negative_loss_weight': 0.0,
        'monotonicity_loss_weight': 0.1,
        'mastery_performance_loss_weight': 0.8,
        'gain_performance_loss_weight': 0.8,
        'sparsity_loss_weight': 0.2,
        'consistency_loss_weight': 0.3
    }
    model_config = {
        'num_c':100,'seq_len':200,'d_model':512,'n_heads':8,'num_encoder_blocks':6,'d_ff':1024,'dropout':0.2,'emb_type':'qid','monitor_frequency':50,'use_mastery_head':True,'use_gain_head':True,
        **base_weights
    }
    model = create_exp_model(model_config).to(device)
    model_core = model.module if isinstance(model, torch.nn.DataParallel) else model

    data_config = {'assist2015': {'dpath':'/workspaces/pykt-toolkit/data/assist2015','num_q':0,'num_c':100,'input_type':['concepts'],'max_concepts':1,'min_seq_len':3,'maxlen':200,'emb_path':'','train_valid_original_file':'train_valid.csv','train_valid_file':'train_valid_sequences.csv','folds':[0,1,2,3,4],'test_original_file':'test.csv','test_file':'test_sequences.csv','test_window_file':'test_window_sequences.csv'}}
    train_loader, valid_loader = init_dataset4train(args.dataset, 'gainakt2exp', data_config, args.fold, args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCELoss(reduction='none')

    run_dir = 'paper/results/sched_corr_runs_v3e'
    os.makedirs(run_dir, exist_ok=True)
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    suffix = args.output_suffix or f'v3e_seed{args.seed}'
    with open(os.path.join(run_dir, f'config_{suffix}_{run_id}.json'),'w') as cf:
        json.dump(vars(args), cf, indent=2)

    corr_targets = [args.initial_corr_target] + [t for t in args.corr_targets if t > args.initial_corr_target]
    current_target_index = 0
    current_target = corr_targets[current_target_index]
    promote_counter = 0
    collapse_counter = 0
    low_variance_counter = 0
    variance_health_streak = 0
    recovery_freeze = False
    semantic_traj = []

    for epoch in range(1, args.epochs+1):
        model.train()
        constraint_scale = compute_constraint_scale(epoch, args.constraint_warmup_epochs, args.constraint_ramp_epochs) if args.enable_constraints else 0.0
        epoch_losses = []
        preds_all = []
        targets_all = []
        batch_mastery_variances = []
        interpretability_components_logged = []
        batch_variance_penalties = []
        batch_spread_penalties = []
        batch_var_values = []

        for batch in train_loader:
            q = batch['cseqs'].to(device)
            r = batch['rseqs'].to(device)
            qsh = batch['shft_cseqs'].to(device)
            rsh = batch['shft_rseqs'].to(device)
            msk = batch['masks'].to(device)
            out = model_core.forward_with_states(q=q, r=r, qry=qsh)
            pm = out.get('projected_mastery')
            preds = out['predictions']
            mask_bool = msk.bool()

            main_loss = bce(preds[mask_bool], rsh[mask_bool].float()).mean()
            total_loss = main_loss

            # Compute variance BEFORE any noise injection for diagnostic
            batch_vars = []
            if pm is not None:
                with torch.no_grad():
                    for i in range(pm.size(0)):
                        sm = msk[i].bool()
                        if int(sm.sum().item()) < 3:
                            continue
                        m_seq_local = pm[i][sm].mean(dim=1)
                        batch_vars.append(m_seq_local.var(unbiased=False))
            if batch_vars:
                var_mean_batch = torch.stack(batch_vars).mean()
                batch_var_values.append(float(var_mean_batch.detach().cpu()))
            else:
                var_mean_batch = torch.tensor(0.0, device=device)

            tiny_threshold = args.variance_floor / 25.0
            emergency_mode = (var_mean_batch.item() < tiny_threshold and epoch >= args.corr_start_epoch)
            if emergency_mode and pm is not None and pm.requires_grad:
                with torch.no_grad():
                    pm.add_(torch.randn_like(pm) * args.emergency_noise_std)

            variance_gap_batch = torch.clamp(args.variance_floor - var_mean_batch, min=0.0)
            linear_w = args.var_linear_weight
            square_w = args.var_square_weight
            if variance_gap_batch.item() > 0 and epoch >= args.corr_start_epoch:
                # escalate if in recovery_freeze or emergency
                if emergency_mode or recovery_freeze:
                    linear_w *= args.escalation_linear_factor
                    square_w *= args.escalation_square_factor
                variance_penalty_batch = variance_gap_batch * linear_w + (variance_gap_batch ** 2) * square_w
                total_loss += variance_penalty_batch
                batch_variance_penalties.append(float(variance_penalty_batch.detach().cpu()))
                # spread regularizer
                if pm is not None:
                    spread_terms = []
                    for i in range(pm.size(0)):
                        sm = msk[i].bool()
                        if int(sm.sum().item()) < 3:
                            continue
                        m_seq_local = pm[i][sm].mean(dim=1)
                        spread_terms.append(mean_absolute_deviation(m_seq_local))
                    if spread_terms:
                        spread_val = torch.stack(spread_terms).mean()
                        spread_penalty = (args.variance_floor - var_mean_batch).clamp(min=0.0) * spread_val * args.spread_weight
                        total_loss += spread_penalty
                        batch_spread_penalties.append(float(spread_penalty.detach().cpu()))

            healthy_for_constraints = (variance_health_streak >= args.health_gate_streak and var_mean_batch.item() >= args.variance_floor)

            if args.enable_constraints and constraint_scale > 0 and 'interpretability_loss_breakdown' in out and healthy_for_constraints:
                for k,v in out['interpretability_loss_breakdown'].items():
                    comp_val = float(v.detach().cpu())
                    if ('performance' in k) and (constraint_scale < args.performance_delay_scale):
                        interpretability_components_logged.append((k,comp_val,0.0))
                        continue
                    if 'monotonicity' in k and (constraint_scale < args.mono_delay_scale):
                        interpretability_components_logged.append((k,comp_val,0.0))
                        continue
                    if 'sparsity' in k:
                        sparsity_subscale = compute_sparsity_subscale(constraint_scale, args.sparsity_ramp_start, args.sparsity_ramp_end)
                        total_loss += v * sparsity_subscale
                        interpretability_components_logged.append((k,comp_val,sparsity_subscale))
                        continue
                    if 'consistency' in k and (constraint_scale < args.staging_scale_threshold):
                        interpretability_components_logged.append((k,comp_val,0.0))
                        continue
                    total_loss += v * constraint_scale
                    interpretability_components_logged.append((k,comp_val,constraint_scale))

            # Correlation regularizer gating
            apply_corr = (epoch >= args.corr_start_epoch and constraint_scale >= args.corr_min_scale and healthy_for_constraints and not emergency_mode)
            corr_term = None
            if apply_corr and pm is not None:
                perf = rsh
                batch_records = []
                for i in range(pm.size(0)):
                    sm = msk[i].bool()
                    L = int(sm.sum().item())
                    if L < 3:
                        continue
                    m_seq = pm[i][sm].mean(dim=1)
                    p_seq = perf[i][sm].float()
                    batch_records.append((m_seq, p_seq))
                reg = mastery_corr_regularizer(batch_records, current_target)
                if reg is not None:
                    corr_term = reg * correlation_weight_schedule(constraint_scale, args.corr_weight_low, args.corr_regularizer_weight, args.corr_ramp_scale)
                    total_loss += corr_term

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(float(total_loss.detach().cpu()))
            preds_all.append(preds[mask_bool].detach().cpu().numpy())
            targets_all.append(rsh[mask_bool].detach().cpu().numpy())
            if pm is not None:
                with torch.no_grad():
                    for i in range(pm.size(0)):
                        sm = msk[i].bool()
                        L = int(sm.sum().item())
                        if L < 3:
                            continue
                        m_seq = pm[i][sm].mean(dim=1)
                        var = float(m_seq.var(unbiased=False).cpu())
                        batch_mastery_variances.append(var)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        try:
            train_auc = float(__import__('sklearn.metrics').metrics.roc_auc_score(np.concatenate(targets_all), np.concatenate(preds_all))) if preds_all else 0.0
        except ValueError:
            train_auc = 0.0

        model.eval()
        val_preds = []
        val_targets = []
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

        records = collect_valid_sequences(model_core, valid_loader, device)
        gm, gg, counts = compute_global_correlations(records, args.global_students, args.seed, union_sampling=args.union_sampling)
        mastery_var_mean = float(np.mean(batch_mastery_variances)) if batch_mastery_variances else 0.0
        mastery_var_std = float(np.std(batch_mastery_variances)) if batch_mastery_variances else 0.0
        epoch_variance_penalty_mean = float(np.mean(batch_variance_penalties)) if batch_variance_penalties else 0.0
        epoch_spread_penalty_mean = float(np.mean(batch_spread_penalties)) if batch_spread_penalties else 0.0
        variance_gate = args.variance_floor * args.variance_gate_factor
        batch_var_min = float(np.min(batch_var_values)) if batch_var_values else 0.0
        batch_var_max = float(np.max(batch_var_values)) if batch_var_values else 0.0
        batch_var_median = float(np.median(batch_var_values)) if batch_var_values else 0.0
        variance_gap_epoch = max(0.0, args.variance_floor - mastery_var_mean)

        # Update health streak
        if mastery_var_mean >= args.variance_floor:
            variance_health_streak += 1
        else:
            variance_health_streak = 0

        # low variance tracking for recovery freeze (we still define it but constraints won't apply anyway if unhealthy)
        if mastery_var_mean < variance_gate and epoch >= args.corr_start_epoch:
            low_variance_counter += 1
        else:
            low_variance_counter = 0
        recovery_freeze = (low_variance_counter >= args.recovery_patience)

        if mastery_var_mean < (args.variance_floor/12.0) and epoch >= args.corr_start_epoch:
            collapse_counter += 1
        else:
            collapse_counter = 0
        early_stop = collapse_counter >= args.collapse_patience

        if epoch >= args.corr_start_epoch and gm >= (current_target - args.target_margin) and mastery_var_mean >= args.variance_floor and variance_health_streak >= args.health_gate_streak:
            promote_counter += 1
            if promote_counter >= args.corr_promote_epochs and current_target_index < len(corr_targets)-1:
                current_target_index += 1
                current_target = corr_targets[current_target_index]
                promote_counter = 0
        else:
            promote_counter = 0

        semantic_traj.append({
            'epoch': epoch,'train_loss': train_loss,'train_auc': train_auc,'val_auc': val_auc,
            'global_alignment_mastery_corr': gm,'global_alignment_gain_corr': gg,'constraint_scale': constraint_scale,
            'current_corr_target': current_target,'mastery_variance_mean': mastery_var_mean,'mastery_variance_std': mastery_var_std,
            'variance_penalty_mean': epoch_variance_penalty_mean,'spread_penalty_mean': epoch_spread_penalty_mean,
            'variance_gate': variance_gate,'variance_health_streak': variance_health_streak,'recovery_freeze': recovery_freeze,
            'promote_counter': promote_counter,'collapse_counter': collapse_counter,'early_stop': early_stop,
            'batch_var_min': batch_var_min,'batch_var_median': batch_var_median,'batch_var_max': batch_var_max,'variance_gap_epoch': variance_gap_epoch,
            'interpretability_components': [
                {'name': n,'raw': rv,'scale_applied': sc} for (n,rv,sc) in interpretability_components_logged
            ],'counts': counts
        })

        print(f"[Epoch {epoch:02d}] TL={train_loss:.4f} TAUC={train_auc:.4f} VAUC={val_auc:.4f} MCorr={gm:.4f} VarMean={mastery_var_mean:.5f} VarMed={batch_var_median:.5f} Target={current_target:.3f} Scale={constraint_scale:.2f} VarPen={epoch_variance_penalty_mean:.5f} SpreadPen={epoch_spread_penalty_mean:.5f} HealthStreak={variance_health_streak} CorrActive={(epoch>=args.corr_start_epoch and constraint_scale>=args.corr_min_scale and variance_health_streak>=args.health_gate_streak)}")
        if early_stop:
            print(f"[EarlyStop] Extreme variance collapse for {collapse_counter} epochs. Aborting.")
            break

    traj_path = os.path.join(run_dir, f'semantic_traj_{suffix}_{run_id}.json')
    with open(traj_path,'w') as f:
        json.dump({'trajectory': semantic_traj,'corr_targets': corr_targets}, f, indent=2)
    print(f'[INFO] Saved semantic trajectory to {traj_path}')

if __name__ == '__main__':
    main()
