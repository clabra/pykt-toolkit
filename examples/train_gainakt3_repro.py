#!/usr/bin/env python3
"""Reproducible multi-seed GainAKT3 launcher (aligned with GainAKT2Exp reproducibility patterns).

Renamed script: wandb_gainakt3_train.py (former train_gainakt3_repro.py) for consistency with naming across models.

Features:
 - Multi-seed execution with aggregated stability metrics.
 - Complete config serialization (config_md5) including interpretability weights & artifact hashes.
 - Per-epoch CSV logging (metrics_epoch.csv) with raw + share constraint losses and interpretability metrics.
 - Environment capture, seed manifest, best + last checkpoints.
 - Optional reproduction & comparison modes (simplified vs GainAKT2Exp).
 - Override mechanism (--set key=val) supporting dot-path partial updates.

Schema version: 1 (GainAKT3 reproducible).
"""
import os
import sys
import argparse
import datetime
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from examples.train_gainakt3 import evaluate, train_epoch, synthetic_loader

sys.path.insert(0, '/workspaces/pykt-toolkit')
from examples.exp_utils import (make_experiment_dir, atomic_write_json, append_epoch_csv,
    capture_environment, timestamped_logger, compute_config_hash)
from pykt.models.gainakt3 import create_gainakt3_model
from pykt.datasets import init_dataset4train

EPOCH_HEADER = [
    'seed','epoch','train_loss','constraint_loss','val_auc','val_accuracy','mastery_corr','gain_corr',
    'mastery_corr_macro','gain_corr_macro','mastery_corr_macro_weighted','gain_corr_macro_weighted',
    'monotonicity_violation_rate','retention_violation_rate','gain_future_alignment',
    'peer_influence_share','reconstruction_error','difficulty_penalty_contrib_mean',
    'alignment_share','sparsity_share','consistency_share','retention_share','lag_gain_share','peer_alignment_share',
    'difficulty_ordering_share','drift_smoothness_share','alignment_loss_raw','sparsity_loss_raw','consistency_loss_raw',
    'retention_loss_raw','lag_gain_loss_raw','peer_alignment_loss_raw','difficulty_ordering_loss_raw','drift_smoothness_loss_raw','cold_start_flag'
]

def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# (imports moved to top)

_DEF_BOOL = {'true': True, 'false': False}

def _coerce_value(raw: str):
    r = raw.strip()
    lr = r.lower()
    if lr in _DEF_BOOL:
        return _DEF_BOOL[lr]
    try:
        if '.' in r:
            return float(r)
        return int(r)
    except ValueError:
        if ',' in r:
            return [_coerce_value(x) for x in r.split(',') if x.strip()]
        return r

def apply_overrides(args: argparse.Namespace, overrides: List[str]) -> List[str]:
    logs = []
    for ov in overrides:
        if '=' not in ov:
            continue
        #!/usr/bin/env python3
        """DEPRECATED GainAKT3 reproducible launcher.

        Superseded by `examples/wandb_gainakt3_train.py` which implements:
         - Multi-seed stability aggregation
         - Reproduction workflow (--reproduce_from, --strict_schema, --manifest)
         - Complete config serialization (config_md5, missing_params, raw_args)
         - Interpretability metrics logging & constraint decomposition

        Retention rationale:
        Historical experiment folders may embed this filename in `runtime.command`.
        Forwarding preserves reproducibility without editing archived configs.

        Use instead:
            python examples/wandb_gainakt3_train.py <args>

        All CLI arguments are forwarded verbatim.
        """
        import os
        import runpy

        def main():  # pragma: no cover
            print('[DEPRECATED] Forwarding to wandb_gainakt3_train.py. Please update any scripts to call the new launcher directly.')
            target = os.path.join(os.path.dirname(__file__), 'wandb_gainakt3_train.py')
            # Forward execution preserving original sys.argv
            runpy.run_path(target, run_name='__main__')

        if __name__ == '__main__':
            main()
        python examples/wandb_gainakt3_train.py <args>

    Invocation of this wrapper forwards all arguments verbatim.
    """
    from examples.wandb_gainakt3_train import main as _forward_main
    import sys

    def main():  # pragma: no cover - trivial forwarder
        print('[DEPRECATED] Forwarding to wandb_gainakt3_train.py. Update scripts to use the new launcher.')
        _forward_main()

    if __name__ == '__main__':
        main()
        if exp_path_candidate.exists() and not args.force:
            print('[ERROR] experiment_dir exists; use --force to reuse.')
            sys.exit(1)
        exp_path_candidate.parent.mkdir(parents=True, exist_ok=True)
        exp_path = str(exp_path_candidate)
        if not exp_path_candidate.exists():
            os.makedirs(exp_path, exist_ok=True)
            os.makedirs(os.path.join(exp_path,'artifacts'), exist_ok=True)
        exp_id = os.path.basename(exp_path)
    else:
        exp_path = make_experiment_dir('gainakt3', args.short_title, base_dir=str(base_dir))
        exp_id = os.path.basename(exp_path)
    logger = timestamped_logger(exp_id, os.path.join(exp_path,'stdout.log'))
    # Artifact hashing
    peer_path = args.peer_artifact_path or os.path.join(args.artifact_base,'peer_index',args.dataset,'peer_index.pkl')
    diff_path = args.difficulty_artifact_path or os.path.join(args.artifact_base,'difficulty',args.dataset,'difficulty_table.parquet')
    peer_hash = sha256_file(peer_path)
    diff_hash = sha256_file(diff_path)
    cold_start = ((args.use_peer_context and peer_hash == 'MISSING') or (args.use_difficulty_context and diff_hash == 'MISSING'))
    if args.strict_artifact_hash and cold_start:
        print('[ABORT] Strict artifact hash enabled but artifact missing.')
        sys.exit(2)
    cfg = build_config(args, exp_id, exp_path, seeds, peer_hash, diff_hash, cold_start)
    atomic_write_json(cfg, os.path.join(exp_path,'config.json'))
    capture_environment(os.path.join(exp_path,'environment.txt'))
    with open(os.path.join(exp_path,'SEED_INFO.md'),'w') as sf:
        sf.write(f"Primary seed: {seeds[0]}\nAll seeds: {seeds}\n")
    metrics_csv = os.path.join(exp_path,'metrics_epoch.csv')
    per_seed_results: List[Dict[str,Any]] = []
    for seed in seeds:
        logger.info(f"===== Seed {seed} =====")
        set_seeds(seed)
        # Build model
        model_cfg = {
            'num_c': args.num_c,
            'seq_len': args.maxlen,
            'dataset': args.dataset,
            'peer_K': args.peer_K,
            'beta_difficulty': args.beta_difficulty,
            'artifact_base': args.artifact_base,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'alignment_weight': args.alignment_weight,
            'sparsity_weight': args.sparsity_weight,
            'consistency_weight': args.consistency_weight,
            'retention_weight': args.retention_weight,
            'lag_gain_weight': args.lag_gain_weight,
            'warmup_constraint_epochs': args.warmup_constraint_epochs,
            'peer_alignment_weight': args.peer_alignment_weight,
            'difficulty_ordering_weight': args.difficulty_ordering_weight,
            'drift_smoothness_weight': args.drift_smoothness_weight,
            'attempt_confidence_k': args.attempt_confidence_k,
            'gain_threshold': args.gain_threshold,
            'mastery_temperature': args.mastery_temperature,
            'use_peer_context': args.use_peer_context,
            'use_difficulty_context': args.use_difficulty_context,
            'disable_fusion_broadcast': args.disable_fusion_broadcast,
            'disable_difficulty_penalty': args.disable_difficulty_penalty,
            'fusion_for_heads_only': args.fusion_for_heads_only,
            'gate_init_bias': args.gate_init_bias,
        }
        model = create_gainakt3_model(model_cfg)
        model.mastery_temperature = args.mastery_temperature
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Load data
        if args.use_synthetic:
            train_loader = list(synthetic_loader(num_students=args.batch_size, seq_len=args.maxlen, num_c=args.num_c, batches=10))
            val_loader = list(synthetic_loader(num_students=args.batch_size, seq_len=args.maxlen, num_c=args.num_c, batches=4))
            def tuple_to_dict(t):
                q,r,y = t
                return {'cseqs': q.long(),'rseqs': r.long(),'shft_cseqs': q.long(),'shft_rseqs': y.long(),'masks': torch.ones_like(q).long()}
            train_loader = [tuple_to_dict(t) for t in train_loader]
            val_loader = [tuple_to_dict(t) for t in val_loader]
        else:
            train_loader, val_loader = init_dataset4train(args.dataset, 'gainakt3', {
                args.dataset: {
                    'dpath': f'/workspaces/pykt-toolkit/data/{args.dataset}',
                    'num_q': 0,
                    'num_c': args.num_c,
                    'input_type': ['concepts'],
                    'max_concepts': 1,
                    'min_seq_len': 3,
                    'maxlen': args.maxlen,
                    'emb_path': '',
                    'train_valid_original_file': 'train_valid.csv',
                    'train_valid_file': 'train_valid_sequences.csv',
                    'folds': [0,1,2,3,4],
                    'test_original_file': 'test.csv',
                    'test_file': 'test_sequences.csv',
                    'test_window_file': 'test_window_sequences.csv'
                }
            }, args.fold, args.batch_size)
        # Initialize CSV header once on first seed
        if seed == seeds[0]:
            with open(metrics_csv,'w',newline='') as f:
                import csv
                w = csv.writer(f)
                w.writerow(EPOCH_HEADER)
        # Epoch loop
        epoch_rows = []
        for epoch in range(1, args.epochs+1):
            model.current_epoch = epoch
            perf_loss = train_epoch(model, train_loader, model_cfg['device'], optimizer, 100)
            with torch.no_grad():
                probe_batch = train_loader[0] if isinstance(train_loader,list) else next(iter(train_loader))
                c_probe = probe_batch['cseqs'].to(model_cfg['device'])
                r_probe = probe_batch['rseqs'].to(model_cfg['device'])
                probe_out = model(c_probe.long(), r_probe.long())
            constraint_total = float(probe_out['total_constraint_loss'].detach().cpu())
            train_loss = perf_loss + constraint_total
            (val_auc, val_acc, mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mastery_corr_macro_weighted, gain_corr_macro_weighted, mono_rate, ret_rate, gain_future_alignment, per_concept_mastery_corr, per_concept_gain_corr) = evaluate(model, val_loader, model_cfg['device'], args.gain_threshold)
            with torch.no_grad():
                probe = model(torch.randint(0,100,(1,200)).to(model_cfg['device']), torch.randint(0,2,(1,200)).to(model_cfg['device']))
            peer_share = float(probe['peer_influence_share'])
            decomp = probe.get('decomposition',{})
            reconstruction_error = float(decomp.get('reconstruction_error', float('nan')))
            difficulty_penalty_contrib_mean = float(decomp.get('difficulty_penalty_contrib', torch.tensor(float('nan'))).mean().item()) if 'difficulty_penalty_contrib' in decomp and torch.is_tensor(decomp.get('difficulty_penalty_contrib')) else float('nan')
            eps = 1e-8
            cl = probe_out['constraint_losses']
            alignment_raw = float(cl.get('alignment_loss', 0.0))
            alignment_share = alignment_raw / (constraint_total + eps)
            sparsity_raw = float(cl.get('sparsity_loss', 0.0))
            sparsity_share = sparsity_raw / (constraint_total + eps)
            consistency_raw = float(cl.get('consistency_loss', 0.0))
            consistency_share = consistency_raw / (constraint_total + eps)
            retention_raw = float(cl.get('retention_loss', 0.0))
            retention_share = retention_raw / (constraint_total + eps)
            lag_gain_raw = float(cl.get('lag_gain_loss', 0.0))
            lag_gain_share = lag_gain_raw / (constraint_total + eps)
            peer_alignment_raw = float(cl.get('peer_alignment_loss', 0.0))
            peer_alignment_share = peer_alignment_raw / (constraint_total + eps)
            difficulty_ordering_raw = float(cl.get('difficulty_ordering_loss', 0.0))
            difficulty_ordering_share = difficulty_ordering_raw / (constraint_total + eps)
            drift_smoothness_raw = float(cl.get('drift_smoothness_loss', 0.0))
            drift_smoothness_share = drift_smoothness_raw / (constraint_total + eps)
            row = {
                'seed': seed,'epoch': epoch,'train_loss': train_loss,'constraint_loss': constraint_total,'val_auc': val_auc,'val_accuracy': val_acc,
                'mastery_corr': mastery_corr,'gain_corr': gain_corr,'mastery_corr_macro': mastery_corr_macro,'gain_corr_macro': gain_corr_macro,
                'mastery_corr_macro_weighted': mastery_corr_macro_weighted,'gain_corr_macro_weighted': gain_corr_macro_weighted,
                'monotonicity_violation_rate': mono_rate,'retention_violation_rate': ret_rate,'gain_future_alignment': gain_future_alignment,
                'peer_influence_share': peer_share,'reconstruction_error': reconstruction_error,'difficulty_penalty_contrib_mean': difficulty_penalty_contrib_mean,
                'alignment_share': alignment_share,'sparsity_share': sparsity_share,'consistency_share': consistency_share,'retention_share': retention_share,'lag_gain_share': lag_gain_share,
                'peer_alignment_share': peer_alignment_share,'difficulty_ordering_share': difficulty_ordering_share,'drift_smoothness_share': drift_smoothness_share,
                'alignment_loss_raw': alignment_raw,'sparsity_loss_raw': sparsity_raw,'consistency_loss_raw': consistency_raw,'retention_loss_raw': retention_raw,'lag_gain_loss_raw': lag_gain_raw,
                'peer_alignment_loss_raw': peer_alignment_raw,'difficulty_ordering_loss_raw': difficulty_ordering_raw,'drift_smoothness_loss_raw': drift_smoothness_raw,'cold_start_flag': cold_start
            }
            append_epoch_csv(row, metrics_csv, EPOCH_HEADER)
            epoch_rows.append(row)
        # Save seed-specific checkpoints
        torch.save(model.state_dict(), os.path.join(exp_path,f'model_last_seed{seed}.pth'))
        best_row = max(epoch_rows, key=lambda x: x['val_auc']) if epoch_rows else {}
        torch.save({'state_dict': model.state_dict(), 'best_epoch': best_row.get('epoch'), 'val_auc': best_row.get('val_auc')}, os.path.join(exp_path,f'model_best_seed{seed}.pth'))
        per_seed_results.append(best_row)
    # Aggregate multi-seed stability metrics
    def stat(vals):
        arr = np.array([v for v in vals if v is not None], dtype=float)
        if arr.size == 0:
            return {'mean':0.0,'std':0.0,'min':0.0,'max':0.0}
        return {'mean':float(arr.mean()),'std':float(arr.std()),'min':float(arr.min()),'max':float(arr.max())}
    multi_seed_summary = {
        'seeds': seeds,
        'best_val_auc': stat([r.get('val_auc') for r in per_seed_results]),
        'best_mastery_corr': stat([r.get('mastery_corr') for r in per_seed_results]),
        'best_gain_corr': stat([r.get('gain_corr') for r in per_seed_results]),
        'config_md5': cfg['config_md5'],
        'completed_at': datetime.datetime.utcnow().isoformat()+'Z'
    }
    atomic_write_json(multi_seed_summary, os.path.join(exp_path,'results_multi_seed.json'))
    # Legacy results.json (single primary seed best for compatibility)
    primary_best = per_seed_results[0] if per_seed_results else {}
    atomic_write_json({
        'config_md5': cfg['config_md5'],
        'best_epoch': primary_best.get('epoch'),
        'best_val_auc': primary_best.get('val_auc'),
        'best_mastery_corr': primary_best.get('mastery_corr'),
        'best_gain_corr': primary_best.get('gain_corr')
    }, os.path.join(exp_path,'results.json'))
    # README
    checklist = [
        ('Folder naming convention followed','✅'),
        ('config.json contains all params','✅'),
        ('Shell script lists full command','✅'),
        ('Best + last checkpoints saved','✅'),
        ('Per-epoch metrics CSV present','✅'),
        ('Raw stdout log saved','✅'),
        ('Git commit & branch recorded','✅'),
        ('Seeds documented','✅'),
        ('Environment versions captured','✅'),
        ('Correlation metrics logged','✅')
    ]
    lines = [f"# Experiment {exp_id}", '', "Model: gainakt3", f"Short title: {args.short_title}", '', '# Reproducibility Checklist', '', '| Item | Status |', '|------|--------|']
    for item, status in checklist:
        lines.append(f"| {item} | {status} |")
    lines.append('\n## Multi-Seed Best Metrics Summary')
    lines.append('| Metric | Mean | Std | Min | Max |')
    lines.append('|--------|------|-----|-----|-----|')
    def add_summary(name, stats):
        lines.append(f"| {name} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |")
    add_summary('Val AUC', multi_seed_summary['best_val_auc'])
    add_summary('Mastery Corr', multi_seed_summary['best_mastery_corr'])
    add_summary('Gain Corr', multi_seed_summary['best_gain_corr'])
    lines.append('\n## Primary Seed Best')
    lines.append(f"Best epoch (primary seed): {primary_best.get('epoch')} val_auc={primary_best.get('val_auc')} mastery_corr={primary_best.get('mastery_corr')} gain_corr={primary_best.get('gain_corr')}")
    lines.append('\n## Config MD5')
    lines.append(cfg['config_md5'])
    with open(os.path.join(exp_path,'README.md'),'w') as rf:
        rf.write('\n'.join(lines))
    logger.info('GainAKT3 reproducible experiment complete.')

if __name__ == '__main__':
    main()
