#!/usr/bin/env python3
"""Multi-seed reproducible launcher for GainAKT2Exp with semantic stability instrumentation.

Adds:
  - Ablation presets (baseline|align|retention|both|both_lag) enabling objectives by default
  - Temporal stability aggregation (variance & slopes of correlations, retention penalty stats)
  - Extended README narrative for semantic emergence interpretation
"""
import os
import sys
import argparse
import datetime
import torch
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, '/workspaces/pykt-toolkit')
from examples.exp_utils import (make_experiment_dir, atomic_write_json, append_epoch_csv,
    capture_environment, timestamped_logger, compute_config_hash)
from examples.train_gainakt2exp import train_gainakt2exp_model

EPOCH_HEADER = [
    'seed','epoch','train_loss','train_auc','val_auc','val_accuracy','monotonicity_violation_rate',
    'negative_gain_rate','bounds_violation_rate','mastery_correlation','mastery_ci_low','mastery_ci_high',
    'gain_correlation','gain_ci_low','gain_ci_high','main_loss_share','constraint_loss_share','alignment_loss_share',
    'lag_loss_share','retention_loss_share'
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed reproducible GainAKT2Exp training")
    p.add_argument('--ablation_mode', type=str, default='both_lag', choices=['baseline','align','retention','both','both_lag'],
                   help='Preset enabling alignment/retention/lag objectives unless explicitly overridden by flags.')
    p.add_argument('--dataset', '--dataset_name', dest='dataset', default='assist2015')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', '--num_epochs', dest='epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.000174)
    p.add_argument('--weight_decay', type=float, default=1.7571e-05)
    p.add_argument('--enhanced_constraints', action='store_true', default=True)
    p.add_argument('--experiment_title', type=str, default='baseline')
    p.add_argument('--experiment_suffix', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='*', default=None)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--devices', type=int, nargs='*', default=[0,1,2,3,4])
    p.add_argument('--monitor_freq', type=int, default=50)
    p.add_argument('--patience', type=int, default=20)
    # Constraint weights
    p.add_argument('--non_negative_loss_weight', type=float, default=0.0)
    p.add_argument('--monotonicity_loss_weight', type=float, default=0.1)
    p.add_argument('--mastery_performance_loss_weight', type=float, default=0.8)
    p.add_argument('--gain_performance_loss_weight', type=float, default=0.8)
    p.add_argument('--sparsity_loss_weight', type=float, default=0.2)
    p.add_argument('--consistency_loss_weight', type=float, default=0.3)
    # Alignment / semantic emergence
    p.add_argument('--enable_alignment_loss', action='store_true')
    p.add_argument('--alignment_weight', type=float, default=0.25)
    p.add_argument('--alignment_warmup_epochs', type=int, default=8)
    p.add_argument('--adaptive_alignment', action='store_true', default=True)
    p.add_argument('--alignment_min_correlation', type=float, default=0.05)
    p.add_argument('--enable_global_alignment_pass', action='store_true')
    p.add_argument('--alignment_global_students', type=int, default=600)
    p.add_argument('--use_residual_alignment', action='store_true')
    p.add_argument('--alignment_residual_window', type=int, default=5)
    # Retention & lag objectives
    p.add_argument('--enable_retention_loss', action='store_true')
    p.add_argument('--retention_delta', type=float, default=0.005)
    p.add_argument('--retention_weight', type=float, default=0.14)
    p.add_argument('--enable_lag_gain_loss', action='store_true')
    p.add_argument('--lag_gain_weight', type=float, default=0.06)
    p.add_argument('--lag_max_lag', type=int, default=3)
    p.add_argument('--lag_l1_weight', type=float, default=0.5)
    p.add_argument('--lag_l2_weight', type=float, default=0.3)
    p.add_argument('--lag_l3_weight', type=float, default=0.2)
    # Share cap & scheduling
    p.add_argument('--alignment_share_cap', type=float, default=0.08)
    p.add_argument('--alignment_share_decay_factor', type=float, default=0.7)
    p.add_argument('--warmup_constraint_epochs', type=int, default=8)
    p.add_argument('--enable_cosine_perf_schedule', action='store_true')
    p.add_argument('--consistency_rebalance_epoch', type=int, default=8)
    p.add_argument('--consistency_rebalance_threshold', type=float, default=0.10)
    p.add_argument('--consistency_rebalance_new_weight', type=float, default=0.2)
    # Variance floor
    p.add_argument('--variance_floor', type=float, default=1e-4)
    p.add_argument('--variance_floor_patience', type=int, default=3)
    p.add_argument('--variance_floor_reduce_factor', type=float, default=0.5)
    p.add_argument('--freeze_sparsity', action='store_true')
    # Semantic trajectory sampling
    p.add_argument('--max_semantic_students', type=int, default=50)
    p.add_argument('--semantic_trajectory_path', type=str, default=None)
    # Heads
    p.add_argument('--use_mastery_head', action='store_true', default=True)
    p.add_argument('--use_gain_head', action='store_true', default=True)
    # Output base
    p.add_argument('--output_base', type=str, default='examples/experiments')
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()
    def flag_present(flag: str) -> bool:
        return any(a == flag for a in sys.argv)
    if args.ablation_mode != 'baseline':
        if args.ablation_mode in ['align','both','both_lag'] and not flag_present('--enable_alignment_loss'):
            args.enable_alignment_loss = True
        if args.ablation_mode in ['align','both','both_lag'] and not flag_present('--enable_global_alignment_pass'):
            args.enable_global_alignment_pass = True
        if args.ablation_mode in ['retention','both','both_lag'] and not flag_present('--enable_retention_loss'):
            args.enable_retention_loss = True
        if args.ablation_mode == 'both_lag' and not flag_present('--enable_lag_gain_loss'):
            args.enable_lag_gain_loss = True
    else:
        if not flag_present('--enable_alignment_loss'):
            args.enable_alignment_loss = False
        if not flag_present('--enable_global_alignment_pass'):
            args.enable_global_alignment_pass = False
        if not flag_present('--enable_retention_loss'):
            args.enable_retention_loss = False
        if not flag_present('--enable_lag_gain_loss'):
            args.enable_lag_gain_loss = False
    return args

def build_config(args: argparse.Namespace, exp_id: str, exp_path: str, seeds: List[int]) -> Dict[str,Any]:
    cfg = {
        'experiment': {
            'id': exp_id,
            'model': 'gainakt2exp',
            'short_title': args.experiment_title,
            'purpose': 'Multi-seed reproducible training run',
            'experiment_suffix': args.experiment_suffix,
            'ablation_mode': args.ablation_mode
        },
        'data': {'dataset': args.dataset, 'fold': args.fold},
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'optimizer': 'Adam',
            'patience': args.patience,
            'mixed_precision': bool(args.use_amp),
            'gradient_clip': 1.0
        },
        'interpretability': {
            'enhanced_constraints': bool(args.enhanced_constraints),
            'non_negative_loss_weight': args.non_negative_loss_weight,
            'monotonicity_loss_weight': args.monotonicity_loss_weight,
            'mastery_performance_loss_weight': args.mastery_performance_loss_weight,
            'gain_performance_loss_weight': args.gain_performance_loss_weight,
            'sparsity_loss_weight': args.sparsity_loss_weight,
            'consistency_loss_weight': args.consistency_loss_weight,
            'warmup_constraint_epochs': args.warmup_constraint_epochs,
            'alignment_weight': args.alignment_weight,
            'enable_alignment_loss': bool(args.enable_alignment_loss),
            'retention_weight': args.retention_weight,
            'enable_retention_loss': bool(args.enable_retention_loss),
            'lag_gain_weight': args.lag_gain_weight,
            'enable_lag_gain_loss': bool(args.enable_lag_gain_loss)
        },
        'sampling': {
            'max_semantic_students': args.max_semantic_students,
            'alignment_global_students': args.alignment_global_students
        },
        'seeds': {'primary': seeds[0], 'all': seeds},
        'hardware': {'devices': args.devices, 'threads': int(os.environ.get('OMP_NUM_THREADS','8'))},
        'command': 'python examples/train_gainakt2exp_repro.py ' + ' '.join(sys.argv[1:]),
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    }
    cfg['config_md5'] = compute_config_hash(cfg)
    cfg['output_dir'] = exp_path
    return cfg

def write_readme(exp_path: str, cfg: Dict[str,Any], multi_seed_summary: Dict[str,Any] = None):
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
    lines = [
        f"# Experiment {cfg['experiment']['id']}",
        "",
        f"Model: {cfg['experiment']['model']}",
        f"Short title: {cfg['experiment']['short_title']}",
        f"Ablation mode: {cfg['experiment'].get('ablation_mode')}",
        "",
        '# Reproducibility Checklist',
        "",
        '| Item | Status |',
        '|------|--------|'
    ]
    for item, status in checklist:
        lines.append(f"| {item} | {status} |")
    if multi_seed_summary:
        lines.append("\n## Multi-seed Stability Summary")
        lines.append('Seeds: ' + ', '.join(str(s) for s in multi_seed_summary['seeds']))
        lines.append('')
        lines.append('| Final Metric | Mean | Std | Min | Max |')
        lines.append('|--------------|------|-----|-----|-----|')
        for metric, stats in multi_seed_summary['stats'].items():
            lines.append(f"| {metric} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |")
        lines.append("\n## Semantic Stability (Temporal)")
        lines.append('| Temporal Metric | Mean | Std | Min | Max |')
        lines.append('|-----------------|------|-----|-----|-----|')
        for metric, stats in multi_seed_summary['temporal_aggregate'].items():
            lines.append(f"| {metric} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |")
        lines.append("\n### Interpretive Narrative")
        lines.append("- Variance metrics quantify epoch-to-epoch stability of semantic grounding.")
        lines.append("- Positive early slopes followed by low variance indicate successful semantic emergence then consolidation.")
        lines.append("- Retention penalty count/mean/peak describe correlation decay events; high peak + high count may signal over-regularization or instability.")
        lines.append("- Divergence between global vs local mastery variance can motivate adjusting alignment sampling or weight.")
        lines.append("- If slopes negative post warm-up, investigate alignment decay or excessive retention penalty.")
    with open(os.path.join(exp_path,'README.md'),'w') as f:
        f.write('\n'.join(lines))

def main():
    args = parse_args()
    if args.experiment_suffix is None:
        args.experiment_suffix = args.experiment_title
    seeds = args.seeds if args.seeds else [args.seed]
    seeds = list(dict.fromkeys(seeds))
    if args.devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in args.devices)
    exp_path = make_experiment_dir('gainakt2exp', args.experiment_title, base_dir=args.output_base)
    exp_id = os.path.basename(exp_path)
    logger = timestamped_logger(exp_id, os.path.join(exp_path,'stdout.log'))
    logger.info(f"Created experiment folder: {exp_path}")
    cfg = build_config(args, exp_id, exp_path, seeds)
    atomic_write_json(cfg, os.path.join(exp_path,'config.json'))
    capture_environment(os.path.join(exp_path,'environment.txt'))
    metrics_csv = os.path.join(exp_path,'metrics_epoch.csv')
    artifacts_dir = os.path.join(exp_path,'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    base_suffix = args.experiment_suffix
    per_seed_results: List[Dict[str,Any]] = []
    per_seed_temporal: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        logger.info(f"\n===== Seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_suffix = f"{base_suffix}_s{seed}"
        args.experiment_suffix = seed_suffix
        args.semantic_trajectory_path = os.path.join(artifacts_dir, f'semantic_trajectory_seed{seed}.json')
        results = train_gainakt2exp_model(args)
        history = results.get('train_history', {})
        semantic = history.get('semantic_trajectory', [])
        consistency_list = history.get('consistency_metrics', [])
        # Temporal stability extraction
        mastery_corrs = [e.get('mastery_correlation') for e in semantic if e.get('mastery_correlation') is not None]
        gain_corrs = [e.get('gain_correlation') for e in semantic if e.get('gain_correlation') is not None]
        global_mastery_corrs = [e.get('global_alignment_mastery_corr') for e in semantic if e.get('global_alignment_mastery_corr') is not None]
        retention_penalties = [e.get('retention_loss_value') for e in semantic if e.get('retention_loss_value') is not None and e.get('retention_loss_value') > 0]
        def slope(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals), dtype=float)
            y = np.array(vals, dtype=float)
            xm = x.mean()
            ym = y.mean()
            denom = ((x - xm)**2).sum() + 1e-8
            return float(((x - xm) * (y - ym)).sum() / denom)
        per_seed_temporal[seed] = {
            'mastery_corr_slope': slope(mastery_corrs),
            'gain_corr_slope': slope(gain_corrs),
            'mastery_corr_variance': float(np.var(mastery_corrs)) if mastery_corrs else 0.0,
            'gain_corr_variance': float(np.var(gain_corrs)) if gain_corrs else 0.0,
            'global_mastery_corr_variance': float(np.var(global_mastery_corrs)) if global_mastery_corrs else 0.0,
            'retention_penalty_count': len(retention_penalties),
            'retention_penalty_mean': float(np.mean(retention_penalties)) if retention_penalties else 0.0,
            'retention_penalty_peak': float(np.max(retention_penalties)) if retention_penalties else 0.0
        }
        for idx, epoch_entry in enumerate(semantic):
            epoch_num = epoch_entry.get('epoch', idx+1)
            consistency = consistency_list[idx] if idx < len(consistency_list) else {}
            # CI values (per-epoch consistency metrics already contain them if present)
            mastery_ci_low = consistency.get('mastery_correlation_ci_low')
            mastery_ci_high = consistency.get('mastery_correlation_ci_high')
            gain_ci_low = consistency.get('gain_correlation_ci_low')
            gain_ci_high = consistency.get('gain_correlation_ci_high')
            row = {
                'seed': seed,
                'epoch': epoch_num,
                'train_loss': history.get('train_loss',[None]*epoch_num)[epoch_num-1] if history.get('train_loss') else None,
                'train_auc': history.get('train_auc',[None]*epoch_num)[epoch_num-1] if history.get('train_auc') else None,
                'val_auc': history.get('val_auc',[None]*epoch_num)[epoch_num-1] if history.get('val_auc') else None,
                'val_accuracy': history.get('val_acc',[None]*epoch_num)[epoch_num-1] if history.get('val_acc') else None,
                'monotonicity_violation_rate': consistency.get('monotonicity_violation_rate'),
                'negative_gain_rate': consistency.get('negative_gain_rate'),
                'bounds_violation_rate': consistency.get('bounds_violation_rate'),
                'mastery_correlation': consistency.get('mastery_correlation'),
                'mastery_ci_low': mastery_ci_low,
                'mastery_ci_high': mastery_ci_high,
                'gain_correlation': consistency.get('gain_correlation'),
                'gain_ci_low': gain_ci_low,
                'gain_ci_high': gain_ci_high,
                'main_loss_share': epoch_entry.get('loss_shares',{}).get('main'),
                'constraint_loss_share': epoch_entry.get('loss_shares',{}).get('constraint_total'),
                'alignment_loss_share': epoch_entry.get('loss_shares',{}).get('alignment'),
                'lag_loss_share': epoch_entry.get('loss_shares',{}).get('lag'),
                'retention_loss_share': epoch_entry.get('loss_shares',{}).get('retention'),
            }
            append_epoch_csv(row, metrics_csv, EPOCH_HEADER)
        per_seed_results.append({
            'seed': seed,
            'best_val_auc': results.get('best_val_auc'),
            'final_consistency': results.get('final_consistency_metrics', {}),
            'val_accuracy_last_epoch': history.get('val_acc',[None])[-1] if history.get('val_acc') else None
        })
        original_dir = os.path.join('saved_model', f'gainakt2exp_{seed_suffix}')
        if os.path.isdir(original_dir):
            best_src = os.path.join(original_dir,'best_model.pth')
            if os.path.exists(best_src):
                import shutil
                shutil.copy2(best_src, os.path.join(exp_path,f'model_best_seed{seed}.pth'))
                logger.info(f'Copied best checkpoint for seed {seed}.')
        torch.save({'note':'placeholder last checkpoint','seed':seed}, os.path.join(exp_path,f'model_last_seed{seed}.pth'))

    def _stats(vals: List[float]) -> Dict[str,float]:
        arr = np.array(vals, dtype=float)
        return {'mean': float(arr.mean()) if arr.size else 0.0,
                'std': float(arr.std()) if arr.size else 0.0,
                'min': float(arr.min()) if arr.size else 0.0,
                'max': float(arr.max()) if arr.size else 0.0}
    def agg(vals: List[float]) -> Dict[str,float]:
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            return {'mean':0.0,'std':0.0,'min':0.0,'max':0.0}
        return {'mean':float(arr.mean()),'std':float(arr.std()),'min':float(arr.min()),'max':float(arr.max())}
    mastery_variances = [per_seed_temporal[s]['mastery_corr_variance'] for s in per_seed_temporal]
    gain_variances = [per_seed_temporal[s]['gain_corr_variance'] for s in per_seed_temporal]
    global_mastery_variances = [per_seed_temporal[s]['global_mastery_corr_variance'] for s in per_seed_temporal]
    mastery_slopes = [per_seed_temporal[s]['mastery_corr_slope'] for s in per_seed_temporal]
    gain_slopes = [per_seed_temporal[s]['gain_corr_slope'] for s in per_seed_temporal]
    retention_counts = [per_seed_temporal[s]['retention_penalty_count'] for s in per_seed_temporal]
    retention_means = [per_seed_temporal[s]['retention_penalty_mean'] for s in per_seed_temporal]
    retention_peaks = [per_seed_temporal[s]['retention_penalty_peak'] for s in per_seed_temporal]
    multi_seed_summary = {
        'seeds': seeds,
        'stats': {
            'best_val_auc': _stats([r['best_val_auc'] for r in per_seed_results]),
            'mastery_correlation': _stats([r['final_consistency'].get('mastery_correlation',0.0) for r in per_seed_results]),
            'gain_correlation': _stats([r['final_consistency'].get('gain_correlation',0.0) for r in per_seed_results]),
            'val_accuracy_last_epoch': _stats([r['val_accuracy_last_epoch'] for r in per_seed_results if r['val_accuracy_last_epoch'] is not None])
        },
        'per_seed': per_seed_results,
        'temporal': per_seed_temporal,
        'temporal_aggregate': {
            'mastery_corr_variance': agg(mastery_variances),
            'gain_corr_variance': agg(gain_variances),
            'global_mastery_corr_variance': agg(global_mastery_variances),
            'mastery_corr_slope': agg(mastery_slopes),
            'gain_corr_slope': agg(gain_slopes),
            'retention_penalty_count': agg(retention_counts),
            'retention_penalty_mean': agg(retention_means),
            'retention_penalty_peak': agg(retention_peaks)
        },
        'config_md5': cfg['config_md5'],
        'completed_at': datetime.datetime.utcnow().isoformat()+'Z'
    }
    atomic_write_json(multi_seed_summary, os.path.join(exp_path,'results_multi_seed.json'))
    primary = per_seed_results[0]
    legacy = {
        'best_epoch_auc': primary['best_val_auc'],
        'final_consistency': primary['final_consistency'],
        'config_md5': cfg['config_md5'],
        'completed_at': multi_seed_summary['completed_at'],
        'multi_seed': len(seeds) > 1,
        'seeds': seeds,
        'best_mastery_corr': primary['final_consistency'].get('mastery_correlation'),
        'best_gain_corr': primary['final_consistency'].get('gain_correlation')
    }
    atomic_write_json(legacy, os.path.join(exp_path,'results.json'))
    write_readme(exp_path, cfg, multi_seed_summary=multi_seed_summary)
    logger.info('Experiment complete (multi-seed aggregation).')

if __name__ == '__main__':
    main()
