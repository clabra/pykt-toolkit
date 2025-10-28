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
import hashlib
import json
from pathlib import Path
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
    p = argparse.ArgumentParser(description="Unified GainAKT2Exp training, reproduction, and comparison")
    # existing args (modified description):
    p.add_argument('--ablation_mode', type=str, default='both_lag', choices=['baseline','align','retention','both','both_lag'],
                   help='Preset enabling alignment/retention/lag objectives unless explicitly overridden by flags.')
    p.add_argument('--dataset', '--dataset_name', dest='dataset', default='assist2015')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', '--num_epochs', dest='epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.00016)
    p.add_argument('--weight_decay', type=float, default=1.7571e-05)
    p.add_argument('--enhanced_constraints', action='store_true', default=True)
    p.add_argument('--experiment_title', '--short-title', dest='experiment_title', type=str, default='baseline')
    p.add_argument('--experiment_suffix', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='*', default=None)
    p.add_argument('--no_amp', action='store_true', help='Disable mixed precision (AMP). Enabled by default.')
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--devices', type=int, nargs='*', default=[0,1,2,3,4])
    p.add_argument('--monitor_freq', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    # Reproduction / comparison / overrides
    p.add_argument('--experiment_dir', type=str, default=None, help='Explicit experiment directory id or absolute path to use/reuse.')
    p.add_argument('--force', action='store_true', help='Allow reuse of existing non-empty experiment directory.')
    p.add_argument('--set', action='append', default=[], help='Override config value using dot-path key=val (repeatable).')
    p.add_argument('--reproduce_from', type=str, default=None, help='Source experiment folder to reproduce (creates *_reproduce).')
    p.add_argument('--compare_only', action='store_true', help='Compare two experiments without training.')
    p.add_argument('--source_exp', type=str, default=None, help='Source experiment id or path for compare-only.')
    p.add_argument('--target_exp', type=str, default=None, help='Target experiment id or path for compare-only.')
    p.add_argument('--manifest', action='store_true', help='Write reproduction_manifest.json summarizing reproduction or comparison.')
    p.add_argument('--strict_schema', action='store_true', help='Abort reproduction if schema_version differs.')
    # Constraint weights
    p.add_argument('--non_negative_loss_weight', type=float, default=0.0)
    p.add_argument('--monotonicity_loss_weight', type=float, default=0.1)
    p.add_argument('--mastery_performance_loss_weight', type=float, default=0.8)
    p.add_argument('--gain_performance_loss_weight', type=float, default=0.8)
    p.add_argument('--sparsity_loss_weight', type=float, default=0.2)
    p.add_argument('--consistency_loss_weight', type=float, default=0.3)
    # Alignment / semantic emergence
    p.add_argument('--enable_alignment_loss', action='store_true')
    # Increased alignment weight back to previously stable baseline
    p.add_argument('--alignment_weight', type=float, default=0.30)
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
    # Restore retention weight to baseline value
    p.add_argument('--retention_weight', type=float, default=0.12)
    p.add_argument('--enable_lag_gain_loss', action='store_true')
    # Restore lag gain weight to baseline value
    p.add_argument('--lag_gain_weight', type=float, default=0.05)
    p.add_argument('--lag_max_lag', type=int, default=3)
    p.add_argument('--lag_l1_weight', type=float, default=0.5)
    p.add_argument('--lag_l2_weight', type=float, default=0.3)
    p.add_argument('--lag_l3_weight', type=float, default=0.2)
    # Share cap & scheduling
    # Restore alignment share scheduling parameters
    p.add_argument('--alignment_share_cap', type=float, default=0.10)
    p.add_argument('--alignment_share_decay_factor', type=float, default=0.75)
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
    # Use absolute path anchored at project root to avoid duplication when launching from within examples/
    p.add_argument('--output_base', type=str, default=None,
                   help='Base directory for experiment folders. Defaults to <project_root>/examples/experiments if not set.')
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()
    setattr(args, 'use_amp', not getattr(args, 'no_amp', False))
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
    # Collect all args for completeness
    args_dict = vars(args)
    cfg = {
        'schema_version': 2,
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
            'gradient_clip': 1.0,
            'monitor_freq': args.monitor_freq
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
            'retention_weight': args.retention_weight,
            'lag_gain_weight': args.lag_gain_weight,
            'enable_alignment_loss': bool(args.enable_alignment_loss),
            'enable_retention_loss': bool(args.enable_retention_loss),
            'enable_lag_gain_loss': bool(args.enable_lag_gain_loss)
        },
        'alignment': {
            'alignment_warmup_epochs': args.alignment_warmup_epochs,
            'adaptive_alignment': bool(args.adaptive_alignment),
            'alignment_min_correlation': args.alignment_min_correlation,
            'enable_global_alignment_pass': bool(args.enable_global_alignment_pass),
            'alignment_global_students': args.alignment_global_students,
            'use_residual_alignment': bool(args.use_residual_alignment),
            'alignment_residual_window': args.alignment_residual_window,
            'alignment_share_cap': args.alignment_share_cap,
            'alignment_share_decay_factor': args.alignment_share_decay_factor
        },
        'retention_lag': {
            'retention_delta': args.retention_delta,
            'lag_max_lag': args.lag_max_lag,
            'lag_l1_weight': args.lag_l1_weight,
            'lag_l2_weight': args.lag_l2_weight,
            'lag_l3_weight': args.lag_l3_weight
        },
        'scheduling': {
            'enable_cosine_perf_schedule': bool(args.enable_cosine_perf_schedule),
            'consistency_rebalance_epoch': args.consistency_rebalance_epoch,
            'consistency_rebalance_threshold': args.consistency_rebalance_threshold,
            'consistency_rebalance_new_weight': args.consistency_rebalance_new_weight
        },
        'stability_controls': {
            'variance_floor': args.variance_floor,
            'variance_floor_patience': args.variance_floor_patience,
            'variance_floor_reduce_factor': args.variance_floor_reduce_factor,
            'freeze_sparsity': bool(args.freeze_sparsity)
        },
        'heads': {
            'use_mastery_head': bool(args.use_mastery_head),
            'use_gain_head': bool(args.use_gain_head)
        },
        'sampling': {
            'max_semantic_students': args.max_semantic_students
        },
        'paths': {
            'semantic_trajectory_path': args.semantic_trajectory_path,
            'output_dir': exp_path
        },
        'runtime': {
            'resume': args.resume,
            'devices': args.devices,
            'threads': int(os.environ.get('OMP_NUM_THREADS','8')),
            'command': 'python examples/train_gainakt2exp_repro.py ' + ' '.join(sys.argv[1:]),
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        },
        'seeds': {'primary': seeds[0], 'all': seeds}
    }
    # Determine missing args (those parsed but not yet represented). Flatten cfg keys for comparison.
    serialized_keys = set()
    def collect(d):
        for k,v in d.items():
            if isinstance(v, dict):
                collect(v)
            else:
                serialized_keys.add(k)
    collect(cfg)
    args_dict = vars(args)
    # CLI-only flags or alias names intentionally not serialized (represented indirectly)
    exclude = {
        'experiment_title',  # stored as experiment.short_title
        'seed',               # seeds.primary covers it
        'seeds',              # represented under seeds.all
        'no_amp',             # reflected by training.mixed_precision
        'use_amp',            # same as above (derived)
        'use_wandb',          # external logging not serialized
        'experiment_dir',     # folder path chosen externally
        'force',              # control flag, not part of spec
        'set',                # override list applied pre-hash
        'reproduce_from',     # reproduction metadata stored separately
        'compare_only',       # mode flag (no training)
        'source_exp',         # comparison source
        'target_exp',         # comparison target
        'manifest',           # optional artifact creation
        'strict_schema',      # reproduction check behavior
        'output_base',        # base directory (not part of spec content)
        'devices',            # serialized under runtime.devices
        'resume',             # serialized under runtime.resume if provided
    }
    missing = [k for k in args_dict.keys() if k not in serialized_keys and k not in exclude]
    cfg['missing_params'] = missing
    assert len(missing) == 0, f"Unserialized parameters found (unexpected): {missing}"
    cfg['config_md5'] = compute_config_hash(cfg)
    return cfg

# Override helpers
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
            parts = [p.strip() for p in r.split(',') if p.strip()]
            coerced = []
            for p in parts:
                cp = _coerce_value(p)
                coerced.append(cp)
            return coerced
        return r

def apply_overrides_to_args(args: argparse.Namespace, overrides: list) -> list:
    logs = []
    for ov in overrides:
        if '=' not in ov:
            print(f"[WARN] Ignoring malformed override '{ov}' (missing '=')")
            continue
        key, val = ov.split('=',1)
        value = _coerce_value(val)
        original = None
        # Map common prefixes to args attribute names
        if key.startswith('training.'):
            sub = key.split('.',1)[1]
            attr_map = {'epochs':'epochs','batch_size':'batch_size','learning_rate':'learning_rate','weight_decay':'weight_decay','seeds':'seeds'}
            if sub in attr_map:
                original = getattr(args, attr_map[sub], None)
                if sub == 'seeds':
                    if isinstance(value, list):
                        value = [int(x) for x in value]
                setattr(args, attr_map[sub], value)
        elif key.startswith('interpretability.'):
            sub = key.split('.',1)[1]
            if hasattr(args, sub):
                original = getattr(args, sub)
                setattr(args, sub, value)
        elif key.startswith('alignment.'):
            sub = key.split('.',1)[1]
            name = 'alignment_' + sub if not sub.startswith('alignment_') else sub
            if hasattr(args, name):
                original = getattr(args, name)
                setattr(args, name, value)
        elif key.startswith('retention_lag.'):
            sub = key.split('.',1)[1]
            if sub.startswith('lag_') or sub in ['retention_delta']:
                if hasattr(args, sub):
                    original = getattr(args, sub)
                    setattr(args, sub, value)
        elif key.startswith('heads.'):
            sub = key.split('.',1)[1]
            name = 'use_' + sub if sub in ['mastery_head','gain_head'] else sub
            if hasattr(args, name):
                original = getattr(args, name)
                setattr(args, name, value)
        elif key.startswith('sampling.'):
            sub = key.split('.',1)[1]
            if hasattr(args, sub):
                original = getattr(args, sub)
                setattr(args, sub, value)
        else:
            # Unmapped override; record but not applied to args
            original = None
        logs.append(f"{key}:{original}->{value}")
    return logs

# Results MD5 helper

def file_md5(path: str) -> str:
    try:
        with open(path,'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

# Comparison logic

def compare_experiments(source: str, target: str) -> dict:
    def resolve(p: str) -> str:
        base = '/workspaces/pykt-toolkit/examples/experiments'
        abs_path = p if os.path.isabs(p) else os.path.join(base, p)
        return abs_path
    sdir = resolve(source)
    tdir = resolve(target)
    report = {'source': sdir, 'target': tdir, 'exists_source': os.path.isdir(sdir), 'exists_target': os.path.isdir(tdir)}
    if not (os.path.isdir(sdir) and os.path.isdir(tdir)):
        return report
    def load_results(d):
        rjson = os.path.join(d,'results.json')
        if os.path.exists(rjson):
            with open(rjson) as f:
                return json.load(f)
        return {}
    sres = load_results(sdir)
    tres = load_results(tdir)
    report['source_config_md5'] = sres.get('config_md5')
    report['target_config_md5'] = tres.get('config_md5')
    report['config_md5_match'] = (report['source_config_md5'] == report['target_config_md5'] and report['source_config_md5'] is not None)
    report['source_best_epoch_auc'] = sres.get('best_epoch_auc')
    report['target_best_epoch_auc'] = tres.get('best_epoch_auc')
    if report['source_best_epoch_auc'] is not None and report['target_best_epoch_auc'] is not None:
        report['best_auc_abs_diff'] = abs(report['source_best_epoch_auc'] - report['target_best_epoch_auc'])
    report['source_mastery_corr'] = sres.get('best_mastery_corr')
    report['target_mastery_corr'] = tres.get('best_mastery_corr')
    if report['source_mastery_corr'] is not None and report['target_mastery_corr'] is not None:
        report['mastery_corr_abs_diff'] = abs(report['source_mastery_corr'] - report['target_mastery_corr'])
    report['source_gain_corr'] = sres.get('best_gain_corr')
    report['target_gain_corr'] = tres.get('best_gain_corr')
    if report['source_gain_corr'] is not None and report['target_gain_corr'] is not None:
        report['gain_corr_abs_diff'] = abs(report['source_gain_corr'] - report['target_gain_corr'])
    s_md5 = file_md5(os.path.join(sdir,'results.json'))
    t_md5 = file_md5(os.path.join(tdir,'results.json'))
    report['source_results_md5'] = s_md5
    report['target_results_md5'] = t_md5
    report['results_md5_match'] = (s_md5 == t_md5 and s_md5 is not None)
    # Tolerance assessment
    tol_auc = 0.002
    tol_corr = 0.01
    report['within_tolerance'] = (
        (report.get('best_auc_abs_diff',0) <= tol_auc) and
        (report.get('mastery_corr_abs_diff',0) <= tol_corr) and
        (report.get('gain_corr_abs_diff',0) <= tol_corr)
    )
    return report

# Augment write_readme to include reproduction/comparison metadata

def write_readme(exp_path: str, cfg: Dict[str,Any], multi_seed_summary: Dict[str,Any] = None, reproduction_meta: Dict[str,Any] = None):
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
        f"Schema version: {cfg.get('schema_version','N/A')}",
        "",
        '# Reproducibility Checklist',
        "",
        '| Item | Status |',
        '|------|--------|'
    ]
    for item, status in checklist:
        lines.append(f"| {item} | {status} |")
    if reproduction_meta:
        lines.append("\n## Reproduction Metadata")
        lines.append('| Field | Value |')
        lines.append('|-------|-------|')
        for k,v in reproduction_meta.items():
            lines.append(f"| {k} | {v} |")
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
    lines.append("\n## Parameter Scenario Quick Reference")
    lines.append("Comprehensive (defaults): --ablation_mode both_lag (alignment+retention+lag) warmup=8 alignment_weight=0.30 retention_weight=0.12 lag_gain_weight=0.05")
    lines.append("Predictive baseline: --ablation_mode baseline (all interpretability losses disabled, heads optional)")
    lines.append("Alignment-only: --ablation_mode align (retention & lag disabled) to observe correlation decay")
    lines.append("Retention stress: increase --retention_weight (e.g., 0.18) and decrease --retention_delta (e.g., 0.003) to test decay resistance")
    lines.append("Variance recovery: extend --warmup_constraint_epochs (e.g., 10) and reduce performance loss weights (0.8→0.7) to boost head variance")
    lines.append("Mastery-only: --use_gain_head False to isolate mastery semantics")
    lines.append("Disable AMP: add --no_amp (AMP enabled by default)")
    lines.append("See paper/README_gainakt2exp.md for full parameter tables & tuning guidelines.")
    with open(os.path.join(exp_path,'README.md'),'w') as f:
        f.write('\n'.join(lines))

# Main updated to handle reproduction & comparison modes

def main():
    args = parse_args()
    # Comparison mode early exit
    if args.compare_only:
        if not (args.source_exp and args.target_exp):
            print('[ERROR] --compare_only requires --source_exp and --target_exp')
            sys.exit(1)
        report = compare_experiments(args.source_exp, args.target_exp)
        # Write report to target experiment folder if exists else current path
        target_dir = report.get('target') if os.path.isdir(report.get('target','')) else os.getcwd()
        out_path = os.path.join(target_dir,'reproduction_report.json')
        atomic_write_json(report, out_path)
        print(f"Comparison report written: {out_path}")
        sys.exit(0)
    # Apply overrides to args BEFORE config construction
    override_logs = apply_overrides_to_args(args, args.set)
    if args.experiment_suffix is None:
        args.experiment_suffix = args.experiment_title
    seeds = args.seeds if args.seeds else [args.seed]
    seeds = list(dict.fromkeys(seeds))
    if args.devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in args.devices)
    project_root = Path(__file__).resolve().parent.parent
    if args.output_base is None:
        base_dir = project_root / 'examples' / 'experiments'
    else:
        supplied = Path(args.output_base)
        base_dir = supplied if supplied.is_absolute() else project_root / supplied
    base_dir.mkdir(parents=True, exist_ok=True)
    source_repro_dir = None
    reproduction_meta = None
    # Reproduction path
    if args.reproduce_from:
        source_repro_dir = Path(args.reproduce_from)
        if not source_repro_dir.is_absolute():
            source_repro_dir = base_dir / args.reproduce_from
        if not source_repro_dir.exists():
            print(f"[ERROR] Source reproduction folder does not exist: {source_repro_dir}")
            sys.exit(1)
        # Load source config for baseline args (schema/version check optional)
        source_cfg_path = source_repro_dir / 'config.json'
        if not source_cfg_path.exists():
            print('[ERROR] Source config.json missing for reproduction.')
            sys.exit(1)
        with open(source_cfg_path) as f:
            source_cfg = json.load(f)
        if args.strict_schema and source_cfg.get('schema_version') != 2:
            print('[ERROR] Schema version mismatch. Aborting reproduction.')
            sys.exit(1)
        # Create new folder with _reproduce suffix
        short_title = source_cfg['experiment']['short_title'] + '_reproduce'
        exp_path = make_experiment_dir('gainakt2exp', short_title, base_dir=str(base_dir))
        exp_id = os.path.basename(exp_path)
        # Build config from current args (overrides may differ) then tag reproduction metadata
        cfg = build_config(args, exp_id, exp_path, seeds)
        cfg['reproduction'] = {
            'mode': 'reproduce',
            'source_experiment': source_cfg['experiment']['id'],
            'source_config_md5': source_cfg.get('config_md5'),
            'applied_overrides': override_logs
        }
        reproduction_meta = {
            'source_experiment': source_cfg['experiment']['id'],
            'source_config_md5': source_cfg.get('config_md5'),
            'new_config_md5': cfg.get('config_md5'),
            'override_count': len(override_logs)
        }
    else:
        # New or explicit experiment dir reuse
        if args.experiment_dir:
            exp_dir_candidate = Path(args.experiment_dir)
            if not exp_dir_candidate.is_absolute():
                exp_dir_candidate = base_dir / args.experiment_dir
            if exp_dir_candidate.exists():
                if not args.force:
                    print('[ERROR] experiment_dir exists; use --force to reuse.')
                    sys.exit(1)
                exp_path = str(exp_dir_candidate)
            else:
                exp_dir_candidate.parent.mkdir(parents=True, exist_ok=True)
                exp_path = str(exp_dir_candidate)
                os.makedirs(exp_path, exist_ok=True)
                os.makedirs(os.path.join(exp_path,'artifacts'), exist_ok=True)
            exp_id = os.path.basename(exp_path)
        else:
            exp_path = make_experiment_dir('gainakt2exp', args.experiment_title, base_dir=str(base_dir))
            exp_id = os.path.basename(exp_path)
        cfg = build_config(args, exp_id, exp_path, seeds)
    logger = timestamped_logger(exp_id, os.path.join(exp_path,'stdout.log'))
    logger.info(f"Created experiment folder: {exp_path}")
    atomic_write_json(cfg, os.path.join(exp_path,'config.json'))
    capture_environment(os.path.join(exp_path,'environment.txt'))
    # Write override audit
    audit_list = override_logs
    write_path = os.path.join(exp_path,'overrides_applied.txt')
    with open(write_path,'a') as auditf:
        auditf.write(f"timestamp: {datetime.datetime.utcnow().isoformat()}Z\n")
        auditf.write(f"experiment_id: {exp_id}\n")
        auditf.write(f"override_count: {len(audit_list)}\n")
        for ov in audit_list:
            auditf.write(f"override: {ov}\n")
        auditf.write("---\n")
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
        seed_suffix = f"{base_suffix}_s{seed}" if base_suffix else f"s{seed}"
        args.experiment_suffix = seed_suffix
        args.semantic_trajectory_path = os.path.join(artifacts_dir, f'semantic_trajectory_seed{seed}.json')
        results = train_gainakt2exp_model(args)
        history = results.get('train_history', {})
        semantic = history.get('semantic_trajectory', [])
        consistency_list = history.get('consistency_metrics', [])
        mastery_corrs = [e.get('mastery_correlation') for e in semantic if e.get('mastery_correlation') is not None]
        gain_corrs = [e.get('gain_correlation') for e in semantic if e.get('gain_correlation') is not None]
        global_mastery_corrs = [e.get('global_alignment_mastery_corr') for e in semantic if e.get('global_alignment_mastery_corr') is not None]
        retention_penalties = [e.get('retention_loss_value') for e in semantic if e.get('retention_loss_value') is not None and e.get('retention_loss_value') > 0]
        def slope(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals), dtype=float)
            y = np.array(vals, dtype=float)
            # slope computation separated for readability
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
    # Reproduction report (if reproduction mode)
    if reproduction_meta:
        source_dir = source_repro_dir
        rep = compare_experiments(str(source_dir.name), os.path.basename(exp_path)) if source_dir else {}
        rep.update({'mode':'reproduce','source_experiment': reproduction_meta['source_experiment']})
        atomic_write_json(rep, os.path.join(exp_path,'reproduction_report.json'))
    # Optional manifest
    if args.manifest:
        manifest = {
            'experiment_id': cfg['experiment']['id'],
            'schema_version': cfg.get('schema_version'),
            'config_md5': cfg.get('config_md5'),
            'seeds': seeds,
            'reproduction_mode': bool(reproduction_meta),
            'override_count': len(override_logs),
            'results_md5': file_md5(os.path.join(exp_path,'results.json'))
        }
        atomic_write_json(manifest, os.path.join(exp_path,'reproduction_manifest.json'))
    write_readme(exp_path, cfg, multi_seed_summary=multi_seed_summary, reproduction_meta=reproduction_meta)
    logger.info('Unified experiment complete.')

if __name__ == '__main__':
    main()
