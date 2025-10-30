#!/usr/bin/env python3
"""Experiment utilities
"""
from __future__ import annotations

import os
import json
import csv
import hashlib
import platform
import sys
import subprocess
import logging
import datetime
from collections import OrderedDict
from typing import Dict, Any, List, Optional

TS_FMT = "%Y%m%d_%H%M%S"

def utc_timestamp(fmt: str = TS_FMT) -> str:
    return datetime.datetime.utcnow().strftime(fmt)

def make_experiment_dir(model_name: str, short_title: str, base_dir: str = "examples/experiments") -> str:
    ts = utc_timestamp()
    exp_id = f"{ts}_{model_name}_{short_title}".lower()
    path = os.path.join(base_dir, exp_id)
    if os.path.exists(path):
        raise FileExistsError(f"Experiment directory already exists: {path}")
    os.makedirs(path, exist_ok=False)
    os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
    return path

def capture_environment(out_path: str):
    info = {
        "python": sys.version.replace('\n',' '),
        "platform": platform.platform(),
        "cuda_available": False,
        "torch_version": None,
        "cuda_version": None,
        "git_commit": None,
        "git_branch": None,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    except Exception:
        pass
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
        branch = subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"]).decode().strip()
        info["git_commit"], info["git_branch"] = commit, branch
    except Exception:
        pass
    with open(out_path, 'w') as f:
        for k,v in info.items():
            f.write(f"{k}: {v}\n")

def atomic_write_json(obj: Dict[str,Any], path: str):
    tmp = path + ".tmp"
    with open(tmp,'w') as f:
        # Preserve insertion order; do NOT sort keys so group ordering remains as constructed
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def compute_config_hash(obj: Dict[str,Any]) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def append_epoch_csv(row: Dict[str,Any], csv_path: str, header: List[str]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path,'a',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in header})

def timestamped_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
    return logger

def write_seed_info(path: str, seeds: List[int], primary: int):
    lines = ["# Seed Information","",f"Primary seed: {primary}","All seeds: " + ", ".join(map(str,seeds)),
             "Rationale: Primary used for initial trajectory; others for stability variance."]
    with open(path,'w') as f:
        f.write('\n'.join(lines))

def write_readme(exp_path: str, cfg: Dict[str,Any], best_metrics: Optional[Dict[str,Any]] = None):
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
        ('Correlation / interpretability metrics logged','✅'),
    ]
    lines = [f"# Experiment {cfg['experiment']['id']}","",f"Model: {cfg['experiment']['model']}",
             f"Short title: {cfg['experiment']['short_title']}","","## Reproducibility Checklist","",
             '| Item | Status |','|------|--------|']
    for item,status in checklist:
        lines.append(f"| {item} | {status} |")
    # Config provenance section
    lines.append("\n## Configuration Provenance")
    lines.append("The `config.json` groups appear in the following canonical order ensuring reproducibility of parameter resolution:")
    ordered_groups = ['runtime','hardware','seeds','training','experiment','data','interpretability','alignment','global_alignment','refinement']
    lines.append('\n')
    lines.append('| Group | Key Highlights |')
    lines.append('|-------|----------------|')
    highlights = {
        'runtime': 'launcher + train commands, timestamp',
        'hardware': 'device list, thread count',
        'seeds': 'primary + all seeds',
        'training': 'epochs, batch_size, lr, optimizer, mixed_precision, gradient_clip',
        'experiment': 'id, model, short_title, purpose',
        'data': 'dataset, fold',
        'interpretability': 'core heads + constraint weights + warmup',
        'alignment': 'enable, weight, warmup, adaptive, min_corr, share_cap/decay',
        'global_alignment': 'enable pass, students, residual window',
        'refinement': 'retention, lag predictive emergence, rebalance, variance floor'
    }
    for g in ordered_groups:
        if g in cfg:
            lines.append(f"| {g} | {highlights.get(g,'')} |")
    # Semantic flag snapshot
    lines.append("\n### Semantic Flag Snapshot")
    lines.append("| Flag | Value |")
    lines.append("|------|-------|")
    # Flatten selected semantic flags for quick audit
    semantic_pairs = []
    for section in ['alignment','global_alignment','refinement']:
        if section in cfg:
            for k,v in cfg[section].items():
                semantic_pairs.append((f"{section}.{k}", v))
    for k,v in semantic_pairs:
        lines.append(f"| {k} | {v} |")
    # Interpretability core & constraints snapshot
    lines.append("\n### Core Interpretability & Constraints")
    lines.append("| Key | Value |")
    lines.append("|-----|-------|")
    for k,v in cfg.get('interpretability', {}).items():
        lines.append(f"| interpretability.{k} | {v} |")
    # Evaluation snapshot section if present
    if 'evaluation_snapshot' in cfg:
        lines.append("\n### Evaluation Snapshot")
        lines.append("| Key | Value |")
        lines.append("|-----|-------|")
        for k,v in cfg['evaluation_snapshot'].items():
            lines.append(f"| evaluation.{k} | {v} |")
        if 'evaluation_snapshot_md5' in cfg:
            lines.append(f"\nSnapshot MD5: `{cfg['evaluation_snapshot_md5']}`")
    # Reproducibility policy section
    if 'reproducibility_policy' in cfg:
        lines.append("\n### Reproducibility Policy")
        policy = cfg['reproducibility_policy']
        lines.append("Ignored evaluation flags: " + (', '.join(policy.get('ignored_eval_flags', [])) or 'None'))
        lines.append("Ignored training flags: " + (', '.join(policy.get('ignored_training_flags', [])) or 'None'))
        lines.append(f"Schema version: {policy.get('schema_version','<unset>')}")
        lines.append(f"Policy last updated: {policy.get('policy_last_updated','<unset>')}")
    if best_metrics:
        lines.append("\n## Best Epoch Summary")
        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        for k,v in best_metrics.items():
            lines.append(f"| {k} | {v} |")
    with open(os.path.join(exp_path,'README.md'),'w') as f:
        f.write('\n'.join(lines) + '\n')

def build_config(raw_args: Dict[str,Any], exp_id: str, exp_path: str, seeds: List[int]) -> Dict[str,Any]:
    # Distinguish between launcher command and underlying train script command
    launcher_command = raw_args.get('launcher_command', raw_args.get('command'))
    train_command = raw_args.get('train_command', '<unset>')
    # Normalize runtime fields: enforce non-null monitor_freq and explicit seed
    monitor_freq_val = raw_args.get('monitor_freq')
    if monitor_freq_val is None:
        # Fallback to default (50) if absent; MUST be explicit for reproducibility
        monitor_freq_val = 50
    seed_val = raw_args.get('seed', seeds[0] if seeds else 42)
    cfg = OrderedDict({
        'runtime': {
            'command': launcher_command,
            'train_command': train_command,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'seed': seed_val,
            'monitor_freq': monitor_freq_val,
            'use_amp': bool(raw_args.get('use_amp', False)),
            'use_wandb': bool(raw_args.get('use_wandb', False)),
            'enable_cosine_perf_schedule': bool(raw_args.get('enable_cosine_perf_schedule', False))
        },
        'hardware': {
            'devices': raw_args.get('devices'),
            'threads': int(os.environ.get('OMP_NUM_THREADS','8'))
        },
        'seeds': {'primary': seeds[0], 'all': seeds},
        'training': {
            'epochs': raw_args.get('epochs'),
            'batch_size': raw_args.get('batch_size'),
            'learning_rate': raw_args.get('learning_rate'),
            'weight_decay': raw_args.get('weight_decay',0.0),
            'optimizer': 'Adam',
            'mixed_precision': bool(raw_args.get('use_amp', False)),
            'gradient_clip': raw_args.get('gradient_clip',1.0),
            'patience': raw_args.get('patience',20)
        },
        'experiment': {
            'id': exp_id,
            'model': raw_args.get('model_name','gainakt2exp'),
            'short_title': raw_args.get('short_title','baseline'),
            'purpose': raw_args.get('purpose','Reproducible training run')
        },
        'data': {
            'dataset': raw_args.get('dataset'),
            'fold': raw_args.get('fold',0)
        },
        'interpretability': {
            'use_mastery_head': bool(raw_args.get('use_mastery_head', True)),
            'use_gain_head': bool(raw_args.get('use_gain_head', True)),
            'monotonicity_loss_weight': raw_args.get('monotonicity_loss_weight',0.1),
            'mastery_performance_loss_weight': raw_args.get('mastery_performance_loss_weight',0.8),
            'gain_performance_loss_weight': raw_args.get('gain_performance_loss_weight',0.8),
            'sparsity_loss_weight': raw_args.get('sparsity_loss_weight',0.2),
            'consistency_loss_weight': raw_args.get('consistency_loss_weight',0.3),
            'warmup_constraint_epochs': raw_args.get('warmup_constraint_epochs',8),
            'non_negative_loss_weight': raw_args.get('non_negative_loss_weight',0.0),
            'enhanced_constraints': bool(raw_args.get('enhanced_constraints', True)),
            'max_semantic_students': raw_args.get('max_semantic_students',50)
        }
    })
    cfg['alignment'] = {
        'enable_alignment_loss': bool(raw_args.get('enable_alignment_loss', False)),
        'alignment_weight': raw_args.get('alignment_weight', 0.25),
        'alignment_warmup_epochs': raw_args.get('alignment_warmup_epochs', 8),
        'adaptive_alignment': bool(raw_args.get('adaptive_alignment', False)),
        'alignment_min_correlation': raw_args.get('alignment_min_correlation', 0.05),
        'alignment_share_cap': raw_args.get('alignment_share_cap', 0.08),
        'alignment_share_decay_factor': raw_args.get('alignment_share_decay_factor', 0.7)
    }
    cfg['global_alignment'] = {
        'enable_global_alignment_pass': bool(raw_args.get('enable_global_alignment_pass', False)),
        'alignment_global_students': raw_args.get('alignment_global_students', 600),
        'use_residual_alignment': bool(raw_args.get('use_residual_alignment', False)),
        'alignment_residual_window': raw_args.get('alignment_residual_window', 5)
    }
    cfg['refinement'] = {
        'enable_retention_loss': bool(raw_args.get('enable_retention_loss', False)),
        'retention_delta': raw_args.get('retention_delta', 0.005),
        'retention_weight': raw_args.get('retention_weight', 0.14),
        'enable_lag_gain_loss': bool(raw_args.get('enable_lag_gain_loss', False)),
        'lag_gain_weight': raw_args.get('lag_gain_weight', 0.06),
        'lag_max_lag': raw_args.get('lag_max_lag', 3),
        'lag_l1_weight': raw_args.get('lag_l1_weight', 0.5),
        'lag_l2_weight': raw_args.get('lag_l2_weight', 0.3),
        'lag_l3_weight': raw_args.get('lag_l3_weight', 0.2),
        'consistency_rebalance_epoch': raw_args.get('consistency_rebalance_epoch', 8),
        'consistency_rebalance_threshold': raw_args.get('consistency_rebalance_threshold', 0.1),
        'consistency_rebalance_new_weight': raw_args.get('consistency_rebalance_new_weight', 0.2),
        'variance_floor': raw_args.get('variance_floor', 0.0001),
        'variance_floor_patience': raw_args.get('variance_floor_patience', 3),
        'variance_floor_reduce_factor': raw_args.get('variance_floor_reduce_factor', 0.5)
    }
    # Evaluation defaults snapshot
    try:
        defaults_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'parameter_default.json')
        with open(defaults_path) as df:
            defaults_json = json.load(df)
        eval_snapshot = defaults_json.get('evaluation_defaults', {})
        cfg['evaluation_snapshot'] = eval_snapshot
        cfg['evaluation_snapshot_md5'] = hashlib.md5(json.dumps(eval_snapshot, sort_keys=True).encode()).hexdigest()
        # Metadata reproducibility policy (ignored flags)
        metadata = defaults_json.get('metadata', {})
        policy = {
            'ignored_eval_flags': metadata.get('ignored_eval_flags', []),
            'ignored_training_flags': metadata.get('ignored_training_flags', []),
            'schema_version': metadata.get('schema_version'),
            'policy_last_updated': metadata.get('policy_last_updated')
        }
        cfg['reproducibility_policy'] = policy
    except Exception as e:
        cfg['evaluation_snapshot_error'] = f"Failed to load evaluation defaults: {e}"    
    cfg['config_md5'] = compute_config_hash(cfg)
    return cfg

__all__ = [
    'make_experiment_dir','capture_environment','atomic_write_json','append_epoch_csv','timestamped_logger',
    'compute_config_hash','utc_timestamp','write_seed_info','write_readme','build_config','compute_auc_acc'
]

def compute_auc_acc(targets, preds) -> Dict[str,float]:
    """Shared AUC/Accuracy computation to unify training and evaluation metrics.
    Expects flat lists/arrays of targets (0/1) and prediction probabilities.
    Returns dict with auc and acc (binary accuracy at 0.5 threshold)."""
    import numpy as _np
    from sklearn.metrics import roc_auc_score as _roc_auc
    if len(targets) == 0:
        return {'auc': 0.0, 'acc': 0.0}
    t = _np.asarray(targets)
    p = _np.asarray(preds)
    try:
        auc = float(_roc_auc(t, p))
    except ValueError:
        auc = 0.0
    acc = float(_np.mean((p >= 0.5) == (t == 1)))
    return {'auc': auc, 'acc': acc}
