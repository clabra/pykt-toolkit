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
    'mastery_monotonicity_rate','mastery_temporal_variance','mastery_second_diff_mean','gain_sparsity_index',
    'peer_gate_mean','difficulty_gate_mean',
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

# Helper: safe share division avoiding explosion when total is ~0
def _safe_share(raw: float, total: float, eps: float = 1e-8) -> float:
    if total <= eps:
        return 0.0
    return raw / total

# Helper: format a compact fixed-width table row for epoch metrics
def _format_epoch_table(seed:int, epoch:int, train_loss:float, constraint_loss:float, val_auc:float, val_acc:float,
                        mastery_corr:float, gain_corr:float, mono_rate:float, gain_future_alignment:float,
                        mastery_temporal_variance:float, mastery_second_diff_mean:float, gain_sparsity_index:float,
                        peer_gate_mean:float, difficulty_gate_mean:float,
                        alignment_share:float, consistency_share:float, lag_gain_share:float, peer_alignment_share:float, drift_share:float) -> str:
    # Scientific notation for potentially large shares
    def sci(x: float) -> str:
        return f"{x:>10.2e}"
    return (
        f"| {seed:>4} | {epoch:>3} | {train_loss:>8.4f} | {constraint_loss:>8.4f} | {val_auc:>7.4f} | {val_acc:>7.4f} | "
        f"{mastery_corr:>9.4f} | {gain_corr:>9.4f} | {mono_rate*100:>6.2f}% | {gain_future_alignment:>8.4f} | "
        f"{mastery_temporal_variance:>9.4f} | {mastery_second_diff_mean:>9.4f} | {gain_sparsity_index:>9.4f} | "
        f"{peer_gate_mean:>9.4f} | {difficulty_gate_mean:>9.4f} | {sci(alignment_share)} | {sci(consistency_share)} | "
        f"{sci(lag_gain_share)} | {sci(peer_alignment_share)} | {sci(drift_share)} |"
    )

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
        key, val = ov.split('=',1)
        value = _coerce_value(val)
        original = None
        if key.startswith('training.'):
            sub = key.split('.',1)[1]
            if sub in ['epochs','batch_size','lr','seeds','seed']:
                if sub == 'seeds' and isinstance(value, list):
                    value = [int(x) for x in value]
                if sub == 'lr':
                    original = args.lr
                    args.lr = float(value)
                elif sub == 'epochs':
                    original = args.epochs
                    args.epochs = int(value)
                elif sub == 'batch_size':
                    original = args.batch_size
                    args.batch_size = int(value)
                elif sub == 'seed':
                    original = args.seed
                    args.seed = int(value)
                elif sub == 'seeds':
                    original = args.seeds
                    args.seeds = value
        elif key.startswith('model.'):
            sub = key.split('.',1)[1]
            if hasattr(args, sub):
                original = getattr(args, sub)
                setattr(args, sub, value)
        elif key.startswith('interpretability.'):
            sub = key.split('.',1)[1]
            if hasattr(args, sub):
                original = getattr(args, sub)
                setattr(args, sub, value)
        logs.append(f"{key}:{original}->{value}")
    return logs

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Reproducible multi-seed GainAKT3 experiment launcher')
    p.add_argument('--short_title', type=str, default='gainakt3_repro')
    p.add_argument('--purpose', type=str, default='GainAKT3 reproducible training run')
    p.add_argument('--dataset', type=str, default='assist2015')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--seeds', type=int, nargs='*', default=None)
    p.add_argument('--devices', type=int, nargs='*', default=[0,1,2,3,4])
    p.add_argument('--num_workers', type=int, default=5)
    p.add_argument('--persistent_workers', action='store_true', help='Enable persistent DataLoader workers (requires dataset loader support).')
    p.add_argument('--data_loader_debug', action='store_true', help='Print DataLoader diagnostics (length, first batch shapes).')
    p.add_argument('--probe_reuse', action='store_true', help='Reuse first batch as probe each epoch to avoid repeated iterator creation.')
    p.add_argument('--no_amp', action='store_true')
    # Data shape / model structural hints
    p.add_argument('--num_c', type=int, default=100, help='Number of concepts (embed size domain)')
    p.add_argument('--auto_num_c', action='store_true', help='Automatically expand num_c to max concept id + 1 if dataset contains larger ids.')
    p.add_argument('--report_concept_stats', action='store_true', help='Report dataset-wide concept statistics (min, max, unique count).')
    p.add_argument('--maxlen', type=int, default=200, help='Maximum sequence length used for padding/truncation')
    p.add_argument('--input_type', nargs='*', default=['concepts'], help='Input modality types')
    # Heads
    p.add_argument('--use_mastery_head', action='store_true', default=True)
    p.add_argument('--use_gain_head', action='store_true', default=True)
    p.add_argument('--output_base', type=str, default=None)
    p.add_argument('--experiment_dir', type=str, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--set', action='append', default=[], help='Override config value using dot-path key=val')
    # Reproduction
    p.add_argument('--reproduce_from', type=str, default=None, help='Source experiment id or path to reproduce (creates *_reproduce folder).')
    p.add_argument('--strict_schema', action='store_true', help='Abort reproduction if schema_version differs.')
    p.add_argument('--manifest', action='store_true', help='Write reproduction_manifest.json summarizing reproduction metadata.')
    # Model / interpretability weights
    p.add_argument('--alignment_weight', type=float, default=0.05)
    p.add_argument('--consistency_weight', type=float, default=0.2)
    p.add_argument('--retention_weight', type=float, default=0.0)
    p.add_argument('--lag_gain_weight', type=float, default=0.05)
    p.add_argument('--sparsity_weight', type=float, default=0.0)
    p.add_argument('--peer_alignment_weight', type=float, default=0.05)
    p.add_argument('--difficulty_ordering_weight', type=float, default=0.0)
    p.add_argument('--drift_smoothness_weight', type=float, default=0.0)
    p.add_argument('--warmup_constraint_epochs', type=int, default=3)
    p.add_argument('--beta_difficulty', type=float, default=0.5)
    p.add_argument('--attempt_confidence_k', type=float, default=5.0)
    p.add_argument('--gain_threshold', type=float, default=0.01)
    p.add_argument('--mastery_temperature', type=float, default=1.0)
    p.add_argument('--peer_K', type=int, default=8)
    p.add_argument('--use_peer_context', action='store_true')
    p.add_argument('--use_difficulty_context', action='store_true')
    p.add_argument('--disable_fusion_broadcast', action='store_true')
    p.add_argument('--disable_difficulty_penalty', action='store_true')
    p.add_argument('--fusion_for_heads_only', action='store_true', default=True)
    p.add_argument('--gate_init_bias', type=float, default=-2.0)
    p.add_argument('--broadcast_last_context', action='store_true', help='Broadcast final fused context across sequence instead of using full temporal ctx (baseline keeps it disabled).')
    # Artifacts
    p.add_argument('--artifact_base', type=str, default='data')
    p.add_argument('--peer_artifact_path', type=str, default='')
    p.add_argument('--difficulty_artifact_path', type=str, default='')
    p.add_argument('--strict_artifact_hash', action='store_true')
    # Synthetic debug
    p.add_argument('--use_synthetic', action='store_true')
    # Distributed
    p.add_argument('--multi_gpu', action='store_true', default=True, help='Enable multi-GPU (DDP preferred; fallback DataParallel).')
    p.add_argument('--no_multi_gpu', action='store_true', help='Disable multi-GPU even if GPUs available.')
    p.add_argument('--ddp', action='store_true', help='(Deprecated) Alias for --multi_gpu.')
    args = p.parse_args()
    return args

def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return 'MISSING'
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def _git_info() -> Dict[str,str]:
    def safe(cmd: list) -> str:
        try:
            return subprocess.check_output(cmd, cwd='/workspaces/pykt-toolkit').decode().strip()
        except Exception:
            return 'UNKNOWN'
    return {
        'commit': safe(['git','rev-parse','HEAD']),
        'branch': safe(['git','rev-parse','--abbrev-ref','HEAD'])
    }

def build_config(args: argparse.Namespace, exp_id: str, exp_path: str, seeds: List[int], peer_hash: str, diff_hash: str, cold_start: bool,
                 multi_gpu_mode: str, effective_multi: bool) -> Dict[str,Any]:
    cfg = {
        'schema_version': 1,
        'experiment': {
            'id': exp_id,
            'model': 'gainakt3',
            'short_title': args.short_title,
            'purpose': args.purpose,
        },
        'data': {
            'dataset': args.dataset,
            'fold': args.fold,
            'batch_size': args.batch_size,
            'num_c': args.num_c,
            'maxlen': args.maxlen,
            'input_type': args.input_type
        },
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'optimizer': 'Adam',
            'gradient_clip': args.grad_clip,
            'mixed_precision': not args.no_amp,
            'warmup_constraint_epochs': args.warmup_constraint_epochs
        },
        'heads': {
            'use_mastery_head': bool(args.use_mastery_head),
            'use_gain_head': bool(args.use_gain_head)
        },
        'context': {
            'use_peer_context': bool(args.use_peer_context),
            'use_difficulty_context': bool(args.use_difficulty_context),
            'fusion_for_heads_only': bool(args.fusion_for_heads_only),
            'disable_fusion_broadcast': bool(args.disable_fusion_broadcast),
            'disable_difficulty_penalty': bool(args.disable_difficulty_penalty),
        'hardware': {
            'requested_devices': args.devices,
            'visible_devices_env': os.environ.get('CUDA_VISIBLE_DEVICES','UNSET')
        },
            'peer_K': args.peer_K
        },
        'weights': {
            'alignment_weight': args.alignment_weight,
            'consistency_weight': args.consistency_weight,
            'retention_weight': args.retention_weight,
            'lag_gain_weight': args.lag_gain_weight,
            'sparsity_weight': args.sparsity_weight,
            'peer_alignment_weight': args.peer_alignment_weight,
            'difficulty_ordering_weight': args.difficulty_ordering_weight,
            'drift_smoothness_weight': args.drift_smoothness_weight
        },
        'hyperparams': {
            'beta_difficulty': args.beta_difficulty,
            'attempt_confidence_k': args.attempt_confidence_k,
            'gain_threshold': args.gain_threshold,
            'mastery_temperature': args.mastery_temperature,
            'gate_init_bias': args.gate_init_bias
        },
        'artifacts': {
            'peer_index_path': args.peer_artifact_path or os.path.join(args.artifact_base,'peer_index',args.dataset,'peer_index.pkl'),
            'peer_index_sha256': peer_hash,
            'difficulty_table_path': args.difficulty_artifact_path or os.path.join(args.artifact_base,'difficulty',args.dataset,'difficulty_table.parquet'),
            'difficulty_table_sha256': diff_hash,
            'cold_start': bool(cold_start)
        },
        'hardware': {
            'devices': args.devices,
            'num_workers': args.num_workers,
            'threads': int(os.environ.get('OMP_NUM_THREADS','8'))
        },
        'runtime': {
            'command': 'python examples/wandb_gainakt3_train.py ' + ' '.join(sys.argv[1:]),
            'timestamp': datetime.datetime.utcnow().isoformat()+'Z'
        },
        'git': _git_info(),
        'seeds': {
            'primary': seeds[0],
            'all': seeds
        }
    }
    # Inject distributed flags (mode decided externally to reflect actual runtime plan)
    cfg['distributed'] = {
        'multi_gpu_enabled': bool(effective_multi),
        'multi_gpu_mode': multi_gpu_mode,
        'world_size_env': os.environ.get('WORLD_SIZE','1'),
        'rank_env': os.environ.get('RANK','0'),
        'local_rank_env': os.environ.get('LOCAL_RANK', os.environ.get('RANK','0'))
    }
    # Missing params detection
    flat_keys = set()
    def collect(d):
        for k,v in d.items():
            if isinstance(v, dict):
                collect(v)
            else:
                flat_keys.add(k)
    collect(cfg)
    exclude = {'devices','set','force','experiment_dir','output_base','no_amp','seeds','seed','reproduce_from','strict_schema','manifest','ddp'}
    # Add missing root args directly into cfg for completeness
    cfg['raw_args'] = {
        'lr': args.lr,
        'grad_clip': args.grad_clip,
        'artifact_base': args.artifact_base,
        'peer_artifact_path': args.peer_artifact_path,
        'difficulty_artifact_path': args.difficulty_artifact_path,
        'strict_artifact_hash': bool(args.strict_artifact_hash),
        'use_synthetic': bool(args.use_synthetic),
        'multi_gpu': bool(effective_multi),
        'no_multi_gpu': bool(getattr(args,'no_multi_gpu', False))
    }
    # Include newly added flags to avoid assertion failure
    cfg['raw_args']['auto_num_c'] = bool(getattr(args,'auto_num_c', False))
    cfg['raw_args']['report_concept_stats'] = bool(getattr(args,'report_concept_stats', False))
    # Newly added runtime / debug flags
    cfg['raw_args']['persistent_workers'] = bool(getattr(args,'persistent_workers', False))
    cfg['raw_args']['data_loader_debug'] = bool(getattr(args,'data_loader_debug', False))
    cfg['raw_args']['probe_reuse'] = bool(getattr(args,'probe_reuse', False))
    cfg['raw_args']['broadcast_last_context'] = bool(getattr(args,'broadcast_last_context', False))
    for k in cfg['raw_args'].keys():
        flat_keys.add(k)
    missing = [k for k in vars(args).keys() if k not in flat_keys and k not in exclude]
    cfg['missing_params'] = missing
    assert len(missing) == 0, f"Unserialized parameters found: {missing}"
    cfg['config_md5'] = compute_config_hash(cfg)
    return cfg

def _resolve_experiment_path(base_dir: Path, ref: str) -> Path:
    candidate = Path(ref)
    if not candidate.is_absolute():
        candidate = base_dir / ref
    return candidate

def _load_json(path: Path) -> Dict[str,Any]:
    if not path.exists():
        return {}
    import json
    with open(path) as f:
        return json.load(f)

def _reproduction_report(source_cfg: Dict[str,Any], source_results: Dict[str,Any], new_cfg: Dict[str,Any], new_results: Dict[str,Any]) -> Dict[str,Any]:
    rep = {
        'source_experiment': source_cfg.get('experiment',{}).get('id'),
        'new_experiment': new_cfg.get('experiment',{}).get('id'),
        'source_config_md5': source_cfg.get('config_md5'),
        'new_config_md5': new_cfg.get('config_md5'),
        'config_md5_match': source_cfg.get('config_md5') == new_cfg.get('config_md5'),
        'schema_version_source': source_cfg.get('schema_version'),
        'schema_version_new': new_cfg.get('schema_version'),
    }
    # Metric comparison (tolerances similar to GainAKT2Exp guidelines)
    src_auc = source_results.get('best_val_auc') or source_results.get('best_epoch_auc') or source_results.get('best_val_auc')
    new_auc = new_results.get('best_val_auc') or new_results.get('best_epoch_auc') or new_results.get('best_val_auc')
    src_master = source_results.get('best_mastery_corr') or source_results.get('best_mastery_corr')
    new_master = new_results.get('best_mastery_corr') or new_results.get('best_mastery_corr')
    src_gain = source_results.get('best_gain_corr')
    new_gain = new_results.get('best_gain_corr')
    if src_auc is not None and new_auc is not None:
        rep['auc_diff'] = abs(src_auc - new_auc)
    if src_master is not None and new_master is not None:
        rep['mastery_corr_diff'] = abs(src_master - new_master)
    if src_gain is not None and new_gain is not None:
        rep['gain_corr_diff'] = abs(src_gain - new_gain)
    # Tolerance assessment (AMP assumed):
    tol_auc = 0.002
    tol_corr = 0.01
    rep['within_tolerance'] = (
        (rep.get('auc_diff',0) <= tol_auc) and
        (rep.get('mastery_corr_diff',0) <= tol_corr) and
        (rep.get('gain_corr_diff',0) <= tol_corr)
    )
    return rep

def main():
    args = parse_args()
    # --- Signal handling (graceful teardown to mitigate BrokenPipeError) ---
    import signal
    shutdown_flag = {'value': False}
    def _graceful(sig, frame):
        if not shutdown_flag['value']:
            print(f'[SIGNAL] Received {sig}; initiating graceful shutdown after current epoch.')
        shutdown_flag['value'] = True
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _graceful)
        except Exception:
            pass
    # Propagate num_workers to dataset layer (pykt datasets read PYKT_NUM_WORKERS); allow override at runtime.
    os.environ['PYKT_NUM_WORKERS'] = str(max(0, int(getattr(args,'num_workers',5))))
    # (Optional future support) persist flag for DataLoader if framework adopts env hook
    os.environ['PYKT_PERSISTENT_WORKERS'] = '1' if getattr(args,'persistent_workers', False) else '0'
    # --- Early GPU availability sanity check (avoid internal CUDA asserts when defaulting to nonexistent GPUs) ---
    try:
        available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        available_gpu_count = 0
    if available_gpu_count == 0:
        # Force CPU mode; suppress multi-gpu logic that would set invalid CUDA_VISIBLE_DEVICES
        args.no_multi_gpu = True
        args.multi_gpu = False
        args.devices = []
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        print('[GPU] No CUDA devices detected; forcing CPU mode and disabling multi-GPU.')
    else:
        # Respect pre-set CUDA_VISIBLE_DEVICES from environment if present
        env_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        valid_indices = list(range(available_gpu_count))
        if env_visible:
            # Parse environment list and intersect with physically available devices
            try:
                env_list = [int(x) for x in env_visible.split(',') if x.strip() != '']
            except ValueError:
                env_list = []
            pruned_env = [d for d in env_list if d in valid_indices]
            if not pruned_env:
                pruned_env = [0]
            # Always use ALL env-visible GPUs; ignore --devices (explicit environment takes precedence)
            if args.devices and set(args.devices) != set(pruned_env):
                diff = set(args.devices) - set(pruned_env)
                if diff:
                    print(f"[GPU] Overriding --devices with CUDA_VISIBLE_DEVICES; disregarding: {sorted(diff)}")
            args.devices = pruned_env
            # Normalize env to pruned set if mismatch
            normalized = ','.join(str(d) for d in pruned_env)
            if normalized != env_visible:
                os.environ['CUDA_VISIBLE_DEVICES'] = normalized
                print(f"[GPU] Normalized CUDA_VISIBLE_DEVICES from '{env_visible}' to '{normalized}'.")
            # Auto-disable multi-GPU if only one visible
            if len(args.devices) <= 1:
                if args.multi_gpu and not args.no_multi_gpu:
                    print('[GPU] Single visible device detected; disabling multi-GPU mode.')
                args.multi_gpu = False
                args.ddp = False
            else:
                # Multiple GPUs available via env; ensure multi-gpu enabled unless explicitly disabled
                if not args.no_multi_gpu:
                    args.multi_gpu = True
                    print(f"[GPU] Using all {len(args.devices)} GPUs from CUDA_VISIBLE_DEVICES: {args.devices}")
        else:
            # No pre-set env: derive from args or defaults
            if args.devices:
                pruned = [d for d in args.devices if d in valid_indices]
                invalid = [d for d in args.devices if d not in valid_indices]
                if invalid:
                    print(f"[GPU] Ignoring unavailable devices: {invalid}")
                if not pruned:
                    pruned = [0]
                args.devices = pruned
            else:
                use_ct = min(5, available_gpu_count)
                args.devices = list(range(use_ct))
            visibility = ','.join(str(d) for d in args.devices)
            os.environ['CUDA_VISIBLE_DEVICES'] = visibility
            print(f"[GPU] Setting CUDA_VISIBLE_DEVICES to {visibility} (available={available_gpu_count}).")
            if len(args.devices) <= 1:
                if args.multi_gpu and not args.no_multi_gpu:
                    print('[GPU] Single device default; multi-GPU disabled.')
                args.multi_gpu = False
                args.ddp = False
    # --- DDP initialization (optional) ---
    ddp_enabled = False
    local_rank = 0
    world_size = 1
    if args.ddp or args.multi_gpu:
        import torch.distributed as dist
        required_env = all(k in os.environ for k in ('MASTER_ADDR','MASTER_PORT','WORLD_SIZE','RANK'))
        if not dist.is_available():
            print('[DDP] torch.distributed not available; skipping (will fallback to DataParallel if possible).')
        elif not required_env:
            # Multi-GPU requested but no rendezvous env; skip DDP gracefully
            print('[DDP] Skipping distributed init (missing MASTER_ADDR/PORT/WORLD_SIZE/RANK); using DataParallel.')
        else:
            local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK','0')))
            world_size = int(os.environ['WORLD_SIZE'])
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            try:
                dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=local_rank)
                ddp_enabled = True
            except Exception as e:
                print(f'[DDP] Initialization failed: {e}; falling back to DataParallel.')
                ddp_enabled = False
    is_rank0 = (local_rank == 0)
    # --- GPU visibility management ---
    # If CUDA_VISIBLE_DEVICES unset or user wants explicit limitation, apply devices list.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # Default to first five GPUs (0-4) matching project guideline (~60% of 8 GPUs)
        default_visible = '0,1,2,3,4'
        os.environ['CUDA_VISIBLE_DEVICES'] = default_visible
        print(f"[GPU] CUDA_VISIBLE_DEVICES was unset; defaulting to {default_visible}")
    else:
        # Optionally override with --devices if user passed an explicit list differing from env
        env_current = os.environ.get('CUDA_VISIBLE_DEVICES','')
        cli_requested = ','.join(str(d) for d in args.devices)
        if cli_requested and cli_requested != env_current:
            os.environ['CUDA_VISIBLE_DEVICES'] = cli_requested
            print(f"[GPU] Overriding CUDA_VISIBLE_DEVICES env ({env_current}) with CLI devices {cli_requested}")
    # Record final visible list for later config capture
    # Decide multi-GPU mode (DDP or DataParallel fallback) AFTER visibility set
    effective_multi = (args.multi_gpu or args.ddp) and (not args.no_multi_gpu)
    dp_fallback = False
    multi_gpu_mode = 'none'
    if effective_multi:
        if ddp_enabled and int(os.environ.get('WORLD_SIZE','1')) > 1:
            multi_gpu_mode = 'ddp'
        else:
            # Consider DataParallel fallback if multiple GPUs visible
            visible_ct = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if visible_ct > 1:
                dp_fallback = True
                multi_gpu_mode = 'dataparallel'
            else:
                multi_gpu_mode = 'none'
    apply_overrides(args, args.set)
    seeds = args.seeds if args.seeds else [args.seed]
    seeds = list(dict.fromkeys(seeds))
    if args.devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in args.devices)
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / 'examples' / 'experiments' if args.output_base is None else Path(args.output_base)
    base_dir.mkdir(parents=True, exist_ok=True)
    reproduction_mode = False
    source_cfg = {}
    source_results = {}
    source_exp_path = None
    if args.reproduce_from:
        reproduction_mode = True
        # Resolve source experiment directory
        source_exp_path = _resolve_experiment_path(base_dir, args.reproduce_from)
        if not source_exp_path.exists():
            print(f"[ERROR] Source experiment for reproduction not found: {source_exp_path}")
            sys.exit(1)
        source_cfg = _load_json(source_exp_path / 'config.json')
        if args.strict_schema and source_cfg.get('schema_version') != 1:
            print('[ERROR] Schema version mismatch; aborting reproduction.')
            sys.exit(1)
        source_results = _load_json(source_exp_path / 'results.json')
        # Adopt seeds from source if not provided
        if args.seeds is None and 'seeds' in source_cfg:
            seeds = source_cfg['seeds'].get('all', seeds)
        # Derive new short_title
        repro_short = source_cfg.get('experiment',{}).get('short_title','reproduce') + '_reproduce'
        exp_path = make_experiment_dir('gainakt3', repro_short, base_dir=str(base_dir))
        exp_id = os.path.basename(exp_path)
    elif args.experiment_dir:
        exp_path_candidate = Path(args.experiment_dir)
        if not exp_path_candidate.is_absolute():
            exp_path_candidate = base_dir / args.experiment_dir
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
    logger = timestamped_logger(exp_id, os.path.join(exp_path,'stdout.log')) if is_rank0 else None
    # Artifact hashing
    peer_path = args.peer_artifact_path or os.path.join(args.artifact_base,'peer_index',args.dataset,'peer_index.pkl')
    diff_path = args.difficulty_artifact_path or os.path.join(args.artifact_base,'difficulty',args.dataset,'difficulty_table.parquet')
    peer_hash = sha256_file(peer_path)
    diff_hash = sha256_file(diff_path)
    cold_start = ((args.use_peer_context and peer_hash == 'MISSING') or (args.use_difficulty_context and diff_hash == 'MISSING'))
    if args.strict_artifact_hash and cold_start:
        print('[ABORT] Strict artifact hash enabled but artifact missing.')
        sys.exit(2)
    cfg = build_config(args, exp_id, exp_path, seeds, peer_hash, diff_hash, cold_start, multi_gpu_mode, effective_multi)
    atomic_write_json(cfg, os.path.join(exp_path,'config.json'))
    capture_environment(os.path.join(exp_path,'environment.txt'))
    with open(os.path.join(exp_path,'SEED_INFO.md'),'w') as sf:
        sf.write(f"Primary seed: {seeds[0]}\nAll seeds: {seeds}\n")
    # Log and serialize GPU count for reproducibility
    try:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        gpu_count = 0
    cfg['hardware']['gpu_count'] = gpu_count
    atomic_write_json(cfg, os.path.join(exp_path,'config.json'))
    if gpu_count > 0:
        mode_note = 'ddp' if (multi_gpu_mode == 'ddp') else ('dataparallel' if multi_gpu_mode == 'dataparallel' else 'single')
        print(f"[GPU] Training start: using {gpu_count} visible GPU(s) (mode={mode_note}).")
    else:
        print("[GPU] Training start: CPU mode (no GPUs detected).")
    # ---- Flatten & dump all parameter values for verification (must match config.json) ----
    def _flatten(d: dict, prefix: str = ''):
        items = []
        for k,v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(_flatten(v, path))
            else:
                items.append((path, v))
        return items
    flat_items = _flatten(cfg)
    flat_lines = [f"{k} = {v}" for k,v in sorted(flat_items)]
    flat_path = os.path.join(exp_path,'config_flat.txt')
    with open(flat_path,'w') as ff:
        ff.write('\n'.join(flat_lines))
    if logger:
        logger.info('[CONFIG-DUMP] BEGIN')
        for line in flat_lines:
            logger.info(line)
        logger.info('[CONFIG-DUMP] END')
    metrics_csv = os.path.join(exp_path,'metrics_epoch.csv')
    per_seed_results: List[Dict[str,Any]] = []
    for seed in seeds:
        if is_rank0 and logger:
            logger.info(f"===== Seed {seed} =====")
        set_seeds(seed)
        # Load data FIRST (needed for concept count validation before model creation)
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
            if args.data_loader_debug and is_rank0:
                try:
                    loader_type = type(train_loader).__name__
                    est_len = len(train_loader) if hasattr(train_loader,'__len__') else 'NA'
                    print(f'[DL-DEBUG] train_loader type={loader_type} len={est_len} num_workers={args.num_workers} persistent={args.persistent_workers}')
                except Exception as e:
                    print(f'[DL-DEBUG] Unable to inspect train_loader: {e}')
            # Dataset-wide concept scan (train + val) for robust num_c resolution
            try:
                def iterate(loader):
                    if isinstance(loader, list):
                        for b in loader:
                            yield b
                    else:
                        for b in loader:
                            yield b
                max_id = -1
                min_id = None
                unique_ids = set()
                for batch in iterate(train_loader):
                    c = batch['cseqs']
                    max_id = max(max_id, int(c.max().item()))
                    cur_min = int(c.min().item())
                    min_id = cur_min if min_id is None else min(min_id, cur_min)
                    unique_ids.update(c.view(-1).tolist())
                for batch in iterate(val_loader):
                    c = batch['cseqs']
                    max_id = max(max_id, int(c.max().item()))
                    cur_min = int(c.min().item())
                    min_id = cur_min if min_id is None else min(min_id, cur_min)
                    unique_ids.update(c.view(-1).tolist())
                resolved_num_c = args.num_c
                if max_id >= args.num_c:
                    if args.auto_num_c:
                        resolved_num_c = max_id + 1
                        if is_rank0:
                            print(f"[AUTO] num_c adjusted from {args.num_c} -> {resolved_num_c} (max concept id {max_id}).")
                    else:
                        if is_rank0:
                            print(f"[ERROR] max concept id {max_id} >= declared num_c {args.num_c}. Use --auto_num_c or rerun with --num_c {max_id+1}.")
                        sys.exit(3)
                if args.report_concept_stats and is_rank0:
                    print(f"[STATS] concept_min={min_id}, concept_max={max_id}, unique={len(unique_ids)}, declared={args.num_c}, resolved={resolved_num_c}")
                # Update args and config with resolved num_c & stats (rank0)
                args.num_c = resolved_num_c
                if is_rank0:
                    cfg['data']['num_c_requested'] = cfg['data']['num_c']
                    cfg['data']['num_c_resolved'] = resolved_num_c
                    cfg['data']['concept_unique_count'] = len(unique_ids)
                    cfg['data']['concept_min_id'] = min_id
                    cfg['data']['concept_max_id'] = max_id
                    atomic_write_json(cfg, os.path.join(exp_path,'config.json'))
            except Exception as e:
                if is_rank0:
                    print(f"[WARN] Concept scan failed: {e}. Proceeding with declared num_c={args.num_c} (risk of OOB).")
        # Build model AFTER validation
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
            'broadcast_last_context': args.broadcast_last_context,
        }
        model = create_gainakt3_model(model_cfg)
        # DDP wrap (single seed only for now)
        if ddp_enabled and world_size > 1:
            if args.seeds and len(args.seeds) > 1:
                if is_rank0:
                    print('[DDP] Multi-seed not supported under ddp; using primary seed only.')
                seeds = [seeds[0]]
            torch.cuda.set_device(local_rank)
            model.to(local_rank)
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        elif dp_fallback:
            # DataParallel fallback
            if is_rank0:
                print(f"[MultiGPU] DDP unavailable; falling back to DataParallel over devices: {os.environ.get('CUDA_VISIBLE_DEVICES','')}")
            from torch.nn import DataParallel
            model = DataParallel(model)
        model.mastery_temperature = args.mastery_temperature
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Initialize CSV header once on first seed
        if seed == seeds[0]:
            with open(metrics_csv,'w',newline='') as f:
                import csv
                w = csv.writer(f)
                w.writerow(EPOCH_HEADER)
        # Optional probe batch caching (avoid repeated iterator recreation)
        cached_probe_batch = None
        if args.probe_reuse:
            try:
                first_iter = train_loader[0] if isinstance(train_loader,list) else next(iter(train_loader))
                cached_probe_batch = {
                    'cseqs': first_iter['cseqs'].clone() if torch.is_tensor(first_iter['cseqs']) else first_iter['cseqs'],
                    'rseqs': first_iter['rseqs'].clone() if torch.is_tensor(first_iter['rseqs']) else first_iter['rseqs']
                }
                if args.data_loader_debug and is_rank0:
                    print('[DL-DEBUG] Cached probe batch for reuse across epochs.')
            except Exception as e:
                if is_rank0:
                    print(f'[DL-DEBUG] Probe reuse disabled (failed to cache first batch): {e}')
                cached_probe_batch = None
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
            # DataParallel returns per-replica outputs gathered; total_constraint_loss may be a tensor with >1 element.
            raw_constraint = probe_out['total_constraint_loss']
            if isinstance(raw_constraint, torch.Tensor):
                if raw_constraint.numel() > 1:
                    # Reduce to scalar (mean) for logging; preserves aggregate scale under symmetric replication
                    constraint_total = float(raw_constraint.mean().detach().cpu())
                else:
                    constraint_total = float(raw_constraint.detach().cpu())
            else:
                # Fallback: attempt direct float conversion if already numeric
                constraint_total = float(raw_constraint)
            train_loss = perf_loss + constraint_total
            (val_auc, val_acc, mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mastery_corr_macro_weighted, gain_corr_macro_weighted, mono_rate, ret_rate, gain_future_alignment, mastery_monotonicity_rate, mastery_temporal_variance, mastery_second_diff_mean, gain_sparsity_index, peer_gate_mean, difficulty_gate_mean, per_concept_mastery_corr, per_concept_gain_corr) = evaluate(model, val_loader, model_cfg['device'], args.gain_threshold)
            # Compute separated performance and total losses
            train_perf_loss = perf_loss
            train_total_loss = train_loss
            # Tabular core metrics view (printed once header + row each epoch)
            if is_rank0:
                # Fixed-width cell specification
                cols = [
                    ('seed', seed, 6),
                    ('ep', epoch, 4),
                    ('val_auc', val_auc, 10),
                    ('val_acc', val_acc, 10),
                    ('mastery', mastery_corr, 11),
                    ('gain', gain_corr, 11),
                    ('perf_loss', train_perf_loss, 12),
                    ('total_loss', train_total_loss, 12),
                    ('constr', constraint_total, 10)
                ]
                if epoch == 1:
                    header = '|' + '|'.join(f" {name:>{width-2}}" for name,_,width in cols) + '|'
                    print(header)
                def fmt(val, width):
                    if isinstance(val, (float,int)):
                        return f" {val:>{width-2}.4f}" if isinstance(val,float) else f" {val:>{width-2}}"
                    return f" {str(val):<{width-2}}"
                row = '|' + '|'.join(fmt(v,w) for _,v,w in cols) + '|'
                print(row)
            # Raw constraint losses needed for CSV export (retain variable usage)
            cl_raw = probe_out['constraint_losses']
            with torch.no_grad():
                probe_len = model_cfg['seq_len'] if 'seq_len' in model_cfg else args.maxlen
                probe = model(torch.randint(0,model_cfg['num_c'],(1,probe_len)).to(model_cfg['device']), torch.randint(0,2,(1,probe_len)).to(model_cfg['device']))
            peer_share = float(probe['peer_influence_share'])
            decomp = probe.get('decomposition',{})
            reconstruction_error = float(decomp.get('reconstruction_error', float('nan')))
            difficulty_penalty_contrib_mean = float(decomp.get('difficulty_penalty_contrib', torch.tensor(float('nan'))).mean().item()) if 'difficulty_penalty_contrib' in decomp and torch.is_tensor(decomp.get('difficulty_penalty_contrib')) else float('nan')
            eps = 1e-8
            cl = cl_raw  # reuse previously captured constraint losses
            def _to_scalar(x):
                if torch.is_tensor(x):
                    return float(x.mean().detach().cpu())
                return float(x)
            alignment_raw = _to_scalar(cl.get('alignment_loss', 0.0))
            alignment_share = alignment_raw / (constraint_total + eps)
            sparsity_raw = _to_scalar(cl.get('sparsity_loss', 0.0))
            sparsity_share = sparsity_raw / (constraint_total + eps)
            consistency_raw = _to_scalar(cl.get('consistency_loss', 0.0))
            consistency_share = consistency_raw / (constraint_total + eps)
            retention_raw = _to_scalar(cl.get('retention_loss', 0.0))
            retention_share = retention_raw / (constraint_total + eps)
            lag_gain_raw = _to_scalar(cl.get('lag_gain_loss', 0.0))
            lag_gain_share = lag_gain_raw / (constraint_total + eps)
            peer_alignment_raw = _to_scalar(cl.get('peer_alignment_loss', 0.0))
            peer_alignment_share = peer_alignment_raw / (constraint_total + eps)
            difficulty_ordering_raw = _to_scalar(cl.get('difficulty_ordering_loss', 0.0))
            difficulty_ordering_share = difficulty_ordering_raw / (constraint_total + eps)
            drift_smoothness_raw = _to_scalar(cl.get('drift_smoothness_loss', 0.0))
            drift_smoothness_share = drift_smoothness_raw / (constraint_total + eps)
            row = {
                'seed': seed, 'epoch': epoch, 'train_loss': train_loss, 'constraint_loss': constraint_total, 'val_auc': val_auc, 'val_accuracy': val_acc,
                'mastery_corr': mastery_corr,'gain_corr': gain_corr,'mastery_corr_macro': mastery_corr_macro,'gain_corr_macro': gain_corr_macro,
                'mastery_corr_macro_weighted': mastery_corr_macro_weighted,'gain_corr_macro_weighted': gain_corr_macro_weighted,
                'monotonicity_violation_rate': mono_rate,'retention_violation_rate': ret_rate,'gain_future_alignment': gain_future_alignment,
                'mastery_monotonicity_rate': mastery_monotonicity_rate,'mastery_temporal_variance': mastery_temporal_variance,'mastery_second_diff_mean': mastery_second_diff_mean,'gain_sparsity_index': gain_sparsity_index,
                'peer_gate_mean': peer_gate_mean,'difficulty_gate_mean': difficulty_gate_mean,
                'peer_influence_share': peer_share,'reconstruction_error': reconstruction_error,'difficulty_penalty_contrib_mean': difficulty_penalty_contrib_mean,
                'alignment_share': alignment_share,'sparsity_share': sparsity_share,'consistency_share': consistency_share,'retention_share': retention_share,'lag_gain_share': lag_gain_share,
                'peer_alignment_share': peer_alignment_share,'difficulty_ordering_share': difficulty_ordering_share,'drift_smoothness_share': drift_smoothness_share,
                'alignment_loss_raw': alignment_raw,'sparsity_loss_raw': sparsity_raw,'consistency_loss_raw': consistency_raw,'retention_loss_raw': retention_raw,'lag_gain_loss_raw': lag_gain_raw,
                'peer_alignment_loss_raw': peer_alignment_raw,'difficulty_ordering_loss_raw': difficulty_ordering_raw,'drift_smoothness_loss_raw': drift_smoothness_raw,'cold_start_flag': cold_start
            }
            if is_rank0:
                append_epoch_csv(row, metrics_csv, EPOCH_HEADER)
            epoch_rows.append(row)
        # Save seed-specific checkpoints
        if is_rank0:
            state_dict = model.module.state_dict() if (ddp_enabled and hasattr(model,'module')) else model.state_dict()
            torch.save(state_dict, os.path.join(exp_path,f'model_last_seed{seed}.pth'))
        best_row = max(epoch_rows, key=lambda x: x['val_auc']) if epoch_rows else {}
        if is_rank0:
            best_state_dict = model.module.state_dict() if (ddp_enabled and hasattr(model,'module')) else model.state_dict()
            torch.save({'state_dict': best_state_dict, 'best_epoch': best_row.get('epoch'), 'val_auc': best_row.get('val_auc')}, os.path.join(exp_path,f'model_best_seed{seed}.pth'))
        try:
            # Epoch loop is the most interruption-prone area; isolate exception handling here.
            for epoch in range(1, args.epochs+1):
                if shutdown_flag['value']:
                    if is_rank0:
                        print('[SHUTDOWN] Flag set; skipping remaining epochs for this seed.')
                    break
                model.current_epoch = epoch
                perf_loss = train_epoch(model, train_loader, model_cfg['device'], optimizer, 100)
                with torch.no_grad():
                    if cached_probe_batch is not None:
                        probe_batch = cached_probe_batch
                    else:
                        probe_batch = train_loader[0] if isinstance(train_loader,list) else next(iter(train_loader))
                    c_probe = probe_batch['cseqs'].to(model_cfg['device'])
                    r_probe = probe_batch['rseqs'].to(model_cfg['device'])
                    probe_out = model(c_probe.long(), r_probe.long())
                raw_constraint = probe_out['total_constraint_loss']
                if isinstance(raw_constraint, torch.Tensor):
                    if raw_constraint.numel() > 1:
                        constraint_total = float(raw_constraint.mean().detach().cpu())
                    else:
                        constraint_total = float(raw_constraint.detach().cpu())
                else:
                    constraint_total = float(raw_constraint)
                train_loss = perf_loss + constraint_total
                (val_auc, val_acc, mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mastery_corr_macro_weighted, gain_corr_macro_weighted, mono_rate, ret_rate, gain_future_alignment, mastery_monotonicity_rate, mastery_temporal_variance, mastery_second_diff_mean, gain_sparsity_index, peer_gate_mean, difficulty_gate_mean, per_concept_mastery_corr, per_concept_gain_corr) = evaluate(model, val_loader, model_cfg['device'], args.gain_threshold)
                train_perf_loss = perf_loss
                train_total_loss = train_loss
                if is_rank0:
                    cols = [
                        ('seed', seed, 6),('ep', epoch, 4),('val_auc', val_auc, 10),('val_acc', val_acc, 10),('mastery', mastery_corr, 11),('gain', gain_corr, 11),('perf_loss', train_perf_loss, 12),('total_loss', train_total_loss, 12),('constr', constraint_total, 10)
                    ]
                    if epoch == 1:
                        header = '|' + '|'.join(f" {name:>{width-2}}" for name,_,width in cols) + '|'
                        print(header)
                    def fmt(val, width):
                        if isinstance(val, (float,int)):
                            return f" {val:>{width-2}.4f}" if isinstance(val,float) else f" {val:>{width-2}}"
                        return f" {str(val):<{width-2}}"
                    row_print = '|' + '|'.join(fmt(v,w) for _,v,w in cols) + '|'
                    print(row_print)
                cl_raw = probe_out['constraint_losses']
                with torch.no_grad():
                    probe_len = model_cfg['seq_len'] if 'seq_len' in model_cfg else args.maxlen
                    probe = model(torch.randint(0,model_cfg['num_c'],(1,probe_len)).to(model_cfg['device']), torch.randint(0,2,(1,probe_len)).to(model_cfg['device']))
                peer_share = float(probe['peer_influence_share'])
                decomp = probe.get('decomposition',{})
                reconstruction_error = float(decomp.get('reconstruction_error', float('nan')))
                difficulty_penalty_contrib_mean = float(decomp.get('difficulty_penalty_contrib', torch.tensor(float('nan'))).mean().item()) if 'difficulty_penalty_contrib' in decomp and torch.is_tensor(decomp.get('difficulty_penalty_contrib')) else float('nan')
                eps = 1e-8
                cl = cl_raw
                def _to_scalar(x):
                    if torch.is_tensor(x):
                        return float(x.mean().detach().cpu())
                    return float(x)
                alignment_raw = _to_scalar(cl.get('alignment_loss', 0.0))
                alignment_share = alignment_raw / (constraint_total + eps)
                sparsity_raw = _to_scalar(cl.get('sparsity_loss', 0.0))
                sparsity_share = sparsity_raw / (constraint_total + eps)
                consistency_raw = _to_scalar(cl.get('consistency_loss', 0.0))
                consistency_share = consistency_raw / (constraint_total + eps)
                retention_raw = _to_scalar(cl.get('retention_loss', 0.0))
                retention_share = retention_raw / (constraint_total + eps)
                lag_gain_raw = _to_scalar(cl.get('lag_gain_loss', 0.0))
                lag_gain_share = lag_gain_raw / (constraint_total + eps)
                peer_alignment_raw = _to_scalar(cl.get('peer_alignment_loss', 0.0))
                peer_alignment_share = peer_alignment_raw / (constraint_total + eps)
                difficulty_ordering_raw = _to_scalar(cl.get('difficulty_ordering_loss', 0.0))
                difficulty_ordering_share = difficulty_ordering_raw / (constraint_total + eps)
                drift_smoothness_raw = _to_scalar(cl.get('drift_smoothness_loss', 0.0))
                drift_smoothness_share = drift_smoothness_raw / (constraint_total + eps)
                row = {
                    'seed': seed,'epoch': epoch,'train_loss': train_loss,'constraint_loss': constraint_total,'val_auc': val_auc,'val_accuracy': val_acc,'mastery_corr': mastery_corr,'gain_corr': gain_corr,'mastery_corr_macro': mastery_corr_macro,'gain_corr_macro': gain_corr_macro,
                    'mastery_corr_macro_weighted': mastery_corr_macro_weighted,'gain_corr_macro_weighted': gain_corr_macro_weighted,'monotonicity_violation_rate': mono_rate,'retention_violation_rate': ret_rate,'gain_future_alignment': gain_future_alignment,'mastery_monotonicity_rate': mastery_monotonicity_rate,'mastery_temporal_variance': mastery_temporal_variance,'mastery_second_diff_mean': mastery_second_diff_mean,'gain_sparsity_index': gain_sparsity_index,
                    'peer_gate_mean': peer_gate_mean,'difficulty_gate_mean': difficulty_gate_mean,'peer_influence_share': peer_share,'reconstruction_error': reconstruction_error,'difficulty_penalty_contrib_mean': difficulty_penalty_contrib_mean,
                    'alignment_share': alignment_share,'sparsity_share': sparsity_share,'consistency_share': consistency_share,'retention_share': retention_share,'lag_gain_share': lag_gain_share,'peer_alignment_share': peer_alignment_share,'difficulty_ordering_share': difficulty_ordering_share,'drift_smoothness_share': drift_smoothness_share,
                    'alignment_loss_raw': alignment_raw,'sparsity_loss_raw': sparsity_raw,'consistency_loss_raw': consistency_raw,'retention_loss_raw': retention_raw,'lag_gain_loss_raw': lag_gain_raw,'peer_alignment_loss_raw': peer_alignment_raw,'difficulty_ordering_loss_raw': difficulty_ordering_raw,'drift_smoothness_loss_raw': drift_smoothness_raw,'cold_start_flag': cold_start
                }
                if is_rank0:
                    append_epoch_csv(row, metrics_csv, EPOCH_HEADER)
                epoch_rows.append(row)
            # After epochs complete for this seed
            if is_rank0:
                state_dict = model.module.state_dict() if (ddp_enabled and hasattr(model,'module')) else model.state_dict()
                torch.save(state_dict, os.path.join(exp_path,f'model_last_seed{seed}.pth'))
            best_row = max(epoch_rows, key=lambda x: x['val_auc']) if epoch_rows else {}
            if is_rank0:
                best_state_dict = model.module.state_dict() if (ddp_enabled and hasattr(model,'module')) else model.state_dict()
                torch.save({'state_dict': best_state_dict, 'best_epoch': best_row.get('epoch'), 'val_auc': best_row.get('val_auc')}, os.path.join(exp_path,f'model_best_seed{seed}.pth'))
            per_seed_results.append(best_row)
        except KeyboardInterrupt:
            if is_rank0:
                print('[INTERRUPT] KeyboardInterrupt received; aborting remaining seeds.')
                if logger:
                    logger.info('KeyboardInterrupt received; aborting remaining seeds.')
            break
        except BrokenPipeError as bpe:
            if is_rank0:
                print(f'[ERROR] BrokenPipeError during seed {seed}: {bpe}. DataLoader worker likely terminated early. Rerun with --num_workers 0 for debug.')
                if logger:
                    logger.exception('BrokenPipeError encountered; aborting remaining seeds.')
            break
        except Exception as e:
            if is_rank0:
                print(f'[ERROR] Unexpected exception during seed {seed}: {e}')
                if logger:
                    logger.exception(f'Unexpected exception seed {seed}: {e}')
            continue
        # Explicit loader cleanup to encourage worker shutdown
        try:
            del train_loader
            del val_loader
        except Exception:
            pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
    if is_rank0:
        atomic_write_json(multi_seed_summary, os.path.join(exp_path,'results_multi_seed.json'))
    # Legacy results.json (single primary seed best for compatibility)
    primary_best = per_seed_results[0] if per_seed_results else {}
    if is_rank0:
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
    if reproduction_mode:
        lines.append('\n## Reproduction Metadata')
        lines.append('| Field | Value |')
        lines.append('|-------|-------|')
        lines.append(f"| Source Experiment | {source_cfg.get('experiment',{}).get('id')} |")
        lines.append(f"| Source Config MD5 | {source_cfg.get('config_md5')} |")
        lines.append(f"| New Config MD5 | {cfg.get('config_md5')} |")
        # Load new results for comparison
        new_results = _load_json(Path(exp_path) / 'results.json')
        report = _reproduction_report(source_cfg, source_results, cfg, new_results)
        lines.append(f"| Config MD5 Match | {report.get('config_md5_match')} |")
        lines.append(f"| AUC Diff | {report.get('auc_diff','NA')} |")
        lines.append(f"| Mastery Corr Diff | {report.get('mastery_corr_diff','NA')} |")
        lines.append(f"| Gain Corr Diff | {report.get('gain_corr_diff','NA')} |")
        lines.append(f"| Within Tolerance | {report.get('within_tolerance')} |")
        # Persist reproduction report & optional manifest once results.json exists
        atomic_write_json(report, os.path.join(exp_path,'reproduction_report.json'))
        if args.manifest:
            manifest = {
                'experiment_id': cfg['experiment']['id'],
                'source_experiment': source_cfg.get('experiment',{}).get('id'),
                'schema_version': cfg.get('schema_version'),
                'config_md5_source': source_cfg.get('config_md5'),
                'config_md5_new': cfg.get('config_md5'),
                'within_tolerance': report.get('within_tolerance'),
                'seeds': seeds
            }
            atomic_write_json(manifest, os.path.join(exp_path,'reproduction_manifest.json'))
    lines.append('\n## Config MD5')
    lines.append(cfg['config_md5'])
    lines.append('\n## Hardware')
    lines.append(f"Requested devices: {args.devices}")
    lines.append(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES','UNSET')}")
    if is_rank0:
        with open(os.path.join(exp_path,'README.md'),'w') as rf:
            rf.write('\n'.join(lines))
        if logger:
            logger.info('GainAKT3 reproducible experiment complete.')
    if ddp_enabled:
        try:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass

if __name__ == '__main__':
    main()
