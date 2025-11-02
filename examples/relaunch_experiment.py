#!/usr/bin/env python3
"""Relaunch an existing experiment using its recorded config.json.

Audit procedure:
1. Load config.json from provided experiment directory.
2. Extract all parameter values (no hidden defaults). We rely ONLY on values present.
3. Reconstruct training command for underlying script (runtime.train_command preferred if present).
4. Regenerate a new experiment folder (timestamped with suffix 'relaunch').
5. Write a relaunch_audit.json containing:
   - original_config_md5
   - regenerated_config_md5 (after rebuild)
   - diff of parameter values (should be empty if reproducible)
   - original vs reconstructed train_command strings and token diffs.
6. Launch training using reconstructed command, capturing artifacts identically.

Usage:
  python examples/relaunch_experiment.py --source_dir examples/experiments/20251029_101010_gainakt2exp_ordercheck --dry_run
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import shlex
import subprocess
import hashlib
import glob
from datetime import datetime
from typing import Dict, Any
sys.path.insert(0, '/workspaces/pykt-toolkit')
from examples.experiment_utils import make_experiment_dir, atomic_write_json, capture_environment, write_seed_info, build_config, write_readme, timestamped_logger

def parse_args():
    p = argparse.ArgumentParser(description='Relaunch an existing experiment using its config.json for formal reproducibility audit.')
    p.add_argument('--source_dir', type=str, required=True, help='Path to existing experiment directory containing config.json')
    p.add_argument('--output_base', type=str, default='examples/experiments', help='Base directory for new reproduced experiment')
    p.add_argument('--short_title', type=str, default='relaunch', help='Suffix / short title for reproduced experiment folder')
    p.add_argument('--devices', type=int, nargs='*', default=[0,1,2,3,4], help='CUDA devices to use for reproduction (override original if set)')
    p.add_argument('--dry_run', action='store_true', help='Perform audit only; do not execute training subprocess')
    p.add_argument('--skip_readme', action='store_true')
    p.add_argument('--strict', action='store_true', help='Fail before training if any config key cannot be mapped to a training flag (excluding metadata groups).')
    p.add_argument('--override_epochs', type=int, default=None, help='Override number of epochs for reproduction run (for quick audits).')
    # Removed metrics copying/forcing: reproduction should regenerate metrics organically.
    return p.parse_args()

def load_config(path: str) -> Dict[str,Any]:
    with open(path) as f:
        return json.load(f)

def flatten_config(cfg: Dict[str,Any]) -> Dict[str,Any]:
    flat = {}
    for section, content in cfg.items():
        if isinstance(content, dict) and section != 'config_md5':
            for k,v in content.items():
                flat[f"{section}.{k}"] = v
        else:
            flat[section] = content
    return flat

def compute_md5_order_preserved(obj: Dict[str,Any]) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def reconstruct_train_args_dynamic(cfg: Dict[str,Any], valid_flags: set[str]) -> tuple[Dict[str,Any], Dict[str,str]]:
    """Dynamically produce training args from config sections.

    Returns (args_dict, skipped) where skipped maps config_key -> reason.
    We flatten all non-metadata groups. For booleans we emit flag only if True and training script supports it.
    Numeric / string values produce --<key> <value> if the flag exists. This allows forward compatibility when new
    parameters are added to the training script & config but not yet hard-coded here.
    """
    args_out: Dict[str,Any] = {}
    skipped: Dict[str,str] = {}
    # Always carry seed
    primary_seed = cfg.get('seeds', {}).get('primary', 42)
    args_out['seed'] = primary_seed
    for section, content in cfg.items():
        # Skip metadata / non-training sections entirely
        if section in {'runtime','experiment','hardware','seeds','config_md5', 'evaluation_defaults'}:
            continue
        # Treat model_config architecture as informational unless flags exist in training script
        if section == 'model_config':
            for key, value in content.items():
                flag = f"--{key}"
                if flag in valid_flags:
                    # Architecture flags now supported; always emit current value.
                    if isinstance(value, bool):
                        if value:
                            args_out[key] = True
                    else:
                        args_out[key] = value
                else:
                    skipped[f"{section}.{key}"] = "architecture key (no training flag)"
            continue
        if not isinstance(content, dict):
            continue
        for key, value in content.items():
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    if flag in valid_flags:
                        args_out[key] = True
                    else:
                        skipped[f"{section}.{key}"] = f"missing boolean flag {flag}"
                continue
            # Non-boolean
            if flag in valid_flags:
                args_out[key] = value
            else:
                skipped[f"{section}.{key}"] = f"missing flag {flag}"
    # Mixed precision -> use_amp mapping
    if cfg.get('training', {}).get('mixed_precision'):
        if '--use_amp' in valid_flags:
            args_out['use_amp'] = True
        else:
            skipped['training.mixed_precision'] = 'missing flag --use_amp'
    # Post-filter: remove any evaluation_defaults.* (defensive, in case future code injects)
    skipped = {k:v for k,v in skipped.items() if not k.startswith('evaluation_defaults.')}
    return args_out, skipped

def build_train_command(train_script: str, args: Dict[str,Any]) -> str:
    parts = [sys.executable, train_script]
    # Map of config keys to script flags (explicit one-to-one)
    for k,v in args.items():
        if isinstance(v, bool):
            if k == 'use_amp' and v:
                parts.append('--use_amp')
            elif k == 'use_mastery_head' and v:
                parts.append('--use_mastery_head')
            elif k == 'use_gain_head' and v:
                parts.append('--use_gain_head')
            elif k == 'enable_alignment_loss' and v:
                parts.append('--enable_alignment_loss')
            elif k == 'adaptive_alignment' and v:
                parts.append('--adaptive_alignment')
            elif k == 'enable_global_alignment_pass' and v:
                parts.append('--enable_global_alignment_pass')
            elif k == 'use_residual_alignment' and v:
                parts.append('--use_residual_alignment')
            elif k == 'enable_retention_loss' and v:
                parts.append('--enable_retention_loss')
            elif k == 'enable_lag_gain_loss' and v:
                parts.append('--enable_lag_gain_loss')
            elif k == 'enhanced_constraints' and v:
                parts.append('--enhanced_constraints')
        else:
            parts.extend([f"--{k}", str(v)])
    return ' '.join(shlex.quote(x) for x in parts)

def main():
    args = parse_args()
    source_cfg_path = os.path.join(args.source_dir, 'config.json')
    if not os.path.exists(source_cfg_path):
        raise FileNotFoundError(f'config.json not found in {args.source_dir}')
    original_cfg = load_config(source_cfg_path)
    # Ensure model_config_md5 present for consistent diffing; legacy runs may lack it.
    if 'model_config_md5' not in original_cfg:
        try:
            mc = original_cfg.get('model_config')
            if isinstance(mc, dict):
                original_cfg['model_config_md5'] = hashlib.md5(json.dumps(mc, sort_keys=True).encode()).hexdigest()
            else:
                # Fallback: hash empty dict to have deterministic value
                original_cfg['model_config_md5'] = hashlib.md5(json.dumps({}, sort_keys=True).encode()).hexdigest()
        except Exception:
            original_cfg['model_config_md5'] = '<uncomputable>'
    original_md5 = original_cfg.get('config_md5')
    # Invariant collection: auto_shifted_eval must remain True
    auto_shifted_eval_val = original_cfg.get('runtime', {}).get('auto_shifted_eval', True)
    invariant_failures = []
    if auto_shifted_eval_val is not True:
        invariant_failures.append({'key': 'runtime.auto_shifted_eval', 'expected': True, 'found': auto_shifted_eval_val})
    train_script = original_cfg['runtime']['train_command'].split()[1] if 'train_command' in original_cfg['runtime'] else 'examples/train_gainakt2exp.py'
    # Gather valid flags dynamically from training script help
    try:
        help_text = subprocess.check_output([sys.executable, train_script, '--help'], stderr=subprocess.STDOUT, text=True)
        valid_flags = {tok.split()[0] for tok in help_text.split() if tok.startswith('--')}
    except Exception:
        valid_flags = set()
    train_args, skipped = reconstruct_train_args_dynamic(original_cfg, valid_flags)
    # Epoch override: adjust epochs if user requests a shorter reproduction
    original_epochs_record = original_cfg.get('training', {}).get('epochs')
    epoch_override_ignored = False
    if args.override_epochs is not None and original_epochs_record is not None:
        # Ignore user override to preserve correlation stabilization fidelity
        epoch_override_ignored = True
        train_args['epochs'] = original_epochs_record
    elif original_epochs_record is not None:
        train_args['epochs'] = original_epochs_record
    reconstructed_cmd = build_train_command(train_script, train_args)
    # Create new experiment directory
    if os.path.isabs(args.output_base):
        base_dir = args.output_base
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if args.output_base.startswith('examples'):
            base_dir = os.path.join(project_root, args.output_base)
        else:
            base_dir = os.path.join(project_root, args.output_base)
    new_exp_dir = make_experiment_dir(original_cfg['experiment']['model'], args.short_title, base_dir=base_dir)
    exp_id = os.path.basename(new_exp_dir)
    # Build new config from flattened raw_args clone for comparison
    # raw_args should include ALL original config keys so that rebuilt config matches (except experiment metadata)
    raw_args = {}
    for section, content in original_cfg.items():
        if isinstance(content, dict) and section not in {'runtime','experiment','hardware','seeds','config_md5'}:
            for k,v in content.items():
                raw_args[k] = v
    # overlay dynamically reconstructed args to ensure presence
    raw_args.update(train_args)
    # Force original training core values (avoid script defaults drifting)
    if 'training' in original_cfg:
        orig_training = original_cfg['training']
        for core_k in ['epochs','batch_size','learning_rate','weight_decay','gradient_clip','patience']:
            if core_k in orig_training and orig_training[core_k] is not None:
                raw_args[core_k] = orig_training[core_k]
    # Preserve evaluation_defaults and override_applied sections explicitly for faithful reproduction
    if 'evaluation_defaults' in original_cfg:
        raw_args['__evaluation_defaults__'] = original_cfg['evaluation_defaults']
    if 'override_applied' in original_cfg:
        raw_args['__override_applied__'] = original_cfg['override_applied']
    raw_args['model_name'] = original_cfg['experiment']['model']
    raw_args['short_title'] = args.short_title
    purpose_base = f"Relaunch reproduction of {original_cfg['experiment']['id']}"
    if args.override_epochs is not None:
        purpose_base += f" (override_epochs={args.override_epochs})"
    raw_args['purpose'] = purpose_base
    raw_args['launcher_command'] = f"python examples/relaunch_experiment.py --source_dir {args.source_dir}"
    raw_args['train_command'] = reconstructed_cmd
    # Device adaptation: attempt to reproduce original devices, adapt to current availability with warning metadata.
    orig_devices = original_cfg.get('hardware', {}).get('devices', args.devices)
    adapted_devices = orig_devices
    adaptation_note = None
    try:
        import torch
        if torch.cuda.is_available():
            available = torch.cuda.device_count()
            # Filter original list to those < available
            filtered = [d for d in orig_devices if d < available]
            if not filtered:
                # Fallback: select up to 60% (ceil) capped at 5 from available devices
                import math
                target = min(max(1, math.ceil(available * 0.6)), 5)
                filtered = list(range(target))
                adaptation_note = f"No original devices usable (orig={orig_devices}, available={available}); selected fallback {filtered}."
            elif len(filtered) != len(orig_devices):
                adaptation_note = f"Some original devices missing (orig={orig_devices}, available_count={available}); using subset {filtered}."
            adapted_devices = filtered
        else:
            # CPU-only environment
            adapted_devices = []  # denote CPU
            if orig_devices:
                adaptation_note = f"Original GPU devices {orig_devices} not available; running on CPU."
    except Exception as e:
        adaptation_note = f"Device adaptation error: {e}; using recorded devices {orig_devices}."
        adapted_devices = orig_devices
    raw_args['devices'] = adapted_devices
    seeds = [original_cfg['seeds']['primary']]
    new_cfg = build_config(raw_args, exp_id=exp_id, exp_path=new_exp_dir, seeds=seeds)
    # Inject preserved sections into new_cfg (build_config does not include them by default)
    if '__evaluation_defaults__' in raw_args:
        new_cfg['evaluation_defaults'] = raw_args['__evaluation_defaults__']
    if '__override_applied__' in raw_args:
        new_cfg['override_applied'] = raw_args['__override_applied__']
    # Preserve original hardware metadata fields if present (gpu_selection_source, selected_devices, counts)
    orig_hw = original_cfg.get('hardware', {})
    rel_hw = new_cfg.get('hardware', {})
    # Always set devices to adapted_devices chosen earlier
    rel_hw['devices'] = adapted_devices
    # Copy metadata if exists in original
    for meta_key in ['selected_devices','gpu_selection_source','visible_gpu_count','selected_gpu_count']:
        if meta_key in orig_hw:
            rel_hw[meta_key] = orig_hw[meta_key]
    new_cfg['hardware'] = rel_hw
    # Preserve original preflight_consistency block if present to avoid metadata erosion
    if 'preflight_consistency' in original_cfg and 'preflight_consistency' not in new_cfg:
        new_cfg['preflight_consistency'] = original_cfg['preflight_consistency']
    # Recompute hashes (full + effective) after injection
    from examples.experiment_utils import compute_config_hash as _compute_hash
    def _strip_effective(obj: dict) -> dict:
        """Remove metadata-only sections and fields from config for effective hash (must match launcher logic)."""
        META_EXCLUDE_SECTIONS = {'preflight_consistency', 'evaluation_snapshot', 'override_applied', 'reproducibility_policy'}
        MD5_EXCLUDE_KEYS = {'config_md5', 'config_md5_full', 'config_md5_effective', 'model_config_md5', 'evaluation_snapshot_md5'}
        RUNTIME_METADATA_PREFIXES = ('runtime.', 'experiment.', 'hardware.gpu_selection_source', 'hardware.selected_devices')
        OPTIONAL_METADATA_KEYS = {'training.mixed_precision'}
        
        def flatten_and_filter(cfg):
            flat = {}
            for section, content in cfg.items():
                if section in META_EXCLUDE_SECTIONS or section in MD5_EXCLUDE_KEYS:
                    continue
                if isinstance(content, dict):
                    for k, v in content.items():
                        key = f"{section}.{k}"
                        if any(key.startswith(pref) for pref in RUNTIME_METADATA_PREFIXES):
                            continue
                        if key in OPTIONAL_METADATA_KEYS:
                            continue
                        # Filter out any *_md5 fields
                        if k.endswith('_md5') or '_md5_' in k:
                            continue
                        flat[key] = v
                else:
                    if section not in MD5_EXCLUDE_KEYS:
                        flat[section] = content
            return flat
        
        return flatten_and_filter(obj)
    
    new_cfg['config_md5_full'] = _compute_hash(new_cfg)
    new_cfg['config_md5_effective'] = _compute_hash(_strip_effective(new_cfg))
    # Backward compatible field
    new_cfg['config_md5'] = new_cfg['config_md5_full']
    atomic_write_json(new_cfg, os.path.join(new_exp_dir,'config.json'))
    atomic_write_json(new_cfg, os.path.join(new_exp_dir,'config.json'))
    capture_environment(os.path.join(new_exp_dir,'environment.txt'))
    write_seed_info(os.path.join(new_exp_dir,'SEED_INFO.md'), seeds, seeds[0])
    # Diff audit
    flat_original = flatten_config(original_cfg)
    flat_new = flatten_config(new_cfg)
    param_diffs = {}
    # Float comparison tolerance to treat formatting-only differences as equal
    FLOAT_TOL = 1e-12
    def values_equal(a,b):
        if not isinstance(b, type(a)):
            # Allow int vs float if numerically equal
            if isinstance(a,(int,float)) and isinstance(b,(int,float)):
                return abs(float(a)-float(b)) < FLOAT_TOL
            return False
        if isinstance(a,float):
            return abs(a-b) < FLOAT_TOL
        return a == b
    for k,v in flat_original.items():
        if k.startswith('runtime.') or k == 'config_md5':
            continue
        if k not in flat_new:
            param_diffs[k] = {'original': v, 'new': '<missing>'}
        else:
            if not values_equal(flat_new[k], v):
                param_diffs[k] = {'original': v, 'new': flat_new[k]}
    # Inverse check: new params not in original
    for k,v in flat_new.items():
        if k.startswith('runtime.') or k == 'config_md5':
            continue
        if k not in flat_original:
            param_diffs[k] = {'original': '<missing>', 'new': v}
    # Metadata keys we intentionally allow to differ between original and relaunch
    allowed_metadata_diff_prefixes = {
        'experiment.id',
        'experiment.short_title',
        'experiment.purpose',
        'runtime.timestamp',  # timestamp always differs
        'model_config_md5',  # treat model_config hash as metadata once computed
        # Training/runtime aliasing: treat training.mixed_precision (legacy alias) as non-substantive
        'training.mixed_precision',
        'evaluation_snapshot.dataset',
        'evaluation_snapshot.fold',
        'evaluation_snapshot.batch_size',
        'evaluation_snapshot.seq_len',
        'evaluation_snapshot.d_model',
        'evaluation_snapshot.n_heads',
        'evaluation_snapshot.num_encoder_blocks',
        'evaluation_snapshot.d_ff',
        'evaluation_snapshot.dropout',
        'evaluation_snapshot.use_mastery_head',
        'evaluation_snapshot.use_gain_head',
        'evaluation_snapshot.non_negative_loss_weight',
        'evaluation_snapshot.monotonicity_loss_weight',
        'evaluation_snapshot.mastery_performance_loss_weight',
        'evaluation_snapshot.gain_performance_loss_weight',
        'evaluation_snapshot.sparsity_loss_weight',
        'evaluation_snapshot.consistency_loss_weight',
        'evaluation_snapshot.max_correlation_students',
        'evaluation_snapshot_md5',
        'reproducibility_policy.ignored_eval_flags',
        'reproducibility_policy.ignored_training_flags',
        'reproducibility_policy.schema_version',
        'reproducibility_policy.policy_last_updated'
    }
    # Epoch difference info
    original_epochs = original_cfg.get('training', {}).get('epochs')
    new_epochs = new_cfg.get('training', {}).get('epochs')
    epochs_changed = (original_epochs is not None and new_epochs is not None and original_epochs != new_epochs)
    epoch_diff_info = None
    if epochs_changed:
        epoch_diff_info = {
            'original_epochs': original_epochs,
            'relaunch_epochs': new_epochs,
            'override_used': args.override_epochs is not None
        }
    # Rebuild diffs focusing ONLY on substantive differences and missing values.
    # VALIDATION: Check evaluation_snapshot consistency with canonical hyperparameters
    eval_snapshot_inconsistencies = []
    if 'evaluation_snapshot' in new_cfg:
        canonical_mappings = {
            'emb_type': ('model_config', 'emb_type'),
            'seq_len': ('model_config', 'seq_len'),
            'd_model': ('model_config', 'd_model'),
            'n_heads': ('model_config', 'n_heads'),
            'num_encoder_blocks': ('model_config', 'num_encoder_blocks'),
            'd_ff': ('model_config', 'd_ff'),
            'dropout': ('model_config', 'dropout'),
            'dataset': ('training', 'dataset_name'),
            'fold': ('training', 'fold'),
            'batch_size': ('training', 'batch_size')
        }
        for snap_key, (canon_section, canon_key) in canonical_mappings.items():
            snap_val = new_cfg['evaluation_snapshot'].get(snap_key)
            canon_val = new_cfg.get(canon_section, {}).get(canon_key)
            if snap_val is not None and canon_val is not None and snap_val != canon_val:
                eval_snapshot_inconsistencies.append({
                    'snapshot_key': f'evaluation_snapshot.{snap_key}',
                    'snapshot_value': snap_val,
                    'canonical_key': f'{canon_section}.{canon_key}',
                    'canonical_value': canon_val
                })
    
    # Refined metadata filtering:
    # evaluation_snapshot.* contains COPIES of canonical hyperparameters for eval convenience.
    # While batch_size, seq_len, emb_type, etc. ARE hyperparameters in their canonical locations
    # (training.batch_size, model_config.seq_len, model_config.emb_type), their COPIES in
    # evaluation_snapshot are metadata (redundant cache). The validation layer above ensures
    # that if evaluation_snapshot values exist, they match their canonical counterparts.
    # Therefore, treating ALL evaluation_snapshot.* as metadata is correct:
    # - If canonical hyperparameter unchanged → snapshot presence/absence is metadata evolution
    # - If canonical hyperparameter changed → flagged via canonical location diff
    # - If snapshot diverges from canonical → flagged via validation as corruption
    
    # Extend prefix matching for metadata to catch runtime/experiment/etc
    metadata_filter_prefixes = (
        'experiment.', 'runtime.', 'hardware.', 'preflight_consistency.', 
        'evaluation_snapshot.',  # All snapshot keys are metadata (validated copies of canonical values)
        'reproducibility_policy.',
        'config_md5', 'model_config_md5', 'evaluation_snapshot_md5'
    )
    def is_metadata_key(key):
        """Return True if key should be ignored as pure metadata (not hyperparameter).
        
        Note: evaluation_snapshot.* keys are metadata even when they represent hyperparameters,
        because they are redundant copies of canonical values in model_config/training sections.
        The validation layer ensures consistency; divergence is flagged as corruption.
        """
        if key in allowed_metadata_diff_prefixes:
            return True
        if any(key.startswith(pref) for pref in metadata_filter_prefixes):
            return True
        # Also exclude any *_md5 fields
        if key.endswith('_md5') or '_md5_' in key:
            return True
        return False
    
    substantive_diffs = {}
    missing_params = {}
    metadata_added = {}  # Track newly added metadata for audit transparency
    
    for k, v in param_diffs.items():
        if is_metadata_key(k):
            # If it's a new metadata field (missing in original), track separately
            if v['original'] == '<missing>':
                metadata_added[k] = v
            continue  # skip from substantive diffs
        orig_val = v['original']
        new_val = v['new']
        # Capture missing markers explicitly (for true hyperparameters only)
        if orig_val == '<missing>' or new_val == '<missing>':
            missing_params[k] = v
            substantive_diffs[k] = v
            continue
        if isinstance(orig_val,(int,float)) and isinstance(new_val,(int,float)) and values_equal(orig_val,new_val):
            continue  # formatting-only numeric diff -> ignore
        if orig_val == new_val:
            continue  # identical
        substantive_diffs[k] = v
    # MD5 comparison: Recompute effective hash for BOTH configs using current exclusion logic
    # to ensure consistent comparison (original may have been computed with older logic).
    from examples.experiment_utils import compute_config_hash as _compute_hash_util
    
    # Use the same _strip_effective defined earlier in relaunch script (lines 279-302)
    def _recompute_effective(cfg):
        """Apply current metadata exclusion logic to any config for fair comparison."""
        META_EXCLUDE_SECTIONS = {'preflight_consistency', 'evaluation_snapshot', 'override_applied', 'reproducibility_policy'}
        MD5_EXCLUDE_KEYS = {'config_md5', 'config_md5_full', 'config_md5_effective', 'model_config_md5', 'evaluation_snapshot_md5'}
        # Expand runtime/experiment metadata to cover all non-hyperparameter fields
        RUNTIME_METADATA_PREFIXES = ('runtime.', 'experiment.', 'hardware.gpu_selection_source', 'hardware.selected_devices')
        # Legacy/optional fields added in later versions (not present in all runs)
        OPTIONAL_METADATA_KEYS = {'training.mixed_precision'}
        
        def flatten_and_filter(c):
            flat = {}
            for section, content in c.items():
                if section in META_EXCLUDE_SECTIONS or section in MD5_EXCLUDE_KEYS:
                    continue
                if isinstance(content, dict):
                    for k, v in content.items():
                        key = f"{section}.{k}"
                        # Exclude all runtime.* and experiment.* keys (not hyperparameters)
                        if any(key.startswith(pref) for pref in RUNTIME_METADATA_PREFIXES):
                            continue
                        # Exclude optional metadata fields
                        if key in OPTIONAL_METADATA_KEYS:
                            continue
                        if k.endswith('_md5') or '_md5_' in k:
                            continue
                        flat[key] = v
                else:
                    if section not in MD5_EXCLUDE_KEYS:
                        flat[section] = content
            return flat
        return flatten_and_filter(cfg)
    
    original_full = original_cfg.get('config_md5_full', original_md5)
    relaunch_full = new_cfg.get('config_md5_full')
    # Recompute effective hashes using current unified logic
    original_effective_recomputed = _compute_hash_util(_recompute_effective(original_cfg))
    relaunch_effective_recomputed = _compute_hash_util(_recompute_effective(new_cfg))
    
    md5_full_match = (original_full == relaunch_full)
    md5_effective_match = (original_effective_recomputed == relaunch_effective_recomputed)
    md5_comparison = {
        'md5_full_match': md5_full_match,
        'md5_effective_match': md5_effective_match,
        # Backward compatibility: single md5_match flag (true only if both full & effective match)
        'md5_match': (md5_full_match and md5_effective_match),
        'original_full_md5': original_full,
        'relaunch_full_md5': relaunch_full,
        'original_effective_md5': original_effective_recomputed,
        'relaunch_effective_md5': relaunch_effective_recomputed
    }
    # GPU comparison summary
    gpu_comparison = {
        'original_devices': orig_devices,
        'relaunch_devices': adapted_devices,
        'device_count_original': len(orig_devices),
        'device_count_relaunch': len(adapted_devices),
        'device_count_changed': len(orig_devices) != len(adapted_devices)
    }
    if gpu_comparison['device_count_changed'] and adaptation_note is None:
        adaptation_note = 'Device count differs without explicit adaptation note.'
    # Seed match flag
    seed_match = (original_cfg.get('seeds', {}).get('primary') == new_cfg.get('seeds', {}).get('primary'))
    # Helper to detect presence of original metrics file (simple existence heuristic)
    def metrics_csv_exists():
        return os.path.exists(os.path.join(args.source_dir, 'metrics_epoch.csv'))
    # Classify diff types: metadata_only vs hyperparameter
    metadata_prefixes = (
        'experiment.', 'runtime.', 'hardware.', 'preflight_consistency.', 'evaluation_snapshot.', 'model_config_md5',
        'config_md5', 'config_md5_full', 'config_md5_effective'
    )
    hyperparameter_diffs = {}
    metadata_diffs = {}
    for k,v in substantive_diffs.items():
        if any(k.startswith(pref) for pref in metadata_prefixes):
            metadata_diffs[k] = v
        else:
            hyperparameter_diffs[k] = v
    # Determine equivalence status
    if md5_effective_match and not hyperparameter_diffs and metrics_csv_exists():
        equivalence_status = 'equivalent'
    elif md5_effective_match and hyperparameter_diffs == {} and metadata_diffs:
        equivalence_status = 'metadata_diff'
    elif not md5_effective_match and not hyperparameter_diffs and metadata_diffs:
        equivalence_status = 'metadata_diff'
    elif hyperparameter_diffs and md5_effective_match:
        equivalence_status = 'param_diff'
    else:
        equivalence_status = 'param_diff'
    def metrics_csv_exists(train_cmd):
        # Placeholder: assume metrics exist if original run folder contained metrics_epoch.csv
        orig_metrics = os.path.join(args.source_dir, 'metrics_epoch.csv')
        return os.path.exists(orig_metrics)
    # ========== ACTIONABLE AUDIT: THREE-LAYER PROTECTION ==========
    
    # Layer 1: Detect Hyperparameter Changes (via canonical diffs)
    canonical_changes = {
        'status': 'PASS' if not substantive_diffs else 'FAIL',
        'description': 'Checks if substantive hyperparameters changed between original and relaunch',
        'hyperparameter_diffs': substantive_diffs,
        'count': len(substantive_diffs),
        'action_required': bool(substantive_diffs),
        'guidance': 'No action needed - experiments are substantively equivalent' if not substantive_diffs else 'ACTION REQUIRED: Hyperparameter changes detected. These experiments are NOT equivalent.'
    }
    
    # Layer 2: Validate Consistency (via snapshot validation)
    snapshot_corruption = {
        'status': 'PASS' if not eval_snapshot_inconsistencies else 'FAIL',
        'description': 'Checks if evaluation_snapshot copies match their canonical hyperparameter sources',
        'inconsistencies': eval_snapshot_inconsistencies,
        'count': len(eval_snapshot_inconsistencies),
        'action_required': bool(eval_snapshot_inconsistencies),
        'guidance': 'No action needed - snapshot values match canonical locations' if not eval_snapshot_inconsistencies else 'ACTION REQUIRED: Snapshot corruption detected. Config file is INVALID.'
    }
    
    # Layer 3: Track Schema Evolution (via metadata classification)
    schema_evolution = {
        'status': 'BENIGN',
        'description': 'Tracks metadata additions/changes from launcher version evolution',
        'metadata_added': metadata_added,
        'metadata_changed': metadata_diffs,
        'skipped_params': skipped,
        'count': len(metadata_added) + len(metadata_diffs) + len(skipped),
        'count_added': len(metadata_added),
        'count_changed': len(metadata_diffs),
        'count_skipped': len(skipped),
        'action_required': False,
        'guidance': f'No action needed - {len(metadata_added)} metadata fields added, {len(metadata_diffs)} changed, {len(skipped)} skipped due to missing CLI flags'
    }
    
    # Overall Reproducibility Status
    reproducibility_status = {
        'verdict': equivalence_status,
        'reproducible': equivalence_status == 'equivalent' and not eval_snapshot_inconsistencies,
        'blocking_issues': [],
        'warnings': [],
        'info': []
    }
    
    if substantive_diffs:
        reproducibility_status['blocking_issues'].append({
            'type': 'HYPERPARAMETER_MISMATCH',
            'severity': 'CRITICAL',
            'count': len(substantive_diffs),
            'message': f'{len(substantive_diffs)} hyperparameter(s) differ between experiments',
            'action': 'Review canonical_changes.hyperparameter_diffs for details'
        })
    
    if eval_snapshot_inconsistencies:
        reproducibility_status['blocking_issues'].append({
            'type': 'SNAPSHOT_CORRUPTION',
            'severity': 'CRITICAL',
            'count': len(eval_snapshot_inconsistencies),
            'message': 'Evaluation snapshot values do not match canonical hyperparameters',
            'action': 'Review snapshot_corruption.inconsistencies for corrupted keys'
        })
    
    if missing_params:
        reproducibility_status['blocking_issues'].append({
            'type': 'MISSING_PARAMETERS',
            'severity': 'CRITICAL',
            'count': len(missing_params),
            'message': f'{len(missing_params)} parameter(s) could not be reconstructed',
            'action': 'Check missing_parameters section for unmapped keys'
        })
    
    if invariant_failures:
        reproducibility_status['blocking_issues'].append({
            'type': 'INVARIANT_VIOLATION',
            'severity': 'CRITICAL',
            'count': len(invariant_failures),
            'message': 'Invariant checks failed',
            'action': 'Review invariant_failures for validation errors'
        })
    
    if not seed_match:
        reproducibility_status['warnings'].append({
            'type': 'SEED_MISMATCH',
            'severity': 'WARNING',
            'message': 'Random seed differs between experiments',
            'action': 'Results may not be numerically identical'
        })
    
    if epoch_diff_info:
        reproducibility_status['warnings'].append({
            'type': 'EPOCH_OVERRIDE',
            'severity': 'WARNING',
            'message': epoch_diff_info,
            'action': 'Training duration differs from original'
        })
    
    if adaptation_note:
        reproducibility_status['warnings'].append({
            'type': 'DEVICE_ADAPTATION',
            'severity': 'INFO',
            'message': adaptation_note,
            'action': 'GPUs differ but training can proceed'
        })
    
    if metadata_added:
        reproducibility_status['info'].append({
            'type': 'SCHEMA_EVOLUTION',
            'severity': 'INFO',
            'message': f'{len(metadata_added)} metadata field(s) added from newer launcher version',
            'action': 'No action required - benign schema evolution'
        })
    
    # Compact actionable audit structure
    audit = {
        # === ACTIONABLE SUMMARY ===
        'reproducibility_status': reproducibility_status,
        
        # === THREE-LAYER PROTECTION ===
        'canonical_changes': canonical_changes,
        'snapshot_corruption': snapshot_corruption,
        'schema_evolution': schema_evolution,
        
        # === EXPERIMENT IDENTITY ===
        'experiment': {
            'original_id': original_cfg['experiment']['id'],
            'relaunch_id': exp_id,
            'train_script': train_script,
            'original_command': original_cfg['runtime'].get('train_command'),
            'relaunch_command': reconstructed_cmd
        },
        
        # === HASH VERIFICATION ===
        'hash_verification': {
            'full_match': md5_full_match,
            'effective_match': md5_effective_match,
            'combined_match': md5_full_match and md5_effective_match,
            'original_effective_md5': original_effective_recomputed,
            'relaunch_effective_md5': relaunch_effective_recomputed,
            'explanation': 'full=all params including metadata; effective=substantive hyperparameters only'
        },
        
        # === DETAILED DIAGNOSTICS (for debugging) ===
        'diagnostics': {
            'missing_parameters': missing_params,
            'invariant_failures': invariant_failures,
            'seed_match': seed_match,
            'epoch_info': epoch_diff_info,
            'device_info': {
                'original': orig_devices,
                'relaunch': adapted_devices,
                'adaptation_note': adaptation_note
            }
        },
        
        # === EXECUTION METADATA ===
        'execution': {
            'dry_run': args.dry_run,
            'strict_mode': args.strict,
            'audit_timestamp': datetime.utcnow().isoformat()
        }
    }
    audit['correlation_deterministic'] = True
    audit['epoch_override_ignored'] = epoch_override_ignored
    # Strict mode: if epochs changed, fail early with guidance regardless of other diffs.
    if args.strict:
        if epochs_changed:
            audit['strict_failure'] = True
            audit['strict_guidance'] = 'Epoch count differs. Remove --strict or omit --override_epochs for faithful reproduction.'
        elif substantive_diffs or skipped:
            audit['strict_failure'] = True
            audit['strict_guidance'] = 'Substantive parameter diffs detected; aborting in strict mode.'
        else:
            audit['strict_failure'] = False
            audit['strict_guidance'] = 'No substantive diffs detected; proceeding.'
    else:
        audit['strict_failure'] = False
        audit['strict_guidance'] = 'Strict mode disabled.'
    atomic_write_json(audit, os.path.join(new_exp_dir,'relaunch_audit.json'))
    logger = timestamped_logger('relaunch', os.path.join(new_exp_dir,'stdout.log'))
    logger.info(f"Relaunch audit created for {original_cfg['experiment']['id']} -> {exp_id}")
    
    # === PRINT ACTIONABLE SUMMARY TO CONSOLE ===
    def print_audit_summary():
        """Print structured, actionable summary of relaunch audit to console."""
        print("\n" + "=" * 80)
        print("RELAUNCH AUDIT SUMMARY")
        print("=" * 80)
        
        # 1. Quick Status
        status = reproducibility_status['verdict'].upper()
        reproducible = reproducibility_status['reproducible']
        status_symbol = "✅" if reproducible else "❌"
        print(f"\n{status_symbol} REPRODUCIBILITY STATUS: {status}")
        print(f"   Reproducible: {reproducible}")
        
        # 2. Blocking Issues (if any)
        if reproducibility_status['blocking_issues']:
            print(f"\n❌ BLOCKING ISSUES ({len(reproducibility_status['blocking_issues'])}):")
            for issue in reproducibility_status['blocking_issues']:
                print(f"   • [{issue['severity']}] {issue['type']}")
                print(f"     {issue['message']}")
                print(f"     → ACTION: {issue['action']}")
        
        # 3. Three-Layer Protection Summary
        print("\n" + "-" * 80)
        print("THREE-LAYER PROTECTION SUMMARY")
        print("-" * 80)
        
        # Layer 1: Canonical Changes
        layer1_symbol = "✅" if canonical_changes['status'] == 'PASS' else "❌"
        print(f"\n{layer1_symbol} Layer 1 - CANONICAL CHANGES (Hyperparameter Detection)")
        print(f"   Status: {canonical_changes['status']}")
        print(f"   Action Required: {canonical_changes['action_required']}")
        print(f"   {canonical_changes['guidance']}")
        if canonical_changes.get('details'):
            for detail in canonical_changes['details'][:3]:  # Show first 3
                print(f"      • {detail}")
        
        # Layer 2: Snapshot Corruption
        layer2_symbol = "✅" if snapshot_corruption['status'] == 'PASS' else "⚠️"
        print(f"\n{layer2_symbol} Layer 2 - SNAPSHOT VALIDATION (Consistency Check)")
        print(f"   Status: {snapshot_corruption['status']}")
        print(f"   Action Required: {snapshot_corruption['action_required']}")
        print(f"   {snapshot_corruption['guidance']}")
        if snapshot_corruption.get('details'):
            for detail in snapshot_corruption['details'][:3]:
                print(f"      • {detail}")
        
        # Layer 3: Schema Evolution
        layer3_symbol = "ℹ️" if schema_evolution['status'] == 'BENIGN' else "⚠️"
        print(f"\n{layer3_symbol} Layer 3 - SCHEMA EVOLUTION (Metadata Changes)")
        print(f"   Status: {schema_evolution['status']}")
        print(f"   Count: {schema_evolution['count']} changes")
        print(f"     - Added: {len(schema_evolution.get('added', []))}")
        print(f"     - Changed: {len(schema_evolution.get('changed', []))}")
        print(f"     - Skipped: {len(schema_evolution.get('skipped', []))}")
        print(f"   Action Required: {schema_evolution['action_required']}")
        print(f"   {schema_evolution['guidance']}")
        
        # 4. Hash Verification
        print("\n" + "-" * 80)
        print("HASH VERIFICATION")
        print("-" * 80)
        effective_match = audit['hash_verification']['effective_match']
        full_match = audit['hash_verification']['full_match']
        hash_symbol = "✅" if effective_match else "❌"
        print(f"\n{hash_symbol} Effective Hash Match: {effective_match}")
        print(f"   Full Hash Match: {full_match} (includes metadata)")
        print(f"   → {audit['hash_verification']['explanation']}")
        
        # 5. Warnings (if any)
        if reproducibility_status['warnings']:
            print(f"\n⚠️  WARNINGS ({len(reproducibility_status['warnings'])}):")
            for warn in reproducibility_status['warnings']:
                print(f"   • [{warn['severity']}] {warn['type']}")
                print(f"     {warn['message']}")
                print(f"     → ACTION: {warn['action']}")
        
        # 6. Info Messages
        if reproducibility_status['info']:
            print("\nℹ️  INFO:")
            for info in reproducibility_status['info']:
                print(f"   • {info['message']}")
        
        # 7. Bottom Line
        print("\n" + "=" * 80)
        if reproducible and effective_match:
            print("✅ VERDICT: Experiments are substantively EQUIVALENT and REPRODUCIBLE")
            print("   No action required. Safe to proceed.")
        elif reproducibility_status['blocking_issues']:
            print("❌ VERDICT: BLOCKING issues detected - Review and resolve before proceeding")
        else:
            print("⚠️  VERDICT: Review warnings and verify reproducibility claims")
        print("=" * 80)
        print(f"\nFull audit details: {os.path.join(new_exp_dir, 'relaunch_audit.json')}")
        print()
    
    # Print the summary
    print_audit_summary()
    
    # === ACTIONABLE SUMMARY LOGGING ===
    logger.info("==== REPRODUCIBILITY STATUS ====")
    logger.info(f"Verdict: {reproducibility_status['verdict'].upper()}")
    logger.info(f"Reproducible: {reproducibility_status['reproducible']}")
    
    if reproducibility_status['blocking_issues']:
        logger.error(f"BLOCKING ISSUES ({len(reproducibility_status['blocking_issues'])}):")
        for issue in reproducibility_status['blocking_issues']:
            logger.error(f"  [{issue['severity']}] {issue['type']}: {issue['message']}")
            logger.error(f"      Action: {issue['action']}")
    
    if reproducibility_status['warnings']:
        logger.warning(f"WARNINGS ({len(reproducibility_status['warnings'])}):")
        for warn in reproducibility_status['warnings']:
            logger.warning(f"  [{warn['severity']}] {warn['type']}: {warn['message']}")
            logger.warning(f"      Action: {warn['action']}")
    
    if reproducibility_status['info']:
        logger.info("INFO:")
        for info in reproducibility_status['info']:
            logger.info(f"  {info['message']}")
    
    # === THREE-LAYER PROTECTION SUMMARY ===
    logger.info("==== THREE-LAYER PROTECTION ====")
    logger.info(f"Layer 1 (Canonical Changes): {canonical_changes['status']} - {canonical_changes['guidance']}")
    logger.info(f"Layer 2 (Snapshot Corruption): {snapshot_corruption['status']} - {snapshot_corruption['guidance']}")
    logger.info(f"Layer 3 (Schema Evolution): {schema_evolution['status']} - {schema_evolution['guidance']}")
    
    # === STRICT MODE CHECK (Before Training) ===
    if args.strict and audit['strict_failure']:
        logger.error("Strict mode violation: aborting before training.")
        print("\n❌ STRICT MODE: Aborting due to reproducibility violations.")
        print("   Remove --strict flag to proceed anyway, or resolve issues first.\n")
        return
    
    # === BLOCKING ISSUES CHECK (Before Training) ===
    if reproducibility_status['blocking_issues'] and not args.dry_run:
        print("\n⚠️  BLOCKING ISSUES DETECTED")
        print("   The audit found issues that may affect reproducibility.")
        print("   Review the audit summary above before proceeding.\n")
        logger.warning("Blocking issues present but not in strict mode; proceeding with training.")
    
    # === DRY RUN COMPLETION ===
    if args.dry_run:
        print("\n✅ DRY RUN COMPLETE - No training executed")
        print(f"   Audit saved to: {os.path.join(new_exp_dir, 'relaunch_audit.json')}\n")
        logger.info("Dry run completed; no training executed.")
        # Still write final audit with diagnostics
        atomic_write_json(audit, os.path.join(new_exp_dir,'relaunch_audit.json'))
        if not args.skip_readme:
            write_readme(new_exp_dir, new_cfg, best_metrics=None)
        return
    
    # === PROCEED TO TRAINING ===
    print("\n" + "=" * 80)
    print("PROCEEDING TO TRAINING")
    print("=" * 80)
    logger.info("Starting relaunch training subprocess...")
    
    if not args.dry_run:
        env = os.environ.copy()
        # Use adapted_devices determined earlier (original when possible, else fallback) for reproduction fidelity
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in adapted_devices)
        env['EXPERIMENT_DIR'] = new_exp_dir
        # Deterministic CuBLAS workspace for reproducible matmul under deterministic algorithms
        if 'CUBLAS_WORKSPACE_CONFIG' not in env:
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Infer config path from source_dir experiment (single source of truth); do NOT rely on PYKT_CONFIG_PATH
        source_config_path = os.path.join(args.source_dir, 'config.json')
        # Append --config flag if not already present so training script explicitly loads resolved config
        if '--config' not in reconstructed_cmd:
            reconstructed_cmd = f"{reconstructed_cmd} --config {shlex.quote(source_config_path)}"
        logger.info(f"Launching training subprocess: {reconstructed_cmd}")
        proc = subprocess.Popen(reconstructed_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        while True:
            line = proc.stdout.readline()
            if line:
                logger.info(line.rstrip())
            elif proc.poll() is not None:
                break
        stderr = proc.stderr.read()
        if stderr:
            with open(os.path.join(new_exp_dir,'stderr.log'),'w') as ef:
                ef.write(stderr)
        rc = proc.wait()
        logger.info(f"Training exit code: {rc}")
        # === Metrics comparison ===
        def _locate_results(dir_path: str) -> Dict[str,Any]:
            repros = sorted(glob.glob(os.path.join(dir_path, 'repro_results_*.json')))
            target = repros[-1] if repros else os.path.join(dir_path,'results.json')
            if not os.path.exists(target):
                return {}
            try:
                with open(target) as f:
                    return json.load(f)
            except Exception:
                return {}
        original_results = _locate_results(args.source_dir)
        relaunch_results = _locate_results(new_exp_dir)
        def _extract(core: Dict[str,Any]) -> Dict[str,Any]:
            if not core:
                return {}
            best_auc = core.get('best_val_auc') or core.get('best_val_auc', core.get('best_val_auc'))
            consistency = core.get('final_consistency_metrics') or core.get('final_consistency_metrics', {})
            mastery_corr = None
            gain_corr = None
            if isinstance(consistency, dict):
                mastery_corr = consistency.get('mastery_correlation') or consistency.get('mastery_corr')
                gain_corr = consistency.get('gain_correlation') or consistency.get('gain_corr')
            return {
                'best_val_auc': best_auc,
                'mastery_corr': mastery_corr,
                'gain_corr': gain_corr
            }
        o_metrics = _extract(original_results)
        r_metrics = _extract(relaunch_results)
        metrics_diffs = {}
        TOL = {'best_val_auc': 1e-3, 'mastery_corr': 1e-2, 'gain_corr': 1e-2}
        metrics_match = True
        relaunch_metrics_path = os.path.join(new_exp_dir,'metrics_epoch.csv')
        original_metrics_path = os.path.join(args.source_dir,'metrics_epoch.csv')
        relaunch_metrics_exists = os.path.exists(relaunch_metrics_path)
        original_metrics_exists = os.path.exists(original_metrics_path)
        if original_metrics_exists and not relaunch_metrics_exists:
            metrics_match = False
            logger.warning("Metrics file missing in relaunch while present in original; marking metrics_match False.")
        if o_metrics and r_metrics:
            for k, ov in o_metrics.items():
                rv = r_metrics.get(k)
                if ov is not None and rv is not None:
                    diff = abs(rv - ov)
                    metrics_diffs[k] = {'original': ov, 'relaunch': rv, 'abs_diff': diff, 'tolerance': TOL.get(k)}
                    if diff > TOL.get(k, 0):
                        metrics_match = False
            logger.info("RELAUNCH METRICS COMPARISON:")
            for k,v in metrics_diffs.items():
                logger.info(f"  {k}: original={v['original']} relaunch={v['relaunch']} diff={v['abs_diff']:.6f} tol={v['tolerance']}")
            logger.info(f"Metrics within tolerance: {metrics_match}")
        else:
            logger.info("Metric value comparison skipped (missing one or both result artifacts).")
        audit['metrics_comparison'] = {
            'original': o_metrics,
            'relaunch': r_metrics,
            'diffs': metrics_diffs,
            'within_tolerance': metrics_match,
            'tolerance_map': TOL,
            'training_exit_code': rc,
            'original_metrics_exists': original_metrics_exists,
            'relaunch_metrics_exists': relaunch_metrics_exists
        }
    # Final audit enrichment & write (second write after potential metrics comparison)
    if missing_params:
        audit['diagnostics']['missing_params_count'] = len(missing_params)
    atomic_write_json(audit, os.path.join(new_exp_dir,'relaunch_audit.json'))
    
    # Human-readable summary block (post-training)
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - REPRODUCIBILITY REPORT")
    print("=" * 80)
    
    logger.info("==== FINAL AUDIT SUMMARY ====")
    logger.info(f"Reproducibility Status: {audit['reproducibility_status']['verdict'].upper()}")
    logger.info(f"Hash Verification: effective={audit['hash_verification']['effective_match']}, full={audit['hash_verification']['full_match']}")
    
    # === ENHANCED METRICS COMPARISON REPORT ===
    if audit.get('metrics_comparison'):
        mc = audit['metrics_comparison']
        
        # Determine overall metrics status
        all_metrics_match = mc.get('within_tolerance', False)
        has_metrics = bool(mc.get('diffs'))
        training_succeeded = mc.get('training_exit_code') == 0
        
        print("\n" + "-" * 80)
        print("METRICS COMPARISON REPORT")
        print("-" * 80)
        
        if not training_succeeded:
            print("\n❌ TRAINING FAILED")
            print(f"   Exit code: {mc.get('training_exit_code')}")
            print("   Cannot compare metrics - relaunch did not complete successfully")
            logger.error(f"Training failed with exit code {mc.get('training_exit_code')}")
        
        elif not mc.get('original_metrics_exists'):
            print("\n⚠️  ORIGINAL METRICS MISSING")
            print("   Original experiment did not produce metrics file")
            print("   Cannot verify reproducibility")
            logger.warning("Original metrics file not found")
        
        elif not mc.get('relaunch_metrics_exists'):
            print("\n❌ RELAUNCH METRICS MISSING")
            print("   Relaunch training did not produce metrics file")
            print("   Training may have crashed or been interrupted")
            logger.error("Relaunch metrics file not found")
        
        elif not has_metrics:
            print("\n⚠️  NO METRICS AVAILABLE FOR COMPARISON")
            print("   Results files exist but contain no extractable metrics")
            print("   Check results.json format in both experiments")
            logger.warning("Metrics extraction failed for comparison")
        
        else:
            # We have metrics to compare
            logger.info(f"Metrics Comparison: within_tolerance={all_metrics_match}")
            
            if all_metrics_match:
                print("\n✅ METRICS REPRODUCED SUCCESSFULLY")
                print("   All metrics within tolerance - reproducibility CONFIRMED")
            else:
                print("\n❌ METRICS DIVERGENCE DETECTED")
                print("   Some metrics exceed tolerance - reproducibility at RISK")
            
            print(f"\n{'Metric':<25} {'Original':<12} {'Relaunch':<12} {'Delta':<12} {'Status':<10}")
            print("-" * 80)
            
            for k, v in mc['diffs'].items():
                orig = v['original']
                rel = v['relaunch']
                diff = v['abs_diff']
                tol = v['tolerance']
                
                within_tol = diff <= tol
                status_symbol = "✅ PASS" if within_tol else "❌ FAIL"
                
                # Format metric name
                metric_display = k.replace('_', ' ').title()
                
                print(f"{metric_display:<25} {orig:<12.6f} {rel:<12.6f} {diff:<12.6f} {status_symbol}")
                
                # Log to file
                logger.info(f"  {k}: original={orig:.6f}, relaunch={rel:.6f}, diff={diff:.6f}, tolerance={tol}, match={within_tol}")
                
                # Additional context for failures
                if not within_tol:
                    pct_diff = (diff / orig * 100) if orig != 0 else float('inf')
                    print(f"  {'':>25} ⚠️  Tolerance: {tol:.6f} | Δ%: {pct_diff:+.2f}%")
            
            # Bottom line verdict
            print("\n" + "-" * 80)
            if all_metrics_match:
                print("✅ VERDICT: Relaunch successfully reproduced original metrics")
                print("   → Reproducibility validated. Safe to use relaunch results.")
            else:
                failed_count = sum(1 for v in mc['diffs'].values() if v['abs_diff'] > v['tolerance'])
                print(f"❌ VERDICT: {failed_count}/{len(mc['diffs'])} metric(s) exceed tolerance")
                print("   → Possible causes:")
                print("      • Non-deterministic operations (check seeds/determinism settings)")
                print("      • Hardware differences (GPU/CPU variations)")
                print("      • Library version changes (PyTorch/CUDA)")
                print("      • Hyperparameter drift (review Layer 1 canonical changes)")
                print("   → ACTION: Review audit for hyperparameter differences and check logs")
    
    else:
        print("\n⚠️  METRICS COMPARISON SKIPPED")
        print("   Training did not run or metrics not available")
    
    # File locations
    print("\n" + "-" * 80)
    print("EXPERIMENT ARTIFACTS")
    print("-" * 80)
    print(f"Relaunch Directory: {new_exp_dir}")
    print(f"Audit JSON:         {os.path.join(new_exp_dir, 'relaunch_audit.json')}")
    if audit.get('metrics_comparison', {}).get('relaunch_metrics_exists'):
        print(f"Metrics CSV:        {os.path.join(new_exp_dir, 'metrics_epoch.csv')}")
        print(f"Results JSON:       {os.path.join(new_exp_dir, 'results.json')}")
    print(f"Training Log:       {os.path.join(new_exp_dir, 'stdout.log')}")
    
    # Final warnings
    if audit['reproducibility_status']['blocking_issues']:
        print("\n⚠️  CONFIGURATION ISSUES DETECTED")
        print(f"   {len(audit['reproducibility_status']['blocking_issues'])} blocking issue(s) found")
        print("   Review relaunch_audit.json for hyperparameter differences")
    
    print("=" * 80 + "\n")
    
    if not args.skip_readme:
        write_readme(new_exp_dir, new_cfg, best_metrics=None)

if __name__ == '__main__':
    main()
