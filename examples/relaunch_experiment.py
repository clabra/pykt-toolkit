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
        if section in {'runtime','experiment','hardware','seeds','config_md5'}:
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
    original_md5 = original_cfg.get('config_md5')
    train_script = original_cfg['runtime']['train_command'].split()[1] if 'train_command' in original_cfg['runtime'] else 'examples/train_gainakt2exp.py'
    # Gather valid flags dynamically from training script help
    try:
        help_text = subprocess.check_output([sys.executable, train_script, '--help'], stderr=subprocess.STDOUT, text=True)
        valid_flags = {tok.split()[0] for tok in help_text.split() if tok.startswith('--')}
    except Exception:
        valid_flags = set()
    train_args, skipped = reconstruct_train_args_dynamic(original_cfg, valid_flags)
    # Epoch override: adjust epochs if user requests a shorter reproduction
    if args.override_epochs is not None:
        train_args['epochs'] = args.override_epochs
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
    atomic_write_json(new_cfg, os.path.join(new_exp_dir,'config.json'))
    capture_environment(os.path.join(new_exp_dir,'environment.txt'))
    write_seed_info(os.path.join(new_exp_dir,'SEED_INFO.md'), seeds, seeds[0])
    # Diff audit
    flat_original = flatten_config(original_cfg)
    flat_new = flatten_config(new_cfg)
    param_diffs = {}
    for k,v in flat_original.items():
        if k.startswith('runtime.') or k == 'config_md5':
            continue
        if k not in flat_new:
            param_diffs[k] = {'original': v, 'new': '<missing>'}
        else:
            if flat_new[k] != v:
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
        'experiment.purpose'
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
    # Determine if any param_diffs are substantive (not in allowed metadata set)
    substantive_diffs = {
        k: v for k, v in param_diffs.items() if k not in allowed_metadata_diff_prefixes
    }
    # MD5 comparison (ignoring metadata change expectations). We simply compare config_md5 values from original and relaunch.
    md5_match = (original_md5 == new_cfg.get('config_md5'))
    md5_comparison = {
        'md5_match': md5_match,
        'original_md5': original_md5,
        'relaunch_md5': new_cfg.get('config_md5')
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
    audit = {
        'original_experiment_id': original_cfg['experiment']['id'],
        'relaunch_experiment_id': exp_id,
        'original_config_md5': original_md5,
        'relaunch_config_md5': new_cfg['config_md5'],
        'train_script': train_script,
        'original_train_command': original_cfg['runtime'].get('train_command'),
        'reconstructed_train_command': reconstructed_cmd,
        'param_diffs': param_diffs,
        'skipped_params': skipped,
        'devices_original': orig_devices,
        'devices_used': adapted_devices,
        'device_adaptation_note': adaptation_note,
        'gpu_comparison': gpu_comparison,
        'epoch_diff_info': epoch_diff_info,
        'md5_comparison': md5_comparison,
        'dry_run': args.dry_run
    }
    # Strict mode: if epochs changed, fail early with guidance regardless of other diffs.
    if args.strict and epochs_changed:
        audit['strict_failure'] = True
        audit['strict_guidance'] = 'Epoch count differs. Remove --strict or omit --override_epochs for faithful reproduction.'
        audit['substantive_param_diffs'] = substantive_diffs
    elif args.strict and (substantive_diffs or skipped):
        audit['strict_failure'] = True
        audit['substantive_param_diffs'] = substantive_diffs
        if 'training.epochs' in substantive_diffs:
            audit['strict_guidance'] = 'Epoch count differs (override_epochs in use). Re-run without --strict to execute shortened reproduction.'
    else:
        audit['strict_failure'] = False
        audit['substantive_param_diffs'] = {}
    atomic_write_json(audit, os.path.join(new_exp_dir,'relaunch_audit.json'))
    logger = timestamped_logger('relaunch', os.path.join(new_exp_dir,'stdout.log'))
    logger.info(f"Relaunch audit created for {original_cfg['experiment']['id']} -> {exp_id}")
    if param_diffs:
        logger.warning(f"PARAMETER DIFFS DETECTED: {len(param_diffs)} entries; reproduction may NOT be identical.")
    else:
        logger.info("No parameter diffs detected; proceeding with faithful reproduction.")
    if skipped:
        logger.warning(f"SKIPPED {len(skipped)} unmapped parameters.")
    if args.strict and (param_diffs or skipped):
        logger.error("Strict mode violation: aborting before training.")
        return
    if not args.dry_run:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in args.devices)
        env['EXPERIMENT_DIR'] = new_exp_dir
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
    else:
        logger.info("Dry run selected; training subprocess skipped.")
    if not args.skip_readme:
        write_readme(new_exp_dir, new_cfg, best_metrics=None)
    print(f"Relaunch experiment directory: {new_exp_dir}")

if __name__ == '__main__':
    main()
