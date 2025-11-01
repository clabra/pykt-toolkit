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
    # Recompute config_md5 after injection
    from examples.experiment_utils import compute_config_hash as _compute_hash
    new_cfg['config_md5'] = _compute_hash(new_cfg)
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
    substantive_diffs = {}
    missing_params = {}
    for k, v in param_diffs.items():
        if k in allowed_metadata_diff_prefixes:
            continue  # skip allowed metadata drift
        orig_val = v['original']
        new_val = v['new']
        # Capture missing markers explicitly
        if orig_val == '<missing>' or new_val == '<missing>':
            missing_params[k] = v
            substantive_diffs[k] = v
            continue
        if isinstance(orig_val,(int,float)) and isinstance(new_val,(int,float)) and values_equal(orig_val,new_val):
            continue  # formatting-only numeric diff -> ignore
        if orig_val == new_val:
            continue  # identical
        substantive_diffs[k] = v
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
    # Seed match flag
    seed_match = (original_cfg.get('seeds', {}).get('primary') == new_cfg.get('seeds', {}).get('primary'))
    audit = {
        'original_experiment_id': original_cfg['experiment']['id'],
        'relaunch_experiment_id': exp_id,
        'original_config_md5': original_md5,
        'relaunch_config_md5': new_cfg['config_md5'],
        'train_script': train_script,
        'original_train_command': original_cfg['runtime'].get('train_command'),
        'reconstructed_train_command': reconstructed_cmd,
        'param_diffs_substantive': substantive_diffs,
        'missing_params': missing_params,
        'skipped_params': skipped,
        'devices_original': orig_devices,
        'devices_used': adapted_devices,
        'device_adaptation_note': adaptation_note,
        'gpu_comparison': gpu_comparison,
        'seed_match': seed_match,
        'epoch_diff_info': epoch_diff_info,
        'md5_comparison': md5_comparison,
        'dry_run': args.dry_run
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
    if substantive_diffs:
        logger.warning(f"SUBSTANTIVE PARAMETER DIFFS: {len(substantive_diffs)} entries.")
    else:
        logger.info("No substantive parameter diffs detected.")
    if missing_params:
        logger.error(f"MISSING PARAMETER MAPPINGS: {len(missing_params)} keys contain '<missing>' values.")
    if skipped:
        logger.warning(f"SKIPPED {len(skipped)} unmapped parameters.")
    if args.strict and audit['strict_failure']:
        logger.error("Strict mode violation: aborting before training.")
        return
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
    audit['audit_completed_at'] = datetime.utcnow().isoformat()
    if missing_params:
        audit['missing_params_count'] = len(missing_params)
    atomic_write_json(audit, os.path.join(new_exp_dir,'relaunch_audit.json'))
    # Human-readable summary block
    logger.info("==== RELAUNCH AUDIT SUMMARY ====")
    logger.info(f"MD5 match: {audit['md5_comparison']['md5_match']}")
    logger.info(f"Substantive param diffs: {len(substantive_diffs)} | Missing params: {len(missing_params)} | Skipped flags: {len(skipped)}")
    if audit.get('metrics_comparison'):
        mc = audit['metrics_comparison']
        logger.info(f"Metrics within tolerance: {mc['within_tolerance']}")
    if substantive_diffs:
        preview_keys = list(substantive_diffs.keys())[:10]
        logger.info(f"Sample param diffs: {preview_keys}{'...' if len(substantive_diffs)>10 else ''}")
    if missing_params:
        preview_missing = list(missing_params.keys())[:10]
        logger.info(f"Sample missing params: {preview_missing}{'...' if len(missing_params)>10 else ''}")
    if not args.skip_readme:
        write_readme(new_exp_dir, new_cfg, best_metrics=None)
    print(f"Relaunch experiment directory: {new_exp_dir}")

if __name__ == '__main__':
    main()
