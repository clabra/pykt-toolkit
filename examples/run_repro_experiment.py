#!/usr/bin/env python3
from __future__ import annotations
# Model-agnostic reproducible experiment launcher.
# We avoid modifying existing training scripts (e.g., `examples/train_gainakt2exp.py`).
# Instead, this wrapper:
#   1. Parses generic experiment arguments (model script, short title, seeds, devices).
#   2. Creates a timestamped experiment directory per AGENTS.md reproducibility spec.
#   3. Serializes a canonical `config.json` (resolved args + defaults) using `tmp/experiment_utils`.
#   4. Captures environment metadata to `environment.txt`.
#   5. Invokes the underlying training script as a subprocess, teeing stdout/stderr into logs with timestamps.
#   6. Expects the training script to optionally emit a JSON metrics file or performs a lightweight post-hoc extraction.
#   7. Saves `model_best.pth` / `model_last.pth` if produced by the training script.
#   8. Generates a README summarizing best metrics if found.
# Usage example:
#   python examples/run_repro_experiment.py \
#       --train_script examples/train_gainakt2exp.py \
#       --short_title baseline \
#       --dataset assist2015 \
#       --epochs 12 \
#       --batch_size 64 \
#       --learning_rate 0.000174 \
#       --seed 42
import argparse
import os
import sys
import subprocess
import shlex
import time
import json
from typing import List, Dict, Any, Optional

sys.path.insert(0, '/workspaces/pykt-toolkit')
from examples.experiment_utils import (
    make_experiment_dir, build_config, atomic_write_json, capture_environment,
    timestamped_logger, write_seed_info, write_readme
)

EPOCH_HEADER = [
    'epoch','train_loss','val_auc','val_accuracy','mastery_corr','gain_corr','constraint_loss_share'
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repro launcher (config-first) with backward-compatible training flags.")
    p.add_argument('--train_script', type=str, required=True, help='Training script path (e.g., examples/train_gainakt2exp.py)')
    p.add_argument('--model_name', type=str, default='gainakt2exp')
    p.add_argument('--short_title', type=str, default='baseline')
    p.add_argument('--purpose', type=str, default='Reproducible training run')
    p.add_argument('--output_base', type=str, default='examples/experiments')
    p.add_argument('--devices', type=int, nargs='*', default=[0,1,2,3,4])
    # Overrides: allow arbitrary --param=value matching defaults JSON; store raw extras
    p.add_argument('--override', action='append', default=[], help='Override in key=value form; can repeat.')
    # Legacy parameter flags (will be transformed into overrides automatically)
    p.add_argument('--dataset', type=str)
    p.add_argument('--fold', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--learning_rate', type=float)
    p.add_argument('--weight_decay', type=float)
    p.add_argument('--optimizer', type=str)
    p.add_argument('--gradient_clip', type=float)
    p.add_argument('--patience', type=int)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--use_wandb', action='store_true')
    # Interpretability & constraints
    p.add_argument('--use_mastery_head', action='store_true')
    p.add_argument('--use_gain_head', action='store_true')
    p.add_argument('--enhanced_constraints', action='store_true')
    p.add_argument('--non_negative_loss_weight', type=float)
    p.add_argument('--monotonicity_loss_weight', type=float)
    p.add_argument('--mastery_performance_loss_weight', type=float)
    p.add_argument('--gain_performance_loss_weight', type=float)
    p.add_argument('--sparsity_loss_weight', type=float)
    p.add_argument('--consistency_loss_weight', type=float)
    p.add_argument('--warmup_constraint_epochs', type=int)
    p.add_argument('--max_semantic_students', type=int)
    # Alignment
    p.add_argument('--enable_alignment_loss', action='store_true')
    p.add_argument('--alignment_weight', type=float)
    p.add_argument('--alignment_warmup_epochs', type=int)
    p.add_argument('--adaptive_alignment', action='store_true')
    p.add_argument('--alignment_min_correlation', type=float)
    p.add_argument('--alignment_share_cap', type=float)
    p.add_argument('--alignment_share_decay_factor', type=float)
    # Global alignment
    p.add_argument('--enable_global_alignment_pass', action='store_true')
    p.add_argument('--alignment_global_students', type=int)
    p.add_argument('--use_residual_alignment', action='store_true')
    p.add_argument('--alignment_residual_window', type=int)
    # Refinement
    p.add_argument('--enable_retention_loss', action='store_true')
    p.add_argument('--retention_delta', type=float)
    p.add_argument('--retention_weight', type=float)
    p.add_argument('--enable_lag_gain_loss', action='store_true')
    p.add_argument('--lag_gain_weight', type=float)
    p.add_argument('--lag_max_lag', type=int)
    p.add_argument('--lag_l1_weight', type=float)
    p.add_argument('--lag_l2_weight', type=float)
    p.add_argument('--lag_l3_weight', type=float)
    p.add_argument('--consistency_rebalance_epoch', type=int)
    p.add_argument('--consistency_rebalance_threshold', type=float)
    p.add_argument('--consistency_rebalance_new_weight', type=float)
    p.add_argument('--variance_floor', type=float)
    p.add_argument('--variance_floor_patience', type=int)
    p.add_argument('--variance_floor_reduce_factor', type=float)
    p.add_argument('--enable_cosine_perf_schedule', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--skip_readme', action='store_true')
    p.add_argument('--allow_drift', action='store_true', default=False)
    p.add_argument('--report_only', action='store_true', default=False)
    p.add_argument('--auto_shifted_eval', action='store_true')
    return p.parse_args()

def build_subprocess_command(train_script: str, resolved: Dict[str,Any]) -> List[str]:
    """Construct training subprocess command from resolved config dictionary (training_defaults merged with overrides).
    We only emit flags necessary for legacy training script compatibility until it is converted to config ingestion.
    """
    td = resolved['training']
    interp = resolved.get('interpretability', {})
    align = resolved.get('alignment', {})
    glob = resolved.get('global_alignment', {})
    refn = resolved.get('refinement', {})
    runtime = resolved.get('runtime', {})
    # Initial unsorted command components (we will canonicalize ordering later)
    cmd = [sys.executable, train_script,
           '--dataset', resolved['data']['dataset'],
           '--fold', str(resolved['data']['fold']),
           '--epochs', str(td['epochs']),
           '--batch_size', str(td['batch_size']),
           '--learning_rate', str(td['learning_rate']),
           '--optimizer', str(td.get('optimizer','Adam')),
           '--seed', str(runtime.get('seed', resolved['seeds']['primary'])),
           '--weight_decay', str(td['weight_decay']),
           '--gradient_clip', str(td['gradient_clip']),
           '--patience', str(td['patience'])]
    cmd.extend([
        '--monotonicity_loss_weight', str(interp['monotonicity_loss_weight']),
        '--mastery_performance_loss_weight', str(interp['mastery_performance_loss_weight']),
        '--gain_performance_loss_weight', str(interp['gain_performance_loss_weight']),
        '--sparsity_loss_weight', str(interp['sparsity_loss_weight']),
        '--consistency_loss_weight', str(interp['consistency_loss_weight']),
        '--non_negative_loss_weight', str(interp['non_negative_loss_weight']),
        '--warmup_constraint_epochs', str(interp['warmup_constraint_epochs']),
        '--max_semantic_students', str(interp['max_semantic_students'])
    ])
    if interp.get('use_mastery_head', True):
        cmd.append('--use_mastery_head')
    else:
        cmd.append('--disable_mastery_head')
    if interp.get('use_gain_head', True):
        cmd.append('--use_gain_head')
    else:
        cmd.append('--disable_gain_head')
    if interp.get('enhanced_constraints', True):
        cmd.append('--enhanced_constraints')
    align_enabled = align.get('enable_alignment_loss', False)
    if align_enabled:
        cmd.append('--enable_alignment_loss')
    cmd.extend([
        '--alignment_weight', str(align['alignment_weight']),
        '--alignment_warmup_epochs', str(align['alignment_warmup_epochs']),
        '--alignment_min_correlation', str(align['alignment_min_correlation']),
        '--alignment_share_cap', str(align['alignment_share_cap']),
        '--alignment_share_decay_factor', str(align['alignment_share_decay_factor'])
    ])
    if align.get('adaptive_alignment', False):
        cmd.append('--adaptive_alignment')
    if glob.get('enable_global_alignment_pass', False):
        cmd.append('--enable_global_alignment_pass')
    cmd.extend([
        '--alignment_global_students', str(glob['alignment_global_students']),
        '--alignment_residual_window', str(glob['alignment_residual_window'])
    ])
    if glob.get('use_residual_alignment', False):
        cmd.append('--use_residual_alignment')
    if refn.get('enable_retention_loss', False):
        cmd.extend(['--enable_retention_loss','--retention_delta', str(refn['retention_delta']), '--retention_weight', str(refn['retention_weight'])])
    if refn.get('enable_lag_gain_loss', False):
        cmd.extend(['--enable_lag_gain_loss','--lag_gain_weight', str(refn['lag_gain_weight']), '--lag_max_lag', str(refn['lag_max_lag']), '--lag_l1_weight', str(refn['lag_l1_weight']), '--lag_l2_weight', str(refn['lag_l2_weight']), '--lag_l3_weight', str(refn['lag_l3_weight'])])
    cmd.extend([
        '--consistency_rebalance_epoch', str(refn['consistency_rebalance_epoch']),
        '--consistency_rebalance_threshold', str(refn['consistency_rebalance_threshold']),
        '--consistency_rebalance_new_weight', str(refn['consistency_rebalance_new_weight']),
        '--variance_floor', str(refn['variance_floor']),
        '--variance_floor_patience', str(refn['variance_floor_patience']),
        '--variance_floor_reduce_factor', str(refn['variance_floor_reduce_factor'])
    ])
    if runtime.get('use_amp', False):
        cmd.append('--use_amp')
    # Canonical ordering (lexicographic) of flags for reproducibility string equivalence.
    # Preserve interpreter + script path; sort the remaining flags treating flag+value as a unit.
    head = cmd[:2]
    tail = cmd[2:]
    paired: List[tuple[str,List[str]]] = []
    i = 0
    while i < len(tail):
        token = tail[i]
        if token.startswith('--'):
            # Boolean flag (no value) if next token starts with '--' or doesn't exist
            if i+1 >= len(tail) or tail[i+1].startswith('--'):
                paired.append((token, [token]))
                i += 1
            else:
                # Flag-value pair
                paired.append((token, [token, tail[i+1]]))
                i += 2
        else:
            # Unexpected stray value without leading flag; treat as standalone (should not happen)
            paired.append((f'__val_{i}__', [token]))
            i += 1
    paired.sort(key=lambda x: x[0])
    canonical_tail: List[str] = []
    for _, group in paired:
        canonical_tail.extend(group)
    cmd = head + canonical_tail
    return cmd

def run_consistency_check(train_script: str) -> Dict[str,Any]:
    """Invoke check_defaults_consistency.py and parse JSON output. Returns dict."""
    checker = os.path.join(os.path.dirname(__file__), 'check_defaults_consistency.py')
    eval_script = os.path.join(os.path.dirname(__file__), 'eval_gainakt2exp.py')
    defaults_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'parameter_default.json')
    cmd = [sys.executable, checker, '--train_script', train_script, '--eval_script', eval_script, '--defaults_path', defaults_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    report = {}
    try:
        import json as _json
        report = _json.loads(out)
    except Exception:
        report = {'parse_error': out, 'stderr': err}
    report['exit_code'] = proc.returncode
    return report

def tail_metrics_from_csv(csv_path: str) -> Optional[Dict[str,Any]]:
    if not os.path.exists(csv_path):
        return None
    try:
        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
            if not rows:
                return None
            last = rows[-1]
            # Convert numeric fields when possible
            parsed = {}
            for k,v in last.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            return parsed

        def main():
            args = parse_args()
            # Pre-flight consistency check
            consistency_report = run_consistency_check(args.train_script)
            if consistency_report.get('drift_detected') and not args.allow_drift:
                print('[ConsistencyCheck] Drift detected between argparse flags and defaults. Aborting launch.', file=sys.stderr)
                print('[ConsistencyCheck] Report:', file=sys.stderr)
                print(str(consistency_report), file=sys.stderr)
                sys.exit(2)
            # Proceed with original logic (moved from bottom). Reconstruct training command etc.
            cmd = build_subprocess_command(args)
            # Device environment setup
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in args.devices)
            exp_dir = make_experiment_dir(args.model_name, args.short_title, base_dir=args.output_base)
            exp_id = os.path.basename(exp_dir)
            logger = timestamped_logger('launcher', os.path.join(exp_dir,'stdout.log'))
            logger.info(f"Launching reproducible experiment: {exp_id}")
            logger.info(f"Training script: {os.path.abspath(args.train_script)}")
            full_command_str = ' '.join(shlex.quote(c) for c in [sys.executable] + cmd[1:]) if cmd and cmd[0] == sys.executable else ' '.join(shlex.quote(x) for x in cmd)
            logger.info(f"Command: {full_command_str}")
            # Build config
            raw_args = vars(args).copy()
            raw_args['launcher_command'] = full_command_str
            raw_args['train_command'] = full_command_str
            raw_args['devices'] = args.devices
            cfg = build_config(raw_args, exp_id=exp_id, exp_path=exp_dir, seeds=[args.seed])
            # Insert eval command for snapshot reproducibility
            cfg['runtime']['eval_command'] = f"python examples/eval_gainakt2exp.py --run_dir {os.path.abspath(exp_dir)} --dataset {args.dataset}"
            atomic_write_json(cfg, os.path.join(exp_dir,'config.json'))
            capture_environment(os.path.join(exp_dir,'environment.txt'))
            write_seed_info(os.path.join(exp_dir,'SEED_INFO.md'), [args.seed], args.seed)
            if not args.skip_readme:
                write_readme(exp_dir, cfg, best_metrics=None)
            # Dry-run exit
            if args.dry_run:
                logger.info('Dry run selected; training subprocess skipped.')
                print(f"Experiment directory created (dry run): {exp_dir}")
                return
            # Launch training subprocess
            logger.info('[Launcher] Set EXPERIMENT_DIR=' + exp_dir)
            env = os.environ.copy()
            env['EXPERIMENT_DIR'] = exp_dir
            proc = subprocess.Popen(' '.join(shlex.quote(x) for x in cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            start = time.time()
            while True:
                line = proc.stdout.readline()
                if line:
                    logger.info(line.rstrip())
                elif proc.poll() is not None:
                    break
            stderr = proc.stderr.read()
            if stderr:
                with open(os.path.join(exp_dir,'stderr.log'),'w') as ef:
                    ef.write(stderr)
            rc = proc.wait()
            dur = time.time() - start
            logger.info(f"Training subprocess exit code: {rc} (duration {dur:.1f}s)")
            # Attempt to attach best metrics
            metrics_csv = os.path.join(exp_dir,'metrics_epoch.csv')
            best_metrics = tail_metrics_from_csv(metrics_csv)
            if best_metrics and not args.skip_readme:
                write_readme(exp_dir, cfg, best_metrics=best_metrics)
            print(f"Experiment directory created: {exp_dir}")

        if __name__ == '__main__':
            main()
    except Exception:
        return None

def main():
    args = parse_args()
    orig_train_script = args.train_script
    if not os.path.isabs(orig_train_script):
        # First try as given relative to project root (parent of this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidate_root = os.path.join(project_root, orig_train_script)
        if not os.path.exists(candidate_root):
            # If we are currently inside examples/ and path starts with 'examples/', try stripping the prefix
            if os.path.basename(os.getcwd()) == 'examples' and orig_train_script.startswith('examples/'):
                stripped = orig_train_script.split('/', 1)[1]
                if os.path.exists(stripped):
                    args.train_script = stripped
                else:
                    # Try relative to project root after stripping
                    stripped_root = os.path.join(project_root, stripped)
                    if os.path.exists(stripped_root):
                        args.train_script = stripped_root
            elif os.path.exists(orig_train_script):
                # Exists relative to current CWD
                args.train_script = orig_train_script
            else:
                # As last fallback keep original; will raise explicit error below if still missing
                pass
        else:
            args.train_script = candidate_root
    # Final existence check
    if not os.path.exists(args.train_script):
        raise FileNotFoundError(f"Training script not found: '{orig_train_script}' (resolved attempt: '{args.train_script}'). If you ran from inside 'examples/', omit the leading 'examples/' in --train_script.")

    # Report-only mode: run consistency check then exit WITHOUT creating experiment directory.
    if getattr(args, 'report_only', False):
        report = run_consistency_check(args.train_script)
        import json as _json
        print(_json.dumps(report, indent=2, sort_keys=True))
        if report.get('drift_detected'):
            print('[ReportOnly] Drift detected between argparse flags and defaults.')
            sys.exit(2)
        else:
            print('[ReportOnly] No drift detected.')
            sys.exit(0)
    # Load canonical defaults JSON
    defaults_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'parameter_default.json')
    with open(defaults_path) as f:
        defaults_json = json.load(f)
    training_defaults = defaults_json['training_defaults']
    # Build initial resolved structure (nested grouping similar to previous build_config output)
    # Map training_defaults flat keys into sections
    def sectionize(td: Dict[str,Any]) -> Dict[str,Any]:
        resolved = {
            'runtime': {
                'seed': td['seed'],
                'monitor_freq': td['monitor_freq'],
                'use_amp': td['use_amp'],
                'use_wandb': td['use_wandb'],
                'enable_cosine_perf_schedule': td['enable_cosine_perf_schedule'],
                'auto_shifted_eval': td.get('auto_shifted_eval', False)
            },
            'training': {
                'epochs': td['epochs'],
                'batch_size': td['batch_size'],
                'learning_rate': td['learning_rate'],
                'weight_decay': td['weight_decay'],
                'optimizer': td['optimizer'],
                'gradient_clip': td['gradient_clip'],
                'patience': td['patience']
            },
            'data': {
                'dataset': td['dataset'],
                'fold': td['fold']
            },
            'interpretability': {
                'use_mastery_head': td['use_mastery_head'],
                'use_gain_head': td['use_gain_head'],
                'enhanced_constraints': td['enhanced_constraints'],
                'non_negative_loss_weight': td['non_negative_loss_weight'],
                'monotonicity_loss_weight': td['monotonicity_loss_weight'],
                'mastery_performance_loss_weight': td['mastery_performance_loss_weight'],
                'gain_performance_loss_weight': td['gain_performance_loss_weight'],
                'sparsity_loss_weight': td['sparsity_loss_weight'],
                'consistency_loss_weight': td['consistency_loss_weight'],
                'warmup_constraint_epochs': td['warmup_constraint_epochs'],
                'max_semantic_students': td['max_semantic_students']
            },
            'alignment': {
                'enable_alignment_loss': td['enable_alignment_loss'],
                'alignment_weight': td['alignment_weight'],
                'alignment_warmup_epochs': td['alignment_warmup_epochs'],
                'adaptive_alignment': td['adaptive_alignment'],
                'alignment_min_correlation': td['alignment_min_correlation'],
                'alignment_share_cap': td['alignment_share_cap'],
                'alignment_share_decay_factor': td['alignment_share_decay_factor']
            },
            'global_alignment': {
                'enable_global_alignment_pass': td['enable_global_alignment_pass'],
                'alignment_global_students': td['alignment_global_students'],
                'use_residual_alignment': td['use_residual_alignment'],
                'alignment_residual_window': td['alignment_residual_window']
            },
            'refinement': {
                'enable_retention_loss': td['enable_retention_loss'],
                'retention_delta': td['retention_delta'],
                'retention_weight': td['retention_weight'],
                'enable_lag_gain_loss': td['enable_lag_gain_loss'],
                'lag_gain_weight': td['lag_gain_weight'],
                'lag_max_lag': td['lag_max_lag'],
                'lag_l1_weight': td['lag_l1_weight'],
                'lag_l2_weight': td['lag_l2_weight'],
                'lag_l3_weight': td['lag_l3_weight'],
                'consistency_rebalance_epoch': td['consistency_rebalance_epoch'],
                'consistency_rebalance_threshold': td['consistency_rebalance_threshold'],
                'consistency_rebalance_new_weight': td['consistency_rebalance_new_weight'],
                'variance_floor': td['variance_floor'],
                'variance_floor_patience': td['variance_floor_patience'],
                'variance_floor_reduce_factor': td['variance_floor_reduce_factor']
            }
        }
        return resolved
    resolved = sectionize(training_defaults)
    # Collect legacy flags into overrides automatically (omit None values)
    legacy_to_key = [
        'dataset','fold','seed','epochs','batch_size','learning_rate','weight_decay','optimizer','gradient_clip','patience',
        'use_amp','use_wandb','use_mastery_head','use_gain_head','enhanced_constraints','non_negative_loss_weight','monotonicity_loss_weight',
        'mastery_performance_loss_weight','gain_performance_loss_weight','sparsity_loss_weight','consistency_loss_weight','warmup_constraint_epochs',
        'max_semantic_students','enable_alignment_loss','alignment_weight','alignment_warmup_epochs','adaptive_alignment','alignment_min_correlation',
        'alignment_share_cap','alignment_share_decay_factor','enable_global_alignment_pass','alignment_global_students','use_residual_alignment',
        'alignment_residual_window','enable_retention_loss','retention_delta','retention_weight','enable_lag_gain_loss','lag_gain_weight','lag_max_lag',
        'lag_l1_weight','lag_l2_weight','lag_l3_weight','consistency_rebalance_epoch','consistency_rebalance_threshold','consistency_rebalance_new_weight',
        'variance_floor','variance_floor_patience','variance_floor_reduce_factor','enable_cosine_perf_schedule'
    ]
    override_pairs = {}
    for k in legacy_to_key:
        if hasattr(args, k):
            val = getattr(args, k)
            if val is not None:
                # For store_true style booleans: only override when user explicitly enabled (True).
                # If flag not passed (False), skip so defaults (possibly True) remain intact.
                if isinstance(val, bool) and val is False:
                    continue
                override_pairs[k] = val
    # Apply overrides from --override key=value pairs (explicit string form)
    for ov in args.override:
        if '=' not in ov:
            raise ValueError(f"Override must be key=value: {ov}")
        kk, vv = ov.split('=',1)
        kk = kk.strip()
        vv = vv.strip()
        if vv.lower() in ('true','false'):
            v_cast = vv.lower() == 'true'
        else:
            try:
                v_cast = int(vv) if vv.isdigit() else float(vv)
            except ValueError:
                v_cast = vv
        override_pairs[kk] = v_cast
    # Validate every override key exists in training_defaults
    missing = [k for k in override_pairs.keys() if k not in training_defaults]
    if missing:
        raise KeyError(f"Overrides contain unknown keys not in training_defaults: {missing}")
    # Merge
    for k,val in override_pairs.items():
        training_defaults[k] = val
    # Re-sectionize after updates
    resolved = sectionize(training_defaults)
    # Integrity: ensure every training_defaults key appears in some resolved section
    flat_keys = set(training_defaults.keys())
    section_keys = set()
    for sec_name, sec_vals in resolved.items():
        for sk in sec_vals.keys():
            section_keys.add(sk)
    missing_mapped = flat_keys - section_keys
    if missing_mapped:
        raise RuntimeError(f"Integrity check failed: some defaults keys not mapped into resolved sections: {sorted(missing_mapped)}")
    resolved['override_applied'] = override_pairs
    # Dynamic multi-GPU selection logic:
    # If user supplies --devices explicitly (non-empty list differing from defaults) we honor it.
    # Otherwise we attempt environment-based discovery:
    #   1. PYKT_VISIBLE_GPUS (comma-separated indices) overrides everything when set.
    #   2. If CUDA is available, query torch.cuda.device_count() and select up to 60% (ceil) capped at 5.
    #      Fallback to first GPU [0] if count==1.
    #   3. Persist chosen list back into args.devices so config.json records actual devices used.
    try:
        import torch  # local import to avoid hard dependency for non-training dry runs
        explicit_devices = args.devices if args.devices != [0,1,2,3,4] else None
        env_devices = os.environ.get('PYKT_VISIBLE_GPUS')
        selected: List[int] = []
        if env_devices:
            try:
                selected = [int(x) for x in env_devices.split(',') if x.strip()!='']
            except ValueError:
                selected = []
        if not selected and explicit_devices:
            selected = explicit_devices
        if not selected:
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 1:
                    import math
                    target = max(1, math.ceil(count * 0.6))
                    target = min(target, 5)  # cap per project guideline
                    selected = list(range(target))
                else:
                    selected = [0]
            else:
                selected = [0]
        args.devices = selected
    except Exception:
        # In case torch import fails (e.g., CPU-only dry environment), keep original value
        pass
    # Normalize output base to absolute path to avoid relative path duplication when launching from subdirectories
    output_base = args.output_base
    # Normalize base path relative to project root (one 'examples')
    if not os.path.isabs(output_base):
        # If script resides in /.../examples/, project root is parent
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Preserve 'examples/experiments' path hierarchy; do not collapse 'examples'
        if output_base.startswith('examples'):
            candidate = os.path.join(project_root, output_base)  # keep full relative path
        else:
            candidate = os.path.join(project_root, output_base)
        output_base = os.path.abspath(candidate)
    # Step 1: Create experiment directory
    exp_dir = make_experiment_dir(args.model_name, args.short_title, base_dir=output_base)
    exp_id = os.path.basename(exp_dir)

    # Step 2: Seeds (single-seed for now; multi-seed orchestration will wrap multiple calls)
    seeds = [args.seed]

    # Step 3: Build and dump config (config-first). Add experiment + hardware metadata.
    # Train/eval command placeholders inserted after constructing resolved.
    resolved['experiment'] = {
        'id': exp_id,
        'model': args.model_name,
        'short_title': args.short_title,
        'purpose': args.purpose
    }
    resolved['hardware'] = {
        'devices': args.devices,
        'threads': int(os.environ.get('OMP_NUM_THREADS','8')),
        'selected_devices': [],  # will be populated after dynamic selection
    }
    resolved['seeds'] = {
        'primary': resolved['runtime']['seed'],
        'all': [resolved['runtime']['seed']]
    }
    # Evaluation defaults snapshot retained from defaults_json
    eval_defaults = defaults_json.get('evaluation_defaults', {})
    resolved['evaluation_defaults'] = eval_defaults
    # Inject architecture model_config from defaults (allow overrides via legacy flags if provided in override_pairs above)
    model_cfg_defaults = defaults_json.get('model_config_defaults', {})
    # Allow user override through --override seq_len=... etc.
    model_cfg_resolved = {}
    for k,v in model_cfg_defaults.items():
        if k in override_pairs:
            model_cfg_resolved[k] = override_pairs[k]
        else:
            model_cfg_resolved[k] = v
    mandatory_arch = ['seq_len','d_model','n_heads','num_encoder_blocks','d_ff','dropout']
    missing_arch = [m for m in mandatory_arch if m not in model_cfg_resolved]
    if missing_arch:
        raise KeyError(f"Missing mandatory architecture keys during launcher resolution: {missing_arch}")
    resolved['model_config'] = model_cfg_resolved
    # Commands
    train_cmd_list = build_subprocess_command(args.train_script, resolved)
    # train_cmd_list is already canonical (flags lexicographically ordered) ensuring stable string equivalence
    verbose_train_cmd_str = ' '.join(shlex.quote(c) for c in train_cmd_list)
    launcher_command = 'python ' + ' '.join(shlex.quote(a) for a in sys.argv)
    resolved['runtime']['command'] = launcher_command
    # Minimal train command (config-first). The verbose version is retained under 'verbose_train_command' for audit.
    minimal_train_cmd_str = f"python {args.train_script} --config {os.path.join(exp_dir, 'config.json')}"
    resolved['runtime']['train_command'] = minimal_train_cmd_str
    resolved['runtime']['verbose_train_command'] = verbose_train_cmd_str
    resolved['runtime']['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    resolved['runtime']['eval_command'] = (
        f"python examples/eval_gainakt2exp.py --run_dir {exp_dir} --dataset {resolved['data']['dataset']} --experiment_id {exp_id}")
    # Minimal reproduction command (config-first): a single invocation pointing to the canonical config.json.
    # Rationale: The training script now supports --config to ingest ALL parameters from the stored file, eliminating
    # the need for verbose override enumeration and reducing drift risk. Reproduction requires copying the experiment
    # directory (or config.json) and executing this command. To retain legacy behaviour, one could reintroduce the
    # enumerated override method behind a feature flag (not requested).
    # Updated reproduction protocol: use relaunch_experiment.py to formally audit and relaunch.
    # short_title for relaunch must be the original user-provided short_title (without timestamp/model prefix) + '_relaunch'.
    # This avoids overly long folder names and keeps semantic continuity with the initial short_title.
    relaunch_short = f"{args.short_title}_relaunch" if args.short_title else "relaunch"
    resolved['runtime']['repro_command'] = (
        f"{sys.executable} examples/relaunch_experiment.py --source_dir {exp_dir} --short_title {relaunch_short}" )
    # Config hash
    from examples.experiment_utils import compute_config_hash
    resolved['config_md5'] = compute_config_hash(resolved)
    # Record placeholder gpu_selection_source; will be updated if training runs.
    resolved['hardware']['gpu_selection_source'] = 'pending'
    atomic_write_json(resolved, os.path.join(exp_dir, 'config.json'))

    # Step 4: Environment + seeds metadata
    capture_environment(os.path.join(exp_dir, 'environment.txt'))
    write_seed_info(os.path.join(exp_dir, 'SEED_INFO.md'), seeds, seeds[0])

    # Step 5: Logger (stdout.log); we still run subprocess for training
    logger = timestamped_logger('repro', os.path.join(exp_dir, 'stdout.log'))
    logger.info(f"Launching reproducible experiment: {exp_id}")
    logger.info(f"Training script: {args.train_script}")
    logger.info(f"Launcher command: {launcher_command}")
    logger.info(f"Train command (minimal): {minimal_train_cmd_str}")
    logger.info(f"Train command (verbose): {verbose_train_cmd_str}")

    # Step 6: Optionally run training
    metrics_csv = os.path.join(exp_dir, 'metrics_epoch.csv')
    if not args.dry_run:
        # Pre-create header to distinguish between "file never written" vs "no epochs" scenarios
        if not os.path.exists(metrics_csv):
            import csv as _csv
            with open(metrics_csv, 'w', newline='') as f:
                writer = _csv.writer(f)
                writer.writerow([
                    'epoch','train_loss','train_auc','val_loss','val_auc','val_accuracy',
                    'monotonicity_violation_rate','negative_gain_rate','bounds_violation_rate',
                    'mastery_correlation','gain_correlation','main_loss_share','constraint_loss_share',
                    'alignment_loss_share','lag_loss_share','retention_loss_share'
                ])
    else:
        # In dry_run: do not create metrics file to avoid confusion about missing values
        metrics_csv = None
    # Launcher now writes a separate summary file to avoid clashing with legacy training results.json
    results_json_path = os.path.join(exp_dir, 'summary.json')
    best_metrics: Optional[Dict[str,Any]] = None

    if not args.dry_run:
        env = os.environ.copy()
        # Dynamic GPU subset selection precedence:
        # 1. If CUDA_VISIBLE_DEVICES already set (explicit device list), use it as-is.
        # 2. Else if PYKT_GPU_IDS provided (comma-separated list), use that.
        # 3. Else if PYKT_NUM_GPUS provided (count), select first N from args.devices.
        # 4. Else apply heuristic: select a subset strictly <70% of available (ceil(avail*0.7)-1 if >1) else 1.
        existing_cuda = env.get('CUDA_VISIBLE_DEVICES')
        gpu_ids_env = env.get('PYKT_GPU_IDS')
        requested_env_gpus = env.get('PYKT_NUM_GPUS')
        selection_source = None
        if existing_cuda:
            # Honor pre-set CUDA_VISIBLE_DEVICES (could be set by external scheduler)
            selected_devices = [d for d in existing_cuda.split(',') if d!='']
            selection_source = 'CUDA_VISIBLE_DEVICES'
        elif gpu_ids_env:
            selected_devices = [d for d in gpu_ids_env.split(',') if d!='']
            selection_source = 'PYKT_GPU_IDS'
        elif requested_env_gpus is not None:
            try:
                num_requested = int(requested_env_gpus)
            except ValueError:
                num_requested = len(args.devices)
            selected_devices = [str(d) for d in args.devices[:num_requested]]
            selection_source = 'PYKT_NUM_GPUS'
        else:
            try:
                import torch
                import math
                avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except Exception:
                avail = 0
            if avail > 0:
                # Heuristic target (<70% but ensure at least 2 if >=2 available)
                raw_target = max(1, math.ceil(avail * 0.7))
                if raw_target >= avail and avail > 1:
                    raw_target = avail - 1  # enforce strictly <100% when possible
                num_select = max(1, raw_target)
                if num_select < 2 and avail >= 2:
                    num_select = 2  # ensure multi-GPU utilization when at least 2 are available
                selected_devices = [str(d) for d in args.devices[:num_select]]
            else:
                selected_devices = []
            selection_source = 'heuristic_<70%_min2'
        # Normalize and set CUDA_VISIBLE_DEVICES to selected list
        env['CUDA_VISIBLE_DEVICES'] = ','.join(selected_devices)
        env['PYKT_VISIBLE_GPUS'] = env['CUDA_VISIBLE_DEVICES']
        logger.info(f"[Launcher] GPU selection (subprocess): source={selection_source} available={args.devices} selected={selected_devices}")
        visible_count = len(args.devices)
        selected_count = len(selected_devices)
        logger.info(f"[Launcher] GPU counts: visible={visible_count} selected={selected_count}")
        # Update config with actual selection source for audit reproducibility
        try:
            import json as _json
            cfg_path = os.path.join(exp_dir, 'config.json')
            with open(cfg_path) as _cf:
                _cfg = _json.load(_cf)
            _cfg['hardware']['gpu_selection_source'] = selection_source
            _cfg['hardware']['visible_gpu_count'] = len(args.devices)
            _cfg['hardware']['selected_gpu_count'] = len(selected_devices)
            _cfg['hardware']['selected_devices'] = selected_devices
            atomic_write_json(_cfg, cfg_path)
        except Exception as e:
            logger.info(f"[Launcher] Warning: unable to update gpu_selection_source in config.json: {e}")
        env['EXPERIMENT_DIR'] = exp_dir
        env['PYKT_CONFIG_PATH'] = os.path.join(exp_dir, 'config.json')
        logger.info(f"[Launcher] Set EXPERIMENT_DIR={exp_dir}")
        logger.info(f"[Launcher] Set PYKT_CONFIG_PATH={env['PYKT_CONFIG_PATH']}")
        start = time.time()
        # Use verbose command for actual execution to avoid relying on file that is simultaneously being written.
        proc = subprocess.Popen(verbose_train_cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, shell=True)
        # Stream output with timestamps
        while True:
            line = proc.stdout.readline()
            if line:
                logger.info(line.rstrip())
            elif proc.poll() is not None:
                break
        stderr_out = proc.stderr.read()
        if stderr_out:
            with open(os.path.join(exp_dir,'stderr.log'),'w') as ef:
                ef.write(stderr_out)
        ret = proc.wait()
        duration = time.time() - start
        logger.info(f"Training subprocess exit code: {ret} (duration {duration:.1f}s)")
        # Post-hoc best metrics extraction attempt
        tail_metrics = tail_metrics_from_csv(metrics_csv) if metrics_csv else None
        # Determine whether epochs were appended (header-only file indicates no data rows)
        data_rows = 0
        if metrics_csv:
            try:
                import csv as _csv
                with open(metrics_csv) as f:
                    reader = _csv.reader(f)
                    next(reader, None)
                    for _ in reader:
                        data_rows += 1
            except Exception:
                pass
        if tail_metrics and data_rows > 0:
            best_metrics = {
                'best_epoch': tail_metrics.get('epoch'),
                'best_val_auc': tail_metrics.get('val_auc'),
                'best_val_accuracy': tail_metrics.get('val_accuracy'),
                'mastery_corr': tail_metrics.get('mastery_corr'),
                'gain_corr': tail_metrics.get('gain_corr'),
            }
            atomic_write_json({'tail_metrics': tail_metrics, 'best_summary': best_metrics, 'config_md5': resolved['config_md5']}, results_json_path)
        else:
            note = 'metrics_epoch.csv present but no data rows appended (training script may not have written metrics).' if metrics_csv and os.path.exists(metrics_csv) else 'No metrics file (dry_run or script not adapted).'
            failure = ret != 0
            payload = {'config_md5': resolved['config_md5'], 'note': note}
            if failure:
                payload['exit_code'] = ret
                if stderr_out:
                    payload['stderr_excerpt'] = stderr_out[:5000]
            atomic_write_json(payload, results_json_path)
    else:
        atomic_write_json({'config_md5': resolved['config_md5'], 'dry_run': True, 'note': 'Dry run: metrics and training skipped.'}, results_json_path)

    # Step 7: README generation (optional)
    if not args.skip_readme:
        # Reuse write_readme with resolved config; adapt format to expected keys
        write_readme(exp_dir, resolved, best_metrics)

    print(f"Experiment directory created: {exp_dir}")

    # Step 8: Optional post-training shifted evaluation (next-step metrics)
    # Guard conditions: auto_shifted_eval enabled AND training actually ran (not dry_run) AND checkpoint exists.
    if getattr(args, 'auto_shifted_eval', False) and not args.dry_run:
        ckpt = os.path.join(exp_dir, 'model_best.pth')
        helper = os.path.join(os.path.dirname(__file__), '..', 'tmp', 'shifted_eval_helper.py')
        if os.path.exists(helper):
            if os.path.exists(ckpt):
                import subprocess as _sp
                import json as _json
                # Build helper command mirroring architectural defaults; rely on eval helper internal defaults for d_model etc if not trained with overrides.
                helper_cmd = [sys.executable, helper,
                              '--run_dir', exp_dir,
                              '--dataset', args.dataset,
                              '--fold', str(args.fold),
                              '--batch_size', str(args.batch_size),
                              '--seq_len', '200',
                              '--d_model', '512', '--n_heads', '8', '--num_encoder_blocks', '6', '--d_ff', '1024', '--dropout', '0.2']
                try:
                    print('[PostEval] Running shifted next-step evaluation helper...')
                    proc = _sp.Popen(helper_cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True)
                    out, err = proc.communicate()
                    if proc.returncode == 0:
                        # Attempt to parse final JSON block from stdout (helper prints results JSON)
                        # Simple heuristic: last '{' to end
                        last_brace = out.rfind('{')
                        shifted_results = None
                        if last_brace != -1:
                            try:
                                shifted_results = _json.loads(out[last_brace:])
                            except Exception:
                                shifted_results = {'parse_error': 'Could not parse helper JSON from stdout'}
                        else:
                            shifted_results = {'parse_error': 'No JSON object found in helper stdout'}
                        # Merge into results.json if present
                        results_path = os.path.join(exp_dir, 'results.json')
                        merged = {}
                        if os.path.exists(results_path):
                            try:
                                with open(results_path) as rf:
                                    merged = _json.load(rf)
                            except Exception:
                                merged = {'load_error': 'Failed to read existing results.json'}
                        merged['shifted_eval'] = shifted_results
                        with open(results_path, 'w') as wf:
                            _json.dump(merged, wf, indent=2)
                        print('[PostEval] Shifted evaluation results merged into results.json')
                    else:
                        print(f'[PostEval] Helper returned non-zero exit code {proc.returncode}. stderr:\n{err[:5000]}')
                except Exception as e:
                    print(f'[PostEval] Exception during shifted evaluation: {e}')
            else:
                print('[PostEval] Skipping shifted evaluation: model_best.pth not found.')
        else:
            print('[PostEval] Skipping shifted evaluation: helper script missing (tmp/shifted_eval_helper.py).')

if __name__ == '__main__':
    main()
