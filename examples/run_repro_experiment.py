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
    p = argparse.ArgumentParser(description="Generic reproducible experiment wrapper")
    p.add_argument('--train_script', type=str, required=True,
                   help='Path to existing training script to execute (e.g., examples/train_gainakt2exp.py).')
    p.add_argument('--model_name', type=str, default='gainakt2exp')
    p.add_argument('--short_title', type=str, default='baseline')
    p.add_argument('--purpose', type=str, default='Reproducible training run')
    p.add_argument('--dataset', type=str, default='assist2015')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=0.000174)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type (currently only Adam supported)')
    p.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping max norm')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    p.add_argument('--extra_args', type=str, nargs='*', default=[],
                   help='Additional raw args forwarded to the training script.')
    p.add_argument('--devices', type=int, nargs='*', default=[0,1,2,3,4])
    p.add_argument('--use_amp', action='store_true')
    # Core interpretability heads & constraint weights
    # Interpretability defaults: ENABLE semantic components unless user disables
    p.add_argument('--use_mastery_head', action='store_true', default=True, help='Enable mastery head (default: enabled). Use --disable_mastery_head to turn off.')
    p.add_argument('--disable_mastery_head', action='store_true', help='Disable mastery head')
    p.add_argument('--use_gain_head', action='store_true', default=True, help='Enable gain head (default: enabled). Use --disable_gain_head to turn off.')
    p.add_argument('--disable_gain_head', action='store_true', help='Disable gain head')
    p.add_argument('--enhanced_constraints', action='store_true', default=True, help='Use enhanced constraint preset (default: enabled). Use --pure_bce to disable.')
    p.add_argument('--pure_bce', action='store_true', help='Disable enhanced constraints (pure BCE baseline)')
    p.add_argument('--non_negative_loss_weight', type=float, default=0.0)
    p.add_argument('--monotonicity_loss_weight', type=float, default=0.1)
    p.add_argument('--mastery_performance_loss_weight', type=float, default=0.8)
    p.add_argument('--gain_performance_loss_weight', type=float, default=0.8)
    p.add_argument('--sparsity_loss_weight', type=float, default=0.2)
    p.add_argument('--consistency_loss_weight', type=float, default=0.3)
    # Semantic interpretability / alignment flags (captured for config reproducibility)
    p.add_argument('--enable_alignment_loss', action='store_true', default=True, help='Enable local alignment loss (default: enabled). Use --disable_alignment_loss to turn off.')
    p.add_argument('--disable_alignment_loss', action='store_true', help='Disable local alignment loss')
    p.add_argument('--alignment_weight', type=float, default=0.25, help='Base alignment weight')
    p.add_argument('--alignment_warmup_epochs', type=int, default=8, help='Warm-up epochs for alignment scaling')
    p.add_argument('--adaptive_alignment', action='store_true', default=True, help='Enable adaptive alignment scaling logic (default: enabled). Use --disable_adaptive_alignment to turn off.')
    p.add_argument('--disable_adaptive_alignment', action='store_true', help='Disable adaptive alignment scaling')
    p.add_argument('--alignment_min_correlation', type=float, default=0.05, help='Minimum correlation to retain/increase alignment weight')
    p.add_argument('--alignment_share_cap', type=float, default=0.08, help='Cap on alignment loss share (fraction of total loss)')
    p.add_argument('--alignment_share_decay_factor', type=float, default=0.7, help='Decay factor applied when alignment exceeds cap without improvement')
    # Global residual alignment
    p.add_argument('--enable_global_alignment_pass', action='store_true', default=True, help='Enable global alignment consistency pass (default: enabled). Use --disable_global_alignment_pass to turn off.')
    p.add_argument('--disable_global_alignment_pass', action='store_true', help='Disable global alignment pass')
    p.add_argument('--alignment_global_students', type=int, default=600, help='Number of students for global alignment sampling')
    p.add_argument('--use_residual_alignment', action='store_true', help='Use residualization for global alignment')
    p.add_argument('--alignment_residual_window', type=int, default=5, help='Window size for residual alignment calculations')
    # Additional semantic sampling / scheduling args used by training script
    p.add_argument('--warmup_constraint_epochs', type=int, default=4, help='Warm-up epochs for performance alignment losses')
    p.add_argument('--max_semantic_students', type=int, default=50, help='Students sampled for consistency semantic correlations')
    # Retention & Lag objectives
    p.add_argument('--enable_retention_loss', action='store_true', default=True, help='Enable retention loss to preserve mastery peaks (default: enabled). Use --disable_retention_loss to turn off.')
    p.add_argument('--disable_retention_loss', action='store_true', help='Disable retention loss')
    p.add_argument('--retention_delta', type=float, default=0.005, help='Minimum decay before retention penalty triggers')
    p.add_argument('--retention_weight', type=float, default=0.14, help='Retention penalty scaling weight')
    p.add_argument('--enable_lag_gain_loss', action='store_true', default=True, help='Enable lag-based gain emergence loss (default: enabled). Use --disable_lag_gain_loss to turn off.')
    p.add_argument('--disable_lag_gain_loss', action='store_true', help='Disable lag-based gain emergence loss')
    p.add_argument('--lag_gain_weight', type=float, default=0.06, help='Base lag gain weight')
    p.add_argument('--lag_max_lag', type=int, default=3, help='Maximum lag depth considered')
    p.add_argument('--lag_l1_weight', type=float, default=0.5, help='Lag level-1 weight')
    p.add_argument('--lag_l2_weight', type=float, default=0.3, help='Lag level-2 weight')
    p.add_argument('--lag_l3_weight', type=float, default=0.2, help='Lag level-3 weight')
    # Consistency rebalance & variance
    p.add_argument('--consistency_rebalance_epoch', type=int, default=8, help='Epoch to potentially rebalance consistency weight')
    p.add_argument('--consistency_rebalance_threshold', type=float, default=0.1, help='Mastery corr threshold for rebalance trigger')
    p.add_argument('--consistency_rebalance_new_weight', type=float, default=0.2, help='New consistency weight after rebalance')
    p.add_argument('--variance_floor', type=float, default=0.0001, help='Floor for mastery variance monitoring')
    p.add_argument('--variance_floor_patience', type=int, default=3, help='Patience epochs before adjusting floor')
    p.add_argument('--variance_floor_reduce_factor', type=float, default=0.5, help='Factor to reduce floor if needed')
    # Use project-root anchored default to avoid examples/examples duplication
    p.add_argument('--output_base', type=str, default='examples/experiments')
    p.add_argument('--skip_readme', action='store_true')
    p.add_argument('--dry_run', action='store_true', help='Create folder & config only; skip training subprocess.')
    return p.parse_args()

def build_subprocess_command(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, args.train_script,
           '--dataset', args.dataset,
           '--fold', str(args.fold),
           '--epochs', str(args.epochs),
           '--batch_size', str(args.batch_size),
           '--learning_rate', str(args.learning_rate),
           '--seed', str(args.seed),
           '--weight_decay', str(args.weight_decay),
           '--gradient_clip', str(args.gradient_clip),
           '--patience', str(args.patience)]
    if args.optimizer:
        cmd.extend(['--optimizer', args.optimizer])
    # Heads & constraints (now enabled by default in training script). We only propagate explicit disables via extra args.
    # Preserve explicit enable flags if user supplied them for clarity.
    if getattr(args, 'disable_mastery_head', False):
        # Training script interprets disable flag; no need to send enable flag
        cmd.append('--disable_mastery_head')
    elif args.use_mastery_head:
        cmd.append('--use_mastery_head')
    if getattr(args, 'disable_gain_head', False):
        cmd.append('--disable_gain_head')
    elif args.use_gain_head:
        cmd.append('--use_gain_head')
    if getattr(args, 'pure_bce', False):
        cmd.append('--pure_bce')
    elif args.enhanced_constraints:
        cmd.append('--enhanced_constraints')
    cmd.extend([
        '--non_negative_loss_weight', str(args.non_negative_loss_weight),
        '--monotonicity_loss_weight', str(args.monotonicity_loss_weight),
        '--mastery_performance_loss_weight', str(args.mastery_performance_loss_weight),
        '--gain_performance_loss_weight', str(args.gain_performance_loss_weight),
        '--sparsity_loss_weight', str(args.sparsity_loss_weight),
        '--consistency_loss_weight', str(args.consistency_loss_weight)
    ])
    # Forward semantic flags if enabled / set (keep defaults for transparency)
    # Alignment & semantic objectives default enabled; only append enable flag if explicitly set (transparency) or disabled flag absent.
    if getattr(args, 'disable_alignment_loss', False):
        cmd.append('--disable_alignment_loss')
    elif args.enable_alignment_loss:
        cmd.append('--enable_alignment_loss')
    cmd.extend([
        '--alignment_weight', str(args.alignment_weight),
        '--alignment_warmup_epochs', str(args.alignment_warmup_epochs),
        '--alignment_min_correlation', str(args.alignment_min_correlation),
        '--alignment_share_cap', str(args.alignment_share_cap),
        '--alignment_share_decay_factor', str(args.alignment_share_decay_factor),
        '--warmup_constraint_epochs', str(args.warmup_constraint_epochs),
        '--max_semantic_students', str(args.max_semantic_students)
    ])
    if getattr(args, 'disable_adaptive_alignment', False):
        cmd.append('--disable_adaptive_alignment')
    elif args.adaptive_alignment:
        cmd.append('--adaptive_alignment')
    if getattr(args, 'disable_global_alignment_pass', False):
        cmd.append('--disable_global_alignment_pass')
    elif args.enable_global_alignment_pass:
        cmd.append('--enable_global_alignment_pass')
    cmd.extend([
        '--alignment_global_students', str(args.alignment_global_students),
        '--alignment_residual_window', str(args.alignment_residual_window)
    ])
    if args.use_residual_alignment:
        cmd.append('--use_residual_alignment')
    if getattr(args, 'disable_retention_loss', False):
        cmd.append('--disable_retention_loss')
    elif args.enable_retention_loss:
        cmd.extend(['--enable_retention_loss', '--retention_delta', str(args.retention_delta), '--retention_weight', str(args.retention_weight)])
    if getattr(args, 'disable_lag_gain_loss', False):
        cmd.append('--disable_lag_gain_loss')
    elif args.enable_lag_gain_loss:
        cmd.extend(['--enable_lag_gain_loss', '--lag_gain_weight', str(args.lag_gain_weight), '--lag_max_lag', str(args.lag_max_lag), '--lag_l1_weight', str(args.lag_l1_weight), '--lag_l2_weight', str(args.lag_l2_weight), '--lag_l3_weight', str(args.lag_l3_weight)])
    cmd.extend([
        '--consistency_rebalance_epoch', str(args.consistency_rebalance_epoch),
        '--consistency_rebalance_threshold', str(args.consistency_rebalance_threshold),
        '--consistency_rebalance_new_weight', str(args.consistency_rebalance_new_weight),
        '--variance_floor', str(args.variance_floor),
        '--variance_floor_patience', str(args.variance_floor_patience),
        '--variance_floor_reduce_factor', str(args.variance_floor_reduce_factor)
    ])
    if args.use_amp:
        cmd.append('--use_amp')
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd

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
    except Exception:
        return None

def main():
    args = parse_args()
    # Normalize / validate train_script path (common user error: running from examples/ and passing 'examples/train_*.py')
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

    # Step 3: Build and dump config
    raw_args = vars(args).copy()
    train_cmd_list = build_subprocess_command(args)
    raw_args['train_command'] = ' '.join(shlex.quote(c) for c in train_cmd_list)
    # Launcher command reproduction (python path + full argv for this script)
    raw_args['launcher_command'] = 'python ' + ' '.join(shlex.quote(a) for a in sys.argv)
    raw_args['command'] = raw_args['launcher_command']  # backward compatibility
    # Pre-build evaluation command template; run_dir placeholder replaced after exp_dir creation
    # We rely on architectural defaults used in training script (d_model, n_heads, etc.). These are not passed here; evaluation script will infer them or user can adjust.
    # Include heads if flags set to ensure correlation computation works identically at eval time.
    eval_parts = [
        'python', 'examples/eval_gainakt2exp.py', '--run_dir', '{EXP_DIR}', '--dataset', args.dataset
    ]
    # Evaluation script defaults:
    eval_defaults = {
        'fold': 0,
        'batch_size': 96,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 8,
        'num_encoder_blocks': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'use_mastery_head': True,
        'use_gain_head': True
    }
    # Add only overrides vs these defaults
    if args.fold != eval_defaults['fold']:
        eval_parts.extend(['--fold', str(args.fold)])
    if args.batch_size != eval_defaults['batch_size']:
        eval_parts.extend(['--batch_size', str(args.batch_size)])
    # Heads: include disable flag if turned off
    if getattr(args,'disable_mastery_head',False) or not args.use_mastery_head:
        eval_parts.append('--disable_mastery_head')
    if getattr(args,'disable_gain_head',False) or not args.use_gain_head:
        eval_parts.append('--disable_gain_head')
    # If any architectural overrides were passed through extra_args or differ from defaults, include them
    # We inspect raw_args for presence
    raw_eval_related = {k: raw_args.get(k) for k in ['seq_len','d_model','n_heads','num_encoder_blocks','d_ff','dropout'] if k in raw_args}
    for k,v in raw_eval_related.items():
        if v is not None and str(v) != str(eval_defaults.get(k)):
            eval_parts.extend([f'--{k}', str(v)])
    raw_args['eval_command_template'] = ' '.join(eval_parts)
    cfg = build_config(raw_args, exp_id=exp_id, exp_path=exp_dir, seeds=seeds)
    # Inject resolved evaluation command now that EXP_DIR known
    eval_cmd = raw_args.get('eval_command_template','').replace('{EXP_DIR}', exp_dir)
    if 'runtime' in cfg:
        cfg['runtime']['eval_command'] = eval_cmd
    else:
        cfg['runtime'] = {'eval_command': eval_cmd}
    atomic_write_json(cfg, os.path.join(exp_dir, 'config.json'))

    # Step 4: Environment + seeds metadata
    capture_environment(os.path.join(exp_dir, 'environment.txt'))
    write_seed_info(os.path.join(exp_dir, 'SEED_INFO.md'), seeds, seeds[0])

    # Step 5: Logger (stdout.log); we still run subprocess for training
    logger = timestamped_logger('repro', os.path.join(exp_dir, 'stdout.log'))
    logger.info(f"Launching reproducible experiment: {exp_id}")
    logger.info(f"Training script: {args.train_script}")
    logger.info(f"Command: {raw_args['command']}")

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
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in args.devices)
        # Also expose chosen devices via PYKT_VISIBLE_GPUS for downstream scripts (evaluation, relaunch) if not already set.
        env.setdefault('PYKT_VISIBLE_GPUS', env['CUDA_VISIBLE_DEVICES'])
        env['EXPERIMENT_DIR'] = exp_dir
        logger.info(f"[Launcher] Set EXPERIMENT_DIR={exp_dir}")
        start = time.time()
        proc = subprocess.Popen(build_subprocess_command(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
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
            atomic_write_json({'tail_metrics': tail_metrics, 'best_summary': best_metrics, 'config_md5': cfg['config_md5']}, results_json_path)
        else:
            note = 'metrics_epoch.csv present but no data rows appended (training script may not have written metrics).' if metrics_csv and os.path.exists(metrics_csv) else 'No metrics file (dry_run or script not adapted).'
            failure = ret != 0
            payload = {'config_md5': cfg['config_md5'], 'note': note}
            if failure:
                payload['exit_code'] = ret
                # Include truncated stderr snippet for quick diagnosis
                if stderr_out:
                    payload['stderr_excerpt'] = stderr_out[:5000]
            atomic_write_json(payload, results_json_path)
    else:
        atomic_write_json({'config_md5': cfg['config_md5'], 'dry_run': True, 'note': 'Dry run: metrics and training skipped.'}, results_json_path)

    # Step 7: README generation (optional)
    if not args.skip_readme:
        write_readme(exp_dir, cfg, best_metrics)

    print(f"Experiment directory created: {exp_dir}")

if __name__ == '__main__':
    main()
