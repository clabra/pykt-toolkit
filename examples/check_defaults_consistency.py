#!/usr/bin/env python3
"""Check argparse flags vs parameter_default.json keys for drift.

Usage:
  python examples/check_defaults_consistency.py \
    --train_script examples/train_gainakt2exp.py \
    --eval_script examples/eval_gainakt2exp.py \
    --defaults_path configs/parameter_default.json

Exit code 0 if consistent, 1 if drift detected.
"""
import argparse
import subprocess
import sys
import json
import re
from typing import Set, Tuple

IGNORED_TRAIN_FLAGS = {
    'help',
    'version'
}
# Flags that appear as negatives or operational toggles not stored in defaults
SPECIAL_BOOLEAN_DISABLE_PREFIXES = (
    'disable_', 'pure_bce'
)
def load_metadata_ignored(defaults: dict) -> Tuple[Set[str], Set[str]]:
    meta = defaults.get('metadata', {})
    ignored_eval = {f for f in meta.get('ignored_eval_flags', [])}
    ignored_train = {f for f in meta.get('ignored_training_flags', [])}
    return ignored_eval, ignored_train

def extract_flags(script_path: str) -> dict:
    """Return mapping flag->info (currently only names)."""
    try:
        out = subprocess.check_output([sys.executable, script_path, '--help'], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        out = e.output
    flags = {}
    pattern = re.compile(r'--([a-zA-Z0-9_\-]+)')
    for token in pattern.findall(out):
        name = token.strip()
        if name in IGNORED_TRAIN_FLAGS:
            continue
        flags[name] = {}
    return flags

def normalize(flag: str) -> str:
    return flag.replace('-', '_')

def load_defaults(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def classify(train_flags: dict, defaults_keys: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    missing_in_json = set()
    missing_in_argparse = set()
    extraneous_defaults = set()
    # Train flags not in defaults
    for f in train_flags:
        norm = normalize(f)
        if any(norm.startswith(pref) for pref in SPECIAL_BOOLEAN_DISABLE_PREFIXES):
            continue
        if norm not in defaults_keys:
            missing_in_json.add(norm)
    # Defaults keys not represented by a train flag
    for k in defaults_keys:
        # Some defaults live only in runtime (seed, monitor_freq) or evaluation; allow those
        if k in {'seed','monitor_freq','use_amp','use_wandb','enable_cosine_perf_schedule'}:
            continue
        if k not in {normalize(f) for f in train_flags}:
            missing_in_argparse.add(k)
    # Extraneous defaults (heuristic: keys that look deprecated)
    for k in defaults_keys:
        if k.startswith('deprecated_'):
            extraneous_defaults.add(k)
    return missing_in_json, missing_in_argparse, extraneous_defaults

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_script', required=True)
    ap.add_argument('--eval_script', required=True)
    ap.add_argument('--defaults_path', default='configs/parameter_default.json')
    args = ap.parse_args()
    defaults = load_defaults(args.defaults_path)
    ignored_eval_flags, ignored_train_flags = load_metadata_ignored(defaults)
    train_defaults = set(defaults.get('training_defaults', {}).keys())
    eval_defaults = set(defaults.get('evaluation_defaults', {}).keys())
    model_config_defaults = set(defaults.get('model_config_defaults', {}).keys())
    train_flags = extract_flags(args.train_script)
    eval_flags = extract_flags(args.eval_script)

    # Remove ignored training flags from train_flags before classification
    train_flags_filtered = {f: v for f, v in train_flags.items() if normalize(f) not in ignored_train_flags}
    miss_json_train, miss_argparse_train, extra_train = classify(train_flags_filtered, train_defaults)
    # Filter evaluation flags before classification
    eval_flags_filtered = {f: v for f,v in eval_flags.items() if normalize(f) not in ignored_eval_flags}
    miss_json_eval, miss_argparse_eval, extra_eval = classify(eval_flags_filtered, eval_defaults)

    # Architecture keys are not expected to appear in training argparse (currently); exclude them from drift unless missing in defaults
    arch_missing_in_defaults = [k for k in ['seq_len','d_model','n_heads','num_encoder_blocks','d_ff','dropout'] if k not in model_config_defaults]
    drift = any([miss_json_train, miss_argparse_train, extra_train, miss_json_eval, miss_argparse_eval, extra_eval, arch_missing_in_defaults])

    report = {
        'train_script': args.train_script,
        'eval_script': args.eval_script,
        'missing_in_json_training': sorted(list(miss_json_train)),
        'missing_in_argparse_training': sorted(list(miss_argparse_train)),
        'extraneous_training_defaults': sorted(list(extra_train)),
        'missing_in_json_evaluation': sorted(list(miss_json_eval)),
        'missing_in_argparse_evaluation': sorted(list(miss_argparse_eval)),
        'extraneous_evaluation_defaults': sorted(list(extra_eval)),
        'model_config_defaults': sorted(list(model_config_defaults)),
        'architecture_missing_in_defaults': arch_missing_in_defaults,
        'ignored_eval_flags': sorted(list(ignored_eval_flags)),
        'ignored_training_flags': sorted(list(ignored_train_flags)),
        'drift_detected': drift
    }
    print(json.dumps(report, indent=2))
    sys.exit(1 if drift else 0)

if __name__ == '__main__':
    main()
