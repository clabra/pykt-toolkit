#!/usr/bin/env python3
"""Reproducible experiment launcher for GainAKT2Exp.

This script centralizes three reproducibility workflows: (1) launching a new experiment; (2) reproducing an existing
experiment; and (3) comparing two finished experiment folders for metric and config hash equivalence.

--------------------------------------------------------------------------------
1) Launch a NEW experiment (creates folder + config.json + runs training)
--------------------------------------------------------------------------------
Minimal:
    python examples/repro_train_gainakt2exp.py --short-title baseline

Specify explicit directory name (must be unique unless --force):
    python examples/repro_train_gainakt2exp.py --experiment-dir 20251026_120000_gainakt2exp_mytrial --short-title mytrial

Override epochs at creation time (applies BEFORE md5 hashing) via generic override:
    python examples/repro_train_gainakt2exp.py --short-title trajtest --set training.epochs=1

Resume from checkpoint (only override honored even if config already exists):
    python examples/repro_train_gainakt2exp.py --experiment-dir 20251026_120000_gainakt2exp_mytrial \
            --resume /path/to/model_last.pth

Force reuse of an existing populated folder (will NOT change parameters unless overrides explicitly allowed):
    python examples/repro_train_gainakt2exp.py --experiment-dir 20251026_120000_gainakt2exp_mytrial --force

--------------------------------------------------------------------------------
2) REPRODUCE an existing experiment (clone config + retrain + compare)
--------------------------------------------------------------------------------
Given a completed source experiment folder (e.g. examples/experiments/20251026_120000_gainakt2exp_baseline):
    python examples/repro_train_gainakt2exp.py --experiment-dir 20251026_120000_gainakt2exp_baseline --reproduce

Optionally adjust epochs for the reproduction (new config hash recorded):
    python examples/repro_train_gainakt2exp.py --experiment-dir 20251026_120000_gainakt2exp_baseline --reproduce --set training.epochs=1

Output: A new folder with suffix *_reproduce containing reproduction_report.json summarizing per-seed best metrics,
max absolute differences vs source, tolerance checks, and fully_reproduced flag.

Legacy source configs (missing structured blocks) are auto-adapted using baseline template (configs/gainakt2_exp_config.json).

--------------------------------------------------------------------------------
3) COMPARE two existing experiments WITHOUT retraining (fast diff)
--------------------------------------------------------------------------------
Use compare-only mode with source and target experiment ids OR absolute paths:
    python examples/repro_train_gainakt2exp.py --compare-only \
            --source-exp 20251026_120000_gainakt2exp_baseline \
            --target-exp 20251026_130505_gainakt2exp_variantA

Absolute paths also allowed:
    python examples/repro_train_gainakt2exp.py --compare-only \
            --source-exp /abs/path/to/examples/experiments/20251026_120000_gainakt2exp_baseline \
            --target-exp /abs/path/to/examples/experiments/20251026_130505_gainakt2exp_variantA

Artifacts loaded in priority order: results.json -> results_multi_seed.json -> results_*.json (first) -> synthesized
from metrics_epoch.csv (best epoch inferred by val_auc). If target artifacts are missing a partial report is emitted.

Comparison report (reproduction_report.json in target experiment) fields:
    source_best_val_auc, new_best_val_auc, max_best_val_auc_abs_diff,
    source_mastery_corr, new_mastery_corr, max_mastery_corr_abs_diff,
    source_gain_corr, new_gain_corr, max_gain_corr_abs_diff,
    tolerances, fully_reproduced (boolean), source_results_md5, new_results_md5, results_md5_match.

--------------------------------------------------------------------------------
CONFIG MD5 INTEGRITY
--------------------------------------------------------------------------------
Each config.json stores a stable content hash (config_md5). The script recomputes the hash (excluding the field itself)
and aborts if mismatched unless --force is supplied. All `--set` overrides are applied BEFORE hashing to guarantee that
reproduced runs using the same command and environment produce identical config_md5 signatures.

To manually verify two experiments share the same resolved configuration:
    grep -H "config_md5" examples/experiments/<exp_id>/config.json
Or use compare-only mode which implicitly checks metrics (hash equality is separately logged in stdout.log).

--------------------------------------------------------------------------------
RESULTS MD5 INTEGRITY (NEW)
--------------------------------------------------------------------------------
After each run we compute MD5 signatures for primary metric artifacts:
    - results.json  -> stored in results_md5.txt (key: results_md5)
    - results_multi_seed.json -> stored in results_multi_seed_md5.txt (key: results_multi_seed_md5)

During reproduction (`--reproduce`) the reproduction_report.json now includes:
    source_results_md5, new_results_md5, results_md5_match (boolean).
In compare-only mode, if both experiments expose results.json the comparison report embeds the same MD5 fields.

Notes:
    - MD5 hashing uses raw file bytes; any manual edit to results.json changes the hash and will appear as a mismatch.
    - Synthesized or fallback artifacts (e.g., results_synthesized.json) will not yield a match with original results.
    - A matching config_md5 but differing results_md5 indicates non-determinism or environment drift.

--------------------------------------------------------------------------------
HOW TO USE (Quick Reference)
--------------------------------------------------------------------------------
1. Create a new experiment:
       python examples/repro_train_gainakt2exp.py --short-title baseline
   The folder is auto-created under examples/experiments with timestamped id; inspect config.json for resolved params.

2. Override parameters at creation (dot-path):
       python examples/repro_train_gainakt2exp.py --short-title trial --set training.epochs=3 --set training.learning_rate=0.00025
   All overrides are applied before computing config_md5 to lock the spec.

3. Reproduce an existing experiment (metrics + hashes):
       python examples/repro_train_gainakt2exp.py --experiment-dir <existing_id> --reproduce
   Produces a new *_reproduce folder and writes reproduction_report.json including diff statistics and MD5 matches.

4. Reproduce with modifications (treated as a variant, new hash):
       python examples/repro_train_gainakt2exp.py --experiment-dir <existing_id> --reproduce --set training.epochs=1
   Expect different config_md5; reproduction report still compares source vs variant metrics.

5. Fast compare without retraining:
       python examples/repro_train_gainakt2exp.py --compare-only --source-exp <id_or_path> --target-exp <id_or_path>
   Emits reproduction_report.json in target folder (or uses existing) with per-metric diffs and MD5 status.

6. Resume training from checkpoint:
       python examples/repro_train_gainakt2exp.py --experiment-dir <id> --resume /path/to/model_last.pth --force
   Only the resume path is honored; other overrides ignored unless forcing new hash with --set.

7. Verify integrity:
       grep -H "config_md5" examples/experiments/<id>/config.json
       cat examples/experiments/<id>/results_md5.txt
   Matching config_md5 across runs indicates identical hyperparameter resolution; matching results_md5 indicates stable metric serialization.

8. Extract metrics quickly:
       jq '.per_seed[].best_epoch_metrics.val_auc' examples/experiments/<id>/results.json
       jq '.aggregated.val_auc_mean' examples/experiments/<id>/results_multi_seed.json

9. Add multiple seeds:
       --set training.seeds=21,42,63,84,105
   Multi-seed aggregation appears in results_multi_seed.json and SEED_INFO.md.

10. Troubleshoot mismatches:
    - If config_md5 differs unexpectedly: ensure no manual config.json edits; re-run with identical --set sequence.
    - If results_md5 differs but config_md5 matches: check environment.txt for version drift; confirm deterministic flags.

All core artifacts: config.json, environment.txt, stdout.log, metrics_epoch.csv, results.json, results_multi_seed.json, SEED_INFO.md, reproduction_report.json (if reproduction / compare-only).

--------------------------------------------------------------------------------
Responsibilities (high level):
    1. Create or reuse experiment folder with naming convention: YYYYMMDD_HHMMSS_gainakt2exp_<short_title>.
    2. Serialize complete config (baseline + overrides) and compute config_md5.
    3. Launch multi-seed training (seeds from config) capturing stdout in stdout.log (Tee).
    4. Produce reproducibility artifacts: config.json, environment.txt, stdout.log, metrics_epoch.csv, results.json, results_multi_seed.json, SEED_INFO.md.
    5. Perform reproduction or comparison if requested and write reproduction_report.json.
    6. Append summary sections to README.md (auto-created on first launch).

Notes:
    - Underlying training logic resides in examples/train_gainakt2exp.py.
    - Parameter injection occurs via build_args_from_config to prevent drift between config and runtime behavior.
    - When reusing an existing folder (without reproduction) hyperparameter CLI overrides are ignored (except --resume).

--------------------------------------------------------------------------------
GENERIC PARAMETER OVERRIDES (--set key=val)
--------------------------------------------------------------------------------
We support overriding ANY parameter present (or to be created) inside the structured config using the repeatable
`--set key=val` flag. Keys use dot notation to navigate the JSON hierarchy.

Examples:
    Override epochs:
        python examples/repro_train_gainakt2exp.py --short-title fasttrial --set training.epochs=1

    Change learning rate and monotonicity loss weight simultaneously:
        python examples/repro_train_gainakt2exp.py --short-title lrmono \
                --set training.learning_rate=0.00025 \
                --set constraints.monotonicity_loss_weight=0.2

    Disable gain head (boolean coercion) and raise sparsity weight:
        python examples/repro_train_gainakt2exp.py --short-title nogain \
                --set model.use_gain_head=false \
                --set constraints.sparsity_loss_weight=0.25

    Provide a multi-seed list (comma separated becomes list of ints):
        python examples/repro_train_gainakt2exp.py --short-title multiseed \
                --set training.seeds=21,42,63

    Mixed types (float + bool + list):
        python examples/repro_train_gainakt2exp.py --short-title mixed \
                --set training.learning_rate=0.0003 \
                --set model.use_mastery_head=true \
                --set sampling.max_semantic_students=75

Type Coercion Rules:
    - Existing value type guides coercion (int stays int, float stays float, bool resolved from 'true'/'false').
    - Lists use comma separation; each element is coerced individually (int/float/bool else string).
    - If key did not exist, we attempt numeric parse (int if no dot, float if dot) else bool else raw string.

Precedence & Hash:
    - All --set overrides apply BEFORE config_md5 recomputation; the new hash records the modified spec.
    - Overrides are logged as: key:old->new (old may be None for newly created fields).

Legacy Configs:
    - If an existing experiment has a legacy results-style config lacking structured blocks, we reconstruct blocks using
        `configs/gainakt2_exp_config.json` before applying overrides. This enables uniform override behavior.

Reproduction:
    - Overrides supplied during `--reproduce` create a modified clone (new config_md5) distinct from the source; the
        reproduction report then compares source metrics with overridden reproduction metrics.

Audit Artifact (implemented):
    - overrides_applied.txt now records every override applied (epochs shortcut + generic --set). See POST-RUN NORMALIZATION section.

Edge Cases:
    - Malformed overrides (missing '=') are ignored with a warning.
    - Non-positive epochs via --set training.epochs=0 are accepted as 0; training will likely fail expected behavior,
        so prefer epochs>=1.
    - Removing keys is not yet supported (planned via future `--unset key.path`).

Recommended Practice:
    - Use explicit seeds lists when exploring sensitivity: `--set training.seeds=21,42,63,84,105`.
    - Keep a changelog outside the folder for interpretability experiments; deterministic reproduction requires matching
        the override sequence and final config_md5.

Copyright (c) 2025 Concha Labra. All Rights Reserved.

--------------------------------------------------------------------------------
POST-RUN NORMALIZATION & OVERRIDE AUDIT (ADDED)
--------------------------------------------------------------------------------
New reproducibility artifacts extend the standard pykt suite:

1. overrides_applied.txt
   Purpose: Immutable chronological log of CLI overrides to ensure forensic traceability.
   Block structure (repeats, separated by ---):
       timestamp: <UTC ISO8601>
       experiment_id: <id>
       reproduction_mode: <True|False>
       source_experiment_dir: <path>            # only when --reproduce
       final_config_md5: <hash after overrides>
       cli_invocation: python examples/repro_train_gainakt2exp.py ...
       override_count: N
       override: training.learning_rate:0.000174->0.00025
       override: constraints.monotonicity_loss_weight:0.1->0.2
       ---
   Integrity: Any manual config.json edit without a matching override line signals potential tampering.

2. config_postrun.json
   Purpose: Canonical normalized configuration AFTER training completes.
   Normalization Rules (v1):
       - Removes legacy fields: training_args, model_config
       - Ensures all structured blocks exist; merges missing keys from baseline template
       - Captures effective epochs (fallback to legacy num_epochs if needed)
       - Canonicalizes seeds into training.seeds list
       - Forces training.output_dir to experiment path
       - Appends config_postrun_md5 (hash excluding itself)
   Use Case: Detect implicit default population & legacy adaptation vs original config.json.

3. config_postrun_diff.json / config_postrun_diff.txt
   Purpose: Structural diff between config.json and config_postrun.json.
   JSON fields:
       added_keys: ["constraints.sparsity_loss_weight", ...]
       removed_keys: ["training_args", "model_config"]
       changed_keys: ["training.epochs", "training.learning_rate"]
       original_config_md5_excluding_field, postrun_config_md5_excluding_field
       config_postrun_md5_match (boolean)
   Text summary: One line per category + MD5 comparison.

4. reproduction_report.json (extended)
   Added fields:
       source_config_postrun_md5
       new_config_postrun_md5
       config_postrun_md5_match
   Interpretation:
       - config_md5_match True but config_postrun_md5_match False => normalization introduced structural additions (expected when baseline fills defaults).
       - Both matches True => perfect structural reproduction including normalization.

EXAMPLE WORKFLOWS (Extended):
Create new experiment with epochs & LR overrides:
    python examples/repro_train_gainakt2exp.py --short-title lrtest \
        --set training.epochs=3 \
        --set training.learning_rate=0.00025

Reproduce and inspect diff:
    python examples/repro_train_gainakt2exp.py --experiment-dir 20251026_120000_gainakt2exp_baseline --reproduce
    cat examples/experiments/20251026_120000_gainakt2exp_baseline_reproduce/config_postrun_diff.txt

Variant reproduction altering epochs only:
    python examples/repro_train_gainakt2exp.py --experiment-dir 20251026_120000_gainakt2exp_baseline --reproduce --set training.epochs=1
    grep -H "source_config_postrun_md5" examples/experiments/20251026_120000_gainakt2exp_baseline_reproduce/reproduction_report.json

Inspect override audit (recent block):
    tail -n 25 examples/experiments/20251026_120000_gainakt2exp_baseline/overrides_applied.txt

Manually recompute post-run hash (exclude self-field):
    python - <<'PY'
    import json,hashlib,sys
    cfg=json.load(open(sys.argv[1]))
    h=cfg.pop('config_postrun_md5',None)
    md5=hashlib.md5(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    print('Recomputed:',md5,'Stored:',h)
    PY examples/experiments/20251026_120000_gainakt2exp_baseline_reproduce/config_postrun.json

Sample reproduction_report.json excerpt:
    {
      "source_best_val_auc": [0.7187],
      "new_best_val_auc": [0.7187],
      "results_md5_match": true,
      "source_config_postrun_md5": "0122622543870a3c41ac18cecbd141e5",
      "new_config_postrun_md5": "b004ac813ee330255348869d3514c8ab",
      "config_postrun_md5_match": false
    }

Planned future extensions:
    - Nested per-block diff granularity
    - Normalization rules version tagging
    - Artifact integrity manifest summarizing all hashes

We maintain academic traceability: every structural transformation is hash-verifiable.

Copyright (c) 2025 Concha Labra. All Rights Reserved.
"""
import os
import sys
import json
import argparse
from datetime import datetime
import torch
import numpy as np

sys.path.insert(0, '/workspaces/pykt-toolkit')

from examples.train_gainakt2exp import train_gainakt2exp_model

CONFIG_BASE = '/workspaces/pykt-toolkit/configs/gainakt2_exp_config.json'
EXPERIMENTS_ROOT = '/workspaces/pykt-toolkit/examples/experiments'
MODEL_NAME = 'gainakt2exp'

MANDATORY_FILES = [
    'config.json','environment.txt','stdout.log','metrics_epoch.csv','results.json'
]

# --- Override Audit & Post-Run Normalization Helpers ---
def write_overrides_audit(exp_dir: str, overrides_applied: list, cfg: dict, reproduction_mode: bool, source_dir: str = None):
    """Persist an audit trail of overrides applied for this launch.
    Always writes (even if no overrides) to strengthen reproducibility traceability.
    File: overrides_applied.txt
    Contents:
        timestamp, experiment id, reproduction_mode flag, source_dir (if reproduction), final config_md5,
        original CLI invocation, count of overrides, list (one per line)
    """
    try:
        audit_path = os.path.join(exp_dir, 'overrides_applied.txt')
        with open(audit_path, 'a') as af:
            af.write(f"timestamp: {datetime.utcnow().isoformat()}Z\n")
            af.write(f"experiment_id: {cfg.get('experiment',{}).get('id')}\n")
            af.write(f"reproduction_mode: {reproduction_mode}\n")
            if reproduction_mode and source_dir:
                af.write(f"source_experiment_dir: {source_dir}\n")
            af.write(f"final_config_md5: {cfg.get('config_md5')}\n")
            af.write(f"cli_invocation: {' '.join(sys.argv)}\n")
            if overrides_applied:
                af.write(f"override_count: {len(overrides_applied)}\n")
                for ov in overrides_applied:
                    af.write(f"override: {ov}\n")
            else:
                af.write("override_count: 0\n")
                af.write("override: <none>\n")
            af.write("---\n")
    except Exception as e:
        print(f"[WARN] Failed to write overrides_applied.txt: {e}")

def postrun_normalize_config(cfg: dict, exp_dir: str, seeds: list, used_epochs: int):
    """Create a normalized post-run config snapshot capturing the exact structured spec used.
    Removes legacy fields (training_args, model_config) and ensures seeds list & epochs are explicit.
    Writes config_postrun.json with a config_postrun_md5 hash (excluding itself from the hash computation).
    """
    try:
        import hashlib  # ensure available within function scope
        norm = json.loads(json.dumps(cfg))  # deep copy via serialization to avoid mutation surprises
        # Capture legacy snapshot BEFORE removal
        legacy_ta_snapshot = norm.get('training_args') if isinstance(norm.get('training_args'), dict) else {}
        # Remove legacy summary-style fields if present
        for legacy_key in ['training_args', 'model_config']:
            if legacy_key in norm:
                norm.pop(legacy_key)
        # Ensure training block exists
        if 'training' not in norm or not isinstance(norm.get('training'), dict):
            norm['training'] = {}
        # Fallback epochs from used_epochs or legacy training_args
        legacy_ta = legacy_ta_snapshot  # use snapshot captured before removal
        epochs_final = used_epochs if isinstance(used_epochs, (int,float)) and used_epochs is not None else legacy_ta.get('num_epochs') or legacy_ta.get('epochs') or norm['training'].get('epochs') or 0
        if (not isinstance(epochs_final,(int,float))) or int(epochs_final) == 0:
            # Attempt stronger fallback from legacy training_args
            epochs_final = legacy_ta.get('num_epochs') or legacy_ta.get('epochs') or 1
        norm['training']['epochs'] = int(epochs_final)
        # Propagate common scalar fields from legacy training_args if present and not yet overridden
        propagate_map = {
            'batch_size':'batch_size',
            'learning_rate':'learning_rate',
            'weight_decay':'weight_decay',
            'gradient_clip':'gradient_clip',
            'patience':'patience',
            'seed':'seed',
            'use_wandb':'use_wandb',
            'enhanced_constraints':'enhanced_constraints'
        }
        for lk, tk in propagate_map.items():
            if lk in legacy_ta:
                norm['training'][tk] = legacy_ta[lk]
        # Scheduler fields
        if 'scheduler' in legacy_ta:
            norm['training']['scheduler'] = legacy_ta['scheduler']
        if 'scheduler_params' in legacy_ta:
            norm['training']['scheduler_params'] = legacy_ta['scheduler_params']
        # Seeds canonicalization: prefer provided seeds list else legacy seed
        if not seeds:
            legacy_seed = legacy_ta.get('seed') or norm['training'].get('seed') or 42
            seeds = [legacy_seed]
        norm['training']['seeds'] = seeds
        # Ensure output_dir correct
        norm['training']['output_dir'] = exp_dir
        # Sort blocks presence (non-destructive - ensure required blocks exist using baseline if missing)
        required_blocks = ['data','training','constraints','semantic_alignment','retention','lag_gain','consistency_rebalance','variance_control','sampling','model']
        try:
            with open(CONFIG_BASE,'r') as bf:
                base_cfg = json.load(bf)
            for b in required_blocks:
                if b not in norm or not isinstance(norm.get(b), dict):
                    norm[b] = base_cfg.get(b, {})
                else:
                    # Merge missing keys from baseline for completeness
                    for k,v in base_cfg.get(b, {}).items():
                        if k not in norm[b]:
                            norm[b][k] = v
        except Exception:
            pass  # baseline merge best-effort
        # Compute hash excluding own hash field if re-run
        if 'config_postrun_md5' in norm:
            norm.pop('config_postrun_md5')
        postrun_md5 = hashlib.md5(json.dumps(norm, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
        norm['config_postrun_md5'] = postrun_md5
        out_path_tmp = os.path.join(exp_dir,'config_postrun.json.tmp')
        out_path = os.path.join(exp_dir,'config_postrun.json')
        with open(out_path_tmp,'w') as pf:
            json.dump(norm, pf, indent=2)
        os.replace(out_path_tmp, out_path)
        print(f"[INFO] Post-run normalized config written: {out_path} (md5={postrun_md5})")
    except Exception as e:
        print(f"[WARN] Failed post-run config normalization: {e}")

# --- Comparison Helper (reused for reproduction and compare-only) ---
def perform_reproduction_comparison(source_exp_id: str, new_exp_id: str, experiments_root: str, seeds_list=None, per_seed_new=None, auc_mean_new=None, mastery_mean_new=None, gain_mean_new=None):
    """Compare metrics between source and target experiment directories.
    If per_seed_new or aggregated means are None (compare-only mode), they will be loaded from target results.json.
    Returns a dict with diff metrics and reproduction flags.
    """
    import json
    import os
    # Allow passing absolute paths or relative IDs
    def resolve(p):
        if os.path.isabs(p):
            return p
        return os.path.join(experiments_root, p)
    source_dir = resolve(source_exp_id)
    target_dir = resolve(new_exp_id)
    src_results_path = os.path.join(source_dir,'results.json')
    tgt_results_path = os.path.join(target_dir,'results.json')
    def resolve_results(path_dir, primary):
        if os.path.exists(primary):
            return primary
        # Try multi-seed file
        alt_multi = os.path.join(path_dir,'results_multi_seed.json')
        if os.path.exists(alt_multi):
            return alt_multi
        # Try any results_*.json (pick first deterministic sorted)
        candidates = [f for f in os.listdir(path_dir) if f.startswith('results_') and f.endswith('.json')]
        if candidates:
            candidates.sort()
            return os.path.join(path_dir,candidates[0])
        return None
    src_results_path = resolve_results(source_dir, src_results_path)
    tgt_results_path = resolve_results(target_dir, tgt_results_path)
    if not src_results_path:
        return {'error': f'source results artifacts missing in {source_dir}'}
    if not tgt_results_path:
        # Try parsing metrics_epoch.csv to synthesize minimal results
        def synthesize_from_metrics(dir_path):
            import csv
            metrics_csv = os.path.join(dir_path,'metrics_epoch.csv')
            if not os.path.exists(metrics_csv):
                return None
            epochs = []
            try:
                with open(metrics_csv,'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Expect columns: epoch,val_auc,mastery_correlation,gain_correlation (best_epoch markers not needed)
                        try:
                            epoch = int(row.get('epoch', len(epochs)+1))
                        except Exception:
                            epoch = len(epochs)+1
                        def sf(k):
                            v = row.get(k)
                            try:
                                return float(v) if v not in (None,'','nan') else None
                            except Exception:
                                return None
                        epochs.append({
                            'epoch': epoch,
                            'val_auc': sf('val_auc'),
                            'mastery_correlation': sf('mastery_correlation'),
                            'gain_correlation': sf('gain_correlation')
                        })
                if not epochs:
                    return None
                # Determine best by val_auc
                best = max([e for e in epochs if e['val_auc'] is not None], key=lambda x: x['val_auc']) if any(e['val_auc'] for e in epochs) else epochs[-1]
                return {
                    'per_seed': [{
                        'best_val_auc': best['val_auc'],
                        'best_mastery_corr': best['mastery_correlation'],
                        'best_gain_corr': best['gain_correlation']
                    }]
                }
            except Exception:
                # Failed parsing metrics csv
                return None
        synthesized = synthesize_from_metrics(target_dir)
        if synthesized is None:
            # Graceful partial report: load source metrics and mark target missing
            try:
                import json as _json
                src = _json.load(open(src_results_path))
                def safe_float(x):
                    return x if isinstance(x,(int,float)) else None
                def extract_per_seed(block):
                    out = {'val_auc': [],'mastery_correlation': [],'gain_correlation': []}
                    for r in block.get('per_seed', []):
                        if 'best_epoch_metrics' in r and isinstance(r['best_epoch_metrics'], dict):
                            out['val_auc'].append(safe_float(r['best_epoch_metrics'].get('val_auc')))
                            out['mastery_correlation'].append(safe_float(r['best_epoch_metrics'].get('mastery_correlation')))
                            out['gain_correlation'].append(safe_float(r['best_epoch_metrics'].get('gain_correlation')))
                        else:
                            out['val_auc'].append(safe_float(r.get('best_val_auc') or r.get('val_auc')))
                            mc = safe_float(r.get('best_mastery_corr'))
                            gc = safe_float(r.get('best_gain_corr'))
                            if mc is None and isinstance(r.get('final_consistency_metrics'), dict):
                                mc = safe_float(r['final_consistency_metrics'].get('mastery_correlation'))
                            if gc is None and isinstance(r.get('final_consistency_metrics'), dict):
                                gc = safe_float(r['final_consistency_metrics'].get('gain_correlation'))
                            out['mastery_correlation'].append(mc)
                            out['gain_correlation'].append(gc)
                    return out
                src_metrics = extract_per_seed(src)
                return {
                    'source_experiment': source_dir,
                    'new_experiment': target_dir,
                    'note': 'target missing results artifacts; comparison incomplete',
                    'source_best_val_auc': src_metrics['val_auc'],
                    'new_best_val_auc': [],
                    'max_best_val_auc_abs_diff': None,
                    'source_mastery_corr': src_metrics['mastery_correlation'],
                    'new_mastery_corr': [],
                    'max_mastery_corr_abs_diff': None,
                    'source_gain_corr': src_metrics['gain_correlation'],
                    'new_gain_corr': [],
                    'max_gain_corr_abs_diff': None,
                    'auc_within_tolerance': False,
                    'mastery_within_tolerance': False,
                    'gain_within_tolerance': False,
                    'fully_reproduced': False,
                    'error': f'target results artifacts missing in {target_dir}'
                }
            except Exception:
                return {'error': f'target results artifacts missing in {target_dir}'}
        # Write synthesized temporary file for consistent downstream parsing
        tgt_results_path = os.path.join(target_dir,'results_synthesized.json')
        try:
            with open(tgt_results_path,'w') as sfh:
                json.dump(synthesized, sfh)
        except Exception as e:
            return {'error': f'failed to synthesize target results: {e}'}
    try:
        src = json.load(open(src_results_path))
        tgt = json.load(open(tgt_results_path))
        def safe_float(x):
            return x if isinstance(x,(int,float)) else None
        def extract_per_seed(block):
            out = {
                'val_auc': [],
                'mastery_correlation': [],
                'gain_correlation': []
            }
            for r in block.get('per_seed', []):
                # AUC
                if 'best_epoch_metrics' in r and isinstance(r['best_epoch_metrics'], dict):
                    out['val_auc'].append(safe_float(r['best_epoch_metrics'].get('val_auc')))
                    out['mastery_correlation'].append(safe_float(r['best_epoch_metrics'].get('mastery_correlation')))
                    out['gain_correlation'].append(safe_float(r['best_epoch_metrics'].get('gain_correlation')))
                else:
                    out['val_auc'].append(safe_float(r.get('best_val_auc') or r.get('val_auc')))
                    # Correlations may be best_mastery_corr / best_gain_corr or inside final_consistency_metrics
                    mc = safe_float(r.get('best_mastery_corr'))
                    gc = safe_float(r.get('best_gain_corr'))
                    if mc is None and isinstance(r.get('final_consistency_metrics'), dict):
                        mc = safe_float(r['final_consistency_metrics'].get('mastery_correlation'))
                    if gc is None and isinstance(r.get('final_consistency_metrics'), dict):
                        gc = safe_float(r['final_consistency_metrics'].get('gain_correlation'))
                    out['mastery_correlation'].append(mc)
                    out['gain_correlation'].append(gc)
            return out
        src_metrics = extract_per_seed(src)
        tgt_metrics = extract_per_seed(tgt) if per_seed_new is None else extract_per_seed({'per_seed': per_seed_new})
        def max_abs_diff(a,b):
            diffs = []
            for x,y in zip(a,b):
                if x is None or y is None:
                    continue
                diffs.append(abs(x-y))
            return max(diffs) if diffs else None
        auc_diff = max_abs_diff(src_metrics['val_auc'], tgt_metrics['val_auc'])
        mastery_diff = max_abs_diff(src_metrics['mastery_correlation'], tgt_metrics['mastery_correlation'])
        gain_diff = max_abs_diff(src_metrics['gain_correlation'], tgt_metrics['gain_correlation'])
        tol_auc = 0.002
        tol_corr = 0.01
        # Aggregated means (compute if not provided)
        def mean(vals):
            vals2 = [v for v in vals if isinstance(v,(int,float))]
            return sum(vals2)/len(vals2) if vals2 else None
        if auc_mean_new is None:
            auc_mean_new = mean(tgt_metrics['val_auc'])
        if mastery_mean_new is None:
            mastery_mean_new = mean(tgt_metrics['mastery_correlation'])
        if gain_mean_new is None:
            gain_mean_new = mean(tgt_metrics['gain_correlation'])
        report = {
            'source_experiment': source_dir,
            'new_experiment': target_dir,
            'seeds': seeds_list or [],
            'source_best_val_auc': src_metrics['val_auc'],
            'new_best_val_auc': tgt_metrics['val_auc'],
            'max_best_val_auc_abs_diff': auc_diff,
            'source_mastery_corr': src_metrics['mastery_correlation'],
            'new_mastery_corr': tgt_metrics['mastery_correlation'],
            'max_mastery_corr_abs_diff': mastery_diff,
            'source_gain_corr': src_metrics['gain_correlation'],
            'new_gain_corr': tgt_metrics['gain_correlation'],
            'max_gain_corr_abs_diff': gain_diff,
            'auc_within_tolerance': auc_diff is not None and auc_diff <= tol_auc,
            'mastery_within_tolerance': mastery_diff is not None and mastery_diff <= tol_corr,
            'gain_within_tolerance': gain_diff is not None and gain_diff <= tol_corr,
            'tolerances': {'auc': tol_auc, 'corr': tol_corr},
            'aggregated_new': {
                'val_auc_mean': auc_mean_new,
                'mastery_correlation_mean': mastery_mean_new,
                'gain_correlation_mean': gain_mean_new
            },
            'fully_reproduced': all([
                auc_diff is not None and auc_diff <= tol_auc,
                mastery_diff is not None and mastery_diff <= tol_corr,
                gain_diff is not None and gain_diff <= tol_corr
            ])
        }
        return report
    except Exception as e:
        return {'error': f'comparison_failed: {e}'}


def timestamp_id(short_title: str) -> str:
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return f"{ts}_{MODEL_NAME}_{short_title}"


def load_or_initialize_config(exp_dir: str, short_title: str) -> dict:
    cfg_path = os.path.join(exp_dir, 'config.json')
    if os.path.isdir(exp_dir) and os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return json.load(f)
    # Need to initialize
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    # Copy baseline template
    with open(CONFIG_BASE, 'r') as bf:
        base_cfg = json.load(bf)
    # Inject experiment id & suffix/title if missing
    exp_id = os.path.basename(exp_dir)
    base_cfg['experiment']['id'] = exp_id
    if 'title' in base_cfg['experiment'] and (base_cfg['experiment']['title'] is None or base_cfg['experiment']['title'] == 'baseline'):
        base_cfg['experiment']['title'] = short_title
    base_cfg['experiment']['suffix'] = short_title
    # Ensure output_dir matches exp_dir
    base_cfg['training']['output_dir'] = exp_dir
    # Add config_md5
    import hashlib
    md5 = hashlib.md5(json.dumps(base_cfg, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
    base_cfg['config_md5'] = md5
    with open(cfg_path,'w') as wf:
        json.dump(base_cfg, wf, indent=2)
    return base_cfg


def write_environment(exp_dir: str):
    env_path = os.path.join(exp_dir,'environment.txt')
    try:
        import platform
        import subprocess
        py_ver = platform.python_version()
        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda if torch.cuda.is_available() else 'None'
        git_commit = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
        git_branch = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD']).decode().strip()
        with open(env_path,'w') as f:
            f.write(f"timestamp: {datetime.utcnow().isoformat()}Z\n")
            f.write(f"python_version: {py_ver}\n")
            f.write(f"torch_version: {torch_ver}\n")
            f.write(f"cuda_version: {cuda_ver}\n")
            f.write(f"git_branch: {git_branch}\n")
            f.write(f"git_commit: {git_commit}\n")
    except Exception as e:
        print(f"[WARN] Failed to write environment.txt: {e}")


def build_args_from_config(cfg: dict, resume_path: str = None):
    """Construct an argparse-like namespace consumed by train_gainakt2exp_model.
    We map config schema fields to expected argument names.
    """
    class Dummy:  # simple namespace
        pass
    args = Dummy()
    # Core
    args.dataset = cfg['data']['dataset']
    args.fold = cfg['data'].get('fold',0)
    args.epochs = cfg['training']['epochs']
    args.batch_size = cfg['training']['batch_size']
    args.learning_rate = cfg['training']['learning_rate']
    args.weight_decay = cfg['training']['weight_decay']
    args.gradient_clip = cfg['training'].get('gradient_clip',1.0)
    args.scheduler = cfg['training'].get('scheduler','none')
    sch_params = cfg['training'].get('scheduler_params',{})
    args.scheduler_mode = sch_params.get('mode','max')
    args.scheduler_factor = sch_params.get('factor',0.5)
    args.scheduler_patience = sch_params.get('patience',5)
    args.patience = cfg['training'].get('patience',20)
    args.use_amp = bool(cfg['training'].get('use_amp', False))
    args.enhanced_constraints = bool(cfg['training'].get('enhanced_constraints', True))
    args.seed = cfg['training'].get('seed',42)
    args.use_wandb = bool(cfg['training'].get('use_wandb', False))
    args.output_dir = cfg['training'].get('output_dir')
    # Heads
    args.use_mastery_head = cfg['model'].get('use_mastery_head', True)
    args.use_gain_head = cfg['model'].get('use_gain_head', True)
    # Constraint weights
    c = cfg['constraints']
    args.non_negative_loss_weight = c.get('non_negative_loss_weight',0.0)
    args.monotonicity_loss_weight = c.get('monotonicity_loss_weight',0.1)
    args.mastery_performance_loss_weight = c.get('mastery_performance_loss_weight',0.8)
    args.gain_performance_loss_weight = c.get('gain_performance_loss_weight',0.8)
    args.sparsity_loss_weight = c.get('sparsity_loss_weight',0.2)
    args.consistency_loss_weight = c.get('consistency_loss_weight',0.3)
    args.warmup_constraint_epochs = c.get('warmup_constraint_epochs',8)
    # Alignment
    a = cfg['semantic_alignment']
    args.enable_alignment_loss = bool(a.get('enable_alignment_loss', False))
    args.alignment_weight = a.get('alignment_weight',0.25)
    args.alignment_warmup_epochs = a.get('alignment_warmup_epochs',8)
    args.adaptive_alignment = bool(a.get('adaptive_alignment', True))
    args.alignment_min_correlation = a.get('alignment_min_correlation',0.05)
    args.enable_global_alignment_pass = bool(a.get('enable_global_alignment_pass', False))
    args.alignment_global_students = a.get('alignment_global_students',600)
    args.use_residual_alignment = bool(a.get('use_residual_alignment', False))
    args.alignment_residual_window = a.get('alignment_residual_window',5)
    args.alignment_share_cap = a.get('alignment_share_cap',0.08)
    args.alignment_share_decay_factor = a.get('alignment_share_decay_factor',0.7)
    # Retention
    r = cfg['retention']
    args.enable_retention_loss = bool(r.get('enable_retention_loss', False))
    args.retention_delta = r.get('retention_delta',0.005)
    args.retention_weight = r.get('retention_weight',0.14)
    # Lag gain
    lg = cfg['lag_gain']
    args.enable_lag_gain_loss = bool(lg.get('enable_lag_gain_loss', False))
    args.lag_gain_weight = lg.get('lag_gain_weight',0.06)
    args.lag_max_lag = lg.get('lag_max_lag',3)
    args.lag_l1_weight = lg.get('lag_l1_weight',0.5)
    args.lag_l2_weight = lg.get('lag_l2_weight',0.3)
    args.lag_l3_weight = lg.get('lag_l3_weight',0.2)
    # Consistency rebalance & variance
    cr = cfg['consistency_rebalance']
    args.enable_cosine_perf_schedule = bool(cr.get('enable_cosine_perf_schedule', False))
    args.consistency_rebalance_epoch = cr.get('consistency_rebalance_epoch',8)
    args.consistency_rebalance_threshold = cr.get('consistency_rebalance_threshold',0.10)
    args.consistency_rebalance_new_weight = cr.get('consistency_rebalance_new_weight',0.2)
    vc = cfg['variance_control']
    args.variance_floor = vc.get('variance_floor',1e-4)
    args.variance_floor_patience = vc.get('variance_floor_patience',3)
    args.variance_floor_reduce_factor = vc.get('variance_floor_reduce_factor',0.5)
    # Sampling
    s = cfg['sampling']
    args.max_semantic_students = s.get('max_semantic_students',50)
    args.final_consistency_max_students = s.get('final_consistency_max_students',200)
    # Flags
    f = cfg.get('flags',{})
    args.freeze_sparsity = bool(f.get('freeze_sparsity', False))
    # Artifacts
    art = cfg.get('artifacts',{})
    args.semantic_trajectory_path = art.get('semantic_trajectory_path', None)
    # Resume path override
    args.resume = resume_path
    # Monitor frequency for InterpretabilityMonitor
    args.monitor_freq = cfg['model'].get('monitor_frequency',50)
    return args


def write_readme_min(exp_dir: str, cfg: dict):
    readme_path = os.path.join(exp_dir,'README.md')
    if os.path.exists(readme_path):
        return
    lines = [
        f"# Experiment {cfg['experiment'].get('id')} (Repro Mode)",
        '',
        'This folder was generated in reproducible mode. All hyperparameters originate from config.json.',
        'Re-run criteria: identical config_md5 and command reconstruction leads to matching best AUC (within stochastic tolerance for seeds).',
        '',
        'Artifacts produced by underlying training script will supplement this README automatically if extended script used.',
        '',
        '## Key Hyperparameters',
        '```json',
        json.dumps({
            'training': cfg['training'],
            'constraints': cfg['constraints'],
            'alignment': cfg['semantic_alignment'],
            'retention': cfg['retention'],
            'lag_gain': cfg['lag_gain']
        }, indent=2),
        '```',
        '',
        '## Reproducibility Checklist (Partial)',
        '| Item | Status |',
        '|------|--------|',
        '| config.json present | ✅ |',
        '| environment.txt present | ✅ |',
        '| stdout.log capturing run | ✅ |',
        '| metrics_epoch.csv (after run) | ⏳ |',
        '| results.json (after run) | ⏳ |',
        '| config_md5 recorded | ✅ |'
    ]
    with open(readme_path,'w') as rf:
        rf.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Reproducible launcher for GainAKT2Exp experiment.')
    parser.add_argument('--experiment-dir', type=str, default=None,
                        help='Explicit experiment directory name under examples/experiments. If exists with config.json, parameters are locked.')
    parser.add_argument('--short-title', type=str, default='baseline', help='Short title used if creating new experiment.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming (only override allowed).')
    parser.add_argument('--force', action='store_true', help='Force reuse of existing experiment directory even if populated (protects against accidental overwrite).')
    parser.add_argument('--reproduce', action='store_true', help='Clone an existing experiment (given by --experiment-dir) into a new folder and verify metric reproduction.')
    parser.add_argument('--compare-only', action='store_true', help='Compare existing source and target experiments without launching training.')
    parser.add_argument('--source-exp', type=str, default=None, help='Source experiment id for comparison (legacy or structured).')
    parser.add_argument('--target-exp', type=str, default=None, help='Target experiment id for comparison (already trained).')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs stored in config before launch.')
    parser.add_argument('--set', action='append', default=[], help='Generic override flag key=val (dot notation allowed, e.g. training.learning_rate=0.0002). Repeat for multiple overrides.')
    args = parser.parse_args()

    global os  # ensure module reference recognized
    if not os.path.isdir(EXPERIMENTS_ROOT):
        os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

    # Fast comparison-only mode (no training)
    if args.compare_only:
        global perform_reproduction_comparison
        if not args.source_exp or not args.target_exp:
            print('[ERROR] --compare-only requires --source-exp and --target-exp.')
            return
        report = perform_reproduction_comparison(args.source_exp, args.target_exp, EXPERIMENTS_ROOT)
        # When absolute paths given, target_dir is resolved inside helper; adapt target_dir for artifact writes
        import os
        target_dir = args.target_exp if os.path.isabs(args.target_exp) else os.path.join(EXPERIMENTS_ROOT, args.target_exp)
        # Write report
        try:
            def normalize(obj):
                if isinstance(obj, dict):
                    return {k: normalize(v) for k,v in obj.items()}
                if isinstance(obj, list):
                    return [normalize(v) for v in obj]
                import numpy as np
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return obj
            report = normalize(report)
            tmp_path = os.path.join(target_dir,'reproduction_report.json.tmp')
            final_path = os.path.join(target_dir,'reproduction_report.json')
            with open(tmp_path,'w') as fh:
                json.dump(report, fh, indent=2)
            os.replace(tmp_path, final_path)
            print(f"[INFO] Comparison report written to {final_path}")
        except Exception as e:
            print(f"[WARN] Failed to write comparison report: {e}")
        # Append README if exists
        readme_path = os.path.join(target_dir,'README.md')
        try:
            if os.path.exists(readme_path):
                with open(readme_path,'a') as rf:
                    rf.write('\n\n## Compare-Only Results\n')
                    if 'error' in report:
                        rf.write(f"Error: {report['error']}\n")
                    else:
                        rf.write(f"Fully Reproduced: {report.get('fully_reproduced')}\n\n")
                        rf.write('| Metric | Max Abs Diff | Within Tolerance |\n')
                        rf.write('|--------|--------------|------------------|\n')
                        def fmt(v):
                            return f"{v:.4f}" if isinstance(v,(int,float)) and v is not None else '-'
                        rf.write(f"| Val AUC | {fmt(report.get('max_best_val_auc_abs_diff'))} | {report.get('auc_within_tolerance')} |\n")
                        rf.write(f"| Mastery Corr | {fmt(report.get('max_mastery_corr_abs_diff'))} | {report.get('mastery_within_tolerance')} |\n")
                        rf.write(f"| Gain Corr | {fmt(report.get('max_gain_corr_abs_diff'))} | {report.get('gain_within_tolerance')} |\n")
        except Exception as e:
            print(f"[WARN] Failed to append compare-only results to README: {e}")
        return

    if args.reproduce:
        if not args.experiment_dir:
            print('[ERROR] --reproduce requires --experiment-dir pointing to the SOURCE experiment to clone.')
            return
        source_dir = os.path.join(EXPERIMENTS_ROOT, args.experiment_dir)
        if not os.path.isdir(source_dir):
            print(f"[ERROR] Source experiment directory '{source_dir}' not found.")
            return
        # Load source config
        source_cfg_path = os.path.join(source_dir,'config.json')
        if not os.path.exists(source_cfg_path):
            print(f"[ERROR] Source config.json missing in {source_dir}")
            return
        with open(source_cfg_path,'r') as sf:
            source_cfg = json.load(sf)
        # Derive suffix robustly (handle legacy config without 'experiment' block)
        suffix = 'reproduce'
        if isinstance(source_cfg, dict) and 'experiment' in source_cfg and isinstance(source_cfg['experiment'], dict):
            suffix = source_cfg['experiment'].get('suffix') or source_cfg['experiment'].get('title') or suffix
        else:
            # Legacy config adaptation: parse directory name pattern YYYYMMDD_HHMMSS_gainakt2exp_<suffix>
            parts = args.experiment_dir.split('_')
            if len(parts) >= 4 and parts[2] == MODEL_NAME:
                suffix = '_'.join(parts[3:])
            elif len(parts) >= 3:
                suffix = parts[-1]
            print(f"[WARN] Legacy source config detected (missing 'experiment' block). Using derived suffix='{suffix}'. Baseline hyperparameters will be used for reproduction unless present.")
            # If legacy summary-style config (results.json style), fabricate minimal training block from training_args
            if 'training' not in source_cfg and 'training_args' in source_cfg and isinstance(source_cfg['training_args'], dict):
                ta = source_cfg['training_args']
                source_cfg['training'] = {
                    'epochs': ta.get('num_epochs', ta.get('epochs', 12)),
                    'batch_size': ta.get('batch_size', 64),
                    'learning_rate': ta.get('learning_rate', 0.000174),
                    'weight_decay': ta.get('weight_decay', 0.0),
                    'seed': ta.get('seed', 42),
                    'output_dir': source_dir
                }
                print(f"[INFO] Synthesized structured training block from legacy training_args (epochs={source_cfg['training']['epochs']}).")
        # Prepare new target directory
        new_exp_id = timestamp_id(suffix) + '_reproduce'
        exp_dir = os.path.join(EXPERIMENTS_ROOT, new_exp_id)
        os.makedirs(exp_dir, exist_ok=False)
        # Remove existing id & md5 so we can regenerate
        tmp_cfg = dict(source_cfg) if isinstance(source_cfg, dict) else {}
        # Ensure required structured sections exist; if not, pull from baseline template
        try:
            if 'experiment' not in tmp_cfg or not isinstance(tmp_cfg.get('experiment'), dict):
                tmp_cfg['experiment'] = {'id': None, 'suffix': suffix, 'title': suffix}
            if 'training' not in tmp_cfg or not isinstance(tmp_cfg.get('training'), dict):
                with open(CONFIG_BASE,'r') as bf:
                    baseline_cfg = json.load(bf)
                # Copy structured fields from baseline
                for key in ['data','training','constraints','semantic_alignment','retention','lag_gain','consistency_rebalance','variance_control','sampling','model']:
                    if key in baseline_cfg and key not in tmp_cfg:
                        tmp_cfg[key] = baseline_cfg[key]
            # If 'data' section exists but missing dataset key, fill from baseline
            if 'data' in tmp_cfg and 'dataset' not in tmp_cfg['data']:
                with open(CONFIG_BASE,'r') as bf:
                    baseline_cfg = json.load(bf)
                if 'data' in baseline_cfg and 'dataset' in baseline_cfg['data']:
                    tmp_cfg['data']['dataset'] = baseline_cfg['data']['dataset']
        except Exception as e:
            print(f"[WARN] Failed legacy config adaptation: {e}")
        if 'config_md5' in tmp_cfg:
            tmp_cfg.pop('config_md5')
        tmp_cfg['experiment']['id'] = new_exp_id
        tmp_cfg['training']['output_dir'] = exp_dir
        import hashlib
        new_md5 = hashlib.md5(json.dumps(tmp_cfg, sort_keys=True, separators=(',',':')).encode()).hexdigest()
        tmp_cfg['config_md5'] = new_md5
        with open(os.path.join(exp_dir,'config.json'),'w') as nf:
            json.dump(tmp_cfg, nf, indent=2)
        print(f"[INFO] Reproduction target directory created: {exp_dir}")
        cfg = tmp_cfg
        new_creation = True
        reproduction_mode = True
    else:
        reproduction_mode = False
        if args.experiment_dir:
            exp_dir = os.path.join(EXPERIMENTS_ROOT, args.experiment_dir)
            exists = os.path.exists(exp_dir)
            if exists and not args.force and os.listdir(exp_dir):
                print(f"[ERROR] Experiment directory '{exp_dir}' already exists and is populated. Use --force to reuse. Aborting to prevent overwrite.")
                return
            new_creation = not exists
        else:
            exp_id = timestamp_id(args.short_title)
            exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_id)
            new_creation = True
        cfg = load_or_initialize_config(exp_dir, args.short_title)

    # --- Legacy adaptation for existing experiment with non-structured config ---
    def adapt_legacy_config(cfg_obj):
        try:
            # Determine if structured keys missing
            required_blocks = ['data','training','constraints','semantic_alignment','retention','lag_gain','consistency_rebalance','variance_control','sampling','model']
            missing_blocks = [b for b in required_blocks if b not in cfg_obj or not isinstance(cfg_obj.get(b), dict)]
            if missing_blocks:
                with open(CONFIG_BASE,'r') as bf:
                    baseline_cfg = json.load(bf)
                for b in required_blocks:
                    base_block = baseline_cfg.get(b)
                    if base_block is None:
                        continue
                    if b not in cfg_obj or not isinstance(cfg_obj.get(b), dict):
                        cfg_obj[b] = base_block
                    else:
                        # Merge missing keys non-destructively
                        for k,v in base_block.items():
                            if k not in cfg_obj[b]:
                                cfg_obj[b][k] = v
            # Ensure experiment block exists
            if 'experiment' not in cfg_obj or not isinstance(cfg_obj.get('experiment'), dict):
                cfg_obj['experiment'] = {
                    'id': os.path.basename(exp_dir),
                    'suffix': cfg_obj.get('experiment_name','recovered'),
                    'title': cfg_obj.get('experiment_name','recovered'),
                    'purpose': None,
                    'split_strategy': 'standard'
                }
            # Map potential legacy fields into structured blocks if present
            if 'model_config' in cfg_obj and isinstance(cfg_obj['model_config'], dict):
                legacy_mc = cfg_obj['model_config']
                model_block = cfg_obj.get('model', {})
                for k in ['monitor_frequency','use_mastery_head','use_gain_head']:
                    if k in legacy_mc and k not in model_block:
                        model_block[k] = legacy_mc[k]
                cfg_obj['model'] = model_block
            if 'training_args' in cfg_obj and isinstance(cfg_obj['training_args'], dict):
                ta = cfg_obj['training_args']
                train_block = cfg_obj.get('training', {})
                mapping = {
                    'num_epochs': 'epochs',
                    'batch_size': 'batch_size',
                    'learning_rate': 'learning_rate',
                    'weight_decay': 'weight_decay',
                    'seed': 'seed',
                    'output_dir': 'output_dir'
                }
                for lk, tk in mapping.items():
                    if lk in ta:
                        # Always overwrite epochs to preserve original legacy value for reproducibility
                        if lk == 'num_epochs':
                            prev = train_block.get(tk, None)
                            train_block[tk] = ta[lk]
                            if prev is not None and prev != ta[lk]:
                                print(f"[INFO] Legacy num_epochs={ta[lk]} overwrote existing training.{tk}={prev} for reproducibility")
                        else:
                            # Only set other fields if missing to avoid clobbering intentional structured values
                            if tk not in train_block:
                                train_block[tk] = ta[lk]
                # scheduler info
                if 'scheduler' in ta and 'scheduler' not in train_block:
                    train_block['scheduler'] = ta['scheduler']
                if 'scheduler_params' in ta and 'scheduler_params' not in train_block:
                    train_block['scheduler_params'] = ta['scheduler_params']
                cfg_obj['training'] = train_block
            return cfg_obj
        except Exception as e:
            print(f"[WARN] Legacy adaptation failed: {e}")
            return cfg_obj

    # Adapt legacy config if needed (only when reusing an existing experiment or legacy reproduction source)
    cfg = adapt_legacy_config(cfg)
    # Persist any legacy adaptation changes (even if no overrides later). Recompute config_md5 for adapted structure.
    try:
        cfg_path_adapt = os.path.join(exp_dir,'config.json')
        if os.path.exists(cfg_path_adapt):
            with open(cfg_path_adapt,'r') as _f:
                before_disk = json.load(_f)
            # Compare excluding config_md5
            before_cmp = dict(before_disk)
            if 'config_md5' in before_cmp:
                before_cmp.pop('config_md5')
            after_cmp = dict(cfg)
            if 'config_md5' in after_cmp:
                after_cmp.pop('config_md5')
            if before_cmp != after_cmp:
                import hashlib
                tmp_cfg = dict(cfg)
                if 'config_md5' in tmp_cfg:
                    tmp_cfg.pop('config_md5')
                new_md5_adapt = hashlib.md5(json.dumps(tmp_cfg, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
                cfg['config_md5'] = new_md5_adapt
                with open(cfg_path_adapt,'w') as cf_adapt:
                    json.dump(cfg, cf_adapt, indent=2)
                print(f"[INFO] Legacy adaptation persisted. New config_md5={new_md5_adapt}")
    except Exception as e:
        print(f"[WARN] Failed to persist legacy adaptation changes: {e}")

    # === Apply CLI overrides (epochs and --set) BEFORE md5 validation so updated config is hashed ===
    overrides_applied = []
    if args.epochs is not None:
        try:
            if args.epochs <= 0:
                print(f"[WARN] Ignoring non-positive epochs override: {args.epochs}")
            else:
                if 'training' not in cfg:
                    cfg['training'] = {}
                prev_epochs = cfg['training'].get('epochs')
                cfg['training']['epochs'] = int(args.epochs)
                overrides_applied.append(f"epochs:{prev_epochs}->{args.epochs}")
        except Exception as e:
            print(f"[WARN] Failed to apply epochs override: {e}")
    # Generic key=val overrides
    def coerce_value(existing, raw):
        if existing is None:
            if raw.lower() in ('true','false'):
                return raw.lower() == 'true'
            try:
                if '.' in raw:
                    return float(raw)
                return int(raw)
            except Exception:
                return raw
        if isinstance(existing, bool):
            return raw.lower() == 'true' if raw.lower() in ('true','false') else existing
        if isinstance(existing, int):
            try:
                return int(raw)
            except Exception:
                return existing
        if isinstance(existing, float):
            try:
                return float(raw)
            except Exception:
                return existing
        if isinstance(existing, (list, tuple)):
            parts = [p.strip() for p in raw.split(',') if p.strip()]
            out = []
            for p in parts:
                try:
                    if '.' in p:
                        out.append(float(p))
                    else:
                        out.append(int(p))
                except Exception:
                    if p.lower() in ('true','false'):
                        out.append(p.lower() == 'true')
                    else:
                        out.append(p)
            return out
        if raw.lower() in ('true','false'):
            return raw.lower() == 'true'
        try:
            if '.' in raw:
                return float(raw)
            return int(raw)
        except Exception:
            return raw
    for ov in getattr(args,'set',[]):
        if '=' not in ov:
            print(f"[WARN] Ignoring malformed override (missing '='): {ov}")
            continue
        kpath, rval = ov.split('=',1)
        kpath = kpath.strip()
        rval = rval.strip()
        if not kpath:
            print(f"[WARN] Empty key in override: {ov}")
            continue
        segs = kpath.split('.')
        ref = cfg
        for seg in segs[:-1]:
            if seg not in ref or not isinstance(ref[seg], dict):
                ref[seg] = {}
            ref = ref[seg]
        leaf = segs[-1]
        prev = ref.get(leaf, None)
        newv = coerce_value(prev, rval)
        ref[leaf] = newv
        overrides_applied.append(f"{kpath}:{prev}->{newv}")
    # If we applied overrides, refresh config_md5 and persist
    if overrides_applied:
        try:
            import hashlib
            tmp_cfg = dict(cfg)
            # remove old md5 before recompute
            if 'config_md5' in tmp_cfg:
                tmp_cfg.pop('config_md5')
            new_md5 = hashlib.md5(json.dumps(tmp_cfg, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
            cfg['config_md5'] = new_md5
            with open(os.path.join(exp_dir,'config.json'),'w') as cf:
                json.dump(cfg, cf, indent=2)
            print(f"[INFO] Applied overrides: {', '.join(overrides_applied)} | new config_md5={new_md5}")
        except Exception as e:
            print(f"[WARN] Failed to persist overrides to config.json: {e}")

    # Lock parameters if config existed (normal mode)
    if not reproduction_mode and (not new_creation) and os.path.exists(os.path.join(exp_dir,'config.json')):
        if overrides_applied:
            print("[INFO] Existing config.json reused WITH overrides applied.")
        else:
            print("[INFO] Reusing existing config.json (parameters locked). CLI overrides ignored except --resume.")

    # Validate config_md5 integrity (recompute excluding field itself)
    cfg_path = os.path.join(exp_dir,'config.json')
    try:
        import hashlib
        with open(cfg_path,'r') as cf:
            raw_cfg = json.load(cf)
        recorded_md5 = raw_cfg.get('config_md5')
        # create a copy without config_md5
        if 'config_md5' in raw_cfg:
            tmp_cfg = dict(raw_cfg)
            tmp_cfg.pop('config_md5')
        else:
            tmp_cfg = raw_cfg
        recomputed_md5 = hashlib.md5(json.dumps(tmp_cfg, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
        if recorded_md5 and recorded_md5 != recomputed_md5:
            print(f"[ERROR] config_md5 mismatch. recorded={recorded_md5} recomputed={recomputed_md5}. Folder: {exp_dir}")
            print("        Possible manual edit of config.json detected. Aborting to preserve reproducibility. Use --force to override.")
            if not args.force:
                return
            else:
                print("[WARN] --force supplied; proceeding despite hash mismatch.")
        else:
            print(f"[INFO] config_md5 validation passed (md5={recorded_md5 or recomputed_md5}).")
            # Ensure in-memory cfg matches exactly what is on disk post-validation.
            # This guards against any accidental divergence between the object we modified and the persisted file.
            cfg = raw_cfg
    except Exception as e:
        print(f"[WARN] Unable to validate config_md5: {e}")

    # Environment & README
    write_environment(exp_dir)
    write_readme_min(exp_dir, cfg)
    # Override audit artifact (always written for traceability)
    try:
        write_overrides_audit(exp_dir, overrides_applied, cfg, reproduction_mode, source_dir=source_dir if 'source_dir' in globals() or 'source_dir' in locals() else None)
    except Exception as e:
        print(f"[WARN] Override audit artifact failed: {e}")

    # Logging redirection (basic): capture stdout to file while still echoing
    stdout_path = os.path.join(exp_dir,'stdout.log')
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    orig_stdout = sys.stdout
    log_fh = open(stdout_path, 'a')
    sys.stdout = Tee(orig_stdout, log_fh)

    # Seed setup from config (single seed retained; multi-seed would be separate wrapper)
    seeds = cfg['training'].get('seeds') or [cfg['training'].get('seed',42)]
    if not isinstance(seeds, list):
        seeds = [seeds]
    print(f"[INFO] Multi-seed run commencing: seeds={seeds}")

    per_seed_results = []

    resume_path = args.resume  # single resume path applied to first seed only
    for idx, seed in enumerate(seeds):
        print(f"[INFO] === Seed {seed} ({idx+1}/{len(seeds)}) ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Build args fresh each seed to update seed-specific field
        cfg['training']['seed'] = seed
        built_args = build_args_from_config(cfg, resume_path=resume_path if idx == 0 else None)
        # Distinct semantic trajectory path per seed (if path template provided)
        if built_args.semantic_trajectory_path:
            root, ext = os.path.splitext(built_args.semantic_trajectory_path)
            built_args.semantic_trajectory_path = f"{root}_seed{seed}{ext}" if ext else f"{root}_seed{seed}.json"
        print(f"[INFO] Output dir: {built_args.output_dir} | seed={seed}")
        seed_result = train_gainakt2exp_model(built_args)
        seed_result['seed'] = seed
        per_seed_results.append(seed_result)
    # Persist per-seed and aggregated results
    results_path = os.path.join(exp_dir,'results.json')
    try:
        with open(results_path,'w') as rf:
            json.dump({'per_seed': per_seed_results}, rf, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write results.json: {e}")
    # Compute MD5 for results.json if written
    results_md5 = None
    if os.path.exists(results_path):
        try:
            import hashlib
            with open(results_path,'rb') as rfh:
                results_md5 = hashlib.md5(rfh.read()).hexdigest()
            with open(os.path.join(exp_dir,'results_md5.txt'),'w') as mh:
                mh.write(f"results_md5={results_md5}\n")
        except Exception as e:
            print(f"[WARN] Unable to hash results.json: {e}")

    # (Deferred) reproduction comparison will run after aggregation if args.reproduce

    # Aggregation (val_auc, mastery_correlation, gain_correlation)
    def extract(metric):
        vals = []
        for r in per_seed_results:
            # use best_epoch metrics if available
            if 'best_epoch_metrics' in r:
                be = r['best_epoch_metrics']
                if metric in be:
                    vals.append(be[metric])
                    continue
            if metric in r:
                vals.append(r[metric])
        return vals
    import math
    def mean_std(v):
        if not v:
            return None, None
        m = sum(v)/len(v)
        var = sum((x-m)**2 for x in v)/len(v)
        return m, math.sqrt(var)
    auc_vals = extract('val_auc')
    mastery_vals = extract('mastery_correlation')
    gain_vals = extract('gain_correlation')
    auc_mean, auc_std = mean_std(auc_vals)
    mastery_mean, mastery_std = mean_std(mastery_vals)
    gain_mean, gain_std = mean_std(gain_vals)
    # Fisher z CI for correlations
    try:
        from pykt.metrics_utils import fisher_z_ci
        mastery_ci = fisher_z_ci(mastery_mean, len(mastery_vals)) if mastery_mean is not None else (None,None)
        gain_ci = fisher_z_ci(gain_mean, len(gain_vals)) if gain_mean is not None else (None,None)
    except Exception as e:
        print(f"[WARN] CI computation failed: {e}")
        mastery_ci = (None,None)
        gain_ci = (None,None)
    aggregated = {
        'seeds': seeds,
        'val_auc_mean': auc_mean,
        'val_auc_std': auc_std,
        'mastery_correlation_mean': mastery_mean,
        'mastery_correlation_std': mastery_std,
        'mastery_correlation_ci_low': mastery_ci[0],
        'mastery_correlation_ci_high': mastery_ci[1],
        'gain_correlation_mean': gain_mean,
        'gain_correlation_std': gain_std,
        'gain_correlation_ci_low': gain_ci[0],
        'gain_correlation_ci_high': gain_ci[1]
    }
    multi_path = os.path.join(exp_dir,'results_multi_seed.json')
    try:
        with open(multi_path,'w') as mf:
            json.dump({'aggregated': aggregated, 'per_seed': per_seed_results}, mf, indent=2)
    except Exception as e:
        print(f"[WARN] Unable to write results_multi_seed.json: {e}")
    multi_md5 = None
    if os.path.exists(multi_path):
        try:
            import hashlib
            with open(multi_path,'rb') as mmfh:
                multi_md5 = hashlib.md5(mmfh.read()).hexdigest()
            with open(os.path.join(exp_dir,'results_multi_seed_md5.txt'),'w') as mh2:
                mh2.write(f"results_multi_seed_md5={multi_md5}\n")
        except Exception as e:
            print(f"[WARN] Unable to hash results_multi_seed.json: {e}")

    # SEED_INFO.md
    seed_info_path = os.path.join(exp_dir,'SEED_INFO.md')
    try:
        lines = ["# Seed Information","","| Seed | Val AUC | Mastery Corr | Gain Corr |","|------|--------|--------------|-----------|"]
        for r in per_seed_results:
            be = r.get('best_epoch_metrics',{})
            auc = be.get('val_auc', r.get('val_auc'))
            mc = be.get('mastery_correlation', r.get('mastery_correlation'))
            gc = be.get('gain_correlation', r.get('gain_correlation'))
            def fmt(v):
                return f"{v:.4f}" if isinstance(v,(int,float)) and v is not None else '-'
            lines.append(f"| {r.get('seed')} | {fmt(auc)} | {fmt(mc)} | {fmt(gc)} |")
        lines.append("")
        lines.append("## Aggregated")
        def fmt_ci(v):
            return f"{v:.4f}" if isinstance(v,(int,float)) and v is not None else '-'
        lines.append(f"Mean Val AUC: {fmt(auc_mean)} (std={fmt(auc_std)})")
        lines.append(f"Mastery Corr Mean: {fmt(mastery_mean)} (std={fmt(mastery_std)}) CI=({fmt_ci(mastery_ci[0])},{fmt_ci(mastery_ci[1])})")
        lines.append(f"Gain Corr Mean: {fmt(gain_mean)} (std={fmt(gain_std)}) CI=({fmt_ci(gain_ci[0])},{fmt_ci(gain_ci[1])})")
        with open(seed_info_path,'w') as sf:
            sf.write('\n'.join(lines))
    except Exception as e:
        print(f"[WARN] Unable to write SEED_INFO.md: {e}")

    # Update config hash file (post-run)
    try:
        import hashlib
        with open(os.path.join(exp_dir,'config.json'),'r') as cf:
            current_cfg_json = cf.read()
        current_hash = hashlib.md5(current_cfg_json.encode('utf-8')).hexdigest()
        with open(os.path.join(exp_dir,'config_hash.txt'),'w') as hf:
            hf.write(f"config_md5={current_hash}\n")
    except Exception as e:
        print(f"[WARN] Unable to refresh config hash: {e}")

    # Append multi-seed summary to README if present
    readme_path = os.path.join(exp_dir,'README.md')
    try:
        if os.path.exists(readme_path):
            with open(readme_path,'a') as rf:
                rf.write('\n\n## Multi-Seed Summary\n')
                rf.write('| Metric | Mean | Std | CI Low | CI High |\n')
                rf.write('|--------|------|-----|--------|---------|\n')
                def fmt(v):
                    return f"{v:.4f}" if isinstance(v,(int,float)) and v is not None else '-'
                rf.write(f"| Val AUC | {fmt(auc_mean)} | {fmt(auc_std)} | - | - |\n")
                rf.write(f"| Mastery Corr | {fmt(mastery_mean)} | {fmt(mastery_std)} | {fmt(mastery_ci[0])} | {fmt(mastery_ci[1])} |\n")
                rf.write(f"| Gain Corr | {fmt(gain_mean)} | {fmt(gain_std)} | {fmt(gain_ci[0])} | {fmt(gain_ci[1])} |\n")
                rf.write('\nReproducibility checklist updated: multi-seed artifacts present.\n')
    except Exception as e:
        print(f"[WARN] Unable to update README with multi-seed summary: {e}")

    # Post-run normalized config snapshot for canonical reproduction (writes config_postrun.json) + diff artifacts
    try:
        used_epochs = cfg.get('training',{}).get('epochs')
        postrun_normalize_config(cfg, exp_dir, seeds, used_epochs)
        # Diff & MD5 comparison between original config and post-run normalized config
        try:
            import json as _json
            import hashlib as _hashlib
            orig_cfg_path = os.path.join(exp_dir,'config.json')
            post_cfg_path = os.path.join(exp_dir,'config_postrun.json')
            if os.path.exists(orig_cfg_path) and os.path.exists(post_cfg_path):
                orig_cfg = _json.load(open(orig_cfg_path))
                post_cfg = _json.load(open(post_cfg_path))
                def md5_excluding(d, field):
                    if not isinstance(d, dict):
                        return None
                    tmp = dict(d)
                    if field in tmp:
                        tmp.pop(field)
                    return _hashlib.md5(_json.dumps(tmp, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
                orig_md5_excl = md5_excluding(orig_cfg,'config_md5')
                post_md5_excl = md5_excluding(post_cfg,'config_postrun_md5')
                def top_level_diff(a,b):
                    added = []
                    removed = []
                    changed = []
                    if not isinstance(a, dict) or not isinstance(b, dict):
                        return added, removed, changed
                    a_keys = set(a.keys())
                    b_keys = set(b.keys())
                    for k in sorted(b_keys - a_keys):
                        added.append(k)
                    for k in sorted(a_keys - b_keys):
                        removed.append(k)
                    for k in sorted(a_keys & b_keys):
                        av = a[k]
                        bv = b[k]
                        if av == bv:
                            continue
                        if isinstance(av, dict) and isinstance(bv, dict):
                            ah = _hashlib.md5(_json.dumps(av, sort_keys=True, separators=(',',':')).encode()).hexdigest()
                            bh = _hashlib.md5(_json.dumps(bv, sort_keys=True, separators=(',',':')).encode()).hexdigest()
                            if ah != bh:
                                changed.append(k)
                        else:
                            changed.append(k)
                    return added, removed, changed
                added, removed, changed = top_level_diff(orig_cfg, post_cfg)
                diff_payload = {
                    'original_config_md5_excluding_field': orig_md5_excl,
                    'postrun_config_md5_excluding_field': post_md5_excl,
                    'config_postrun_md5_match': (orig_md5_excl == post_md5_excl),
                    'top_level_added': added,
                    'top_level_removed': removed,
                    'top_level_changed': changed
                }
                with open(os.path.join(exp_dir,'config_postrun_diff.json'),'w') as dfh:
                    _json.dump(diff_payload, dfh, indent=2)
                with open(os.path.join(exp_dir,'config_postrun_diff.txt'),'w') as tfh:
                    tfh.write(f"Original(excl field) MD5: {orig_md5_excl}\n")
                    tfh.write(f"Postrun(excl field) MD5: {post_md5_excl}\n")
                    tfh.write(f"Match: {orig_md5_excl == post_md5_excl}\n")
                    tfh.write(f"Added: {', '.join(added) if added else '<none>'}\n")
                    tfh.write(f"Removed: {', '.join(removed) if removed else '<none>'}\n")
                    tfh.write(f"Changed: {', '.join(changed) if changed else '<none>'}\n")
        except Exception as e:
            print(f"[WARN] Failed to compute config_postrun diff: {e}")
    except Exception as e:
        print(f"[WARN] Post-run normalization failed: {e}")

    # Perform reproduction comparison at end (after aggregation) for clearer ordering
    if args.reproduce:
        def perform_reproduction_comparison(source_exp_id: str, new_exp_dir: str, seeds_list, per_seed_new, auc_mean_new, mastery_mean_new, gain_mean_new):
            report = {}
            try:
                source_dir = os.path.join(EXPERIMENTS_ROOT, source_exp_id)
                src_results_path = os.path.join(source_dir,'results.json')
                if not os.path.exists(src_results_path):
                    return {'error': 'source results.json missing'}
                src = json.load(open(src_results_path))
                def safe_float(x):
                    return x if isinstance(x,(int,float)) else None
                def extract_per_seed(block, attr_map):
                    out = {k: [] for k in attr_map}
                    for r in block.get('per_seed', []):
                        for metric, keys in attr_map.items():
                            val = None
                            # new structured best_epoch_metrics path
                            if 'best_epoch_metrics' in r and isinstance(r['best_epoch_metrics'], dict):
                                if metric == 'val_auc':
                                    val = safe_float(r['best_epoch_metrics'].get('val_auc'))
                                else:
                                    val = safe_float(r['best_epoch_metrics'].get(metric))
                            if val is None:
                                for k in keys:
                                    if k in r:
                                        val = safe_float(r.get(k))
                                        break
                            if val is None and isinstance(r.get('final_consistency_metrics'), dict):
                                val = safe_float(r['final_consistency_metrics'].get(metric))
                            out[metric].append(val)
                    return out
                attr_map = {
                    'val_auc': ['best_val_auc','val_auc'],
                    'mastery_correlation': ['best_mastery_corr','mastery_correlation'],
                    'gain_correlation': ['best_gain_corr','gain_correlation']
                }
                src_metrics = extract_per_seed(src, attr_map)
                new_metrics = extract_per_seed({'per_seed': per_seed_new}, attr_map)
                def max_abs_diff(a,b):
                    diffs = []
                    for x,y in zip(a,b):
                        if x is None or y is None:
                            continue
                        diffs.append(abs(x-y))
                    return max(diffs) if diffs else None
                auc_diff = max_abs_diff(src_metrics['val_auc'], new_metrics['val_auc'])
                mastery_diff = max_abs_diff(src_metrics['mastery_correlation'], new_metrics['mastery_correlation'])
                gain_diff = max_abs_diff(src_metrics['gain_correlation'], new_metrics['gain_correlation'])
                tol_auc = 0.002
                tol_corr = 0.01
                report = {
                    'source_experiment': source_dir,
                    'new_experiment': new_exp_dir,
                    'seeds': seeds_list,
                    'source_best_val_auc': src_metrics['val_auc'],
                    'new_best_val_auc': new_metrics['val_auc'],
                    'max_best_val_auc_abs_diff': auc_diff,
                    'source_mastery_corr': src_metrics['mastery_correlation'],
                    'new_mastery_corr': new_metrics['mastery_correlation'],
                    'max_mastery_corr_abs_diff': mastery_diff,
                    'source_gain_corr': src_metrics['gain_correlation'],
                    'new_gain_corr': new_metrics['gain_correlation'],
                    'max_gain_corr_abs_diff': gain_diff,
                    'auc_within_tolerance': auc_diff is not None and auc_diff <= tol_auc,
                    'mastery_within_tolerance': mastery_diff is not None and mastery_diff <= tol_corr,
                    'gain_within_tolerance': gain_diff is not None and gain_diff <= tol_corr,
                    'tolerances': {'auc': tol_auc, 'corr': tol_corr},
                    'aggregated_new': {
                        'val_auc_mean': auc_mean_new,
                        'mastery_correlation_mean': mastery_mean_new,
                        'gain_correlation_mean': gain_mean_new
                    },
                    'fully_reproduced': all([
                        auc_diff is not None and auc_diff <= tol_auc,
                        mastery_diff is not None and mastery_diff <= tol_corr,
                        gain_diff is not None and gain_diff <= tol_corr
                    ])
                }
            except Exception as e:
                report = {'error': f'comparison_failed: {e}'}
            return report
        reproduction_report = perform_reproduction_comparison(args.experiment_dir, exp_dir, seeds, per_seed_results, auc_mean, mastery_mean, gain_mean)
        # Inject results md5 info for source & new if available
        try:
            import hashlib
            source_results_path = os.path.join(EXPERIMENTS_ROOT, args.experiment_dir,'results.json')
            source_md5 = None
            if os.path.exists(source_results_path):
                with open(source_results_path,'rb') as srfh:
                    source_md5 = hashlib.md5(srfh.read()).hexdigest()
            reproduction_report['source_results_md5'] = source_md5
            reproduction_report['new_results_md5'] = results_md5
            reproduction_report['results_md5_match'] = (source_md5 is not None and results_md5 is not None and source_md5 == results_md5)
            # Inject post-run config md5 comparison if both available
            try:
                import json as _json
                src_post_path = os.path.join(EXPERIMENTS_ROOT, args.experiment_dir,'config_postrun.json')
                new_post_path = os.path.join(exp_dir,'config_postrun.json')
                src_post_md5 = None
                new_post_md5 = None
                if os.path.exists(src_post_path):
                    src_post_md5 = _json.load(open(src_post_path)).get('config_postrun_md5')
                if os.path.exists(new_post_path):
                    new_post_md5 = _json.load(open(new_post_path)).get('config_postrun_md5')
                reproduction_report['source_config_postrun_md5'] = src_post_md5
                reproduction_report['new_config_postrun_md5'] = new_post_md5
                reproduction_report['config_postrun_md5_match'] = (src_post_md5 is not None and new_post_md5 is not None and src_post_md5 == new_post_md5)
            except Exception as e3:
                reproduction_report['config_postrun_md5_error'] = str(e3)
        except Exception as e:
            reproduction_report['results_md5_error'] = str(e)
        try:
            # Normalize any numpy scalar types to Python native
            def normalize(obj):
                if isinstance(obj, dict):
                    return {k: normalize(v) for k,v in obj.items()}
                if isinstance(obj, list):
                    return [normalize(v) for v in obj]
                import numpy as np
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return obj
            reproduction_report = normalize(reproduction_report)
            tmp_path = os.path.join(exp_dir,'reproduction_report.json.tmp')
            final_path = os.path.join(exp_dir,'reproduction_report.json')
            with open(tmp_path,'w') as rr:
                json.dump(reproduction_report, rr, indent=2)
            os.replace(tmp_path, final_path)
        except Exception as e:
            print(f"[WARN] Failed to write reproduction_report.json: {e}")
        # Append reproduction section
        try:
            if os.path.exists(readme_path):
                with open(readme_path,'a') as rf:
                    rf.write('\n\n## Reproduction Comparison (Post-Aggregation)\n')
                    if 'error' in reproduction_report:
                        rf.write(f"Error: {reproduction_report['error']}\n")
                    else:
                        rf.write(f"Fully Reproduced: {reproduction_report['fully_reproduced']}\n\n")
                        rf.write('| Metric | Max Abs Diff | Within Tolerance |\n')
                        rf.write('|--------|--------------|------------------|\n')
                        def fmt(v):
                            return f"{v:.4f}" if isinstance(v,(int,float)) and v is not None else '-'
                        rf.write(f"| Val AUC | {fmt(reproduction_report.get('max_best_val_auc_abs_diff'))} | {reproduction_report.get('auc_within_tolerance')} |\n")
                        rf.write(f"| Mastery Corr | {fmt(reproduction_report.get('max_mastery_corr_abs_diff'))} | {reproduction_report.get('mastery_within_tolerance')} |\n")
                        rf.write(f"| Gain Corr | {fmt(reproduction_report.get('max_gain_corr_abs_diff'))} | {reproduction_report.get('gain_within_tolerance')} |\n")
        except Exception as e:
            print(f"[WARN] Unable to append post-aggregation reproduction comparison to README: {e}")

    sys.stdout = orig_stdout
    log_fh.close()
    print(f"[INFO] Experiment completed. Folder: {exp_dir}")


if __name__ == '__main__':
    main()
