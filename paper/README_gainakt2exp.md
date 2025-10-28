# GainAKT2Exp Experiment Launch & Reproducibility Guide

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

We document how to launch, reproduce, and interpret GainAKT2Exp experiments using the canonical launcher `examples/train_gainakt2exp_repro.py`. The objective is to ensure academically rigorous reproducibility while enabling controlled ablations and interpretability analyses (alignment, retention, lag gain).

---
## 1. Purpose & Scope
This guide covers:
- Baseline multi-seed launch pattern
- Default hyperparameter values (performance + interpretability constraints)
- Scenario-based invocation (fast test, full interpretability, ablation cases)
- Reproducibility artifacts & integrity checks
- Recommended override etiquette
- Common pitfalls and mitigations

---
## 2. Canonical Launcher
Script: `examples/train_gainakt2exp_repro.py`
Role: Create timestamped experiment folders under `examples/experiments/` honoring structure:
```
YYYYMMDD_HHMMSS_gainakt2exp_<short_title>
```
Artifacts (per run):
- `config.json` (resolved parameters + `config_md5`)
- `environment.txt` (Python, PyTorch, CUDA, git commit/branch)
- `stdout.log` (timestamped console output)
- `metrics_epoch.csv` (per-epoch AUC, accuracy, correlations, loss shares)
- `results.json` (per-seed best metrics)
- `results_multi_seed.json` (if >1 seed)
- `SEED_INFO.md` (seed list and aggregate summaries)
- `model_best.pth`, `model_last.pth`
- `README.md` (run summary + Parameter Scenario Quick Reference)

AMP is enabled by default (mixed precision) unless overridden (`--no_amp`). Alignment, retention, and lag gain losses are enabled via the default `--ablation_mode both_lag` configuration.

---
## 3. Default Parameter Values
Below we list principal defaults (launcher perspective). Some internal model/training defaults may differ; the launcher overrides them to enforce consistency.

### 3.1 Training
| Parameter | Default | Notes |
|-----------|---------|-------|
| epochs | 40 | Adjust for exploratory runs (e.g., 3–5 for fast sanity). |
| batch_size | 64 | AMP allows this size; reduce if OOM. |
| learning_rate | 0.00016 | Tuned for stable constraint integration. |
| weight_decay | 1.7571e-05 | Light regularization; can set 0 for pure comparison. |
| optimizer | Adam | Standard; no scheduler unless specified. |
| gradient_clip | 1.0 | Prevent exploding gradients under constraint mix. |
| mixed_precision (AMP) | True | Disable with `--no_amp`. |
| patience | 10 | Early stopping patience for potential scheduler logic. |

### 3.2 Interpretability & Constraints
| Parameter | Default | Effect |
|-----------|---------|--------|
| monotonicity_loss_weight | 0.1 | Penalize temporal mastery regressions. |
| mastery_performance_loss_weight | 0.8 | Contribution of mastery head to main performance objective. |
| gain_performance_loss_weight | 0.8 | Contribution of gain head to main performance objective. |
| sparsity_loss_weight | 0.2 | Drives compact attention/representation usage. |
| consistency_loss_weight | 0.3 | Maintains stable mastery/gain trajectories. |
| warmup_constraint_epochs | 8 | Gradual ramp of constraint influence. |
| alignment_weight | 0.30 | Semantic alignment of mastery/gain progression. |
| retention_weight | 0.12 | Penalizes premature forgetting / drift. |
| lag_gain_weight | 0.05 | Encourages lag-aware progressive gain consistency. |

### 3.3 Alignment & Sampling
| Parameter | Default | Notes |
|-----------|---------|-------|
| alignment_global_students | 600 | Batch-scope for global semantic alignment pass. |
| max_semantic_students | 50 | Sample size per epoch for correlation calculations. |
| alignment_share_cap | 0.10 | Upper bound for alignment loss share (adaptive). |
| alignment_share_decay_factor | 0.75 | Decays cap after warmup. |

### 3.4 Hardware
| Parameter | Default | Notes |
|-----------|---------|-------|
| devices | 0 1 2 3 4 | Use first five GPUs (≈60% utilization). |
| threads | 8 | Limit CPU threads for reproducibility. |

---
## 4. Core Launch Patterns

### 4.1 Baseline Multi-Seed Run
```
python examples/train_gainakt2exp_repro.py \
  --experiment_title baseline \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag
```
Produces folder: `YYYYMMDD_HHMMSS_gainakt2exp_baseline`.

### 4.2 Fast Sanity (3 Epochs, Single Seed)
```
python examples/train_gainakt2exp_repro.py \
  --experiment_title quickcheck \
  --seeds 21 \
  --devices 0 \
  --ablation_mode both_lag \
  --epochs 3
```
Use to verify environment & artifact generation quickly.

### 4.3 Disable Alignment / Retention (Performance Focus)
```
python examples/train_gainakt2exp_repro.py \
  --experiment_title perf_only \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode none
```
Ablation mode `none` expected to disable alignment, retention, lag gain (depends on script implementation of mode mapping).

### 4.4 Alignment Only (Interpretability Emphasis)
```
python examples/train_gainakt2exp_repro.py \
  --experiment_title align_only \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode alignment_only
```
If not directly supported, mirror via manual flags (future extension).**

### 4.5 No AMP (Deterministic Reference)
```
python examples/train_gainakt2exp_repro.py \
  --experiment_title no_amp_ref \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --no_amp
```
Use when exact reproducibility metrics (bit-level) required.

### 4.6 Extended Warmup (Interpretability Stabilization)
```
python examples/train_gainakt2exp_repro.py \
  --experiment_title warmup12 \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --warmup_constraint_epochs 12
```
Slows constraint pressure onset; may preserve early AUC.

### 4.7 High Sparsity Exploration
```
python examples/train_gainakt2exp_repro.py \
  --experiment_title high_sparsity \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --sparsity_loss_weight 0.35
```
Document justification (e.g., interpretability emphasis) in README.

---
## 5. Scenario Matrix
| Scenario | Goal | Key Flags | Expected Outcome |
|----------|------|-----------|------------------|
| Baseline | Balanced performance + interpretability | `--ablation_mode both_lag` | Stable AUC peak early, mild decline; full constraints active. |
| Performance Focus | Maximize raw AUC | `--ablation_mode none` | Higher best AUC; reduced interpretability metrics. |
| Interpretability Max | Strong semantic signals | Increase alignment/retention/lag weights | Potential modest AUC trade-off; improved correlation stability. |
| Deterministic Reference | Audit-grade reproducibility | `--no_amp` | Slightly higher memory use; identical metric hashes. |
| Fast Sanity | Environment check | `--epochs 3 --seeds 21` | Quick artifact generation; not for publication. |
| High Sparsity | Feature compression | `--sparsity_loss_weight ≥0.3` | Sparser representations; monitor for AUC drop. |
| Extended Warmup | Delay constraint pressure | `--warmup_constraint_epochs >8` | Smoother early AUC trajectory. |

---
## 6. Reproducibility Protocol
1. Read `schema_version` and ensure it matches the expected launcher version (currently `2`). If versions differ, reproduction claims must note the discrepancy.
2. Confirm `missing_params` is an empty array in `config.json`. Non-empty indicates a serialization regression; do not cite such runs.
3. Record `config_md5` from `config.json` (this hash already excludes no fields; it is computed over the entire serialized object including timestamp). For normalized structural comparison you may recompute a hash with volatile keys removed (see Step 8).
4. Preserve `environment.txt` (Python, torch, CUDA, git commit, branch). Optionally compute an environment hash: `md5sum environment.txt`.
5. For a reproduction attempt, re-run with identical CLI and verify:
   - Matching `config_md5`.
   - Identical `schema_version`.
   - Empty `missing_params`.
   - Per-seed best metrics within tolerance (AUC drift ≤0.002; correlation drift ≤0.01) or bit-identical if AMP disabled.
6. Archive `stdout.log` (timestamped lines serve as temporal audit). Consider computing a rolling hash: `sha256sum stdout.log`.
7. Include experiment folder name verbatim in reporting (tables, appendices) and list `config_md5`.
8. (Optional) Normalized structural hash excluding volatile fields (if future schema adds more):
   ```bash
   jq 'del(.runtime.timestamp, .config_md5)' config.json | jq -S '.' | md5sum
   ```
   This allows verification that only timestamp differs when repeating an identical run.
9. (Optional) Create an audit manifest capturing: `config_md5`, normalized hash, `environment.txt` hash, first & last 5 lines of `stdout.log`.

---
## 6.a Step-by-Step Experiment Reproduction (Schema v2)
Audit-grade procedure for reproducing an experiment created with `train_gainakt2exp_repro.py`.

### A. Identify Source Experiment
Locate original folder, e.g.:
```
examples/experiments/20251027_180238_gainakt2exp_remediate
```
Verify presence of: `config.json`, `results.json`, `metrics_epoch.csv`, `environment.txt`, `README.md`.

### B. Inspect Integrity Anchors
```bash
jq '.schema_version, .missing_params, .config_md5' examples/experiments/20251027_180238_gainakt2exp_remediate/config.json
```
Expected output:
- `schema_version` = 2
- `missing_params` = []
- `config_md5` = <hash>
If `missing_params` not empty, halt and patch launcher before reproduction.

### C. Optional Normalized Config Hash
Remove ephemeral timestamp & `config_md5` then hash:
```bash
jq 'del(.runtime.timestamp, .config_md5)' examples/experiments/20251027_180238_gainakt2exp_remediate/config.json | jq -S '.' | md5sum
```
Record as `normalized_config_hash`.

### D. Reconstruct Original Command (Exact Flags)
Extract needed flags directly:
```bash
python examples/train_gainakt2exp_repro.py \
  --experiment_title remediate \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --learning_rate 0.00016 \
  --batch_size 64 \
  --epochs 40 \
  --weight_decay 1.7571e-05
```
Include any non-default flags if they differ from defaults (e.g., changed `alignment_weight`). Do NOT add flags that were absent originally.

### E. Launch Reproduction Run
Use distinct title (e.g., `remediate_repro`) and identical flags:
```bash
python examples/train_gainakt2exp_repro.py \
  --experiment_title remediate_repro \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --learning_rate 0.00016 \
  --batch_size 64 \
  --epochs 40 \
  --weight_decay 1.7571e-05
```
Ensure no manual edits to generated `config.json`.

### F. Verification Checklist
| Item | Verification Command | Pass Criterion |
|------|----------------------|----------------|
| Schema version | `jq '.schema_version' config.json` | Equals 2 |
| Missing params | `jq '.missing_params' config.json` | Empty array |
| Config hash | `jq -r '.config_md5' config.json` | Matches original |
| Normalized hash | recompute (Step C) | Matches original normalized hash |
| Environment | `diff source/environment.txt repro/environment.txt` | No diff (or documented benign version bump) |
| Seeds | `jq '.seeds.all' config.json` | Identical list/order |
| Best metrics | compare `results.json` | Within tolerance |

### G. Metric Tolerances
| Metric | Strict (AMP off) | Standard (AMP on) |
|--------|------------------|-------------------|
| Best Val AUC diff | 0.0000–0.0002 | ≤0.002 |
| Mastery Corr diff | 0.0000–0.0005 | ≤0.01 |
| Gain Corr diff | 0.0000–0.0005 | ≤0.01 |

### H. Failure Diagnostics Enhancements
| Symptom | Cause | Additional Probe | Remedy |
|---------|-------|------------------|--------|
| Hash mismatch | Flag drift | `diff <normalized configs>` | Re-run with corrected flags |
| Large AUC drift | Dataset variant changed | Check dataset file checksum | Ensure identical data split |
| Corr instability | Sampling variance | Increase `--max_semantic_students` | Reproduce with higher sample |
| `missing_params` non-empty | Launcher update unsynced | Inspect build_config | Patch & rerun |

### I. Strengthening Options
- Disable AMP (`--no_amp`) for deterministic floating point.
- Set backend determinism in script (future: add flags `--torch_deterministic`).
- Capture RNG states per seed into `artifacts/rng_seed<N>.json` (future extension).

### J. Archival Manifest Template
Create `reproduction_manifest.json` in reproduction folder:
```json
{
  "source_experiment": "20251027_180238_gainakt2exp_remediate",
  "reproduction_experiment": "20251029_093312_gainakt2exp_remediate_repro",
  "schema_version": 2,
  "config_md5_source": "<hash>",
  "config_md5_repro": "<hash>",
  "normalized_config_hash_source": "<hash>",
  "normalized_config_hash_repro": "<hash>",
  "environment_match": true,
  "auc_diff": 0.0004,
  "mastery_corr_diff": 0.0002,
  "gain_corr_diff": 0.0003,
  "within_tolerance": true,
  "amp_used": true,
  "notes": "All criteria satisfied; minor AUC drift within tolerance."
}
```

### K. Citation Wording Example
> We reproduced experiment `20251027_180238_gainakt2exp_remediate` using identical schema (v2) and matched `config_md5` (83f8…e2). Metric deltas (AUC +0.0004; mastery corr +0.0002; gain corr +0.0003) fall below defined AMP tolerance thresholds; environment and seeds identical; `missing_params=[]`.

This updated protocol supersedes earlier Section 6 instructions; all reported reproductions must satisfy schema + hash + tolerance requirements.

---
## 15. Unified Script Enhancements (October 2025)
We merged the former reproduction utility (`examples/repro_train_gainakt2exp.py`) into the canonical launcher `examples/train_gainakt2exp_repro.py`, deprecating the separate script. The unified launcher now supports training, reproduction, and comparison workflows with a single interface.

### 15.1 Deprecation Notice
`examples/repro_train_gainakt2exp.py` is deprecated and replaced by a stub that instructs users to invoke `train_gainakt2exp_repro.py`. All future reproducibility claims must reference the unified script.

### 15.2 New CLI Flags
| Flag | Purpose | Notes |
|------|---------|-------|
| --experiment_dir <id|path> | Use or reuse a specific experiment directory (manual naming) | Requires `--force` if directory exists. |
| --force | Allow reuse of an existing populated directory | Prevents accidental overwrite protection. |
| --set key=val | Dot-path override applied before hashing | Repeatable; affects `config_md5`. |
| --reproduce_from <exp_id|path> | Clone an existing experiment (creates *_reproduce) | Source `config_md5` recorded. |
| --compare_only | Compare two finished experiments without training | Requires `--source_exp` and `--target_exp`. |
| --source_exp <id|path> | Source experiment for compare-only | Can be relative id or absolute path. |
| --target_exp <id|path> | Target experiment for compare-only | Comparison report stored in target folder. |
| --manifest | Writes `reproduction_manifest.json` summarizing run metadata | Optional audit artifact. |
| --strict_schema | Abort reproduction if schema_version mismatch | Ensures stable config structure. |

### 15.3 Override Semantics (`--set`)
Overrides use dot-path navigation (e.g., `training.epochs=3`). Coercion rules:
- Booleans: `true|false`
- Floats: contain at least one decimal point
- Integers: numeric without decimal
- Lists: comma separated; each element individually coerced
All overrides are applied prior to computing `config_md5`. The audit artifact (`overrides_applied.txt`) logs each override in the form `key:old->new` to enforce forensic traceability.

### 15.4 Reproduction Workflow (`--reproduce_from`)
When reproduction is requested:
1. Source `config.json` loaded and schema checked (if `--strict_schema`).
2. New folder is created with `_reproduce` suffix based on source short title.
3. Current CLI overrides applied; new `config_md5` computed.
4. Reproduction metadata embedded (`reproduction` block in config + README table).
5. Optional manifest created if `--manifest` is supplied.
6. A comparison report (`reproduction_report.json`) generated using unified comparison logic.

Example: 
`python examples/train_gainakt2exp_repro.py 
  --reproduce_from 20251027_180238_gainakt2exp_remediate 
  --devices 0 1 2 3 4`
`

### 15.5 Compare-Only Mode (`--compare_only`)
Performs a fast diff without retraining:
- Loads `results.json` from both experiments (falls back to empty if missing).
- Computes absolute differences (AUC, mastery corr, gain corr).
- Evaluates tolerance (AUC ≤0.002; correlations ≤0.01).
- Produces `reproduction_report.json` in target folder with MD5 status for both configs and results artifacts.

### 15.6 Results MD5 Hashing
The unified script computes MD5 for `results.json` on completion. Comparison and reproduction reports include:
- `source_results_md5` / `target_results_md5`
- `results_md5_match` boolean
A matching `config_md5` but differing `results_md5` indicates environment or stochastic drift; investigate `environment.txt` and seeds.

### 15.7 README Extensions
Experiment `README.md` now includes:
- Schema version line
- Reproduction metadata table (if reproduction)
- Reference to full parameter tables in this guide

### 15.8 Manifest File (`--manifest`)
Optional `reproduction_manifest.json` fields:
| Field | Description |
|-------|-------------|
| experiment_id | Folder identifier |
| schema_version | Config schema version |
| config_md5 | Canonical configuration hash |
| seeds | Executed seeds list |
| reproduction_mode | True if `--reproduce_from` used |
| override_count | Number of applied overrides |
| results_md5 | MD5 of `results.json` |

### 15.9 Tolerance Criteria (Recap)
| Metric | Threshold |
|--------|-----------|
| AUC absolute diff | ≤0.002 (AMP) / ≤0.0002 (no AMP) |
| Mastery corr absolute diff | ≤0.01 (AMP) / ≤0.0005 (no AMP) |
| Gain corr absolute diff | ≤0.01 (AMP) / ≤0.0005 (no AMP) |

### 15.10 Recommended Citation Update
Cite unified script and hash posture:
```
All GainAKT2Exp experiments launched with unified script (train_gainakt2exp_repro.py, schema v2). Overrides applied (n=2): training.epochs:40->20, alignment_weight:0.30->0.28. Reproduction of 20251027_180238_gainakt2exp_remediate achieved matching config_md5 (83f8…e2) and results within tolerance (AUC diff 0.0004).
```

### 15.11 Migration Notes
Legacy folders created with the deprecated script remain valid; note their original tool in any citation. New experiments should not use the deprecated script stub.

### 15.12 Future Consolidation Tasks
Planned additions to unified script:
- `--from_config <path>` direct config ingestion (bypass CLI divergence)
- `--unset key.path` for removing keys
- Deterministic backend flag group (`--torch_deterministic`) capturing cudnn settings
- RNG state artifact per seed (`artifacts/rng_seed<N>.json`)
- Head variance/skew instrumentation flags (`--log_head_stats`) and normalization (`--enable_head_normalization`)

These enhancements will bump `schema_version` upon structural additions; reproduction documentation will be updated accordingly.

---
Unified scripting consolidates reproducibility, reduces maintenance surface, and strengthens audit trails via integrated override and comparison reporting.
