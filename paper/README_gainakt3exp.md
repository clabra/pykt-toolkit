# GainAKT3 Experiment Launch & Reproducibility Guide

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

We document how to launch, reproduce, and interpret GainAKT3 experiments using the canonical launcher `examples/wandb_gainakt3_train.py` (formerly `train_gainakt3_repro.py`). The objective is to ensure academically rigorous reproducibility while supporting interpretability (alignment, consistency, retention, lag gain, peer alignment, difficulty ordering) and contextual fusion (peer + difficulty).

---

## 1. Purpose & Scope

This guide covers:

- Baseline multi-seed launch pattern
- Default hyperparameter values (performance + interpretability constraints)
- Scenario-based invocation (fast sanity, alignment emphasis, retention emphasis, heads-only fusion)
- Reproducibility artifacts & integrity checks
- Override etiquette (`--set key=val`)
- Common pitfalls and mitigations

---
 
## 2. Canonical Launcher

Script: `examples/wandb_gainakt3_train.py`

Creates timestamped experiment folders under `examples/experiments/`:

```text
YYYYMMDD_HHMMSS_gainakt3_<short_title>
```

Artifacts per run:

- `config.json` (all serialized parameters + `config_md5` + `missing_params=[]`)
- `environment.txt` (Python, PyTorch, CUDA, git commit/branch)
- `stdout.log` (timestamped console output)
- `metrics_epoch.csv` (per-epoch AUC, accuracy, correlations, constraint loss shares + raw values)
- `results.json` (primary seed best metrics)
- `results_multi_seed.json` (multi-seed aggregated stats: AUC, mastery corr, gain corr)
- `SEED_INFO.md` (seed manifest)
- `model_best_seed<N>.pth`, `model_last_seed<N>.pth`
- `README.md` (summary + reproducibility checklist)

Mixed precision (AMP) is enabled by default unless `--no_amp` is passed.

---
 
## 3. Default Parameter Values

Launcher defaults (see script for full list).

### 3.1 Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| epochs | 10 | Increase for convergence (e.g., 30–50). |
| batch_size | 64 | Adjust for GPU memory. |
| learning_rate (`--lr`) | 0.0003 | Tuned from early sweeps. |
| weight_decay | 0.0 | Set >0 for regularization comparisons. |
| gradient_clip (`--grad_clip`) | 1.0 | Stabilizes under constraint sums. |
| mixed_precision | True | Disable with `--no_amp`. |
| warmup_constraint_epochs | 3 | Ramp constraint influence; extend for stability. |

### 3.2 Interpretability & Constraint Weights

| Parameter | Default | Effect |
|-----------|---------|--------|
| alignment_weight | 0.05 | Encourages temporal alignment of mastery vs gain. |
| consistency_weight | 0.2 | Stabilizes head trajectories. |
| retention_weight | 0.0 | Penalizes premature decay (off by default). |
| lag_gain_weight | 0.05 | Lag-aware smooth gain evolution. |
| sparsity_weight | 0.0 | Enables representation sparsity when >0. |
| peer_alignment_weight | 0.05 | Aligns peer-influenced adjustments. |
| difficulty_ordering_weight | 0.0 | Enforces ordered difficulty progression. |
| drift_smoothness_weight | 0.0 | Smoothens concept difficulty drift. |

### 3.3 Context & Fusion

| Parameter | Default | Notes |
|-----------|---------|-------|
| use_peer_context | False | Enable to incorporate peer similarity. |
| use_difficulty_context | False | Enable to incorporate difficulty priors. |
| fusion_for_heads_only | True | Restricts fusion to prediction heads (interpretability). |
| disable_fusion_broadcast | False | If True, disables feature broadcasting. |
| disable_difficulty_penalty | False | Turns off difficulty penalty path. |
| peer_K | 8 | Peer neighborhood size. |
| gate_init_bias | -2.0 | Negative bias starts gate mostly closed. |

### 3.4 Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| beta_difficulty | 0.5 | Scales difficulty features. |
| attempt_confidence_k | 5.0 | Confidence shaping factor. |
| gain_threshold | 0.01 | Threshold for gain-based metrics. |
| mastery_temperature | 1.0 | Temperature for mastery head. |

### 3.5 Heads & Data Shape

| Parameter | Default | Notes |
|-----------|---------|-------|
| use_mastery_head | True | Predictive + interpretability mastery signal. |
| use_gain_head | True | Predictive + interpretability gain signal. |
| num_c | 100 | Concept cardinality (set per dataset). |
| maxlen | 200 | Max sequence length. |
| input_type | concepts | Input modality. |

### 3.6 Hardware

| Parameter | Default | Notes |
|-----------|---------|-------|
| devices | 0 1 2 3 4 | First five GPUs (≈60% utilization). |
| num_workers | 5 | DataLoader workers. |
| threads | 8 | CPU threads (from env). |

---
 
## 4. Launch Patterns

### 4.1 Baseline (Single Seed)

```bash
python examples/wandb_gainakt3_train.py \
  --dataset assist2015 \
  --short_title baseline
```

### 4.2 Multi-Seed Stability

```bash
python examples/wandb_gainakt3_train.py \
  --dataset assist2015 \
  --short_title stability_run \
  --seeds 21 42 63 84 105 \
  --alignment_weight 0.15 \
  --consistency_weight 0.25 \
  --warmup_constraint_epochs 8
```

### 4.3 Enable Context Fusion

```bash
python examples/wandb_gainakt3_train.py \
  --dataset assist2015 \
  --short_title ctx_fusion \
  --use_peer_context \
  --use_difficulty_context \
  --alignment_weight 0.15 \
  --consistency_weight 0.25
```

### 4.4 Retention Emphasis

```bash
python examples/wandb_gainakt3_train.py \
  --dataset assist2015 \
  --short_title retention_focus \
  --retention_weight 0.10 \
  --warmup_constraint_epochs 6
```

### 4.5 Heads-Only Fusion + Strong Alignment

```bash
python examples/wandb_gainakt3_train.py \
  --dataset assist2015 \
  --short_title heads_align \
  --fusion_for_heads_only \
  --alignment_weight 0.25 \
  --warmup_constraint_epochs 8
```

### 4.6 High Sparsity Exploration

```bash
python examples/wandb_gainakt3_train.py \
  --dataset assist2015 \
  --short_title high_sparsity \
  --sparsity_weight 0.3
```

### 4.7 Fast Synthetic Smoke Test

```bash
python examples/wandb_gainakt3_train.py \
  --use_synthetic \
  --epochs 2 \
  --short_title synthetic_smoke
```

---
 
## 5. Overrides (`--set`)

Use dot-path to modify parameters prior to hashing:

```bash
python examples/wandb_gainakt3_train.py \
  --dataset assist2015 \
  --short_title override_demo \
  --set training.epochs=20 \
  --set weights.alignment_weight=0.12 \
  --set context.use_peer_context=true
```

Coercion rules: `true|false` → boolean; numeric with `.` → float; else int; comma lists become arrays (each element coerced).

---
 
## 6. Reproducibility Protocol (Primary Run)

Primary (non-reproduction) runs establish the canonical reference state.

Checklist:

1. Ensure `missing_params` is empty in `config.json` (guarantees no silent defaults were omitted).
2. Record `config_md5` for citation (exact hash of serialized resolved config JSON; cosmetic CLI changes alter it).
3. Verify artifact hashes (`peer_index_sha256`, `difficulty_table_sha256`) for contextual reproducibility.
4. Archive `README.md` autogenerated reproducibility checklist (all ✅).
5. Inspect `metrics_epoch.csv` for monotonic stabilization of interpretability metrics (alignment, consistency, retention shares).
6. Multi-seed stability (if multiple seeds provided) summarized in `results_multi_seed.json` (mean, std, min, max for AUC, mastery_corr, gain_corr).
7. Cite experiment folder name exactly (e.g., `20251028_092134_gainakt3_audit_ext`) plus `config_md5` in papers and logs.

Tolerance heuristics (initial reference):

- Validation AUC drift ≤ 0.002
- Mastery / gain correlation drift ≤ 0.01

These apply when reproducing AMP-enabled runs. For stricter comparison disable AMP (`--no_amp`).

Hash Interpretation:

- `config_md5` changes if ANY serialized value changes (including `short_title`).
- A reproduction with identical semantic hyperparameters but a different `short_title` will have a different hash yet metrics may still fall within tolerance; cite both original and reproduction hashes when reporting.
- Future enhancement will add a `normalized_config_md5` excluding cosmetic fields (not yet present).

---
 
## 7. Common Pitfalls & Mitigations

| Pitfall | Symptom | Mitigation |
|---------|---------|-----------|
| Missing artifact → cold start | Increased early variance | Pre-generate peer/difficulty artifacts; verify hashes. |
| Constraint weight imbalance | AUC collapse | Increase warmup epochs; reduce alignment_weight. |
| Oversparsity | Flat correlations | Lower sparsity_weight or raise alignment_weight gradually. |
| Gate never opens | Low peer influence share | Reduce negative gate_init_bias (e.g., -1.0). |
| Retention over-regularization | Mastery corr stagnation | Lower retention_weight or raise warmup. |

---
 
## 8. Citation Template

> Experiment `20251028_092134_gainakt3_audit_ext` (GainAKT3, schema v1) executed with config_md5 `a7ce617c...edab`, seeds `[21]`, mixed_precision enabled. Alignment_weight=0.05, retention_weight=0.0, peer_alignment_weight=0.05. Artifact hashes recorded; `missing_params=[]`.

---
 
## 9. Future Enhancements

Planned additions:

- Temporal stability metrics (variance/slopes) analogous to GainAKT2Exp.
- Resume protocol (`--resume`) with RNG state snapshot.
- Direct config ingestion (`--from_config <path>`).
- Deterministic backend flag group.

 
## 10. Reproduction Workflow (`--reproduce_from`)

We provide an explicit reproduction mode to evaluate metric drift relative to a source experiment.

 
### 10.1 Flags

| Flag | Purpose |
|------|---------|
| `--reproduce_from <folder_name>` | Points to an existing experiment folder (basename only) under `examples/experiments/`. Triggers reproduction mode. |
| `--strict_schema` | Aborts if the source and current launcher schema differ (structural safety). |
| `--manifest` | Saves a `reproduction_manifest.json` enumerating source vs reproduced artifact hashes and metric diffs. |

 
### 10.2 Steps

1. Run a source experiment normally (e.g., `20251028_093235_gainakt3_base_src`).
2. Invoke reproduction:

```bash
python examples/wandb_gainakt3_train.py \
  --reproduce_from 20251028_093235_gainakt3_base_src \
  --devices 0 \
  --seed 21 \
  --epochs 1 \
  --short_title base_src \
  --use_synthetic
```

1. A new folder is created with suffix `_reproduce` (e.g., `20251028_101522_gainakt3_base_src_reproduce`).
2. Generated artifacts include `reproduction_report.json` and optionally `reproduction_manifest.json` (if `--manifest`).
3. Inspect drift values; confirm `within_tolerance=true`.

 
### 10.3 `reproduction_report.json` Schema

```json
{
  "source_experiment": "20251028_093235_gainakt3_base_src",
  "reproduction_experiment": "20251028_101522_gainakt3_base_src_reproduce",
  "config_md5_match": false,
  "source_config_md5": "bd94b21338e2a6a80f30ebcf5aea1d7a",
  "reproduction_config_md5": "c3e1b5f0d17c4fa6a5d1bc934e9f72ab",
  "auc_source": 0.7421,
  "auc_reproduction": 0.7421,
  "auc_diff": 0.0,
  "mastery_corr_source": 0.5632,
  "mastery_corr_reproduction": 0.5632,
  "mastery_corr_diff": 0.0,
  "gain_corr_source": 0.5874,
  "gain_corr_reproduction": 0.5874,
  "gain_corr_diff": 0.0,
  "within_tolerance": true,
  "tolerance": {"auc": 0.002, "corr": 0.01},
  "amp_enabled": true
}

```
Values above are illustrative; actual numbers depend on dataset and seed.

 
### 10.4 Interpretation

- `config_md5_match=false` with `within_tolerance=true` indicates purely cosmetic or non-semantic divergence (e.g., different `short_title`).
- If metrics drift exceeds tolerance thresholds investigate: random seed mismatch, AMP instability, artifact hash differences, constraint warmup variations.
- Use `--strict_schema` to guard against unnoticed launcher structural changes between source and reproduction runs.

 
### 10.5 Reporting Reproductions

When citing a reproduced experiment include:

```text
Original: 20251028_093235_gainakt3_base_src (config_md5=bd94b2...d7a)
Reproduction: 20251028_101522_gainakt3_base_src_reproduce (config_md5=c3e1b5...72ab) within_tolerance (AUC diff 0.0000, mastery corr diff 0.0000, gain corr diff 0.0000)
```

 
### 10.6 Manifest File (optional)

`reproduction_manifest.json` extends the report with raw artifact hashes and environment diffs enabling external auditing. Include when preparing supplementary material for publication.

 
### 10.7 Future Normalized Hash

We plan `normalized_config_md5` (excluding `short_title`, timestamp, non-semantic toggles) to distinguish semantic vs cosmetic changes; until then treat `config_md5` as strict.

Schema version will increment upon structural changes; reproduction claims must reference the schema version.

---
 
## 11. Summary

GainAKT3 launcher provides complete parameter serialization, artifact hashing, and multi-seed stability aggregation. Through controlled overrides and documented defaults we preserve experimental integrity while allowing interpretability-driven explorations. All published results must cite folder name and `config_md5` and validate `missing_params` is empty to meet reproducibility standards.

---

## 12. Newly Added Parameters & Diagnostics (Extended Instrumentation)

We integrated additional parameters and instrumentation to strengthen robustness, interpretability auditing, and configuration transparency. This section documents each new element, its rationale, and usage conventions.

### 12.1 Automatic Concept Cardinality Resolution

| Parameter | Type | Default | Purpose | Notes |
|-----------|------|---------|---------|-------|
| `--auto_num_c` | flag (store_true) | off | Expands `num_c` automatically to `max_concept_id + 1` if dataset contains unseen higher concept ids. | Prevents silent out-of-range embedding indices; recommended when experimenting with new or merged datasets. |
| `--report_concept_stats` | flag (store_true) | off | Prints dataset-wide concept statistics (min, max, unique count) prior to model instantiation. | Logs resolved vs requested concept cardinality; aids reproducibility reviews. |

Concept scan traverses both training and validation loaders before allocating embeddings. The launcher serializes both requested and resolved cardinalities: `data.num_c_requested`, `data.num_c_resolved`, plus `concept_min_id`, `concept_max_id`, and `concept_unique_count` in `config.json` to ensure exact reconstruction conditions.

### 12.2 Extended Interpretability Constraint Set

Beyond previously documented weights we expose additional structural penalties:

| Parameter | Type | Default | Functional Role |
|-----------|------|---------|-----------------|
| `difficulty_ordering_weight` | float | 0.0 | Encourages predictions to respect relative difficulty ordering during final step pairwise comparisons. |
| `drift_smoothness_weight` | float | 0.0 | Penalizes high second derivative (temporal curvature) in difficulty logits to promote smooth drift. |

These augment alignment, consistency, retention, lag gain, sparsity, and peer alignment to form a comprehensive interpretability regularization suite. When introducing new penalties we maintain a conservative warm-up (`warmup_constraint_epochs`) to stabilize baseline convergence before applying structure.

### 12.3 Peer & Difficulty Artifact Hashing

`peer_index_sha256` and `difficulty_table_sha256` are serialized after computing SHA256 digests of contextual artifact files. A missing artifact activates `cold_start` mode which is recorded explicitly (`artifacts.cold_start = true|false`). This prevents ambiguity in early-phase interpretability metrics influenced by absent context priors.

### 12.4 GPU Visibility Sanitization

We now sanitize GPU selection to avoid internal CUDA assertions when requested devices exceed physical availability:

| Behavior | Description |
|----------|-------------|
| Device pruning | Requested indices not present are dropped; a warning is emitted with the invalid subset. |
| Default selection | If no devices provided and GPUs are available, we expose up to five (project policy: ~60% utilization). |
| CPU fallback | If no CUDA device is detected we disable multi-GPU (`multi_gpu=false`) and unset `CUDA_VISIBLE_DEVICES`. |

The final device list is reflected under `hardware.devices` and the effective environment string captured as `context.hardware.visible_devices_env`. This enables post-hoc verification of multi-GPU assumptions and prevents misinterpretation of performance scaling claims.

### 12.5 Boolean Output Normalization for Parallelism

To ensure compatibility with `DataParallel` gathering semantics we convert raw Python boolean outputs in the model forward pass (e.g., `cold_start`) to CUDA tensors. This avoids scatter/gather TypeErrors (`'bool' object is not iterable`) and preserves semantic content for interpretability summaries. No metric meaning is altered; representation change is purely structural.

### 12.6 Configuration Flattening Dump

Each experiment now includes a deterministic flattened parameter listing (`config_flat.txt`) alongside `config.json`. It enumerates all nested keys as dot-path entries (`section.subsection.key = value`) enabling rapid diffing and external auditing without reparsing hierarchical JSON. A bracketed block `[CONFIG-DUMP]` is emitted in `stdout.log` to allow streaming capture.

Rationale:

1. Guards against partial serialization regressions (quickly reveals missing subtree keys).  
2. Facilitates reproducibility peer review by enabling trivial textual comparison across experiments.  
3. Establishes a stable interface for automated provenance checkers (e.g., building a `normalized_config_md5` in future work).

### 12.7 Safety & Diagnostic Environment Flags

| Env Var | Default | Purpose |
|---------|---------|---------|
| `GAINAKT3_INDEX_DEBUG` | `1` (enabled) | Activates one-time index range summary (concept id, interaction token extrema, embedding cardinalities) to pre-empt opaque device-side index asserts. Set to `0` to silence. |

### 12.8 Recommended Usage Patterns for New Diagnostics

| Scenario | Recommended Flags |
|----------|-------------------|
| Unknown dataset concept range | `--report_concept_stats --auto_num_c` |
| First run after dataset augmentation | `--report_concept_stats` (retain original `num_c` to compare) |
| Investigating index OOB errors | Ensure `GAINAKT3_INDEX_DEBUG=1` (default) |
| Artifact integrity audit | Inspect `artifacts.*_sha256` and `cold_start` entries in `config_flat.txt` |

### 12.9 Interaction with Reproduction Mode

Reproduction runs inherit flattened configuration generation and concept statistics; any divergence in resolved concept cardinality or artifact hashes is immediately visible via dot-path diff. A reproduction claiming equivalence SHOULD have identical lines for all non-cosmetic keys (pending future normalized hash implementation removing `short_title` and timestamp).

### 12.10 Parameter Change Impact Notes

| Change | Primary Impact | Secondary Considerations |
|--------|----------------|--------------------------|
| Increase `alignment_weight` | Reduces negative mastery deltas; may slow convergence | Consider raising warmup epochs to mitigate early constraint dominance. |
| Enable `auto_num_c` | Mitigates embedding OOB; may increase memory footprint | Document resulting `num_c_resolved` for paper tables. |
| Add `drift_smoothness_weight` > 0 | Smoother difficulty trajectory | Monitor reconstruction error; excessive smoothing may mask genuine concept transitions. |
| Raise `peer_alignment_weight` | Stronger conformity to peer priors | Ensure artifact hash stability; drift due to changing peer index invalidates comparisons. |
| High `sparsity_weight` | Promotes gain sparsity | Track gain sparsity index; may depress correlation metrics if over-regularized. |

---

## 13. Audit Checklist Extension

Augment the existing reproducibility checklist with:

| Item | Status Criterion |
|------|------------------|
| Concept stats recorded | `concept_min_id`, `concept_max_id`, `concept_unique_count` present in `config.json`. |
| Flattened config present | `config_flat.txt` exists and `[CONFIG-DUMP]` block appears in `stdout.log`. |
| Artifact hashes present | `peer_index_sha256` and `difficulty_table_sha256` not `MISSING` (unless intentional cold start). |
| Resolved vs requested `num_c` | Both `num_c_requested` and `num_c_resolved` serialized when auto expansion invoked. |
| Index debug summary | `[GainAKT3][INDEX-SUMMARY]` appears (or explicitly disabled). |
| Boolean normalization | No DataParallel gather TypeError encountered. |

All items above must be satisfied for an experiment to be eligible for inclusion in comparative tables or publication appendices.

---
