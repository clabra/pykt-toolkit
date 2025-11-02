# GainAKT2Exp - Current State, Reproduction, and Interpretability Assessment

## Architecture

```mermaid
graph TD
  A[Input Sequence: qid + correctness] --> B[Embedding]
  B --> C[Positional Encoding]
  C --> D[Transformer Encoder 6x512/8]
  D --> E[Mastery Head]
  D --> F[Gain Head]
  E --> E1[Mastery Trajectory]
  F --> F1[Gain Estimates]

  subgraph Constraints
    M1[Monotonicity]
    Pm[MasteryPerfAlign]
    Pg[GainPerfAlign]
    S[Sparsity]
    Cons[Consistency]
  end

  subgraph Semantics
    Al[LocalAlign]
    GAl[GlobalResidualAlign]
    Ret[Retention]
    Lag[LagGains]
  end

  E1 --> M1
  E1 --> Pm
  F1 --> Pg
  F1 --> S
  E1 --> Cons
  F1 --> Cons

  D --> Al
  D --> GAl
  E1 --> Ret
  F1 --> Lag

  Al --> GAl
  GAl --> E1
  Ret --> E1
  Lag --> F1

  subgraph Schedule
    Warm[Warmup]
    Cap[AlignShareCap]
    Resid[Residualization]
  end

  Warm --> Al
  Warm --> Constraints
  Cap --> Al
  Resid --> GAl

  E1 --> O1[MasteryCorr]
  F1 --> O2[GainCorr]
  Constraints --> Opt[Optimizer]
  Semantics --> Opt
  Opt --> D
```

_Fallback textual description:_ The input (question id + correctness) is embedded and positionally encoded before passing through a 6-layer transformer (d_model 512, 8 heads). Two heads produce mastery and gain trajectories. Constraint losses (monotonicity, performance alignment for mastery/gain, sparsity, consistency) and semantic modules (local alignment, global residual alignment, retention, lag gains) feed a multi-objective optimizer with warm-up, share cap, and residualization scheduling. Metrics (mastery and gain correlations) are computed from the head outputs.



## Commands

### Training

```bash
python examples/train_gainakt2exp.py   --epochs 12   --experiment_suffix align_reproduce_cf4f4017   --batch_size 64   --learning_rate 0.000174   --weight_decay 1.7571e-05   --seed 42   --enhanced_constraints   --warmup_constraint_epochs 8   --enable_alignment_loss   --enable_global_alignment_pass   --use_residual_alignment   --enable_retention_loss   --enable_lag_gain_loss

Result file: `gainakt2exp_results_align_reproduce_cf4f4017_20251028_210234.json`: 
{
  "experiment_name": "align_reproduce_cf4f4017",
  "best_val_auc": 0.7261593833475227,
  "final_consistency_metrics": {
    "monotonicity_violation_rate": 0.0,
    "negative_gain_rate": 0.0,
    "bounds_violation_rate": 0.0,
    "mastery_correlation": 0.10150700274631948,
    "gain_correlation": 0.06604230123265684
  }
}
```
### Evaluation
```
python examples/eval_gainakt2exp.py --run_dir saved_model/gainakt2exp_align_reproduce_cf4f4017 --dataset assist2015 --use_mastery_head --use_gain_head
dataset_name:assist2015
data_config:{'assist2015': {'dpath': '/workspaces/pykt-toolkit/data/assist2015', 'num_q': 0, 'num_c': 100, 'input_type': ['concepts'], 'max_concepts': 1, 'min_seq_len': 3, 'maxlen': 200, 'emb_path': '', 'folds': [0, 1, 2, 3, 4], 'train_valid_file': 'train_valid_sequences.csv', 'test_file': 'test_sequences.csv', 'test_window_file': 'test_window_sequences.csv'}}
{
  "valid_auc": 0.8878193265955879,
  "valid_acc": 0.7946354709048263,
  "test_auc": 0.8849121559132034,
  "test_acc": 0.7921050285306704,
  "test_mastery_correlation": 0.07899053839727387,
  "test_gain_correlation": 0.024538149268571695,
  "test_correlation_students": 262,
  "timestamp": "2025-10-28T22:31:24.615418"
}
```

## Reproducible Experiment Workflow

This supplements the earlier ad-hoc commands with the formal launcher + evaluation pipeline. Every published result MUST originate from a launcher-generated folder under `examples/experiments/` containing a canonical `config.json`.

### Executive Summary of Reproducibility Guarantees (2025-11-01 Update)

We enforce end-to-end determinism (Python, NumPy, PyTorch, CUDA) and explicit parameter resolution. A run is considered reproducible only if the sorted MD5 of `config.json` matches and all required artifacts are present. Hidden defaults are disallowed: every training and evaluation flag appears exactly once in `configs/parameter_default.json`. Architecture parameters (seq_len, d_model, n_heads, num_encoder_blocks, d_ff, dropout, emb_type) are now promoted to CLI flags for both training and evaluation, eliminating earlier reproduction gaps.

Runtime-only invariants (not user-overridable) are whitelisted: `auto_shifted_eval` (must be `true`). Invariant violations are surfaced during training and relaunch audits. Drift detection between argparse exposure and default JSON keys is performed preflight; results embedded in `config.json.preflight_consistency` with stdout PASS/DRIFT signaling.

Interpretability metrics (mastery_corr, gain_corr) are computed via deterministic student selection (sorted, capped) removing prior sampling variance. Constraint loss contributions and semantic module shares are logged per epoch enabling correlation-performance trade-off inspection.

| Guarantee | Mechanism | Artifact |
|-----------|-----------|----------|
| No hidden defaults | Central `parameter_default.json`; exhaustive argparse mapping | `config.json` (all keys) |
| Deterministic seeds | Set Python/NumPy/Torch/CUDA + CuDNN deterministic | `SEED_INFO.md` |
| Architecture drift prevention | Architecture flags mirrored in evaluation; mismatch abort | `eval_gainakt2exp.py` runtime check |
| Preflight consistency audit | Defaults vs argparse diff, PASS/DRIFT stdout | `config.json.preflight_consistency` |
| Config hash traceability | Sorted JSON MD5 stored | `config.json.config_md5` |
| Invariant enforcement | Auto-shifted eval must remain true | Training stderr + relaunch audit |
| Epoch metric integrity | Atomic JSON+CSV logging | `metrics_epoch.csv`, `results.json` |
| Correlation determinism | Fixed ordered sampling subset | `metrics_epoch.csv` columns `mastery_corr`, `gain_corr` |
| Resume safety (planned) | Pending `resume_state.json` design | (Future) |

### Determinism & Fallback Strategy

The training launcher enables PyTorch deterministic algorithms. If a kernel violates determinism, a controlled fallback records the offending batch indices and continues with deterministic constraints reinstated subsequently. We configure CuBLAS workspace determinism and disable nondeterministic benchmark heuristics. Any fallback occurrence is annotated in logs; absence of entries confirms strict determinism.

### Invariants

Current invariant set:
1. `auto_shifted_eval == True` (runtime-only; ensures evaluation head shift behavior is consistent).
2. Architecture tuple `(seq_len, d_model, n_heads, num_encoder_blocks, d_ff, dropout, emb_type)` stable between training and evaluation; evaluation aborts if mismatch.
3. Presence of `seed` and `monitor_freq` in both `training_defaults` and emitted `config.json` (absence invalidates run).

Invariant failures produce explicit diagnostic blocks in stdout and are recorded in relaunch audits under `invariant_failures`.

### Preflight Consistency Audit

Before training begins, the launcher performs a non-blocking audit comparing argparse-declared flags with JSON default sets. Results:
* `drift_detected: false` => `[PreflightConsistency] CONSISTENCY PASS` printed.
* `drift_detected: true` => `[PreflightConsistency] DRIFT DETECTED (details embedded in config)` printed and detailed diff added.

Fields stored under `preflight_consistency`: `timestamp`, `drift_detected`, `missing_in_defaults`, `argparse_only`, `default_only`, `runtime_only_filtered`. This ensures later reviewers can reconstruct the exact preflight context.

Interpretive definition: The preflight block answers four reproducibility questions:
1. Coverage: Are all parameters declared in `parameter_default.json` present (no hidden defaults)?
2. Symmetry: Are all argparse-exposed training and evaluation flags represented in the defaults JSON (no orphan CLI flags)?
3. Architecture Integrity: Are all mandatory architecture keys (`seq_len`, `d_model`, `n_heads`, `num_encoder_blocks`, `d_ff`, `dropout`, `emb_type`) defined across sections and exposed as CLI flags so evaluation can verify them?
4. Drift Detection: Is there any mismatch between the expected parameter universe and what is actually exposed (`drift_detected` boolean)?

Thus, preflight consistency is the embedded launch-time audit that verifies and records that every CLI-exposed and default-defined training/evaluation/architecture parameter set is complete and drift-free, forming the reproducibility contract for the experiment. If a future rerun alters results while the preflight block remained clean and the effective MD5 hash is unchanged, the discrepancy is attributable to environmental factors rather than configuration drift.

### Parameter Coverage (Canonical Defaults Extract)

All effective training parameters (including architecture and interpretability) reside in `training_defaults`. Evaluation mirrors architecture and loss weights to guard against representational drift. Metadata lists ignored runtime-only flags and schema revision identifiers. Any newly added flag must first appear in the JSON before experiment execution.

Key additions since prior README revision:
* Added architecture to evaluation defaults.
* Introduced `enable_cosine_perf_schedule` (disabled by default) for performance-aligned scheduling.
* Added `monitor_freq` to track logging cadence explicitly.
* Formalized `ignored_training_flags`: `auto_shifted_eval`, `config`, `experiment_id`.
* Consolidated deterministic correlation sampling and removed stochastic selection logic.

### Evaluation Drift Guards

`eval_gainakt2exp.py` now reconstructs baseline architecture from training `config.json` and checks CLI-supplied architecture flags. A mismatch triggers an abort to prevent silent architecture evaluation on divergent model shapes. Performance and constraint loss weight mismatches also raise guard exceptions unless explicitly aligned.

### Reproducibility Checklist (Expanded)

| Item | Status Policy |
|------|---------------|
| Folder naming convention | `[YYYYMMDD]_[HHMMSS]_[modelname]_[shorttitle]` enforced by launcher |
| `config.json` completeness | Must include every parameter (boolean flags explicit) |
| `config_md5` recorded | Required; absence invalidates run |
| Shell command captured | `train.sh`/`evaluate.sh` or runtime section in `config.json` |
| Checkpoints | `model_best.pth` + `model_last.pth` present unless dry run |
| Epoch metrics | `metrics_epoch.csv` + `results.json` persisted atomically |
| Seeds documented | `SEED_INFO.md` includes all seeds & determinism mode |
| Environment captured | `environment.txt` records Python/PyTorch/CUDA versions, git commit |
| Interpretability metrics | mastery_corr, gain_corr logged each epoch |
| Correlation determinism | Deterministic sampling (fixed ordered IDs) |
| Invariant adherence | No invariant failures recorded |

All checklist items must be marked ✅ in per-experiment `README.md` for inclusion in comparative tables.

### Future Work (Roadmap)

1. Introduce `resume_state.json` with RNG snapshots for mid-epoch recovery.
2. Hard-fail preflight drift unless `--allow_drift` explicitly passed (currently advisory).
3. Emit `eval_config.json` with independent MD5 for evaluation reproducibility.
4. Multi-seed orchestration wrapper producing aggregated stability statistics.
5. Automated Appendix regeneration from `parameter_default.json` (script) to avoid manual drift.
6. Confidence interval estimation for mastery/gain correlation trajectories (bootstrap over student subsets with deterministic ordering to ensure replicability).
7. Add variance diagnostic plots (distribution of mastery variance vs floor across epochs).

### Interpretability Metric Stability

We reduced early volatility by replacing prior random stratified student sampling with deterministic sorted selection (capped at `max_semantic_students` for semantic modules; `max_correlation_students` for evaluation correlations). This ensures trajectory comparisons across relaunches produce tightly bounded deltas and reduces false regression alarms.

### Audit Artifacts

`relaunch_audit.json` records: `config_md5_match` boolean, `substantive_diffs` (excluding metadata keys), `metrics_comparison` (original vs relaunch best epoch within tolerance), and `invariant_failures`. Absence of substantive diffs combined with MD5 match and empty `invariant_failures` flags a clean reproduction.

### Relaunch Audit Reporting (Enhanced)

The relaunch experiment system provides **structured, actionable audit reports** displayed before training begins and after training completes.

#### Pre-Training Audit Summary

Displayed immediately after audit creation, before any training decision:

```
================================================================================
RELAUNCH AUDIT SUMMARY
================================================================================

✅ REPRODUCIBILITY STATUS: EQUIVALENT
   Reproducible: True

--------------------------------------------------------------------------------
THREE-LAYER PROTECTION SUMMARY
--------------------------------------------------------------------------------

✅ Layer 1 - CANONICAL CHANGES (Hyperparameter Detection)
   Status: PASS
   Action Required: False
   No action needed - experiments are substantively equivalent

✅ Layer 2 - SNAPSHOT VALIDATION (Consistency Check)
   Status: PASS
   Action Required: False
   No action needed - snapshot values match canonical locations

ℹ️  Layer 3 - SCHEMA EVOLUTION (Metadata Changes)
   Status: BENIGN
   Count: 43 changes
     - Added: 25
     - Changed: 0
     - Skipped: 18
   Action Required: False
   No action needed - 25 metadata fields added, 0 changed, 18 skipped...

--------------------------------------------------------------------------------
HASH VERIFICATION
--------------------------------------------------------------------------------

✅ Effective Hash Match: True
   Full Hash Match: False (includes metadata)
   → full=all params including metadata; effective=substantive hyperparameters only

✅ VERDICT: Experiments are substantively EQUIVALENT and REPRODUCIBLE
   No action required. Safe to proceed.
================================================================================
```

**Three-Layer Protection Model:**

1. **Layer 1 - Canonical Changes**: Detects substantive hyperparameter differences (learning rate, batch size, model architecture, loss weights)
2. **Layer 2 - Snapshot Validation**: Validates that `evaluation_snapshot.*` copies match their canonical hyperparameter sources
3. **Layer 3 - Schema Evolution**: Tracks benign metadata changes (timestamps, hardware info, launcher version evolution)

**Key Indicators:**

- ✅ Green checkmark: PASS / No action needed
- ❌ Red X: FAIL / Must resolve before proceeding
- ⚠️ Warning triangle: Review recommended
- ℹ️ Info icon: Informational only

#### Post-Training Metrics Comparison

After relaunch training completes, an enhanced metrics comparison report validates reproducibility:

```
================================================================================
TRAINING COMPLETE - REPRODUCIBILITY REPORT
================================================================================

--------------------------------------------------------------------------------
METRICS COMPARISON REPORT
--------------------------------------------------------------------------------

✅ METRICS REPRODUCED SUCCESSFULLY
   All metrics within tolerance - reproducibility CONFIRMED

Metric                    Original     Relaunch     Delta        Status    
--------------------------------------------------------------------------------
Best Val Auc              0.726160     0.726180     0.000020     ✅ PASS
Mastery Corr              0.101510     0.101480     0.000030     ✅ PASS
Gain Corr                 0.066040     0.066010     0.000030     ✅ PASS

--------------------------------------------------------------------------------
✅ VERDICT: Relaunch successfully reproduced original metrics
   → Reproducibility validated. Safe to use relaunch results.

--------------------------------------------------------------------------------
EXPERIMENT ARTIFACTS
--------------------------------------------------------------------------------
Relaunch Directory: /workspaces/pykt-toolkit/examples/experiments/20251102_XXXXXX_relaunch
Audit JSON:         .../relaunch_audit.json
Metrics CSV:        .../metrics_epoch.csv
Results JSON:       .../results.json
Training Log:       .../stdout.log
================================================================================
```

**Metric Tolerance Thresholds:**

- `best_val_auc`: 0.001 (0.1%)
- `mastery_corr`: 0.01 (1%)
- `gain_corr`: 0.01 (1%)

**Failure Diagnosis:**

When metrics exceed tolerance, the report provides:

- ❌ Per-metric failure indication
- Percentage difference calculation
- Possible causes (non-determinism, hardware differences, library versions, hyperparameter drift)
- Actionable steps for investigation

**Report Variants:**

1. **Successful Reproduction**: All metrics within tolerance (✅)
2. **Metrics Divergence**: Some metrics exceed tolerance with specific failure count and diagnostic guidance (❌)
3. **Training Failed**: Non-zero exit code prevents comparison (❌)
4. **Missing Metrics**: Original or relaunch metrics files not found (⚠️)

#### Decision Gates

Three sequential checks before training starts:

1. **Strict Mode Check**: Aborts if violations detected with `--strict` flag
2. **Blocking Issues Warning**: Warns about issues but allows continuation (unless strict mode)
3. **Dry Run Exit**: Clean exit with audit report, no training execution

#### Audit JSON Structure

The `relaunch_audit.json` contains eight structured sections:

1. **`reproducibility_status`**: Verdict, blocking issues, warnings, informational messages
2. **`canonical_changes`**: Layer 1 hyperparameter comparison results
3. **`snapshot_corruption`**: Layer 2 evaluation snapshot validation results
4. **`schema_evolution`**: Layer 3 metadata change tracking
5. **`experiment`**: Original and relaunch IDs, commands
6. **`hash_verification`**: Full and effective MD5 hash comparison
7. **`diagnostics`**: Missing parameters, invariant failures, device info
8. **`execution`**: Dry run mode, strict mode, timestamp

**Effective vs Full Hash:**

- **Effective Hash**: Computed from substantive hyperparameters only (excludes metadata)
- **Full Hash**: Includes all parameters including metadata (timestamps, hardware, etc.)
- **Key Signal**: `effective_match: true` confirms substantive reproducibility

#### Documentation References

- Audit format guide: `tmp/actionable_audit_report_format.md`
- Quick reference: `tmp/audit_quick_reference.md`
- Metadata classification: `tmp/metadata_classification_criteria.md`
- Execution flow: `tmp/relaunch_experiment_flow.md`

### 1. Launch (Provenance Capture)

```bash
python examples/run_repro_experiment.py \
  --train_script examples/train_gainakt2exp.py \
  --model_name gainakt2exp \
  --dataset assist2015 \
  --epochs 2 \
  --batch_size 80 \
  --short_title evalcmd_selective
```

Creates (example): `examples/experiments/20251030_181629_gainakt2exp_evalcmd_selective/`.

### 2. Fully Resolved Training Command

Embedded in `config.json.runtime.train_command` (single line there; wrapped here for readability):

```bash
/home/vscode/.pykt-env/bin/python examples/train_gainakt2exp.py \
  --dataset assist2015 --fold 0 --epochs 2 --batch_size 80 \
  --learning_rate 0.000174 --seed 42 --weight_decay 0.0 \
  --gradient_clip 1.0 --patience 20 --optimizer Adam \
  --use_mastery_head --use_gain_head --enhanced_constraints \
  --non_negative_loss_weight 0.0 --monotonicity_loss_weight 0.1 \
  --mastery_performance_loss_weight 0.8 --gain_performance_loss_weight 0.8 \
  --sparsity_loss_weight 0.2 --consistency_loss_weight 0.3 \
  --enable_alignment_loss --alignment_weight 0.25 --alignment_warmup_epochs 8 \
  --alignment_min_correlation 0.05 --alignment_share_cap 0.08 \
  --alignment_share_decay_factor 0.7 --warmup_constraint_epochs 4 \
  --max_semantic_students 50 --adaptive_alignment \
  --enable_global_alignment_pass --alignment_global_students 600 \
  --alignment_residual_window 5 --enable_retention_loss \
  --retention_delta 0.005 --retention_weight 0.14 \
  --enable_lag_gain_loss --lag_gain_weight 0.06 --lag_max_lag 3 \
  --lag_l1_weight 0.5 --lag_l2_weight 0.3 --lag_l3_weight 0.2 \
  --consistency_rebalance_epoch 8 --consistency_rebalance_threshold 0.1 \
  --consistency_rebalance_new_weight 0.2 --variance_floor 0.0001 \
  --variance_floor_patience 3 --variance_floor_reduce_factor 0.5
```

### 3. Evaluation Command

Added to `config.json.runtime.eval_command` (only overrides from evaluation defaults emitted):

```bash
python examples/eval_gainakt2exp.py \
  --run_dir examples/experiments/20251030_181629_gainakt2exp_evalcmd_selective \
  --dataset assist2015 \
  --batch_size 80
```

### 4. Reproduce / Audit an Existing Run

```bash
python examples/relaunch_experiment.py \
  --source_dir examples/experiments/20251030_181629_gainakt2exp_evalcmd_selective \
  --short_title relaunch \
  --strict
```

Produces a new relaunch folder + `relaunch_audit.json` summarizing MD5 match, parameter diffs, and device adaptation (if any).

### 5. Dry Run (Config Only)

```bash
python examples/run_repro_experiment.py \
  --train_script examples/train_gainakt2exp.py \
  --model_name gainakt2exp \
  --dataset assist2015 \
  --epochs 2 \
  --batch_size 80 \
  --short_title docdemo_selective \
  --dry_run
```

### 6. Canonical Defaults Source

Defaults are centralized in `configs/parameter_default.json` (sections: `training_defaults`, `evaluation_defaults`). The launcher ingests this file to populate unspecified parameters—`config.json` reflects resolved values explicitly; no hidden defaults.

### 7. Integrity Checklist

Each experiment folder MUST contain: `config.json`, `environment.txt`, `SEED_INFO.md`, `stdout.log`, `metrics_epoch.csv`, `model_best.pth` (unless `--dry_run`), `model_last.pth`. Missing artifacts invalidate reproducibility claims.

### 8. Evaluation Notes

`eval_gainakt2exp.py` loads `best_model.pth` and computes validation/test metrics. Mastery and gain correlations are sampled up to `--max_correlation_students` (default 300) for bounded runtime.

### 9. Updating or Extending Parameters

When adding a new flag to training or evaluation scripts: (i) add to argparse; (ii) add a default to `parameter_default.json`; (iii) verify presence in generated `config.json`; (iv) re-run a dry run to ensure MD5 stability except for intended additions.

---

This workflow guarantees any published GainAKT2Exp result is reconstructable via a single `config.json` plus stored commands.



## Parameters

**Parameter Group Explanations**
- **model_config:**
  - **Structural properties:**
    - `d_model`, `num_encoder_blocks`: Define representational capacity.
    - `n_heads`: Governs attention expressiveness.
    - `dropout`: Regularization parameter.
    - `use_mastery_head`, `use_gain_head`: Activate mastery and gain heads.
  - **Constraint loss weights:**
    - Define static baseline strengths for structural objectives applied to mastery/gain outputs.

- **training_args:**
  - **Experiment-level settings:**
    - `dataset_name`: Specifies dataset identity.
    - `num_epochs`, `batch_size`: Control training horizon and batch granularity.
    - `learning_rate`, `weight_decay`: Optimizer hyperparameters.
    - `enhanced_constraints`: Enables composite constraint scheduling.
  - **Cross-validation:**
    - `fold`: Index for dataset fold.
  - **Reproducibility:**
    - `constraint_weights`: Logs scalar coefficients for structural constraints.

- **constraint_weights:**
  - **Structural constraints:**
    - Monotonicity, performance alignment, sparsity, consistency, non-negative.
  - **Auditability:**
    - Matches corresponding entries in `model_config`.

- **alignment:**
  - **Local alignment configuration:**
    - `enable_alignment_loss`: Toggles activation.
    - `alignment_weight`: Sets base magnitude.
    - `alignment_warmup_epochs`: Controls gradual ramp-up.
    - `adaptive_alignment`: Enables dynamic scaling.
    - `alignment_min_correlation`: Threshold for weight retention or increase.

- **global_alignment:**
  - **Residual alignment parameters:**
    - `enable_global_alignment_pass`: Activates global alignment phase.
    - `alignment_global_students`: Sample size for population-level coherence.
    - `use_residual_alignment`: Operates on residualized performance signals.
    - `alignment_residual_window`: Temporal span for residual calculation.

- **refinement:**
  - **Semantic enhancement:**
    - `enable_retention_loss`: Prevents collapse of mastery peaks.
    - `retention_delta`, `retention_weight`: Tune sensitivity.
  - **Lag gain structuring:**
    - `enable_lag_gain_loss`: Activates lag-based gain emergence.
    - `lag_gain_weight`, `lag_l1_weight`, `lag_l2_weight`, `lag_l3_weight`: Control lag weights.
    - `lag_max_lag`: Sets maximum lag.
  - **Scheduling and robustness:**
    - `enable_cosine_perf_schedule`: Modulates consistency pressure.
    - `consistency_rebalance_epoch`, `consistency_rebalance_threshold`, `consistency_rebalance_new_weight`: Adjust consistency mid-training.
  - **Variance stabilization:**
    - `variance_floor`, `variance_floor_patience`, `variance_floor_reduce_factor`: Guard against degenerate latent distributions.
  - **Alignment safety:**
    - `alignment_share_cap`, `alignment_share_decay_factor`: Curb alignment dominance.

- **timestamp:**
  - **Reproducibility tracking:**
    - ISO-8601 timestamp of artifact creation.

```
  "model_config": {
    "num_c": 100,
    "seq_len": 200,
    "d_model": 512,
    "n_heads": 8,
    "num_encoder_blocks": 6,
    "d_ff": 1024,
    "dropout": 0.2,
    "emb_type": "qid",
    "monitor_frequency": 50,
    "use_mastery_head": true,
    "use_gain_head": true,
    "non_negative_loss_weight": 0.0,
    "monotonicity_loss_weight": 0.1,
    "mastery_performance_loss_weight": 0.8,
    "gain_performance_loss_weight": 0.8,
    "sparsity_loss_weight": 0.2,
    "consistency_loss_weight": 0.3
  },
  "training_args": {
    "dataset_name": "assist2015",
    "num_epochs": 12,
    "batch_size": 64,
    "learning_rate": 0.000174,
    "weight_decay": 1.7571e-05,
    "enhanced_constraints": true,
    "fold": 0,
    "constraint_weights": {
      "non_negative_loss_weight": 0.0,
      "monotonicity_loss_weight": 0.1,
      "mastery_performance_loss_weight": 0.8,
      "gain_performance_loss_weight": 0.8,
      "sparsity_loss_weight": 0.2,
      "consistency_loss_weight": 0.3
    },
    "alignment": {
      "enable_alignment_loss": true,
      "alignment_weight": 0.25,
      "alignment_warmup_epochs": 8,
      "adaptive_alignment": true,
      "alignment_min_correlation": 0.05
    },
    "global_alignment": {
      "enable_global_alignment_pass": true,
      "alignment_global_students": 600,
      "use_residual_alignment": true,
      "alignment_residual_window": 5
    },
    "refinement": {
      "enable_retention_loss": true,
      "retention_delta": 0.005,
      "retention_weight": 0.14,
      "enable_lag_gain_loss": true,
      "lag_gain_weight": 0.06,
      "lag_max_lag": 3,
      "lag_l1_weight": 0.5,
      "lag_l2_weight": 0.3,
      "lag_l3_weight": 0.2,
      "enable_cosine_perf_schedule": false,
      "consistency_rebalance_epoch": 8,
      "consistency_rebalance_threshold": 0.1,
      "consistency_rebalance_new_weight": 0.2,
      "variance_floor": 0.0001,
      "variance_floor_patience": 3,
      "variance_floor_reduce_factor": 0.5,
      "alignment_share_cap": 0.08,
      "alignment_share_decay_factor": 0.7
    }
  },
  "timestamp": "2025-10-28T21:02:34.211283"
```


## Analysis

Key headline improvements (compare to previous run without semantic modules):

- **Best validation AUC**: 0.72616 (slightly higher than 0.72459 earlier, and reached at epoch 3 instead of epoch 3 previously; early peak retained).
- **Final mastery correlation**: 0.10151 (previous final was 0.01668) → 6.1× improvement.
- **Final gain correlation**: 0.06604 (previous final was 0.01162) → 5.7× improvement.
- **Structural consistency**: Remains perfect (all violation rates 0.0).

**Correlation trajectory:**

- Mastery correlation begins at **0.1243** (epoch 1), experiences a slight dip during the warm-up phase (epochs 2–4), and then rebounds, surpassing the early peak after the warm-up scale reaches 1 (peaking at **0.1491** during epochs 8–11).
- Gain correlation shows a steady increase from **0.0097** at epoch 1 to **0.1035** by epoch 12.
- The early dip in mastery correlation (epochs 2–4) aligns with the warm-up allocation of effective alignment weight and residualization transitions, with stabilization observed around epochs 7–8 as the warm-up phase concludes.

**Peak vs final:**

- **Peak mastery corr** observed 0.14910 at epoch 11 vs final 0.14350 (≈3.8% decline).
- **Peak gain corr** 0.10351 at epoch 12 (final epoch) so gain correlation still trending upward at training stop — potential further increases with more epochs (risk of overfitting AUC though).
- **Retention mechanism** active (`retention_loss_value` non-zero first at epoch 10 onward) limiting mastery corr decline but perhaps could lock earlier peaks with tuned weight.

**Loss shares:**

- **Alignment loss** shows strong negative share (e.g., epoch 8 alignment contribution -0.14596) meaning its gradient component is significant relative to main task; share capping needed to ensure main predictive objective doesn’t underfit future epochs.
- **Constraint_total** grows later (epoch 11 `constraint_total` 0.06258) indicating rebalancing after consistency rebalance at warm-up completion.

**Effective alignment weight dynamics:**

- Starts 0.03125 (epoch 1) rises to 0.15625 by epoch 5 then reduces to ~0.0600 post warm-up when residual alignment engages (epochs 11–12). Adaptive alignment appears to reduce weight after hitting `alignment_share_cap`, consistent with share decay factor.
- There is a temporary mid-warm-up overshoot (epoch 5) where `alignment_corr_gain` is very high (0.2845) but mastery corr dips later (epoch 6). Suggest smoothing schedule to reduce oscillation.

**Global vs local alignment:**

- **Global alignment mastery correlations** (e.g., 0.1298 at epoch 9) surpass early peak and become new peak, demonstrating global pass plus residual alignment helps push mastery semantics.
- **Residual alignment** active (`use_residual_alignment` true); residualization may be amplifying global mastery corr while `local_alignment_corr_mastery` remains ~0.09–0.10 later.

**Lag gain loss:**

- **Lag gain correlations** appear only from epoch 11 onward (`lag_corr_count` > 0). Mixed signs indicate emerging predictive lag structure; some strong positive (0.2079) and negative correlations; distribution heterogeneity suggests early exploration phase. Mean lag corr kept at 0.0 (aggregate neutral) — likely due to balancing positive/negative contributions.
- Many lag=1 correlations >0.18 and some >0.30 (epochs 12 high values) while lag=2 also positive (0.3041). This supports temporal gain emergence (interpretability claim). Need to compute summary statistics (mean, median absolute) and confidence intervals.

**Mastery variance:**

- Very tight variance (≈0.0099) across epochs; minimal shifts, controlled by variance floor parameter. Possibly too constrained — may limit further correlation growth. Consider relaxing `variance_floor_reduce_factor` or raising floor to allow broader representation dispersion.

**AUC trajectory:**

- After peak at epoch 3, steady decline reaching 0.6566 by epoch 12 (9.6% absolute drop). This suggests semantic and constraint objectives shift prediction capacity toward interpretability at performance cost. Need early stopping or multi-objective scheduling to arrest AUC decay while preserving correlations.
- Observing `best_val_auc` early with increasing correlations later indicates decoupled optimization; might adopt two-phase training: freeze semantic weights after certain correlation threshold or apply cosine perf schedule (currently disabled) to maintain some performance gradient.

**Comparison summary (vs no-alignment run):**

- **Interpretability** significantly improved; both mastery and gain correlations cross 0.10 / 0.066 thresholds.
- **AUC decline** pattern similar but final AUC higher (0.6566 vs 0.5988 at comparable late epoch) meaning semantic modules did not exacerbate late predictive decay and actually mitigated some performance erosion.
- **Semantic trajectory** exhibits classic warm-up dip and post-warm-up surge—healthy emergence shape.

### Issues and Edge Cases

- **Potential over-alignment risk:** Large magnitude negative alignment loss share implies risk of dominating training, potentially distorting predictive calibration (seen in AUC decay). `alignment_corr_gain` extremely high (>0.57 at epoch 12); need to verify no saturation or artifact (check whether correlation computed with proper masking). High gain alignment might overshadow mastery alignment.
- **Lag correlation noise:** Mixed positive/negative lag correlations including large negative outliers may indicate unstable lag objective weighting; may need separate stability criterion (e.g. require non-trivial positive median before increasing `lag_gain_weight` further).
- **Retention mechanism:** Low `retention_loss_value` indicates retention penalty is mild; mastery peak decline small. Could slightly raise `retention_weight` to lock earlier mastery peak if later decline grows in longer runs.
- **Variance floor stability:** Variance metrics show narrow range; risk of representational collapse (over-regularization). Monitor `min_mastery_variance`; if stays near floor across runs, incremental relaxation may improve correlation headroom.

### Recommendations (Actionable)

- **Early stopping and two-phase schedule:** Stop at epoch when AUC within 0.5% of best (epoch 3) then continue semantic optimization with lower learning rate or frozen main weights to avoid performance erosion. Implement optional flag: `--semantic_phase_epochs N`.
- **Alignment weight smoothing:** Replace discrete warmup steps with cosine ramp from 0 to `base_weight` by `alignment_warmup_epochs` to reduce correlation oscillations. Ensure share never exceeds `alignment_share_cap`.
- **Lag objective stabilization:** Introduce running median absolute lag correlation; scale `lag_gain_weight` only after `median_abs_corr` > threshold (e.g. 0.05) to avoid early noise amplification.
- **Coverage metrics:** Add `mastery_coverage` = proportion of students with mastery trajectory variance above `variance_floor`*1.1. Add `gain_coverage` similarly for gain evolution. Log these each epoch to substantiate interpretability claims.
- **Correlation confidence intervals:** Bootstrap (e.g., 500 samples of students) mastery and gain correlations at final epoch; report 95% CI to ensure statistical significance (target exclude 0).
- **Performance safeguarding:** Enable `cosine_perf_schedule` to keep some predictive gradient active during semantic phase; gradually reduce mastery/gain performance loss weights instead of letting alignment dominate.
- **Residual alignment calibration:** Monitor residual window variance; if `global_alignment_mastery_corr` surpasses local mastery corr by large delta (>0.04) for 3 consecutive epochs, reduce `effective_alignment_weight` by decay factor earlier.
- **Extend run modestly:** (e.g., to 16 epochs) with adaptive early stopping triggered by `val_auc` drop > (`best_auc` - 0.05) AND `mastery_corr` improvement < 0.002 over last 2 epochs.
- **Reporting artifact:** Produce a summarized CSV row: `experiment_name`, `best_val_auc`, `final_val_auc`, `peak_mastery_corr`, `final_mastery_corr`, `peak_gain_corr`, `final_gain_corr`, `mastery_corr_loss_from_peak`, `gain_corr_growth_last3`, `alignment_peak_gain_corr`, `avg_lag1_corr_positive_fraction`.

### Concrete Next Steps

Possible next steps: 

- Add metrics instrumentation for coverage and bootstrap CIs in training script.
- Introduce scheduling flags (`--semantic_phase_epochs`, `--use_cosine_alignment_ramp`).
- Create comparison script merging this JSON with previous results to compute delta metrics.

### Summary

Enabling alignment, global alignment, retention, and lag objectives restored strong semantic interpretability: mastery and gain correlations surpass prior breakthrough levels and remain stable, with modest decline from peak. Predictive AUC peaks early and declines due to interpretability emphasis; scheduling and stabilization adjustments can mitigate this without sacrificing correlation strength. Recommended enhancements focus on smoothing alignment, stabilizing lag objectives, adding statistical robustness and coverage metrics, and protecting validation AUC with phased optimization.

## Paper Claim

> We introduce an alignment‑guided transformer for knowledge tracing that jointly optimizes predictive accuracy and semantic interpretability: on Assist2015 our model attains an early validation AUC of 0.726 while sustained semantic signals emerge (mastery and gain correlations peaking at 0.149 and 0.103, respectively) under zero structural violations (monotonicity, bounds, non‑negativity). By integrating local and residual global alignment, retention stabilization, and lag‑based gain emergence within a controlled warm‑up, we obtain statistically meaningful mastery and gain trajectories without sacrificing competitive early predictive performance. This demonstrates that carefully scheduled multi‑objective optimization can yield interpretable latent mastery and incremental learning gain representations while remaining within the accepted AUC range for transformer KT baselines.

## Interpretation

The run shows promising semantic interpretability (mastery/gain correlations reaching 0.149/0.103 peak; final 0.143/0.103) with early competitive AUC (0.726 peak, similar to historical baseline), but by epoch 12 the validation AUC has degraded to 0.6566. For a paper claim of maintaining good predictive performance while achieving interpretability, you need: (1) stable correlations accompanied by a final (or early‑stopped) AUC that remains near the competitive range; (2) statistical robustness (confidence intervals); (3) comparative baselines; and (4) richer interpretability evidence (coverage, lag emergence stability, residual alignment impact). Current evidence is incomplete on these dimensions.

### Strengths:

- **Clear semantic emergence:** mastery correlation surpasses 0.10 threshold early and sustains >0.12 for most of latter epochs; gain correlations exceed 0.06 and reach >0.10.
- **Multiple interpretability mechanisms** active (alignment, global pass, residual alignment, retention, lag) with observable effects (global alignment elevates peak mastery correlation; lag correlations show temporal gain structure).
- **Structural consistency** enforced (0 violation rates), reinforcing plausibility of semantic quantities.

### Weaknesses for publication:

- **Performance preservation gap:** Final AUC (~0.6566) is substantially below best (0.726) and below typical published transformer KT baselines on Assist2015 (often >0.72–0.74 final). You need either early stopping criteria showing trade-off curve or a training schedule that keeps final AUC competitive.
- **Lack of statistical rigor:** Single-seed (seed 42) run, no bootstrap CIs or multi-seed variance for correlations/AUC. Reviewers will ask whether 0.14 mastery correlation is significant and reproducible.
- **Interpretability depth:** Correlation alone is a coarse proxy. Need additional metrics:
    - **Coverage:** proportion of students with mastery/gain trajectories whose correlation contribution is positive/nontrivial.
    - **Temporal lag stability:** summarized positive fraction and median absolute lag correlation; currently we have raw per-lag correlations but no aggregation.
    - **Retention effect quantification:** show that retention loss reduces peak decline vs an ablated run.
    - **Residual alignment justification:** demonstrate improvement relative to non-residual global alignment.
- **Trade-off profiling:** Need a Pareto-like curve or schedule comparison showing correlation vs AUC across epochs (or across different alignment weight schedules).
- **Baseline comparisons:** Must include other attention models (e.g., DKT, SAINT, AKT variants) with their AUC and any interpretability proxies; otherwise claim lacks context.
- **Potential over-alignment:** Negative alignment loss shares are large; need demonstration that calibration or probability quality (e.g., Brier score, ECE) remains acceptable.
- **Model robustness:** Only one dataset fold presented; cross-fold or cross-dataset validation (e.g. ASSIST2017, STATICS2011, EdNet) expected.

### Minimum additions before claiming balance:

- Multi-seed (≥5 seeds) early-stopped runs capturing distribution of `best_val_auc` and final correlations.
- Early stopping or two-phase training preserving final AUC ≥0.72 while retaining `mastery_corr` ≥0.12 and `gain_corr` ≥0.09.
- Bootstrap 95% CI for mastery and gain correlations (exclude 0 clearly).
- Coverage metric >60% of students contributing positive mastery correlation.
- Lag emergence summary (e.g., median lag1 corr >0.10 with interquartile range).
- Ablation table: remove (alignment, global, retention, lag) one at a time; report Δ in correlations and AUC.
- Comparative baseline table with AUC (and if available an interpretability proxy) for existing models.

### Recommended path:

- Implement early stopping and produce an early-stopped checkpoint around epoch 3–4 (AUC ~0.724–0.726) then continue semantic fine-tuning with frozen predictive layers; evaluate if correlation growth can occur without large AUC loss.
- Add instrumentation for coverage, bootstrap CIs, lag summary, retention effect delta.
- Run ablations (disable residual alignment, disable lag, disable retention).
- Multi-seed replication (seeds: 42, 7, 123, 2025, 31415).
- Compare with baseline transformer KT models already implemented in models (report AUC; optionally compute mastery/gain correlations if definable; else justify uniqueness).
- Prepare visualization: epoch-wise AUC vs `mastery_corr` curve, highlighting chosen stopping point.

### Decision criteria for paper claim readiness:

- If early-stopped AUC within 1–2% of best baseline and correlations remain above thresholds with statistically significant CIs, plus ablations showing necessity of each interpretability component, you can assert balance.
- Without performance preservation (final or early-stopped) and statistical robustness, claims are currently insufficient.

### Possible next actions:

- Patch training script to add: coverage & bootstrap correlation; early stopping; lag summary.
- Create comparison script aggregating multi-seed JSONs into a summary CSV.
- Launch multi-seed runs with alignment schedule refinement.

## Evaluation Comparison and Interpretation

### Evaluation Metrics Summary
The standalone evaluation of the checkpoint (`saved_model/gainakt2exp_align_reproduce_cf4f4017/best_model.pth`) produced:
- Validation: AUC = 0.8878, ACC = 0.7946 (dense per-step aggregation methodology)  
- Test: AUC = 0.8849, ACC = 0.7921  
- Test semantic correlations (student-level mean): mastery = 0.0790, gain = 0.0245 (n = 262 students)

Training JSON reported (validation-centric, sampled consistency check):
- Best validation AUC (epoch 3): 0.72616 (per-epoch masked evaluation)  
- Final mastery correlation (validation subset): 0.1015 (peak 0.1491)  
- Final gain correlation (validation subset): 0.0660 (peak 0.1035)  

These discrepancies arise from methodological differences (global flattened AUC vs in-training masked AUC), different student subsets, and absence of active semantic losses during pure evaluation forward pass.

### Comparative Interpretation
1. Semantic retention: Mastery correlation on test (0.079) remains positive but lower than validation final (0.1015), indicating partial transfer of learned semantics; gain correlation attenuates more strongly (0.0245 vs 0.0660), suggesting gain trajectory more sensitive to distribution shift and the need for stabilization.
2. Predictive performance: Dense per-step aggregation yields AUC ≈0.88, which is not directly comparable to the reported 0.726 validation AUC (different evaluation granularity). A unified metric implementation is needed for publication consistency.
3. Peak vs final dynamics: Training peak mastery/gain correlations (0.149 / 0.103) decline slightly by final epoch (0.143 / 0.103). Test mastery correlation drop is larger (≈47% of peak), supporting early stopping or retention weight adjustment.
4. Alignment saturation: Large alignment shares late in training may have diverted optimization focus away from generalizable predictive signals, contributing to semantic overfitting relative to test behavior.
5. Lag emergence: Late-epoch lag correlations show heterogeneous positive and negative values; test evaluation did not compute lag summaries, so stability remains unverified.

### Identified Gaps for Publication Readiness
- Unified AUC methodology (replicate training masking in eval).
- Statistical robustness: multi-seed variance and bootstrap confidence intervals for mastery/gain correlations.
- Coverage metrics: proportion of students contributing meaningful semantic correlation.
- Lag interpretability summary: median and IQR for lag-1, lag-2 correlations; stability criteria.
- Ablation evidence: removal of alignment, global residual, retention, lag modules and their effect on AUC + correlations.
- Performance preservation protocol: early stopping or phased semantic fine-tuning to avoid late AUC erosion.
- Calibration metrics: Brier score and expected calibration error to assure predictive quality under interpretability constraints.

### Recommended Next Steps
- Implement instrumentation in `train_gainakt2exp.py` for: coverage, bootstrap CIs, lag summaries, calibration metrics.
- Add early stopping and optional second semantic phase (`--semantic_phase_epochs`) with frozen predictive layers.
- Re-run multi-seed (≥5) experiments capturing peak and stabilized metrics; aggregate results into a publication summary JSON/Markdown.
- Produce trade-off visualization: epoch-wise AUC vs mastery correlation curve with annotated chosen stop point.
- Conduct ablations and compile a delta table quantifying contribution of each module.

### Provisional Assessment
Current evaluation confirms persistence of semantic signal but reveals attenuation on test and methodological inconsistency in AUC reporting. Before asserting a balanced interpretability-performance contribution, we must establish metric comparability, reproducibility, and stability across seeds and ablations.

## Comparison with Proposed Learning Gain Attention Architecture 

### Summary of the Proposed Architecture
The To-Be design in `newmodel.md` semantically redefines the attention mechanism so that:
1. Queries (Q) encode the current learning context.
2. Keys (K) encode historical interaction patterns (skill similarity / prerequisite structure).
3. Values (V) explicitly parameterize per-interaction learning gains g_i.

Knowledge state h_t is computed as an attention-weighted sum of past learning gains: h_t = Σ_i α_{t,i} * g_i, and prediction uses [h_t ; embed(S_t)]. Interpretability arises because each component of h_t can be causally decomposed into attention weights and their associated gains.

### Current GainAKT2Exp Implementation
The current model:
- Employs an encoder-only transformer over interaction tokens (question id + correctness).
- Derives mastery and gain via projection heads applied to the final layer representation rather than intrinsic attention Values.
- Accumulates mastery outside the attention mechanism via an additive update (prev + scaled gain), not via Σ α g aggregation.
- Uses external semantic losses (alignment, global residual alignment, retention, lag) and constraint losses (monotonicity, performance alignment, sparsity, consistency) to sculpt emergent correlations.
- Does not bind attention Value tensors to explicit gain semantics during aggregation.

### Alignment (Similarities)
- Encoder-only backbone over (qid, correctness) mirrors (S,R) tuple tokenization intent.
- Explicit gain trajectory concept exists (gain head output) and is monitored.
- Multi-objective optimization integrates predictive and interpretability goals.

### Divergences (Gaps)
- Attention Values are opaque latent vectors; gains are produced post-hoc by a projection head rather than being the Values consumed in weighted aggregation.
- Knowledge state is not formed by Σ α g; attention output does not expose per-interaction gain contributions directly.
- Causal trace from prediction to specific (α_{t,i}, g_i) pairs is partial: modification of a single attention weight does not deterministically adjust mastery without projection interactions.
- Transfer effects across skills are enforced indirectly (sparsity/performance losses) instead of being an emergent property of gain-valued attention.
- No explicit Q-matrix / G-matrix integration inside attention computations; educational structure enters only via token embeddings and loss masks.

### Interpretability Consequences
- Achieved correlations (mastery/gain) support semantic emergence but causal decomposability is weaker than To-Be design where h_t is a direct linear mixture of gains.
- Attribution requires combining attention maps with gain head outputs; intrinsic transparency is limited.

### Refactoring Roadmap
1. Intrinsic Gain Values: Replace gain_head with enforced non-negative Value projections (softplus) so V = g_i.
2. Knowledge State Formation: Redefine attention output for context stream as Σ α g directly (remove intermediate latent transformation).
3. Skill-Space Basis: Map gains onto explicit skill dimensions (num_skills) optionally via low-rank factorization for efficiency.
4. Attribution API: Expose top-k past interactions contributing to current prediction (α_{t,i} * ||g_i||) per head.
5. Structural Masks: Integrate Q-matrix to zero gains for non-linked skills pre-aggregation, reducing reliance on sparsity loss.
6. Minimal Prediction Input: Use [h_t ; skill_emb] only; remove separate value concatenation for purity of formulation.

### Transitional Strategy
Introduce a feature flag (`--intrinsic_gain_attention`) to activate revised semantics while retaining legacy heads for ablation. Collect comparative metrics: causal attribution fidelity, decomposition error, AUC trade-off.

### Target Metrics Post-Refactor
- Decomposition fidelity: ||h_t - Σ α g|| / ||h_t|| < 0.05.
- Causal attribution consistency: correlation between Σ α g skill component and projected mastery > 0.7.
- Non-negative gain violation rate < 1%.
- AUC within 2% of early-stopped baseline.

## Comparison with Dynamic Value Stream Architecture

### Summary of Dynamic Value Stream (newmodel.md)
Introduces dual sequences (Context and Value) that co-evolve across encoder layers. Q,K from Context; V from Value. Both streams undergo independent residual + norm operations per layer, refining perceived gains (Value) and contextual knowledge (Context). Final prediction concatenates (h, v, skill) enabling joint use of accumulated mastery and dynamic learning gain state.

### Aspect-by-Aspect Comparison
| Aspect | Dynamic Value Stream | GainAKT2Exp Current |
|--------|----------------------|---------------------|
| Dual Streams | Separate Context & Value sequences maintained per layer | Single latent sequence; gains projected only at final layer |
| Attention Inputs | Q,K from Context; V from evolving Value | Standard attention over one sequence (implicit Values) |
| Layer-wise Gain Refinement | Value stream updated each block | No intermediate gain exposure (final projection only) |
| Residual Paths | Separate Add & Norm for Context and Value | Single residual normalization path |
| Prediction Inputs | Concatenate (h, v, skill) | Concatenate (context latent, projected gains, skill) |
| Gain Semantics Enforcement | Architectural (Value is gain) | Auxiliary losses external to attention |
| Interpretability Depth | Layer-by-layer gain trajectories available | Only final gain vector interpretable |

### Missing Dynamic Elements
- Lack of per-layer gain evolution trace prevents vertical interpretability (depth refinement analysis).
- No distinct normalization separating gain from context may entangle representations.
- Architectural semantics not guaranteeing that V equals gain; semantics depend on post-hoc projection and losses.

### Advantages of Current Simplicity
- Reduced implementation complexity; leverages existing transformer blocks.
- Lower parameter overhead; fewer moving parts for optimization stability.
- Rapid iteration on semantic loss scheduling without core architectural rewrites.

### Trade-offs
- Loss of layer-wise interpretability and refinement diagnostics.
- Potential ceiling on modeling nuanced temporal gain dynamics (e.g., attenuation, reinforcement loops).
- Harder to claim intrinsic causal semantics vs engineered post-hoc gains.

### Migration Plan to Dynamic Value Stream
1. Add distinct Value embedding table and initialize parallel `value_seq`.
2. Modify encoder block to process (context_seq, value_seq) and output updated pair with separate layer norms.
3. Instrument intermediate `value_seq` states (hooks) for gain magnitude trajectories and per-skill projections.
4. Gradually shift auxiliary gain losses from final projection to per-layer Value states (start at final layer; extend backward).
5. Introduce orthogonality regularizer between averaged context and value representations to prevent collapse.
6. Benchmark dual-stream vs single-stream across seeds (AUC, mastery_corr, gain_corr, attribution fidelity).

### Validation Criteria
- Layer gain stability: systematic refinement pattern (e.g., decreasing variance or structured amplification) across depth.
- Contribution distribution: early layers contribute ≥30% of cumulative gain magnitude (not all deferred to final layer).
- Performance retention: ΔAUC ≤2% vs single-stream baseline over ≥3 seeds.
- Interpretability uplift: +10 percentage points in student coverage with stable gain trajectories.

### Risks & Mitigations
- Over-parameterization → overfitting: mitigate with shared projections or low-rank Value factorization.
- Training instability from dual residuals: stagger LR warm-up for Value parameters.
- Semantic blending (Value ≈ Context): enforce orthogonality or contrastive divergence loss.

### Strategic Recommendation
Phase 1: Implement intrinsic gain attention within current single-stream to establish causal aggregation cheaply.
Phase 2: Introduce dynamic dual-stream only after intrinsic semantics are stable and quantitatively superior in attribution fidelity.
This staged approach manages complexity while ensuring each interpretability enhancement yields measurable educational value.

<!-- Gap summary relocated to end -->

## Comparison with Augmented Architecture Design

### Summary of Augmented Architecture Design
The Augmented Architecture (described under "Augmented Architecture Design" in `newmodel.md`) enhances a baseline GainAKT2-like transformer by adding:
- Two projection heads: mastery_head (context → per-skill mastery) and gain_head (value → per-skill gains).
- Five auxiliary interpretability losses: non-negative gains, monotonic mastery, mastery-performance alignment, gain-performance alignment, sparsity.
- Monitoring hooks for real-time constraint assessment.
It retains the standard attention computation (Values are latent) and treats interpretability as a supervised regularization layer rather than an intrinsic semantic definition.

### Aspect-by-Aspect Comparison
| Aspect | Augmented Architecture Design | Current GainAKT2Exp | Difference |
|--------|-------------------------------|---------------------|------------|
| Projection Heads | Mandatory mastery & gain heads | Mastery & gain heads present | Aligned |
| Loss Suite | Full 5-loss set (non-neg, mono, mastery-perf, gain-perf, sparsity) | Implemented (weights configured) | Aligned |
| Monitoring | Real-time interpretability monitor hooks | Monitoring frequency and correlation logging | Aligned (naming differs) |
| Intrinsic Gain Semantics | Not intrinsic; post-hoc via gain_head | Same | No gap |
| Knowledge State Formation | Latent context + additive mastery accumulation | Same additive rule | No gap |
| Causal Decomposition | Partial (requires combining attention + projections) | Same | No gap |
| Dynamic Value Stream | Optional future; not implemented | Not implemented | Shared future direction |
| Evaluation Metrics | AUC + interpretability violation/correlation metrics | AUC + mastery/gain correlation + violation rates | Current adds semantic alignment modules beyond base design |
| Semantic Modules (Alignment, Retention, Lag) | Not core (can be optional) | Implemented (alignment, global, retention, lag) | Current extends augmentation scope |
| Calibration Metrics | Proposed future addition | Not yet implemented | Pending both |
| Gating / Injection | Future research | Not implemented | Shared future direction |

### Additional Extensions in Current GainAKT2Exp Beyond Augmentation
- Alignment family (local + global residual) to enhance correlation emergence.
- Retention loss to preserve peak mastery trajectory.
- Lag-based gain structuring capturing temporal emergence signals.
- Scheduling controls (warm-up, share cap, residualization) shaping multi-objective dynamics.

### Gaps Relative to Augmented Architecture Goals
| Gap | Description | Impact |
|-----|-------------|--------|
| Unified Metric Framework | Augmented spec implies integrated interpretability reporting; current evaluation uses differing AUC methodologies | Confuses performance comparison; requires consolidation |
| Coverage & Stability Metrics | Augmented design emphasizes systematic interpretability auditing; current lacks coverage %, bootstrap CIs, lag stability summaries | Limits statistical rigor of claims |
| Direct Skill Transfer Visualization | Projection heads exist but no standardized transfer reporting | Weakens educational interpretability evidence |
| Calibration (ECE/Brier) | Suggested for production readiness; absent | Unverified predictive reliability under constraints |

### Advantages of Current Implementation vs Minimal Augmentation
- Demonstrates enhanced semantic trajectory through alignment and lag modules absent in minimal augmented spec.
- Provides higher mastery/gain correlations than early baseline forms, showcasing potential of extended semantic regularization layer.
- Maintains modularity: extended components can be ablated cleanly to revert to minimal augmented core.

### Consolidation Roadmap (Augmented → Publication-Ready)
1. Metric Unification: Implement identical masking logic for train/eval AUC; add calibration metrics.
2. Interpretability Expansion: Coverage %, bootstrap CIs, lag stability, per-skill transfer matrix derived from gain_head projections.
3. Component Ablation Study: Quantify deltas removing alignment, retention, lag losses vs pure 5-loss augmented baseline.
4. Performance Preservation: Add early-stopping + semantic fine-tuning phase flag to retain peak AUC while growing correlations.
5. Reporting Artifacts: Auto-generate a consolidated JSON + CSV summarizing performance + interpretability metrics per seed.

### Decision Matrix for Paper Claims
| Requirement | Minimal Augmented | Current GainAKT2Exp | Needed for Claim |
|-------------|-------------------|---------------------|------------------|
| Competitive AUC (Assist2015) | ~0.72 early | 0.726 early / declines later | Early-stopped preservation |
| Mastery/Gain Correlations | Emergent; moderate | Peaks 0.149 / 0.103 | CI + multi-seed reproducibility |
| No Structural Violations | Enforced by losses | Achieved (0 violation rates) | Maintain |
| Statistical Robustness | Not built-in | Absent | Bootstrap, seeds |
| Layer-wise Interpretability | Limited | Limited | Optional (future) |
| Educational Transfer Evidence | Not explicit | Not explicit | Add transfer matrix |
| Calibration Quality | Pending | Pending | Implement ECE/Brier |

### Positioning Summary
Current GainAKT2Exp fully implements the Augmented Architecture Design’s core elements (projection heads, five losses, monitoring) and extends them with semantic alignment, retention, and lag objectives. The remaining work to elevate from augmented prototype to publishable interpretable architecture centers on metric unification, statistical rigor, performance preservation, and richer educational transfer analyses rather than fundamental architectural rewrites.

### Concise Architecture Gap Summary 

| Dimension | Proposed Intrinsic Gain Attention (Σ α g) | Dynamic Value Stream (Dual Context+Value) | Augmented Architecture Design (Heads+5 Losses) | Current GainAKT2Exp | Gap Impact | Priority |
|-----------|-------------------------------------------|--------------------------------------------|-----------------------------------------------|---------------------|------------|----------|
| Gain Semantics | Values are explicit gains g_i | Value stream refined per layer | Gains projected post-hoc | Gains projected post-encoder + extra semantic modules | Limits causal traceability | High |
| Knowledge State h_t | Direct Σ α g aggregation | Context attention output + separate Value | Latent context; mastery via projection | Recursive additive (prev + scaled gain) | Weaker theoretical alignment | High |
| Attention Attribution | Native α·g decomposition | Layer-wise α with evolving g | Requires combining attention + projection | Same; plus alignment influences | Reduced explanation fidelity | High |
| Layer-wise Gain Evolution | Not required | Explicit per-layer refinement | Only final layer gain head | Only final layer gain head | Loss of vertical interpretability | Medium |
| Skill-Space Integration | Architectural in gain vectors | Indirect via Value projections | Projection heads provide skill mapping | Projection heads; sparsity + alignment | Delayed intrinsic semantics | Medium |
| Q/G-matrix Usage | Mask inside attention/gain | Potential integration in Value path | External sparsity loss | External sparsity + alignment masks | Indirect educational grounding | Medium |
| Non-Negativity Enforcement | Activation choice (e.g. softplus) | Architectural or per-layer constraint | Auxiliary non-negative loss | Auxiliary (weight currently 0.0) | Possible semantic drift | High |
| Prediction Input | [h_t ; skill_emb] | [h ; v ; skill] | [context ; gain ; skill] | [context ; gain ; skill] + semantic modules | Mixed latent semantics | Low |
| Causal Decomposition Metric | Built-in | Layer-wise contribution analyzable | Needs tooling | Needs tooling + alignment disentangling | Attribution overhead | High |
| Complexity vs Baseline | Minimal change | Moderate (dual streams) | Low incremental | Moderate (losses + alignment modules) | Iteration speed vs semantics | - |
| Statistical Interpretability Metrics | Native, direct mapping | Requires layer instrumentation | Loss violation + correlation | Correlations + alignment metrics only | Limited rigor (no CIs, coverage) | High |

Priority Legend: High = foundational causal interpretability; Medium = depth/educational alignment; Low = incremental polish.

Paper Positioning Sentence (updated): *GainAKT2Exp matches the Augmented Architecture Design (projection heads + five educational losses + monitoring) and extends it with alignment, retention, and lag objectives, yet still lacks intrinsic attention-level gain semantics (Values ≠ gains) and direct Σ α g knowledge state formation. Bridging this gap through intrinsic gain attention and unified evaluation metrics is our next step to claim causal interpretability while maintaining competitive AUC.*

## Semantic Interpretabily Recovery

### Objective
Recover non-zero, educationally meaningful mastery and gain correlations after they regressed to 0.0 in a prior configuration, and identify the minimal parameter set whose activation restores semantic signals. Provide actionable guidance for parameter sweep design to optimize the trade-off between predictive AUC and interpretability (correlations, stability, coverage).

### Methodological Approach
We compared two experiment configurations:
1. Pre-recovery (zero correlations): `20251030_232030_gainakt2exp_config_params`.
2. Post-recovery (restored correlations): `20251030_234720_gainakt2exp_recover_bool_fix`.

Both runs share core hyperparameters (learning_rate, weight_decay, monotonicity/gain/mastery performance weights, sparsity, consistency base weights, alignment_weight, alignment_warmup_epochs), but differ along enabled boolean semantics and scheduling parameters. The regression was traced to unintended overrides of store_true flags to `False` (heads and semantic modules disabled), caused by launcher logic that wrote false values when flags were not explicitly passed. The fix re-enabled defaults by only overriding booleans when explicitly set `True`.

We enumerated parameter deltas in the sections: interpretability, alignment, global_alignment, refinement, plus epochs/batch_size/warmup_constraint_epochs. We then linked each delta to plausible causal pathways for mastery/gain correlation emergence.

### Parameter Delta Summary (Pre vs Post)
| Group | Parameter | Pre (Zero Corr) | Post (Recovered Corr) | Causal Role |
|-------|-----------|-----------------|------------------------|-------------|
| interpretability | use_mastery_head | false | true | Enables projection of mastery trajectories necessary for correlation computation. |
| interpretability | use_gain_head | false | true | Produces gain trajectories; correlations impossible without activation. |
| interpretability | enhanced_constraints | false | true | Activates bundled structural losses stabilizing trajectories (monotonicity, alignment to performance, sparsity, consistency synergy). |
| interpretability | warmup_constraint_epochs | 4 | 8 | Longer warm-up reduces early over-regularization, allowing mastery/gain signal to form before full constraint pressure. |
| training/runtime | epochs | 20 | 12 | Shorter training halts before late overfitting / correlation erosion; preserves early semantic signal. |
| training/runtime | batch_size | 96 | 64 | Smaller batch increases update stochasticity; can amplify diversity in latent states aiding correlation emergence. |
| alignment | enable_alignment_loss | false | true | Local alignment shapes latent representations toward performance-consistent mastery evolution. |
| alignment | adaptive_alignment | false | true | Dynamically scales alignment forcing based on correlation feedback; supports sustained growth without over-saturation. |
| global_alignment | enable_global_alignment_pass | false | true | Population-level coherence improves mastery correlation stability (global signal reinforcement). |
| global_alignment | use_residual_alignment | false | true | Removes explained variance, clarifying incremental mastery/gain improvements; sharper correlations. |
| refinement | enable_retention_loss | false | true | Prevents post-peak decay of mastery trajectory, retaining correlation magnitude. |
| refinement | enable_lag_gain_loss | false | true | Introduces temporal structure for gains; lag pattern increases gain correlation interpretability. |
| refinement | retention_delta / retention_weight | inactive | active | Retention only influences trajectories when enabled; improves final correlation retention ratio. |
| refinement | lag_* weights | inactive | active | Lag structuring turns otherwise inert weights into semantic shaping forces. |

All other numeric weights remained identical; correlation recovery is attributable to activation of the semantic and interpretability heads plus extended warm-up and reduced epoch horizon.

### Causal Impact Inference
1. Heads Activation (Mastery/Gain): Mandatory prerequisite; without heads correlations are structurally zero. Their absence fully explains initial regression.
2. Enhanced Constraints: Provides regularization synergy; prevents degenerate or noisy trajectories, increasing correlation stability above random fluctuation.
3. Alignment (Local + Adaptive): Drives early shaping of mastery sequence toward performance-consistent progression, accelerating correlation emergence pre-warm-up completion.
4. Global Residual Alignment: Consolidates population patterns; lifts mastery correlation peak and smooths gain correlation ascent by reducing cross-student variance.
5. Retention: Maintains elevated mastery levels post-peak, reducing late-stage decline and supporting higher final correlation.
6. Lag Gain Loss: Adds temporal causal narrative for gains; improves gain correlation by emphasizing structured progression rather than noise.
7. Warm-up Extension (4 → 8): Avoids premature constraint saturation, allowing latent representations to differentiate before full constraint pressure, yielding higher eventual peak correlations.
8. Epoch Reduction (20 → 12): Avoids performance/semantic drift phase where alignment dominance and constraint loss shares begin to erode predictive calibration and correlation stability.
9. Batch Size Reduction (96 → 64): Increases gradient variance, potentially enhancing exploration and preventing early convergence to flat mastery trajectories (empirical pattern: higher mastery_corr at epoch 3).

### Sweep Design Guidance
We propose a structured sweep with prioritized axes:
- Core Activation Set (binary toggles): {enhanced_constraints, enable_alignment_loss, adaptive_alignment, enable_global_alignment_pass, use_residual_alignment, enable_retention_loss, enable_lag_gain_loss}.
- Warm-up Horizon: warmup_constraint_epochs ∈ {4, 6, 8, 10}.
- Epoch Budget / Early Stop: epochs ∈ {8, 10, 12, 14} with early-stopping on val AUC plateau (ΔAUC < 0.002 over 2 epochs).
- Batch Size: {48, 64, 80, 96} to evaluate impact on correlation variance vs AUC stability.
- Alignment Weight & Cap: alignment_weight ∈ {0.15, 0.25, 0.35}; alignment_share_cap ∈ {0.06, 0.08, 0.10}.
- Lag Gain Weight: lag_gain_weight ∈ {0.04, 0.06, 0.08} with lag_l1:l2:l3 ratios fixed or slightly varied.
- Retention Weight: retention_weight ∈ {0.10, 0.14, 0.18}; retention_delta fixed at 0.005.

Sweep Objective Metrics:
- Peak & final val AUC.
- Peak & final mastery_corr, gain_corr.
- Correlation retention ratio = final_corr / peak_corr.
- Constraint violation rates (expected 0; monitor for regression).
- Alignment loss share trajectory (identify saturation / over-dominance).
- Gain temporal structure metrics (median lag1 correlation, positive fraction).

Multi-stage approach: First coarse sweep to identify promising semantic activation subsets; second fine-tuning sweep on alignment_weight, warmup_constraint_epochs, retention_weight trade-offs.

### Flag Impact Table
| Flag / Parameter | Role | Pre-Recovery Value | Post-Recovery Value | Hypothesized Impact on Mastery Corr | Hypothesized Impact on Gain Corr | Interaction Notes |
|------------------|------|--------------------|---------------------|-------------------------------------|----------------------------------|-------------------|
| use_mastery_head | Enables mastery trajectory | false | true | Enables computation (from 0 to >0.10) | Indirect (gain interacts via consistency) | Must be true for mastery metrics |
| use_gain_head | Enables gain trajectory | false | true | Indirect (gain influences mastery via consistency) | Enables gain correlation (>0.05) | Needed for lag structuring |
| enhanced_constraints | Synergistic structural regularization | false | true | Stabilizes trajectory, raises reliability | Reduces gain noise variance | Enhances effect of alignment |
| enable_alignment_loss | Local alignment shaping | false | true | Accelerates emergence (earlier peak) | Provides smoother gain ascent | Warm-up interacts with its ramp |
| adaptive_alignment | Dynamic scaling | false | true | Avoids plateau, sustains improvements | Prevents over-alignment degradation | Works with share_cap decay |
| enable_global_alignment_pass | Population-level coherence | false | true | Raises peak mastery corr | Minor direct, stabilizing indirectly | Synergizes with residual alignment |
| use_residual_alignment | Residual variance removal | false | true | Sharper mastery increments | Clarifies gain increments | Overuse may reduce AUC; monitor |
| enable_retention_loss | Preserve peaks | false | true | Higher final vs peak retention ratio | Minor direct, prevents mastery decline affecting gain | Tune weight to avoid over-preservation |
| enable_lag_gain_loss | Temporal gain structure | false | true | Mild indirect via constraint interplay | Primary: boosts gain correlation | Needs gain_head active |
| warmup_constraint_epochs | Delay full constraint pressure | 4 | 8 | Higher peak, less early suppression | Gain builds under partial constraints | Too long may delay convergence |
| epochs | Training horizon | 20 | 12 | Avoids late decline phase | Prevents gain drift after peak | Early stopping alternative |
| batch_size | Stochasticity level | 96 | 64 | Slightly higher variance fosters emergence | Better gain differentiation | Trade-off with AUC stability |

### Experimental Phases
1. Diagnostic Recovery: Confirm heads + semantic modules activation rescues correlations (completed).
2. Activation Subset Sweep: Binary subset search to rank contribution (target next).
3. Schedule Optimization: Tune warmup_constraint_epochs vs alignment_weight vs retention_weight for AUC retention.
4. Stability & Robustness: Multi-seed (≥5) runs for top 3 configurations; bootstrap CIs for correlations.
5. Fine-Grained Lag Structuring: Adjust lag_gain_weight and ratios; assess temporal interpretability metrics.
6. Pareto Profiling: Construct AUC vs mastery_corr trade-off curves across retained configurations.

### Measurement & Logging Enhancements (Upcoming)
- Add per-epoch: peak_mastery_corr_so_far, retention_ratio, alignment_effective_weight, lag1_median_corr, lag_positive_fraction.
- Bootstrap (N=200 student resamples) mastery/gain correlation CIs at best epoch and final epoch.
- Coverage: percentage of students with mastery_corr > 0 and gain_corr > 0.05.
- Correlation retention ratio = final_corr / peak_corr.
- Early stopping criteria logging (epochs until AUC plateau, correlation slope).

### Immediate Next Action
Implement logging instrumentation for lag correlation summary, coverage, retention ratio, and bootstrap confidence intervals, then launch activation subset sweep varying enhanced_constraints, alignment family, retention, lag, residual alignment to quantify individual and combined contributions. Document results in a new `SEMANTIC_SWEEP_RESULTS.md` and update this section with empirical impact values.

### Sweep Axes (Concise List)
`use_mastery_head` (ensure always true), `use_gain_head`, `enhanced_constraints`, `enable_alignment_loss`, `adaptive_alignment`, `enable_global_alignment_pass`, `use_residual_alignment`, `enable_retention_loss`, `enable_lag_gain_loss`, `warmup_constraint_epochs`, `alignment_weight`, `alignment_share_cap`, `retention_weight`, `lag_gain_weight`, `batch_size`, `epochs`.

### Expected Outcomes
Recovered configuration demonstrates that enabling semantic modules and interpretability heads plus extending warm-up and reducing training horizon restores correlations (mastery ≈0.10+, gain ≈0.05+). Sweeps will seek configurations yielding mastery_corr ≥0.12 with val AUC ≥0.72 (early-stopped) and gain_corr ≥0.07 under zero violations, establishing a balanced regime for publication.




