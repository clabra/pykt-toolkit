# GainAKT2Exp Model Documentation

## 1. Overview
GainAKT2Exp is an attention-based Knowledge Tracing (KT) model designed to balance predictive performance and semantic interpretability. It extends earlier GainAKT variants by introducing multi-head cumulative mastery representation, explicit gain dynamics, alignment-driven semantic emergence, retention-based peak preservation, and multi-lag predictive gain objectives. All interpretability mechanisms are enabled by default to promote meaningful latent trajectories without requiring manual flag specification.

## 2. Design Principles
We follow three core principles:
1. Semantic First: Mastery and gain heads generate trajectories correlated with performance early in training.
2. Constraint-Guided Emergence: Monotonicity, sparsity, consistency, and performance alignment losses sculpt interpretable latent states.
3. Reproducibility: Every experiment launched through `examples/run_repro_experiment.py` produces a complete artifact set (config, logs, metrics, checkpoints) allowing exact reconstruction.

## 3. Default Semantic Interpretability Configuration
The training script `examples/train_gainakt2exp.py` and launcher `examples/run_repro_experiment.py` enable the following components by default:

| Component | Purpose | Default | Disable Flag |
|-----------|---------|---------|--------------|
| Mastery head | Produces per-skill mastery trajectory | Enabled | `--disable_mastery_head` |
| Gain head | Captures incremental skill acquisition signals | Enabled | `--disable_gain_head` |
| Enhanced constraints | Monotonicity, performance alignment (mastery & gain), sparsity, consistency | Enabled | `--pure_bce` |
| Local alignment loss | Correlates mastery/gains with performance locally | Enabled | `--disable_alignment_loss` |
| Adaptive alignment scaling | Increases alignment pressure if correlations lag | Enabled | `--disable_adaptive_alignment` |
| Global alignment pass | Stratified sequence-level correlation consolidation | Enabled | `--disable_global_alignment_pass` |
| Retention loss | Penalizes decay after a mastered peak (preserves semantic peaks) | Enabled | `--disable_retention_loss` |
| Multi-lag gain emergence objective | Encourages gains predicting future correctness | Enabled | `--disable_lag_gain_loss` |

All enable states are serialized in `config.json` under the interpretability sections (`interpretability`, `alignment`, `global_alignment`, `refinement`). Disable flags are explicitly recorded when used, ensuring transparency.

## 4. Disable Flags: Obtaining Predictive Baselines
To revert to a pure predictive (non-semantic) baseline, pass the following flags:
```
--pure_bce \
--disable_mastery_head \
--disable_gain_head \
--disable_alignment_loss \
--disable_adaptive_alignment \
--disable_global_alignment_pass \
--disable_retention_loss \
--disable_lag_gain_loss
```
A typical predictive baseline launch (1 epoch smoke test):
```
python examples/run_repro_experiment.py \
  --train_script examples/train_gainakt2exp.py \
  --model_name gainakt2exp \
  --dataset assist2015 \
  --epochs 1 \
  --short_title pure_bce_baseline \
  --pure_bce --disable_mastery_head --disable_gain_head \
  --disable_alignment_loss --disable_global_alignment_pass \
  --disable_retention_loss --disable_lag_gain_loss
```

## 5. Training Usage (Direct Script)
Direct invocation without any flags automatically enables semantic mechanisms:
```
python examples/train_gainakt2exp.py --dataset assist2015 --epochs 20
```
Common optional arguments:
- `--learning_rate` (default 1.74e-4)
- `--batch_size` (default 96)
- `--patience` (early stopping, default 20)
- `--gradient_clip` (default 1.0)
- `--alignment_weight` (default 0.25)
- `--warmup_constraint_epochs` (default 4)

Disable examples:
```
python examples/train_gainakt2exp.py --dataset assist2015 --epochs 12 --disable_alignment_loss --pure_bce
```

## 6. Reproducible Experiment Launcher
Preferred path for formal experiments:
```
python examples/run_repro_experiment.py \
  --train_script examples/train_gainakt2exp.py \
  --model_name gainakt2exp \
  --dataset assist2015 \
  --epochs 20 \
  --short_title semantic_default
```
Artifacts stored under `examples/experiments/[TIMESTAMP]_gainakt2exp_semantic_default/`:
- `config.json`: Complete resolved parameters + command provenance + MD5 hash
- `stdout.log`: Timestamped training console output
- `stderr.log`: Error stream (only if non-empty)
- `metrics_epoch.csv`: Per-epoch metrics including loss shares and correlations
- `results.json`: Legacy-format summary (best AUC + consistency metrics)
- `summary.json`: Launcher-tail metrics or diagnostic note
- `model_best.pth` / `model_last.pth`: Checkpoints
- `environment.txt`: Python/PyTorch/CUDA versions + git commit
- `SEED_INFO.md`: Documented seeds
- `README.md`: Human-readable experiment summary
- `artifacts/` (optional): Semantic trajectory JSONs or plots

## 7. Metrics Logged Per Epoch
Minimum set appended to `metrics_epoch.csv`:
- `epoch`, `train_loss`, `val_loss`, `val_auc`, `val_accuracy`
- Consistency: `monotonicity_violation_rate`, `negative_gain_rate`, `bounds_violation_rate`
- Correlations: `mastery_correlation`, `gain_correlation`
- Loss decomposition: `main_loss_share`, `constraint_loss_share`, `alignment_loss_share`, `lag_loss_share`, `retention_loss_share`
Alignment share may appear negative because alignment is added as a negative correlation reward inside the loss; sign encodes contribution direction.

## 8. Recommended Default Epoch Count
We default to 20 epochs to allow early emergence (often peaked by epoch 3–5) while retaining capacity for stabilization of global alignment and lag objectives. For quick reproductions use the relaunch script with `--override_epochs`.

## 9. Relaunching Experiments
Use:
```
python examples/relaunch_experiment.py --source_dir <experiment_folder> --short_title reproduce --strict
```
To truncate epochs for audit-only reproduction:
```
python examples/relaunch_experiment.py --source_dir <experiment_folder> --short_title reproduce3 --override_epochs 3
```
The relaunch audit records MD5 consistency, device adaptation, and epoch differences. Strict mode aborts if core training parameters differ (unless only metadata changes).

## 10. Baseline Comparison Strategy
For fair comparison:
1. Semantic Default (all enabled) – report AUC, accuracy, mastery/gain correlations.
2. Pure BCE baseline (all disabled) – same metrics; expect reduced correlation and slightly different AUC.
3. Progressive Ablations: disable alignment, retention, lag objectives individually to quantify contributions.
All runs must satisfy reproducibility checklist (see `AGENTS.md`).

## 11. Troubleshooting
| Symptom | Likely Cause | Mitigation |
|---------|--------------|-----------|
| Empty `metrics_epoch.csv` | Heads/semantic disabled in older script version | Re-run with updated script or ensure disable flags not passed inadvertently |
| Negative alignment share large magnitude | Alignment weight too high early | Adjust `--alignment_weight` or increase `--alignment_warmup_epochs` |
| Low mastery correlation plateau | Correlation below min target | Tune `--alignment_weight` or disable retention if premature decay |
| OOM on large batch | GPU memory pressure | Lower `--batch_size` or reduce `d_model` (requires model code change) |

## 12. Citation & Usage Notes
When referencing GainAKT2Exp experiments externally, cite the full experiment folder name and include the `config.json` MD5 hash to assert reproducibility.

## 13. License & Restrictions
This documentation and associated model code are private and confidential (see repository LICENSE). Redistribution or external model training using this content is prohibited.

---
We will iterate this README as architecture refinements or new interpretability diagnostics are added.
