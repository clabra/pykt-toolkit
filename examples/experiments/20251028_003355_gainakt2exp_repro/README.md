# Experiment 20251028_003355_gainakt2exp_repro

Model: gainakt2exp
Short title: repro
Ablation mode: both_lag
Schema version: 2

# Reproducibility Checklist

| Item | Status |
|------|--------|
| Folder naming convention followed | ✅ |
| config.json contains all params | ✅ |
| Shell script lists full command | ✅ |
| Best + last checkpoints saved | ✅ |
| Per-epoch metrics CSV present | ✅ |
| Raw stdout log saved | ✅ |
| Git commit & branch recorded | ✅ |
| Seeds documented | ✅ |
| Environment versions captured | ✅ |
| Correlation metrics logged | ✅ |

## Multi-seed Stability Summary
Seeds: 42

| Final Metric | Mean | Std | Min | Max |
|--------------|------|-----|-----|-----|
| best_val_auc | 0.724398 | 0.000000 | 0.724398 | 0.724398 |
| mastery_correlation | -0.014939 | 0.000000 | -0.014939 | -0.014939 |
| gain_correlation | 0.064288 | 0.000000 | 0.064288 | 0.064288 |
| val_accuracy_last_epoch | 0.753496 | 0.000000 | 0.753496 | 0.753496 |

## Semantic Stability (Temporal)
| Temporal Metric | Mean | Std | Min | Max |
|-----------------|------|-----|-----|-----|
| mastery_corr_variance | 0.000092 | 0.000000 | 0.000092 | 0.000092 |
| gain_corr_variance | 0.000144 | 0.000000 | 0.000144 | 0.000144 |
| global_mastery_corr_variance | 0.000145 | 0.000000 | 0.000145 | 0.000145 |
| mastery_corr_slope | 0.000410 | 0.000000 | 0.000410 | 0.000410 |
| gain_corr_slope | -0.000888 | 0.000000 | -0.000888 | -0.000888 |
| retention_penalty_count | 32.000000 | 0.000000 | 32.000000 | 32.000000 |
| retention_penalty_mean | 0.002166 | 0.000000 | 0.002166 | 0.002166 |
| retention_penalty_peak | 0.003806 | 0.000000 | 0.003806 | 0.003806 |

### Interpretive Narrative
- Variance metrics quantify epoch-to-epoch stability of semantic grounding.
- Positive early slopes followed by low variance indicate successful semantic emergence then consolidation.
- Retention penalty count/mean/peak describe correlation decay events; high peak + high count may signal over-regularization or instability.
- Divergence between global vs local mastery variance can motivate adjusting alignment sampling or weight.
- If slopes negative post warm-up, investigate alignment decay or excessive retention penalty.

## Parameter Scenario Quick Reference
Comprehensive (defaults): --ablation_mode both_lag (alignment+retention+lag) warmup=8 alignment_weight=0.30 retention_weight=0.12 lag_gain_weight=0.05
Predictive baseline: --ablation_mode baseline (all interpretability losses disabled, heads optional)
Alignment-only: --ablation_mode align (retention & lag disabled) to observe correlation decay
Retention stress: increase --retention_weight (e.g., 0.18) and decrease --retention_delta (e.g., 0.003) to test decay resistance
Variance recovery: extend --warmup_constraint_epochs (e.g., 10) and reduce performance loss weights (0.8→0.7) to boost head variance
Mastery-only: --use_gain_head False to isolate mastery semantics
Disable AMP: add --no_amp (AMP enabled by default)
See paper/README_gainakt2exp.md for full parameter tables & tuning guidelines.