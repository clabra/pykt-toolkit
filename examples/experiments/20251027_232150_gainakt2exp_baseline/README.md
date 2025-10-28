# Experiment 20251027_232150_gainakt2exp_baseline

Model: gainakt2exp
Short title: baseline
Ablation mode: both_lag

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
| best_val_auc | 0.721008 | 0.000000 | 0.721008 | 0.721008 |
| mastery_correlation | -0.002138 | 0.000000 | -0.002138 | -0.002138 |
| gain_correlation | 0.066271 | 0.000000 | 0.066271 | 0.066271 |
| val_accuracy_last_epoch | 0.751754 | 0.000000 | 0.751754 | 0.751754 |

## Semantic Stability (Temporal)
| Temporal Metric | Mean | Std | Min | Max |
|-----------------|------|-----|-----|-----|
| mastery_corr_variance | 0.000130 | 0.000000 | 0.000130 | 0.000130 |
| gain_corr_variance | 0.000058 | 0.000000 | 0.000058 | 0.000058 |
| global_mastery_corr_variance | 0.000194 | 0.000000 | 0.000194 | 0.000194 |
| mastery_corr_slope | 0.000640 | 0.000000 | 0.000640 | 0.000640 |
| gain_corr_slope | -0.001059 | 0.000000 | -0.001059 | -0.001059 |
| retention_penalty_count | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| retention_penalty_mean | 0.001757 | 0.000000 | 0.001757 | 0.001757 |
| retention_penalty_peak | 0.003213 | 0.000000 | 0.003213 | 0.003213 |

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
See paper/PARAMETERS.md for full tables & tuning guidelines.