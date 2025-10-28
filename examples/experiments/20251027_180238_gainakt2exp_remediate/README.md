# Experiment 20251027_180238_gainakt2exp_remediate

Model: gainakt2exp
Short title: remediate
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
Seeds: 21, 42, 63

| Final Metric | Mean | Std | Min | Max |
|--------------|------|-----|-----|-----|
| best_val_auc | 0.720951 | 0.000173 | 0.720716 | 0.721128 |
| mastery_correlation | -0.001871 | 0.000732 | -0.002603 | -0.000872 |
| gain_correlation | 0.061731 | 0.004180 | 0.056183 | 0.066271 |
| val_accuracy_last_epoch | 0.752348 | 0.000485 | 0.751754 | 0.752942 |

## Semantic Stability (Temporal)
| Temporal Metric | Mean | Std | Min | Max |
|-----------------|------|-----|-----|-----|
| mastery_corr_variance | 0.000107 | 0.000023 | 0.000076 | 0.000130 |
| gain_corr_variance | 0.000154 | 0.000141 | 0.000050 | 0.000354 |
| global_mastery_corr_variance | 0.000181 | 0.000012 | 0.000165 | 0.000194 |
| mastery_corr_slope | 0.000246 | 0.000345 | -0.000200 | 0.000640 |
| gain_corr_slope | -0.001476 | 0.000801 | -0.002597 | -0.000773 |
| retention_penalty_count | 11.333333 | 0.471405 | 11.000000 | 12.000000 |
| retention_penalty_mean | 0.002204 | 0.000318 | 0.001757 | 0.002461 |
| retention_penalty_peak | 0.004113 | 0.000673 | 0.003213 | 0.004831 |

### Interpretive Narrative
- Variance metrics quantify epoch-to-epoch stability of semantic grounding.
- Positive early slopes followed by low variance indicate successful semantic emergence then consolidation.
- Retention penalty count/mean/peak describe correlation decay events; high peak + high count may signal over-regularization or instability.
- Divergence between global vs local mastery variance can motivate adjusting alignment sampling or weight.
- If slopes negative post warm-up, investigate alignment decay or excessive retention penalty.