# Experiment 20251026_101736_gainakt2exp_baseline

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
Seeds: 21, 42, 63

| Final Metric | Mean | Std | Min | Max |
|--------------|------|-----|-----|-----|
| best_val_auc | 0.720954 | 0.000325 | 0.720656 | 0.721407 |
| mastery_correlation | -0.003057 | 0.001252 | -0.004462 | -0.001421 |
| gain_correlation | 0.063613 | 0.006936 | 0.057310 | 0.073274 |
| val_accuracy_last_epoch | 0.752348 | 0.000508 | 0.751637 | 0.752796 |

## Semantic Stability (Temporal)
| Temporal Metric | Mean | Std | Min | Max |
|-----------------|------|-----|-----|-----|
| mastery_corr_variance | 0.000120 | 0.000022 | 0.000089 | 0.000137 |
| gain_corr_variance | 0.000171 | 0.000135 | 0.000051 | 0.000360 |
| global_mastery_corr_variance | 0.000474 | 0.000121 | 0.000323 | 0.000619 |
| mastery_corr_slope | -0.000040 | 0.000330 | -0.000506 | 0.000212 |
| gain_corr_slope | -0.001573 | 0.000600 | -0.002397 | -0.000987 |
| retention_penalty_count | 11.000000 | 0.816497 | 10.000000 | 12.000000 |
| retention_penalty_mean | 0.003304 | 0.000433 | 0.002909 | 0.003907 |
| retention_penalty_peak | 0.007803 | 0.001020 | 0.006386 | 0.008748 |

### Interpretive Narrative
- Variance metrics quantify epoch-to-epoch stability of semantic grounding.
- Positive early slopes followed by low variance indicate successful semantic emergence then consolidation.
- Retention penalty count/mean/peak describe correlation decay events; high peak + high count may signal over-regularization or instability.
- Divergence between global vs local mastery variance can motivate adjusting alignment sampling or weight.
- If slopes negative post warm-up, investigate alignment decay or excessive retention penalty.