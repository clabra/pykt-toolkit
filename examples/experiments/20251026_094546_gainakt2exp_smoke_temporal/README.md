# Experiment 20251026_094546_gainakt2exp_smoke_temporal

Model: gainakt2exp
Short title: smoke_temporal
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
Seeds: 21, 42

| Final Metric | Mean | Std | Min | Max |
|--------------|------|-----|-----|-----|
| best_val_auc | 0.687542 | 0.000204 | 0.687338 | 0.687746 |
| mastery_correlation | -0.000759 | 0.000755 | -0.001514 | -0.000004 |
| gain_correlation | 0.092977 | 0.003181 | 0.089796 | 0.096158 |
| val_accuracy_last_epoch | 0.743277 | 0.000107 | 0.743170 | 0.743384 |

## Semantic Stability (Temporal)
| Temporal Metric | Mean | Std | Min | Max |
|-----------------|------|-----|-----|-----|
| mastery_corr_variance | 0.000228 | 0.000015 | 0.000213 | 0.000243 |
| gain_corr_variance | 0.000115 | 0.000072 | 0.000043 | 0.000187 |
| global_mastery_corr_variance | 0.000798 | 0.000529 | 0.000269 | 0.001327 |
| mastery_corr_slope | -0.030192 | 0.001016 | -0.031208 | -0.029176 |
| gain_corr_slope | 0.020239 | 0.007103 | 0.013137 | 0.027342 |
| retention_penalty_count | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| retention_penalty_mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| retention_penalty_peak | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### Interpretive Narrative
- Variance metrics quantify epoch-to-epoch stability of semantic grounding.
- Positive early slopes followed by low variance indicate successful semantic emergence then consolidation.
- Retention penalty count/mean/peak describe correlation decay events; high peak + high count may signal over-regularization or instability.
- Divergence between global vs local mastery variance can motivate adjusting alignment sampling or weight.
- If slopes negative post warm-up, investigate alignment decay or excessive retention penalty.