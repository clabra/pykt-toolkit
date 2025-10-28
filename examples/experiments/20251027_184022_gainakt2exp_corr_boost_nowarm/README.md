# Experiment 20251027_184022_gainakt2exp_corr_boost_nowarm

Model: gainakt2exp
Short title: corr_boost_nowarm
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
| best_val_auc | 0.721115 | 0.000155 | 0.720970 | 0.721331 |
| mastery_correlation | -0.000093 | 0.002680 | -0.003789 | 0.002482 |
| gain_correlation | 0.062977 | 0.006430 | 0.055824 | 0.071415 |
| val_accuracy_last_epoch | 0.752423 | 0.000137 | 0.752241 | 0.752572 |

## Semantic Stability (Temporal)
| Temporal Metric | Mean | Std | Min | Max |
|-----------------|------|-----|-----|-----|
| mastery_corr_variance | 0.000122 | 0.000022 | 0.000100 | 0.000151 |
| gain_corr_variance | 0.000412 | 0.000086 | 0.000300 | 0.000510 |
| global_mastery_corr_variance | 0.000186 | 0.000018 | 0.000169 | 0.000211 |
| mastery_corr_slope | -0.000053 | 0.000419 | -0.000446 | 0.000528 |
| gain_corr_slope | -0.000556 | 0.000400 | -0.001042 | -0.000064 |
| retention_penalty_count | 18.666667 | 0.471405 | 18.000000 | 19.000000 |
| retention_penalty_mean | 0.003019 | 0.000211 | 0.002774 | 0.003289 |
| retention_penalty_peak | 0.005580 | 0.000712 | 0.004688 | 0.006430 |

### Interpretive Narrative
- Variance metrics quantify epoch-to-epoch stability of semantic grounding.
- Positive early slopes followed by low variance indicate successful semantic emergence then consolidation.
- Retention penalty count/mean/peak describe correlation decay events; high peak + high count may signal over-regularization or instability.
- Divergence between global vs local mastery variance can motivate adjusting alignment sampling or weight.
- If slopes negative post warm-up, investigate alignment decay or excessive retention penalty.