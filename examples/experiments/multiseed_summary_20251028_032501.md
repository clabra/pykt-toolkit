# GainAKT3 Multi-Seed Variance Summary

Seeds: [15, 25]

## Aggregate Metrics

| Metric | Mean | Std |
|--------|------|-----|
| val_auc | 0.5034 | 0.0007 |
| val_accuracy | 0.5017 | 0.0026 |
| mastery_corr | -0.0169 | 0.0646 |
| gain_corr | 0.0005 | 0.0065 |
| mastery_corr_macro | 0.0827 | 0.2032 |
| gain_corr_macro | -0.0007 | 0.0075 |
| mastery_corr_macro_weighted | 0.0992 | 0.1796 |
| gain_corr_macro_weighted | -0.0007 | 0.0075 |
| peer_influence_share | 0.4878 | 0.0388 |
| monotonicity_violation_rate | 0.5001 | 0.0003 |
| gain_future_alignment | -0.0006 | 0.0044 |
| reconstruction_error | 0.0004 | 0.0000 |
| difficulty_penalty_contrib_mean | 0.0218 | 0.0311 |

## Per-Seed Best Metrics

| Seed | Folder | val_auc | val_accuracy | mastery_corr | gain_corr | mastery_corr_macro | gain_corr_macro | mastery_corr_macro_weighted | gain_corr_macro_weighted | peer_influence_share | monotonicity_violation_rate | gain_future_alignment | reconstruction_error | difficulty_penalty_contrib_mean |
|------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 15 | 20251028_032433_gainakt3_calibtest_s15 | 0.5027 | 0.4991 | -0.0815 | -0.0059 | -0.1205 | -0.0082 | -0.0805 | -0.0082 | 0.5266 | 0.5005 | 0.0038 | 0.0003 | 0.0529 |
| 25 | 20251028_032450_gainakt3_calibtest_s25 | 0.5042 | 0.5043 | 0.0477 | 0.0070 | 0.2859 | 0.0068 | 0.2788 | 0.0068 | 0.4489 | 0.4998 | -0.0051 | 0.0004 | -0.0093 |

### Notes
- Mean/std computed over best epoch per seed (selected by val_auc).
- Gain threshold applied: 0.0000.
- Use peer context: True | Use difficulty context: True
- Constraint weights: alignment=0.050, consistency=0.100, peer_alignment=0.200, difficulty_ordering=0.100, drift_smoothness=0.050.
