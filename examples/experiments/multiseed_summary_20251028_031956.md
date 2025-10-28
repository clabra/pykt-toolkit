# GainAKT3 Multi-Seed Variance Summary

Seeds: [11, 12]

## Aggregate Metrics

| Metric | Mean | Std |
|--------|------|-----|
| val_auc | 0.4996 | 0.0023 |
| val_accuracy | 0.5012 | 0.0005 |
| mastery_corr | -0.0533 | 0.0765 |
| gain_corr | 0.0002 | 0.0028 |
| mastery_corr_macro | -0.0960 | 0.2522 |
| gain_corr_macro | -0.0003 | 0.0015 |
| peer_influence_share | 0.5035 | 0.0102 |
| monotonicity_violation_rate | 0.5001 | 0.0003 |
| gain_future_alignment | 0.0015 | 0.0008 |

## Per-Seed Best Metrics

| Seed | Folder | val_auc | val_accuracy | mastery_corr | gain_corr | mastery_corr_macro | gain_corr_macro | peer_influence_share | monotonicity_violation_rate | gain_future_alignment |
|------|--------|------|------|------|------|------|------|------|------|------|
| 11 | 20251028_031929_gainakt3_drysynthetic_s11 | 0.5019 | 0.5017 | -0.1298 | -0.0026 | 0.1562 | -0.0019 | 0.5137 | 0.5004 | 0.0007 |
| 12 | 20251028_031942_gainakt3_drysynthetic_s12 | 0.4973 | 0.5008 | 0.0231 | 0.0030 | -0.3482 | 0.0012 | 0.4932 | 0.4998 | 0.0022 |

### Notes
- Mean/std computed over best epoch per seed (selected by val_auc).
- Gain threshold applied: 0.0000.
- Use peer context: True | Use difficulty context: True
- Constraint weights: alignment=0.050, consistency=0.100, peer_alignment=0.200, difficulty_ordering=0.100, drift_smoothness=0.050.
