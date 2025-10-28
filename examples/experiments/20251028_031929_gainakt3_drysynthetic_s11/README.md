# Experiment 20251028_031929_gainakt3_drysynthetic_s11

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.5019 val_acc=0.5017 mastery_corr=-0.1298 gain_corr=-0.0026 mastery_corr_macro=0.1562 gain_corr_macro=-0.0019

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | -0.1298 |
| Gain Correlation (global) | -0.0026 |
| Mastery Correlation (macro) | 0.1562 |
| Gain Correlation (macro) | -0.0019 |
| Monotonicity Violation Rate | 0.5004 |
| Retention Violation Rate | 0.5004 |
| Gain Future Alignment | 0.0007 |
| Peer Influence Share | 0.5137 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
