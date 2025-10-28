# Experiment 20251028_032450_gainakt3_calibtest_s25

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 2 val_auc=0.5042 val_acc=0.5043 mastery_corr=0.0477 gain_corr=0.0070 mastery_corr_macro=0.2859 gain_corr_macro=0.0068

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0477 |
| Gain Correlation (global) | 0.0070 |
| Mastery Correlation (macro) | 0.2859 |
| Gain Correlation (macro) | 0.0068 |
| Monotonicity Violation Rate | 0.4998 |
| Retention Violation Rate | 0.4998 |
| Gain Future Alignment | -0.0051 |
| Peer Influence Share | 0.4489 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
