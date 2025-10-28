# Experiment 20251028_035054_gainakt3_real_both_disabled_e5

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 2 val_auc=0.6818 val_acc=0.7453 mastery_corr=0.1088 gain_corr=0.0077 mastery_corr_macro=0.6540 gain_corr_macro=-0.0008

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.1088 |
| Gain Correlation (global) | 0.0077 |
| Mastery Correlation (macro) | 0.6540 |
| Gain Correlation (macro) | -0.0008 |
| Monotonicity Violation Rate | 0.4991 |
| Retention Violation Rate | 0.4991 |
| Gain Future Alignment | 0.0015 |
| Peer Influence Share | 0.6681 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
