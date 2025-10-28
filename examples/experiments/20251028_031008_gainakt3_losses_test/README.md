# Experiment 20251028_031008_gainakt3_losses_test

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.4992 val_acc=0.4948 mastery_corr=-0.0851 gain_corr=0.0037 mastery_corr_macro=0.3333 gain_corr_macro=0.0006

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | -0.0851 |
| Gain Correlation (global) | 0.0037 |
| Mastery Correlation (macro) | 0.3333 |
| Gain Correlation (macro) | 0.0006 |
| Monotonicity Violation Rate | 0.4999 |
| Retention Violation Rate | 0.4999 |
| Gain Future Alignment | -0.0052 |
| Peer Influence Share | 0.4867 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
