# Experiment 20251028_031138_gainakt3_losses_test

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 2 val_auc=0.5015 val_acc=0.4967 mastery_corr=-0.0547 gain_corr=0.0038 mastery_corr_macro=0.3333 gain_corr_macro=0.0010

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | -0.0547 |
| Gain Correlation (global) | 0.0038 |
| Mastery Correlation (macro) | 0.3333 |
| Gain Correlation (macro) | 0.0010 |
| Monotonicity Violation Rate | 0.4997 |
| Retention Violation Rate | 0.4997 |
| Gain Future Alignment | -0.0054 |
| Peer Influence Share | 0.4183 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
