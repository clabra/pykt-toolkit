# Experiment 20251028_035849_gainakt3_real_heads_only_test

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 2 val_auc=0.6818 val_acc=0.7450 mastery_corr=0.0951 gain_corr=0.0084 mastery_corr_macro=0.6538 gain_corr_macro=-0.0005

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0951 |
| Gain Correlation (global) | 0.0084 |
| Mastery Correlation (macro) | 0.6538 |
| Gain Correlation (macro) | -0.0005 |
| Monotonicity Violation Rate | 0.4994 |
| Retention Violation Rate | 0.4994 |
| Gain Future Alignment | 0.0020 |
| Peer Influence Share | 0.6663 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
