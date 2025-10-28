# Experiment 20251028_163024_gainakt3_realdata_dev

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.5554 val_acc=0.7368 mastery_corr=0.4569 gain_corr=0.0121 mastery_corr_macro=1.0000 gain_corr_macro=-0.0010

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.4569 |
| Gain Correlation (global) | 0.0121 |
| Mastery Correlation (macro) | 1.0000 |
| Gain Correlation (macro) | -0.0010 |
| Monotonicity Violation Rate | 0.4992 |
| Retention Violation Rate | 0.4992 |
| Gain Future Alignment | -0.0017 |
| Peer Influence Share | 0.1570 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
