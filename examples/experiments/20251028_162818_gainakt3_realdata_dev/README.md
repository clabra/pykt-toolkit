# Experiment 20251028_162818_gainakt3_realdata_dev

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.4947 val_acc=0.4970 mastery_corr=0.0857 gain_corr=0.0047 mastery_corr_macro=0.0992 gain_corr_macro=0.0005

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0857 |
| Gain Correlation (global) | 0.0047 |
| Mastery Correlation (macro) | 0.0992 |
| Gain Correlation (macro) | 0.0005 |
| Monotonicity Violation Rate | 0.4999 |
| Retention Violation Rate | 0.4999 |
| Gain Future Alignment | -0.0045 |
| Peer Influence Share | 0.1345 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
