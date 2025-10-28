# Experiment 20251028_162959_gainakt3_realdata_dev

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.5562 val_acc=0.7319 mastery_corr=0.3675 gain_corr=0.0123 mastery_corr_macro=1.0000 gain_corr_macro=-0.0007

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.3675 |
| Gain Correlation (global) | 0.0123 |
| Mastery Correlation (macro) | 1.0000 |
| Gain Correlation (macro) | -0.0007 |
| Monotonicity Violation Rate | 0.4990 |
| Retention Violation Rate | 0.4990 |
| Gain Future Alignment | -0.0011 |
| Peer Influence Share | 0.1544 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
