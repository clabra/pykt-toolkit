# Experiment 20251028_033241_gainakt3_auc_baseline_noctx

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 2 val_auc=0.4998 val_acc=0.4999 mastery_corr=0.0267 gain_corr=0.0000 mastery_corr_macro=-0.0424 gain_corr_macro=-0.0011

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0267 |
| Gain Correlation (global) | 0.0000 |
| Mastery Correlation (macro) | -0.0424 |
| Gain Correlation (macro) | -0.0011 |
| Monotonicity Violation Rate | 0.5007 |
| Retention Violation Rate | 0.5007 |
| Gain Future Alignment | -0.0045 |
| Peer Influence Share | 0.4542 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
