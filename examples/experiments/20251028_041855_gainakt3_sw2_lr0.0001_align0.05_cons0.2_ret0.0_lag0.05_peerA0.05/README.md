# Experiment 20251028_041855_gainakt3_sw2_lr0.0001_align0.05_cons0.2_ret0.0_lag0.05_peerA0.05

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.4974 val_acc=0.4978 mastery_corr=-0.0874 gain_corr=0.0031 mastery_corr_macro=-0.0838 gain_corr_macro=0.0004

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | -0.0874 |
| Gain Correlation (global) | 0.0031 |
| Mastery Correlation (macro) | -0.0838 |
| Gain Correlation (macro) | 0.0004 |
| Monotonicity Violation Rate | 0.4998 |
| Retention Violation Rate | 0.4998 |
| Gain Future Alignment | -0.0051 |
| Peer Influence Share | 0.3283 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
