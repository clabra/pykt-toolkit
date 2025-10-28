# Experiment 20251028_044428_gainakt3_sw8_lr0.0003_align0.05_cons0.2_ret0.0_lag0.05_peerA0.05

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 4 val_auc=0.6825 val_acc=0.7465 mastery_corr=0.2548 gain_corr=0.0121 mastery_corr_macro=0.6598 gain_corr_macro=-0.0024

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.2548 |
| Gain Correlation (global) | 0.0121 |
| Mastery Correlation (macro) | 0.6598 |
| Gain Correlation (macro) | -0.0024 |
| Monotonicity Violation Rate | 0.4990 |
| Retention Violation Rate | 0.4990 |
| Gain Future Alignment | -0.0101 |
| Peer Influence Share | 0.1398 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
