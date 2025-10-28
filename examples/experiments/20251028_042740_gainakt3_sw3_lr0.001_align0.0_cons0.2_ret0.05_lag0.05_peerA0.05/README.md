# Experiment 20251028_042740_gainakt3_sw3_lr0.001_align0.0_cons0.2_ret0.05_lag0.05_peerA0.05

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 2 val_auc=0.6818 val_acc=0.7450 mastery_corr=0.1130 gain_corr=0.0175 mastery_corr_macro=0.6536 gain_corr_macro=-0.0006

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.1130 |
| Gain Correlation (global) | 0.0175 |
| Mastery Correlation (macro) | 0.6536 |
| Gain Correlation (macro) | -0.0006 |
| Monotonicity Violation Rate | 0.4992 |
| Retention Violation Rate | 0.4992 |
| Gain Future Alignment | 0.0039 |
| Peer Influence Share | 0.4308 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
