# Experiment 20251028_045746_gainakt3_sw12_lr0.0001_align0.0_cons0.0_ret0.0_lag0.05_peerA0.0

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 5 val_auc=0.6785 val_acc=0.7461 mastery_corr=0.3394 gain_corr=0.0100 mastery_corr_macro=0.6663 gain_corr_macro=-0.0026

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.3394 |
| Gain Correlation (global) | 0.0100 |
| Mastery Correlation (macro) | 0.6663 |
| Gain Correlation (macro) | -0.0026 |
| Monotonicity Violation Rate | 0.4998 |
| Retention Violation Rate | 0.4998 |
| Gain Future Alignment | -0.0094 |
| Peer Influence Share | 0.2590 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
