# Experiment 20251028_044107_gainakt3_sw7_lr0.0001_align0.05_cons0.2_ret0.0_lag0.0_peerA0.0

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 5 val_auc=0.6785 val_acc=0.7460 mastery_corr=0.3387 gain_corr=0.0101 mastery_corr_macro=0.6661 gain_corr_macro=-0.0025

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.3387 |
| Gain Correlation (global) | 0.0101 |
| Mastery Correlation (macro) | 0.6661 |
| Gain Correlation (macro) | -0.0025 |
| Monotonicity Violation Rate | 0.4998 |
| Retention Violation Rate | 0.4998 |
| Gain Future Alignment | -0.0097 |
| Peer Influence Share | 0.1138 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
