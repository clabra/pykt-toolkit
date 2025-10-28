# Experiment 20251028_034014_gainakt3_real_baseline_e5

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 3 val_auc=0.7307 val_acc=0.7573 mastery_corr=0.2212 gain_corr=0.0024 mastery_corr_macro=0.6685 gain_corr_macro=-0.0019

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.2212 |
| Gain Correlation (global) | 0.0024 |
| Mastery Correlation (macro) | 0.6685 |
| Gain Correlation (macro) | -0.0019 |
| Monotonicity Violation Rate | 0.4992 |
| Retention Violation Rate | 0.4992 |
| Gain Future Alignment | -0.0029 |
| Peer Influence Share | 0.5730 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
