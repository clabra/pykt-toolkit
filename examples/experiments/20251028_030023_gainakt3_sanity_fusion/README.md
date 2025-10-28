# Experiment 20251028_030023_gainakt3_sanity_fusion

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.5085 val_acc=0.5052 mastery_corr=0.0514 gain_corr=0.0039 mastery_corr_macro=0.1376 gain_corr_macro=0.0013

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0514 |
| Gain Correlation (global) | 0.0039 |
| Mastery Correlation (macro) | 0.1376 |
| Gain Correlation (macro) | 0.0013 |
| Monotonicity Violation Rate | 0.5000 |
| Retention Violation Rate | 0.5000 |
| Gain Future Alignment | -0.0026 |
| Peer Influence Share | 0.5079 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
