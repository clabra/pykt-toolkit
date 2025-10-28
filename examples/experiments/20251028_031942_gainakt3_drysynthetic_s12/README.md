# Experiment 20251028_031942_gainakt3_drysynthetic_s12

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.4973 val_acc=0.5008 mastery_corr=0.0231 gain_corr=0.0030 mastery_corr_macro=-0.3482 gain_corr_macro=0.0012

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0231 |
| Gain Correlation (global) | 0.0030 |
| Mastery Correlation (macro) | -0.3482 |
| Gain Correlation (macro) | 0.0012 |
| Monotonicity Violation Rate | 0.4998 |
| Retention Violation Rate | 0.4998 |
| Gain Future Alignment | 0.0022 |
| Peer Influence Share | 0.4932 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
