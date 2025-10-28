# Experiment 20251028_181403_gainakt3_realdata_dev

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 4 val_auc=0.7583 val_acc=0.7679 mastery_corr=0.3768 gain_corr=0.0018 mastery_corr_macro=0.6630 gain_corr_macro=0.0023

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.3768 |
| Gain Correlation (global) | 0.0018 |
| Mastery Correlation (macro) | 0.6630 |
| Gain Correlation (macro) | 0.0023 |
| Monotonicity Violation Rate | 0.4994 |
| Retention Violation Rate | 0.4994 |
| Gain Future Alignment | -0.0042 |
| Peer Influence Share | 0.5034 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=MISSING, difficulty=MISSING, cold_start=True)
- NOTE: cold_start=True (peer/difficulty artifacts missing); interpretability metrics limited.
