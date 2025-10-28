# Experiment 20251028_023534_gainakt3_real_corr_stability

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 4 val_auc=0.7642 val_acc=0.7691 mastery_corr=0.0549 gain_corr=-0.0006 mastery_corr_macro=0.6534 gain_corr_macro=0.0017

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0549 |
| Gain Correlation (global) | -0.0006 |
| Mastery Correlation (macro) | 0.6534 |
| Gain Correlation (macro) | 0.0017 |
| Monotonicity Violation Rate | 0.4987 |
| Retention Violation Rate | 0.4987 |
| Gain Future Alignment | -0.0032 |
| Peer Influence Share | 0.5068 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=MISSING, difficulty=MISSING, cold_start=True)
- NOTE: cold_start=True (peer/difficulty artifacts missing); interpretability metrics limited.
