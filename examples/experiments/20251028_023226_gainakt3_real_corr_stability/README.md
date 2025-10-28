# Experiment 20251028_023226_gainakt3_real_corr_stability

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 3 val_auc=0.7597 val_acc=0.7657 mastery_corr=0.1074 gain_corr=-0.0026 mastery_corr_macro=0.6528 gain_corr_macro=-0.0026

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.1074 |
| Gain Correlation (global) | -0.0026 |
| Mastery Correlation (macro) | 0.6528 |
| Gain Correlation (macro) | -0.0026 |
| Monotonicity Violation Rate | 0.4989 |
| Retention Violation Rate | 0.4989 |
| Gain Future Alignment | -0.0052 |
| Peer Influence Share | 0.4413 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=MISSING, difficulty=MISSING, cold_start=True)
- NOTE: cold_start=True (peer/difficulty artifacts missing); interpretability metrics limited.
