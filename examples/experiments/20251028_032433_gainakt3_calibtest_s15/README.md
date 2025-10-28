# Experiment 20251028_032433_gainakt3_calibtest_s15

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 2 val_auc=0.5027 val_acc=0.4991 mastery_corr=-0.0815 gain_corr=-0.0059 mastery_corr_macro=-0.1205 gain_corr_macro=-0.0082

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | -0.0815 |
| Gain Correlation (global) | -0.0059 |
| Mastery Correlation (macro) | -0.1205 |
| Gain Correlation (macro) | -0.0082 |
| Monotonicity Violation Rate | 0.5005 |
| Retention Violation Rate | 0.5005 |
| Gain Future Alignment | 0.0038 |
| Peer Influence Share | 0.5266 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
