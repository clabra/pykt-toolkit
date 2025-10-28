# Experiment 20251028_040653_gainakt3_gatebias_smoke

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.4945 val_acc=0.4968 mastery_corr=0.0743 gain_corr=0.0030 mastery_corr_macro=0.1063 gain_corr_macro=0.0009

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0743 |
| Gain Correlation (global) | 0.0030 |
| Mastery Correlation (macro) | 0.1063 |
| Gain Correlation (macro) | 0.0009 |
| Monotonicity Violation Rate | 0.5000 |
| Retention Violation Rate | 0.5000 |
| Gain Future Alignment | -0.0036 |
| Peer Influence Share | 0.1430 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
