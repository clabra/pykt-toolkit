# Experiment 20251028_033503_gainakt3_auc_ctx_constraints

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 20 val_auc=0.5047 val_acc=0.5054 mastery_corr=0.0294 gain_corr=-0.0003 mastery_corr_macro=0.0384 gain_corr_macro=-0.0006

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | 0.0294 |
| Gain Correlation (global) | -0.0003 |
| Mastery Correlation (macro) | 0.0384 |
| Gain Correlation (macro) | -0.0006 |
| Monotonicity Violation Rate | 0.5007 |
| Retention Violation Rate | 0.5007 |
| Gain Future Alignment | -0.0033 |
| Peer Influence Share | 0.4866 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
