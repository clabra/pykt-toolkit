# Experiment 20251028_033852_gainakt3_smoke_ablate

GainAKT3 training on real dataset split (assist2015) with masked losses.

## Summary
Best epoch: 1 val_auc=0.5000 val_acc=0.4997 mastery_corr=-0.1017 gain_corr=0.0026 mastery_corr_macro=0.3333 gain_corr_macro=-0.0000

## Interpretability Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Mastery Correlation (global) | -0.1017 |
| Gain Correlation (global) | 0.0026 |
| Mastery Correlation (macro) | 0.3333 |
| Gain Correlation (macro) | -0.0000 |
| Monotonicity Violation Rate | 0.4996 |
| Retention Violation Rate | 0.4996 |
| Gain Future Alignment | -0.0055 |
| Peer Influence Share | 0.4460 |

Reproducibility Checklist (partial)
- Config saved with MD5
- Environment captured
- Seeds recorded
- Train/validation loaded via init_dataset4train
- Artifact hashes logged (peer=2dfdaf0fbd2d9090, difficulty=e5ab11a146b9c0e1, cold_start=False)
