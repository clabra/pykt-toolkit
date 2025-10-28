# Mastery Temperature Calibration Summary

Experiment: 20251028_032433_gainakt3_calibtest_s15

Primary selection metric: mastery_corr_macro_weighted

| Temp | val_auc | val_acc | mastery_corr | mastery_macro | mastery_wmacro |
|------|---------|---------|--------------|---------------|---------------|
| 0.75 | 0.5007 | 0.6047 | 0.5751 | 0.6230 | -0.1296 |
| 1.00 | 0.5007 | 0.6047 | 0.5633 | 0.6210 | -0.1355 |
| 1.25 | 0.5007 | 0.6047 | 0.5568 | 0.6200 | -0.1385 |

## Best Temperature
Chosen: 0.75 (max mastery_corr_macro_weighted)

Reproducibility:
- config_md5: fea87d88722f2eca991dcf1548519377
- checkpoint: model_best.pth
- gain_threshold: 0.0

Selection rationale: maximize primary metric, break ties by val_auc then absolute mastery_corr proximity to zero (stability heuristic).
