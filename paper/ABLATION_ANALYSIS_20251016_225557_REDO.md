# Ablation Study Re-Run Report

**Generated**: 2025-10-16 23:10:02  
**Dataset**: assist2015  

## Configurations

### ablation_enhanced_20251016_225557

- Enhanced Constraints: Yes
- Resolved Weights:
  - Non-negative: 0.0
  - Monotonicity: 0.1
  - Mastery-Performance: 0.8
  - Gain-Performance: 0.8
  - Sparsity: 0.2
  - Consistency: 0.3

### ablation_purebce_20251016_225557

- Enhanced Constraints: No (Pure BCE)
- Resolved Weights:
  - Non-negative: 0.0
  - Monotonicity: 0.0
  - Mastery-Performance: 0.0
  - Gain-Performance: 0.0
  - Sparsity: 0.0
  - Consistency: 0.0

## Results Summary

| Configuration | Best Val AUC | Final AUC | Mastery Corr | Gain Corr | Train Time (s) |
|---------------|--------------|-----------|--------------|-----------|----------------|
| ablation_enhanced_20251016_225557 | 0.7260 | 0.6268 | 0.026 | 0.002 | 464.7 |
| ablation_purebce_20251016_225557 | 0.7258 | 0.6236 | 0.029 | -0.025 | 381.2 |

## Difference Analysis

- Δ Best AUC (Enhanced - Pure BCE): +0.0002
- Enhanced added +83.5s training time

## Notes

- Pure BCE run now truly zeros all constraint weights as per new logic.
- Consistency loss term newly implemented; values reflect alignment residual penalty.
- Architectural enforcement (non-negative gains + cumulative mastery) still active in both runs.
- For a stricter architectural ablation, disable mastery/gain heads in a follow-up baseline.