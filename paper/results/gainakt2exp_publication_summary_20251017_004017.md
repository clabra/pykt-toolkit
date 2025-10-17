# GainAKT2Exp Publication-Grade Summary
Generated: 2025-10-17T00:40:17.849862
## Aggregate Metrics
| Variant | Seeds | Mean AUC | Std AUC | 95% CI Low | 95% CI High | ΔAUC vs Heads-Off | Mastery Corr | Gain Corr | Mono Viol. | Neg Gain | Bounds Viol. |
|---------|-------|----------|---------|------------|-------------|-------------------|--------------|-----------|------------|----------|-------------|
| heads_off | 2 | 0.7166 | 0.0006 | 0.7158 | 0.7174 | +0.0000 | 0.000 | 0.000 | 0.00% | 0.00% | 0.00% |
| arch_only | 2 | 0.7166 | 0.0006 | 0.7158 | 0.7174 | +0.0000 | 0.029 | -0.016 | 0.00% | 0.00% | 0.00% |
| full | 2 | 0.7171 | 0.0005 | 0.7164 | 0.7179 | +0.0005 | 0.017 | -0.004 | 0.00% | 0.00% | 0.00% |

## Success Criteria Evaluation
| Criterion | Value | Pass |
|-----------|-------|------|
| Predictive Parity ΔAUC (Full-HeadsOff) | +0.0005 | True |
| Stability Std(AUC) (Heads-Off) | 0.0006 | True |
| Architecture-Only ΔAUC vs Heads-Off | +0.0000 | n/a |
| Constraints Incremental ΔAUC (Full-ArchOnly) | +0.0005 | n/a |

### Interpretation
- Full model performance-neutral (ΔAUC +0.0005).
- Architecture-only effect vs Heads-Off: +0.0000.
- Incremental constraints effect (Full − ArchOnly): +0.0005.

**NOTE:** Add per-term constraint loss shares + correlation CIs for publication tables.
