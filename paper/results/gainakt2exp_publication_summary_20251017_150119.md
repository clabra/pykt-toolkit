# GainAKT2Exp Publication-Grade Summary
Generated: 2025-10-17T15:01:19.157816
## Aggregate Metrics
| Variant | Seeds | Mean AUC | Std AUC | 95% CI Low | 95% CI High | ΔAUC vs Heads-Off | Mastery Corr | Gain Corr | Mono Viol. | Neg Gain | Bounds Viol. |
|---------|-------|----------|---------|------------|-------------|-------------------|--------------|-----------|------------|----------|-------------|
| heads_off | 5 | 0.6406 | 0.0037 | 0.6373 | 0.6438 | +0.0000 | 0.000 | 0.000 | 0.00% | 0.00% | 0.00% |
| arch_only | 5 | 0.6406 | 0.0037 | 0.6373 | 0.6438 | +0.0000 | 0.032 | -0.009 | 0.00% | 0.00% | 0.00% |
| full | 5 | 0.6520 | 0.0023 | 0.6501 | 0.6540 | +0.0115 | 0.025 | -0.005 | 0.00% | 0.00% | 0.00% |

## Success Criteria Evaluation
| Criterion | Value | Pass |
|-----------|-------|------|
| Predictive Parity ΔAUC (Full-HeadsOff) | +0.0115 | True |
| Stability Std(AUC) (Heads-Off) | 0.0037 | True |
| Architecture-Only ΔAUC vs Heads-Off | +0.0000 | n/a |
| Constraints Incremental ΔAUC (Full-ArchOnly) | +0.0115 | n/a |

### Interpretation
- Full model positive gain (ΔAUC +0.0115); verify CI excludes 0.
- Architecture-only effect vs Heads-Off: +0.0000.
- Incremental constraints effect (Full − ArchOnly): +0.0115.

**NOTE:** Add per-term constraint loss shares + correlation CIs for publication tables.
