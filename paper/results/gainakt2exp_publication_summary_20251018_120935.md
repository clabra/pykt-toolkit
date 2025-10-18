# GainAKT2Exp Publication-Grade Summary
Generated: 2025-10-18T12:09:35.127115
## Aggregate Metrics
| Variant | Seeds | Mean AUC | Std AUC | 95% CI Low | 95% CI High | ΔAUC vs Heads-Off | Mastery Corr | Gain Corr | Mono Viol. | Neg Gain | Bounds Viol. |
|---------|-------|----------|---------|------------|-------------|-------------------|--------------|-----------|------------|----------|-------------|
| heads_off | 5 | 0.6502 | 0.0030 | 0.6475 | 0.6528 | +0.0000 | 0.000 | 0.000 | 0.00% | 0.00% | 0.00% |
| arch_only | 5 | 0.6502 | 0.0030 | 0.6475 | 0.6528 | +0.0000 | 0.031 | -0.009 | 0.00% | 0.00% | 0.00% |
| full | 5 | 0.7169 | 0.0012 | 0.7159 | 0.7179 | +0.0667 | 0.003 | 0.046 | 0.00% | 0.00% | 0.00% |

## Success Criteria Evaluation
| Criterion | Value | Pass |
|-----------|-------|------|
| Predictive Parity ΔAUC (Full-HeadsOff) | +0.0667 | True |
| Stability Std(AUC) (Heads-Off) | 0.0030 | True |
| Architecture-Only ΔAUC vs Heads-Off | +0.0000 | n/a |
| Constraints Incremental ΔAUC (Full-ArchOnly) | +0.0667 | n/a |

### Interpretation
- Full model positive gain (ΔAUC +0.0667); verify CI excludes 0.
- Architecture-only effect vs Heads-Off: +0.0000.
- Incremental constraints effect (Full − ArchOnly): +0.0667.

**NOTE:** Add per-term constraint loss shares + correlation CIs for publication tables.
