## H1.1 Structural Encoding - Diagnostic Probing Results

**Campaign**: 20260202_222258_benchpaper_assist2009_893468

**Experiment ID**: 893468


| Dataset | Construct | Fidelity (R²) | Pearson (r) | Selectivity (Δ R² Standard) | Selectivity (Δ R² Strict) |
|---------|-----------|---------------|-------------|----------------------------|---------------------------|
| assist2009 | Initial Mastery (L₀) | 0.549 ± 0.064 | 0.743 ± 0.041 | **0.619 ± 0.061** | -0.360 ± 0.058 |
| assist2009 | Learning Rate (T) | 0.515 ± 0.080 | 0.721 ± 0.050 | **0.570 ± 0.080** | -0.373 ± 0.108 |

**Notes**:
- Results computed from campaign: 20260202_222258_benchpaper_assist2009_893468
- All metrics averaged across 5-fold cross-validation
- **Selectivity (Δ R²)**: Difference between fidelity and control task R²
  - **Standard**: Observation-level shuffling (baseline control)
  - **Strict**: Skill-consistent shuffling (stronger control, prevents skill-level artifacts)
- **Threshold > 0.5 indicates strong structural encoding** (marked in bold)
- Control R² < 0 confirms probes cannot recover shuffled targets
