## H1.1 Structural Encoding - Diagnostic Probing Results

**Campaign**: 20260202_222106_benchpaper_698838

**Experiment ID**: 698838


| Dataset | Construct | Fidelity (R²) | Pearson (r) | Selectivity (Δ R² Standard) | Selectivity (Δ R² Strict) |
|---------|-----------|---------------|-------------|----------------------------|---------------------------|
| algebra2005 | Initial Mastery (L₀) | 0.367 ± 0.060 | 0.610 ± 0.046 | 0.420 ± 0.066 | -0.615 ± 0.061 |
| algebra2005 | Learning Rate (T) | 0.163 ± 0.109 | 0.435 ± 0.099 | 0.248 ± 0.107 | -0.808 ± 0.120 |
| bridge2algebra2006 | Initial Mastery (L₀) | 0.393 ± 0.129 | 0.625 ± 0.104 | 0.433 ± 0.137 | -0.232 ± 0.115 |
| bridge2algebra2006 | Learning Rate (T) | 0.468 ± 0.145 | 0.683 ± 0.103 | **0.539 ± 0.131** | -0.256 ± 0.092 |
| nips_task34 | Initial Mastery (L₀) | 0.452 ± 0.034 | 0.678 ± 0.023 | **0.513 ± 0.035** | -0.512 ± 0.070 |
| nips_task34 | Learning Rate (T) | -0.043 ± 0.038 | 0.065 ± 0.029 | 0.006 ± 0.022 | -1.025 ± 0.033 |

**Notes**:
- Results computed from campaign: 20260202_222106_benchpaper_698838
- All metrics averaged across 5-fold cross-validation
- **Selectivity (Δ R²)**: Difference between fidelity and control task R²
  - **Standard**: Observation-level shuffling (baseline control)
  - **Strict**: Skill-consistent shuffling (stronger control, prevents skill-level artifacts)
- **Threshold > 0.5 indicates strong structural encoding** (marked in bold)
- Control R² < 0 confirms probes cannot recover shuffled targets
