## H1.1 Structural Encoding - Diagnostic Probing Results

**Campaign**: 20260202_222106_benchpaper_datasets_698838

**Experiment ID**: 698838


| Dataset | Construct | Fidelity (R²) | Pearson (r) | Selectivity (Δ R² Standard) | Selectivity (Δ R² Strict) |
|---------|-----------|---------------|-------------|----------------------------|---------------------------|
| nips_task34 | Initial Mastery (L₀) | 0.452 ± 0.034 | 0.678 ± 0.023 | **0.529 ± 0.027** | -0.512 ± 0.070 |
| nips_task34 | Learning Rate (T) | -0.043 ± 0.038 | 0.065 ± 0.029 | 0.006 ± 0.022 | -1.025 ± 0.033 |

**Notes**:
- Results computed from campaign: 20260202_222106_benchpaper_datasets_698838
- All metrics averaged across 5-fold cross-validation
- **Selectivity (Δ R²)**: Difference between fidelity and control task R²
  - **Standard**: Observation-level shuffling (baseline control)
  - **Strict**: Skill-consistent shuffling (stronger control, prevents skill-level artifacts)
- **Threshold > 0.5 indicates strong structural encoding** (marked in bold)
- Control R² < 0 confirms probes cannot recover shuffled targets
