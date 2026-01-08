# Integration of Diagnostic Probing Validation into Paper

## Summary

Successfully integrated diagnostic probing methodology as a **validation framework** for the interpretability metrics in `paper/latex/paper.tex`. This addresses your concern about choosing between the Weighted Semantic Alignment Score and Probing methods by using them **hierarchically** rather than as alternatives.

## Changes Made

### 1. **Enhanced Hypothesis 1 (Line 205)**
- **Before**: Mentioned "high linear recoverability" without explaining validation method
- **After**: Explicitly states that linear recoverability will be validated through "diagnostic probing with control tasks"
- **Rationale**: Creates clear connection between hypothesis and validation methodology

### 2. **New Subsubsection: "Latent Structure Validation via Diagnostic Probing" (After line 429)**
- **Location**: Inserted after the discussion of I₁/I₂ divergence, before the Trade-Off section
- **Content**:
  - Explains the motivation: distinguishing genuine encoding from superficial pattern matching
  - Describes the probing protocol: extracting hidden states (h_t) and training linear probes
  - Introduces the control task methodology (Hewitt & Liang 2019)
  - Defines the Selectivity metric: ΔR² = R²_true - R²_control
  - Presents results in Table (tab_probing):
    - AS2009_S: Selectivity = 0.6870 (Robust Structural Encoding)
    - AS2015: Selectivity = 0.6515 (Strong Alignment)
  - Interprets results: validates that I₁ correlations reflect deep structural encoding
  - Concludes with "dual validation" framework

### 3. **Updated Conclusions (Line 599)**
- **Before**: Only mentioned convergent validity (I₁)
- **After**: Mentions both I₁ and probing selectivity (ΔR² > 0.65)
- **Added**: "This dual validation—combining correlation-based alignment with rigorous probing selectivity—establishes that the semantic alignment reflects genuine structural encoding rather than superficial pattern matching."

## Methodological Framework

The paper now uses a **two-tier validation approach**:

### Primary Metrics (For Pareto Analysis)
- **I₁** (Convergent Validity): Pearson correlation between projected parameters and BKT values
- **I₂** (Predictor Equivalence): Correlation between induced and reference trajectories
- **Composite I**: I = (I₁ₗ + I₁ₜ + I₂) / 3

### Validation Metric (For Rigor)
- **Probing Selectivity**: ΔR² = R²_true - R²_control
- **Purpose**: Proves that I₁ correlations reflect genuine structural encoding
- **Threshold**: ΔR² > 0.5 indicates "strong encoding"

## Key Benefits

1. **Avoids Overcomplexity**: Primary interpretability metric (I) remains simple and pedagogically interpretable
2. **Adds Rigor**: Probing provides gold-standard validation expected by ML reviewers
3. **Complete Story**: 
   - I₁, I₂ → "The model aligns with theory"
   - Probing → "This alignment is structurally real, not accidental"
4. **Aligns with Abstract**: Delivers the "formal validation framework" promised in the abstract

## Bibliography

Both required citations are already present in `biblio.bib`:
- `hewitt2019designing`: Hewitt & Liang (2019) - Selectivity metric
- `alain2018understanding`: Alain & Bengio (2018) - Probing methodology

## Answer to Your Question

**"High linear recoverability"** means:
- BKT constructs can be **accurately recovered** from iDKT's latent representations
- Using **simple linear transformations** (linear probes)
- Validated by high R² on true task AND high selectivity (true >> control)

This demonstrates that the latent space is **structurally organized** around pedagogical constructs, not just correlated with them by chance.

## Next Steps

The paper now has a complete, rigorous interpretability validation framework that:
1. Uses simple, interpretable metrics (I₁, I₂) for primary analysis
2. Validates these with gold-standard probing methodology
3. Tells a coherent story from hypothesis → validation → conclusions

No need to choose between approaches—both work together to provide pedagogical clarity AND methodological rigor.
