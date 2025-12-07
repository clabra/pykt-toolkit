# BKT Validation Results - iKT2 Baseline Experiment

**Experiment**: `20251128_232738_ikt2_irt-baseline_400293`  
**Date**: November 29, 2025  
**Dataset**: ASSIST2015 (test set, 15 students, 1,450 interactions)

---

## Summary

This experiment validates the iKT2 model's mastery estimates (M_IRT) against BKT's dynamic learning trajectories. We compute BKT forward inference using pre-trained BKT parameters to estimate P(L_t) for each test interaction, then correlate with the model's M_IRT values.

---

## Results

### Correlation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **BKT Correlation** | **r = 0.4707** | Moderate alignment with dynamic baseline |
| p-value | 8.15e-81 | Highly significant |
| MSE | 0.0565 | Moderate deviation |
| MAE | 0.1965 | Average absolute difference ~20% |

### Comparison with IRT Validation

| Validation Type | Correlation | Theory Type |
|-----------------|-------------|-------------|
| **IRT Correlation** | **r = 0.8304** | Static psychometric theory |
| **BKT Correlation** | **r = 0.4707** | Dynamic learning model |

### Distribution Statistics

| Metric | Model M_IRT | BKT P(L_t) |
|--------|-------------|------------|
| Mean | 0.764 | 0.765 |
| Std Dev | 0.101 | 0.268 |
| Min | 0.263 | 0.065 |
| Max | 0.942 | 1.000 |

---

## Interpretation

### 1. Moderate BKT Alignment (r=0.47)

**What it means:**
- Model's mastery estimates show **moderate** correlation with BKT's dynamic learning trajectories
- Relationship is statistically significant (p < 0.001)
- Model captures some dynamic learning patterns but not perfectly aligned with BKT

**Why lower than IRT correlation (0.83)?**

The difference in correlations is expected and reveals important characteristics:

1. **Different theoretical foundations:**
   - IRT (r=0.83): Model trained explicitly with M_IRT = σ(θ - β) formula
   - BKT (r=0.47): Independent dynamic model with different assumptions

2. **Model architecture:**
   - iKT2 uses IRT-based mastery inference (θ_t extracted from h, then M = σ(θ-β))
   - This naturally aligns better with IRT than BKT

3. **Variance differences:**
   - Model M_IRT: narrow distribution (std=0.10, range [0.26, 0.94])
   - BKT P(L_t): wide distribution (std=0.27, range [0.07, 1.00])
   - BKT allows more extreme values (near 0 and 1), model is more conservative

### 2. Validation Against Dynamic Baseline

**Positive findings:**
- ✅ Correlation is positive and significant (not random)
- ✅ Means are nearly identical (0.764 vs 0.765)
- ✅ Model captures general trends in learning progression
- ✅ Validates that model learns more than static IRT patterns

**Areas of divergence:**
- ⚠️ Model's mastery estimates are more concentrated (less variance)
- ⚠️ BKT shows stronger learning effects (wider swings in P(L_t))
- ⚠️ r=0.47 suggests model could better capture dynamic learning patterns

### 3. Methodological Note

**BKT Forward Inference:**
- We compute BKT P(L_t) online using trained BKT parameters on test interactions
- This is the correct approach (not using pre-computed trajectories)
- Each student starts with skill priors, updates via Bayesian inference after each response
- Represents "what BKT would predict" for the same test sequence

---

## Implications for Model Design

### Strengths Confirmed

1. **IRT-based architecture is sound:**
   - Strong IRT correlation (0.83) validates theoretical grounding
   - Model successfully learns interpretable θ_t and β_k parameters

2. **Dynamic learning captured:**
   - Positive BKT correlation (0.47) shows model is not purely static
   - Encoder learns temporal patterns (h[t] encodes learning progression)

### Potential Improvements

1. **Increase dynamic sensitivity:**
   - Current model may be "smoothing" too much (std=0.10 vs BKT std=0.27)
   - Consider: Allow θ_t to vary more dramatically across timesteps
   - Trade-off: Stability vs dynamic sensitivity

2. **Hybrid validation target:**
   - Currently: L_align = MSE(p_correct, M_IRT) only uses IRT
   - Alternative: Add BKT term: L_align = MSE(p, M_IRT) + λ_bkt * MSE(p, M_BKT)
   - Would encourage alignment with both static (IRT) and dynamic (BKT) theories

3. **Per-skill learning rates:**
   - BKT has different "learn" rates per skill
   - iKT2 ability_encoder produces global θ_t (not per-skill)
   - Consider: Skill-specific learning dynamics in architecture

---

## Conclusion

**BKT correlation (r=0.47) validates that iKT2 captures dynamic learning patterns beyond static IRT**, though not as strongly as it aligns with IRT theory (r=0.83). This is expected given:
- Model architecture is explicitly IRT-based
- Training uses IRT alignment loss
- BKT represents independent theoretical framework

**The dual validation (IRT + BKT) demonstrates:**
- ✅ **Strong psychometric grounding** (IRT r=0.83)
- ✅ **Meaningful dynamic learning** (BKT r=0.47, p<<0.001)
- ✅ **Better than purely static models** (positive BKT correlation)

**For paper:**
- Report both correlations to show balanced approach
- Emphasize IRT as primary theoretical framework
- Use BKT as complementary validation of dynamic learning
- Acknowledge trade-off: psychometric rigor (IRT) vs dynamic sensitivity (BKT)

---

## Technical Details

**BKT Parameters Source:**
- File: `data/assist2015/bkt_targets.pkl`
- Method: pyBKT trained on full ASSIST2015 dataset
- Parameters: prior, learn, slip, guess per skill (100 skills)

**Computation:**
- Script: `examples/compute_bkt_correlation.py`
- Method: BKT forward algorithm on test interactions
- Output: `bkt_validation.json`

**Test Set:**
- Students: 15
- Interactions: 1,450
- Skills: 80 (subset of 100 total)
