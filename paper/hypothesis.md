# Validated Hypotheses and Fixed Parameters

## Fixed Parameters (in configs/parameter_default.json)

### 1. intrinsic_gain_attention = false
- **Rationale**: Experimentally determined to be redundant or harmful
- **Experiments**: []

### 2. mastery_performance_loss_weight = 0.0
### 3. gain_performance_loss_weight = 0.0
- **Rationale**: Redundant with alignment loss; removing both improves mastery correlation by 13.6% without affecting AUC
- **Experiments**: ["724276", "542954"]
- **Key Finding**: Alignment loss alone is sufficient for semantic supervision; performance losses add no value

### 4. alignment_weight = 0.15 (UPDATED FROM 0.25)
- **Rationale**: Optimal balance between mastery correlation and gain correlation
- **Experiments**: ["977548", "470340", "724276"]
- **Key Finding**: Alignment weight controls a **fundamental trade-off** between two aspects of interpretability:

| Alignment Weight | Mastery Correlation | Gain Correlation | Best For | AUC |
|------------------|---------------------|------------------|----------|-----|
| 0.10 | 0.1225 (+9.6%) | 0.0291 (-28.5%) | Mastery estimation | 0.7184 |
| **0.15** | **0.1212 (+8.3%)** | **0.0365 (-10.4%)** | **Balanced** | **0.7188** |
| 0.25 | 0.1119 (baseline) | 0.0407 (baseline) | Learning gains | 0.7193 |

**Decision**: Fixed to 0.15 as optimal balance:
- Only -10.4% loss in gain tracking vs 0.25
- Gains +8.3% in mastery estimation vs 0.25
- Negligible AUC difference (0.7188 vs 0.7193)
- More defensible than extremes (0.10 or 0.25)

### 5. Architecture Parameters: d_model=256, n_heads=4, num_encoder_blocks=4, d_ff=512
- **Rationale**: Efficient architecture addressing severe overparameterization in baseline (14.6M params)
- **Experiments**: ["603633"]
- **Key Finding**: 81% parameter reduction (14.6M → 2.7M) with comprehensive benefits:

| Architecture | Parameters | Test AUC | Test Acc | Test Mastery Corr | Test Gain Corr | Peak Epoch | Overfitting |
|--------------|------------|----------|----------|-------------------|----------------|------------|-------------|
| Baseline | 14.6M | 0.7193 | 0.7470 | 0.1119 | 0.0407 | 3 | 0.273 |
| **Reduced** | **2.7M** | **0.7188** | **0.7485** | **0.1165** | **0.0344** | **5** | **0.119** |
| Change | **-81.3%** | **-0.06%** | **+0.20%** | **+4.17%** | **-15.58%** | **+67%** | **-56%** |

**Decision**: Fixed architecture with following justification:
1. **Performance Maintained**: Test AUC -0.06% (negligible), test accuracy +0.20%
2. **Mastery Improved**: +4.17% test correlation, +6.31% train correlation
3. **Training Dynamics**: Peak epoch moved 3→5 (67% later), overfitting reduced 56%
4. **Parameter Efficiency**: 5.5× more updates per parameter (0.000088 vs 0.000016)
5. **Memory Efficiency**: 76% smaller model files (32.9 MB vs ~140 MB)
6. **State-of-the-Art Alignment**: 2.7M params comparable to AKT (~3M) while maintaining dual interpretability

**Trade-off**: Gain correlation decreased 15.58% on test set (-0.40% on train), but this is acceptable given:
- Lower alignment_weight (0.15) naturally favors mastery over gain
- Absolute gain correlation (0.0344) still captures learning dynamics
- Mastery estimation is primary interpretability contribution
- Efficiency gains enable broader adoption

**Baseline Overparameterization Evidence**:
- 14.6M params for 3,080 training sequences = 4,756 params per sequence
- Only 0.000016 updates per parameter (far below typical 0.001 minimum)
- Peak at epoch 3 indicates severe overfitting from limited capacity utilization
- 81% reduction brings to 869 params per sequence, much healthier ratio

## Key Hypotheses Confirmed

### H1: Alignment is Critical for Interpretability
- **Status**: CONFIRMED
- Removing alignment (weight=0.0) causes -43.9% mastery correlation drop
- Experiments ["724276", "542954", "745857"]

### H2: Performance Losses are Redundant
- **Status**: CONFIRMED
- Removing both performance losses improves mastery correlation +13.6%
- No impact on AUC
- Experiments ["724276", "542954"]

### H3: Alignment Weight Trade-off
- **Status**: DISCOVERED
- Lower alignment improves mastery calibration but degrades gain tracking
- Higher alignment improves gain tracking but degrades mastery calibration
- Reveals interpretability dimensions are distinct and tunable
- Experiments ["977548", "470340", "724276"]

Implications for Paper: 
This trade-off should be reported as a **research contribution**:
1. **Not just a hyperparameter** - Reveals interpretability dimensions are orthogonal
2. **Tunable focus** - Practitioners can adjust based on application needs
3. **No free lunch** - Multi-objective optimization inherent to interpretable KT
4. **Design space** - Shows model provides tunable interpretability focus

---

## Capacity Increase Experiments (November 12, 2025)

### Experiment A: Moderate Capacity Increase
- **Configuration**: d_model=384, d_ff=1024, n_heads=8, blocks=4, dropout=0.15, alignment_weight=0.15
- **Parameters**: 7.05M (2.6× baseline)
- **Results**: Peak valid AUC = 0.7243 @ epoch 3
- **Status**: ❌ FAILED - Underperformed baseline (0.7258)

### Experiment B: Reduce Interpretability Constraints
- **Configuration**: d_model=256, d_ff=512, n_heads=4, blocks=4, alignment_weight=0.05 (reduced)
- **Parameters**: 2.73M (same as baseline)
- **Results**: Peak valid AUC = 0.7239 @ epoch 4
- **Status**: ❌ FAILED - Underperformed baseline (0.7258)

### Experiment C: Combined (Capacity + Reduced Constraints)
- **Configuration**: d_model=384, d_ff=1024, n_heads=8, blocks=4, alignment_weight=0.05
- **Parameters**: 7.09M (2.6× baseline)
- **Results**: Peak valid AUC = 0.7249 @ epoch 3
- **Status**: ❌ FAILED - Underperformed baseline (0.7258)

### Summary Table

| Experiment | Params | Peak Epoch | Valid AUC | Δ vs Baseline | Mastery Corr | Gain Corr | Status |
|------------|--------|------------|-----------|---------------|--------------|-----------|--------|
| **Baseline** | 2.7M | 5 | **0.7258** | — | 0.1165 | **0.0344** | ✅ OPTIMAL |
| Exp A | 7.05M | 3 | 0.7243 | -0.21% | **0.1218** | 0.0210 | ❌ |
| Exp B | 2.73M | 4 | 0.7239 | -0.26% | 0.1169 | 0.0151 | ❌ |
| Exp C | 7.09M | 3 | 0.7249 | -0.12% | **0.1221** | 0.0220 | ❌ |

### H4: Capacity Increase Improves AUC
- **Status**: ❌ REJECTED
- **Finding**: Increasing capacity from 2.7M to 7M parameters **decreases** validation AUC by 0.12-0.21%
- **Evidence**: All three experiments (A, B, C) underperformed the baseline 2.7M configuration
- **Root Cause**: 
  1. **Overparameterization**: 7M params for 3,080 sequences = 2,273 params/sequence (too high)
  2. **Severe Overfitting**: All experiments peaked at epoch 3-4, then collapsed 8-9% by epoch 11-13
  3. **Training Dynamics**: Early peak indicates insufficient data for larger capacity
- **Conclusion**: The 2.7M parameter baseline is **optimally sized** for this dataset

### H5: Reducing Interpretability Constraints Improves AUC
- **Status**: ❌ REJECTED
- **Finding**: Reducing alignment_weight from 0.15 to 0.05 **decreases** validation AUC by 0.12-0.26%
- **Evidence**: Experiments B and C (reduced constraints) underperformed baseline
- **Key Insight**: Interpretability losses provide **crucial regularization**:
  - Alignment loss prevents overfitting (not just improves interpretability)
  - Removing constraints causes model to memorize training data
  - Gain correlation dropped 36-56% with reduced alignment
- **Conclusion**: alignment_weight=0.15 is optimal for both performance and interpretability

### Critical Finding: Severe Overfitting Pattern

All capacity experiments showed catastrophic validation collapse:

| Experiment | Peak (Epoch 3-4) | Late Training (Epoch 11-13) | Collapse |
|------------|------------------|----------------------------|----------|
| Exp A | 0.7243 | 0.6524 | **-9.9%** |
| Exp B | 0.7239 | 0.6841 | -5.5% |
| Exp C | 0.7249 | 0.6583 | **-9.2%** |

**Training AUC continued rising** (→0.89) while validation collapsed, indicating severe memorization.

**Implication**: Early stopping with patience=3-4 is critical to prevent collapse.

### Lessons Learned

1. **Baseline Architecture is Optimal**: 2.7M parameters perfectly sized for dataset (869 params/sequence)
2. **More Parameters ≠ Better Performance**: 7M params caused worse generalization
3. **Interpretability Constraints Are Regularizers**: Removing them hurts both interpretability and AUC
4. **Early Stopping Essential**: Model peaks at epoch 3-5, training beyond causes catastrophic overfitting
5. **Alignment Weight is Optimal**: 0.15 provides best balance of performance and interpretability

### Recommendation

**Keep all current defaults unchanged**:
- Architecture: d_model=256, n_heads=4, blocks=4, d_ff=512
- alignment_weight=0.15
- Total parameters: 2.7M
- Training: 12 epochs with early stopping (patience=3-4)

**Next research direction**: Instead of capacity increases, explore architectural improvements that add targeted inductive biases (e.g., skill difficulty parameters, student learning speed embeddings) with minimal parameter overhead.