# iKT2 As-Is

Summary of current implementation of the ik2 model. 

## Total Loss
```
Phase 1
L_total = L_BCE + λ_reg × L_reg
```

```
Phase 2
L_total = L_BCE + λ_reg × L_reg + λ_align × L_align
```

## Loss Functions

### L_BCE (Performance)
```
L_BCE = -1/N ∑ᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)] (Binary Cross-Entropy)
```

```
Metrics:
- val_per_auc
- val_per_acc
```

### L_align (Interpretability)

In current approach, **interpretability is measured in terms of alignment between Heads**, i.e. alignment between Head 2, that makes predictions based on the reference model (IRT in this case) and Head 1, based on optimizing performance prediction. This means that system is guided toward theta and beta parameters that make predictions aligned with H1 no with the theoretical model. 

```
L_align = 1/N ∑ᵢ (ŷᵢ - M_IRT)² (Mean Squared Error - MSE)
```

```
Compared with Head 1 predictions, no with true labels. 

M_IRT is infered from context h vector: 
    # === HEAD 2: IRT-Based Mastery Inference ===
    # Step 1: Infer student ability from knowledge state
    theta_t = self.ability_encoder(h).squeeze(-1)  # [B, L]

    # Step 2: Extract skill difficulties for questions being answered
    beta_k = self.skill_difficulty_emb(qry).squeeze(-1)  # [B, L]

    # Step 3: Compute mastery probability (according to IRT reference model)
    mastery_irt = torch.sigmoid(theta_t - beta_k)  # [B, L]
```

```
Metrics:
- val_align_mse (MSE between BCE predictions and IRT mastery)
- val_align_mae (MAE between BCE predictions and IRT mastery)

```

### L_reg (Regularization)
```
L_reg = 1/K ∑ₖ (β_learned - β_IRT)² (Mean Squared Error - MSE)
```
```
Metrics:
- val_reg_mse (MSE between learned and IRT-calibrated difficulties)
```

##  Interpretability Diagnostic (Monitoring, Not Enforcing)
- val_heads_corr (Pearson correlation between mastery_irt and p_correct)
  - p_correct = Head 1 predictions
  - mastery_irt = Head 2 predictions
  - Measures how well Head 2's IRT mastery aligns with Head 1's performance predictions

# Issues of Current Approach

In Head 2, what we want to measure is consistency with the reference, theoretical, model.

The current approach achieves internal consistency (high head agreement) but NOT external alignment with Rasch reference (poor Kendall and Spearman correlations). 

This is because:
- L_align = MSE(p_correct, M_IRT) only aligns the two heads with each other
- Neither head is directly optimized to match Rasch reference M_ref
- Both heads can agree but still diverge from ground truth IRT calibration

## Empirical Validation: Experiment 20251206_173247_ikt2_lrefmetrics_fixed_636735

Training configuration:
- Dataset: ASSISTments 2015 (fold 0)
- Epochs: 17 (Phase 1: epochs 1-11, Phase 2: epochs 12-17)
- Hyperparameters: λ_align=1.0, λ_reg=0.01
- Rasch reference: data/assist2015/rasch_per_skill_targets_fold0.pkl

### Results Summary

**Performance Metrics:**
- Test AUC: 0.7151
- Test Accuracy: 74.58%

**Internal Consistency (Head Agreement):**
- Head Agreement (r = 0.8801): **EXCELLENT** - Strong correlation between M_IRT and p_correct
- Interpretation: Both heads produce coherent predictions, demonstrating meaningful learned parameters

**External Alignment with Rasch Reference:**
- L_ref MSE: 0.6602
- L_ref MAE: 0.7737
- Ref Kendall: **-0.0253** (negative correlation)
- Ref Spearman: **-0.0311** (negative correlation)
- Interpretation: **POOR** - Model predictions inversely correlated with Rasch calibration

**Other Interpretability Metrics:**
- Difficulty Fidelity (r = 0.8372): **STRONG** - Learned β closely matches IRT β_IRT
- BKT Correlation (r = 0.2177): **POOR** - Weak alignment with Bayesian Knowledge Tracing

### Phase Transition Analysis

**Phase 1 → Phase 2 Transition (Epoch 11 → 12):**
- Head Agreement: 0.40 → 0.87 (**+118% improvement**)
- L_ref MSE: 0.835 → 0.698 (**-16% improvement**)
- Validation AUC: 0.724 → 0.721 (-0.4%, acceptable trade-off)

**Key Observation:** Adding L_align in Phase 2 dramatically improved **internal consistency** (head agreement) but **did not establish external alignment** with Rasch reference (correlations remained negative).

### Conclusion: Current Approach Limitations Confirmed

The experiment provides strong empirical evidence for the theoretical critique:

1. **Internal consistency ≠ External validity**: High head agreement (0.88) does not guarantee alignment with reference model (negative correlations)

2. **L_align optimizes wrong target**: MSE(p_correct, M_IRT) forces both heads to agree with each other, not with M_ref from Rasch calibration

3. **Missing direct supervision**: Neither head receives gradients from comparison with Rasch reference M_ref during training (L_ref only used for monitoring in validation)

4. **Phase 2 improves wrong metric**: While Phase 2 successfully increases head agreement, it does not address the fundamental issue of misalignment with theoretical model



