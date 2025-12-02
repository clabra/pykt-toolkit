# iKT2 Model

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
```
L_align = 1/N ∑ᵢ (ŷᵢ - M_IRT)² (Mean Squared Error - MSE)
```

```
Compared with model predictions, no with true labels. 

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
- val_align_mse ( MSE between BCE predictions and IRT mastery)
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