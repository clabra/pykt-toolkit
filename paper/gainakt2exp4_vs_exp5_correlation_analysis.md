# Correlation Analysis: gainakt2exp4 (single-seed remediation) vs gainakt2exp5 (multi-seed remediation)

## 1. Objective
We examine why mastery and gain correlation metrics did not substantially improve despite parameter adjustments (increased performance loss weights, removal of warm-up constraints in subsequent exploratory runs). We quantify differences between the single-seed remediation artifact (`gainakt2exp4_metrics.csv`, seed 63) and the multi-seed remediation aggregation (`gainakt2exp5_metrics.csv`, seeds 21, 42, 63).

## 2. Key Metrics Extracted

| Metric | gainakt2exp4 (seed 63) | gainakt2exp5 (mean across 3 seeds) | Delta (mean - seed63) |
|--------|------------------------|------------------------------------|-----------------------|
| Final AUC | 0.72113 | 0.72095 (mean best) | -0.00018 |
| Final Gain Corr | 0.05618 | 0.06173 | +0.00555 |
| Final Mastery Corr | -0.00087 | -0.00187 | -0.00100 |
| Gain Corr Max (across seeds) | 0.05618 | 0.06627 | +0.01009 |
| Gain Corr Min (across seeds) | 0.05618 | 0.05618 | 0.00000 |
| Gain Corr Std (across seeds) | — | 0.00418 | — |
| Mastery Corr Std (across seeds) | — | 0.00073 | — |

Observation: The average gain correlation increased modestly (+0.0055 absolute, ~9.9% relative to 0.05618). Mastery correlation drifted slightly more negative and remains near zero. Thus, the assertion that correlation "didn't improve" holds qualitatively for mastery (remains negligible) but gain correlation did exhibit a small uptick that may be within noise given narrow confidence intervals and small sample size (3 seeds).

## 3. Diagnosis of Limited Correlation Improvement
### 3.1 Loss Weight Coupling
Increasing `mastery_performance_loss_weight` and `gain_performance_loss_weight` prioritizes predictive accuracy. This can compress variance in head outputs (pushing them toward calibrated probability ranges), reducing Pearson correlation potential with broader latent constructs (e.g., cumulative knowledge or performance deltas).

### 3.2 Early Constraint Removal (No Warm-Up)
Removing warm-up (`warmup_constraint_epochs=0`) means alignment / retention / lag regularizers exert full force from epoch 1, potentially damping exploratory shaping of representation variance. This often yields prematurely stabilized (low-variance) auxiliary head activations, constraining correlation growth.

### 3.3 Measurement Geometry
Correlations are computed on raw head outputs whose scale may be influenced by shared backbone gradients. Without explicit normalization (e.g., z-scoring, layernorm, temperature scaling) correlations can be artificially suppressed due to range compression or skew.

### 3.4 Target Proxy Misalignment
If mastery correlation is measured against a simplistic proxy (e.g., immediate correctness, running average) while the mastery head is implicitly optimized toward next-step prediction, semantic mismatch produces near-zero correlations even when head is useful for prediction (the head learns a *decision boundary* rather than a *latent mastery trajectory*).

### 3.5 Interference Between Heads
Gain and mastery heads may share upstream representation channels without orthogonality or disentanglement penalties, generating collinearity and reducing distinct variance explanatory power needed for correlation amplification.

### 3.6 Variance Collapse via Constraints
Retention / alignment penalties may indirectly penalize deviations that would increase head variance. A tight constraint regime (alignment weight 0.30, retention 0.12, lag 0.05) without scheduled annealing can suppress dynamic range.

## 4. Hypothesized Root Causes Ranked
1. Variance suppression (constraint + performance weight synergy) – High likelihood.
2. Proxy target mismatch for mastery – High likelihood.
3. Lack of correlation-focused auxiliary objectives – High likelihood.
4. Absence of output normalization – Moderate likelihood.
5. Head interference (no orthogonality) – Moderate likelihood.
6. Small seed set (3) causing statistical uncertainty in perceived improvement – Supporting factor.

## 5. Recommended Strategy for Meaningful Correlation Gains
### 5.1 Add Explicit Correlation Maximization Losses
- `L_mastery_corr = -corr( master_out, mastery_proxy )`
- `L_gain_corr = -corr( gain_out, gain_proxy )`
Use moving-window mastery proxy (smoothed cumulative correctness) and gain proxy (short-horizon performance delta). Apply after epoch warm-up (e.g., start at epoch 4) to avoid destabilizing early learning.

### 5.2 Preserve Variance
Introduce variance floor penalty: `L_var = max( var_target_floor - Var(head_out), 0 )` for each head. Floor can decay slowly (e.g., start at 0.05 then anneal).

### 5.3 Output Normalization & Scaling
Apply layer normalization + learnable temperature (scalar `tau`) before computing correlation metrics. This ensures stable scale and reduces saturation.

### 5.4 Two-Phase Training Schedule
Phase 1 (epochs 1–4): Focus on predictive performance (existing losses), relaxed constraints (scale alignment/retention by 0.5). Phase 2 (epochs 5+): Reintroduce full constraint weights + activate correlation losses + raise sparsity/consistency if needed.

### 5.5 Orthogonality / Disentanglement
Penalty: `L_ortho = || (W_mastery^T W_gain) ||_F^2` to encourage distinct feature subspaces improving independent variance for correlations.

### 5.6 Proxy Refinement
- Mastery proxy: Exponentially weighted moving average of correctness with decay factor (e.g., 0.9) plus difficulty adjustment.
- Gain proxy: Short-term improvement: difference between last k (e.g., 5) accuracy and preceding k-window accuracy, clipped to reduce noise.

### 5.7 Constraint Scheduling
Introduce alignment weight ramp: `alignment_weight(t) = alignment_base * sigmoid( (t - t0)/s )`. This avoids early variance collapse.

### 5.8 Diagnostic Instrumentation Enhancements
Add per-epoch logs: `mastery_var`, `gain_var`, `mastery_zskew`, `gain_zskew`, `corr_loss_share`. This enables tracing whether low correlations stem from variance or alignment mismatch.

## 6. Experimental Plan (Sequential)
1. Implement instrumentation (variance & skew logging).  
2. Add normalization layers to heads; re-run baseline to measure correlation shift.  
3. Introduce orthogonality penalty (low weight, e.g., 0.01).  
4. Add phased correlation losses with warm-up; track improvement delta.  
5. Evaluate scheduling of alignment & retention vs constant regime.  
6. Integrate refined proxies including difficulty adjustment (prep for GainAKT3 modules).  
7. Multi-seed run (≥5 seeds) compute statistical significance (confidence interval for mean gain/mastery correlations).  

## 7. Success Criteria
- Gain correlation mean ≥ 0.08 sustained final epochs without AUC degradation (>0.720).  
- Mastery correlation mean ≥ 0.03 (indicates proxy alignment) with variance > baseline variance floor.  
- Violation rates remain 0.0 or near 0.0.  
- Overfit decay (peak - final AUC) ≤ 0.0005 ensuring stability.  

## 8. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Correlation loss destabilizes early training | Reduced AUC | Phase scheduling + gradual weight ramp. |
| Variance floor induces noisy activations | Increased loss variance | Anneal floor; monitor var trajectories. |
| Orthogonality penalty conflicts with alignment | Lower correlation | Keep weight small; evaluate ablation. |
| Proxy noise (gain delta volatility) | Unstable correlation metric | Use smoothed rolling windows with clipping. |

## 9. Immediate Next Steps
- Patch missing MD5 in `gainakt2exp4_metrics.csv` for reproducibility.  
- Extend training script to log head variances and add optional normalization.  
- Prepare correlation loss module (switchable flags: `--enable_corr_objectives`, `--corr_warmup_epochs`).  

## 10. Summary
Mastery correlation stagnation and modest gain correlation improvement likely arise from strong performance weighting, early constraints without warm-up, and absence of explicit correlation-targeted objectives or variance preservation. The proposed multi-component enhancement focuses on controlled variance, structured scheduling, direct correlation losses, orthogonality, and refined proxies to unlock meaningful interpretability gains without sacrificing predictive performance.

## 11. Parameter-Only Correlation Boost Variants

We outline CLI-only variants (no code changes) aimed at increasing mastery/gain correlations. These adjust existing loss weights, warm-up regime, and regularization balance. Expectations remain moderate; substantial gains beyond the listed targets likely require instrumentation (Sections 5–7).

### Variant A (Gentle Rebalance)
Goal: Small correlation uplift with minimal AUC risk.
```bash
python examples/train_gainakt2exp_repro.py \
  --experiment_title corr_varA \
  --experiment_suffix corr_varA \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --warmup_constraint_epochs 4 \
  --learning_rate 0.00015 \
  --mastery_performance_loss_weight 0.9 \
  --gain_performance_loss_weight 0.9 \
  --alignment_weight 0.30 \
  --retention_weight 0.08 \
  --lag_gain_weight 0.04 \
  --sparsity_loss_weight 0.25 \
  --consistency_loss_weight 0.25 \
  --use_amp
```
Expected final gain_corr ≈ 0.07–0.075; mastery_corr ≈ 0.01–0.02.

### Variant B (Variance Liberation)
Goal: Increase early variance; moderate AUC stability risk.
```bash
python examples/train_gainakt2exp_repro.py \
  --experiment_title corr_varB \
  --experiment_suffix corr_varB \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --warmup_constraint_epochs 6 \
  --learning_rate 0.00014 \
  --mastery_performance_loss_weight 0.8 \
  --gain_performance_loss_weight 0.85 \
  --alignment_weight 0.28 \
  --retention_weight 0.06 \
  --lag_gain_weight 0.03 \
  --sparsity_loss_weight 0.30 \
  --consistency_loss_weight 0.20 \
  --use_amp
```
Expected peak gain_corr ≈ 0.10 early; final ≈ 0.085; mastery_corr ≈ 0.02–0.025.

### Variant C (Aggressive Exploration)
Goal: Maximize head variance; higher instability risk.
```bash
python examples/train_gainakt2exp_repro.py \
  --experiment_title corr_varC \
  --experiment_suffix corr_varC \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --warmup_constraint_epochs 8 \
  --learning_rate 0.00013 \
  --mastery_performance_loss_weight 0.7 \
  --gain_performance_loss_weight 0.75 \
  --alignment_weight 0.26 \
  --retention_weight 0.05 \
  --lag_gain_weight 0.025 \
  --sparsity_loss_weight 0.35 \
  --consistency_loss_weight 0.18 \
  --use_amp
```
Expected early gain_corr ≈ 0.11; final ≈ 0.09; mastery_corr ≈ 0.025–0.03. Monitor val_auc; abort if < 0.715 mid-run.

### Variant D (Extended Horizon Shaping)
Goal: Allow semantic heads longer stabilization.
```bash
python examples/train_gainakt2exp_repro.py \
  --experiment_title corr_varD \
  --experiment_suffix corr_varD \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --epochs 28 \
  --warmup_constraint_epochs 6 \
  --learning_rate 0.00015 \
  --mastery_performance_loss_weight 0.85 \
  --gain_performance_loss_weight 0.85 \
  --alignment_weight 0.30 \
  --retention_weight 0.07 \
  --lag_gain_weight 0.04 \
  --sparsity_loss_weight 0.28 \
  --consistency_loss_weight 0.22 \
  --use_amp
```
Expected finals similar to Variant B but with reduced decay from peak.

### Variant E (Variance Floor Emulation)
Goal: Prevent collapse via slightly raised implicit variance floor.
```bash
python examples/train_gainakt2exp_repro.py \
  --experiment_title corr_varE \
  --experiment_suffix corr_varE \
  --seeds 21 42 63 \
  --devices 0 1 2 3 4 \
  --ablation_mode both_lag \
  --warmup_constraint_epochs 5 \
  --variance_floor 0.0003 \
  --learning_rate 0.00015 \
  --mastery_performance_loss_weight 0.85 \
  --gain_performance_loss_weight 0.9 \
  --alignment_weight 0.29 \
  --retention_weight 0.07 \
  --lag_gain_weight 0.04 \
  --sparsity_loss_weight 0.30 \
  --consistency_loss_weight 0.20 \
  --use_amp
```
Expected gain_corr ≈ 0.085–0.09; mastery_corr ≈ 0.02–0.025 with improved variance stability.

### Selection & Escalation
1. Start Variant A; evaluate improvement after 20 epochs.  
2. If gain_corr < 0.07, escalate to Variant B.  
3. If decay >40% from peak to final, try Variant D.  
4. If variance collapse observed, move to Variant E.  
5. If mastery_corr remains <0.015, proceed with instrumentation (normalization + correlation losses) beyond parameter-only tuning.

### Monitoring Checklist
Per epoch capture: gain_corr, mastery_corr, AUC, constraint_loss_share, variance (if available). Abort criteria: persistent AUC <0.715; gain_corr early peak >0.10 with final <0.07 (excessive decay indicates need for correlation loss or relaxed retention).

### Limitations
Parameter-only adjustments cannot guarantee large semantic correlation gains; structural instrumentation is the next necessary phase if targets in Section 7 are unmet.
