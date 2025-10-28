# Semantic Interpretability and Correlation Enhancement Toolkit

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

## 1. Purpose and Scope
We outline principled mechanisms to enhance semantic interpretability in knowledge tracing models while preserving competitive predictive performance (AUC, accuracy). Our emphasis is on improving and robustly validating (i) Mastery–Performance Correlation (MPC) and (ii) Gain–Performance Correlation (GPC), alongside creating auxiliary stability, robustness, and causal attribution metrics. The proposed components are modular and can be incrementally integrated into a new Transformer‑attention based model.

## 2. Interpretability Criteria
We define semantic interpretability through:
1. **Stability**: Mastery trajectories evolve smoothly, avoiding implausible volatility.
2. **Monotonic Educational Consistency**: Mastery should rarely decrease absent long inactivity or forgetting events explicitly modeled.
3. **Causal Alignment**: Changes in mastery and gain should exhibit predictive and counterfactual influence on performance.
4. **Disentanglement**: Distinct latent factors (mastery, gain, retention) occupy orthogonal representational subspaces.
5. **Robust Correlation**: MPC and GPC hold under perturbations (noise, dropout) and across random seeds.
6. **Taxonomic Binding**: Latent skill representations align with curricular or ontology prototypes.

## 3. Measurement Enhancements
We recommend extending per‑epoch logging and post‑run summaries:
- Pearson, Spearman, and Kendall rank correlations (raw and partial controlling for item difficulty, attempt index, and time gaps).
- Lagged correlations ρ_lag(k) between mastery_t and performance_{t+k}.
- Robustness metrics: correlation shift after injecting Gaussian noise ε ~ N(0, σ²) into mastery/gain (δρ_noise).
- Stability index: 1 − (std_seed(ρ_mastery) / mean_seed(ρ_mastery)).
- Gain elasticity: regression coefficient of ΔPerformance on gain controlling for mastery.
- Attribution consistency: correlation between gradient×input (or SHAP) feature attributions and mastery changes.
- Calibration over mastery deciles: Expected Calibration Error for mastery‑conditioned prediction bins (ECE_mastery).

All metrics must be written to `metrics_epoch.csv` and aggregated into a structured `results.json` block.

## 4. Architectural Extensions
### 4.1 Dual Subspace Encoders
Separate linear (or low‑rank) projectors for long‑term mastery (slow dynamics) and short‑term gain (fast dynamics). This structural separation facilitates higher disentanglement and clearer causal attribution.

### 4.2 Retention (Decay) Cell
Introduce a gated decay mechanism with learned half‑life per skill. This provides explicit temporal forgetting rather than implicit volatility, improving monotonic interpretability.

### 4.3 Hierarchical Skill Embeddings
Model concept → subskill factorization; mastery is aggregated via attention over subskills, while performance prediction conditions on both levels. Disentangles difficulty and skill granularity, reducing spurious correlation distortions.

### 4.4 Dual‑Path Attention
Path A: predictive causal masked temporal attention for performance. Path B: differential (delta) attention emphasizing recent changes for gain extraction. Enforce orthogonality between path outputs.

### 4.5 Structural Equation Auxiliary Network (SEAN)
An auxiliary interpretable regression head: Performance_t ≈ α·Mastery_{t−1} + β·Gain_{t−1} + γ·Difficulty + δ·TimeGap + ε. Record standardized coefficients and partial R² per epoch.

## 5. Regularizers and Loss Terms
Let M_t, G_t denote mastery and gain vectors at time t.
1. **Orthogonality**: L_orth = ||M^T G||_F² / (||M||_F² ||G||_F² + ε).
2. **Monotonicity**: L_mono = Σ_t ReLU(M_t − M_{t+1}) · 1[Δt < θ].
3. **Gain Sparsity & Positivity**: L_gain_sparse = Σ_t ||G_t||_1; L_gain_neg = Σ_t ReLU(−G_t).
4. **Variance Normalization**: L_var = Σ_s (Var_t(G_{t,s}) − v₀)².
5. **Rank Alignment**: L_rank = 1 − τ(M_students, P_students) (Kendall τ between average mastery and performance probabilities).
6. **Correlation Targeting**: L_corrM = (ρ_target − ρ_mastery)^2; L_corrG similarly.
7. **Dropout Consistency**: Forward twice; L_cons = ||M^(a) − M^(b)||².
8. **Trend Filtering**: L_trend = λ_tf Σ_t |M_{t+2} − 2 M_{t+1} + M_t|.
9. **Channel Separation** (two gain channels): L_sep = cos_sim(G_immediate, G_consolidated) minimized.

Composite loss:
L = L_perf + λ_mperf L_master_perf + λ_gperf L_gain_perf + λ_orth L_orth + λ_mono L_mono + λ_rank L_rank + λ_sparse L_gain_sparse + λ_var L_var + λ_corrM L_corrM + λ_corrG L_corrG + λ_cons L_cons + λ_trend L_trend + λ_sep L_sep.

## 6. Dynamic Loss Scheduling
Adaptive scaling of λ_mperf, λ_gperf to steer correlations to targets without sacrificing AUC:
λ_mperf^{t+1} = λ_mperf^{t} ( (ρ_target) / (ρ_mastery(t)+ε) )^α; α ∈ [0.3, 0.7].
Freeze escalation if ΔAUC < δ_auc_min while Δρ_mastery marginal (< δ_corr_gain). Maintain moving average smoothing over last n epochs.

## 7. Contrastive and Alignment Objectives
1. **Skill Prototype Alignment**: InfoNCE pulling mastery slices toward active skill prototypes and away from others.
2. **Temporal Gain Contrast**: Positive pair (G_t, ΔPerformance_{t+1}); negatives are mismatched time steps.
3. **Hard Negative Mining**: Focus on interactions where predicted mastery high but outcome incorrect to refine boundaries between mastery and gain.

## 8. Causal and Invariance Probes
- Regression probes for causal coefficients (α, β, γ, δ) with stability tracking across epochs and perturbed batches (skill masking, temporal shuffling).
- Counterfactual simulation: replace Mastery_t with Mastery_{t−Δ} keeping Gain_t; measure predicted performance delta (quantifying mastery causal contribution).
- Invariance audit: correlation shifts under distribution perturbations (time compression, difficulty stratification).

## 9. Calibration Techniques
Apply temperature scaling to predicted probabilities to reduce miscalibration that attenuates linear relationships with mastery. Post‑hoc isotonic regression on mastery vs empirical success rate to detect local non‑monotonic regions (report deviation curve and integrate a penalty if severe).

## 10. Temporal Modeling Strategies
Smoothing mastery (trend filtering) while leaving gain unsmoothed preserves the interpretive dichotomy: mastery = stable knowledge; gain = marginal learning increment. Multi‑resolution time features (short‑gap, medium‑gap, long‑gap embeddings) help separate recency effects from cumulative mastery.

## 11. Disentanglement Approaches
Group latent dimensions: mastery block, gain block, retention block. Penalize cross mutual information via contrastive predictive coding (CPC) negatives. Evaluate semantic purity: mutual information of each latent dimension with skill labels and difficulty strata.

## 12. Evaluation Protocol
For each added mechanism:
1. Perform an ablation (on vs off) recording ΔAUC, Δρ_mastery, Δρ_gain.
2. Multi‑seed (≥5) distribution: mean, std, coefficient of variation.
3. Overfit metrics: peak vs final correlation drop; correlation retention ratio ρ_final / ρ_peak.
4. Robustness: δρ under noise injection; target |δρ| < threshold.
5. Causal coefficient stability: variance of probe coefficients across seeds.
6. Report all metrics in standardized experiment folder (see reproducibility guidelines). Avoid reuse of folder names unless resuming.

## 13. Implementation Roadmap (Prioritized)
Immediate:
- Logging: partial & lagged correlations, Kendall τ, stability index.
- Introduce orthogonality, monotonicity, rank alignment losses.
- Add adaptive λ scaling controller.
Short‑Term:
- Prototype alignment (InfoNCE) and temporal gain contrast.
- Trend filtering regularizer.
- Causal regression probe with coefficient logging.
Medium‑Term:
- Dual‑path attention, retention cell, hierarchical skill embeddings.
- Dual gain channel separation.
Advanced:
- Counterfactual simulation, invariance audit suite.
- Mutual information disentanglement penalties.

## 14. Risk Analysis and Mitigations
Risk: Correlation maximization may degrade AUC (representation collapse). Mitigations:
- AUC guardrail: halt or roll back increased λ if AUC slope negative for p consecutive epochs.
- Minimum entropy constraint on mastery distribution.
- Monitor correlation retention ratio; avoid excessive smoothing causing delayed responsiveness.

Risk: Over‑regularization reduces adaptability on sparse skills. Mitigation: skill‑adaptive λ scaling (higher weights only for skills with sufficient interaction counts).

Risk: Prototype misalignment due to poor initial embeddings. Mitigation: initialize prototypes via average of early mastery vectors or curriculum metadata embeddings.

## 15. Documentation and Reporting Standards
Each experiment report must:
- Explicitly list enabled interpretability mechanisms and λs.
- Provide a table of metrics (AUC, accuracy, MPC, GPC, lagged MPC/GPC, stability index, elasticity, calibration error).
- Describe qualitative trajectory examples (mastery vs performance evolution for selected students).
- Include reproducibility checklist.
- Cite experiment folder path and config hash.

## 16. Future Extensions
- Bayesian hierarchical priors over mastery dynamics (student‑level random effects) for clearer variance attribution.
- Skill clustering diagnostics to merge latent dimensions with low purity.
- Educational outcome linkage (e.g., long‑term retention tests) for external validity of mastery semantics.

## 17. Summary
We propose a comprehensive toolkit to enhance semantic interpretability while retaining predictive strength. The approach centers on measurement expansion, architectural disentanglement, targeted regularization, adaptive loss scheduling, contrastive alignment, causal probing, and rigorous evaluation. By integrating these components systematically and documenting them within the established reproducibility framework, we enable credible, stable, and causally meaningful mastery and gain representations suitable for scholarly dissemination.
