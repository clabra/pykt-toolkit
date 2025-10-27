# Interpretability Framework for GainAKT2Exp

## 1. Interpretability Approach

We pursue a balance between predictive performance (AUC / accuracy) and semantic interpretability.  

### Semantic Requirements

To be considered interpretable and explainable, the model should adhere to the following semantic requirements:

- **Monotonicity of Mastery**: A student's mastery of a skill should not decrease over time. An interaction can result in zero gain, but never a negative gain.
- **Non-Negative Learning Gains**: Skill mastery increases through cumulative learning gains resulting from interactions. Each learning must be greater than or equal to zero.
- **Mastery-Performance Correlation**: The likelihood of answering a question correctly increases with the model’s estimated mastery of the linked skills, as mapped by the Q-matrix.
- **Gain-Performance Correlation**: Correct answers should, on average, reflect greater learning gains for the linked skills than incorrect ones.
- **Sparsity of Gains (Desirable)**: The learning gain from an interaction with a specific question should primarily involve only the skills linked to that question.

### Heads

Interpretability in GainAKT2Exp is grounded in two latent heads:

 - **Mastery head (M)**: a cumulative latent knowledge representation intended to monotonically reflect a learner's long-run competence; it should preserve relative ordering among students' historical performance profiles and remain stable under minor sequence perturbations.
 - **Gain head (G)**: a prospective latent signal estimating expected incremental improvement conditional on the next interaction; it emphasizes short-horizon responsiveness and should correlate with near-term performance deltas without collapsing into noise.

Both heads are constrained by **architectural guarantees (non-negativity, bounded outputs)** and **shaped via loss functions**.

### Loss Functions 

We define proper loss functions to force the model to be aligned with semantic requirements. To this end, the objective is decomposed into predictive, structural, and semantic components. 

Let \(y_i^t\in\{0,1\}\) be correctness, \(\hat{p}_i^t\) the predicted probability, \(M_i^t\) mastery, \(G_i^t\) gain, \(C_i^t\) a cumulative performance proxy (e.g., running average correctness), and \(\Delta_i^t\) a short-horizon performance delta. 

The full epoch objective is:
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{corrM}}\,\mathcal{L}_{\text{corrM}} + \lambda_{\text{corrG}}\,\mathcal{L}_{\text{corrG}} + \lambda_{\text{align}}\,\mathcal{L}_{\text{align}} + \lambda_{\text{ret}}\,\mathcal{L}_{\text{ret}} + \lambda_{\text{lag}}\,\mathcal{L}_{\text{lag}} + \lambda_{\text{mon}}\,\mathcal{L}_{\text{mon}} + \lambda_{\text{sparse}}\,\mathcal{L}_{\text{sparse}} + \lambda_{\text{cons}}\,\mathcal{L}_{\text{cons}}.
\]

Core components (abbreviated formulations):
1. Predictive loss \(\mathcal{L}_{\text{pred}}\): binary cross-entropy \(-\sum_{t,i} y_i^t\log \hat{p}_i^t + (1-y_i^t)\log(1-\hat{p}_i^t)\).
2. Mastery correlation loss \(\mathcal{L}_{\text{corrM}}\): \(-\operatorname{corr}(M^t, C^t)\) (implemented via differentiable centered cosine similarity) encouraging explanatory ordering.
3. Gain correlation loss \(\mathcal{L}_{\text{corrG}}\): \(-\operatorname{corr}(G^t, \Delta^t)\) aligning prospective signal with short-term improvement.
4. Alignment loss \(\mathcal{L}_{\text{align}}\): combines local/global sampled correlation objectives and (optionally) a residual term focusing on misaligned subsets; weight scheduled post warm-up.
5. Retention loss \(\mathcal{L}_{\text{ret}}\): penalizes correlation decay beyond tolerance \(\delta_{\text{ret}}\): \(\sum_e \max(0,(r^{*}-r^{(e)})-\delta_{\text{ret}})\) with running peak \(r^{*}\).
6. Lag gain loss \(\mathcal{L}_{\text{lag}}\): temporal coherence; discourages large gains before mastery rises. Surrogate: \(\sum_t \max(0, G_i^t - f(M_i^{t-1}))\) with monotone \(f\), or hinge on ordering between \(G\) and \(\Delta M\).
7. Monotonicity penalty \(\mathcal{L}_{\text{mon}}\): \(\sum_{t>0}\operatorname{ReLU}(M^{t-1}-M^{t})\) enforcing non-decreasing mastery.
8. Sparsity loss \(\mathcal{L}_{\text{sparse}}\): masked L1 outside Q-matrix linked skills: \(\sum_t \|(1-Q_t)\odot G^t\|_1\).
9. Cross-head consistency \(\mathcal{L}_{\text{cons}}\): couples gains to mastery increments: \(\sum_t |(M^t-M^{t-1}) - \gamma G^t|\) or correlation between \(G^t\) and \(\Delta M^t\).
10. Non-negativity (architectural): softplus/ReLU ensures \(G^t \ge 0\); explicit penalty optional.

Scheduling principles: 

(i) Warm-up epochs suppress \(\lambda_{\text{align}}, \lambda_{\text{ret}}, \lambda_{\text{lag}}\) to allow stable predictive calibration; 
(ii) Linear ramp introduces alignment once mastery variance decreases; 
(iii) Adaptive decay reduces \(\lambda_{\text{align}}\) if global CI lower bound stays sub-emerging while local upper bound exceeds threshold; 
(iv) Retention activates only after first confirmed emergence (CI lower bound > 0).

Gradient role partitioning: predictive loss anchors accuracy; structural penalties (monotonicity, sparsity, non-negativity) constrain feasible latent geometries; semantic losses (correlation, alignment, consistency) inject educational meaning; temporal robustness (retention, lag) shapes stability of interpretability rather than single-epoch spikes.

We cap alignment gradient share to avoid interpretability dominating predictive improvement, monitoring \( (\lambda_{\text{align}}\,\mathcal{L}_{\text{align}})/\mathcal{L}_{\text{total}} \).

Implementation note: Pearson correlation surrogates use centered covariance divided by product of centered norms with \(\epsilon\) stabilizer; Fisher z is applied post hoc for confidence intervals, not inside gradients.







## 2. Mastery & Gain Correlations: Basis for Semantic Interpretability
We quantify whether latent variables encode meaningful educational constructs by examining their Pearson correlation with observed performance signals. Let \(M_i^t\) denote the model's mastery estimate for student \(i\) at time \(t\) and \(G_i^t\) its gain estimate (expected incremental improvement). We compare:

- **Mastery correlation**: correlation between \(M_i^t\) (or an epoch-aggregated statistic) and empirical cumulative performance (e.g., average correctness or calibrated prediction accuracy). High positive values indicate that the latent mastery dimension preserves ordinal relationships among learners' actual knowledge levels.
- **Gain correlation**: correlation between \(G_i^t\) and short-term performance deltas (e.g., change in correctness probability over a recent window). This evaluates whether the gain head captures prospective learning responsiveness rather than static difficulty.

We adopt these correlations because (i) they provide scale-invariant, unit-free comparability across seeds and datasets, (ii) they are robust to monotonic transformations of latent scores, and (iii) they directly test explanatory fidelity: a latent dimension purporting to represent knowledge should co-vary with realized performance outcomes.

### Local vs Global Estimation
To mitigate sampling artefacts, we distinguish local and global correlation estimates:

- **Local correlations**: computed each epoch over a small random subset of students (size \(n_{local}\)); provide rapid optimization feedback with minimal computational overhead.
- **Global correlations**: computed over a larger, diverse subset (size \(n_{global}\), with \(n_{global} \gg n_{local}\)); assess generalization and guard against overfitting the local slice.

Interpretability maturation follows the pattern: early local emergence (mastery correlation \(>0.10\)), subsequent global convergence (global approaching local), and variance contraction. Persistent divergence (local \(\gg\) global) or negative global correlation after local emergence signals misalignment needing adjustment (e.g., alignment weight scheduling, retention relaxation).

### Statistical Considerations
Using Fisher's z-transform \(z = \tfrac{1}{2}\ln\frac{1+r}{1-r}\) with standard error \(SE_z = 1/\sqrt{n-3}\), small \(n_{local}\) yields wide confidence intervals; thus high early local correlation spikes may be unreliable. Increasing \(n_{local}\) primarily narrows interval width, whereas increasing \(n_{global}\) probes semantic robustness across heterogeneous interaction patterns.

Operational uses of Fisher's z in GainAKT2Exp:
1. Per-epoch confidence intervals: each sampled correlation (local and global) is transformed to \(z\), a 95% interval \(z \pm 1.96 SE_z\) is computed, then back-transformed to obtain lower/upper bounds recorded (planned addition) in `metrics_epoch.csv` and semantic trajectory artifacts.
2. Multi-seed aggregation: correlations from different seeds are first transformed to \(z\); we compute the arithmetic mean \(\bar{z}\) and then back-transform to a pooled correlation \(\bar{r}\). This prevents bias from direct averaging of \(r\) when magnitudes are large.
3. Plateau detection (forthcoming): stability is declared when successive epochs' confidence intervals overlap and the absolute change in \(\bar{z}\) stays below a tolerance (e.g., 0.02) for \(K\) epochs post warm-up.
4. Adaptive objective triggers: if the global correlation's lower CI bound remains below zero while the local upper bound exceeds the emerging threshold (0.10) for \(\ge 3\) consecutive post warm-up epochs, we schedule a mid-run decay of alignment weight and relaxation of retention penalties.
5. Reproducibility criterion: a re-run is considered semantically consistent if its plateau epoch and final \(r\) lie within the original run's Fisher-transformed CI bounds after back-transformation.

This z-based protocol standardizes stability interpretation, mitigates overreaction to noisy early spikes, and provides statistically grounded triggers for adaptive loss scheduling.

### Practical Thresholds
Empirical working bands: 

emerging (\(0.10 \le r < 0.20\)), 
moderate (\(0.20 \le r < 0.30\)), 
strong (\(0.30 \le r < 0.40\)), 
very strong (\(\ge 0.40\), requires overfitting scrutiny). 

Gain correlations typically trail mastery by \(0.03{-}0.10\) due to short-term noise.

Here, \(r\) denotes the sample Pearson correlation between a latent head (mastery \(M\) or gain \(G\)) and its corresponding performance target (cumulative accuracy for mastery, short-term performance delta for gain). These threshold bands operationalize semantic emergence stages: values below 0.10 typically reflect noise; entering the emerging band signals initial alignment; moderate and strong bands indicate increasingly reliable explanatory ordering; very strong values (>0.40) require scrutiny for overfitting or collapse.

---

## 3. Correlation Estimation & Stability (Consolidated)
### Correlation Estimation: Local vs Global

We distinguish between local and global correlation estimates to balance rapid feedback with robustness:

- **Local correlations**: Computed each epoch over a small random subset of students (\(n_{local}\)). These provide quick optimization feedback with minimal computational overhead.
- **Global correlations**: Computed over a larger, diverse sample (\(n_{global}\), where \(n_{global} \gg n_{local}\)). These assess generalization and guard against overfitting the local slice.

#### Desired Progression
1. **Early Local Emergence**: Mastery correlation (\(r > 0.10\)) appears first in local estimates.
2. **Global Convergence**: Global correlations approach local values as training progresses.
3. **Variance Reduction**: Decreasing variance in both local and global estimates signals stabilization.

#### Practical Working Bands
- **Emerging**: \(0.10 \leq r < 0.20\)
- **Moderate**: \(0.20 \leq r < 0.30\)
- **Strong**: \(0.30 \leq r < 0.40\)
- **Very Strong**: \(r \geq 0.40\) (requires scrutiny for overfitting).

Gain correlations typically trail mastery correlations by \(0.03{-}0.10\) due to short-term noise.

#### Statistical Reliability
We use Fisher's z-transform to assess statistical reliability:
- Confidence intervals are derived to distinguish true semantic stabilization from transient spikes.
- Plateau detection and adaptive triggers are enabled based on these intervals (details in Section 2).

#### Temporal Stability Diagnostics
Temporal stability is judged using:
- Variance trends: Declining post-warm-up variance indicates stabilization.
- Slopes: Mastery slope near zero at a positive plateau reflects durable semantics.
- Retention penalty statistics: Moderate retention events signal healthy regularization.

#### Loss Contributions
Alignment, retention, lag, and correlation losses jointly target:
1. **Emergence**: Initial alignment of latent variables with performance signals.
2. **Sustained Plateau**: Long-term stability of semantic correlations.
3. **Avoidance of Over-Regularization**: Preventing excessive constraints that hinder learning dynamics.

- **Retention Loss**: Penalizes post-peak decay beyond a defined tolerance.
- **Lag Loss**: Enforces temporal coherence between mastery increases and gain realization.

A healthy stability pattern includes declining variance, mastery slope stabilization, and moderate retention events post-warm-up.

## 4. Empirical Results Summary (Current Runs)
| Run | Epochs | Seeds | Mean Best AUC | Mean Final Mastery Corr | Mean Final Gain Corr | Notes |
|-----|--------|-------|---------------|-------------------------|----------------------|-------|
| Smoke (initial) | 2 | 21,42 | 0.6875 | -0.00076 | 0.09298 | Early emergence; high gain corr for size; mastery near 0. |
| Extended (tuned) | 20 | 21,42,63 | 0.72095 | -0.00306 | 0.06361 | AUC improved; correlations regressed; frequent retention events. |

Observation: Increasing alignment weight and global sampling improved AUC but did not yield positive mastery correlation; gain correlation declined. Retention penalties (mean count ≈11 of 20 epochs) may have over-constrained upward semantic drift.

## 5. Diagnostic Indicators
| Indicator | Desired | Current (20e Mean) | Interpretation |
|-----------|---------|--------------------|---------------|
| Mastery corr > 0.20 | Yes | No (negative) | Semantic grounding failed to consolidate. |
| Gain corr > 0.15 | Yes | No (0.064) | Gain head under-aligned post warm-up. |
| Global vs local gap narrows | Yes | Not achieved | Possible over-regularization or mis-weighted alignment. |
| Variance decreasing | Yes | Mastery variance lower | Stability at poor level. |
| Retention penalties moderate (≤5) | Yes | 11 | Over-suppression of correlation growth. |

## 6. Remediation Plan
1. **Reduce Retention Pressure**: Lower retention weight (0.12 → 0.08) and increase delta (0.005 → 0.007) to allow exploratory correlation rises.
2. **Adjust Alignment Scheduling**: Maintain alignment_weight=0.30 but reduce share cap to 0.07 and increase decay factor to 0.85 (faster attenuation of alignment dominance). Enable residual alignment to focus on misaligned epochs selectively.
3. **Boost Head Predictive Anchoring**: Increase mastery/gain performance loss weights (0.8 → 0.9) after warm-up to directly reinforce correlation signals.
4. **Confidence Interval Logging**: Add per-epoch Fisher CI to distinguish true semantic stagnation from estimation noise.
5. **Early Divergence Alert**: If mastery corr remains ≤0 for 3 consecutive post-warm-up epochs with negative slope, adaptively decay alignment_weight mid-run.
6. **Semantic Sample Calibration**: Retain n_local=40; consider n_global increment (150 → 200) only after initial stabilization.

## 7. Reproducibility Linkage
Every interpretability result is tied to a uniquely hashed configuration (`config_md5`). Reproduction requires:
- Exact `config.json` argument set.
- Deterministic seeds recorded (primary + list).
- Preservation of per-epoch metrics (`metrics_epoch.csv`) and semantic trajectories (`artifacts/semantic_trajectory_seed*.json`).
A run is considered semantically reproducible only if re-execution regenerates correlation trajectories whose plateau epoch and final mean correlation lie within Fisher CI bounds of the original.

## 8. Future Work
- Plateau detection (epoch of stable mastery correlation change < threshold for K epochs).
- Cross-seed semantic convergence metric (e.g., average pairwise correlation difference trajectory).
- Causal probing: perturb input sequence ordering to assess robustness of latent mastery variance.

## 9. Guidelines
We adopt a precise statistical framing. Correlation magnitudes are contextualized with sample size and confidence intervals to prevent misinterpretation. Negative or near-zero correlations after convergence phases are treated as actionable diagnostic failures, not silently ignored.

## 10. Summary
The architecture provides a rich suite of interpretability constraints and temporal diagnostics. Initial extended training shows strong predictive improvement but insufficient semantic alignment; proposed remediation targets controlled relaxation of retention and refinement of alignment scheduling. Future iterations will incorporate statistical interval logging and adaptive objective modulation to achieve stable, meaningful latent educational representations.

## 11. Implementation Status (Design vs Current Model)
This section summarizes the interpretability design (as specified in `assistant/newmodel.md`) and the **current implementation state in `pykt/models/gainakt2_exp.py`**.

Implemented components:
- Dual-head structure (mastery and gain) with non-negative gain enforcement and cumulative mastery accumulation (scaled additive update + clamp).
- Core consistency losses: non-negative gains, monotonicity of mastery, mastery-performance coupling, gain-performance hinge, sparsity (mask outside Q-linked skill), and mastery–gain increment consistency.
- Exposure of internal sequences (`context_seq`, `value_seq`) and head projections via `forward_with_states` for monitoring.
- Factory creation (`create_exp_model`) with configurable weights for interpretability penalties.
- Basic monitoring hook mechanism (`set_monitor`, frequency-based invocation) enabling external logging.

Not yet implemented / pending per design specification:
- Explicit learning gain semantics inside attention weights (current model treats value stream as embeddings; learning gains are head outputs rather than raw attention values).
- Q·K skill similarity-specific gain attribution audit (design calls for tracing gains through attention matrix; current code does not return attention weights nor a gain attribution map).
- Dynamic G-matrix estimation (learning gains mapped to a persistent matrix structure) distinct from per-step projected gains.
- Fisher z-based confidence interval logging, plateau detection, and adaptive scheduling triggers (referenced in interpretability plan; not in `gainakt2_exp.py`).
- Retention and lag-specific losses (currently only monotonicity/consistency; retention decay penalty and lag coherence loss absent).
- Residual alignment scheduling and alignment share cap logic (managed externally in training scripts, not integrated in model code).
- Cross-seed semantic convergence metrics (aggregation logic handled outside; model only supplies per-seed states).
- Causal perturbation probes (sequence order robustness tests) and skill transfer visualization utilities.

Partial or simplified implementations:
- Mastery accumulation uses a fixed scaling factor (0.1); design envisions adaptive scaling conditioned on context/difficulty.
- Gain-performance loss uses mean difference hinge with margin; design suggests correlation-based (Pearson surrogate) objective for scale invariance.
- Sparsity uses masked L1; design anticipates potential structured sparsity (group lasso) for multi-skill interactions.

Near-term priorities to align with design:
1. Return attention weights and compute per-head gain attribution matrices for traceable causal explanations.
2. Introduce Fisher z CI logging and adaptive triggers into training loop (retain model interface unchanged).
3. Add retention and lag losses consistent with documented formulations; expose weight hyperparameters.
4. Replace fixed mastery gain scaling with learnable or context-conditioned factor.
5. Implement dynamic G-matrix snapshot export for longitudinal interpretability artifacts.

This status ensures clarity on current capabilities (structural consistency + dual heads) versus planned causal and statistical interpretability enhancements required for the full paper contribution.
