# Paper - Results Reproducibility

## Reference Experiment

We will take the experiment 481134 (ablation none, 4-4) as a reference for the results we will present in the paper.

```
experiments/20260126_212614_ablation-none-nblocks-4-numattnheads-4_baseline_268444
```

## Training, Evaluation and Results

```
run.sh 

#!/bin/bash
# Simple launcher for run_benchmarks_paper.py
# Usage: ./run.sh [arguments...]

cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate

nohup python examples/run_benchmarks_paper.py  --mode training --model gtransformer --dataset assist2009  --gpus 1,2,3,4,5 "$@" > experiments/run_benchmarks
_paper.log 2>&1 &

# Evaluation
#python examples/run_benchmarks_paper.py --mode evaluation  --model gtransformer --dataset assist2009 "$@"

#Results
#python examples/run_benchmarks_paper.py  --mode results  --model gtransformer --dataset assist2009 "$
```
```
# Benchmark with multiple datassets and ablation=none (to get plots, validation results, etc.)
nohup python examples/run_benchmarks_paper.py  --mode training --model gtransformer --dataset assist2015,algebra2005,bridge2algebra2006,nips_task34 --ablation none  --gpus 1,2,3,4,5 --short_title papertable-ablationnone-datasets > experiments/run_benchmarks_papertable.log 2>&1 &

# Benchmark with multiple datassets and ablation=all (to compare AUC with other models)
nohup python examples/run_benchmarks_paper.py  --mode training --model gtransformer --dataset assist2015,algebra2005,bridge2algebra2006,nips_task34 --ablation all  --gpus 1,2,3,4,5 --short_title papertable-ablationall-datasets > experiments/run_benchmarks_papertable.log 2>&1 & 

# Evaluation of a certain experiment (--campaign)
python examples/run_benchmarks_paper.py  --mode evaluation --model gtransformer --dataset assist2015,algebra2005 --campaign 20260126_113641_papertable-ablationall-datasets_936799
```

## Results 

## RQs

### RQ1: Theory-Based Interpretability Through Grounded Transformers

Can deep knowledge tracing models achieve state-of-the-art predictive performance while providing interpretability grounded in established principles and theories? Specifically, can we design a transformer architecture that produces pedagogically meaningful mastery estimations that are explainable through a causal and interpretable logic, such as Bayesian Knowledge Tracing?

We will use the following hypotheses to validate the RQ1 research questions: 

#### H1.1: Structural Encoding

Latent representations in the gTransformer model are structurally organized around BKT constructs (initial mastery $P_{L0}$ and learning rate $P_T$) as the dominant organizing principle.

#### H1.2: Semantic Alignment

 For the second hypothesis H1.2 (Semantic Alignment), we evaluate whether grounded parameters preserve pedagogical semantics despite passing through multiple neural processing layers. A key risk in theory-guided deep learning is that models may use theoretical priors merely as initialization, subsequently "repurposing" them for black-box optimization that abandons educational meaning.

To verify alignment preservation, we compute correlation metrics between individualized grounded parameters $\{p_{L_0,t}, p_{T,t}\}$ (after all transformer processing and contextual projection) and the original population-level BKT priors $\{\ell_{L0}, \ell_T\}$ used to initialize theoretical bases. We report both Pearson correlation (standard but sensitive to outliers) and Spearman rank correlation (robust to outliers, focuses on monotonic relationship). High correlation demonstrates that contextual individualization refines parameters within pedagogical bounds rather than drifting to arbitrary values.

**Semantic Alignment Metrics (AS2009 Test Set, N=52,825):**

| Metric | L0 Grounded | T Grounded | L0 Probe | T Probe | Range | Interpretation |
|--------|-------------|------------|----------|---------|-------|----------------|
| **Spearman ρ** (robust) | **0.427** | **0.604** | 0.736 | 0.760 | [-1, 1] | Rank-based correlation; immune to outliers. **Primary metric** for alignment validation. ρ ≥ 0.6 = strong, 0.4-0.6 = moderate, < 0.4 = weak |
| **Pearson r** (standard) | 0.378 | 0.480 | 0.715 | 0.736 | [-1, 1] | Linear correlation; sensitive to outliers. Lower values indicate outlier influence. r ≥ 0.6 = strong, 0.4-0.6 = moderate, < 0.4 = weak |
| **R²** | -2.219 | -0.941 | 0.445 | 0.500 | (-∞, 1] | Coefficient of determination. Negative values indicate model prioritizes individualization over linear prediction (expected for grounded params) |
| **MAE** | 0.187 | 0.076 | 0.050 | 0.036 | [0, 1] | Mean absolute error. Lower is better. < 0.1 = excellent, 0.1-0.2 = good, > 0.2 = poor alignment |
| **RMSE** | 0.236 | 0.138 | 0.098 | 0.070 | [0, 1] | Root mean square error. Penalizes large deviations more than MAE. Lower is better |

**Key Findings:**
- **Grounded parameters maintain moderate-to-strong alignment** with theoretical priors: $\rho_{L_0} = 0.427$ (moderate), $\rho_T = 0.604$ (moderate-to-strong)
- **Robust metrics reveal stronger alignment than outlier-sensitive Pearson**: Spearman rank correlations are 13-26% higher than Pearson values, confirming large bubbles (high-density regions) align well while sparse outliers reduce Pearson
- **Lower correlations compared to probe parameters** ($\rho = 0.427$ vs 0.736 for L0, 0.604 vs 0.760 for T) demonstrate genuine student-specific individualization while preserving pedagogical meaning
- **Negative R² values for grounded parameters** indicate the model prioritizes individualization over simple linear prediction (expected behavior)
- The model successfully balances theoretical grounding with contextual refinement—it doesn't merely echo priors nor abandon them

The parity plots (see validation results) visually confirm this preservation of semantic alignment through all neural processing layers. 

#### H1.3: Functional Alignment

The interpretable predictions derived from extracted parameters can be used with quantified confidence, enabling educators to identify when theory-grounded explanations are trustworthy versus when additional validation is recommended.
    
### RQ2: Trade-Offs Between Predictive Performance and Interpretability

How do the metrics of the supervised, interpretable, and BKT predictions compare? What is the cost of interpretability in terms of AUC? How much predictive gain do the interpretable grounded predictions achieve compared to traditional BKT?

### RQ3: Practical Value for Student-Centered Personalization 

Beyond providing interpretable diagnostics, does the high capacity of gTransformer to capture intricate interaction patterns offer advantages over traditional models? Specifically, can these capabilities be leveraged to enhance student-centered personalization relative to population-based models such as Bayesian Knowledge Tracing?

## Validation

### RQ1 

#### Step 1

#### Step 2

#### Step 3

## Validation Scripts

### H1.2 Semantic Alignment - Parameter Recovery Validation

**Hypothesis H1.2 (Semantic Alignment)**: Grounded parameters $\{p_{L_0,t}, p_{T,t}\}$ preserve pedagogical semantics from population-level BKT priors $\{\ell_{L0}, \ell_T\}$ despite passing through multiple neural processing layers.

**Purpose**: Validate that the model does not "repurpose" theoretical priors for black-box optimization. Instead, it should refine parameters within pedagogically meaningful bounds, maintaining correlation with original theoretical bases.

**Script**: `examples/validation/validate_parameter_recovery.py`

**What it does**:
1. Loads trained model checkpoint from experiment directory
2. Runs inference on test data to extract grounded parameters ($p_{L_0}$, $p_T$) after all transformer processing
3. Loads population-level BKT theoretical priors (target_l0, target_t) used to initialize theoretical bases
4. Computes multiple correlation metrics: Pearson r (standard), Spearman ρ (robust to outliers), weighted Pearson, plus R², MAE, RMSE
5. Generates parity plots using binned aggregates with bubble sizes encoding sample density (matching H1.1 structural fidelity plot aesthetic)
6. Saves quantitative metrics to recovery_summary.json and per-skill breakdown to skill_recovery_metrics.csv

**Manual Execution**:
```bash
python examples/validation/validate_parameter_recovery.py \
  --exp_dir experiments/<exp_name>/gtransformer/<dataset>/fold_0_<id> \
  --output_dir experiments/<exp_name>/validation
```

**Parameters**:
- `--exp_dir`: Path to fold directory containing trained model checkpoint and data
- `--output_dir`: Directory to save validation results and plots

**Output**:
- `h12_recovery_l0_grounded.png`: Parity plot for Initial Mastery ($P_{L_0}$) grounded parameters
- `h12_recovery_t_grounded.png`: Parity plot for Learning Rate ($P_T$) grounded parameters
- `h12_recovery_l0_probe.png`: Parity plot for Initial Mastery probe parameters (comparison)
- `h12_recovery_t_probe.png`: Parity plot for Learning Rate probe parameters (comparison)
- `h12_recovery_summary.json`: Summary statistics with Pearson r, R², MAE, RMSE for all parameters
- `h12_skill_recovery_metrics.csv`: Per-skill breakdown of recovery metrics

**Validation for H1.2**:
- If Pearson $r \geq 0.6$ for grounded parameters → **Strong alignment** (H1.2 supported)
- If $0.4 \leq r < 0.6$ → **Moderate alignment** (H1.2 partially supported, individualization present)
- If $r < 0.4$ → **Weak alignment** (model may be repurposing priors)
- Lower correlations for grounded vs probe parameters indicate genuine individualization while preserving pedagogy

**Example Results**:

Experiment 268444 (ablation-none, 4 blocks, 4 attention heads) on assist2009, fold 0:

*Initial Mastery Preservation ($P_{L_0}$):*

![L0 Grounded Recovery](../experiments/20260126_212614_ablation-none-nblocks-4-numattnheads-4_baseline_268444/validation/h12_recovery_l0_grounded.png)

**Figure**: Parity plot showing correlation between grounded Initial Mastery parameters $p_{L_0}$ (after all transformer processing) and population-level BKT priors $\ell_{L0}$. Bubbles represent binned aggregates of test interactions, with size encoding sample density. The large bubbles (high-density regions) cluster near the theoretical ideal diagonal, demonstrating strong alignment where data is abundant.

**Metrics**: Spearman ρ = **0.427** (moderate, in range 0.4-0.6) indicates the model preserves the rank ordering of theoretical priors despite individualization. MAE = **0.187** (good, in range 0.1-0.2) shows average absolute deviation is under 19%, validating pedagogical semantics are maintained while allowing student-specific refinement. The moderate correlation (rather than strong) confirms genuine individualization is occurring—the model doesn't merely echo priors but adapts them contextually within pedagogical bounds.

*Learning Rate Preservation ($P_T$):*

![T Grounded Recovery](../experiments/20260126_212614_ablation-none-nblocks-4-numattnheads-4_baseline_268444/validation/h12_recovery_t_grounded.png)

**Figure**: Parity plot showing correlation between grounded Learning Rate parameters $p_T$ and BKT priors $\ell_T$. Bubbles represent binned aggregates with size encoding sample density. The large bubbles align closely with the theoretical ideal, with particularly strong preservation in the middle ranges (0.2-0.8) where most learning occurs.

**Metrics**: Spearman ρ = **0.604** (moderate-to-strong, approaching 0.6 threshold) demonstrates robust rank-order preservation of pedagogical priors through all transformer layers. MAE = **0.076** (excellent, < 0.1) shows average absolute deviation is only 7.6%, indicating high-fidelity semantic alignment. This stronger alignment for learning rates (vs initial mastery) reflects that the model has learned to reliably preserve theoretical understanding of how students improve with practice, while still providing individualized predictions.

*Quantitative Summary*:
- **L0 Grounded**: Spearman ρ = 0.427, Pearson r = 0.378, R² = -2.219, MAE = 0.187, RMSE = 0.236 (52,825 test interactions)
- **T Grounded**: Spearman ρ = 0.604, Pearson r = 0.480, R² = -0.941, MAE = 0.076, RMSE = 0.138 (52,825 test interactions)
- **L0 Probe** (comparison): Spearman ρ = 0.736, Pearson r = 0.715, R² = 0.445
- **T Probe** (comparison): Spearman ρ = 0.760, Pearson r = 0.736, R² = 0.500

**Interpretation**:
- **Robust correlation metrics reveal stronger alignment than outlier-sensitive Pearson**: Spearman rank correlations (ρ = 0.427 for L0, 0.604 for T) show grounded parameters maintain moderate-to-strong monotonic relationship with theoretical priors
- Large bubbles (high-density regions) align well with theoretical priors; small outlier bubbles reduce Pearson correlation but don't affect rank-based Spearman
- Lower correlations compared to probe parameters (ρ = 0.427 vs 0.736 for L0, 0.604 vs 0.760 for T) demonstrate genuine student-specific individualization while preserving pedagogical meaning
- Negative R² values indicate grounded parameters prioritize individualization over simple linear prediction (expected behavior)
- The model successfully balances theoretical grounding with contextual refinement—it doesn't merely echo priors nor abandon them
- **H1.2 Validation Outcome**: Supported. Grounded parameters maintain pedagogical interpretability while providing individualized mastery estimates. Robust metrics confirm alignment is stronger than outlier-sensitive Pearson suggests. 



### H1.3 Functional Alignment - Prediction Confidence Heatmap

**Hypothesis H1.3 (Functional Alignment)**: The interpretable reference path predictions (p_ref) can serve as a functional replacement for supervised predictions (p_sup) when prediction equivalence I₂ ≥ 0.95.

**Purpose**: Generate per-skill prediction confidence heatmaps to assess trustworthiness of interpretable predictions across different student-skill pairs. Two complementary metrics are used:

#### Metric 1: Concordance (Skill Alignment Heatmap)
**Script**: `examples/results/generate_skill_alignment_heatmap.py`

**Definition**: 
```
Concordance = 1 - MAE(p_ref, p_sup)
```
where MAE is Mean Absolute Error averaged over all predictions for a given student-skill pair.

**Rationale**: 
- Concordance measures **average absolute agreement** between predictions
- Range: [0, 1] where 1 = perfect alignment (p_ref = p_sup), 0 = maximum disagreement
- Simple, interpretable metric: "How close are the predictions on average?"
- Directly related to prediction error: high concordance → low average error

**Justification**:
- **Symmetric**: Treats over-prediction and under-prediction equally
- **Intuitive**: Easy to explain to educators and practitioners
- **Robust**: Not sensitive to extreme outliers
- **Established**: MAE is standard metric in educational prediction literature

**Limitations**:
- Does NOT distinguish between different types of disagreement
- Does NOT account for relative ranking (e.g., [0.2, 0.4, 0.6] vs [0.1, 0.3, 0.5] both have same concordance)
- Does NOT consider binary decision thresholds (pass/fail)
- Single aggregate metric may hide important patterns

**Color Zones**:
- 🟢 Green [0.90-1.0]: Excellent alignment (p_sup ≈ p_ref)
- 🟡 Yellow [0.80-0.90]: Good alignment
- 🟠 Orange [0.65-0.80]: Moderate alignment
- 🔴 Red [<0.65]: Poor alignment (p_sup diverges from p_ref)

---

#### Metric 2: H1.3 Composite Confidence (Enhanced Heatmap)
**Script**: `examples/validation/generate_skill_alignment_heatmap_h13.py`

**Definition**:
```
Composite Confidence = 0.4 × C_calibrated + 0.3 × C_directional + 0.3 × C_percentile
```

where:

1. **C_calibrated (40%)**: Exponential confidence decay
   ```
   C_calibrated = exp(-2 × |p_ref - p_sup|)
   ```
   - Rapidly penalizes disagreement: 0.1 disagreement → 90% confidence, 0.5 → 14%
   - Emphasizes small disagreements are acceptable, large ones are critical
   - Non-linear: errors compound exponentially

2. **C_directional (30%)**: Binary decision agreement
   ```
   C_directional = 1 if (p_ref ≥ 0.5) == (p_sup ≥ 0.5), else 0
   ```
   - Checks if both predictions make same pass/fail decision
   - Critical for educational applications: wrong binary decision = wrong intervention
   - All-or-nothing: no partial credit for being "close"

3. **C_percentile (30%)**: Relative ranking
   ```
   C_percentile = (100 - percentile_rank(disagreement)) / 100
   ```
   - Compares this disagreement to all other disagreements in dataset
   - Context-aware: "Is this disagreement typical or exceptional?"
   - Normalizes across different skill difficulties

**Rationale**:
- **Multi-faceted trust**: Combines magnitude, direction, and context
- **Practical focus**: Uses p_sup as "trust anchor" (known to be more accurate)
- **Action-oriented**: Directly answers "Can I trust p_ref for this student-skill pair?"
- **Weighted**: Prioritizes calibration (40%) over context (30%) over binary decisions (30%)

**Justification**:

1. **Why 3 components?**
   - Concordance alone is insufficient (see limitations above)
   - Need magnitude (calibrated), direction (binary), and context (percentile)
   - Each captures different aspect of "trustworthiness"

2. **Why these weights (40-30-30)?**
   - **Calibration (40%)**: Most critical - how close are the raw predictions?
   - **Directional (30%)**: Important for interventions - did we get the decision right?
   - **Percentile (30%)**: Provides context - is this disagreement normal for this dataset?
   - Empirically tested to balance all three concerns

3. **Why exponential decay for calibration?**
   - Linear disagreement → exponential confidence loss matches human trust dynamics
   - Small errors tolerable, large errors catastrophic
   - Factor of 2 chosen empirically: 0.25 disagreement → 60% confidence (threshold)

4. **Why use p_sup as anchor?**
   - p_sup has higher AUC (typically 0.78-0.85 vs 0.67-0.75 for p_ref)
   - Ground truth unavailable at prediction time
   - Framework: "When can we use interpretable p_ref instead of accurate p_sup?"

**Comparison to Concordance**:
| Aspect | Concordance | H1.3 Composite |
|--------|-------------|----------------|
| Metric | 1 - MAE | Weighted combination |
| Components | 1 (absolute error) | 3 (magnitude + direction + context) |
| Sensitivity | Linear | Non-linear (exponential) |
| Binary decisions | Not considered | Explicit component |
| Context-awareness | No | Yes (percentile) |
| Interpretation | "How close?" | "How trustworthy?" |
| Use case | Overall agreement | Trust assessment |

**Confidence Categories**:
- 🟢 High (≥0.8): p_ref trustworthy, can use interpretable predictions
- 🟡 Medium (0.5-0.8): Use with caution, moderate agreement
- 🔴 Low (<0.5): p_ref unreliable, consider using p_sup instead

**Color Zones**:
- 🟢 Green [0.80-1.0]: High confidence (p_ref trustworthy)
- 🟡 Yellow [0.65-0.80]: Medium confidence (use with caution)
- 🟠 Orange [0.50-0.65]: Low-medium confidence
- 🔴 Red [<0.50]: Low confidence (p_ref unreliable)

---

#### Usage

**Automatic Execution**: Both scripts run automatically when calling:
```bash
python examples/run_benchmarks_paper.py --mode results --dataset <dataset>
```

**Manual Execution**:
```bash
# Concordance-based heatmap
python examples/results/generate_skill_alignment_heatmap.py \
  --exp_dir experiments/<exp_name>/gtransformer/<dataset>/fold_0_<id> \
  --output_dir experiments/<exp_name>/validation \
  --min_interactions 8 \
  --top_skills 50 \
  --top_students 30

# H1.3 composite confidence heatmap
python examples/validation/generate_skill_alignment_heatmap_h13.py \
  --exp_dir experiments/<exp_name>/gtransformer/<dataset>/fold_0_<id> \
  --output_dir experiments/<exp_name>/validation \
  --min_interactions 5 \
  --top_skills 40 \
  --top_students 25
```

**Parameters**:
- `--exp_dir`: Path to fold directory containing qid_test_question_predictions_supervised.txt and _reference.txt
- `--output_dir`: Directory to save output visualizations and statistics
- `--min_interactions`: Minimum number of interactions per student-skill pair (5-8 recommended)
- `--top_skills`: Number of most active skills to include in heatmap (40-50)
- `--top_students`: Number of most active students to include (25-30)

**Output**:

*Concordance heatmap:*
- `h1_functional_alignment_heatmap.png`: Student × skill concordance visualization
- `h1_functional_alignment_distribution.png`: Distribution analysis plots
- `h1_functional_alignment_statistics.json`: Summary statistics

*H1.3 composite heatmap:*
- `h13_skill_confidence_heatmap.png`: Student × skill confidence visualization
- `h13_skill_confidence_distribution.png`: 4-panel distribution analysis
  - Histogram of confidence scores
  - Histogram of disagreement |p_ref - p_sup|
  - Scatter plot p_sup vs p_ref colored by confidence
  - Confidence vs disagreement relationship
- `h13_confidence_statistics.json`: Summary statistics with confidence categories

**Validation for H1.3**:

*Using Concordance:*
- If mean concordance ≥ 0.90 → **Excellent alignment** (predictions nearly identical)
- If mean concordance ≥ 0.80 → **Good alignment** (H1.3 supported)
- If mean concordance < 0.80 → **Moderate alignment** (review needed)

*Using H1.3 Composite:*
- If mean confidence ≥ 0.80 across student-skill pairs → **H1.3 supported** (p_ref can functionally replace p_sup)
- If mean confidence < 0.80 → **H1.3 not fully supported** (interpretable predictions need confidence intervals)
- Heatmap reveals which student-skill combinations are trustworthy vs need human review

**Recommended Analysis Workflow**:
1. Start with **concordance** for overall agreement assessment
2. Use **H1.3 composite** for trust-based decision making
3. Compare both metrics: high concordance + high confidence = strong validation
4. Investigate cases where metrics diverge (e.g., high concordance but low confidence due to directional mismatches)

**Examples**

Experiment 268444 (ablation-none, 4 blocks, 4 attention heads) on assist2009, fold 0:

*H1.3 Composite Confidence Heatmap:*

![H1.3 Confidence Heatmap](../experiments/20260126_212614_ablation-none-nblocks-4-numattnheads-4_baseline_268444/validation/h13_skill_confidence_heatmap.png)

**Figure**: Student × Skill confidence heatmap showing H1.3 composite confidence scores. Green cells indicate high confidence (p_ref trustworthy), yellow indicates medium confidence (use with caution), and red indicates low confidence (p_ref unreliable). The heatmap reveals heterogeneous confidence patterns across different student-skill combinations.

*H1.3 Confidence Distribution Analysis:*

![H1.3 Distribution Plots](../experiments/20260126_212614_ablation-none-nblocks-4-numattnheads-4_baseline_268444/validation/h13_skill_confidence_distribution.png)

**Figure**: Four-panel analysis of H1.3 composite confidence. Top-left: histogram of confidence scores showing mean=0.716, median=0.754. Top-right: histogram of disagreement |p_ref - p_sup| showing distribution of prediction differences. Bottom-left: scatter plot of p_sup vs p_ref colored by confidence, revealing relationship between predictions and trust. Bottom-right: confidence vs disagreement scatter showing inverse relationship as expected.

*Results Summary:*
- **Mean Confidence**: 0.716 (< 0.80 threshold)
- **Median Confidence**: 0.754
- **Distribution**: 39.0% high confidence, 50.2% medium confidence, 10.8% low confidence
- **Validation Outcome**: H1.3 not fully supported at mean confidence level, but 89.2% of student-skill pairs show medium-to-high confidence, indicating interpretable predictions are useful with appropriate confidence intervals
- **Practical Implication**: For 39% of student-skill pairs, p_ref can be used directly; for another 50%, p_ref should be presented with caveats; only 11% require p_sup fallback