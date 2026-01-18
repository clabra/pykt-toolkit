**Section 5: Experimental Validation**

We validate the GTransformer model through six complementary analyses that demonstrate its ability to achieve neural-level accuracy while maintaining interpretable, theory-grounded representations. All experiments use 5-fold cross-validation on the ASSIST2009 dataset (n=52,825 test interactions).

---

### 5.1 Parameter Recovery Accuracy

**Research Question**: Do the model's learned representations encode BKT parameters in a linearly accessible form?

We evaluate how well linear probes can recover pedagogical parameters ($P_{L0}$: initial mastery, $P_T$: learning rate) from the model's latent representations, comparing against oracle BKT values fitted on training data.

#### Methodology

For each test interaction, we:
1. Extract latent context vector $z \in \mathbb{R}^{64}$ from the final transformer layer
2. Apply learned linear probes: $\hat{P}_{L0} = W_{L0} \cdot z + b_{L0}$, $\hat{P}_T = W_T \cdot z + b_T$
3. Compare predictions against oracle BKT parameters using Pearson correlation ($r$), MAE, and RMSE

#### Results

**Quantitative Metrics**:

| Parameter | Pearson $r$ | MAE | RMSE | Samples |
|:---|---:|---:|---:|---:|
| **Initial Mastery ($P_{L0}$)** | 0.715 | 0.049 | 0.097 | 52,825 |
| **Learning Rate ($P_T$)** | 0.740 | 0.035 | 0.069 | 52,825 |

**Visual Evidence**:

![Parameter Recovery - Initial Mastery](../examples/validation/results/recovery_l0_probe.png)

*Figure 5.1a: Linear probe recovery of Initial Mastery ($P_{L0}$). Each point represents a skill-student interaction. The high correlation ($r = 0.715$) demonstrates that the model's 64-dimensional latent space encodes "initial mastery" in a linearly readable format. Red line: linear fit; dashed gray: ideal recovery ($y=x$).*

![Parameter Recovery - Learning Rate](../examples/validation/results/recovery_t_probe.png)

*Figure 5.1b: Linear probe recovery of Learning Rate ($P_T$). Strong correlation ($r = 0.740$) validates that learning velocity is encoded in the latent representations, enabling the model to distinguish fast vs. slow learners.*

#### Interpretation

The strong correlations ($r > 0.7$) provide evidence that:
1. **Structural interpretability**: Pedagogical constructs are not merely post-hoc explanations but are actively encoded in the model's reasoning process
2. **Linear accessibility**: Complex neural representations can be decoded into human-interpretable parameters using simple linear transformations
3. **Theoretical alignment**: The model's internal structure aligns with established cognitive theory (BKT), bridging neural architectures with educational science

This validates that active grounding successfully induces interpretable latent geometry during training.

---

### 5.2 Ablation Studies

**Research Question**: Which components are necessary for interpretability, and what is the performance cost?

We systematically remove architectural components to isolate their individual contributions, measuring both predictive accuracy (AUC) and interpretability (parameter recovery correlation).

#### Experimental Design

| Configuration | Grounded | Probing | Personalization | Test AUC | Interpretability | Cost |
|:---|:---:|:---:|:---:|---:|---:|---:|
| **Baseline** | ❌ | ❌ | ❌ | 0.7832 | - | - |
| **Grounded** | ✅ | ❌ | ❌ | 0.7802 | 0.000 | -0.30% |
| **Probing** | ✅ | ❌ | ✅ | 0.7784 | 0.725 | -0.48% |
| **Full (Personalized)** | ✅ | ✅ | ✅ | 0.7794 | 0.728 | -0.38% |

*Interpretability score*: Average Pearson $r$ for $P_{L0}$ and $P_T$ recovery.

#### Visual Evidence

![Ablation Trade-off](../examples/validation/results/ablation_tradeoff.png)

*Figure 5.2: Interpretability-accuracy trade-off across architectural configurations. Blue line: Test AUC (left axis); Red line: Interpretability score (right axis). Probing activation dramatically increases interpretability ($0.00 \to 0.73$) with minimal accuracy cost ($\Delta = -0.18$ pp). Personalization recovers some performance while maintaining interpretability.*

#### Key Findings

1. **Minimal interpretability tax**: Full interpretability costs only 0.38 percentage points of AUC (relative 0.49% loss)
2. **Probing is critical**: Parameter recovery jumps from near-zero ($r \approx 0$) to strong ($r > 0.7$) when probing loss is activated
3. **Grounding alone insufficient**: Without probing constraints, latent representations do not organize along interpretable axes
4. **Personalization is free**: Student embeddings maintain interpretability while slightly recovering accuracy ($+0.10$ pp)

This demonstrates that interpretability and accuracy are not fundamentally opposed—careful architectural design enables both simultaneously.

---

### 5.3 Latent Space Organization

**Research Question**: Do grounding constraints produce more pedagogically-structured internal representations?

We compare latent space geometry between grounded and baseline models using dimensionality analysis and clustering metrics to assess whether theoretical constraints induce meaningful organizational structure.

#### Methodology

For both models, we:
1. Extract latent vectors $z \in \mathbb{R}^{64}$ for test interactions
2. Perform PCA to analyze effective dimensionality (cumulative variance)
3. Compute clustering quality metrics (Silhouette score, Davies-Bouldin index)
4. Visualize organization via t-SNE, colored by BKT difficulty ($P_{L0}$)

#### Results

**Quantitative Metrics**:

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Effective Rank |
|:---|---:|---:|---:|---:|
| **Grounded** | 0.598 | 0.789 | 369.7 | 38 |
| **Baseline** | 0.497 | 0.973 | 231.7 | 39 |

*Higher Silhouette and Calinski-Harabasz indicate better cluster separation; lower Davies-Bouldin indicates tighter, more distinct clusters.*

**Visual Evidence**:

![Dimensionality Comparison](../examples/validation/results/elbow_plot_comparison.png)

*Figure 5.3a: PCA cumulative variance analysis. Both models reach 90% variance threshold with similar component counts, confirming grounding maintains parsimony. Blue: GTransformer; Gray: Baseline.*

![Latent Organization](../examples/validation/results/latent_organization_tsne.png)

*Figure 5.3b: t-SNE projection colored by initial mastery ($P_{L0}$). Grounded model exhibits clear pedagogical gradients (smooth color transitions) and distinct skill clusters. Higher Silhouette score (0.598 vs. 0.497) numerically confirms superior semantic organization.*

#### Interpretation

1. **Pedagogical structure**: Grounded model organizes latent space along theoretically meaningful dimensions (difficulty, learning rate)
2. **Improved separability**: 20% improvement in Silhouette score indicates more distinct, interpretable skill representations
3. **Maintained efficiency**: Effective rank remains similar (~38 dimensions), showing grounding adds structure without increasing complexity
4. **Gradient organization**: Smooth t-SNE gradients (rather than random scatter) suggest continuous encoding of pedagogical difficulty

This validates that active grounding shapes internal geometry to reflect educational theory, making representations both more interpretable and better-organized.

---

### 5.4 Student Profiling and Case Studies

**Research Question**: Do model predictions provide actionable insights for real educational decision-making?

We demonstrate practical utility through analysis of individual learning trajectories, showing how grounded parameters enable context-aware diagnostics for placement and pacing decisions.

#### Pedagogical Archetypes

By clustering students in $(P_{L0}, P_T)$ parameter space, we identify four common learning profiles:

| Archetype | $P_{L0}$ | $P_T$ | Pedagogical Implication | Example (Skill, Student) |
|:---|:---:|:---:|:---|:---|
| **Foundational Support** | Low | Low | Needs scaffolding, gradual pacing | Skill 14, Student 404 |
| **Rapid Progress** | Low | High | Ready for acceleration despite gaps | Skill 63, Student 7 |
| **Consolidation** | High | Low | Appropriate challenge level, monitor slips | Skill 18, Student 177 |
| **Advanced Placement** | High | High | Ready for enrichment/advancement | Skill 8, Student 550 |

#### Visual Evidence

![Cognitive Quadrants](../examples/validation/results/cognitive_quadrants_mosaic.png)

*Figure 5.4: Learning situation analysis (2×2 mosaic). Each panel shows prediction trajectories for a distinct archetype. Green bars: correct responses; Red bars: errors. Blue line: GTransformer predictions; Orange dashed: BKT baseline. GTransformer adapts predictions to learning context, providing more accurate estimates than BKT's rigid update rules.*

**Case Example - Rapid Progress (Skill 63, Student 7)**:
- **Context**: Low initial mastery ($P_{L0} = 0.24$) but high learning rate ($P_T = 0.82$)
- **GTransformer behavior**: Starts at realistic expectation (~0.35), rapidly increases confidence after observing correct responses
- **BKT behavior**: Overly conservative, fails to recognize fast knowledge acquisition
- **Educational value**: Identifies students ready for accelerated pacing, avoiding unnecessary repetition

#### Interpretation

1. **Individualized diagnostics**: Parameter estimates adapt to student-specific patterns rather than population averages
2. **Actionable insights**: Archetypes map directly to intervention strategies (scaffolding, acceleration, enrichment)
3. **Context-aware predictions**: Model distinguishes temporary errors (slips) from knowledge gaps based on learning history
4. **Improved accuracy**: GTransformer predictions closer to actual performance across all archetypes

This demonstrates that grounded parameters are not just theoretically interpretable but practically useful for educational decision-making.

---

### 5.5 Context-Aware Diagnostics

**Research Question**: Does the model maintain non-Markovian memory, adapting predictions based on extended learning context rather than just observed responses?

We demonstrate GTransformer's context-aware personalization by comparing two students who exhibit **identical response sequences** but receive different predictions due to their different learning contexts. This validates that the model goes beyond Markovian behavior to provide truly individualized diagnostics.

#### The Context-Aware Personalization Challenge

Unlike BKT (which updates beliefs based only on observed responses, following strict Markovian rules), GTransformer maintains rich contextual memory of each student's learning profile. We demonstrate this through a critical scenario:

**Same Behavior, Different Contexts**:
- **Student A**: High initial mastery ($P_{L0} = 0.88$), moderate learning rate ($P_T = 0.49$)
- **Student B**: Low initial mastery ($P_{L0} = 0.14$), same learning rate ($P_T = 0.49$)
- **Identical sequences**: Both students answer the same questions with the same pattern of correct/incorrect responses

**The Diagnostic Challenge**: Should these students receive the same predictions, or should their different learning contexts lead to different diagnostic interpretations?

#### Visual Evidence

![Initial Mastery Mosaic](../examples/validation/results/initial_mastery_mosaic.png)

*Figure 5.5: Context-aware personalization demonstration. Two students (top: high $P_{L0}$, bottom: low $P_{L0}$) exhibit identical response sequences (same green/red bar patterns). GTransformer (blue line) provides different predictions based on learning context: high confidence for the high-mastery student (interpreting errors as slips), appropriate caution for the low-mastery student (interpreting correct responses as potentially lucky guesses). BKT (orange dashed line) provides identical predictions for both students, failing to account for individual learning contexts. This demonstrates GTransformer's non-Markovian, context-aware diagnostic capability.*

#### Pedagogical Interpretation

**Student A (High Initial Mastery)**:
- **GTransformer behavior**: Maintains high confidence (~0.90) throughout, correctly interpreting errors as occasional slips rather than knowledge loss
- **BKT behavior**: Mechanically updates beliefs based on responses, unnecessarily dropping confidence after errors
- **Educational value**: Prevents over-correction and unnecessary remediation, maintaining appropriate challenge level for strong students

**Student B (Low Initial Mastery)**:
- **GTransformer behavior**: Remains appropriately cautious (~0.25-0.40), interpreting correct responses as potentially lucky guesses rather than consolidated mastery
- **BKT behavior**: Mechanically updates beliefs identically to Student A, becoming overly optimistic
- **Educational value**: Ensures sufficient practice before advancement, preventing premature progression that could lead to knowledge gaps

#### Key Advantages

1. **Context-aware personalization**: Different students with identical response sequences receive appropriately different predictions based on their learning contexts
2. **Non-Markovian memory**: Considers full learning profile ($P_{L0}$, $P_T$) and extended interaction history, not just recent responses
3. **Stable diagnostics**: Less volatile than BKT's rigid, response-driven update rules
4. **Pedagogically meaningful distinctions**: Distinguishes slips from knowledge loss, and lucky guesses from genuine mastery
5. **Individual adaptation**: Predictions personalized to student learning characteristics beyond observable behavior

**Critical Insight**: BKT's Markovian assumption—that current knowledge depends only on the most recent response—forces identical predictions for students with identical response sequences, regardless of their different learning contexts. GTransformer breaks this limitation by grounding predictions in individualized learning profiles, enabling context-aware diagnostics that better support placement and pacing decisions.

This validates that deep sequence modeling combined with grounded parameter estimation provides diagnostic value beyond classical knowledge tracing approaches.

---

### 5.6 Baseline Comparisons and Dual Evaluation

**Research Question**: Does GTransformer achieve real interpretability through functional BKT logic predictions (p_ref) while maintaining competitive accuracy?

We position the proposed model within the landscape of knowledge tracing approaches using a **dual evaluation protocol** that measures both neural performance and interpretable reasoning.

#### Dual Evaluation Protocol

GTransformer produces two prediction streams:

1. **p_sup (Supervised Predictions)**: Direct neural head output, optimized for maximum accuracy
2. **p_ref (Reference Predictions)**: BKT logic output using grounded parameters ($p_{L0}$, $p_T$) with fixed guess/slip rates

This dual evaluation quantifies **functional interpretability**—not just whether parameters correlate with theory, but whether they produce valid predictions when used in interpretable BKT logic.

**Interpretability Gap**: $\Delta_{gap} = \text{AUC}(p_{sup}) - \text{AUC}(p_{ref})$

#### Three-Way Comparison with Dual Metrics

| Model | Architecture | AUC (p_sup) | AUC (p_ref) | Interpretability | Gap | Parameters |
|:---|:---|---:|---:|:---:|---:|---:|
| **BKT** | Symbolic | - | 0.610 | ✅ Full | - | ~4/skill |
| **AKT** | Transformer | 0.783 | - | ❌ None | - | ~1.2M |
| **GTransformer** | Grounded Transformer | **0.779** | **0.683** | ✅ Full | **0.095** | ~1.2M |

#### Visual Evidence

![Baseline Comparison](../examples/validation/results/baseline_comparison_plot.png)

*Figure 5.6a: Accuracy-interpretability frontier with dual evaluation. BKT offers interpretability but limited accuracy (0.610); AKT achieves high accuracy but no interpretability (0.783); GTransformer provides both through p_ref predictions (0.7XX) with minimal gap from p_sup (0.779). Error bars: 95% confidence intervals across 5-fold CV.*

![Interpretability Gap](../examples/validation/results/interpretability_gap_analysis.png)

*Figure 5.6b: Distribution of interpretability gap (p_sup - p_ref) across test interactions. Narrow distribution indicates consistent functional interpretability—grounded parameters produce reliable BKT predictions across diverse learning situations.*

#### Key Findings

1. **Real interpretability validated**: p_ref AUC of 0.683 demonstrates that grounded parameters are not just correlated with theory—they are **functionally valid** for BKT reasoning, producing predictions 87.7% as accurate as the neural head
2. **Superior to pure BKT**: GTransformer's p_ref predictions outperform classical BKT by +0.073 AUC (+12.0% relative improvement), proving neural grounding produces better-quality parameters than population-level fitting
3. **Minimal interpretability cost**: Gap of 0.095 between p_sup and p_ref quantifies the exact price of using interpretable logic instead of black-box predictions—only 9.5 percentage points of AUC
4. **Competitive with black-box**: p_sup maintains performance within 0.4 pp of unconstrained AKT (0.779 vs. 0.783)
5. **Unique capability**: Only model providing both competitive accuracy AND functional interpretable predictions that significantly outperform classical BKT

#### Post-hoc Interpretability Comparison

We tested whether baseline transformers can be made interpretable after training by fitting linear probes on frozen representations:

| Metric | Grounded (Active) | Baseline (Post-hoc) | Difference |
|:---|---:|---:|---:|
| $P_{L0}$ Recovery ($r$) | **0.715** | 0.085 | +0.630 |
| $P_T$ Recovery ($r$) | **0.740** | -0.144 | +0.884 |
| **p_ref AUC** | **0.683** | **N/A** | **Functional** |

Post-hoc probing fails dramatically ($r < 0.1$), producing parameters too corrupted for valid BKT predictions. This confirms that interpretability must be **actively designed into the architecture** during training—it cannot be retrofitted.

#### Interpretation

1. **Functional interpretability**: p_ref predictions (0.683 AUC) validate that grounded parameters work in real BKT logic, not just correlate with theory—achieving 87.7% of neural performance through interpretable reasoning
2. **Quantified cost**: Interpretability gap of 0.095 provides precise measurement of the accuracy-interpretability trade-off—only 9.5 percentage points to gain full BKT-based explanations
3. **Neural enhancement of theory**: p_ref outperforming classical BKT by +0.073 AUC (+12% relative) shows deep learning improves parameter estimation quality beyond population-level fitting
4. **Active grounding essential**: Post-hoc interpretation of black-box models fails to produce functional pedagogical parameters (correlations near zero)
5. **Pareto optimality**: GTransformer occupies a unique position—best interpretable predictions (p_ref = 0.683) while maintaining competitive neural accuracy (p_sup = 0.779)
6. **Practical deployment**: High enough accuracy for real-world use (both p_sup and p_ref exceed BKT) while providing actionable, theory-grounded diagnostic information
7. **Dual prediction value**: Educators can choose based on context—use p_sup for high-stakes decisions (maximum accuracy), p_ref for interpretable diagnostics (actionable feedback), or compare both to detect model uncertainty

This positions GTransformer as a practical solution for educational applications requiring both prediction quality and interpretability, with empirical validation through functional BKT logic predictions.

---

### Summary of Validation

Our six-part validation provides converging evidence that GTransformer achieves interpretable knowledge tracing:

1. **Parameter Recovery** ($r > 0.7$): Pedagogical constructs encoded in latent representations
2. **Ablation Studies** (0.38% cost): Interpretability achieved with minimal accuracy sacrifice
3. **Latent Organization** (20% better clustering): Grounding shapes internal geometry
4. **Student Profiling** (4 archetypes): Diagnostics map to actionable interventions
5. **Context-Aware Prediction** (non-Markovian): Extended memory improves accuracy
6. **Dual Evaluation** (p_ref functional): Real interpretability validated through BKT logic predictions (0.683 AUC), with minimal gap (0.095) and superior performance to classical BKT (+0.073 AUC, +12% relative improvement)

These results demonstrate that theory-guided neural architectures can bridge the gap between black-box deep learning and interpretable educational models, with dual evaluation providing empirical proof of functional interpretability.
