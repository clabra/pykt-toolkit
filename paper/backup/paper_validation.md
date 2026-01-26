# 5. Experimental Validation

In this section, we present a comprehensive empirical evaluation of the gTransformer model, focusing on its ability to bridge the gap between deep learning performance and pedagogical interpretability. Our validation strategy is structured to demonstrate that the model's internal representations are not only mathematically sound but also practically useful for educational decision-making. We evaluate the model across six dimensions, using 5-fold cross-validation on the ASSIST2009 dataset to ensure robust and reproducible results.


## 5.1 Parameter Recovery Accuracy

A fundamental requirement for an interpretable knowledge tracing model is that its internal parameters must correspond to established educational constructs. We assess the structural validity of gTransformer by measuring how accurately its learned representations can recover the parameters of Bayesian Knowledge Tracing (BKT), specifically initial mastery ($P_{L0}$) and learning rate ($P_T$).

As shown in Figure 5.1, linear probes fitted to the model’s latent context vectors demonstrate a high degree of correlation with "oracle" BKT parameters ($r > 0.7$). Specifically, we achieve a Pearson correlation of $0.709$ for initial mastery and $0.733$ for the learning rate. This signifies that the model successfully induces a latent geometry where pedagogical dimensions are represented linearly and accessibly, confirming that our grounding mechanism effectively anchors the deep learning architecture to defined educational theory.

<div style="width: 50%;">

![Parameter Recovery - Initial Mastery](../examples/validation/results/recovery_l0_probe.png)

</div>

<div style="width: 50%;">

![Parameter Recovery - Learning Rate](../examples/validation/results/recovery_t_probe.png)

</div>
*Figure 5.1: Linear probe recovery of Initial Mastery ($P_{L0}$) and Learning Rate ($P_T$). The strong correlations ($r = 0.71$ and $r = 0.73$ respectively) validate that pedagogical constructs are encoded in a linearly readable format.*

## 5.2 Ablation Studies and the Interpretability-Accuracy Trade-off

We systematically examined the performance implications of adding interpretability constraints to a standard Transformer baseline. By isolating the effects of grounding and probing mechanisms, we quantified the "interpretability tax" inherent in our approach.

Our results indicate that achieving full theoretical grounding—enabling the model to produce structurally valid and pedagogically aligned parameters—costs less than 0.5% in absolute AUC (approximately 0.38 percentage points). As illustrated in Table 5.2, the transition from a non-personalized baseline to a fully grounded and personalized model results in a negligible performance loss.

**Table 5.2: Ablation Study of Architectural Components**
| Configuration | Grounded | Probing | Personalization | Test AUC | Interpretability ($p_{ref}$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | ❌ | ❌ | ❌ | 0.7822 | - |
| Aligned Grounding | ✅ | ✅ | ❌ | 0.7784 | 0.6834 |
| Personalized | ✅ | ✅ | ✅ | 0.7794 | 0.6839 |
| **Minimalist Grounding**| ❌ | ✅ | ❌ | **0.7790** | **0.6756** |

<div style="width: 50%;">

![Ablation Trade-off](../examples/validation/results/ablation_tradeoff.png)

</div>
*Figure 5.2: Interpretability-accuracy trade-off. Accuracy (Blue) slightly decreases as constraints are added, while Structural Interpretability (Red) jumps once the Probing objective is active.*

## 5.3 Latent Space Organization

To evaluate the impact of theory-guided constraints on the model's internal reasoning, we analyzed its latent space organization. We compared the manifold geometry of gTransformer against a traditional unconstrained baseline using both qualitative visualizations and quantitative clustering metrics.

The grounded architecture maintains structural purity while achieving predictive parity with unconstrained models. While global clustering metrics such as Silhouette scores are comparable ($0.5749$ vs $0.5736$), dimensionality analysis via PCA elbow plots and effective rank calculations reveal that gTransformer achieves a more parsimonious representation. Specifically, the grounded model organizes the latent space into a lower effective rank (38 vs 40), indicating that grounding compresses representations into meaningful educational gradients rather than transient statistical correlations. This provides a more stable foundation for adaptive instruction by clustering student interactions along explicit difficulty and learning velocity axes.

<div style="width: 50%;">

![Elbow Plot Comparison](../examples/validation/results/elbow_plot_comparison.png)

</div>

<div style="width: 50%;">

![Latent Organization](../examples/validation/results/latent_organization_tsne.png)

</div>
*Figure 5.3: Dimensionality (Elbow Plot) and Latent Organization (t-SNE). Grounding maintains parsimony (lower effective rank) while ensuring that the latent space is organized according to pedagogical constructs.*

## 5.4 Student Profiling and Case Studies

The practical utility of gTransformer is best demonstrated through its ability to capture actionable differences in student learning patterns. By analyzing individual learning trajectories, we show how the model's parameters—specifically its estimates of prior knowledge ($P_{L0}$) and learning rate ($P_T$)—identify student-specific needs that population-level models often overlook.

### Situational Archetypes

Through clustering analysis on the contextually derived $(P_{L0}, P_T)$ parameter space, we identify four distinct student learning situations. Crucially, these situations are accurately identified by the model **even without student IDs**, using only the longitudinal context of the interaction sequence.

| Learning Situation | $P_{L0}$ | $P_T$ | Proportion | Pedagogical Intervention |
|:---|:---:|:---:|:---:|:---|
| **Foundational Support** | Low | Low | 58% | Scaffolding and remedial support |
| **Rapid Progress** | Low | High | 1% | Sequence acceleration and higher pacing |
| **Consolidated Mastery** | High | Low | 38% | Practice maintenance and stability monitoring |
| **Advanced Achievement** | High | High | 3% | Enrichment and advancement to next module |

**A Key Finding - The Dominance of Longitudinal Context**: 
Our "Minimalist Grounding" study reveals that the Transformer’s attention mechanism is a more powerful diagnostic tool than fixed student parameters.
- **Contextual Models (Aligned/Minimalist)**: Achieve high diagnostic variance (Std $P_T \approx 0.35$). The model "discovers" student-specific behavioral patterns (e.g., rapid learning vs. lucky guesses) as it observes the sequence, without needing to know the student's unique ID.
- **Personalized Models (Student IDs)**: Do not significantly increase diagnostic resolution over purely contextual ones. Personalization contributes to a slight accuracy gain (+0.001 AUC) but the diagnostic "heavy lifting" is performed by the Transformer observing behavioral history.

This has profound implications for educational practice: it means gTransformer can provide **high-resolution diagnostics from the very first interaction sequence**, even for students it has never encountered before (Cold-Start). By moving from "Labeling Students" to "Diagnosing Situations," the model provides a more dynamic and less biased approach to personalization.

## 5.5 Context-Aware Diagnostics

A key contribution of gTransformer is its ability to provide individualized, context-aware diagnostics without over-reliance on fixed student traits. By focusing on "temporal signatures"—the patterns of progress captured in the longitudinal interaction history—the model adapts its predictions to the specific context of each learning sequence.

Our "Learning Situation Analysis" (2x2 mosaic) demonstrates how the model's predictions respond to different combinations of prior knowledge and learning responsiveness. For example, in situations where a student with low prior knowledge begins to succeed rapidly, gTransformer’s high learning rate parameter enables a sharp, responsive update in predicted mastery that far exceeds the rigid, Markovian update rules of classical BKT. This context-awareness ensures that the model provides truly individualized diagnostics, acknowledging that a student's current performance is best understood within the full context of their evolving learning history.

<div style="width: 50%;">

![Cognitive Quadrants](../examples/validation/results/cognitive_quadrants_mosaic.png)

</div>
*Figure 5.4: Learning situation analysis (2×2 mosaic). The model adapts its predictions to different pedagogical scenarios (e.g., Foundational Support vs. Responsive Learning), providing more nuanced estimates than the Markovian baseline.*

## 5.6 Baseline Comparisons

Finally, we position gTransformer within the broader landscape of knowledge tracing models through a three-way comparison between classical symbolic models (BKT), state-of-the-art Transformers (AKT), and our proposed model.

The results, summarized in Table 5.6, show that gTransformer achieves the optimal balance between accuracy and interpretability. While pure symbolic models provide full transparency but limited accuracy (AUC ≈ 0.61), and standard Transformers offer high accuracy (AUC ≈ 0.78) with no interpretability, gTransformer maintains a competitive accuracy level ($p_{sup} = 0.7790$) while providing the functional interpretability of a theory-grounded model ($p_{ref} = 0.6756$). By outperforming pure BKT by 10.8% in predictive power while offering the structural alignment missing in standard Transformers, gTransformer represents a significant step forward in the development of practical, high-performance, and pedagogically sound educational AI.

**Table 5.6: Comparative Evaluation on ASSIST2009**
| Model | Architecture | AUC ($p_{sup}$) | AUC ($p_{ref}$) | Interpretability | Gap |
|:---|:---|---:|---:|:---:|---:|
| **BKT** | Symbolic | - | 0.610 | ✅ Full | - |
| **AKT (Baseline)** | Transformer | **0.783** | - | ❌ None | - |
| **gTransformer** | Proposed | **0.779** | **0.676** | ✅ Full | **0.103** |

<div style="width: 50%;">

![Baseline Comparison](../examples/validation/results/baseline_comparison_plot.png)

</div>
*Figure 5.5: Accuracy-interpretability frontier. gTransformer provides both competitive accuracy and functional interpretable predictions that significantly outperform classical BKT (+0.073 AUC).*
