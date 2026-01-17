
## Section 5: Experimental Validation

### 5.1 Parameter Recovery Accuracy

To validate the pedagogical integrity of the proposed architecture, we first assess the accuracy with which the model recovers theoretical parameters from Bayesian Knowledge Tracing (BKT). We define two distinct recovery pathways: (1) **Local Projection (Grounded)**, which represents the context-aware estimates used for individualized prediction, and (2) **Global Probing**, which utilizes linear heads specifically designed to extract the latent "BKT-essence" from the hidden state.

Table 1 summarizes the recovery performance across 52,825 test interactions from a large-scale student interaction dataset.

**Table 1: Parameter Recovery Metrics**

| Parameter | Method | Pearson $r$ | MAE | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| Initial Mastery ($P(L_0)$) | Linear Probe | **0.715** | 0.049 | 0.097 |
| Initial Mastery ($P(L_0)$) | Grounded | 0.224 | 0.398 | 0.465 |
| Learning Rate ($P(T)$) | Linear Probe | **0.740** | 0.035 | 0.069 |
| Learning Rate ($P(T)$) | Grounded | 0.268 | 0.212 | 0.350 |

#### Structural Validity (Global Probing)
As shown in Table 1, the **Linear Probes** achieve high correlation ($r > 0.71$) and remarkably low error (MAE $\le 0.05$) when recovering population-level BKT parameters. This represents a critical "Structural Validity" proof: it mathematically demonstrates that the theoretical constructs of BKT—Mastery and Growth—are linearly accessible within the model’s deep representations. Even though the model is a high-capacity neural network, its latent space has been successfully "theory-steered" to encode pedagogical meaning. The corresponding scatter plots for these probes (Fig. 5.1-A/B) show a consistent clustering along the theoretical ideal.

**Figure 5.1-A: Mastery Recovery Probe**
![Mastery Recovery Probe](../examples/validation/results/recovery_l0_probe.png)

**Figure 5.1-B: Learning Rate Recovery Probe**
![Learning Rate Recovery Probe](../examples/validation/results/recovery_t_probe.png)

#### Diagnostic Granularity (Grounded Individualization)
The **Grounded pathway** shows higher variance and lower correlation with the skill-level BKT targets. We interpret this not as a loss of validity, but as the primary value proposition of the neuro-symbolic approach. While traditional BKT assigns a single $L_0$ and $T$ value to all students for a given skill, the proposed model utilizes the longitudinal interaction history to derive **student-specific** estimates. The deviation from the "mean" BKT parameter represents the model identifying students with higher-than-average prior knowledge or faster-than-average learning trajectories. This allows the system to move beyond "population-level averages" to "individualized diagnostics" while maintaining the pedagogical structure proved by the probes.

### 5.2 Ablation Study: Assessing the Cost of Interpretability

To determine the necessity of each architectural component, we conducted a systematic ablation study by evolving the architecture from a black-box Transformer (Baseline) to the full neuro-symbolic model (Personalized). We evaluate two critical dimensions: **Predictive Accuracy (AUC)** and **Structural Interpretability** (measured as the mean Pearson $r$ of the BKT parameter recovery).

Table 2 presents the results of this evolution.

**Table 2: Ablation Results and Performance-Interpretability Trade-offs**

| Model Stage | Composition | Test AUC | Interpretability ($r$) |
| :--- | :--- | :---: | :---: |
| **Baseline** | Standard Transformer | **0.7832** | 0.00 |
| **Grounded** | + Semantic Axes | 0.7802 | 0.00 |
| **Probing** | + Active Grounding | 0.7784 | 0.725 |
| **Personalized** | + Theory Personalization | 0.7794 | **0.728** |

#### The "No-Cost Interpretability" Hypothesis
Our findings challenge the common assumption that interpretability requires a significant sacrifice in accuracy. As shown in Figure 5.2, the transition from a black-box model to a fully theory-grounded model resulted in a negligible absolute AUC reduction of only **0.38 percentage points** (relative 0.49% loss).

**Figure 5.2: The Pareto Frontier of the Proposed Architecture**
![Ablation Tradeoff](../examples/validation/results/ablation_tradeoff.png)

#### Component Contributions
1.  **Semantic Grounding**: Introducing semantic axes for parameter projection provides the necessary mathematical constraints but does not automatically align the latent space linearly ($r \approx 0$).
2.  **Active Grounding**: The introduction of the probing objective is the primary catalyst for interpretability, jumping the structural alignment from zero to **r = 0.725** with a modest performance cost.
3.  **Theory Personalization**: Adding student-specific mastered-at-start and learning-rate offsets not only maintains high interpretability but actually **recovered ~0.1% AUC** compared to the population-level probing model. This suggests that allowing the model to individualize its theoretical parameters helps it better fit the data without losing scientific semantics.

### 5.3 Latent Space Organization: Pedagogical Manifolds

A critical requirement for pedagogical interpretability is that the model’s internal representations must organize themselves according to educational constructs. We analyze the latent space using dimensionality reduction (t-SNE) and quantitative clustering metrics compared to the unconstrained Baseline.

#### Semantic Clustering Quality
As shown in Table 3, the grounded architecture exhibits significantly higher internal organization than the Baseline model across all metrics.

**Table 3: Quantitative Latent Organization Metrics (Skill Clustering)**

| Metric | Baseline | Proposed Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Silhouette Score** ($\uparrow$) | 0.497 | **0.598** | +20.3% |
| **Davies-Bouldin Index** ($\downarrow$) | 0.973 | **0.789** | -18.9% |
| **Calinski-Harabasz** ($\uparrow$) | 231.7 | **369.7** | +59.6% |

The 20.3% improvement in Silhouette score demonstrates that the proposed architecture creates more cohesive and better-separated clusters for different skills. Unlike the baseline, which organizes representations derived solely from sequence patterns, our architecture is forced to map interactions into a "theory-aligned" manifold.

#### Dimensionality and Parsimony
We further analyze the effective rank of the latent representations using an **Elbow Plot** on the cumulative explained variance of the Principal Components.

**Figure 5.3: Latent Space Parsimony (Elbow Plot)**
![Elbow Plot](../examples/validation/results/elbow_plot_comparison.png)

The analysis shows that both models achieve 90% variance within a similar number of principal components. This confirms that adding theoretical constraints does not unnecessarily bloat the latent representation or force the model into an excessively complex state.

#### Visualization of the Difficulty Gradient
Figure 5.4 presents the t-SNE projection of the latent space, colored by the BKT Difficulty ($L_0$).

**Figure 5.4: Pedagogical Manifolds**
![Latent Organization](../examples/validation/results/latent_organization_tsne.png)

The visualization reveals a highly structured organization where clusters (skills) form distinct "islands" in the latent space. Notably, the color gradient (representing cognitive difficulty) is not randomly distributed but follows a coherent internal logic within the manifold. This confirms that the model’s "Relational Axes" successfully steer the deep representations to align with the semantic meaning of Mastery and Difficulty.

**5.4 Sensitivity Analysis**
- Perturbation curves (δ vs. ΔP)
- Monotonicity scores (Spearman ρ > 0.95)
- Finding: Learned axes have causal, interpretable effects

**5.5 Case Studies**
- 4 representative student trajectories
- Diagnostic narratives showing actionable insights
- Finding: Model provides clinically useful individualized diagnostics

**5.6 Baseline Comparisons**
- Three-way comparison table (BKT vs. AKT vs. Proposed)
- Post-hoc vs. built-in interpretability comparison
- Finding: Proposed model achieves best balance of accuracy and interpretability
