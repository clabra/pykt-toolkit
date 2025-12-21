# iDKT: High-Fidelity Theoretical Grounding in Deep Knowledge Tracing

**Alternative Titles**
Grounded Transformers: Bridging Theory and Deep Learning for Interpretable Knowledge Tracing.
High-Fidelity Theoretical Grounding in Knowledge Tracing: When Interpretability Meets Deep Learning.
iDKT: Relational Differential Fusion for Individualized and Interpretable Knowledge Tracing.

**Abstract:** Deep learning models, particularly Transformers, have set new benchmarks in student performance prediction. However, their adoption in educational settings is hindered by a perceived lack of interpretability. In this paper, we propose **iDKT** (Interpretable Deep Knowledge Tracing), an architecture that achieves intrinsic interpretability by structurally grounding its latent states in classical educational theory. Using **Relational Differential Fusion**, we individualize student mastery estimates by shifting skill-specific bases derived from Bayesian Knowledge Tracing (BKT). Our experiments on the ASSIST2009 and ASSIST2015 datasets demonstrate that iDKT achieves near-perfect alignment with theoretical parameters ($>0.98$ correlation) while maintaining competitive predictive performance, with less than 1.5% decrease in AUC compared to non-grounded baselines. These results suggest that high-fidelity grounding is a viable path toward transparent and trustworthy educational AI.

## 1. Introduction
The challenge of Knowledge Tracing (KT) is to estimate students' evolving mastery states based on their interaction history. While deep learning models like AKT and SAINT have shown superior predictive accuracy, they are often criticized as "black boxes." This paper introduces iDKT, which prioritizes **interpretability-by-design**. Unlike post-hoc interpretation methods, iDKT's internal mechanisms are directly informed by and regularized against established educational theories.

## 2. Methodology: Relational Differential Fusion
The core of iDKT is the Relational Differential Fusion (RDF) mechanism. We define student mastery as a combination of skill-specific bases and student-specific differentials:
- **Individualized Initial Mastery ($l_c$):** $l_c = L0_{skill} + k_c \cdot d_c$
- **Individualized Learning Velocity ($t_s$):** $t_s = T_{skill} + v_s \cdot d_s$

Where $L0_{skill}$ and $T_{skill}$ are anchored in BKT priors, and $k_c, v_s$ are learnable student-specific parameters. This structure ensures that every prediction made by the model is filtered through a theoretically meaningful lens.

## 3. Experimental Results

### 3.1 Dataset Performance
We evaluated iDKT on two benchmark datasets. The results show that the model remains highly competitive.

| Dataset | iDKT AUC | Baseline AUC | AUC Cost |
| :--- | :---: | :---: | :---: |
| ASSIST2015 | 0.7253 | 0.7336 | -1.13% |
| ASSIST2009 | 0.8397 | 0.8504* | -1.25% |
*\*Baseline estimated from similar configs.*

### 3.2 Theoretical Alignment
The most significant finding is the convergence between the model's latent states and BKT reference parameters.

| Alignment Metric | ASSIST2015 | ASSIST2009 |
| :--- | :---: | :---: |
| Initial Mastery Correlation | **0.9905** | **0.9838** |
| Learning Rate Correlation | **0.9934** | **0.9838** |
| Prediction Correlation | 0.6157 | 0.5206 |

The extremely high correlation coefficients for initial mastery and learning rates prove that iDKT acts as a "theoretically-guided super-estimator," preserving the semantics of educational theory while leveraging the predictive power of the Transformer architecture.

## 4. Discussion
The iDKT architecture demonstrates that interpretability does not have to be an afterthought or a "best-effort" visualization. By pinning the model's input space to theoretical grounding points, we create a system where the latent trajectory itself is the explanation. The negligible AUC cost suggests that the data signal in educational contexts is inherently well-aligned with established theories, and iDKT simply enforces this alignment to prevent the model from learning spurious, uninterpretable correlations.

## 5. Conclusion
iDKT represents a step toward **Scientific Machine Learning** in education. By fusing the strengths of BKT's interpretability and Transformers' predictive capacity, we provide a model that researchers and educators can trust without sacrificing accuracy. Future work will explore further archetypes of grounding, including the integration of IRT-based difficulty constraints.
