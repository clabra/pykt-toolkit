# Proposal

## Abstract

Deep learning models for knowledge tracing aim to predict learner performance over time, but most existing approaches emphasize predictive accuracy at the cost of interpretability. We present **iDKT** (Interpretable Deep Knowledge Tracing), a novel framework that achieves **interpretability-by-design** through Representational Grounding of latent states in classical pedagogical theory. By using **Relational Differential Fusion (RDF)**, iDKT individualizes student mastery estimates by shifting skill-specific bases derived from Bayesian Knowledge Tracing (BKT).

Our results on benchmark datasets (ASSIST2009, ASSIST2015) demonstrate that iDKT achieves near-perfect alignment with theoretical parameters ($>0.98$ correlation) while maintaining competitive predictive performance, with less than 1.5% decrease in AUC compared to non-grounded baselines. These results suggest that high-fidelity grounding is a viable path toward transparent and trustworthy educational AI.

## The Interpretability Challenge in Knowledge Tracing

### The Black Box Problem

Traditional deep learning models for knowledge tracing achieve high predictive accuracy but suffer from a fundamental interpretability deficit. During training and deployment, these models operate as opaque black boxes: their internal representations evolve without semantic grounding and they provide predictions about the future performance of the students but no information about their knowledge states or learning trajectories.

1. **Hidden Knowledge Evolution**: We cannot observe how the model's internal knowledge states change as it processes student interaction sequences, making it impossible to verify whether learned representations correspond to meaningful learning constructs.

2. **Unverified Mastery Estimates**: When they try to project latent states into skill mastery vectors, they tend to exhibit patterns that violate pedagogical principles—they might decrease over time (contradicting the monotonicity principle), take negative values (lacking interpretable semantics), or show no correlation with observed performance (breaking the fundamental link between internal state and external behavior).

3. **Unconstrained Architectural Freedom**: Without explicit constraints, deep learning models can learn representations that optimize predictive loss while producing nonsensical intermediate states. The model might internally represent "mastery" as any arbitrary vector that happens to minimize cross-entropy, regardless of whether those values have educational meaning.

4. **Post-hoc Opacity**: Even when models incorporate mechanisms such as attention weights or skill embeddings, they don't translate into interpretable output. We cannot verify in real-time whether architectural constraints like positivity or monotonicity are actually satisfied during optimization, nor can we detect when the model strays into semantically inconsistent regions of the parameter space.

This interpretability gap has profound implications: educators cannot trust model recommendations, researchers cannot validate learning theories through model introspection, and the deployment of KT systems in high-stakes educational contexts remains problematic.

## Our Proposal: Interpretability-by-Design with Semantic Alignment

**Core Innovation**: iDKT embeds interpretability directly into the architecture through **Relational Differential Fusion (RDF)**. Rather than treating interpretability as a post-hoc analysis problem, iDKT's latent representations are structurally grounded in BKT priors ($L0_{skill}$, $T_{skill}$) and individualized via learned student-specific scalars ($v_s$, $k_c$). This ensures that the model's internal knowledge states are, by design, semantically aligned with established pedagogical constructs.

## Related Work

### Interpretability and Explainability in Transformers

Attention-based methods, both alone and in conjunction with activation-based and gradient-based methods, are the most employed ones. A growing attention is also devoted to the deployment of visualization techniques to help the explanation process @fantozzi2024explainability.

Classification proposed in @fantozzi2024explainability employs the following classes and their combinations:

- Activation: based on identifying the contribution of each input feature to the output through identigying waht neurons are activated.
- Attention: there are three streams: (1) the papers proposing the usage of attention weights; (2) the papers using the Attention Rollout technique (a chain of cumulative attention matrices is formed by multiplying the attention matrix Al of the l-th layer by the attention matrices of the subsequent layers); and (3) the papers exploiting visualization techniques for attention weights.
- Gradient: based on different functions of the gradient computed at different points in the neural network.
- Perturbation: The perturbation approach identifies the relevance of a portion of the input by masking it and checking the consequences on the output.

### Theory-Guided Deep Learning

Theory-Guided Data Science (TGDS) is a paradigm introduced by @karpatne2017theory that leverages the wealth of scientific knowledge (theory) to improve the effectiveness of data science models in scientific disciplines.

The most relevant method for our approach is based in the use of some variant of an augmented loss function:

```math
    Loss = Loss_{SUP}(Ytrue,Ypred) + \lambda R(W) +\gamma Loss_{PHY}(Ypred)
```

Where $Loss_{SUP}$ is the supervised training loss, $Loss_{PHY}$ is the physics loss, $R(W)$ is a regularization term, and $\lambda$ and $\gamma$ are hyperparameters.

See `bibliography/theory-guided/theory_guided.md` for background on theory-guided learning. Some relevant papers are:

- @karpatne2017theory was a foundational work defining Theory-Guided Deep Learning (TGDL). It covers:

  - **Motivation:** Purely data-driven models (black-box DL) often fail to generalize to unseen scenarios (e.g., changing climate conditions) and may produce results inconsistent with known physical laws.
  - **Core Concept:** TGDS introduces scientific consistency as an explicit requirement. It integrates domain knowledge into the learning process to ensure models are scientifically plausible and interpretable.
  - **Approaches:**
  - **Theory-guided Learning:** Incorporating physical laws (e.g., conservation of mass/energy) as regularization terms in the loss function.
  - **Theory-guided Architecture:** Designing neural network architectures that respect domain structure (e.g., connectivity based on physical interactions).
  - **Theory-guided Refinement:** Post-processing predictions to enforce consistency.

- @vonrueden2021informed reviews the field including how algebraic equations and inequalities can be integrated into learning algorithms via additional loss terms or, more generally, via constrained problem formulation.

- @willard2023theory reviews the field including various ways to integrate theory into machine learning including loss functions, initialization (using the physics-based model’s simulated data to pre-train the ML model), physics-guided architecture, and hybrid Physics-ML Models

- @nasir2025understanding focuses on the mechanics of training deep networks using theoretical constraints:

  - **Physics-Informed Neural Networks (PINNs):** A dominant approach where differential equations (PDEs) are embedded into the loss function.
  - **Optimization:** Discusses challenges in optimizing these hybrid loss functions (balancing data loss vs. physics loss) and methods like meta-learning or evolutionary strategies to find optimal architectures.
  - **Design:** Argues for designing networks that are inherently constrained by theory rather than just regularized by it.

### Theory-Guided Loss Functions

### Theory-Guided Deep Knowledge Tracing

Concepts and methods from Theory-Guided Deep Learning have not been widely applied in Knowledge Tracing, except for some works that applied contrastive contrastive knowledge tracing algorithms and prediction-consistency @shukurlu2025survey, and regularization losses imposing certain consistency or monotonicity biases on model’s predictions to improve the generalization ability of KT models @lee2021consistency.

### Latent Space

The two main methods to factorise the latent space are:

#### Latent Space Factorisation

@li2020latent shows how to learn disentangled latent representations via matrix subspace projection. This then allows to change selected attributes while preserving other information. The method is much simpler than previous approaches to latent space factorisation, for example not requiring multiple epochs of training ... The variables representing attributes are fully disentangled, with one isolated variable for each attribute of the training set. In conditional generation, we can assign pure attributes combined with other latent data which does not conflict, so that the generated pictures are of high quality and not contaminated with spurious attributes. The model is a universal plugin. In theory, it can be applied to any existing AEs (if and only if the AEs use a latent vector).

The approach is based on:

![Latent Space Factorization](/bibliography/transformers/msp.png)

$L_{MSP} = L_1 + L_2$

$L_1 = ||\hat{y} − y||2 = ||M · \hat{z} − y||2$

$L_2 = ||\hat{s}||2$

```
Given a latent vector z encoding x and an arbitrarily complex invertible function H(·), H(·) transforms z to a new linear space (ˆz = H(z)) such that one can find a matrix M where:

(a) the projection of ˆz on M (denoted by ˆy) approaches y (i.e., ˆy captures attribute information), M · ˆz = ˆy; ˆy → y
(b) there is an orthogonal matrix U ≡ [M; N], where N is the null space of M (i.e., M ⊥ N) and the projection of ˆz on N (denoted by ˆs) captures non-attribute information.

Here L1 and L2 encode the above two constraints, respectively, and ˆy is the predicted attributes. Given that the AE relies on the information of ˆz to reconstruct x, the optimisation constraints of LAE and L2 essentially introduce an adversarial process: on the one hand, it discourages any information of ˆz to be stored in ˆs due to the penalty from L2; on the other hand, the AE requires information from ˆz to reconstruct x. So, the best solution is to only restore the essential information for reconstruction (except the attribute information) in ˆs. By optimising LMSP , we cause ˆz to be factorised, with the attribute information stored in ˆy, while ˆs only retains non-attribute information.
```

#### Concept Whitening

@chen2020concept shows how to add concept whitening modules to a CNN, to align the axes of the latent space with known concepts of interest.

### Relational Differential Fusion (Archetype 1)

The core innovation of iDKT is the **Relational Differential Fusion (RDF)** layer, which bridges psychometric theory and deep learning. We define individualized student embeddings as:
- **Individualized Initial Mastery ($l_c$):** $l_c = L0_{skill} + k_c \cdot d_c$
- **Individualized Learning Velocity ($t_s$):** $t_s = T_{skill} + v_s \cdot d_s$

Where $L0_{skill}$ and $T_{skill}$ are anchored in BKT priors, and $k_c, v_s$ are learnable student-specific scalars. This implements the **Relational Inductive Bias** ($\text{Challenge} - \text{Capability}$), ensuring that the Transformer's attention is semantically anchored.

### Loss Function

We use a multi-objective loss function to enforce Representational Grounding:

$L_{\text{total}} = L_{\text{SUP}} + \lambda_{\text{ref}} L_{\text{ref}} + \sum_{i} \lambda_{\text{p},i} L_{\text{param},i} + \lambda_{reg} L_{reg}$

Where:
- $L_{\text{SUP}}$: Standard Binary Cross-Entropy (BCE) for correctness prediction.
- $L_{\text{ref}}$: Mean Squared Error (MSE) between iDKT and BKT correctness predictions.
- $L_{\text{param}}$: MSE between model latent projections and theoretical BKT parameters ($L0, T$).
- $L_{reg}$: L2 penalty on difficulty ($u_q$) and student-specific parameters ($v_s, k_c$).

**Key Advantages**:
- **High-Fidelity Alignment**: Achieving $>0.98$ correlation between model states and theory ensures that the "hidden knowledge evolution" is no longer hidden but semantically verifiable.
- **Negligible Performance Cost**: Our "interpretability-by-design" approach incurs a minimal (<1.5%) AUC cost, invalidating the common assumption that Transformers must be black boxes to be accurate.
- **Scientific Machine Learning (SciML)**: iDKT represents a fusion of the interpretability of probabilistic models with the predictive power of deep learning, providing a robust framework for educational accountability.

**Summary**: iDKT demonstrates that we can have models that are simultaneously accurate, interpretable, and theoretically justified—addressing the core limitations of current deep knowledge tracing models.
