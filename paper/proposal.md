# Proposal

## Abstract

Deep learning models for knowledge tracing aim to predict learner performance over time, but most existing approaches emphasize predictive accuracy at the cost of interpretability. We present iDKT, a novel framework that achieves interpretability-by-design through semantic alignment of latent states. iDKT restricts the solution space to representations that are both predictive and consistent with pedagogical theories, ensuring that internal states correspond to meaningful learning concepts.

This is accomplished via mechanisms that enforce semantic consistency and guide the model toward valid configurations. By adopting an interpretability-by-design paradigm, iDKT offers transparent insights into knowledge evolution, enhances trustworthiness, and provides actionable guidance for educators. Experiments on benchmark knowledge tracing datasets show that iDKT matches or surpasses state-of-the-art performance while delivering interpretable outputs on knowledge states and their progression along students' learning paths.

## The Interpretability Challenge in Knowledge Tracing

### The Black Box Problem

Traditional deep learning models for knowledge tracing achieve high predictive accuracy but suffer from a fundamental interpretability deficit. During training and deployment, these models operate as opaque black boxes: their internal representations evolve without semantic grounding and they provide predictions about the future performance of the students but no information about their knowledge states or learning trajectories.

1. **Hidden Knowledge Evolution**: We cannot observe how the model's internal knowledge states change as it processes student interaction sequences, making it impossible to verify whether learned representations correspond to meaningful learning constructs.

2. **Unverified Mastery Estimates**: When they try to project latent states into skill mastery vectors, they tend to exhibit patterns that violate pedagogical principles—they might decrease over time (contradicting the monotonicity principle), take negative values (lacking interpretable semantics), or show no correlation with observed performance (breaking the fundamental link between internal state and external behavior).

3. **Unconstrained Architectural Freedom**: Without explicit constraints, deep learning models can learn representations that optimize predictive loss while producing nonsensical intermediate states. The model might internally represent "mastery" as any arbitrary vector that happens to minimize cross-entropy, regardless of whether those values have educational meaning.

4. **Post-hoc Opacity**: Even when models incorporate mechanisms such as attention weights or skill embeddings, they don't translate into interpretable output. We cannot verify in real-time whether architectural constraints like positivity or monotonicity are actually satisfied during optimization, nor can we detect when the model strays into semantically inconsistent regions of the parameter space.

This interpretability gap has profound implications: educators cannot trust model recommendations, researchers cannot validate learning theories through model introspection, and the deployment of KT systems in high-stakes educational contexts remains problematic.

## Our Proposal: Interpretability-by-Design with Semantic Alignment

**Core Innovation**: Rather than treating interpretability as an afterthought or post-hoc analysis problem, iDKT embeds interpretability directly into the learning process through **semantic alignment of latent states**. The model's internal representations are constrained from the outset to remain within a solution space that is both predictive and pedagogically meaningful.

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

## Approach

### Loss Function

We'll use a combination of supervised and semantic alignment losses to train the model.

$L_{\text{total}} = L_{\text{SUP}} + sum (\lambda_{\text{int}} \times L_{\text{int}}) + sum (\lambda_{\text{reg}} \times L_{\text{reg}})$,

where:

- $L_{\text{SUP}} = f(p_{\text{correct}}, y)$ enforces consistency between predictions and ground truth.
- $L_{\text{int}} = f(p_{\text{correct}}, M_{\text{IRT}})$ enforces interpretability by constraining the model to remain within a solution space that is semantically meaningful.
- $L_{\text{reg}} = f(p_{\text{correct}}, M_{\text{IRT}})$ enforces regularization.

**Key Advantages**:

- **Theoretical Grounding**: By anchoring to pedagogical reference models, we connect deep learning to educational principles. The model's internal states are not arbitrary neural activations—they are constrained to approximate quantities that have established pedagogical interpretations.

- **Verifiable Interpretability**: Unlike post-hoc explanations, our approach provides _guarantees_ about semantic consistency through alignment with a reference model that is pedagogically sound.

- **Transparent Trade-offs**: The hyperparameter λ_sem makes the performance-interpretability balance explicit. Higher values enforce stronger semantic consistency but may reduce AUC, while lower values prioritize performance. Our approach will systematically explore this trade-off to find configurations that are both accurate and interpretable.

**Practical Impact**:

This approach bridges the gap between deep learning performance and educational accountability. Users can inspect model-estimated mastery levels with confidence that they reflect pedagogically meaningful constructs. It enables validation of the learning trajectories and support interpretability. It allows for models with competitive prediction performance while adding interpretability guarantees that purely black-box models don't provide.

**In Summary**: Our appraoch shows that interpretability need not be sacrificed for performance. By constraining the solution space to representations that are both predictive and semantically grounded, we get models that are simultaneously accurate, interpretable, and theoretically justified—addressing the core limitations of current deep knowledge tracing models designed without interpretability in mind.
