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

See `bibliography/theory-guided/theory_guided.md` for background on theory-guided learning.

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
