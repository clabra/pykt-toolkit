# A More Interpretable Architecture: Explicit State-Space GainAKT

This document outlines a proposal for a new model architecture that builds upon GainAKT2 to create a more interpretable and causally-explainable model. The core idea is to move from a latent representation of knowledge to an explicit state-space that is constrained by educational theory.

## Core Idea: The Explicit State-Space Architecture

Instead of having latent `context` and `value` streams of size `d_model`, the model's core will now be an explicit **Mastery State Vector** of size `num_skills`. Each element in this vector will directly represent the mastery of a specific skill.

## Consistency Requirements for Explainability

To be considered interpretable and explainable, the model should adhere to the following consistency requirements:

*   **Monotonicity of Mastery:** A student's mastery of a skill should not decrease over time. An interaction can result in zero gain, but never a negative gain.
*   **Non-Negative Learning Gains:** The learning gain from any interaction should be greater than or equal to zero.
*   **Mastery-Performance Correlation:** The predicted probability of correctly answering a question for a given skill should be positively correlated with the model's estimated mastery of that skill.
*   **Gain-Performance Correlation:** A correct answer should, on average, lead to a higher learning gain than an incorrect answer for the same skill.
*   **Sparsity of Gains (Desirable):** The learning gain from an interaction with a specific skill should primarily affect that skill and only a few closely related skills.

## Architectural Modifications

Here are the architectural modifications needed to implement this new, more constrained model:

### Idea 1: The "Gain Computer" Module

This module will be responsible for computing the learning gain for each interaction.

*   **What it does:** For each interaction `(S_t, R_t)`, it computes a **Gain Vector** of size `num_skills`. This vector represents how much the mastery of *every* skill increases as a result of this single interaction.
*   **Inputs:**
    1.  The current interaction embedding (`(S_t, R_t)`).
    2.  The student's mastery state *before* the interaction (`Mastery_{t-1}`). This is crucial, as the learning gain should depend on the student's current knowledge.
*   **Architectural Change:** We would replace the `value_embedding` stream with this new, more complex module. It would likely be a small feed-forward network.
*   **Constraint - Non-Negative Gains:** To ensure learning gains are never negative, the output of this module would be passed through a `ReLU` or `Softplus` activation function. This enforces the `gain >= 0` constraint directly in the architecture.

### Idea 2: The "State Updater" and the Explicit Mastery Vector

This is the heart of the new architecture.

*   **What it does:** It maintains and updates the student's **Mastery State Vector** over time. This vector has a size of `num_skills`, and `Mastery_t[k]` is the mastery of skill `k` at time `t`.
*   **How it works:** The update rule is simple and interpretable:
    `Mastery_t = Mastery_{t-1} + Gain_t`
    Where `Gain_t` is the output from the "Gain Computer" for the current interaction.
*   **Constraint - Monotonic Mastery:** Because the gains are constrained to be non-negative (from Idea 1), the mastery level for any skill can only increase or stay the same. This enforces monotonicity directly. It may also be desirable to apply a `torch.clamp(..., min=0, max=1)` to keep mastery levels within a probabilistic range.

### Idea 3: The "Mastery-Aware" Prediction Head

This modification makes the model's predictions directly dependent on the explicit skill mastery levels.

*   **What it does:** Predicts the probability of a correct response for the current interaction `(S_t, R_t)`.
*   **Inputs:**
    1.  The current **Mastery State Vector** (`Mastery_t`).
    2.  The current skill being interacted with (`S_t`).
*   **How it works:**
    1.  From the `Mastery_t` vector, it "looks up" the mastery level for the current skill `S_t`.
    2.  This mastery level is then fed into a small network (or even just a linear layer + sigmoid) to produce the final prediction.
*   **Causal Link:** This creates a direct, causal link: `Mastery -> Prediction`. The model is forced to use its explicit representation of skill mastery to make its prediction.

## Guiding Training with Auxiliary Losses

The main prediction loss (Binary Cross-Entropy) will still be the primary driver of training. However, we can add auxiliary losses to further enforce the desired properties:

*   **Consistency Loss:** We could add a loss term that encourages the mastery level of a skill to be consistent with the performance on questions related to that skill. For example, the mastery of "Algebra" should be highly correlated with the probability of answering "Algebra" questions correctly.
*   **Sparsity Loss:** It may be desirable to encourage the gain vectors to be sparse (i.e., an interaction with one skill should only affect a few other related skills). An L1 regularization term on the output of the "Gain Computer" could achieve this.

## Recommendations

1.  **Create a New Model:** This is a significant departure from `gainakt2.py`. We'll create a new model file (e.g., `gainakt_explicit.py`) to implement this new architecture.
2.  **Start with the Architecture:** Focus on implementing the three core architectural ideas first: the explicit mastery vector, the gain computer, and the mastery-aware prediction head.
3.  **Iterate on the Loss:** Once the architecture is in place, we can experiment with different auxiliary losses to see how they affect the model's performance and the interpretability of the learned representations.