---
bibliography: [../../bibliography/biblio.bib]
---

# Theory-Guided and Scientific Machine Learning: A Summary

This document summarizes key literature on Theory-Guided Data Science (TGDS), Scientific Machine Learning (SciML), and their applications, with a specific focus on recent developments in educational data mining.

## 1. The Paradigm of Theory-Guided Data Science

**Source:** _Karpatne et al. (2017) - "Theory-guided Data Science: A New Paradigm for Scientific Discovery from Data"_

Foundational work defining Theory-Guided Data Science (TGDS) as a paradigm that leverages the wealth of scientific knowledge (theory) to improve the effectiveness of data science models in scientific disciplines.

- **Motivation:** Purely data-driven models (black-box DL) often fail to generalize to unseen scenarios (e.g., changing climate conditions) and may produce results inconsistent with known physical laws.
- **Core Concept:** TGDS introduces scientific consistency as an explicit requirement. It integrates domain knowledge into the learning process to ensure models are scientifically plausible and interpretable.
- **Approaches:**
  - **Theory-guided Learning:** Incorporating physical laws (e.g., conservation of mass/energy) as regularization terms in the loss function.
  - **Theory-guided Architecture:** Designing neural network architectures that respect domain structure (e.g., connectivity based on physical interactions).
  - **Theory-guided Refinement:** Post-processing predictions to enforce consistency.

## 2. Scientific Discovery in the Age of AI

**Source:** _Wang et al. (2023) - "Scientific discovery in the age of artificial intelligence"_

This paper provides a broad overview of how AI is transforming scientific discovery (AI4Science).

- **Impact:** AI is enabling new capabilities in disciplines ranging from molecular biology (protein folding) to high-energy physics.
- **Integration:** It highlights the synergy between "first-principles" methods and "data-driven" methods.
- **Trends:** Increasing use of Generative AI and Foundation Models in scientific contexts, allowing for hypothesis generation and simulation acceleration.

## 3. Methodological Advances: Theory-Guided Training

**Source:** _Nasir et al. (2025) - "Understanding and Designing Deep Neural Networks Through Theory-Guided Training"_

Focuses on the mechanics of training deep networks using theoretical constraints.

- **Physics-Informed Neural Networks (PINNs):** A dominant approach where differential equations (PDEs) are embedded into the loss function.
- **Optimization:** Discusses challenges in optimizing these hybrid loss functions (balancing data loss vs. physics loss) and methods like meta-learning or evolutionary strategies to find optimal architectures.
- **Design:** Argues for designing networks that are inherently constrained by theory rather than just regularized by it.

## 4. Application in Education: The TGEL-Transformer

**Source:** _Gong et al. (2025) - "TGEL-transformer: Fusing educational theories with deep learning for interpretable student performance prediction"_

This paper applies the TGDS paradigm to the domain of Knowledge Tracing and Student Performance Prediction, directly relevant to `pykt-toolkit`.

- **Problem:** Existing Deep Learning Knowledge Tracing (DLKT) models (like DKT, AKT) significantly outperform traditional models but lack interpretability and often ignore established educational theories (e.g., forgetting curves, self-regulated learning).
- **Solution (TGEL-Transformer):**
  - **Theory-Guided Educational Learning (TGEL):** A framework fusing educational theory with deep learning.
  - **Mechanism:** It likely uses attention mechanisms (Transformer) constrained or guided by theoretical factors. For example, ensuring that the "forgetting" behavior of the model aligns with cognitive science theories (Ebbinghaus curve).
  - **Interpretability:** By grounding the model's internal states in educational theory, the predictions become more explainable to educators and students (e.g., "performance dropped because of lack of recent practice," rather than just "probability score decreased").
- **Significance:** Demonstrates that identifying and incorporating domain-specific "invariants" or theories (similar to physics laws in SciML) can improve both accuracy and trust in educational AI.

## 5. 2025 Monaco - Theory-guided Data Science models (Phd Thesis)

Focuses on the mechanics of training deep networks using theoretical constraints.

- Current issues with black-box deep learning models and advantages of the theory-guided approach: "the end of theory", lack of theoretical grounding, without explicit constraints to enforce domain knowledge, these models risk producing spurious correlations rather than capturing causative or physically meaningful relationships.
- **Rise of the emerging paradigm of Theory-Guided Data Science** [@karpatne2017theory]
- The **benefits** of Theory-guided Data Science include:
  - Improved generalizability to OOD data and domains.
  - Enhanced robustness in low-data regimes.
  - Increased interpretability of model predictions.
  - Ensuring physically consistent and meaningful output
- Traditional **techniques** of integrating knowledge into machine learning include
  - feature engineering
  - domain-aware labeling
  - structured regularization
- More recent **techniques** introduce deeper forms of knowledge integration
  - **logical constraints** [@diligenti2017integrating]
  - **algebraic formulations** [@stewart2017label, @daw2017physics]
  - differential equations embedded within neural network architectures [@raissi2017physics]
- [@vonrueden2021informed] provided a comprehensive **taxonomy** of the avaliable techniques.All approaches are categorized on three key aspects:

  - Knowledge source – the origin of the integrated knowledge, whether it stems from established scientific theories, empirical observations, or expert intuition.
  - Knowledge representation – the format in which the knowledge is encoded, such as mathematical equations, logical rules, or probabilistic models. - When differential or algebraic equations are involved, the final solution may exhibit a partially known behavior or be subject to constraints that can be formalized
    mathematically. **Constraints** are typically represented using algebraic equations
    or inequalities.
    - In [@muralidhar2018incorporating], the authors investigate methods for **embedding priors such as bounds and monotonicity constraints** into learning processes
  - Knowledge integration – the stage in the machine learning pipeline where domain knowledge is incorporated

    - at the beginning, by **embedding it within the training dataset**
    - in the middle, through the **design of tailored architectures and learning**
    - at the end, by influencing the model’s **output**.

    ```
    Learning with Regularization Terms

    Another way to integrate prior knowledge is by constraining the learning process through the introduction of a physics-informed loss function alongside standard supervised objectives. This general approach can be formalized as follows [@willard2022integrating]:

    L = LSUP(YTRUE,YPRED) + γLPHY(YPRED) + λ R(W ) ; (1)

    where LSUP represents the supervised loss (e.g., Mean Squared Error, cross-entropy), R is an additional regularization term that limits model complexity, and LPHY incorporates physics-informed constraints. The coefficients γ and λ control the relative contributions of these terms. The physics-based term may include algebraic, differential, or logical constraints.
    ```

    ```
    Architectures

    - that inherently respect domain-specific principles [@vonrueden2021informed].
    - structured to incorporate relational inductive biases [@battaglia2018relational] from the outset, shaping their ability to learn meaningful representations even before training begin
    ```

## Integrating Physics-Based Modeling With Machine Learning: A Survey (Willard et al., 2022)

see paper pdf in: `bibliography/theory-guided/2020 Willard _ Integrating Physics-Based Modeling With Machine Learning.pdf`

### 2. Objectives of Physics-ML Integration

### 3. Physics-ML Methods

#### 3.1 Physics-Guided Loss Function

Loss = LossTRN(Ytrue,Ypred) + λR(W) +γLossPHY(Ypred)

#### 3.2 Physics-Guided Initialization

> Since many ML models require an initial choice of model parameters before training, researchers
> have explored different ways to physically inform a model starting state. For example, in NNs,
> weights are often initialized according to a random distribution prior to training. Poor initialization
> can cause models to anchor in local minima, which is especially true for deep neural networks.
> However, if physical or other contextual knowledge can be used to help inform the initialization
> of the weights, model training can be accelerated or improved [@jia2020physics]. One way to inform the
> initialization to assist in model training and escaping local minima is to use an ML technique
> known as transfer learning. In transfer learning, a model can be pre-trained on a related task prior
> to being fine-tuned with limited training data to fit the desired task. The pre-trained model serves
> as an informed initial state that ideally is closer to the desired parameters for the desired task
> than random initialization. One way to harness physics-based modeling knowledge is to use the
> physics-based model’s simulated data to pre-train the ML model, which also alleviates data paucity
> issues.
>
> — (Willard et al., 2022)

#### 3.3 Physics-Guided Architecture

- Intermediate Physical Variables: ascribe physical meaning for certain neurons in the NN ... they can help extract physically meaningful hidden representation that can be interpreted
- Encoding invariances and symmetries: state-of-the-art deep learning architectures already encode certain types of invariance; for example, RNNs encode time invariance and CNNs can implicitly encode spatial translation, rotation, scale variance. In the same way, scientific modeling tasks may require other invariances based on physical laws ... e.g. adding a higher-order multiplicative layer that ensures the prediction lies on a rotationally invariant tensor basis
- Physics-informed architectures for discovering governing equations
- In Section 2.7, symbolic regression is mentioned as an approach that has shown success.
- Encoding other domain-specific physical knowledge. Various other domain-specific physical information can be encoded into architecture that doesn’t exactly correspond to known invariances but provides meaningful structure to the optimization process depending on the task at hand.
- Currently, human experts have manually developed the majority of domain knowledge-encoded employed architectures, which can be a time-intensive and error-prone process. Because of this, there is increasing interest in automated neural architecture search methods [13, 73, 115]. A young but promising direction in ML architecture design is to embed prior physical knowledge into neural architecture searches. Ba et al. [12] adds physically meaningful input nodes and physical operations between nodes to the neural architecture search space to enable the search algorithm to discover more ideal physics-guided ML architectures.

- Auxiliary Task in Multi-Task Learning. An example of an auxiliary task in a multi-task learning framework might be related to ensuring physically consistent solutions in addition to accurate predictions ... a task-constrained loss function can be formulated to allow errors of related tasks to be back-propagated jointly to improve model generalization ... Early work in a computational chemistry application showed that a NN could be trained to predict energy by constructing a loss function that had penalties for both inaccuracy and inaccurate energy derivatives with respect to time as determined by the surrounding energy force field [199].
- Physics-guided Gaussian process regression. Gaussian process regression (GPR) [265] is a nonpara
  metric, Bayesian approach to regression that is increasingly being used in ML applications ... In GPR, first a Gaussian process prior must be assumed in the form of a mean function and a matrix-valued kernel or covariance function.

#### 3.4 Residual modeling

The oldest and most common approach for directly addressing the imperfection of physics-based
models in the scientific community is residual modeling, where an ML model (usually linear
regression) learns to predict the errors, or residuals, made by a physics-based model [82, 247].

#### 3.5 Hybrid Physics-ML Models

One straightforward method to combine physics-based and ML models is to feed the output
of a physics-based model as input to an ML model. Karpatne et al [128] showed that using the
output of a physics-based model as one feature in an ML model along with inputs used to drive the
physics-based model for lake temperature modeling can improve predictions.

## Conclusion

The literature indicates a strong convergence towards **Neuro-symbolic** and **Theory-guided** AI. In the context of `pykt-toolkit`, moving beyond pure Transformer-based models (like AKT/SAKT) to models that explicitly encode educational psychology principles (as seen in TGEL-Transformer) represents the current State-of-the-Art (SOTA) direction.
