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

- Hard constraints (HC): reducinng the network output space in order to make the solution exactly fulfill the known hard constraints (HC) ... to give a hysical value to some output neurons, e.g., by means of a loss function enforcing them o be equal to those variables [19], or by using models pretrained on an intermediate task [ 28].

## Integrating Physics-Based Modeling With Machine Learning: A Survey (Willard et al., 2022)

see paper pdf in: `bibliography/theory-guided/2020 Willard _ Integrating Physics-Based Modeling With Machine Learning.pdf`

### 2. Objectives of Physics-ML Integration

### 3. Physics-ML Methods

#### 3.1 Physics-Guided Loss Function

$Loss = Loss_{TRN}(Ytrue,Ypred) + \lambda R(W) +\gamma Loss_{PHY}(Ypred)$

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

## Process-based Models

- @karpatne2024physics

"The conventional approach for scientific modeling is to use process-based models (also referred to asphysics-based or science-based models), where the solution structure of relationships between inputs and response variables is rooted in scientific equations (e.g., laws of energy and mass conservation). In particular, process-based models make use of scientific equations to infer the evolution of time-varyinglatent or hidden variables of the system, also referred to as system states (see Figure 1). Additionally, process-based models often involve model parameters that need to be specified or calibrated (oftenwith the help of observational data) for approximating real-world phenomena ... Taken together, the conventional paradigm of scientific modeling—developed over centuries of systematic research forms the foundation of our present-day understanding of scientific systems across a wide spectrum of problemsin environmental sciences. A key feature of process-based models is their ability to provide a mechanistic understanding of the cause-effect mechanisms between input and output variables that can be used as a building blockfor advancing scientific knowledge. As a result, process-based models are continually updated andimproved by the scientific community to fill knowledge gaps in current modeling standards and discovernew theories and formulations of scientific equations that better match with observations and arescientifically meaningful and explainable. Since process-based models are rooted in scientific equationsthat are assumed to hold true in any testing scenario, they are also expected to easily generalize evenoutside the data used during model building. For example, process-based models can be made toextrapolate in space (e.g., over different geographic regions), in time (e.g., forecasting the future of asystem under varying forcings of input drivers), or in scale (e.g., discovering emergent properties oflarger-scale systems using models of smaller-scale system components)"

## Methods

- @karpatne2024physics

Main methods: knowledge-guided learning algorithms (e.g., using loss functions), knowledge-guided architecture of ML models, and knowledge-guided pretraining or initialization of ML models.

Knowledge-guided learning: the predominant method for integrating scientific knowledge into ML is to directly introduce physical principles into the training objective of ML models [20, 71, 78, 79, 128].

- @evopinn

the multi-objective optimization must reconcile inherently conflicting
objectives between data fitting and physics adherence. Iden-
tifying an appropriate aggregation of objectives to arrive at
good (physically satisfactory) trade-off solution is also non-
trivial [124] Moreover, the shape of the Pareto front formed
by the multiple objectives is not known beforehand [125],
with simple weighted-sum aggregations being insufficient to
support all parts of non-convex fronts [126].
...
Multi-objective EAs (MOEAs) are tailor-made to address
these issues [127]. In particular, Pareto dominance-based vari-
ants of MOEAs [128] do not require the multiple loss terms to
be scalarized, and hence are less sensitive to incommensurable
objectives.

[124] Z. Wei et al., “How to select physics-informed neural networks in the absence of ground truth: a pareto front-based strategy,”
in 1st Workshop on the Synergy of Scientific and Machine Learning Modeling@ ICML2023, 2023.
[125] F. M. Rohrhofer, S. Posch, C. G ̈oßnitzer, and B. C. Geiger, “On the apparent pareto front of physics-informed neural networks,” IEEE Access, 2023.
[127] C. A. C. Coello, Evolutionary algorithms for solving multi-objective problems. Springer, 2007.
[128] K. Deb and M. Ehrgott, “On generalized dominance structures for multi-objective optimization,” Mathematical and Computational Applications, vol. 28, no. 5, p. 100, 2023

- @vonrueden2021informed

a knowledge-based loss term Lk can be built into the objective function [10], [12]:

Whereas L is the usual label-based loss and R is a regularization function, Lk quantifies the violation of given
prior-knowledge equations.
Note that **Lk only depends on the input features x i and the learned function f** and thus offers the possibility of label-free supervision [13]
...
Algebraic equations and inequalities can be integrated into learning algorithms via **additional loss terms** [12], [13], [33], [35] or, **more generally, via
constrained problem formulation [36], [37], [39]**

[10] M. Diligenti, S. Roychowdhury, and M. Gori, “Integrating prior
knowledge into deep learning,” in Proc. Int. Conf. Mach. Learn.
Appl. 2017, pp. 920–923.
[12] A. Karpatne, W. Watkins, J. Read, and V. Kumar, “Physics-
guided neural networks (PGNN): An application in lake temper-
ature modeling,” 2017, arXiv:1710.11431.
[13] R. Stewart and S. Ermon, “Label-free supervision of neural net-
works with physics and domain knowledge,” in Proc. Conf. Artif.
Intell., 2017, pp. 2576–2582

[33] N. Muralidhar, M. R. Islam, M. Marwah, A. Karpatne, and N. Ram-
akrishnan, “Incorporating prior domain knowledge into deep neu-
ral networks,” in Proc. Int. Conf. Big Data, 2018, pp. 36–45.
[35] R. Heese, M. Walczak, L. Morand, D. Helm, and M. Bortz, “The
good, the bad and the ugly: Augmenting a black-box model with
expert knowledge,” in Proc. Int. Conf. Artif. Neural Netw., 2019,
pp. 391–395.
[36] G. M. Fung, O. L. Mangasarian, and J. W. Shavlik, “Knowledge-
based support vector machine classifiers,” in Proc. 15th Int. Conf.
Neural Inf. Process. Syst., 2003, pp. 537–544.
[37] O. L. Mangasarian and E. W. Wild, “Nonlinear knowledge-
based classification,” IEEE Trans. Neural Netw., vol. 19,
no. 10, pp. 1826–1832, Oct. 2008.
[39] M. von Kurnatowski, J. Schmid, P. Link, R. Zache, L. Morand,
T. Kraft, I. Schmidt, and A. Stoll, “Compensating data shortages
in manufacturing with monotonicity knowledge,” 2020,
arXiv:2010.15955.

@elhamod2023understanding
The Effect of Knowledge Guidance on Learning A Scientifically Meaningful Latent Space

Can the KGML framework (Knowledge-Guided Machine Learning) help devise a method that discovers biologically-valid and anatomically-relevant species traits
and, as a by-product, deliver better generalization performance on downstream ML tasks?
...
Answering this question entails learning species traits as features in the latent space. The
properties of this learned space and how it corresponds to domain knowledge (e.g., biology
in this application) is the target of my study.

- @theory_guided_data_science

  - 4.1 Theory-guided Initialization
  - 4.2 Theory-guided Probabilistic Models

    Another approach to reduce the variance of model pa-
    rameters (and thus avoid model overfitting) is to introduce
    priors in the model space. An example of the use of theory-
    guided priors is the problem of non-invasive electrophysi-
    ological imaging of the heart.
    ...
    it's easy that the system learns spurious patterns.
    ...
    Incorporating such theory-guided spatial distributions as
    priors and using it along with externally collected ECG data
    in a hierarchical Bayesian model has been shown to provide
    promising results over traditional data science models [26],
    [27]. Another example of theory-guided priors can be found
    in the field of geophysics [51], where the knowledge of
    convection-diffusion equations was used as priors for de-
    termining the connectivity structure of subsurface aquifers.

  - 4.3 Theory-guided Constrained Optimization
  - 4.4 Theory-guided Regularization

  - Hybrid TGDS Models
    An alternate way of creating a hybrid TGDS model is to
    use data science methods to predict intermediate quantities
    in theory-based models that are currently being missed or
    inaccurately estimated. By feeding data science outputs into
    theory-based models, such a hybrid model can not only
    show better predictive performance but also amend the
    deficiencies in existing theory-based models. Further, the
    outputs of theory-based models may also be used as training
    samples in data science components [72], thus creating a
    two-way synergy between them. Depending on the nature
    of the model and the requirements of the application, there
    can be multiple ways of introducing data science outputs
    in theory-based models. In the following, we provide an
    illustrative example of this theme of TGDS research in the
    field of turbulence modeling.

---

## Practical Challenges in Optimization

Optimising hybrid loss functions remains a major implementation challenge:
• Loss Imbalance: The scale and convergence speeds of different losses (e.g., data loss, boundary loss, PDE residual loss) often conflict. The simplest approach uses static hyperparameters (λ) to weigh the terms.
• Adaptive Weighting: More sophisticated methods dynamically adjust these weights during training (loss re-weighting) to balance the optimisation process and avoid convergence issues. Methods include:
◦ Gradient Norm Reweighting: Adjusting λ based on the maximum gradient norm of one loss relative to another, to mitigate dominance by high-frequency losses (like PDE residuals).
◦ NTK Reweighting: Using Neural Tangent Kernel (NTK) analysis to dynamically tune λ based on the kernel trace to balance optimisation across different frequencies.
◦ Inverse-Dirichlet Weighting: Using gradient variance to adjust λ and alleviate issues like vanishing gradients in multi-scale modeling.
• Adaptive Data Re-sampling: Instead of adjusting weights, collocation points (samples used to evaluate losses) can be dynamically re-sampled to focus training effort on areas with higher error (higher residual loss).
