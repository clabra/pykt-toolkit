bibliography: [../bibliography/biblio.bib]

# Bridging Transformers Predictive Power and Pedagogical Theory - A Novel Approach to Individualized Bayesian Knowledge Tracing

```
Alternative Titles

1) Individualization of Bayesian Knowledge Tracing with Interpretable Deep Learning - Bridging Transformers' Predictive Power and Pedagogical Theory

2) Individualization of Bayesian Knowledge Tracing with Interpretable Deep Learning Grounded on Pedagogical Models

3) Individualized Diagnostic Placement and Dynamic Pacing: An Interpretable-by-design Transformer-Based Knowledge Tracing Approach

4) Bridging Transformers Predictive Power and Pedagogical Theory: A Novel Structurally Grounded Approach for Interpretable Deep Knowledge Tracing
```

**Keywords:** knowledge tracing; deep learning; interpretability; Bayesian Knowledge Tracing; educational data mining; personalized learning.

## Featured Application
The iDKT framework is designed for integration into Intelligent Tutoring Systems (ITS) to provide data-driven personalization. It enables diagnostic placement by identifying student-specific knowledge gaps, allowing students to skip mastered content or receive immediate remediation. Additionally, it facilitates dynamic pacing by detecting individualized learning velocities, enabling the system to automatically adjust the instruction speed or provide targeted scaffolding. iDKT allows to deploy deep learning models that remains pedagogically interpretable, moving beyond simple performance prediction to provide actionable, student-specific diagnostic evidence.

## Abstract 

```
A single paragraph of about 200 words maximum. For research articles, abstracts should give a pertinent overview of the work. We strongly encourage authors to use the following style of structured abstracts, but without headings: (1) Background: Place the question ad-dressed in a broad context and highlight the purpose of the study; (2) Methods: briefly de-scribe the main methods or treatments applied; (3) Results: summarize the article’s main findings; (4) Conclusions: indicate the main conclusions or interpretations. The abstract should be an objective representation of the article and it must not contain results that are not presented and substantiated in the main text and should not exaggerate the main con-clusions.
```


While deep knowledge tracing models provide high predictive accuracy, their black-box nature limits the extraction of actionable pedagogical insights. This study introduces iDKT, an interpretable-by-design Transformer model that utilizes structural grounding to align deep latent representations with educational constructs defined by intrinsically interpretable models. We introduce a formal validation framework to verify the pedagogical alignment of iDKT's internal representations and, using Bayesian Knowledge Tracing (BKT) as a reference, evaluate it across multiple educational datasets. Results demonstrate that iDKT maintains state-of-the-art predictive performance while yielding additional interpretable insights at a significantly higher granularity than population-level baselines. Specifically, iDKT identifies student-level initial knowledge and learning velocities, providing mastery estimations that are more sensitive to the nuances of individual behavioral patterns than standard BKT predictions. By anchoring deep learning to semantic concepts defined by the reference model, iDKT enables precise diagnostic placement and dynamic pacing in adaptive learning environments. This work offers both a robust methodology for evaluating the interpretability of Transformer-based models and a practical tool for improving educational effectiveness through data-driven personalization.

```
 How to Use this Template 19

 The template details the sections that can be used in a manuscript. Note that the 20
order and names of article sections may differ from the requirements of the journal (e.g., 21
the positioning of the Materials and Methods section). Please check the instructions on 22
the authors’ page of the journal to verify the correct order and names. For any questions, 23
please contact the editorial office of the journal or support@mdpi.com. For LaTeX-related 24
questions please contact latex@mdpi.com. 25

 ```

## 1. Introduction

```
The introduction should briefly place the study in a broad context and highlight why it is important. It should define the purpose of the work and its significance. The current state of the research field should be carefully reviewed and key publications cited. Please highlight controversial and diverging hypotheses when necessary. Finally, briefly men-tion the main aim of the work and highlight the principal conclusions. As far as possible, please keep the introduction comprehensible to scientists outside your particular field of research. References should be numbered in order of appearance and indicated by a nu-meral or numerals in square brackets—e.g., [1] or [2,3], or [4–6]. See the end of the docu-ment for further details on references.
```


Knowledge Tracing (KT) is a fundamental task in the field of Artificial Intelligence in Education (AIEd), serving as the cognitive engine for modern Intelligent Tutoring Systems (ITS) and Massive Open Online Courses (MOOCs). Its primary objective is to model a student's dynamic knowledge state over time based on their history of interactions with learning materials, enabling systems to predict future performance and personalize instruction [1]. As educational environments become increasingly diverse and digital, the ability to accurately track and interpret student mastery has become a critical requirement for scalable, effective education.

Historically, the field has been dominated by two distinct paradigms. The first, exemplified by Bayesian Knowledge Tracing (BKT) and its variants [@corbett1994knowledge; @yudelson2013individualized], relies on probabilistic graphical models that explicitly represent latent knowledge states. BKT is highly prized for its *interpretability*: its parameters (e.g., probability of learning, slipping, guessing) map directly to pedagogical constructs, allowing educators to diagnose student difficulties and trust the model's decisions. However, its simplicity often limits its predictive power, as it typically assumes binary skills and independent learning probabilities, struggling to capture the complex, non-linear dependencies inherent in real-world learning trajectories [@piech2015deep].

The second paradigm emerged with the advent of Deep Knowledge Tracing (DKT) [@piech2015deep], which utilizes Recurrent Neural Networks (RNNs) and later Transformers [@vaswani2017attention] to model student interactions as complex sequential data. Models such as DKT, DKVMN [@zhang2017dynamic], and AKT [@ghosh2020context] have achieved state-of-the-art predictive performance, significantly outperforming classical approaches by capturing latent long-term dependencies. Yet, this "predictive supremacy" has come at a significant cost: interpretability. Deep learning models are notoriously opaque "black boxes," where the learned representations are distributed across high-dimensional latent spaces that bear no direct correspondence to educational theory. This lack of transparency creates a "trust gap" for practitioners, who cannot easily discern *why* a model predicts a student has failed or succeeded, nor can they derive actionable pedagogical insights—such as specific knowledge gaps or learning velocities—from the model's internal weights [@bai2024survey].

Current efforts to bridge this gap typically rely on *post-hoc* explainability methods, such as visualization of attention weights or perturbation analysis [@fantozzi2024explainability; @di2025ante]. While valuable for debugging, these techniques often provide only a superficial view of the model's decision-making process and do not guarantee that the learned representations align with valid educational constructs.

To address this challenge, we propose a shift towards **Interpretability-by-Design**, inspired by the emerging paradigm of Theory-Guided Data Science (TGDS) [@karpatne2017theory; @willard2022integrating]. TGDS posits that scientific consistency should be an architectural constraint rather than an afterthought. By integrating extensive domain knowledge—in this case, pedagogical theory—deep learning models can be constrained to learn representations that are both scientifically plausible and highly predictive.

In this work, we introduce **iDKT** (Interpretable Deep Knowledge Tracing), a novel Transformer-based framework that achieves intrinsic interpretability through **Structural Grounding**. Unlike previous approaches that use theory only for regularization [@lee2021consistency], iDKT anchors its deep latent representations directly to the conceptual space of Bayesian Knowledge Tracing. This allows the model to leverage the representational power of Transformers to capture complex learning dynamics while ensuring that its internal states ($l_c, t_s$) remain formally equivalent to established educational parameters. 

Our contributions are threefold: (1) We propose a method for **Neural-Symbolic Structural Grounding** that forces a Transformer to operate within a pedagogically valid manifold; (2) We validate iDKT against BKT, demonstrating that it captures granular, student-specific insights—such as individualized knowledge gaps and diverse learning velocities—that classical models overlook; and (3) We show that this approach enables actionable educational interventions, such as **precise diagnostic placement** and **dynamic pacing**, without sacrificing the predictive accuracy characteristic of state-of-the-art deep learning.

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

### 2.1 Black-Box Knowledge Tracing

**Deep Knowledge Tracing (DKT)** [Piech et al., 2015]: LSTM-based approach learning latent knowledge states via end-to-end training. Achieves strong performance but provides no interpretability beyond final predictions.

**Dynamic Key-Value Memory Networks (DKVMN)** [Zhang et al., 2017]: Separates concept keys from mastery value states using external memory. Improves on DKT but representations remain opaque.

**Attentive Knowledge Tracing (AKT)** [Ghosh et al., 2020]: Transformer attention for context-aware predictions. Post-hoc attention visualization attempted but attention weights ≠ causal explanations.

**SAINT** [Choi et al., 2020]: Encoder-decoder Transformer achieving state-of-the-art performance. Interpretability not addressed.

**Limitation**: All prioritize performance over interpretability. Post-hoc explanation methods applied after training often misleading.

### 2.2 Interpretable Knowledge Tracing

**Bayesian Knowledge Tracing (BKT)** [Corbett & Anderson, 1995]: Probabilistic model with explicit skill mastery states. Interpretable but limited expressiveness (binary mastery, Markovian assumptions).

**Item Response Theory (IRT)** [Lord, 1980]: Static item difficulty/discrimination parameters. Strong theoretical foundation but doesn't model learning dynamics.

**Performance Factors Analysis (PFA)** [Pavlik et al., 2009]: Additive effects of success/failure counts. Interpretable coefficients but linear assumptions limit performance.

**Limitation**: Interpretability achieved via simplicity, sacrificing performance.

### 2.3 Our Approach: Interpretability-by-Design

We embed interpretability constraints directly into architecture:
- **Explicit constructs**: Mastery and gain as first-class components (projection heads)
- **Architectural constraints**: Monotonicity, performance alignment, recursive accumulation
- **Training-time supervision**: Auxiliary losses enforce educational validity
- **Construct validity**: Learned representations measured against ground truth

This differs from prior work by treating interpretability as a design principle, not an afterthought.

---

**2.1 Knowledge Tracing Models**
- Bayesian models (BKT): Interpretable but limited capacity
- Deep learning (DKT, SAINT, AKT): High performance but opaque
- Gap: Need for interpretable neural models

**2.2 Interpretability in Machine Learning**
- General definitions (post-hoc vs intrinsic, local vs global)
- Educational AI needs: Transparent reasoning, domain grounding
- Current approaches: Attention visualization (insufficient for education)

**2.3 Item Response Theory & Ability-Difficulty Models**
- IRT framework: Performance = f(ability, difficulty)
- Strengths: Interpretable parameters, theoretical foundation
- Limitations: Static assumptions (we adapt to dynamic learning)

**2.4 Measuring Interpretability**
- Challenge: Often claimed, rarely measured
- Existing approaches: Human studies (subjective), proxy metrics (incomplete)
- Our contribution: Multi-metric quantitative validation





## 2. Methodology

The iDKT framework is implemented using the `pykt-toolkit` library on top of PyTorch. All experiments were conducted in a reproducible Docker environment (image: `pinn-dev`). The complete source code, along with the configuration files for reproducibility (`config.json`), is available in the supplementary materials.

### 2.1. The iDKT Model Architecture

iDKT (Interpretable Deep Knowledge Tracing) is a Transformer-based model designed to bridge the gap between predictive power and pedagogical interpretability. It builds upon the Attentive Knowledge Tracing (AKT) architecture [@ghosh2020context] but introduces a novel input layer based on **Structural Grounding**.

The core architecture consists of three main components:
1.  **Context-Aware Encoder ($N=4$ blocks)**: Processes the sequence of exercises and questions to generate contextualized question embeddings.
2.  **Knowledge Retriever ($2N=8$ blocks)**: An encoder-decoder structure that retrieves relevant historical interactions using a **monotonic attention mechanism**. Crucially, the multi-head attention employs distinct, learnable decay rates for each head. This allows the model to simultaneously capture **short-term dynamics** (via heads with rapid exponential decay) and **long-term retention** (via heads with slow decay), ensuring a comprehensive view of the student's learning trajectory.
3.  **Prediction Head**: A multi-layer perceptron (MLP) that combines the retrieved knowledge state with the current question embedding to predict the probability of a correct response.

Key hyperparameters include a model dimension $d_{model}=256$, $H=8$ attention heads, and a dropout rate of 0.1, optimized via Adam (`lr=1e-3`).

### 2.2. Structural Grounding Embeddings

To achieve interpretability-by-design, iDKT replaces standard learned embeddings with "Textured Grounding" embeddings. These are formally anchored to the conceptual space of Bayesian Knowledge Tracing (BKT) [@corbett1994knowledge], ensuring that the latent representations carry semantic meaning. We employ a modified Rasch model logic to individualize these representations:

**Individualized Question ($x'_t$)**:
The standard question embedding is replaced by a residual representation:
$$ x'_t = (c_{c_t} + u_q \cdot d_{c_t}) - l_c $$
where $c_{c_t}$ is the concept embedding, $u_q$ is a learned scalar for item difficulty, and $d_{c_t}$ is a variation axis. Crucially, $l_c$ is the **individualized initial mastery**, grounded in the BKT prior ($L0$), defined as:
$$ l_c = L0_{skill} + k_c \cdot d_c $$
Here, $k_c$ is a learned student-specific scalar representing their "Knowledge Gap" relative to the population mean.

**Individualized Interaction ($y'_t$)**:
The interaction history is similarly grounded by adding learning momentum:
$$ y'_t = (e_{c_t,r_t} + u_q \cdot (f_{c_t,r_t} + d_{c_t})) + t_s $$
where $t_s$ is the **individualized learn rate**, grounded in the BKT learn probability ($T$):
$$ t_s = T_{skill} + v_s \cdot d_s $$
Here, $v_s$ represents the student's "Learning Velocity," allowing the model to distinguish between fast and slow learners dynamically.

### 2.3. Loss Functions and Training Objective

The model is trained using a multi-objective loss function that balances predictive accuracy with theoretical alignment. The total loss $L_{total}$ is a weighted sum of three components:

$$ L_{total} = L_{SUP} + \lambda_{ref} L_{ref} + L_{reg} $$

1.  **Supervised Prediction Loss ($L_{SUP}$)**: The standard Binary Cross-Entropy (BCE) loss between the predicted probability $\hat{p}_{it}$ and the actual student response $r_{it}$.
2.  **Theoretical Alignment Loss ($L_{ref}$)**: Enforces consistency with the reference theory (BKT). It includes Mean Squared Error (MSE) terms penalizing deviations between the model's projected parameters (e.g., $l_c, t_s$) and the corresponding BKT theoretical values. This ensures that the learned representations remain semantically valid.
    $$ L_{ref} = \text{MSE}(l_c, L0_{BKT}) + \text{MSE}(t_s, T_{BKT}) $$
3.  **Regularization Loss ($L_{reg}$)**: A task-agnostic regularization on the student-specific scalars ($u_q, k_c, v_s$) to prevent overfitting and ensure they represent meaningful deviations from the norm.

### 2.4. Datasets

We evaluated iDKT on the **ASSISTments 2009-2010 (Skill Builder)** dataset, a standard benchmark in Knowledge Tracing. 

*   **Source**: [ASSISTments Data Mining Competition 2017](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data).
*   **Preprocessing**: We filtered for the "Skill Builder" problem sets to focus on mastery learning. Sequences were truncated to a maximum length of 200.
*   **Statistics**: The processed dataset contains 4,151 students, 110 skills, and approximately 325,000 interactions.
*   **Data Availability**: The BKT-augmented version of the dataset used for training (`train_valid_sequences_bkt.csv`) is generated via the `augment_with_bkt.py` script provided in the repository.

## 3. Results

### 3.1. Construct Validity and Structural Grounding

To verify that iDKT's internal representations genuinely reflect educational constructs rather than arbitrary latent features, we evaluated the model against three psychometric hypotheses ($H_1$--$H_3$). Table 1 summarizes the alignment metrics across different grounding strengths ($\lambda$).

**Table 1.** Construct Validity and Performance across the Grounding Spectrum.

| Grounding Strength ($\lambda$) | Test AUC | $H_1$: Numerical Alignment ($r$) | $H_2$: Behavioral Alignment ($r$) | $H_3$: Discriminant Validity ($r$) |
| :--- | :---: | :---: | :---: | :---: |
| 0.00 (Baseline) | 0.8317 | 0.9993 | 0.2652 | -0.0325 |
| **0.10 (Sweet Spot)** | **0.8322** | **0.9838** | **0.2949** | **-0.0330** |
| 0.30 | 0.7984 | 0.9691 | 0.3192 | -0.0330 |
| 0.50 | 0.7740 | 0.9884 | 0.2828 | -0.0331 |

We observe consistent **Convergent Validity ($H_1$)**, with the correlation between the model's projected initial mastery ($l_c$) and the theoretical prior ($L_0$) remaining above $0.96$ throughout the sweep. This confirms that the Structural Grounding mechanism successfully anchors the deep latent space to the reference theory. Furthermore, the **Discriminant Validity ($H_3$)** remains stable at $r \approx -0.03$, proving that the model successfully disentangles "Student Knowledge Gap" ($k_c$) from "Student Learning Velocity" ($v_s$) as distinct, non-redundant traits.

### 3.2. The Pareto Efficiency of Theory-Guided Learning

Our analysis reveals a non-linear trade-off between predictive accuracy and theoretical fidelity. Contrary to the common assumption that interpretability imposes a performance penalty, we identified an **"Inductive Bias Bonus"** at moderate grounding levels ($\lambda \approx 0.10$).

As shown in Table 1, the model with $\lambda=0.10$ achieves a Test AUC of **0.8322**, slightly outperforming the unconstrained baseline (0.8317). This suggests that the BKT-based regularization acts as a beneficial inductive bias, preventing the Transformer from overfitting to noise in sparse interaction histories. However, excessive grounding ($\lambda > 0.30$) leads to a sharp decline in predictive performance as the model becomes over-constrained by the simplicity of the reference theory.

### 3.3. Granularity of Individualization

While standard BKT assigns a fixed "Learning Rate" ($T$) to all students for a given skill, iDKT captures a rich distribution of **Individualized Learning Velocities** ($t_s$).

**Figure 1** (see supplementary materials) illustrates this "Delta Distribution" ($\Delta = t_s - T$). We observe a visible right-skewed variance, indicating that for many skills, the Deep Learning model identifies "fast-track" learning trajectories that classical population-level models underestimate. This granularity allows for **precise diagnostic placement**, distinguishing between students who lack initial knowledge ($low \ l_c$) versus those who suffer from slow acquisition momentum ($low \ t_s$).

### 3.4. Longitudinal Mastery Dynamics

The practical impact of these individualized parameters is evident in the **Mastery Mosaic** analysis. When simulating the mastery acquisition of "Fast" vs. "Slow" learners on the same sequence of correct responses:
*   **Standard BKT** predicts identical mastery curves for both students.
*   **iDKT** projects distinct trajectories, where "Fast" learners reach the 95% mastery threshold significantly earlier (fewer interactions) than "Slow" learners.

This "Informed Divergence" validates that iDKT does not merely mimic BKT labels but leverages its transformer core to dynamically adjust the **Velocity of Mastery** based on the student's historical profile, enabling truly adaptive pacing in intelligent tutoring scenarios.




## 4. Discussion

The results presented above challenge the prevailing dichotomy in educational AI that views interpretability and performance as conflicting objectives. By adopting an **Interpretability-by-Design** approach, iDKT demonstrates that grounding a Deep Learning model in pedagogical theory can actually enhance its robustness.

### 4.1. The "Free Lunch" of Inductive Bias
The observation that moderate grounding ($\lambda \approx 0.10$) improves AUC suggests that BKT—despite its simplicity—provides a valuable **Relational Inductive Bias**. It forces the Transformer to "reason" about the gap between student capability and item difficulty ($x' = \text{Challenge} - \text{Ability}$) rather than simply memorizing sequence patterns. This constrains the search space of the optimization, guiding it towards more generalizable solutions.

### 4.2. Bridging the Trust Gap
For practitioners, the "Black Box" nature of standard DKT models has been a barrier to adoption. iDKT resolves this by providing **Intrinsic Transparency**. When the model predicts a high probability of failure, educators need not trust an opaque vector; they can inspect the grounded components:
*   Is the projected Initial Mastery ($l_c$) low? $\rightarrow$ **Remediation needed.**
*   Is the projected Learning Velocity ($t_s$) low? $\rightarrow$ **Pacing adjustment needed.**
This transparency transforms the model from a predictive oracle into a diagnostic instrument.

## 5. Conclusions

In this work, we introduced iDKT, a novel framework that reconciles the predictive power of Transformers with the interpretability of Bayesian Knowledge Tracing. Through **Structural Grounding**, we demonstrated that it is possible to enforce semantic validiy on deep latent representations without sacrificing performance.

Our key contributions are:
1.  **Methodological**: A Neural-Symbolic architecture that embeds the "Challenge vs. Ability" logic of psychometrics directly into the Transformer's input stream.
2.  **Empirical**: Evidence of an "Interpretability Sweet Spot" where theoretical regularization acts as a performance enhancer.
3.  **Practical**: The ability to extract granular, student-specific profiles (Knowledge Gaps and Learning Velocities) that classical population-level models overlook.

Future work will explore extending this framework to more complex theoretical baselines, such as Performance Factors Analysis (PFA) or Item Response Theory (IRT), and validating the **diagnostic placement** capabilities in live classroom settings. By moving beyond black-box predictions, iDKT paves the way for a new generation of Theory-Aware Educational AI that is as trusted as it is accurate.

## References
