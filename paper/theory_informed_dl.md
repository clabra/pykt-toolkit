## References
https://arxiv.org/pdf/2403.07279



## What are good explanations for knowledge tracing? 
Goal-oriented:

- Comprehension of learning processes
- Trust and Acceptance: encourage the result acceptance and boost model satisfaction
- Model Improvement: enables model optimization
- Credibility: improve the model’s credibility


## Theoretical Models
Most of the theoretical models mentioned are based in:

- IRT
- Forgetting Curves


## Taxonomy Category
Ante-hoc methods -> Transparent Models

- Bayesian Knowledge Tracing (BKT)
- Item Response Theory (IRT)
- Factor Analysis Model (FAM)

Ante-hoc methods -> Model-intrinsic interpretability
- Incorporate attention mechanism
- Integrating educational psychology and other theories


Integrating Educational Psychology and Other Theories
This approach integrates educational principles directly into the model structure, using psychologically or mathematically meaningful parameters to enhance transparency.



## Key Theoretical Integrations

### Item Response Theory (IRT)

IRT provides predefined interpretable parameters (such as student ability and item difficulty) to describe student behaviour.

#### Deep-IRT
This model integrates Dynamic Key-Value Memory Networks (DKVMNs) with IRT. DKVMN infers learners' abilities and item difficulties via neural networks, and these network-derived parameters are then used within the IRT framework to predict answer correctness, combining DLKT's predictive strength with IRT's interpretability

#### Time-and-Concept Enhanced Deep Multidimensional Item Response Theory (TC-MIRT)

This advanced method integrates the interpretable parameters of multi-dimensional IRT (MIRT) into a recurrent neural network. This integration allows the model to predict student states and generate interpretable parameters for each specific knowledge field, addressing the limitation of one-dimensional IRT parameters in explaining complex behaviours.

#### Knowledge Interaction-Enhanced Knowledge Tracing (KIKT)

This model utilizes the IRT framework to simulate learners' performance and establish an interpretable relationship between learner proficiency and project characteristics.

### Learning and Forgetting Curve Theories

Models explicitly integrate the mechanisms of learning and forgetting, rooted in Ebbinghaus's forgetting curve theory (which posits memory decline over time) and learning curve theory (which sees knowledge acquisition as a result of repeated practice).

#### Time-and-Content Balanced Attention (TCBA)

AlignKT utilizes TCBA, inspired by the Ebbinghaus Forgetting Curve, to model forgetting behaviour. This method uses a fitting formula that regulates the significance of temporal distance and mastery level to determine memory retention, leveraging this cognitive principle in its attention calculation.

#### PKT and Others

Models like Progressive Knowledge Tracing (PKT) are designed based on constructive learning and IRT, explicitly incorporating interpretable and educationally meaningful parameters. Other models have constructed learning and forgetting factors at the learner level as additional features to track and explain changes in knowledge proficiency.

### Constructivist Learning Theory

Models like Ability Boosted Knowledge Tracing (ABKT) and PKT are inspired by this theory, which focuses on knowledge internalization as the basis for mastery differences, employing methods like continuous matrix factorization to simulate this process and enhance interpretability.

While integrating psychological theories offers interpretability framed within established academic frameworks, these methods can be limited by the specificity and scope of the underlying theories, potentially failing to capture the dynamic and multifaceted nature of real-world learning fully.

---

The "Implementing Knowledge-Based Loss Functions in Machine Learning" note previously provided a framework for incorporating domain knowledge via regularization (or penalty) terms, primarily drawing on Physics-Informed Machine Learning (PIML) and Bayesian Knowledge Tracing (BKT) concepts.
The new sources reinforce and expand upon the practical applications of this methodology, particularly in Deep Learning-based Knowledge Tracing (DLKT) and Physics-Informed Graph Neural Networks (PIGNNs), offering concrete examples of how theory and constraints are translated into loss or penalty terms, often referred to as L 
physics
​
  or L 
k
​
 .
Here is the enhanced information categorized by method and specific implementation:
1. Augmenting Loss Functions with Physics-Informed Penalties (PIML/PIGNN)
The general objective function integrates the knowledge-based loss term L 
k
​
  (or L 
physics
​
 ) into the optimization goal, typically alongside the data-fitting loss L 
data
​
  and standard regularization R:
f 
∗
 =arg min 
f
​
 (λ 
l
​
  
i
∑
​
 L(f(x 
i
​
 ),y 
i
​
 )+λ 
r
​
 R(f)+λ 
k
​
 L 
k
​
 (f(x 
i
​
 ),x 
i
​
 ))
A. Differential Equations as Loss Terms (Residual Loss)
This method ensures the model adheres to governing laws by penalising how much the output u 
w
​
  violates the equation F:
• PDE Residual Minimization: For Physics-Informed Neural Networks (PINNs) and PIGNNs, the loss function explicitly includes the PDE residual loss (L 
r
​
 ), which quantifies the deviation of the GNN’s output from satisfying the known governing equation, forcing the model toward physically plausible solutions.
• Total Loss Structure: In a full PINN setup, the total loss (L) includes terms for the PDE residual (L 
r
​
 ), boundary condition (L 
b
​
 ), initial condition (L 
i
​
 ), and standard data loss (L 
d
​
 ):
L≈ 
N 
r
​
 
λ 
r
​
 
​
 ∑∣∣F(u 
w
​
 ;θ)(x 
i
​
 )∣∣ 
2
 + 
N 
i
​
 
λ 
i
​
 
​
 ∑∣∣I(u 
w
​
 ;θ)(x 
i
​
 )∣∣ 
2
 + 
N 
b
​
 
λ 
b
​
 
​
 ∑∣∣B(u 
w
​
 ;θ)(x 
i
​
 )∣∣ 
2
 + 
N 
d
​
 
λ 
d
​
 
​
 ∑∣∣u 
w
​
 (x 
i
​
 )−u(x 
i
​
 )∣∣ 
2
 
• Variational Formulation: Instead of using the strong-form residual, some PIGNNs minimize the weak or variational form of the governing equations (e.g., based on the Galerkin method). This often results in lower-order derivatives and can be more stable and robust against noise than minimizing strong-form residuals.
B. Algebraic Constraints and Conservation Laws
These are typically added as penalty terms (L 
k
​
 ) to enforce fundamental domain rules:
• Conservation Laws: Physical principles like conservation of mass or energy can be encoded directly as loss terms in PIGNNs.
• Constrained Optimisation (Hard Constraints): While soft penalties are common, some model architectures and specialized layers enforce constraints strictly (hard constraints) rather than relying on penalties in the loss function. For example, for simple boundary conditions, a post-processing layer can be designed to force the solution to zero at boundaries. However, complex or arbitrary boundary conditions are challenging to enforce strictly, often necessitating alternative approaches or loss term adjustments.
2. Integrating Cognitive Principles via Learning Algorithms
In Knowledge Tracing (KT), domain knowledge related to learning and memory is incorporated, acting as a form of weak regularization or bias within the optimization routine.
A. Monotonicity and Forgetting Mechanisms
Cognitive principles, particularly those related to forgetting and temporal decay, are translated into constraints on attention weights or loss terms:
• Monotonic Multi-Head Attention (MMHA): This mechanism explicitly introduces temporal monotonicity constraints by applying an exponential decay to the attention weights, thereby diminishing the influence of distant exercises. The calculation includes a trainable decay coefficient (θ) and temporal distance (d(t,τ)).
max( 
d 
k
​
 

​
 
exp(−θ⋅d(t,τ))q 
t
​
 k 
τ
T
​
 
​
 )
• Attention with Linear Bias Multi-Head Attention (AMHA): This enhances local attention by adding a position-specific bias term (λ 
h
​
 ) based on relative temporal distances (t−τ), mimicking cognitive forgetting patterns.
softmax( 
d 
k
​
 

​
 
q 
t
​
 k 
τ
T
​
 +λ 
h
​
 ⋅(t−τ)
​
 )
• Time-and-Content Balanced Attention (TCBA): AlignKT incorporates TCBA which is inspired by the Ebbinghaus Forgetting Curve. The attention score (α 
t,i
​
 ) is regulated using an exponential decay function based on temporal distance (∣t−i∣) and the current mastery level (q 
t
T
​
 k 
i
​
 ), which ensures that forgetting is explicitly modeled.
B. Enhancing Model Robustness (Contrastive Loss)
In deep KT (DLKT), complex models benefit from auxiliary loss terms to stabilize training and improve generalisation:
• Contrastive Learning Module (CL): Models like AlignKT incorporate contrastive learning modules that use InfoNCE (Information Noise-Contrastive Estimation) to define contrastive loss (l 
CL
​
 ), promoting robustness against minor perturbations in data. This is achieved by penalizing representations of augmented (positive) samples and dissimilar (negative) samples, helping the model discriminate semantics more clearly. The final objective function combines the standard binary cross-entropy loss (l 
BCE
​
 ) with the contrastive loss via a hyperparameter (λ):
loss=l 
BCE
​
 +λ⋅(l 
CLc
​
 +l 
CLs
​
 )
3. Practical Challenges in Optimization
Optimising hybrid loss functions remains a major implementation challenge:
• Loss Imbalance: The scale and convergence speeds of different losses (e.g., data loss, boundary loss, PDE residual loss) often conflict. The simplest approach uses static hyperparameters (λ) to weigh the terms.
• Adaptive Weighting: More sophisticated methods dynamically adjust these weights during training (loss re-weighting) to balance the optimisation process and avoid convergence issues. Methods include:
    ◦ Gradient Norm Reweighting: Adjusting λ based on the maximum gradient norm of one loss relative to another, to mitigate dominance by high-frequency losses (like PDE residuals).
    ◦ NTK Reweighting: Using Neural Tangent Kernel (NTK) analysis to dynamically tune λ based on the kernel trace to balance optimisation across different frequencies.
    ◦ Inverse-Dirichlet Weighting: Using gradient variance to adjust λ and alleviate issues like vanishing gradients in multi-scale modeling.
• Adaptive Data Re-sampling: Instead of adjusting weights, collocation points (samples used to evaluate losses) can be dynamically re-sampled to focus training effort on areas with higher error (higher residual loss).
In essence, whether the prior knowledge is a physical law (PDE) or a cognitive principle (forgetting curve), the learning algorithm incorporates it by formulating a differentiable cost or penalty term (L 
k
​
 ) which is then carefully balanced within the comprehensive optimization objective.

 