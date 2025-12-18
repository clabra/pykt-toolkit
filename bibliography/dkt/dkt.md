# DKT and Interpretability

## @bai2024survey

### What are good explanations for knowledge tracing?

Goal-oriented:

- Comprehension of learning processes
- Trust and Acceptance: encourage the result acceptance and boost model satisfaction
- Model Improvement: enables model optimization
- Credibility: improve the model’s credibility

### Theoretical Models

Most of the theoretical models mentioned are based in:

- IRT
- Forgetting Curves

### Taxonomy Category

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

These method-
ologies include the deployment of standardized objective metrics, such as stability [80],
fidelity [80], and sensibility [81], to appraise the congruence in xKT’s interpretation of
proximate or analogous data instances and the accuracy in approximating black-box
model predictions.

@article{vilone2021notions,
title={Notions of explainability and evaluation approaches for explainable artificial intelligence},
author={Vilone, G. and Longo, L.},
journal={Information Fusion},
volume={76},
pages={89--106},
year={2021}
}

@article{alvarez2018towards,
title={Towards robust interpretability with self-explaining neural networks},
author={Alvarez Melis, D. and Jaakkola, T.},
journal={Advances in neural information processing systems},
volume={31},
year={2018}
}

causality

This approach aims to uncover the actual causal relationships within learning pro-
cesses, moving beyond the limitations of correlation-based analysis prevalent in many
machine learning models @li2023genetic. By applying techniques such as counterfactual reason-
ing @cornacchia2023auditing, researchers can explore various hypothetical scenarios, and such alternative
learning strategies might lead to diverse learning outcomes. This method enables a
more profound understanding of the direct impact of specific learning activities on
educational results. Such in-depth causal analysis not only improves the scientific rigor
and accuracy of knowledge tracing models but also offers valuable insights for the
development of effective and personalized educational interventions @li2023genetic, @azhunova2022causal. Conse-
quently, knowledge tracing technology has advanced from simply predicting outcomes
to providing actionable insights for enhancing educational practices.

@article{li2023genetic,
title={A genetic causal explainer for deep knowledge tracing},
author={Li, Q. and Yuan, X. and Liu, S. and Gao, L. and Wei, T. and Shen, X. and Sun, J.},
journal={IEEE Transactions on Evolutionary Computation},
year={2023}
}

@article{cornacchia2023auditing,
title={Auditing fairness under unawareness through counterfactual reasoning},
author={Cornacchia, G. and Anelli, V.W. and Biancofiore, G.M. and Narducci, F. and Pomo, C. and Ragone, A. and Di Sciascio, E.},
journal={Information Processing & Management},
volume={60},
number={2},
pages={103224},
year={2023}
}

@article{zhu2023stable,
title={Stable knowledge tracing using causal inference},
author={Zhu, J. and Ma, X. and Huang, C.},
journal={IEEE Transactions on Learning Technologies},
year={2023}
}

The Trade-Off Between Model Performance and Interpretability in
Knowledge Tracing.
In knowledge tracing, the critical future challenge is to
balance model accuracy with interpretability. Achieving this balance requires creat-
ing algorithms that are both precise in prediction and intuitive in understanding.
Future research is expected to focus on refining deep learning architectures to sim-
plify structures and integrate advanced attention mechanisms, aiming to balance
high performance with better interpretability [152].

@article{lee2022monacobert,
title={Monacobert: Monotonic attention based convbert for knowledge tracing},
author={Lee, U. and Park, Y. and Kim, Y. and Choi, S. and Kim, H.},
journal={arXiv preprint arXiv:2208.12615},
year={2022}
}

---
