
## Abstract

Deep knowledge tracing models obtaing excelent resuls in terms of predicting student performance but they lack the interpretability that is inherent to other appoaches like bayesian knowledge tracing. We propose iDKT, a novel model that combines the best of both worlds. iDKT is based in a Transformer architecture with interpreatbility-by-design. Rather than treating interpretability as a post-hoc diagnostic task—an attempt to explain what a black-box model has already learned—we propose **Structural Grounding** as a core architectural principle.

We propose a definition of interpretability in relation to a reference model that can be considered as intrinsecally interpretable. Through the architectura design we guide the training process to learn internal representations that are formally anchored to the conceptual space of the reference theory. This provides a interpretable framework that remains valid for diverse reference models and provides a theoretically-sound bridge between data-driven insights and pedagogical interpretations. 

We achieve **Intrinsic Interpretability** through a **Relational Differential Fusion** method where the transformer internal representations are anchored to the conceptual space of the reference theory. 

To move beyond heuristic interpretation, we establish a validation framework based on these formal hypotheses:  
| ID | Hypothesis | Validation Method |
| :--- | :--- | :--- |
| **$H_1$** | **Convergent Validity** | Pearson correlation between latent projections ($l_c, t_s$) and BKT intrinsic parameters. |
| **$H_2$** | **Predictor Equivalence** | Functional substitutability of iDKT parameters into canonical BKT mastery recurrence equations. |
| **$H_3$** | **Discriminant Validity** | Non-collinearity check between Learner Gap ($k_c$) and Learning Speed ($v_s$) to ensure identifiability. |

This framework proves that iDKT’s internal representations ($l_c$, $t_s$, $u_q$) represent the intended educational constructs. 



