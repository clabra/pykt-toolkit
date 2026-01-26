## Introduction

### Requirements
*The introduction should briefly place the study in a broad context and highlight why it is important. It should define the purpose of the work and its significance. The current state of the research field should be carefully reviewed and key publications cited. Please highlight controversial and diverging hypotheses when necessary. Finally, briefly mention the main aim of the work and highlight the principal conclusions. As far as possible, please keep the introduction comprehensible to scientists outside your particular field of research.
References should be numbered in order of appearance and indicated by a numeral or numerals in square brackets—e.g., [1] or [2,3], or [4–6]. See the end of the document for further details on references.*

### Place the study in a broad context and highlight why it is important.

#### Points to include

- Artificial intelligence (AI) is profoundly transforming the educational landscape
- Deep Learning (DL) is an esential part of modern AI
- DL models are increasingly used to model student behavior and predict learning outcomes
- They are the protagonisst of the state of the art in Knowledge Tracing
- Definition of Knowledge Tracing
- Its role in  educational systems to dynamically adapt to individual needsfacilitating the personalisation of learning experiences.
- Its main advantage of DL: high accuracy
- Its main drawback: lack of interpretability and explainability
- The need to solve this in the field of education

#### Text

**Artificial intelligence (AI) is profoundly transforming the educational landscape [1], with deep learning (DL) architectures serving as an essential pillar of modern AI applications [2]. These models are increasingly utilized to analyze student behavior and predict longitudinal learning outcomes, currently representing the state-of-the-art in Knowledge Tracing [3,4]. Knowledge Tracing is formally defined as the task of modeling a student's evolving mastery of specific concepts based on their history of interactions with learning materials [5], providing the necessary mechanism for educational systems to dynamically adapt to individual needs and facilitate highly personalized learning experiences. While the primary advantage of DL-based approaches lies in their superior predictive accuracy compared to traditional probabilistic models [6], they suffer from a fundamental drawback: a lack of interpretability and explainability, often operating as opaque "black boxes" where internal reasoning remains hidden [7]. Resolving this tension is critical in the educational domain, where stakeholders require transparent models that yield actionable pedagogical insights.**

Key References Used (Mapped to biblio.bib)
[1] Khosravi et al., 2022 (khosravi2022explainable) — AI in Education.
[2] Karpatne et al., 2017 (karpatne2017theory) — DL as essential modern AI.
[3] Abdelrahman et al., 2023 (abdelrahman2023knowledge) — DKT as state-of-the-art.
[4] Liu et al., 2022 (liu2022pykt) — pykt benchmarking.
[5] Corbett & Anderson, 1994 (corbett1994knowledge) — Definition of Knowledge Tracing.
[6] Piech et al., 2015 (piech2015deep) — DL accuracy advantage.
[7] Bai et al., 2024 (bai2024survey) — Lack of interpretability.
[8] Zanellati et al., 2024 (zanellati2024hybrid) — Need for interpretability in real-world pedagogy.

### Define the purpose of the work and its significance

#### Points to include

- The purpose of this work is to contribute to bring interpretability to the field of Deep Learning models for Knowledge Tracing. A kind of Interpretability that can be applied by practitioners in the field of educational technology and allow to leverage the potential of adaptive educational systems with a learner-centred approach that allows to individualized educational diagnosis and adaptation.
- For this we need a new type of Interpretable Deep Learning models adapted to the educational needs, that: 1) provide additional estimations beyond mere predictions about the student responses; 2) That these estimations are based on constructs with a clear semantic educational meaning; 3) That these estimations allow to explain predictions according to sound pedagogical principles or theories.  
- Estos requisitos son cumplidos por los modelos tradicionales a los que los modelos DKT han desbancado en cuanto a accuracy, pero no en cuanto a interpretabilidad. 
- Poner ejempls de como un modelo BKT es interpretable que explican el aprendizaje en terminos de un proceso de Markov donde la evolución del estudiante se analiza mediante un modelo probabilistico donde el cambio de un estado cognitivo a otro se explica en terminos del conocimiento incial del aluno y de la probabilidad de aprendizaje de cada concepto.
- Extender y generalizar a otras teorias como los modelos factoriales basados en regreasion lineal que explican el aprendizaje en terminos de (como la matesria aumenta con las oportunidades de aprendizaje y trata de estimar en que medida se produce ese aumento)

#### Text

**The primary purpose of this work is to bridge the interpretability gap in Deep Learning (DL) models for Knowledge Tracing, providing a framework that is directly applicable by practitioners in educational technology. By making these high-capacity models transparent, we aim to leverage the full potential of adaptive educational systems within a learner-centered approach. This transparency is not merely a technical requirement but a pedagogical necessity, as it enables individualized educational diagnosis and the fine-grained adaptation of learning pathways based on a clear understanding of a student's cognitive state.

To meet these educational needs, we propose a new class of Interpretable Deep Learning models that go beyond binary performance prediction. These models must satisfy three critical requirements: first, they must provide multi-dimensional estimations that characterize the learning process itself; second, these estimations must be anchored to constructs with clear semantic educational meaning; and third, the resulting estimations must allow for explanations grounded in sound pedagogical principles or theories. Such a shift in design ensures that the model's output is not just a probability of success, but an actionable insight that can inform instructional design and personalized support.

These requirements have traditionally been met by classical probabilistic models, such as Bayesian Knowledge Tracing (BKT) [5,131]. BKT can be considered a gold standard for interpretability because it explains learning as a Markov process, where the evolution of a student's knowledge is estimated through parameters—such as initial knowledge and learning transition probabilities—that map directly to pedagogical constructs. While Deep Knowledge Tracing (DKT) models have largely surpassed these traditional approaches in terms of predictive accuracy [6], they have done so at the cost of this intrinsic interpretability, leaving educators without the "why" behind model decisions.

The requirements for interpretability are also met by other established educational theories, such as Factor Analysis models based on linear regression (e.g., AFM and PFM [185]). These models explain mastery as a function of learning opportunities, seeking to estimate the rate at which competence increases over time. We aim to achieve this level of interpretability within a deep learning model, demonstrating that it is possible to maintain a high predictive performance while achieving the kind of interpretability that made traditional models so valuable. This synthesis represents a significant step toward "interpretable-by-design" systems that are both state-of-the-art and pedagogically responsible.**

Key References Used (Mapped to biblio.bib)
[5] Corbett & Anderson, 1994 (corbett1994knowledge) — BKT fundamentals.
[6] Piech et al., 2015 (piech2015deep) — DKT vs. traditional accuracy.
[131] Šarić-Grgić et al., 2022 (twenty_five_years_of_bkt) — Systematic review of BKT interpretability.
[185] — Factor Analysis and procedural knowledge acquisition.


### The current state of the research field 

#### Points to include

- There has been a big evolution of Deep Learning models for Knowledge Tracing, but it has been guided for the most part by accuracy, with interpretability being a secondary concern.
- A first line of evolution has been about how to exploit new information sources to improve accuracy. Esas fuentes de informacion han estado relacionadas mayormente con el modelo de Dominio, el modelo de Estudiante y en algunos casos con el modelo Tutorial, incluyendo algunos principios teoricos tomados de la Pedagogia, la Psicometria o las Ciencias Cognitivas.
- En la mayoria de los modelos que se pueden clasificar en esta linea la informacion adicional ha sido utilizada generalmente para mejorar la precision del modelo, pero no para resolver el problema de la interpretabilidad.
- A second line of evolution has been about exploring new model architectures.
- A concise summary of the evolution of the field of Deep Knowledge Tracing starting with the model proposed by Piech et al. (piech2015deep) to current Transformer-based models.
- De nuevo el objeto de la mayoria de los modelos en esta segunda linea de evolution ha sido la mejora en la precision del modelo a la hora de predecir la respuesta del estudiante a la pregunta. 
- Ha habido algunos intentos de resolver el problema de la interpretabilidad pero con resultados limitados.
- Entre estos intentos se pueden destacar el uso de tecnicas post-hoc genericas para cualquier modelo de Deep Learning. Estas tecnicas tienen una aplicabilidad limitada en el campo de la Educacion ya que requieren un conocimiento profundo para ser aplicadas y comprendidas. 
- Una segunda aproximacion en busqueda de la interpretabilidad han sido los modelo ante-hoc que incorporan componentes a la arquitectura para hacerlos mas transparentes. 
- La review [] menciona los mecanismos de atencion de los Transformers como componentes de este segundo tipo ante-hoc pero puede afirmarse que la atencion en si misma no implica interpretabilidad, al menos no en terminos de conceptos con significado semantico que permitan dar explicaciones basadas en teorias o principios. 
- Tambien se mencionan modelos que incorporan informacion proveniente de modelos intrinsecamente explicables, sobre todo IRT. Pero por lo general esa informacion se ha utilizado para mejorar las predicciones, dentro de una aproximacion IML; sin tampoco dotar al modelo de interpretabilidad en los terminos mencionados. 
- Lo cierto es que actualmente la interpretabilidad sigue siendo uno de los puntos debiles de los modelos DKT frente a los modelos tradicionales. 

#### Text 

**The rapid evolution of Deep Learning (DL) models for Knowledge Tracing has been characterized by a pursuit of predictive accuracy, often at the expense of model transparency [284]. Since the inception of Deep Knowledge Tracing (DKT), the research community has focused on optimizing the ability to forecast student responses, historically leaving interpretability as a secondary or incidental concern. This trajectory has followed two primary lines of development: the exploitation of multi-modal information sources and the exploration of increasingly sophisticated neural architectures.

The first line of evolution involves the integration of auxiliary information related to domain, student, and tutorial models. Researchers have sought to enhance data-driven models by incorporating principles from pedagogy, psychometrics, and cognitive sciences [450]. These features include task difficulty, student learning styles, and concept hierarchies, often represented through relational graphs or expert-defined metadata. While these extensions have significantly improved model precision, their primary goal has reached a bottleneck in terms of transparency.

In most models within this category, additional information is utilized as an "informed" input to boost predictive performance, rather than to resolve the underlying interpretability problem [450]. By treating pedagogical theories merely as data features, the models remain functionally opaque; they may reach the correct prediction more often, but they do not necessarily provide a reason that is semantically aligned with the theories that informed their inputs.

The second line of evolution has focused on architectural innovation, shifting from the initial Recurrent Neural Networks (RNNs) proposed by Piech et al. [268] to the current Transformer-based models [276]. This transition has allowed models to capture long-range temporal dependencies and complex interaction patterns that were previously inaccessible. However, much like the first line, this structural advancement has been evaluated almost exclusively through the lens of prediction task accuracy [284].

Despite the dominance of predictive objectives, several attempts have been made to address the interpretability problem, though with limited success in educational practice. One common approach involves applying general-purpose post-hoc techniques, such as Layer-wise Relevance Propagation (LRP) or SHAP values, to extract explanations from trained models [150]. While technically robust, these methods often fail in real-world educational settings because they require deep technical expertise to apply and interpret, making them inaccessible to the teachers and students they are meant to serve.

A separate ante-hoc approach has sought to build interpretability directly into the model architecture. Extensive literature highlights the attention mechanisms of Transformers as a means of transparency, suggesting that attention maps provide a window into the model's "focus" [150]. However, it is increasingly argued that attention weights in themselves do not constitute true interpretability; they reveal data correlations but lack the semantic meaning necessary to provide explanations based on sound learning theories or pedagogical principles.

Other notable efforts have incorporated information from intrinsically explainable models, particularly Item Response Theory (IRT) [485]. Within the framework of Informed Machine Learning (IML), these models use IRT-derived parameters to calibrate task embeddings and improve accuracy. Yet, these implementations often stop short of providing a fully interpretable latent space, as the deep representations remain mathematically distant from the human-understandable constructs defined by the original psychometric models.

Ultimately, interpretability remains a critical weakness for DKT models when compared to their traditional counterparts [150]. While deep learning provides unparalleled accuracy, the field currently lacks a standard methodology to ensure that these sophisticated architectures reason in a way that is verifiable by educational experts. Bridging this gap requires moving beyond post-hoc visualizations toward "interpretable-by-design" architectures that are anchored to established pedagogical constructs.**

Key References Used (Mapped to biblio.bib)
[150] Bai et al., 2024 (bai2024survey) — Survey of Explainable Knowledge Tracing.
[268] Piech et al., 2015 (piech2015deep) — Deep Knowledge Tracing.
[276] Vaswani et al., 2017 (vaswani2017attention) — Attention Is All You Need.
[284] Abdelrahman et al., 2023 (abdelrahman2023knowledge) — Knowledge Tracing Survey.
[450] Zanellati et al., 2024 (zanellati2024hybrid) — Review of Informed Machine Learning in KT.
[485] Ghosh et al., 2020 (ghosh2020context) — AKT and IRT integration.

### Finally, briefly mention the main aim of the work and highlight the principal conclusions

#### Points to include

- This work is aimed to do so, proposing a novel "interpretable-by-design" DKT model, called iDKT that provide the kind of interpretability required by educational stakeholders. 

- The iDKT model is based on a novel approach that we term Theory-Grounded Embeddings that consist of ...

- We evaluate the proposal ...

- The principal conclusions are ...

#### Text

**This research introduces iDKT (Interpretable Deep Knowledge Tracing), a Transformer-based model specifically engineered to provide the semantic transparency required by educational stakeholders without sacrificing the high predictive accuracy characteristic of deep learning. Unlike previous attempts that rely on post-hoc visualizations and generic explainability techniques, iDKT adopts an "interpretable-by-design" approach. The primary aim of this work is to demonstrate that deep latent representations can be formally anchored to pedagogically valid constructs, thereby transforming the "black box" of student modeling into a transparent tool.

The core of iDKT lies in a novel mechanism we term Representational Grounding. This methodology utilizes a multi-objective loss pipeline to constrain the Transformer's high-dimensional embeddings to align with external theoretical parameters, such as those derived from Bayesian Knowledge Tracing (BKT). By projecting deep latent states into a semantically meaningful space, iDKT ensures that its internal inference mechanism reflects established learning theories. This allows the model to not only predict the probability of a correct response but also to provide a justification for that prediction based on human-understandable educational parameters.

We evaluate the proposed framework across five large-scale benchmark datasets (ASSISTments 2009, ASSISTments 2015, Algebra 2005, Bridge to Algebra 2006, and NIPS 2020) using a rigorous diagnostic probing protocol. Beyond standard predictive metrics (AUC and Accuracy), we introduce a formal validation framework that operationalizes interpretability through measurable **Probing Metrics**. This approach allows us to quantify a conceptual property typically treated in a vague or non-rigorous manner, transforming it into a rigorous and objective dimension for the analysis of deep knowledge tracing models. 

The principal conclusions of this study are that iDKT achieves high-fidelity pedagogical interpretability without compromising state-of-the-art predictive performance. Our results demonstrate that theoretical grounding can be significantly augmented with a negligible loss in accuracy; moreover, for specific benchmarks such as ASSISTments 2009, the model reaches a sweet spot where improving interpretability actually enhances predictive power. By enabling its latent embeddings to capture individualized student-level estimations of parameters typically treated as population-level averages, iDKT achieves a diagnostic granularity inaccessible to traditional models. Ultimately, by operationalizing interpretability through a formal validation framework, this work transforms opaque predictors into rigorous, verifiable tools for personalized education, enabling precise diagnostic placement and dynamic pacing grounded in sound pedagogical reasoning.**

Key References Used (Mapped to biblio.bib)
- iDKT — Described in paper/latex/paper.tex
- Representational Grounding — Defined in paper/latex/paper.tex (Methodology)
- Validation Protocol (Probing) — Detailed in paper/latex/paper.tex (Section 5)

