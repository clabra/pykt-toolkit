# BKT

## @twenty_five_years_of_bkt

- Two types of evaluation approaches found in the literature, including the prediction of student answers and the ability to estimate knowledge mastery.
- The most frequently investigated enhancements extended the vanilla BKT model by including **student characteristics** and tutor interventions.
- The expectation–maximization algorithm practically became the standard in estimating BKT parameters.
- While the enhanced BKT models generally overperformed the vanilla model in predicting the student answer by using the measures such as RMSE (root mean square error), AUC–ROC (area under curve, receiver operating characteristics curve) and accuracy, only a few studies further investigated the systems’ estimations of knowledge mastery by correlating it to knowledge on post-tests. The most frequently used educational platforms included ITSs, Massive Open Online Courses (MOOCs) and simulated environments.
- Table 1 lists reviews with proposed taxonomies of student modelling approaches (mainly Probabilistic models, Logistic models, Deep learning-based models)
- The main task of the vanilla model is to estimate the probability that a student has mastered the knowledge at time step t, denoted by a learning parameter P(Lt, t ≥ 0). The model updates the probability P(Lt) after each opportunity to apply knowledge given an observed correct or incorrect response

```Vanilla model parameters are P(L0), P(T), P(G), and P(S).
P(T) is the probability of a knowledge transitioning from unlearned state to learned state
P(L 0 ) is the initial
probability of knowledge before any opportunity of applying it (prior knowledge)
P(G) guess prpbabilty
P(S) slip probability
```

- The original BKT publication was (Corbett & Anderson 1995)
- Regarding the BKT parameter estimation procedure, Corbett and Anderson (1995) discussed individualization per skill and individualization per student of all four BKT parameters. The individualized BKT model resulted in a better correlation between actual and expected accuracy across student results than the non-individualized BKT model whose accuracy of predicting student test scores (after a period of working with a tutoring system) did not improve tangibly (Yudelson et al. 2013). Finally, the parameter fitting procedure of the vanilla model relates to expert-based estimations of the four BKT parameters per skill.
- Table 8 list enhanced BKT models per enhancement aspect

## @intro_pyBKT

- In the field of psychometrics, researchers have extensively studied various test theories to model students’ knowledge states within a test session. Test theories differ from KT in that they are primarily designed for tests where students’ knowledge states are assumed to be static (i.e., students’ knowledge remains unchanged as they respond to a set of questions) [ 30]. As a modern test theory, IRT stands out as another notable approach for modeling students’ (current) knowledge states.

- IRT is a modeling framework used for constructing and analyzing learning and assessment data [31].

```
The probability of person j answering item i correctly, P(Yij), can be expressed as a function of:

 ai, bi, and ci refer to the discrimination, difficulty, and guessing parameters of item i, respectively

 θj represents person j’s latent trait (or ability) level on the construct being measured by the item.
```

- IRT is more tailored for cross-sectional data, aiming to analyze item responses to assess learners’ latent traits, such as knowledge or ability, while BKT, specifically designed for longitudinal data, focuses on modeling and tracking learners’ knowledge states over time [29, 32].
- Deonovic et al. [29] discussed the connection between BKT models and IRT models and highlighted how the limitations of each model can be mitigated by integrating concepts from the other.
- Wang et al. [33] proposed a novel approach that combines IRT and BKT... in their model, first, IRT estimates the learner’s ability for skill. Next, this information is combined with the estimated difficulty and discrimination level of each skill to calculate the probability of a learner already knowing a skill before practicing it.
- In their study, Pardos and Heffernan [26] explored how the addition of question-level difficulty can be incorporated into the guess and slip parameters. The intuitive notion is that skills associated with a high guess rate (guess parameter) can be considered easier.
- pyBKT allows to estimate the four parameters (plus forgetting) for each skills and also the AUC/accuracy of the model.
- Case Study 2: Comparing IRT and BKT in Modeling Response Accuracy
- The results shown in Table 2 indicate the **Pearson correlation calculated between the IRT difficulty parameters and the reconstructed difficulty parameters from BKT log(πφkp)**, which represents the forget parameter φ for the specific problem p in the knowledge component k. The results suggest that most of the difficulty parameters from IRT (bkp) and the converted difficulty parameters from BKT (log(πφkp)) generally were **highly correlated, indicating the close alignment between the IRT and BKT models**.
