# Bayesian Knowledge Tracing (BKT) Implementation

## Overview

Bayesian Knowledge Tracing (BKT) is a classical probabilistic model used as a dynamic baseline to validate deep knowledge tracing models. In our framework, we use BKT to compute mastery trajectories that serve as external validation metrics for the iDKT model's interpretability claims.

## pyBKT: An Accessible Library for BKT Modeling

We use the **pyBKT** library, which implements Expectation-Maximization (EM) to learn BKT parameters from dataset sequences @badrinath2021pybkt. 

BKT is a probabilistic model widely used in the field of educational data mining and learning analytics. BKT offers a dynamic process of modeling the probability that a learner possesses knowledge of each skill throughout their learning journey.

As an HMM with observable nodes, the goal of BKT is to estimate the knowledge states of individual learners over time based on their interactions with educational tasks or assessments. In BKT, the learner’s knowledge is defined as a latent variable while the learner’s responses to items (i.e., their performance) are treated as observable variables. The knowledge is represented as a binary variable, indicating whether the learner has learned the skill. The items are also scored dichotomously, with answers categorized as either correct or incorrect.

At its core, BKT leverages Bayesian inference to update and refine its estimates of a learner’s knowledge state at a particular time step t as the learner responds to questions or completes learning activities (obst). The model includes **two knowledge parameters for each learner: the probability of knowing a concept before encountering a task (i.e., prior knowledge; P(L0)), and the probability of learning or acquiring knowledge from the task at time step t (i.e., learned knowledge; P(Lt))**. The **probability of transitioning from the not-known state to the known state after each answer is the *learning* parameter denoted as P(T)**. In addition to knowledge parameters, the model also involves two performance parameters: **the probability of making a mistake when applying a known skill (slip; P(S))** and **the probability of answering an item correctly with a not-known skill (guess; P(G))** @qiu2011does.

**These four parameters in BKT are utilized to update the probability of learning**, representing the likelihood of the learner’s knowledge of a specific skill. More specifically, as the learner responds to the items, BKT updates P(Lt) based on the accuracy of their response (correct or incorrect):

   $$P(Lt|obst = 1) = \frac{P(Lt)(1 − P(S))}{P(Lt)(1 − P(S)) + (1 − P(Lt))P(G)}$$

   $$P(Lt|obst = 0) = \frac{P(Lt)P(S)}{P(Lt)P(S) + (1 − P(Lt))(1 − P(G))}$$

As the learner transitions from one step (t) to the next (t + 1), the updated prior for the following time step can be calculated by @badrinath2021pybkt: 

$$P(Lt+1) = P(Lt|obst) + (1 − P(Lt|obst))P(T)$$

which suggests that the learner is likely to transition from a not-known state to a known state by learning from immediate feedback and any other instructional support; however, the probability of forgetting (P(F)) remains zero @badrinath2021pybkt. 

Many versions have followed the original BKT proposal @corbett1994knowledge including individualized variants discussing individualization per skill and individualization per student of all four BKT parameters. The individualized BKT model resulted in a better correlation between actual and expected accuracy across student results than the non-individualized BKT model @vsaric2024twenty. 

The individualized models could significantly improve the accuracy of predicting the student success. An interesting finding was that adding student-specific probability of learning parameter proved more beneficial for the model accuracy than adding student-specific probability of prior knowledge @yudelson2013individualized.

Student characteristics have been used in many enhanced BKT models, making it the most frequent aspect of BKT enhancements @vsaric2024twenty.

## Training and Evaluation Script
**Script**: `examples/train_bkt.py`

This script computes the mastery state of all skills for each student interaction:
- **Objective**: Estimate $P(L_t)$ and learned parameters $\{L_0, T, S, G\}$ initial knowledge, learning, slip and guess probabilities per skill.
- **Inputs**: Reads from `data/{dataset}/train_valid_sequences.csv`.
- **Outputs**: Saves a trained model as a `.pkl` file and reports RMSE, AUC, and MAE.
- **Key Function**: `prepare_bkt_data(df)` converts sequence-based data into row-per-interaction format for pyBKT.

### The Roster

The Roster is a specialized tracking component in the pyBKT framework (implemented in `pyBKT/models/Roster.py`) that manages the real-time mastery state estimation for a group of students across multiple skills. While the core BKT model handles parameter estimation (fitting), the Roster acts as a "live dashboard" for practitioners.

#### Implementation in `train_bkt.py`
The training script utilizes the Roster to perform a longitudinal inspection of student learning trajectories. It performs the following steps:
1. **Roster Initialization**: A `Roster` object is created for a sample of students and all identified skills, linked to the trained BKT model.
2. **Sequential Replay**: The script iterates through a student's historical interactions in chronological order.
3. **State Updating**: For each interaction, it calls `roster.update_state(skill_name, student_id, correct)`. This triggers a forward inference step in the underlying BKT model, updating the latent mastery probability $P(L_t)$ based on the student's performance (correct vs. incorrect).
4. **Mastery Retrieval**: After each update, the current mastery probability is retrieved using `roster.get_mastery_prob(skill, student_id)`.

#### Output Interpretation
The Roster inspection generates a step-by-step matrix for sampled students:

```text
Student ID: 1234
  Step | Skill | Cor ||  S1      |  S2      |  S3      |
  -------------------------------------------------------
     1 |    S1 |   1 ||  0.4250  |  0.1500  |  0.2000  |
     2 |    S1 |   1 ||  0.7820  |  0.1500  |  0.2000  |
     3 |    S2 |   0 ||  0.7820  |  0.0840  |  0.2000  |
```

- **Skill Columns (e.g., S1, S2)**: Represent the latent mastery probability for that specific skill.
- **Mastery Jump**: Observe how the probability increases significantly after correct responses (showing learning) and stays constant when other skills are practiced (since standard BKT skills are independent).
- **Mastery Threshold**: The Roster uses a default threshold of **0.95** (classifying students as `MASTERED` vs. `UNMASTERED`). This allows practitioners to implement "Mastery Learning" strategies, transitioning students to new topics only once they cross this threshold.

#### iDKT Roster Equivalent
In our theory-guided framework, we maintain parity with this functionality through the **`IDKTRoster`** ([`pykt/models/idkt_roster.py`](file:///home/conchalabra/projects/dl/pykt-toolkit/pykt/models/idkt_roster.py)). This class exposes the same `update_state` and `get_mastery_probs` interface but leverages the iDKT Transformer's attention mechanism to compute the probabilities. This allows for direct, interaction-by-interaction comparison of how a deep learning model perceives mastery evolution versus the classical BKT baseline.

#### How to Inspect the Roster-Equivalent Data

The "Roster Matrix" (longitudinal mastery tracking) can be inspected through two primary channels, each providing different levels of detail:

1. **Console Inspection (Sampling)**:
   When running `examples/train_bkt.py`, the script prints a longitudinal matrix for a random sample of students. This matrix shows the **latent mastery probability** $P(L_t)$ as it evolves after each interaction.

2. **Persistent Experiment Files (Evaluation Subset)**:
   For formal iDKT experiments, interaction-level alignment data is dumped into the experiment results folder for a **subset of students** (default: first 1000). 
   - **File Location**: `experiments/[RUN_DIR]/`
   - **`traj_predictions.csv`**: Correlates iDKT predicted correctness with BKT's ground truth correctness predictions ($p_{idkt}$ vs. $p_{bkt}$).
   - **`traj_initmastery.csv`**: Correlates iDKT's initial mastery projection with the static BKT prior ($idkt\_im$ vs. $bkt\_im$).
   - **`traj_rate.csv`**: Correlates iDKT's dynamic learning rate with BKT's static transition parameter ($idkt\_rate$ vs. $bkt\_rate$).
   - `roster_bkt.csv` / `roster_idkt.csv`: longitudinal "wide" tables for BKT and iDKT respectively, allowing for side-by-side comparison of mastery probabilities across all skills.

> [!IMPORTANT]
> To save disk space, the evaluation scripts limit the interaction-level export to **1000 students**. For a full population analysis, this limit must be adjusted in `examples/eval_idkt_interpretability.py` by modifying `max_export_students`.

---

## Integration with iDKT Model

To enable the multi-objective optimization of iDKT through alignment with a reference BKT model, we must provide BKT-based mastery trajectories and skill parameters during training.

### BKT-Augmented Data Generation
**Script**: `examples/augment_with_bkt.py`

This script bridges the gap between the BKT model and the iDKT training pipeline:
1. **Parameter Extraction**: Loads the trained `.pkl` BKT model and extracts skill-level parameters.
2. **Mastery Replay**: For every interaction in the training set, it re-runs the BKT forward algorithm to compute the predictive mastery $P(L_t)$ and expected correctness $P(corr_t)$.
3. **Sequence Augmentation**: Stores computed trajectories as new columns (`bkt_mastery`, `bkt_p_correct`) in the CSV, preserving the `pykt` sequence format.
4. **Reference Model Output**: Generates `train_valid_sequences_bkt.csv`, providing data for supervised learning and theory-guided alignment outcomes.

### Skill Parameter Registry
The script also saves `bkt_skill_params.pkl` containing static estimates for each skill's BKT parameters, serving as the target reference values ($\mu_{ref,i}$) for the parameter alignment loss term in iDKT.



