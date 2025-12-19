# Bayesian Knowledge Tracing (BKT) Implementation

## Overview

Bayesian Knowledge Tracing (BKT) is a classical probabilistic model used as a dynamic baseline to validate deep knowledge tracing models. In our framework, we use BKT to compute mastery trajectories that serve as external validation metrics for the iDKT model's interpretability claims.

## Theoretical Foundation

### The BKT Model
BKT models student learning as a Hidden Markov Model (HMM) where:
- **Hidden State**: Binary mastery status $L_t \in \{0, 1\}$ for each skill at time $t$.
- **Observable**: Student responses (correct/incorrect) on problems practicing that skill.

### Core Parameters (Per Skill)
BKT learns four parameters for each skill:
1. **P($L_0$) - Prior Knowledge**: Probability student has mastered the skill before any practice.
2. **P($T$) - Learning Rate**: Probability of learning (transitioning from unmastered to mastered) after each practice opportunity.
3. **P($S$) - Slip Probability**: Probability of incorrect response despite mastery.
4. **P($G$) - Guess Probability**: Probability of correct response without mastery.

### Mastery Probability
BKT computes the mastery probability $P(L_t)$ for each interaction using Bayesian inference:

**Step 1: Prediction**
$$ P(correct) = P(L_t) \cdot (1 - P(S)) + (1 - P(L_t)) \cdot P(G) $$

**Step 2: Bayesian Update (after observing response)**
- If response is correct: $P(L_t | correct) = \frac{P(L_t) \cdot (1 - P(S))}{P(correct)}$
- If response is incorrect: $P(L_t | incorrect) = \frac{P(L_t) \cdot P(S)}{1 - P(correct)}$

**Step 3: Learning Transition**
$$ P(L_{t+1}) = P(L_t | response) + (1 - P(L_t | response)) \cdot P(T) $$

## Parameter Estimation

We use the **pyBKT** library, which implements Expectation-Maximization (EM) to learn BKT parameters from dataset sequences.

### Training and Evaluation Script
**Script**: `examples/train_bkt.py`

This script computes the mastery state of all skills for each student interaction:
- **Objective**: Estimate $P(L_t)$ and learned parameters $\{L_0, T, S, G\}$ per skill.
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



