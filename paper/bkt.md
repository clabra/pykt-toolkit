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
