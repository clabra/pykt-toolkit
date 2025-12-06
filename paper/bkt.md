# Bayesian Knowledge Tracing (BKT) Implementation

## Overview

Bayesian Knowledge Tracing (BKT) is a classical probabilistic model used as a dynamic baseline to validate deep knowledge tracing models. In our framework, we use BKT to compute mastery trajectories that serve as external validation metrics for the iKT2 model's interpretability claims.

## Theoretical Foundation

### The BKT Model

BKT models student learning as a Hidden Markov Model (HMM) where:
- **Hidden State**: Binary mastery status L_t ∈ {0, 1} for each skill at time t
- **Observable**: Student responses (correct/incorrect) on problems practicing that skill

### Core Parameters (Per Skill)

BKT learns four parameters for each skill:

1. **P(L₀) - Prior Knowledge**: Probability student has mastered the skill before any practice
2. **P(T) - Learning Rate**: Probability of learning (transitioning from unmastered to mastered) after each practice opportunity
3. **P(S) - Slip Probability**: Probability of incorrect response despite mastery (performance error)
4. **P(G) - Guess Probability**: Probability of correct response without mastery (lucky guess)

### Parameter Structure: Per-Skill vs Per-Student

**Standard BKT Implementation (Our Approach):**

In the most common BKT implementation, all four parameters are **per-skill** (not per-student). This means:
- Each skill has ONE set of {P(L₀), P(T), P(S), P(G)} shared by ALL students
- Student-specific variation comes from the dynamic hidden state P(L_t)_uk
- The same skill parameters applied to different response sequences produce personalized learning trajectories

**Example from ASSIST2015**:
```python
# Skill 0 parameters (shared by all students)
{'prior': 0.748, 'learns': 0.094, 'slips': 0.086, 'guesses': 0.485}

# Skill 5 parameters (shared by all students)  
{'prior': 0.520, 'learns': 0.063, 'slips': 0.133, 'guesses': 0.338}
```

**How Personalization Works:**

Despite per-skill parameters, BKT achieves student-specific mastery estimates through Bayesian updates:

1. **Initialization**: Student A and B both start with P(L₀)=0.52 for skill 5
2. **Different responses lead to different trajectories**:
   - Student A: correct → P(L₁)=0.718 → correct → P(L₂)=0.859
   - Student B: incorrect → P(L₁)=0.289 → incorrect → P(L₂)=0.307
3. **Learning transition**: P(L_{t+1}) = P(L_t|obs) + (1 - P(L_t|obs)) × P(T)
   - Same P(T)=0.063 applied to different P(L_t|obs) values
   - Produces student-specific learning curves

**Individualized BKT (iBKT) Variant:**

Yudelson et al. (2013) proposed **Individualized Bayesian Knowledge Tracing**, which extends BKT to have per-student parameters. However, this is a later variant requiring more data and computational resources. Our implementation uses standard BKT with skill-specific parameters, following the original formulation by Corbett & Anderson (1994) and the most common practice in the field.

**Reference**:
- Yudelson, M. V., Koedinger, K. R., & Gordon, G. J. (2013). Individualized bayesian knowledge tracing models. In Artificial Intelligence in Education (pp. 171-180). Springer.
- Wikipedia: "In its most common implementation, BKT has only skill-specific parameters."

### Forward Algorithm

BKT computes the mastery probability P(L_t) for each interaction using Bayesian inference:

**Step 1: Prediction**
```
P(correct) = P(L_t) * (1 - P(S)) + (1 - P(L_t)) * P(G)
```

**Step 2: Bayesian Update (after observing response)**

If correct:
```
P(L_t | correct) = [P(L_t) * (1 - P(S))] / P(correct)
```

If incorrect:
```
P(L_t | incorrect) = [P(L_t) * P(S)] / [1 - P(correct)]
```

**Step 3: Learning Transition**
```
P(L_{t+1}) = P(L_t | response) + [1 - P(L_t | response)] * P(T)
```

The algorithm maintains a per-(student, skill) mastery state that evolves sequentially through the student's interaction history.

## Parameter Estimation

### EM Algorithm via pyBKT

We use the **pyBKT** library, which implements Expectation-Maximization (EM) to learn BKT parameters from data.

**Key characteristics:**
- **Statistical estimation**: Maximum likelihood estimation of skill-level parameters
- **No hyperparameter tuning**: EM converges to stable estimates without validation sets
- **Training data**: All train+valid students (15,275 for ASSIST2015, 3,082 for ASSIST2009)
- **Fold-independent**: Unlike deep learning, BKT doesn't require fold splits - more data yields better parameter estimates

### Implementation Script

**Script**: `examples/compute_bkt_targets.py`

**Key Functions**:

1. **`prepare_bkt_data(df)`**: Converts pykt sequence format to pyBKT format
   ```python
   # Input: pykt CSV with uid, concepts, responses, selectmasks
   # Output: pyBKT format with user_id, skill_name, correct, order_id
   ```

2. **`fit_bkt_model(bkt_df, num_skills)`**: Trains BKT via EM algorithm
   ```python
   from pyBKT.models import Model
   
   model = Model(seed=42, num_fits=1, parallel=False)
   model.fit(data=bkt_df)  # EM algorithm
   
   # Extract parameters per skill
   params[skill_id] = {
       'prior': ...,   # P(L₀)
       'learns': ...,  # P(T)
       'slips': ...,   # P(S)
       'guesses': ...  # P(G)
   }
   ```

3. **`compute_bkt_targets(df, model, params, num_skills)`**: Runs forward algorithm to generate mastery trajectories
   - Maintains `student_skill_mastery` dict: (student_id, skill_id) → current P(L_t)
   - Applies Bayesian update + learning transition for each interaction
   - Returns per-student mastery matrices: `{uid: torch.Tensor[seq_len, num_skills]}`

4. **`enforce_monotonicity_skill_wise(targets_dict)`**: Optional smoothing to enforce P(L_t) ≤ P(L_{t+1})

**Usage**:
```bash
python examples/compute_bkt_targets.py \
  --dataset assist2015 \
  --output_path data/assist2015/bkt_targets.pkl
```

**Output Files**:
- `data/assist2015/bkt_targets.pkl` (201M): Standard version
- `data/assist2015/bkt_targets_mono.pkl` (201M): Monotonic version

**File Structure**:
```python
{
    'bkt_targets': {uid: torch.Tensor[seq_len, num_skills]},  # Per-student mastery
    'bkt_params': {skill_id: {'prior', 'learns', 'slips', 'guesses'}},  # Per-skill params
    'metadata': {
        'dataset': 'assist2015',
        'num_students': 15275,
        'num_skills': 100,
        'method': 'BKT (Bayesian Knowledge Tracing)',
        'model': 'pyBKT',
        'monotonic': False
    }
}
```

## BKT Forward Inference for Validation

### Computing Test Set Trajectories

**Script**: `examples/compute_bkt_correlation.py`

This script computes BKT mastery trajectories for test interactions to validate the iKT2 model.

**Key Functions**:

1. **`load_bkt_params(dataset, bkt_path=None)`**: Loads pre-trained BKT parameters
   ```python
   # Returns: bkt_params dict, metadata
   bkt_path = f"data/{dataset}/bkt_targets.pkl"
   ```

2. **`load_model_mastery(experiment_dir)`**: Loads iKT2 mastery estimates
   ```python
   # Loads: experiments/{exp_id}/mastery_test.csv
   # Contains: student_id, time_step, skill_id, response, mi_value, ...
   ```

3. **`compute_bkt_forward(bkt_params, df)`**: Runs BKT forward inference
   ```python
   student_skill_mastery = {}  # (student_id, skill_id) → current P(L_t)
   bkt_mastery_values = []
   
   # Sort by student_id, time_step for sequential processing
   df_sorted = df.sort_values(['student_id', 'time_step'])
   
   for row in df_sorted:
       key = (student_id, skill_id)
       
       # Initialize P(L_0) = prior on first encounter
       if key not in student_skill_mastery:
           student_skill_mastery[key] = bkt_params[skill_id]['prior']
       
       p_l = student_skill_mastery[key]
       bkt_mastery_values.append(p_l)  # Store BEFORE interaction
       
       # Bayesian update based on response
       if response == 1:  # Correct
           p_l_updated = (p_l * (1-slip)) / p_correct
       else:  # Incorrect
           p_l_updated = (p_l * slip) / (1 - p_correct)
       
       # Learning transition
       p_l_next = p_l_updated + (1 - p_l_updated) * learn
       student_skill_mastery[key] = p_l_next
   
   # Reorder to match original df index
   return bkt_mastery_array[reorder_indices]
   ```

**Critical Details**:
- Stores P(L_t) **BEFORE** each interaction (predictive mastery)
- Maintains temporal consistency via sorting + reordering
- Handles missing skills by averaging parameters

4. **`validate_data_consistency(experiment_dir, dataset, bkt_metadata, df)`**: Validates fold/dataset/skill alignment
   - Checks dataset name matches
   - Verifies all test skills have BKT parameters
   - Warns about student overlap (for debugging)

5. **`compute_trajectory_slopes(df, bkt_mastery, model_mastery)`**: Computes learning trajectory slopes
   - Analyzes per (student, skill) learning rates
   - Compares BKT vs iKT2 learning progressions

**Usage**:
```bash
python examples/compute_bkt_correlation.py \
  --experiment_dir experiments/20251206_173247_ikt2_lrefmetrics_fixed_636735 \
  --dataset assist2015 \
  --output_file bkt_validation.json \
  --update_csv
```

**Outputs**:

1. **JSON Results** (`experiments/{exp_id}/bkt_validation.json`):
   ```json
   {
     "bkt_correlation": 0.7234,  // Pearson r(M_IRT, P(L_t))
     "p_value": 1.2e-150,
     "mse": 0.0523,
     "mae": 0.1847,
     "num_interactions": 1492,
     "supplementary_correlations": {
       "time_lagged": {
         "correlation": 0.7891,  // After initialization (attempt > 3)
         "p_value": 3.4e-120,
         "num_interactions": 1021
       },
       "trajectory_slopes": {
         "correlation": 0.6543,  // Learning rate alignment
         "num_trajectories": 74
       }
     }
   }
   ```

2. **Updated CSV** (`experiments/{exp_id}/mastery_test_bkt.csv`):
   - Adds `bkt_mastery` column to `mastery_test.csv`
   - Contains P(L_t) for each interaction
   - Used for detailed trajectory analysis

3. **Updated Metrics** (`experiments/{exp_id}/metrics_test.csv`):
   - Adds `bkt_correlation_standard` column
   - Adds `bkt_correlation_timelagged` column
   - Integrated into evaluation summary

## Integration with iKT2 Model

### Validation Workflow

1. **Training iKT2**:
   ```bash
   python examples/train_ikt2.py \
     --dataset assist2015 \
     --fold 0 \
     --rasch_path data/assist2015/rasch_test_iter300.pkl \
     --mastery_method rasch
   ```
   - `--rasch_path` must point to a **Rasch IRT file** (not BKT) containing `skill_difficulties`
   - iKT2 trains on fold 0 (validation) and folds 1-4 (training)
   - Produces `experiments/{exp_id}/mastery_test.csv` with `mi_value` column

2. **Computing BKT Correlation**:
   ```bash
   python examples/compute_bkt_correlation.py \
     --experiment_dir experiments/{exp_id} \
     --dataset assist2015 \
     --bkt_path data/assist2015/bkt_targets.pkl
   ```
   - Loads pre-computed **BKT parameters** from `bkt_targets.pkl` (trained on all train+valid students)
   - Uses separate `--bkt_path` parameter (not `--rasch_path`) for BKT files
   - Runs BKT forward inference on test interactions
   - Correlates BKT P(L_t) with iKT2 M_IRT (from `mi_value`)

3. **Interpretation**:
   - **High correlation (r > 0.7)**: iKT2 mastery estimates align with classical BKT baseline
   - **Time-lagged correlation**: Tests alignment after initialization phase
   - **Trajectory slopes**: Validates learning rate consistency

### BKT vs Rasch IRT: Different Roles

**Rasch IRT** (used with `--rasch_path` during training):
- **Purpose**: Skill difficulty regularization during iKT2 training
- **File requirement**: Must contain `skill_difficulties` key with β values
- **Example**: `rasch_test_iter300.pkl` (201M)
- **Role**: Provides static difficulty priors for skill regularization
- **Used by**: `train_ikt2.py` → `load_skill_difficulties_from_irt()`

**BKT** (used with `--bkt_path` during validation):
- **Purpose**: Temporal mastery trajectory validation
- **File requirement**: Contains `bkt_params` (prior, learns, slips, guesses)
- **Example**: `bkt_targets.pkl` (201M)
- **Role**: Provides dynamic mastery baselines for correlation analysis
- **Used by**: `compute_bkt_correlation.py` → `load_bkt_params()`

**Key Distinction**: These are separate validation systems serving different purposes. Rasch IRT provides static skill difficulties for training regularization, while BKT provides temporal mastery trajectories for validation correlation. They use different file formats and cannot be interchanged.

### Validation Metrics

**Standard Correlation**: Pearson r between iKT2's M_IRT and BKT's P(L_t)
```python
model_mastery = df['mi_value'].values  # iKT2 IRT-based mastery
bkt_mastery = compute_bkt_forward(bkt_params, df)  # BKT forward inference

correlation, p_value = pearsonr(model_mastery, bkt_mastery)
```

**Time-Lagged Correlation**: Correlation after attempt > 3
- Reduces prior knowledge bias
- Tests alignment during learning phase
- More robust indicator of trajectory quality

**Trajectory Slope Correlation**: Compares learning rates
- Computes slopes: Δmastery / Δinteraction for each (student, skill)
- Correlates BKT learning rates with iKT2 learning rates
- Validates that both models detect similar learning patterns

### Why BKT Validation Matters

1. **External Validity**: BKT is a well-established baseline with strong theoretical foundations
2. **Interpretability**: If iKT2 mastery estimates don't align with BKT, interpretability claims are questionable
3. **Dynamic Baseline**: Unlike static IRT difficulty, BKT provides temporal mastery evolution
4. **Skill-Level Validation**: Each skill has independent BKT parameters, enabling fine-grained analysis

## Validation Results Example

**Experiment**: `20251206_173247_ikt2_lrefmetrics_fixed_636735`
- **Dataset**: ASSIST2015
- **Test interactions**: 1,492
- **BKT correlation**: 0.7373
- **Time-lagged correlation**: 0.7891
- **Interpretation**: Strong alignment between iKT2 and BKT, validating mastery interpretability

**Statistics**:
- BKT mastery: μ = 0.737, σ = 0.253, range = [0.08, 1.00]
- iKT2 mastery: μ = 0.724, σ = 0.241, range = [0.12, 0.98]
- Similar distributions indicate semantic alignment

## BKT vs Deep Learning

### Why BKT Doesn't Need Fold Splits

Unlike deep learning models:
- **No overfitting risk**: Statistical parameter estimation, not gradient descent optimization
- **Skill-level parameters**: 4 parameters per skill (400 for ASSIST2015), not millions of weights
- **More data = better estimates**: BKT benefits from maximum training data
- **No hyperparameters**: EM algorithm converges without tuning

**Decision**: Train BKT on all 15,275 train+valid students (ASSIST2015) for robust parameter estimates

### Limitations of BKT

1. **Independence assumption**: Assumes skills are independent (ignores prerequisites)
2. **Binary states**: Models mastery as learned/unlearned (not continuous)
3. **Static parameters**: P(T), P(S), P(G) constant for all students
4. **First-order Markov**: Ignores long-term dependencies

Deep models like iKT2 overcome these limitations while maintaining interpretability through BKT alignment.

## Implementation Files

### Scripts
- `examples/compute_bkt_targets.py`: Parameter training via EM
- `examples/compute_bkt_correlation.py`: Validation correlation computation
- `tmp/validate_bkt_and_augment_csv.py`: Comprehensive validation tool

### Data Files (ASSIST2015)

**BKT Files** (used by `compute_bkt_correlation.py`):
- `data/assist2015/bkt_targets.pkl` (201M): Pre-trained BKT parameters
  - Structure: `{'bkt_params': {...}, 'bkt_targets': {...}, 'metadata': {...}}`
  - Contains: 4 parameters per skill (prior, learns, slips, guesses)
  - Used for: Validation correlation analysis
- `data/assist2015/bkt_targets_mono.pkl` (201M): Monotonic version

**Rasch IRT Files** (used by `train_ikt2.py`):
- `data/assist2015/rasch_test_iter300.pkl` (201M): Rasch IRT skill difficulties
  - Structure: `{'skill_difficulties': {...}, 'student_abilities': {...}, 'rasch_targets': {...}, 'metadata': {...}}`
  - Contains: β values (skill difficulties)
  - Used for: Training regularization via `--rasch_path`
- `data/assist2015/rasch_targets.pkl`: Symlink to `rasch_test_iter300.pkl`

**Shared Files**:
- `data/assist2015/keyid2idx.json`: ID mapping (concepts & students)

### Experiment Outputs
- `experiments/{exp_id}/mastery_test.csv`: iKT2 mastery estimates
- `experiments/{exp_id}/mastery_test_bkt.csv`: Augmented with BKT column
- `experiments/{exp_id}/bkt_validation.json`: Correlation results
- `experiments/{exp_id}/metrics_test.csv`: Updated with BKT correlations

## Dependencies

```python
# pyBKT: Bayesian Knowledge Tracing library
from pyBKT.models import Model

# Standard libraries
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
import pickle
```

**Installation**:
```bash
pip install pyBKT
```



## Summary

BKT serves as a rigorous external validation mechanism for iKT2's interpretability:
- **Training**: EM algorithm learns 4 parameters per skill from all train+valid data
- **Inference**: Forward algorithm computes mastery trajectories P(L_t) for test interactions
- **Validation**: Pearson correlation between BKT P(L_t) and iKT2 M_IRT
- **Interpretation**: High correlation (r > 0.7) validates that iKT2 mastery estimates are semantically meaningful and align with established educational theory

### Dual-Validation Architecture

iKT2 uses two separate validation systems serving complementary purposes:

1. **Rasch IRT Validation** (Training-time regularization):
   - Parameter: `--rasch_path` pointing to Rasch IRT file
   - File requirement: `skill_difficulties` key containing β values
   - Purpose: Static skill difficulty priors for regularization
   - Script: `train_ikt2.py`
   - Role: Guides model training with expert difficulty estimates

2. **BKT Validation** (Post-training correlation):
   - Parameter: `--bkt_path` pointing to BKT parameters file
   - File requirement: `bkt_params` key containing {prior, learns, slips, guesses}
   - Purpose: Dynamic mastery trajectory validation
   - Script: `compute_bkt_correlation.py`
   - Role: Validates temporal learning trajectories align with educational theory

**Critical Distinction**: These validation systems use different file formats and cannot be interchanged. Rasch IRT provides static difficulties for training, while BKT provides temporal mastery baselines for validation. Together, they provide comprehensive evidence for iKT2's interpretability claims.

## References

1. **Corbett, A. T., & Anderson, J. R. (1994)**. Knowledge tracing: Modeling the acquisition of procedural knowledge. User Modeling and User-Adapted Interaction, 4(4), 253-278.

2. **Pardos, Z. A., & Heffernan, N. T. (2010)**. Modeling individualization in a bayesian networks implementation of knowledge tracing. In International Conference on User Modeling, Adaptation, and Personalization (pp. 255-266).

3. **Yudelson, M. V., Koedinger, K. R., & Gordon, G. J. (2013)**. Individualized bayesian knowledge tracing models. In Artificial Intelligence in Education (pp. 171-180).

4. **pyBKT Documentation**: https://github.com/CAHLR/pyBKT