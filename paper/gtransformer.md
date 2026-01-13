# GTransformer (Grounded Transformer)

The GTransformer model is a "Grounded" version of the Context-Aware Attentive Knowledge Tracing (AKT) architecture. It incorporates **Representational Grounding** by leveraging Bayesian Knowledge Tracing (BKT) as a theoretical reference model. This allows the model to learn student-specific parameters (initial mastery and learning rate) that are anchored to established pedagogical theory while maintaining the predictive power of Transformers.

---

This dsection summarizes the results and methodology used to verify the parity between the newly implemented `gTransformer` model (a direct architectural copy of the Context-Aware Attentive Knowledge Tracing or AKT) and the original `AKT` implementation within the `pykt-toolkit` framework.

### 1. Objective
The primary goal was to ensure that the `gTransformer` implementation is functionally identical to the reference `AKT` model. Verification was conducted by comparing training trajectories, validation metrics, and final test performance across different execution methodologies (Direct Scripts vs. Reproduction Launcher).

### 2. Experimental Setup & Parameters

Experiments were conducted on the **ASSIST2009** dataset (Fold 0) using the following hyperparameters:

| Parameter | Value | Source |
| :--- | :--- | :--- |
| **Dataset** | `assist2009` | Override (`assist2009_S`) |
| **Epochs** | `2` | Override (`200`) for parity fast-track |
| **Batch Size** | `128` | Standard Launcher Default |
| **Seed** | `42` | Reproducibility Standard |
| **Fold** | `0` | Reference Fold |
| **Fusion Type** | `late_fusion` | Question-level aggregation |
| **Learning Rate** | `1e-4` | System Default |
| **Optimizer** | `Adam` | System Default |
| **Dropout** | `0.3` | System Default |
| **d_model** | `256` | AKT Architecture |
| **d_ff** | `512` | AKT Architecture |
| **n_blocks** | `4` | AKT Architecture |
| **num_attn_heads** | `8` | AKT Architecture |

## 3. Reproduction Launcher Methodology

The `run_repro_experiment.py` script was used to launch both models. This approach ensures that all parameters are explicitly recorded in a `config.json` file and that the project's reproducibility protocol (Audit) is satisfied.

### AKT Run Command
```bash
python examples/run_repro_experiment.py \
    --model akt \
    --dataset assist2009 \
    --seed 42 \
    --fold 0 \
    --epochs 2 \
    --use_wandb 0 \
    --short_title akt_parity_v4 \
    --batch_size 128 \
    --fusion_type late_fusion
```

### gTransformer Run Command
```bash
python examples/run_repro_experiment.py \
    --model gtransformer \
    --dataset assist2009 \
    --seed 42 \
    --fold 0 \
    --epochs 2 \
    --use_wandb 0 \
    --short_title gtrans_parity_v4 \
    --batch_size 128 \
    --fusion_type late_fusion
```

### 4. Direct Script Methodology (Standard PyKT)

To further validate the framework's internal consistency, a "Direct" run was performed bypassing the launcher, using the original standard scripts.

#### Manual Training Command
```bash
# Executed from within the 'examples/' directory
python wandb_akt_train.py \
    --d_ff 512 \
    --d_model 256 \
    --dataset_name assist2009 \
    --dropout 0.3 \
    --final_fc_dim 512 \
    --fold 0 \
    --l2 1e-05 \
    --learning_rate 0.0001 \
    --n_blocks 4 \
    --num_attn_heads 8 \
    --num_epochs 2 \
    --seed 42 \
    --use_wandb 0 \
    --add_uuid 0 \
    --save_dir ../experiments/manual_akt_standard
```

#### Manual Prediction Command (Predict-only)
The standard PyKT prediction script (`wandb_predict.py`) was used to obtain the same metrics as the launcher. Note that `wandb_eval.py` follows a different "Multi-step" methodology and was excluded from this exact parity table.

```bash
python wandb_predict.py \
    --save_dir ../experiments/manual_akt_standard/[NESTED_FOLDER] \
    --bz 128 \
    --use_wandb 0 \
    --fusion_type late_fusion
```

### 5. Performance Parity Results

The following table summarizes the metrics obtained across all three runs. The metrics for `AKT` and `gTransformer` are **mathematically identical** up to the 6th decimal place.

| Metric | AKT (Launcher) | gTransformer (Launcher) | AKT (Direct Script) |
| :--- | :--- | :--- | :--- |
| **Valid AUC (Epoch 2)** | 0.791929 | 0.791929 | 0.791929 |
| **Valid ACC (Epoch 2)** | 0.756479 | 0.756479 | 0.756479 |
| **Test AUC (KC-Level)** | 0.783603 | 0.783603 | 0.783603 |
| **Test ACC (KC-Level)** | 0.744443 | 0.744443 | 0.744443 |
| **Question AUC (Late Mean)** | 0.705707 | 0.705707 | 0.705707 |
| **Question ACC (Late Mean)** | 0.696272 | 0.696272 | 0.696272 |

### 6. Comparison: Direct Execution vs. Launcher

While the final metrics are identical, the two methodologies serve different roles in the development lifecycle:

| Feature | Direct Scripts (`wandb_*.py`) | Reproduction Launcher (`run_repro_experiment.py`) |
| :--- | :--- | :--- |
| **Primary Goal** | Ad-hoc testing and framework baseline. | Scientific reproducibility and audit compliance. |
| **Parameter Source**| Explicit CLI flags or hardcoded defaults in code. | `configs/parameter_default.json` (Single source of truth). |
| **Audit Trail** | None (unless manually logged). | Mandatory `config.json` with MD5 and parameter audit. |
| **Output Structure** | Flat (or standard `saved_model` dir). | Structured timestamped folders with explicit metadata. |
| **Error Handling** | Standard Python tracebacks. | Pre-flight checks for parameter inconsistencies. |

### 7. Grounded Implementation (Representational Grounding)

The `GTransformer` implementation extends the base `AKT` architecture with a Grounding Layer that aligns its latent projections with pedagogical priors.

#### Theory-Guided Components:
1. **Student Capability ($v_s$)**: A learned student embedding that scales the Rasch-based difficulty.
2. **Knowledge Gap ($k_s$)**: A student-specific parameter representing the variance in prerequisite proficiency.
3. **Induced Mastery Trajectory**: The model generates explicit predictions for initial mastery ($L_0$) and learning rate ($T$) that are regularized against BKT-derived priors.

### 8. BKT Preprocessing & Augmentation Pipeline

To support theory-guided training, the standard PyKT pipeline is augmented with a BKT reference stage.

#### Step 1: BKT Parameter Calculation
Calculate population-level BKT parameters (Prior, Learn, Guess, Slip) for each skill in the dataset.
```bash
python examples/train_bkt.py --dataset assist2009
```
*Output: `data/assist2009/bkt_skill_params.pkl`*

#### Step 2: Dataset Augmentation
Align BKT predictions with original interactions using a robust left-merge to handle missing/skipped skills.
```bash
python examples/augment_with_bkt.py --dataset assist2009
```
*Output: `data/assist2009/skill_builder_data_corrected_collapsed_bkt_augmented.csv`*

#### Step 3: BKT-Augmented Preprocessing
Generate 8-line format sequences (adding `bkt_p_correct` and `bkt_mastery` rows) and perform K-Fold splitting.
```bash
python examples/data_preprocess_bkt_augmented.py --dataset assist2009
```
*Output: `data/assist2009/bkt_augmented/train_valid_sequences_bkt.csv`*

### 9. GTransformer Training & Loss Calibration

The `gtransformer` model supports a specialized training loop (`train_gtransformer.py`) that handles the theory-guided loss and provides automatic loss calibration.

#### Execution Command
```bash
python examples/run_repro_experiment.py \
    --model gtransformer \
    --dataset assist2009_bkt \
    --short_title grounding_verify \
    --epochs 100 \
    --theory_guided 1 \
    --calibrate 1
```

#### Loss Calibration Logic
Because theory-guided signals (alignment with BKT) and supervised signals (performance on next-item prediction) can have different magnitudes, `GTransformer` performs a **warm-up calibration pass**. It calculates the initial MSE of theory components and normalizes their contribution to match a target ratio of the supervised loss (default 10%).

### 10. Question-Level Evaluation Alignment

To ensure a fair comparison with BKT and other baselines, we use a specialized Question-Level evaluation script that implements Late Fusion (Mean) for multi-concept questions.

```bash
# Evaluate GTransformer
python examples/eval_gtransformer.py --dataset assist2009_bkt --fold 0

# Evaluate BKT Baseline (Fair Comparison)
python examples/eval_bkt_question_level.py --dataset assist2009_bkt --fold 0
```

### 11. Comprehensive Analysis Pipeline

Once training is complete, the `generate_gtransformer_full_analysis.py` pipeline (automatically triggered by the launcher) regenerates all interpretability artifacts:
- **Interpretability Alignment**: Trajectory and Roster comparisons (`traj_predictions.csv`, `roster_gtransformer.csv`).
- **Probing Validation**: Diagnostic validation metrics and Pareto plots.
- **Advanced Visualizations**: Student clusters, Skill Mastery Maps, and Curriculum Heatmaps.

