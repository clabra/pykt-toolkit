# GTransformer (Grounded Transformer)

The GTransformer model is a "Grounded" version of the Context-Aware Attentive Knowledge Tracing (AKT) architecture. It grounds output estimations ot parameter values given by an intrinsic interprtable reference model like Bayesian Knowledge Tracing (BKT). This allows the model to learn student-specific parameters (initial mastery and learning rate) that are anchored to established pedagogical theory while maintaining the predictive power of Transformers.

---

This section summarizes the results and methodology used to verify the parity between the newly implemented `gTransformer` model (a direct architectural copy of the Context-Aware Attentive Knowledge Tracing or AKT) and the original `AKT` implementation within the `pykt-toolkit` framework.

### 1. Objective
The primary goal was to ensure that the `gTransformer` implementation is functionally identical to the reference `AKT` model. Verification was conducted by comparing training trajectories, validation metrics, and final test performance across a full **5-Fold Cross-Validation** campaign on the `assist2009` dataset.

### 2. Experimental Setup & Parameters

Experiments were conducted on the **ASSIST2009** dataset (Folds 0-4) using the standard benchmark hyperparameters:

| Parameter | Value | Source |
| :--- | :--- | :--- |
| **Dataset** | `assist2009` | Benchmark Standard |
| **Epochs** | `200` | Full converged training |
| **Batch Size** | `64` | Benchmark Standard |
| **Seed** | `3407` | Benchmark Standard |
| **Folds** | `0-4` | Full Cross-Validation |
| **Fusion Type** | `late_fusion` | Question-level aggregation |
| **Learning Rate** | `1e-4` | AKT Optimal |
| **Optimizer** | `Adam` | System Default |
| **Dropout** | `0.1` | AKT Optimal |
| **d_model** | `64` | AKT Optimal |
| **d_ff** | `256` | AKT Optimal |
| **n_blocks** | `4` | AKT Optimal |
| **num_attn_heads** | `4` | AKT Optimal |

## 3. Reproduction Launcher Methodology

The `run_repro_experiment.py` script was used to launch both models. This approach ensures that all parameters are explicitly recorded in a `config.json` file and that the project's reproducibility protocol (Audit) is satisfied.

### AKT Run Command
```bash
python examples/run_repro_experiment.py --model_name akt --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1
```

### gTransformer Run Command
```bash
python examples/run_repro_experiment.py --model_name gtransformer --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1
```

### 4. Direct Script Methodology (Standard PyKT)
*(Section omitted for brevity, focusing on Benchmark Launcher results)*

### 5. Performance Parity Results (5-Fold CV)

The following table summarizes the metrics obtained across the full 5-fold cross-validation. The metrics for `AKT` and `gTransformer` are **functionally identical**, confirming successful replication.

| Metric | AKT (Benchmark) | gTransformer (Benchmark) | Delta |
| :--- | :--- | :--- | :--- |
| **Test AUC (Mean)** | **0.7825** ± 0.0017 | **0.7825** ± 0.0017 | **0.0000** |
| **Experiment ID** | `557255` | `282848` | - |
| **PyKT Reference** | 0.7853 (Δ-0.0028) | - | - |

**Conclusion**: The `gTransformer` model is a reliable architectural reproduction of `AKT`.

### 6. Comparison: Direct Execution vs. Launcher

While the final metrics are identical, the two methodologies serve different roles in the development lifecycle:

| Feature | Direct Scripts (`wandb_*.py`) | Reproduction Launcher (`run_repro_experiment.py`) |
| :--- | :--- | :--- |
| **Primary Goal** | Ad-hoc testing and framework baseline. | Scientific reproducibility and audit compliance. |
| **Parameter Source**| Explicit CLI flags or hardcoded defaults in code. | `configs/parameter_default.json` (Single source of truth). |
| **Audit Trail** | None (unless manually logged). | Mandatory `config.json` with MD5 and parameter audit. |
| **Output Structure** | Flat (or standard `saved_model` dir). | Structured timestamped folders with explicit metadata. |
| **Error Handling** | Standard Python tracebacks. | Pre-flight checks for parameter inconsistencies. |

### 7. Grounded Implementation (Priors Grounding)

The `GTransformer` implementation extends the base `AKT` architecture by grounding latent projections with pedagogical priors.

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

To ensure a rigorous comparison with baselines, evaluation is performed using the standard PyKT prediction script (`wandb_predict.py`) configured for **Question-Level Late Fusion**. This methodology aggregates predictions for multi-concept questions by averaging the probabilities of their constituent concepts, yielding the `oriauclate_mean` metric used in the benchmark.

```bash
# Standard Evaluation Command (Invoked by Launcher)
python examples/wandb_predict.py \
    --save_dir [EXPERIMENT_FOLD_DIR] \
    --bz 64 \
    --fusion_type late_fusion \
    --use_wandb 0
```

### 11. Comprehensive Analysis Pipeline

Once training is complete, the `generate_gtransformer_full_analysis.py` pipeline (automatically triggered by the launcher) regenerates all interpretability artifacts:
- **Interpretability Alignment**: Trajectory and Roster comparisons (`traj_predictions.csv`, `roster_gtransformer.csv`).
- **Probing Validation**: Diagnostic validation metrics and Pareto plots.
- **Advanced Visualizations**: Student clusters, Skill Mastery Maps, and Curriculum Heatmaps.

