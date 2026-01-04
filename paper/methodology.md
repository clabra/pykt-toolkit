# Experimental Methodology: iDKT Validation

This document details the technical implementation and scientific rationale behind the evaluation of the iDKT model. It explains the divergence from standard benchmarking protocols in favor of interpretability-focused metrics and provides a guide for fair comparison.

## 1. Evaluation Granularity: Concept-Level vs. Question-Level

In Knowledge Tracing, datasets often contain questions associated with multiple Knowledge Components (KCs). Standard benchmarks (like the original PyKT paper) typically emphasize **Question-Level** performance.

### The iDKT Approach: Pure Concept Evaluation
iDKT experiments launched via `run_repro_experiment.py` and `eval_idkt.py` utilize **Concept-Level (Skill-Level)** evaluation without fusion.

*   **Mechanism**: Performance metrics (AUC, ACC) are calculated directly on the raw predictions of individual KCs.
*   **Rationale**: To validate **Representational Grounding**, we must maintain a 1-to-1 link between a specific pedagogical construct (e.g., "Addition") and its latent representation in the model.
*   **Avoidance of Fusion**: "Early Fusion" (averaging hidden states) and "Late Fusion" (averaging predictions like Mean/Vote/All) are skipped because they act as ensemble mechanisms that obscure the diagnostic accuracy of specific skill embeddings.

## 2. Temporal Logic: One-by-One vs. All-in-One

The "All-in-One" approach requires estimating all KCs of a multi-KC question simultaneously using only history *prior* to that question. iDKT instead uses **One-by-One** (or point-to-point) evaluation.

### The "Grounding Chain" Argument
iDKT is designed to align with a **BKT Reference Model**. BKT updates its mastery estimation after every single interacton (point-to-point). 

*   **Logic**: If iDKT were evaluated using "All-in-One" while BKT remains "One-by-One," a state mismatch would occur. The iDKT prediction would be "lagging" behind the BKT mastery state for multi-KC questions.
*   **Scientific Validity**: By using one-by-one evaluation, we ensure that iDKT and BKT are updated on the exact same temporal schedule. This is mandatory for our interpretability metrics ($I_1$: Convergent Validity and $I_2$: Functional Substitutability) to accurately measure the alignment between the deep latent space and the psychometric manifold.

## 3. Comparative Benchmarking Protocol

To perform a fair comparison between iDKT and other models (like AKT, SAINT, or DKT), two approaches are recommended depending on the goal:

### Approach A: Scientific Alignment (Baseline forced to KC-level)
To see if iDKT's interpretability provides a better "signal" than standard models, run baselines and disable fusion.

1.  **Launch iDKT**:
    ```bash
    python examples/run_repro_experiment.py --model idkt --dataset [DATASET] --short_title iDKT_Science
    ```
2.  **Launch Baseline (e.g., AKT)**:
    Since `wandb_predict.py` defaults to fusion, you must override it during evaluation:
    ```bash
    python examples/wandb_predict.py --model_name akt --fusion_type "none" --use_all_in_one False --test_filename [FILE]
    ```

### Approach B: Competition Mode (iDKT forced to Fused-Question)
To report iDKT on standard leaderboard terms, use the standard PyKT evaluation script instead of `eval_idkt.py`.

```bash
# iDKT is registered in evaluate_model.py hasearly list, 
# so it supports fusion and all-in-one in the standard script.
python examples/wandb_predict.py --model_name idkt --fusion_type "early_fusion,late_fusion" --use_all_in_one True
```

## 4. Scientific Rigor: 5-Fold Cross-Validation

Every AUC score reported in the final paper must be the **mean of 5 folds** with a reported standard deviation.

### Execution Workflow
The `run_repro_experiment.py` script is an atomic launcher. It does not iterate folds automatically. You must launch them explicitly:

```bash
# Example shell loop for a 5-fold run
for f in 0 1 2 3 4
do
   python examples/run_repro_experiment.py --model idkt --dataset assist2009_S --fold $f --short_title CrossVal_Run
done
```

### Aggregation
After the 5 runs complete, collect the results from the `eval_results.json` file in each experiment folder.
*   **Metric to Report**: Average Test AUC across the 5 folds.
*   **Uncertainty**: Calculation of the standard deviation ($\sigma$) to demonstrate model stability.

## 5. Parameter Consistency

All experiments must use the single source of truth: `configs/parameter_default.json`. 

*   **Zero Defaults Policy**: All scripts are designed to fail if a parameter isn't provided. 
*   **Launcher Role**: The launcher manages over 60 parameters (architecture, weights, optimizer). Manual overrides via CLI (e.g., `--learning_rate 0.0005`) are documented in the generated `config.json` for perfect provenance.
*   **Baseline Alignment**: When comparing models, ensure that base hyperparameters (like `d_model`, `n_heads`, and `dropout`) are identical unless the model's paper specifically dictates otherwise.

### Default Embedding Sizes by Model Family
The following table summarizes the default embedding dimensions ($d_{model}$ or $emb\_size$) defined in `configs/kt_config.json` for standard PyKT models:

| Model Family | Parameter Name | Default Value |
| :--- | :--- | :--- |
| **iDKT (Reproduction)** | `d_model` | **256** |
| **Attention-based (AKT, SAINT, SAKT, SimAKT)** | `d_model` / `emb_size` | **256** |
| **RNN-based (DKT, DKT+, DKVMN, Deep-IRT)** | `emb_size` / `dim_s` | **200** |
| **Graph-based (GKT)** | `emb_size` | **32** |
| **Time-aware (Hawkes)** | `emb_size` | **64** |
| **Convolutional (ATKT)** | `skill_dim` | **80** |

### Default Batch Sizes and Memory Safeguards
To maintain training stability and prevent Out-of-Memory (OOM) errors across different architectures, the following batch size protocols are applied:

| Protocol Source | Default Batch Size | Scope / Application |
| :--- | :--- | :--- |
| **iDKT Reproduction** | **64** | Standard setting in `parameter_default.json`. |
| **PyKT Base Config** | **64** | Global default in `kt_config.json`. |
| **Standard Overrides** | **64** | Forced override in `wandb_train.py` for AKT, SAINT, DKT, and DKVMN. |
| **Graph Safeguard** | **16** | Forced reduction for GKT due to graph memory intensity. |
| **HPC Optimized** | **1024** | Specialized setting for SimAKT and DTransformer. |

For all comparative experiments, a unified batch size of **64** is recommended to ensure that performance gains are not skewed by gradients derived from differing batch statistics.

## 6. Hyperparameter Search Space (Table 7)

To ensure a fair and rigorous comparison, iDKT follows the standard hyperparameter search spaces utilized in the official PyKT benchmark (detailed in Table 7 of the PyKT framework paper). This prevents "cherry-picking" results and ensures that all models are evaluated under similar levels of optimization.

### Common Search Space
The following ranges are utilized for all Deep Learning Knowledge Tracing (DLKT) models in our experiments:

| Parameter | Search Space / Values | Description |
| :--- | :--- | :--- |
| **Learning Rate** | {$10^{-3}$, $10^{-4}$, $10^{-5}$} | Optimizer step size. |
| **Dropout** | {0.05, 0.1, 0.3, 0.5} | Regularization rate. |
| **Seed** | {42, 3407} | Random initialization seeds. |
| **Embedding Size** | 256 (see section 5) | Latent dimension ($d_{model}$). |
| **Number of Blocks** | {1, 2, 4} | Number of Transformer/Encoder layers. |
| **Heads (Attention)** | {4, 8} | Number of multi-head attention heads. |
| **Batch Size** | 64 (see section 5) | Training batch size. |

### Sweep Configuration for iDKT
When launching sweeps for iDKT via W&B, use the same grid as the baseline models. For example, a W&B YAML configuration for an iDKT search would include:

```yaml
method: grid
parameters:
  learning_rate:
    values: [0.001, 0.0001, 0.00001]
  dropout:
    values: [0.05, 0.1, 0.3, 0.5]
  seed:
    values: [42, 3407]
  d_model:
    values: [128, 256]
```

### Reproducibility via Launcher
When using `run_repro_experiment.py`, these values should be passed as CLI overrides to the default values in `parameter_default.json`:

```bash
python examples/run_repro_experiment.py \
   --model idkt \
   --learning_rate 0.0001 \
   --dropout 0.2 \
   --d_model 256 \
   --short_title Sweep_Candidate_X
```

By adhering to the Table 7 search space, we validate that iDKT's performance is not a result of excessive tuning but is robust within the standard operational parameters of the Knowledge Tracing domain.

## 7. Dataset Variants: The "S" Protocol

To ensure the integrity of the **Representational Grounding** process, all iDKT experiments are conducted using the **"S" (Short/Stable)** versions of the educational datasets.

### Definition of the S Version
The "S" variant of a dataset (e.g., `assist2009_S`, `assist2015_S`) includes only student interaction sequences that are less than or equal to **200 interactions** in length.

### Rationale: Avoiding BKT Window Issues
The use of the S Protocol is a prerequisite for valid alignment with the BKT reference model:

1.  **Alignment Stability**: Standard BKT implementations can suffer from "mastery saturation" or numerical instability when processed over extremely long sequences. This drift makes the BKT mastery state an unreliable ground truth for deep model grounding.
2.  **Window Synchronization**: Many Knowledge Tracing benchmarks utilize windowing for long sequences. However, windowing can break the temporal continuity required for iDKT's **One-by-One** evaluation. By constraining the absolute sequence length, we eliminate the need for windowing and preserve a clean, continuous "Grounding Chain."
3.  **Representational Fidelity**: Filtering to $L \le 200$ ensures that the student-level parameters learned by iDKT ($v_s$: Velocity, $k_c$: Gap) are derived from a cohesive behavioral period, maximizing the correlation with the BKT reference parameters.

### Creating the "S" Variant
The generation of the "S" version is an automated two-step process to ensure reproducibility across any dataset supported by the framework.

#### 1. Execute the Filtering Script
Use the specialized script `examples/create_set_s.py` to filter out learners with extensive histories.
```bash
python examples/create_set_s.py --dataset assist2009 --max_len 200
```
**Effect**: This script scans the original sequence files (e.g., `train_valid_sequences.csv`) in the dataset directory and produces new filtered files with the `_S` suffix (e.g., `train_valid_sequences_S.csv`).

#### 2. Update Configuration
Once the files are generated, a new entry must be added to `configs/data_config.json` to define the search paths for the "S" version.
```json
// Example entry for assist2009_S
"assist2009_S": {
    "dpath": "../data/assist2009_S",
    "num_q": 17737,
    "num_c": 123,
    "input_type": ["questions", "concepts"],
    "train_valid_file": "train_valid_sequences_S.csv",
    "test_file": "test_sequences_S.csv"
}
```

### Launching with S-Data
To utilize these stable versions, specify the `_S` suffix in the dataset argument:

```bash
# Correct usage for Reproducibility Launcher
python examples/run_repro_experiment.py \
   --model idkt \
   --dataset assist2009_S \
   --short_title IDKT_Stable_Run
```
