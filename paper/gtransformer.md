# GTransformer (Grounded Transformer)

The GTransformer model is a "Grounded" version of the Context-Aware Attentive Knowledge Tracing (AKT) architecture. It grounds output estimations of parameter values given by an intrinsic interpretable reference model like Bayesian Knowledge Tracing (BKT). This allows the model to learn student-specific parameters (initial mastery and learning rate) that are anchored to established pedagogical theory while maintaining the predictive power of Transformers. By anchoring deep representations to defined concepts, gTransformer offers a pedagogically interpretable alternative for data-driven personalization.

We will launch experiments to demonstrate state-of-the-art accuracy showing that our model achieves superior diagnostic granularity by identifying student-specific parameters—such as initial knowledge and learning rates—that capture individual longitudinal contexts. 

## Design Guidelines
- gTransformer is a novel model that unifies the high predictive performance of deep learning with the intrinsic interpretability of traditional approaches like Bayesian Knowledge Tracing. 
- The design employs an encoder-decoder Transformer that enriches input sequences with parameter estimates from the interpretable model. Current implementation can be found in `pykt/models/gtransformer.py`. It borrows from the akt model described in `bibliography/papers-pykt/2020 Ghosh - AKT - Context-Aware Attentive Knowledge Tracing.pdf`.
- Attention mechanisms integrate the embeddings into a latent context vector, which is projected into updated parameter values constrained to remain semantically grounded to educational constructs while incorporating rich temporal dependencies learned by the network. The embeddings are escribed in detail in the section "Embeddings" of `paper/latex/paper.tex`.
- These enriched parameters then drive predictions through interpretable Bayesian logic. 
- We will use a loss function compound by 3 types of losses: 
    - A typical supervised classification loss for the prediction of student responses. See "Supervised Alignment (Lsup)" in the section "Loss Function" of `paper/latex/paper.tex`.
    - Regularization losses (one per parameter) for the parameter values to ensure they remain within valid ranges, similar to parameter values given by the reference model (BKT) and only diverge whent the evidence is strong enough. See "Regularization Term (Lreg)" in the section "Loss Function" of `paper/latex/paper.tex`.
    - Probing losses (one per parameter) to ensure that the model is learning the right things, i.e. that the values obtained by projection of the z context vector actually represent the intended concepts. To use this loss we extract the task-contextualized representations (zt)—the concatenation of the decoder’s hidden state and the task embedding—and train a simple linear regression probe to recover the BKT mastery probability P(Lt). However, high performance  on this true task alone is insufﬁcient, as high-capacity models might memorize arbitrary  patterns. To rigorously distinguish genuine encoding from superﬁcial correlation, we  implement a control task [ 35] by training an identical probe to predict a randomly shufﬂed  version of the theoretical labels. We quantify the integrity of the representation via the Selectivity metric. 

    ```
    Selectivity = R^2_{true} − R^2_{control}
    where R^2_{true} measures the variance explained when predicting actual BKT mastery, and R^2_{control} measures the same for the shufﬂed control. A high positive selectivity score (∆R2 > 0.5) provides terminal evidence of Structural Encoding, confirming that the pedagogical constructs remain the primary informative features through the model’s entire reasoning chain.
    ```
A digram of the gtransformer intended architecture, described in d2 language, can be found in `paper/latex/d2/arch.d2`. 

---

## Baseline Benchmark 
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
## BKT Augmented Datasets

To implement Informed Machine Learning with theoretical grounding, we augment the raw interaction datasets with population-level parameter estimates from a Bayesian Knowledge Tracing (BKT) model. This provides the GTransformer with a semantically grounded reference point for its latent representations.

### 1. Augmentation Methodology
The process involves training a population-level BKT model on the raw interaction history and projecting its parameters back into the original feature space.

- **Script**: `examples/augment_raw_with_bkt.py`
- **Informed Logic**: The script fits a BKT model using the Expectation-Maximization (EM) algorithm to identify the optimal Prior ($L_0$), Learn Rate ($T$), Guess ($G$), and Slip ($S$) parameters for every skill in the dataset.
- **Scientific Rigor**: Row-level predictions ($P(C_t)$) are **excluded** at this stage to prevent data leakage. The augmentation focuses solely on the four population parameters which represent invariant pedagogical characteristics of the Knowledge Components (KCs).

### 2. Column Mapping Resolution
The pipeline manages dataset-specific schemas (e.g., Assistments csv vs. Carnegie Learning txt) using a three-tier strategy:
1. **User Override**: Prioritizes any existing `data/[dataset]/bkt/default_dictionary.json`.
2. **Central Registry**: Pulls from a hardcoded registry in the script for standard benchmarks (Assist2009, AL2005, etc.).
3. **Generic Fallback**: Uses standard EDM headers (`user_id`, `correct`, etc.) if no specific mapping is found.

### 3. Technical Robustness
- **Sanitization**: Skill names containing special characters (common in `KC(Default)` fields) are temporarily mapped to numeric IDs during fitting to prevent regex compilation errors in the underlying `pyBKT` engine.
- **Delimiters**: The script dynamically handles both comma and tab-separated formats.

### 4. Replication Command
To generate the augmented version of a dataset (e.g., `assist2009`), execute:
```bash
docker exec pinn-dev /bin/bash -c "source /home/vscode/.pykt-env/bin/activate && \
python examples/augment_raw_with_bkt.py --dataset assist2009"
```

### 5. Input and Output Structure
- **Input**: The raw CSV file specified in the `dname2paths` mapping within the script (e.g., `data/assist2009/skill_builder_data_corrected_collapsed.csv`).
- **Outputs** (Saved in `data/[dataset]/bkt/`):
    - `[file_name]_bkt.csv`: The augmented dataset.
    - `parameters.json`: Human-readable population-level parameters per skill.
    - `model.pkl`: The serialized `pyBKT` model object.
    - `default_dictionary.json`: The column mapping used for the specific dataset.

### 6. Added Features and Semantic Meaning
The following columns are appended to the raw dataset to serve as theoretical priors for the GTransformer embedding layer:

| Column | Parameter | Pedagogical Meaning |
| :--- | :--- | :--- |
| `bkt_p_l0` | $P(L_0)$ | **Initial Knowledge**: Probability the student knows the skill before any interaction. |
| `bkt_p_t` | $P(T)$ | **Learning Rate**: Probability of transitioning from non-mastery to mastery after an interaction. |
| `bkt_p_g` | $P(G)$ | **Guessing**: Probability of a correct response despite lacking mastery. |
| `bkt_p_s` | $P(S)$ | **Slipping**: Probability of an incorrect response despite having achieved mastery. |

These features are consumed by the GTransformer during the embedding stage to anchor its latent context vector to established educational constructs.
