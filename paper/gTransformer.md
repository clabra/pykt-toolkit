# gTransformer 

## Performance Verification

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

### Conclusion
The successful parity test confirms that `gTransformer` is a faithful implementation of the AKT architecture. The identical performance across the Launcher and Direct Scripts further validates that our Reproducibility Layer effectively wraps the original PyKT business logic without introducing side effects or performance degradation.
