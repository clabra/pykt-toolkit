# GTransformer Ablation Strategy

This document outlines the ablation strategy for the GTransformer model, specifically focusing on how the system reverts to a baseline state and how individual Neuro-Symbolic features are toggled.

## 1. The Baseline Codebase (`--ablation all`)

The **Ablation Baseline** refers to the state of the model where it is functionally equivalent to a standard Context-Aware Attentive Knowledge Tracing (**AKT**) architecture. This serves as the predictive performance floor.

When `ablation` is set to `"all"`, the following deactivations occur:

*   **Step 2 (Grounded Outputs deactivation)**: 
    *   The model skips the generation of the `reference_preds` (the Symbolic/BKT head).
    *   The **Differentiable BKT Layer** is bypassed; no Bayesian "walk" over history occurs during the forward pass.
*   **Step 3 (Textured Grounding deactivation)**:
    *   The **Semantic Axes** (`Axis_Know`, `Axis_Vel`) and concept-specific **BKT Bases** are not created or initialized.
    *   The latent context vector $z$ is not projected into pedagogical constructs; it flows directly to the standard supervised head.
*   **Step 4 (Individualization deactivation)**: 
    *   Student-specific latent biases ($v_s$ for velocity and $k_s$ for knowledge gap) are omitted.
*   **Loss Function Simplification**:
    *   The model calculates only the standard **Supervised Loss** (BCE). 
    *   The multi-component loss terms ($\lambda_{ref}, \lambda_{init}, \lambda_{rate}$) are excluded.

## 2. Parameter Control Summary

We use the `ablation` parameter (defined in `parameter_default.json` and overridable via CLI) to control this behavior.

| Scenario | Parameter Setting | Key Functional Changes | Expected Performance |
| :--- | :--- | :--- | :--- |
| **Full Model** | `ablation="none"` | Step 2, 3, and 4 active. Dual loss + Semantic axes + Student biases. | Best (Neuro-Symbolic) |
| **All Features Ablated** | `ablation="all"` | Reverts to AKT baseline. BKT head and semantic grounding disabled. | Parity with AKT (0.7825 AUC on AS2009) |
| **Regularization Ablated** | `ablation="regularization"` | Grounds parameters via axes but removes the MSE regularization ($\mathcal{L}_{param\_L0}, \mathcal{L}_{param\_T}$). | Tests impact of the prior constraint. |

## 3. Implementation Logic

### Forward Pass Mechanism
In `pykt/models/gtransformer.py`, the `forward` method uses conditional logic to prune the graph:

```python
if self.ablation == "all":
    outputs = {'predictions': preds}
    return outputs, c_reg_loss
```

This ensuring that no unnecessary computations (like the expensive BKT walk) are performed during baseline runs.

### Loss Calculation Mechanism
In `pykt/models/train_model.py`, the `cal_loss` function handles the ablation logic:

```python
if model.ablation != "all":
    # Add Reference Loss (BKT head)
    loss = loss + model.lambda_ref * loss_ref_sup + \
           model.lambda_initmastery * loss_l0 + \
           model.lambda_rate * loss_t
```

## 4. Operational Requirements

1.  **Parity Verification**: Any significant change to the GTransformer architecture must be verified by running `--ablation all` on `assist2009`. The resulting Mean AUC must match the AKT benchmark (**0.7825 ± 0.0017**) to ensure the foundation remains sound.
2.  **Audit Compliance**: Even in ablation mode, all parameters must be explicitly passed to the training script to satisfy reproducibility audits.
