# GTransformer (Grounded Transformer)

The GTransformer model is a "Grounded" version of the Context-Aware Attentive Knowledge Tracing (AKT) architecture. It grounds output estimations of parameter values given by an intrinsic interpretable reference model like Bayesian Knowledge Tracing (BKT). This allows the model to learn student-specific parameters (initial mastery and learning rate) that are anchored to established pedagogical theory while maintaining the predictive power of Transformers. By anchoring deep representations to defined concepts, gTransformer offers a pedagogically interpretable alternative for data-driven personalization.

The main difference between gtransformer and the `pykt/models/idkt.py` implementation is that `idkt` grounded inputs/embeddings, whereas `gtransformer` grounds the **output parameter estimation**. The latent context vector $z$ is projected into enriched context-aware parameters ($p_{L0}, p_T$), which are then fed into a differentiable BKT logic layer.

## Architecture & Implementation Steps

The implementation of `gTransformer` follows a Neuro-Symbolic architecture, progressively built in four verified steps:

### Step 1: Baseline Architecture (AKT-like)
*   **Foundation**: The core is the `AKT` model (Transformer Encoder with monotonic attention).
*   **Verification**: We established functional parity with the standard `AKT` model in `pykt-toolkit` by running `gtransformer` with `--ablation all`.
*   **Result**: The baseline gTransformer achieves identical predictive performance to AKT on `assist2009` (AUC ~0.7825).

### Step 2: Grounded Outputs & Reference BKT Logic
*   **Symbolic Output**: Instead of just predicting correctness $P(y)$, the model outputs two grounded BKT parameters for every timestep:
    *   $p_{L0}$: Context-aware Initial Mastery probability.
    *   $p_T$: Context-aware Learning Rate (Transition) probability.
*   **Reference Output (differentiable BKT)**: These parameters ($p_{L0}, p_T$) are fed into a **differentiable BKT layer** implemented directly in the forward pass. This layer performs a retrospective Bayesian update walk using:
    *   The model's generated parameters ($p_{L0}, p_T$).
    *   Fixed, population-level BKT Guess ($G$) and Slip ($S$) parameters (loaded from pre-fit BKT models).
*   **Dual Loss**: The model minimizes a combined loss:
    *   **Supervised Loss**: Standard BCE on the Transformer's direct prediction.
    *   **Reference Loss**: BCE on the BKT layer's prediction (forcing parameters to be valid for BKT logic).

### Step 3: Textured Grounding (Semantic Axes)
*   **Motivation**: To ensure the parameters imply "Knowledge" and "Learning Ability" rather than arbitrary values.
*   **Implementation**: Instead of a black-box linear layer, parameters are projected using concept-specific semantic axes:
    *   $p_{L0} = \sigma(\text{Base}_{L0} + z \cdot \text{Axis}_{Know})$
    *   $p_{T} = \sigma(\text{Base}_{T} + z \cdot \text{Axis}_{Vel})$
*   **BKT Anchoring**: The `Base` terms are initialized from population-level BKT parameters, ensuring the model starts from a theoretically sound prior.

### Step 4: Individualization (Representational Grounding)
*   **Student Logic**: When student IDs are available (`n_uid > 0`), the model learns static student-specific biases:
    *   $v_s$: Student Velocity bias (added to $p_T$).
    *   $k_s$: Student Knowledge Gap bias (added to $p_{L0}$).
*   **Integration**: These are additive terms in the logit space, allowing the model to capture that some students are systematically faster learners or have higher prior knowledge, independent of the context.

## Loss Function

The model is trained using a multi-component loss function to enforce grounding:

$$ \mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_{ref}\mathcal{L}_{ref} + \lambda_{L0}\mathcal{L}_{param\_L0} + \lambda_{T}\mathcal{L}_{param\_T} $$

1.  **$\mathcal{L}_{sup}$**: Standard binary cross-entropy on the transformer's direct prediction ($y_{pred}$ vs $y_{true}$).
2.  **$\mathcal{L}_{ref}$**: Binary cross-entropy on the BKT Reference Output ($y_{bkt}$ vs $y_{true}$). This forces the learned $p_{L0}, p_T$ to be useful for BKT reasoning.
3.  **$\mathcal{L}_{param}$**: MSE regularization penalizing deviation of $p_{L0}, p_T$ from their population-level BKT priors, ensuring they don't drift into theoretically invalid regions.

## Baseline Benchmark Verification

This section summarizes the results verifying Step 1 (Baseline Parity) between `gTransformer` (in ablation mode) and `AKT`.

### Performance Parity Results (5-Fold CV)

| Metric | AKT (Benchmark) | gTransformer (Baseline Mode) | Delta |
| :--- | :--- | :--- | :--- |
| **Test AUC (Mean)** | **0.7825** ± 0.0017 | **0.7825** ± 0.0017 | **0.0000** |
| **Experiment ID** | `557255` | `282848` | - |

**Conclusion**: The `gTransformer` model is a reliable architectural reproduction of `AKT` when BKT components are disabled.

## Current Training Campaign (Neuro-Symbolic)

We are currently running a **5-Fold Cross-Validation Campaign** on `assist2009` with the full Neuro-Symbolic architecture enabled:
*   **Dataset**: `assist2009`
*   **Configuration**:
    *   `lambda_ref`: 0.5
    *   `lambda_initmastery`: 0.1
    *   `lambda_rate`: 0.1
    *   `ablation`: `none` (Full architecture active)
*   **Live Experiments**: Fold 0-4 are running on dedicated GPUs.
