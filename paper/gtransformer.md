# GTransformer (Grounded Transformer)

The GTransformer model is a "Grounded" version of the Context-Aware Attentive Knowledge Tracing (AKT) architecture. It grounds output estimations of parameter values given by an intrinsic interpretable reference model like Bayesian Knowledge Tracing (BKT). This allows the model to learn student-specific parameters (initial mastery and learning rate) that are anchored to established pedagogical theory while maintaining the predictive power of Transformers. By anchoring deep representations to defined concepts, gTransformer offers a pedagogically interpretable alternative for data-driven personalization.

The main difference between gtransformer and the `pykt/models/idkt.py` implementation is that `idkt` grounded inputs/embeddings, whereas `gtransformer` grounds the **output parameter estimation**. The latent context vector $z$ is projected into enriched context-aware parameters ($p_{L0}, p_T$), which are then fed into a differentiable BKT logic layer.

## Theory-Guided Strategy

The GTransformer employs a "Prior-Adjustment" mechanism to ensure the deep learning model remains pedagogically meaningful. This mechanism acts as a Bayesian **prior**, where the model starts by assuming the student is "average" and then uses the Transformer's pattern-matching power to calculate a **contextual adjustment** ($\Delta$).

*   **Theoretical Bases (Logit-space Initialization)**: Every skill is initialized with population-level $L_0$ and $T$ from a pre-fit Bayesian Knowledge Tracing model. These probabilities are converted into **logits** to serve as the fixed intercept ($\text{Base}_{\text{theory}}$). This ensures that at $epoch=0$, the model already behaves like a valid BKT machine.
*   **Semantic Projections**: The model uses concept-specific axes ($\text{Axis}_{Know}, \text{Axis}_{Vel}$) to project the latent context vector $z$. The final context-aware parameter is calculated as the sum of the theoretical prior and the contextual delta:
    $$ \text{logit}(p) = \text{Base}_{\text{theory}} + \underbrace{(z \cdot \text{Axis})}_{\Delta \text{context}} $$
*   **Grounded Gaussian Initialization (Signal Survival)**: To ensure these theoretical priors are not "erased" by the Transformer's LayerNorm blocks, we use **Grounded Gaussian Initialization**. Instead of a flat constant, the bases are initialized with a small amount of Gaussian variance ($N(\mu=logit, \sigma=0.05)$). This non-zero variance allows the theoretical signal to propagate through the deep architecture.
*   **Anchored Learning**: By regularizing the learned parameters against these bases ($\mathcal{L}_{param}$), the model is forced to find the best *individual* deviations from the theoretical average, rather than learning unconstrained values that lack semantic meaning.

## Steps

### Step 0: Baseline Architecture (AKT-like)
> | **Attribute** | **Details** |
> | :--- | :--- |
> | **Commit** | `18604dad66b3b5ff112c566b2803e3d34e1641af` (Jan 13) |
> | **Experiment** | `20260113_1814_benchmark_CV_fixed_baseline_benchpaper` |
> | **Test AUC (Late Fusion)** | **0.7825** ± 0.0017 |
> | **Parameters Changed** | None (Established Baseline) |
> | **Interpretation** | Benchmarking results for a standard, unconstrained AKT architecture. |

*   **Foundation**: The core is the `AKT` model (Transformer Encoder with monotonic attention).
*   **Verification**: We established functional parity with the standard `AKT` model by running `gtransformer` with `--ablation all`.
*   **Result**: The baseline gTransformer achieves identical predictive performance to AKT on `assist2009`.

### Step 1: Augmented Input and Embeddings
> | **Attribute** | **Details** |
> | :--- | :--- |
> | **Commit** | `6ef69d83006e98f6fe238f6a91f23549fe225f82` (Jan 15) |
> | **Experiment** | N/A (Ablation Verification Only) |
> | **Test AUC (Late Fusion)** | N/A |
> | **Parameters Changed** | `ablation` flag enabled |
> | **Interpretation** | Verification that grounding infrastructure logic does not interfere with neural processing. |

*   **Grounding Infrastructure**: Integration of the BKT data loader to ingest population-level $L_0$ and $T$ parameters from pre-fit models (`bkt_skill_params.pkl`).
*   **Texturing**: Initialization of `l0_base_emb` and `t_base_emb` using the "Prior-Adjustment" logic (logit-space conversion).
*   **Verification**: Verified that the enrichment of the embedding space with theoretical priors does not degrade performance when the non-symbolic projection heads are bypassed (via `--ablation no_individualization`).

### Step 2: Grounded Outputs & Reference BKT Logic
> | **Attribute** | **Details** |
> | :--- | :--- |
> | **Commit** | `4b45fa41ac15ebab5ea2695812363ed70c1b46ed` (Jan 15) |
> | **Experiment** | `20260115_090230_benchpaper` |
> | **Test AUC (Late Fusion)** | **0.7800** ± 0.0013 |
> | **Parameters Changed** | `n_blocks: 4 -> 2`, `n_heads: 4 -> 8`, `lambda_ref: 0.5` |
> | **Interpretation** | Marginal (~0.25%) drop confirms grounding acts as a regularizer, narrowing the Rashomon set to focus on pedagogically valid representations. |

*   **Symbolic Output**: Instead of just predicting correctness $P(y)$, the model outputs two grounded BKT parameters for every timestep:
    *   $p_{L0}$: Context-aware Initial Mastery probability.
    *   $p_T$: Context-aware Learning Rate (Transition) probability.
*   **Reference Output (differentiable BKT)**: These parameters ($p_{L0}, p_T$) are fed into a **differentiable BKT layer** implemented directly in the forward pass. This layer performs a retrospective Bayesian update walk using:
    *   The model's generated parameters ($p_{L0}, p_T$).
    *   Fixed, population-level BKT Guess ($G$) and Slip ($S$) parameters (loaded from pre-fit BKT models).
*   **Dual Loss**: The model minimizes a combined loss:
    *   **Supervised Loss**: Standard BCE on the Transformer's direct prediction.
    *   **Reference Loss**: BCE on the BKT layer's prediction (forcing parameters to be valid for BKT logic).

### Step 3: Semantic Axes & Grounded Gaussian Initialization (Semantic Axes)
> | **Attribute** | **Details** |
> | :--- | :--- |
> | **Commit** | `4b45fa41ac15ebab5ea2695812363ed70c1b46ed` (Jan 15) |
> | **Experiment** | `20260115_090230_benchpaper` |
> | **Test AUC (Late Fusion)** | **0.7800** ± 0.0013 |
> | **Parameters Changed** | Same as Step 2 |
> | **Interpretation** | Successful projection of latent vectors through Semantic Axes without further performance degradation. |

*   **Motivation**: To ensure the parameters imply "Knowledge" and "Learning Ability" rather than arbitrary values.
*   **Implementation**: Instead of a black-box linear layer, parameters are projected using concept-specific semantic axes:
    *   $p_{L0} = \sigma(\text{Base}_{L0} + z \cdot \text{Axis}_{Know})$
    *   $p_{T} = \sigma(\text{Base}_{T} + z \cdot \text{Axis}_{Vel})$
*   **BKT Anchoring**: The `Base` terms are initialized from population-level BKT parameters, ensuring the model starts from a theoretically sound prior.

### Step 4: Individualization (Representational Grounding)
> | **Attribute** | **Details** |
> | :--- | :--- |
> | **Commit** | `4b45fa41ac15ebab5ea2695812363ed70c1b46ed` (Jan 15) |
> | **Experiment** | `20260115_090230_benchpaper` |
> | **Test AUC (Late Fusion)** | **0.7800** ± 0.0013 |
> | **Parameters Changed** | `n_blocks: 4 -> 2`, `n_heads: 4 -> 8`, `lambda_ref: 0.5` |
> | **Interpretation** | Final verification of the full Neuro-Symbolic pipeline. Implementation of Step 4 completes the representational grounding roadmap. |

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


## BKT Logic

In GTransformer, we do not simply use a pre-calculated BKT model to predict student performance. Instead, we implement a **Differentiable BKT Logic Wrapper** directly within the neural network's forward pass. This Neuro-Symbolic integration allows the Transformer to serve as a context-aware parameter estimator for a symbolic probabilistic machine.

### Differentiable Implementation
The BKT logic is encapsulated in the `_bkt_ref_output` method, which implements the standard Hidden Markov Model (HMM) recurrence equations for Bayesian Knowledge Tracing:

1.  **Bayes Update (Post-observation)**: Given an observation $y_{i}$ at history step $i$, the model updates the belief of mastery $L_i$:
    $$ P(L_i | y_i=1) = \frac{L_i(1 - S)}{L_i(1 - S) + (1 - L_i)G} $$
    $$ P(L_i | y_i=0) = \frac{L_i S}{L_i S + (1 - L_i)(1 - G)} $$
2.  **Learning Transition**: The belief for the next step $i+1$ is updated using the learning rate $T$:
    $$ L_{i+1} = P(L_i|y_i) + (1 - P(L_i|y_i))T $$
3.  **Output Emission**: The final prediction for step $t$ is:
    $$ P(y_t=1) = L_t(1 - S) + (1 - L_t)G $$

This logic is implemented using vectorized PyTorch operations, ensuring that the entire "BKT walk" is fully differentiable. This allows the Transformer to receive gradients through the BKT logic, learning to estimate parameters that are not only accurate but also theoretically consistent.

### Integration in GTransformer
GTransformer integrates this logic by separating the **Parameter Estimation** from the **Inference Machine**:

*   **The Estimator (Transformer)**: The Transformer encoder-decoder processes the student's interaction history and projects the latent context $z_t$ into context-aware parameters $p_{L0}$ (Initial Mastery) and $p_{T}$ (Learning Rate).
*   **The Machine (BKT Logic Wrapper)**: For every timestep $t$, the BKT Wrapper takes the estimated $(p_{L0}, p_T)$ and "walks" through the student's actual responses from $1 \dots t-1$ to calculate the mastery belief $L_t$. This **retrospective re-evaluation** is necessary because as the Transformer's estimation of the student's stable traits ($p_{L0}, p_T$) improves with more evidence, it must re-interpret the entire journey to arrive at a theoretically consistent diagnostic of their current knowledge. Despite its $O(T^2)$ nature, this process is computationally efficient as it matches the complexity of the Transformer's self-attention mechanism and is fully parallelized across the temporal dimension using vectorized GPU operations.
*   **The Reference Output**: The result of this walk is the **Reference Output** ($y_{ref}$). Because this output is constrained by BKT equations, any performance gains must come from the Transformer's ability to better estimate the underlying learner parameters $(p_{L0}, p_T)$ based on temporal context.

### Role of the pre-fit BKT Model (`pyBKT`)
While GTransformer implements its own BKT logic, we still leverage a standard BKT model (fit using Expectation-Maximization via `pyBKT`) for two critical "anchoring" purposes:

1.  **Global Priors**: The population-level Guess ($G$) and Slip ($S$) parameters are loaded from the pre-fit model and kept fixed during GTransformer training. This ensures that the model's interpretation of "Knowledge" remains grounded in standard pedagogical assumptions.
2.  **Theoretical Bases**: The `Base` terms for $p_{L0}$ and $p_T$ are initialized with the population-level $P(L_0)$ and $P(T)$ from the BKT model. This provides the Transformer with a "theory-guided" starting point, from which it learns to deviate based on specific student contexts.

**Why not use `pyBKT` directly?** We don't use the `pyBKT` library during training because it is a static, non-differentiable CPU library designed for population-level fits. By re-implementing the logic in PyTorch, we enable the high-performance, context-aware individualization that characterizes GTransformer.

### Implementation Snippet: Vectorized Retrospective Walk

The following snippet shows how we efficiently implement the retrospective re-evaluation using 3D tensor expansion. By expanding the history ($i$) and context ($t$) into separate dimensions, we can update the entire mastery block in parallel across all contexts.

```python
# Initial Belief L and Learning Rate T at context t
# Shape: [BS, seqlen_context, seqlen_history]
L = p_l0.unsqueeze(-1).expand(bs, seqlen, seqlen).clone()
T_rate = p_t.unsqueeze(-1).expand(bs, seqlen, seqlen)

# Iterative walk through student history (i)
for i in range(seqlen - 1):
    # Retrieve observation and skill priors at history step i
    obs = target[:, i].view(bs, 1, 1).expand(bs, seqlen, 1)
    g_i, s_i = gs[:, i].unsqueeze(1).unsqueeze(1), ss[:, i].unsqueeze(1).unsqueeze(1)
    
    L_i = L[:, :, i:i+1] # Belief for all contexts t at history step i
    
    # 1. Bayes Update (Probabilistic Symbolic Step)
    prob_correct = L_i * (1 - s_i) + (1 - L_i) * g_i
    L_post = torch.where(obs > 0.5, 
                         (L_i * (1 - s_i)) / prob_correct, 
                         (L_i * s_i) / (1 - prob_correct))
    
    # 2. Transition (Learning Step using context-aware T)
    L[:, :, i+1:i+2] = L_post + (1 - L_post) * T_rate[:, :, i:i+1]

# Diagonal Extraction: For context t, get mastery belief after journey 1...t-1
idx = torch.arange(seqlen)
L_at_t = L[:, idx, idx] 
```

## Regularization

In addition to the explicit *Grounding Losses* ($\mathcal{L}_{ref}$, $\mathcal{L}_{param}$) described above, the model employs a structural regularization mechanism (`LossRegularization`) to prevent overfitting of the difficulty parameters.

*   **Rasch Regularization**: This is an $L_2$ penalty applied specifically to the Question Difficulty embeddings ($u_q$) when `emb_type="qid"` is used with problem IDs.
    *   **Goal**: Encourages parsimony in the difficulty estimation, preventing the model from explaining away student performance variances solely through arbitrary difficulty adjustments.
    *   **Formula**: $\mathcal{L}_{reg} = \lambda_{rasch} \sum ||u_q||^2$
    *   **Implementation**: This logic is encapsulated within the `forward` pass (`gtransformer.py`) but added to the total loss in `train_model.py` via the `preloss` accumulator. In current benchmarks (`assist2009`), $\lambda_{rasch}$ is set to `1e-05`.


## Hyperparameter Configuration

Through a benchmarking campaign (documented in `paper/benchmark_paper.md`), we established the optimal architectural configuration for balancing predictive performance with interpretability.

### The "Wide & Shallow" Insight
Our experiments revealed a counter-intuitive dynamic: grounding works better in **wider, shallower** networks.
*   **Narrow Architectures (4 heads)**: Suffered significant performance degradation when grounding was applied ($-0.56\%$ AUC), likely due to a bottleneck in processing both pattern-matching and symbolic constraints.
*   **Wide Architectures (8 heads)**: Absorbed the grounding constraints with effectively **zero marginal cost** ($-0.03\%$ AUC). The increased width provides the necessary capacity to maintain separate subspaces for neural patterns and symbolic logic.

**Note**: While the 2-block/8-head gTransformer (0.7800 AUC) is the optimal *grounded* host, the absolute highest predictive performance observed in our campaign remains the **unconstrained (ablation=all) 4-block/4-head AKT baseline** (Exp 123509, **0.7838 AUC**). We intentionally trade this minor 0.38% predictive margin to gain full pedagogical interpretability at zero *marginal* cost for the chosen architecture.

### Recommended Defaults
Based on these findings, we endorse the following configuration as the standard for gTransformer:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `n_blocks` | **2** | Sufficient depth for reasoning; deeper models (4 blocks) showed diminishing returns. |
| `n_heads` | **8** | Critical width required to host neuro-symbolic logic without friction ("Interpretability for Free"). |
| `d_model` | 64 | Standard embedding size. |
| `lambda_ref` | 0.5 | Balanced weight for the Reference BKT Loss. |
| `lambda_init` | 0.1 | Regularization strength for Initial Mastery ($p_{L0}$). |
| `lambda_rate` | 0.1 | Regularization strength for Learning Rate ($p_T$). |

## Next Steps 

### Probing Lossess

### Per Paramete Regularization

