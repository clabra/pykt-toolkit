# GTransformer Ablation Strategy

This document outlines the simplified ablation strategy for the GTransformer model. 

## 1. Ablation Baseline Codebase (Case 1)

The **Ablation Baseline Codebase** refers to the state of the code functionally equivalent to the commit `de277669539f91c1bed8ba1f09917ce17cc7091c`. 

**Functional Requirement**: When `ablation` is set to "all", the model must produce results that match the `AKT` metrics reported in `paper/benchmark_paper.md`.

## 2. Parameter Strategy

We use a single `ablation` parameter to control the model's behavior. 

1.  **Defaults**: `ablation` is set to "none" by default in `configs/parameter_default.json`. This ensures that a standard run includes the full gTransformer feature set.
2.  **Explicit Overrides**: For ablation runs, set `ablation` to "all" or "regularization" via the CLI.

## 3. Ablation Scenarios

| Scenario | Parameter Setting | Description | Reversion Behavior |
| :--- | :--- | :--- | :--- |
| **Full Model** | `ablation="none"` | The complete gTransformer model with all theory-guided components active. | N/A |
| **All Features Ablated** | `ablation="all"` | Reverts the model to the Baseline Codebase (AKT functionality). | 1. **Output**: Supervised Output only (Bayesian head disabled).<br>2. **Loss**: $L_{total} = L_{sup}$ (no $L_{reg}$ or $L_{gro}$).<br>3. **Input**: Standard data path (no BKT priors).<br>4. **Embeddings**: Standard size constraints (no augmented features). |
| **Regularization Ablated** | `ablation="regularization"` | Tests the impact of semantic regularization ($L_{reg}$). | 1. **Loss**: $L_{total}$ excludes $L_{reg}$.<br>2. **Note**: Other components (Input, Output, Embeddings) remain active. |

## 4. Implementation Details

- **Input Path Logic**:
    - If `ablation="all"`, read from `data/[dataset_name]` (Standard).
    - Otherwise, read from `data/[dataset_name]/bkt` (Augmented).

- **Loss Calculation**:
    - If `ablation="all"`, $L_{total} = L_{sup}$.
    - If `ablation="regularization"`, $L_{total} = L_{sup} + L_{gro}$ (exclude $L_{reg}$).
    - Else ($total$), $L_{total} = L_{sup} + L_{reg} + L_{gro}$.

- **Embedding & Output**:
    - If `ablation="all"`, disable parameter projection layers and Bayesian output head. Use standard MLP output.
    - Ensure embedding dimensions are adjusted to exclude augmented feature slots.
