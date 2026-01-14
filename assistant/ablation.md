# GTransformer Ablation Strategy

This document outlines the systematic ablation strategy for the GTransformer model. Ablation studies are essential to decompose the contribution of each theoretical component to the overall predictive performance and interpretability of the model.

## 1. Ablation Baseline Codebase

The **Ablation Baseline Codebase** refers to the state of the code functionally equivalent to the commit `de277669539f91c1bed8ba1f09917ce17cc7091c`. In this scenario, all augmented features are disabled, and the model behaves as a standard Context-Aware Attentive Knowledge Tracing (AKT) architecture. 

The baseline results for this version are documented in `paper/benchmark_paper.md`. Any addition of new features must ensure that the model returns to this exact level of performance when all ablation toggles are set to `false`.

## 2. Parameter On/Off Strategy

To ensure scientific rigor and adherence to the "Explicit Parameters, Zero Defaults" reproducibility guidelines, the following protocol is followed for all ablation runs:

1.  **Defaults**: All ablation toggles are set to `true` by default in `configs/parameter_default.json`. This ensures that a standard run includes the full feature set.
2.  **Explicit Overrides**: For ablation studies, toggles must be explicitly set to `false` via the CLI or a dedicated experiment configuration file.
3.  **Data Path Switching**: The `prior_augmentation` toggle controls the data ingestion path:
    - `true`: Read from `data/[dataset_name]/bkt` (BKT-augmented sequences).
    - `false`: Read from `data/[dataset_name]` (Standard interaction sequences).
4.  **Baseline Reversion**: When a feature is toggled off, the model architecture must revert to its fallback behavior (functionally copy of AKT) without hidden side-effects or residual parameters being active in the computation graph.

## 3. Ablation Parameters Table

The following toggle in `GTransformer` controls its theory-guided features:

| Parameter | Type | Default | Component | Description |
| :--- | :--- | :---: | :--- | :--- |
| `prior_augmentation` | `bool` | `true` | **Theoretical Priors** | When `true`, input data is enriched with population-level parameters from a reference model (e.g., BKT $L_0, T, G, S$). Data is read from `data/[dataset_name]/bkt`. |

## 4. Ablation Matrix

The following table defines the combination of parameters required to ablate specific features and reach the AKT baseline.

| Experiment Label | `prior_augmentation` | Target Insight |
| :--- | :---: | :--- |
| **GTransformer (Full)** | `true` | Performance of the architecture anchored to theoretical BKT population estimates. |
| **Ablation Baseline** | `false` | Reverts the model to the original AKT architecture (Standard Benchmark). |

## 5. Update Protocol

This document must be updated whenever a new toggleable feature is added. Each update must:
- Define the new parameter and its pedagogical purpose.
- Expand the Ablation Matrix.
- Verify that the parameter follows the "Explicitness" protocol in `configs/parameter_default.json`.
