# Benchmark Results: Question-Level Late Fusion (Mean Average)

This document records the benchmark results for baseline Knowledge Tracing models on the `assist2009` dataset, evaluated using **Question-level Late Fusion (Mean Average)** methodology.

## Evaluation Methodology

- **Dataset**: assist2009
- **Cross-Validation**: 5-fold CV
- **Evaluation Level**: Question-level (not KC-level)
- **Fusion Type**: Late Fusion with Mean Average
- **Metrics**: `oriauclate_mean` (AUC), `oriacclate_mean` (ACC)

## Baseline Model Results

| Model | Mean AUC (Late Fusion) | Std AUC | Mean ACC (Late Fusion) | Std ACC | Experiment ID | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DKT** | **0.7528** | 0.0013 | **0.7236** | 0.0016 | 349457 | ✅ Complete |
| **DKVMN** | - | - | - | - | - | 🔄 Running |
| **SAKT** | - | - | - | - | - | ⏳ Queued |
| **SAINT** | - | - | - | - | - | ⏳ Queued |
| **AKT** | - | - | - | - | - | ⏳ Queued |

## Reference: AKT vs. gTransformer Parity (Folds 0-3)

| Metric | AKT (Exp 616804) | gTransformer (Exp 877631) | Status |
| :--- | :--- | :--- | :---: |
| **Folds Considered** | 0, 1, 2, 3 | 0, 1, 2, 3 | - |
| **Mean AUC (Late Fusion)** | 0.7720 | 0.7720 | **MATCH** |
| **Std AUC (Late Fusion)** | 0.0016 | 0.0016 | **MATCH** |
| **Mean ACC (Late Fusion)** | 0.7328 | 0.7328 | **MATCH** |
| **Std ACC (Late Fusion)** | 0.0020 | 0.0020 | **MATCH** |

**Parity Status: VERIFIED**
The metrics for gTransformer (a direct architectural port of AKT) are identical to the reference AKT implementation for Question-level Late Fusion (Mean) across folds 0-3.

---

## Experiment Details

### DKT
- **Experiment Path**: `experiments/20260113_181451_dkt_baseline_bench_CV_349457`
- **Hyperparameters**: Optimized from `configs/kt_config_assist2009.json`
  - `dropout`: 0.5
  - `emb_size`: 256
  - `learning_rate`: 0.001
  - `seed`: 42 (reproducibility)
- **Training Duration**: ~20 minutes (5 folds, 200 epochs with early stopping)
- **Results File**: `cv_results.json`

---

*Last updated: 2026-01-13 18:47 UTC*
*Benchmark launched with fixed subprocess handling to prevent process duplication*
