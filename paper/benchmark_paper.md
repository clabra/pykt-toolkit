# Benchmark Results for Paper

## Test AUC Table: Question Level - Late Fusion (Mean Average)

**Evaluation**: 5-fold Cross-Validation  
**Metric**: Test AUC - Question-level Late Fusion (Mean Average) - `oriauclate_mean`

| Model | AS2009 | AL2005 | BD2006 | NIPS34 |
| :--- | :---: | :---: | :---: | :---: |
| **DKT** | **0.7528** ± 0.0013<br/>(PyKT: 0.7541, Δ-0.0013) | -<br/>(PyKT: 0.8149) | -<br/>(PyKT: 0.8015) | -<br/>(PyKT: 0.7689) |
| **DKVMN** | **0.7440** ± 0.0011<br/>(PyKT: 0.7473, Δ-0.0033) | -<br/>(PyKT: 0.8054) | -<br/>(PyKT: 0.7983) | -<br/>(PyKT: 0.7673) |
| **ATKT** | **0.7715** ± 0.0011<br/>(PyKT: 0.7470, Δ+0.0245) | -<br/>(PyKT: 0.7995) | -<br/>(PyKT: 0.7889) | -<br/>(PyKT: 0.7665) |
| **SAKT** | **0.7263** ± 0.0013<br/>(PyKT: 0.7246, Δ+0.0017) | -<br/>(PyKT: 0.7880) | -<br/>(PyKT: 0.7740) | -<br/>(PyKT: 0.7517) |
| **SAINT** | **0.6921** ± 0.0028<br/>(PyKT: 0.6958, Δ-0.0037) | -<br/>(PyKT: 0.7775) | -<br/>(PyKT: 0.7781) | -<br/>(PyKT: 0.7873) |
| **AKT** | **0.7825** ± 0.0017<br/>(PyKT: **0.7853**, Δ-0.0028) | -<br/>(PyKT: **0.8306**) | -<br/>(PyKT: **0.8208**) | -<br/>(PyKT: **0.8033**) |
| **gTransformer** | **0.7825** ± 0.0017<br/>(Parity Check: Δ0.0000) | - | - | - |

**Notes**:
- **Our Results**: Mean ± Std across 5 folds using Question-level Late Fusion (Mean Average)
- **PyKT Reference**: Values in parentheses from PyKT paper Table 2 (Liu et al. 2023) using "All-in-One" evaluation
- **Delta (Δ)**: Difference between our result and PyKT reference (Our - PyKT). Negative = we're lower, Positive = we're higher
- **DKT AS2009**: Δ-0.0013 is within our standard deviation (±0.0013), confirming excellent reproducibility
- AS2009 = assist2009, AL2005 = algebra2005, BD2006 = bridge2algebra2006, NIPS34 = nips_task34
- All models use optimized hyperparameters from `configs/kt_config_[dataset].json`
- Current benchmark run: AS2009 only (other datasets to be added)

**Experiments**:
- **DKT (AS2009)**: Experiment ID `349457`, Path: `experiments/20260113_1814_benchmark_CV_fixed_baseline/20260113_181451_dkt_baseline_bench_CV_349457/`
- **DKVMN (AS2009)**: Experiment ID `333762`, Path: `experiments/20260113_1814_benchmark_CV_fixed_baseline/20260113_183428_dkvmn_baseline_bench_CV_333762/`
- **AKT (AS2009)**: Experiment ID `557255`, Path: `experiments/20260113_1814_benchmark_CV_fixed_baseline/20260113_202934_akt_baseline_bench_CV_557255/` (Completed)
- **gTransformer (AS2009)**: Experiment ID `282848`, Path: `experiments/20260113_1814_benchmark_CV_fixed_baseline/20260113_202934_gtransformer_baseline_bench_CV_282848/` (Completed)
- **SAKT (AS2009)**: Experiment ID `675489`, Path: `experiments/20260113_1814_benchmark_CV_fixed_baseline/20260113_203934_sakt_baseline_bench_CV_675489/` (Completed)
- **SAINT (AS2009)**: Experiment ID `925894`, Path: `experiments/20260113_1814_benchmark_CV_fixed_baseline/20260113_214623_saint_baseline_bench_CV_925894/` (Completed)
- **ATKT (AS2009)**: Experiment ID `528418`, Path: `experiments/20260113_1814_benchmark_CV_fixed_baseline/20260113_232605_atkt_baseline_bench_CV_528418/` (Completed)

**Commands**:
To reproduce these results, run the following commands from the project root:

```bash
# DKT
python examples/run_repro_experiment.py --model_name dkt --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1

# DKVMN
python examples/run_repro_experiment.py --model_name dkvmn --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1

# AKT
python examples/run_repro_experiment.py --model_name akt --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1

# gTransformer (Our Model)
python examples/run_repro_experiment.py --model_name gtransformer --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1

# SAKT
python examples/run_repro_experiment.py --model_name sakt --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1

# SAINT
python examples/run_repro_experiment.py --model_name saint --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1

# ATKT
python examples/run_repro_experiment.py --model_name atkt --dataset assist2009 --cv --short_title baseline_bench --num_gpus 1
```

---

## Hyperparameters Table

**Overview of hyperparameters used for all models** (adapted from PyKT benchmark Table 7)

### Model-Specific Hyperparameters

| Hyperparameter | DKT | DKVMN | SAKT | SAINT | AKT | ATKT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Embedding Type** | qid | qid | qid | qid | qid | qid |
| **Embedding Size** | 256 | - | 64 | 256 | 64 (d_model) | - |
| **Hidden Dimension** | - | 256 (dim_s) | - | - | 64 (d_model) | 256 |
| **Memory Size** | - | 64 (size_m) | - | - | - | - |
| **Attention Heads** | - | - | 4 | 8 | 4 | - |
| **Encoder Blocks** | - | - | 1 (num_en) | 4 (n_blocks) | 4 (n_blocks) | - |
| **Feed-Forward Dim** | - | - | - | - | 256 (d_ff) | - |
| **Skill Dimension** | - | - | - | - | - | 64 |
| **Answer Dimension** | - | - | - | - | - | 256 |
| **Attention Dimension** | - | - | - | - | - | 256 |
| **Dropout** | 0.5 | 0.1 | 0.5 | 0.3 | 0.1 | 0.5 |
| **Learning Rate** | 0.001 | 0.001 | 0.001 | 0.001 | 0.0001 | 0.001 |
| **Epsilon (ε)** | - | - | - | - | - | 5 |
| **Beta (β)** | - | - | - | - | - | 1.0 |
| **Seed** | 3407 | 42 | 3407 | 3407 | 3407 | 3407 |

### Common Hyperparameters (All Models)

| Hyperparameter | Value |
| :--- | :---: |
| **Batch Size** | 64 |
| **Sequence Length** | 200 |
| **Max Epochs** | 200 |
| **Optimizer** | Adam |
| **Early Stopping Patience** | 10 |
| **Dataset** | assist2009 |
| **Cross-Validation** | 5-fold |

### Notes

- All hyperparameters are optimized values from `configs/kt_config_assist2009.json` (PyKT team's hyperparameter tuning results)
- Common parameters are from `configs/parameter_default.json`
- **qid**: Question ID embedding type
- Early stopping is applied based on validation AUC
- Seeds vary by model based on optimal tuning results (3407 for most, 42 for DKVMN)

---

*Table will be updated as each model completes its 5-fold CV run*  
*Last updated: 2026-01-13 19:11 UTC*
