This file provides guidance to  work with code in this repository. The files referenced here can be found in the `assistant` folder that contains documentation with context and guidelines for assistants. 

## Project Overview

This project, forked from pykt-toolkit, contains the implementation of many deep learning KT models in the pykt/models folder. Our objective is to implement a new model, add the code to the 'models' folder as well as to implement training and evaluation scripts to be added to the 'examples' folder. The implementation will be used to train a model on various datasets containing student interactions and evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder. 

The key contributions of the paper will be to contribute a new Transformer attention-based model witha good balance between performance (is competive in terms of AUC with state of the art attention-based models) and interpretability. 


Project Focus:
- The aproach to build the model will be detailed in a document (`newmodel.md` if not specified otherwise)
- Higlight interpretability and causal explanations as key contributions

Reference Documents:

- `papers-pykt` folder contains papers for most of the models included in `pykt/models`
- `taxonomy.md` - a taxonomy to classify attention-based models, most of them are included in the `pykt` framework, so the code can be found in the `pykt/models` folder
- `quickstart.pdf` - explains how to train and evaluate models, follow these guidelines when it comes to generating scripts for training and testing
- `contribute.pdf` - explains how to proceed to add new models and datasets, follow these guidelines when it comes to code generation, scripts, and documentation
- `datasets.pdf` - explains what datasets are used to evaluate models and how to get them
- `models.pdf` - lists models implemented in `pykt/models` folder, includes a description and the name of the paper explaining the model (you can find the papers in `papers-pykt` folder)


## Data directories:
- `/data`: Processed datasets ready for training
- `/data_original`: Raw datasets (do not modify)


## Development Commands

### Environment Setup
```bash
# Install dependencies and create virtual environment
pip install -e .
```
The machine has 8 GPUs. Use 5 GPUs to run commands as default. 

### Important Constraints
- Always work within the activated conda pykt virtual environment (`.pykt-env`)
- Do NOT modify files in `/data_original` directory
- Do NOT modify existent files in `/data` directory (only modify files created for the new model/s)
- DO NOT modify existent models in `pykt/models` (only the new created model/s)
- The code and scripts for existent models in the pykt framework mustn't be changed. We only want to contribute a new model, without modifing existent ones. 


## General Guidelines

### pykt Standards
- We are creating a new model to contribute to the pykt framework. Follow guidelines in `contribute.pdf` and `quickstart.pdf` when it comes to create model code, training and evaluation scripts, model parameters, etc, following pykt standards.   
- Most scripts are in `examples` folder
- New files and scripts that don't adhere to the standard guidelines described in `contribute.pdf` and `quickstart.pdf` should be created in the `tmp` dir. Create them only when it's neccessary; if they are temporal or merely informative, perhaps is enough with providing the apporpiate feedback without the need of create too many new files. The objective is maintain the original structure as similar as possible to that in the `main` branch while isolating auxiliary files in `tmp` folder. This way it will be easier to main contributions to the upstream folder we forked to develop new models.  


### Operational standards

- Avoid monitoring commands that interrupt scripts running in the terminal (tail command, for instance, can cause KeyboardInterrupt exceptions). Launch this kind of commands in such a way that terminating ongoing processes is avoided. 
- Launch scripts in such a way that we leverage available GPUs (60% as much if not set otherwise) and less than 60% of CPU power. 

### Reproducibility

We treat every training or evaluation run as a formal experiment requiring full reconstruction capability. All resolved parameters originate from a single source of truth: `configs/parameter_default.json`. This file now includes architectural, interpretability, and runtime parameters (`seed`, `monitor_freq`, `use_amp`, `use_wandb`, `enable_cosine_perf_schedule`). CLI flags override individual defaults; absence of a CLI flag implies the default recorded in the experiment's `config.json` (no hidden or implicit defaults allowed). The following standards must be met for an experiment to be considered reproducible.

We want to avoid the risks of having parameter defaults hardcoded. Changes in hardcoded values would not be reflected unless parameter_default.json is manually update first; moreover, evaluation could keep using another values, producing divergent checkpoints and invalid reproducibility claims. Hard-coding also prevents per-experiment architectural variation via overrides.

#### 1. Experiment Folder Structure
Each experiment creates a dedicated folder under `examples/experiments` using the convention:
```
[YYYYMMDD]_[HHMMSS]_[modelname]_[shorttitle]
```
Example:
```
20251025_141237_gainakt3_warmup8_align_retention
```

Mandatory contents within this folder:
| File / Dir | Purpose |
|------------|---------|
| `config.json` | Canonical source of ALL resolved arguments (explicit + defaults) and environment metadata. |
| `train.sh` / `evaluate.sh` | Shell wrapper showing exact command, pinned devices, thread limits, and any env vars. |
| `results.json` | Per-epoch metrics + best metrics block (AUC, accuracy, loss, interpretability metrics). |
| `metrics_epoch.csv` | Tabular epoch log for quick external analysis. |
| `stdout.log` | Raw console log (include timestamp prefix per line). |
| `stderr.log` (optional) | Captured error output if separated. |
| `model_best.pth` | Best checkpoint (selection criterion recorded). |
| `model_last.pth` | Last epoch checkpoint for recovery. |
| `environment.txt` | Python version, PyTorch version, CUDA version, git commit hash. |
| `SEED_INFO.md` | Seeds used and rationale (e.g., multi-seed stability assessment). |
| `README.md` | Human-readable summary (objective, config highlights, table with key results (val AUC, accuracy, loss, master correlation, gain correlation, etc.), results interpretation (including comparison with metrics in the previous experiment), reproducibility checklist). |
| `artifacts/` | Auxiliary plots or correlation trajectory JSONs. |


#### 2. Parameters Consistency 

`configs/parameter_default.json` contains two sections:
1. `training_defaults`
2. `evaluation_defaults`

Coverage rules:
- Every argparse-defined training/evaluation parameter must appear exactly once in the corresponding section.
- Runtime and monitoring parameters (now included): `seed`, `monitor_freq`, `use_amp`, `use_wandb`, `enable_cosine_perf_schedule`.
- If a new parameter is added to a script, add it to the JSON before running experiments; otherwise reproduction claims are invalid.

Override mechanics:
1. Training: `examples/run_repro_experiment.py` loads defaults, applies CLI overrides; resolved dictionary is serialized immediately into `config.json`.
2. Evaluation: `examples/eval_gainakt2exp.py` loads `evaluation_defaults`, applies CLI overrides, writes resolved evaluation command into `config.json.runtime.eval_command`.

Drift prevention:
- After modifying any script's argparse block, run a consistency check (to be added) that diffs argparse parameter names against JSON keys.
- Update the appendix in `paper/README_gainakt2exp.md` only after the JSON change; the README must never introduce parameters not present in the JSON.

Config hash protocol:
- Compute MD5 on a sorted key serialization of `config.json` immediately after writing (stored in `config.json.config_md5`).
- Any change in defaults or overrides results in a new hash; prior runs become non-equivalent for reproduction unless re-launched.

Appendix synchronization:
- The Appendix in `paper/README_gainakt2exp.md` is regenerated from `configs/parameter_default.json` (manual or scripted) to ensure alignment.
- If a parameter is removed, mark it deprecated in commit message; do not silently delete without experiment archival.

No hidden defaults:
- Absence of a value in `config.json` is not permitted. Every parameter must be explicit. Boolean flags appear as `true`/`false`.

Failure conditions:
- If `seed` or `monitor_freq` missing from either JSON or `config.json`, mark run as non-reproducible.
- If MD5 hash not recorded, run cannot be audited.

#### 3. Config File Schema (`config.json`)

When an experiment is launched, the launcher copies structured defaults into `config.json` and applies CLI overrides; it does not embed unresolved implicit values. The full resolved set (including runtime/meta groups) is the canonical configuration.


#### 3. Launch experiment

The config.json file will contain the commands used to launch training and to evaluate the model. 


#### 4. Metric & Logging Standards
Per epoch log at minimum: `epoch`, `train_loss`, `val_auc`, `val_accuracy`, `mastery_corr`, `gain_corr`, `mastery_variance`, `gain_variance`, `constraint_loss_share`. Log in both JSON and CSV. Provide final block with best epoch summary and selection criterion (e.g., best validation AUC).

#### 5. Integrity & Determinism
- Set torch, numpy, random seeds before data loading.
- Record whether cudnn deterministic & benchmark modes are set.
- Capture Python / library versions in `environment.txt`.

#### 6. Reproducibility Checklist (to include in each experiment `README.md`)
| Item | Status |
|------|--------|
| Folder naming convention followed | ✅ |
| `config.json` contains all params (no omissions) | ✅ |
| Shell script lists full command | ✅ |
| Best + last checkpoints saved | ✅ |
| Per-epoch metrics CSV present | ✅ |
| Raw stdout log saved | ✅ |
| Git commit & branch recorded | ✅ |
| Seeds documented | ✅ |
| Environment versions captured | ✅ |
| Correlation / interpretability metrics logged | ✅ |

#### 7. Common Pitfalls & Mitigations
| Pitfall | Symptom | Mitigation |
|---------|---------|------------|
| Missing default parameters | Re-run differs subtly (learning rate, weight decay) | Always serialize resolved args dict immediately after parsing. |
| Non-deterministic data ordering | Different epoch trajectories | Fix seed and disable workers shuffling beyond DataLoader. |
| Overwriting experiment folder | Loss of original logs | Disallow reuse: script should abort if target folder exists unless `--resume`. |
| Partial logs on crash | Truncated JSON/CSV | Write metrics atomically (temp file rename). |
| Invisible environment drift | AUC shifts between runs | Record environment.txt and compare in review. |
| Device mismatch | Different GPU mapping changes perf | Pin `CUDA_VISIBLE_DEVICES` explicitly in shell script. |

#### 8. Resume Protocol
If a run halts mid-epoch, include a `resume_state.json` with last completed epoch and RNG states the training script can ingest via `--resume`. On successful completion, remove or archive this file.

#### 9. Documentation of Experiments in Papers / READMEs
When referencing results externally, cite the experiment folder name exactly and link relative path (e.g., `examples/experiments/20251025_141237_gainakt3_warmup8_align_retention`). Summaries must state: dataset, seeds, epochs, primary metrics, and reproducibility confirmation (all checklist items passed).

#### 10. Minimal Acceptance Criteria for Reproducibility
An experiment is considered reproducible only if: 

(i) config hash (sorted JSON MD5) is identical; 
(ii) rerunning `train.sh` regenerates matching best epoch metrics (within stochastic tolerance if multi-seed)

Failure to meet any mandatory item above requires remediation before inclusion in comparative tables or publication assets.

#### 11. Results

Once a experiment finishes, add a row to examples/experiments/RESULTS.csv with the key results: val AUC, accuraccy, loss, mastery correlation, gains correlation, etc. Fill also the column `best AUC` with `True` and `best Master Corr` with `True`when some of these results are the best compared to ealier experiments (and put earlier result to `False` in this case). 


### Code and Style Guidelines
- Scripts should accept command-line arguments for flexibility. Include default parameter values in script documentation
- Reproducibility of experiments is a must, so all parameter values should be saved and read from a config file 
- Code documentation should include usage examples with command-line arguments
- Use markdown format for documentation files
- Use an academic professional tone, avoiding the use of emojis, icons, exclamations, informal language or marketing jargon.
- Use "we" instead of "you" following the academy convention. 
- Ensure that all documentation is clear, concise, and accessible to a PhD-level audience.



## Copyright and License

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

This project and all its contents, including but not limited to source code, documentation, and data, are private and confidential.

**Strictly Prohibited:**
- Sharing, distributing, or disclosing any part of this project to third parties.
- Using any content from this repository to train, fine-tune, or otherwise improve any machine learning or artificial intelligence models.
