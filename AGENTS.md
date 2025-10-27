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
- Launch scripts in such a way that we leverage around 60% of available GPUs (in current machine, for instance, we should GPUs 0 to 5) and less than 60% of CPU power. 

### Reproducibility

We treat every training or evaluation run as a formal experiment requiring full reconstruction capability. The following standards must be met for an experiment to be considered reproducible.

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

#### 2. Config File Schema (`config.json`)
The config file must include both user-specified and defaulted parameters. Suggested top-level keys:
```json
{
	"experiment": {
		"id": "20251025_141237_gainakt3_warmup8_align_retention",
		"model": "gainakt3",
		"short_title": "warmup8_align_retention",
		"purpose": "Assess sustained semantic alignment with extended warm-up and retention penalty"
	},
	"data": {
		"dataset": "assist2015",
		"split_strategy": "standard",
		"num_students": 17746
	},
	"training": {
		"epochs": 12,
		"batch_size": 64,
		"learning_rate": 0.000174,
		"optimizer": "Adam",
		"weight_decay": 0.0,
		"scheduler": "None",
		"mixed_precision": true,
		"gradient_clip": 1.0
	},
	"interpretability": {
		"use_mastery_head": true,
		"use_gain_head": true,
		"monotonicity_loss_weight": 0.1,
		"mastery_performance_loss_weight": 0.8,
		"gain_performance_loss_weight": 0.8,
		"sparsity_loss_weight": 0.2,
		"consistency_loss_weight": 0.3,
		"warmup_constraint_epochs": 8,
		"alignment_weight": 0.25,
		"retention_weight": 0.1,
		"lag_gain_weight": 0.05
	},
	"sampling": {
		"max_semantic_students": 200,
		"global_alignment_students": 600
	},
	"seeds": {
		"primary": 21,
		"all": [21,42,63,84,105]
	},
	"hardware": {
		"devices": [0,1,2,3,4],
		"num_workers": 5,
		"threads": 8
	},
	"command": "bash train.sh",
	"git": {
		"commit": "<hash>",
		"branch": "v0.0.9-gainakt3"
	},
	"timestamp": "2025-10-25T14:12:37Z"
}
```
Adjust values accordingly; record all added flags to prevent ambiguity.

#### 3. Launch experiment




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
