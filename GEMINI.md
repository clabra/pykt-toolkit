This file provides guidance to work with code in this repository. The files referenced here can be found in the `assistant` folder that contains documentation with context and guidelines for assistants.

## Project Overview

This project, forked from pykt-toolkit, contains the implementation of many deep learning models for Knowledge Tracing in the pykt/models folder. Our objective is to implement a new model, add the code to the 'models' folder as well as to implement training and evaluation scripts to be added to the 'examples' folder. We will train the model on various datasets containing student interactions and will evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder.

The key contributions of the paper will be a new Transformer attention-based model witha good balance between performance (i.e. is competive in terms of AUC with state of the art attention-based models) and interpretability. Our approach higlights interpretability and causal explanations as key contributions

Reference Documents:

- `papers-pykt` folder contains papers for most of the models included in `pykt/models`
- `assistant/taxonomy.md` - a taxonomy to classify attention-based models, most of them are included in the `pykt` framework, so the code can be found in the `pykt/models` folder
- `assistant/quickstart.pdf` - explains how to train and evaluate models, follow these guidelines when it comes to generating scripts for training and testing
- `assistant/contribute.pdf` - explains how to proceed to add new models and datasets, follow these guidelines when it comes to code generation, scripts, and documentation
- `assistant/datasets.pdf` - explains what datasets are used to evaluate models and how to get them
- `assistant/models.pdf` - a description and a paper for each model in `pykt/models`. Papers can be found in `papers-pykt` folder)

## Datasets

- `/data`: Processed datasets ready for training
- `/data_original`: Raw datasets (do not modify)
- `data/datasets.md`: details about the datasets and `keyid2idx.json`, a dictionary mapping original dataset IDs and zero-based sequential indices used internally by the pykt models.

## Environment Setup

Commands should be launched inside a virtual environment that can be activated with:

```bash
source /home/vscode/.pykt-env/bin/activate
```

Always check that the terminal used to launch commands runs inside a container (e.g. docker container).

The machine we are currently using has 8 GPUs. Use 5 GPUs when running commands.

## Reproducibility

We treat every training or evaluation run as a formal experiment requiring full reproductibility as detailed in `examples/reproducibility.md`. All default values for parameters should be specified in a single source of truth: `configs/parameter_default.json`. CLI flags override individual defaults; absence of a CLI flag implies the default recorded in the experiment's `config.json` (no hidden or implicit defaults allowed). The following standards must be met for an experiment to be considered reproducible.

We want to avoid the risks of having parameter defaults hardcoded. Changes in hardcoded values would not be reflected unless parameter_default.json is manually update first; moreover, evaluation could keep using another values, producing divergent checkpoints and invalid reproducibility claims. Hard-coding also prevents per-experiment architectural variation via overrides.

when you change any parameter default value (the reference values are in paper/parameters.cvs) follow guidelines in "Parameter Evolution Protocol" section.

## Important Constraints

- Always work within the activated conda pykt virtual environment (`.pykt-env`)
- Do NOT modify files in `/data_original` directory
- Do NOT modify existent files in `/data` directory (only modify files created for the new model/s)
- DO NOT modify existent models in `pykt/models` (only the new created model/s)
- The code and scripts for existent models in the pykt framework mustn't be changed. We only want to contribute a new model, without modifing existent ones.

## Guidelines

### pykt Standards

- We are creating a new model to contribute to the pykt framework. Follow guidelines in `assistant/contribute.pdf` and `assistant/quickstart.pdf` when it comes to create model code, training and evaluation scripts, model parameters, etc, following pykt standards.
- Most scripts are in `examples` folder
- New files and scripts that don't adhere to the standard guidelines described in `assistant/contribute.pdf` and `assistant/quickstart.pdf` should be created in the `pykt-toolkit/tmp` dir. Create them only when it's neccessary; if they are temporal or merely informative, perhaps is enough with providing the apporpiate feedback without the need of create too many new files. The objective is maintain the original structure as similar as possible to that in the `main` branch while isolating auxiliary files in `pykt-toolkit/tmp` folder. This way it will be easier to do contributions to the upstream repo from which we forked.

### Operational standards

- Training and evaluation should be launched using the commands desribed in ´examples/reproducibility.md´
- Avoid launching commands terminating scripts tha are running in the terminal (tail command, for instance, can cause KeyboardInterrupt exceptions).
- Launch scripts in such a way that we leverage available GPUs (less than 75% if not set otherwise) and CPUs (less than 75% of CPU power).

### Code and Style Guidelines

- Code documentation should include usage examples
- Use markdown format for documentation files
- Use an academic professional tone, avoiding the use of emojis, icons, exclamations, informal language or marketing jargon.
- Use "we" instead of "you" following academic writing conventions.
- Ensure that all documentation is clear, concise, and accessible to a PhD-level audience.
- If you need to create documentation files, create them in `./tmp`folder, unless I specifically ask for them.
- Follow guidelines in `reproducibility.md` to avoid hardcoded default values for parameters. Don't avoid audits.
- After changes in codebase, always check if parameters in `configs/parameter_default.json` were added or modified. If so, apply guidelines described in "### Parameter Evolution Protocol" to propagate the changes in order to have proper reproducibility guarantees.
- When creating or modifyng texts don't include time estimations
- Only do commits when I ask for. In general, I prefer commit after testing with experiments. Don't add nothing to the commit unless it is explicitly asked for.
- In general, try to avoid fallbacks. I prefer fail as early as possible, throwing exceptions in case something doesn't match what is expected.

## Models architecture and implementation

- `assistant/newmodel.md`: points to the doc containing the description of current version of the model being tested
- `paper/proposal.md`: description of the appraoch we are proposing as a base for the model and paper experiments
- `paper/models.md`: a summary with the models we explored, issues, changes and evolution
- `paper/implementation.md`: implementation details. A kind of history with details related to different implementations. If you don't find information about some detail in the rest of documents in the `paper` folder, look here.
- `paper/rasch.md`: IRT model theory and principles
- `paper/bkt.md`: Bayesian Knowledge Tracing (BKT) theory and principles
- `paper/theory_informed_dl.md`: Informed Deep Learning, theory and principles about how to use theoretical models to inform a deep learning model.

## Instructions

Your are an expert in deep learning models applied to knowledge tracing. You are also an expert in the field of knowledge tracing, educational assessment, statistics and machine learning, with a strong background in experimental design and reproducibility. You know the pykt-toolkit framework aimed to implement and compare many deep kwnowledge tracing models.

### Role-Specific Guidelines

#### 👨‍💻 For Coding & Implementation

When you are writing or fixing code (Coder Agent):

- The code of the models are in `pykt/models`. The scripts to train and evaluate them in `examples`. The papers about these models can be found in `bibliography/papers-pykt`.
- Prioritize modifications in `pykt/models` for model architecture but only for new models we are implementing, not for existent models.
- Follow the stricter `assistant/contribute.pdf` guidelines for code style.
- **Do not** modify the `data_original` directory.
- Always run a small test script (e.g., in `tmp/`) before committing major changes.

#### 📝 For Documentation & Writing

When you are updating the paper or documentation (Writer Agent):

- Use the academic "we" instead of "you".
- Maintain a tone suitable for a PhD-level audience.
- Use/create info in `paper` folder for knowledge about our approach (paper and proposed models), and experiment results.
- Use papers in `bibliography` for theoretical alignment and get state-of-the-art knowledge about knowledge tracing and related topics. The file `bibliography/biblio.bib` contains the bibliography that is referenced in other documents using `@` followed by the key of the entry in the biblio.bib file.

#### 📊 For Experiments & Reproducibility

When you are running experiments (Experiment Agent):

- Follow `assistant/quickstart.pdf` guidelines.
- Ensure all default parameters are in `configs/parameter_default.json`.
- Strictly following the reproducibility protocol in `examples/reproducibility.md`.

## Copyright and License

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

This project and all its contents, including but not limited to source code, documentation, and data, are private and confidential.

**Strictly Prohibited:**

- Sharing, distributing, or disclosing any part of this project to third parties.
- Using any content from this repository to train, fine-tune, or otherwise improve any machine learning or artificial intelligence models.
