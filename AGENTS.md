This file provides guidance to  work with code in this repository. The files referenced here can be found in the `assistant` folder that contains documentation with context and guidelines for assistants. 

## Project Overview

This project, forked from pykt-toolkit, contains the implementation of many deep learning models for Knowledge Tracing in the pykt/models folder. Our objective is to implement a new model, add the code to the 'models' folder as well as to implement training and evaluation scripts to be added to the 'examples' folder. We will train the model on various datasets containing student interactions and will evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder. 

The key contributions of the paper will be a new Transformer attention-based model witha good balance between performance (i.e. is competive in terms of AUC with state of the art attention-based models) and interpretability. Our approach higlights interpretability and causal explanations as key contributions

Reference Documents:

- `newmodel.md` details our aproach including the architecture of the model 
- `papers-pykt` folder contains papers for most of the models included in `pykt/models`
- `taxonomy.md` - a taxonomy to classify attention-based models, most of them are included in the `pykt` framework, so the code can be found in the `pykt/models` folder
- `quickstart.pdf` - explains how to train and evaluate models, follow these guidelines when it comes to generating scripts for training and testing
- `contribute.pdf` - explains how to proceed to add new models and datasets, follow these guidelines when it comes to code generation, scripts, and documentation
- `datasets.pdf` - explains what datasets are used to evaluate models and how to get them
- `models.pdf` - a description and a paper for each model in `pykt/models`. Papers can be found in `papers-pykt` folder)

## Data directories:
- `/data`: Processed datasets ready for training
- `/data_original`: Raw datasets (do not modify)


### Environment Setup

Commands should be launched inside a virtual environment that can be activated with: 
```bash
source /home/vscode/.pykt-env/bin/activate
```
The machine we are currently using has 8 GPUs. Use 5 GPUs when running commands. 

### Reproducibility

We treat every training or evaluation run as a formal experiment requiring full reproductibility as detailed in `examples/reproducibility.md`. All default values for parameters should be specified in a single source of truth: `configs/parameter_default.json`. CLI flags override individual defaults; absence of a CLI flag implies the default recorded in the experiment's `config.json` (no hidden or implicit defaults allowed). The following standards must be met for an experiment to be considered reproducible.

We want to avoid the risks of having parameter defaults hardcoded. Changes in hardcoded values would not be reflected unless parameter_default.json is manually update first; moreover, evaluation could keep using another values, producing divergent checkpoints and invalid reproducibility claims. Hard-coding also prevents per-experiment architectural variation via overrides.

when you change any parameter default value (the reference values are in paper/parameters.cvs) follow guidelines in "Parameter Evolution Protocol" section. 


### Important Constraints
- Always work within the activated conda pykt virtual environment (`.pykt-env`)
- Do NOT modify files in `/data_original` directory
- Do NOT modify existent files in `/data` directory (only modify files created for the new model/s)
- DO NOT modify existent models in `pykt/models` (only the new created model/s)
- The code and scripts for existent models in the pykt framework mustn't be changed. We only want to contribute a new model, without modifing existent ones. 


## Guidelines

### pykt Standards
- We are creating a new model to contribute to the pykt framework. Follow guidelines in `contribute.pdf` and `quickstart.pdf` when it comes to create model code, training and evaluation scripts, model parameters, etc, following pykt standards.   
- Most scripts are in `examples` folder
- New files and scripts that don't adhere to the standard guidelines described in `contribute.pdf` and `quickstart.pdf` should be created in the `tmp` dir. Create them only when it's neccessary; if they are temporal or merely informative, perhaps is enough with providing the apporpiate feedback without the need of create too many new files. The objective is maintain the original structure as similar as possible to that in the `main` branch while isolating auxiliary files in `tmp` folder. This way it will be easier to do contributions to the upstream repo from which we forked.  

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
- Avoid creating documentation files unless specifically asked for them. Provide feedback directly in responses instead.


## Copyright and License

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

This project and all its contents, including but not limited to source code, documentation, and data, are private and confidential.

**Strictly Prohibited:**
- Sharing, distributing, or disclosing any part of this project to third parties.
- Using any content from this repository to train, fine-tune, or otherwise improve any machine learning or artificial intelligence models.
