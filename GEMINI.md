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

- The scripts used to train and evaluate the models should be ready for experiment reproduction. Create config files and log mechanisms that guarantees easy reproducibility. 
- To guarantee reproducibility is mandatory that all parameter values used in the experiment are properly recorded in a config file to be used to set the parameter values in a later run. 
- When asked to document experiments in README files, include files sucha as config, logs, etc. and the command that is needed to exactly reproduce the experiment again


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
