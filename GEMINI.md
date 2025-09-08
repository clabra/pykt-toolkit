# GEMINI.md

This file provides guidance to  work with code in this repository.

## Project Overview

This project, forked from pykt-toolkit, contains the implementation of many deep learning KT models in the pykt/models folder. Our objective is to implement a new model, add the code to the 'models' folder as well as to implement training and evaluation scripts to be added to the 'examples' folder. The implementation will be used to train a model on various datasets containing student interactions and evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder. 

The key contributions of the paper will be to contribute a new Transformer attention-based model witha good balance between performance (is competive in terms of AUC with state of the art attention-based models) and interpretability. 


Project Focus:
- The aproach to build the model will be detailed in a document (newmodel.md if not specified otherwise)
- Higlight interpretability and causal explanations as key contributions

Reference Documents:
papers-pykt folder - contains papers for most of the models included in pykt/models
- taxonomy.md - a taxonomy to classify attention-based models, most of them are included in the the pykt framework, so the code can be found in the pykt/models folder. 
- quickstart.pdf - explains how train and evaluate models, follow these guidelines when it comes to generate scripts for training and testing 
- contribute.pdf - explains how to proceed to add new models and datasets, follow these guidelined when it comes to code generation, scripts and documentation
- datasets.pdf - explains what datasets are used to evaluate models and how to get them
- models.pdf - list of models implemented in pykt/model foldes, includes a description and the name of the paper explaining the model (you can find the papers in papers-pykt folder)
- taxonomy.md - explains seq2seq to KT adaptation and model taxonomy


## Data directories:
- `/data`: Processed datasets ready for training
- `/data_original`: Raw datasets (do not modify)


## Development Commands

### Environment Setup
```bash
# Install dependencies and create virtual environment
pip install -e .
```

### Important Constraints
- Do NOT modify files in `/data_original` directory
- Do NOT modify existent files in `/data` directory (only modify files created for the new model/s)
- DO NOT modify existent models in pykt/models (only the new created model/s)
- Always work within the activated conda pykt virtual environment


## General Guidelines

### pykt Standards
We are creating a new model to contribute to the pykt framework. Follow guidelines in contribute.pdf and quickstart.pdf when it comes to create model code, training and evaluation scripts, model parameters, etc, following pykt standards.   

### Non-intrusive changes
The code and scripts for existent models in the pykt framework mustn't be changed. We only want to contribute a new model, without modifing existent models. 


## Some Terms and Concepts

- **item**: A problem in the dataset
- **learning path**: Student's sequence of interactions with problems 
- **relevant knowledge components**: Knowledge Components (KC) related with a given item or problem according to the q-matrix
- **relevant skills**: Skills that problems are designed to develop/assess. The relevant KC indicate the skills that the student will develop by working with the problem. 
- **interaction**: A student's attempt at solving a problem
- **mastery**: Probability that a student has mastered a knowledge component
- **Q-matrix**: Binary matrix linking problems and knowledge components (KCs)
- **G-matrix**: Matrix with the same shape than the Q-matrix. Each value is a real number between 0 and 1 that indicates to what extent the interaction of the student with a problem will develop the relevant knowledge compponents.
- **learning gain**: Value of a cell in the G-matrix, indicating the expected increase in mastery for a knowledge component after an interaction with the corresponding problem. The value of a skill is the sum of the learning gains after all interactions with the problems related to that skill.
- **gain signature**: A vector of learning gains for all knowledge components
- **gain token**: A token representing the gain signature, used as input to the Transformer
- A complete description of concepts in assistment datasets can be found in the `assist09.md` file.

## Output and Style Guidelines

- Most scripts are in `examples` folder
- Scripts should accept command-line arguments for flexibility. Include default parameter values in script documentation
- Reproducibility of experiments is a must, so all parameter values should be saved and read from a config file 
- Code documentation should include usage examples with command-line arguments
- Use markdown format for documentation files
- Use an academic professional tone, avoiding the use of emojis, exclamations, informal language or marketing jargon.
- Talk to the user in the second person (you) in documentation. 
- Ensure that all documentation is clear, concise, and accessible to a PhD-level audience.

### Integration with Project

You can analyze and reference project's key documents:
  - workspace code and documentation for implementation tasks
  - read contribute.pdf to know how to make contributions to the to create the code and scripts to add a new model, and evaluate it, using current practices and project standards
  - put additional documentation and auxiliary scripts in the 'assistant' folder. Try to avoid put them in other folders. 
  - taxonomy.md - for models context and taxonomy with special focus in attention mechnisms

## Copyright and License

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

This project and all its contents, including but not limited to source code, documentation, and data, are private and confidential.

**Strictly Prohibited:**
- Sharing, distributing, or disclosing any part of this project to third parties.
- Using any content from this repository to train, fine-tune, or otherwise improve any machine learning or artificial intelligence models.
