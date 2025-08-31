# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimAKT is a Knowledge Tracing (KT) model designed to predict the responses given by a student to each question/problem in a sequence of interactions, while also estimating the evolution of the skills mastery. It is based on a Transformer-based architecture that processes sequences of student interactions. 

This project, pykt-toolkit, contains the implementation of many deep learning KT models in the pykt/models folder. Our objective is to implement a new model, called SimAKT, add the code to the 'models' folder and training and evaluation scripts to the 'examples' folder. The implementation will be used to train a model on various datasets containing student interactions and evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder. 

The doc `attention_taxonomy.md` explains this issue and set a taxonomy to classify the models included in the the pykt framwork (and other ones). The `seq2seq-transformers.pdf` doc deeps dive in this modality of Transformers and `similarity-transformers.pdf` explore the state of the art regarding the use of similarity approaches in attention-based models. 

The key contributions of the paper will be to present an alternative to current Transformer architectures using a similarity-based approach that improves interpretability and causal explanations. 

The abstract of the paper is as follows: 

    Modeling the dynamic evolution of student knowledge states is essential for advancing personalized education. This paper introduces a novel Transformer-based model that leverages attention mechanisms grounded in the similarity of learning trajectories to predict learning curves that allows not only to predict the correctness of responses but also to track the evolution of skills mastery.

    Our evaluation across multiple educational datasets demonstrates that the proposed model achieves competitive predictive performance compared to state-of-the-art Deep Knowledge Tracing models. More importantly, it offers significantly improved interpretability by explicitly modeling the evolution of knowledge states and the underlying causal mechanisms based on learning curves. This advancement enhances the explainability of predictions, making the model more transparent and actionable for educators and learners.

    This research contributes to the development of intelligent tutoring systems by enabling accurate and interpretable predictive modeling. It lays the groundwork for open learning models, where students can visualize their learning progress. Additionally, it provides a robust framework for personalized learning design and targeted educational interventions, empowering teachers to monitor student progress, deliver precise feedback, and design courses tailored to individual learning needs.

Project Focus:
- The model now uses similarity-based attention mechanisms grounded in learning trajectory similarity
- Goal is to predict learning curves (not just binary responses) to track skills mastery evolution
- Focus on interpretability and causal explanations as key contributions

Reference Documents:
- quickstart.pdf - explains how start to train and evaluate models, it's useful to understand project organization
- contribute.pdf - explains how to proceed to add new models and datasets, follow these guidelined when it comes to code generation, scripts and documentation
- datasets.pdf - explains what datasets are used to evaluate models and how to get them
- models.pdf - list of models implemented in pykt/model foldes, includes a description and the name of the paper explaining the model (you can find the papers in papers-pykt folder)
- attention_taxonomy.md - explains seq2seq to KT adaptation and model taxonomy
- seq2seq-transformers.pdf - deep dive into seq2seq Transformer modalities
- similarity-transformers.pdf - explores similarity approaches in attention-based models

Research Approach:
- SimAKT model: Transformer-based architecture using similarity-based attention
- Learning curves prediction: Beyond correctness to track skills mastery evolution
- Causal mechanisms: Explicit modeling based on learning curves for better interpretability


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
- Always work within the activated conda pykt virtual environment

## Documentation
`papers-pykt` folder contains papers for most of the models included in pykt/models


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
- Parameter values precedence: config_file > user_params > optimized_defaults
- Code documentation should include usage examples with command-line arguments
- Use markdown format for documentation files
- Use an academic professional tone, avoiding the use of emojis, exclamations, informal language or marketing jargon.
- Talk to the user in the second person (you) in documentation. 
- Ensure that all documentation is clear, concise, and accessible to the intended audience.

### Integration with Project

You can analyze and reference project's key documents:
  - workspace code and documentation for implementation tasks
  - read contribute.pdf to know how to make contributions to the to create the code and scripts to add a new model, and evaluate it, using current practices and project standards
  - put additional documentation and auxiliary scripts in the 'assistant' folder. Try to avoid changes in existent codebase.
  - attention_taxonomy.md - for models context and taxonomy with special focus in attention mechnisms
  - assitant/simakt.md for a detailed description of the requirements and guidelines to dessign and implement the new SimAKT model
  - seq2seq-transformers.pdf - for architectural guidance
  - similarity-transformers.pdf - for similarity approaches 

## Copyright and License

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

This project and all its contents, including but not limited to source code, documentation, and data, are private and confidential.

**Strictly Prohibited:**
- Sharing, distributing, or disclosing any part of this project to third parties.
- Using any content from this repository to train, fine-tune, or otherwise improve any machine learning or artificial intelligence models.
