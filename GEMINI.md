This file provides context and guidelines about this project and code in the repository.

## Project Overview

This project uses pykt-toolkit as a starting point, forked from pykt-toolkit github repository. It contains, in the pykt/models folder, the implementation of many deep learning models for Knowledge Tracing. We want to implement a new model, add the code to the 'models' folder as well as to implement training and evaluation scripts to be added to the 'examples' folder. We will train the model on various datasets containing student interactions and will evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder.

The key contributions of the paper will be a new Transformer attention-based model with a good balance between performance (i.e. is competive in terms of AUC with state of the art attention-based models) and interpretability. Our approach higlights interpretability and explanability as key contributions

## Paper Publication

The paper is intended to be published in this [MDPI special issue](https://www.mdpi.com/journal/applsci/special_issues/KODK4051XS) that, ultimately, aims to compile rigorous research that explores the impact of AI, EDM and advanced technological tools on evolving educational models. Its target is researchers and practitioners interested in original ideas that improve the effectiveness and quality of education in increasingly diverse learning contexts. So, the abstract should highlight the practical outcomes of the approach more than architecture approaches or technical novelties. We should try to answer the question: what can be done with our proposal than can't be done without it?.

## MDPI Special Issue Information

Artificial intelligence (AI) and educational data mining (EDM) are profoundly transforming the educational landscape, reshaping the paradigms of teaching and learning. AI applications in education enable the analysis of large datasets, revealing intricate patterns in student behaviour and facilitating the personalisation of learning experiences. The GED, meanwhile, provides detailed information that enables evidence-based decision-making in education. Likewise, the integration of human–computer interaction (HCI) improves the accessibility and personalisation of educational interfaces, making learning tools more intuitive and responsive, and more advanced methods, including predictive modelling and machine learning algorithms, allow educational systems to dynamically adapt to individual needs, fostering a more inclusive and learner-centred approach to education. This Special Issue calls on the academic community to examine how these technologies are driving the development of adaptive learning environments, enabling early identification of student needs and promoting accessibility and equity in education.

Ultimately, this Special Issue aims to compile rigorous research that explores the impact of AI, EDM and advanced technological tools on evolving educational models. Researchers and practitioners are invited to contribute original ideas that improve the effectiveness and quality of education in increasingly diverse learning contexts.

The scope of the Special Issue includes, but is not limited to, the following topics:

- Artificial intelligence in education;
- Educational data mining (EDM);
- Interactive machine learning (IML);
- Human-in-the-loop machine learning and machine teaching;
- Educational data mining and learning analytics;
- Predictive modelling in education;
- Machine learning for learning analytics;
- Neural networks in educational AI;
- Natural language processing (NLP) in education;
- Intelligent tutoring systems (ITS);
- AI ethics in education;
- Inclusion and accessibility in AI tools.

## Paper Abstract

"Knowledge Tracing, which enables the estimation of how students' knowledge evolves as they interact with educational content, is a cornerstone of intelligent tutoring systems. Traditional approaches coexist with deep knowledge tracing models that deliver better predictions but suffer from a critical drawback: lack of interpretability, which is particularly problematic in educational contexts. To overcome this limitation, we propose gTransformer, a novel model that unifies the high predictive performance of deep learning with the intrinsic interpretability of traditional approaches like Bayesian Knowledge Tracing. Our design employs an encoder-decoder Transformer that enriches input sequences with parameter estimates from the interpretable model. Attention mechanisms integrate these embeddings into a latent context vector, which is projected into updated parameter values constrained to remain semantically grounded to educational constructs while incorporating rich temporal dependencies learned by the network. These enriched parameters then drive predictions through interpretable Bayesian logic. Experiments demonstrating state-of-the-art accuracy show that our model achieves superior diagnostic granularity by identifying student-specific parameters—such as initial knowledge and learning rates—that capture individual longitudinal contexts. By anchoring deep representations to defined concepts, gTransformer offers a pedagogically interpretable alternative for data-driven personalization."


## Reference Documents

- `papers-pykt` folder contains papers for most of the models included in `pykt/models`
- `assistant/taxonomy.md` - a taxonomy to classify attention-based models, most of them are included in the `pykt` framework, so the code can be found in the `pykt/models` folder
- `assistant/quickstart.pdf` - explains how to train and evaluate models, follow these guidelines when it comes to generating scripts for training and testing
- `assistant/contribute.pdf` - explains how to proceed to add new models and datasets, follow these guidelines when it comes to code generation, scripts, and documentation
- `assistant/datasets.pdf` - explains datasets used in the pykt-toolkit framework and how to get them
- `assistant/models.pdf` - prvides a list with a description and a paper for each model in `pykt/models`. Many of the cited papers can be found in `papers-pykt` folder

## Datasets

- `/data`: Processed datasets ready for training
- `/data_original`: Raw datasets (do not modify)
- `assistant/datasets.md`: details about some datasets that we'll use to test models, and an explanation of `data/[DATASET]/keyid2idx.json`, a bidirectional mapping dictionary that converts between original dataset IDs and zero-based sequential indices used internally by the pykt framework.

## Environment Setup

The project is run inside a Docker container (container name: `pinn-dev`). All commands related to development, training, and evaluation **MUST** be executed within this container using the dedicated virtual environment.

### Command Execution Protocol

To launch commands properly, follow these steps:

1.  **Terminal Entry**: Ensure you are using a terminal that is attached to the container or use `docker exec`.
2.  **Environment Activation**: The virtual environment is located at `/home/vscode/.pykt-env`. It must be activated before running any scripts:
    ```bash
    source /home/vscode/.pykt-env/bin/activate
    ```
3.  **Working Directory**: The project root inside the container is `/workspaces/pykt-toolkit`.
4.  **Host Machine Execution**: If running commands from the host machine, use `docker exec` to target the container:
    ```bash
    docker exec -w /workspaces/pykt-toolkit pinn-dev /bin/bash -c "source /home/vscode/.pykt-env/bin/activate && python examples/run_repro_experiment.py ..."
    ```

### GPU Resources
The machine has 8 GPUs.
- **Standard Allocation**: Use 5 GPUs for training runs (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3,4`).
- **Monitoring**: Always verify GPU availability before launching multi-GPU experiments.

## Reproducibility

We treat every training or evaluation run as a formal experiment requiring full reproductibility as detailed in `examples/reproducibility.md`. All default values for parameters should be specified in a single source of truth: `configs/parameter_default.json`. CLI flags override individual defaults; absence of a CLI flag implies the default recorded in the experiment's `config.json` (no hidden or implicit defaults allowed). The following standards must be met for an experiment to be considered reproducible.

We want to avoid the risks of having parameter defaults hardcoded. Changes in hardcoded values would not be reflected unless parameter_default.json is manually update first; moreover, evaluation could keep using another values, producing divergent checkpoints and invalid reproducibility claims. Hard-coding also prevents per-experiment architectural variation via overrides.

when you change any parameter default value (the reference values are in paper/parameters.cvs) follow guidelines in "Parameter Evolution Protocol" section.

## Ablation Studies

See `assistant/ablation.md` for guidelines on how to augment or modify the model code in order to be able to properly perform ablation studies.

## Important Constraints

- Always work within the activated .pykt-env virtual environment
- Do NOT modify files in `/data_original` directory
- Do NOT modify existent files in `/data` directory (only modify files created for the new model/s)
- DO NOT modify existent models in `pykt/models` (only the new created model/s). The code and scripts for existent models in the pykt framework mustn't be changed. We only want to contribute a new model, without modifing existent ones.
- DO NOT modify scripts in examples such as wandb_train.py, wandb_predict.py or wandb_[model_name]_train.py that are use by pykt framework to train and evaluate models. 
## Guidelines

### pykt Standards

- We are creating a new model to contribute to the pykt framework. Follow guidelines in `assistant/contribute.pdf` and `assistant/quickstart.pdf` when it comes to create model code, training and evaluation scripts, model parameters, etc, following pykt standards.
- Most scripts are in `examples` folder
- New files and scripts that don't adhere to the standard guidelines described in `assistant/contribute.pdf` and `assistant/quickstart.pdf` should be created in the `pykt-toolkit/tmp` dir. Create them only when it's neccessary; if they are temporal or merely informative, perhaps is enough with providing the apporpiate feedback without the need of create too many new files. The objective is maintain the original structure as similar as possible to that in the `main` branch while isolating auxiliary files in `pykt-toolkit/tmp` folder. This way it will be easier to do contributions to the upstream repo from which we forked.

### Operational standards

- Training and evaluation should be launched using the commands described in `examples/reproducibility.md`
- Avoid launching commands that terminate scripts tha are running in the terminal
- Launch scripts in such a way that we leverage available GPUs (around 75% if not set otherwise) and CPUs (around 75% of CPU power)

### Code and Style Guidelines

- Code documentation should include usage examples
- Use markdown format for documentation files
- Use an academic professional tone, avoiding the use of emojis, icons, exclamations, informal language or marketing jargon. Use "we" instead of "you" following academic writing conventions.
- Ensure that all documentation is clear, concise, and accessible to a PhD-level audience.
- If you need to create new documentation files, create them in `./tmp`folder, unless I specifically ask for them.
- Follow guidelines in `reproducibility.md` to avoid hardcoded default values for parameters. Don't avoid audits.
- After changes in codebase, always check if parameters in `configs/parameter_default.json` were added or modified. If so, apply guidelines described in `examples/reproducibility.md` to propagate the changes in order to have proper reproducibility guarantees.
- When creating or updating plans don't include time estimations
- Only do commits when I ask for. In general, I prefer to commit after testing with experiments. Don't add nothing to the commit unless it is explicitly asked for.
- In general, try to avoid fallbacks. I prefer fail as early as possible, throwing exceptions, in case something doesn't match what is expected.

## Models architecture and implementation

- `assistant/newmodel.md`: points to the doc containing the description of current version of the model being developed and tested
- `paper/bkt.md`: Bayesian Knowledge Tracing (BKT) theory and principles

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
- The paper master is the `paper/latex/paper.tex` file, in Latex format.
- The folder `paper/` contains also other auxiliar files for the paper, such as the bibliography file `paper/latex/biblio.bib` and `paper/gtransformer.md` containing a detailed description of our approach, including the gtransformer model implemented in `pykt/models/gtransformer.py`, scripts to run experiments in `pykt/examples` and results of the experiments in `experiments` folder. **Most of the information in the latex paper is based on teh info in this file**.
- Use papers in `bibliography/` for theoretical alignment and get state-of-the-art knowledge about knowledge tracing and related topics. The file `paper/latex/biblio.bib` contains the bibliography that is referenced in other documents using `@` followed by the key of the entry in the biblio.bib file (in markdown documennts) or \citep{key} in LaTeX .tex documents.

#### 📊 For Experiments & Reproducibility

When you are running experiments (Experiment Agent):

- Follow `assistant/quickstart.pdf` guidelines.
- Ensure all default parameters are in `configs/parameter_default.json`.
- Use `configs/data_config.json` for datasets path and configuration.
- Strictly following the reproducibility protocol in `examples/reproducibility.md`.

## Copyright and License

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

This project and all its contents, including but not limited to source code, documentation, and data, are private and confidential.

**Strictly Prohibited:**

- Sharing, distributing, or disclosing any part of this project to third parties.
- Using any content from this repository to train, fine-tune, or otherwise improve any machine learning or artificial intelligence models.
