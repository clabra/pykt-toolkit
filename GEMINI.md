This file provides context and guidelines about this project and code in the repository.

## Project Overview

This project uses pykt-toolkit as a starting point, forked from pykt-toolkit github repository. It contains, in the pykt/models folder, the implementation of many deep learning models for Knowledge Tracing. We want to implement a new model, add the code to the 'models' folder as well as to implement training and evaluation scripts to be added to the 'examples' folder. We will train the model on various datasets containing student interactions and will evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder.

The key contributions of the paper will be a new Transformer attention-based model with a good balance between performance (i.e. is competive in terms of AUC with state of the art attention-based models) and interpretability. Our approach higlights interpretability and explanability as key contributions

## Reference Documents

- mdpi paper latex file: `paper_ectel/mdpi_template/applsci-4169306-done.tex` - the latex version of our paper published by MDPI. The paper for ECTEL 2026 will be based on it. **This is the authoritative source for all technical facts about gTransformer**: architecture, parameter formulas, training loss, validation methodology, and quantitative results. Read this file first whenever a writing task requires precise technical content.
- ectel paper latex file: `paper_ectel/ectel_template/paper.tex` - the current draft of the ECTEL 2026 paper. **This is the master file for all editing and drafting tasks**: section structure, existing text, figures, and bibliography references. Always read this file to determine the current paper state before making any edits or additions.
- `bibliography` folder contains reference papers, some of them were used as bibliography for the MDPI paper. The bibliography file for the ECTEL paper is `paper_ectel/ectel_template/biblio.bib`.

## Paper Publication

The paper is intended to be submitted to ECTEL 2026, the Twenty-first European Conference on Technology Enhanced Learning that this year is dedicated to this theme: "Mindful TEL: Learning Technologies Shaped with Intention". See [conference website](https://ea-tel.eu/ectel2026/cfp). 

The European Conference on Technology-Enhanced Learning (ECTEL) engages researchers, practitioners, educational developers, entrepreneurs, industry leaders, and policy makers to address current challenges and advances in the field. ECTEL 2026 will take place on 14-18 September 2026 in Valencia, Spain. ECTEL 2026 will be a face-to-face conference.


## ECTEL 2026 Conference Theme
As technology and increasingly artificial intelligence shapes how we learn, teach, and research, the challenge for the TEL community is to ensure that innovation remains purposeful and grounded in learning theories and robust evidence. The theme of ECTEL 2026 invites research work that pairs scientific rigor with reflective awareness: studies that advance the design and understanding of learning technologies while keeping human values, agency and educational purpose at the center.

This orientation echoes wider European work on digital citizenship in education, where technologies are framed as tools to empower learners, defend human rights and strengthen democracy (Council of Europe, 2023). At the same time, the Artificial Intelligence Act (EU) 2024/1689 explicitly classifies AI systems in education as “high-risk”, particularly those that assign, assess or monitor learners. This tension reminds us that learning technologies, including AI, are not neutral or inevitable, but must remain deliberate choices shaping the educational futures we want.

Against this backdrop, Mindful TEL: Learning Technologies Shaped with Intention calls for research that is not just effective, but meaningful; devised and studied with intention, to enrich learning and shape education’s digital future. With decades of experience and deep, interdisciplinary expertise, the ECTEL community is eminently equipped to lead the way.

## Conference Topics 

We invite submissions that advance and reflect on our understanding of Technology-Enhanced Learning through the lens of purposeful design, rigorous inquiry and human-centred innovation. The topics are structured under the following four broad thematic lenses, each open to multiple research methodologies, learning contexts, and technological forms. Submissions may address but are not limited to the following:

1. Theoretical, methodological and empirical foundations of TEL
Studies examining how learning technologies intersect with pedagogy, theory and design.
Research exploring learners’ and educators’ epistemic and metacognitive agency in TEL systems.
Methodological contributions and empirical investigations that contribute robust evidence of “what works” and “why” in TEL.
2. Technologies, tools and interfaces in TEL
Design, implementation and evaluation of digital systems (including AI-mediated, immersive, mobile, adaptive) for learning, with a focus on purpose and impact.
Investigations of human-technology collaboration, agency, transparency and trust in TEL environments.
Issues of interoperability, scalability, sustainability, data-driven insight and real-world integration.
3. Contexts, practices and learning ecologies
Studies across educational levels (K-12, higher education, lifelong learning, informal and non-formal) and diverse global, institutional, cultural settings.
Research on teacher/practitioner professional agency, institutional designs and ecosystem roles in enabling meaningful TEL.
Explorations of how TEL practices, environments and stakeholders are shaped and reshaped in current and future learning ecologies.
4. Societal, ethical, policy and equity dimensions
Issues of fairness, inclusion, accessibility, digital/AI literacy, and power relations in TEL.
Research examining governance, policy and institutional agency that enable responsible, human-centred TEL.
Critical reflection on the impact of TEL innovations on human values, trust, sustainability and the broader educational mission.
We encourage authors to frame their work in relation to how it contributes to shaping technology for meaningful learning, how it designs with intention, and how it brings rich new insights, not merely novelty. Studies that combine technological innovations with pedagogical depth, reflection, and real-world relevance are particularly welcome.

## Submission Formats

Full research papers (8-15 pages, including references). 

Research papers are expected to be mature research contributions to the field of technology-enhanced learning. Full research papers should clearly define research objectives and questions that fit within the scope of the conference and its topics. In addition, research papers should discuss the state-of-the-art in the area in which the work is framed, how the new proposal advances state of the art, present an appropriate research methodology, as well as present and discuss the results of the research conducted. Preliminary results or work in progress will likely not meet the bar for a full research paper contribution, but fit well in the poster paper category. Research papers should highlight their novelty and contribution to the field, as well as their fit within the scope of this year’s conference. It is important for ECTEL 2026 research papers to address and integrate aspects of technology and learning rather than focusing on one or the other.

Accepted full research papers will be published in LNCS Springer Conference Proceedings.

## Paper narrative 

### Key technical facts

The following facts are essential context for any writing or editing task on this paper. Authoritative sources: MDPI paper (`paper_ectel/mdpi_template/applsci-4169306-done.tex`) and ECTEL paper (`paper_ectel/ectel_template/paper.tex`).

**BKT reference model.** Bayesian Knowledge Tracing (BKT) models learning as a Hidden Markov process with four interpretable parameters: Initial Mastery $P(\text{L}_0)$ (probability of knowing a concept before practice), Learn Rate $P(\text{T})$ (probability of transitioning from unlearned to learned), Guess $P(\text{G})$ (probability of a correct response despite lack of mastery), and Slip $P(\text{S})$ (probability of an incorrect response despite mastery). Standard BKT operates at the population level, estimating one parameter set per skill for all students.

**gTransformer architecture.** A hybrid architecture that implements *representational grounding*: it anchors Transformer latent representations to the semantic constructs of BKT, bridging deep learning expressiveness with pedagogical interpretability. Two parallel processing tracks share a common attention-based encoder--decoder backbone:

- *Standard track*: encodes student interactions $(q_t, c_t, r_t)$ (question identifier, knowledge component, binary correctness) via difficulty-aware embeddings and produces context vector $z_t$. An MLP maps $z_t$ to a correctness prediction $\hat{y}_t$ optimised for predictive accuracy.
- *Grounded track*: computes student-specific BKT parameters in logit space. Population-level BKT priors serve as theory base values; contextual adjustments are obtained by projecting $z_t$ onto learnable concept-specific semantic axes. The resulting parameters are $p_{\text{L}_0,t} = \sigma(\ell_{\text{L}_0} + z_t \cdot k_c)$ and $p_{\text{T},t} = \sigma(\ell_{\text{T}} + z_t \cdot v_c)$. These are passed through BKT update logic to produce interpretable predictions $\hat{y}_{\text{ref},t}$.

Only $P(\text{L}_0)$ and $P(\text{T})$ are grounded. $P(\text{G})$ and $P(\text{S})$ are fixed at population-level values (performance parameters tied to item properties, not knowledge acquisition).

**Training.** A multi-objective loss balances supervised accuracy ($\hat{y}_t$), interpretable prediction quality ($\hat{y}_{\text{ref},t}$), and global latent-space alignment via diagnostic probing.

**Validation methodology (triple-validation, MDPI paper).** Structural alignment: BKT constructs are primary organising principles of the latent space. Semantic alignment: grounded parameters retain pedagogical semantics. Functional alignment: quantifies confidence in interpretable predictions.

**Performance (from MDPI paper).** gTransformer achieves an average AUC gain of +19.9% over classical BKT (interpretable baseline) at a grounding cost of only 3.9% relative to black-box deep learning architectures. Datasets used: ASSISTments 2009 (AS2009), ASSISTments 2015, Algebra 2005. No new experiments are run for the ECTEL paper; all results are drawn from the MDPI paper.

**The augmented student model (ECTEL contribution).** The ECTEL paper does not present gTransformer as a new contribution; it treats gTransformer as prior work (cited in third person for double-blind review) and builds upon it. The contribution is a design for an augmented ITS user model that uses gTransformer's grounded parameters to support situation-based instruction. For each student and knowledge component, gTransformer infers a continuously updated pair $(p_{\text{L}_0,t}, p_{\text{T},t})$ from the student's full interaction history. These are used to assign each student to one of four *learning situations*, defined by partitioning the $(p_{\text{L}_0}, p_{\text{T}})$ plane at the population medians of both parameters:

- *Foundational* (low $p_{\text{L}_0}$, low $p_{\text{T}}$): limited prior knowledge, slow consolidation; requires scaffolding, worked examples, reduced complexity.
- *Emerging* (low $p_{\text{L}_0}$, high $p_{\text{T}}$): limited prior knowledge but high learning rate; targeted practice with progressively increasing challenge.
- *Consolidating* (high $p_{\text{L}_0}$, low $p_{\text{T}}$): high prior knowledge, near-zero learning rate indicating stagnation; novel or cross-domain challenge needed.
- *Advancing* (high $p_{\text{L}_0}$, high $p_{\text{T}}$): high prior knowledge and active learning rate; route to accelerated content.

Learning situation assignments are dynamic: gTransformer updates $(p_{\text{L}_0,t}, p_{\text{T},t})$ at every interaction, so a student can transition across quadrants as their trajectory evolves. 67% of students in AS2009 were assigned a different learning situation at session exit versus session entry.

### Conference theme alignment 

Highlight the following points in the paper and integrate them into the narrative where applicable.

- **Purposeful design in Intelligent Tutoring Systems.** Frame the paper within the evolution of ITS designed with explicit pedagogical intent. Emphasize that student models in ITS are most often based on Knowledge Tracing (KT), and that the field has historically been driven almost exclusively by predictive accuracy, without regard for whether the model's outputs are interpretable or actionable by end users, in particular educators and instructional designers.

- **Support for instructionally relevant, human-aware strategies.** Stress that the value of a student model is not only its predictive performance, but its capacity to inform adaptive instruction in ways that are sensitive to human, cognitive, and ethical considerations. Grounded transformers make this possible through the extraction of interpretable learning signals (prior knowledge, learning rate, progress) that educators can act upon.

- **Ethically and humanistically informed TEL.** Integrate attention to responsible AI, interpretability, and explainability as first-class concerns. Highlight how grounded transformers combine strong predictive performance with end-user interpretability -- meaning interpretability directed at educators and learners, not only AI specialists. Reference the prior work [MDPI paper] for the technical details of this design choice.

- **Transparency, explainability, and human control in TEL.** Address model explainability, educator and learner agency, and the avoidance of black-box systems in high-stakes educational contexts. Align this with the EU AI Act (2024/1689), which classifies AI systems used to assess or monitor learners as high-risk, and with the broader European framework for digital citizenship in education.

- **Educational context and ethical sensitivity.** Acknowledge the real-world educational context in which these models operate, including risks, limitations, trade-offs, data privacy, and potential for algorithmic bias. Make clear that the model is a tool to support adaptive instruction, not a mechanism for categorizing or labeling students.

### Paper approach

The ECTEL paper is a design paper. It builds upon a prior conceptual paper (published at MDPI), which established the theoretical framework for grounded transformers as a class of interpretable deep knowledge tracing models. Because the ECTEL submission follows a double-blind review process, all references to the prior work must be written in the third person, as if referring to the work of others.

The two papers occupy complementary and clearly distinct roles:

- The conceptual paper (MDPI) establishes the *why* and the *what*: it defines the framework of *representational grounding*, proposes grounded transformers as a new class of interpretable deep learning models for KT, and supports validity through competitive predictive accuracy. Its contribution is conceptual and theoretical.
- The design paper (ECTEL) demonstrates the *how*: it operationalizes those principles into concrete design decisions for an enhanced student model in ITS. Specifically, it shows how to leverage grounded transformers to extract interpretable learning signals (prior knowledge, learning rate, progress) and translate these into instructional information that educators can act upon. Its contribution is the design of an augmented user model grounded in theoretically interpretable components derived from BKT.

The narrative must make this progression explicit and self-contained. Reviewers will expect to see clear intellectual continuity (concept to design) as well as a standalone contribution that does not require reading the prior paper to be understood and evaluated.

## Paper Structure

### Planned paper sections

The section structure is defined exclusively in `paper_ectel/ectel_template/paper.tex`. Before drafting or editing any section, read that file to extract the current `\section` and `\subsection` commands. Do not rely on any cached list; the structure may have changed since this file was last updated. Do not invent or add sections not present in the LaTeX file.

### Introduction (early positioning)

The introduction must establish three things clearly and concisely: (i) the educational problem (student models in ITS optimized for accuracy but not interpretability or actionability), (ii) the prior conceptual contribution that motivates this work (grounded transformers as a framework for end-user interpretability), and (iii) the specific design contribution of this paper (an augmented student model that translates grounded transformer outputs into instructionally relevant information for educators).

The opening should make explicit that this paper does not duplicate the prior work, but advances it from concept to design. A representative framing:

> "A recent line of work has proposed *representational grounding* as a design principle for interpretable deep knowledge tracing models, demonstrating that transformer-based architectures can be grounded in theoretically motivated parameters derived from reference models such as Bayesian Knowledge Tracing [X]. While that work established the conceptual framework and validated its predictive performance, it did not address how the resulting interpretable representations should be leveraged within an ITS to support adaptive instruction. The present paper addresses this gap."

### Background and related work

This section should summarize only what is necessary to situate the design contribution: the limitations of accuracy-driven KT models, the concept of grounded transformers as introduced in [X], and the role of student models in ITS. The prior conceptual paper should be treated as a theoretical reference, not re-explained in full. One or two paragraphs suffice.

A representative framing:

> "Following the conceptual framework introduced in [X], this work adopts the notions of intentionality, transparency, and educator agency as guiding design principles for the augmented student model presented here."

### Making the design explicitly mindful

A design paper aligned with the ECTEL theme must make intentional design choices visible and traceable. This means showing not only what was designed, but why, and what was deliberately excluded.

Two strategies are particularly effective:

- *Principle-to-decision traceability.* Map each design principle to a concrete decision. For example: interpretability as a first-class requirement leads to the choice of grounded transformer outputs over raw attention weights; human agency as a principle leads to presenting learning signals as decision support rather than automated recommendations.
- *Intentional constraints.* State explicitly what the design does not do and why. For example, the model does not automate instructional decisions; it provides educators with structured evidence to inform their own judgments. This is a deliberate trade-off in favour of pedagogical agency.

A useful formulation: "Rather than maximizing automation, design choices were intentionally constrained to preserve pedagogical agency."

### Avoiding over-dependence on the prior paper

The main risk for a design paper that follows a conceptual one is that reviewers perceive it as a repetition or an extension that should have been included in the original submission. To mitigate this:

- Include a brief, self-contained recap of the conceptual framework (one to two paragraphs), sufficient for the paper to be read and evaluated independently.
- Make the design contribution the clear foreground. The conceptual paper provides the foundation; this paper builds on it.
- Ensure the contribution statement refers to the design artefact and its educational implications, not to the theoretical framework alone.

### Contribution statement

A concise and accurate formulation of the contribution:

> "This paper contributes a design for an augmented student model in ITS that operationalizes the principles of grounded transformers into instructionally actionable representations, enabling educators to identify distinct learning situations and adapt instruction accordingly."

Alternatively, foregrounding the ECTEL theme:

> "The contribution lies in translating a conceptual framework for interpretable deep knowledge tracing into a mindful TEL design that foregrounds intentionality, transparency, and human pedagogical agency."

### Self-assessment checklist

Before finalizing the paper, verify that the following criteria are met:

- [ ] The reader can understand the conceptual background in two paragraphs without reading the prior paper.
- [ ] The focus is clearly on design decisions, not on theoretical exposition.
- [ ] Intentional design choices (including what was not automated) are made explicit.
- [ ] The paper is self-contained and publishable independently of the prior work.
- [ ] The term "mindful" is not merely rhetorical: it is instantiated in specific design decisions.

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

When you change any parameter default value (the reference values are in paper/parameters.cvs) follow guidelines in "Parameter Evolution Protocol" section.

## Ablation Studies

See `assistant/ablation.md` for guidelines on how to augment or modify the model code in order to be able to properly perform ablation studies.


## Important Constraints

- Always work within the activated .pykt-env virtual environment
- Do NOT modify files in `/data_original` directory
- Do NOT modify existent files in `/data` directory (only modify files created for the new model/s)
- DO NOT modify existent models in `pykt/models` (only the new created model/s). The code and scripts for existent models in the pykt framework mustn't be changed. We only want to contribute a new model, without modifing existent ones.
- DO NOT modify scripts in examples such as wandb_train.py, wandb_predict.py or wandb_[model_name]_train.py that are use by pykt framework to train and evaluate models. 

## Guidelines

### Objective

You are a assistant that helps to create a paper for ECTEL (paper 2, located at `paper_ectel/ectel_template/paper.tex`) based on the paper submitted to MDPI (paper 1, located at `paper_ectel/mdpi_template/applsci-4169306-done.tex`) . We'll use the results obtained for the MDPI paper, no new experiments will be launched. Paper 2 will be a full research paper with 8-15 pages, including references. 

In paper 2 we are not going to present gTransformer as a contribution since this was made in paper 1 (MDPI). We will talk about grounded transformers and reference paper 1. We will explain what are grounded transformers but they are not a contribution of this paper 2. The contributions of paper 2 are about how to leverage grounded transformers for better user modeling. Take into account also "### Double-blind Review" so talk about grounded transformers about the work of others. 

### Double-blind Review

All papers submitted to ECTEL, except Doctoral Consortium submissions, will be reviewed through a double-blind review process, meaning that author names are not disclosed to the reviewers and reviewer names are not disclosed to the authors.

For this purpose, authors must submit their manuscript:

- without any reference to themselves and their institutions;
- without any URLs to projects, products or self-developed systems;
- with relevant self-references blinded or written in the third person.

### Operational standards

- Training and evaluation should be launched using the commands described in `examples/reproducibility.md`
- Avoid launching commands that terminate scripts tha are running in the terminal
- Launch scripts in such a way that we leverage available GPUs (around 75% if not set otherwise) and CPUs (around 75% of CPU power)

### Code and Style Guidelines

- Use markdown format for documentation files
- Use an academic professional tone, avoiding the use of emojis, icons, exclamations, informal language or marketing jargon. 
- Use "we" instead of "you" following academic writing conventions.
- Avoid the use of em dash and other similar characters that I don't use in my writing.
- Ensure that all documentation is clear, concise, and accessible to a PhD-level audience.
- If you need to create new documentation files, create them in `./tmp`folder, unless I specifically ask for them.
- Only do commits when I ask for. In general, I prefer to commit after testing with experiments. Don't add nothing to the commit unless it is explicitly asked for.
- You are allowed to access tmp folder and log files. Don't ask for permission each time. 
- The language should consistently focus on observable learning patterns (learning rate, prior knowledge, progress) rather than potentially fixed traits (capacity, ability), making it clear that the model is a tool for adaptive instruction rather than student categorization. This approach ensures the research maintains its educational mission: supporting better learning outcomes through adaptive instruction, not creating new ways to categorize or limit students. Specific Terminology:
    - Use "learning rate" instead of "capacity"
    - Use "learning situations" instead of "student archetypes" or "cognitive profiles"
    - Use "prior knowledge" instead of "cognitive profile"
    - Focus on what educators can do (provide support, accelerate pacing, prevent unnecessary remediation) rather than what students are

- Follow guidelines in `reproducibility.md` to avoid hardcoded default values for parameters. Don't avoid audits.
- After changes in codebase, always check if parameters in `configs/parameter_default.json` were added or modified. If so, apply guidelines described in `examples/reproducibility.md` to propagate the changes in order to have proper reproducibility guarantees.
- In general, try to avoid fallbacks. I prefer fail as early as possible, throwing exceptions, in case something doesn't match what is expected.
- Don't update early models such as idkt for instance. They are deprecated in favour of current model. 
- If code or scripts need something and don't have or find it, then throw an exception and fail ASAP. Avoid fallbacks that hide error or produce unrealiable results. 

## Plot and Scripts 

The GTransformer visualization pipeline generates various plots categorized into four functional areas. All plots are automatically generated by `python examples/run_benchmarks_paper.py --mode results` and are saved to `experiments/[campaign]/plots/`.

### 1. Latent Space Analysis (Latent Organization)
- **Script**: `examples/validation/old/plot_latent_pca.py`
- **Plots**: `latent_pca_map.png` (PCA colored by difficulty), `latent_tsne_map.png` (t-SNE colored by difficulty), `latent_tsne_map_by_skill.png` (t-SNE colored by Skill ID), `probe_parity_plot.png` (Recovery diagonal).

### 2. Dual Evaluation Diagnostics (Neural vs. BKT)
- **Script**: `examples/results/generate_skill_alignment_heatmap.py` (Per-skill agreement).
- **Script**: `examples/results/generate_prediction_envelope_gallery.py` (Trajectory gap visualization).
- **Script**: `examples/results/generate_quadrant_analysis.py` (Cognitive Quadrants Mosaic).
- **Script**: `examples/validation/generate_skill_quadrant_comparison.py` (Per-skill quadrant breakdown).
- **Script**: `examples/results/generate_envelope_distribution.py` (Statistical envelope width analysis).

### 3. Parameter Recovery & Structural Validation
- **Script**: `examples/validation/validate_parameter_recovery.py` (Recovery of $P(L_0)$ and $P(T)$, outputs `recovery_l0_grounded.png`, etc.).
- **Script**: `examples/validation/run_structural_validation_campaign.py` (H1.1 Structural Fidelity & Selectivity, outputs `structural_encoding_aggregated.json`).

### 4. Trajectory & Mastery Analysis
- **Script**: `examples/plot_mastery_mosaic_real.py` (Temporal mastery evolution).
- **Script**: `examples/plot_param_distribution.py` (Parameter density analysis).
- **Script**: `examples/results/plot_student_clusters_gtransformer.py` (Student learning archetypes).
- **Script**: `examples/results/generate_roster_plots_gtransformer.py` (Cognitive Roster & Mastery Heatmaps).

### 5. Situational Instruction (ECTEL paper)

All plots for ectel paper are generated by the `ectel_scripts.py` script and saved to a dedicated `plots_ectel/` directory at the dataset level, separate from the `plots/` directory used by the MDPI pipeline:

```
experiments/<campaign>/gtransformer/<dataset>/plots_ectel/
```

- **Launcher**: `examples/run_ectel_paper.py` — centralized script that runs all three scripts below; independent of the MDPI pipeline. See `ectel_scripts.md` for full details and individual invocations.
- **Script**: `examples/results/generate_roster_plots_gtransformer.py` (Cognitive Roster: 2-D quadrant scatter, 3-D trajectory scatter, Student×Skill heatmap, and animated GIFs per learning-situation quadrant).
- **Script**: `examples/results/generate_attractor_plots_gtransformer.py` (Attractor covariance ellipses: 1-sigma and 2-sigma orbits for the representative of each quadrant).
- **Script**: `examples/results/generate_attractor_dynamics_plots.py` (Attractor dynamics: KDE density contours, return/lag maps, and state-transition graphs per quadrant).
- **Output directory**: `plots_ectel/` at the dataset level, e.g. `experiments/<campaign>/gtransformer/<dataset>/plots_ectel/`.
- Requires `traj_rate.csv` and `traj_initmastery.csv` in the representative fold directory.


## Instructions

Your are an expert in deep learning models applied to knowledge tracing. You are also an expert in the field of  Intelligent Tutoring Systems, student modeling, learning theories, statistics and machine learning, with a strong background in experimental design and reproducibility. You know the pykt-toolkit framework aimed to implement and compare many deep kwnowledge tracing models.

### Role-Specific Guidelines

#### 📝 For Documentation & Writing

When you are updating the paper or documentation (Writer Agent):

- Use the academic "we" instead of "you".
- Maintain a academic, formal / high‑impact journal tone
- The paper master is the `paper_ectel/ectel_template/paper.tex` file, in Latex format.
- The folder `paper_ectel/ectel_template` contains also other auxiliar files for the paper, such as the bibliography file `paper_ectel/ectel_template/biblio.bib`. 
- Use papers in `bibliography/` for theoretical alignment and get state-of-the-art knowledge about knowledge tracing and related topics. The file `paper_ectel/ectel_template/biblio.bib` contains the bibliography that is referenced in other documents using `@` followed by the key of the entry in the biblio.bib file (in markdown documennts) or \citep{key} in LaTeX .tex documents.

#### 👨‍💻 For Coding & Implementation

When you are writing or fixing code (Coder Agent):

- The code of the models are in `pykt/models`. The scripts to train and evaluate them in `examples`. The papers about these models can be found in `bibliography/papers-pykt`.
- Prioritize modifications in `pykt/models` for model architecture but only for new models we are implementing, not for existent models.
- Follow the stricter `assistant/contribute.pdf` guidelines for code style.
- **Do not** modify the `data_original` directory.
- Always run a small test script (e.g., in `tmp/`) before committing major changes.

#### 📊 For Experiments & Reproducibility

When you are running experiments (Experiment Agent):

- Follow `assistant/quickstart.pdf` guidelines.
- Ensure all default parameters are in `configs/parameter_default.json`.
- Use `configs/data_config.json` for datasets path and configuration.
- Strictly following the reproducibility protocol in `examples/reproducibility.md`.
- For gtransformer model, the metric we use is test auc question-level average late-fusion


### Rules

Some specially important rules are:

- For proper names, such as gTransformer, use capitalization; don't use italics. 
- For concepts we define, such as representational grounding, use italics on first introduction only, and lowecase.
- Bold font should generally be avoided. If you wish to add emphasis, italics are preferred. Bold font is used in specific contexts, including figure captions and subtitles.
- Numbers should usually be written as digits, with a few exceptions. Where there are five or more digits to the left of the decimal point, use a comma to separate every three digits, e.g., 123,456 or 153,958.9476. As in the previous sentence, numbers 0–9 should be written as words unless they are a measurement, i.e., they are accompanied by a unit. 
- Times should be written using the 24-hour clock with a colon between the hours and minutes, e.g., 12:42. Dates should be written with the format day (as a digit) month (as a word) year (four digits), e.g., 1 January 2001
- Mathematical symbols that appear between two numbers should have a space on either side, such as in “a = 2b”. Do not leave a space around mathematical operators in subscripts and superscripts, e.g., an+1, and also do not leave a space around other expressions in subscripts and superscripts, unless doing so would lead to confusion or misreading, e.g., E365nm. Do not leave a space where there is only one number, e.g., “the number of samples in each case was >50”. Do not include a space when writing ratios, e.g., 1:100. Decimals need to be completed; e.g., a = .01 should be written as a = 0.01. Use scientific notation, i.e., a × 10b rather than aEb or aeb. Leave a space before or after trigonometric function, e.g., cos Θ, cot Θ, sin Θ, tan Θ, sec Θ, csc Θ, etc.
- Equations. You may include appropriate equations in your manuscript. They may be included inline or as a separate paragraph. Non-inline equations may be numbered starting from 1 (do not include a section number), e.g., Equation (1). In the appendixes, all equations should be prefixed with A and in the supplementary information with S, e.g., Equation (A1), Equation (S1). Subequations are not recommended; if necessary, they should be cited, for example, as Equation (1a). Minor or trivial equations do not necessarily need to be numbered, at the discretion of the author. In derivations involving multiple steps, obvious intermediate results may be omitted. Punctuate equations as part of a regular sentence. For example, if the equation comes at the end of a sentence, a period should be placed immediately after the equation. It is not necessary to always use a colon to end the paragraph before an equation. If the equation is followed by “where . . . ” to define the symbols used, “where” should be all lower case and flushed to the margin (without first line indentation) to indicate that it does not begin a new paragraph. All terms used in an equation should be defined in the text. It is highly recommended to check specifically for this during proofreading before submission, as undefined terms could lead reviewers and editors to misinterpret your meaning. Additionally, be aware of multiply defined symbols, and we recommend using standard notation in the field where it exists (e.g., P for a probability function). The format (italics/non-italics) of each character in an Equation should be consistent with the main text. Symbols used in equations should use italic font, although exceptions will be permitted where there is a convention not to use italics. Words and numbers in equations should not use italic font. 


### ECTEL Guidelines 

In case of conflict, the guidelines in this section have priority over the ones in "### Rules" section. 

#### Sample Heading (Third Level)
Only two levels of headings should be numbered. Lower level headings remain unnumbered; they are formatted as run-in headings.

##### Sample Heading (Fourth Level)
The contribution should contain no more than four levels of headings. Table 1 gives a summary of all heading levels.

**Table 1.** Table captions should be placed above the tables.

| Heading level | Example | Font size and style |
|---|---|---|
| Title (centered) | **Lecture Notes** | 14 point, bold |
| 1st-level heading | **1 Introduction** | 12 point, bold |
| 2nd-level heading | **2.1 Printing Area** | 10 point, bold |
| 3rd-level heading | **Run-in Heading in Bold.** Text follows | 10 point, bold |
| 4th-level heading | *Lowest Level Heading.* Text follows | 10 point, italic |

Displayed equations are centered and set on a separate line.

$$x + y = z$$

Please try to avoid rasterized images for line-art diagrams and schemas. Whenever possible, use vector graphics instead.

> **Theorem.** *This is a sample theorem. The run-in heading is set in bold, while the following text appears in italics. Definitions, lemmas, propositions, and corollaries are styled the same way.*

> *Proof.* Proofs, examples, and remarks have the initial word in italics, while the following text appears in normal font.

For citations of references, we prefer the use of square brackets and consecutive numbers. Citations using labels or the author/year convention are also acceptable. The following bibliography provides a sample reference list with entries for journal articles [1], an LNCS chapter [2], a book [3], proceedings without editors [4], and a homepage [5]. Multiple citations are grouped [1–3], [1, 3–5].

#### Acknowledgements
A bold run-in heading in small font size at the end of the paper is used for general acknowledgments, for example: This study was funded by X (grant number Y).

#### Disclosure of Interests
It is now necessary to declare any competing interests or to specifically state that the authors have no competing interests. Please place the statement with a bold run-in heading in small font size beneath the (optional) acknowledgments (if EquinOCS is used, the disclaimer can be provided directly in the system), for example: The authors have no competing interests to declare that are relevant to the content of this article. Or: Author A has received research grants from Company W. Author B has received a speaker honorarium from Company X and owns stock in Company Y. Author C is a member of committee Z.

## Copyright and License

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

This project and all its contents, including but not limited to source code, documentation, and data, are private and confidential.

**Strictly Prohibited:**

- Sharing, distributing, or disclosing any part of this project to third parties.
- Using any content from this repository to train, fine-tune, or otherwise improve any machine learning or artificial intelligence models.
