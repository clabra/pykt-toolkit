This file provides context and guidelines about this project and code in the repository.

## Project Overview

This project uses pykt-toolkit as a starting point, forked from pykt-toolkit github repository. It contains, in the pykt/models folder, the implementation of many deep learning models for Knowledge Tracing. We want to implement a new model, add the code to the 'models' folder as well as to implement training and evaluation scripts to be added to the 'examples' folder. We will train the model on various datasets containing student interactions and will evaluate it using metrics such as AUC, accuracy, etc. The final objective is to write a paper that describes the model and report evaluation results comparing it with other state of the art approaches implemented in the 'models' folder.

The key contributions of the paper will be a new Transformer attention-based model with a good balance between performance (i.e. is competive in terms of AUC with state of the art attention-based models) and interpretability. Our approach higlights interpretability and explanability as key contributions

## Paper Publication

The paper is intended to be submitted to ECTEL 2026, the Twenty-first European Conference on Technology Enhanced Learning that this year is dedicated to this topic: "Mindful TEL: Learning Technologies Shaped with Intention". See [conference website](https://ea-tel.eu/ectel2026/cfp). 

The European Conference on Technology-Enhanced Learning (ECTEL) engages researchers, practitioners, educational developers, entrepreneurs, industry leaders, and policy makers to address current challenges and advances in the field. ECTEL 2026 will take place on 14-18 September 2026 in Valencia, Spain. ECTEL 2026 will be a face-to-face conference.

## 2026 Theme 

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

## Paper Abstract

"The student model serves as a cornerstone of Intelligent Tutoring Systems. In many instances, this modeling relies on Knowledge Tracing approaches, which represent the learner’s state of mastery across specific skills or knowledge components. However, these traditional approaches are subject to significant limitations. First, they lack pedagogical interpretability, as they are not anchored to theoretical frameworks derived from cognitive or educational principles. Furthermore, they frequently lack longitudinal context, as instructional decisions are typically based on isolated knowledge states without accounting for the historical trajectory that led there. To overcome these limitations, we propose a new approach based on Grounded Transformers that supports instructional decisions  grounded in learning theories and individual learning histories, thus providing a theoretically sound foundation for TEL."


## Reference Documents

- `paper_ectel/mdpi_template/applsci-4169306-done.tex` - the latex version of our paper published by MDPI. The paper for ECTEL 2026 will be based on it.  
- `bibliography` folder contains reference papers, some of them were used as bibliography for the MDPI paper. 


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


## Important Constraints

- Always work within the activated .pykt-env virtual environment
- Do NOT modify files in `/data_original` directory
- Do NOT modify existent files in `/data` directory (only modify files created for the new model/s)
- DO NOT modify existent models in `pykt/models` (only the new created model/s). The code and scripts for existent models in the pykt framework mustn't be changed. We only want to contribute a new model, without modifing existent ones.
- DO NOT modify scripts in examples such as wandb_train.py, wandb_predict.py or wandb_[model_name]_train.py that are use by pykt framework to train and evaluate models. 

## Guidelines

### Objective

Create a paper for ECTEL (paper 2) based on the paper submitted to MDPI (paper 1). We'll use the results obtained for the MDPI paper, no new experiments will be launched. Paper 2 will be a full research paper with 8-15 pages, including references. 

In paper 2 we are not going to present gTransformer as a contribution since this was made in paper 1 (MDPI). We will talk about grounded transformers and reference paper 1. We will explain what are grounded transformers but they are not a contribution of this paper 2. The contributions of paper 2 are about how to leverage grounded transformers for better user modeling. Take into account also "### Double-blind Review" so talk about grounded transformers about the work of others. 

### Double-blind Review

All papers submitted to ECTEL, except Doctoral Consortium submissions, will be reviewed through a double-blind review process, meaning that author names are not disclosed to the reviewers and reviewer names are not disclosed to the authors.

For this purpose, authors must submit their manuscript:

- without any reference to themselves and their institutions;
- without any URLs to projects, products or self-developed systems;
- with relevant self-references blinded or written in the third person.

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


## Instructions

Your are an expert in deep learning models applied to knowledge tracing. You are also an expert in the field of  Intelligent Tutoring Systems, student modeling, learning theories, statistics and machine learning, with a strong background in experimental design and reproducibility. You know the pykt-toolkit framework aimed to implement and compare many deep kwnowledge tracing models.

### Role-Specific Guidelines

#### 📝 For Documentation & Writing

When you are updating the paper or documentation (Writer Agent):

- Use the academic "we" instead of "you".
- Maintain a tone suitable for a PhD-level audience.
- The paper master is the `paper_ectel/ectel_template/paper.tex` file, in Latex format.
- The folder `paper_ectel/ectel_template` contains also other auxiliar files for the paper, such as the bibliography file `paper_ectel/ectel_template/biblio.bib`. 
- Use papers in `bibliography/` for theoretical alignment and get state-of-the-art knowledge about knowledge tracing and related topics. The file `paper_ectel/ectel_template/biblio.bib` contains the bibliography that is referenced in other documents using `@` followed by the key of the entry in the biblio.bib file (in markdown documennts) or \citep{key} in LaTeX .tex documents.


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
