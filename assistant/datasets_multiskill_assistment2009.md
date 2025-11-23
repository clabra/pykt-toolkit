# Single and Multi Skills datasets

## Single-Skill Datasets

These datasets, in their raw form, primarily or exclusively contain problems that are associated with a single KC. This makes them suitable for direct use with single-skill KT models without the need for the "repeat format" preprocessing.

- ASSISTments 2015: This dataset is characterized by its "Skill Builder" design, where each problem is intentionally designed to target only one skill, making it a true single-skill benchmark.

- Algebra 2005: From a cognitive tutor, this dataset is structured so that each problem is linked to a unique KC.

- Bridge to Algebra 2006 (Bridge2006): Similar to Algebra 2005, this dataset from the same source is single-skill in nature.

- Ednet: A large-scale dataset used for TOEIC preparation, it primarily focuses on individual skills within a hierarchical structure, classifying it as a single-skill dataset.

- POJ (Programming on Judge): This dataset contains programming problems that are tagged with specific, single concepts or algorithms, making it a single-skill dataset.

ASSISTments 2015 is a single-skill dataset. This means that each problem is labeled with only one Knowledge Component (KC) or skill. The ASSISTments 2015 dataset, specifically the "Skill Builder" data, is structured to support single-skill practice. The "Skill Builder" assignments are designed for students to practice problems that correspond to a single, specific skill. This design ensures that each interaction record (a student's attempt at a problem) can be uniquely mapped to one KC, simplifying the task for knowledge tracing models. For models implemented in frameworks like pyKT-toolkit, the single-skill nature of ASSISTments 2015 is a crucial characteristic. Many traditional and deep knowledge tracing models, including those with attention mechanisms, are primarily designed for single-skill scenarios. They model the student's mastery of a set of distinct skills over time. Datasets like ASSISTments 2015 provide a clean and widely used benchmark for evaluating these models.

Single-Skill Focus: Models like DKT, DKVMN, and SAKT, which are available in pyKT, treat each interaction as a tuple of (skill, correctness). The single-skill nature of the dataset aligns perfectly with this input format, where the model only needs to consider one KC at a time.

Contrasting with Multi-Skill Datasets: In contrast, multi-skill datasets, such as the Statics dataset, require more complex modeling approaches. For these, a single question might involve multiple skills. In the ASSISTments system, when a problem is linked to multiple skills, a common pre-processing step, as noted in documentation for other ASSISTments datasets, is to duplicate the interaction record for each skill, effectively transforming it into a single-skill format for model input.

The ASSISTments 2009-2010 dataset is generally considered a single-skill dataset, similar to the 2015 version. While the original data might have some problems tagged with multiple skills, the widely used "Skill Builder" subset and standard preprocessing procedures, including those within pyKT, format it as a single-skill dataset by duplicating interaction records for each skill.

Conceptual Distinction
Intrinsic Nature (Multi-Skill): The original, raw data from an educational platform like ASSISTments can be multi-skill. A single question (e.g., a word problem) might require a student to apply several different cognitive skills to solve it. In this raw form, the student's single response is associated with a set of skills.

Modeling Paradigm (Single-Skill): Many foundational and widely-used deep knowledge tracing models, such as DKT, DKVMN, and SAKT, are built on the assumption that a single interaction corresponds to a single skill. Their architectural design, whether it's a recurrent neural network or an attention mechanism, is often optimized to process a sequence of (skill, correctness) pairs.

The Role of Data Preprocessing
The pyKT-toolkit, along with most other knowledge tracing research, adopts the repeat format as a standardized preprocessing step. The goal is not to deny the multi-skill nature of the original problem but to transform the data so that it can be used by models designed for a single-skill input.

Let's illustrate with an example:
Suppose a student answers a question q1 that is associated with two skills, k1 and k2, and the student answers it correctly.
In the raw data, this might be a single record: (student_id, q1, {k1, k2}, correct).

Using the repeat format, this single record is expanded into two separate records for the purpose of input to the model:

(student_id, q1, k1, correct)

(student_id, q1, k2, correct)

This transformation effectively creates two "opportunities" for skill practice from a single student action. When the model receives this sequence, it processes it as a single-skill sequence, but the historical context includes these duplicated records.

This preprocessing strategy allows for a fair comparison of models across different datasets and simplifies the implementation for many deep learning architectures. It's a pragmatic solution that enables single-skill models to be applied to a wider range of datasets, even those with multi-skill properties.

So, to summarize your point: the ASSISTments 2009-2010 dataset is inherently multi-skill in its original form, but due to the widespread adoption of the "repeat format" for preprocessing, it is treated as a single-skill dataset for most knowledge tracing experiments.

## Multi-Skill Datasets

These datasets intrinsically contain problems that are linked to more than one knowledge component (KC). This is a crucial distinction as it reflects the reality of many educational assessments, where a single question can measure multiple skills.

- ASSISTments 2009-2010: This dataset is confirmed to have multi-skill problems, and the "skill builder" format is a pre-processing step to adapt it for single-skill models. The raw data logs indicate that problems can be associated with multiple skills.

- ASSISTments 2017: This dataset is also known to contain problems that map to multiple skills.

- Statics 2011: A well-known example of a multi-skill dataset from a college-level physics course. Each problem on this assessment is explicitly tagged with multiple skills.

- NIPS34: This dataset, which is a version of the ASSISTments data, is also considered multi-skill due to the nature of the original problems.

# Evaluation Protocol

With multi-skill datasets we need to consider approaches for Fusion Strategies and Testing Data Formats. See "Testing Data Format" section of conribute.pdf and "1.3 Evaluating Your Mode" of readthedocs.pdf for more info. 

### Multi-KC Question Evaluation

Questions in educational datasets often relate to multiple Knowledge Components (KCs). To ensure pyKT's evaluation aligns with real-world prediction scenarios, models are trained at the KC level but evaluated at the question level. This approach addresses the fundamental challenge that student performance on a question depends on mastery of all its associated KCs.

### Fusion Strategies

When a question involves multiple KCs, pyKT employs two main fusion approaches to obtain question-level predictions:

#### Early Fusion
Averages the hidden states at the KC level before feeding them into the prediction layer. For example, if question q₃ relates to KCs k₅ and k₆, the hidden states h₅ and h₆ are averaged before prediction, producing a single question-level prediction p₃.

#### Late Fusion
Generates KC-level predictions first, then combines them using three strategies:
- **Mean**: Averages all KC-level prediction probabilities as the final question-level prediction
- **Vote**: Uses the median value of KC predictions as the final prediction (robust to outliers)
- **All**: Requires all KC predictions to exceed the threshold for a positive question-level prediction

### Testing Data Formats

Three Testing Data Formats in pyKT

1. Repeat Format

- What it does: Expands questions to KC level by repeating responses for each KC
- Example: Question with KCs {k₁,k₃} and response r₁ becomes: [k₁,r₁], [k₃,r₁]
- Used for: Maintaining consistency with training data format
- Current status: This is what train_valid.csv uses

2. Question Level Format

- What it does: Creates separate KC-response subsequences for each KC in a question
- Example: For question q₃ with KCs {k₃,k₄}, creates two separate evaluation sequences
- Used for: Granular evaluation of each KC's contribution to question prediction
- Advantage: Better reflects real-world scenarios where KCs may have different mastery levels

3. Window Format

- What it does: Applies sliding window to limit sequence length to M interactions
- Example: If M=200 and sequence has 201 interactions, keeps the most recent 200
- Used for: Managing very long sequences that exceed model capacity


train_valid.csv, for instance, uses the repeat format. This measn that: 
  - Re-exposure patterns might be inflated because the same question appears multiple times through KC expansion
  - The "3 correct in a row" mastery criterion becomes more complex when the same question response is counted multiple times
  - SimAKT's similarity-based attention needs to distinguish between actual skill practice vs. repeated question exposure. This understanding is crucial for designing the SimAKT model's attention mechanisms and evaluation strategy.
  
# Sparsity 

## Data Sparsity

The finding of an average of 6.5 skills per student in the ASSISTments 2015 dataset seems quite low, although it is not an error but an expected characteristic of the data. The low average number of skills per student is a direct result of the dataset's origin: the ASSISTments Skill Builder system. This system is not a comprehensive curriculum but rather a supplementary tool for targeted practice and mastery.

Targeted Practice: The Skill Builder feature is designed for students to practice a specific skill until they achieve a certain level of mastery (e.g., getting three problems in a row correct). A teacher would typically assign a Skill Builder on a particular topic to reinforce learning from a recent class lesson or to address a specific weakness.

This dataset is single-skill in nature, meaning each problem is tied to only one skill. Students do not typically work on a wide variety of skills in a single session. Instead, their activity is concentrated on a small set of skills assigned by their teacher.

Data Sparsity: This concentrated, single-skill usage leads to data sparsity. The student-skill interaction matrix is very sparse, with most students interacting with only a small fraction of the total number of skills in the dataset. This sparsity is a significant challenge for knowledge tracing models, as they must predict future performance with limited historical data for each student.

Implications for Knowledge Tracing
The low skill count per student is a key feature that makes the ASSISTments 2015 dataset a valuable, though challenging, benchmark. It simulates a realistic scenario in which a student's educational history, as captured by a single platform, is incomplete and focused.

Model Evaluation: Knowledge Tracing models, particularly those with attention mechanisms, are evaluated on their ability to make accurate predictions despite this data sparsity. The goal is to see how well a model can generalize from a student's limited history on a few skills to their future performance.

Contextualization: The limited number of skills per student underscores that the dataset does not represent a student's entire academic life. It is a snapshot of their interactions with a specific tutoring system for a limited duration, not a complete record of their learning.

This characteristic is a fundamental aspect of the dataset and not a data collection or processing error. It is a reflection of how the ASSISTments platform is used in real-world educational 

## General References for Knowledge Tracing Datasets and Models

The data sparsity explanations are based on the common understanding and preprocessing practices described in foundational papers and resources, which form the basis for implementations like those in the pyKT-toolkit framework. 

pyKT-toolkit: For implementation details and model architecture, the primary reference is the pyKT-toolkit repository on GitHub and its official documentation. The framework is described in the paper:

Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., & Luo, W. (2022). PYKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models. arXiv preprint arXiv:2206.11460.

References for ASSISTments Datasets
The ASSISTments platform has been a cornerstone of educational data mining research, with different datasets released over time, each with its own characteristics.

ASSISTments 2009-2010: This dataset is widely recognized and used as a benchmark. The original release and a detailed description of the data can be found on the ASSISTmentsData website. A key paper that describes the data is:

Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). An intelligent tutoring system for learning how to solve word problems. In Proceedings of the 12th International Conference on Artificial Intelligence in Education.

For the specific "skill builder" and multi-skill nature, the description of the data is provided in the ASSISTmentsData 2009-2010 documentation, which notes that for the skill builder dataset, multi-skill problems are duplicated to create single-skill opportunities.

ASSISTments 2015: This dataset is a curated collection of data from multiple randomized controlled experiments. It's known for its clean, single-skill structure.

Botelho, A., Heffernan, N., Heffernan, C., & Razzaq, L. (2016). ASSISTments dataset from multiple randomized controlled experiments. In Proceedings of the 9th International Conference on Educational Data Mining.

This paper explicitly explains that the dataset leverages the "Skill Builder" infrastructure, which focuses on targeted, single-skill practice.

References for Multi-Skill Datasets
Statics 2011: This dataset is a classic example of a multi-skill dataset, originating from the Open Learning Initiative (OLI) platform. The data's details and properties are documented on the DataShop at Carnegie Mellon University.

The dataset's documentation on DataShop provides information about its content from the OLI Engineering Statics course, which is known for its multi-skill problems.



# ASSISTments 2009-2010

See [How to Interpret ASSISTments Datasets](https://sites.google.com/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets)

Many log files are from the **problem-logs** table, where each row represents a single problem for each student. Extra information includes hints used, time to first response, and more. Some fields may be duplicated when joined with student or assignment-level data.

---

## Column Descriptions

### Student Information
- **user_id**: Student ID
- **prior_problem_count**: Number of problems completed before this assignment
- **prior_correct**: Number of problems answered correctly before this assignment
- **prior_percent_correct**: Percent of past problems answered correctly
- **guessed_gender** / **guessed_gender_2**: Guessed gender (Male, Female, Unknown)
- **user_details_grade**: Current grade of the student

### Assignment Information
- **assignment_id**: Assignment ID
- **assignment_started_count**: Number of students who started the assignment
- **assignment_finished_count**: Number of students who finished the assignment
- **assignment_homework_count**: Number of students who finished outside school hours
- **homework_percent**: Percent of students who did the assignment as homework
- **due_date**: Assignment due date
- **release_date**: When the assignment became visible
- **assigned_date**: Date the teacher assigned the assignment
- **assignment_type**: Type (ClassAssignment, ARRS, remedial)
- **class_assignments_id**: Same as assignment_id
- **class_assignments_sequence_id**: Refers to the problem set (sequence)
- **student_class_id**: Class ID

### Problem Information
- **problem_logs_id**: Problem log ID
- **problem_logs_assignment_id**: Assignment ID for the problem log
- **problem_logs_user_id**: Student ID for the problem log
- **assistment_id**: Problem ID in the builder
- **problem_id**: Main problem ID
- **original**: 1 for main problem, 0 for scaffolding
- **correct**: 1 if correct on first attempt, 0 otherwise
- **answer_id**: Selected answer ID (MCQ)
- **answer_text**: Text of student's answer
- **first_action**: First action (0: Attempt, 1: Hint, 2: Scaffolding)
- **hint_count**: Number of hints requested
- **bottom_hint**: 1 if bottom-out hint requested
- **attempt_count**: Number of attempts
- **problem_start_time**: Time started
- **problem_end_time**: Time finished (ms after start)
- **first_response_time**: Time to first action (ms)
- **overlap_time**: Total time spent (may be inaccurate)
- **tutor_strategy_id**: Tutoring strategy ID

### ARRS Columns
- **ARRS Correctness**: 1 if correct on first reassessment, 0 if incorrect
- **ARRS Delay Days**: Days between skill builder completion and ARRS test
- **ARRS Adaptive Mode**: Whether assigned in adaptive mode

### Assignment Logs
- **assignment_logs_id**: Assignment log ID
- **assignment_logs_assignment_id**: Assignment ID
- **assignment_logs_user_id**: Student ID
- **assignment_start_time**: Assignment start time
- **assignment_end_time**: Assignment end time
- **last_worked_on**: Last date student worked on assignment
- **mastery_status**: For skill builder: mastered, limit exceeded, not mastered exhausted, blank
- **status_id**: 10 if excused

### Problem Set / Sequence Columns
- **sequence_id**: Same as class_assignments_sequence_id
- **sequence_name**: Title of the problem set
- **sequence_description**: Description (older sets only)
- **quality_level_id**: 1: Uncertified, 3: WPI certified
- **sequences_created_at**: Date created
- **sequences_updated_at**: Last updated

### Skill and Opportunity Columns
- **skill_id**: Skill ID(s) for the problem
- **skill_name**: Skill name(s)
- **hint_total**: Total possible hints
- **template_id**: Template ID
- **opportunity**: Number of opportunities to practice skill
- **opportunity_original**: Opportunities for original problems only

---

## Notes
- For skill builder datasets, multi-skill questions are duplicated across rows, each tagged with a skill and opportunity count.
- For non-skill builder datasets, skills and opportunities are comma-separated in a single row.
- Many columns are not currently used or are for administrative purposes.

---

For more details, see the [official documentation](https://sites.google.com/site/assistmentsdata/an-explanation-on-how-to-interpret-our_data_sets).