# SimAKT Architecture

Guidelines for the design and implementation of a new SimAKT (Similarity-based Attention for Knowledge Tracing) model. 

## Introduction

The original Transformer was designed for seq2seq tasks like machine translation, where both the input and output are sequences of the same type of tokens (e.g., words). In contrast, Knowledge Tracing (KT) tasks involve input sequences composed of interaction data, including concept IDs, responses, and sometimes additional information such as problem/question IDs or timestamps. The output, is typically a prediction about the student's next response. 

The 'taxonomy.md' file provides a overview of the main models and challengues of the Transformer approach applied to the field of Konwledge Tracing and specifically, of the models implemented in this project. 


### From Intra-Student to Inter-Student Modeling: A Paradigm Shift 
 
A significant challenge in personalized learning systems is data sparsity. For new students or 
those with limited interaction history, the "intra-student information" is sparse and insufficient 
to train a reliable model. This issue mirrors a classic problem in recommender systems, 
where a cold-start user with no history is difficult to make recommendations for. The solution, 
in both domains, involves a shift from a purely individualized approach to one that leverages 
"inter-student information"—the collective intelligence of a peer group.

The central thesis of the paper is that a next major conceptual leap in knowledge tracing might be 
the integration of collaborative information. By identifying and leveraging the learning 
behaviors of "students who have similar question-answering experiences," a model can inform 
predictions for a given student, even when their own history is limited.

This paradigm shift addresses the fundamental limitation of data sparsity by allowing the model to draw on a 
richer, more extensive set of data from similar peers, providing a powerful supplement to a 
student's own historical sequence. 
 

### Defining "Collaborative Information" in Knowledge Tracing 
 
Within the context of knowledge tracing, "collaborative information" refers to the insights and 
signals derived from the learning behaviors of a group of learners, particularly those identified 
as similar to a target student. This goes beyond the traditional intra-student focus by 
explicitly modeling the relationships and collective patterns that exist across a student 
population. This approach is motivated by the observation that learners sharing similar 
cognitive states often display comparable problem-solving performances. 

Collaborative signals can manifest in several forms, each with its own architectural 
implications. The most direct form involves retrieving the full question-answering sequences 
of peers who have a history of similar interactions. A more abstract approach leverages 
pre-calculated or learned patterns, such as "Follow-up Performance Trends" (FPTs), that 
represent common learning trajectories derived from the entire student corpus. These 
trends, while not tied to a specific individual, still represent an aggregate form of collaborative 
information. The efficacy of a collaborative model is therefore fundamentally dependent on 
the definition of what constitutes "similarity" and how these external signals are integrated. 


### The Mechanisms of Similarity-Based Attention 
 
Traditional Transformer-based models like SAINT employ a self-attention mechanism that 
computes attention weights based on the relationships between tokens within a single 
sequence, such as a student's past interactions with exercises. For collaborative knowledge 
tracing, this mechanism must be redefined to calculate attention based on the similarity 
between different students or between a student and a pre-defined learning pattern. 
The core of this "similarity-based attention" involves a creative adaptation of the standard 
attention architecture. In a cross-attention setup, a "query" vector representing the current 
student's learning state can be used to query a set of "key" vectors derived from the 
representations of similar peers or collaborative patterns. The resulting attention score 
becomes a measure of semantic or behavioral similarity, which allows the model to assign 
higher weights to the most relevant peer interactions or patterns. The model's hidden 
representation for the current time step is then a weighted sum of the "value" vectors from 
these similar peers. This process allows the model to selectively and dynamically leverage the 
most pertinent collaborative information, thereby enhancing its ability to make accurate 
predictions, particularly when the intra-student data is sparse. The choice of what 
constitutes "similarity"—be it a simple metric on question-answering history or a complex, 
learned embedding—is a crucial design decision that fundamentally determines the model's 
capability and its computational complexity. 

In our case we will use an approach based in similarity of learning paths. The sequence of student interactions is preprocessed to have a sequence of tuples where each tuple can be consideerd a point in a trajectory. Then we will applied existent techniques to encode trajectories as h vectors. Two similar trajectories will have similar h vectors. 

    Each tuple (S, N, M) will contain information about: 
    - question or skill (S)
    - number of attempts (N): number of interactions of the student with the question or skill
    - mastery (M): level of mastery skill acquired after the number of attemps

Each tuple defines the learning curve for this skill. We use a sigmoid curve to model a monotonic learning curve that follows a pattern cahractrized by slow start, rapid improvement phase and a plateau effect once the skill has been mastered. 

 
### Current Similarity-based Models  
 
Some models exemplify the shift towards collaborative and similarity-based attention 
mechanisms. They each address the problem from a distinct architectural perspective, 
highlighting a growing consensus that collaborative information is a vital component for 
robust knowledge tracing. See section "3. In-Depth Examination of Relevant Models" of similarity-transformers.pdf" for a description of CoKT, FINER and Coral. 

## Definition of Learning Trajectory Similarity for SimAKT

Learning trajectories can be represented as: 

  - Sequence of Interactions 
  
  (St, Rt), where S: Skill, R: Correct or Wrong response (1 or 0)

  - Skill Mastery Evolution

  S1[0.2→0.6→0.8], S2[0.1→0.3→0.7], S3[0.0→0.5]

- Sequence of Skill Learning Curves

The learning curve can be modeled through a sigmoid calculated from a (S, N, M) tuple that represent a point in the learning trajectory. 

    Each learning tuple (S, N, M) will contain information about: 
    - question or skill (S)
    - number of attempts (N): number of interactions of the student with the question or skill
    - mastery (M: level of mastery skill acquired after the number of attemps

We have M as the y and N as the x coordinate in the sigmoid curve characteristic of each S skill. After N attempts the student learns (achieves skill mastery level) or fails. 

**SimAKT, unlike other models, uses this format.**

As we characterize students by their sequences of learning curves, two similar trajectories in SimAKT mean that the students have been exposed to similar concepts (i.e. questions to train similar skills) with similar performance (i.e. they got simialr mastery levels after similar number of attempts).  



## SimAKT Implementation Requirements


The SimAKT  model implementation follows the guidelines defined in contribute.pdf to add a new model to the pyKT framework. This model introduces a novel Transformer-based architecture that uses attention mechanisms based on trajectory similarity. 

### Integration with pyKT Framework

**Compatibility**: Full integration with existing pyKT infrastructure
- Standard data loaders and preprocessing
- Evaluation metrics (AUC, accuracy, precision, recall)
- Cross-validation and model comparison tools
- WandB experiment tracking

**Configuration**: Supports all standard pyKT parameters plus SimAKT-specific options:
- `mastery_threshold`: Threshold for response correctness prediction


### Key Components

**Core SimAKT Model** (`pykt/models/simakt.py`)

**Model Initialization** (`pykt/models/init_model.py`):
- SimAKT registered in model factory (line 145)
- Compatible with standard pyKT configuration parameters
- Integrated with existing data loading and preprocessing

**Training Integration**:
- `compute_loss()` method for PyKT compatibility
- Loss function 
- Proper sequence mask handling
- Compatible with existing training loops

**Training Script** (`examples/wandb_simakt_train.py`):
- Command-line interface following pyKT patterns
- SimAKT-specific hyperparameters (similarity_cache_size, mastery_threshold, curve_dim)
- WandB integration support
- Standard training parameters (num_epochs, batch_size, learning_rate)

**Configuration Setup**:
- Model parameters defined in `configs/kt_config.json` 
- Standard pyKT configuration pattern followed
- SimAKT hyperparameters properly configured

## Baseline Models

We take as baselines these attention-based models: 

- SAKT: one of the first attention-based models that remains competitive and serves as a baseline for subsequent attention-based models
- AKT: consistently outperforms other models in many evaluation with different datasets and scenarios 
- DKVMN: a competitive variant that is relevant for our approach because is based on the use of memory
- SAINT: usually don't outperform AKT but it's interesting due to its encoder-decoder architecture. A variant, SAINT+, is reported to show top performance with the EdNet dataset
- DTransformer (2023): tne most recent of the chosen models, outperforms the rest of models (including AKT) in most evaluations with different datasets and scenarios 
- CL4KT?: Contrastive Learning for Knowledge Tracing (https://drive.google.com/file/d/1JtJNKr1tHU5lxLHy0zEms3_I-aYv2Ph0/view)

Other models show promising performance and could outperform AKT, including SAINT+ (2021), DIMKT (2023), stableKT (2024), and extraKT (2024). However, these models have been excluded because they lack evaluation on datasets that would enable meaningful comparison with the selected baseline models.

The KDD Cup 2010 datasets became a kind of standard for benchmarking new Knowledge Tracing models. While the original competition used RMSE, much of the subsequent academic literature, particularly in deep learning, has evaluated performance using Area Under the Curve (AUC) and Accuracy (ACC).

Unlike the feature-engineering and logistic regression-based models that were common in 2010, recent challenges have been dominated by deep learning, specifically Transformer and attention-based models (AKT, SAINT, etc.).

Below are indicative performance metrics for some well-known Knowledge Tracing models on a version of the 'Bridge to Algebra 2008-2009' dataset, often referred to as 'kddcup' in research papers.

Model	AUC	ACC
DKT (Deep Knowledge Tracing)	~0.83-0.85	~0.76-0.78
DKVMN (Dynamic Key-Value Memory Networks)	~0.84-0.86	~0.77-0.79
AKT (Attentive Knowledge Tracing)	~0.86-0.88	~0.78-0.80
SAINT+ (Separated Self-Attentive Neural KT)	~0.87-0.89	~0.79-0.81

State-of-the-Art AUC: 

- AUC: According to recent, large-scale challenges (AAAI 2023 Global Knowledge Tracing Challenge, The NeurIPS 2020 Education Challenge, Kaggle, 2020-2021), the state-of-the-art **AUC for knowledge tracing on complex datasets typically falls in the 0.81 to 0.87 range**. The specific value depends heavily on the dataset's characteristics and cleanliness.
- Transformer Dominance: In all cases, the core of the winning solutions was a Transformer-based architecture, confirming that **models like AKT and [SAINT+](https://arxiv.org/pdf/2010.12042) are the foundational building blocks for top performance**.
- Ensembling and Feature Engineering are Crucial: Achieving the highest scores requires more than just a single, well-designed model. The winning solutions consistently use ensembles of multiple models and incorporate carefully engineered features related to timing, past performance, and question characteristics to gain a competitive edge.

## Architecture Decissions

We will use the DTransformer model as the foundation for our architecture due to its superior performance among the baseline models. Additionally, we will incorporate key concepts from the AKT model, as DTransformer builds upon several ideas originally proposed in AKT.

**AKT:**
- Uses two encoders: the Question Encoder (which considers only the questions) and the Knowledge Encoder (which considers both the questions and the responses), along with a Knowledge Retriever that determines the knowledge state based on both encoders.
- Employs the traditional attention mechanism with modifications aimed at: 1) monotonic attention in the Knowledge Retriever, 2) reducing the relevance of exercises completed a long time ago, and 3) reducing the relevance of exercises involving different concepts.
- Each encoder and the Knowledge Retriever has a key, query, and value embedding layer that maps the input into output queries, keys, and values of dimensions \(D_q = D_k\), \(D_k\), and \(D_v\), respectively.
- Uses question embeddings to map both queries and keys (while SAKT, for example, uses question embeddings for queries and response embeddings to map keys and values).

**DTransformer:**
- Uses an approach based on "Temporal and Cumulative Attention (TCA)" that considers the cumulative effort in the learning process when implementing the attention mechanism. This allows it to infer the student's knowledge state at each moment and predict responses accordingly. Additionally, DTransformer uses Contrastive Learning in the loss function to enforce the monotonicity of knowledge. In contrast, the AKT model infers the knowledge state based on responses, which can lead to unstable and non-monotonic knowledge states.
- Uses knowledge as a query (in addition to questions as queries) in the attention mechanism, enabling it to extract the knowledge state.

**Similarities:**
- Both DTransformer and AKT use a Rasch Model that accounts for the difficulty of exercises. This explains why a skill might be considered mastered at a given time, yet later, an exercise targeting the same skill could be answered incorrectly if its difficulty is high.
- DTransformer leverages the modified attention mechanism from AKT.


## Architecture Design Requirements

- The SimAKT model is ready as an copy of DTransformer. This provides a clean foundation to build upon with novel similarity-based attention mechanisms and learning trajectory improvements.
- I'll provide instructions for evolving this DTransformer-based foundation into the SimAKT model



## Architecture Design



## Testing

python wandb_dtransformer_train.py --dataset_name=assist2015 --use_wandb=0

python wandb_simakt_train.py --dataset_name=assist2015 --use_wandb=0

