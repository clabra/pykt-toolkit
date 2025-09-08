# SimAKT Architecture

Guidelines for the design and implementation of a new SimAKT (Similarity-based Attention for Knowledge Tracing) model. 

Approach: we start with a As-Is initial architecture for the SimAKT model and evolve it progresively towards a To-Be architecture.  

## Requirements for the To-Be Architecture Design 

1. Goal

To augment a standard encoder-only Transformer architecture for Knowledge Tracing. The objective is to incorporate information from similar students to enhance prediction accuracy for a target student. This looks as a promising direction, moving beyond purely intra-student sequential modeling to leverage collaborative patterns.

2. Augmented architecture

We look for a non-invasive approach that takes one model and augment it to improve performance disrupting existing arquitecture as less as possible.


## Similarity-Based Attention 

The original Transformer was designed for seq2seq tasks like machine translation, where both the input and output are sequences of the same type of tokens (e.g., words). In contrast, Knowledge Tracing (KT) tasks involve input sequences composed of interaction data, including concept IDs, responses, and sometimes additional information such as problem/question IDs or timestamps. The output, is typically a prediction about the student's next response. 

The 'taxonomy.md' file provides a overview of the main models and challengues of the Transformer approach applied to the field of Konwledge Tracing and specifically, of the models implemented in this project. 


### From Intra-Student to Inter-Student Modeling: A Paradigm Shift 
 
The central thesis of the approach is the shift from a purely individualized approach to collaborative filetring that leverages inter-users information. By identifying and leveraging the learning behaviors of "students who have similar question-answering experiences," 
a model can inform predictions for a given student. This paradigm shift allows the model to draw on a richer, more extensive 
set of data from similar peers, providing a powerful supplement to a student's own historical sequence. 
 

### Defining "Collaborative Information" in Knowledge Tracing 
 
Within the context of knowledge tracing, "collaborative information" refers to the insights and 
signals derived from the learning behaviors of a group of learners, particularly those identified 
as similar to a target student. This goes beyond the traditional intra-student focus by 
explicitly modeling the relationships and collective patterns that exist across a student 
population. This approach is motivated by the observation that learners sharing similar 
cognitive states often display comparable problem-solving performances. 


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
constitutes "similarity" —be it a simple metric on question-answering history or a complex, 
learned embedding— is a crucial design decision that fundamentally determines the model's 
capability and its computational complexity. 

In our case we will use an approach based in similarity of learning trajectories.  

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

**SimAKT, unlike other models, uses this format.**. As we characterize students by their sequences of learning curves, two similar trajectories in SimAKT mean that the students have been exposed to similar concepts (i.e. questions to train similar skills) with similar performance (i.e. they got simialr mastery levels after similar number of attempts).  

The sequence of student interactions is preprocessed to have the sequence of tuples where each tuple can be consideerd a point in a trajectory.

 
### Current Similarity-based Models  
 
Some models exemplify the shift towards collaborative and similarity-based attention 
mechanisms. They each address the problem from a distinct architectural perspective, 
highlighting a growing consensus that collaborative information is a vital component for 
robust knowledge tracing. 

- CokT: uses similarity calculated using IDF (Inverse Document Frequency) and BM (Best Match). The architecture uses RNN + Attention instead of a Transformer archuitecture. The cokt-ktsimilarity.pdf paper describes the approach in detail. 
- FINER: uses similarity calculated using extracted so-called Follow-up Performance Trends (FPTs) and historical data to improve the predictions of a model based on Long Short Term Memory (LSTM) networks + Attention. The paper finer-ktsimilarity.pdf describes the approach in detail. 


## Integration with pyKT Framework

The SimAKT  model implementation follows the guidelines defined in contribute.pdf to add a new model to the pyKT framework. This model introduces a novel Transformer-based architecture that uses attention mechanisms based on trajectory similarity. 

### Compatibility

Full integration with existing pyKT infrastructure
- Standard data loaders and preprocessing
- Evaluation metrics (AUC, accuracy, precision, recall)
- Cross-validation and model comparison tools
- WandB experiment tracking



### Key Components

1. **Argument Parsing** (wandb_simakt_train.py)
    - Parses command-line arguments
    - Sets default values for SimAKT-specific parameters
    - Passes parameters to main training function

2. **Configuration Loading**
    - **kt_config.json**: Training hyperparameters (batch_size=32 for SimAKT)
    - **data_config.json**: Dataset specifications (assist2015: 100 concepts, maxlen=200)

3. **Data Loading** (init_dataset4train)
    - Loads preprocessed sequences from CSV files
    - Creates PyTorch DataLoaders
    - Handles train/valid/test splits based on fold

4. **Model Initialization** (init_model)
    - Creates SimAKT model instance with specified architecture
    - Initializes embeddings for questions and interactions
    - Sets up Transformer blocks with similarity-based attention

5. **Training Loop** (train_model)
    - Iterates through epochs
    - For each batch:
      - Forward pass through model
      - Calculate loss (BCE + Contrastive Loss)
      - Backward pass and optimization
      - Track training metrics
    - Validates after each epoch
    - Saves best model based on validation AUC
    - `compute_loss()` method for PyKT compatibility
    - Loss function 
    - Proper sequence mask handling
    - Compatible with existing training loops

6. **Evaluation** (evaluate)
    - Runs model in eval mode
    - Computes AUC and accuracy on validation/test sets
    - No gradient computation during evaluation

7. **Model Saving**
    - Saves model checkpoint when validation improves
    - Stores configuration alongside model
    - Enables model restoration for inference

### Data Flow

1. **Input**: Student interaction sequences (questions, responses)
2. **Embedding**: Convert discrete tokens to continuous representations
3. **Attention Processing**: Apply similarity-based attention mechanisms
4. **Prediction**: Output probability of correct response
5. **Loss Calculation**: Compare predictions with ground truth
6. **Optimization**: Update model parameters via backpropagation

### Key Parameters

- **Dataset**: assist2015 (100 concepts, educational dataset)
- **Model Architecture**: 
  - d_model=256 (embedding dimension)
  - n_blocks=4 (Transformer layers)
  - num_attn_heads=8 (attention heads)
  - dropout=0.3
- **Training**:
  - batch_size=32
  - learning_rate=0.001
  - optimizer=Adam with weight_decay=1e-5
- **Loss Function**: Binary Cross-Entropy + Contrastive Loss (λ=0.1)


## Baseline Models

We will take as baselines for metrics comparison these attention-based models (all of them implemented in pykt/models folder): 

- SAKT: one of the first attention-based models that remains competitive and serves as a baseline for subsequent attention-based models
- AKT: consistently outperforms other models in many evaluation with different datasets and scenarios 
- DKVMN: a competitive variant that is relevant for our approach because is based on the use of memory
- SAINT: usually don't outperform AKT but it's interesting due to its encoder-decoder architecture. A variant, SAINT+, is reported to show top performance with the EdNet dataset
- DTransformer (2023): tne most recent of the chosen models, outperforms the rest of models (including AKT) in most evaluations with different datasets and scenarios 

Other models show promising performance and could outperform AKT, including SAINT+ (2021), DIMKT (2023), stableKT (2024), and extraKT (2024). However, these models have been excluded because they lack evaluation on datasets that would enable meaningful comparison with the selected baseline models.

The KDD Cup 2010 datasets became a kind of standard for benchmarking new Knowledge Tracing models. While the original competition used RMSE, much of the subsequent academic literature, particularly in deep learning, has evaluated performance using Area Under the Curve (AUC) and Accuracy (ACC).

Unlike the feature-engineering and logistic regression-based models that were common in 2010, recent challenges have been dominated by deep learning, specifically Transformer and attention-based models (AKT, SAINT, etc.).

Below are indicative performance metrics for some well-known Knowledge Tracing models on a version of the 'Bridge to Algebra 2008-2009' dataset, often referred to as 'kddcup' in research papers.

```
Model	AUC	ACC
DKT (Deep Knowledge Tracing)	~0.83-0.85	~0.76-0.78
DKVMN (Dynamic Key-Value Memory Networks)	~0.84-0.86	~0.77-0.79
AKT (Attentive Knowledge Tracing)	~0.86-0.88	~0.78-0.80
SAINT+ (Separated Self-Attentive Neural KT)	~0.87-0.89	~0.79-0.81
```

State-of-the-Art AUC: 

- AUC: According to recent, large-scale challenges (AAAI 2023 Global Knowledge Tracing Challenge, The NeurIPS 2020 Education Challenge, Kaggle, 2020-2021), the state-of-the-art **AUC for knowledge tracing on complex datasets typically falls in the 0.81 to 0.87 range**. The specific value depends heavily on the dataset's characteristics and cleanliness.
- Transformer Dominance: In all cases, the core of the winning solutions was a Transformer-based architecture, confirming that **models like AKT and [SAINT+](https://arxiv.org/pdf/2010.12042) are the foundational building blocks for top performance**.
- Ensembling and Feature Engineering are Crucial: Achieving the highest scores requires more than just a single, well-designed model. The winning solutions consistently use ensembles of multiple models and incorporate carefully engineered features related to timing, past performance, and question characteristics to gain a competitive edge.


### Next Steps

#### General guidelines to take into account in next steps:

1) Follow an approach based in 3 phases: 1) architecture design exploration and refinement, 2) architectural decisions after careful analysis and evaluation of the alternatives, 3) implementation. In all the phases we'll collaborate together, going step by step and checking each step before continuing to the next one. In the implementation phase, we'll go step by step, evaluatin the model after each step to check that it works as expected before continuing with the next steps. 

3) Make changes according to contribute.pdf and quickstart.pdf, taking into account that our goal is to contribute a new simakt model, so change only what is neccessary without modifing the pykt framework or other models and follow the standard way to add code for a new model, and the corresponding scripts to train, evaluate, etc.   


#### Phase 1 - Architecture Design Exploration and Refinement

##### Explored approaches

#### 1. Encoder-only - Additional inter-student head

This approach involves incorporating an inter-student attention head into an encoder-only architecture initially inspired by DTransformer. The design leverages attention heads that utilize a *Temporal and Cumulative Attention (TCA)* mechanism to estimate knowledge states. These states are inferred by explicitly accounting for: (1) the temporal dynamics of the learning process, and (2) the cumulative effort exerted by the student. The inter-student head provides another complementary view about how to predict responses by lookin what happened in the past with similar students. 

The approach has been explored, analyzed and described in detail in *simakt_phase1_proposal_encoder_head.md* but is left aside, for now, in favor of a decoder-only approach that looks more promising when it comes to leverage inter-student information. The problem with the encoder-only design is that the **encoding of the students is not learned but calculated previously** using heuristic approaches, saved in a bank memory and then retrieved using clustering or k-nn techniques. 

A decoder-only approach looks more prommising when it comes to **encode student trajectories through model training, with a proper loss to match students in such a way that response predictions are optimized**. 

#### 2. Decoder-only - Self-attention on interaction (S, R) tuples to get similar learning trajectories

This approach implements decoder-only architecture using self-attention on interaction (S, R) tuples. 

The fundamental innovation is that Q·K^T matching learns to identify similar trajectories through backpropagation on prediction loss. When trajectories with similar patterns lead to similar outcomes, the model automatically learns to give them high attention scores. This emergent similarity is more powerful than any hand-crafted metric or external memory system.

The approach has been explored, analyzed and described in detail in *simakt_phase1_proposal_decoder.md* but is left aside, for now, in favor of a more simple encoder-only approach that also looks promising and is more aligned with the initial paper proposal. 









