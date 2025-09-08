# GainAKT Architecture

Guidelines for the design and implementation of a new model called GainAKT (Gains-based Attention for Knowledge Tracing). 

Approach: we start with a As-Is initial architecture for the GainAKT model and evolve it progresively towards a To-Be architecture.  

## Requirements for the To-Be Architecture Design 

1. Goal

To evolve an existent encoder-only Transformer architecture. 

Priorities to choose the starting model: 1) only attention-based Transformers from taxonomy.md, 2) architecture similarity to GainAKT (single block preferred), and 3) best performance among compliant models.

2. Non-intrusive changes

We look for a non-intrusive approach that takes one model and change it as less as possible to implement the approach and improve performance disrupting existing arquitecture as less as possible.

3. Guidelines

We are creating a new model to contribute to the pykt framewotk. Follow guidelines in contribute.pdf and quickstart.pdf when it comes to create model code and evaluation scripts that follow pykt standards.   


## The Approach

The new model is an encoder-only with self-attention on interaction (S, R) tuples to learn tuple learning gains. 

The fundamental innovation of this approach lies in the reformulation of the attention mechanism to directly compute and aggregate learning gains. It is described in sections below. 


## Architectural Design

### Core Innovation

The fundamental innovation of this approach lies in the reformulation of the attention mechanism to directly compute and aggregate learning gains. Instead of treating attention weights as abstract importance scores, this architecture learns to:

1. **Identify relevant interaction tuples**: Through Q·K^T matching, the model learns to identify (S, R) tuples that involve similar knowledge components
2. **Quantify learning gains**: Values in the attention mechanism represent the learning gains induced by specific interactions
3. **Aggregate knowledge states**: Knowledge states are computed as weighted linear combinations of learning gains

### Token Representation

```text
Tokens = (S, R) tuples
Where:
- S: Skill/Concept/Knowledge Component identifier
- R: Response (0 for incorrect, 1 for correct)
```

Each token represents a discrete learning interaction, encapsulating both the skill being practiced and the outcome of that practice.

### Architecture Components

#### 1. Embedding Layer

- **Input**: (S, R) tuples from learning trajectories
- **Output**: Dense embeddings that capture both skill semantics and response patterns
- **Function**: Maps discrete (S, R) pairs to continuous vector space

#### 2. Self-Attention Mechanism with Learning Gains

The core architectural innovation is the redefinition of the attention mechanism components:

```text
Query (Q): Learned representation of current interaction context
Key (K): Learned representation of historical interaction patterns  
Value (V): Learning gains induced by (S, R) interactions
```

The attention computation follows:

```text
Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
```

However, the semantic meaning is fundamentally different:

- **QK^T matching**: Learns to identify relevant historical interactions based on skill similarity
- **Attention weights**: Represent the relevance of past learning gains to current knowledge state
- **Weighted sum**: Aggregates relevant learning gains to compute current knowledge state

#### 3. Knowledge State Computation

Knowledge states at each time step are calculated as:

```text
Knowledge_State_t = sum(attention_weights_i * learning_gains_i)
```

This formulation provides direct interpretability: each component of the knowledge state can be traced back to specific learning gains from relevant interactions.

### Learning Mechanism

#### 1. Objective Function

The model learns through backpropagation on prediction loss, which drives the attention mechanism to:

- Assign high attention weights to interactions with similar knowledge components
- Learn appropriate learning gain values that contribute to accurate predictions
- Develop representations that capture skill relationships and learning dynamics

#### 2. Training Dynamics

During training, the model automatically learns to:

- **Match similar skills**: Q·K^T computation learns to identify interactions involving the same or related knowledge components
- **Quantify learning impact**: Values learn to represent the actual learning gains from specific interactions
- **Aggregate effectively**: Attention weights learn to combine learning gains optimally for prediction

#### 3. Emergent Similarity

Unlike hand-crafted similarity metrics, the model develops an emergent understanding of skill similarity through the optimization process. When interactions with similar knowledge components consistently lead to similar learning outcomes, the model learns to give them high attention scores.

## Mathematical Formulation

### Attention Mechanism as Learning Gain Aggregation

Given a sequence of interactions up to time step t, the knowledge state is computed as:

```text
h_t = sum_{i=1}^{t-1} alpha_{t,i} * g_i
```

Where:

- `h_t`: Knowledge state at time t
- `alpha_{t,i}`: Attention weight between current context t and past interaction i
- `g_i`: Learning gain induced by interaction i (stored as value in attention)

The attention weights are computed as:

```text
alpha_{t,i} = softmax(q_t^T k_i / sqrt(d_k))
```

Where:

- `q_t`: Query vector representing current learning context
- `k_i`: Key vector representing past interaction pattern
- `d_k`: Scaling dimension

### Learning Gain Representation

Each learning gain `g_i` is represented as a dense vector that captures:

- **Skill-specific gains**: Components corresponding to different knowledge components
- **Response-dependent effects**: How correct/incorrect responses affect learning
- **Transfer effects**: How learning in one skill affects related skills

### Prediction Layer

The final response prediction is computed as:

```text
P(R_t = 1 | S_t, h_t) = sigma(W_out [h_t; embed(S_t)] + b_out)
```

Where:

- `sigma`: Sigmoid activation
- `W_out`: Output projection weights
- `embed(S_t)`: Embedding of current skill
- `h_t`: Aggregated knowledge state


## Advantages Over Alternative Approaches

### 1. Interpretability

- **Direct Learning Gain Computation**: Values directly represent learning gains, enabling clear interpretation
- **Traceable Knowledge States**: Each knowledge state component can be attributed to specific interactions
- **Causal Explanations**: The model provides clear explanations for why certain interactions influence predictions

### 2. Simplicity

- **Single Architecture**: No need for separate encoders or complex memory systems
- **End-to-end Learning**: All components learned jointly through standard backpropagation
- **Fewer Hyperparameters**: Simpler architecture reduces hyperparameter tuning complexity

### 3. Educational Alignment

- **Learning Gain Semantics**: Directly models educational concepts familiar to practitioners
- **Skill-based Organization**: Natural alignment with Q-matrix and G-matrix educational frameworks
- **Progressive Learning**: Models cumulative learning through gain aggregation

### 4. Computational Efficiency

- **Standard Transformer**: Leverages well-optimized transformer implementations
- **No External Memory**: No need for student encoding banks or similarity computations
- **Scalable**: Standard attention complexity O(n^2) with established optimization techniques

## Theoretical Foundation

### Learning Gain Theory

This approach is grounded in educational psychology's learning gain theory, which posits that:

- Learning occurs through discrete interactions with problems/skills
- Each interaction produces a measurable learning gain
- Knowledge states evolve through accumulation of learning gains
- Similar skills produce transferable learning gains

### Attention as Learning Gain Aggregation

The reformulation of attention as learning gain aggregation provides:

- **Theoretical Justification**: Aligns with established educational theory
- **Interpretable Weights**: Attention scores represent educational relevance
- **Causal Modeling**: Direct modeling of learning cause-and-effect relationships

### Connection to Q-matrix and G-matrix

This approach naturally aligns with established educational frameworks:

- **Q-matrix Integration**: The skill identifiers S directly correspond to Q-matrix knowledge components
- **G-matrix Learning**: The learned values (learning gains) approximate the G-matrix entries, but are learned from data rather than pre-specified
- **Dynamic G-matrix**: Unlike static G-matrices, learned gains can capture individual differences and contextual effects

## Requirements for a GainKT Architectural Diagram

```mermaid
flowchart TD
    %% Input Layer
    A["`**Input Sequence**
    (S₁,R₁), (S₂,R₂), ..., (Sₜ₋₁,Rₜ₋₁)
    _Learning Interaction Tuples_`"] --> B
    
    %% Embedding Layer
    B["`**Embedding Layer**
    Maps (S,R) → Dense Vectors
    _Skill + Response Embeddings_`"] --> C["`**Embedded Sequence**
    [e₁, e₂, ..., eₜ₋₁]
    _d_model dimensional vectors_`"]
    
    %% Self-Attention Mechanism
    C --> D{"`**Self-Attention Mechanism**
    _Learning Gain Aggregation_`"}
    
    %% Q, K, V Computation
    D --> E["`**Query (Q)**
    qₜ = eₜ × W_Q
    _Current Context_`"]
    D --> F["`**Key (K)**
    K = [e₁, e₂, ..., eₜ₋₁] × W_K
    _Historical Patterns_`"]
    D --> G["`**Value (V) - Learning Gains**
    V = [g₁, g₂, ..., gₜ₋₁] × W_V
    _**Novel: Learned Gains**_`"]
    
    %% Attention Computation
    E --> H["`**Attention Weights**
    α = softmax(QK^T / √d_k)
    _Skill Similarity Scores_`"]
    F --> H
    
    %% Learning Gain Aggregation
    H --> I["`**Knowledge State**
    hₜ = Σᵢ αₜᵢ × gᵢ
    _Weighted Learning Gains_`"]
    G --> I
    
    %% Prediction Layer
    I --> J["`**Concatenation**
    [hₜ ; embed(Sₜ)]
    _Current State + Target Skill_`"]
    
    J --> K["`**Prediction Head**
    P(Rₜ=1|Sₜ) = σ(W_out[hₜ;embed(Sₜ)] + b)
    _Response Probability_`"]
    
    %% Final Output
    K --> L["`**Output**
    Response Prediction
    _Binary Classification_`"]
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef embedding fill:#f3e5f5
    classDef attention fill:#fff3e0
    classDef novel fill:#ffebee
    classDef knowledge fill:#e8f5e8
    classDef prediction fill:#fce4ec
    classDef output fill:#f1f8e9
    
    class A input
    class B,C embedding
    class D,E,F,H attention
    class G,I novel
    class J knowledge
    class K prediction
    class L output
    
```

## Key Innovation

The fundamental innovation in this SimAKT approach lies in the **semantic redefinition of attention mechanism components** to directly model educational learning processes:

### Semantic Component Redefinition

- **Traditional Attention**: Abstract importance weighting between sequence elements
- **SimAKT Innovation**: Direct computation and aggregation of concrete learning gains from educational interactions

### Novel Value Semantics

- **Query (Q)**: Represents current learning context and knowledge state requirements
- **Key (K)**: Represents historical interaction patterns and skill-based similarities  
- **Value (V)**: **Explicitly models learning gains** induced by specific (S,R) interactions

### Emergent Skill Similarity Learning

- No hand-crafted similarity metrics required
- Model learns to identify related skills through Q·K^T matching during training
- Attention weights naturally emerge to represent educational relevance between interactions

## Interpretability Benefits

This architectural design provides interpretability for knowledge tracing models through multiple complementary mechanisms:

### Direct Learning Gain Interpretation

- **Traceable Knowledge States**: Each component of the knowledge state vector can be directly attributed to specific learning gains from identifiable interactions
- **Educational Semantics**: Values in the attention mechanism have clear educational meaning (actual learning increments)
- **Quantifiable Impact**: The contribution of each past interaction to current predictions is explicitly computed and interpretable

### Causal Explanation Capabilities

- **Interaction-Level Causality**: Model can explain which specific past interactions (S,R pairs) most influenced a prediction
- **Skill Transfer Visualization**: Attention weights reveal how learning in one skill affects performance in related skills
- **Learning Trajectory Analysis**: Complete learning progression can be reconstructed through the sequence of aggregated gains

### Educational Alignment

- **Q-matrix Compatibility**: Natural integration with established knowledge component frameworks
- **G-matrix Learning**: Learned gains approximate and extend traditional G-matrix concepts with data-driven insights
- **Practitioner-Friendly**: Outputs align with familiar educational concepts (skills, mastery, learning gains)


## Typical Encoder

```mermaid
graph TD
    subgraph Input Processing
        A[Input Sequence of Tokens]
        B(Token Embeddings)
        C(Positional Embeddings)
        D["Sum (+)"]
        
        A --> B
        A --> C
        B --> D
        C --> D
    end

    D --> E[Input to Encoder Stack]

    subgraph "Encoder Stack (N x Blocks)"
        subgraph "Encoder Block 1"
            E_In_1[Block Input] --> F_1(Multi-Head Self-Attention)
            F_1 --> G_1("Add & Norm")
            E_In_1 -- Residual Connection --> G_1
            
            G_1 --> H_1(Feed-Forward Network)
            H_1 --> I_1("Add & Norm ")
            G_1 -- Residual Connection --> I_1
        end

        E --> E_In_1
        I_1 --> J[...]

        subgraph "Encoder Block N"
            J_In_N[Block Input] --> F_N(Multi-Head Self-Attention)
            F_N --> G_N("Add & Norm")
            J_In_N -- Residual Connection --> G_N

            G_N --> H_N(Feed-Forward Network)
            H_N --> I_N("Add & Norm ")
            G_N -- Residual Connection --> I_N
        end
        
        J --> J_In_N
        I_N --> K[Output from Encoder Stack]
    end

    K --> L[Output: Sequence of Contextualized Embeddings]
```

## Expected Contributions

### 1. Methodological Contributions

- Novel attention mechanism semantics for knowledge tracing
- Direct modeling of learning gains in neural architectures
- Unified framework for prediction and interpretation

### 2. Educational Contributions

- Interpretable knowledge state evolution modeling
- Causal explanations for learning predictions
- Alignment with established educational theory

### 3. Technical Contributions

- Efficient implementation of learning gain aggregation
- Scalable architecture for large educational datasets
- Framework for educational AI interpretability

## Comparison with Previous Approaches

### Versus Encoder-only with Inter-student Head

- **Simplicity**: No need for external student encoding or memory banks
- **End-to-end Learning**: All similarity learning happens through backpropagation
- **Interpretability**: Direct learning gain interpretation vs. abstract similarity scores

### Versus Decoder-only Trajectory Matching

- **Complexity**: Simpler architecture with clearer educational semantics
- **Alignment**: Better aligned with initial paper proposal focusing on learning gains
- **Implementation**: More straightforward implementation path

This encoder-only approach with learning gains represents a promising direction for SimAKT that balances predictive performance with interpretability requirements, providing a solid foundation for both research contributions and practical educational applications.






## Metrics

## simakt.py

```
Default parameters
Epoch: 28, validauc: 0.6868, validacc: 0.7475, best epoch: 18, best auc: 0.6868, train loss: 0.529769870185798, emb_type: qid, model: gainsakt, save_dir: saved_model/assist2015_gainsakt_qid_saved_model_42_0_0.2_128_0.001_8_1_200_0_1
            testauc: -1, testacc: -1, window_testauc: -1, window_testacc: -1
fold    modelname       embtype testauc testacc window_testauc  window_testacc  validauc        validacc        best_epoch
0       gainsakt        qid     -1      -1      -1      -1      0.686758830678621       0.7475401220449834      18
end:2025-09-08 06:13:59.018062
```

## simakt2.py

```
Default parameters
Epoch: 3, **validauc: 0.7184, validacc: 0.7507**, best epoch: 3, best auc: 0.7184, train loss: 0.5138106416820177, emb_type: qid, model: gainakt2, save_dir: saved_model/assist2015_gainakt2_qid_saved_model_42_0_128_0.001_8_2_256_0.1_200_0_1
            testauc: -1, testacc: -1, window_testauc: -1, window_testacc: -1
```

```
Tuned parameters
python wandb_gainakt2_train.py \
    --dataset_name=assist2015 \
    --use_wandb=0 \
    --learning_rate=2e-4 \
    --d_model=256 \
    --num_encoder_blocks=4 \
    --d_ff=1024 \
    --dropout=0.2

Epoch: 1, validauc: 0.6934, validacc: 0.7452, best epoch: 1, best auc: 0.6934, train loss: 0.557500138811338, emb_type: qid, model: gainakt2, save_dir: saved_model/assist2015_gainakt2_qid_saved_model_42_0_256_0.0002_8_4_1024_0.2_200_0_1
            testauc: -1, testacc: -1, window_testauc: -1, window_testacc: -1

Epoch: 2, validauc: 0.7113, validacc: 0.7494, best epoch: 2, best auc: 0.7113, train loss: 0.5275374122159148, emb_type: qid, model: gainakt2, save_dir: saved_model/assist2015_gainakt2_qid_saved_model_42_0_256_0.0002_8_4_1024_0.2_200_0_1

Epoch: 3, validauc: 0.7182, validacc: 0.7514, best epoch: 3, best auc: 0.7182, train loss: 0.5186076481321434, emb_type: qid, model: gainakt2, save_dir: saved_model/assist2015_gainakt2_qid_saved_model_42_0_256_0.0002_8_4_1024_0.2_200_0_1
            testauc: -1, testacc: -1, window_testauc: -1, window_testacc: -1

Epoch: 4, validauc: 0.7215, validacc: 0.7527, best epoch: 4, best auc: 0.7215, train loss: 0.5125869734824042, emb_type: qid, model: gainakt2, save_dir: saved_model/assist2015_gainakt2_qid_saved_model_42_0_256_0.0002_8_4_1024_0.2_200_0_1
            testauc: -1, testacc: -1, window_testauc: -1, window_testacc: -1
```

## Baseline models
```
PYKT Benchmark Results Summary (Question-Level AUC):
- AKT: 0.7853 (AS2009), 0.8306 (AL2005), 0.8208 (BD2006), 0.8033 (NIPS34) - **Best overall**
- SAKT: 0.7246 (AS2009), 0.7880 (AL2005), 0.7740 (BD2006), 0.7517 (NIPS34)
- SAINT: 0.6958 (AS2009), 0.7775 (AL2005), 0.7781 (BD2006), 0.7873 (NIPS34)

Other benchmarks: 
- simpleKT 0.7744 (AS2009) 0.7248 (AS2015) - Reported as strong baseline with minimal complexity
```

## Comparison


| Model | Dataset | Test AUC | Test ACC | Valid AUC | Valid ACC | Best Epoch | Notes |
|-------|---------|----------|----------|-----------|-----------|------------|--------|
| **GainSAKT** | ASSIST2015 | -1 | -1 | 0.6868 | 0.7475 | 18 | Early implementation |
| **GainAKT2** | ASSIST2015 | -1 | -1 | 0.7184 | 0.7507 | 3 | Default parameters (quick) |
| **GainAKT2** | ASSIST2015 | -1 | -1 | **0.7215** | **0.7527** | 3 | Tuned parameters (slow) |

| Model | AS2009 | AS2015 | AL2005 | BD2006 | NIPS34 | Notes |
|-------|--------|--------|--------|--------|--------|--------|
| **AKT** | 0.7853 | **0.7281** |*0.8306 | 0.8208 | 0.8033 | Best overall |
| **SAKT** | 0.7246 | **0.7114** | 0.7880 | 0.7740 | 0.7517 | Strong attention baseline |
| **SAINT** | 0.6958 | **0.7020** | 0.7775 | 0.7781 | 0.7873 | Encoder-decoder |
| **simpleKT** | 0.7744 | **0.7248** | - | - | - | Simple but effective |

1. **GainAKT2 shows improvement over GainSAKT**: 
   - GainAKT2 achieved 0.7184 valid AUC vs GainSAKT's 0.6868
   - This represents a ~4.6% improvement in validation AUC

2. **Performance vs Baselines**:
   - GainAKT2 (0.7184) is competitive with simpleKT on AS2015 (0.7248)
   - Still below top performers like AKT, but approaching strong baselines
   - Shows promise for the learning gains approach

3. **Training Efficiency**:
   - GainAKT2 converged quickly (best epoch 3 vs 18 for GainSAKT)
   - Suggests better optimization dynamics with the revised architecture

4. **Parameter Sensitivity**:
   - Default parameters performed better than the tuned configuration
   - Indicates the model may prefer simpler configurations initially

The results show that the learning gains approach is viable and improving, with GainAKT2 demonstrating competitive performance against established baselines while maintaining the interpretability advantages of explicit learning gain modeling.

