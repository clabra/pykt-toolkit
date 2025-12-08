
## Approach

Deep learning models for knowledge tracing aim to predict learner performance over time, but most existing approaches emphasize predictive accuracy at the cost of interpretability. We present iKT, a novel framework that achieves interpretability-by-design through semantic alignment of latent states. iKT restricts the solution space to representations that are both predictive and consistent with pedagogical principles, ensuring that internal states correspond to meaningful learning concepts. This is accomplished via mechanisms that enforce semantic consistency and guide the model toward valid configurations. By adopting an interpretability-by-design paradigm, iKT offers transparent insight into knowledge evolution, enhances trustworthiness, and provides actionable guidance for educators. Experiments on benchmark knowledge tracing datasets show that iKT matches or surpasses state-of-the-art performance while delivering interpretable outputs on knowledge states and their progression along students' learning paths.

### The Interpretability Challenge in Knowledge Tracing

**The Black Box Problem**: 

Traditional deep learning models for knowledge tracing achieve high predictive accuracy but suffer from a fundamental interpretability deficit. During training and deployment, these models operate as opaque black boxes: their internal representations evolve without semantic grounding and they provide predictions about the future performance of the students but no information about their knowledge states or learning trajectories. 

1. **Hidden Knowledge Evolution**: We cannot observe how the model's internal knowledge states change as it processes student interaction sequences, making it impossible to verify whether learned representations correspond to meaningful learning constructs.

2. **Unverified Mastery Estimates**: When they try to project latent states into skill mastery vectors, they tend to exhibit patterns that violate pedagogical principles—they might decrease over time (contradicting the monotonicity principle), take negative values (lacking interpretable semantics), or show no correlation with observed performance (breaking the fundamental link between internal state and external behavior).

3. **Unconstrained Architectural Freedom**: Without explicit constraints, deep learning models can learn representations that optimize predictive loss while producing nonsensical intermediate states. The model might internally represent "mastery" as any arbitrary vector that happens to minimize cross-entropy, regardless of whether those values have educational meaning.

4. **Post-hoc Opacity**: Even when models incorporate mechanisms such as attention weights or skill embeddings, they don't translate into interpretable output. We cannot verify in real-time whether architectural constraints like positivity or monotonicity are actually satisfied during optimization, nor can we detect when the model strays into semantically inconsistent regions of the parameter space.

This interpretability gap has profound implications: educators cannot trust model recommendations, researchers cannot validate learning theories through model introspection, and the deployment of KT systems in high-stakes educational contexts remains problematic.

### Our Proposal: Interpretability-by-Design with Semantic Alignment

**Core Innovation**: Rather than treating interpretability as an afterthought or post-hoc analysis problem, iKT embeds interpretability directly into the learning process through **semantic alignment of latent states**. The model's internal representations are constrained from the outset to remain within a solution space that is both predictive and pedagogically meaningful.

**Two Phases Approach**:

1. **Phase 1 - Warmup with Performance Learning**: 
   We initialize the model by training it to predict student performance while regularizing skill difficulty embeddings to IRT-calibrated values. The loss is $L_{\text{total}} = L_{\text{BCE}} + \lambda_{\text{reg}} \times L_{\text{reg}}$, where **L_BCE optimizes prediction accuracy** and **L_reg prevents difficulty embedding drift**. During this phase, the ability encoder learns to extract meaningful student ability θ_i(t) from the hidden state, and the model builds good performance-predictive representations without interpretability constraints yet.
   
2. **Phase 2 - IRT Alignment for Interpretability**:

After the warmup period, we add an interpretability constraint to ensure predictions align with IRT-based mastery expectations. The loss becomes $L_{\text{total}} = L_{\text{BCE}} + \lambda_{\text{align}} \times L_{\text{align}} + \lambda_{\text{reg}} \times L_{\text{reg}}$, where **L_align = MSE(p_correct, mastery_irt)** enforces consistency between predicted probabilities and IRT-based mastery $M_{\text{IRT}} = \sigma(\theta_i(t) - \beta_k)$. This allows the model to maintain high AUC while ensuring its predictions are consistent with psychometric theory—students with higher ability relative to skill difficulty should have higher mastery probabilities.

**Key Advantages**:

- **Verifiable Interpretability**: Unlike post-hoc explanations, our approach provides *guarantees* about semantic consistency through IRT alignment. We measure interpretability using Pearson correlation r between predicted probabilities p_correct and IRT-based mastery M_IRT = σ(θ - β), with target r > 0.85 indicating strong alignment with psychometric theory.

- **Transparent Trade-offs**: The hyperparameter λ_align makes the performance-interpretability balance explicit. Higher values enforce stronger IRT consistency (higher r) but may slightly reduce AUC, while lower values prioritize performance. The approach systematically explores this trade-off to find configurations that are both accurate and interpretable.

- **Real-time Monitoring**: The model captures intermediate states during training, enabling real-time verification that:
  - Student ability θ_i(t) increases over time (learning progression)
  - Skill difficulties β_k remain aligned with IRT calibration (corr_beta > 0.8)
  - IRT alignment quality stays strong (irt_correlation > 0.85)
  - Predictions are consistent with ability-difficulty relationships

- **Theoretical Grounding**: By anchoring to Rasch/IRT models, we connect deep learning to psychometric research. The model's internal states are not arbitrary neural activations—they are constrained to approximate quantities (ability, difficulty, mastery) that have established educational interpretations.

- **Minimal Overhead**: The monitoring mechanisms introduce negligible computational cost (~1-2% slowdown).

**Practical Impact**:

This approach bridges the gap between deep learning performance and educational accountability. Users can inspect model-estimated mastery levels with confidence that they reflect pedagogically meaningful constructs. It enables validation of the model's internal learning trajectories and alignment with educational theories. And it has competitive AUC while adding interpretability guarantees that purely black-box models don't provide.

**In Summary**: iKT demonstrates that interpretability need not be sacrificed for performance. By constraining the solution space to representations that are both predictive and semantically grounded, we achieve a model that is simultaneously accurate, interpretable, and theoretically justified—addressing the core limitations of existing deep knowledge tracing models. 

## Architecture 

### iKT3

<div style="width: 1200px;">

```mermaid
graph TD
    subgraph "Input Layer"
        Input_q[["Input Questions q<br/>[B, L]"]]
        Input_r[["Input Responses r<br/>[B, L]"]]
        Ground_Truth_r[["Ground Truth Responses<br/>[B, L]"]]
    end
    
    subgraph "Reference Model Targets (Pre-computed)"
        RefTargets[["IRT Reference Targets<br/>β_IRT: [num_skills]<br/>θ_IRT: {uid: scalar}<br/>M_ref: {uid: [seq_len]}"]]
    end
    
    subgraph "Embedding Layer"
        Tokens[["Interaction Tokens<br/>(q, r) pairs"]]
        Context_Emb["Context Embedding<br/>[B, L, d_model]"]
        Value_Emb["Value Embedding<br/>[B, L, d_model]"]
        Skill_Emb["Skill Embedding<br/>[B, L, d_model]"]
        Pos_Emb["Positional Embeddings<br/>[1, L, d_model]"]
        
        Context_Seq[["Context Sequence<br/>c = emb(q,r) + pos<br/>[B, L, d_model]"]]
        Value_Seq[["Value Sequence<br/>v = emb(r) + pos<br/>[B, L, d_model]"]]
    end
    
    subgraph "Transformer Encoder"
        direction TB
        
        subgraph "Context Stream Attention"
            Q_c["Q_context = Linear(c)<br/>[B, L, d_model]"]
            K_c["K_context = Linear(c)<br/>[B, L, d_model]"]
            V_c["V_context = Linear(c)<br/>[B, L, d_model]"]
            Attn_c["Multi-Head Attention<br/>softmax(QK^T/√d)V"]
            Out_c["h' = Attention(Q_c, K_c, V_c)<br/>[B, L, d_model]"]
        end
        
        subgraph "Value Stream Attention"
            Q_v["Q_value = Linear(v)<br/>[B, L, d_model]"]
            K_v["K_value = Linear(v)<br/>[B, L, d_model]"]
            V_v["V_value = Linear(v)<br/>[B, L, d_model]"]
            Attn_v["Multi-Head Attention<br/>softmax(QK^T/√d)V"]
            Out_v["v' = Attention(Q_v, K_v, V_v)<br/>[B, L, d_model]"]
        end
        
        FFN_C["Feed-Forward Context<br/>× N blocks"]
        FFN_V["Feed-Forward Value<br/>× N blocks"]
        
        Note_Stack["× N Transformer Blocks<br/>(N=8, d_model=256)"]
    end
    
    subgraph "Encoder Output"
        Final_h[["Knowledge State h<br/>(final context)<br/>[B, L, d_model]"]]
        Final_v[["Value State v<br/>(final value)<br/>[B, L, d_model]"]]
    end
    
    subgraph "Head 1: Performance Prediction (BCE)"
        Concat1["Concat[h, v, skill_emb]<br/>[B, L, 3·d_model]"]
        PredHead["MLP Prediction Head<br/>Linear → ReLU → Dropout → Linear"]
        Logits[["Logits<br/>[B, L]"]]
        BCEPred[["p_correct = σ(logits)<br/>[B, L]"]]
    end
    
    subgraph "Head 2: IRT-Based Mastery (Pluggable Reference Model)"
        Head2["IRT Mastery Estimator<br/>Inputs: h (knowledge state), q (skills)<br/>Outputs: M_IRT (mastery probabilities)<br/>[B, L] → [B, L]"]
        MasteryIRT[["M_IRT<br/>IRT-based Mastery<br/>[B, L] probabilities"]]
    end
    
    subgraph "Loss Computation (via Reference Model Interface)"
        direction LR
        
        L_BCE["l_bce<br/>BCE(p_correct, targets)<br/>Performance Loss"]
        
        L_21["l_21 (performance)<br/>BCE(M_IRT, M_ref)<br/>Prediction alignment"]
        
        L_22["l_22 (difficulty)<br/>MSE(β_learned[q], β_IRT[q])<br/>Difficulty regularization<br/>(always active)"]
        
        L_23["l_23 (ability)<br/>MSE(mean(θ_learned), θ_IRT)<br/>Ability alignment"]
        
        LambdaSchedule["λ(epoch) Warm-up<br/>λ = λ_target × min(1, epoch/warmup)<br/>λ_target=0.5, warmup=50"]
    end
    
    subgraph "Combined Loss (Single-Phase Training)"
        LTotal["L_total = (1-λ)×l_bce + c×l_22 + λ×(l_21 + l_23)<br/><br/>λ: interpretability weight (warm-up)<br/>c: stability regularization (fixed, c=0.01)"]
        Backprop["Backpropagation<br/>Updates: θ encoder, β embeddings,<br/>prediction head, encoder"]
    end
    
    %% Input to Embedding flow
    Input_q --> Tokens
    Input_r --> Tokens
    Tokens --> Context_Emb
    Tokens --> Value_Emb
    Input_q --> Skill_Emb
    
    Context_Emb --> Context_Seq
    Value_Emb --> Value_Seq
    Pos_Emb --> Context_Seq
    Pos_Emb --> Value_Seq
    
    %% Encoder processing - Context stream
    Context_Seq --> Q_c
    Context_Seq --> K_c
    Context_Seq --> V_c
    Q_c --> Attn_c
    K_c --> Attn_c
    V_c --> Attn_c
    Attn_c --> Out_c
    Out_c --> FFN_C
    
    %% Encoder processing - Value stream
    Value_Seq --> Q_v
    Value_Seq --> K_v
    Value_Seq --> V_v
    Q_v --> Attn_v
    K_v --> Attn_v
    V_v --> Attn_v
    Attn_v --> Out_v
    Out_v --> FFN_V
    
    %% Final encoder outputs
    FFN_C --> Final_h
    FFN_V --> Final_v
    
    %% Head 1 - Performance prediction
    Final_h --> Concat1
    Final_v --> Concat1
    Skill_Emb --> Concat1
    Concat1 --> PredHead --> Logits --> BCEPred
    
    %% Head 2 - IRT mastery inference
    Final_h --> Head2
    Skill_Emb --> Head2
    Head2 --> MasteryIRT
    
    %% Loss computation flows
    BCEPred --> L_BCE
    Ground_Truth_r --> L_BCE
    
    MasteryIRT --> L_21
    RefTargets --> L_21
    
    Head2 --> L_22
    RefTargets --> L_22
    
    Head2 --> L_23
    RefTargets --> L_23
    
    %% Loss aggregation with lambda schedule
    L_BCE --> LTotal
    L_21 --> LTotal
    L_22 --> LTotal
    L_23 --> LTotal
    LambdaSchedule --> LTotal
    
    LTotal --> Backprop
    
    %% Gradient flow (dotted lines)
    Backprop -.->|∂L/∂h| Final_h
    Backprop -.->|∂L/∂v| Final_v
    Backprop -.->|∂L/∂Head2| Head2
    
    %% Styling
    classDef input_style fill:#ffffff,stroke:#333333,stroke-width:2px
    classDef ref_style fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef emb_style fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef encoder_style fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef head1_style fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    classDef head2_style fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    classDef loss_style fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    classDef combined_style fill:#f3e5f5,stroke:#6a1b9a,stroke-width:4px
    
    class Input_q,Input_r,Ground_Truth_r input_style
    class RefTargets ref_style
    class Tokens,Context_Emb,Value_Emb,Skill_Emb,Pos_Emb,Context_Seq,Value_Seq emb_style
    class Q_c,K_c,V_c,Attn_c,Out_c,Q_v,K_v,V_v,Attn_v,Out_v,FFN_C,FFN_V,Note_Stack,Final_h,Final_v encoder_style
    class Concat1,PredHead,Logits,BCEPred head1_style
    class Head2,MasteryIRT head2_style
    class L_BCE,L_21,L_22,L_23,LambdaSchedule loss_style
    class LTotal,Backprop combined_style
```
</div>

### AKT

<div style="width: 1000px;">

```mermaid
graph TD
    subgraph "Input Layer"
        Input_q[["Questions q<br/>[B, L]"]]
        Input_r[["Responses r<br/>[B, L]"]]
        Input_pid[["Problem IDs pid<br/>[B, L]"]]
    end
    
    subgraph "Embedding Layer"
        Q_Emb["Question Embedding<br/>q_embed: [B, L, d_model]"]
        R_Emb["Response Embedding<br/>r_embed: [2] → [B, L, d_model]"]
        QA_Emb["Interaction Embedding<br/>qa = q_embed + r_embed<br/>[B, L, d_model]"]
        
        Diff_Param["Difficulty Parameter<br/>difficult_param(pid)<br/>[B, L, 1]"]
        Q_Diff_Emb["Question Difficulty Emb<br/>q_embed_diff(q)<br/>[B, L, d_model]"]
        QA_Diff_Emb["Interaction Difficulty Emb<br/>qa_embed_diff(r)<br/>[B, L, d_model]"]
        
        Final_Q[["Enhanced Q Embedding<br/>q_final = q_embed + uq·d_ct<br/>[B, L, d_model]"]]
        Final_QA[["Enhanced QA Embedding<br/>qa_final = qa_embed + uq·(h_rt+d_ct)<br/>[B, L, d_model]"]]
    end
    
    subgraph "Encoder: Blocks_1 (N blocks)"
        direction TB
        Enc_Attn["Multi-Head Self-Attention<br/>Q = K = V = Linear(qa_final)<br/>Attention(Q, K, V)<br/>with Monotonic Attention"]
        Enc_Norm1["Layer Norm"]
        Enc_FFN["Feed-Forward Network<br/>Linear → ReLU → Linear"]
        Enc_Norm2["Layer Norm"]
        Enc_Out[["Encoded QA: y^<br/>[B, L, d_model]"]]
    end
    
    subgraph "Knowledge Retriever: Blocks_2 (2N blocks)"
        direction TB
        
        subgraph "First Layer (Self-Attention on Q)"
            KR_Attn1["Multi-Head Self-Attention<br/>Query=Key=Value=x (questions)<br/>mask=1, apply_pos=False"]
            KR_Norm1["Layer Norm"]
            KR_Out1[["x^ (enhanced questions)<br/>[B, L, d_model]"]]
        end
        
        subgraph "Subsequent Layers (Cross-Attention)"
            KR_Attn2["Multi-Head Attention<br/>Query=x (questions)<br/>Key=Value=y^ (encoded QA)<br/>mask=0, apply_pos=True"]
            KR_Norm2a["Layer Norm"]
            KR_FFN["Feed-Forward Network"]
            KR_Norm2b["Layer Norm"]
            KR_Out2[["Knowledge State x<br/>[B, L, d_model]"]]
        end
    end
    
    subgraph "Prediction Head"
        Concat["Concat[x, q_final]<br/>[B, L, 2·d_model]"]
        FC1["Linear(2·d_model, 512) → ReLU → Dropout"]
        FC2["Linear(512, 256) → ReLU → Dropout"]
        FC3["Linear(256, 1)"]
        Sigmoid["Sigmoid"]
        Pred[["Predictions<br/>[B, L]"]]
    end
    
    subgraph "Loss"
        L_BCE["BCE Loss"]
        L_Reg["Rasch Regularization<br/>L_reg = ||uq||²"]
        L_Total["L_total = L_BCE + λ·L_reg"]
    end
    
    %% Input to Embedding flow
    Input_q --> Q_Emb
    Input_r --> R_Emb
    Q_Emb --> QA_Emb
    R_Emb --> QA_Emb
    
    %% Difficulty enhancement
    Input_pid --> Diff_Param
    Input_q --> Q_Diff_Emb
    Input_r --> QA_Diff_Emb
    
    Diff_Param --> Final_Q
    Q_Emb --> Final_Q
    Q_Diff_Emb --> Final_Q
    
    Diff_Param --> Final_QA
    QA_Emb --> Final_QA
    QA_Diff_Emb --> Final_QA
    Q_Diff_Emb --> Final_QA
    
    %% Encoder flow
    Final_QA --> Enc_Attn
    Enc_Attn --> Enc_Norm1
    Enc_Norm1 --> Enc_FFN
    Enc_FFN --> Enc_Norm2
    Enc_Norm2 --> Enc_Out
    
    %% Knowledge Retriever flow
    Final_Q --> KR_Attn1
    KR_Attn1 --> KR_Norm1
    KR_Norm1 --> KR_Out1
    
    KR_Out1 --> KR_Attn2
    Enc_Out --> KR_Attn2
    KR_Attn2 --> KR_Norm2a
    KR_Norm2a --> KR_FFN
    KR_FFN --> KR_Norm2b
    KR_Norm2b --> KR_Out2
    
    %% Prediction flow
    KR_Out2 --> Concat
    Final_Q --> Concat
    Concat --> FC1
    FC1 --> FC2
    FC2 --> FC3
    FC3 --> Sigmoid
    Sigmoid --> Pred
    
    %% Loss computation
    Pred --> L_BCE
    Diff_Param --> L_Reg
    L_BCE --> L_Total
    L_Reg --> L_Total
    
    %% Styling
    classDef input_style fill:#ffffff,stroke:#333333,stroke-width:2px
    classDef emb_style fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef encoder_style fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef retriever_style fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    classDef pred_style fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    classDef loss_style fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    
    class Input_q,Input_r,Input_pid input_style
    class Q_Emb,R_Emb,QA_Emb,Diff_Param,Q_Diff_Emb,QA_Diff_Emb,Final_Q,Final_QA emb_style
    class Enc_Attn,Enc_Norm1,Enc_FFN,Enc_Norm2,Enc_Out encoder_style
    class KR_Attn1,KR_Norm1,KR_Out1,KR_Attn2,KR_Norm2a,KR_FFN,KR_Norm2b,KR_Out2 retriever_style
    class Concat,FC1,FC2,FC3,Sigmoid,Pred pred_style
    class L_BCE,L_Reg,L_Total loss_style
```
</div>

**Key Features:**
- **Dual Encoding Streams**: Encodes past interactions (qa) separately from current questions (q)
- **Monotonic Attention**: Uses distance-based attention weighting with learnable γ parameters
- **Difficulty-Enhanced Embeddings**: Incorporates problem-specific difficulty via Rasch parameters
- **Knowledge Retriever**: Two-phase retrieval (self-attention on questions, then cross-attention with encoded interactions)
- **Rasch Regularization**: Penalizes large difficulty parameters to prevent overfitting
