## Approach

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

### iDKT

<div style="width: 1000px;">

```mermaid
graph TD
    subgraph "Input Layer"
        Input_q[["Questions q<br/>[B, L]"]]
        Input_r[["Responses r<br/>[B, L]"]]
        Input_pid[["Problem IDs pid<br/>[B, L]"]]
    end

    subgraph "Embedding Layer"
        Q_Emb["Question Embedding<br/>q_embed[q]: [B, L, d]"]
        R_Emb["Response Embedding<br/>r_embed[r]: [B, L, d]"]
        QA_Emb["Interaction Embedding<br/>qa = q_embed + r_embed<br/>[B, L, d]"]

        Diff_Param["Difficulty Parameter<br/>uq = difficult_param[pid]<br/>[B, L, 1]"]
        Q_Diff_Emb["q_embed_diff[q]<br/>[B, L, d]"]
        QA_Diff_Emb["qa_embed_diff[r]<br/>[B, L, d]"]

        Final_Q[["Enhanced Q Embedding<br/>x (Questions)<br/>[B, L, d]"]]
        Final_QA[["Enhanced QA Embedding<br/>y (Interactions)<br/>[B, L, d]"]]
    end

    subgraph "Encoder: N Blocks"
        direction TB

        subgraph "Enc_MHA [Multi-Head Self-Attention]"
            direction TB
            E_Split["Embedding split 8 segments"]

            subgraph "E_Heads [8 Parallel Attention Heads]"
                direction LR
                E_H1["Head 1<br/>Attn(Q1,K1,V1)<br/>γ1 (Short-term?)"]
                E_H2["Heads 2..7<br/>...<br/>Diverse γ"]
                E_H8["Head 8<br/>Attn(Q8,K8,V8)<br/>γ8 (Long-term?)"]
            end

            E_Concat["Concatenate<br/>[B, L, d]"]
            E_Wo["Linear Output W_o"]
        end

        Enc_Norm1["Layer Norm + Residual"]
        Enc_FFN["Feed-Forward Network"]
        Enc_Norm2["Layer Norm + Residual"]
        Enc_Out[["Encoded Interactions: y^<br/>[B, L, d]"]]
    end

    subgraph "Decoder (Knowledge Retriever): 2N Blocks"
        direction TB

        subgraph "Odd Layers: Self-Attention (Questions)"
            direction TB

            subgraph "KR1_MHA [Multi-Head Self-Attention]"
                direction TB
                KR1_Split["Embedding split 8 segments"]

                subgraph "KR1_Heads [8 Parallel Heads]"
                    direction LR
                    KR1_H1["Head 1<br/>γ1"]
                    KR1_H2["Heads 2..7"]
                    KR1_H8["Head 8<br/>γ8"]
                end

                KR1_Concat["Concatenate"]
                KR1_Wo["Linear Output W_o"]
            end

            KR_Norm1["Layer Norm + Residual"]
        end

        subgraph "Even Layers: Cross-Attention"
            direction TB

            subgraph "KR2_MHA [Multi-Head Cross-Attention]"
                direction TB
                KR2_Proj_Q["Linear W_q (from Questions)"]
                KR2_Proj_KV["Linear W_k, W_v (from Encoded y^)"]

                subgraph "KR2_Heads [8 Parallel Heads]"
                    direction LR
                    KR2_H1["Head 1<br/>γ1"]
                    KR2_H2["Heads 2..7"]
                    KR2_H8["Head 8<br/>γ8"]
                end

                KR2_Concat["Concatenate"]
                KR2_Wo["Linear Output W_o"]
            end

            KR_Norm2a["Layer Norm + Residual"]
            KR_FFN["Feed-Forward Network"]
            KR_Norm2b["Layer Norm + Residual"]
        end

        KR_Out[["Knowledge State: x^<br/>[B, L, d]"]]
    end

    subgraph "Prediction Head"
        Concat["Concat[x^, x]<br/>[B, L, 2d]"]
        mlp_layers["MLP Layers"]
        Pred[["Predictions p<br/>[B, L]"]]
    end

    subgraph "Loss"
        L_BCE["L_bce"]
        L_Reg["L_reg (Rasch)"]
        L_Total["L_total"]
    end

    %% Wiring - Input/Emb
    Input_q --> Q_Emb
    Input_r --> R_Emb
    Q_Emb --> QA_Emb
    R_Emb --> QA_Emb

    %% Difficulty enhancement
    Input_pid --> Diff_Param
    Input_q --> Q_Diff_Emb
    Input_r --> QA_Diff_Emb

    Q_Emb --> Final_Q
    Diff_Param --> Final_Q
    Q_Diff_Emb --> Final_Q

    QA_Emb --> Final_QA
    Diff_Param --> Final_QA
    QA_Diff_Emb --> Final_QA
    Q_Diff_Emb --> Final_QA

    %% Encoder Wiring
    Final_QA --> E_Split
    E_Split --> E_H1 & E_H2 & E_H8
    E_H1 & E_H2 & E_H8 --> E_Concat
    E_Concat --> E_Wo

    %% Skip connection around attention
    Final_QA -.-> Enc_Norm1
    E_Wo --> Enc_Norm1

    Enc_Norm1 --> Enc_FFN
    Enc_Norm1 -.-> Enc_Norm2
    Enc_FFN --> Enc_Norm2
    Enc_Norm2 --> Enc_Out

    %% Decoder Wiring - Odd (Self Attn)
    Final_Q --> KR1_Split
    KR1_Split --> KR1_H1 & KR1_H2 & KR1_H8 --> KR1_Concat --> KR1_Wo

    Final_Q -.-> KR_Norm1
    KR1_Wo --> KR_Norm1

    %% Decoder Wiring - Even (Cross Attn)
    KR_Norm1 --> KR2_Proj_Q
    Enc_Out --> KR2_Proj_KV

    KR2_Proj_Q --> KR2_H1 & KR2_H2 & KR2_H8
    KR2_Proj_KV --> KR2_H1 & KR2_H2 & KR2_H8

    KR2_H1 & KR2_H2 & KR2_H8 --> KR2_Concat --> KR2_Wo

    KR_Norm1 -.-> KR_Norm2a
    KR2_Wo --> KR_Norm2a

    KR_Norm2a --> KR_FFN
    KR_Norm2a -.-> KR_Norm2b
    KR_FFN --> KR_Norm2b
    KR_Norm2b --> KR_Out

    %% Prediction
    KR_Out --> Concat
    Final_Q --> Concat
    Concat --> mlp_layers --> Pred

    %% Loss (Abstracted)
    Pred --> L_BCE
    Diff_Param --> L_Reg
    L_BCE & L_Reg --> L_Total

    %% Styling
    classDef plain fill:#fff,stroke:#333,stroke-width:1px;
    classDef emb fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef attn fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef heads fill:#c8e6c9,stroke:#2e7d32,stroke-dasharray: 5 5;

    class Input_q,Input_r,Input_pid plain
    class Final_Q,Final_QA,Enc_Out,KR_Out emb
    class E_Proj,E_Concat,E_Wo,E_Split,KR1_Split,KR1_Concat,KR1_Wo,KR2_Proj_Q,KR2_Proj_KV,KR2_Concat,KR2_Wo attn
    class E_H1,E_H2,E_H8,KR1_H1,KR1_H2,KR1_H8,KR2_H1,KR2_H2,KR2_H8 heads
```

</div>

**Key Features:**

- **Context-Aware Representations** (Paper §3.1):

  - Question encoder produces contextualized question embeddings: `ˆxt = fenc1(x1,...,xt)`
  - Knowledge encoder produces contextualized interaction embeddings: `ˆyt-1 = fenc2(y1,...,yt-1)`
  - Reflects that learner comprehension and knowledge acquisition depend on personal response history
  - Two learners with different past sequences understand the same question differently

- **Monotonic Attention Mechanism** (Paper §3.2, Eq. 1):

  - Formula: `st,τ = exp(-θ·d(t,τ)) · q⊺tkτ/√Dk` where θ > 0 is learnable decay rate
  - Distance-based weighting: `scores × exp(γ × distance)` where γ is learnable per head
  - Temporal decay: past experiences from too long ago are less relevant
  - Intuition: "past experiences i) on unrelated concepts and ii) that are from too long ago are not likely to be highly relevant"
  - Applied in both Encoder (self-attention) and Knowledge Retriever

- **Context-Aware Distance Measure** (Paper §3.2):

  - Adjusts temporal distance based on concept similarity: `d(t,τ) = |t-τ| · Σ γt,t'`
  - Example: If learner practices Venn Diagram repeatedly then Prime Numbers, distance accounts for concept relevance
  - Previous Prime Numbers practice at t=1 remains relevant despite intermediate Venn Diagram practices
  - Uses softmax function to weight consecutive time indices by concept relatedness

- **Rasch Model-Based Embeddings** (Paper §3.4):

  - Questions: `xt = cct + µqt·dct` where cct is concept embedding, µqt is difficulty scalar
  - Interactions: `yt = e(ct,rt) + µqt·f(ct,rt)` where `e(ct,rt) = cct + grt`
  - Implementation formulas:
    - `x = q_embed + uq × q_embed_diff`
    - `y = qa_embed + uq × (qa_embed_diff + q_embed_diff)`
  - Balances modeling individual question differences with avoiding overparameterization
  - Total parameters: (C+2)D + Q instead of QD (where C≪Q and D≫1)
  - Regularized via L₂ penalty: `L_reg = ||uq||²`

- **Knowledge Retriever** (Paper §3.1, §3.2):

  - Architecture: 2N blocks (default 8) with alternating pattern
  - **Odd layers**: Self-attention on questions Q=K=V=x (no FFN)
  - **Even layers**: Cross-attention Q=x (questions), K=V=ˆy (encoded interactions) + FFN
  - Uses question embeddings for both queries and keys (more effective than SAKT's approach)
  - Outputs context-aware knowledge state: `ht = fkr(ˆx1,...,ˆxt, ˆy1,...,ˆyt-1)`
  - First question (t=0) receives zero-padded attention → no historical information available

- **Multi-Head Attention** (Paper §3.2):

  - H independent attention heads (default H=8)
  - Each head has its own learnable decay rate θ
  - Enables summarizing past performance at multiple time scales
  - Output: concatenate (Dv·H)×1 vector, pass to next layer

    ```
    The Multi-Head Attention mechanism in iDKT (inherited from AKT) typically consists of **8 parallel heads** (default configuration).

    - Structure: Each head operates independently on a subspace of the embedding dimension ($d_{model}/8$).
    - Splitting: The full embedding vector (e.g., $d_{model}=256$) is split into 8 smaller segments of size 32 ($256 \div 8 = 32$). Each head only sees this 32-dimensional "slice" of the data.
    - Independence: The attention calculation—queries, keys, values, and the learnable decay $\gamma$—happens separately and in parallel for each head. Head 1 calculates its own weighted sum without knowing what Head 2 is doing.
    - Specialization: This allows each head to specialize. One might focus on short-term mastery (via a high $\gamma$), while another tracks long-term concept retention (via a low $\gamma$).
    - Recombination: After processing, the 8 outputs are concatenated back to form the full 256-dimensional vector, combining the specialized insights from all heads.
    ```

- **Learnable Decay**: Crucially, each head $h$ possesses a unique, learnable decay parameter $\gamma_h$.
- $\gamma_h$ controls the rate of exponential decay for the monotonic attention mechanism in that specific head.
- Heads with **large $\gamma$** values decay information rapidly, focusing on **short-term** (recent) context.
- Heads with **small $\gamma$** values decay information slowly, allowing the model to retain **long-term** context.
- **Diversity**: This diversity allows the iDKT model to simultaneously attend to immediate prerequisites and foundational concepts learned much earlier in the sequence.

- **Response Prediction Model** (Paper §3.3):

- Input: Concatenates retrieved knowledge ht and current question embedding xt
- Architecture: Fully-connected network + sigmoid
- Implementation: Linear(2d, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 1) → Sigmoid
- Loss: Binary cross-entropy `ℓ = Σi Σt -(rit log ˆrit + (1-rit)log(1-ˆrit))`
- End-to-end training of all parameters

- **Mask Semantics**:

- `mask=1`: Causal masking in encoders (can see current + past positions)
- `mask=0`: Strict past-only in Knowledge Retriever (current position masked) + zero-padding for first row

- **Architecture Size**: N=4 encoder blocks, 2N=8 retriever blocks, d_model=256, H=8 heads (default)
