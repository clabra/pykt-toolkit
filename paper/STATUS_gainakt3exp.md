# GainAKT3Exp Model Status

## Model 

The model implementations is in ´pykt/models/gainakt3_exp.py´.  

## Architecture

The diagram below is described in detail in `assistant/gainakt3exp_architecture_approach.md`. 

It illustrates the Learning Gains approach based on an Encoder-only Transformer, augmented with these features:
- **Green components**: Core augmented architecture (Skill Embedding, Dynamic Value Stream, Constraint Losses, Monitoring)
- **Blue components**: Recursive Mastery Accumulation (deterministic temporal constraint: mastery_{t+1} = mastery_t + α·ReLU(gain_t))
- **Orange components**: Semantic modules (Alignment, Global Alignment, Retention, Lag Gains) that enable interpretability recovery
- **Red components**: DEACTIVATED features - **Gain Projection Head** (Values used directly as gains), Intrinsic gain attention mode (intrinsic_gain_attention=false), Mastery-Performance loss (mastery_performance_loss_weight=0.0), Gain-Performance loss (gain_performance_loss_weight=0.0). The Gain Projection Head is architecturally bypassed - encoder Values flow directly to the Recursive Mastery Accumulation as learning gains, enabling the attention mechanism to learn gain representations directly.
- **Yellow diamonds (OR gates)**: Mutually exclusive paths - Standard Mode (via projection heads) OR Intrinsic Mode (via attention weights), never both simultaneously
- **Circles (Hubs)**: Convergence/distribution points where multiple data flows aggregate and route to multiple outputs

**GainAKT3Exp Key Innovation**: Unlike GainAKT2Exp, the Gain Projection Head is deactivated. Encoder **Values** (v) are treated directly as **learning gains**, learned via the attention mechanism. This architectural simplification enables:
- Direct interpretability: attention values represent gains without intermediate projections
- Reduced architectural complexity: one fewer projection head
- Enhanced semantic alignment: values optimize directly for gain-related losses

```mermaid
graph TD
    subgraph "Input Layer"
        direction LR
        Input_q["Input Questions (q)<br/>Shape: [B, L]"]
        Input_r["Input Responses (r)<br/>Shape: [B, L]"]
        Ground_Truth["Ground Truth Responses"]
    end

    subgraph "Tokenization & Embedding"
        direction TB

        
        Tokens["Interaction Tokens<br/>(q + num_c * r)<br/>Shape: [B, L]"]
        
        Context_Emb["Context Embedding Table"]
        Value_Emb["Value Embedding Table"]
        Skill_Emb["Skill Embedding Table"]

        Tokens --> Context_Emb
        Tokens --> Value_Emb
        Input_q --> Skill_Emb

        Context_Seq["Context Sequence<br/>Shape: [B, L, D]"]
        Value_Seq["Value Sequence<br/>Shape: [B, L, D]"]
        Pos_Emb["Positional Embeddings<br/>Shape: [B, L, D]"]
        
        Context_Emb --> Context_Seq
        Value_Emb --> Value_Seq

        Context_Seq_Pos["Context + Positional<br/>Shape: [B, L, D]"]
        Value_Seq_Pos["Value + Positional<br/>Shape: [B, L, D]"]
        
        Context_Seq --"Add"--> Context_Seq_Pos
        Pos_Emb --"Add"--> Context_Seq_Pos
        Value_Seq --"Add"--> Value_Seq_Pos
        Pos_Emb --"Add"--> Value_Seq_Pos
    end

    Input_q --> Tokens
    Input_r --> Tokens

    subgraph "Dynamic Encoder Block"
        direction TB
        
        Encoder_Input_Context["Input: Context Sequence<br/>[B, L, D]"]
        Encoder_Input_Value["Input: Value Sequence<br/>[B, L, D]"]

        subgraph "Attention Mechanism"
            direction TB
            
            Attn_Input_Context["Input: Context<br/>[B, L, D]"]
            Attn_Input_Value["Input: Value<br/>[B, L, D]"]

            Proj_Q["Q = Linear(Context)<br/>[B, H, L, Dk]"]
            Proj_K["K = Linear(Context)<br/>[B, H, L, Dk]"]
            Proj_V["V = Linear(Value)<br/>[B, H, L, Dk]"]
            
            Attn_Input_Context --> Proj_Q
            Attn_Input_Context --> Proj_K
            Attn_Input_Value --> Proj_V

            Scores["Scores = $\\frac{Q \\cdot K^T}{\\sqrt{D_k}}$<br/>[B, H, L, L]"]
            Proj_Q --> Scores
            Proj_K --> Scores
            
            Weights["Weights = $\\text{softmax}(\\text{Scores})$<br/>[B, H, L, L]"]
            Scores --> Weights

            Attn_Output_Heads["Attn Output (Heads)<br/>[B, H, L, Dk]"]
            Weights --> Attn_Output_Heads
            Proj_V --> Attn_Output_Heads

            Attn_Output["Reshaped Attn Output<br/>[B, L, D]"]
            Attn_Output_Heads --> Attn_Output
        end

        Encoder_Input_Context --> Attn_Input_Context
        Encoder_Input_Value --> Attn_Input_Value

        AddNorm_Ctx["Add & Norm (Context)"]
        Attn_Output --> AddNorm_Ctx
        Encoder_Input_Context --"Residual"--> AddNorm_Ctx

        AddNorm_Val["Add & Norm (Value)<br/>"]
        Attn_Output --> AddNorm_Val
        Encoder_Input_Value --"Residual"--> AddNorm_Val

        FFN["Feed-Forward Network"]
        AddNorm_Ctx --> FFN
        
        AddNorm2["Add & Norm"]
        FFN --> AddNorm2
        AddNorm_Ctx --"Residual"--> AddNorm2
        
        Encoder_Output_Ctx["Output: Context (h)<br/>[B, L, D]"]
        AddNorm2 --> Encoder_Output_Ctx

        Encoder_Output_Val["Output: Value (v)<br/>[B, L, D]"]
        AddNorm_Val --> Encoder_Output_Val
    end

    Context_Seq_Pos --> Encoder_Input_Context
    Value_Seq_Pos --> Encoder_Input_Value

    subgraph "Prediction Head"
        direction TB
        
        Pred_Input_h["Input: Knowledge State (h)<br/>[B, L, D]"]
        Pred_Input_v["Input: Value State (v)<br/>[B, L, D]"]
        Pred_Input_s["Input: Target Skill (s)<br/>[B, L, D]"]

        Concat["Concatenate<br/>[h, v, s]<br/>[B, L, 3*D]"]
        MLP["MLP Head"]
        Sigmoid["Sigmoid"]
        
        Pred_Input_h --> Concat
        Pred_Input_v --> Concat
        Pred_Input_s --> Concat
        Concat --> MLP
        MLP --> Sigmoid
    end
    
    Encoder_Output_Ctx --> Pred_Input_h
    Encoder_Output_Val --> Pred_Input_v
    Skill_Emb --"Lookup"--> Pred_Input_s

    subgraph "Final Output"
        direction LR
        Predictions["Predictions<br/>[B, L]"]
    end

    Sigmoid --> Predictions

    %% Projection Heads (Baseline Mode)
    Proj_Mastery["Mastery Projection Head<br/>Linear(D, num_skills)<br/>(Baseline Mode)"]
    Proj_Gain["Gain Projection Head<br/>Linear(D, num_skills)<br/>(DEACTIVATED - Values used directly)"]
    
    %% Intrinsic Mode Components (Red - DEACTIVATED)
    Attention_Derived_Gains["Attention-Derived Gains<br/>Cumulative mastery from<br/>attention weights<br/>(DEACTIVATED: intrinsic_gain_attention=false)"]
    
    %% OR Gates showing mutually exclusive paths
    Mastery_OR{"OR<br/>Mastery<br/>Source"}
    Gain_OR{"OR<br/>Gain<br/>Source"}
    
    %% Baseline Mode Paths
    Encoder_Output_Ctx -->|"Standard Mode"| Proj_Mastery
    Encoder_Output_Val -->|"BYPASSED - Direct to Gains"| Proj_Gain
    Proj_Mastery --> Mastery_OR
    Proj_Gain -.->|"DEACTIVATED"| Gain_OR
    
    %% GainAKT3Exp Direct Path: Values → Gains (KEY INNOVATION)
    Encoder_Output_Val -->|"Direct Gains (GainAKT3Exp)"| Gain_OR
    
    %% Intrinsic Mode Paths
    Weights -->|"Intrinsic Mode"| Attention_Derived_Gains
    Attention_Derived_Gains -->|"bypasses heads"| Mastery_OR
    Attention_Derived_Gains -->|"bypasses heads"| Gain_OR
    
    %% Output after OR gates (mutually exclusive sources converge)
    Projected_Mastery_Output["Projected Mastery<br/>[B, L, num_skills]"]
    Projected_Gain_Output["Encoder Values (Direct Gains)<br/>[B, L, D]<br/>(GainAKT3Exp: bypass projection)"]
    
    Mastery_OR --> Projected_Mastery_Output
    Gain_OR --> Projected_Gain_Output

    %% Recursive Mastery Accumulation (Key Architectural Constraint)
    subgraph "Recursive Mastery Accumulation"
        direction TB
        
        Gain_Input["Gains at t<br/>gain_t[c]<br/>[B, num_c]"]
        Mastery_Prev["Previous Mastery<br/>mastery_{t-1}[c]<br/>[B, num_c]"]
        
        ReLU_Op["ReLU<br/>(non-negativity)"]
        Scale_Op["× α<br/>(α=0.1)"]
        Sum_Op["+ (accumulation)<br/>mastery_t = mastery_{t-1} + α·ReLU(gain_t)"]
        Clamp_Op["Clamp[0,1]<br/>(bounded mastery)"]
        
        Mastery_Current["Current Mastery<br/>mastery_t[c]<br/>[B, num_c]"]
        
        Gain_Input --> ReLU_Op
        ReLU_Op --> Scale_Op
        Scale_Op --> Sum_Op
        Mastery_Prev --> Sum_Op
        Sum_Op --> Clamp_Op
        Clamp_Op --> Mastery_Current
        
        %% Temporal feedback loop
        Mastery_Current -.->|"becomes mastery_{t-1}<br/>for next timestep"| Mastery_Prev
    end
    
    %% Connect to recursive accumulation
    Projected_Gain_Output -->|"gain_t"| Gain_Input
    Mastery_Current -->|"accumulated<br/>mastery"| Projected_Mastery_Output

    %% Circle Connectors (Aggregation/Distribution Hubs)
    Mastery_Hub(("Mastery<br/>Hub"))
    Gain_Hub(("Gain<br/>Hub"))
    Encoder_Hub(("Encoder<br/>Hub"))
    Pred_Hub(("Predictions<br/>Hub"))
    
    Projected_Mastery_Output --> Mastery_Hub
    Projected_Gain_Output --> Gain_Hub
    Encoder_Output_Ctx --> Encoder_Hub
    Encoder_Output_Val --> Encoder_Hub
    Predictions --> Pred_Hub

    %% Semantic Feedback Loop (orange)
    Global_Alignment["Global Alignment Pass<br/>population coherence"]
    Residual_Alignment["Residual Alignment<br/>variance capture"]
    
    Mastery_Hub -->|"Global Align"| Global_Alignment
    Global_Alignment --> Residual_Alignment
    Residual_Alignment -.feedback.-> Projected_Mastery_Output

    %% Loss Framework
    subgraph "Loss Framework"
        direction LR
        
        subgraph "Primary Loss"
            BCE_Loss["BCE Loss"]
        end
        
        subgraph "Constraint Losses (Green)"
            direction TB
            Monotonicity_Loss["Monotonicity"]
            Mastery_Perf_Loss["Mastery-Perf<br/>(DEACTIVATED: weight=0.0)"]
            Gain_Perf_Loss["Gain-Perf<br/>(DEACTIVATED: weight=0.0)"]
            Sparsity_Loss["Sparsity"]
            Consistency_Loss["Consistency"]
            NonNeg_Loss["Non-Negativity"]
        end
        
        subgraph "Semantic Losses (Orange)"
            direction TB
            Alignment_Loss["Local Alignment"]
            Retention_Loss["Retention"]
            Lag_Gain_Loss["Lag Gain"]
        end
        
        Total_Loss["Total Loss<br/>BCE + Constraints + Semantics<br/>Warmup & Share Cap Scheduling"]
    end

    %% Connections via Diamond Hubs
    Pred_Hub -->|"BCE"| BCE_Loss
    Ground_Truth --> BCE_Loss

    Mastery_Hub -->|"Monotonicity"| Monotonicity_Loss
    Mastery_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss
    Mastery_Hub -->|"Consistency"| Consistency_Loss
    Mastery_Hub -->|"Retention"| Retention_Loss
    
    Gain_Hub -->|"Gain-Perf"| Gain_Perf_Loss
    Gain_Hub -->|"Sparsity"| Sparsity_Loss
    Gain_Hub -->|"Consistency"| Consistency_Loss
    Gain_Hub -->|"NonNeg"| NonNeg_Loss
    Gain_Hub -->|"Lag"| Lag_Gain_Loss
    
    Pred_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss
    Pred_Hub -->|"Gain-Perf"| Gain_Perf_Loss
    
    Encoder_Hub -->|"Alignment"| Alignment_Loss
    Pred_Hub -->|"Alignment"| Alignment_Loss

    %% All losses to Total
    BCE_Loss --> Total_Loss
    Monotonicity_Loss --> Total_Loss
    Mastery_Perf_Loss --> Total_Loss
    Gain_Perf_Loss --> Total_Loss
    Sparsity_Loss --> Total_Loss
    Consistency_Loss --> Total_Loss
    NonNeg_Loss --> Total_Loss
    Alignment_Loss --> Total_Loss
    Retention_Loss --> Total_Loss
    Lag_Gain_Loss --> Total_Loss

    %% Monitoring
    Monitor_Hub(("Monitor<br/>Inputs"))
    Monitor_Hook["Interpretability Monitor<br/>Real-time Analysis"]
    
    Mastery_Hub -->|"to Monitor"| Monitor_Hub
    Gain_Hub -->|"to Monitor"| Monitor_Hub
    Pred_Hub -->|"to Monitor"| Monitor_Hub
    Monitor_Hub -->|"Monitor Output"| Monitor_Hook

    %% Styling
    classDef new_component fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef semantic_component fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    classDef intrinsic_component fill:#ffcdd2,stroke:#c62828,stroke-width:3px,stroke-dasharray:5 5,font-style:italic
    classDef or_gate fill:#fff59d,stroke:#f57f17,stroke-width:3px
    classDef accumulation_component fill:#b3e5fc,stroke:#0277bd,stroke-width:3px
    
    %% Individual hub colors with distinct visual styles
    classDef mastery_hub fill:#e8f5e8,stroke:#00ff00,stroke-width:4px
    classDef gain_hub fill:#e8f5e8,stroke:#008800,stroke-width:4px
    classDef encoder_hub fill:#fff3e0,stroke:#ffa500,stroke-width:4px
    classDef pred_hub fill:#e3f2fd,stroke:#888888,stroke-width:4px
    classDef monitor_hub fill:#f3e5f5,stroke:#800080,stroke-width:4px

    class Proj_Mastery,Ground_Truth,Skill_Emb,BCE_Loss,Monotonicity_Loss,Sparsity_Loss,Consistency_Loss,NonNeg_Loss,Total_Loss,Monitor_Hook new_component
    class Proj_Gain,Projected_Gain_Output,Attention_Derived_Gains,Mastery_Perf_Loss,Gain_Perf_Loss intrinsic_component
    class Projected_Mastery_Output new_component
    class Gain_Input,Mastery_Prev,ReLU_Op,Scale_Op,Sum_Op,Clamp_Op,Mastery_Current accumulation_component
    class Alignment_Loss,Global_Alignment,Residual_Alignment,Retention_Loss,Lag_Gain_Loss semantic_component
    class Mastery_OR,Gain_OR or_gate
    
    class Mastery_Hub mastery_hub
    class Gain_Hub gain_hub
    class Encoder_Hub encoder_hub
    class Pred_Hub pred_hub
    class Monitor_Hub monitor_hub

    %% Link Styling - Match output lines to hub colors
    %% NOTE: Mermaid does not support class-based link styling; manual indexing required
    
    %% Pred_Hub outputs (black) - ALL 5 connections from Pred_Hub styled blue
    %% Pred_Hub -->|"BCE"| BCE_Loss (line 224)
    %% Pred_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss (line 238)
    %% Pred_Hub -->|"Gain-Perf"| Gain_Perf_Loss (line 239)
    %% Pred_Hub -->|"Alignment"| Alignment_Loss (line 242)
    %% Pred_Hub -->|"to Monitor"| Monitor_Hub (line 262)
    linkStyle 60 stroke:#888888,stroke-width:3px
    linkStyle 71 stroke:#888888,stroke-width:3px
    linkStyle 72 stroke:#888888,stroke-width:3px
    linkStyle 74 stroke:#888888,stroke-width:3px
    linkStyle 87 stroke:#888888,stroke-width:3px
    
    %% Mastery_Hub outputs (red) - ALL 6 connections from Mastery_Hub styled red
    %% Mastery_Hub -->|"Global Align"| Global_Alignment (line 195)
    %% Mastery_Hub -->|"Monotonicity"| Monotonicity_Loss (line 227)
    %% Mastery_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss (line 228)
    %% Mastery_Hub -->|"Consistency"| Consistency_Loss (line 229)
    %% Mastery_Hub -->|"Retention"| Retention_Loss (line 230)
    %% Mastery_Hub -->|"to Monitor"| Monitor_Hub (line 260)
    linkStyle 57 stroke:#00ff00,stroke-width:3px
    linkStyle 62 stroke:#00ff00,stroke-width:3px
    linkStyle 63 stroke:#00ff00,stroke-width:3px
    linkStyle 64 stroke:#00ff00,stroke-width:3px
    linkStyle 65 stroke:#00ff00,stroke-width:3px
    linkStyle 85 stroke:#00ff00,stroke-width:3px
    
    %% Gain_Hub outputs (green) - ALL 6 connections from Gain_Hub styled green
    %% Gain_Hub -->|"Gain-Perf"| Gain_Perf_Loss (line 232)
    %% Gain_Hub -->|"Sparsity"| Sparsity_Loss (line 233)
    %% Gain_Hub -->|"Consistency"| Consistency_Loss (line 234)
    %% Gain_Hub -->|"NonNeg"| NonNeg_Loss (line 235)
    %% Gain_Hub -->|"Lag"| Lag_Gain_Loss (line 236)
    %% Gain_Hub -->|"to Monitor"| Monitor_Hub (line 261)
    linkStyle 66 stroke:#008800,stroke-width:3px
    linkStyle 67 stroke:#008800,stroke-width:3px
    linkStyle 68 stroke:#008800,stroke-width:3px
    linkStyle 69 stroke:#008800,stroke-width:3px
    linkStyle 70 stroke:#008800,stroke-width:3px
    linkStyle 86 stroke:#008800,stroke-width:3px
    
    %% Encoder_Output_Ctx outputs (purple) - ALL 3 connections from Encoder_Output_Ctx styled purple
    %% Encoder_Output_Ctx --> Pred_Input_h (line 135)
    %% Encoder_Output_Ctx --> Proj_Mastery (line 157)
    %% Encoder_Output_Ctx --> Encoder_Hub (line 186)
    linkStyle 38 stroke:#800080,stroke-width:3px
    linkStyle 42 stroke:#800080,stroke-width:3px
    linkStyle 54 stroke:#800080,stroke-width:3px
    
    %% Encoder_Output_Val outputs (pink) - ALL 3 connections from Encoder_Output_Val styled pink
    %% Encoder_Output_Val --> Pred_Input_v (line 136)
    %% Encoder_Output_Val --> Proj_Gain (line 158)
    %% Encoder_Output_Val --> Encoder_Hub (line 187)
    linkStyle 39 stroke:#ff69b4,stroke-width:3px
    linkStyle 43 stroke:#ff69b4,stroke-width:3px
    linkStyle 55 stroke:#ff69b4,stroke-width:3px
    
    %% Encoder_Hub outputs (orange) - 1 connection from Encoder_Hub styled orange
    %% Encoder_Hub -->|"Alignment"| Alignment_Loss (line 241)
    linkStyle 73 stroke:#ffa500,stroke-width:3px
    
    %% Monitor_Hub output (purple)
    linkStyle 81 stroke:#800080,stroke-width:3px
```

**Tensor Dimensions Legend:**
- **B**: Batch size (number of students in a training/evaluation batch)
- **L**: Sequence length (maximum number of interactions per student)
- **D**: Model dimension (hidden size of transformer layers, e.g., 256)
- **num_c**: Number of concepts/skills in the dataset (e.g., 100 for ASSIST2015)
- **num_q**: Number of questions in the dataset (varies by dataset)
- **H**: Number of attention heads in multi-head attention

### Key Architectural Feature: Recursive Mastery Accumulation

The **blue subgraph** in the diagram above illustrates a critical architectural constraint that enforces interpretability-by-design. Unlike black-box models where knowledge states are opaque, our architecture implements a **deterministic recursive accumulation** mechanism:

$$\text{mastery}_{t+1}^{(c)} = \text{mastery}_t^{(c)} + \alpha \cdot \text{ReLU}(\text{gain}_t^{(c)})$$

This is implemented in the model's forward pass (`gainakt3_exp.py` lines 145, 162):

```python
accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
```

**Input/Output Dimensions:**
- **Input `projected_mastery[:, t-1, :]`**: Previous mastery levels, shape `[B, num_c]` (mastery for all concepts at time t-1)
- **Input `projected_gains[:, t, :]`**: Current learning gains, shape `[B, num_c]` (gains for all concepts at time t)
- **Output `projected_mastery[:, t, :]`**: Updated mastery levels, shape `[B, num_c]` (mastery for all concepts at time t)
- **Scaling factor α**: 0.1 (controls accumulation rate to prevent runaway growth)

**Key Properties:**
1. **Non-negative gains**: ReLU ensures learning gains ≥ 0 (no unlearning)
2. **Monotonic mastery**: Mastery can only increase or stay constant over time
3. **Bounded accumulation**: Clamp keeps mastery in [0,1] range (probabilistic interpretation)
4. **Transparent causality**: Each mastery state is the sum of all previous learning gains
5. **Evidence-based confidence**: More interactions ($n_t^{(c)}$) → more accumulated gains → higher confidence

This architectural constraint makes the relationship between evidence accumulation and knowledge states **explicit and interpretable**, supporting the confidence interval estimation methods described in the paper (Monte Carlo Dropout, Attention Entropy, Evidence Accumulation).

### GainAKT3Exp Architectural Innovation: Values as Direct Gains

**GainAKT3Exp** introduces a fundamental architectural simplification compared to GainAKT2Exp:

**GainAKT2Exp Flow:**
```
Encoder Values (v) → Gain Projection Head → Projected Gains → Recursive Mastery
```

**GainAKT3Exp Flow:**
```
Encoder Values (v) → Direct Gains → Recursive Mastery
               └──────────────────→ Gain Hub (for losses)
```

**Key Differences:**

1. **Direct Gain Representation**: Encoder attention Values are treated directly as learning gains without intermediate projection:
   - **Eliminated**: `Gain Projection Head` Linear(D, num_skills)
   - **New**: Values maintain dimension `[B, L, D]` as gains (vs. projected `[B, L, num_skills]`)
   
2. **Attention-Learned Gains**: The attention mechanism learns to produce Values that directly represent meaningful learning gains:
   - Values optimize for gain-related constraints (sparsity, non-negativity, consistency)
   - Attention learns gain semantics intrinsically (not via post-hoc projection)
   - Simpler architecture with fewer trainable parameters

3. **Dual Routing**: Same Value output serves two purposes:
   - **Mastery Accumulation**: `mastery_t = mastery_{t-1} + α·ReLU(values_t)`
   - **Loss Computation**: Gain Hub routes Values to constraint losses

**Hypothesized Benefits:**
- **Enhanced Interpretability**: Attention values have direct semantic meaning (gains), no projection indirection
- **Stronger Constraints**: Gain-related losses optimize attention directly, not projection weights
- **Architectural Simplicity**: One fewer component to tune/debug
- **Potential Performance**: Direct optimization may improve mastery correlation

**Implementation Status**: Currently GainAKT3Exp is initialized identical to GainAKT2Exp (training baseline). The architectural modification to bypass the Gain Projection Head and use Values directly will be implemented after baseline training completes.

## Implementation Summary

The `pykt/models/gainakt3_exp.py` model is currently initialized identical to GainAKT2Exp as a baseline. The GainAKT3Exp architectural innovation (Values as Direct Gains, bypassing the Gain Projection Head) will be implemented after baseline training completes.

This section describes the current implementation, which successfully implements all five augmented features from the **Augmented Architecture Design** shown in the architecture diagram above (GainAKT2Exp baseline). The diagram shows the TARGET GainAKT3Exp architecture with the Gain Projection Head marked as DEACTIVATED (red) to illustrate the planned modification.

**Important Context:** The "Concise Architecture Gap Summary" section (below) compares the baseline implementation against THREE additional, more advanced architectural proposals from `assistant/gainakt2exp_architecture_approach.md`, including the "Intrinsic Gain Attention" design where attention Values directly represent learning gains (enabling h_t = Σ α g knowledge state formation). The current GainAKT3Exp baseline uses the "Augmented Design" (GainAKT2Exp), which achieves interpretability through post-hoc projection heads and auxiliary losses. The planned GainAKT3Exp modification moves toward the intrinsic attention semantics by using Values directly as gains.

Below is a comprehensive analysis of each component's implementation status and alignment with the Augmented Architecture Design shown in the diagram.

### Feature 1: Skill Embedding Table 

**Expected (from diagram):** A separate embedding table that maps question IDs to skill representations, used in the prediction head to provide skill-specific context for response prediction.

**Implementation Status:**
- **Location:** `gainakt3.py` line 198: `self.concept_embedding = nn.Embedding(num_c, d_model)`
- **Usage:** Lines 272-273 in forward pass:
  ```python
  target_concept_emb = self.concept_embedding(target_concepts)
  concatenated = torch.cat([context_seq, value_seq, target_concept_emb], dim=-1)
  ```
- **Architecture Alignment:** 
  - Separate embedding table for skills/concepts (distinct from interaction embeddings)
  - Embedded size: `d_model` (consistent with context/value streams)
  - Concatenated with context and value sequences as input to prediction head
  - Supports both direct question IDs (`q`) and query questions (`qry`)

**Verification:** The prediction head receives `[context_seq, value_seq, target_concept_emb]` with shape `[B, L, 3*d_model]`, exactly as specified in the diagram node "Concatenate [h, v, s]".

---

### Feature 2: Dynamic Value Stream 

**Expected (from diagram):** Dual-stream architecture where context and value sequences evolve independently through encoder blocks, with Q/K computed from context and V from value stream.

**Implementation Status:**
- **Dual Embeddings:** `gainakt3.py` lines 195-196:
  ```python
  self.context_embedding = nn.Embedding(num_c * 2, d_model)
  self.value_embedding = nn.Embedding(num_c * 2, d_model)
  ```
- **Dual Stream Processing:** Lines 263-269:
  ```python
  context_seq = self.context_embedding(interaction_tokens)
  value_seq = self.value_embedding(interaction_tokens)
  # ... add positional encodings to both ...
  for block in self.encoder_blocks:
      context_seq, value_seq = block(context_seq, value_seq, mask)
  ```
- **Separate Residual Paths:** `EncoderBlock` (lines 124-153) implements:
  - `norm1_ctx` and `norm1_val` - separate layer norms for each stream after attention
  - `norm2_ctx` - final layer norm for context after FFN
  - Value stream updated: `value_sequence + attn_output` 
  - Context stream updated: `context_sequence + attn_output + ffn_output`
  
- **Attention Mechanism:** `MultiHeadAttention.forward()` (lines 40-89):
  ```python
  Q = self.query_proj(context_sequence)  # Q from context
  K = self.key_proj(context_sequence)    # K from context
  V = self.value_proj(value_sequence)    # V from value stream
  ```

**Architecture Alignment:** 
- Dual independent sequences maintained throughout encoder stack
- Separate Add & Norm operations for context and value (as shown in diagram)
- Q/K from context, V from value exactly as specified
- Both streams contribute to final prediction

**Verification:** The architecture diagram shows "AddNorm_Ctx" and "AddNorm_Val" as separate nodes—implementation has `norm1_ctx`, `norm1_val`, and `norm2_ctx` implementing this exactly.

---

### Feature 3: Ground Truth Responses / Training-time Monitoring 

**Expected (from diagram):** Ground truth responses flow into loss calculation; interpretability monitor hook for real-time constraint analysis with configurable frequency.

**Implementation Status:**

**3a. Ground Truth Usage:**
- Ground truth `r` (responses) used in:
  - Interaction token creation (line 91): `interaction_tokens = q + self.num_c * r_int`
  - All auxiliary loss computations (lines 202-277) via `responses` parameter
  - Mastery-performance alignment: separates correct/incorrect responses (lines 236-243)
  - Gain-performance alignment: compares gains for correct vs incorrect (lines 246-254)

**3b. Training-time Monitoring Integration:**
- **Monitor Hook:** `gainakt3_exp.py` lines 40-41, 54-56:
  ```python
  self.interpretability_monitor = None
  def set_monitor(self, monitor): 
      self.interpretability_monitor = monitor
  ```
- **Periodic Execution:** Lines 164-178:
  ```python
  if (self.interpretability_monitor is not None and 
      batch_idx is not None and 
      batch_idx % self.monitor_frequency == 0 and primary_device):
      with torch.no_grad():
          self.interpretability_monitor(
              batch_idx=batch_idx,
              context_seq=context_seq,
              value_seq=value_seq,
              projected_mastery=projected_mastery,
              projected_gains=projected_gains,
              predictions=predictions,
              questions=q,
              responses=r
          )
  ```
- **Configurable Frequency:** `monitor_frequency` parameter (default: 50 batches)
- **DataParallel Safety:** Primary device guard prevents duplicate monitoring under multi-GPU training

**Architecture Alignment:** 
- Ground truth responses integrated into all constraint loss computations
- Monitoring hook provides real-time interpretability analysis
- Frequency control matches diagram's "Configurable frequency" specification
- All internal states exposed: context, value, mastery, gains, predictions, questions, responses

**Verification:** The diagram shows "Ground Truth Responses" flowing into "BCE Loss" and monitoring receiving multiple state tensors—implementation provides this via `forward_with_states()` returning all required outputs.

---

### Feature 4: Mastery and Gain Projection Heads 

**Expected (from diagram):** Two linear projection heads mapping internal representations to per-skill mastery and gain estimates (shape: `[B, L, num_skills]`). **Note:** In GainAKT3Exp, the Gain Projection Head is architecturally DEACTIVATED—encoder Values flow directly as learning gains.

**Implementation Status (Current Baseline):**
- **Head Creation:** `gainakt3.py` lines 216-219:
  ```python
  if self.use_mastery_head:
      self.mastery_head = nn.Linear(self.d_model, self.num_c)
  if self.use_gain_head:
      self.gain_head = nn.Linear(self.d_model, self.num_c)
  ```
- **Mastery Projection:** `gainakt3_exp.py` lines 115-127:
  ```python
  projected_mastery_raw = self.mastery_head(context_seq)
  initial_mastery = torch.sigmoid(projected_mastery_raw)
  # Recursive accumulation with clamping:
  projected_mastery = torch.zeros_like(initial_mastery)
  projected_mastery[:, 0, :] = initial_mastery[:, 0, :]
  for t in range(1, seq_len):
      accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
      projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
  ```
- **Gain Projection (TO BE MODIFIED):** Lines 113-114:
  ```python
  projected_gains_raw = self.gain_head(value_seq)
  projected_gains = torch.relu(projected_gains_raw)  # enforce non-negativity
  ```

**GainAKT3Exp Target Architecture (Planned Modification):**
```python
# CURRENT (Baseline - GainAKT2Exp approach):
projected_gains_raw = self.gain_head(value_seq)  # Project via linear layer
projected_gains = torch.relu(projected_gains_raw)

# TARGET (GainAKT3Exp - Direct Values as Gains):
projected_gains = torch.relu(value_seq)  # Use Values directly, enforce non-negativity
# Gain Projection Head bypassed - no self.gain_head() call
```

**Architecture Alignment:** 
- Mastery head: `Linear(d_model, num_c)` projects context → per-skill mastery
- **Gain head (GainAKT3Exp):** DEACTIVATED - encoder Values used directly as gains (dimension `[B, L, D]` instead of `[B, L, num_skills]`)
- Output shapes: Mastery `[batch_size, seq_len, num_c]`, Gains `[batch_size, seq_len, d_model]` (GainAKT3Exp modification)
- **Educational Enhancement:** Recursive mastery accumulation enforces consistency between mastery changes and scaled gains (factor: 0.1)
- Non-negativity enforced architecturally via ReLU on gains (applies to raw Values in GainAKT3Exp)
- Bounded mastery [0, 1] via sigmoid initialization and clamping

**Verification:** The diagram shows "Encoder Values (Direct Gains)" with shape `[B, L, D]` and the Gain Projection Head marked RED (DEACTIVATED)—target implementation will use Values directly, bypassing the projection layer.

---

### Feature 5: BCE + Auxiliary Loss Functions 

**Expected (from diagram):** BCE loss for prediction accuracy plus five auxiliary losses (Non-Negative, Monotonicity, Mastery-Performance, Gain-Performance, Sparsity) with configurable weights, all integrated into total loss.

**Implementation Status:**

**5a. BCE Loss:**
- Computed externally in training script using `predictions` output
- Model provides both `predictions` (sigmoid) and `logits` for flexible loss computation

**5b. Auxiliary Losses (all in `compute_interpretability_loss()` lines 202-277):**

1. **Non-Negative Gains Loss** (lines 217-220):
   ```python
   negative_gains = torch.clamp(-projected_gains, min=0)
   non_negative_loss = negative_gains.mean()
   total_loss += self.non_negative_loss_weight * non_negative_loss
   ```

2. **Monotonicity Loss** (lines 222-226):
   ```python
   mastery_decrease = torch.clamp(projected_mastery[:, :-1] - projected_mastery[:, 1:], min=0)
   monotonicity_loss = mastery_decrease.mean()
   total_loss += self.monotonicity_loss_weight * monotonicity_loss
   ```

3. **Mastery-Performance Alignment Loss** (lines 228-243):
   ```python
   relevant_mastery = projected_mastery[skill_masks]
   correct_mask = (responses == 1).flatten()
   incorrect_mask = (responses == 0).flatten()
   low_mastery_on_correct = torch.clamp(1 - relevant_mastery[correct_mask], min=0)
   high_mastery_on_incorrect = torch.clamp(relevant_mastery[incorrect_mask], min=0)
   mastery_performance_loss = low_mastery_on_correct.mean() + high_mastery_on_incorrect.mean()
   total_loss += self.mastery_performance_loss_weight * mastery_performance_loss
   ```

4. **Gain-Performance Alignment Loss** (lines 245-254):
   ```python
   relevant_gains = projected_gains[skill_masks]
   correct_gains = relevant_gains[(responses == 1).flatten()]
   incorrect_gains = relevant_gains[(responses == 0).flatten()]
   if correct_gains.numel() > 0 and incorrect_gains.numel() > 0:
       gain_performance_loss = torch.clamp(incorrect_gains.mean() - correct_gains.mean() + 0.1, min=0)
       total_loss += self.gain_performance_loss_weight * gain_performance_loss
   ```

5. **Sparsity Loss** (lines 256-259):
   ```python
   non_relevant_gains = projected_gains[~skill_masks]
   sparsity_loss = torch.abs(non_relevant_gains).mean()
   total_loss += self.sparsity_loss_weight * sparsity_loss
   ```

6. **Consistency Loss** (lines 261-266):
   ```python
   mastery_delta = projected_mastery[:, 1:, :] - projected_mastery[:, :-1, :]
   scaled_gains = projected_gains[:, 1:, :] * 0.1
   consistency_residual = torch.abs(mastery_delta - scaled_gains)
   consistency_loss = consistency_residual.mean()
   total_loss += self.consistency_loss_weight * consistency_loss
   ```

**5c. Integration:**
- All losses computed in single `compute_interpretability_loss()` method
- Returned as `interpretability_loss` in `forward_with_states()` output dict (line 149)
- Each loss has configurable weight parameter (constructor lines 27-32)
- Skill masks computed from Q-matrix structure (line 213)

**Architecture Alignment:** 
- All 5 diagram losses implemented exactly as shown
- 6th loss (Consistency) added for tighter mastery-gain coupling
- All weights configurable
- Total loss formula: `BCE + w1×NonNeg + w2×Monotonicity + w3×Mastery_Perf + w4×Gain_Perf + w5×Sparsity + w6×Consistency`

**Verification:** The diagram shows 5 auxiliary loss nodes feeding into "Total Loss"—implementation provides these plus an additional consistency loss, all with independently tunable weights.

---

### Feature 6: Intrinsic Gain Attention Mode ❌ DEACTIVATED

**Objective:** Provide an alternative architectural mode that achieves parameter efficiency by deriving gains directly from attention mechanisms, eliminating the need for post-hoc projection heads. This explores the trade-off between model compactness and interpretability while maintaining competitive predictive performance.

**Expected (from architectural exploration):** A feature flag (`--intrinsic_gain_attention`) that conditionally disables projection heads and computes mastery/gains from cumulative attention weights, reducing parameter count while preserving the ability to track learning trajectories.

**Current Status:**: Given the results detailed before we will **deactivate the Intrinsic Gain Attention Mode by default**. So, we set "intrinsic_gain_attention": false in configs/parameter_default.json

**Implementation Status:**

**6a. Architectural Constraint Enforcement:**
- **Location:** `gainakt3_exp.py` lines 58-74
- **Mechanism:** 
  ```python
  if self.intrinsic_gain_attention:
      # Override projection head flags - intrinsic mode incompatible with heads
      self.use_mastery_head = False
      self.use_gain_head = False
      
      if use_mastery_head or use_gain_head:
          print("WARNING: Intrinsic gain attention mode enabled. "
                "Projection heads (use_mastery_head, use_gain_head) will be disabled.")
  ```
- **Rationale:** Prevents conflicting architectural configurations where both projection-based and attention-derived gains would coexist, ensuring clean experimental comparison.

**6b. Attention-Derived Gain Computation:**
- **Location:** `gainakt3_exp.py` lines 102-111
- **Implementation:**
  ```python
  if self.intrinsic_gain_attention and not (self.use_mastery_head or self.use_gain_head):
      # Derive gains from attention patterns
      # Extract attention weights from last encoder layer
      last_block = self.encoder_blocks[-1]
      attn_module = last_block.attn
      
      # Aggregate attention weights across heads as proxy for learning gains
      # Shape: [batch_size, seq_len, seq_len] -> [batch_size, seq_len, num_c]
      attention_gains = self._compute_attention_derived_gains(
          attn_module.attention_weights, questions, batch_size, seq_len
      )
  ```
- **Gain Extraction:** Uses attention weights from final encoder layer as indicators of "information flow" between timesteps, treating high attention as proxy for learning influence.

**6c. Cumulative Mastery from Attention:**
- **Location:** `gainakt3_exp.py` lines 113-133
- **Recursive Accumulation:**
  ```python
  # Initialize mastery from attention-derived gains
  projected_gains = attention_gains  # [batch_size, seq_len, num_c]
  
  # Compute cumulative mastery via recursive addition
  projected_mastery = torch.zeros_like(projected_gains)
  projected_mastery[:, 0, :] = torch.sigmoid(projected_gains[:, 0, :])
  
  for t in range(1, seq_len):
      # Accumulate previous mastery + scaled current gains
      accumulated = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
      projected_mastery[:, t, :] = torch.clamp(accumulated, min=0.0, max=1.0)
  ```
- **Educational Semantics:** Treats attention weights as learning increments, cumulative mastery as integrated knowledge over time.

**6d. Parameter Reduction:**
- **Baseline Mode:** 14,658,761 params
- **Intrinsic Mode:** 12,738,265 params
- **Reduction: 1,920,496 params (13.1%)**
  
  *Note: Reduction comes from disabled projection heads (mastery_head + gain_head) plus associated architectural optimizations.*

**6e. CLI Integration:**
- **Location:** `examples/run_repro_experiment.py` line 89
- **Usage:** `python examples/run_repro_experiment.py --intrinsic_gain_attention --epochs 12`
- **Default:** `False` (baseline mode with projection heads)
- **Parameter File:** Added to `configs/parameter_default.json` as `"intrinsic_gain_attention": false`

**Architecture Alignment:** Complete implementation with validated trade-offs

**Verification:** The updated architecture diagram (red components) shows intrinsic mode as conditional bypass of projection heads, with attention-derived gains feeding directly to mastery/gain outputs.


#### Feature 6 Experimental Validation: Multi-Seed Analysis

**Validation Protocol:**
- **Dataset:** ASSIST2015, fold 0
- **Seeds:** 42, 7, 123, 2025, 31415 (same as baseline for paired comparison)
- **Epochs:** 12 (matching baseline configuration)
- **Hardware:** 8× Tesla V100-SXM2-32GB
- **Training Duration:** ~25 minutes per experiment

**6f. Predictive Performance Results:**

| Metric | Baseline (N=5) | Intrinsic (N=5) | Difference | Statistical Sig. |
|--------|----------------|-----------------|------------|------------------|
| **Test AUC** | 0.7196 ± 0.0005 | 0.7142 ± 0.0007 | -0.0054 (-0.75%) | p=0.0001 (t=16.35) |
| **Test Accuracy** | 0.7474 ± 0.0002 | 0.7460 ± 0.0006 | -0.0014 (-0.19%) | p=0.0131 (t=4.25) |
| **Valid AUC** | 0.7255 ± 0.0002 | 0.7204 ± 0.0006 | -0.0051 (-0.70%) | p=0.0001 (t=15.98) |
| **Reproducibility (CV)** | 0.07% | 0.10% | +0.03% | Both excellent |

**Key Findings:**
- ✅ **Performance preserved:** <1% AUC degradation (statistically significant but practically negligible)
- ✅ **Excellent reproducibility:** CV < 0.2% for both modes
- ✅ **Parameter efficiency validated:** 13.1% reduction with minimal performance cost

**6g. Interpretability Analysis Results:**

| Metric | Baseline (N=5) | Intrinsic (N=5) | Difference | Statistical Sig. |
|--------|----------------|-----------------|------------|------------------|
| **Mastery Correlation** | 0.0952 ± 0.0177 | 0.0322 ± 0.0008 | -0.0630 (-66.2%) | p=0.0013 (t=8.02) |
| **Gain Correlation** | 0.0276 ± 0.0035 | **-0.0065 ± 0.0238** | -0.0341 (-124%) | p=0.0329 (t=3.20) |
| **Seeds with Negative Gains** | 0/5 (0%) | **3/5 (60%)** | - | Critical issue |
| **Gain Correlation Range** | [0.0216, 0.0312] | **[-0.0362, +0.0234]** | - | Extreme variance |

**Critical Observations:**

1. **Mastery Correlation Collapse (66% loss):**
   - Baseline maintains stable ~0.095 correlation across all seeds
   - Intrinsic mode converges to near-zero ~0.032 with tight variance
   - **Interpretation:** Attention-derived mastery lacks educational grounding; cumulative sum of attention weights does not reflect true learning progression

2. **Gain Correlation Instability (negative mean):**
   - **Seed-level breakdown:**
     - Seed 42: +0.0234 ✓ (positive, acceptable)
     - Seed 7: +0.0084 ✓ (positive, weak)
     - Seed 123: **-0.0362** ✗ (negative, violates pedagogy)
     - Seed 2025: **-0.0226** ✗ (negative, violates pedagogy)
     - Seed 31415: **-0.0055** ✗ (negative, violates pedagogy)
   - **Mean:** -0.0065 (negative!)
   - **CV:** 366% (extreme instability)
   - **Pedagogical Violation:** Negative correlations imply "unlearning" when students answer correctly—educationally nonsensical

3. **Reproducibility Paradox:**
   - Predictive metrics: Excellent reproducibility (CV < 0.2%)
   - Interpretability metrics: Catastrophic variance (CV > 300%)
   - **Conclusion:** Model converges reliably for prediction, but interpretability is unstable and unreliable

**6h. Root Cause Analysis:**

**Why Intrinsic Mode Fails at Interpretability:**

1. **Lack of Explicit Supervision:**
   - Baseline: Projection heads trained with `mastery_performance_loss` and `gain_performance_loss`
   - Intrinsic: Gains derived purely from attention (no educational constraint)
   - Result: Gains optimize information flow for prediction, not pedagogical validity

2. **Attention ≠ Learning Gains (Conceptual Mismatch):**
   - Attention weights reflect **"what information is useful for next prediction"**
   - Learning gains reflect **"how much knowledge increased from interaction"**
   - These are fundamentally different: high attention can occur for remediation (reviewing weak skills) or consolidation (reinforcing strong skills)
   - Intrinsic mode conflates predictive utility with learning magnitude

3. **Cumulative Drift:**
   - Recursive mastery = Σ attention-derived "gains" over time
   - Without explicit mastery supervision, cumulative sum drifts from true skill levels
   - Small biases in per-timestep gains amplify across sequence length

4. **Loss Function Mismatch:**
   - Baseline: BCE + mastery_perf_loss + gain_perf_loss + sparsity + ...
   - Intrinsic: Only BCE (interpretability losses disabled when heads are off)
   - Result: No gradient signal to shape attention weights into educationally meaningful gains

**6i. Trade-off Assessment:**

| Dimension | Baseline Mode | Intrinsic Mode | Winner |
|-----------|---------------|----------------|--------|
| **Predictive AUC** | 0.7196 ± 0.0005 | 0.7142 ± 0.0007 | Baseline (marginal) |
| **Parameter Count** | 14.66M | 12.74M | Intrinsic (-13%) |
| **Mastery Interpretability** | 0.095 ± 0.018 | 0.032 ± 0.001 | Baseline (3× stronger) |
| **Gain Interpretability** | 0.028 ± 0.004 | -0.007 ± 0.024 | Baseline (stable & positive) |
| **Reproducibility (Prediction)** | CV=0.07% | CV=0.10% | Both excellent |
| **Reproducibility (Interp)** | CV=13% | CV=366% | Baseline (stable) |
| **Educational Validity** | ✅ All seeds positive | ❌ 60% seeds negative | Baseline (clear) |

**Verdict:**
- **Use Baseline Mode:** For any application requiring interpretability, educational analysis, or student modeling
- **Use Intrinsic Mode:** Only for resource-constrained deployment where prediction suffices (edge devices, mobile apps) and interpretability is optional

**6j. Publication Positioning:**

**Original Hypothesis (Pre-Validation):**
> "Intrinsic gain attention enables native causal decomposability where attention weights directly represent learning gains."

**Empirical Reality (Post-Validation):**
> "Intrinsic mode achieves 13% parameter reduction with <1% AUC loss, but sacrifices interpretability: 66% weaker mastery correlations and 60% of seeds exhibit negative gain correlations. Attention weights optimize prediction, not pedagogical semantics. Explicit supervision via projection heads is necessary for educationally valid gain estimation."

**Reporting Recommendation:**
```
Multi-seed validation (N=5) revealed intrinsic mode interpretability instability: 
60% of seeds exhibited negative gain correlations (range: -0.036 to +0.023, CV=366%), 
violating pedagogical assumptions. In contrast, baseline mode with projection heads 
achieved stable positive correlations across all seeds (gain: 0.0276±0.0035, CV=13%). 

We conclude attention weights optimize predictive utility rather than educational 
semantics, necessitating explicit supervision (projection heads + constraint losses) 
for reliable interpretability. Intrinsic mode is suitable for parameter-efficient 
deployment when interpretability is not required.
```


#### Feature 6 Recommendations

**For Current Paper (Immediate Publication Goal):**

1. **Primary Contribution:** Baseline mode (projection heads + auxiliary losses)
   - Report: Test AUC 0.7196 ± 0.0005, Mastery Corr 0.095 ± 0.018, Gain Corr 0.028 ± 0.004
   - Emphasize: Stable interpretability across all seeds, zero pedagogical violations
   - Positioning: "Interpretability-first transformer for knowledge tracing"

2. **Intrinsic Mode as Ablation/Alternative:**
   - Present as architectural variant demonstrating parameter efficiency
   - Transparent reporting: 13% reduction, <1% AUC loss, but 66% interpretability loss
   - Use Case Table:
     | Application | Recommended Mode | Rationale |
     |-------------|------------------|-----------|
     | Pedagogical analysis | Baseline | Stable, valid correlations |
     | Student modeling | Baseline | Explicit educational supervision |
     | Edge deployment | Intrinsic | 13% smaller, prediction-only |
     | Mobile apps | Intrinsic | Efficiency critical, interp optional |

3. **Paper Structure Adjustment:**
   ```
   Section 4: Experimental Validation
     4.1 Multi-Seed Reproducibility (N=5, CV=0.07%)
     4.2 Baseline Performance & Interpretability (PRIMARY)
     4.3 Architectural Ablation: Intrinsic Mode Analysis
     4.4 Interpretability Trade-offs & Limitations
     4.5 Use Case Guidance
   ```

4. **Avoid Over-Claiming:**
   - ❌ Do NOT claim: "Intrinsic mode provides causal interpretability"
   - ❌ Do NOT claim: "Attention weights represent learning gains"
   - ✅ DO claim: "Projection heads with explicit supervision necessary for interpretable gains"
   - ✅ DO claim: "13% parameter reduction feasible for prediction-only applications"

**For Future Work (Post-Publication Improvements):**

1. **Hybrid Architecture:**
   - Retain attention-derived gains for parameter efficiency
   - Add lightweight projection heads (e.g., 64-dim intermediate) with partial supervision
   - Goal: Balance efficiency and interpretability

2. **Supervised Attention:**
   - Modify attention mechanism to receive pedagogical gradients
   - Add per-head gain losses to shape attention semantics
   - Requires architectural innovation

3. **Q-Matrix Integration (Phase 2):**
   - Architectural sparsity enforcement via pre-attention masking
   - May stabilize intrinsic gains by constraining to relevant skills
   - Lower priority given current instability

---

## Overall Architecture Compliance

| **Feature**                | **Diagram Specification**                                      | **Implementation Details**                                                                 | **Status**          |
|----------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------|
| **Skill Embedding Table**  | Separate embedding for target skills in prediction            | `concept_embedding` used in `[h, v, s]` concatenation                                     | ✅ Activated         |
| **Dynamic Value Stream**   | Dual context/value sequences, separate norms, Q/K from context, V from value | Dual embeddings + separate `norm1_ctx/val`, `norm2_ctx` + correct attention               | ✅ Activated         |
| **Ground Truth Integration** | Used in loss calculation + monitoring hooks                  | Integrated in all losses + `set_monitor()` + periodic execution                           | ✅ Activated         |
| **Projection Heads**       | Mastery (context→skills), Gain (value→skills)                 | `mastery_head`, `gain_head` with recursive accumulation                                   | ✅ Activated + enhanced |
| **Auxiliary Losses**       | 5 losses (NonNeg, Monotonicity, Mastery-Perf, Gain-Perf, Sparsity) | All 5 + Consistency (bonus) with configurable weights                                     | ✅ Activated         |
| **Monitoring**             | Real-time interpretability analysis, configurable frequency   | `interpretability_monitor` hook + `monitor_frequency` + DataParallel safety              | ✅ Activated         |
| **Intrinsic Gain Attention** | Alternative parameter-efficient mode                        | `--intrinsic_gain_attention` flag, architectural constraint enforcement, attention-derived gains | ❌ Deactivated       |

### Key Implementation Strengths

1. **Modular Design:** Projection heads and auxiliary losses controlled by boolean flags (`use_mastery_head`, `use_gain_head`) enabling clean ablation studies.

2. **Educational Consistency:** Recursive mastery accumulation (mastery[t] = mastery[t-1] + 0.1 × gains[t]) enforces architectural constraint beyond just auxiliary losses.

3. **Training Stability:** Mixed-precision safety (dtype-aware masking in attention), DataParallel guards, gradient-friendly operations (clamp, hinge losses).

4. **PyKT Framework Compliance:** Dual forward methods (`forward()` for compatibility, `forward_with_states()` for monitoring) maintain integration with existing evaluation scripts.

5. **Interpretability Infrastructure:** Complete state exposure (context, value, mastery, gains, predictions, questions, responses) enables rich post-hoc analysis and real-time monitoring.

### Minor Observations
  
- **Non-Negative Loss Default:** The `non_negative_loss_weight` is currently 0.0 in default parameters because non-negativity is enforced architecturally (ReLU on gains). The loss remains available for architectural variants.

- **Skill Mask Computation:** Uses Q-matrix structure via `skill_masks.scatter_(2, questions.unsqueeze(-1), 1)` to identify relevant skills—correctly implements sparsity constraint based on problem-skill mappings.

### Conclusion

**The current baseline implementation in `pykt/models/gainakt3_exp.py` achieves 100% compliance with the GainAKT2Exp Augmented Architecture Design.** The target GainAKT3Exp architecture (shown in the diagram with Gain Projection Head marked RED/DEACTIVATED) represents a planned modification to use encoder Values directly as learning gains. All six augmented features are currently implemented following the GainAKT2Exp baseline:

1. **Features 1-5 (Baseline Mode):** Production-ready with excellent interpretability (mastery corr: 0.095 ± 0.018, gain corr: 0.028 ± 0.004)
2. **Feature 6 (Intrinsic Mode):** Validated alternative achieving 13% parameter reduction with <1% AUC loss, but 66% interpretability degradation

**GainAKT3Exp Planned Modification:**
- **Target:** Bypass Gain Projection Head, use `value_seq` directly as gains (dimension `[B, L, D]`)
- **Benefit:** Direct attention-learned gains, simpler architecture, potentially stronger mastery correlations
- **Implementation:** Modify `gainakt3_exp.py` forward pass to route Values directly to recursive mastery accumulation and Gain Hub

**Multi-Seed Validation (N=5) Establishes (for current GainAKT2Exp baseline):**
- ✅ **Baseline Mode:** Reproducible, interpretable, educationally valid (CV=0.07%, all seeds positive)
- ⚠️ **Intrinsic Mode:** Reproducible for prediction, unstable for interpretation (60% seeds with negative gain correlations). 


> **Publication Readiness:**
> - Primary contribution: Baseline mode demonstrates that **projection heads + explicit supervision are necessary and sufficient for educationally meaningful interpretability**
> - GainAKT3Exp innovation: Direct Values as Gains represents an architectural simplification hypothesis to be validated after baseline training
> - **Ablation** contribution: Intrinsic mode validates the necessity of explicit supervision through its interpretability failure
> - Practical outcome: Clear use-case guidance (baseline for analysis, intrinsic for edge deployment)
>
> The model is ready for paper writeup with reporting of both successes (baseline interpretability) and limitations (intrinsic mode trade-offs).


## Parameters

The complete list of parameters including category and description is in ´paper/parameters.csv´. 

## Evolving the Model

We'll try to improve and evaluate variants of the model, being essential to maintain consistency in the definition of the parameters across all these possible changes. 

Do these consistency updates/tests after making changes to the codebase (model, training/evaluation/reproduction scripts, etc.) in any of the scenarios described below 
  - Check that there are no hidden parameters with hardcoded default values that can change without notice, distorting the interpretation of the impact of hyperparameter changes.
  - If parameters are added or changed (name or default value), update "defaults" section of configs/parameter_default.json needs to be updated accordingly. 

Launch a consistency test after each change in the codebase and output warnings (only inform) or errors (inform and await for remediation)

### Scenarios

#### Scenario 1: Hyperparameter Sweep

Objective: By doing a sweep, we mean systematically exploring different combinations of hyperparameters to find the configuration that yields the best performance.
Guidelines: We can use default values in configs/parameter_default.json as starting points. Once a optimal combination is found, current defaults in configs/parameter_default.json should be updated. 

#### Scenario 2: Ablation Studies

Objective: in a ablation studio we deactivate parameters one by one to measure the impact
Guidelines: Check current value of the parameter to ablate in configs/parameter_default.json and deactivate (changing a boolean value, setting a weight to 0, etc.). 

#### Scenario 3: Benchmark

Objective: compare metrics of different models or model variants. 
Guidelines: use defaults to launch training and evaluation.


### Parameter Evolution Best Practices

When adding/changing parameters:

1. **Update `configs/parameter_default.json`**
   ```bash
   # Edit the file to add new parameter
   # Then recompute MD5:
   python -c "
   import json, hashlib
   data = json.load(open('configs/parameter_default.json'))
   md5 = hashlib.md5(json.dumps(data['defaults'], sort_keys=True).encode()).hexdigest()
   data['md5'] = md5
   json.dump(data, open('configs/parameter_default.json', 'w'), indent=2)
   print(f'Updated MD5: {md5}')
   "
   ```

2. **Update training/evaluation scripts**
   - Add argparse parameter with `required=True` (no default!)
   - Ensure parameter name matches exactly

3. **Test with dry run**
   ```bash
   python examples/run_repro_experiment.py \
     --short_title test_new_param \
     --epochs 1
   ```

4. **Verify in config.json**
   - Check that parameter appears in `defaults` section
   - Check that it appears in `train_explicit` or `eval_explicit` command

5. **Update this documentation**
   - Add to appropriate category table above
   - Document purpose and default value

## Loss Functions

Total Loss = BCE Loss + Constraint Losses + Semantic Module Losses

### Loss Parameters

| Category | Name | Parameter Name | Default Value | Description |
|----------|------|----------------|---------------|-------------|
| **Main** | BCE Loss | - | - | Binary cross-entropy for response prediction |
| **Constraint** | Non-Negative Gains | `non_negative_loss_weight` | 0.0 | Penalizes negative learning gains (disabled) |
| **Constraint** | Monotonicity | `monotonicity_loss_weight` | 0.1 | Enforces non-decreasing mastery over time |
| **Constraint** | Mastery-Performance | `mastery_performance_loss_weight` | 0.8 | Penalizes low mastery on correct, high on incorrect |
| **Constraint** | Gain-Performance | `gain_performance_loss_weight` | 0.8 | Enforces higher gains for correct responses |
| **Constraint** | Sparsity | `sparsity_loss_weight` | 0.2 | Penalizes gains on non-relevant skills |
| **Constraint** | Consistency | `consistency_loss_weight` | 0.3 | Aligns mastery changes with scaled gains |
| **Semantic** | Alignment (Local) | `alignment_weight` | 0.25 | Maximizes correlation between mastery/gains and performance |
| **Semantic** | Global Alignment | `enable_global_alignment_pass` | true | Population-level mastery coherence regularization |
| **Semantic** | Residual Alignment | `use_residual_alignment` | true | Alignment on variance unexplained by global signal |
| **Semantic** | Retention | `retention_weight` | 0.14 | Prevents post-peak mastery decay |
| **Semantic** | Lag Gain | `lag_gain_weight` | 0.06 | Introduces temporal structure to gains (lag-1,2,3) |
| **Schedule** | Constraint Warmup | `warmup_constraint_epochs` | 8 | Epochs to ramp constraint losses from 0 to full |
| **Schedule** | Alignment Warmup | `alignment_warmup_epochs` | 8 | Epochs to ramp alignment loss from 0 to full |
| **Schedule** | Alignment Share Cap | `alignment_share_cap` | 0.08 | Maximum proportion of total loss from alignment |

### BCE Loss

Binary Cross-Entropy (BCE) Loss: Core loss for response correctness prediction. 

### Constraint Losses

Constraint losses enforce structural validity and educational plausibility of the projected mastery and gain trajectories. Implemented in the model's `compute_interpretability_loss()` method (`pykt/models/gainakt3_exp.py`), these losses operate at the **interaction level**, penalizing specific violations of educational expectations. Unlike semantic module losses that shape overall trajectory correlations, constraint losses act as **hard regularizers** preventing degenerate or nonsensical states.

**Non-Negative Gains** (`non_negative_loss_weight = 0.0`): Penalizes negative learning gains by computing `clamp(-projected_gains, min=0).mean()`. Currently disabled (weight 0.0) as gains are naturally non-negative due to model architecture, but available for architectural variants.

**Monotonicity** (`monotonicity_loss_weight = 0.1`): Enforces non-decreasing mastery over time by penalizing `clamp(mastery[t] - mastery[t+1], min=0).mean()`. Ensures mastery cannot regress, reflecting the assumption that learning is cumulative and students do not "unlearn" previously mastered skills.

**Mastery-Performance Alignment** (`mastery_performance_loss_weight = 0.8`): Penalizes interaction-level mismatches between mastery and performance. Specifically: (1) penalizes low mastery (`clamp(1 - mastery, min=0)`) when students answer correctly, and (2) penalizes high mastery (`clamp(mastery, min=0)`) when students answer incorrectly. This hinge-style constraint prevents obvious violations (e.g., mastery=0.1 on correct response, mastery=0.9 on incorrect response) and complements the trajectory-level Alignment Loss by enforcing point-wise consistency.

**Gain-Performance Alignment** (`gain_performance_loss_weight = 0.8`): Enforces that correct responses should yield higher gains than incorrect responses via hinge loss: `clamp(mean(incorrect_gains) - mean(correct_gains) + 0.1, min=0)`. The 0.1 margin ensures a clear separation, reflecting the educational assumption that successful problem-solving produces greater learning increments.

**Sparsity** (`sparsity_loss_weight = 0.2`): Penalizes non-zero gains for skills not directly involved in the current interaction via `abs(non_relevant_gains).mean()`. Encourages skill-specific learning (gains concentrated on the question's target skill) rather than diffuse updates across all skills, improving interpretability and alignment with skill-specific educational theories.

**Consistency** (`consistency_loss_weight = 0.3`): Enforces temporal coherence between mastery changes and scaled gains via `|mastery_delta - scaled_gains * 0.1|.mean()`. Ensures that mastery increments align with the projected gain magnitudes, preventing the model from producing contradictory mastery and gain trajectories (e.g., large gains with flat mastery, or mastery jumps with zero gains).

All constraint losses are subject to warm-up scheduling (`warmup_constraint_epochs = 8`), gradually ramping from zero to full weight to allow the model to establish baseline representations before enforcing strict constraints. Violation rates are monitored and logged; current optimal configuration achieves **zero violations** across all constraints.

### Semantic Module Losses

Enabling alignment, global alignment, retention, and lag objectives restored strong semantic interpretability: mastery and gain correlations surpass prior breakthrough levels and remain stable, with modest decline from peak. Predictive AUC peaks early and declines due to interpretability emphasis; scheduling and stabilization adjustments can mitigate this without sacrificing correlation strength. Recommended enhancements focus on smoothing alignment, stabilizing lag objectives, adding statistical robustness and coverage metrics, and protecting validation AUC with phased optimization.

**Alignment Loss (Local)** (`alignment_weight = 0.25`): Encourages the model's projected mastery estimates to align with actual student performance on individual interactions. Specifically, it penalizes low mastery when students answer correctly and high mastery when they answer incorrectly. This local constraint shapes mastery trajectories to be performance-consistent at the interaction level, accelerating the emergence of educationally meaningful correlations.

**Global Alignment Pass** (`enable_global_alignment_pass = true`): Computes population-level mastery statistics (mean/variance across students) and uses them to regularize individual mastery trajectories toward global coherence patterns. This cross-student alignment improves mastery correlation stability by reducing inter-student variance and reinforcing common learning progressions.

**Residual Alignment** (`use_residual_alignment = true`): Applied after global alignment to capture unexplained variance. By removing the global signal component, residual alignment clarifies incremental mastery improvements specific to individual learning contexts, yielding sharper and more interpretable correlation patterns.

**Retention Loss** (`retention_weight = 0.14`): Prevents post-peak decay of mastery trajectories by penalizing decreases in mastery levels after they reach local maxima. This ensures that once students demonstrate mastery, the model maintains elevated mastery estimates rather than allowing degradation, supporting higher final correlation retention ratios.

**Lag Gain Loss** (`lag_gain_weight = 0.06`): Introduces temporal structure to learning gains by encouraging gains at timestep t to correlate with gains at previous timesteps (lag-1, lag-2, lag-3). This creates a coherent temporal narrative where gains emerge systematically rather than randomly, enhancing gain correlation interpretability and capturing causal learning progression patterns.

#### Alignment Schedule Parameters

The semantic module losses, particularly alignment loss, use scheduling mechanisms to balance interpretability emergence with predictive performance:

**Warm-up Scheduling** (`alignment_warmup_epochs = 8`): Alignment loss is gradually ramped from zero to full weight over the first 8 epochs, allowing the model to establish baseline representations before enforcing strict alignment constraints. This prevents early optimization conflicts where the model hasn't yet learned discriminative features, which could cause training instability or degrade predictive performance.

**Share Cap** (`alignment_share_cap = 0.08`): Limits the maximum proportion of total loss contributed by alignment to 8%. This prevents alignment from dominating the optimization objective, which could sacrifice predictive accuracy (BCE loss) for interpretability. The cap acts as a soft constraint ensuring that performance remains competitive while still benefiting from alignment-driven trajectory shaping.

**Rationale:** Early experiments showed that uncapped alignment loss could improve mastery correlation by 15-20% but degrade AUC by 2-3%. The combination of warm-up + share cap enables a balanced regime where interpretability improves (mastery correlation: 0.095±0.018) while maintaining competitive AUC (0.720±0.001). The 8-epoch warm-up aligns with constraint warm-up (`warmup_constraint_epochs = 8`), creating a coordinated two-phase training strategy: (1) Phase 1 (epochs 1-8): representation learning with gradual constraint introduction, (2) Phase 2 (epochs 9-12): full multi-objective optimization with alignment capped at 8% of total loss.

**Implementation Note:** The share cap is dynamically computed per batch as `min(alignment_loss * alignment_weight, total_loss * alignment_share_cap)`, ensuring the constraint applies regardless of batch-level loss magnitude fluctuations. This provides stable training dynamics across different dataset characteristics and batch compositions.



## Semantic Interpretabily Recovery

### Objective
Recover non-zero, educationally meaningful mastery and gain correlations after they regressed to 0.0 in a prior configuration, and identify the minimal parameter set whose activation restores semantic signals. Provide actionable guidance for parameter sweep design to optimize the trade-off between predictive AUC and interpretability (correlations, stability, coverage).

### Recovery Summary

Mastery and gain correlations regressed to zero when projection heads (`use_mastery_head`, `use_gain_head`) and semantic modules (alignment, global alignment, retention, lag) were inadvertently disabled by launcher logic overriding boolean flags to `false`. Recovery was achieved by re-enabling these modules plus extending constraint warm-up (4→8 epochs), reducing training horizon (20→12 epochs), and decreasing batch size (96→64). 

**Key Recovery Mechanisms:**
1. **Heads Activation:** Mandatory for producing mastery/gain trajectories (correlation computation impossible without)
2. **Alignment Family:** Local + adaptive + global residual alignment accelerates correlation emergence and stabilizes trajectories via performance-consistency shaping and population-level coherence
3. **Retention + Lag:** Prevents post-peak mastery decay and introduces temporal gain structure, improving final correlation retention and interpretability
4. **Scheduling:** Extended warm-up (8 epochs) allows latent representations to differentiate before full constraint pressure; shorter training (12 epochs) avoids late-stage correlation erosion

**Outcome:** Mastery correlation peaked at 0.149 (final: 0.124), gain correlation at 0.103 (final: 0.093), with zero constraint violations and early validation AUC of 0.726.

**Next Steps:** Multi-seed validation, early stopping to preserve AUC, ablation studies quantifying individual component contributions, and expansion to cross-dataset evaluation.

### Expected Outcomes
Recovered configuration demonstrates that enabling semantic modules and interpretability heads plus extending warm-up and reducing training horizon restores correlations (mastery ≈0.10+, gain ≈0.05+). Sweeps will seek configurations yielding mastery_corr ≥0.12 with val AUC ≥0.72 (early-stopped) and gain_corr ≥0.07 under zero violations, establishing a balanced regime for publication.




## Paper Claim 

**UPDATED (GainAKT3Exp Development Status):**

> We introduce GainAKT3Exp, an interpretability-first transformer for knowledge tracing that models learning gains and mastery trajectories through a simplified architecture where encoder **Values directly represent learning gains** (bypassing the Gain Projection Head). This architectural innovation builds on GainAKT2Exp's validated baseline (mastery correlation: 0.095±0.018, gain correlation: 0.028±0.004, test AUC: 0.720±0.001, N=5 seeds) by enabling the attention mechanism to learn gain semantics intrinsically rather than through post-hoc projection. GainAKT3Exp hypothesizes that direct Value-as-Gain representation will improve mastery correlations while maintaining competitive predictive performance through tighter coupling between attention learning and pedagogical constraints.

**GainAKT2Exp Baseline Reference (Validated):**

> GainAKT2Exp is an interpretability-first transformer for knowledge tracing that explicitly models learning gains and mastery trajectories through projection heads and auxiliary pedagogical losses. Multi-seed validation (N=5) on ASSIST2015 demonstrates stable interpretability (mastery correlation: 0.095±0.018, gain correlation: 0.028±0.004, CV<1%) with competitive predictive performance (test AUC: 0.720±0.001). Through architectural ablation comparing projection-based (baseline) versus attention-derived (intrinsic) gain estimation, we empirically established that explicit supervision is necessary for educationally valid interpretability: intrinsic mode achieves 13% parameter reduction with <1% AUC loss but suffers 66% interpretability degradation and negative gain correlations in 60% of seeds.

### Current Status & Publication Readiness

**🔄 IN DEVELOPMENT (GainAKT3Exp):**

1. **Baseline Training Complete (Epoch 9/12 in progress):**
   - Model: GainAKT3Exp initialized identical to GainAKT2Exp
   - Current metrics: Valid AUC 0.7101, Best AUC 0.7243
   - Correlations: Mastery 0.087, Gains 0.048
   - Purpose: Establish baseline before architectural modification

2. **Architectural Modification Planned:**
   - Target: Bypass Gain Projection Head, use Values directly as gains
   - Expected benefit: Enhanced mastery correlations through direct attention-gain coupling
   - Implementation: Modify `gainakt3_exp.py` forward pass after baseline completes

**✅ VALIDATED (GainAKT2Exp Baseline - Reference Architecture):**

1. **Multi-Seed Statistical Validation (N=5):**
   - Baseline mode: Test AUC 0.7196 ± 0.0005 (CV: 0.07%)
   - Mastery correlation: 0.0952 ± 0.0177 (all seeds positive)
   - Gain correlation: 0.0276 ± 0.0035 (all seeds positive)
   - Paired t-tests demonstrate statistical significance

2. **Architectural Ablation Complete:**
   - Baseline mode (14.66M params): Interpretable, educationally valid
   - Intrinsic mode (12.74M params): Parameter-efficient, interpretability compromised
   - Rigorous paired comparison establishes necessity of explicit supervision

3. **Reproducibility Demonstrated:**
   - Excellent predictive reproducibility (CV < 0.2% for both modes)
   - Stable interpretability in baseline mode (CV = 13-19%)
   - Complete experimental tracking (config.json, MD5 checksums)

4. **Zero Constraint Violations:**
   - Monotonicity, non-negativity, bounded mastery maintained
   - Educational plausibility constraints satisfied

5. **Honest Limitation Reporting:**
   - Intrinsic mode interpretability failure documented
   - Clear trade-off analysis provided
   - Use-case guidance table included

**✅ PUBLICATION-READY STRENGTHS:**

- **Novel Contribution:** First systematic multi-seed validation of interpretability metrics in transformer KT
- **Methodological Rigor:** Paired statistical testing, architectural ablation, transparent limitation reporting
- **Practical Value:** Clear guidance on when to use projection heads vs attention-derived gains
- **Reproducibility:** Full parameter manifests, deterministic training, open implementation

**⚠️ OPTIONAL ENHANCEMENTS (Can defer to future work):**

1. **Baseline Model Comparisons (Low priority for v1):**
   - AKT, SAKT, SAINT comparisons on same dataset
   - Can cite published numbers, add full reproduction in extended version
   
2. **Coverage Metrics (Medium priority):**
   - Per-student correlation distributions
   - % students with positive mastery/gain correlations
   - Can include in supplementary materials

3. **Cross-Dataset Validation (Future work):**
   - ASSIST2017, STATICS2011, EdNet
   - Demonstrates generalization but not critical for initial publication
   
4. **Ablation of Semantic Modules (Medium priority):**
   - Retention, lag, global alignment individual contributions
   - Strong ablation already exists (baseline vs intrinsic)
   - Finer-grained ablations can be future work

### Immediate Publication Strategy (Speed-Focused)

**Target Venue:** Educational Data Mining (EDM) or AI in Education (AIED) conference

**Core Narrative:**
1. **Problem:** Transformer KT models lack interpretable learning gain estimation
2. **Solution:** Projection heads + auxiliary pedagogical losses
3. **Validation:** Multi-seed reproducibility (N=5) demonstrates stable interpretability
4. **Key Finding:** Attention-derived gains (intrinsic mode) fail interpretability despite parameter efficiency
5. **Impact:** First rigorous demonstration that explicit supervision is necessary for interpretable gains

**Paper Structure (Optimized for Speed):**

```
1. Introduction (2 pages)
   - Knowledge tracing + interpretability challenge
   - Contribution: Reproducible interpretability with explicit supervision
   
2. Related Work (1.5 pages)
   - Transformer KT (AKT, SAKT, SAINT)
   - Interpretability in KT (DKT interpretability issues)
   - Learning gain modeling
   
3. Method (3 pages)
   - Architecture: Projection heads, auxiliary losses
   - Training: Multi-objective optimization, warm-up scheduling
   - Intrinsic mode: Attention-derived gains (ablation)
   
4. Experimental Setup (1.5 pages)
   - Dataset: ASSIST2015 (fold 0)
   - Multi-seed protocol (N=5)
   - Metrics: AUC, accuracy, mastery/gain correlations
   
5. Results (3 pages)
   - Table 1: Baseline multi-seed performance (AUC, correlations)
   - Table 2: Baseline vs Intrinsic paired comparison
   - Figure 1: Seed-level gain correlation distribution
   - Figure 2: Mastery trajectory example (high vs low performing student)
   
6. Discussion (2 pages)
   - Why explicit supervision matters (empirical evidence)
   - Trade-off analysis (efficiency vs interpretability)
   - Use case guidance (when to use each mode)
   
7. Limitations & Future Work (1 page)
   - Single dataset (cross-dataset validation deferred)
   - Hybrid architectures for future exploration
   
8. Conclusion (0.5 page)
   - Reproducible interpretability achieved
   - Clear design guidelines for practitioners
```

**Total:** ~14-15 pages (typical conference format)

**Timeline for Draft:**
- **Week 1:** Write Sections 1-3 (Intro, Related Work, Method)
- **Week 2:** Write Sections 4-5 (Experiments, Results)
- **Week 3:** Write Sections 6-8, polish, proofread
- **Week 4:** Internal review, revisions, submit

**No Additional Experiments Needed:**
- ✅ All data collected (baseline N=5, intrinsic N=5)
- ✅ Statistical analysis complete (paired t-tests, CIs)
- ✅ Key figures ready (can generate from existing logs)

### Decision Criteria (UPDATED)

**✅ MINIMUM CRITERIA MET FOR PUBLICATION:**
- ✅ Multi-seed validation (N=5) with excellent reproducibility
- ✅ Statistically significant correlations (paired t-tests p<0.05)
- ✅ Architectural ablation (baseline vs intrinsic) demonstrating necessity
- ✅ Honest limitation reporting (intrinsic mode interpretability failure)
- ✅ Competitive AUC (0.720, within range of SAKT 0.711, simpleKT 0.725)

**Current verdict:** ✅ **READY FOR PAPER WRITING.** All critical experiments complete, core claims validated, narrative clear. Focus should shift to writing, not additional experiments.

## Architectural Modes Comparison (Implemented & Validated)

| Dimension | Baseline Mode (Projection Heads) | Intrinsic Mode (Attention-Derived) | Empirical Winner | Publication Status |
|-----------|----------------------------------|-------------------------------------|------------------|--------------------|
| **Parameter Count** | 14.66M | 12.74M (-13.1%) | Intrinsic (efficiency) | ✅ Validated N=5 |
| **Test AUC** | 0.7196 ± 0.0005 | 0.7142 ± 0.0007 (-0.75%) | Baseline (marginal) | ✅ Validated N=5 |
| **Mastery Correlation** | 0.0952 ± 0.0177 | 0.0322 ± 0.0008 (-66%) | Baseline (3× stronger) | ✅ Validated N=5 |
| **Gain Correlation** | 0.0276 ± 0.0035 | -0.0065 ± 0.0238 (-124%) | Baseline (positive) | ✅ Validated N=5 |
| **Reproducibility (AUC)** | CV = 0.07% | CV = 0.10% | Both excellent | ✅ Validated N=5 |
| **Reproducibility (Gain)** | CV = 13% | CV = 366% | Baseline (stable) | ✅ Validated N=5 |
| **Educational Validity** | 5/5 seeds positive | 2/5 seeds positive | Baseline (100% vs 40%) | ✅ Validated N=5 |
| **Explicit Supervision** | ✅ mastery_perf_loss + gain_perf_loss | ❌ Only BCE loss | Baseline (necessary) | ✅ Empirically proven |
| **Use Cases** | Analysis, modeling, research | Edge deployment, mobile | Context-dependent | ✅ Clear guidance |

**Key Empirical Finding (GainAKT2Exp - Validated, GainAKT3Exp - Under Development):**

**GainAKT2Exp (Reference Architecture):**
> Multi-seed validation (N=5) demonstrates that **explicit supervision via projection heads is necessary for educationally valid interpretability**. Intrinsic mode (attention-derived gains) achieves parameter efficiency (13% reduction, <1% AUC loss) but fails interpretability: 60% of seeds exhibit negative gain correlations (mean: -0.007 ± 0.024, CV=366%), violating pedagogical assumptions. Baseline mode maintains stable positive correlations across all seeds (gain: 0.028 ± 0.004, CV=13%), supporting its use for student modeling and educational analysis.

**GainAKT3Exp (Current Development):**
> GainAKT3Exp explores whether **Values-as-Direct-Gains** can improve upon GainAKT2Exp's baseline by enabling the attention mechanism to learn gain semantics intrinsically while maintaining explicit supervision through auxiliary losses. Unlike the failed intrinsic mode (which eliminated ALL supervision), GainAKT3Exp retains constraint losses but routes them to raw attention Values instead of projected outputs. Hypothesis: Direct Value optimization for gain-related losses may strengthen mastery correlations beyond GainAKT2Exp's 0.095±0.018 baseline.

**Paper Positioning (Updated for GainAKT3Exp Development):**
*GainAKT2Exp established projection heads + auxiliary losses as a sufficient architecture for interpretable transformer KT through rigorous multi-seed ablation. GainAKT3Exp extends this by testing whether Values themselves can serve as gains when supervised by the same auxiliary losses, potentially achieving the benefits of intrinsic representation (direct attention-gain coupling) without sacrificing interpretability. We provide empirical comparison of three architectural variants: projection-based (GainAKT2Exp baseline), attention-derived without supervision (intrinsic mode - failed), and attention-derived with supervision (GainAKT3Exp - under evaluation).*


## Benchmark

### Baseline models
```
PYKT Benchmark Results Summary (Question-Level AUC):
- AKT: 0.7853 (AS2009), 0.8306 (AL2005), 0.8208 (BD2006), 0.8033 (NIPS34) - **Best overall**
- SAKT: 0.7246 (AS2009), 0.7880 (AL2005), 0.7740 (BD2006), 0.7517 (NIPS34)
- SAINT: 0.6958 (AS2009), 0.7775 (AL2005), 0.7781 (BD2006), 0.7873 (NIPS34)

Other benchmarks: 
- simpleKT 0.7744 (AS2009) 0.7248 (AS2015) - Reported as strong baseline with minimal complexity
```


### GinAKT Results Comparison

#### GainAKT2Exp Results
| Split      | AUC       | Accuracy  | Mastery Correlation | Gain Correlation | Correlation Students | Timestamp                  |
|------------|-----------|-----------|---------------------|------------------|----------------------|---------------------------|
| Training   | 0.7549    | 0.7631    | 0.1222              | 0.0544           | 3334                 | 2025-11-12T11:45:29.442074 |
| Validation | 0.7243    | 0.7538    | N/A                 | N/A              | N/A                  | 2025-11-12T11:45:29.442074 |
| Test       | 0.7188    | 0.7485    | 0.1165              | 0.0344           | 3177                 | 2025-11-12T11:45:29.442074 |

#### GainAKT3Exp Results
| Split      | AUC       | Accuracy  | Mastery Correlation | Gain Correlation | Correlation Students | Timestamp                  |
|------------|-----------|-----------|---------------------|------------------|----------------------|---------------------------|
| Training   | 0.7242    | 0.7510    | 0.0260              | 0.0257           | 3334                 | 2025-11-14T18:20:58.091915 |
| Validation | 0.7139    | 0.7512    | N/A                 | N/A              | N/A                  | 2025-11-14T18:20:58.091915 |
| Test       | 0.7095    | 0.7452    | 0.0221              | 0.0216           | 3177                 | 2025-11-14T18:20:58.091915 |
