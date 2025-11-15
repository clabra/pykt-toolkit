# GainAKT3Exp Model Status

**Document Version**: Updated 2025-11-15  
**Model Version**: GainAKT3Exp - Dual-stream transformer with Mastery Accumulation (Gains Head output deactivated)  
**Status**: Active implementation with full training/evaluation pipeline

**⚠️ CURRENT CONFIGURATION (2025-11-15)**: 
- **Dual-Prediction Architecture**: ✅ TWO independent prediction branches with TWO loss functions
  - **Base Predictions** → BCE Loss (primary)
  - **Incremental Mastery Predictions** → Incremental Mastery Loss (weight=0.1)
- **Mastery Accumulation**: ✅ ACTIVE (`use_mastery_head=true`) - Recursive mastery tracking with learnable thresholds
- **Gains Head Output**: ❌ DEACTIVATED (`use_gain_head=false`) - Gains computed internally for mastery but not exposed as output
- **Architecture Flow**: 
  - Path 1: Encoder → [Context, Value, Skill] → Prediction Head → Base Predictions → BCE Loss
  - Path 2: Values → Learning Gains → Mastery → Threshold → Incremental Mastery Predictions → IM Loss
- **Code Location**: `gainakt3_exp.py` lines 318-516 (dual predictions, dual losses)

See "Architecture Summary" section below for detailed flow.

---

## Model Overview

**Implementation Files**:
- **Model**: `pykt/models/gainakt3_exp.py` (477 lines) - extends `GainAKT3` base class
- **Training**: `examples/train_gainakt3exp.py` (1893 lines) - zero defaults, explicit parameters
- **Evaluation**: `examples/eval_gainakt3exp.py` (308 lines) - test metrics + correlations
- **Trajectories**: `examples/learning_trajectories.py` (365 lines) - individual student analysis
- **Launcher**: `examples/run_repro_experiment.py` - loads defaults, manages experiments
- **Factory**: `create_exp_model(config)` (line 435 in `gainakt3_exp.py`) - requires 22 explicit parameters


## Core Architecture: Dual-Stream Transformer with Dual-Prediction Mechanism

**GainAKT3Exp Current State**: The model uses recursive mastery accumulation to track skill mastery over time and generates **TWO SEPARATE** prediction outputs with **TWO SEPARATE** loss functions.

**Dual-Prediction Architecture**:
1. **Base Predictions Path**: Encoder → [Context, Value, Skill] → Prediction Head → **Base Predictions** → **BCE Loss**
2. **Incremental Mastery Path**: Values → Learning Gains → Mastery Accumulation → Threshold Mechanism → **Incremental Mastery Predictions** → **Incremental Mastery Loss**

**Architecture Notes**:
- ✅ **Dual Predictions**: TWO independent prediction branches (base + incremental mastery)
- ✅ **Mastery Accumulation**: ACTIVE (`use_mastery_head=true`) - Recursive tracking of skill mastery with learnable thresholds
- ✅ **Learning Gains Computation**: ACTIVE - Computed from Values to update mastery (internal use only)
- ✅ **Incremental Mastery Predictions**: ACTIVE - Separate predictions from threshold mechanism (sigmoid((mastery - threshold) / temperature))
- ✅ **Base Predictions**: ACTIVE - Standard predictions from concatenation head (NOT overridden)
- ✅ **Dual Loss Functions**: 
  - BCE Loss on Base Predictions (standard)
  - Incremental Mastery Loss on Threshold Predictions (weight=0.1, new)
- ✅ **Constraint Losses**: ACTIVE - Interpretability losses computed on mastery and internal gains
- ❌ **Gains Head Output**: DEACTIVATED (`use_gain_head=false`) - Gains not exposed in model output
- ❌ **Gains D-dimensional Output**: DEACTIVATED - `projected_gains_d` not included in output
- ❌ **Attention-Derived Gains** (intrinsic_gain_attention mode): DEACTIVATED (all related code commented out)

**Result**: The model produces two independent predictions: (1) Base predictions from the standard prediction head for primary BCE loss, and (2) Incremental mastery predictions from the threshold mechanism for interpretability-driven mastery loss. Mastery and base predictions are included in output, gains remain internal.

## Architecture

The architecture follows a modular design where features such as projection heads or interpretability computation can be controlled via code paths in GainAKT3Exp.

The diagram below illustrates the **complete architecture** inherited from GainAKT3 base class:

**Visual Legend:**
- **Double-border boxes** (`[[...]]`): **Input/Output data** (tensors, embeddings, intermediate representations) - white background with dark borders
- **Single-border boxes** (`[...]`): **Processing operations** (embeddings tables, transformations, neural network layers)
- **Green components**: Core augmented architecture (Skill Embedding, Dynamic Value Stream, Auxiliary Losses, Monitoring)
- **Blue components**: Recursive Mastery Accumulation (deterministic temporal constraint: mastery_{t+1} = mastery_t + α·gain_t where gain_t comes from Value stream) - **ACTIVE**
- **Orange components**: Semantic modules (Alignment, Global Alignment, Retention, Lag Gains) that enable interpretability recovery
- **Red components with ⚠️**: Features with restricted output visibility
  - **Gains Output** (use_gain_head=false) - Computed internally for mastery but NOT exposed in model output
  - **Attention-Derived Gains** (intrinsic_gain_attention=false) - **CODE COMMENTED OUT**
- **Circles (Hubs)**: Convergence/distribution points where multiple data flows aggregate and route to multiple outputs

**Key Architectural Note**: 
- **Mastery Accumulation**: ✅ FULLY ACTIVE (`use_mastery_head=true`) - Gains computed from Values, accumulated into mastery, used for threshold predictions, mastery included in output
- **Gains Head**: ⚠️ OUTPUT SUPPRESSED (`use_gain_head=false`) - Gains computed internally for mastery accumulation but NOT included in model output (lines 479-482)
- **Data Flow**: Values → ReLU → Aggregated Gains → Mastery Accumulation (recursive) → Threshold Predictions → Output (predictions + mastery only)


**⚠️ ARCHITECTURE STATUS**: The diagram shows the complete architecture. All mastery-related computation is ACTIVE (lines 318-465). Gains are computed internally for mastery accumulation but the `use_gain_head` parameter controls whether gains appear in the output dictionary. Current config: mastery output ✅ included, gains output ❌ suppressed.

```mermaid
graph TD
    subgraph "Input Layer"
        direction LR
        Input_q[["Input Questions (q)<br/>Shape: [B, L]"]]
        Input_r[["Input Responses (r)<br/>Shape: [B, L]"]]
        Ground_Truth[["Ground Truth Responses"]]
    end

    subgraph "Tokenization & Embedding"
        direction TB

        
        Tokens[["Interaction Tokens<br/>(q + num_c * r)<br/>Shape: [B, L]"]]
        
        Context_Emb["Context Embedding Table"]
        Value_Emb["Value Embedding Table"]
        Skill_Emb["Skill Embedding Table"]

        Tokens --> Context_Emb
        Tokens --> Value_Emb
        Input_q --> Skill_Emb

        Context_Seq[["Context Sequence<br/>Shape: [B, L, D]"]]
        Value_Seq[["Value Sequence<br/>Shape: [B, L, D]"]]
        Pos_Emb[["Positional Embeddings<br/>Shape: [B, L, D]"]]
        
        Context_Emb --> Context_Seq
        Value_Emb --> Value_Seq

        Context_Seq_Pos[["Context + Positional<br/>Shape: [B, L, D]"]]
        Value_Seq_Pos[["Value + Positional<br/>Shape: [B, L, D]"]]
        
        Context_Seq --"Add"--> Context_Seq_Pos
        Pos_Emb --"Add"--> Context_Seq_Pos
        Value_Seq --"Add"--> Value_Seq_Pos
        Pos_Emb --"Add"--> Value_Seq_Pos
    end

    Input_q --> Tokens
    Input_r --> Tokens

    subgraph "Dynamic Encoder Block"
        direction TB
        
        Encoder_Input_Context[["Input: Context Sequence<br/>[B, L, D]"]]
        Encoder_Input_Value[["Input: Value Sequence<br/>[B, L, D]"]]

        subgraph "Attention Mechanism"
            direction TB
            
            Attn_Input_Context[["Input: Context<br/>[B, L, D]"]]
            Attn_Input_Value[["Input: Value<br/>[B, L, D]"]]

            Proj_Q["Q = Linear(Context)<br/>[B, H, L, Dk]"]
            Proj_K["K = Linear(Context)<br/>[B, H, L, Dk]"]
            Proj_V["V = Linear(Value)<br/>[B, H, L, Dk]"]
            
            Attn_Input_Context --> Proj_Q
            Attn_Input_Context --> Proj_K
            Attn_Input_Value --> Proj_V

            Scores["Scores = $\\frac{Q \\cdot K^T}{\\sqrt{D_k}}$<br/>[B, H, L, L]"]
            Proj_Q --> Scores
            Proj_K --> Scores
            
            Weights[["Weights = softmax(Scores)<br/>[B, H, L, L]"]]
            Scores --> Weights

            Attn_Output_Heads[["Attn Output (Heads)<br/>[B, H, L, Dk]"]]
            Weights --> Attn_Output_Heads
            Proj_V --> Attn_Output_Heads

            Attn_Output[["Reshaped Attn Output<br/>[B, L, D]"]]
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
        
        Encoder_Output_Ctx[["Output: Context (h)<br/>[B, L, D]"]]
        AddNorm2 --> Encoder_Output_Ctx

        Encoder_Output_Val[["Output: Value (v)<br/>[B, L, D]"]]
        AddNorm_Val --> Encoder_Output_Val
    end

    Context_Seq_Pos --> Encoder_Input_Context
    Value_Seq_Pos --> Encoder_Input_Value

    subgraph "Prediction Head (Base Predictions)"
        direction TB
        
        Pred_Input_h[["Input: Knowledge State (h)<br/>[B, L, D]"]]
        Pred_Input_v[["Input: Value State (v)<br/>[B, L, D]"]]
        Pred_Input_s[["Input: Target Skill (s)<br/>[B, L, D]"]]

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

    subgraph "Final Outputs (Dual Prediction Architecture)"
        direction TB
        Base_Predictions[["Base Predictions<br/>(from Prediction Head)<br/>[B, L]"]]
        IM_Predictions[["Incremental Mastery Predictions<br/>(from Threshold Mechanism)<br/>[B, L]<br/>✅ ACTIVE when use_mastery_head=true"]]
    end

    Sigmoid --> Base_Predictions

    %% Learning Gains and Mastery Computation (ACTIVE in GainAKT3Exp)
    Learning_Gains_D[["Learning Gains (D-dim)<br/>ReLU(Value Sequence)<br/>[B, L, D]"]]
    Aggregated_Gains[["Aggregated Gains (scalar)<br/>mean(Learning_Gains_D, dim=-1)<br/>[B, L, 1]"]]
    Projected_Gains_Internal[["Projected Gains (per-skill)<br/>sigmoid(Aggregated) expanded<br/>[B, L, num_skills]<br/>(Internal only - not in output⚠️)"]]
    
    Encoder_Output_Val --> Learning_Gains_D
    Learning_Gains_D --> Aggregated_Gains
    Aggregated_Gains --> Projected_Gains_Internal
    
    %% Mastery accumulation output (ACTIVE - included in output)
    Projected_Mastery_Output[["Projected Mastery<br/>[B, L, num_skills]<br/>(✅ Included in output)"]]

    %% Recursive Mastery Accumulation (ACTIVE in GainAKT3Exp)
    subgraph "Recursive Mastery Accumulation<br/>(✅ ACTIVE - use_mastery_head=true)"
        direction TB
        
        Gain_Input[["Learning Gain from Projected Gains<br/>gain_t (interaction's contribution)<br/>[B, 1] per skill"]]
        Mastery_Prev[["Previous Skill Mastery<br/>mastery_{t-1}[skill]<br/>[B, 1]"]]
        
        Scale_Op["× α<br/>(learning rate α=0.1)"]
        Sum_Op["+ (mastery accumulation)<br/>mastery_t[skill] = mastery_{t-1}[skill] + α·gain_t"]
        Tanh_Op["tanh normalization<br/>(soft boundary)"]
        Clamp_Op["Clamp[0,1]<br/>(bounded mastery)"]
        
        Mastery_Current[["Updated Skill Mastery<br/>mastery_t[skill]<br/>[B, 1]"]]
        
        Gain_Input --> Scale_Op
        Scale_Op --> Sum_Op
        Mastery_Prev --> Sum_Op
        Sum_Op --> Tanh_Op
        Tanh_Op --> Clamp_Op
        Clamp_Op --> Mastery_Current
        
        %% Temporal feedback loop - mastery persists across interactions
        Mastery_Current -.->|"persists as mastery_{t-1}<br/>for next interaction with this skill"| Mastery_Prev
    end
    
    %% Active connections in GainAKT3Exp
    Projected_Gains_Internal -->|"✅ ACTIVE<br/>(used for mastery)"| Gain_Input
    Mastery_Current -->|"✅ ACTIVE<br/>(accumulates to output)"| Projected_Mastery_Output
    
    %% Incremental Mastery Predictions (ACTIVE - separate from base predictions)
    Learnable_Threshold[["Learnable Mastery Threshold<br/>per skill [num_skills]<br/>(trainable parameter)"]]
    Threshold_Pred["Incremental Mastery Prediction<br/>sigmoid((mastery - threshold) / temperature)<br/>✅ Separate prediction branch"]
    
    Projected_Mastery_Output --> Threshold_Pred
    Learnable_Threshold --> Threshold_Pred
    Threshold_Pred --> IM_Predictions

    %% Circle Connectors (Aggregation/Distribution Hubs)
    Mastery_Hub(("Mastery<br/>Hub<br/>✅ ACTIVE"))
    Gain_Hub(("Gain<br/>Hub<br/>⚠️ INTERNAL"))
    Encoder_Hub(("Encoder<br/>Hub"))
    Base_Pred_Hub(("Base<br/>Predictions<br/>Hub"))
    IM_Pred_Hub(("Incremental<br/>Mastery<br/>Predictions<br/>Hub"))
    
    Projected_Mastery_Output --> Mastery_Hub
    Projected_Gains_Internal --> Gain_Hub
    Encoder_Output_Ctx --> Encoder_Hub
    Encoder_Output_Val --> Encoder_Hub
    Base_Predictions --> Base_Pred_Hub
    IM_Predictions --> IM_Pred_Hub

    %% Semantic Feedback Loop (orange)
    Global_Alignment["Global Alignment Pass<br/>population coherence"]
    Residual_Alignment["Residual Alignment<br/>variance capture"]
    
    Mastery_Hub -->|"Global Align"| Global_Alignment
    Global_Alignment --> Residual_Alignment
    Residual_Alignment -.feedback.-> Projected_Mastery_Output

    %% Loss Framework
    subgraph "Loss Framework (Dual-Prediction Architecture)"
        direction LR
        
        subgraph "Primary Losses"
            direction TB
            BCE_Loss["BCE Loss<br/>(Base Predictions)"]
            IM_Loss["Incremental Mastery Loss<br/>weight=0.1<br/>(✅ NEW - Threshold Predictions)"]
        end
        
        subgraph "Constraint Losses (✅ ACTIVE)"
            direction TB
            Monotonicity_Loss["Monotonicity<br/>weight=0.1<br/>(✅ ACTIVE on mastery)"]
            Mastery_Perf_Loss["Mastery-Perf<br/>weight=0.5<br/>(✅ ACTIVE on mastery)"]
            Gain_Perf_Loss["Gain-Perf<br/>weight=0.5<br/>(✅ ACTIVE on internal gains)"]
            Sparsity_Loss["Sparsity<br/>weight=0.2<br/>(✅ ACTIVE on internal gains)"]
            Consistency_Loss["Consistency<br/>weight=0.3<br/>(✅ ACTIVE on mastery/gains)"]
            NonNeg_Loss["Non-Negativity<br/>weight=0.0<br/>(✅ ACTIVE - effectively 0)"]
        end
        
        subgraph "Semantic Losses (Orange)"
            direction TB
            Alignment_Loss["Local Alignment"]
            Retention_Loss["Retention"]
            Lag_Gain_Loss["Lag Gain"]
        end
        
        Total_Loss["Total Loss<br/>BCE + IM_Loss + Constraints + Semantics<br/>Warmup & Share Cap Scheduling"]
    end

    %% Connections via Hubs - Dual Prediction Architecture
    Base_Pred_Hub -->|"BCE"| BCE_Loss
    Ground_Truth --> BCE_Loss
    
    IM_Pred_Hub -->|"✅ NEW"| IM_Loss
    Ground_Truth --> IM_Loss

    Mastery_Hub -->|"Monotonicity"| Monotonicity_Loss
    Mastery_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss
    Mastery_Hub -->|"Consistency"| Consistency_Loss
    Mastery_Hub -->|"Retention"| Retention_Loss
    
    Gain_Hub -->|"Gain-Perf"| Gain_Perf_Loss
    Gain_Hub -->|"Sparsity"| Sparsity_Loss
    Gain_Hub -->|"Consistency"| Consistency_Loss
    Gain_Hub -->|"NonNeg"| NonNeg_Loss
    Gain_Hub -->|"Lag"| Lag_Gain_Loss
    
    Base_Pred_Hub -->|"Mastery-Perf"| Mastery_Perf_Loss
    Base_Pred_Hub -->|"Gain-Perf"| Gain_Perf_Loss
    
    Encoder_Hub -->|"Alignment"| Alignment_Loss
    Base_Pred_Hub -->|"Alignment"| Alignment_Loss

    %% All losses to Total
    BCE_Loss --> Total_Loss
    IM_Loss --> Total_Loss
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
    Base_Pred_Hub -->|"to Monitor"| Monitor_Hub
    IM_Pred_Hub -->|"to Monitor"| Monitor_Hub
    Monitor_Hub -->|"Monitor Output"| Monitor_Hook

    %% Styling
    classDef new_component fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef semantic_component fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    classDef deactivated_component fill:#ffcdd2,stroke:#c62828,stroke-width:3px,stroke-dasharray:5 5,font-style:italic
    classDef accumulation_component fill:#e0e0e0,stroke:#757575,stroke-width:2px,stroke-dasharray:3 3
    classDef io_data fill:#ffffff,stroke:#333333,stroke-width:3px
    classDef io_data_unused fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,stroke-dasharray:3 3
    
    %% Individual hub colors with distinct visual styles
    classDef mastery_hub fill:#e8f5e8,stroke:#00ff00,stroke-width:4px
    classDef gain_hub fill:#e8f5e8,stroke:#008800,stroke-width:4px
    classDef encoder_hub fill:#fff3e0,stroke:#ffa500,stroke-width:4px
    classDef pred_hub fill:#e3f2fd,stroke:#888888,stroke-width:4px
    classDef monitor_hub fill:#f3e5f5,stroke:#800080,stroke-width:4px

    class Ground_Truth,Skill_Emb,BCE_Loss,Total_Loss,Monitor_Hook new_component
    class Proj_Mastery,Proj_Gain,Monotonicity_Loss,Mastery_Perf_Loss,Gain_Perf_Loss,Sparsity_Loss,Consistency_Loss,NonNeg_Loss deactivated_component
    class Gain_Input,Mastery_Prev,Mastery_Current,ReLU_Op,Scale_Op,Sum_Op,Clamp_Op accumulation_component
    class Alignment_Loss,Global_Alignment,Residual_Alignment,Retention_Loss,Lag_Gain_Loss semantic_component
    class Projected_Mastery_Output,Projected_Gain_Output io_data_unused
    
    class Mastery_Hub mastery_hub
    class Gain_Hub gain_hub
    class Encoder_Hub encoder_hub
    class Pred_Hub pred_hub
    class Monitor_Hub monitor_hub
    
    %% Input/Output data boxes with darker borders
    class Input_q,Input_r,Tokens,Context_Seq,Value_Seq,Pos_Emb,Context_Seq_Pos,Value_Seq_Pos io_data
    class Encoder_Input_Context,Encoder_Input_Value,Attn_Input_Context,Attn_Input_Value io_data
    class Weights,Attn_Output_Heads,Attn_Output,Encoder_Output_Ctx,Encoder_Output_Val io_data
    class Pred_Input_h,Pred_Input_v,Pred_Input_s,Predictions io_data
    class Projected_Mastery_Output,Projected_Gain_Output io_data

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

---

### Architecture Summary - Current State

**What's ACTIVE** (actually executed):
- ✅ **Dual-Stream Encoder**: Context and Value streams through transformer blocks
- ✅ **Learning Gains Computation**: ReLU(Values) → aggregated to scalar → expanded per-skill
- ✅ **Recursive Mastery Accumulation**: For t in 1..seq_len: mastery[skill,t] = mastery[skill,t-1] + α × gain[t]
- ✅ **Dual-Prediction Architecture**: TWO independent prediction outputs:
  - **Base Predictions**: From standard prediction head [context, value, skill] → MLP → sigmoid
  - **Incremental Mastery Predictions**: From threshold mechanism sigmoid((mastery - learnable_threshold) / temperature)
- ✅ **Mastery Output**: `projected_mastery` included in model output dictionary
- ✅ **Base Predictions Output**: `predictions` from prediction head in model output dictionary
- ✅ **Incremental Mastery Predictions Output**: `incremental_mastery_predictions` from threshold mechanism in output dictionary
- ✅ **Dual Loss Functions**:
  - **BCE Loss**: Binary cross-entropy on **base predictions** vs ground truth (primary loss)
  - **Incremental Mastery Loss**: Binary cross-entropy on **incremental mastery predictions** vs ground truth (weight=0.1)
- ✅ **Constraint Losses**: ALL ACTIVE - `compute_interpretability_loss()` called with mastery and internal gains:
  - Monotonicity Loss (weight=0.1) - ✅ enforces mastery non-decrease
  - Mastery-Performance Loss (weight=0.5) - ✅ aligns mastery with correctness
  - Gain-Performance Loss (weight=0.5) - ✅ correlates gains with performance
  - Sparsity Loss (weight=0.2) - ✅ encourages sparse gain attribution
  - Consistency Loss (weight=0.3) - ✅ aligns mastery changes with gains
  - Non-Negativity Loss (weight=0.0) - ✅ active but zero weight
- ✅ **Semantic Losses**: Alignment, Retention, Lag Gain (optional, controlled by weights)

**What's SUPPRESSED** (computed but not included in output):
- ⚠️ **Gains Head Output**: `projected_gains` computed internally for mastery but **NOT included** in output dictionary (controlled by `use_gain_head=false`)
- ⚠️ **Gains D-dimensional Output**: `projected_gains_d` not included in output (requires both heads enabled)

**What's INACTIVE** (code commented out):
- ❌ **Attention-Derived Gains**: Intrinsic gain attention mode completely commented out

**Code Location**: The dual-prediction computation is in `gainakt3_exp.py` lines 318-470:
```python
# Standard prediction head computes base predictions
logits = self.prediction_head(concatenated).squeeze(-1)
predictions = torch.sigmoid(logits)  # Base predictions - NOT overridden

if self.use_mastery_head:  # Currently true
    # Compute learning gains from Values
    learning_gains_d = torch.relu(value_seq)
    aggregated_gains = learning_gains_d.mean(dim=-1, keepdim=True)
    projected_gains = torch.sigmoid(aggregated_gains).expand(-1, -1, self.num_c)
    
    # Recursive mastery accumulation
    for t in range(1, seq_len):
        # Update mastery for practiced skill
        ...
    
    # Incremental mastery predictions (SEPARATE from base predictions)
    incremental_mastery_predictions = sigmoid((mastery - threshold) / temperature)
    # NOTE: Does NOT override predictions - both coexist for dual loss
```

Output control (lines 479-490):
```python
output = {
    'predictions': predictions,  # Base predictions from prediction head
    'incremental_mastery_predictions': incremental_mastery_predictions  # Threshold predictions
}
if projected_gains is not None and self.use_gain_head:  # Currently false
    output['projected_gains'] = projected_gains  # Suppressed
```

Loss computation (lines 498-516):
```python
# Standard BCE loss on base predictions
# (computed in training script)

# New: Incremental mastery loss on threshold predictions
if incremental_mastery_predictions is not None:
    incremental_mastery_loss = F.binary_cross_entropy(
        incremental_mastery_predictions, r.float(), reduction='mean'
    )
```

**Result**: The model produces TWO independent predictions and computes TWO losses. Base predictions train the standard prediction head, while incremental mastery predictions provide interpretability-driven supervision on the mastery evolution. Gains remain internal.

---

**Tensor Dimensions Legend:**
- **B**: Batch size (number of students in a training/evaluation batch)
- **L**: Sequence length (maximum number of interactions per student)
- **D**: Model dimension (hidden size of transformer layers, e.g., 256)
- **num_c**: Number of concepts/skills in the dataset (e.g., 100 for ASSIST2015)
- **num_q**: Number of questions in the dataset (varies by dataset)
- **H**: Number of attention heads in multi-head attention


## Sequence

This section shows the temporal flow of operations during training and evaluation, illustrating how components interact across time.

### Model Sequence

GainAKT3Exp

### Training Sequence

The training workflow involves the experiment launcher loading defaults, creating the model, iterating through epochs with monitoring hooks, and saving artifacts.

```mermaid
sequenceDiagram
    participant User
    participant Launcher as run_repro_experiment.py
    participant Config as parameter_default.json
    participant TrainScript as train_gainakt3exp.py
    participant Factory as create_exp_model()
    participant Model as GainAKT3Exp
    participant Monitor as InterpretabilityMonitor
    participant Data as DataLoader
    participant Optimizer
    participant Disk as File System
    
    User->>Launcher: python run_repro_experiment.py<br/>--model gainakt3exp<br/>--dataset assist2015<br/>--epochs 12<br/>[--param overrides...]
    
    activate Launcher
    Launcher->>Config: Load defaults
    Config-->>Launcher: All parameter defaults
    
    Launcher->>Launcher: Apply CLI overrides
    Launcher->>Launcher: Generate experiment directory<br/>{timestamp}_{model}_{title}_{uid}
    
    Launcher->>TrainScript: Launch training with<br/>all explicit parameters
    deactivate Launcher
    
    activate TrainScript
    TrainScript->>Factory: create_exp_model(config)
    activate Factory
    Factory->>Factory: Validate all 22 required params
    Factory->>Model: Initialize GainAKT3Exp<br/>(d_model, n_heads, etc.)
    Factory-->>TrainScript: model instance
    deactivate Factory
    
    TrainScript->>Monitor: Create monitor instance
    TrainScript->>Model: set_monitor(monitor)
    Model-->>TrainScript: Monitor hook registered
    
    TrainScript->>Disk: Save initial config.json<br/>(defaults + train/eval/trajectory commands)
    
    loop Each Epoch (1 to 12)
        TrainScript->>TrainScript: Reset epoch metrics
        
        loop Each Batch
            TrainScript->>Data: Get batch (q, r, qry)
            Data-->>TrainScript: Batch tensors
            
            TrainScript->>Model: forward_with_states(q, r, qry,<br/>batch_idx=current_batch)
            activate Model
            
            Model->>Model: Tokenize inputs<br/>Embed context & value streams
            Model->>Model: Pass through encoder blocks<br/>(dual-stream attention)
            Model->>Model: Generate predictions<br/>(MLP head)
            Model->>Model: Compute mastery/gains<br/>(projection heads + recursion)
            Model->>Model: Compute interpretability_loss<br/>(constraint losses)
            
            alt batch_idx % monitor_frequency == 0
                Model->>Monitor: Call monitor hook<br/>(context, value, mastery, gains, predictions)
                activate Monitor
                Monitor->>Monitor: Compute correlations<br/>Log statistics
                Monitor-->>Model: Monitoring complete
                deactivate Monitor
            end
            
            Model-->>TrainScript: {predictions, logits,<br/>context_seq, value_seq,<br/>mastery, gains, interp_loss}
            deactivate Model
            
            TrainScript->>TrainScript: Compute BCE loss<br/>(from logits)
            TrainScript->>TrainScript: total_loss = BCE + interp_loss
            
            TrainScript->>Optimizer: Backward pass
            Optimizer->>Model: Update parameters
            Optimizer-->>TrainScript: Step complete
            
            TrainScript->>TrainScript: Accumulate batch metrics
        end
        
        TrainScript->>TrainScript: Compute validation metrics<br/>(AUC, accuracy)
        
        alt Best validation AUC
            TrainScript->>Disk: Save checkpoint<br/>model_epoch{n}.pt
        end
        
        TrainScript->>Disk: Save trajectory data<br/>(mastery/gain correlations)
        
        TrainScript->>TrainScript: Log epoch summary<br/>(train/val AUC, correlations)
    end
    
    TrainScript->>Disk: Save final metrics.json<br/>(all epochs' results)
    TrainScript->>Disk: Save best checkpoint metadata
    
    TrainScript-->>User: Training complete<br/>Saved to {run_dir}
    deactivate TrainScript
```

**Key Training Flow Characteristics:**
- **Zero Defaults**: All parameters explicitly loaded from config or CLI
- **Monitoring Hooks**: Periodic state capture every N batches (default: 50)
- **Dual Loss**: BCE (prediction accuracy) + interpretability loss (constraints + semantics)
- **Artifact Persistence**: Complete reproducibility via saved config + checkpoints
- **Correlation Tracking**: Mastery/gain correlations computed per epoch

### Evaluation Sequence

The evaluation workflow loads a trained model checkpoint, runs inference on test data, computes metrics including correlations, and saves results.

```mermaid
sequenceDiagram
    participant User
    participant EvalScript as eval_gainakt3exp.py
    participant Config as run_dir/config.json
    participant Factory as create_exp_model()
    participant Model as GainAKT3Exp
    participant Checkpoint as model_*.pt
    participant Data as Test DataLoader
    participant Disk as File System
    
    User->>EvalScript: python eval_gainakt3exp.py<br/>--run_dir {experiment_dir}<br/>--ckpt_name model_best.pt<br/>[--num_corr_students 3000]
    
    activate EvalScript
    EvalScript->>Config: Load config.json
    Config-->>EvalScript: All parameters + metadata
    
    EvalScript->>Factory: create_exp_model(config)
    activate Factory
    Factory->>Factory: Validate parameters
    Factory->>Model: Initialize GainAKT3Exp
    Factory-->>EvalScript: model instance
    deactivate Factory
    
    EvalScript->>Checkpoint: Load checkpoint state_dict
    Checkpoint-->>EvalScript: Trained parameters
    EvalScript->>Model: load_state_dict(checkpoint)
    Model-->>EvalScript: Parameters loaded
    
    EvalScript->>Model: eval() mode
    EvalScript->>Data: Load test dataset
    Data-->>EvalScript: Test data ready
    
    EvalScript->>EvalScript: Initialize result accumulators<br/>(predictions, labels, mastery, gains)
    
    loop Each Test Batch
        EvalScript->>Data: Get batch (q, r, qry)
        Data-->>EvalScript: Batch tensors
        
        EvalScript->>Model: forward_with_states(q, r, qry)
        activate Model
        Model->>Model: Tokenize & embed
        Model->>Model: Encoder blocks (no gradients)
        Model->>Model: Generate predictions
        Model->>Model: Compute mastery/gains<br/>(for correlation)
        Model-->>EvalScript: {predictions, mastery, gains}
        deactivate Model
        
        EvalScript->>EvalScript: Accumulate predictions & labels
        EvalScript->>EvalScript: Accumulate mastery/gains<br/>(for selected students)
    end
    
    EvalScript->>EvalScript: Compute test AUC<br/>(all predictions vs labels)
    EvalScript->>EvalScript: Compute test accuracy
    
    alt Correlation computation requested
        EvalScript->>EvalScript: Sample students<br/>(default: 3000 students)
        
        loop Each sampled student
            EvalScript->>EvalScript: Extract student trajectory<br/>(mastery_seq, gains_seq, responses)
            EvalScript->>EvalScript: Compute Pearson correlation<br/>(mastery vs performance)
            EvalScript->>EvalScript: Compute Pearson correlation<br/>(gains vs performance)
            EvalScript->>EvalScript: Accumulate valid correlations
        end
        
        EvalScript->>EvalScript: Average mastery correlation<br/>across students
        EvalScript->>EvalScript: Average gain correlation<br/>across students
    end
    
    EvalScript->>Disk: Save eval_results_{timestamp}.json<br/>{AUC, accuracy, correlations, num_students}
    
    EvalScript-->>User: Evaluation complete<br/>Test AUC: {auc}<br/>Mastery Corr: {mastery_corr}<br/>Gain Corr: {gain_corr}
    deactivate EvalScript
```

**Key Evaluation Flow Characteristics:**
- **Checkpoint Loading**: Restores exact trained model state from disk
- **No Gradient Computation**: Model in eval mode, torch.no_grad() context
- **Correlation Sampling**: Configurable student sample size (default: 3000)
- **Trajectory Analysis**: Per-student mastery/gain sequences correlated with performance
- **Result Persistence**: Timestamped evaluation results saved to run directory

### Learning Trajectory Analysis Sequence

Individual student learning progressions can be extracted post-hoc using the trajectory analysis script (command auto-saved in config.json).

```mermaid
sequenceDiagram
    participant User
    participant TrajScript as learning_trajectories.py
    participant Config as run_dir/config.json
    participant Factory as create_exp_model()
    participant Model as GainAKT3Exp
    participant Checkpoint as model_best.pt
    participant Data as Test DataLoader
    participant Console
    
    User->>TrajScript: python learning_trajectories.py<br/>--run_dir {experiment_dir}<br/>--num_students 10<br/>--min_steps 10
    
    activate TrajScript
    TrajScript->>Config: Load config.json
    Config-->>TrajScript: Parameters
    
    TrajScript->>Factory: create_exp_model(config)
    Factory->>Model: Initialize model
    Factory-->>TrajScript: model instance
    
    TrajScript->>Checkpoint: Load best checkpoint
    Checkpoint-->>TrajScript: Trained parameters
    TrajScript->>Model: load_state_dict(checkpoint)
    
    TrajScript->>Data: Load test dataset
    TrajScript->>TrajScript: Filter students with<br/>≥ min_steps interactions
    TrajScript->>TrajScript: Sample diverse students<br/>(varied sequence lengths)
    
    loop Each selected student
        TrajScript->>Model: forward_with_states(student_seq)
        Model-->>TrajScript: {predictions, mastery, gains}
        
        TrajScript->>TrajScript: Extract trajectory:<br/>per-timestep (skill, true, pred, mastery, gain)
        
        TrajScript->>Console: Print student header:<br/>Global Index | Interactions | Unique Skills | Accuracy
        
        TrajScript->>Console: Print trajectory table:<br/>Step | Skill | True | Pred | Match | Gain | Mastery
        
        loop Each timestep
            TrajScript->>Console: t | skill_id | 0/1 | 0/1 | ✓/✗ | gain_val | mastery_val
        end
        
        TrajScript->>Console: Print separator
    end
    
    TrajScript-->>User: Trajectory analysis complete<br/>{num_students} students displayed
    deactivate TrajScript
```

**Key Trajectory Analysis Characteristics:**
- **Compact Format**: Tabular display with student summary statistics
- **Match Indicator**: Visual ✓/✗ showing prediction correctness
- **Per-Timestep Detail**: Shows skill practiced, true/predicted response, gains, mastery
- **Diverse Sampling**: Selects students with varying interaction counts
- **Post-Hoc Analysis**: No training required, works with any saved checkpoint

### Internal Model Flow: GainAKT3Exp forward_with_states()

This detailed sequence diagram shows the internal data flow within a single forward pass, tracking how Context (h) and Value (v) streams flow through the model and feed into predictions, interpretability projections, and losses.

```mermaid
sequenceDiagram
    participant Input as Input Tensors<br/>(q, r, qry)
    participant Tokenize as Tokenization
    participant CtxEmb as Context Embedding
    participant ValEmb as Value Embedding
    participant PosEmb as Positional Embedding
    participant Encoder as Encoder Blocks<br/>(N layers)
    participant CtxStream as Context Stream (h)<br/>[B, L, D]
    participant ValStream as Value Stream (v)<br/>[B, L, D]
    participant SkillEmb as Skill Embedding
    participant PredHead as Prediction Head<br/>(MLP)
    participant MasteryHead as Mastery Head<br/>Linear(D, num_c)
    participant GainHead as Gain Head<br/>Linear(D, num_c)
    participant Recursion as Recursive<br/>Accumulation
    participant Losses as Loss Computation
    participant Monitor as Monitor Hook
    participant Output as Output Dict
    
    Input->>Tokenize: q, r (questions, responses)
    Tokenize->>Tokenize: interaction_tokens = q + num_c * r
    
    Note over CtxEmb,ValEmb: Dual Embedding Tables
    Tokenize->>CtxEmb: interaction_tokens
    Tokenize->>ValEmb: interaction_tokens
    CtxEmb->>CtxStream: context_seq [B, L, D]
    ValEmb->>ValStream: value_seq [B, L, D]
    
    PosEmb->>CtxStream: Add positional encodings
    PosEmb->>ValStream: Add positional encodings
    
    Note over CtxStream,ValStream: Pass through N encoder blocks<br/>with dual-stream attention
    
    loop Each Encoder Block
        CtxStream->>Encoder: context_seq
        ValStream->>Encoder: value_seq
        
        Note over Encoder: Q, K = Linear(context_seq)<br/>V = Linear(value_seq)<br/>Attention = softmax(QK^T/√d)V
        
        Encoder->>Encoder: attn_output = Attention(Q, K, V)
        Encoder->>Encoder: context_seq = LayerNorm(context + attn + FFN(attn))
        Encoder->>Encoder: value_seq = LayerNorm(value + attn)
        
        Encoder->>CtxStream: Updated context_seq
        Encoder->>ValStream: Updated value_seq
    end
    
    rect rgb(200, 230, 255)
        Note over CtxStream: Context Output (h)<br/>Encoder_Output_Ctx [B, L, D]<br/>FLOWS TO 3 DESTINATIONS ↓
    end
    
    rect rgb(255, 220, 230)
        Note over ValStream: Value Output (v)<br/>Encoder_Output_Val [B, L, D]<br/>FLOWS TO 3 DESTINATIONS ↓
    end
    
    par Context Stream (h) flows to 3 destinations
        Note over CtxStream,PredHead: FLOW 1: Context → Prediction Head
        CtxStream->>PredHead: context_seq (h) [B, L, D]
    and
        Note over CtxStream,MasteryHead: FLOW 2: Context → Mastery Projection
        CtxStream->>MasteryHead: context_seq (h) [B, L, D]
    and
        Note over CtxStream,Monitor: FLOW 3: Context → Monitor/Output
        CtxStream->>Output: context_seq (for monitoring)
    end
    
    par Value Stream (v) flows to 3 destinations
        Note over ValStream,PredHead: FLOW 1: Value → Prediction Head
        ValStream->>PredHead: value_seq (v) [B, L, D]
    and
        Note over ValStream,Recursion: FLOW 2: Value → Recursive Mastery Accumulation<br/>(GainAKT3Exp Innovation: Values ARE learning gains)
        ValStream->>Recursion: value_seq = learning gains<br/>[B, L, D]<br/>(each interaction's contribution)
    and
        Note over ValStream,Monitor: FLOW 3: Value → Monitor/Output
        ValStream->>Output: value_seq (for monitoring)
    end
    
    Note over PredHead: Concatenate [h, v, s]
    Input->>SkillEmb: qry (target skills)
    SkillEmb->>PredHead: target_skill_emb [B, L, D]
    PredHead->>PredHead: concat = [context, value, skill]<br/>[B, L, 3*D]
    PredHead->>PredHead: logits = MLP(concat)<br/>[B, L]
    PredHead->>PredHead: predictions = sigmoid(logits)
    PredHead->>Output: predictions [B, L]
    PredHead->>Output: logits [B, L]
    
    Note over MasteryHead,Recursion: Mastery Projection Path
    MasteryHead->>MasteryHead: mastery_raw = Linear(context)<br/>[B, L, num_c]
    Note over MasteryHead: (Initial estimate - refined by recursion)
    
    rect rgb(180, 230, 255)
        Note over Recursion: Recursive Mastery Accumulation<br/>Values (learning gains) increment skill mastery<br/>For each interaction's skill:<br/>mastery[skill, t] = mastery[skill, t-1] + α × ReLU(value[t])
    end
    
    Input->>Recursion: questions (q) [B, L]<br/>(to identify relevant skill)
    
    loop Each timestep t (1 to L)
        Recursion->>Recursion: skill = question[t]
        Recursion->>Recursion: gain = ReLU(value[t]) × α (α=0.1)
        Recursion->>Recursion: mastery[skill, t] = mastery[skill, t-1] + gain
        Recursion->>Recursion: mastery[skill, t] = clamp(mastery[skill, t], 0, 1)
    end
    
    Recursion->>Output: projected_mastery [B, L, num_c]<br/>(accumulated across timesteps)
    Recursion->>Output: learning_gains [B, L, D]<br/>(directly from Values)
    
    Note over Losses: Loss Computation Sources
    
    rect rgb(255, 240, 200)
        Note over Losses: LOSS INPUTS:<br/>1. predictions (from Pred Head)<br/>2. projected_mastery (from Recursion)<br/>3. projected_gains (from Recursion)<br/>4. responses (ground truth)<br/>5. questions (for skill masks)
    end
    
    Output->>Losses: predictions [B, L]
    Output->>Losses: projected_mastery [B, L, num_c]
    Output->>Losses: projected_gains [B, L, num_c]
    Input->>Losses: responses (r) [B, L]
    Input->>Losses: questions (q) [B, L]
    
    Losses->>Losses: Compute skill masks<br/>(Q-matrix structure)
    
    par Constraint Losses (6 losses computed in parallel)
        Losses->>Losses: L1: Non-Negative Gains<br/>clamp(-gains, min=0).mean()
    and
        Losses->>Losses: L2: Monotonicity<br/>clamp(mastery[t] - mastery[t+1], min=0).mean()
    and
        Losses->>Losses: L3: Mastery-Performance<br/>low_mastery_correct + high_mastery_incorrect
    and
        Losses->>Losses: L4: Gain-Performance<br/>clamp(incorrect_gains - correct_gains + 0.1, min=0)
    and
        Losses->>Losses: L5: Sparsity<br/>abs(non_relevant_gains).mean()
    and
        Losses->>Losses: L6: Consistency<br/>abs(mastery_delta - scaled_gains).mean()
    end
    
    Losses->>Losses: Total interpretability_loss =<br/>w1*L1 + w2*L2 + w3*L3 + w4*L4 + w5*L5 + w6*L6
    Losses->>Output: interpretability_loss (scalar)
    
    alt Monitoring enabled AND batch_idx % monitor_frequency == 0
        Output->>Monitor: Pass all states:<br/>context_seq, value_seq,<br/>mastery, gains, predictions,<br/>questions, responses
        Monitor->>Monitor: Compute correlations<br/>Log statistics
        Monitor-->>Output: Monitoring complete
    end
    
    Output->>Output: Assemble final dict:<br/>{predictions, logits,<br/>context_seq, value_seq,<br/>projected_mastery, projected_gains,<br/>interpretability_loss}
```

**Key Flow Insights:**

### Context Stream (h) - 3 Destinations:
1. **→ Prediction Head**: Concatenated with value and skill embeddings for response prediction
2. **→ Mastery Head**: Provides initial mastery estimate (refined by recursive accumulation)
3. **→ Output/Monitor**: Returned for monitoring and analysis

### Value Stream (v) - 3 Destinations (GainAKT3Exp Core Innovation):
**Values ARE learning gains** - each interaction's Value output directly represents the learning gain for that (skill, response) tuple.

1. **→ Prediction Head**: Concatenated with context and skill embeddings for response prediction
2. **→ Recursive Mastery Accumulation**: **Direct flow as learning gains**
   - Each Value output represents: "How much did the student learn from this interaction?"
   - For the skill associated with each question: `mastery[skill, t] = mastery[skill, t-1] + α × ReLU(value[t])`
   - Scaling factor α = 0.1 ensures bounded increments
   - ReLU ensures non-negative gains (no knowledge loss)
   - Mastery clamped to [0, 1] range (normalized competence)
3. **→ Output/Monitor**: Returned for monitoring and analysis

**Educational Semantics**: The transformer learns to output Values that encode meaningful learning gains per interaction. When a student interacts with a question (targeting specific skill) and provides a response (correct/incorrect), the Value output quantifies the learning gain from that experience.

### Loss Computation Sources:
Interpretability losses receive inputs from multiple stages:
- **predictions**: From Prediction Head (sigmoid outputs)
- **projected_mastery**: From Recursive Accumulation (mastery trajectory across timesteps)
- **learning_gains**: Directly from Value stream (per-interaction learning)
- **responses (r)**: Ground truth from input (for performance alignment)
- **questions (q)**: Input questions (for Q-matrix skill masks)

**Recursive Accumulation**: The key architectural principle where Values directly flow into mastery computation:
```
For each interaction t with skill s:
  learning_gain[t] = ReLU(value[t])  # Non-negative learning
  mastery[s, t] = clamp(mastery[s, t-1] + α × learning_gain[t], min=0, max=1)
```

This direct mapping from Values → Learning Gains → Mastery Accumulation enforces interpretability-by-design and provides educational transparency.


## Implementation Summary

The GainAKT3Exp model (`pykt/models/gainakt3_exp.py`) is an enhanced version of the GainAKT3 base model that adds training-time interpretability monitoring and auxiliary loss computation. The implementation follows PyKT framework standards:

**Training Pipeline** (`examples/train_gainakt3exp.py`):
- Zero hardcoded defaults—all 60+ parameters must be explicit
- Launched via `run_repro_experiment.py` which loads defaults from `configs/parameter_default.json`
- Saves complete experiment artifacts: checkpoints, config, trajectories, metrics
- Supports semantic trajectory tracking (mastery/gain correlations per epoch)

**Evaluation Pipeline** (`examples/eval_gainakt3exp.py`):
- Loads trained model and computes test metrics (AUC, accuracy)
- Computes mastery/gain correlations (configurable student sample size)
- Saves evaluation results with timestamp

**Learning Trajectory Analysis** (`examples/learning_trajectories.py`):
- Standalone script to extract and display individual student learning trajectories
- Shows timestep-by-timestep: skills practiced, gains, mastery, predictions vs truth
- Compact tabular format with student summary statistics
- Command automatically added to `config.json` for easy access

**Model Creation**: Models are instantiated via `create_exp_model(config)` which requires all parameters in the config dict (no defaults), ensuring reproducibility.

### Training and Evaluation Workflow

Following PyKT framework standards (see `assistant/quickstart.pdf` and `assistant/contribute.pdf`):

**1. Launch Training:**
```bash
python examples/run_repro_experiment.py \
  --model gainakt3exp \
  --dataset assist2015 \
  --short_title baseline_test \
  --epochs 12 \
  [--param_override value ...]
```

This launcher:
- Loads defaults from `configs/parameter_default.json`
- Applies CLI overrides for specified parameters
- Creates timestamped experiment directory: `saved_model/{timestamp}_{model}_{title}_{uid}/`
- Saves complete config (including trajectory command) to `{run_dir}/config.json`
- Saves checkpoints, metrics, and trajectory data

**2. Evaluate Model:**
```bash
python examples/eval_gainakt3exp.py \
  --run_dir saved_model/{experiment_dir} \
  --ckpt_name {checkpoint}.pt
```

Outputs test AUC, accuracy, and mastery/gain correlations.

**3. Analyze Learning Trajectories:**
```bash
python examples/learning_trajectories.py \
  --run_dir saved_model/{experiment_dir} \
  --num_students 10 \
  --min_steps 10
```

Displays individual student learning progressions with mastery/gain states and prediction accuracy.

**Note:** The trajectory command is automatically included in `config.json` during training, enabling easy post-hoc analysis without manual parameter reconstruction.

### Reproducibility System

Following the "zero defaults" pattern documented in `examples/reproducibility.md`:

**Experiment Structure**: Each training run creates a timestamped directory:
```
saved_model/{timestamp}_{model}_{title}_{uid}/
  ├── config.json           # Complete parameter set + trajectory command
  ├── model_*.pt            # Checkpoints
  ├── metrics.json          # Training metrics (AUC, accuracy, correlations)
  ├── trajectory_*.json     # Semantic trajectory data (optional)
  └── eval_results_*.json   # Evaluation results
```

**Config File Contents**:
- `defaults`: All parameter default values (from `parameter_default.json`)
- `train_explicit`: Full training command with all parameters
- `eval_explicit`: Full evaluation command with all parameters  
- `trajectory_command`: Trajectory analysis command (10 students, min 10 steps)
- `metadata`: Timestamp, git hash, hostname, GPU info

**Reproducibility Guarantees**:
1. No hardcoded defaults in model or training code
2. All parameters explicitly passed via CLI
3. Complete config saved with every experiment
4. Commands reconstructable from config alone
5. Git hash and environment info captured

See `examples/reproducibility.md` for complete parameter evolution protocol.

Below is a comprehensive analysis of each architectural component's implementation status.

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


### Feature 4: Recursive Mastery Accumulation from Value Stream

**Expected (from diagram):** Values from encoder directly represent learning gains per interaction. These gains drive recursive accumulation of skill mastery across timesteps.

**Implementation Status:**

**4a. Mastery Head** (`gainakt3.py` lines 216-217):
```python
if self.use_mastery_head:
    self.mastery_head = nn.Linear(self.d_model, self.num_c)
```
Provides initial mastery estimate from context, used as starting point.

**4b. GainAKT3Exp Core Innovation: Values ARE Learning Gains** (`gainakt3_exp.py` lines 189-205):
```python
# GainAKT3Exp: Values directly represent learning gains
# Each interaction's Value output = how much the student learned from (skill, response) tuple
learning_gains = torch.relu(value_seq)  # [B, L, D] - Non-negative learning only

# For multi-skill scenarios, values can be mapped to per-skill gains via question mapping
# (Implementation detail: gains applied to skill associated with each question)
```

**Key Conceptual Shift**: 
- **Old interpretation**: "Values are processed via ReLU then projected to gains via gain_head"
- **Clearer interpretation**: "Values ARE the learning gains—each interaction's Value output directly quantifies the learning contribution"
- **Educational meaning**: When a student interacts with a question (skill, response), the transformer learns to output a Value that represents: "How much did they learn from this experience?"

**4c. Recursive Mastery Accumulation** (`gainakt3_exp.py` lines 208-217):
```python
# Initialize skill mastery tracking
projected_mastery = torch.zeros(batch_size, seq_len, self.num_c, device=q.device)

# For each timestep, increment the mastery of the relevant skill by the learning gain
for t in range(seq_len):
    skill_idx = question[t]  # Which skill this interaction targets
    learning_gain_t = ReLU(value_seq[t]) * α  # α = 0.1 (scaling factor)
    
    if t == 0:
        mastery[skill_idx, t] = clamp(learning_gain_t, 0, 1)
    else:
        mastery[skill_idx, t] = clamp(mastery[skill_idx, t-1] + learning_gain_t, 0, 1)
```

**Architecture Alignment:**
- **Value Stream Output**: `[B, L, D]` represents learning gains per interaction
- **Mastery Head**: Provides initial context-based mastery estimate (refined by recursion)
- **Recursive Update**: For each interaction with skill s: `mastery[s, t] = mastery[s, t-1] + α × ReLU(value[t])`
- **Output Shapes**: Mastery produces `[B, L, num_c]` tensors tracking per-skill mastery evolution
- **Educational Semantics**: 
  - **Learning Gains**: Value output directly represents knowledge increment from this interaction
  - **Mastery Trajectory**: Accumulated learning across all interactions with each skill
  - **Scaling Factor**: α=0.1 bounds individual increments (max +0.1 per interaction)
  - **Clamping**: Ensures mastery ∈ [0, 1] (normalized competence scale)
  - **Non-Negativity**: ReLU ensures gains ≥ 0 (no knowledge loss)

**Interpretability Guarantee**: The direct mapping from Values → Learning Gains → Mastery Accumulation means the model's internal representations have clear educational meaning. We can inspect any interaction's Value output and understand: "This is how much the student learned." We can trace mastery evolution and understand: "This is the accumulated learning for each skill."

**Verification:** The architecture enforces that Values ARE learning gains by design—no intermediate projection layer obscures this relationship. The recursive accumulation directly uses these gains to build interpretable mastery trajectories.


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


### Feature 6: Monitoring

**Expected (from diagram):** Real-time interpretability analysis during training via a monitoring hook that periodically captures internal model states (context, value, mastery, gains, predictions) for analysis, with configurable frequency to balance overhead and insight granularity.

**Implementation Status:**

**6a. Monitor Hook Infrastructure:**
- **Location:** `gainakt3_exp.py` lines 40-41, 54-56
- **Hook Registration:**
  ```python
  self.interpretability_monitor = None
  
  def set_monitor(self, monitor): 
      """Set the interpretability monitor hook."""
      self.interpretability_monitor = monitor
  ```
- **Usage Pattern:** Training scripts instantiate a monitor object and inject it via `model.set_monitor(monitor_instance)`, enabling modular monitoring strategies without model code changes.

**6b. Periodic State Capture:**
- **Location:** `gainakt3_exp.py` lines 164-178 (within `forward_with_states()`)
- **Execution Logic:**
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
- **State Exposure:** Captures all interpretability-critical tensors at training time
- **No-Gradient Context:** Monitoring wrapped in `torch.no_grad()` to prevent gradient computation overhead

**6c. Configurable Frequency:**
- **Parameter:** `monitor_frequency` (default: 50 batches)
- **Location:** Constructor parameter (`gainakt3_exp.py` line 35)
- **Purpose:** Controls monitoring granularity—higher values reduce overhead but provide coarser temporal resolution
- **CLI Integration:** `--monitor_freq` parameter in training scripts

**6d. Multi-GPU Safety:**
- **Primary Device Guard:** `primary_device = (not hasattr(self, 'device_ids') or q.device == torch.device(f'cuda:{self.device_ids[0]}'))`
- **Rationale:** Under `DataParallel`, multiple model replicas process different batches; guard ensures monitoring executes only once per global batch (on primary GPU)
- **Location:** `gainakt3_exp.py` lines 160-163

**6e. State Dictionary Returned:**
- **Location:** `forward_with_states()` return statement (line 182-189)
- **Contents:**
  ```python
  return {
      'predictions': predictions,
      'logits': logits,
      'context_seq': context_seq,
      'value_seq': value_seq,
      'projected_mastery': projected_mastery,
      'projected_gains': projected_gains,
      'interpretability_loss': interpretability_loss
  }
  ```
- **Purpose:** Enables both real-time monitoring (via hook) and post-hoc analysis (via returned states)

**Architecture Alignment:** 
- Complete monitoring infrastructure matching diagram's "Monitor Hub" and "Interpretability Monitor" nodes
- Configurable frequency control as specified in diagram annotation
- All internal states exposed for comprehensive interpretability analysis
- Multi-GPU safe implementation for production training environments
- Zero-gradient overhead via `torch.no_grad()` wrapper

**Verification:** The architecture diagram shows "Monitor Hub" receiving inputs from Mastery Hub, Gain Hub, and Predictions Hub, then routing to "Interpretability Monitor"—implementation provides exactly this via the `forward_with_states()` method capturing all relevant tensors and passing them to the registered monitor hook.

---

### Feature 7: Intrinsic Gain Attention Mode ❌ DEACTIVATED

**Objective:** Provide an alternative architectural mode that achieves parameter efficiency by deriving gains directly from attention mechanisms, eliminating the need for post-hoc projection heads. This explores the trade-off between model compactness and interpretability while maintaining competitive predictive performance.

**Expected (from architectural exploration):** A feature flag (`--intrinsic_gain_attention`) that conditionally disables projection heads and computes mastery/gains from cumulative attention weights, reducing parameter count while preserving the ability to track learning trajectories.

**Current Status:**: Given the results detailed before we will **deactivate the Intrinsic Gain Attention Mode by default**. So, we set "intrinsic_gain_attention": false in configs/parameter_default.json

**Implementation Status:**

**7a. Architectural Constraint Enforcement:**
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

**7b. Attention-Derived Gain Computation:**
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

**7c. Cumulative Mastery from Attention:**
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

**7d. Parameter Reduction:**
- **Baseline Mode:** 14,658,761 params
- **Intrinsic Mode:** 12,738,265 params
- **Reduction: 1,920,496 params (13.1%)**
  
  *Note: Reduction comes from disabled projection heads (mastery_head + gain_head) plus associated architectural optimizations.*

**7e. CLI Integration:**
- **Location:** `examples/run_repro_experiment.py` line 89
- **Usage:** `python examples/run_repro_experiment.py --intrinsic_gain_attention --epochs 12`
- **Default:** `False` (baseline mode with projection heads)
- **Parameter File:** Added to `configs/parameter_default.json` as `"intrinsic_gain_attention": false`

**Architecture Alignment:** Complete implementation with validated trade-offs

**Verification:** The updated architecture diagram (red components) shows intrinsic mode as conditional bypass of projection heads, with attention-derived gains feeding directly to mastery/gain outputs.


### Feature 8: Recursive Mastery Accumulation

Values are considere4d as Direct Gains
```
Encoder Values (v) → Direct Gains → Recursive Mastery
               └──────────────────→ Gain Hub (for losses)
```

The **blue subgraph** in the diagram above illustrates a critical architectural constraint that enforces interpretability-by-design. Unlike black-box models where knowledge states are opaque, our architecture implements a **deterministic recursive accumulation** mechanism:

$$\text{mastery}_{t+1}^{(c)} = \text{mastery}_t^{(c)} + \alpha \cdot \text{ReLU}(\text{gain}_t^{(c)})$$

This is implemented in the model's forward pass (`gainakt3_exp.py` lines 145, 162):

```python
accumulated_mastery = projected_mastery[:, t-1, :] + projected_gains[:, t, :] * 0.1
projected_mastery[:, t, :] = torch.clamp(accumulated_mastery, min=0.0, max=1.0)
```

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
| **Recursive Mastery Accumulation** | Deterministic temporal constraint: mastery_{t+1} = mastery_t + α·ReLU(gain_t) | Recursive loop with scaling (α=0.1) and clamping [0,1], enforces consistency between mastery and gains | ✅ Activated         |



## Parameters

The complete list of parameters including category and description is in `paper/parameters.csv`.

### Model Instantiation

Models are created via `create_exp_model(config)` (`gainakt3_exp.py` line 435), which requires all parameters explicitly in the config dictionary:

**Required Parameters** (22 total):
- **Architecture**: `num_c`, `seq_len`, `d_model`, `n_heads`, `num_encoder_blocks`, `d_ff`, `dropout`, `emb_type`
- **Interpretability Features**: `use_mastery_head`, `use_gain_head`, `intrinsic_gain_attention`, `use_skill_difficulty`, `use_student_speed`
- **Training Context**: `num_students` (for student_speed embedding when enabled)
- **Loss Weights** (6): `non_negative_loss_weight`, `monotonicity_loss_weight`, `mastery_performance_loss_weight`, `gain_performance_loss_weight`, `sparsity_loss_weight`, `consistency_loss_weight`
- **Monitoring**: `monitor_frequency` (batches between monitoring calls)

**Zero Defaults Policy**: The factory function raises errors if required parameters are missing, ensuring no hidden defaults. All defaults are defined in `configs/parameter_default.json`, which is loaded by `run_repro_experiment.py` and can be overridden via CLI. 

## Evolving the Model

We follow a rigorous parameter evolution protocol to maintain reproducibility across model variants and hyperparameter sweeps. See `examples/reproducibility.md` for complete details.

### Parameter Evolution Protocol

All parameter defaults live in `configs/parameter_default.json`. When adding or modifying parameters:

1. **Update `parameter_default.json`** with new parameter and default value
2. **Update `examples/run_repro_experiment.py`** to add CLI argument for new parameter
3. **Update `paper/parameters.csv`** with parameter documentation (category, description, default value)
4. **Verify no hardcoded defaults** exist in model/training code that could diverge from `parameter_default.json`

### Consistency Verification

After any codebase modification (model, training/evaluation scripts, etc.):
- Check that no hidden parameters with hardcoded defaults exist
- Verify all model parameters are passed explicitly via `create_exp_model(config)`
- Ensure training scripts require all parameters via CLI (no fallback defaults)
- Confirm `parameter_default.json` matches actual code behavior

### Hyperparameter Optimization

**Objective**: Systematically explore parameter combinations to find optimal configuration.

**Process**:
1. Use defaults from `configs/parameter_default.json` as starting point
2. Run experiments with parameter variations via CLI overrides
3. Document results in experiment-specific config files (auto-generated in run directories)
4. When optimal configuration is found, update `configs/parameter_default.json` with new defaults
5. Document change in `paper/parameters.csv` with rationale and date 

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

## Trajectories



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

**Semantic Constraints**
1. **Non-negative gains**: Learning gains are always positive (>=0)
2. **Monotonic mastery**: Mastery can only increase or stay constant over time. Mastery level is in [0,1] range (probabilistic interpretation)
3. **Consistency**: Mastery increments are consistent with learning gains
4. **Sparsity**: Practice with an item/question produces mastery increments only in the relevant skills (those skills related to the item according to the Q-Matrix)


**Non-Negative Gains** (`non_negative_loss_weight = 0.0`): Penalizes negative learning gains by computing `clamp(-projected_gains, min=0).mean()`. Currently disabled (weight 0.0) as gains are naturally non-negative due to model architecture, but available for architectural variants.

**Monotonicity** (`monotonicity_loss_weight = 0.1`): Enforces non-decreasing mastery over time by penalizing `clamp(mastery[t] - mastery[t+1], min=0).mean()`. Ensures mastery cannot regress, reflecting the assumption that learning is cumulative and students do not "unlearn" previously mastered skills.

**Mastery-Performance Alignment** (`mastery_performance_loss_weight = 0.8`): Penalizes interaction-level mismatches between mastery and performance. Specifically: (1) penalizes low mastery (`clamp(1 - mastery, min=0)`) when students answer correctly, and (2) penalizes high mastery (`clamp(mastery, min=0)`) when students answer incorrectly. This hinge-style constraint prevents obvious violations (e.g., mastery=0.1 on correct response, mastery=0.9 on incorrect response) and complements the trajectory-level Alignment Loss by enforcing point-wise consistency.

**Gain-Performance Alignment** (`gain_performance_loss_weight = 0.8`): Enforces that correct responses should yield higher gains than incorrect responses via hinge loss: `clamp(mean(incorrect_gains) - mean(correct_gains) + 0.1, min=0)`. The 0.1 margin ensures a clear separation, reflecting the educational assumption that successful problem-solving produces greater learning increments.

**Sparsity** (`sparsity_loss_weight = 0.2`): Penalizes non-zero gains for skills not directly involved in the current interaction via `abs(non_relevant_gains).mean()`. Encourages skill-specific learning (gains concentrated on the question's target skill) rather than diffuse updates across all skills, improving interpretability and alignment with skill-specific educational theories. **Skill Mask Computation:** Uses Q-matrix structure via `skill_masks.scatter_(2, questions.unsqueeze(-1), 1)` to identify relevant skills—correctly implements sparsity constraint based on problem-skill mappings.

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
