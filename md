direction: up

** {
  style: {
    font-size: 85
  }
}

title: "Model Architecture" {
  near: top-center
  shape: text
  style: {
    font-size: 100
  }
}

# ============================================================================
# 1: INPUT COMPONENTS
# ============================================================================

Section1: {
  label: "1: Input Components"
  style: {
    stroke: "#000000"
    stroke-width: 3
    fill: "#fafafa"
    border-radius: 8
    font-size: 100
  }
  
  BKT: {
    label: |md
      BKT

      Reference Model
    |
    shape: cylinder
    style: {
      fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
    }
  }


  InputData: {
    label: |md
      Interaction 
      
      Data
    |
    shape: cylinder
    style: {
      fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
    }
  }

}

# ============================================================================
# 2: EMBEDDINGS LAYER
# ============================================================================

Section2: {
  label: "2: Embeddings Layer"
  style: {
    stroke: "#000000"
    stroke-width: 3
    fill: "#fafafa"
    border-radius: 8
    font-size: 100
  }
  
  TaskHistoryEmb: {
    label: |md
      Task & History
      
      x'ₜ (Current Task)

      y'ₜ (History)
    |
    shape: rectangle
    style: {
      fill: "#e8f4f8"
      stroke: "#000000"
      stroke-width: 2
      3d: true
    }
  }
  
}

# ============================================================================
# 3: TRANSFORMER ARCHITECTURE
# ============================================================================

Section3: {
  label: "3: Transformer Architecture"
  style: {
    stroke: "#000000"
    stroke-width: 3
    fill: "#fafafa"
    border-radius: 8
    font-size: 100
  }
  
  Encoder: {
    label: |md
      Encoder

      (Self-Attention)

      y'_{1:t-1}
    |
    shape: rectangle
    style: {
      fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
      multiple: true
    }
  }
  
  Decoder: {
    label: |md
      Decoder

      (Cross-Attention)

      x'ₜ
    |
    shape: rectangle
    style: {
      fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
      multiple: true
    }
  }
  
  DecoderOutput: {
    label: "Decoder Output\nd_output\n\nR^d_model"
    shape: rectangle
    style: {
      fill: "#e8f4f8"
      stroke: "#000000"
      stroke-width: 2
      bold: false
    }
  }
  
  QuestionEmbed: {
    label: "Question Embedding\nq_embed\n\nR^d_model"
    shape: rectangle
    style: {
      fill: "#e8f4f8"
      stroke: "#000000"
      stroke-width: 2
      bold: false
    }
  }
  
  ContextVector: {
    label: "Context Vector\nz_t = [d_output concat q_embed]\n\nR^(2*d_model)"
    shape: rectangle
    style: {
      fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
      3d: true
      bold: false
    }
  }
  
  Encoder -> Decoder: "History Context" { style: { stroke: "#000000"; font-size: 80 } }
  Decoder -> DecoderOutput { style: { stroke: "#000000" } }
  DecoderOutput -> ContextVector: "concat" { style: { stroke: "#000000"; font-size: 80 } }
  QuestionEmbed -> ContextVector: "concat" { style: { stroke: "#000000"; font-size: 80 } }
}

# ============================================================================
# 4: GROUNDED PARAMETERS (THREE-COMPONENT COMPOSITION)
# ============================================================================

Section4: {
  label: "4: Grounded Parameters (Three-Term Decomposition v2.0)"
  style: {
    stroke: "#000000"
    stroke-width: 3
    fill: "#fafafa"
    border-radius: 8
    font-size: 100
  }
  
  PopulationLevel: {
    label: "Population-Level \n (from BKT, fixed) \n mu_L0[q], mu_T[q] \n \nR^1 per skill"
    shape: rectangle
    style: {
        fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
      bold: false
    }
  }
  
  StudentTraits: {
    label: "Per-Student Traits \n (cluster embeddings) \n delta_L0[uid], delta_T[uid] \n \n R^2 per student"
    shape: rectangle
    style: {
        fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
      bold: false
    }
  }
  
  SkillResiduals: {
    label: "Per-Skill Residuals \n (from z context) \n epsilon_L0 = proj(z) \n epsilon_T = proj(z) \n \n R^1 per interaction"
    shape: rectangle
    style: {
        fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 2
      bold: false
    }
  }
  
  tSNEReference: {
    label: "t-SNE Reference \n (from BKT) \n Z_ref[uid] in R^2 \n \nCluster targets"
    shape: cylinder
    style: {
        fill: "#fce4ec"
      stroke: "#000000"
      stroke-width: 2
      bold: false
    }
  }
  
  FinalParams: {
    label: "Final Parameters \n p_L0 = sigma(mu + delta + epsilon)\\np_T = sigma(mu + delta + epsilon)"
    shape: rectangle
    style: {
      fill: "#d4e6f1"
      stroke: "#000000"
      stroke-width: 3
      bold: false
    }
  }
  
  PopulationLevel -> FinalParams: "+" { style: { stroke: "#000000"; font-size: 80 } }
  StudentTraits -> FinalParams: "+" { style: { stroke: "#000000"; font-size: 80 } }
  SkillResiduals -> FinalParams: "+" { style: { stroke: "#000000"; font-size: 80 } }
  
  tSNEReference -> StudentTraits: "L_tsne" { style: { stroke: "#e91e63"; stroke-dash: 5; font-size: 80 } }
}

# ============================================================================
# 5: OUTPUTS & MULTI-OBJECTIVE LOSS
# ============================================================================

Section5: {
  label: "5: Outputs & Multi-Objective Loss"
  style: {
    stroke: "#000000"
    stroke-width: 3
    fill: "#fafafa"
    border-radius: 8
    font-size: 100
  }
  
  Outputs: {
    label: "Output Heads"
    style: {
      stroke: "#000000"
      stroke-width: 2
      stroke-dash: 5
      fill: "#ffffff"
      font-size: 90
    }
    
    SupervisedOut: {
      label: |md
        Supervised Prediction
        
        ŷ = MLP(zₜ)
      |
      shape: rectangle
      style: {
          fill: "#d4e6f1"
        stroke: "#000000"
        stroke-width: 2
      }
    }
    
    ReferenceOut: {
      label: |md
        Reference Output

        (BKT Logic)
        
        ŷ_ref = BKT(p_L0, p_T)
      |
      shape: rectangle
      style: {
          fill: "#d4e6f1"
        stroke: "#000000"
        stroke-width: 2
      }
    }
    
    DiagnosticProbes: {
      label: |md
        Probe Grounding
        
        Probe_L0(zₜ)
        Probe_T(zₜ)
      |
      shape: rectangle
      style: {
          fill: "#d4e6f1"
        stroke: "#000000"
        stroke-width: 2
      }
    }
  }
  
  Loss: {
    label: "Multi-Objective Loss"
    style: {
      stroke: "#000000"
      stroke-width: 2
      stroke-dash: 5
      fill: "#ffffff"
      font-size: 90
    }
    
    LossSup: {
      label: |md
        ℒ_sup L
      |
      shape: rectangle
      style: {
        fill: "#d4e6f1"
        stroke: "#000000"
        stroke-width: 2
      }
    }
    
    LossRef: {
      label: |md
        ℒ_ref L 
      |
      shape: rectangle
      style: {
        fill: "#d4e6f1"
        stroke: "#000000"
        stroke-width: 2
      }
    }
    
    LossProbe: {
      label: "L_tsne\\nCluster coherence"
      shape: rectangle  
      style: {
        fill: "#fce4ec"
        stroke: "#e91e63"
        stroke-width: 2
        bold: false
      }
    }
    
    TotalLoss: {
      label: "Total Loss\\nL = lambda_sup * L_sup +\\nlambda_ref * L_ref +\\nlambda_tsne * L_tsne"
      shape: rectangle
      style: {
        fill: "#d4e6f1"
        stroke: "#000000"
        stroke-width: 3
        bold: false
      }
    }
    
    LossSup -> TotalLoss: "lambda_sup" { style: { stroke: "#000000"; font-size: 80 } }
    LossRef -> TotalLoss: "lambda_ref" { style: { stroke: "#000000"; font-size: 80 } }
    LossProbe -> TotalLoss: "lambda_tsne" { style: { stroke: "#e91e63"; font-size: 80 } }
  }
  
  Outputs.SupervisedOut -> Loss.LossSup { style: { stroke: "#000000" } }
  Outputs.ReferenceOut -> Loss.LossRef { style: { stroke: "#000000" } }
  Outputs.DiagnosticProbes -> Loss.LossProbe { style: { stroke: "#e91e63"; stroke-dash: 5 } }
}



# ============================================================================
# DATA FLOW CONNECTIONS BETWEEN SECTIONS
# ============================================================================

# 1 -> 2
Section1.BKT -> Section1.InputData: "p_L0, p_T" { style: { stroke: "#000000"; font-size: 80 } }
Section1.InputData -> Section2.TaskHistoryEmb: "{(q, c, r, p_L0, p_T)}" { style: { stroke: "#000000"; font-size: 80 } }

# 2 -> 3
Section2.TaskHistoryEmb -> Section3.Encoder: "y'_{1:t-1}" { style: { stroke: "#000000"; font-size: 80 } }
Section2.TaskHistoryEmb -> Section3.Decoder: "x'ₜ" { style: { stroke: "#000000"; font-size: 80 } }
Section2.TaskHistoryEmb -> Section3.QuestionEmbed: "q_embed" { style: { stroke: "#000000"; font-size: 80 } }

# 3 -> 4
Section1.BKT -> Section4.PopulationLevel: "mu_L0, mu_T" { style: { stroke: "#000000"; font-size: 80 } }
Section1.BKT -> Section4.tSNEReference: "Z_ref" { style: { stroke: "#e91e63"; stroke-dash: 5; font-size: 80 } }
Section3.ContextVector -> Section4.SkillResiduals: "proj(z)" { style: { stroke: "#000000"; font-size: 80 } }

# 4 -> 5
Section4.FinalParams -> Section5.Outputs.ReferenceOut: "p_L0, p_T" { style: { stroke: "#000000"; font-size: 80 } }
Section3.ContextVector -> Section5.Outputs.SupervisedOut: "Direct MLP" { style: { stroke: "#000000"; font-size: 80 } }
Section4.StudentTraits -> Section5.Outputs.DiagnosticProbes: "delta traits" { style: { stroke: "#000000"; font-size: 80 } }

# Ground Truth -> Loss
Section1.InputData -> Section5.Loss.LossSup: "y_true" {
  style: {
    stroke-dash: 5
    opacity: 0.5
    font-size: 80
    stroke: "#000000"
  }
}
Section1.BKT -> Section5.Loss.LossProbe: "p_L0, p_T" {
  style: {
    stroke-dash: 5
    opacity: 0.5
    font-size: 80
    stroke: "#000000"
  }
}


