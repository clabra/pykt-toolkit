# iDKT Architecture Diagram - LaTeX TikZ Implementation

## Overview

Created a detailed TikZ diagram for the iDKT architecture that replaces the PNG image reference. The diagram is designed for academic publication with appropriate balance between detail and clarity.

## Diagram Components

### 1. **Input Layer** (Blue boxes)
- Questions ($q$)
- Responses ($r$)
- Problem IDs
- Student IDs
- **BKT Reference Data** (gray dashed boxes): $L_0$, $T$, $p_{BKT}$

### 2. **Representational Grounding** (Cyan boxes)
Shows the two parallel paths:

**Task Path (x')**:
- Concept embedding ($c$)
- Grounded initial knowledge: $l_c = L_0 + k_c \cdot d_c$
- Final task embedding: $x' = (c + u \cdot d) - l_c$

**History Path (y')**:
- Interaction embedding ($e$)
- Grounded learning rate: $t_c = T + v_s \cdot d_s$
- Final history embedding: $y' = (e + u \cdot f) + t_c$

### 3. **Transformer Core** (Green boxes)
- **Encoder** (N blocks): Self-attention on $y'$ → produces $y^$ (encoded history)
- **Decoder** (2N blocks): Self-Attn ⊕ Cross-Attn → produces $x^$ (decoder output)
- Shows cross-attention connection from encoder to decoder

### 4. **Prediction Pipeline** (Purple boxes)
- Concatenation: $\text{concat}_q = [x^, x']$
- MLP layers: 512 → 256 → 1
- Final prediction: $p_{iDKT}$

### 5. **Monitoring Outputs** (Purple boxes, right side)
- Initial Mastery: $\sigma(\text{mean}(l_c))$
- Learning Rate: $\sigma(\text{mean}(t_c))$

### 6. **Loss Functions** (Orange boxes, right side)
- $L_{sup}$: BCE$(p, r)$ - Supervised loss
- $L_{ref}$: MSE$(p, p_{BKT})$ - Alignment loss
- $L_{init}$: MSE$(l_c, L_0)$ - Initial knowledge grounding
- $L_{rate}$: MSE$(t_c, T)$ - Learning rate grounding

### 7. **Diagnostic Probing Validation** (Yellow dashed boxes, left side)
- Extract $\text{concat}_q$ (qtest=True)
- Linear Probe → $P(L_t)_{BKT}$ (true task)
- Control Probe → Shuffled (control task)
- Selectivity: $\Delta R^2 > 0.5$

## Visual Design Choices

### Color Scheme
- **Blue**: Input data
- **Cyan**: Embedding/grounding layer
- **Green**: Transformer components
- **Purple**: Output/prediction
- **Orange**: Loss functions
- **Yellow**: Probing validation
- **Gray**: BKT reference data

### Arrow Types
- **Solid arrows**: Data flow
- **Dashed gray arrows**: Reference/grounding connections
- **Dashed black arrows**: Probing validation flow

### Layout
- **Vertical flow**: Main pipeline (top to bottom)
- **Left side**: Probing validation (auxiliary)
- **Right side**: Monitoring outputs and loss functions

## Technical Features

1. **Resizable**: Uses `\resizebox{\textwidth}{!}` to fit page width
2. **Figure***: Uses two-column spanning for MDPI template
3. **Placement**: `[t]` for top placement
4. **Labels**: Section labels for each major component
5. **Mathematical notation**: Proper LaTeX math formatting
6. **Compact**: Efficient use of space while maintaining readability

## Key Improvements Over PNG

1. **Vector graphics**: Scalable without quality loss
2. **Editable**: Easy to modify colors, layout, or content
3. **Consistent typography**: Matches paper's font
4. **Integrated**: No external file dependency
5. **Professional**: Publication-ready quality
6. **Complete**: Includes probing validation (not in original PNG)

## Caption

The comprehensive caption explains all six components:
1. Input Layer with BKT reference
2. Representational Grounding mechanism
3. Encoder-Decoder Transformer architecture
4. Prediction via concatenated representations
5. Multi-objective loss functions
6. Diagnostic probing validation

## Compilation Requirements

Requires TikZ package (already included in MDPI template):
```latex
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes}
```

The diagram should compile without issues in the existing MDPI template.
