# D2 Architecture Diagrams for GTransformer

This directory contains D2 diagram source files for visualizing the GTransformer architecture and its personalization mechanism.

## Files

### Main Architecture
- **`arch.d2`**: Complete end-to-end architecture diagram showing all components

### Sectional Diagrams
- **`section1_inputs.d2`**: Input components (BKT, Oracle, Data, Student IDs)
- **`section2_embeddings.d2`**: Embeddings layer (Task/History vs. Student Memory)
- **`section3_transformer.d2`**: Transformer encoder-decoder architecture
- **`section4_grounded_params.d2`**: Three-component parameter composition
- **`section5_outputs_loss.d2`**: Output heads and multi-objective loss

### Data Flow Diagrams
- **`data_flow_baseline.d2`**: Complete flow when `personalization=false`
- **`data_flow_hybrid.d2`**: Complete flow when `personalization=true`

## Rendering Diagrams

### Install D2
```bash
curl -fsSL https://d2lang.com/install.sh | sh -s --
```

### Render Individual Diagrams
```bash
# Main architecture
d2 arch.d2 arch.svg
d2 arch.d2 arch.png

# Sectional diagrams
d2 section1_inputs.d2 section1_inputs.svg
d2 section2_embeddings.d2 section2_embeddings.svg
d2 section3_transformer.d2 section3_transformer.svg
d2 section4_grounded_params.d2 section4_grounded_params.svg
d2 section5_outputs_loss.d2 section5_outputs_loss.svg

# Data flow diagrams
d2 data_flow_baseline.d2 data_flow_baseline.svg
d2 data_flow_hybrid.d2 data_flow_hybrid.svg
```

### Render All at Once
```bash
# SVG format
for f in *.d2; do d2 "$f" "${f%.d2}.svg"; done

# PNG format (for LaTeX)
for f in *.d2; do d2 "$f" "${f%.d2}.png"; done

# PDF format (high quality)
for f in *.d2; do d2 "$f" "${f%.d2}.pdf"; done
```

## Visual Design Conventions

### Color Scheme
- **Blue tones** (`#e8f4f8`, `#d4e6f1`, `#aed6f1`): Always-active components
- **Orange tones** (`#fff8e8`, `#ff8800`): Personalization-dependent components
- **Green tones** (`#d5f4e6`, `#abebc6`): Grounded parameters and outputs
- **Red tones** (`#f8d7da`, `#ffcccc`): Loss functions
- **Gray tones** (`#f0f0f0`, `#f9f9f9`): Legend and info boxes

### Border Styles
- **Solid borders** (`stroke-width: 2`): Always active
- **Dashed borders** (`stroke-dash: 3`): Conditional (personalization=true)
- **Thick borders** (`stroke-width: 3`): Emphasis on key components

### Shape Semantics
- **Cylinders**: Data sources (inputs, ground truth)
- **Rectangles**: Processing components (embeddings, layers)
- **Hexagons**: Intermediate representations (context vector)
- **Circles**: Loss functions
- **3D effect**: Embedding layers

## Usage in Paper

### LaTeX Integration

#### Single Figure (Main Architecture)
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{d2/arch.png}
    \caption{GTransformer architecture showing the three-component additive 
    composition for parameter estimation. Dashed borders indicate components 
    activated only when \texttt{personalization=true}.}
    \label{fig:gtransformer_arch}
\end{figure}
```

#### Multi-Panel Figure (Sectional Views)
```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{d2/section2_embeddings.png}
        \caption{Embeddings Layer}
        \label{fig:arch_embeddings}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{d2/section4_grounded_params.png}
        \caption{Grounded Parameters}
        \label{fig:arch_params}
    \end{subfigure}
    \caption{Key architectural components of GTransformer.}
    \label{fig:arch_components}
\end{figure}
```

#### Side-by-Side Comparison (Baseline vs. Hybrid)
```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{d2/data_flow_baseline.png}
        \caption{Baseline Mode (personalization=false)}
        \label{fig:flow_baseline}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{d2/data_flow_hybrid.png}
        \caption{Hybrid Mode (personalization=true)}
        \label{fig:flow_hybrid}
    \end{subfigure}
    \caption{Data flow comparison showing how personalization adds student-specific 
    biases to the theory-guided and contextual components.}
    \label{fig:flow_comparison}
\end{figure}
```

## Diagram Descriptions

### Section 1: Inputs
Shows the four input sources:
- BKT Reference Model (always active)
- Oracle Labels (always active)
- Interaction Data (always active)
- Student IDs (conditional, dashed border)

### Section 2: Embeddings
Illustrates the split embeddings layer:
- Task & History Embeddings (always active)
- Student Memory Embeddings (conditional, dashed border)

### Section 3: Transformer
Depicts the encoder-decoder architecture:
- Encoder: Self-attention on interaction history
- Decoder: Cross-attention with current task
- Context Vector: Rich representation for downstream tasks

### Section 4: Grounded Parameters
**Most important diagram** - shows the three-component additive composition:
1. **Theory Base**: Population-level BKT priors (always active)
2. **Contextual Projection**: Transformer's temporal reasoning (always active)
3. **Student Memory**: Individual-level biases (conditional, dashed border)

All three components feed into "Final Parameters" with "+" connections, making the additive nature explicit.

### Section 5: Outputs & Loss
Shows the three output heads:
- Supervised Prediction (standard MLP)
- Reference Output (BKT logic wrapper)
- Diagnostic Probes (linear interpretability)

And the multi-objective loss function combining all three.

### Data Flow: Baseline
Complete end-to-end flow when `personalization=false`:
- Only solid-bordered components active
- Two-component composition: Theory + Context
- Cold-start capable, privacy-preserving

### Data Flow: Hybrid
Complete end-to-end flow when `personalization=true`:
- Orange-highlighted components show active personalization path
- Three-component composition: Theory + Context + Student
- Personalized diagnostics, captures individual traits

## Customization

### Changing Colors
Edit the `style.fill` property in each component:
```d2
ComponentName: {
  style: {
    fill: "#YOUR_COLOR_HEX"
  }
}
```

### Changing Layout Direction
Change the `direction` at the top of each file:
```d2
direction: down   # or: up, left, right
```

### Adding Annotations
Use text shapes for notes:
```d2
Note: {
  shape: text
  label: "Your annotation here"
  style: {
    font-size: 14
  }
}
```

## References

- **D2 Documentation**: https://d2lang.com/
- **Model Implementation**: `pykt/models/gtransformer.py`
- **Paper Documentation**: `paper/gtransformer.md`
- **Visual Summary**: `tmp/arch_diagram_visual_summary.md`
