# iKT3 Implementation

## Overview

iKT3 is a simplified interpretable knowledge tracing model with alignment-based interpretability. It improves upon iKT2 by:

- **Single Output Head**: Ability encoder only (no separate mastery head)
- **Fixed β_IRT**: Pre-computed skill difficulties (eliminates scale drift)
- **BCE Losses**: Both L_per and L_ali use Binary Cross-Entropy
- **Smart Scale Initialization**: Automatic θ/β ratio control based on IRT statistics
- **Cleaner Architecture**: Simpler gradient flow, fewer hyperparameters

## Architecture

Based on the specification in `paper/ikt3_architecture_approach.md`:

```
Input (q,r) → Embeddings → Transformer Encoder → Ability Encoder (θ) → IRT Formula (σ(θ-β)) → Predictions
                                                                     ↑
                                                          Fixed β_IRT (pre-computed)
```

## Files

- **Model**: `pykt/models/ikt3.py`
- **Training**: `examples/train_ikt3.py`
- **Evaluation**: `examples/eval_ikt3.py`
- **Config**: `configs/parameter_default.json` (model="ikt3")

## Two-Phase Training

### Phase 1: Performance Learning (epochs 1-10)
- **Objective**: Learn good representations
- **Loss**: L_total = L_per (BCE)
- **Focus**: Maximize prediction accuracy

### Phase 2: Alignment (epochs 11-30)
- **Objective**: Balance performance and interpretability
- **Loss**: L_total = (1-λ_int)×L_per + λ_int×L_ali (both BCE)
- **Focus**: Align with reference IRT predictions

## Key Parameters

### Model Architecture
- `d_model`: 256 (hidden dimension)
- `n_heads`: 4 (attention heads)
- `num_encoder_blocks`: 8 (transformer layers)
- `d_ff`: 1536 (feedforward dimension)
- `dropout`: 0.2
- `seq_len`: 200 (max sequence length)

### iKT3-Specific
- `target_ratio`: 0.4 (θ/β scale ratio, valid range 0.3-0.5)
- `phase1_epochs`: 10 (performance learning phase)
- `lambda_int`: 0.5 (alignment weight for Phase 2)
- `rasch_path`: Path to pre-computed IRT difficulties
- `reference_path`: Path to reference predictions (optional)

### Training
- `learning_rate`: 0.0001
- `batch_size`: 64
- `epochs`: 30
- `optimizer`: Adam
- `gradient_clip`: 1.0
- `patience`: 4 (early stopping)

## Usage

### Training with Reproducible Experiment Launcher

```bash
# Default training (uses parameter_default.json)
python examples/run_repro_experiment.py --short_title "ikt3_baseline"

# Override specific parameters
python examples/run_repro_experiment.py \
    --short_title "ikt3_higher_lambda" \
    --lambda_int 0.7 \
    --phase1_epochs 15

# Different dataset
python examples/run_repro_experiment.py \
    --short_title "ikt3_assist2009" \
    --dataset assist2009 \
    --rasch_path data/assist2009/rasch_targets.pkl
```

### Direct Training (Not Recommended - No Reproducibility Guarantees)

```bash
python examples/train_ikt3.py \
    --dataset assist2015 \
    --rasch_path data/assist2015/rasch_targets.pkl \
    --save_dir saved_model/ikt3/exp001 \
    --epochs 30 \
    --phase1_epochs 10 \
    --lambda_int 0.5 \
    --target_ratio 0.4
```

### Evaluation

```bash
python examples/eval_ikt3.py \
    --model_path saved_model/ikt3/exp001/best_model.pt \
    --dataset assist2015 \
    --split test
```

## Scale Health Monitoring

iKT3 automatically monitors the θ/β scale ratio during training:

- **Target Range**: 0.3 - 0.5 (empirically validated)
- **Monitoring Frequency**: Every 5 epochs (configurable via `monitor_freq`)
- **Health Check**: Prints θ/β statistics and validates ratio

Example output:
```
[Epoch 20] Scale Health Check:
  θ: mean=-0.052, std=0.335
  β: mean=-2.004, std=0.848
  θ/β ratio: 0.395 (target: 0.3-0.5)
  Learned scale parameter: 0.327
  ✅ Scale ratio healthy
```

## Interpretability Metrics

After evaluation, iKT3 reports:

- **Performance**: AUC, Accuracy
- **Ability Statistics**: θ mean, θ std
- **Difficulty Statistics**: β mean, β std
- **Scale Health**: θ/β ratio (should be 0.3-0.5)
- **Mastery Statistics**: m_pred mean, m_pred std

## Key Differences from iKT2

| Aspect | iKT2 | iKT3 |
|--------|------|------|
| **Output Heads** | 2 (Performance + Mastery) | 1 (Ability Encoder) |
| **Skill Difficulties** | Learnable β_k with L_reg | Fixed β_IRT (pre-computed) |
| **Losses** | L_per + L_align + L_reg | L_per + L_ali (no L_reg) |
| **Loss Functions** | Mixed (BCE + MSE) | BCE for both |
| **Scale Control** | Manual monitoring | Automatic smart initialization |
| **Hyperparameters** | λ_align, λ_reg, epsilon | λ_int, target_ratio (simpler) |
| **Gradient Flow** | Complex (dual heads) | Simplified (single head) |

## Pre-computed IRT Requirements

iKT3 requires pre-computed IRT skill difficulties. Generate them using:

```bash
python examples/compute_rasch_targets.py --dataset assist2015
```

This creates `data/assist2015/rasch_targets.pkl` with:
- `skill_difficulties`: β_IRT values for each skill
- `student_abilities`: Global θ per student (not used by iKT3)
- `metadata`: Dataset statistics

## Expected Performance

Based on architecture design:
- **Target AUC**: ~0.71 (competitive with baselines)
- **Phase 1 AUC**: 0.71-0.73 (pure performance)
- **Phase 2 AUC**: 0.70-0.72 (small drop for interpretability)
- **Scale Ratio**: 0.3-0.5 throughout training

## Troubleshooting

### Scale Ratio Outside Range

If θ/β ratio drifts outside [0.3, 0.5]:
1. Check `target_ratio` parameter (default 0.4)
2. Verify IRT difficulties loaded correctly
3. Try adjusting `phase1_epochs` (more warmup helps)
4. Consider adding soft regularization (see architecture doc)

### Poor Phase 2 Performance

If AUC drops significantly in Phase 2:
1. Reduce `lambda_int` (less alignment weight)
2. Increase `phase1_epochs` (better Phase 1 warmup)
3. Check reference predictions quality
4. Verify BCE losses are computed correctly

### Memory Issues

If running out of memory:
1. Reduce `batch_size` (default 64)
2. Reduce `seq_len` (default 200)
3. Reduce `num_encoder_blocks` (default 8)
4. Enable `use_amp` (mixed precision training)

## Citation

If you use iKT3 in your research, please cite:

```bibtex
@article{ikt3,
  title={iKT3: Interpretable Knowledge Tracing with Alignment-Based Interpretability},
  author={Your Name},
  year={2025}
}
```

## References

- Architecture Specification: `paper/ikt3_architecture_approach.md`
- Reproducibility Protocol: `examples/reproducibility.md`
- PyKT Framework: `assistant/quickstart.pdf`
- Rasch IRT Theory: `paper/rasch_model.md`
