# Encoder Consistency Regularization

## Implementation Summary (2025-11-20)

### Objective
Enable the Interpretability Encoder (IM - Encoder 2) to guide the learning of the BCE Encoder (Encoder 1) through prediction consistency regularization.

### Approach
**Consistency Regularization**: Force both encoders to produce consistent predictions, allowing the interpretable IM encoder to guide BCE learning toward more interpretable representations.

### Mathematical Formulation

```python
# Total loss composition
total_loss = λ₁ * L_BCE + λ₂ * L_IM + α * L_consistency

where:
  L_BCE = BCE(predictions_BCE, targets)           # Encoder 1 prediction loss
  L_IM = BCE(predictions_IM, targets)             # Encoder 2 prediction loss  
  L_consistency = MSE(predictions_BCE, predictions_IM.detach())  # Consistency loss

  λ₁ = bce_loss_weight (default: 0.9)             # BCE encoder weight
  λ₂ = 1 - λ₁ (default: 0.1)                      # IM encoder weight
  α = encoder_consistency_weight (default: 0.1)   # Consistency weight
```

### Key Design Decision

```python
# IM predictions are DETACHED from gradient flow
consistency_loss = MSE(bce_predictions, im_predictions.detach())
```

**Rationale**: 
- IM acts as a **guide/teacher** for BCE
- Gradients only flow to BCE encoder
- IM encoder learns independently from its own loss
- BCE encoder learns to match IM's interpretable predictions while optimizing for accuracy

### Implementation Details

**Files Modified**:
1. `examples/train_gainakt3exp.py`:
   - Added `enable_encoder_consistency` flag
   - Added `encoder_consistency_weight` parameter
   - Implemented consistency loss computation
   - Added tracking and logging

2. `configs/parameter_default.json`:
   - `enable_encoder_consistency`: false (disabled by default)
   - `encoder_consistency_weight`: 0.1

### Configuration

**CLI Arguments**:
```bash
--enable_encoder_consistency          # Enable regularization
--encoder_consistency_weight 0.1      # Weight for consistency loss
```

**Example Usage**:
```bash
python examples/run_repro_experiment.py \
  --short_title encoder-consistency \
  --epochs 12 \
  --enable_encoder_consistency \
  --encoder_consistency_weight 0.1 \
  --bce_loss_weight 0.9
```

### Expected Benefits

1. **Improved BCE Representations**: BCE learns patterns from interpretable IM encoder
2. **Better Generalization**: Consistency regularization acts as implicit regularization
3. **Interpretability Transfer**: BCE predictions become more aligned with interpretable mastery states
4. **Stable Training**: IM provides consistent guidance throughout training

### Monitoring

The training logs show:
```
Encoder Consistency Regularization ENABLED: IM encoder guides BCE encoder (weight=0.100)

Train - Loss: X.XXXX (BCE: X.XXXX, IM: X.XXXX, Consistency: X.XXXX)
```

### Comparison with Alternatives

| Approach | Description | Gradient Flow | Complexity |
|----------|-------------|---------------|------------|
| **Consistency Regularization** ✓ | Force prediction alignment | IM→BCE only | Low |
| Loss Warm-up | Gradually shift loss weights | Both encoders | Low |
| Knowledge Distillation | IM teaches BCE via soft labels | IM→BCE | Medium |
| Auxiliary Features | IM states as BCE input | Both encoders | High |
| Progressive Training | Train IM first, then BCE | Sequential | Medium |

**Why Consistency Regularization?**
- ✅ Simple to implement and debug
- ✅ Clear gradient flow (IM guides BCE)
- ✅ Both encoders remain active
- ✅ No architectural changes required
- ✅ Interpretable and controllable via single weight parameter

### Hyperparameter Tuning

**Recommended values for `encoder_consistency_weight`**:

| Weight | Effect | Use Case |
|--------|--------|----------|
| 0.0 | Disabled | Baseline comparison |
| 0.05 | Light guidance | Minimal intervention |
| **0.1** | **Moderate guidance** | **Recommended starting point** |
| 0.2 | Strong guidance | IM highly interpretable |
| 0.5+ | Very strong | Risk of underfitting |

**Tuning Guidelines**:
1. Start with 0.1 (default)
2. If BCE AUC drops: reduce to 0.05
3. If IM interpretability needed: increase to 0.2
4. Monitor both `bce_auc` and `im_auc` in metrics

### Validation

Check that consistency regularization is working:

1. **Consistency loss should decrease**: `Consistency: X.XXXX` should go down over epochs
2. **BCE predictions should approach IM**: Measure MSE(BCE_pred, IM_pred) over time
3. **BCE AUC should remain high**: Should not drop significantly compared to baseline
4. **Global AUC should improve**: Weighted combination may benefit from consistency

### Ablation Study Design

To evaluate impact:

```bash
# Baseline (no consistency)
python examples/run_repro_experiment.py --short_title baseline --epochs 12

# With consistency
python examples/run_repro_experiment.py --short_title consistency-0.1 --epochs 12 --enable_encoder_consistency --encoder_consistency_weight 0.1

# Compare metrics_epoch.csv:
# - bce_auc: Should remain similar or improve slightly
# - im_auc: May improve if BCE helps IM indirectly
# - global_auc: Target for improvement
```

### Theoretical Foundation

**Consistency Regularization** is a form of:
1. **Semi-Supervised Learning**: One encoder guides another
2. **Teacher-Student Learning**: IM (teacher) guides BCE (student)
3. **Multi-Task Learning**: Shared representations between tasks

**Related Work**:
- Mean Teacher (Tarvainen & Valpola, 2017)
- Π-Model (Laine & Aila, 2017)
- Temporal Ensembling (Laine & Aila, 2017)

### Future Extensions

1. **Bidirectional Consistency**: Allow BCE to also guide IM (remove detach)
2. **Dynamic Weight Schedule**: Adjust α over training epochs
3. **Layer-wise Consistency**: Match intermediate representations, not just predictions
4. **Confidence-Weighted**: Weight consistency by prediction confidence

### Troubleshooting

**Issue**: Consistency loss stays high
- **Solution**: Increase `encoder_consistency_weight` or train longer

**Issue**: BCE AUC drops significantly
- **Solution**: Reduce `encoder_consistency_weight` to 0.05 or disable

**Issue**: No improvement in global AUC
- **Solution**: IM predictions may need improvement first (see IMPROVE_IM_LOSS.md)

## Conclusion

Encoder Consistency Regularization provides a simple yet effective mechanism for the interpretable IM encoder to guide the BCE encoder toward more interpretable representations while maintaining prediction accuracy. This approach requires minimal code changes and is controlled by a single hyperparameter, making it easy to tune and deploy.

**Status**: ✅ Implemented and tested
**Default**: Disabled (enable with `--enable_encoder_consistency`)
**Recommended**: Enable for interpretability-focused experiments
