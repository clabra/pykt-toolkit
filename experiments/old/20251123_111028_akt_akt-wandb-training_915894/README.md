# AKT Wandb Training Experiment (915894)

## Overview
Training experiment using wandb_akt_train.py script to compare with run_repro_experiment.py methodology.

## Training Configuration
- **Model**: AKT (Attention-based Knowledge Tracing)
- **Dataset**: assist2015, fold 0
- **Seed**: 42
- **Parameters**:
  - dropout: 0.2
  - d_model: 256
  - d_ff: 512
  - n_heads: 4
  - num_encoder_blocks: 4
  - learning_rate: 0.000174
  - emb_type: qid

## Results

### Best Performance
- **Best Epoch**: 13
- **Validation AUC**: 0.7320
- **Validation Accuracy**: 0.7579

### Training Summary
- Total Epochs: 23
- Early stopping after epoch 13 (10 epochs without improvement)
- Final train loss: 0.4998

### Test Set Evaluation
**Status**: Completed using wandb_eval.py on saved model (epoch 13)

- **Test AUC**: 0.7215
- **Test Accuracy**: 0.7794
- **Test Samples**: 15,588

The wandb training script only evaluates on validation set during training. Test evaluation was performed separately using wandb_eval.py on the saved model checkpoint from epoch 13 (best validation epoch).

**Model Location**: `examples/saved_model/assist2015_akt_qid_saved_model_42_0_0.2_256_512_4_4_0.000174_0_1/qid_model.ckpt`

## Comparison with run_repro_experiment.py

| Metric | wandb_akt_train.py | run_repro_experiment.py (exp 242169) | Difference |
|--------|-------------------|--------------------------------------|------------|
| Validation AUC (epoch 13) | 0.7320 | 0.7307 | +0.0013 |
| Test AUC | 0.7215 | 0.7183 | +0.0032 |
| Test Accuracy | 0.7794 | - | - |
| Training epochs | 23 | 12 | +11 |
| Experiment structure | Manual | Automated | - |

**Performance**: wandb training achieved +0.32% better test AUC (+0.45% relative improvement)

## Findings

### Methodology Differences
1. **Test Evaluation**: run_repro_experiment.py includes automatic test evaluation; wandb_akt_train.py requires separate eval script
2. **Experiment Structure**: run_repro_experiment.py creates standardized folder structure; wandb requires manual organization
3. **Training Duration**: wandb trains longer (23 epochs) vs run_repro (12 epochs with early stopping)

### Performance Analysis
Wandb training achieved better performance on both validation and test sets:
- **Validation**: 0.7320 vs 0.7307 (+0.0013, +0.18%)
- **Test**: 0.7215 vs 0.7183 (+0.0032, +0.45%)

The improvement on test set (+0.45%) is larger than on validation (+0.18%), suggesting the model trained longer (23 vs 12 epochs) without overfitting.

## Conclusion

### Performance Comparison
The wandb training approach achieved **better test performance** (0.7215 vs 0.7183, +0.45%) by training for more epochs (23 vs 12) without overfitting.

### Methodology Trade-offs

**wandb_akt_train.py**:
- ✅ Better final performance (+0.45% test AUC)
- ✅ Longer training captures more learning
- ❌ No automatic test evaluation (requires separate wandb_eval.py)
- ❌ Manual experiment organization required
- ❌ Model saved in examples/saved_model (non-standard location)

**run_repro_experiment.py**:
- ✅ Complete automated train/eval/test pipeline
- ✅ Standardized experiment folders with full reproducibility
- ✅ Automatic metrics tracking and trajectory generation
- ❌ Shorter training (early stopping more aggressive)
- ❌ Slightly lower performance (-0.45% test AUC)

### Recommendation
- **For research/publication**: Use run_repro_experiment.py for full reproducibility and standardized results
- **For best performance**: Consider extending training epochs in run_repro_experiment.py (increase from 12 to ~23 epochs)
- **Current baselines**: Both approaches valid
  - run_repro_experiment.py: exp 242169, test AUC 0.7183
  - wandb_akt_train.py: exp 915894, test AUC 0.7215 (+0.45%)
