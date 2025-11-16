# Dual-Encoder Metrics Implementation Summary

**Date**: 2025-11-16  
**Implementer**: AI Assistant  
**Model**: GainAKT3Exp (Dual-Encoder Architecture)

## Overview

Implemented comprehensive dual-encoder metrics tracking and visualization across the entire GainAKT3Exp pipeline, following the 3-step plan to enhance observability of both encoder paths (Performance and Interpretability).

## Step 1: CSV Metrics Enhancement ✅

### Changes to `examples/train_gainakt3exp.py`:

1. **Updated CSV Header** (lines 617-640):
   - Added 28 new columns for comprehensive dual-encoder tracking
   - Overall metrics (combined): train/val loss, AUC, accuracy
   - Loss components (unweighted): BCE loss, IM loss
   - Weighted losses: weighted BCE, weighted IM, total weighted
   - Loss shares: percentage contribution of each loss component
   - Encoder-specific metrics: AUC and accuracy for both encoders

2. **Updated Training Loop** (lines 800-803):
   - Added separate prediction tracking lists:
     - `total_predictions_encoder1`: Base predictions from Encoder 1
     - `total_predictions_encoder2`: IM predictions from Encoder 2

3. **Updated Prediction Collection** (lines 1176-1198):
   - Collect predictions from both encoders during training
   - Extract `incremental_mastery_predictions` from model outputs
   - Handle cases where IM predictions not available (fallback to zeros)

4. **Updated Validation Loop** (lines 1243-1249):
   - Added separate validation tracking for both encoders
   - Track validation loss components separately

5. **Updated Validation Prediction Collection** (lines 1280-1305):
   - Collect predictions from both encoders during validation
   - Track incremental mastery loss per batch

6. **Added Encoder Metrics Computation** (lines 1312-1358):
   - Compute AUC/accuracy for Encoder 1 (train & validation)
   - Compute AUC/accuracy for Encoder 2 (train & validation)
   - Compute weighted losses and shares for both train and validation

7. **Updated CSV Writing** (lines 1708-1738):
   - Write all 28 metrics to CSV with 6 decimal precision
   - Include both encoders' performance metrics

### Changes to `examples/eval_gainakt3exp.py`:

1. **Updated `evaluate_predictions` Function** (lines 71-146):
   - Returns dict with metrics for combined, encoder1, and encoder2
   - Collects predictions from both encoders
   - Handles missing `incremental_mastery_predictions` gracefully

2. **Updated Main Evaluation** (lines 303-340):
   - Call updated `evaluate_predictions` for all splits
   - Store metrics for both encoders in results dict
   - Include encoder-specific AUC and accuracy

3. **Updated CSV Output** (lines 361-411):
   - Added encoder1/encoder2 columns to metrics_epoch_eval.csv
   - Write dual-encoder metrics for train/valid/test splits

### Changes to `configs/parameter_default.json`:

- Added `bce_loss_weight` to `interpretability` type group (line 119)

## Step 2: Monitoring Hook Enhancement ✅

### Changes to `pykt/models/gainakt3_exp.py`:

**Updated Monitor Call** (lines 710-735):
- Pass outputs from both encoders to monitor
- Encoder 1 outputs: `context_seq_1`, `value_seq_1`, `base_predictions`
- Encoder 2 outputs: `context_seq_2`, `value_seq_2`, `projected_mastery`, `projected_gains`, `incremental_mastery_predictions`
- Common inputs: `questions`, `responses`

### Changes to `examples/interpretability_monitor.py`:

1. **Updated `__call__` Signature** (lines 32-60):
   - Accept outputs from both encoders as separate parameters
   - Added comprehensive docstring explaining dual-encoder architecture

2. **Updated Metric Computation** (lines 61-99):
   - `encoder1_mastery_corr`: Correlation between Encoder 1 predictions and mastery
   - `encoder2_mastery_corr`: Correlation between Encoder 2 predictions and mastery
   - Gain/correctness correlation from Encoder 2
   - Non-negative gains violation rate (Encoder 2)
   - Mastery monotonicity violations (Encoder 2)
   - Added None checks for all Encoder 2-specific outputs

## Step 3: Learning Trajectories Enhancement ✅

### Changes to `examples/learning_trajectories.py`:

1. **Updated `extract_trajectory` Function** (lines 140-230):
   - **Dual-Encoder Predictions**:
     - Extract `predictions` (Encoder 1) and `incremental_mastery_predictions` (Encoder 2)
     - Store both in trajectory data
   
   - **Enhanced Gain Computation**:
     - Priority: `projected_gains` > `projected_gains_d` > compute from `value_seq_2` > zeros
     - Added fallback to compute gains from Encoder 2 value sequence
   
   - **Multi-Skill Support**:
     - Handle questions with multiple skills (generic case)
     - Parse skill IDs as list or single value
     - Store all skills practiced per step
     - Collect gains/mastery for all involved skills
   
   - **Updated Step Data Structure**:
     ```python
     {
       'timestep': t,
       'skills_practiced': [skill_id1, skill_id2, ...],  # List of skills
       'gains': {skill_id: gain_val, ...},  # Dict per skill
       'mastery': {skill_id: mastery_val, ...},  # Dict per skill
       'true_response': 0 or 1,
       'prediction_encoder1': float,  # Base prediction
       'prediction_encoder2': float or None  # IM prediction
     }
     ```

2. **Updated `print_trajectory` Function** (lines 233-320):
   - **Dual-Encoder Display**:
     - Show predictions from both encoders side-by-side
     - Display match indicators for both encoders (M1, M2)
     - Columns: Step | Skill(s) | True | Enc1 | Enc2 | M1 | M2 | Gain | Mastery
   
   - **Multi-Skill Display**:
     - Show all skills in compact format (e.g., "45,67,89" or "45,+2")
     - Display gains/mastery for primary skill
   
   - **Adaptive Layout**:
     - Automatically adjusts display based on encoder availability
     - Falls back to single-encoder mode if Encoder 2 predictions unavailable
   
   - **Legend**:
     - Prints legend explaining Enc1/Enc2 and M1/M2 when in dual-encoder mode

## New CSV Schema

### `metrics_epoch.csv` (28 columns):

| Column | Description |
|--------|-------------|
| epoch | Epoch number |
| train_loss | Combined training loss |
| train_auc | Combined training AUC |
| train_acc | Combined training accuracy |
| val_loss | Combined validation loss |
| val_auc | Combined validation AUC |
| val_acc | Combined validation accuracy |
| train_bce_loss | Unweighted BCE loss (training) |
| train_im_loss | Unweighted IM loss (training) |
| val_bce_loss | Unweighted BCE loss (validation) |
| val_im_loss | Unweighted IM loss (validation) |
| train_weighted_bce | Weighted BCE loss (λ₁ × BCE) |
| train_weighted_im | Weighted IM loss ((1-λ₁) × IM) |
| train_total_weighted | Total weighted loss (training) |
| val_weighted_bce | Weighted BCE loss (validation) |
| val_weighted_im | Weighted IM loss (validation) |
| val_total_weighted | Total weighted loss (validation) |
| train_bce_share | BCE loss percentage (training) |
| train_im_share | IM loss percentage (training) |
| val_bce_share | BCE loss percentage (validation) |
| val_im_share | IM loss percentage (validation) |
| train_encoder1_auc | Encoder 1 AUC (training) |
| train_encoder1_acc | Encoder 1 accuracy (training) |
| val_encoder1_auc | Encoder 1 AUC (validation) |
| val_encoder1_acc | Encoder 1 accuracy (validation) |
| train_encoder2_auc | Encoder 2 AUC (training) |
| train_encoder2_acc | Encoder 2 accuracy (training) |
| val_encoder2_auc | Encoder 2 AUC (validation) |
| val_encoder2_acc | Encoder 2 accuracy (validation) |
| ... | (interpretability metrics) |

### `metrics_epoch_eval.csv` (11 columns):

| Column | Description |
|--------|-------------|
| split | train/validation/test |
| auc | Combined AUC |
| accuracy | Combined accuracy |
| encoder1_auc | Encoder 1 AUC |
| encoder1_acc | Encoder 1 accuracy |
| encoder2_auc | Encoder 2 AUC |
| encoder2_acc | Encoder 2 accuracy |
| mastery_correlation | Mastery-performance correlation |
| gain_correlation | Gain-correctness correlation |
| correlation_students | Number of students in correlation sample |
| timestamp | Evaluation timestamp |

## Benefits

1. **Complete Observability**: Track both encoder paths independently
2. **Loss Analysis**: See unweighted, weighted, and percentage contributions
3. **Performance Comparison**: Compare Encoder 1 (performance) vs Encoder 2 (interpretability) predictions
4. **Multi-Skill Support**: Generic framework handles datasets with multi-skill questions
5. **Trajectory Visualization**: See how both encoders learn over student interaction sequences
6. **Backward Compatibility**: Legacy single-encoder displays work when Encoder 2 unavailable

## Files Modified

1. `examples/train_gainakt3exp.py` - Training metrics and CSV
2. `examples/eval_gainakt3exp.py` - Evaluation metrics and CSV
3. `pykt/models/gainakt3_exp.py` - Monitor call with dual-encoder outputs
4. `examples/interpretability_monitor.py` - Monitor handler for dual encoders
5. `examples/learning_trajectories.py` - Trajectory extraction and display
6. `configs/parameter_default.json` - Parameter categorization

## Testing Recommendation

Run a short experiment to verify:
```bash
python examples/run_repro_experiment.py \
  --short_title "test_dual_metrics" \
  --epochs 2 \
  --bce_loss_weight 0.5
```

Then check:
- `metrics_epoch.csv` has 28 columns with dual-encoder metrics
- `metrics_epoch_eval.csv` has encoder1/encoder2 columns
- Learning trajectories show Enc1/Enc2 predictions side-by-side

