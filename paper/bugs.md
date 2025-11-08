# Bug Report and Resolution Plan

This document outlines critical bugs discovered in the `GainAKT2Exp` model. We will address them in the following order, as agreed:
1. The Loss Normalization Problem: To ensure all loss components are stable and interpretable.
2. The Legacy Problem: To ensure auxiliary losses are correctly applied during training.
3. The Intrinsic Problem: To fix the broken monitoring pathway for the intrinsic model.

---

## 1. The Loss Normalization Problem: Incorrect Loss Share Calculation

### Symptom

In the `metrics_epoch.csv` logs, the `main_loss_share` can be greater than 1.0, while other loss shares (e.g., `alignment_loss_share`) can be negative. This is mathematically confusing and makes it difficult to interpret the contribution of each loss component.

### Root Cause

This happens when an auxiliary loss component becomes negative. The logging logic calculates shares relative to a total that is now smaller than the main loss, leading to distorted proportions. The negative values likely stem from correlation-based losses that are not constrained to be positive.

### Steps to Fix

The fix involves ensuring all loss components are non-negative and then normalizing them correctly for logging.

1.  **Step 1: Ensure Non-Negative Losses**
    *   **File**: `pykt/models/gainakt2_exp.py`
    *   **Function**: `compute_interpretability_loss`
    *   **Action**: Review each loss calculation. For any loss that can become negative (like correlation-based ones), ensure it is clamped at zero. The `mastery_performance_loss` and `gain_performance_loss` are the primary candidates for this change.

2.  **Step 2: Return and Log Unweighted Losses**
    *   **File**: `pykt/models/gainakt2_exp.py`
    *   **Action**: Modify `compute_interpretability_loss` to return a dictionary of the individual, *unweighted* loss values (e.g., `{'monotonicity': mono_loss, 'sparsity': sparse_loss}`).

3.  **Step 3: Normalize Shares in the Training Loop**
    *   **File**: `examples/train_gainakt2exp.py`
    *   **Function**: `train_epoch`
    *   **Action**:
        *   In the training loop, collect the main loss and the dictionary of unweighted auxiliary losses.
        *   Create the `total_loss` for backpropagation by applying weights to the auxiliary losses and summing them with the main loss.
        *   For logging, create a separate dictionary of the *weighted* loss components. Sum them up to get a final total.
        *   Calculate the share of each component by dividing its weighted value by the final total. This will ensure all shares are positive and sum to 1.0.

---

## 2. The Legacy Problem: Auxiliary Losses Have No Effect

### Symptom

In the standard `GainAKT2Exp` model (where `intrinsic_gain_attention=False`), auxiliary interpretability losses are calculated but have no actual impact on the model's training.

### Root Cause

The main training loop in `examples/train_gainakt2exp.py` calls the model's `forward` pass, but the value that `loss.backward()` is called on is *only* the main prediction loss. The auxiliary losses are effectively ignored by the optimizer.

### Steps to Fix

The fix involves modifying the training script to correctly combine the main loss with the auxiliary losses before backpropagation. This will be done as part of the fix for the "Loss Normalization Problem".

1.  **Open the training script**:
    *   File: `examples/train_gainakt2exp.py`

2.  **Locate the loss composition**:
    *   Find the `train_epoch` function. Inside the loop, find the lines where the total loss for backpropagation is defined.

3.  **Update the backpropagation call**:
    *   Ensure that `loss.backward()` is called on a combined loss value that includes the main prediction loss and all weighted auxiliary losses.

---

## 3. The Intrinsic Problem: Monitoring Pathway is Broken

### Symptom

When `intrinsic_gain_attention=True`, the model fails to produce any interpretability metrics. All correlations are zero, and all auxiliary losses are disabled.

### Root Cause

There is no mechanism to pass the `aggregated_gains` (calculated in the `MultiHeadAttention` layer) up to the `GainAKT2Exp` class, where the monitoring and loss calculations occur. The values are computed but never retrieved.

### Steps to Fix

This fix involves creating a direct channel to pass the intrinsic gains from the attention mechanism up to the loss calculation logic.

1.  **Step 1: Expose Gains from the Attention Layer**
    *   **File**: `pykt/models/gainakt2.py`
    *   **Class**: `MultiHeadAttention`
    *   **Action**: In the `forward` method, after `aggregated_gains` is calculated, add a line to store it as an instance variable: `self.last_aggregated_gains = aggregated_gains`.

2.  **Step 2: Create a Getter Method in the Main Model**
    *   **File**: `pykt/models/gainakt2.py`
    *   **Class**: `GainAKT2`
    *   **Action**: Add a new method to the class to retrieve the stored gains from the final encoder block: `get_aggregated_gains`.

3.  **Step 3: Implement the Intrinsic Monitoring Pathway**
    *   **File**: `pykt/models/gainakt2_exp.py`
    *   **Class**: `GainAKT2Exp`
    *   **Action**: Modify the `forward_with_states` method. Find the `if self.intrinsic_gain_attention:` block and replace its contents with logic to call the new getter and use its results to compute `projected_mastery` and `projected_gains`.


