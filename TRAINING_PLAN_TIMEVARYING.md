# Training Plan: Time-Varying BKT Parameters

## Context
We implemented a fix for static BKT parameters (TIME_VARYING_BKT_FIX.md), but the gallery regeneration showed the same flattening patterns because:
- **Old model**: Trained with static parameter bug (Exp 533154)
- **New code**: Evaluating with time-varying fix
- **Mismatch**: Model weights are calibrated for wrong computation

**Conclusion**: We need to train a NEW model from scratch with the time-varying fix.

---

## Training Configuration (Based on Exp 533154 - Minimalist Grounding)

### Core Architecture
```json
{
  "model": "gtransformer",
  "dataset": "assist2009",
  "fold": 0,
  "seed": 42,
  
  "d_model": 64,
  "n_heads": 8,
  "n_blocks": 2,
  "d_ff": 256,
  "dropout": 0.1,
  
  "batch_size": 64,
  "learning_rate": 0.0001,
  "epochs": 200,
  "patience": 10
}
```

### Grounding Configuration (Minimalist)
```json
{
  "lambda_sup": 1.0,          // Supervised loss (primary)
  "lambda_ref": 0.5,          // Reference BKT loss
  "lambda_probe": 1.0,        // Active grounding (global)
  "lambda_initmastery": 0.0,  // Removed (redundant)
  "lambda_rate": 0.0,         // Removed (redundant)
  "active_grounding": 1       // Enable probing
}
```

### Personalization
```json
{
  "n_uid": 0,                 // Context-only (no student embeddings)
  "personalization": false
}
```

### Regularization
```json
{
  "l2": 1e-05,                // Question difficulty regularization
  "l2_rasch": 1e-05,          // Rasch regularization
  "lambda_student": 1e-05,    // Student bias regularization
  "lambda_gap": 1e-05         // Gap parameter regularization
}
```

---

## Quick Sanity Check Plan (Option B)

Before full training, run a **mini-experiment** to verify the fix works:

### Quick Test Configuration
- **Dataset**: assist2009 (same as baseline)
- **Fold**: 0 (same as baseline)
- **Epochs**: 10 (instead of 200)
- **Subset**: First 1000 students only (if possible)
- **Goal**: Verify that p_ref shows MORE variance than old model

### Success Criteria
1. Training completes without errors
2. Loss curves look normal (no NaN/Inf)
3. Quick gallery generation (5 cases) shows MORE dynamic p_ref than old gallery
4. No regression in p_sup predictions

### Execution Time
- ~10-20 minutes (10 epochs on assist2009)

---

## Full Training Plan (Option A)

### Experiment Setup
```bash
# Command to launch full training
python examples/run_repro_experiment.py \
  --model_name gtransformer \
  --train_script examples/wandb_gtransformer_train.py \
  --dataset assist2009 \
  --fold 0 \
  --short_title timevarying_bkt \
  --active_grounding 1 \
  --lambda_probe 1.0 \
  --lambda_initmastery 0.0 \
  --lambda_rate 0.0 \
  --num_gpus 1
```

### Expected Outputs
1. **New experiment directory**: `experiments/[timestamp]_timevarying_bkt_[exp_id]/`
2. **Trained checkpoint**: `gtransformer/assist2009/fold_0_[id]/qid_model.ckpt`
3. **Evaluation results**: `eval_results.json`
4. **Training logs**: Loss curves, probe correlations, AUC per epoch

### Execution Time
- **Training**: ~2-4 hours (200 epochs with early stopping on assist2009)
- **Evaluation**: ~10 minutes (test set predictions)

### Success Criteria
1. **Test AUC (p_sup)**: ≥0.7788 (no regression from baseline)
2. **p_ref AUC**: >0.6756 (improvement from baseline)
3. **Probe correlation**: R² ≥0.509 (maintain or improve)
4. **Visual validation**: Gallery shows more dynamic p_ref trajectories

---

## Post-Training Analysis Plan

Once training completes, regenerate ALL plots for comparison:

### 1. Envelope Plots (4 plots)
```bash
# Quadrant analysis
python3 examples/validation/generate_quadrant_analysis.py \
  --exp_dir experiments/[NEW_EXP]/gtransformer/assist2009/fold_0_[ID] \
  --output_dir examples/validation/results_timevarying

# Gallery (3×3)
python3 examples/validation/generate_prediction_envelope_gallery.py \
  --exp_dir experiments/[NEW_EXP]/gtransformer/assist2009/fold_0_[ID] \
  --output_dir examples/validation/results_timevarying

# Disagreement heatmap
python3 examples/validation/generate_disagreement_heatmap.py \
  --exp_dir experiments/[NEW_EXP]/gtransformer/assist2009/fold_0_[ID] \
  --output_dir examples/validation/results_timevarying

# Distribution analysis
python3 examples/validation/generate_envelope_distribution.py \
  --exp_dir experiments/[NEW_EXP]/gtransformer/assist2009/fold_0_[ID] \
  --output_dir examples/validation/results_timevarying
```

### 2. Latent Space Plots (3 plots)
```bash
python tmp/plot_latent_pca.py \
  --exp_dir experiments/[NEW_EXP]/gtransformer/assist2009/fold_0_[ID] \
  --output_dir examples/validation/results_timevarying
```

### 3. Comparative Analysis
```bash
python compare_pref_variance.py \
  --old_dir examples/validation/results_exp533154 \
  --new_dir examples/validation/results_timevarying
```

---

## Key Metrics to Track

### Quantitative (from eval_results.json)
| Metric | Baseline (Exp 533154) | Target (Time-Varying) | Improvement |
|:-------|:---------------------:|:---------------------:|:-----------:|
| Test AUC (p_sup) | 0.7790 ± 0.0015 | ≥0.7788 | No regression |
| p_ref AUC | 0.6756 ± 0.0028 | **TBD** | **Expected ↑** |
| Probe R² (L0) | 0.509 | **TBD** | Maintain |
| Mean Envelope Width | 0.2086 | **TBD** | **Expected ↓** |

### Qualitative (from plots)
- **p_ref variance**: Should increase (less flat trajectories)
- **Envelope width**: Should decrease (better agreement with p_sup)
- **Disagreement heatmap**: Low L0/High T quadrant should improve
- **Gallery diversity**: Should show richer temporal dynamics

---

## Risk Assessment

### Low Risk
- Code changes are minimal (4 lines)
- All validation tests passed
- Backward compatible (no API changes)

### Medium Risk
- Training time investment (~2-4 hours)
- May need hyperparameter tuning if convergence differs

### Mitigation Strategy
- **Quick test first** (Option B) to catch issues early
- Monitor training logs closely (especially probe losses)
- Compare loss curves with baseline experiment

---

## Decision Point

**Ready to proceed?**
1. ✅ Configuration reviewed (this document)
2. ⏳ Quick sanity check (Option B) - **NEXT**
3. ⏳ Full training run (Option A) - **AFTER B**
