# Pure Interpretability + Active Probing Campaign

**Launch Date**: January 16, 2026  
**Campaign PID**: 250263  
**Log File**: `experiments/pure_interp_probing_20260116_*.log`

## Experiment Design

This campaign tests whether **Active Probing** (forcing latent representations to be linearly organized) can enable learning with **pure interpretability** (no direct supervised loss).

### Research Question

Can a model learn effectively using only:
1. **BKT Reference Loss** (`lambda_ref=1.0`) - Indirect supervision via BKT predictions
2. **Probing Loss** (`lambda_probe=1.0`) - Force latent space to encode BKT parameters
3. **Parameter Grounding** (`lambda_initmastery=0.1`, `lambda_rate=0.1`) - Constrain parameters to pedagogical ranges

Without:
- **Direct Supervised Loss** (`lambda_sup=0.0`) - No BCE on transformer predictions

### Comparison Baseline: Exp 090230

| Parameter | Exp 090230 (Baseline) | This Campaign | Change |
|-----------|----------------------|---------------|--------|
| **lambda_sup** | 1.0 | **0.0** | ❌ Removed supervised loss |
| **lambda_ref** | 0.5 | **1.0** | ✅ Doubled BKT reference weight |
| **active_grounding** | 0 | **1** | ✅ Enabled probing |
| **lambda_probe** | 0.0 | **1.0** | ✅ Added probing loss |
| lambda_initmastery | 0.1 | 0.1 | Same |
| lambda_rate | 0.1 | 0.1 | Same |
| n_blocks | 2 | 2 | Same |
| n_heads | 8 | 8 | Same |
| d_model | 64 | 64 | Same |
| d_ff | 256 | 256 | Same |
| dropout | 0.1 | 0.1 | Same |

### Previous Results

**Exp 229843** (Pure Interpretability WITHOUT Probing):
- Configuration: `lambda_sup=0.0`, `lambda_ref=0.5`, `active_grounding=0`
- Result: **AUC 0.5137 ± 0.0002** (CATASTROPHIC FAILURE)
- Interpretation: Near-random predictions, supervised loss is essential

**Exp 090230** (Baseline with Supervised Loss):
- Configuration: `lambda_sup=1.0`, `lambda_ref=0.5`, `active_grounding=0`
- Result: **AUC 0.7800 ± 0.0013**
- Interpretation: Strong performance with direct supervision

**Exp 636452** (Active Probing WITH Supervised Loss):
- Configuration: `lambda_sup=1.0`, `lambda_ref=0.5`, `active_grounding=1`, `lambda_probe=1.0`
- Result: **AUC 0.7758 ± 0.0037**
- Interpretation: Probing maintains performance when supervised loss is present

## Expected Outcomes

### Scenario 1: Success (AUC ≥ 0.75)
**Hypothesis**: Active probing provides sufficient gradient signal through latent space organization, allowing the model to learn meaningful representations even without direct supervision.

**Mechanism**: 
- Probing loss forces `z_t` to encode BKT parameters linearly
- BKT reference loss provides indirect supervision (predictions → true labels)
- Combined, these create a learning path that approximates supervised learning

**Implication**: "Probing-Guided Learning" is a viable alternative to direct supervision for interpretable models.

### Scenario 2: Moderate Success (0.60 < AUC < 0.75)
**Hypothesis**: Probing helps but is insufficient to fully replace supervised loss.

**Mechanism**: 
- Probing provides some structure to latent space
- BKT reference loss is too indirect (gradients diluted through O(T²) logic)
- Performance is better than random but below supervised baseline

**Implication**: Probing reduces but doesn't eliminate the need for direct supervision.

### Scenario 3: Failure (AUC ≤ 0.60)
**Hypothesis**: Without direct supervision, the model cannot learn effectively regardless of probing.

**Mechanism**:
- Probing loss only affects latent space organization, not prediction quality
- BKT reference loss alone is insufficient (as shown in Exp 229843)
- Increased `lambda_ref` from 0.5 to 1.0 is not enough to compensate

**Implication**: Direct supervision is fundamentally necessary; interpretability constraints are secondary.

## Loss Function Breakdown

The total loss during training is:

```
L_total = λ_sup·L_sup + λ_ref·L_ref + λ_init·L_L0 + λ_rate·L_T + λ_probe·L_probe + L_reg
```

For this campaign:
- **L_sup**: DISABLED (`lambda_sup=0.0`)
  - `BCE(y_pred, y_true)` where y_pred are transformer predictions
  - This is the primary supervision signal in standard models

- **L_ref**: STRONG (`lambda_ref=1.0`)
  - `BCE(y_bkt, y_true)` where y_bkt are BKT reference predictions
  - Gradients flow backward through differentiable BKT walk
  - Indirect supervision path

- **L_probe**: ENABLED (`lambda_probe=1.0`)
  - `MSE(probe_L0(z_t), oracle_L0) + MSE(probe_T(z_t), oracle_T)`
  - Forces latent vectors to encode BKT parameters linearly
  - Provides structure to representation learning

- **L_param**: ENABLED (`lambda_init=0.1`, `lambda_rate=0.1`)
  - `MSE(p_L0, oracle_L0) + MSE(p_T, oracle_T)`
  - Constrains projected parameters to match BKT oracle
  - Ensures pedagogical validity

## Monitoring Commands

```bash
# Check campaign status
tail -f experiments/pure_interp_probing_*.log

# Monitor GPU usage
nvidia-smi

# Check individual fold progress
ls -ltr experiments/2026*/gtransformer/assist2009/

# View training logs for a specific fold
tail -f experiments/2026*/gtransformer/assist2009/fold_0_*/execution_history.log

# After training completes (check for all 5 folds finished):
python3 examples/run_benchmarks_paper.py --mode evaluation --dataset assist2009 --model gtransformer

# Gather final results:
python3 examples/run_benchmarks_paper.py --mode results --dataset assist2009 --model gtransformer
```

## Expected Timeline

- **Training**: ~30-40 minutes per fold (200 epochs)
- **Total**: ~2-3 hours (5 folds in parallel on GPUs 0-4)
- **Evaluation**: ~5-10 minutes after training completes
- **Results**: Immediate after evaluation

## Documentation Plan

After results are available:
1. Update `paper/benchmark_paper.md` with new experiment section
2. Compare with Exp 229843 (pure interp without probing)
3. Compare with Exp 636452 (probing with supervised loss)
4. Analyze whether probing "rescues" pure interpretability mode
5. Document findings in research narrative

---

**Status**: Training in progress (PID 250263)  
**Next Step**: Wait for training to complete, then run evaluation
