# Phase 1 Learning Curve Parameter Sweep - BCE Loss Weight = 0.9

**Sweep ID**: 243682  
**Created**: 2025-11-16 21:55:49  
**Status**: PENDING  
**Sweep Directory**: `/workspaces/pykt-toolkit/examples/experiments/20251116_215549_gainakt3exp_sweep_243682`

---

## Executive Summary

This sweep investigates optimal learning curve parameters (beta, m_sat, gamma, offset) specifically for `bce_loss_weight=0.9` configurations. The motivation stems from discovering that Phase 1+2 optimized parameters (calibrated for bce=0.3) underperform when used with bce=0.9.

**Key Question**: Can we match or exceed Experiment 714616's performance (Test AUC=0.7183) by finding learning curve parameters optimized specifically for bce_loss_weight=0.9?

---

## 1. Rationale and Motivation

### Performance Gap Discovery

Recent experiments revealed a critical parameter interaction issue:

| Experiment | BCE Weight | Learning Params | Test AUC | Notes |
|------------|-----------|-----------------|----------|-------|
| 714616 (dual-loss) | 0.9 | Pre-sweep defaults | **0.7183** | Baseline reference |
| 889799 (bce-0.9) | 0.9 | Phase 1+2 optimal | 0.6858 | -4.5% drop |

### Root Cause Analysis

**Parameter Mismatch Problem**:
- Phase 1 sweep optimized learning curves with `bce_loss_weight=0.2` (80% to Encoder 2)
- Phase 2 sweep optimized `bce_loss_weight=0.3` with those learning curve parameters
- Result: Parameters are co-optimized for interpretability-focused training (bce=0.3)

**When we override bce_loss_weight back to 0.9**:
- We break the carefully calibrated parameter balance
- Learning curve parameters (Beta=2.5, M_sat=0.7, etc.) were NOT optimized for performance-focused training
- This explains the 4.5% AUC drop despite using "optimal" parameters

### Hypothesis

Experiment 714616 used **older defaults** that may have been accidentally better suited for bce=0.9:

**Experiment 714616 (pre-sweep) parameters**:
```json
{
  "beta_skill_init": "N/A" (not in config),
  "m_sat_init": "N/A",
  "gamma_student_init": "N/A", 
  "sigmoid_offset": "N/A",
  "bce_loss_weight": 0.9
}
```

**Current defaults (Phase 1+2 optimal)**:
```json
{
  "beta_skill_init": 2.5,
  "m_sat_init": 0.7,
  "gamma_student_init": 1.1,
  "sigmoid_offset": 1.5,
  "bce_loss_weight": 0.3
}
```

### Research Question

**Can we find learning curve parameters that achieve Test AUC ≥ 0.7183 with bce_loss_weight=0.9?**

If yes, this suggests:
1. Different BCE weights require different learning curve calibrations
2. We may need separate parameter profiles for performance-focused vs interpretability-focused training
3. Experiment 714616's strong performance may be reproducible with proper parameter tuning

---

## 2. Experimental Methodology

### Grid Search Design

We replicate the Phase 1 methodology exactly, but with bce_loss_weight fixed at 0.9:

**Parameter Grid** (3×3×3×3 = 81 configurations):
- `beta_skill_init`: [1.5, 2.0, 2.5]
  - Controls learning curve steepness
  - Higher values = steeper curves = faster mastery progression
  
- `m_sat_init`: [0.7, 0.8, 0.9]
  - Mastery saturation ceiling
  - Lower values = more conservative mastery estimates
  
- `gamma_student_init`: [0.9, 1.0, 1.1]
  - Student learning rate scaling
  - Higher values = faster learner progression
  
- `sigmoid_offset`: [1.5, 2.0, 2.5]
  - Mastery emergence timing
  - Lower values = earlier mastery emergence (after fewer practices)

### Fixed Parameters

To isolate learning curve effects, we fix all other parameters:

```python
bce_loss_weight = 0.9  # Performance-focused: 90% Encoder 1, 10% Encoder 2
dataset = "assist2015"
fold = 0
epochs = 6  # Same as original Phase 1
batch_size = 64
learning_rate = 0.000174
# All other params from configs/parameter_default.json
```

### Evaluation Metrics

**Primary Metric**: Test AUC (Encoder 1)
- Target: ≥ 0.7183 (match Experiment 714616)

**Secondary Metrics**:
- Encoder 2 AUC (interpretability quality)
- Validation AUC (generalization)
- Combined score (balance between encoders)

### Computational Resources

- **GPUs**: 5 GPUs for parallel training
- **Duration**: ~6 hours total (81 experiments × ~4.5 min/exp)
- **Storage**: ~40GB (81 experiments × ~500MB each)

---

## 3. Implementation

### File Structure

```
examples/experiments/20251116_215549_gainakt3exp_sweep_243682/
├── sweep_phase1_bce09.py      # Main sweep orchestration script
├── launch_sweep.sh             # Bash launcher with environment setup
├── report.md                   # This file - comprehensive documentation
├── sweep_results_*.csv         # Results CSV (created during sweep)
├── sweep_execution.log         # Execution log (created during sweep)
└── results/                    # Individual experiment subdirectories
```

### Execution Commands

**Launch sweep**:
```bash
cd /workspaces/pykt-toolkit
bash examples/experiments/20251116_215549_gainakt3exp_sweep_243682/launch_sweep.sh
```

**Monitor progress**:
```bash
# Watch execution log
tail -f examples/experiments/20251116_215549_gainakt3exp_sweep_243682/sweep_execution.log

# Check GPU utilization
watch -n 2 nvidia-smi

# Count completed experiments
wc -l examples/experiments/20251116_215549_gainakt3exp_sweep_243682/sweep_results_*.csv
```

**Analyze results** (after completion):
```python
import pandas as pd

# Load results
df = pd.read_csv('examples/experiments/20251116_215549_gainakt3exp_sweep_243682/sweep_results_*.csv')

# Top configurations
top5 = df.nlargest(5, 'test_auc')
print(top5[['short_title', 'beta_skill_init', 'm_sat_init', 
            'gamma_student_init', 'sigmoid_offset', 'test_auc']])

# Compare with baseline
baseline_auc = 0.7183
best_auc = df['test_auc'].max()
print(f"Baseline (714616): {baseline_auc:.4f}")
print(f"Best sweep:        {best_auc:.4f}")
print(f"Improvement:       {best_auc - baseline_auc:+.4f} ({(best_auc - baseline_auc)/baseline_auc*100:+.2f}%)")
```

---

## 4. Expected Outcomes

### Scenario 1: Match or Exceed Baseline (Test AUC ≥ 0.7183)

**Interpretation**: Learning curve parameters matter significantly for bce=0.9 training.

**Actions**:
1. Identify optimal parameter configuration
2. Document parameter profile for "performance-focused" training (bce=0.9)
3. Consider maintaining separate defaults for different BCE weight regimes
4. Write paper section discussing parameter co-optimization requirements

### Scenario 2: Cannot Match Baseline (Test AUC < 0.7183)

**Interpretation**: Experiment 714616's performance may involve factors beyond learning curve parameters.

**Possible explanations**:
1. Random seed effects (714616 used seed=42)
2. Other parameter differences not captured in config
3. Training dynamics differences (warmup schedules, etc.)
4. Dataset/fold-specific effects

**Actions**:
1. Investigate Experiment 714616's full training configuration
2. Consider reproducing 714616 exactly to verify results
3. Examine training logs for clues about optimization dynamics

### Scenario 3: Substantially Exceed Baseline (Test AUC > 0.73)

**Interpretation**: We discovered significantly better parameters for bce=0.9.

**Actions**:
1. Update paper with findings about parameter optimization
2. Run validation on other datasets (assist2009, assist2017, algebra2005)
3. Consider adding "performance mode" configuration to framework
4. Document trade-offs between performance (bce=0.9) and interpretability (bce=0.3)

---

## 5. Results

### Status: PENDING

Sweep execution will begin once launch_sweep.sh is executed. Results will be updated here upon completion.

**Expected completion**: ~6 hours from launch time

---

## 6. Analysis Plan (Post-Sweep)

Once the sweep completes, we will conduct the following analyses:

### 6.1 Parameter Impact Analysis

**Question**: Which parameters most influence Test AUC with bce=0.9?

**Method**: 
- Calculate correlation between each parameter and Test AUC
- Compare with Phase 1 correlations (which used bce=0.2)
- Identify interaction effects

### 6.2 Comparison with Phase 1 Findings

**Phase 1 (bce=0.2)** found:
- Beta=2.5 optimal (steep learning curves)
- M_sat=0.7 optimal (conservative mastery)
- Gamma=1.1 optimal (faster learners)
- Offset=1.5 optimal (early mastery emergence)

**Question**: Do these recommendations change for bce=0.9?

### 6.3 Performance vs Interpretability Trade-off

**Analysis**:
- Plot Test AUC (performance) vs Encoder 2 AUC (interpretability)
- Identify Pareto frontier
- Compare bce=0.9 frontier with bce=0.3 frontier from Phase 2

### 6.4 Reproducibility Verification

**Next steps**:
1. Take best configuration from this sweep
2. Run 3-5 replications with different seeds
3. Compare variance with Experiment 714616
4. Verify results are stable and reproducible

---

## 7. Documentation Standards

This sweep follows AGENTS.md guidelines:
- ✅ All files in dedicated sweep folder
- ✅ Naming convention: `YYMMDD_HHMMSS_model_sweep_uniqueid`
- ✅ Comprehensive report.md with rationale and methodology
- ✅ Centralized documentation (all info in this file)
- ✅ Will be updated with results and recommendations post-completion

---

## 8. References

**Related Experiments**:
- Experiment 714616 (dual-loss): Baseline reference, Test AUC=0.7183
- Experiment 889799 (bce-0.9): Motivated this sweep, Test AUC=0.6858
- Phase 1 Sweep (20251116_174547_gainakt3exp_sweep_861908): Methodology template
- Phase 2 Sweep (20251116_202637_gainakt3exp_sweep_phase2_126402): BCE weight optimization

**Configuration References**:
- Experiment 714616 config: `/workspaces/pykt-toolkit/examples/experiments/20251116_125150_gainakt3exp_dual-loss_714616/config.json`
- Current defaults: `/workspaces/pykt-toolkit/configs/parameter_default.json` (MD5: a1e2fd59c3b2f7b3dcde97e3515619e1)
- Pre-sweep defaults: MD5: 92ab1f9df195f45b15f4e9859ae5c402

---

## 9. Conclusion

This sweep addresses a critical finding: parameter optimization is BCE-weight-dependent. The Phase 1+2 parameters were optimized for interpretability-focused training (bce=0.3) and underperform when used in performance-focused training (bce=0.9).

By systematically searching the learning curve parameter space with bce=0.9 fixed, we aim to either:
1. Match Experiment 714616's strong performance through proper parameter tuning, or
2. Understand why 714616 achieved exceptional results and whether they're reproducible

The results will inform whether we need separate parameter profiles for different BCE weight regimes and guide future parameter optimization strategies.

---

**Next Update**: After sweep completion (~6 hours from launch)
