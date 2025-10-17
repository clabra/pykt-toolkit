# Experimental Configurations: GainAKT2Exp Interpretability vs. Baselines

This document specifies the experimental conditions used to assess the interpretable **GainAKT2Exp** architecture and its ablations. It formalizes what we refer to as the *full model* and the *heads‑disabled baseline*, and proposes additional optional variants for more granular attribution. The goal is to ensure transparent, reproducible comparisons isolating the contribution of interpretability heads and constraint-based auxiliary objectives.

---
## 1. Model Components

| Component | Purpose | Active in Full Model | Active in Heads‑Disabled Baseline |
|-----------|---------|----------------------|-----------------------------------|
| Two‑stream Transformer encoder (context/value) | Core sequence modeling | Yes | Yes |
| Mastery projection head (`mastery_head`) | Estimates cumulative latent mastery per concept | Yes | No |
| Gain projection head (`gain_head`) | Estimates instantaneous learning gains per concept | Yes | No |
| Architectural non‑negativity enforcement (ReLU on gains) | Guarantees gains ≥ 0 | Yes (applied) | Not instantiated |
| Cumulative mastery accumulator (monotonic enforcement) | Enforces monotonic mastery trajectory | Yes | Not executed |
| Auxiliary interpretability loss terms | Regularize projections for semantic alignment | Yes | No (loss = 0) |
| Prediction head (primary BCE objective) | Response correctness probability | Yes | Yes |

---
## 2. Definition of Experimental Conditions

### 2.1 Full Model (Interpretability‑Enabled)
The *full model* activates the complete interpretability stack:
- `use_mastery_head = True`
- `use_gain_head = True`
- `enhanced_constraints = True`
- Constraint weights resolve (unless explicitly overridden) to tuned non‑zero values (example preset used in current sweeps):
  - `non_negative_loss_weight = 0.0` (architecturally enforced)
  - `monotonicity_loss_weight = 0.1`
  - `mastery_performance_loss_weight = 0.8`
  - `gain_performance_loss_weight = 0.8`
  - `sparsity_loss_weight = 0.2`
  - `consistency_loss_weight = 0.3`
- `interpretability_loss` (sum of active components) is added to the primary BCE loss.
- Outputs include: `predictions`, `projected_mastery`, `projected_gains` (plus internal states in monitoring mode).

### 2.2 Heads‑Disabled Baseline (Pure Predictive Backbone)
This baseline removes all interpretability-specific capacity:
- `use_mastery_head = False`
- `use_gain_head = False`
- `enhanced_constraints = False` → forces all constraint weights to `0.0`.
- No auxiliary computation: `interpretability_loss = 0` (returned as a zero tensor for gradient safety).
- No projected mastery/gains in outputs; cumulative mastery loop & gain ReLU are bypassed.
- Sole objective: BCE over predicted response correctness.

### 2.3 Recommended Optional Intermediate Ablations (If Needed)
| Variant | Motivation | Settings |
|---------|------------|----------|
| Projections w/o Constraints | Isolate architectural inductive bias of the heads (capacity + cumulative mechanism) separate from loss shaping. | Heads ON, `enhanced_constraints = False`, all weights 0.0 |
| Minimal Constraints | Test necessity of richer constraint suite. | Activate only monotonicity + sparsity (others 0.0) |
| Consistency Only | Isolate the newly added consistency term. | Only `consistency_loss_weight > 0` |
| Performance Correlation Only | Assess effect of aligning mastery/gain to outcome signals exclusively. | Only performance correlation weights > 0 |

---
## 3. Auxiliary Constraint Terms (Activated in Full Model)

| Term | Symbolic Intent | Operational Definition (High-Level) |
|------|-----------------|--------------------------------------|
| Non‑negative gains | Gains ≥ 0 | ReLU(gain_raw); residual penalty only if weight > 0 (currently weight often 0) |
| Monotonic mastery | Mastery[t] ≥ Mastery[t−1] | Architectural cumulative accumulation + penalty on decreases |
| Mastery–performance alignment | Correct answers ↔ higher mastery | Penalize low mastery on correct, high mastery on incorrect |
| Gain–performance alignment | Correct answers ↔ higher gains | Margin hinge: mean(gain_incorrect) − mean(gain_correct) + margin |
| Sparsity of irrelevant gains | Focused learning attribution | L1-like mean abs of gains for non‑queried concepts |
| Consistency (delta vs scaled gain) | Temporal coherence of mastery evolution | Mean abs(Δ mastery − α·gains) (α = 0.1) |

All terms sum linearly after individual weighting coefficients.

---

## 4. Evaluation Metrics

Primary:
- AUC (validation) — discrimination of correctness.
- Accuracy (validation) — complementary baseline reference.

Interpretability Diagnostics (Full Model only):
- Monotonicity violation rate (expected 0% under cumulative design; residuals test safety).
- Negative gain frequency (expected 0%).
- Bounds violation rate (mastery ∉ [0,1]).
- Mean correlation (mastery trajectory vs. performance) — coarse alignment.
- Mean correlation (gain trajectory vs. performance).
- Constraint loss share = interpretability_loss / total_batch_loss (averaged) — regularization intensity.

Optional Efficiency / Stability:
- Wall-clock epoch time (compare added overhead).
- Peak GPU memory.
- Gradient norm statistics (sanity for auxiliary scaling).

---

## 5. Experimental Protocol

| Step | Action | Notes |
|------|--------|-------|
| 1 | Fix random seed(s) | Ensures variance attribution; document seed used. |
| 2 | Train Heads‑Disabled baseline (N epochs or early stopping). | Collect best val AUC. |
| 3 | Train Full Model under identical scheduler & stopping criteria. | Report same metrics + diagnostics. |
| 4 | (Optional) Run intermediate ablations if effect attribution needed. | Keep epoch budget symmetric. |
| 5 | Summarize results in a markdown table with ΔAUC and % overhead. | Place in `paper/` and index via `INDEX.md`. |

Early stopping: Use validation AUC. Patience identical across variants. Scheduler (ReduceLROnPlateau) mode = `max` (AUC).

---

## 6. Reporting Template (Example)

```markdown
| Variant | AUC | Acc | ΔAUC vs. Heads-Off | Epoch of Best | Constraint Loss Share | Mono Viol. | Neg Gains | Mastery Corr | Gain Corr | Time/Epoch (s) |
|---------|-----|-----|--------------------|--------------|----------------------|------------|-----------|-------------|-----------|---------------|
| Heads-Off | 0.7258 | 0.68 | — | 3 | 0.000 | 0.0% | 0.0% | — | — | 22.4 |
| Full Model | 0.7260 | 0.68 | +0.0002 | 3 | 0.074 | 0.0% | 0.0% | 0.41 | 0.38 | 24.9 |
```

Interpret the ΔAUC in context of stochastic variance (include standard deviation over ≥3 seeds if difference <0.002).

---

## 7. Reproducibility Controls

| Control | Implementation |
|---------|----------------|
| Deterministic seed | Seed Torch / NumPy before data loaders & model init. |
| Logged resolved weights | Training script prints final constraint coefficients. |
| Model snapshot | Save `best_model.pth` + `model_config` + `train_history`. |
| Environment | Record Python, PyTorch versions (optionally via a lightweight env dump). |
| Exact args | Persist args dict in results JSON (already implemented). |

---

## 8. Criteria for Declaring Success

This section formalizes quantitative and qualitative thresholds used to claim that the interpretability-augmented GainAKT2Exp model is successful. A "success" determination requires meeting (or exceeding) ALL mandatory criteria and documenting any conditional criteria not yet satisfied.

### 8.1 Mandatory Core Performance Criteria
| Dimension | Metric / Test | Threshold | Rationale |
|----------|---------------|-----------|-----------|
| Predictive Parity (Neutrality) | Mean ΔAUC (Full − Heads-Off) over ≥3 seeds | abs(ΔAUC) ≤ 0.002 (unless positive) | Ensures interpretability does not degrade core predictive power. |
| Predictive Gain (Optional Bonus) | Mean ΔAUC (Full − Heads-Off) | > 0.002 and 95% CI excludes 0 | Indicates regularization or inductive bias benefit. |
| Stability | Std(AUC) across seeds | ≤ 0.004 | Avoids fragile dependence on random initialization. |
| Training Convergence | Best epoch / max epochs | ≤ 0.50 | Early stabilization reduces risk of overfitting constraints. |
| Failure Robustness | Divergence events (NaNs / inf gradients) | 0 | Shows constraints are numerically safe. |

### 8.2 Interpretability Quality Criteria
| Aspect | Metric | Threshold | Interpretation |
|--------|--------|-----------|----------------|
| Structural Validity | Monotonicity / Negative Gain / Bounds violation rates | 0% (deterministic) | Architecture + ReLU + accumulation effective. |
| Semantic Alignment (Mastery) | Mean mastery-performance correlation | ≥ 0.25 (moderate) | Mastery trajectory reflects performance trend. |
| Semantic Alignment (Gain) | Mean gain-performance correlation | ≥ 0.20 (allow slightly lower early) | Gains approximate incremental learning signals. |
| Temporal Coherence | Consistency residual (abs(Δmastery − α·gain)) normalized | < 0.35 (scaled) | Mastery changes largely explained by gains. |
| Sparsity Focus | Mean absolute irrelevant gain / mean relevant gain | < 0.40 | Gains concentrated on queried concepts. |

### 8.3 Regularization Balance Criteria
| Check | Metric | Threshold | Action if Violated |
|-------|--------|-----------|--------------------|
| Auxiliary Weight Dominance | (Interpretability loss) / (Total loss) averaged | ≤ 0.25 | Reduce large weights (0.8→0.5) or introduce warm-up. |
| Inactive Terms | Term contribution / total aux loss | Each active > 5% (except non-negative if weight 0) | Revisit weight scaling or formulation. |
| Over-Constraint Risk | Drop in train AUC relative to heads-off after warm-up | ≤ 0.01 absolute | Schedule constraints later or anneal upwards. |

### 8.4 Reproducibility & Traceability Criteria
| Requirement | Evidence |
|-------------|----------|
| Seeds logged | JSON bundles include `seed`. |
| Resolved weights logged | Training script prints final weight vector. |
| Config snapshot | `model_config` + optimizer state stored in checkpoint. |
| Environment trace (optional) | Python & PyTorch versions recorded (if added). |
| Determinism toggles | cudnn deterministic + manual seeds set. |

### 8.5 Efficiency Criteria
| Metric | Threshold | Notes |
|--------|-----------|-------|
| Wall-clock overhead (Full vs Heads-Off) | ≤ 15% per epoch | Acceptable trade-off for added explanatory value. |
| Parameter overhead | < 2% of total | Heads are small linear layers. |
| Memory overhead | ≤ 10% peak GPU memory | Monitor with nvidia-smi (manual). |

### 8.6 Ablation Clarity Criteria
1. Heads-Off vs Full (baseline vs interpretability) — ALWAYS present.
2. Architecture-Only (Heads ON, Constraints OFF) — Present if ΔAUC not clearly neutral OR correlations weak.
3. Per-Term Ablations (e.g., drop each constraint singly) — Required if any single term plausibly causes degradation (>0.003 ΔAUC loss when removed negative vs baseline).

### 8.7 Decision Matrix
| Scenario | Outcome | Next Action |
|----------|---------|-------------|
| Meets all mandatory + alignment ≥ targets | Declare success | Prepare paper tables. |
| Predictive neutral but alignment below thresholds | Tune weights / schedule; extend epochs | Re-test after adjustments. |
| Predictive degradation | Run architecture-only & single-term ablations | Isolate harmful term(s). |
| Alignment strong but overhead high | Profile & optimize (batch size, mixed precision) | Keep interpretability if AUC neutral. |

### 8.8 Rationale Summary
These criteria balance three pillars: (1) Performance neutrality (or improvement), (2) Demonstrable interpretability leverage (structural + semantic), (3) Practical viability (reproducibility + efficiency). Only when all three align does the interpretability layer justify inclusion in a benchmarked research contribution.

---

## 9. Interpretation Guidelines

1. If AUC improves with negligible overhead, interpretability adds *regularization benefit*.
2. If AUC is unchanged (±0.001), interpretability is *performance-neutral* and defensible as added value.
3. If AUC drops beyond noise (>0.003), conduct per-term ablations to identify degrading constraint(s).
4. Positive mastery / gain correlations > 0.3 indicate meaningful alignment with observed performance sequences.
5. Constraint loss share > ~0.15: verify that predictive loss is not being dominated (risk of over-regularization).

---

## 10. Known Limitations / Future Refinements
- Current cumulative mastery scaling factor (0.1) is fixed; adaptive scaling could better calibrate gain-to-mastery translation.
- Correlations are coarse (sequence-level means). Future: per-concept conditional calibration curves.
- Consistency loss treats all temporal steps uniformly; weighting by recency or uncertainty might improve robustness.
- Sparsity penalty is globally averaged; a group-lasso variant might yield sharper per-concept sparsity.

---

## 11. Summary
The *full model* couples architectural inductive bias (monotonic cumulative mastery + non‑negative gains) with explicit semantic constraints to improve interpretability and potentially stabilize learning. The *heads‑disabled baseline* isolates the predictive backbone, anchoring claims about added interpretability against a fair reference. This layered evaluation protocol supports transparent reporting for the forthcoming paper and aligns with reproducibility standards.

---

## 12. Quick Baseline vs. Full Model Diagnostic Run

### 12.1 How to Run
Short diagnostic (3 epochs) on ASSIST2015 with a fixed seed:

```bash
python tmp/run_gainakt2exp_baseline_compare.py --dataset assist2015 --epochs 3 --batch_size 96 --seed 4
```

What this does:
1. Constructs two configurations programmatically:
  - Heads-Off (pure predictive backbone): no mastery/gain heads; constraints disabled.
  - Full Model: mastery & gain heads enabled; constraint weights set to tuned preset.
2. Runs each for the specified number of epochs with identical optimizer, seed, and data split.
3. Collects best validation AUC, consistency diagnostics (neutral zeros for heads-off), and writes:
  - `tmp/gainakt2exp_baseline_compare_<timestamp>.md` (markdown table)
  - `tmp/gainakt2exp_baseline_compare_<timestamp>.json` (structured results)

### 12.2 Example Result (Seed=42, Epochs=3)
| Variant | Best Val AUC | ΔAUC vs Heads-Off | Mastery Corr | Gain Corr | Monotonicity Viol. | Neg Gain Rate | Bounds Viol. |
|---------|--------------|-------------------|--------------|-----------|--------------------|---------------|--------------|
| Heads-Off | 0.7258 | —        | 0.000 | 0.000 | 0.0% | 0.0% | 0.0% |
| Full Model | 0.7260 | +0.0002 | 0.013 | -0.008 | 0.0% | 0.0% | 0.0% |

Interpretation (early, 3 epochs):
- ΔAUC (+0.0002) is negligible—performance-neutral interpretability stack.
- Correlations near zero: expected for such a short horizon; alignment signals often emerge later.
- Perfect structural consistency (no monotonicity, gain sign, or bounds violations) validates architectural enforcement.

### 12.3 Recommended Next Steps
| Objective | Action | Rationale |
|-----------|--------|-----------|
| Robust performance claim | Run ≥10 epochs, seeds {21,42,84}; report mean ± std AUC | Distinguish real effect from variance |
| Architectural vs. regularization disentanglement | Add third variant: heads ON + constraints OFF | Measures inductive bias of projections alone |
| Constraint contribution clarity | Log each constraint term separately per epoch | Detect dominating or inactive terms |
| Correlation maturation | Plot mastery/gain correlations over epochs | Verify monotonic emergence or plateau |
| Regularization calibration | If correlations stay low, reduce large weights (0.8→0.4) or schedule warm-up | Prevent early suppression of predictive features |
| Efficiency audit | Record wall-clock per epoch & memory | Quantify cost of interpretability for paper |

### 12.4 Reporting Guidance
For paper tables:
- Present Heads-Off vs Full (and optionally Architecture-Only) with multi-seed mean ± std.
- Include correlation metrics only for variants where heads are enabled; mark N/A otherwise.
- Note if ΔAUC < 0.002 and correlations are positive—that supports a “no performance penalty” claim.

---
**Document version:** 2025-10-16
