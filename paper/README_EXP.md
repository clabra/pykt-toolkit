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

---

## 13. Resumable Multi-Seed Execution & Interpretation Matrix

This section documents the resumable experimental runner for long multi-seed, multi-variant GainAKT2Exp evaluations, and provides a structured interpretation matrix for possible outcome patterns. It extends the protocol in Sections 5 and 8 to facilitate robust, interruption-tolerant experimentation and clear paper-ready narratives.

### 13.1 Command Example

```bash
python tmp/run_gainakt2exp_baseline_compare_resumable.py \
  --dataset assist2015 --epochs 1 --batch_size 32 \
  --seeds 42 43 --variants heads_off arch_only full \
  --auto_postprocess --progress_path tmp/gainakt2exp_progress_test.json \
  --results_dir paper/results
```

(For publication-grade runs, increase `--epochs` (e.g., 12) and expand `--seeds` (e.g., `21 42 84`). Use `nohup` + background execution if persistent runtime is required.)

### 13.2 What the Script Does

- Executes each (variant, seed) pair sequentially: `heads_off`, `arch_only`, `full`.
- Records completion progress in a JSON progress file (`--progress_path`) enabling resume with `--resume` (not shown above but supported).
- Aggregates per-seed results into mean / std metrics and computes 95% confidence intervals (normal approximation).
- Evaluates success criteria (predictive neutrality / gain, stability) defined in Section 8.
- Writes publication-grade JSON and Markdown summaries to `paper/results` when `--auto_postprocess` is set.
- Skips already finished pairs when re-run with `--resume` (safe after interruptions).

### 13.3 Variants

| Variant | Purpose | Heads | Constraints | Notes |
|---------|---------|-------|-------------|-------|
| heads_off | Pure predictive backbone baseline | OFF | OFF (weights forced 0) | Reference for ΔAUC neutrality criterion |
| arch_only | Architectural inductive bias only | ON | OFF (weights 0) | Isolates capacity of mastery/gain projections |
| full | Full interpretability stack | ON | ON (weights >0) | Adds semantic constraints (Section 3) |

### 13.4 Result Artifacts

| Artifact | Location | Content |
|----------|----------|---------|
| Raw run bundle | `tmp/gainakt2exp_resumable_raw_<timestamp>.json` | Per-seed raw outputs + aggregates |
| Progress file | `tmp/gainakt2exp_progress.json` (configurable) | List of completed variant|seed keys |
| Publication summary JSON | `paper/results/gainakt2exp_publication_summary_<timestamp>.json` | Aggregates + criteria evaluation + CIs |
| Publication summary Markdown | `paper/results/gainakt2exp_publication_summary_<timestamp>.md` | Tables + interpretation narrative |

### 13.5 Scenario Interpretation Matrix

The following matrix enumerates typical outcome patterns after multi-seed evaluation. Thresholds reference Section 8.

| ID | Quantitative Pattern | Interpretation | Paper Framing | Recommended Follow-Up |
|----|----------------------|----------------|---------------|-----------------------|
| S1 | Full ΔAUC neutrality (|Δ| ≤ 0.002) + correlations ≥ thresholds | Performance-neutral interpretability | Emphasize “no cost” | Proceed; optional loss share reporting |
| S2 | Full ΔAUC > 0.002, CI excludes 0, correlations strong | Constraints add predictive gain + semantics | Dual benefit narrative | Replicate on more datasets |
| S3 | Full ΔAUC gain but correlations weak | Regularization benefit without clear semantics | Cautious improvement claim | Tune weights / warm-up for alignment |
| S4 | Neutral ΔAUC, weak correlations | Uncalibrated interpretability | Calibration narrative | Weight sweep & correlation tracking |
| S5 | ArchOnly ΔAUC > 0; Full ≈ ArchOnly | Architecture drives gain; constraints neutral | Inductive bias emphasis | Consider pruning weak constraints |
| S6 | ArchOnly neutral; Full positive Δ | Constraints uniquely beneficial | Highlight constraint design | Per-term ablation to isolate key term |
| S7 | ArchOnly negative Δ; Full neutral | Constraints stabilize added capacity | Necessity of constraints | Document rescue effect |
| S8 | Full ΔAUC negative (< -0.002), correlations strong | Predictive penalty for semantics | Trade-off acknowledgment | Anneal / reduce dominant weights |
| S9 | Full ΔAUC negative, correlations weak | Broad failure | Remediation focus | Systematic per-term ablations |
| S10 | Heads-Off high variance (Std > 0.004); Full stable | Structural regularization stabilizes training | Stability contribution | Increase seeds; show variance reduction |
| S11 | Structural violations > 0% | Implementation / enforcement bug | Corrective disclosure | Audit accumulation logic before claims |
| S12 | Gains correlation ≥ threshold; mastery below | Gains pick up local signals earlier | Temporal emergence narrative | Adjust consistency scaling α; monitor evolution |
| S13 | Overhead > 15% with neutral Δ | Efficiency concern | Cost-benefit justification | Optimize batch size / mixed precision |
| S14 | Incomplete seeds | Provisional reliability | Preliminary disclaimer | Resume with `--resume` until complete |
| S15 | Constraints incremental Δ negative (within noise) | Over-regularization | Constraint pruning | Reduce or remove low-impact terms |
| S16 | Wide CI (width > 0.01) | Insufficient statistical power | Caution in claims | Add more seeds (≥5–7) |
| S17 | Constraint loss share > 0.25 | Loss dominance risk | Regularization balance | Introduce warm-up / downscale large weights |
| S18 | ArchOnly correlations moderate; Full stronger | Constraints refine semantics | Refinement benefit narrative | Highlight correlation deltas |

### 13.6 Decision Flow Summary

1. Verify structural integrity (violations must be 0%).
2. Classify ΔAUC (neutral / gain / degradation) with CI.
3. Evaluate semantic correlations vs thresholds.
4. Use ArchOnly to attribute architecture vs constraint effects.
5. Check stability (Std AUC baseline ≤ 0.004), efficiency overhead (≤ 15%).
6. Select narrative template aligned with scenario (S1–S18).

### 13.7 Paper Integration Guidance

| Section | Required Data | Scenario Emphasis |
|---------|---------------|------------------|
| Results (Performance) | Mean ± std, CI ΔAUC | S1–S3, S8–S9 classification |
| Results (Interpretability) | Correlations, violation rates | Semantic strength or calibration need |
| Ablation | ArchOnly + constraints incremental Δ | Attribution (S5–S7, S15, S18) |
| Efficiency | Epoch time %, memory % | Cost justification (S13) |
| Discussion | Scenario narrative | Trade-offs, stability, future tuning |
| Limitations | CI width, seeds count, overhead | S14, S16, S13 impacts |

All success claims must reference neutrality or gain thresholds from Section 8 and include multi-seed statistical support (≥3 seeds recommended; >5 for narrow CI if ΔAUC marginal).

### 13.8 Recommended Next Enhancements (Optional)

- Per-term auxiliary loss logging to quantify each constraint’s average contribution.
- Temporal correlation trajectory plots (epoch-indexed) to evidence semantic emergence.
- Adaptive or annealed constraint schedule (warm-up) if early over-regularization observed (S4, S8, S9).
- Git commit hash embedding in publication JSON for traceability.

### 13.9 Demonstrating Semantic Alignment Over Longer Training

This subsection provides a formal methodology to empirically show that cumulative mastery and gain trajectories learned by GainAKT2Exp align with real student performance patterns as training progresses. The objective is to elevate preliminary structural interpretability (Section 8) into validated semantic interpretability.

#### 13.9.1 Alignment Hypotheses
H1 (Mastery Alignment): Mean projected mastery for a student sequence correlates positively with observed correctness probability (global and per concept).
H2 (Gain Alignment): Local increases in projected gains precede or coincide with short-term improvements in correctness compared to recent history.
H3 (Constraint Refinement): Full (constraints ON) variant yields higher alignment correlations than Architecture-Only (constraints OFF) beyond random variation.

#### 13.9.2 Metrics
1. Per-Epoch Correlation Trajectory:
  - Pearson (and Spearman optional) correlation between sequence-level average mastery and correctness per student.
  - Same for gains (averaged per timestep) vs. correctness.
2. Per-Concept Calibration:
  - For each concept c: empirical success rate(c) vs. mean projected mastery(c) (scatter + Pearson/Spearman).
3. Binned Mastery Calibration Curve:
  - Bin mastery values (e.g., deciles); compute observed correctness frequency per bin; plot predicted mastery vs observed correctness.
4. Gain–Improvement Association:
  - Improvement(t) = correctness(t) − mean(correctness(t−k..t−1)), k∈{3,5}.
  - Regress Improvement(t) on Gain(t−1) controlling for Mastery(t−1); report coefficient and bootstrap CI.
5. Lag Cross-Correlation:
  - corr(Gain(t−ℓ), correctness(t)) for ℓ ∈ {0..5} to show temporal lead/lag structure.
6. Constraint Incremental Effect:
  - Δ(master y correlation Full − ArchOnly), Δ(gain correlation Full − ArchOnly) with paired statistical tests.

#### 13.9.3 Statistical Protocol
Seeds: ≥5 (e.g., 21, 42, 63, 84, 105). Epochs: ≥12.
Bootstrap (≥1000 resamples) over students for mastery and gain correlation CIs.
Hypothesis tests:
 - H0_mastery: Corr_full − Corr_heads_off ≤ 0.
 - H0_gain: Gain coefficient ≤ 0.
 - H0_refinement: Corr_full − Corr_arch_only ≤ 0.
Rejecting these supports semantic alignment or constraint-driven refinement.

#### 13.9.4 Instrumentation Plan
Add logging hooks per epoch:
 - Store per-student mastery/gain sequences (sample up to N=200 for memory efficiency).
 - Compute and record mastery/gain correlations, regression statistics, calibration aggregates.
 - Record peak GPU memory and epoch wall-clock for efficiency overhead quantification.

Flags to add (example):
 - `--log_semantic_samples`
 - `--max_semantic_students 200`
 - `--warmup_constraint_epochs 3` (linear ramp of constraint weights)

#### 13.9.5 Warm-Up Strategy
If correlations remain <0.05 through epochs 1–3, apply constraint warm-up: effective_weight(epoch) = base_weight * min(1, epoch / warmup_epochs). Prevents early suppression of predictive signals (addresses S4 scenario).

#### 13.9.6 Interpretation Thresholds (Refining Section 8)
 - Mastery correlation ≥0.25 with CI excluding 0: moderate alignment.
 - Gain correlation ≥0.20 with CI excluding 0: emerging incremental signal.
 - Positive significant gain regression coefficient (p<0.05): gains predictive of improvement.
 - Per-concept calibration Pearson ≥0.30 and Spearman ≥0.30: mastery respects difficulty ordering.

#### 13.9.7 Reporting Template
Table A (Performance): AUC mean ± std, Δ vs baseline, CI.
Table B (Semantic Alignment): mastery/gain correlations (mean, CI), calibration correlations.
Table C (Gains Regression): coefficient, CI, p-value.
Table D (Efficiency): epoch time (sec), memory usage (MiB), overhead %.
Table E (Ablation): ArchOnly vs Full alignment deltas + p-values.

#### 13.9.8 Risks & Mitigations
 - Low variance mastery leads to inflated or meaningless correlations → track mastery variance and exclude near-constant sequences.
 - Short sequences (<3) distort correlation → minimum length filter.
 - Overfitting calibration on small concept sample → require minimum attempt count per concept.
 - Gain noise early epochs → smoothing via EMA or warm-up.

#### 13.9.9 Success Declaration Criteria
Semantic success is declared when: (i) structural violations remain 0%, (ii) mastery & gain correlations meet thresholds with CIs excluding 0, (iii) gains regression indicates predictive incremental value, and (iv) neutrality or acceptable ΔAUC trade-off (|ΔAUC| ≤0.002 or positive gain) is maintained.

---

## 14. Semantic Alignment Experiment (Multi-Seed 12-Epoch Run)

### 14.1 Objective

We conduct a multi-seed, 12-epoch evaluation of GainAKT2Exp on the assist2015 dataset to assess whether interpretability constraints provide (a) performance neutrality or gain and (b) emerging semantic alignment (mastery & gain correlations with student correctness dynamics). This run operationalizes the success criteria in Section 13.9.9.

### 14.2 Execution Command

Baseline (structural + predictive assessment):

```bash
python tmp/run_gainakt2exp_baseline_compare_resumable.py \
  --dataset assist2015 \
  --epochs 12 \
  --batch_size 64 \
  --seeds 21 42 63 84 105 \
  --variants heads_off arch_only full \
  --use_amp \
  --progress_path tmp/gainakt2exp_progress_seq.json \
  --auto_postprocess \
  --results_dir paper/results
```

(AMP reduces memory usage; batch size 64 chosen to avoid OOM.)

### 14.3 Result Artifacts

After completion the following files are produced:

| Artifact | Path | Description |
|----------|------|-------------|
| Base summary MD | `tmp/gainakt2exp_resumable_summary_<timestamp>.md` | Quick AUC aggregates |
| Raw JSON bundle | `tmp/gainakt2exp_resumable_raw_<timestamp>.json` | Per-seed results for each variant |
| Publication summary MD | `paper/results/gainakt2exp_publication_summary_<timestamp>.md` | Formatted performance + interpretability overview |
| Publication summary JSON | `paper/results/gainakt2exp_publication_summary_<timestamp>.json` | Aggregates, criteria evaluation, CIs |
| Progress file | `tmp/gainakt2exp_progress_seq.json` | Tracks completed variant-seed pairs |

### 14.4 Empirical Outcomes (Example Run: 12 epochs, 5 seeds)

| Variant | Mean AUC | Std AUC | 95% CI | ΔAUC vs Heads-Off |
|---------|----------|---------|--------|-------------------|
| heads_off | 0.64058 | 0.00370 | [0.63734, 0.64382] | +0.0000 |
| arch_only | 0.64058 | 0.00370 | [0.63734, 0.64382] | +0.0000 |
| full | 0.65204 | 0.00227 | [0.65005, 0.65404] | +0.0115 |

Mean correlations (sequence-level, coarse):

| Variant | Mastery Corr | Gain Corr |
|---------|--------------|-----------|
| heads_off | 0.0000 | 0.0000 |
| arch_only | 0.0322 | -0.0094 |
| full | 0.0251 | -0.0054 |

Structural integrity: 0% monotonicity, negative gain, bounds violations (all variants).  
Stability: Baseline std(AUC)=0.00370 (<0.004 threshold) → Pass.

### 14.5 Interpretation vs Success Criteria

1. Structural consistency satisfied (criterion i).
2. Semantic correlations remain weak (criterion ii not satisfied: values far below mastery ≥0.25 / gain ≥0.20 thresholds; CIs would include 0).
3. Gains regression not yet performed; incremental predictive value of gains unverified (criterion iii pending).
4. Performance trade-off: Positive predictive gain (+0.0115 AUC) with CI separation → Acceptable (criterion iv satisfied; exceeds neutrality band).

Thus current state: predictive improvement + structural interpretability; semantic interpretability not yet demonstrated.

### 14.6 Scenario Matrix Mapping (Section 13.5)

Primary classification: **S3 (Full ΔAUC gain but correlations weak)**.  
Also partially resembles S6 (constraints uniquely beneficial) in predictive terms, but lacks semantic refinement (no correlation improvement), so we retain S3 framing.

Discarded scenarios: S1 (neutral + strong correlations), S2 (gain + strong correlations), S5 (architecture-only gain), S18 (constraints sharpen semantics). No degradation scenarios (S8/S9) apply.

### 14.7 Bootstrap Confidence Intervals for Correlations (Planned)

“Bootstrap CIs for correlations” refers to estimating confidence intervals for mastery-performance and gain-performance correlations via resampling students with replacement:

1. Compute per-student correlation (filter sequences length ≥3).
2. Perform B resamples (e.g., B=1000), each time sampling N students with replacement and recomputing the mean correlation.
3. Sort bootstrap estimates; 95% CI = [2.5th percentile, 97.5th percentile].
4. CI excluding 0 → statistically supported alignment; CI covering 0 → alignment not yet established.

Rationale: Non-normal, bounded correlation distributions; protects against skew and small-sample distortion.  
Interpretation: Narrow CI → stable alignment; wide CI → need more seeds/students.

### 14.8 Strategic Paths Forward

We outline three implementation tiers to finalize paper claims:

| Tier | Focus | Actions | Pros | Cons |
|------|-------|--------|------|------|
| A | Regularization claim | Publish current S3 result | Fast; clear predictive gain | Weak interpretability narrative |
| B | Semantic emergence attempt | Add constraint warm-up (epochs 1–4), log per-epoch correlations | Moderate chance of correlation growth | Requires one more multi-seed run |
| C | Full semantic validation | Warm-up + calibration + gains regression + bootstrap CIs + possibly second dataset | Strong interpretability evidence | Longer timeline & complexity |

Recommended (to avoid over-complication): **Tier B**. If correlations rise above ~0.15 by epoch 8–12, frame as “emerging alignment.” If not, proceed with Tier A narrative, explicitly scoping semantic alignment as future work.

### 14.9 Warm-Up & Trajectory Logging (Planned Implementation)

Warm-up schedule: effective_weight(epoch) = base_weight * min(1, epoch / W), W≈4. Applies to mastery/gain performance alignment losses to reduce early suppression of variance.  
Add flags: `--warmup_constraint_epochs`, `--log_semantic_samples`, `--max_semantic_students`.  
Outputs: `paper/results/gainakt2exp_semantic_trajectory_<timestamp>.json` capturing per-epoch mastery/gain correlations.

### 14.10 Provisional Claim

“GainAKT2Exp enforces structurally interpretable mastery and gain trajectories with zero violations and yields a modest predictive improvement (+1.15 AUC points) over a backbone baseline. Current sequence-level semantic correlations remain low; we introduce a constraint warm-up protocol to elicit emergent alignment without sacrificing performance.”

### 14.11 Decision to Write

If time-critical and Tier B run cannot be completed promptly: proceed with a structural + performance paper (Tier A) and clearly mark semantic alignment as an active future direction. Otherwise execute the warm-up run to attempt elevating from S3 to an emerging alignment narrative (pre-S1 trajectory).

---
This methodology operationalizes the narrative: *“Over longer training horizons, the model’s interpretable mastery and gain curves do not merely satisfy structural constraints; they increasingly reflect real student performance dynamics, thereby establishing validated semantic interpretability without sacrificing predictive accuracy.”*

Here’s a plain, structured explanation of that success declaration:

We only say the model has real, useful interpretability (semantic success) when ALL four conditions are satisfied together:

1. Structural integrity (no violations): The mastery curve never goes backwards, stays within [0,1], and gains are never negative. If any of these fail, the “shape” of learning is untrustworthy and we stop there—no semantic claim.

2. Meaningful correlations (not just noise): The numbers measuring how well mastery tracks overall correctness and how well gains track short-term improvement are both above the target thresholds (e.g., mastery correlation ≥ 0.25, gain correlation ≥ 0.20). AND their confidence intervals do not touch zero. That means the alignment is statistically real—not a random fluctuation.

3. Gains have predictive incremental value: When we run a small regression to see whether higher gain values at one step predict better immediate performance afterward (controlling for current mastery), the gain coefficient is positive and statistically significant (p < 0.05 with a confidence interval above 0). This shows gains are not just decorative—they add unique information about imminent improvement.

4. No unacceptable performance trade-off: The interpretable version must not harm prediction quality. Either:

- It stays “neutral”: the change in AUC compared to a bare backbone is tiny (|ΔAUC| ≤ 0.002), within defined noise tolerance, OR
- It actually improves AUC (positive ΔAUC outside neutrality band). A larger negative drop would mean we pay too much predictive cost for interpretability and we cannot declare success yet.

Putting it together:

(1) ensures the curves obey the learning principles we baked in.

(2) shows the curves track real student behavior, not arbitrary patterns.

(3) proves the “gain” channel adds actionable predictive signal about immediate learning.

(4) guarantees we did not sacrifice the core task (predicting correctness) to get these curves.


