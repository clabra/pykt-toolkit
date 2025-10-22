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

---

#### 13.9.5 Warm-Up Strategy
If correlations remain <0.05 through epochs 1–3, apply constraint warm-up: effective_weight(epoch) = base_weight * min(1, epoch / warmup_epochs). Prevents early suppression of predictive signals.

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

### 14.7 Bootstrap CI (Single Seed Snapshot Limitation)
Current bootstrap file (seed105 only) produced trivial CIs (single value) and is not representative. Need multi-file aggregation including all seeds for meaningful CIs. Next action: extend bootstrap script to ingest multiple per-seed final global correlations instead of a single matching glob pattern.

Planned fix: rerun bootstrap with pattern `gainakt2exp_results_full_seed*_20251018_07*.json` capturing all seed JSONs sharing final timestamp window, or explicitly enumerate file list.

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

## 15. Implemented Tier B to try semantic emergence upgrades

### Approach 

1. Added warm-up scaling for mastery/gain performance-alignment losses (epochs ramp to full weight).
2. Added epoch-level semantic trajectory logging (mastery & gain correlations + applied warm-up scale).
3. Introduced new args: `warmup_constraint_epochs`, `max_semantic_students`, `semantic_trajectory_path`.
4. Persisted semantic trajectory both inside final results and a standalone JSON file.
5. Repaired earlier patch corruption; restored clean training loop and added warm-up before batch loop.
6. Fixed all syntax issues; file now lint/error free.

Next optional step: propagate these new args into the multi-seed runner (add flags and pass through) so Tier B runs can be automated across variants. Let me know if you want that integrated now.

### Command
Command below launches a Tier B 12-epoch multi-seed experiment with warm-up and semantic correlation sampling. First run (heads_off seed21) is underway on CUDA with AMP; semantic trajectory file will be written to `paper/results/gainakt2exp_semantic_trajectory_heads_off_seed21_20251017_154306.json` after completion of that seed: 

```
python tmp/run_gainakt2exp_baseline_compare_resumable.py --dataset assist2015 --epochs 12 --batch_size 64 --seeds 21 42 63 84 105 --variants heads_off arch_only full --use_amp --auto_postprocess --results_dir paper/results --warmup_constraint_epochs 4 --max_semantic_students 50 --devices 0 1 2 3 4 --max_workers 5 --resume
```

For parallel GPU usage, we use --devices and --max_workers.

After final run, collect:
- Publication summary JSON & MD (paper/results/gainakt2exp_publication_summary_*.{json,md}).
- All semantic trajectory JSON files produced in results.
- Aggregate AUC means/std, CI, and warm-up correlation trajectory (for full variant).
- Provide a concise interpretation: warm-up effect on correlations, whether semantic emergence improved vs earlier run, scenario matrix reclassification.


## 16. Alignment Loss Pilot (Tier B Initial Attempt)

### 16.1 Objective
Assess whether a batch-level correlation alignment auxiliary loss (mastery & gains vs correctness) with 8-epoch warm-up improves final global semantic correlations while preserving validation AUC neutrality/gain.

### 16.2 Pilot Configuration
Two-seed, 8-epoch run (assist2015):

```bash
python tmp/run_full_alignment_pilot.py \
  --seeds 21 42 \
  --epochs 8 \
  --batch_size 64 \
  --dataset assist2015 \
  --alignment_weight 0.1 \
  --alignment_warmup_epochs 8 \
  --alignment_min_correlation 0.05
```

Alignment active only for full variant; adaptive alignment enabled.

### 16.3 Results Summary
| Seed | Best Val AUC | Best Epoch | Final Mastery Corr | Final Gain Corr | Batch-Level Alignment Corr (Mastery) | Observation |
|------|--------------|-----------|--------------------|----------------|--------------------------------------|-------------|
| 21   | 0.6583       | 1         | 0.0253             | 0.0187         | ~0.060–0.065 plateau                 | Early AUC peak; correlations flat |
| 42   | 0.6541       | 1         | 0.0232             | 0.0179         | ~0.061–0.066 plateau                 | Similar pattern |

Predictive performance sustained; semantic correlations remain near noise (global consistency correlations ≈0.02–0.025).

### 16.4 Diagnosis
1. Locality: Alignment computed only on mini-batch; global sequence-level semantics not reinforced.
2. Insufficient Weight: Base 0.1 with full warm-up yields limited gradient influence relative to other constraint weights (0.8 each for performance alignment terms).
3. Metric Gap: Batch correlation does not directly optimize sequence-aggregated global consistency correlation metric.
4. Early Convergence: Best AUC at epoch 1 reduces representational pressure for subsequent semantic reshaping.

### 16.5 Planned Refinements
| Refinement | Description | Expected Benefit |
|-----------|-------------|------------------|
| Global Validation Alignment Pass | Post-epoch correlation on stratified validation sample; adaptive weight based on global corr | Aligns optimization target with evaluation metric |
| Weight Escalation | Increase alignment_weight to 0.25 (warm-up to epoch 8) | Stronger semantic gradient signal |
| Constraint Rebalancing | Temporarily reduce performance-alignment weights (0.8→0.5) during warm-up | Preserve variance for correlation growth |
| Residual Correlation | Correlate gains with correctness residual (delta vs short-term mean) | Targets incremental learning signal |
| Adaptive Consistency Scaling | Downscale consistency_loss_weight by 0.1 if early correlations very low | Frees capacity for semantic shaping |
| Variance Guard | Skip alignment update when mastery variance < 1e-4 | Avoid noisy correlation gradients |

### 16.6 Emerging Alignment Targets
| Metric (Epoch 12) | Emerging Threshold | Rationale |
|------------------|-------------------|-----------|
| Mastery Corr | ≥ 0.10 (CI excl. 0) | Early semantic emergence |
| Gain Corr | ≥ 0.06 (CI excl. 0) | Initial incremental signal |
| Alignment Loss Share | ≤ 0.20 total loss | Prevent dominance |
| ΔAUC vs Heads-Off | ≥ 0.000 | Neutral or positive performance |

### 16.7 Next Steps
1. Patch training loop: add global alignment sampling & residual correlation option (flags: `--enable_global_alignment_pass`, `--alignment_global_students`, `--use_residual_alignment`).
2. Extend adaptive weighting to use global correlation outcomes.
3. Re-run full variant seeds {21, 42, 63, 84, 105} (skip retraining baselines) with updated alignment design.
4. Compute bootstrap CIs for correlations; integrate into publication summary and Section 14 tables.

### 16.8 Documentation Plan
Upon completion, revise Sections 14.4–14.6 with updated correlation metrics and scenario reclassification (aiming to progress from S3 toward S1). Append an alignment formulation appendix for reproducibility.

### 16.9 Provisional Statement
“Local batch correlation alignment preserved predictive improvement but did not elevate global semantic correlations. We will introduce global validation sampling, residual correctness correlation, and adaptive reweighting to stimulate emerging semantic alignment while maintaining performance neutrality.”
### 16.10 Alignment-Enabled Multi-Seed Run Results (12 epochs, assist2015, seeds 21/42/63/84/105)

| Seed | Best Val AUC | Final Mastery Corr | Final Gain Corr | Peak Mastery Corr (Epoch) | Peak Negative Gain Corr Phase | Notes |
|------|--------------|--------------------|-----------------|---------------------------|-------------------------------|-------|
| 21   | 0.7173       | 0.0766             | 0.0339          | 0.1497 (E6)               | Early epochs gains <0 then recover | Strong AUC; mastery declines slightly late |
| 42   | 0.6922       | 0.0904             | 0.0137          | 0.1497 (E8)               | Gains negative mid-run        | Lower AUC outlier seed |
| 63   | 0.6952       | 0.0841             | 0.0175          | 0.1507 (E7)               | Gains negative mid-run        | Mid-tier performance |
| 84   | 0.7066       | 0.0843             | 0.0194          | 0.1510 (E7)               | Gains negative mid-run        | Stable trajectory |
| 105  | 0.7172       | 0.0772             | 0.0358          | 0.1501 (E6)               | Gains negative mid-run        | Mirrors seed 21 |

Aggregate (full variant only):
- Mean AUC = 0.7057 (Std 0.0106, 95% CI [0.6965, 0.7150])
- Mean final mastery correlation = 0.0825 (below emerging threshold 0.10)
- Mean final gain correlation = 0.0241 (well below emerging threshold 0.06)
- Structural violations: 0% (monotonicity, sign, bounds) for all seeds.

Temporal behavior:
- Mastery correlations peak around epochs 6–8 (~0.15) before decaying to ~0.12–0.08 by epoch 12 (regularization drag / over‑smoothing late).
- Gain correlations start slightly positive (some seeds) then dip negative during constraint ramp (epochs 4–8), partially recovering but not exceeding ~0.035 at end.
- Global alignment correlations remain substantially lower than batch alignment_corr_gain (~0.18–0.22), indicating local correlation does not propagate to global sequence-level semantics.

Scenario Classification (Updated):
- Still S3 (Full predictive gain but weak correlations) — semantic emergence not yet achieved.
- Alignment modifications improved stability of gain positivity late (recovery to >0) but insufficient magnitude.

### 16.11 Bootstrap CI (Single Seed Snapshot Limitation)
Current bootstrap file (seed105 only) produced trivial CIs (single value) and is not representative. Need multi-file aggregation including all seeds for meaningful CIs. Next action: extend bootstrap script to ingest multiple per-seed final global correlations instead of a single matching glob pattern.

Planned fix: rerun bootstrap with pattern `gainakt2exp_results_full_seed*_20251018_07*.json` capturing all seed JSONs sharing final timestamp window, or explicitly enumerate file list.

### 16.12 Diagnosis vs Emerging Targets
| Metric | Observed | Target | Status |
|--------|----------|--------|--------|
| Final Mastery Corr (mean) | 0.0825 | ≥0.10 | Not met |
| Final Gain Corr (mean) | 0.0241 | ≥0.06 | Not met (improved) |
| Peak Mastery Corr (transient) | ~0.15 | ≥0.10 | Met transiently (not retained) |
| Gain Corr Recovery (final positivity) | Mild | Sustained ≥0.06 | Not met |
| Performance Δ vs Baseline | Positive (needs baseline reference) | ≥0.000 | Presumed met (prior baseline 0.6406) |

Interpretation: Constraints and alignment elevate transient mastery correlation early but lack retention. Gain alignment loss improves local batch correlations, yet global semantic translation remains shallow—suggesting mismatch in optimization target vs evaluation metric and potential late over-regularization (consistency + performance alignment weights).

### 16.13 Proposed Iteration Path
1. Retention Strategy: Introduce late-epoch correlation preservation term (epoch ≥ warm-up) penalizing decay: L_ret = max(0, corr_peak - corr_current - δ).
2. Dynamic Rebalancing: Reduce consistency_loss_weight from 0.3 → 0.2 after epoch 8 if mastery_corr < 0.10.
3. Performance Alignment Anneal: Implement cosine schedule for mastery/gain performance loss weights instead of fixed post warm-up plateau (prevents continued shrinkage of variance).
4. Cross-Lag Gain Objective: Add auxiliary lag correlation term Corr(Gain_t, Correctness_{t+1..t+2}) to promote predictive gains semantics.
5. Multi-Seed Global Sampling: Increase alignment_global_students from 300 → 600 (stratified by sequence length deciles) for richer global signal.
6. Variance Floor Logging: Track mastery variance per epoch; if variance < threshold for >3 consecutive epochs, auto reduce sparsity_loss_weight by 50%.

### 16.14 Decision
Given time constraints, we either:
- Publish Tier A + partial Tier B attempt (documenting transient peak semantics not retained) and position retention + lag alignment as future work, OR
- Execute one more refinement cycle focusing on retention + lag gains (estimated +6 hours total runtime) to attempt sustained mastery_corr ≥0.10.

Recommended: Perform one refinement cycle (low-risk modifications) before locking claims; if no uplift, finalize Tier A narrative.

### 16.15 Summary Statement
“Alignment-enabled GainAKT2Exp maintains structural interpretability and improves predictive performance over prior baselines while producing transient mastery correctness correlations (~0.15 mid-training) that decay by epoch end and low final gain correlations. We identify retention and lag-predictive alignment objectives as the next leverage points to convert transient local alignment into stable global semantic interpretability.”

---

## 17. Refinement Cycle (Retention + Lag Gain) Multi-Seed Evaluation

### 17.1 Objective
Assess whether introducing (i) a retention penalty to preserve peak mastery correlation and (ii) lag gain correctness alignment to strengthen incremental semantic signaling produces sustained final semantic correlations (mastery ≥ 0.10, gain ≥ 0.06) without degrading AUC relative to the alignment-enabled configuration in Section 16.

### 17.2 Configuration 

| Parameter | Value |
|-----------|-------|
| Dataset | assist2015 |
| Seeds | 21, 42, 63, 84, 105 |
| Epochs | 12 |
| Batch size | 64 |
| Learning rate | 1.74e-4 |
| Alignment weight | 0.25 (warm-up 4 epochs) |
| Retention (δ, weight) | 0.01, 0.10 (logging-only) |
| Lag gain (max_lag, weight) | 3, 0.05 |
| Global alignment sampling | 300 validation students |
| Residual alignment | Enabled (window = 5) |
| Consistency rebalancing | Epoch ≥ 8 if mastery_corr < 0.10 → weight 0.2 |
| Variance floor | 1e-4 (patience 3, sparsity weight ×0.5) |

### 17.3 Command

```bash
python tmp/run_gainakt2exp_baseline_compare_resumable.py \
  --dataset assist2015 --epochs 12 --batch_size 64 \
  --seeds 21 42 63 84 105 --variants full \
  --enable_alignment_loss --alignment_weight 0.25 --alignment_warmup_epochs 4 \
  --enable_global_alignment_pass --alignment_global_students 300 \
  --use_residual_alignment \
  --enable_retention_loss --retention_delta 0.01 --retention_weight 0.1 \
  --enable_lag_gain_loss --lag_gain_weight 0.05 --lag_max_lag 3 \
  --warmup_constraint_epochs 8 --use_amp \
  --devices 0 1 2 3 4 --max_workers 5 \
  --output_dir tmp --progress_path tmp/gainakt2exp_progress_refine.json
```

### 17.5 Final Performance Metrics

| Metric | Mean | Std | 95% CI (t, df=4) | Notes |
|--------|------|-----|------------------|-------|
| Best Val AUC | 0.71851 | 0.00070 | [0.71789, 0.71912] | Improved vs prior alignment (0.7057) |
| ΔAUC vs Section 16 | +0.01270 | — | — | Clear positive gain (CI excludes prior mean) |
| Mean Best Epoch | 11.8 | — | — | Late stabilization (epochs 11–12) |
| Seeds (n) | 5 | — | — | 21,42,63,84,105 |

Interpretation: Performance not only neutral but improved; late best-epoch suggests auxiliary objectives delay AUC peak relative to early plateaus observed previously.

### 17.6 Final Semantic Metrics

| Metric | Mean | Std | 95% CI (t) | Target | Status |
|--------|------|-----|-------------|--------|--------|
| Final Mastery Corr | 0.07168 | 0.00399 | [0.06676, 0.07660] | ≥ 0.10 | Not met |
| Final Gain Corr | 0.04582 | 0.00449 | [0.04022, 0.05142] | ≥ 0.06 | Not met (improved vs 0.0241 prior) |
| Peak Mastery Corr (consistency metric) | ~0.146 | — | — | ≥ 0.15 | Slightly below (some seeds reach ~0.150) |
| Peak Global Align Mastery Corr | 0.06509 | 0.00194 | — | ≥ 0.15 | Far below (different scale) |
| Peak Global Align Gain Corr | 0.13161 | 0.00435 | — | ≥ 0.06 | Met (transient, alignment metric) |
| Retention Decay Gap (mean) | -0.05935 | — | — | ≤ 0 | Met (all seeds gap ≤ 0) |
| Lag Gain Mean Corr (avg lags) | NA | NA | NA | Positive | Not logged (instrumentation pending) |

Notes:

- Mastery and gain performance correlations remain well below emerging semantic thresholds despite gain correlation improvement.
- High peak global alignment gain correlation indicates local batch/global alignment mechanism effective; translation to global performance semantics limited.
- Retention objective (logging-only) preserved or exceeded peak vs final mastery correlation under defined decay criterion (negative gaps across all seeds).

### 17.7 Retention Analysis

All five seeds exhibit negative decay gaps (mean -0.0593), satisfying the retention success indicator despite retention loss being non-gradient. This suggests natural late-epoch stabilization rather than active preservation; enabling gradient for retention could further enforce plateau if needed.

### 17.8 Lag Gain Correlation Analysis

Not yet instrumented. Current JSON artifacts do not include per-lag correctness correlations, preventing evaluation of temporal predictive signaling. Action: add logging for Corr(Gain_t, Correctness_{t+ℓ}), ℓ=1..3, with optional bootstrap. Without this, we cannot distinguish immediate vs lagged gain semantics.

### 17.9 Planned Loss Share Instrumentation (Pre-Refinement)

This section defines the planned regularization balance instrumentation to be captured by the refined training loop (implemented in Section 17.14). It enumerates target early vs. late epoch shares for each auxiliary component. Numerical results are NOT yet available; they will be reported in Section 17.15 after the refinement multi‑seed run completes. Instrumentation records per‑epoch decomposed loss shares (`loss_shares` object) across: main BCE, total constraint (original interpretability heads), alignment (mastery + gain + adaptive scaling), lag (future correctness correlation), and retention (decay gap penalty).

| Component                | Epochs 1–4 Mean Share | Epochs 9–12 Mean Share | Target / Expectation | Interpretation |
|-------------------------|-----------------------|------------------------|----------------------|----------------|
| Main BCE                | >50%                       | >50%                      | Dominant anchor        | Core predictive objective remains anchor |
| Constraint Total        | ≤25%                       | ≤25% or ↓                 | Balance                | Excess indicates over‑regularization risk |
| Alignment (corr only)   | Gradual ramp (≤15%)        | Stable 8–15%              | Avoid spikes           | High early share may suppress variance |
| Lag (future corr)       | ~0 (inactive gate)         | ≤8%                       | Post warm-up only      | Drives predictive incremental semantics |
| Retention               | 0                          | ≤5% (only if decay)       | Decay-triggered        | Penalizes mastery correlation erosion |
| Consistency (current wt)| 10–15%                     | 5–10% (after rebalance)   | Possible reduction     | Overweighting reduces variance for alignment |
| Sparsity (adjusted)     | 5–10%                      | 5–10% or ↓ if variance low| Variance floor reactive| Ensures focused gains; excessive share may cause collapse |

Provisional Observations (before data collection):
 
1. We expect alignment share to rise gradually, not spike; a spike >20% in epoch 1–2 would justify increasing warm‑up length or lowering base alignment_weight.
2. Retention share should remain negligible until a measurable decay gap emerges (post warm‑up); non‑zero early retention implies scheduling bug.
3. Lag share should appear only after activation gate ((epoch > warmup + 2)); presence earlier indicates gating failure.
4. Constraint Total share exceeding 30% sustained across late epochs suggests need to down‑scale performance alignment weights or increase cosine scheduling.
5. Consistency share reduction after rebalancing epoch confirms adaptive mechanism responding to low mastery correlation.
 

Replacement Procedure (populate Section 17.15 after run):
 
1. Use `tmp/aggregate_semantic_metrics.py` adding logic (or a small helper) to average per‑epoch component shares over epochs 1–4 and 9–12.
2. Substitute placeholders with formatted percentages (e.g., 0.12 → 12.0%).
3. Add a one‑sentence interpretation beneath the table noting any deviations (e.g., “Alignment peaked at 18% in epoch 3 then stabilized at 10%, within acceptable range”).
 

Pending Data Note: Until Section 17.15 is populated, regularization balance claims remain qualitative; structural interpretability and performance claims stand, but semantic regularization cost will be quantitatively justified once empirical shares are inserted.

### 17.10 Scenario Classification

Classification: **S3 (Full ΔAUC gain but correlations weak)** persists. Partial movement toward improved gain semantics (final gain corr increased) but mastery correlation declined vs prior run (0.0825 → 0.0724). No upgrade toward S1/S2; semantic emergence still incomplete.

### 17.11 Interpretation

Retention + lag modifications recovered and exceeded prior AUC (+0.0127 over Section 16) while modestly increasing gain-performance correlation (0.0241 → 0.0433) and reducing mastery-performance correlation (0.0825 → 0.0724). Transient mastery peaks (~0.146–0.150) remain unretained at evaluation time in terms of improved final correlation; logging-only retention objective yields negative decay gaps but does not elevate absolute final alignment. Global alignment gain correlations are strong (>0.13) yet fail to propagate to sequence-level gain-performance correlations, indicating a mismatch between optimization signal (alignment with residual correctness samples) and evaluation metric (per-sequence correctness correlation). Overall, refinement enhanced predictive performance and stabilized gain positivity without achieving sustained semantic alignment thresholds.
 
Plain summary: Performance improved, but the mastery and gain curves still do not consistently reflect student correctness by the final epoch. We therefore have structural interpretability plus a predictive benefit, but not yet stable semantic alignment.



### 17.12 Conditional Actions

| Condition | Action |
|-----------|--------|
| Mastery ≥ 0.10, Gain ≥ 0.06 | Proceed to broader datasets |
| Mastery improves, Gain stagnant | Increase lag_gain_weight to 0.075 or lag_max_lag to 4 |
| Neither improves | Restore longer alignment warm-up (8), reduce alignment_weight to 0.15, consider batch_size 96 |
| AUC drop > 0.01 | Delay alignment activation (later epochs) |

Chosen action: Because mastery (0.07168) and gain (0.04582) correlations remain below emerging targets (0.10 / 0.06) despite improved AUC, we will run one additional refinement iteration before broadening datasets. This iteration enables gradient retention, redesigns the lag objective, extends alignment warm-up, increases global sampling, and instruments lag correlations (ℓ=1..3). Isolation ablations (A–E) are deferred until we evaluate uplift from these changes.

### 17.13 Open Questions

1. Shortened warm-up (4 vs 8) did not harm predictive AUC (improved overall) but likely contributed to early constraint pressure limiting mastery correlation growth; extending to 8 may allow higher sustained mastery semantics.
2. Residual correctness transformation appears neutral to positive for gain alignment (high global gain corr) but may dilute mastery signal; an ablation without residualization is warranted.
3. Increasing global sample size (>600) could reduce variance and potentially raise global alignment mastery corr (current peaks ~0.065); stratified sampling by sequence length recommended.
4. Performance alignment (0.8 weights) combined with consistency (0.3 initial, rebalanced to 0.2) likely over-regularized mastery, contributing to late decay; introducing cosine anneal or lowering early performance weights may preserve variance needed for semantic correlation.

Additional Considerations:

- Enable gradient-based retention penalty to actively counter late decay rather than passive monitoring.
- Instrument lag gain correlations to validate temporal predictive semantics before further weight increases.

Answers / decisions following current results:
 
1. Warm-up extension: Will extend to 8 epochs (implemented next run) to allow mastery variance and reduce early suppression.
2. Residual correctness: Retain for next run; remove only if mastery correlation still <0.09 after extended warm-up.
3. Global sampling: Increase to 600 students with stratified length buckets to strengthen global alignment estimation.
4. Performance alignment weights: Keep current schedule; add conditional cosine anneal after epoch 8 if mastery_corr <0.09.
5. Retention: Activate gradient-based retention penalty to attempt preservation of mid-epoch mastery peaks.
6. Lag semantics: Log Corr(Gain_t, Correctness_{t+ℓ}) for ℓ=1..3 to assess predictive lag structure.
7. Ablations: Postpone Isolation Ablations (A–E) until after semantic uplift attempt; trigger if thresholds remain unmet.
 


### 17.14 Next Refinement Iteration (Scheduled Changes)

We initiate a follow-up refinement run to attempt semantic uplift beyond S3 toward emerging alignment. Implemented code modifications (see updated `examples/train_gainakt2exp.py`) include:

1. Gradient-Based Retention Penalty: Peak mastery decay gap now translates into a scheduled penalty applied across batches in the subsequent epoch (rather than logging-only), aiming to preserve transient mastery correlations.
2. Delayed Lag Objective Start: Lag gain correlation loss activates only after warm-up + 2 epochs to prevent early temporal noise from destabilizing predictive features.
3. Lag Correlation Logging: `mean_lag_corr` and `lag_corr_count` recorded per epoch to enable temporal semantic analysis and future CIs.
4. Variance Gating for Alignment: Alignment loss is zeroed when mastery variance falls below `variance_floor`, preventing over-regularization under collapsed representations.
5. Expanded Global Sampling: Planned increase to 600 validation students (stratified) for more robust global alignment signal estimation.
6. Retention Penalty Distribution: Epoch-level retention penalty divided across batches for stable gradient contribution without spikes.
7. Alignment Warm-Up Extension: Warm-up length set to 8 epochs to reduce early constraint pressure and allow mastery variance to form.
8. Future Annealing (Planned): Option to add cosine anneal of performance alignment weights (currently warm-up scaling; switch conditional if needed in subsequent patch) to mitigate late decay.

Target outcome metrics for this iteration:
 
- Final Mastery Corr ≥ 0.10 (CI excluding 0)
- Final Gain Corr ≥ 0.06 (CI excluding 0)
- Mean Lag Corr (ℓ=1..3) positive and > 0.05 for at least one lag
- No AUC regression (mean AUC ≥ 0.718) and ΔAUC vs baseline |Δ| ≤ 0.002.

If uplift criteria fail, we will proceed to Isolation Ablations (Experiments A–E) to attribute component effects before deciding on further semantic optimization or publication with structural interpretability focus.

See Section 17.15 for forthcoming empirical loss share outcomes once this refinement run completes.

---
## 23.1 Forensic Analysis: Legacy Phase 3 Semantic Emergence vs Current Runs

### Summary of Divergence
Legacy Phase 3 semantic emergence (commit `cf4f4017`) reported a final global mastery–performance correlation of ≈0.113 (AUC ≈0.7175). Current reproductions using nominally equivalent settings yield substantially lower global mastery correlations (≈0.02–0.04 under the adaptive training script; ≈0.056 under a stripped helper without constraints). Gain correlation remains below the legacy ≈0.046 mark.

### Investigated Factors and Findings
| Factor | Legacy Behavior (Inferred) | Current Behavior | Effect on Mastery Corr | Conclusion |
|--------|----------------------------|------------------|------------------------|------------|
| Sampling (accumulate + stratified replace) | Present | Identical | Neutral | Not causal |
| Correlation function | Pearson-like | Same (+ unbiased variant) | Small inflation only | Not causal |
| Alignment decay | Static weight | Adaptive decay (can disable) | Minor uplift when disabled | Secondary |
| Warm-up scheduling | 8 epochs | Tested 4 & 8 | Limited effect | Not primary |
| Variance floor & sparsity | Absent (legacy) | Present (can freeze) | Freeze lifts peak to ~0.04 | Insufficient |
| Constraints (performance + consistency) | Active | Active; OFF tests performed | OFF increases corr to ~0.056 | Constraints currently suppress |
| Mastery accumulation scaling (gain * 0.1) | Assumed | Confirmed | May reduce variance | Possible minor contributor |
| Architecture (512/6/1024) | Used | Matched | Neutral | Not causal |
| Retention mechanism | Disabled | Enabled/disabled tests | Negligible | Not causal |

### Hypotheses
1. Legacy performance alignment loss formulation differed (e.g., direct correlation penalty) producing higher mastery-performance coupling.
2. Dynamic or larger effective gain scaling previously preserved more mastery variance.
3. Preprocessing differences altered sequence length / variance characteristics exploited by projection heads.
4. Current joint performance + consistency constraints over-regularize early epochs, damping correlation growth.

### Key Experiments (Seed 21)
| Experiment | Setup | Peak Mastery Corr | Final | Best Val AUC | Observation |
|-----------|-------|-------------------|-------|--------------|-------------|
| Adaptive baseline (freeze sparsity) | Constraints ON | ~0.029 | 0.022 | 0.724 | Low plateau |
| Alignment weight 0.4 | Constraints ON | ~0.031 | 0.015 | 0.723 | Higher weight ineffective |
| Warm-up 4 + freeze | Constraints ON | 0.0399 | 0.0399 | 0.724 | Highest adaptive peak |
| Helper, constraints OFF, union | Pure BCE | 0.0572 | 0.0561 | 0.724 | Variance boost without constraints |
| Helper, constraints ON, union | Full weights | 0.0475 | 0.0374 | 0.725 | Constraints reduce corr |
| Helper, constraints ON, replace | Full weights | 0.0501 | 0.0362 | 0.725 | Sampling mode not decisive |

### Interpretation
Constraint formulations presently reduce alignment instead of enhancing it; legacy correlation magnitude likely required a different auxiliary shaping or gain accumulation dynamic. Removing constraints partially restores variance but not to legacy level, implying structural/optimization differences beyond constraint presence.

### Recovery Plan
1. Introduce `--gain_scale` sweep (0.05–0.2) to test variance sensitivity.
2. Prototype correlation-based performance alignment loss replacing clamp/hinge.
3. Implement linear warm-up for performance & consistency weights (epochs 1–6) to preserve early variance.
4. Log concept-level mastery vs correctness correlations and mastery variance trajectories.
5. Reconstruct historical loss (if source diff unobtainable) using correlation + variance encouragement.
6. Multi-seed evaluation across variants; target mean global mastery corr ≥0.09 without AUC degradation.

### Reproducibility Additions
Record in results JSON: `gain_scale`, `loss_variant`, `constraint_schedule`, concept-level alignment CSV path, mastery variance stats.

### Success Criteria (Recovery)
| Metric | Target |
|--------|--------|
| Global mastery corr (mean ± std) | ≥0.09 |
| Global gain corr | ≥0.04 |
| ΔAUC vs baseline | |Δ| ≤ 0.002 |
| Structural violations | 0% |
| Epoch time overhead | ≤15% |

### Immediate Next Action (Outside This Doc)
Implement gain scaling flag and run sensitivity test (seed 21) before reformulating alignment loss.

### Referenced Artifacts
`paper/results/gainakt2exp_semantic_trajectory_warmup4_freezesparsity_seed21.json`  
`paper/results/legacy_phase3_semantic_trajectory_seed21_union_unbiased.json`  
`paper/results/legacy_phase3_semantic_trajectory_seed21_union_unbiased_constrained.json`  
`paper/results/legacy_phase3_semantic_trajectory_seed21_replace_unbiased_constrained.json`

Commit anchor: `cf4f4017` (`configs/gainakt2_phase3_semantic.yaml`).

---

## 24. Phase 3 Semantic Recovery Attempts (Mastery Correlation Restoration)

This section documents the iterative efforts undertaken to reproduce the previously observed "semantic breakthrough" state (legacy mastery–performance correlation ≈ 0.113) which has not been recovered under the current GainAKT2Exp training regime. We explored progressively more elaborate scheduling, variance preservation, and correlation regularization strategies. Despite stabilizing latent representation variance (VarMean) in later iterations, the mastery correlation plateaued near ~0.055 and did not ascend toward the target threshold (>0.10).

### 24.1 Target and Evaluation Context
| Metric | Legacy Reference | Recovery Target | Achieved (Best) |
|--------|------------------|-----------------|-----------------|
| Mastery–Performance Correlation (Global) | ~0.113 | ≥0.10 sustained post scale ≥0.5 | ~0.0573 (transient, v3c) / ~0.0552 (stable, v3e) |
| Gain–Performance Correlation | ~0.046 | ≥0.04 | ~0.04–0.055 (varied, not focus) |
| Validation AUC | ~0.7175 | ≥0.70 with improved semantics | 0.6457 (v3e final epoch) |
| Mastery Variance Mean (VarMean) | Non-collapsing (>0.002) | ≥0.002 during ramp | Collapsed (v3d), Preserved (v3e ≈0.0096–0.0100) |

### 24.2 Attempt Trajectory Summary
| Attempt | Key Strategy Elements | Variance Behavior | Peak MCorr | Post-Ramp MCorr Trajectory | Outcome Assessment |
|---------|-----------------------|-------------------|-----------|----------------------------|--------------------|
| Baseline (unscheduled) | Standard constraints, no correlation scheduling | Mild early compression | ~0.03 | Decline / flat | Insufficient semantic alignment |
| Scheduled v2 | Warm-up + ramp, dynamic target ladder, variance instrumentation | Collapse mid-ramp | ~0.056 | Decay after mid-ramp | Temporary uplift; unstable |
| v3c | Gating + mixed variance penalties + restoration noise (light) | Severe collapse (VarMean → 1e-5) | ~0.0565 | Unsustained; forced early stop | Variance safeguards inadequate |
| v3d | Stronger penalties, recovery freeze, promotion guard, batch noise | Collapse persists (early stop epoch 7) | ~0.0573 | Aborted pre full ramp | Stability not achieved; penalties inactive (VarPen=0) |
| v3e | Hard variance gate, health streak, delayed correlation (corr_min_scale), adaptive escalation (unused due to stability) | Stable (VarMean ≈ 0.0096–0.0100) | ~0.0552 | Plateau ~0.055 (target 0.08 unmet) | Stability success; semantic growth failure |

### 24.3 Diagnostic Insights
1. Early correlation application prior to variance consolidation increased representational compression leading to catastrophic variance collapse (v3c–v3d).
2. Batch-level variance penalties often did not activate because VarMean dropped precipitously under no_grad manipulations or gating prevented regularizer gradients from flowing; redesign in v3e avoided collapse without needing penalty activation.
3. Once variance was preserved (v3e), mastery correlation remained pinned ~0.055; hinge-style regularizer with static targets did not induce further semantic differentiation despite healthy variance.
4. Promotion to higher correlation targets (0.08+) in v3e lacked upward response, indicating either (a) insufficient gradient signal for semantic separation or (b) architectural bottlenecks limiting attainable correlation under current constraint weighting.
5. Validation AUC trade-off: stronger constraint activation coincided with monotonic decrease in VAUC (0.7172 → 0.6457), suggesting over-regularization relative to predictive neutrality objectives.

### 24.4 Reasons for Abandonment of Current Recovery Path
| Limitation | Evidence | Implication |
|------------|----------|-------------|
| Correlation Plateau | MCorr stable ≈0.055 across epochs 6–12 in v3e | Scheduling + gating insufficient to drive target semantics |
| Diminishing Returns | Added complexity (freeze, noise, escalation logic) produced no MCorr improvement | Further engineering cost unlikely to yield step-change |
| Predictive Degradation | VAUC fell below neutrality threshold (<0.70) while correlations stagnated | Trade-off unfavorable for publication claims |
| Unused Penalties | Variance penalties / spread regularizer never activated in stable regime | Design complexity not leveraged |
| Architectural Ceiling Hypothesis | Recurrent plateau across variants despite variance stability | May require architectural innovation, not further scheduling tweaks |

### 24.5 Alternative Future Directions (Not Pursued Here)
- Architectural modification: Introduce per-concept calibration layers or contrastive temporal alignment instead of scalar correlation hinge.
- Metric reshaping: Replace Pearson hinge with differentiable rank-based or mutual information estimators for richer gradient signal.
- Adaptive gain-to-mastery scaling: Learn scaling coefficient (currently fixed at 0.1) to allow dynamic temporal sensitivity.
- Per-batch correlation curriculum: Begin with local (short-window) alignment before global sequence-level correlations.

### 24.6 Final Status
The stability objective (preserving latent variance through the constraint ramp) was achieved in the v3e configuration; however, the originally reported semantic breakthrough mastery correlation (>0.10) was not recovered. Correlation improvements plateaued at approximately half the target despite multiple intervention layers. Further incremental schedule / penalty adjustments are assessed as low expected value relative to architectural overhaul.

> **Abandonment Notice (Phase 3 Recovery):** After extensive iterative experimentation (baseline, v2 scheduling, v3c/v3d variance-preservation attempts, and v3e stabilized regime) we were not able to reproduce the legacy semantic breakthrough (mastery–performance correlation > 0.10). We therefore formally discontinue this specific recovery approach and will redirect effort toward architectural and metric design alternatives rather than further schedule or penalty tuning.

**Document version update:** 2025-10-22 (Phase 3 recovery attempt summary relocated to end.)



