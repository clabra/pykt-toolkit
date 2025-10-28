# GainAKT3 Architecture: Leveraging Peer Similarity and Historical Difficulty for Enhanced Semantic Interpretability

**Copyright (c) 2025 Concha Labra. All Rights Reserved.**

## 1. Motivation and Gap Analysis
Previous semantic interpretability enhancements (branch `v0.0.8-gainakt2-semantic`) focused on internal constraints (monotonicity, non‑negative gains, sparsity, performance correlations) but did **not** exploit two external, high‑signal information sources:
1. **Peer Response Patterns**: How other students with similar historical knowledge trajectories answered the same item.
2. **Dynamic Skill / Item Difficulty**: Time‑ and cohort‑conditioned difficulty estimates derived from aggregate historical interactions.

Limitations of current `GainAKT2`/`newmodel.md` design:
 
 - Purely single‑student sequence modeling; ignores cross‑student statistical regularities and cohort evolution.
- Item and skill difficulty treated implicitly via embeddings; no explicit calibration channel or temporal difficulty drift modeling.
- Learning gains derived from attention over a student's own past only; no peer‑informative prior for expected gain magnitude.

`GainAKT3` introduces structured external context streams (Peer Similarity Context and Difficulty Context) to enrich gain attribution and mastery estimation while preserving Transformer efficiency and reproducibility standards.

## 2. High‑Level Architectural Additions
We augment the dynamic value stream encoder with two orthogonal, pluggable modules:
 - **Peer Similarity Module (PSM)**: Retrieves or approximates a set of peer summary vectors for the current item/skill conditioned on student state similarity.
- **Historical Difficulty Module (HDM)**: Provides calibrated difficulty embeddings (item‑level d_item_t, skill‑level d_skill_t) with temporal drift capture and uncertainty quantification.

Both modules feed lightweight conditioning vectors into attention and projection heads, enabling:
 
 - Context‑adaptive learning gain scaling (anticipated difficulty × peer correctness distribution).
- Improved interpretability: explicit decomposition of predicted performance into (mastery, difficulty, peer_prior, residual).

## 3. Data Structures and Precomputation
To maintain runtime efficiency and reproducibility, heavy aggregation is precomputed offline and versioned.

### 3.1 Peer Response Index
Stored under `data/peer_index/<dataset>/peer_index.pkl`:
 - For each (item_id) and optionally (skill_id):
  - `peer_correct_rate`: float in [0,1]
  - `peer_attempt_count`: int
  - `skill_local_transfer_vector`: average projected gains for peers after item attempt (vector length = num_skills)
  - `peer_state_cluster_centroids`: K centroids in latent mastery space (K configurable, e.g. 8)
  - Timestamp buckets for temporal drift (weekly or monthly)

### 3.2 Difficulty History Table
Stored under `data/difficulty/<dataset>/difficulty_table.parquet`:
 - Rows: (time_bucket, item_id, skill_id)
- Columns:
  - `difficulty_logit`: model‑agnostic estimate (e.g. Rasch difficulty or logistic calibration)
  - `attempt_count`, `correct_rate`
  - `moving_variance`
  - `stability_score` (1 − normalized variance over last N buckets)
  - Optional: `difficulty_confidence_interval_low/high`

### 3.3 Versioning & Hashing
Each artifact accompanied by `METADATA.json` with:
```json
{
  "source": "assist2015",
  "generated_at": "2025-10-27T18:42:00Z",
  "generator_commit": "<hash>",
  "parameters": {"K": 8, "time_bucket": "week"},
  "sha256": "..."
}
```
Training scripts will log these hashes into `config.json` to preserve reproducibility.

## 4. Core Architectural Flow (Conceptual)

```mermaid
flowchart LR
  %% Subgraph syntax corrected: use id [label] to avoid parser errors on multi-word titles
  subgraph SSEQ[Student Sequence Encoder]
    A((Interaction Tokens q,r)) --> B[Context Embeddings]
    A --> C[Value Embeddings]
    B --> D[Dynamic Encoder Blocks]
    C --> D
    D --> H[Context State h]
    D --> I[Value State v]
  end

  subgraph EXT[External Context]
    E1[Peer Response Index Lookup] --> F1[Peer Features p_t]
    E2[Difficulty Table Lookup] --> F2[Difficulty Features d_t]
  end

  F1 --> G[Context Fusion Gate]
  F2 --> G
  H --> G
  G --> J[Augmented State h']

  subgraph HEADS[Prediction Heads]
    J --> K1[Mastery Head]
    I --> K2[Gain Head]
    J --> K3[Difficulty Calibration Head]
    K1 --> L1[Mastery Vector]
    K2 --> L2[Gain Vector]
    K3 --> L3[Difficulty Logit]
  end

  L1 --> M[Performance Head]
  L3 --> M
  I --> M
  M --> O[Prediction]

  subgraph DECOMP[Interpretability Decomposition]
    O --> P1[Mastery Contribution]
    O --> P2[Difficulty Contribution]
    O --> P3[Peer Prior Contribution]
    O --> P4[Residual]
  end
```

## 5. Detailed Module Specifications
### 5.1 Peer Similarity Module (PSM)
**Inputs**: current item_id, skill_id(s), current mastery embedding h_t.

**Retrieval Strategy** (configurable):
1. Centroid Cosine Match: select top‑K peer_state_cluster_centroids maximizing cos(h_t, c_k).
2. Weighted Aggregate Peer Vector:
   \( p_t = \sum_{k=1}^K w_k c_k \), \( w_k = \text{softmax}(\gamma \cdot \text{cos}(h_t, c_k)) \)
3. Peer Correctness Scalar: peer_correct_rate[item_id] (optionally time‑bucketed) transformed to embedding via learned MLP.
4. Transfer Embedding: project skill_local_transfer_vector[item_id] through linear layer to dimension d_model.

**Output**: Peer context vector p_t ∈ R^{d_model} plus scalar meta‑features (attempt_count, stability_score) optionally appended.

**Interpretability Hooks**:
 - Log selected centroid IDs & weights.
- Store peer_correct_rate and attempt_count used.

### 5.2 Historical Difficulty Module (HDM)
**Inputs**: item_id, skill_id(s), time_index (epoch, absolute timestamp).

**Difficulty Feature Construction**:
1. Base difficulty embedding: map difficulty_logit via learned projection (scalar → R^{d_model}).
2. Drift vector: difference between current bucket logit and moving average last N buckets.
3. Stability scalar: stability_score → gating coefficient.
4. Confidence interval range width → uncertainty scalar.

**Output**: Difficulty context vector d_t ∈ R^{d_model} + scalars (stability, uncertainty).

**Interpretability Hooks**:
 - Log raw difficulty_logit, drift, stability_score, uncertainty per batch.

### 5.3 Context Fusion Gate
We fuse h_t (student intrinsic state), p_t (peer context), d_t (difficulty context):

\[
\tilde{h}_t = h_t + g_p \odot p_t + g_d \odot d_t
\]
Where gates:
\[
[g_p, g_d] = \sigma( W_g [h_t; p_t; d_t] + b_g )
\]
Alternative residual mixture: use LayerNorm after fusion. Provide ablation hooks to disable each gate independently (`--disable_peer`, `--disable_difficulty`).

### 5.4 Difficulty Calibration Head
Predicts calibrated difficulty logit \( \hat{d}_{item} \) used to decompose final prediction:
\[
P(R_t=1) = \sigma( W [\tilde{h}_t; v_t] + b - \beta \hat{d}_{item} )
\]
Where \( \beta \) is a learnable scaling or fixed hyperparameter. This allows explicit subtraction of difficulty: improves explainability (higher mastery must overcome higher difficulty).

### 5.5 Performance Decomposition
After computing logits \( z_t \):
 - Mastery contribution: gradient * (components of \tilde{h}_t).
- Difficulty contribution: \( \beta \hat{d}_{item} \).
- Peer prior contribution: difference when zeroing p_t gate.
- Residual: remainder after removing other contributions.
All stored per epoch in interpretability artifacts.

## 6. Loss Extensions
We integrate new auxiliary objectives supporting the added modules:
1. **Peer Alignment Loss**: Encourage projected mastery to align with peer correctness distribution.
  \( L_{peer} = \text{MSE}( \text{mean}(mastery_{skills}) , peer\_correct\_rate ) \)
2. **Difficulty Ordering Loss**: Pairwise ranking over items: if difficulty_logit_i > difficulty_logit_j then predicted mastery‑adjusted success probability should reflect ordering.
3. **Drift Smoothness Loss**: Penalize excessive volatility in predicted \( \hat{d}_{item} \) over consecutive time buckets: \( L_{drift} = \sum_t |\hat{d}_{t} - 2\hat{d}_{t-1} + \hat{d}_{t-2}| \).
4. **Peer Gate Sparsity**: Encourage selective peer influence: \( L_{gate} = ||g_p||_1 \) (option to reverse sign depending on scaling).
5. **Decomposition Consistency**: Reconstructed probability from components must approximate original prediction: enforce \( L_{decomp} = \text{MSE}( \sigma(z_t), \sigma(z^{recon}_t)) \).

Total extended loss:
\[
L = L_{perf} + L_{existing\_aux} + \lambda_{peer} L_{peer} + \lambda_{diff} L_{difficulty} + \lambda_{drift} L_{drift} + \lambda_{gate} L_{gate} + \lambda_{decomp} L_{decomp}
\]

## 7. Training & Reproducibility Integration
Add new CLI flags to the training script (`examples/train_gainakt3.py`):
 - `--use_peer_context`, `--use_difficulty_context`
- `--peer_K`, `--peer_similarity_gamma`
- `--difficulty_time_bucket` (week|month)
- `--difficulty_drift_window`
- `--lambda_peer`, `--lambda_diff`, `--lambda_drift`, `--lambda_gate`, `--lambda_decomp`
- Hash artifacts: compute SHA256 of peer index and difficulty table → store under `config.json.hardware.artifacts`.
Abort run if artifact hash mismatch and `--strict_artifact_hash` enabled.

## 8. Interpretability Metrics (Extended)
Add per‑epoch logging:
 - `peer_influence_share`: proportion of logit attributable to peer context (via gating analysis).
- `difficulty_adjustment_magnitude`: mean \( \beta \hat{d}_{item} \).
- `mastery_adjusted_accuracy`: accuracy after subtracting difficulty penalty.
- `peer_alignment_error`: current epoch value of L_peer (raw, not weighted).
- `difficulty_rank_accuracy`: proportion of sampled item pairs where predicted ordering matches difficulty ordering.
- `gate_sparsity`: mean |g_p|.
- `decomposition_reconstruction_error`: L_decomp.

## 9. Edge Cases & Robustness Considerations
 - Cold Start Items: Fallback to global average difficulty & peer stats; log `cold_start_flag`.
- Sparse Peer Data: If attempt_count < threshold, reduce g_p via confidence‑based scaling.
- Temporal Drift Spikes: Cap drift magnitude by percentile clipping to prevent destabilizing updates.
- Multi‑Skill Items: Aggregate per‑skill difficulty by mean or weighted by historical attempt density.
- Large num_skills: Use low‑rank adapters for projection heads to keep parameter growth controlled.

## 10. Computational Complexity Impact
 - Peer/Difficulty lookups: O(1) per interaction with hash maps (preloaded in RAM); negligible vs O(L^2 H D) attention.
- Additional heads/gates: O(L D) linear projections.
- Loss additions: Minor overhead for pairwise difficulty ranking (sampled pairs, not full cartesian: O(B * P) with P small, e.g. 64).

## 11. Incremental Implementation Plan
Phase 1 (Scaffolding): Data loaders for peer/difficulty artifacts; gating fusion; logging metrics.
Phase 2 (Heads & Losses): Difficulty calibration head, peer alignment, decomposition consistency.
Phase 3 (Ranking & Drift): Difficulty ordering loss, drift smoothness, robust clipping.
Phase 4 (Ablations & Validation): Systematic on/off toggling; compare AUC, interpretability metrics vs GainAKT2.
Phase 5 (Optimization): Low‑rank gating, memory footprint profiling, seed reproducibility validation.

## 12. Experiment Design & Reporting
Each experiment folder must record artifact hashes and gating configuration. Comparative tables will include:
 - Base (GainAKT2) vs GainAKT3 (+peer, +difficulty, +both)
- Metrics: AUC, mastery_perf_corr, gain_perf_corr, peer_alignment_error, difficulty_rank_accuracy, decomposition_reconstruction_error.
Interpretability improvement claim centered on reduced peer_alignment_error and meaningful difficulty contribution variance.

## 13. Expected Benefits
 - Higher semantic alignment: mastery estimates contextualized by cohort performance and calibrated difficulty.
- Better early prediction on sparse student sequences via peer priors.
- Transparent decomposition: educators see if errors arise from high difficulty rather than low mastery.
- Enhanced gain estimation stability (peer‑regularized expected magnitude).

## 14. Risk & Mitigation
| Risk | Symptom | Mitigation |
|------|---------|------------|
| Peer leakage / label proxy | Overreliance on peer correctness inflates AUC artificially | Gate regularization + monitor peer_influence_share ceiling |
| Artifact drift | AUC inconsistency across runs | Hash & log artifact versions; strict mode abort on mismatch |
| Difficulty miscalibration | Negative impact on mastery correlation | Recalibrate baseline difficulty with held‑out fold; add temperature scaling |
| Increased latency | Slowed batch throughput | Preload artifacts; vectorize lookups; microbench gating |
| Overfitting to dense skills | Sparse skill performance degrades | Confidence scaling based on attempt_count; auxiliary loss weighting by skill coverage |

## 15. Summary
`GainAKT3` extends the dynamic gain aggregation paradigm with externally informed context: peer similarity and historical difficulty. These additions preserve architectural modularity, enhance interpretability through explicit performance decomposition, and introduce new auxiliary losses aligning latent representations with educational cohort dynamics. Implementation follows a phased, reproducible plan with rigorous artifact hashing and extended metrics, positioning GainAKT3 as a robust, semantically enriched evolution over GainAKT2.

## 16. Next Steps
1. Implement Phase 1 scaffolding (artifact loaders + gating) in a temporary `tmp/gainakt3_prototype.py` file.
2. Create preprocessing script `examples/build_peer_difficulty_artifacts.py` with deterministic aggregation & hashing.
3. Add initial experiment `examples/experiments/<timestamp>_gainakt3_peer_only_baseline`.
4. Evaluate effect on cold‑start student sequences (first 10 interactions) vs GainAKT2.
5. Progressively enable difficulty calibration and decomposition metrics.

---
We will proceed with Phase 1 upon confirmation.

\n## 17. Phase2 Implementation Status (2025-10-28)
The production model file `pykt/models/gainakt3.py` has been extended to include auxiliary interpretability constraint losses. These are implemented directly in the forward pass and exposed for the training script to incorporate.

\n### Implemented Auxiliary Losses
\n| Loss | Purpose | Formula (Simplified) | Config Weight |
|------|---------|----------------------|---------------|
| Alignment | Encourage monotonic mastery trajectory | mean(ReLU(-ΔM)) | `alignment_weight` |
| Retention | Discourage mastery decay (separated for logging clarity) | mean(ReLU(-ΔM)) | `retention_weight` |
| Sparsity | Promote gain activation sparsity | mean(|G|) | `sparsity_weight` |
| Consistency | Align gains with positive mastery increments | mean(ReLU(0.5 - cos(ΔM⁺, G))) | `consistency_weight` |
| Lag Gain | Encourage gains to precede future mastery increases | mean(ReLU(0.3 - cos(G_t, ΔM⁺_{t+1}))) | `lag_gain_weight` |

Warm-up control via `warmup_constraint_epochs`: if `current_epoch < warmup_constraint_epochs`, aggregated constraint loss is suppressed (zero) while per-component values are still computed for diagnostic purposes (optional extension: currently they are not computed before warm-up to save compute; we can enable conditional logging later).

### Forward Output Additions
`forward()` now returns:
- `constraint_losses`: dict of detached component losses.
- `total_constraint_loss`: aggregated (weighted) constraint penalty (tensor) used to augment performance loss.
- Existing interpretability metrics (`peer_influence_share`, `difficulty_adjustment_magnitude`, artifact hashes, cold_start flag) retained.

### Trainer Integration Pattern
```
out = model(q, r)
bce_loss = BCE(preds_active, targets_active)
constraint = out['total_constraint_loss']
loss = bce_loss + constraint
loss.backward()
```
Model epoch warm-up set per epoch:
```
model.current_epoch = epoch
```

### Architectural Deviations from Original Design Document
1. Fusion gate currently reports interpretability but does not yet directly fuse peer/difficulty vectors into hidden state (planned future enhancement).
2. Difficulty decomposition into explicit contributions (mastery, difficulty, peer, residual) is not yet serialized; current implementation focuses on constraint scaffolding first.
3. Peer alignment and difficulty ordering losses described in Section 6 of the earlier document have not yet been ported; replaced by a minimal, stable core of five constraint losses to avoid premature complexity.
4. Decomposition consistency (reconstruction) postponed; will be added after baseline stability evaluation of current constraints.

### Rationale for Selected Initial Losses
The chosen subset balances computational simplicity (O(B·L·C)) and immediate interpretability benefits while minimizing risk of training instability. Ranking-based and second-order drift penalties require additional buffered state and sampling logic, deferred to a later milestone.

### Edge Case Handling Implemented
- Short sequences (L ≤ 2): lag gain loss automatically returns zero (no temporal lead window).
- Cold start artifacts: peer vector defaults to zero; difficulty subtraction still applied with learned head.
- Constraint weights set to zero disable term entirely (no unnecessary tensor ops).

### Pending Enhancements
| Feature | Status | Planned Action |
|---------|--------|----------------|
| Peer alignment MSE | Not implemented | Add after verifying centroid stability |
| Difficulty ordering ranking | Not implemented | Introduce sampled pair ranking module |
| Drift smoothness | Not implemented | Maintain circular buffer of recent difficulty logits |
| Decomposition reconstruction | Not implemented | Implement component isolation and recomposition head |
| Gate-based representation fusion | Partial (metrics only) | Replace reporting-only with actual residual integration |

### Configuration Additions (create_gainakt3_model)
New keys exposed for reproducibility:
```
alignment_weight, sparsity_weight, consistency_weight,
retention_weight, lag_gain_weight, warmup_constraint_epochs
```
All default to 0.0 (disabled) unless explicitly set; ensures backwards compatibility with earlier experiments.

### Reproducibility Considerations
- All weights must be serialized in `config.json` and contribute to MD5 hash.
- Forward determinism preserved (no stochastic operations introduced in constraints).
- External artifacts hashing unchanged; constraint logic independent of artifact content.

### Validation
Smoke test executed (2×50 sequence batch) produced non-zero component losses with expected magnitudes and aggregated constraint loss ≈ sum(weighted components.

### Next Implementation Milestone
Integrate constraint loss logging into `metrics_epoch.csv` and expand README template with per-component summaries and interpretability trend plots.

---
End of Phase2 update.
