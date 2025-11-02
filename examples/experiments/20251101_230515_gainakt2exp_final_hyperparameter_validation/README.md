# Experiment 20251101_230515_gainakt2exp_final_hyperparameter_validation

Model: gainakt2exp
Short title: final_hyperparameter_validation

## Reproducibility Checklist

| Item | Status |
|------|--------|
| Folder naming convention followed | ✅ |
| config.json contains all params | ✅ |
| Shell script lists full command | ✅ |
| Best + last checkpoints saved | ✅ |
| Per-epoch metrics CSV present | ✅ |
| Raw stdout log saved | ✅ |
| Git commit & branch recorded | ✅ |
| Seeds documented | ✅ |
| Environment versions captured | ✅ |
| Correlation / interpretability metrics logged | ✅ |

## Configuration Provenance
The `config.json` groups appear in the following canonical order ensuring reproducibility of parameter resolution:


| Group | Key Highlights |
|-------|----------------|
| runtime | launcher + train commands, timestamp |
| hardware | device list, thread count |
| seeds | primary + all seeds |
| training | epochs, batch_size, lr, optimizer, mixed_precision, gradient_clip |
| experiment | id, model, short_title, purpose |
| data | dataset, fold |
| interpretability | core heads + constraint weights + warmup |
| alignment | enable, weight, warmup, adaptive, min_corr, share_cap/decay |
| global_alignment | enable pass, students, residual window |
| refinement | retention, lag predictive emergence, rebalance, variance floor |

### Semantic Flag Snapshot
| Flag | Value |
|------|-------|
| alignment.enable_alignment_loss | True |
| alignment.alignment_weight | 0.25 |
| alignment.alignment_warmup_epochs | 8 |
| alignment.adaptive_alignment | True |
| alignment.alignment_min_correlation | 0.05 |
| alignment.alignment_share_cap | 0.08 |
| alignment.alignment_share_decay_factor | 0.7 |
| global_alignment.enable_global_alignment_pass | True |
| global_alignment.alignment_global_students | 600 |
| global_alignment.use_residual_alignment | True |
| global_alignment.alignment_residual_window | 5 |
| refinement.enable_retention_loss | True |
| refinement.retention_delta | 0.005 |
| refinement.retention_weight | 0.14 |
| refinement.enable_lag_gain_loss | True |
| refinement.lag_gain_weight | 0.06 |
| refinement.lag_max_lag | 3 |
| refinement.lag_l1_weight | 0.5 |
| refinement.lag_l2_weight | 0.3 |
| refinement.lag_l3_weight | 0.2 |
| refinement.consistency_rebalance_epoch | 8 |
| refinement.consistency_rebalance_threshold | 0.1 |
| refinement.consistency_rebalance_new_weight | 0.2 |
| refinement.variance_floor | 0.0001 |
| refinement.variance_floor_patience | 3 |
| refinement.variance_floor_reduce_factor | 0.5 |

### Core Interpretability & Constraints
| Key | Value |
|-----|-------|
| interpretability.use_mastery_head | True |
| interpretability.use_gain_head | True |
| interpretability.monotonicity_loss_weight | 0.1 |
| interpretability.mastery_performance_loss_weight | 0.8 |
| interpretability.gain_performance_loss_weight | 0.8 |
| interpretability.sparsity_loss_weight | 0.2 |
| interpretability.consistency_loss_weight | 0.3 |
| interpretability.warmup_constraint_epochs | 8 |
| interpretability.non_negative_loss_weight | 0.0 |
| interpretability.enhanced_constraints | True |
| interpretability.max_semantic_students | 50 |

### Evaluation Snapshot
| Key | Value |
|-----|-------|
| evaluation.dataset | assist2015 |
| evaluation.fold | 0 |
| evaluation.batch_size | 64 |
| evaluation.seq_len | 200 |
| evaluation.d_model | 512 |
| evaluation.n_heads | 8 |
| evaluation.num_encoder_blocks | 6 |
| evaluation.d_ff | 1024 |
| evaluation.dropout | 0.2 |
| evaluation.emb_type | qid |
| evaluation.use_mastery_head | True |
| evaluation.use_gain_head | True |
| evaluation.non_negative_loss_weight | 0.0 |
| evaluation.monotonicity_loss_weight | 0.1 |
| evaluation.mastery_performance_loss_weight | 0.8 |
| evaluation.gain_performance_loss_weight | 0.8 |
| evaluation.sparsity_loss_weight | 0.2 |
| evaluation.consistency_loss_weight | 0.3 |
| evaluation.max_correlation_students | 300 |

Snapshot MD5: `5d4c1b259dc8ae0277fd1ce352dbd27e`

### Reproducibility Policy
Ignored evaluation flags: device, run_dir
Ignored training flags: auto_shifted_eval, config, experiment_id
Schema version: 1.0.0
Policy last updated: 2025-10-30
