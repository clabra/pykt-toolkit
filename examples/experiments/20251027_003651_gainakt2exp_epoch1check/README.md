# Experiment 20251027_003651_gainakt2exp_epoch1check (Repro Mode)

This folder was generated in reproducible mode. All hyperparameters originate from config.json.
Re-run criteria: identical config_md5 and command reconstruction leads to matching best AUC (within stochastic tolerance for seeds).

Artifacts produced by underlying training script will supplement this README automatically if extended script used.

## Key Hyperparameters
```json
{
  "training": {
    "epochs": 1,
    "batch_size": 64,
    "learning_rate": 0.000174,
    "weight_decay": 1.7571e-05,
    "optimizer": "Adam",
    "scheduler": "reduce_on_plateau",
    "scheduler_params": {
      "mode": "max",
      "factor": 0.5,
      "patience": 5
    },
    "gradient_clip": 1.0,
    "patience": 20,
    "use_amp": false,
    "enhanced_constraints": true,
    "seed": 42,
    "seeds": [
      42
    ],
    "use_wandb": false,
    "output_dir": "/workspaces/pykt-toolkit/examples/experiments/20251027_003651_gainakt2exp_epoch1check"
  },
  "constraints": {
    "non_negative_loss_weight": 0.0,
    "monotonicity_loss_weight": 0.1,
    "mastery_performance_loss_weight": 0.8,
    "gain_performance_loss_weight": 0.8,
    "sparsity_loss_weight": 0.2,
    "consistency_loss_weight": 0.3,
    "warmup_constraint_epochs": 8
  },
  "alignment": {
    "enable_alignment_loss": false,
    "alignment_weight": 0.25,
    "alignment_warmup_epochs": 8,
    "adaptive_alignment": true,
    "alignment_min_correlation": 0.05,
    "enable_global_alignment_pass": false,
    "alignment_global_students": 600,
    "use_residual_alignment": false,
    "alignment_residual_window": 5,
    "alignment_share_cap": 0.08,
    "alignment_share_decay_factor": 0.7
  },
  "retention": {
    "enable_retention_loss": false,
    "retention_delta": 0.005,
    "retention_weight": 0.14
  },
  "lag_gain": {
    "enable_lag_gain_loss": false,
    "lag_gain_weight": 0.06,
    "lag_max_lag": 3,
    "lag_l1_weight": 0.5,
    "lag_l2_weight": 0.3,
    "lag_l3_weight": 0.2
  }
}
```

## Reproducibility Checklist (Partial)
| Item | Status |
|------|--------|
| config.json present | ✅ |
| environment.txt present | ✅ |
| stdout.log capturing run | ✅ |
| metrics_epoch.csv (after run) | ⏳ |
| results.json (after run) | ⏳ |
| config_md5 recorded | ✅ |

## Multi-Seed Summary
| Metric | Mean | Std | CI Low | CI High |
|--------|------|-----|--------|---------|
| Val AUC | - | - | - | - |
| Mastery Corr | - | - | - | - |
| Gain Corr | - | - | - | - |

Reproducibility checklist updated: multi-seed artifacts present.
