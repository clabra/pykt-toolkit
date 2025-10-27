# Experiment 20251027_020809_gainakt2exp_aucup2 (Repro Mode)

This folder was generated in reproducible mode. All hyperparameters originate from config.json.
Re-run criteria: identical config_md5 and command reconstruction leads to matching best AUC (within stochastic tolerance for seeds).

Artifacts produced by underlying training script will supplement this README automatically if extended script used.

## Key Hyperparameters
```json
{
  "training": {
    "epochs": 12,
    "batch_size": 64,
    "learning_rate": 0.00018,
    "weight_decay": 1.7571e-05,
    "optimizer": "Adam",
    "scheduler": "reduce_on_plateau",
    "scheduler_params": {
      "mode": "max",
      "factor": 0.5,
      "patience": 3
    },
    "gradient_clip": 1.0,
    "patience": 20,
    "use_amp": false,
    "enhanced_constraints": true,
    "seed": 42,
    "seeds": [
      42,
      63,
      84
    ],
    "use_wandb": false,
    "output_dir": "/workspaces/pykt-toolkit/examples/experiments/20251027_020809_gainakt2exp_aucup2"
  },
  "constraints": {
    "non_negative_loss_weight": 0.0,
    "monotonicity_loss_weight": 0.1,
    "mastery_performance_loss_weight": 0.7,
    "gain_performance_loss_weight": 0.7,
    "sparsity_loss_weight": 0.2,
    "consistency_loss_weight": 0.3,
    "warmup_constraint_epochs": 6
  },
  "alignment": {
    "enable_alignment_loss": true,
    "alignment_weight": 0.12,
    "alignment_warmup_epochs": 4,
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
    "enable_retention_loss": true,
    "retention_delta": 0.005,
    "retention_weight": 0.08
  },
  "lag_gain": {
    "enable_lag_gain_loss": true,
    "lag_gain_weight": 0.03,
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