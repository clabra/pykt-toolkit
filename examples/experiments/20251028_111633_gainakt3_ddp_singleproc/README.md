# Experiment 20251028_111633_gainakt3_ddp_singleproc

Model: gainakt3
Short title: ddp_singleproc

# Reproducibility Checklist

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
| Correlation metrics logged | ✅ |

## Multi-Seed Best Metrics Summary
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Val AUC | 0.501637 | 0.000000 | 0.501637 | 0.501637 |
| Mastery Corr | 0.061670 | 0.000000 | 0.061670 | 0.061670 |
| Gain Corr | -0.023436 | 0.000000 | -0.023436 | -0.023436 |

## Primary Seed Best
Best epoch (primary seed): 1 val_auc=0.501637308039747 mastery_corr=0.06166952331908138 gain_corr=-0.02343613127409267

## Config MD5
87f5d957b25dfb13ea16ecfb0631a0a3

## Hardware
Requested devices: [0, 1, 2, 3, 4]
CUDA_VISIBLE_DEVICES: 0,1,2,3,4