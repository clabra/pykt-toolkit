# Experiments Reproducibility Guide

Approach based on **Explicit Parameters, Zero Defaults**.

## Executive Summary

This document describes the reproducibility infrastructure for training, evaluating, and reproducing experiments oriented to evaluate Deep Knowledge Tracing models. The system enforces **zero hidden defaults**: all training parameters and evaluation parameters must be explicitly specified via command line. A single launcher script (`examples/run_repro_experiment.py`) manages the complete workflow.

**Key Features:**
- ✅ Single source of truth: `configs/parameter_default.json`
- ✅ Explicit commands: ALL parameters visible in generated commands
- ✅ MD5 integrity: Compares that experiment defaults in config.json match source of truth 
- ✅ Experiment id: Use 6-digit experiment ID to identify experiments
- ✅ Parameter provenance: Clear trail from defaults → CLI overrides → execution

**Quick Commands:**
```bash
# Train
python examples/run_repro_experiment.py --short_title test --epochs 12

# Reproduce (relaunch experiment)
python examples/run_repro_experiment.py --repro_experiment_id 584063

# Evaluate (copy eval_explicit command from config.json)
cd examples/experiments/[experiment_folder]
cat config.json | grep eval_explicit
# → Copy and run the command

# Compare
python examples/compare_reproduction.py 584063
```

---

## Quick Start

### Reproducibility Philosophy: No Hidden Defaults

All scripts follow a **strict no-defaults** design:
- Every parameter must be **explicitly specified** via command line
- No hardcoded fallback values in argparse (all parameters use `required=True`)
- Single source of truth: `configs/parameter_default.json`
- Launcher generates explicit commands with ALL ~60+ parameters visible
- Config integrity verified via MD5 hash

### 1. Launch Training Experiment

```bash
python examples/run_repro_experiment.py \
  --short_title baseline \
  --epochs 12 \
  --batch_size 64
```

**What happens:**
1. Loads all defaults from `configs/parameter_default.json` (63 parameters)
2. Applies CLI overrides (e.g., epochs, batch_size)
3. Generates 6-digit experiment ID (e.g., `423891`)
4. Creates experiment folder: `20251102_143210_gainakt2exp_baseline_423891/`
5. Builds **explicit training command** with ALL parameters:
   ```bash
   python examples/train_gainakt2exp.py \
     --dataset assist2015 --fold 0 --seed 42 --epochs 12 --batch_size 64 \
     --learning_rate 0.000174 --weight_decay 1e-05 --optimizer adam \
     --seq_len 200 --d_model 512 --n_heads 8 --num_encoder_blocks 6 \
     ... (50+ more explicit parameters)
   ```
6. Saves config.json with:
   - `defaults`: Pristine copy from parameter_default.json
   - `overrides`: CLI parameters that differ from defaults
   - `commands.train_explicit`: Complete command with all parameters
   - `md5`: Hash of original defaults (tamper detection)

### 2. Evaluate Trained Model

Get evaluation command from config.json

```bash
cd examples/experiments/20251102_143210_gainakt2exp_baseline_423891
# Copy the eval_explicit command from config.json and run it
```

Evaluation command set all parameters explicitly, example: 
```bash
python examples/eval_gainakt2exp.py \
  --run_dir examples/experiments/20251102_143210_gainakt2exp_baseline_423891 \
  --max_correlation_students 300 \
  --dataset assist2015 --fold 0 --batch_size 64 \
  --seq_len 200 --d_model 512 --n_heads 8 --num_encoder_blocks 6 \
  --d_ff 1024 --dropout 0.2 --emb_type qid \
  --non_negative_loss_weight 0.0 --monotonicity_loss_weight 0.1 \
  --mastery_performance_loss_weight 0.8 --gain_performance_loss_weight 0.8 \
  --sparsity_loss_weight 0.2 --consistency_loss_weight 0.3 \
  --use_mastery_head --use_gain_head
```

**What it does:**
- Loads `model_best.pth` from experiment folder
- Computes validation/test AUC, accuracy
- Computes mastery/gain correlations (up to a limited number of students for bounded runtime)
- Saves `eval_results.json`, `config_eval.json`, `metrics_epoch_eval.csv`

**Important:** Evaluation requires ~20 architecture/constraint parameters to match training configuration. Using the `eval_explicit` command from config.json ensures perfect parameter alignment.

### 3. Reproduce Existing Experiment

```bash
python examples/run_repro_experiment.py \
  --repro_experiment_id 423891
```

**What happens:**
1. Searches for experiment folder containing ID `423891`
2. Loads original `config.json` (perfect defaults + original overrides)
3. Creates reproduction folder: `20251102_154320_gainakt2exp_423891_repro/`
4. Copies config.json **unchanged** (byte-for-byte identical)
5. Executes the **same explicit training command** from original config

**Key principle:** No parameter inference, no adaptation, no modifications—exact rerun.

### 4. Compare Original vs Reproduction

```bash
python examples/compare_reproduction.py 423891
```

Auto-detects reproduction folder and shows:
```
Metric                    Original      Repro         Diff          Status  
----------------------------------------------------------------------------------------------------
Best Val AUC              0.7412        0.7411        0.0001        ✅ PASS
Best Val Accuracy         0.6812        0.6811        0.0001        ✅ PASS
Mastery Correlation       0.5234        0.5231        0.0003        ✅ PASS
Gain Correlation          0.4123        0.4122        0.0001        ✅ PASS
```

## Reproducibility Architecture

### Core Principle: Explicit Parameters, Zero Defaults

**Design Philosophy:**
- **No hardcoded defaults**: Every parameter in training/evaluation scripts uses `required=True`
- **Single source of truth**: `configs/parameter_default.json` (63 parameters)
- **Explicit commands**: Launcher generates commands with ALL parameters visible
- **Tamper detection**: MD5 hash verifies config integrity
- **Parameter provenance**: Clear trail from defaults → CLI overrides → explicit command

### Parameter Flow

```
┌─────────────────────────────────┐
│ configs/parameter_default.json  │  ← Single source of truth
│ - training_defaults (63 params) │
│ - md5: ca1ef5c... (integrity)   │
└────────────┬────────────────────┘
             │
             ↓ (load all defaults)
┌─────────────────────────────────┐
│ run_repro_experiment.py  │  ← Launcher
│ - Apply CLI overrides           │
│ - Build explicit commands       │
└────────────┬────────────────────┘
             │
             ↓ (save to experiment folder)
┌─────────────────────────────────┐
│ config.json                     │
│ ├─ defaults: {...perfect...}   │  ← Unchanged from parameter_default.json
│ ├─ overrides: {epochs: 12, ...} │  ← CLI parameters that differ
│ ├─ commands:                    │
│ │  ├─ train_explicit: "python   │  ← ALL 60+ params explicit
│ │  │    train_gainakt2exp.py    │
│ │  │    --param1 val1 ...       │
│ │  ├─ eval_explicit: "python    │  ← ALL 20+ params explicit
│ │  │    eval_gainakt2exp.py..." │
│ │  └─ reproduce: "python run... │
│ └─ md5: ca1ef5c...              │  ← Hash of perfect defaults
└────────────┬────────────────────┘
             │
             ↓ (execute explicit command)
┌─────────────────────────────────┐
│ train_gainakt2exp.py            │  ← Training script
│ - Receives ALL params via CLI   │
│ - No config.json reading        │
│ - No hardcoded defaults         │
└─────────────────────────────────┘
```

### Training Mode vs Reproduction Mode

**Training Mode** (default):
```bash
python examples/run_repro_experiment.py \
  --short_title baseline --epochs 12 --batch_size 64
```
1. Loads 63 defaults from `configs/parameter_default.json`
2. Applies CLI overrides (epochs=12, batch_size=64)
3. Generates 6-digit experiment ID (e.g., `584063`)
4. Creates folder: `20251102_202046_gainakt2exp_baseline_584063/`
5. Builds explicit command with **all 60+ parameters**
6. Saves config.json with perfect defaults + overrides
7. Executes explicit training command

**Reproduction Mode**:
```bash
python examples/run_repro_experiment.py \
  --repro_experiment_id 584063
```
1. Searches for experiment folder containing `584063`
2. Loads original `config.json` (perfect defaults + original overrides)
3. Creates reproduction folder: `20251102_203015_gainakt2exp_584063_repro/`
4. Copies config.json **byte-for-byte unchanged**
5. Executes the **same explicit training command** from original

### Config.json Structure

Each experiment's config.json contains:

```json
{
  "input": {
    "dataset": "assist2015",
    "fold": 0,
    "model": "gainakt2exp",
    "train_script": "examples/train_gainakt2exp.py",
    "eval_script": "examples/eval_gainakt2exp.py"
  },
  "commands": {
    "run_repro_original": "python examples/run_repro_experiment.py --short_title baseline --epochs 12",
    "train_explicit": "EXPERIMENT_DIR=/path/to/exp python examples/train_gainakt2exp.py --dataset assist2015 --fold 0 --seed 42 --epochs 12 --batch_size 64 --learning_rate 0.000174 ... (60+ params)",
    "eval_explicit": "python examples/eval_gainakt2exp.py --run_dir /path/to/exp --max_correlation_students 300 --dataset assist2015 --fold 0 ... (20+ params)",
    "reproduce": "python examples/run_repro_experiment.py --repro_experiment_id 584063"
  },
  "experiment": {
    "id": "20251102_202046_gainakt2exp_baseline_584063",
    "short_title": "baseline",
    "experiment_id": "584063",
    "created": "20251102_202046"
  },
  "seeds": {
    "primary": 42,
    "all": [42]
  },
  "defaults": {
    "model": "gainakt2exp",
    "dataset": "assist2015",
    "fold": 0,
    "seed": 42,
    "epochs": 12,
    "batch_size": 64,
    ... (63 total perfect defaults)
  },
  "overrides": {
    "epochs": 12,
    "batch_size": 64
  },
  "types": { ... },
  "md5": "ca1ef5c58232b47017ec7d1ba70e17d7",
  "reference": {
    "parameter_default_json": "configs/parameter_default.json"
  }
}
```

**Key sections:**
- **`defaults`**: Pristine copy from `parameter_default.json` (unchanged)
- **`overrides`**: Only parameters that differ from defaults (CLI args)
- **`commands.train_explicit`**: Complete command with ALL parameters explicit
- **`commands.eval_explicit`**: Complete evaluation command with ALL architecture params
- **`md5`**: Hash of `defaults` section for tamper detection

### MD5 Integrity Verification

**Purpose:** Detect manual tampering of config.json defaults

**Computation:**
```python
import json, hashlib
defaults_str = json.dumps(config['defaults'], sort_keys=True)
md5 = hashlib.md5(defaults_str.encode()).hexdigest()
# Result: ca1ef5c58232b47017ec7d1ba70e17d7
```

**Verification flow:**
1. Launcher saves config.json with MD5 of perfect defaults
2. On reproduction/relaunch, recompute MD5 from `config['defaults']`
3. Compare with stored `config['md5']`
4. If mismatch → WARNING: defaults have been manually modified

**What it catches:**
- ✅ Manual edits to defaults in config.json
- ✅ Config corruption or file tampering
- ✅ Drift between parameter_default.json versions

**What it doesn't catch:**
- Changes to `overrides` section (intentional user modifications)
- Changes to `commands` section (regenerated on relaunch)
- Metadata changes (timestamps, folder paths)

### Experiment ID System

**Format:** 6-digit random number (100000-999999)

**Purpose:**
- Uniquely identifies each experiment
- Enables simple reproduction: `--repro_experiment_id XXXXXX`
- Persists across reproductions (same ID + `_repro` suffix)
- Easy reference in papers, discussions, logs

**Folder naming:**
```
YYYYMMDD_HHMMSS_modelname_shorttitle_XXXXXX
20251102_202046_gainakt2exp_baseline_584063       ← Original
20251102_203015_gainakt2exp_584063_repro          ← Reproduction
```

### Parameter Management

**Single Source of Truth:** `configs/parameter_default.json`

Structure:
```json
{
  "defaults": {
    "model": "gainakt2exp",
    "dataset": "assist2015",
    "fold": 0,
    "seed": 42,
    "train_script": "examples/train_gainakt2exp.py",
    "eval_script": "examples/eval_gainakt2exp.py",
    "epochs": 12,
    "batch_size": 64,
    "learning_rate": 0.000174,
    "weight_decay": 1e-05,
    "optimizer": "adam",
    "seq_len": 200,
    "d_model": 512,
    "n_heads": 8,
    ... (63 total parameters)
  },
  "types": {
    "seed": "int",
    "epochs": "int",
    "batch_size": "int",
    "learning_rate": "float",
    "use_mastery_head": "bool",
    ... (type information for validation)
  },
  "md5": "ca1ef5c58232b47017ec7d1ba70e17d7"
}
```

**Key parameters categories:**
1. **Runtime**: seed, epochs, batch_size, learning_rate, optimizer
2. **Architecture**: seq_len, d_model, n_heads, num_encoder_blocks, d_ff, dropout
3. **Embeddings**: emb_type, emb_size
4. **Constraints**: non_negative_loss_weight, monotonicity_loss_weight, mastery_performance_loss_weight, gain_performance_loss_weight, sparsity_loss_weight, consistency_loss_weight
5. **Interpretability**: use_mastery_head, use_gain_head, use_cumulative_mastery
6. **Evaluation**: max_correlation_students (eval-only, not used in training)
7. **Launcher-only**: model, train_script, eval_script (not passed to training script)

### Training Script Design

**File:** `examples/train_gainakt2exp.py`

**Key principles:**
- **Zero hardcoded defaults**: All argparse parameters use `required=True`
- **No config.json reading**: Only accepts CLI parameters
- **Architecture from CLI**: All model parameters (d_model, n_heads, etc.) from CLI

**Parameter count:** ~60 required parameters

**Example invocation** (generated by launcher):
```bash
EXPERIMENT_DIR=/path/to/exp python examples/train_gainakt2exp.py \
  --dataset assist2015 \
  --fold 0 \
  --seed 42 \
  --epochs 12 \
  --batch_size 64 \
  --learning_rate 0.000174 \
  --weight_decay 1e-05 \
  --optimizer adam \
  --seq_len 200 \
  --d_model 512 \
  --n_heads 8 \
  --num_encoder_blocks 6 \
  --d_ff 1024 \
  --dropout 0.2 \
  --emb_type qid \
  --non_negative_loss_weight 0.0 \
  --monotonicity_loss_weight 0.1 \
  --mastery_performance_loss_weight 0.8 \
  --gain_performance_loss_weight 0.8 \
  --sparsity_loss_weight 0.2 \
  --consistency_loss_weight 0.3 \
  --use_mastery_head \
  --use_gain_head \
  --use_cumulative_mastery \
  ... (40+ more parameters)
```

**Why explicit parameters?**
- Complete provenance: See ALL hyperparameters in one command
- No hidden state: What you see is what you get
- Perfect reproduction: Copy-paste the command to rerun
- Version safety: No risk of default value drift over time

### Evaluation Script Design

**File:** `examples/eval_gainakt2exp.py`

**Key principles:**
- **Zero hardcoded defaults**: All argparse parameters use `required=True`
- **No config.json reading**: Only accepts CLI parameters (like training)
- **Architecture must match training**: Requires same d_model, n_heads, etc.

**Parameter count:** ~20 required parameters

**Why architecture params needed:**
- Model checkpoint contains weights but not architecture metadata
- Evaluation must recreate exact model structure to load weights
- Architecture mismatch → model loading failure

**Example invocation** (generated by launcher):
```bash
python examples/eval_gainakt2exp.py \
  --run_dir /path/to/experiment \
  --max_correlation_students 300 \
  --dataset assist2015 \
  --fold 0 \
  --batch_size 64 \
  --seq_len 200 \
  --d_model 512 \
  --n_heads 8 \
  --num_encoder_blocks 6 \
  --d_ff 1024 \
  --dropout 0.2 \
  --emb_type qid \
  --non_negative_loss_weight 0.0 \
  --monotonicity_loss_weight 0.1 \
  --mastery_performance_loss_weight 0.8 \
  --gain_performance_loss_weight 0.8 \
  --sparsity_loss_weight 0.2 \
  --consistency_loss_weight 0.3 \
  --use_mastery_head \
  --use_gain_head
```

**Best practice:** Copy `eval_explicit` command from experiment's config.json

### Launcher Parameter Filtering

The launcher (`run_repro_experiment.py`) excludes certain parameters from training/evaluation:

**Launcher-only parameters** (excluded from training):
- `model`: Used for folder naming and script selection
- `train_script`: Path to training script
- `eval_script`: Path to evaluation script
- `max_correlation_students`: Evaluation-only parameter

**Training parameters:** 59 (63 total - 4 launcher-only)
**Evaluation parameters:** 20 (architecture + constraints + eval-specific)

**Why exclude?**
- Training script doesn't have `--model` argparse (model is hardcoded as gainakt2exp)
- `max_correlation_students` only used in evaluation (correlation sampling limit)
- Keeps training command clean and focused

### Reproducibility Checklist

Each experiment folder **MUST** contain these artifacts:

| File/Dir | Purpose | Verification |
|----------|---------|--------------|
| `config.json` | Complete parameter record + explicit commands | MD5 integrity check |
| `environment.txt` | Python/PyTorch/CUDA versions, git commit | Version match |
| `SEED_INFO.md` | Seeds used and rationale | Determinism audit |
| `stdout.log` | Raw console output with timestamps | Training trajectory |
| `stderr.log` | Error output (if any) | Debug failures |
| `metrics_epoch.csv` | Per-epoch metrics (loss, AUC, correlations) | Performance tracking |
| `results.json` | Best epoch metrics + final summary | Key results |
| `model_best.pth` | Best checkpoint (by validation AUC) | Evaluation/deployment |
| `model_last.pth` | Last epoch checkpoint | Recovery/debugging |
| `README.md` | Human-readable summary + results table | Documentation |

**Missing artifacts = Invalid reproducibility claim**

### Common Workflows

**Workflow 1: Train → Evaluate → Compare**
```bash
# 1. Train new experiment
python examples/run_repro_experiment.py \
  --short_title new_arch \
  --epochs 12 \
  --batch_size 128
# → Creates: 20251102_150000_gainakt2exp_new_arch_123456/

# 2. Evaluate (copy eval_explicit from config.json)
cd examples/experiments/20251102_150000_gainakt2exp_new_arch_123456
cat config.json | grep eval_explicit
# → Copy and run the eval command

# 3. Compare with baseline
python examples/compare_reproduction.py 123456
```

**Workflow 2: Reproduce for Verification**
```bash
# 1. Reproduce experiment 423891
python examples/run_repro_experiment.py \
  --repro_experiment_id 423891
# → Creates: 20251102_160000_gainakt2exp_423891_repro/

# 2. Compare metrics
python examples/compare_reproduction.py 423891
# → Shows original vs reproduction side-by-side
```

**Workflow 3: Parameter Sweep**
```bash
# Train multiple configurations
for lr in 0.0001 0.00017 0.0003; do
  python examples/run_repro_experiment.py \
    --short_title lr_${lr} \
    --learning_rate ${lr}
done

# Compare results in examples/experiments/RESULTS.csv
```

**Workflow 4: Debug Reproduction Failure**
```bash
# 1. Check config MD5 integrity
cd examples/experiments/20251102_150000_gainakt2exp_test_123456
python -c "
import json, hashlib
config = json.load(open('config.json'))
computed = hashlib.md5(json.dumps(config['defaults'], sort_keys=True).encode()).hexdigest()
print('Stored MD5:  ', config['md5'])
print('Computed MD5:', computed)
print('Match:', config['md5'] == computed)
"

# 2. Check environment consistency
cat environment.txt
git log -1 --oneline

# 3. Compare explicit commands
cat config.json | jq '.commands.train_explicit'

# 4. Re-run with same seed
# Use the exact train_explicit command from config.json
```

### Config Integrity: MD5 Deep Dive

**Why MD5 for defaults?**
- Detect manual tampering of config.json
- Ensure perfect defaults match parameter_default.json
- Catch config corruption or file damage
- Prevent silent drift in reproducibility claims

**How it works:**
1. Launcher loads `configs/parameter_default.json`
2. Copies perfect defaults to `config.json['defaults']`
3. Computes MD5: `md5(json.dumps(defaults, sort_keys=True))`
4. Stores in `config.json['md5']`
5. On reproduction: recompute MD5 and compare
6. Mismatch → WARNING: defaults have been modified

**What triggers mismatch:**
- ✅ Manual edit of `config.json['defaults']` values
- ✅ Corruption during file transfer
- ✅ Different parameter_default.json version
- ❌ Changes to `overrides` section (not checked)
- ❌ Changes to `commands` section (not checked)
- ❌ Metadata changes (timestamps, paths)

**Current MD5:** `ca1ef5c58232b47017ec7d1ba70e17d7`

**Verification example:**
```python
import json, hashlib

# Load parameter_default.json
defaults = json.load(open('configs/parameter_default.json'))

# Compute MD5
defaults_str = json.dumps(defaults['defaults'], sort_keys=True)
md5 = hashlib.md5(defaults_str.encode()).hexdigest()

print(md5)  # Should match: ca1ef5c58232b47017ec7d1ba70e17d7
```

### Parameter Evolution Protocol

When adding/modifying parameters in `configs/parameter_default.json`:

**Step 1: Update parameter_default.json**
```bash
# Edit configs/parameter_default.json
# Add new parameter or change existing default
```

**Step 2: Recompute MD5**
```bash
python -c "
import json, hashlib
data = json.load(open('configs/parameter_default.json'))
md5 = hashlib.md5(json.dumps(data['defaults'], sort_keys=True).encode()).hexdigest()
data['md5'] = md5
json.dump(data, open('configs/parameter_default.json', 'w'), indent=2)
print(f'Updated MD5: {md5}')
"
```

**Step 3: Update training/evaluation scripts**
```bash
# Add argparse parameter to train_gainakt2exp.py and/or eval_gainakt2exp.py
# Ensure required=True (no hardcoded defaults)
```

**Step 4: Test with dry run**
```bash
python examples/run_repro_experiment.py \
  --short_title test_new_param \
  --epochs 1
# Check that new parameter appears in config.json
```

**Step 5: Document in README**
```bash
# Update this README with parameter description
# Add to parameter categories section
```

**Failure to follow protocol:**
- Training script rejects unknown parameter
- MD5 mismatch warnings
- Reproducibility chain broken
- Evaluation may fail (architecture mismatch)

- Adding first-time parameter (no prior baseline exists)
- Modifying architecture parameters (affects evaluation compatibility)
- Changing parameter structure in parameter_default.json
- After launcher code changes
- Before multi-seed production runs (catches issues before scaling)
- Before paper submission (validates reproducibility claims)
