# GainAKT2Exp


## Quick Start

We follow a reproducibility approach based on Explicit Parameters, Zero Defaults described in detail in ´examples/reproducibility.md´ The system enforces **zero hidden defaults**: all training parameters and evaluation parameters must be explicitly specified via command line. A single launcher script (`run_repro_experiment.py`) manages the complete workflow.


**Quick Commands:**
```bash
# Train
python examples/run_repro_experiment.py --short_title test --epochs 12

# Reproduce
python examples/run_repro_experiment.py --repro_experiment_id 584063

# Evaluate (copy eval_explicit command from config.json)
cd examples/experiments/[experiment_folder]
cat config.json | grep eval_explicit
# → Copy and run the command

# Compare
python examples/compare_reproduction.py 584063
```

## Commands

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




## Parameters

### Current Parameter Structure

All parameters are stored in a **flat structure** in `configs/parameter_default.json`. The launcher (`run_repro_experiment.py`) loads these defaults and generates explicit commands with all parameters visible.

See ´paper/parameters.csv´ for a description of each parameter. 

