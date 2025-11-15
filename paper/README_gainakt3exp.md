# GainAKT3Exp


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
4. Creates experiment folder: `20251102_143210_gainakt3exp_baseline_423891/`
5. Builds **explicit training command** with ALL parameters:
   ```bash
   python examples/train_gainakt3exp.py \
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
cd examples/experiments/20251102_143210_gainakt3exp_baseline_423891
# Copy the eval_explicit command from config.json and run it
```

Evaluation command set all parameters explicitly, example: 
```bash
python examples/eval_gainakt3exp.py \
  --run_dir examples/experiments/20251102_143210_gainakt3exp_baseline_423891 \
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
3. Creates reproduction folder: `20251102_154320_gainakt3exp_423891_repro/`
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


## Trajectories

The trajectory system captures the evolution of semantic metrics throughout training, providing insight into how the model learns to align internal representations (mastery and gains) with observed student performance. Trajectories are recorded per-epoch and saved as JSON files for post-training analysis.

### Purpose

Trajectories serve multiple analysis objectives:

1. **Training Diagnostics**: Monitor correlation evolution to detect learning progress or stagnation
2. **Loss Decomposition**: Track how different loss components contribute to training dynamics
3. **Architectural Validation**: Verify that semantic modules (alignment, retention, lag-gain) produce expected effects
4. **Ablation Studies**: Compare trajectory patterns across architectural variants to isolate feature contributions
5. **Interpretability Research**: Provide time-series data for analyzing how semantic constraints shape model behavior

### Data Structure

The trajectory is stored under `train_history.semantic_trajectory` as a list of per-epoch dictionaries. Each epoch entry contains both **evolving metrics** (that change as training progresses) and **configuration parameters** (that remain constant or follow predetermined schedules).

#### Evolving Metrics (Change Each Epoch)

**Correlation Metrics** - Measure alignment between model internals and student performance:
- `mastery_correlation`: Pearson correlation between mastery states and performance (50 sampled students)
- `gain_correlation`: Pearson correlation between learning gains and performance (50 sampled students)
- `alignment_corr_mastery`: Local alignment loss mastery correlation (batch-level, last batch of epoch)
- `alignment_corr_gain`: Local alignment loss gain correlation (batch-level, last batch of epoch)
- `global_alignment_mastery_corr`: Population-level mastery correlation (600 students)
- `global_alignment_gain_corr`: Population-level gain correlation (600 students)
- `peak_mastery_corr`: Highest mastery correlation achieved up to this epoch (monotonically increasing)

**Loss Decomposition** (`loss_shares` sub-object) - Proportional contribution of each loss component:
- `main`: BCE loss proportion (typically 50-75%, decreases as constraints activate)
- `constraint_total`: Combined constraint loss proportion (increases during warmup)
- `alignment`: Alignment loss proportion (can be negative when anti-correlated)
- `lag`: Lag-gain loss proportion (typically 0 or small positive)
- `retention`: Retention loss proportion (0 until post-peak decay occurs)

**Variance Statistics** - Mastery distribution characteristics:
- `mean_mastery_variance`: Average variance in mastery states across sampled students
- `min_mastery_variance`: Minimum variance observed across skills
- `max_mastery_variance`: Maximum variance observed across skills

**Lag-Gain Metrics** (if lag-gain loss enabled):
- `mean_lag_corr`: Average correlation across lag positions (1, 2, 3)
- `lag_corr_count`: Number of valid lag correlations computed
- `per_lag_correlations`: List of correlations for each lag position

**Retention Metrics** (if retention loss enabled):
- `retention_loss_value`: Magnitude of retention loss at this epoch

#### Configuration Parameters (Constant or Scheduled)

**Warmup Schedule** - These follow predetermined warmup schedules (constant after warmup completes):
- `warmup_scale`: Constraint loss multiplier (ramps 0.0 → 1.0 over `warmup_constraint_epochs`, then stays 1.0)
- `effective_alignment_weight`: Alignment loss weight after adaptive adjustments (follows schedule)

**Loss Weights** - These are typically constant (unless dynamic rebalancing is enabled):
- `consistency_loss_weight_current`: Consistency loss weight (usually constant at 0.3)
- `sparsity_loss_weight_current`: Sparsity loss weight (usually constant at 0.2)

#### Summary: What Actually Changes?

**True evolving metrics** (reflect model learning):
- Correlations (mastery, gain, alignment, global)
- Loss shares (change as different components activate/deactivate)
- Variance statistics (change as model learns better representations)
- Peak mastery correlation (cumulative maximum)
- Retention/lag values (respond to model state)

**Scheduled parameters** (follow predetermined paths):
- `warmup_scale`: 0.125 → 0.25 → 0.375 → ... → 1.0 (linear ramp over 8 epochs)
- `effective_alignment_weight`: Follows adaptive schedule based on alignment share cap

**Constant parameters** (recorded for convenience but don't change):
- `consistency_loss_weight_current`: 0.3 (unless dynamic rebalancing enabled)
- `sparsity_loss_weight_current`: 0.2 (constant)

The trajectory's value is in tracking the **evolving metrics** to see if semantic constraints are working (correlations improving) and how loss components balance over time (loss shares stabilizing).

### File Locations and Naming

Trajectory data is saved embedded within experiment result files in the experiment directory:

1. **Reproducibility Results File** (primary):
   - **Path**: `examples/experiments/{experiment_id}/repro_results_{timestamp}.json`
   - **Key**: `train_history.semantic_trajectory`
   - **Format**: Complete trajectory embedded within reproducibility results
   - **Purpose**: Self-contained reproducibility record with all training artifacts
   - **Example**: `examples/experiments/20251115_164618_gainakt3exp_baseline_defaults_114045/repro_results_20251115_170523.json`

2. **Legacy Results File** (compatibility):
   - **Path**: `examples/experiments/{experiment_id}/results.json`
   - **Key**: `train_history.semantic_trajectory`
   - **Format**: Trajectory embedded within legacy format results
   - **Purpose**: Backward compatibility with prior experiment analysis tools
   - **Example**: `examples/experiments/20251115_164618_gainakt3exp_baseline_defaults_114045/results.json`

All trajectory data resides within the experiment directory to maintain reproducibility containment and avoid scattered artifacts.

### Script Integration

Trajectories are calculated and saved by `examples/train_gainakt3exp.py`:

**Initialization** (line 699):
```python
train_history['semantic_trajectory'] = []
```

**Per-Epoch Data Collection** (lines 1418-1448):
After each epoch's training and validation, a trajectory entry is constructed from:
- Consistency validation metrics (mastery/gain correlations from 50 sampled students)
- Loss decomposition statistics (shares computed from logged loss components)
- Global alignment pass results (600 students, if enabled)
- Retention loss values
- Lag-gain correlation statistics
- Variance metrics from mastery states

**Saving** (lines 1630-1645):
At training completion, the trajectory is embedded in two JSON files within the experiment directory:
1. `repro_results_{timestamp}.json` - Primary reproducibility format
2. `results.json` - Legacy compatibility format

Both files contain the complete trajectory under `train_history.semantic_trajectory`.

### Access and Analysis

To analyze trajectories:

1. **Load from experiment results file**:
   ```python
   import json
   # Load from repro_results file
   with open('examples/experiments/{experiment_id}/repro_results_{timestamp}.json', 'r') as f:
       results = json.load(f)
   trajectory = results['train_history']['semantic_trajectory']
   
   # Or load from legacy results.json
   with open('examples/experiments/{experiment_id}/results.json', 'r') as f:
       results = json.load(f)
   trajectory = results['train_history']['semantic_trajectory']
   ```

2. **Extract specific metrics**:
   ```python
   epochs = [entry['epoch'] for entry in trajectory]
   mastery_corrs = [entry['mastery_correlation'] for entry in trajectory]
   gain_corrs = [entry['gain_correlation'] for entry in trajectory]
   ```

3. **Plot correlation evolution**:
   ```python
   import matplotlib.pyplot as plt
   plt.plot(epochs, mastery_corrs, label='Mastery Correlation')
   plt.plot(epochs, gain_corrs, label='Gain Correlation')
   plt.xlabel('Epoch')
   plt.ylabel('Pearson Correlation')
   plt.legend()
   plt.show()
   ```

4. **Analyze loss decomposition**:
   ```python
   main_shares = [entry['loss_shares']['main'] for entry in trajectory]
   constraint_shares = [entry['loss_shares']['constraint_total'] for entry in trajectory]
   ```

### Example Trajectory Entry

```json
{
  "epoch": 2,
  "mastery_correlation": 0.0712,
  "gain_correlation": 0.0433,
  "warmup_scale": 0.25,
  "alignment_corr_mastery": 0.0357,
  "alignment_corr_gain": 0.0291,
  "global_alignment_mastery_corr": 0.0359,
  "global_alignment_gain_corr": 0.0291,
  "effective_alignment_weight": 0.0375,
  "peak_mastery_corr": 0.0712,
  "retention_loss_value": 0.0,
  "mean_lag_corr": null,
  "lag_corr_count": 0,
  "consistency_loss_weight_current": 0.3,
  "sparsity_loss_weight_current": 0.2,
  "loss_shares": {
    "main": 0.679,
    "constraint_total": 0.321,
    "alignment": -0.006,
    "lag": 0.0,
    "retention": 0.0
  },
  "mean_mastery_variance": 0.142,
  "min_mastery_variance": 0.001,
  "max_mastery_variance": 0.385,
  "per_lag_correlations": []
}
```

### Trajectory Configuration Parameters

Trajectory behavior is controlled by several parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_semantic_students` | 50 | Number of students sampled for per-epoch correlation computation |
| `alignment_global_students` | 600 | Number of students sampled for global alignment pass |

Note: Trajectory calculation has minimal performance overhead (< 2 seconds per epoch on assist2015) and is always enabled. Trajectories are automatically embedded in experiment result files.
