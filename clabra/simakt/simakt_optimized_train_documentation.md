# SimAKT Optimized Training Script Documentation

This document provides comprehensive guidance for using the parameterized SimAKT training script (`simakt_optimized_train.py`) with configurable resource management for different computing environments.

## Overview

The SimAKT optimized training script allows you to train the SimAKT (Similarity-based Attention Knowledge Tracing) model with flexible resource configuration. The script automatically adapts to your hardware capabilities or can be manually configured for shared computing environments.

## Features

- **Parameterized Resource Management**: Configure CPU threads, data workers, and batch sizes
- **Preset Resource Modes**: Easy-to-use presets for different computing environments
- **Manual Override Options**: Fine-grained control over all resource parameters
- **Shared Machine Safe**: Conservative defaults prevent system overload
- **Hardware Detection**: Automatically detects available CPU cores and memory
- **Progress Monitoring**: Configurable progress reporting frequency

## Usage

### Basic Usage

```bash
# Default conservative mode (recommended for shared machines)
python assistant/simakt_optimized_train.py --dataset_name=assist2015 --use_wandb=0
```

### Resource Mode Presets

#### Minimal Resources (Safest for Heavily Shared Machines)
```bash
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=minimal \
    --use_wandb=0
```
- **Batch Size**: 16
- **Workers**: 0 (no multiprocessing)
- **CPU Threads**: 1
- **Use Case**: Heavily loaded shared machines, containers with strict limits

#### Conservative Resources (Default - Good for Shared Machines)
```bash
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=conservative \
    --use_wandb=0
```
- **Batch Size**: 64
- **Workers**: 2
- **CPU Threads**: 2
- **Use Case**: Shared machines with moderate resource availability

#### Moderate Resources
```bash
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=moderate \
    --use_wandb=0
```
- **Batch Size**: 256
- **Workers**: 8 (or half of available cores)
- **CPU Threads**: 8 (or half of available cores)
- **Use Case**: Lightly shared machines or dedicated development environments

#### Aggressive Resources (Maximum Performance)
```bash
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=aggressive \
    --use_wandb=0
```
- **Batch Size**: 1024
- **Workers**: 16 (or 75% of available cores)
- **CPU Threads**: 20 (or 50% of available cores)
- **Use Case**: Dedicated machines with abundant resources

### Manual Resource Configuration

Override specific resource parameters for fine-grained control:

```bash
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --batch_size=32 \
    --num_workers=1 \
    --cpu_threads=2 \
    --progress_freq=100 \
    --use_wandb=0
```

## Command-Line Arguments

### Core Training Parameters
- `--dataset_name`: Dataset to use (default: "assist2015")
- `--model_name`: Model name (default: "simakt")
- `--emb_type`: Embedding type (default: "qid_cl")
- `--save_dir`: Directory to save models (default: "saved_model")
- `--seed`: Random seed for reproducibility (default: 3407)
- `--fold`: Cross-validation fold (default: 0)
- `--use_wandb`: Enable Weights & Biases logging (default: 0)

### Model Architecture Parameters
- `--d_model`: Model dimension (default: 256)
- `--d_ff`: Feed-forward dimension (default: 256)
- `--num_attn_heads`: Number of attention heads (default: 8)
- `--n_blocks`: Number of transformer blocks (default: 4)
- `--dropout`: Dropout rate (default: 0.3)
- `--n_know`: Number of knowledge states (default: 16)
- `--lambda_cl`: Contrastive learning weight (default: 0.1)
- `--window`: Attention window size (default: 1)
- `--proj`: Use projection layer (default: True)
- `--hard_neg`: Use hard negatives in contrastive learning (default: False)
- `--learning_rate`: Learning rate (default: 0.001)

### Resource Management Parameters
- `--resource_mode`: Preset resource mode ("minimal", "conservative", "moderate", "aggressive")
- `--batch_size`: Override automatic batch size calculation
- `--num_workers`: Override automatic worker count calculation
- `--cpu_threads`: Override automatic CPU thread calculation
- `--progress_freq`: Progress reporting frequency in batches (default: 200)

## Resource Mode Details

| Parameter | Minimal | Conservative | Moderate | Aggressive |
|-----------|---------|--------------|----------|------------|
| Batch Size | 16 | 64 | 256 | 1024 |
| Data Workers | 0 | 2 | min(8, cores/2) | min(16, cores*0.75) |
| CPU Threads | 1 | 2 | min(8, cores/2) | min(20, cores*0.5) |
| Memory Usage | Very Low | Low | Medium | High |
| Training Speed | Slowest | Slow | Fast | Fastest |
| System Impact | Minimal | Low | Medium | High |

## Environment-Specific Recommendations

### Shared Computing Clusters
```bash
# Start with minimal resources
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=minimal \
    --use_wandb=0

# If successful, try conservative mode
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=conservative \
    --use_wandb=0
```

### Personal Development Machine
```bash
# Use moderate resources for good balance
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=moderate \
    --use_wandb=0
```

### High-Performance Computing (HPC)
```bash
# Use aggressive mode for maximum performance
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=aggressive \
    --use_wandb=0
```

### Container Environments (Docker/Singularity)
```bash
# Start with conservative mode
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=conservative \
    --progress_freq=500 \
    --use_wandb=0
```

## Advanced Usage Examples

### Custom Resource Allocation
```bash
# Fine-tuned for 8-core machine with limited memory
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --batch_size=128 \
    --num_workers=4 \
    --cpu_threads=6 \
    --progress_freq=50 \
    --use_wandb=0
```

### Memory-Constrained Environment
```bash
# Small batch size with minimal workers
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --batch_size=16 \
    --num_workers=0 \
    --cpu_threads=1 \
    --progress_freq=1000 \
    --use_wandb=0
```

### Quick Testing/Debugging
```bash
# Minimal resources with frequent progress updates
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=minimal \
    --progress_freq=10 \
    --use_wandb=0
```

## Output and Monitoring

### Console Output
The script provides detailed information about resource allocation:
```
Resource Configuration:
  Resource mode: conservative
  Available CPU cores: 40
  Using data workers: 2
  Batch size: 64
  Computation threads: 2
  Progress frequency: 200

Training SimAKT model (Conservative Resource Mode)
Dataset: assist2015
Save directory: saved_model/assist2015_simakt_qid_cl_conservative_3407
```

### Progress Monitoring
Training progress is reported at configurable intervals:
```
Epoch 1/200 - Training...
  Batch 0: Loss 0.6931
  Batch 200: Loss 0.4523
  Batch 400: Loss 0.3847
Epoch 1/200 - Validating...
Epoch 1/200
  Train Loss: 0.4234
  Valid AUC: 0.7123
  Valid ACC: 0.6789
  New best model saved! AUC: 0.7123
```

### Saved Outputs
- **Model Checkpoint**: `saved_model/{dataset}_{model}_{emb_type}_{mode}_{seed}/simakt_model.ckpt`
- **Configuration**: `saved_model/{dataset}_{model}_{emb_type}_{mode}_{seed}/config.json`

## Troubleshooting

### Process Killed/Out of Memory
1. Reduce resource mode: `aggressive → moderate → conservative → minimal`
2. Manually set smaller batch size: `--batch_size=16`
3. Disable multiprocessing: `--num_workers=0`
4. Reduce CPU threads: `--cpu_threads=1`

### Slow Training
1. Increase resource mode if system allows
2. Increase batch size if memory permits
3. Enable more workers if CPU cores available
4. Reduce progress reporting frequency: `--progress_freq=1000`

### Resource Contention on Shared Systems
```bash
# Ultra-conservative settings
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --batch_size=8 \
    --num_workers=0 \
    --cpu_threads=1 \
    --progress_freq=500 \
    --use_wandb=0
```

## Performance Optimization Tips

1. **Start Conservative**: Begin with `minimal` or `conservative` mode and scale up
2. **Monitor System Resources**: Use `top` or `htop` to monitor CPU and memory usage
3. **Test Incrementally**: Gradually increase resources until you find the optimal setting
4. **Consider I/O**: More workers help with data loading but may cause I/O bottlenecks
5. **Memory vs Speed**: Larger batch sizes are faster but require more memory

## Integration with PyKT Framework

This script follows PyKT framework conventions:
- Model saved in standard PyKT checkpoint format
- Configuration saved for reproducibility
- Compatible with PyKT evaluation scripts
- Follows GEMINI.md guidelines for non-intrusive changes

## Example Training Sessions

### Session 1: Finding Optimal Settings
```bash
# Step 1: Test minimal (should always work)
python assistant/simakt_optimized_train.py --dataset_name=assist2015 --resource_mode=minimal --use_wandb=0

# Step 2: Try conservative (if minimal worked)
python assistant/simakt_optimized_train.py --dataset_name=assist2015 --resource_mode=conservative --use_wandb=0

# Step 3: Try moderate (if conservative worked)
python assistant/simakt_optimized_train.py --dataset_name=assist2015 --resource_mode=moderate --use_wandb=0
```

### Session 2: Production Training
```bash
# Use the optimal setting found in Session 1
python assistant/simakt_optimized_train.py \
    --dataset_name=assist2015 \
    --resource_mode=conservative \
    --progress_freq=100 \
    --use_wandb=1 \
    --seed=42
```

This documentation provides comprehensive guidance for using the parameterized SimAKT training script across different computing environments while maintaining system stability and optimal performance.