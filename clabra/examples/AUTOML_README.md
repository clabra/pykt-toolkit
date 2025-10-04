# 🚀 GainAKT2 AutoML Hyperparameter Optimizer

Advanced AutoML system for automatically finding optimal GainAKT2 parameters using multiple optimization strategies.

## 🎯 Overview

This AutoML system starts from the current best known parameters (AUC: 0.7233) and intelligently searches for even better combinations using:

- **Bayesian Optimization** with Gaussian Processes
- **Adaptive Parameter Sampling** around baseline 
- **Smart Grid Search** with intelligent prioritization
- **Early Stopping** to avoid wasted computation
- **Parallel Evaluation** for faster results

## 📊 Starting Baseline (Current Best)

```yaml
d_model: 256
learning_rate: 0.0002
dropout: 0.2
num_encoder_blocks: 4
d_ff: 768            # Key improvement from previous optimization
n_heads: 8
Baseline AUC: 0.7233 (72.33%)
```

## 🚀 Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
cd /workspaces/pykt-toolkit/examples

# Quick test (20 evaluations, ~30 minutes)
./launch_automl.sh quick

# Standard optimization (50 evaluations, ~2 hours) 
./launch_automl.sh standard

# Intensive search (100 evaluations, ~4 hours)
./launch_automl.sh intensive

# Overnight run (200 evaluations, ~8+ hours)
./launch_automl.sh overnight

# Custom configuration
./launch_automl.sh custom
```

### Option 2: Direct Python Usage
```bash
cd /workspaces/pykt-toolkit/examples

# Standard configuration
python automl_gainakt2_optimizer.py \
    --max_evaluations 50 \
    --parallel_jobs 2 \
    --early_stopping_patience 10 \
    --target_auc 0.735

# Custom configuration
python automl_gainakt2_optimizer.py \
    --max_evaluations 100 \
    --parallel_jobs 3 \
    --early_stopping_patience 15 \
    --target_auc 0.740
```

## 🧠 Optimization Strategies

### Phase 1: Bayesian Optimization (40% of budget)
- Uses Gaussian Process to model parameter-performance relationship
- Intelligently explores promising regions 
- Balances exploration vs exploitation
- Most likely to find global optimum

### Phase 2: Fine-tuning Grid Search (30% of budget)  
- Focused grid around baseline parameters
- Tests small variations that might be missed
- Validates Bayesian findings
- Ensures local optimum exploration

### Phase 3: Exploration Random Search (30% of budget)
- Adaptive sampling with 3 phases:
  - Close to baseline (local search)
  - Medium exploration 
  - Wide exploration
- Prevents getting stuck in local minima

## 🎛️ Parameter Ranges

### Fine-tuning Ranges (Phase 2)
```yaml
d_model: [192, 224, 256, 288, 320, 384]      # Around 256
learning_rate: [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4]  # Around 2e-4  
dropout: [0.15, 0.18, 0.2, 0.22, 0.25]      # Around 0.2
num_encoder_blocks: [3, 4, 5]                # Around 4
d_ff: [512, 640, 768, 896, 1024]             # Around 768
n_heads: [6, 8, 10, 12]                      # Around 8
```

### Exploration Ranges (Phase 3)
```yaml
d_model: [128, 160, 192, 256, 320, 384, 448, 512]
learning_rate: [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]
dropout: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
num_encoder_blocks: [2, 3, 4, 5, 6]
d_ff: [256, 384, 512, 768, 1024, 1280, 1536] 
n_heads: [4, 6, 8, 10, 12, 16]
```

## 🛡️ Built-in Safeguards

### Parameter Validation
- Ensures `d_model` is divisible by `n_heads`
- Memory constraint checking (50M parameter limit)
- Reasonable range validation
- Invalid combinations are automatically skipped

### Early Stopping
- Stops if target AUC is achieved 
- Stops if no improvement for N evaluations
- Prevents wasted computation on poor regions
- Configurable patience parameter

### Timeout Protection  
- 10-minute timeout per evaluation
- Prevents hanging on problematic configurations
- Automatic failure handling and logging

## 📈 Expected Outcomes

### Realistic Targets
- **Conservative**: AUC > 0.7250 (+0.0017 improvement)
- **Optimistic**: AUC > 0.7280 (+0.0047 improvement) 
- **Stretch Goal**: AUC > 0.7350 (+0.0117 improvement)

### Success Indicators
- New configurations beating baseline 0.7233
- Consistent improvements across multiple runs
- Discovery of new parameter combinations
- Validation of current best practices

## 📊 Results and Logging

### Output Files
```
examples/automl_results/
├── automl_results_YYYYMMDD_HHMMSS.json    # Detailed results
├── automl_log_YYYYMMDD_HHMMSS.log         # Execution log  
└── best_command_YYYYMMDD_HHMMSS.sh        # Reproduce best result
```

### Real-time Monitoring
- Live progress updates with AUC scores
- Best result tracking with improvement metrics
- Strategy performance comparison
- Failure rate and timeout monitoring

### Final Report Includes
- Optimization summary with timing
- Best parameters and performance
- Improvement over baseline  
- Success/failure statistics
- Reproduction command for best result

## 🔧 Configuration Options

### Command Line Arguments
```bash
--max_evaluations INT     # Maximum combinations to test (default: 50)
--parallel_jobs INT       # Concurrent training jobs (default: 2)  
--early_stopping_patience INT  # Stop after N no improvements (default: 10)
--target_auc FLOAT        # Stop when this AUC is achieved (default: 0.735)
```

### Resource Considerations
- Each evaluation takes ~3-8 minutes depending on parameters
- Memory usage scales with `d_model * d_ff * num_encoder_blocks`
- Parallel jobs increase CPU/GPU utilization
- Larger models may require more time/memory

## 🚨 Troubleshooting

### Common Issues
1. **Import Error**: scikit-optimize installation (handled automatically)
2. **Memory Issues**: Reduce `parallel_jobs` or `max_evaluations`  
3. **Timeout Errors**: Check GPU availability and dataset loading
4. **No Improvements**: Try longer runs or different parameter ranges

### Performance Tuning
- Increase `parallel_jobs` on multi-GPU systems
- Reduce `max_evaluations` for quick testing
- Adjust `target_auc` based on realistic expectations
- Monitor system resources during execution

## 🏆 Success Stories

The system builds on previous optimization work that achieved:
- **AUC: 0.7233** with optimal d_ff=768 discovery  
- **Key insight**: d_ff=768 significantly outperformed 512 and 1024
- **Validation**: 4 encoder blocks optimal for depth vs efficiency
- **Learning rate**: 2e-4 consistently optimal across configurations

## 💡 Tips for Best Results

1. **Start with `standard` mode** for good balance of speed/thoroughness
2. **Use `intensive` mode** if you have time and want comprehensive search
3. **Monitor the logs** to understand which strategies work best
4. **Run multiple times** with different random seeds for robustness
5. **Combine results** from multiple runs for ensemble insights

## 🎯 Next Steps After Optimization

1. **Validate best result** with longer training (10+ epochs)
2. **Test on multiple datasets** to ensure generalization  
3. **Ensemble multiple** good configurations for robustness
4. **Document insights** from parameter relationships discovered
5. **Update baseline** for future optimization rounds

---

**Happy Optimizing! 🚀**

For questions or issues, check the logs in `automl_results/` directory.