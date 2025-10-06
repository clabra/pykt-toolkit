# GainAKT2 Training-Time Interpretability Monitoring Implementation

## Overview

This implementation provides a comprehensive solution for training GainAKT2 models with **real-time interpretability constraint monitoring** during the training process, addressing the limitations of post-hoc interpretability analysis.

## Key Components

### 1. GainAKT2Monitored Model (`pykt/models/gainakt2_monitored.py`)

Enhanced version of GainAKT2 that supports training-time monitoring:

**New Features:**
- `forward_with_states()` method that returns internal states for monitoring
- Auxiliary loss computation based on interpretability constraints
- Integration hooks for training-time monitoring
- Built-in support for interpretability projection heads

**Auxiliary Loss Functions:**
- **Non-negative gains constraint**: Penalizes negative learning gains (forgetting)
- **Consistency constraint**: Ensures gains align with performance changes

### 2. InterpretabilityMonitor (`examples/interpretability_monitor.py`)

Training-time monitoring hook that tracks 4 key interpretability constraints:

**Monitored Constraints:**
1. **Mastery-Performance Correlation**: Skill mastery should correlate positively with performance on that skill
2. **Gain-Correctness Correlation**: Learning gain magnitude should correlate with response correctness  
3. **Non-Negative Gains**: Learning gains should be non-negative (no forgetting)
4. **Mastery Monotonicity**: Mastery should generally increase after correct responses

**Features:**
- Real-time metric calculation during training
- Configurable monitoring frequency
- Logging to console and wandb
- Exception-safe wandb integration

### 3. Enhanced Training Script (`train_gainakt2_monitored.py`)

Complete training pipeline with interpretability monitoring:

**Key Features:**
- Integration of GainAKT2Monitored with InterpretabilityMonitor
- Combined loss function (main BCE + auxiliary interpretability losses)
- Real-time logging of both performance and interpretability metrics
- Gradient clipping for training stability
- Model checkpointing based on validation AUC

**Training Configuration:**
- Optimal hyperparameters from previous experiments
- Auxiliary loss weights: non_negative=0.1, consistency=0.05
- Monitoring frequency: every 25 batches
- 15 epochs with StepLR scheduling

### 4. Configuration Management (`interpretability_config.py`)

Comprehensive configuration system:

**Configuration Sections:**
- Model architecture parameters
- Training hyperparameters  
- Interpretability monitoring settings
- Performance benchmarks and targets
- Wandb logging configuration

### 5. Launch and Testing Infrastructure

**Launch Script** (`launch_monitored_training.py`):
- Environment setup and dependency checking
- GPU availability verification
- One-command training launch

**Setup Testing** (`test_monitored_setup.py`):
- Model functionality verification
- Monitoring system testing
- Component integration validation

## Training-Time vs Post-Hoc Approach

### Previous Approach (Post-Hoc Analysis)
```
Training → Trained Model → Post-Hoc Interpretability Analysis
```
**Problems:**
- No guidance during training to satisfy constraints
- ~50% violation rates for educational constraints
- Reactive analysis after training completion

### New Approach (Training-Time Monitoring)
```
Training + Real-Time Monitoring + Auxiliary Losses → Interpretable Model
```
**Benefits:**
- **Proactive constraint satisfaction** during training
- **Auxiliary losses guide model** toward interpretable solutions
- **Real-time feedback** on interpretability metrics
- **Combined optimization** of performance and interpretability

## Usage Instructions

### 1. Quick Test
```bash
cd /workspaces/pykt-toolkit
python test_monitored_setup.py
```

### 2. Launch Training
```bash
python launch_monitored_training.py
```

### 3. Monitor Progress
- Check console logs for interpretability metrics every 25 batches
- View wandb dashboard for detailed analysis and visualizations
- Monitor both performance (AUC/accuracy) and constraint satisfaction

## Expected Outcomes

### Performance Targets
- **Target AUC**: ≥0.72 (maintaining competitive performance)
- **Minimum acceptable AUC**: ≥0.70

### Interpretability Targets
- **Mastery-Performance Correlation**: ≥0.4
- **Gain-Correctness Correlation**: ≥0.3  
- **Negative Gains**: <10%
- **Monotonicity Violations**: <20%

## Key Advantages

1. **Real-Time Feedback**: Monitor interpretability during training, not after
2. **Guided Learning**: Auxiliary losses steer model toward interpretable solutions
3. **Balanced Optimization**: Simultaneously optimize performance and interpretability
4. **Educational Alignment**: Constraints based on educational theory and pedagogical principles
5. **Proactive Approach**: Address interpretability issues during training, not post-hoc

## Technical Details

### Model Architecture
- Enhanced GainAKT2 with dual-stream attention (context/value)
- Interpretability projection heads for mastery and gains
- 256-dim embeddings, 8 attention heads, 4 encoder blocks

### Training Strategy
- Combined loss: `total_loss = bce_loss + aux_loss`
- Auxiliary loss weights tuned for balance
- Gradient clipping for training stability
- Real-time constraint monitoring every 25 steps

### Monitoring Framework  
- Hook-based architecture for minimal training overhead
- Educational constraint validation during forward pass
- Comprehensive logging and visualization support

This implementation represents a significant advancement from post-hoc interpretability analysis to **training-time interpretability optimization**, ensuring that educational constraints are satisfied during the learning process rather than verified afterwards.