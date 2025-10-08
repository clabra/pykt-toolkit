# GainAKT2Exp Focused Parameter Sweep Setup

## Problem Analysis
- Current setup getting **0.7242 AUC** instead of target **0.7259 AUC**
- Need to find better parameter combinations around current defaults
- Parameter precision was corrected but still not achieving target

## Solution: Focused Parameter Sweep

Created comprehensive sweep system to find optimal parameters around current defaults:

### 📁 Files Created:

#### 1. **`run_focused_sweep.py`** - Main Sweep Runner
- **Purpose**: All-in-one parameter sweep script
- **Features**: 
  - Generates 20 parameter combinations around defaults
  - Runs experiments sequentially with timeout protection
  - Real-time progress tracking and results analysis
  - Automatically identifies configurations achieving AUC >= 0.7259

#### 2. **`launch_focused_sweep.py`** - Simple Launcher
- **Purpose**: Interactive launcher for the sweep
- **Usage**: `python launch_focused_sweep.py`

#### 3. **`create_focused_sweep.py`** - Wandb Sweep Creator
- **Purpose**: Creates wandb sweep configuration (alternative approach)
- **Features**: Bayesian optimization for efficient parameter search

### 🎯 Search Configuration:

**Parameter Ranges (around current defaults):**
```
Base values: lr=0.0003, wd=0.000059, bs=128, epochs=20

Search ranges:
├── learning_rate: 0.0001 to 0.0008 (33% to 267% of base)
├── weight_decay:  0.00002 to 0.0002 (33% to 339% of base) 
├── batch_size:    [64, 96, 128, 160, 192]
├── num_epochs:    [15, 18, 20, 22, 25, 30]
├── enhanced_constraints: [True, False]
└── patience:      [15, 20, 25]
```

### 🚀 How to Run:

#### Option 1: Simple Launch (Recommended)
```bash
cd /workspaces/pykt-toolkit/examples
python launch_focused_sweep.py
```

#### Option 2: Direct Run
```bash
cd /workspaces/pykt-toolkit/examples  
python run_focused_sweep.py
```

#### Option 3: Wandb Sweep (Advanced)
```bash
cd /workspaces/pykt-toolkit/examples
python create_focused_sweep.py
# Then run: wandb agent <sweep_id>
```

### 📊 Expected Results:

The sweep will:
1. **Generate** 20 parameter combinations using smart sampling
2. **Run** each experiment with 1-hour timeout per run  
3. **Track** progress and best AUC in real-time
4. **Identify** configurations achieving AUC >= 0.7259
5. **Save** detailed results to `focused_sweep_results_TIMESTAMP.txt`

### 🎉 Success Criteria:

- **Target**: Find parameter combinations with AUC >= **0.7259**
- **Expected**: 3-5 configurations should achieve target  
- **Outcome**: New optimal parameters for consistent high performance

### 📈 Sample Output:
```
🥇 TOP 5 RESULTS:
Rank AUC      LR         WD         BS   Epochs Enhanced
1    0.7265   0.000287   0.000071   128  22     True
2    0.7261   0.000341   0.000045   160  20     True  
3    0.7258   0.000298   0.000063   128  25     False
```

### ⚡ Key Features:
- **Smart Sampling**: Focuses search around proven good ranges
- **Fault Tolerant**: Handles crashes/timeouts gracefully
- **Progress Tracking**: Real-time status and best results so far
- **Auto Analysis**: Identifies best configurations automatically
- **Comprehensive Logging**: Saves all results for later analysis

**Ready to launch when you are!** 🚀