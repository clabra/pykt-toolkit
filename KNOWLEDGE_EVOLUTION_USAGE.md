# Knowledge Evolution Extraction - Usage Guide

## Overview

The `extract_knowledge_evolution.py` script provides comprehensive analysis of how student knowledge states evolve over time using the trained GainAKT2Monitored model. It can extract, analyze, and visualize the learning journey of individual students.

## Prerequisites

1. **Trained GainAKT2Monitored model** - You need a saved model checkpoint
2. **Python environment** - Activated pykt environment with required packages
3. **Data** - Student interaction sequences to analyze

## Quick Start

### 1. Basic Usage - Run the Demo

```bash
# Activate the environment
source /home/vscode/.pykt-env/bin/activate

# Navigate to the project directory  
cd /workspaces/pykt-toolkit

# Run the demonstration
python extract_knowledge_evolution.py
```

This will run a demo with synthetic data and show you the capabilities.

### 2. Programmatic Usage

```python
from extract_knowledge_evolution import KnowledgeStateEvolutionExtractor
import torch

# Model configuration (should match your trained model)
model_config = {
    'num_c': 100,
    'seq_len': 200,
    'd_model': 512,
    'n_heads': 4,
    'num_encoder_blocks': 4,
    'd_ff': 512,
    'dropout': 0.4,
    'emb_type': 'qid',
    'non_negative_loss_weight': 0.485828,
    'consistency_loss_weight': 0.173548,
    'monitor_frequency': 25
}

# Initialize the extractor
extractor = KnowledgeStateEvolutionExtractor(
    model_path="saved_model/gainakt2_enhanced_auc_0.7253/model.pth",
    model_config=model_config
)

# Student interaction data
questions = torch.tensor([12, 35, 67, 12, 89, 35, 45, 67, 12, 89])
responses = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1, 1, 1])

# Extract evolution
evolution_data = extractor.extract_student_journey(
    questions, responses, student_id="student_001"
)

# Generate report
report = extractor.generate_learning_report(evolution_data)

# Create visualization
extractor.visualize_mastery_evolution(
    evolution_data, 
    save_path="student_001_evolution.png"
)
```

## Detailed Usage Examples

### Example 1: Analyze Single Student from Real Data

```python
#!/usr/bin/env python3
"""
Example: Analyze a real student from the dataset
"""
import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

from extract_knowledge_evolution import KnowledgeStateEvolutionExtractor
from pykt.datasets import init_dataset4train
import torch

def analyze_real_student():
    # Load dataset
    dataset_name = "assist2015"
    model_name = "gainakt2"
    data_config = {
        "assist2015": {
            "dpath": "/workspaces/pykt-toolkit/data/assist2015",
            "num_q": 0,
            "num_c": 100,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": 3,
            "maxlen": 200,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1, 2, 3, 4],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv"
        }
    }
    
    train_loader, valid_loader = init_dataset4train(
        dataset_name, model_name, data_config, 0, 32
    )
    
    # Get first student from validation set
    for batch in valid_loader:
        questions = batch['cseqs'][0]  # First student
        responses = batch['rseqs'][0]
        break
    
    # Model configuration
    model_config = {
        'num_c': 100,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 4,
        'num_encoder_blocks': 4,
        'd_ff': 512,
        'dropout': 0.4,
        'emb_type': 'qid',
        'non_negative_loss_weight': 0.485828,
        'consistency_loss_weight': 0.173548,
        'monitor_frequency': 25
    }
    
    # Initialize extractor
    extractor = KnowledgeStateEvolutionExtractor(
        model_path="saved_model/gainakt2_enhanced_auc_0.7253/model.pth",
        model_config=model_config
    )
    
    # Extract and analyze
    evolution_data = extractor.extract_student_journey(
        questions, responses, student_id="real_student_001"
    )
    
    # Generate comprehensive analysis
    report = extractor.generate_learning_report(
        evolution_data, 
        save_path="real_student_001_report.json"
    )
    
    # Create visualization
    extractor.visualize_mastery_evolution(
        evolution_data,
        save_path="real_student_001_evolution.png"
    )
    
    print("Analysis completed!")
    print(f"Files generated:")
    print(f"  - real_student_001_report.json")
    print(f"  - real_student_001_evolution.png")

if __name__ == "__main__":
    analyze_real_student()
```

### Example 2: Batch Analysis of Multiple Students

```python
#!/usr/bin/env python3
"""
Example: Batch analysis of multiple students
"""
import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

from extract_knowledge_evolution import KnowledgeStateEvolutionExtractor
from pykt.datasets import init_dataset4train
import torch
import os
import json

def batch_analyze_students(num_students=10):
    """Analyze multiple students and generate comparative reports."""
    
    # Setup (same as previous example)
    dataset_name = "assist2015"
    model_name = "gainakt2"
    data_config = {
        "assist2015": {
            "dpath": "/workspaces/pykt-toolkit/data/assist2015",
            "num_q": 0,
            "num_c": 100,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": 3,
            "maxlen": 200,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1, 2, 3, 4],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv"
        }
    }
    
    train_loader, valid_loader = init_dataset4train(
        dataset_name, model_name, data_config, 0, 32
    )
    
    model_config = {
        'num_c': 100,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 4,
        'num_encoder_blocks': 4,
        'd_ff': 512,
        'dropout': 0.4,
        'emb_type': 'qid',
        'non_negative_loss_weight': 0.485828,
        'consistency_loss_weight': 0.173548,
        'monitor_frequency': 25
    }
    
    extractor = KnowledgeStateEvolutionExtractor(
        model_path="saved_model/gainakt2_enhanced_auc_0.7253/model.pth",
        model_config=model_config
    )
    
    # Create output directory
    os.makedirs("batch_analysis", exist_ok=True)
    
    # Analyze students
    all_reports = []
    student_count = 0
    
    for batch in valid_loader:
        batch_size = batch['cseqs'].size(0)
        
        for i in range(min(batch_size, num_students - student_count)):
            student_id = f"student_{student_count:03d}"
            questions = batch['cseqs'][i]
            responses = batch['rseqs'][i]
            
            # Extract evolution
            evolution_data = extractor.extract_student_journey(
                questions, responses, student_id=student_id
            )
            
            # Generate individual report
            report = extractor.generate_learning_report(evolution_data)
            all_reports.append(report)
            
            # Save individual files
            with open(f"batch_analysis/{student_id}_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            extractor.visualize_mastery_evolution(
                evolution_data,
                save_path=f"batch_analysis/{student_id}_evolution.png"
            )
            
            student_count += 1
            print(f"Analyzed {student_id}")
            
            if student_count >= num_students:
                break
        
        if student_count >= num_students:
            break
    
    # Generate comparative summary
    summary = generate_comparative_summary(all_reports)
    
    with open("batch_analysis/comparative_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nBatch analysis completed!")
    print(f"Analyzed {student_count} students")
    print(f"Results saved in: batch_analysis/")

def generate_comparative_summary(reports):
    """Generate comparative statistics across students."""
    import numpy as np
    
    accuracies = [r['overall_accuracy'] for r in reports]
    final_masteries = [r['avg_final_mastery'] for r in reports]
    mastery_growths = [r['mastery_growth'] for r in reports]
    
    return {
        'total_students': len(reports),
        'accuracy_stats': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies)
        },
        'final_mastery_stats': {
            'mean': np.mean(final_masteries),
            'std': np.std(final_masteries),
            'min': np.min(final_masteries),
            'max': np.max(final_masteries)
        },
        'mastery_growth_stats': {
            'mean': np.mean(mastery_growths),
            'std': np.std(mastery_growths),
            'min': np.min(mastery_growths),
            'max': np.max(mastery_growths)
        }
    }

if __name__ == "__main__":
    batch_analyze_students(num_students=5)
```

## Key Functions and Outputs

### Core Functions

1. **`extract_student_journey(questions, responses, student_id)`**
   - Input: Question sequence, response sequence, optional ID
   - Output: Complete evolution dictionary with mastery trajectories

2. **`get_skill_mastery_trajectory(evolution_data, skill_ids)`**
   - Input: Evolution data, optional skill filter
   - Output: Time-series mastery data for specific skills

3. **`get_interaction_effects(evolution_data)`**
   - Input: Evolution data
   - Output: DataFrame with per-interaction analysis

4. **`visualize_mastery_evolution(evolution_data, save_path)`**
   - Input: Evolution data, save path
   - Output: Comprehensive visualization plots

5. **`generate_learning_report(evolution_data, save_path)`**
   - Input: Evolution data, optional save path
   - Output: Structured learning analytics report

### Output Files

1. **Visualization PNG**: 4-panel plot showing:
   - Skill mastery trajectories over time
   - Learning gains heatmap
   - Mastery vs prediction scatter plot
   - Cumulative learning progress

2. **JSON Report**: Structured data including:
   - Overall student statistics
   - Per-skill breakdown
   - Learning gains analysis
   - Performance metrics

3. **Console Output**: Real-time progress and summary statistics

## Command Line Usage

```bash
# Run demo with synthetic data
python extract_knowledge_evolution.py

# Run with custom model path (modify the script)
# Edit the model_path variable in demo_knowledge_evolution()

# Run batch analysis (using Example 2 script)
python batch_analysis_example.py
```

## Troubleshooting

### Common Issues

1. **Model not found**:
   ```
   FileNotFoundError: Model file not found
   ```
   - Solution: Check the model path in your script
   - Ensure the model was saved correctly

2. **GPU memory issues**:
   ```
   CUDA out of memory
   ```
   - Solution: Add `.cpu()` to move tensors to CPU
   - Use smaller batch sizes

3. **Shape mismatches**:
   ```
   RuntimeError: Expected tensor to have shape [...]
   ```
   - Solution: Ensure model_config matches the saved model
   - Check sequence lengths and dimensions

### Performance Tips

1. **Use CPU for extraction** if you have memory constraints
2. **Batch process** multiple students for efficiency
3. **Save intermediate results** for large analyses
4. **Use specific skill filtering** for faster visualization

This script provides a complete framework for analyzing how students' knowledge evolves over their learning journey, making the interpretability of the GainAKT2Monitored model actionable for educational research and applications.