# Paper - Results Reproducibility

## Reference Experiment

We will take the experiment 481134 (ablation none, 4-4) as a reference for the results we will present in the paper.

```
experiments/20260124_234359_ablation-none-4-4_baseline_481134
```
## Training, Evaluation and Results

```
run.sh 

#!/bin/bash
# Simple launcher for run_benchmarks_paper.py
# Usage: ./run.sh [arguments...]

cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate

nohup python examples/run_benchmarks_paper.py  --mode training --model gtransformer --dataset assist2009  --gpus 1,2,3,4,5 "$@" > experiments/run_benchmarks
_paper.log 2>&1 &

# Evaluation
#python examples/run_benchmarks_paper.py --mode evaluation  --model gtransformer --dataset assist2009 "$@"

#Results
#python examples/run_benchmarks_paper.py  --mode results  --model gtransformer --dataset assist2009 "$
```
```
# Benchmark with multiple datassets and ablation=none (to get plots, validation results, etc.)
nohup python examples/run_benchmarks_paper.py  --mode training --model gtransformer --dataset assist2015,algebra2005,bridge2algebra2006,nips_task34 --ablation none  --gpus 1,2,3,4,5 --short_title papertable-ablationnone-datasets > experiments/run_benchmarks_papertable.log 2>&1 &

# Benchmark with multiple datassets and ablation=all (to compare AUC with other models)
nohup python examples/run_benchmarks_paper.py  --mode training --model gtransformer --dataset assist2015,algebra2005,bridge2algebra2006,nips_task34 --ablation all  --gpus 1,2,3,4,5 --short_title papertable-ablationall-datasets > experiments/run_benchmarks_papertable.log 2>&1 & 

# Evaluation of a certain experiment (--campaign)
python examples/run_benchmarks_paper.py  --mode evaluation --model gtransformer --dataset assist2015,algebra2005 --campaign 20260126_113641_papertable-ablationall-datasets_936799
```

## Results 

## RQs

- **RQ1: Theory-Based Interpretability Through Grounded Transformers**: Can deep knowledge tracing models achieve state-of-the-art predictive performance while providing interpretability grounded in established principles and theories? Specifically, can we design a transformer architecture that produces pedagogically meaningful mastery estimations that are explainable through a causal and interpretable logic, such as Bayesian Knowledge Tracing?
- **RQ2: Trade-Offs Between Predictive Performance and Interpretability**: How do the metrics of the supervised, interpretable, and BKT predictions compare? What is the cost of interpretability in terms of AUC? How much predictive gain do the interpretable grounded predictions achieve compared to traditional BKT?
- **RQ3: Practical Value for Student-Centered Personalization**: Beyond providing interpretable diagnostics, does the high capacity of gTransformer to capture intricate interaction patterns offer advantages over traditional models? Specifically, can these capabilities be leveraged to enhance student-centered personalization relative to population-based models such as Bayesian Knowledge Tracing?

## Validation

### RQ1 

#### Step 1

#### Step 2

#### Step 3


