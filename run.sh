#!/bin/bash
# Simple launcher for run_benchmarks_paper.py
# Usage: ./run.sh [arguments...]

cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate

nohup python examples/run_benchmarks_paper.py  --mode training --model gtransformer --dataset assist2009  --gpus 1,2,3,4,5 "$@" > experiments/run_benchmarks_paper.log 2>&1 &

# Evaluation
#python examples/run_benchmarks_paper.py --mode evaluation  --model gtransformer --dataset assist2009 "$@"

#Results
#python examples/run_benchmarks_paper.py  --mode results  --model gtransformer --dataset assist2009 "$@"
