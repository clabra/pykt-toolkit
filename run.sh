#!/bin/bash
# Simple launcher for run_repro_experiment.py
# Usage: ./run.sh [arguments...]

cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate
python examples/run_repro_experiment.py "$@"
