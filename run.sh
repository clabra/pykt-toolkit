#!/bin/bash
# Reproducibility-compliant launcher for iKT experiments
# 
# This script follows the "Explicit Parameters, Zero Defaults" philosophy:
# - All parameters loaded from configs/parameter_default.json
# - CLI overrides applied explicitly
# - Complete parameter provenance maintained
# - MD5 integrity verification enabled
# - Automatic parameter audit before launch
#
# Usage:
#   ./run.sh --short_title test --epochs 12              # Train new experiment
#   ./run.sh --repro_experiment_id 584063                # Reproduce experiment
#   ./run.sh --short_title test --phase 1 --epsilon 0.0  # Phase 1 training
#   ./run.sh --short_title test --phase 2 --epsilon 0.10 # Phase 2 training
#   ./run.sh --short_title test --phase null             # Automatic two-phase
#
# See examples/reproducibility.md for complete documentation

set -e  # Exit on error

# Activate virtual environment
cd /workspaces/pykt-toolkit
source /home/vscode/.pykt-env/bin/activate

# Launch with reproducibility infrastructure
python examples/run_repro_experiment.py "$@"
