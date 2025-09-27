#!/bin/bash

# INSTALLATION.sh - Set up PyKT environment locally with GPU support (User-space installation)
# Based on the Dockerfile configuration - No sudo required

set -e  # Exit on any error

echo "🚀 Starting PyKT environment setup (user-space installation)..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# 1. Check system dependencies (inform user what might be missing)
print_status "Checking system dependencies..."
missing_deps=()
for cmd in curl wget git; do
    if ! command -v $cmd &> /dev/null; then
        missing_deps+=($cmd)
    fi
done

# Check for tkinter availability
if ! python3 -c "import tkinter" 2>/dev/null; then
    missing_deps+=(python3-tk)
fi

if [ ${#missing_deps[@]} -ne 0 ]; then
    print_warning "Missing system dependencies: ${missing_deps[*]}"
    if [[ " ${missing_deps[*]} " =~ " python3-tk " ]]; then
        print_warning "python3-tk is required for some models (turtle/tkinter graphics)"
        print_warning "To install: sudo apt update && sudo apt install python3-tk"
    fi
    print_warning "Please ask your system administrator to install: ${missing_deps[*]}"
    print_warning "Continuing anyway - some features may not work..."
fi

# 2. Check CUDA availability
print_status "Checking NVIDIA CUDA availability..."
if command -v nvcc &> /dev/null; then
    print_status "CUDA toolkit found: $(nvcc --version | grep release)"
elif command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA driver found: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
    print_warning "CUDA toolkit not found - GPU acceleration may not work"
else
    print_warning "No NVIDIA drivers detected - GPU acceleration will not work"
fi

# 3. Download corporate CA certificate (to user directory)
print_status "Setting up corporate CA certificate..."
mkdir -p ~/.local/share/ca-certificates
curl -O https://nexus.roqs.basf.net/repository/cdn/ca/BASF_internal_and_public_ca_bundle.crt --insecure || print_warning "Failed to download corporate certificate"

if [ -f "BASF_internal_and_public_ca_bundle.crt" ]; then
    cp BASF_internal_and_public_ca_bundle.crt ~/.local/share/ca-certificates/
    print_status "Corporate certificate saved to ~/.local/share/ca-certificates/"
    rm BASF_internal_and_public_ca_bundle.crt
fi

# 4. Install Node.js (user-space using Node Version Manager)
print_status "Installing Node.js via NVM..."
if ! command -v node &> /dev/null; then
    # Install NVM
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    # Install Node.js 20
    nvm install 20
    nvm use 20
    nvm alias default 20
else
    print_warning "Node.js already installed, skipping..."
fi

# 5. Install Miniconda
print_status "Installing Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda3
    rm ~/miniconda.sh
else
    print_warning "Miniconda already installed, skipping..."
fi

# 6. Set up conda in PATH
print_status "Setting up conda in PATH..."
export PATH="$HOME/miniconda3/bin:$PATH"

# Add to bashrc if not already there
if ! grep -q "miniconda3/bin" ~/.bashrc; then
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
fi

# Initialize conda
$HOME/miniconda3/bin/conda init bash

# 7. Configure conda for corporate environment
print_status "Configuring conda..."
$HOME/miniconda3/bin/conda config --set always_yes yes --set changeps1 no
$HOME/miniconda3/bin/conda config --set ssl_verify false

# Update conda
$HOME/miniconda3/bin/conda update -q conda || print_warning "Conda update failed, continuing..."

# 8. Create pykt environment with Python 3.7.5
print_status "Creating pykt conda environment with Python 3.7.5..."
$HOME/miniconda3/bin/conda create --name=pykt python=3.7.5 -y

# 9. Install GPU packages in the environment
print_status "Installing CUDA and PyTorch packages..."
$HOME/miniconda3/bin/conda run -n pykt conda install -c conda-forge cudatoolkit=11.8 -y || print_warning "CUDA toolkit installation failed"
$HOME/miniconda3/bin/conda run -n pykt conda install -c pytorch pytorch torchvision torchaudio pytorch-cuda=11.8 -y || {
    print_warning "PyTorch with CUDA failed, installing CPU-only version..."
    $HOME/miniconda3/bin/conda run -n pykt conda install -c pytorch pytorch torchvision torchaudio cpuonly -y
}

# 10. Install data science packages
print_status "Installing data science packages..."
$HOME/miniconda3/bin/conda run -n pykt conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn jupyter -y

# 11. Install additional Python packages
print_status "Installing additional ML packages..."
$HOME/miniconda3/bin/conda run -n pykt pip install tensorboard wandb
$HOME/miniconda3/bin/conda run -n pykt pip install transformers datasets accelerate
$HOME/miniconda3/bin/conda run -n pykt pip install torch-geometric || print_warning "torch-geometric installation failed"
$HOME/miniconda3/bin/conda run -n pykt pip install dgl-cu118 -f https://data.dgl.ai/wheels/cu118/repo.html || {
    print_warning "DGL CUDA installation failed, installing CPU version..."
    $HOME/miniconda3/bin/conda run -n pykt pip install dgl
}

# 12. Install global Node.js package (user-space)
print_status "Installing Anthropic Claude package..."
if command -v npm &> /dev/null; then
    npm install -g @anthropic-ai/claude-code --force --no-os-check || print_warning "Failed to install claude-code package"
else
    print_warning "npm not available, skipping claude-code installation"
fi

# 13. Set up environment variables
print_status "Setting up environment variables..."
ENV_VARS="
# NVIDIA/CUDA environment variables (adjust paths as needed)
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Corporate CA certificate environment variables
export REQUESTS_CA_BUNDLE=~/.local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
export SSL_CERT_FILE=~/.local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
export CURL_CA_BUNDLE=~/.local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt

# Node.js (NVM)
export NVM_DIR=\"\$HOME/.nvm\"
[ -s \"\$NVM_DIR/nvm.sh\" ] && \\. \"\$NVM_DIR/nvm.sh\"
[ -s \"\$NVM_DIR/bash_completion\" ] && \\. \"\$NVM_DIR/bash_completion\"

# Auto-activate pykt environment
conda activate pykt
"

# Add environment variables to bashrc if not already there
if ! grep -q "NVIDIA_VISIBLE_DEVICES" ~/.bashrc; then
    echo "$ENV_VARS" >> ~/.bashrc
fi

# 14. Test the installation
print_status "Testing the installation..."
$HOME/miniconda3/bin/conda run -n pykt python -c "
import sys
print('Python version:', sys.version)
try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA devices:', torch.cuda.device_count())
        print('GPU name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU detected - using CPU mode')
except ImportError as e:
    print('Error importing PyTorch:', e)

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    print('Data science packages installed successfully')
except ImportError as e:
    print('Error importing data science packages:', e)

try:
    import transformers
    print('Transformers version:', transformers.__version__)
except ImportError as e:
    print('Error importing transformers:', e)
"

print_status "✅ Installation completed successfully!"
echo ""
echo "To use the environment:"
echo "  1. Restart your terminal or run: source ~/.bashrc"
echo "  2. The pykt environment will be activated automatically"
echo "  3. Or manually activate with: conda activate pykt"
echo ""
echo "To test GPU access:"
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""
echo "To run PyKT toolkit examples:"
echo "  cd examples/"
echo "  python wandb_dkt_train.py"
echo ""
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  Note: CUDA toolkit not detected. If you need GPU acceleration:"
    echo "   Ask your system administrator to install CUDA toolkit 11.8"
    echo "   Or install it yourself if you have permissions"
fi