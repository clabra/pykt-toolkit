#!/bin/bash
# Comprehensive GPU Environment Setup Script for PyKT Toolkit
# This script dynamically finds and sets the necessary environment variables
# to enable PyTorch to access GPUs within the dev container.

echo "🔧 Setting up GPU environment for PyKT Toolkit..."

# --- 1. Set Core CUDA Environment Variables ---
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_VISIBLE_DEVICES=all
echo "✅ Core CUDA environment variables set."

# --- 2. Dynamically Find and Set LD_LIBRARY_PATH ---
PYKT_ENV_PATH="/home/vscode/.pykt-env"
SITE_PACKAGES_PATH="${PYKT_ENV_PATH}/lib/python3.8/site-packages"

# Standard system paths
SYSTEM_PATHS=(
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/targets/x86_64-linux/lib"
)

# Dynamically find pip-installed NVIDIA library paths
NVIDIA_PACKAGE_BASE="${SITE_PACKAGES_PATH}/nvidia"
PIP_NVIDIA_PATHS=()
if [ -d "$NVIDIA_PACKAGE_BASE" ]; then
    for dir in $(find "$NVIDIA_PACKAGE_BASE" -type d -name "lib"); do
        PIP_NVIDIA_PATHS+=("$dir")
    done
fi

# Combine all paths, ensuring they exist and are unique.
# CRITICAL: The system's main CUDA library path must come first to ensure
# the correct driver-related libraries are found by PyTorch.
ALL_PATHS=("/usr/local/cuda/lib64" "${PIP_NVIDIA_PATHS[@]}" "${SYSTEM_PATHS[@]}")
UNIQUE_PATHS=()
for path in "${ALL_PATHS[@]}"; do
    if [[ -d "$path" && ! " ${UNIQUE_PATHS[*]} " =~ " ${path} " ]]; then
        UNIQUE_PATHS+=("$path")
    fi
done
 
# Unset the old path and export the new, clean one.
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(IFS=:; echo "${UNIQUE_PATHS[*]}")
echo "✅ LD_LIBRARY_PATH configured dynamically."
# This special line is for the Dockerfile build process to capture the final path.
echo "FINAL_LD_LIBRARY_PATH=export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\""


# --- 3. Test GPU Access ---
echo "Testing GPU access..."

${PYKT_ENV_PATH}/bin/python -c "
import torch
print('=== GPU Access Test ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print('🎉 SUCCESS! GPUs detected:')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} - {props.total_memory // 1024**3} GB')
    
    # Quick computation test
    device = torch.device('cuda:0')
    x = torch.randn(10, 10, device=device)
    y = x @ x.T
    print(f'✅ GPU computation test passed on {y.device}')
else:
    print('❌ GPU access failed')
    print('This indicates a container runtime configuration issue.')
"

echo -e "\n📝 To use GPUs in your current terminal session, run:\nsource /workspaces/pykt-toolkit/enable_gpu.sh"
