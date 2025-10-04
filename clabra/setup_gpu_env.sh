#!/bin/bash
# GPU Environment Setup Script for PyKT Toolkit
# This script fixes PyTorch GPU access by setting correct library paths

echo "=== Setting up GPU environment for PyTorch ==="

# Add NVIDIA library paths to environment
export LD_LIBRARY_PATH="/home/vscode/.pykt-env/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/vscode/.pykt-env/lib/python3.8/site-packages/nvidia/cuda_cupti/lib:/home/vscode/.pykt-env/lib/python3.8/site-packages/nvidia/cublas/lib:/home/vscode/.pykt-env/lib/python3.8/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# Additional CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

echo "Library paths updated. Testing GPU access..."

# Test GPU access
/home/vscode/.pykt-env/bin/python -c "
import torch
print('=== GPU Access Test ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print('✅ GPUs are accessible!')
    
    # Test actual computation
    try:
        device = torch.device('cuda:0')
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)
        print(f'✅ GPU computation test successful on: {z.device}')
    except Exception as e:
        print(f'❌ GPU computation failed: {e}')
else:
    print('❌ GPUs still not accessible')
    print('Trying alternative fix...')
"

echo "=== GPU setup complete ==="
