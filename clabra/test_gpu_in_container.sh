#!/bin/bash
# Container-based GPU test that respects corporate certificates and environment

echo "=== Testing GPU Access in Fresh Container ==="

# Use the same base image as your dev container
docker run --rm --gpus all -it \
  -v /etc/ssl/certs:/etc/ssl/certs:ro \
  -v /usr/local/share/ca-certificates:/usr/local/share/ca-certificates:ro \
  -e SSL_CERT_FILE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt \
  -e REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt \
  nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 bash -c "
    echo '=== Setting up environment ==='
    apt update -qq
    apt install -y python3 python3-pip curl ca-certificates
    
    # Copy corporate certificates
    update-ca-certificates
    
    echo '=== Installing PyTorch with CUDA support ==='
    pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
      --trusted-host download.pytorch.org \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org
    
    echo '=== Testing GPU access ==='
    python3 -c \"
import torch
print('=' * 50)
print('GPU Test Results:')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print('✅ SUCCESS: GPUs accessible in container!')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ FAILED: No GPU access in container')
print('=' * 50)
\"
"
