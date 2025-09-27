#!/usr/bin/env python3
"""
Test GPU access using a temporary virtual environment
This runs inside the current container without affecting the main environment
"""

import subprocess
import sys
import tempfile
import os

def test_gpu_in_venv():
    print("=== Testing GPU Access in Temporary Virtual Environment ===")
    
    # Create temporary directory for virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = os.path.join(temp_dir, "test_venv")
        
        print(f"Creating temporary virtual environment: {venv_path}")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        
        # Get paths for virtual environment
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip")
            python_path = os.path.join(venv_path, "Scripts", "python")
        else:  # Unix/Linux
            pip_path = os.path.join(venv_path, "bin", "pip")
            python_path = os.path.join(venv_path, "bin", "python")
        
        print("Installing PyTorch with CUDA support...")
        try:
            subprocess.run([
                pip_path, "install", 
                "torch==2.0.1+cu118", "torchvision==0.15.2+cu118",
                "--extra-index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True, capture_output=True, text=True)
            
            print("Testing GPU access...")
            result = subprocess.run([
                python_path, "-c", """
import torch
print('=' * 50)
print('GPU Test Results (Virtual Environment):')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print('✅ SUCCESS: GPUs accessible in virtual environment!')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ FAILED: No GPU access in virtual environment')
print('=' * 50)
"""
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"Failed to install PyTorch: {e}")
            
        print("Virtual environment test completed and cleaned up.")

if __name__ == "__main__":
    test_gpu_in_venv()
