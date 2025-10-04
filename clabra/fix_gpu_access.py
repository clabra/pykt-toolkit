#!/usr/bin/env python3
"""
Comprehensive GPU Access Fix for PyKT Toolkit
This script attempts to resolve PyTorch GPU access issues in containerized environments
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_nvidia_environment():
    """Set up NVIDIA library paths and environment variables"""
    print("🔧 Setting up NVIDIA environment...")
    
    # Base paths
    pykt_env = "/home/vscode/.pykt-env"
    nvidia_base = f"{pykt_env}/lib/python3.8/site-packages/nvidia"
    
    # NVIDIA library directories
    nvidia_lib_paths = [
        f"{nvidia_base}/cuda_runtime/lib",
        f"{nvidia_base}/cuda_cupti/lib", 
        f"{nvidia_base}/cublas/lib",
        f"{nvidia_base}/cudnn/lib",
        f"{nvidia_base}/curand/lib",
        f"{nvidia_base}/cusolver/lib",
        f"{nvidia_base}/cusparse/lib",
        f"{nvidia_base}/cufft/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64"
    ]
    
    # Filter existing paths
    existing_paths = [p for p in nvidia_lib_paths if Path(p).exists()]
    print(f"✅ Found {len(existing_paths)} NVIDIA library directories")
    
    # Set environment variables
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = ':'.join(existing_paths + [current_ld_path] if current_ld_path else existing_paths)
    
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['CUDA_ROOT'] = '/usr/local/cuda'  
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
    
    return existing_paths

def install_missing_nvidia_packages():
    """Install any missing NVIDIA packages via pip"""
    print("📦 Checking NVIDIA package installation...")
    
    nvidia_packages = [
        "nvidia-cuda-runtime-cu11",
        "nvidia-cuda-cupti-cu11", 
        "nvidia-cudnn-cu11",
        "nvidia-cublas-cu11",
        "nvidia-curand-cu11",
        "nvidia-cusolver-cu11",
        "nvidia-cusparse-cu11",
        "nvidia-cufft-cu11"
    ]
    
    pip_path = "/home/vscode/.pykt-env/bin/pip"
    
    for package in nvidia_packages:
        try:
            result = subprocess.run([pip_path, "install", package, "-q"], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {package}")
            else:
                print(f"⚠️  {package} - already installed or failed")
        except Exception as e:
            print(f"❌ {package} - error: {e}")

def test_gpu_access():
    """Test PyTorch GPU access"""
    print("\n🧪 Testing PyTorch GPU access...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            print("\n🎉 SUCCESS! GPUs are accessible:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
            
            # Quick computation test
            device = torch.device('cuda:0')
            x = torch.randn(100, 100, device=device)
            y = torch.mm(x, x.T)
            print(f"✅ GPU computation successful on: {y.device}")
            return True
        else:
            print("❌ GPUs still not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GPU access: {e}")
        return False

def main():
    print("=" * 50)
    print("PyKT Toolkit GPU Access Fix")
    print("=" * 50)
    
    # Step 1: Set up environment
    nvidia_paths = setup_nvidia_environment()
    
    # Step 2: Install missing packages  
    install_missing_nvidia_packages()
    
    # Step 3: Test GPU access
    success = test_gpu_access()
    
    if success:
        print("\n🎉 GPU setup completed successfully!")
        print("\nTo make this permanent, add this to your shell profile:")
        print(f'export LD_LIBRARY_PATH="{os.environ["LD_LIBRARY_PATH"]}"')
    else:
        print("\n❌ GPU setup failed. This may be a container runtime issue.")
        print("The container may need to be started with proper GPU support.")
        print("\nDebugging info:")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        
        # Check nvidia-smi access
        try:
            result = subprocess.run(["nvidia-smi", "--list-gpus"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ nvidia-smi can see GPUs")
                print(result.stdout.strip())
            else:
                print("❌ nvidia-smi failed")
        except:
            print("❌ nvidia-smi not accessible")

if __name__ == "__main__":
    main()
