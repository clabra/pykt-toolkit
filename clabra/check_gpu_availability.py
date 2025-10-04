#!/usr/bin/env python3
"""
GPU Availability Comprehensive Check Script
Run this to verify GPU access from Python
"""
import os
import sys
import subprocess
from pathlib import Path

def bootstrap_environment():
    """
    Checks for GPU access and, if it fails, attempts to apply the environment fix
    by sourcing the enable_gpu.sh script. This makes the training script self-sufficient.
    """
    # Define paths
    pykt_env_path = "/home/vscode/.pykt-env"
    site_packages_path = f"{pykt_env_path}/lib/python3.8/site-packages"
    nvidia_lib_path = f"{site_packages_path}/nvidia"
    cuda_runtime_lib_path = f"{nvidia_lib_path}/cuda_runtime/lib"

    # If the path exists but is not in LD_LIBRARY_PATH, we need to re-exec
    if os.path.exists(cuda_runtime_lib_path) and cuda_runtime_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
        print("---")
        print("🔧 NVIDIA libraries not found in LD_LIBRARY_PATH. Attempting to bootstrap a correct environment...")
        
        # Dynamically find all pip-installed NVIDIA library paths
        pip_nvidia_paths = []
        if os.path.isdir(nvidia_lib_path):
            for lib_name in os.listdir(nvidia_lib_path):
                lib_path = os.path.join(nvidia_lib_path, lib_name, 'lib')
                if os.path.isdir(lib_path):
                    pip_nvidia_paths.append(lib_path)

        # Standard system paths
        system_paths = ["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib", "/usr/lib/x86_64-linux-gnu"]

        # CRITICAL: The system's main CUDA library path must come first.
        all_paths = ["/usr/local/cuda/lib64"] + pip_nvidia_paths + system_paths
        unique_paths = list(dict.fromkeys([p for p in all_paths if os.path.isdir(p)]))
        os.environ['LD_LIBRARY_PATH'] = ':'.join(unique_paths)
        
        # Re-execute the script with the new environment
        print("✅ Environment configured. Re-executing script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    # This function will check and re-launch the script if needed.
    bootstrap_environment()

    print("GPU AVAILABILITY COMPREHENSIVE CHECK")
    print("=" * 50)
    
    # Run all checks
    nvidia_gpu_count = check_nvidia_smi()
    check_environment()
    pytorch_available = check_pytorch()
    tensorflow_available = check_tensorflow()
    computation_works = test_gpu_computation()
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Nvidia-smi GPUs: {nvidia_gpu_count}")
    print(f"PyTorch: {'✅ Available' if pytorch_available else '❌ Not Available'}")
    print(f"TensorFlow: {'✅ Available' if tensorflow_available else '❌ Not Available'}")
    print(f"GPU Computation: {'✅ Working' if computation_works else '❌ Not Working'}")
    
    if pytorch_available and computation_works:
        print(f"\n🎉 Overall GPU Status: ✅ READY FOR TRAINING")
        print(f"You can now train models with GPU acceleration!")
    else:
        print(f"\n⚠️ Overall GPU Status: ❌ NEEDS ATTENTION")
        print(f"Recommendations:")
        print(f"- Rebuild dev container: Ctrl+Shift+P → 'Dev Containers: Rebuild Container'")
        print(f"- Or reinstall PyTorch: pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")
        print(f"- See troubleshooting section in README_GPU.md")

def check_nvidia_smi():
    """Check if nvidia-smi is available and shows GPUs"""
    print("=== NVIDIA-SMI Detection ===")
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        gpu_lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        print(f"nvidia-smi available:")
        for line in gpu_lines:
            print(line)
        return len(gpu_lines)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ nvidia-smi not available or no GPUs detected")
        return 0

def check_pytorch():
    """Check PyTorch GPU detection"""
    print("\n=== PyTorch GPU Detection ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("❌ No CUDA GPUs detected by PyTorch")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_tensorflow():
    """Check TensorFlow GPU detection"""
    print("\n=== TensorFlow GPU Detection ===")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow GPU count: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
            return True
        else:
            print("❌ No GPUs detected by TensorFlow")
            return False
    except ImportError:
        print("❌ TensorFlow not available")
        return False

def check_environment():
    """Check CUDA environment variables"""
    print("\n=== CUDA Environment Variables ===")
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

    print("\n=== System GPU Information ===")
    try:
        gpu_devices = [f for f in os.listdir('/dev') if 'nvidia' in f]
        print(f"GPU device files: {gpu_devices}")
    except FileNotFoundError:
        print("No /dev/nvidia* files found.")

    try:
        with open('/proc/driver/nvidia/version', 'r') as f:
            print(f"NVIDIA driver version: {f.readline().strip()}")
    except FileNotFoundError:
        print("NVIDIA driver version file not found.")

    try:
        result = subprocess.run(['gcc', '--version'], capture_output=True, text=True, check=True)
        print(f"GCC version:  {result.stdout.splitlines()[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GCC not found.")

def test_gpu_computation():
    """Test actual GPU computation"""
    print("\n=== GPU Computation Test ===")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print(f"✅ GPU computation successful on: {z.device}")
            return True
        else:
            print("❌ CUDA not available for computation test")
            return False
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

if __name__ == "__main__":
    main()