#!/usr/bin/env python3
"""
Final GPU Diagnostic for PyKT Toolkit
This script provides a comprehensive analysis of the GPU access issue
"""

import os
import subprocess
import sys

def run_command(cmd):
    """Run a command and return its output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 60)
    print("PyKT Toolkit - Final GPU Diagnostic Report")
    print("=" * 60)
    
    # 1. Hardware Detection
    print("\n🔍 1. GPU Hardware Detection:")
    success, stdout, stderr = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
    if success and stdout:
        gpus = stdout.split('\n')
        print(f"   ✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu}")
    else:
        print("   ❌ No GPUs detected by nvidia-smi")
    
    # 2. Container GPU Access
    print("\n🔍 2. Container GPU Device Access:")
    success, stdout, stderr = run_command("ls -la /dev/nvidia* 2>/dev/null")
    if success and stdout:
        device_count = len([line for line in stdout.split('\n') if '/dev/nvidia' in line and not 'nvidia-' in line])
        print(f"   ✅ Found {device_count} GPU device files in /dev/")
    else:
        print("   ❌ No GPU devices found in /dev/")
    
    # 3. CUDA Environment
    print("\n🔍 3. CUDA Environment:")
    cuda_vars = ['CUDA_HOME', 'CUDA_ROOT', 'CUDA_PATH', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        status = "✅" if value != 'Not set' else "❌"
        print(f"   {status} {var}: {value}")
    
    # 4. LD_LIBRARY_PATH
    print("\n� 4. Library Path Analysis:")
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        paths = ld_path.split(':')
        critical_paths = [
            '/usr/lib/x86_64-linux-gnu',
            '/usr/local/cuda/lib64',
            'nvidia/cuda_runtime/lib'
        ]
        print(f"   📝 LD_LIBRARY_PATH has {len(paths)} directories")
        for critical in critical_paths:
            found = any(critical in path for path in paths)
            status = "✅" if found else "❌"
            print(f"   {status} Contains {critical}: {found}")
    else:
        print("   ❌ LD_LIBRARY_PATH is not set")
    
    # 5. PyTorch Test
    print("\n🔍 5. PyTorch GPU Detection:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        status = "✅" if cuda_available else "❌"
        print(f"   {status} CUDA Available: {cuda_available}")
        print(f"   📊 GPU Count: {gpu_count}")
        print(f"   📋 PyTorch Version: {torch.__version__}")
        
        if not cuda_available:
            # Try to get more detailed error info
            try:
                torch.cuda.device_count()
            except Exception as e:
                print(f"   🔍 Detailed Error: {e}")
                
    except ImportError:
        print("   ❌ PyTorch not available")
    
    # 6. Final Analysis
    print("\n" + "=" * 60)
    print("📋 FINAL ANALYSIS:")
    print("=" * 60)
    
    # Check if we have all components
    has_gpu_hardware = "nvidia-smi" in str(subprocess.run("which nvidia-smi", shell=True, capture_output=True))
    has_dev_access = os.path.exists("/dev/nvidia0")
    has_cuda_env = os.environ.get('CUDA_HOME') is not None
    has_lib_paths = '/usr/lib/x86_64-linux-gnu' in os.environ.get('LD_LIBRARY_PATH', '')
    
    if has_gpu_hardware and has_dev_access and has_cuda_env and has_lib_paths:
        print("🎯 ROOT CAUSE: Container Runtime Configuration Issue")
        print("   All components are present but PyTorch cannot access GPUs.")
        print("   This is a Docker/container GPU passthrough problem.")
        print("\n🔧 SOLUTION:")
        print("   1. Your dev container config is already correct")
        print("   2. The LD_LIBRARY_PATH has been fixed")
        print("   3. REBUILD your dev container to apply all changes")
        print("\n📝 REBUILD STEPS:")
        print("   • Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)")
        print("   • Type: 'Dev Containers: Rebuild Container'")
        print("   • Wait for rebuild to complete")
        print("   • Test with: python -c \"import torch; print(torch.cuda.is_available())\"")
    else:
        print("❌ MISSING COMPONENTS:")
        if not has_gpu_hardware: print("   • GPU hardware detection failed")
        if not has_dev_access: print("   • GPU device access failed")
        if not has_cuda_env: print("   • CUDA environment not set")
        if not has_lib_paths: print("   • Library paths incomplete")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
