#!/usr/bin/env python3
"""
Test script to verify if the rebuild actually applied our GPU fixes
"""
import os

def test_rebuild_success():
    print("🧪 Testing if rebuild applied our GPU fixes...")
    
    # Check if our LD_LIBRARY_PATH fix is active
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    critical_path = '/usr/lib/x86_64-linux-gnu'
    
    if critical_path in ld_path:
        print(f"✅ REBUILD SUCCESS: {critical_path} found in LD_LIBRARY_PATH")
        
        # Now test PyTorch
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()
            
            if cuda_available and gpu_count > 0:
                print(f"🎉 COMPLETE SUCCESS: PyTorch detects {gpu_count} GPUs!")
                return True
            else:
                print(f"❌ PARTIAL SUCCESS: Library path fixed but PyTorch still can't access GPUs")
                print("   This indicates a deeper host-level Docker/NVIDIA configuration issue")
                return False
                
        except ImportError:
            print("❌ PyTorch not available")
            return False
    else:
        print(f"❌ REBUILD FAILED: {critical_path} still missing from LD_LIBRARY_PATH")
        print("   The devcontainer configuration changes were not applied")
        return False

if __name__ == "__main__":
    success = test_rebuild_success()
    if not success:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Verify you rebuilt from the correct devcontainer configuration")
        print("2. Check if there are host-level Docker/NVIDIA runtime issues")
        print("3. Contact your system administrator about NVIDIA Container Runtime")

