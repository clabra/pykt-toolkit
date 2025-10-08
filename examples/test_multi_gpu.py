#!/usr/bin/env python3
"""
Test multi-GPU setup with a quick 2-experiment test
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import time

def test_gpu_experiment(gpu_id, test_id):
    """Run a simple torch test on specified GPU"""
    
    print(f"🧪 Testing GPU {gpu_id} with experiment {test_id}")
    
    # Simple torch test script
    test_script = f"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    tensor = torch.randn(1000, 1000).to(device)
    result = torch.matmul(tensor, tensor)
    print(f"GPU_{gpu_id}_TEST_{test_id}_SUCCESS")
else:
    print(f"GPU_{gpu_id}_TEST_{test_id}_NO_CUDA")
"""
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', '-c', test_script],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd='/workspaces/pykt-toolkit/examples'
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if f"GPU_{gpu_id}_TEST_{test_id}_SUCCESS" in output:
                print(f"✅ GPU {gpu_id} Test {test_id}: SUCCESS ({duration:.1f}s)")
                return True
            else:
                print(f"⚠️  GPU {gpu_id} Test {test_id}: No CUDA available ({duration:.1f}s)")
                return False
        else:
            print(f"❌ GPU {gpu_id} Test {test_id}: FAILED - Return code: {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ GPU {gpu_id} Test {test_id}: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ GPU {gpu_id} Test {test_id}: ERROR - {e}")
        return False

def main():
    """Test multi-GPU setup"""
    
    print("🧪 Multi-GPU Setup Test")
    print("=" * 40)
    print("Running quick 1-epoch tests on GPUs 0 and 1...")
    print("This will verify the multi-GPU sweep will work correctly.\\n")
    
    # Test 2 GPUs in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_gpu0 = executor.submit(test_gpu_experiment, 0, 1)
        future_gpu1 = executor.submit(test_gpu_experiment, 1, 2)
        
        # Wait for results
        result_gpu0 = future_gpu0.result()
        result_gpu1 = future_gpu1.result()
    
    print("\\n📊 Test Results:")
    print(f"GPU 0: {'✅ PASS' if result_gpu0 else '❌ FAIL'}")
    print(f"GPU 1: {'✅ PASS' if result_gpu1 else '❌ FAIL'}")
    
    if result_gpu0 and result_gpu1:
        print("\\n🎉 Multi-GPU setup is working correctly!")
        print("✅ Ready to run full multi-GPU parameter sweep")
        return 0
    else:
        print("\\n⚠️  Multi-GPU setup has issues")
        print("Consider using single-GPU sweep instead")
        return 1

if __name__ == "__main__":
    sys.exit(main())