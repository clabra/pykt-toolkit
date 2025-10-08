#!/usr/bin/env python3
"""
GPU availability checker for multi-GPU sweep
"""

import subprocess
import sys

def check_gpu_availability():
    """Check GPU availability and utilization"""
    
    try:
        # Get GPU info
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ nvidia-smi not available")
            return False
            
        gpus = []
        for line in result.stdout.strip().split('\\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_info = {
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_total': int(parts[4])
                    }
                    gpus.append(gpu_info)
        
        print("🔍 GPU Availability Check")
        print("=" * 50)
        
        available_gpus = []
        for gpu in gpus:
            memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            status = "🟢 AVAILABLE" if gpu['utilization'] < 10 and memory_percent < 20 else "🔴 BUSY"
            
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"   Utilization: {gpu['utilization']}%, Memory: {memory_percent:.1f}% - {status}")
            
            if gpu['utilization'] < 10 and memory_percent < 20:
                available_gpus.append(gpu['index'])
        
        print(f"\\n📊 Summary: {len(available_gpus)}/{len(gpus)} GPUs available")
        
        if len(available_gpus) >= 5:
            print("✅ Sufficient GPUs for multi-GPU sweep (need 5, have {})".format(len(available_gpus)))
            return True
        elif len(available_gpus) >= 1:
            print(f"⚠️  Limited GPUs available ({len(available_gpus)}). Consider single-GPU sweep.")
            return True
        else:
            print("❌ No GPUs available. All GPUs are currently busy.")
            return False
            
    except Exception as e:
        print(f"❌ Error checking GPUs: {e}")
        return False

def main():
    """Main function"""
    
    if not check_gpu_availability():
        print("\\nRecommendation: Wait for GPUs to become available or use CPU training.")
        return 1
    
    print("\\n🚀 GPUs ready for parameter sweep!")
    return 0

if __name__ == "__main__":
    sys.exit(main())