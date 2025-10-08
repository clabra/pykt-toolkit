#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/workspaces/pykt-toolkit')

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_improved_multi_gpu_sweep import ImprovedMultiGPUSweep

def main():
    print("🔬 Testing AUC Parsing with 2 Quick Experiments")
    print("=" * 60)
    
    # Create sweep instance
    sweep = ImprovedMultiGPUSweep()
    
    # Generate 2 test parameter combinations
    test_params = [
        {
            'lr': 0.000348,
            'wd': 5.857e-05,
            'bs': 64,
            'epochs': 10,
            'enhanced': True
        },
        {
            'lr': 0.000400,
            'wd': 6.0e-05,
            'bs': 96,
            'epochs': 10,
            'enhanced': False
        }
    ]
    
    print(f"📋 Running {len(test_params)} test experiments...")
    print(f"🎯 Target AUC: >= 0.7259\n")
    
    # Run experiments
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        
        for i, params in enumerate(test_params):
            gpu_id = i  # Use GPU 0 and 1
            run_id = i + 1
            future = executor.submit(sweep.run_single_gpu_experiment, gpu_id, run_id, params)
            futures[future] = (gpu_id, run_id, params)
        
        # Collect results
        for future in as_completed(futures):
            gpu_id, run_id, params = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"✅ Test {run_id}: SUCCESS - AUC parsing worked!")
                else:
                    print(f"❌ Test {run_id}: FAILED - AUC parsing failed")
            except Exception as e:
                print(f"💥 Test {run_id}: ERROR - {str(e)}")
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Total tests: {len(test_params)}")
    print(f"   Successful: {len(results)}")
    print(f"   Failed: {len(test_params) - len(results)}")
    
    if results:
        print(f"\n🎉 AUC Parsing is working! Sample results:")
        for result in results[:2]:
            print(f"   Run {result['run_id']}: AUC = {result['auc']:.4f}")
        
        # Save results
        timestamp = "auc_parsing_test"
        results_file = f"cumulative_mastery_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    else:
        print(f"\n❌ AUC parsing still needs debugging")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())