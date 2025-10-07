#!/usr/bin/env python3
"""
Wait for and display results from the current sweep
"""

import os
import json
import time
import glob
from datetime import datetime

def find_newest_results():
    """Find the newest results file."""
    pattern = "parallel_sweep_results_20251006_084*.json"
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getctime)
    return None

def display_progress():
    """Display current progress."""
    print("🔍 Checking for current sweep results...")
    
    results_file = find_newest_results()
    if not results_file:
        print("⏳ No results file found yet - training runs still in progress")
        return False
    
    print(f"📊 Found results: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        total_runs = len(results)
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') in ['failed', 'error', 'timeout'])
        
        print(f"📋 Progress: {total_runs}/8 runs completed")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if successful > 0:
            successful_runs = [r for r in results if r.get('status') == 'success' and 'metrics' in r]
            best_auc_run = max(successful_runs, key=lambda x: x['metrics'].get('best_val_auc', 0))
            best_auc = best_auc_run['metrics'].get('best_val_auc', 0)
            
            print(f"\\n🏆 BEST AUC SO FAR: {best_auc:.4f}")
            print(f"   Run {best_auc_run['run_id']} on GPU {best_auc_run['gpu_id']}")
            
            # Show parameters
            params = best_auc_run['parameters']
            print("   Parameters:")
            for key, value in params.items():
                if isinstance(value, float):
                    print(f"     {key}: {value:.4f}")
                else:
                    print(f"     {key}: {value}")
        
        return total_runs >= 8  # All runs completed
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False

if __name__ == "__main__":
    print("🚀 WAITING FOR CURRENT SWEEP RESULTS")
    print("=" * 50)
    print("Expected completion: ~10:00 AM")
    print("Checking every 2 minutes...")
    print("=" * 50)
    
    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\\n⏰ {current_time}")
        
        completed = display_progress()
        
        if completed:
            print("\\n🎉 SWEEP COMPLETED!")
            break
        
        print("\\n⏳ Checking again in 2 minutes...")
        time.sleep(120)  # Wait 2 minutes