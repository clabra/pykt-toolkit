#!/usr/bin/env python3

import json
import os
from datetime import datetime
import glob

def analyze_sweep_results():
    print("🔍 GainAKT2Exp Sweep Results Analysis")
    print("=" * 60)
    
    # Find all result files
    result_files = glob.glob("cumulative_mastery_results_*.json")
    result_files += glob.glob("*sweep_results_*.json")
    result_files += glob.glob("gainakt2exp_results_*.json") 
    
    if not result_files:
        print("❌ No result files found")
        return
    
    # Sort by modification time (most recent first)
    result_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"📊 Found {len(result_files)} result files")
    print(f"🕐 Analyzing most recent sweep results...\n")
    
    # Analyze recent files
    all_results = []
    target_auc = 0.7259
    
    for i, file_path in enumerate(result_files[:10]):  # Top 10 most recent
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different file formats
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict) and 'results' in data:
                results = data['results']
            elif isinstance(data, dict) and 'sweep_results' in data:
                results = data['sweep_results']
            elif isinstance(data, dict) and 'best_val_auc' in data:
                # Single experiment result format
                results = [{
                    'auc': data['best_val_auc'],
                    'experiment': data.get('experiment_name', 'unknown'),
                    'consistency_metrics': data.get('final_consistency_metrics', {}),
                    'train_history': data.get('train_history', {})
                }]
            else:
                results = [data] if 'auc' in data else []
            
            if results:
                timestamp = os.path.basename(file_path).split('_')[-1].replace('.json', '')
                file_info = {
                    'file': os.path.basename(file_path),
                    'timestamp': timestamp,
                    'results': results,
                    'count': len(results)
                }
                all_results.append(file_info)
                
        except Exception as e:
            continue
    
    if not all_results:
        print("❌ No valid results found in files")
        return
    
    print("📈 RECENT SWEEP PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Analyze each sweep
    for sweep in all_results[:5]:  # Top 5 sweeps
        results = sweep['results']
        if not results:
            continue
            
        aucs = [r.get('auc', 0) for r in results if 'auc' in r]
        if not aucs:
            continue
            
        best_auc = max(aucs)
        avg_auc = sum(aucs) / len(aucs)
        target_hits = sum(1 for auc in aucs if auc >= target_auc)
        
        print(f"\n🔬 {sweep['file']}")
        print(f"   📅 Timestamp: {sweep['timestamp']}")
        print(f"   🧪 Experiments: {len(aucs)}")
        print(f"   🎯 Best AUC: {best_auc:.4f}")
        print(f"   📊 Average AUC: {avg_auc:.4f}")
        print(f"   🏆 Target Hits (>= {target_auc}): {target_hits}/{len(aucs)}")
        
        if target_hits > 0:
            print(f"   ✨ SUCCESS RATE: {target_hits/len(aucs)*100:.1f}%")
        
        # Find best performing config
        best_result = max(results, key=lambda x: x.get('auc', 0))
        if 'params' in best_result:
            params = best_result['params']
            print(f"   🏅 Best Config:")
            print(f"      lr: {params.get('lr', 'N/A'):.6f}")
            print(f"      wd: {params.get('wd', 'N/A'):.6f}")  
            print(f"      bs: {params.get('bs', 'N/A')}")
            print(f"      enhanced: {params.get('enhanced', 'N/A')}")
    
    # Overall performance analysis
    print(f"\n🎯 OVERALL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    all_aucs = []
    all_configs = []
    
    for sweep in all_results:
        for result in sweep['results']:
            if 'auc' in result:
                all_aucs.append(result['auc'])
                all_configs.append(result)
    
    if all_aucs:
        total_experiments = len(all_aucs)
        best_overall = max(all_aucs)
        avg_overall = sum(all_aucs) / len(all_aucs)
        target_hits_total = sum(1 for auc in all_aucs if auc >= target_auc)
        
        print(f"📊 Total Experiments: {total_experiments}")
        print(f"🏆 Best AUC Ever: {best_overall:.4f}")
        print(f"📈 Overall Average: {avg_overall:.4f}")
        print(f"🎯 Total Target Hits: {target_hits_total}/{total_experiments} ({target_hits_total/total_experiments*100:.1f}%)")
        
        # Best configuration overall
        best_config = max(all_configs, key=lambda x: x.get('auc', 0))
        if 'params' in best_config:
            print(f"\n🥇 BEST CONFIGURATION EVER:")
            print(f"   AUC: {best_config['auc']:.4f}")
            params = best_config['params']
            print(f"   Learning Rate: {params.get('lr', 'N/A')}")
            print(f"   Weight Decay: {params.get('wd', 'N/A')}")
            print(f"   Batch Size: {params.get('bs', 'N/A')}")
            print(f"   Enhanced: {params.get('enhanced', 'N/A')}")
            if 'duration_minutes' in best_config:
                print(f"   Duration: {best_config['duration_minutes']:.1f} min")
        
        # Parameter analysis
        print(f"\n🔬 PARAMETER INSIGHTS:")
        
        # Enhanced vs non-enhanced
        enhanced_aucs = [r['auc'] for r in all_configs if r.get('params', {}).get('enhanced') == True]
        non_enhanced_aucs = [r['auc'] for r in all_configs if r.get('params', {}).get('enhanced') == False]
        
        if enhanced_aucs and non_enhanced_aucs:
            print(f"   Enhanced=True avg: {sum(enhanced_aucs)/len(enhanced_aucs):.4f} ({len(enhanced_aucs)} experiments)")
            print(f"   Enhanced=False avg: {sum(non_enhanced_aucs)/len(non_enhanced_aucs):.4f} ({len(non_enhanced_aucs)} experiments)")
        
        # Learning rate ranges
        lrs = [r['params'].get('lr') for r in all_configs if 'params' in r and 'lr' in r['params']]
        if lrs:
            print(f"   Learning Rate range: {min(lrs):.6f} - {max(lrs):.6f}")
        
        # Top 3 configurations
        top_configs = sorted(all_configs, key=lambda x: x.get('auc', 0), reverse=True)[:3]
        print(f"\n🏆 TOP 3 CONFIGURATIONS:")
        for i, config in enumerate(top_configs, 1):
            if 'auc' in config:
                print(f"   #{i}: AUC={config['auc']:.4f}")
                if 'params' in config:
                    p = config['params']
                    print(f"       lr={p.get('lr', 'N/A'):.6f}, wd={p.get('wd', 'N/A'):.6f}, bs={p.get('bs', 'N/A')}, enhanced={p.get('enhanced', 'N/A')}")

if __name__ == "__main__":
    analyze_sweep_results()