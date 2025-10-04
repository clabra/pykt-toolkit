#!/usr/bin/env python3
"""
Simple GPU Usage Monitor Script

Usage:
    python gpu_monitor.py          # Show current status
    python gpu_monitor.py --watch  # Continuous monitoring
    python gpu_monitor.py --json   # JSON output
"""

import subprocess
import json
import argparse
import time
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi"""
    try:
        # Get GPU utilization and memory info
        cmd = [
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
            
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    gpu_stats.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization_gpu': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                        'utilization_memory': int(parts[3]) if parts[3] != '[Not Supported]' else 0,
                        'memory_used_mb': int(parts[4]),
                        'memory_total_mb': int(parts[5]),
                        'temperature_c': int(parts[6]) if parts[6] != '[Not Supported]' else 0,
                        'power_draw_w': float(parts[7]) if parts[7] != '[Not Supported]' else 0,
                        'power_limit_w': float(parts[8]) if parts[8] != '[Not Supported]' else 0
                    })
        
        return gpu_stats
        
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def format_memory(mb):
    """Format memory in MB to human readable format"""
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    else:
        return f"{mb}MB"

def print_gpu_status(gpu_stats, show_header=True):
    """Print GPU status in a formatted table"""
    if not gpu_stats:
        print("❌ No GPU statistics available")
        return
        
    if show_header:
        print(f"\n🖥️  GPU Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 90)
        print(f"{'GPU':<3} {'Name':<20} {'GPU%':<6} {'Mem%':<6} {'Memory Used':<12} {'Temp':<6} {'Power':<10}")
        print("-" * 90)
    
    for gpu in gpu_stats:
        memory_pct = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
        memory_str = f"{format_memory(gpu['memory_used_mb'])}/{format_memory(gpu['memory_total_mb'])}"
        power_str = f"{gpu['power_draw_w']:.0f}W/{gpu['power_limit_w']:.0f}W"
        
        # Color coding for utilization
        gpu_util = gpu['utilization_gpu']
        if gpu_util >= 90:
            util_indicator = f"{gpu_util}% 🔥"
        elif gpu_util >= 50:
            util_indicator = f"{gpu_util}% ⚡"
        elif gpu_util > 0:
            util_indicator = f"{gpu_util}% 💡"
        else:
            util_indicator = f"{gpu_util}% 💤"
            
        print(f"{gpu['index']:<3} {gpu['name'][:18]:<20} {util_indicator:<6} "
              f"{memory_pct:.1f}%{'':<2} {memory_str:<12} {gpu['temperature_c']}°C{'':<2} {power_str:<10}")

def main():
    parser = argparse.ArgumentParser(description='GPU Usage Monitor')
    parser.add_argument('--watch', '-w', action='store_true', help='Continuous monitoring')
    parser.add_argument('--json', '-j', action='store_true', help='Output in JSON format')
    parser.add_argument('--interval', '-i', type=int, default=2, help='Update interval for watch mode (seconds)')
    
    args = parser.parse_args()
    
    try:
        if args.watch:
            print("🔍 Continuous GPU monitoring (Press Ctrl+C to stop)")
            print("=" * 90)
            
            while True:
                gpu_stats = get_gpu_stats()
                if gpu_stats:
                    # Clear screen and print updated status
                    print("\033[2J\033[H")  # Clear screen and move cursor to top
                    print_gpu_status(gpu_stats)
                    
                    # Summary statistics
                    active_gpus = [g for g in gpu_stats if g['utilization_gpu'] > 0]
                    total_memory_used = sum(g['memory_used_mb'] for g in gpu_stats)
                    total_memory = sum(g['memory_total_mb'] for g in gpu_stats)
                    
                    print("\n📊 Summary:")
                    print(f"   Active GPUs: {len(active_gpus)}/{len(gpu_stats)}")
                    print(f"   Total Memory: {format_memory(total_memory_used)}/{format_memory(total_memory)} ({total_memory_used/total_memory*100:.1f}%)")
                    
                    if active_gpus:
                        avg_util = sum(g['utilization_gpu'] for g in active_gpus) / len(active_gpus)
                        print(f"   Average GPU Utilization: {avg_util:.1f}%")
                else:
                    print("❌ Failed to get GPU statistics")
                
                time.sleep(args.interval)
                
        else:
            gpu_stats = get_gpu_stats()
            
            if args.json:
                print(json.dumps(gpu_stats, indent=2))
            else:
                print_gpu_status(gpu_stats)
                
                if gpu_stats:
                    # Summary
                    active_gpus = [g for g in gpu_stats if g['utilization_gpu'] > 0]
                    print(f"\n📊 Summary: {len(active_gpus)}/{len(gpu_stats)} GPUs active")
                    
                    if active_gpus:
                        max_util_gpu = max(active_gpus, key=lambda x: x['utilization_gpu'])
                        print(f"🔥 Highest utilization: GPU {max_util_gpu['index']} at {max_util_gpu['utilization_gpu']}%")
            
    except KeyboardInterrupt:
        print("\n\n👋 GPU monitoring stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()