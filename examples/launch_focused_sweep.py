#!/usr/bin/env python3
"""
Launcher for GainAKT2Exp focused parameter sweep with GPU options
"""

import subprocess
import sys

def main():
    # Check for non-interactive mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "multi" or mode == "multigpu" or mode == "1":
            print("🚀 Non-interactive Improved Multi-GPU Sweep Launch")
            print("=" * 50)
            script_name = 'run_improved_multi_gpu_sweep.py'
        elif mode == "single" or mode == "singlegpu" or mode == "2":
            print("🔄 Non-interactive Single-GPU Sweep Launch") 
            print("=" * 50)
            script_name = 'run_focused_sweep.py'
        else:
            print(f"Invalid mode: {mode}")
            print("Use: python launch_focused_sweep.py [multi|single|improved]")
            return 1
    else:
        # Interactive mode
        print("🎯 GainAKT2Exp Focused Parameter Sweep Launcher")
        print("=" * 60)
        print("Goal: Find parameter combinations achieving AUC >= 0.7259")
        print()
        
        print("Choose sweep mode:")
        print("1. 🚀 Improved Multi-GPU Sweep (3 GPUs, 21 experiments, ~45-60 min)")
        print("2. 🔄 Single-GPU Sweep (1 GPU, 20 experiments, ~2-3 hours)")
        print("3. ❌ Cancel")
        print()
        
        choice = input("Enter choice [1/2/3]: ").strip()
        
        if choice == '1':
            script_name = 'run_improved_multi_gpu_sweep.py'
            print("\\n🚀 Launching Improved Multi-GPU sweep across 3 GPUs...")
            
        elif choice == '2':
            script_name = 'run_focused_sweep.py'
            print("\\n🔄 Launching Single-GPU sweep...")
            
        elif choice == '3' or choice.lower() == 'n':
            print("Sweep cancelled.")
            return 0
            
        else:
            print("Invalid choice. Sweep cancelled.")
            return 1
    
    try:
        # Run the chosen sweep script with auto-confirm for non-interactive mode
        if len(sys.argv) > 1:
            # Non-interactive mode - pass auto-confirm flag
            result = subprocess.run([sys.executable, script_name, '--yes'], 
                                  cwd='/workspaces/pykt-toolkit/examples')
        else:
            # Interactive mode - let script handle its own prompts
            result = subprocess.run([sys.executable, script_name], 
                                  cwd='/workspaces/pykt-toolkit/examples')
        return result.returncode
        
    except KeyboardInterrupt:
        print("\\n⏹️ Sweep interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Error running sweep: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())