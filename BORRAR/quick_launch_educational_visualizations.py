#!/usr/bin/env python3
"""
Quick launcher for educational visualization tools.
Generates comprehensive educational interpretability visualizations.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def find_latest_model(base_dir="saved_model"):
    """Find the most recently trained cumulative mastery model."""
    model_dirs = []
    
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and "cumulative_mastery" in item:
                # Look for model files
                model_files = ["best_model.pth", "model.pth"]
                for model_file in model_files:
                    model_path = os.path.join(item_path, model_file)
                    if os.path.exists(model_path):
                        model_dirs.append((item_path, model_path, os.path.getmtime(model_path)))
                        break
    
    if model_dirs:
        # Return the most recent model
        model_dirs.sort(key=lambda x: x[2], reverse=True)
        return model_dirs[0][1]  # Return the model path
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Quick launch educational visualizations')
    
    # Model selection
    parser.add_argument('--model_path', type=str, help='Path to specific model (auto-detect if not provided)')
    parser.add_argument('--auto_find', action='store_true', default=True,
                       help='Automatically find latest cumulative mastery model')
    
    # Visualization options
    parser.add_argument('--num_students', type=int, default=30,
                       help='Number of students to analyze (default: 30)')
    parser.add_argument('--output_dir', type=str, 
                       default=f"educational_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Output directory for visualizations')
    
    # Preset configurations
    parser.add_argument('--preset', type=str, default='standard',
                       choices=['quick', 'standard', 'comprehensive'],
                       help='Visualization preset')
    
    args = parser.parse_args()
    
    # Configure presets
    presets = {
        'quick': {'num_students': 10},
        'standard': {'num_students': 30},
        'comprehensive': {'num_students': 50}
    }
    
    if args.preset in presets:
        if args.num_students == 30:  # Default value, apply preset
            args.num_students = presets[args.preset]['num_students']
    
    # Find model if not specified
    if not args.model_path:
        print("🔍 Looking for latest cumulative mastery model...")
        args.model_path = find_latest_model()
        
        if args.model_path:
            print(f"✓ Found model: {args.model_path}")
        else:
            print("❌ No cumulative mastery model found!")
            print("Available options:")
            print("1. Train a model first: python quick_launch_cumulative_mastery.py --preset quick")
            print("2. Specify model path: --model_path path/to/your/model.pth")
            sys.exit(1)
    
    # Verify model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    print("🎨 LAUNCHING EDUCATIONAL VISUALIZATION SUITE")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Preset: {args.preset}")
    print(f"Students to analyze: {args.num_students}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Build command
    cmd = [
        sys.executable, 'educational_visualizer.py',
        '--model_path', args.model_path,
        '--output_dir', args.output_dir,
        '--num_students', str(args.num_students)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Launch visualization
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("\\n🎉 VISUALIZATION GENERATION COMPLETED!")
        print("=" * 60)
        print("📊 Generated Visualizations:")
        print("  • Individual student learning trajectories")
        print("  • Comprehensive consistency dashboard")
        print("  • Concept-level mastery evolution")
        print("  • Educational interpretability report")
        print("=" * 60)
        print(f"📁 All files saved to: {args.output_dir}")
        print("\\n💡 Next steps:")
        print(f"1. Open the visualizations: ls {args.output_dir}/")
        print("2. Review the educational report for detailed analysis")
        print("3. Use visualizations to demonstrate educational validity")
        
        if result.stdout:
            print("\\n📋 Detailed output:")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Visualization generation failed with exit code {e.returncode}")
        if e.stderr:
            print("Error details:")
            print(e.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\\n⏹️  Visualization interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()