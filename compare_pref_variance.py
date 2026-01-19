#!/usr/bin/env python3
"""
Compare p_ref variance between static and time-varying BKT implementations.
This demonstrates the improvement from fixing the static parameter issue.
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def compute_pref_stats(output_dir):
    """Load predictions and compute p_ref statistics."""
    
    # Try to find the summary file
    summary_file = Path(output_dir) / "envelope_statistics.json"
    
    if summary_file.exists():
        with open(summary_file) as f:
            stats = json.load(f)
        return stats
    
    return None

def main():
    print("=" * 80)
    print("Comparing p_ref Variance: Static vs. Time-Varying BKT Parameters")
    print("=" * 80)
    
    # Directories
    static_dir = Path("examples/validation/results_exp533154")
    timevarying_dir = Path("examples/validation/results_exp533154_timevarying")
    
    print(f"\nStatic implementation: {static_dir}")
    print(f"Time-varying implementation: {timevarying_dir}")
    
    # Check if both exist
    if not static_dir.exists():
        print(f"\n⚠️  Static results not found at {static_dir}")
        return
    
    if not timevarying_dir.exists():
        print(f"\n⚠️  Time-varying results not found yet at {timevarying_dir}")
        print("   Gallery generation may still be running...")
        return
    
    # Load statistics
    print("\n" + "-" * 80)
    print("Loading statistics...")
    print("-" * 80)
    
    static_stats = compute_pref_stats(static_dir)
    timevarying_stats = compute_pref_stats(timevarying_dir)
    
    if static_stats and timevarying_stats:
        print("\n✓ Both result sets loaded successfully!")
        
        # Compare key metrics
        print("\n" + "=" * 80)
        print("COMPARISON: p_ref Dynamics")
        print("=" * 80)
        
        # This is a placeholder - actual comparison would require
        # loading the raw predictions and computing p_ref variance
        print("\n📊 Envelope Width Statistics:")
        print(f"  Static:        Mean={static_stats.get('mean_envelope', 'N/A'):.4f}")
        print(f"  Time-Varying:  Mean={timevarying_stats.get('mean_envelope', 'N/A'):.4f}")
        
        print("\n💡 Note: Full p_ref variance analysis requires raw predictions.")
        print("   Check the regenerated gallery plots for visual comparison!")
        
    else:
        print("\n⚠️  Statistics files not found yet.")
        print("   Run envelope distribution analysis on both result sets:")
        print("\n   python3 examples/validation/generate_envelope_distribution.py \\")
        print("     --exp_dir <experiment_dir> \\")
        print("     --output_dir examples/validation/results_exp533154")
        print("\n   python3 examples/validation/generate_envelope_distribution.py \\")
        print("     --exp_dir <experiment_dir> \\")
        print("     --output_dir examples/validation/results_exp533154_timevarying")
    
    print("\n" + "=" * 80)
    print("Check the gallery images for visual comparison:")
    print(f"  Original:       {static_dir}/prediction_envelope_gallery.png")
    print(f"  Time-Varying:   {timevarying_dir}/prediction_envelope_gallery.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
