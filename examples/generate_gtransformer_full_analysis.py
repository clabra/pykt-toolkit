"""
Comprehensive GTransformer Analysis Pipeline
Regenerates all interpretability outputs, plots, and probing visualizations.

This script orchestrates the complete post-training analysis for GTransformer experiments:
1. Interpretability alignment (trajectories, rosters)
2. Analysis plots (loss evolution, parameter distributions, mastery alignment)
3. Probing validation (train probes, generate visualizations)
4. Advanced visualizations (clusters, skill maps, curriculum heatmaps)

Usage:
    python examples/generate_gtransformer_full_analysis.py --experiment_dir experiments/YYYYMMDD_HHMMSS_gtransformer_...
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate complete GTransformer analysis")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--skip_interpretability", action="store_true", help="Skip interpretability alignment (if already done)")
    parser.add_argument("--skip_probing", action="store_true", help="Skip probing analysis")
    parser.add_argument("--skip_viz", action="store_true", help="Skip advanced visualizations")
    return parser.parse_args()

def run_step(cmd, step_name):
    """Execute a pipeline step and report status."""
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  {step_name} failed with exit code {result.returncode}")
        return False
    print(f"✅ {step_name} completed successfully")
    return True

def main():
    args = parse_args()
    exp_dir = Path(args.experiment_dir).absolute()
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    # Load config to get dataset and fold info
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"❌ config.json not found in {exp_dir}")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    dataset = config['input'].get('dataset', 'assist2009_S')
    fold = config['input'].get('fold', 0)
    checkpoint = exp_dir / "best_model.pt"
    
    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint}")
        return
    
    python_path = sys.executable
    project_root = exp_dir.parent.parent
    
    print(f"\n{'='*80}")
    print(f"GTransformer COMPREHENSIVE ANALYSIS PIPELINE")
    print(f"{'='*80}")
    print(f"Experiment: {exp_dir.name}")
    print(f"Dataset: {dataset}")
    print(f"Fold: {fold}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*80}\n")
    
    # Step 1: Interpretability Alignment
    if not args.skip_interpretability:
        interp_cmd = (
            f"{python_path} examples/eval_gtransformer_interpretability.py "
            f"--checkpoint {checkpoint} "
            f"--output_dir {exp_dir} "
            f"--dataset {dataset} "
            f"--fold {fold}"
        )
        run_step(interp_cmd, "Interpretability Alignment (Trajectories & Rosters)")
    
    # Step 2: Analysis Plots
    plots_cmd = (
        f"{python_path} examples/generate_analysis_plots.py "
        f"--run_dir {exp_dir}"
    )
    run_step(plots_cmd, "Analysis Plots (Loss Evolution, Parameter Distributions)")
    
    # Step 3: Validation Plots
    validation_cmd = (
        f"{python_path} examples/generate_validation_plots.py "
        f"--run_dir {exp_dir}"
    )
    run_step(validation_cmd, "Validation Plots (Consensus, Residuals, Uncertainty)")
    
    # Step 4: Probing Analysis
    if not args.skip_probing:
        traj_preds = exp_dir / "traj_predictions.csv"
        if traj_preds.exists():
            probing_output = exp_dir / "probing_plots"
            probing_output.mkdir(exist_ok=True)
            
            probe_cmd = (
                f"{python_path} examples/train_probe.py "
                f"--checkpoint {checkpoint} "
                f"--bkt_preds {traj_preds} "
                f"--output_dir {exp_dir} "
                f"--dataset {dataset} "
                f"--fold {fold}"
            )
            run_step(probe_cmd, "Probing Analysis (Diagnostic Validation)")
            
            # Probing visualizations
            viz_probe_cmd = f"{python_path} examples/viz_probing_report.py --experiment_dir {exp_dir}"
            run_step(viz_probe_cmd, "Probing Visualizations (Distributions)")
        else:
            print(f"⚠️  Skipping probing: traj_predictions.csv not found")
    
    # Step 5: Advanced Visualizations
    if not args.skip_viz:
        roster_gtransformer = exp_dir / "roster_gtransformer.csv"
        if roster_gtransformer.exists():
            # Student clusters
            viz_clusters_cmd = f"{python_path} examples/viz_student_clusters.py --experiment_dir {exp_dir}"
            run_step(viz_clusters_cmd, "Student Cluster Analysis")
            
            # Skill mastery map
            viz_skill_map_cmd = f"{python_path} examples/viz_skill_mastery_map.py --experiment_dir {exp_dir} --dataset {dataset}"
            run_step(viz_skill_map_cmd, "Pedagogical Skill Map")
            
            # Curriculum calibration heatmap
            viz_curriculum_cmd = f"{python_path} examples/viz_curriculum_heatmap.py --experiment_dir {exp_dir}"
            run_step(viz_curriculum_cmd, "Curriculum Calibration Heatmap")
            
            # Student trajectories
            viz_traj_cmd = f"{python_path} examples/viz_student_trajectories.py --experiment_dir {exp_dir}"
            run_step(viz_traj_cmd, "Student Learning Trajectories")
            
            # t-SNE probing
            viz_tsne_cmd = f"{python_path} examples/viz_tsne_probing.py --experiment_dir {exp_dir}"
            run_step(viz_tsne_cmd, "t-SNE Latent Space Visualization")
            
            # Skill bifurcation analysis
            viz_bifurc_cmd = f"{python_path} examples/viz_skill_bifurcation.py --experiment_dir {exp_dir} --skill_id 68"
            run_step(viz_bifurc_cmd, "Skill Bifurcation Analysis")
            
            # Skill zoom analysis
            viz_zoom_cmd = f"{python_path} examples/viz_skill_zoom.py --experiment_dir {exp_dir} --skill_id 68"
            run_step(viz_zoom_cmd, "Skill-Level Zoom Analysis")
        else:
            print(f"⚠️  Skipping advanced visualizations: roster_gtransformer.csv not found")
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated outputs in: {exp_dir}")
    print(f"  - Trajectories: traj_*.csv")
    print(f"  - Rosters: roster_*.csv")
    print(f"  - Plots: plots/")
    print(f"  - Probing: probing_plots/")
    print(f"  - Alignment: interpretability_alignment.json")

if __name__ == "__main__":
    main()
