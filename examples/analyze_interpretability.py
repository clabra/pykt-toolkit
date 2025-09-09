#!/usr/bin/env python
# coding=utf-8
"""
This script analyzes the interpretability of a trained GainAKT2 model by calculating
the correlation between the model's projected skill mastery and its predictions.
"""

import argparse
import numpy as np
import torch
from scipy.stats import pearsonr
import json

from pykt.models import init_model
from pykt.datasets import init_dataset4train

device = "cpu" if not torch.cuda.is_available() else "cuda"

def main(params):
    # Load model
    model_config = params.copy()
    
    with open(f"../configs/data_config.json") as fin:
        data_config = json.load(fin)

    # Clean up model_config
    for key in ["model_name", "dataset_name", "load_model_path", "emb_type", "batch_size", "fold"]:
        if key in model_config:
            del model_config[key]

    # Pass use_mastery_head and use_gain_head from params to model_config
    model_config['use_mastery_head'] = params.get('use_mastery_head', 0)
    model_config['use_gain_head'] = params.get('use_gain_head', 0)

    model = None # Initialize model to None for debugging
    try:
        print("Attempting to initialize model...")
        model = init_model(params["model_name"], model_config, data_config[params["dataset_name"]], params["emb_type"])
        print(f"Model initialized: {model is not None}")
        assert model is not None, "Model initialization failed: init_model returned None."
        
        print(f"Attempting to load state_dict from: {params['load_model_path']}")
        net = torch.load(params["load_model_path"], map_location=device)
        model.load_state_dict(net)
        print("State dict loaded.")
        
        model.to(device)
        model.eval()
        print("Model moved to device and set to eval mode.")

    except Exception as e:
        print(f"Error during model initialization or loading: {e}")
        import traceback
        traceback.print_exc()
        return # Exit if model loading fails

    # Load data
    train_loader, valid_loader = init_dataset4train(params["dataset_name"], params["model_name"], data_config, params["fold"], params["batch_size"])

    all_projections = []
    all_predictions = []
    all_skills = []
    all_masks = []

    with torch.no_grad():
        for data in valid_loader:
            dcur = data
            c, r, cshft = dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["shft_cseqs"].to(device)
            sm = dcur["smasks"].to(device)

            output = model(c.long(), r.long(), cshft.long())

            # Ensure the model output is as expected
            if isinstance(output, dict) and 'predictions' in output and 'projected_mastery' in output:
                predictions = output['predictions']
                projected_mastery = torch.sigmoid(output['projected_mastery']) # Apply sigmoid to get probabilities

                all_predictions.append(predictions.cpu().numpy())
                all_projections.append(projected_mastery.cpu().numpy())
                all_skills.append(cshft.cpu().numpy())
                all_masks.append(sm.cpu().numpy())
            else:
                print("Warning: Model output is not in the expected format. Skipping batch.")
                continue

    # Process results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_projections = np.concatenate(all_projections, axis=0)
    all_skills = np.concatenate(all_skills, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    num_skills = all_projections.shape[-1]
    skill_correlations = {}

    for skill_id in range(num_skills):
        # Find all interactions where this skill was the target
        skill_mask = (all_skills == skill_id) & (all_masks == 1)

        if np.sum(skill_mask) > 1: # Need at least 2 data points to calculate correlation
            skill_preds = all_predictions[skill_mask]
            skill_projs = all_projections[skill_mask, skill_id]

            # Calculate Pearson correlation
            corr, _ = pearsonr(skill_preds, skill_projs)
            skill_correlations[skill_id] = corr

    # Print results
    print("\n--- Mastery-Performance Correlation Analysis ---")
    valid_correlations = [c for c in skill_correlations.values() if not np.isnan(c)]
    if valid_correlations:
        average_correlation = np.mean(valid_correlations)
        print(f"Average correlation across all skills: {average_correlation:.4f}")
    else:
        print("Could not calculate correlations for any skill.")

    print("\nCorrelation for a few sample skills:")
    for i, skill_id in enumerate(list(skill_correlations.keys())[:10]):
        print(f"  Skill {skill_id}: {skill_correlations[skill_id]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gainakt2")
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--load_model_path", type=str, required=True)
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fold", type=int, default=0)
    # Add any other necessary model config arguments here
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_encoder_blocks", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--use_mastery_head", type=int, default=1)
    parser.add_argument("--use_gain_head", type=int, default=1)

    args = parser.parse_args()
    params = vars(args)
    main(params)
