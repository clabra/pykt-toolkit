#!/usr/bin/env python
# coding=utf-8
"""
This script analyzes the interpretability of a trained GainAKT2 model by calculating
metrics for four key consistency requirements.
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

    model = init_model(params["model_name"], model_config, data_config[params["dataset_name"]], params["emb_type"])
    assert model is not None, "Model initialization failed: init_model returned None."
    net = torch.load(params["load_model_path"], map_location=device)
    model.load_state_dict(net, strict=False)
    model.to(device)
    model.eval()

    # Load data
    train_loader, valid_loader = init_dataset4train(params["dataset_name"], params["model_name"], data_config, params["fold"], params["batch_size"])

    all_projections = []
    all_predictions = []
    all_skills = []
    all_masks = []
    all_gains = []
    all_responses = []

    with torch.no_grad():
        for data in valid_loader:
            dcur = data
            c, r, cshft = dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["shft_cseqs"].to(device)
            sm = dcur["smasks"].to(device)

            output = model(c.long(), r.long(), cshft.long())

            # Ensure the model output is as expected
            if isinstance(output, dict) and 'predictions' in output:
                all_predictions.append(output['predictions'].cpu().numpy())
                all_skills.append(cshft.cpu().numpy())
                all_masks.append(sm.cpu().numpy())
                all_responses.append(r.cpu().numpy())

                if 'projected_mastery' in output:
                    all_projections.append(torch.sigmoid(output['projected_mastery']).cpu().numpy())
                
                if 'projected_gains' in output:
                    all_gains.append(output['projected_gains'].cpu().numpy())
            else:
                print("Warning: Model output is not in the expected format. Skipping batch.")
                continue

    # Process results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_skills = np.concatenate(all_skills, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_responses = np.concatenate(all_responses, axis=0)
    
    if all_projections:
        all_projections = np.concatenate(all_projections, axis=0)
    if all_gains:
        all_gains = np.concatenate(all_gains, axis=0)

    num_skills = data_config[params["dataset_name"]]["num_c"]

    # Metric 1: Mastery-Performance Correlation
    print("\n--- 1. Mastery-Performance Correlation Analysis ---")
    if len(all_projections) > 0:
        skill_correlations = {}
        for skill_id in range(num_skills):
            skill_mask = (all_skills == skill_id) & (all_masks == 1)
            if np.sum(skill_mask) > 1:
                skill_preds = all_predictions[skill_mask]
                skill_projs = all_projections[skill_mask, skill_id]
                corr, _ = pearsonr(skill_preds, skill_projs)
                skill_correlations[skill_id] = corr
        
        valid_correlations = [c for c in skill_correlations.values() if not np.isnan(c)]
        if valid_correlations:
            average_correlation = np.mean(valid_correlations)
            print(f"Average correlation across all skills: {average_correlation:.4f}")
        else:
            print("Could not calculate correlations for any skill.")
    else:
        print("Skipping: 'projected_mastery' not found in model output.")

    # Metric 2: Gain-Performance Correlation
    print("\n--- 2. Gain-Performance Correlation Analysis ---")
    if len(all_gains) > 0:
        gains_for_correct_responses = []
        gains_for_incorrect_responses = []
        
        # We extract the gain for the specific skill practiced in each interaction
        relevant_gains = all_gains[np.arange(all_gains.shape[0])[:, None], np.arange(all_gains.shape[1]), all_skills]
        
        # Apply the sequence mask
        active_gains = relevant_gains[all_masks == 1]
        active_responses = all_responses[all_masks == 1]

        if len(active_gains) > 1:
            gain_perf_corr, _ = pearsonr(active_gains, active_responses)
            print(f"Correlation between learning gain and response correctness: {gain_perf_corr:.4f}")
        else:
            print("Not enough data to calculate gain-performance correlation.")
    else:
        print("Skipping: 'projected_gains' not found in model output.")

    # Metric 3: Non-Negative Gains Violation Rate
    print("\n--- 3. Non-Negative Gains Violation Rate ---")
    if len(all_gains) > 0:
        active_gains_all_skills = all_gains[all_masks == 1]
        negative_gains = active_gains_all_skills[active_gains_all_skills < 0]
        violation_rate = len(negative_gains) / active_gains_all_skills.size if active_gains_all_skills.size > 0 else 0
        print(f"Percentage of projected learning gains that are negative: {violation_rate:.4%}")
    else:
        print("Skipping: 'projected_gains' not found in model output.")

    # Metric 4: Mastery Monotonicity Violation Rate
    print("\n--- 4. Mastery Monotonicity Violation Rate ---")
    if len(all_projections) > 0:
        # Compare mastery at step t with step t-1
        mastery_t = all_projections[:, 1:, :]
        mastery_t_minus_1 = all_projections[:, :-1, :]
        
        # Mask for valid transitions (where both t and t-1 are not padding)
        mask_t = all_masks[:, 1:]
        mask_t_minus_1 = all_masks[:, :-1]
        valid_transitions_mask = (mask_t == 1) & (mask_t_minus_1 == 1)
        
        # Find violations where mastery decreases
        mastery_diff = mastery_t - mastery_t_minus_1
        violations = mastery_diff[valid_transitions_mask] < 0
        
        monotonicity_violation_rate = np.sum(violations) / violations.size if violations.size > 0 else 0
        print(f"Percentage of instances where skill mastery decreases: {monotonicity_violation_rate:.4%}")
    else:
        print("Skipping: 'projected_mastery' not found in model output.")


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
