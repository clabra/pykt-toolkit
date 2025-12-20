
import os
import sys
import torch
import pandas as pd
import numpy as np
import json
import argparse
import pickle
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.models.idkt import iDKT
from pykt.models.init_model import load_model

def get_student_accuracy(dataset_path):
    print(f"Calculating student historical accuracy from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Each row is a student sequence. Calculate accuracy per row.
    def calc_acc(row):
        responses = [int(r) for r in row["responses"].split(",")]
        masks = [int(m) for m in row["selectmasks"].split(",")]
        # Filter by masks
        filtered = [r for r, m in zip(responses, masks) if m == 1]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)
    
    df["acc"] = df.apply(calc_acc, axis=1)
    # Average accuracy per student UID
    student_stats = df.groupby("uid")["acc"].mean().dropna().to_dict()
    return student_stats

def validate_capability(model_path, dataset_path, data_config_path, model_config):
    # 1. Get student accuracy
    student_acc = get_student_accuracy(dataset_path)
    
    # 2. Load model
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    
    # Dynamically detect num_students from checkpoint
    num_students = state_dict['student_param.weight'].shape[0] - 1
    print(f"Detected {num_students} students in checkpoint.")
    
    # Load training data to get the same UID mapping
    train_df = pd.read_csv(dataset_path)
    unique_uids = sorted(train_df["uid"].unique())
    uid_to_index = {uid: idx for idx, uid in enumerate(unique_uids)}
    
    model = iDKT(
        n_question=model_config["num_c"],
        n_pid=model_config["num_q"],
        d_model=model_config["d_model"],
        n_blocks=model_config["n_blocks"],
        dropout=model_config["dropout"],
        d_ff=model_config["d_ff"],
        final_fc_dim=model_config["final_fc_dim"],
        n_uid=num_students,
        lambda_student=model_config.get("lambda_student", 1e-5)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Extract vs weights
    print("Extracting student capability embeddings (v_s)...")
    vs_weights = model.student_param.weight.data.cpu().numpy().flatten()
    
    # 4. Correlation Analysis
    proficiencies = []
    accuracies = []
    
    # Track which UIDs we actually have weights for
    valid_count = 0
    for uid, acc in student_acc.items():
        idx = int(uid)
        if idx < len(vs_weights):
            proficiencies.append(vs_weights[idx])
            accuracies.append(acc)
            valid_count += 1
    
    print(f"Matched {valid_count} students with learned embeddings.")
    
    if len(proficiencies) < 2:
        print("Error: Not enough data points for correlation.")
        return
    
    corr, p_value = pearsonr(proficiencies, accuracies)
    
    print("\n" + "="*50)
    print("STUDENT CAPABILITY VALIDATION RESULTS")
    print("="*50)
    print(f"Number of Students: {len(proficiencies)}")
    print(f"Pearson Correlation (v_s vs Accuracy): {corr:.4f}")
    print(f"P-Value: {p_value:.4e}")
    print("="*50)
    
    if corr > 0.5:
        print("✓ STRONG POSITIVE CORRELATION: v_s successfully captures student capability.")
    elif corr > 0.3:
        print("✓ MODERATE POSITIVE CORRELATION: v_s captures student capability.")
    else:
        print("⚠ WEAK CORRELATION: check training convergence or regularization.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="assist2015")
    parser.add_argument("--config_path", type=str, default="configs/parameter_default.json")
    args = parser.parse_args()
    
    # Load configs
    with open(args.config_path, "r") as f:
        config = json.load(f)["defaults"]
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)[args.dataset]
        
    dataset_path = os.path.join(project_root, data_config["dpath"].replace("../", ""), data_config["train_valid_file"])
    
    # Update model_config with dataset stats
    config["num_c"] = data_config["num_c"]
    config["num_q"] = data_config["num_q"]
    
    validate_capability(args.model_path, dataset_path, data_config_path, config)
