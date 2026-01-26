
import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from pykt.models import init_model
from pykt.datasets.gtransformer_dataloader import GTransformerDataset

def calculate_selectivity(X, y, skills, construct_name="Initial Mastery (L0)"):
    """
    Calculates various Selectivity metrics.
    1. Selectivity (Full): R2_true - R2_shuffled_obs
    2. Selectivity (Skill-level): R2_true - R2_shuffled_skills
    """
    print(f"\n--- Calculating Selectivity for {construct_name} ---")
    
    # 1. True Task: Recover BKT Parameters
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    r2_true = r2_score(y_test, y_pred)
    
    # 2. Control Task (Standard Shuffled): Shuffled targets (observation-level)
    # This matches the result in many interpretability papers including H1.1 in our validation.md.
    y_shuffled_obs = np.random.permutation(y)
    _, _, _, y_test_s = train_test_split(X, y_shuffled_obs, test_size=0.2, random_state=42)
    # R2 of a random guesser on shuffled data should be ~0 or negative
    r2_control_obs = r2_score(y_test_s, probe.predict(X_test)) # Evaluated with SAME probe? 
    # Actually, standard is to RETRAIN the probe on control.
    control_probe_obs = Ridge(alpha=1.0)
    control_probe_obs.fit(X_train, np.random.permutation(y_train))
    r2_control_obs = r2_score(y_test_s, control_probe_obs.predict(X_test))

    # 3. Control Task (Skill-consistent): Shuffled labels PER SKILL
    unique_skills = np.unique(skills)
    skill_to_true_val = {s: np.mean(y[skills == s]) for s in unique_skills}
    shuffled_vals = list(skill_to_true_val.values())
    np.random.seed(42)
    np.random.shuffle(shuffled_vals)
    skill_to_shuffled_val = dict(zip(unique_skills, shuffled_vals))
    y_shuffled_skills = np.array([skill_to_shuffled_val[s] for s in skills])
    
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y_shuffled_skills, test_size=0.2, random_state=42)
    control_probe_skill = Ridge(alpha=1.0)
    control_probe_skill.fit(X_train_k, y_train_k)
    r2_control_skill = r2_score(y_test_k, control_probe_skill.predict(X_test_k))
    
    print(f"{'Task':<25} | {'R^2':<10}")
    print("-" * 40)
    print(f"{'True Task':<25} | {r2_true:>10.4f}")
    print(f"{'Control (Shuffled Obs)':<25} | {r2_control_obs:>10.4f}")
    print(f"{'Control (Shuffled Skill)':<25} | {r2_control_skill:>10.4f}")
    print("-" * 40)
    print(f"{'SELECTIVITY (Standard)':<25} | {r2_true - r2_control_obs:>10.4f}")
    print(f"{'SELECTIVITY (Strict)':<25} | {r2_true - r2_control_skill:>10.4f}")
    
    return {
        "r2_true": r2_true,
        "r2_control_obs": r2_control_obs,
        "r2_control_skill": r2_control_skill,
        "selectivity_std": r2_true - r2_control_obs,
        "selectivity_strict": r2_true - r2_control_skill
    }

def extract_latent_all(model, loader, device):
    """
    Extracts z_context, l0, and t_targets.
    """
    model.eval()
    all_z = []
    all_l0 = []
    all_t = []
    all_skills = []
    
    steps = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            sm = data["smasks"].long().to(device)
            
            # For GTransformer, forward with qtest=True returns z_context
            _, _, z_context = model(c, r, pid_data=q, qtest=True)
            
            l0_targets = data["target_l0"].to(device) 
            t_targets = data["target_t"].to(device) 
            
            mask = sm.bool() 
            all_z.append(z_context[mask].cpu().numpy())
            all_l0.append(l0_targets[mask].cpu().numpy())
            all_t.append(t_targets[mask].cpu().numpy())
            all_skills.append(c[mask].cpu().numpy())
            
            steps += 1
            if steps > 200: break
                
    X = np.concatenate(all_z, axis=0)
    y_l0 = np.concatenate(all_l0, axis=0)
    y_t = np.concatenate(all_t, axis=0)
    skills = np.concatenate(all_skills, axis=0)
    
    return X, y_l0, y_t, skills

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment fold directory")
    parser.add_argument("--output_file", type=str, help="Where to save metrics JSON")
    args = parser.parse_args()

    EXP_DIR = args.exp_dir
    OUTPUT_FILE = args.output_file if args.output_file else os.path.join(EXP_DIR, "selectivity_metrics.json")
    
    # Checkpoint Discovery
    found_ckpt = None
    for root, dirs, files in os.walk(EXP_DIR):
        for file in files:
            if file.endswith(".ckpt"):
                found_ckpt = os.path.join(root, file)
                break
        if found_ckpt: break
    
    if not found_ckpt:
        print(f"Error: No checkpoint found in {EXP_DIR}")
        return

    CONFIG_PATH = os.path.join(EXP_DIR, "config.json")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    model_config = config.get('params', config.get('train_config', {}))
    dataset_name = model_config.get("dataset", "assist2009")
    fold = model_config.get("fold", 0)
    
    # Load data config
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    dpath = data_config[dataset_name]['dpath'].replace("../", "")
    dpath = os.path.join(project_root, dpath)
    data_config[dataset_name]['dpath'] = dpath
            
    # Init Dataset
    target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
    train_file = os.path.join(dpath, "train_valid_sequences.csv")
    test_dataset = GTransformerDataset(train_file, data_config[dataset_name]["input_type"], {fold}, target_path=target_path)
    loader = DataLoader(test_dataset, batch_size=model_config.get('batch_size', 64), shuffle=False, num_workers=4)
    
    # Model Setup
    checkpoint = torch.load(found_ckpt, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    bkt_params_path = os.path.join(dpath, "bkt_skill_params.pkl")
    with open(bkt_params_path, "rb") as f:
        bkt_skill_params = pickle.load(f)
    
    model = init_model('gtransformer', model_config, data_config[dataset_name], model_config.get('emb_type', 'qid'))
    model.load_theory_params(bkt_skill_params)
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extraction
    print("Extracting latent representations...")
    X, y_l0, y_t, skills = extract_latent_all(model, loader, device)
    
    # Downsample if extreme
    if len(X) > 10000:
        idx = np.random.choice(len(X), 10000, replace=False)
        X, y_l0, y_t, skills = X[idx], y_l0[idx], y_t[idx], skills[idx]
        
    # Results 
    res_l0 = calculate_selectivity(X, y_l0, skills, "Initial Mastery (L0)")
    res_t = calculate_selectivity(X, y_t, skills, "Learning Rate (T)")
    
    combined_results = {
        "fold": fold,
        "l0": res_l0,
        "t": res_t
    }
    
    # Save results in validation subfolder
    validation_dir = os.path.join(os.path.dirname(EXP_DIR.rstrip('/')), "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    output_filename = f"selectivity_metrics_fold_{fold}.json"
    output_path = os.path.join(validation_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=4)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
