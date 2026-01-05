#!/usr/bin/env python3
"""
Benchmark script for iDKT Paper.
Launches training and evaluation for models in PyKT Table 8 across 4 'S' datasets.
Supports 5-fold cross-validation, resource optimization, and background queueing.

This script acts as a scheduler for 'run_repro_experiment.py', ensuring full 
compliance with the project's reproducibility standards.

Models (Liu et al. 2023, Table 8): 
dkt, dkt+, dkt_forget, kqn, dkvmn, atkt, gkt, sakt, saint, akt + idkt

Datasets:
assist2009_S, assist2015_S, bridge2algebra2006_S, nips_task34_S

Usage:
    # ALWAYS run inside the Docker container (pinn-dev)
    
    # 1. Start training in the background
    nohup python3 examples/run_benchmarks_paper.py --mode training > experiments/benchmark_paper_queue.log 2>&1 &

    # 2. Monitor overall status
    tail -f experiments/benchmark_paper_queue.log

    # 3. Launch evaluation (after training finishes)
    python3 examples/run_benchmarks_paper.py --mode evaluation

    # 4. Process results
    python3 examples/run_benchmarks_paper.py --mode results

Expected Outputs & Metrics Location:
- Reproduced Experiment Folder (Grouped):
    Location: experiments/YYYYMMDD_HHMMSS_[model]_[dataset]_benchpaper/
    - fold_[X]_[ID]/: Individual fold directory.
        - config.json: Reproducibility audit, parameters, and explicit commands.
        - results.json: Summary of best epoch performance (AUC, ACC).
        - [params_str]/: (Baselines ONLY) Internal PyKT structure containing model checkpoints.
        - model_best.pth: (iDKT ONLY) Saved model weights.
        - metrics_epoch.csv: (iDKT ONLY) Full training history.
        - execution_history.log: Verbatim stdout of the process.

- NOTE on 'execution_history.log':
    This file is written DIRECTLY to the experiment folder from the start of training. 
    It captures the reproducibility audit, the training progress, and stdout/stderr. 
    For baselines, this is the only source for epoch-by-epoch AUC results.

- NOTE on 'saved_model/':
    By default, PyKT scripts would save to 'examples/saved_model'. This script overrides 
    this by passing the experiment folder as '--save_dir'.
"""

import os
import sys
import json
import argparse
import subprocess
import time
import glob
import shutil
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Models from PyKT paper (Liu et al., 2023)
BENCHMARK_MODELS = [
    "dkt", "dkt+", "dkt_forget", "kqn", "dkvmn", "atkt", "gkt", "sakt", "saint", "akt"
]

#BENCHMARK_MODELS = [
#    "akt"
#]

# 4 'S' datasets (truncated to sequence length 200)
BENCHMARK_DATASETS = [
    "assist2009_S", "assist2015_S", "bridge2algebra2006_S", "nips_task34_S"
]

#BENCHMARK_DATASETS = [
#    "assist2009_S"
#]


# Mapping models to training scripts
MODEL_SCRIPTS = {
    "dkt": "wandb_dkt_train.py",
    "dkt+": "wandb_dkt_plus_train.py",
    "dkt_forget": "wandb_dkt_forget_train.py",
    "kqn": "wandb_kqn_train.py",
    "dkvmn": "wandb_dkvmn_train.py",
    "atkt": "wandb_atkt_train.py",
    "gkt": "wandb_gkt_train.py",
    "sakt": "wandb_sakt_train.py",
    "saint": "wandb_saint_train.py",
    "akt": "wandb_akt_train.py",
    "idkt": "train_idkt.py"
}

# Environment Config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PYTHON = sys.executable

def run_cmd(cmd, log_path, env=None, cwd=None):
    """Run a process and direct output to log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        # start_new_session=True makes it immune to terminal signals
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        return process.wait()

def train_worker(model, dataset, fold, gpu_id, start_delay=0, parent_folder=None, dry_run=False):
    """
    Worker that uses run_repro_experiment.py to launch a training task.
    Logs are written directly into the experiment folder.
    """
    if start_delay > 0:
        time.sleep(start_delay)

    # Generate the 6-digit ID here so we know the path beforehand
    exp_id = str(random.randint(100000, 999999))
    
    # Construct paths
    if parent_folder:
        exp_dir = Path(parent_folder) / f"fold_{fold}_{exp_id}"
    else:
        # Fallback to legacy structure if no parent (unlikely now)
        exp_dir = Path(PROJECT_ROOT) / "experiments" / f"auto_{model}_{dataset}_fold_{fold}_{exp_id}"

    scheduler_log = exp_dir / "execution_history.log"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # CPU Throttling: 80% of 40 cores / 6 workers = ~5 threads each
    env["OMP_NUM_THREADS"] = "5"
    env["MKL_NUM_THREADS"] = "5"
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}"

    script = MODEL_SCRIPTS.get(model)
    if not script:
        return f"Error: No script for {model}"

    # Command construction using run_repro_experiment.py
    cmd = [
        VENV_PYTHON,
        "examples/run_repro_experiment.py",
        "--model_name", model,
        "--train_script", f"examples/{script}",
        "--dataset", dataset,
        "--fold", str(fold),
        "--short_title", "benchpaper",
        "--force_id", exp_id,
        "--num_gpus", "1" # Already limited by CUDA_VISIBLE_DEVICES
    ]
    if dry_run:
        cmd.append("--dry_run")

    if parent_folder:
        cmd.extend(["--parent_folder", parent_folder])
    
    # run_repro_experiment MUST be run from PROJECT_ROOT for path consistency
    cwd = Path(PROJECT_ROOT)
    exit_code = run_cmd(cmd, str(scheduler_log), env=env, cwd=cwd)

    return (model, dataset, fold, exit_code)

def find_experiment_folder(model, dataset, fold):
    """Locate the experiment folder created by run_repro_experiment.py (recursive)"""
    # Pattern: Search PROJECT_ROOT/experiments for folders containing fold_X
    # We use rglob to find nested folders
    base_path = Path(PROJECT_ROOT) / "experiments"
    matches = list(base_path.rglob(f"fold_{fold}_*"))
    
    # Also check legacy flat structure for backward compatibility
    legacy_matches = list(base_path.glob(f"*_{model}_benchpaper_*"))
    matches.extend(legacy_matches)
    
    # Filter by dataset and fold inside config.json
    valid_matches = []
    for m in matches:
        config_path = Path(m) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
                # Resolve actual dataset/fold/model from config
                # Priority: input (user overrides) > defaults (config fallback)
                cfg_in = cfg.get("input", {})
                cfg_def = cfg.get("defaults", {})
                
                c_data = cfg_in.get("dataset", cfg_def.get("dataset"))
                c_fold = cfg_in.get("fold", cfg_def.get("fold"))
                c_model = cfg_in.get("model", cfg_def.get("model"))

                if c_data == dataset and c_fold == fold and c_model == model:
                    valid_matches.append((os.path.getmtime(m), m))
    
    if not valid_matches:
        return None
    
    # Return the latest one
    return Path(sorted(valid_matches)[-1][1])

def evaluate_worker(model, dataset, fold, gpu_id):
    """
    Worker that finds the experiment folder and runs its eval_explicit command.
    """
    exp_dir = find_experiment_folder(model, dataset, fold)
    if not exp_dir:
        return (model, dataset, fold, -1, "Experiment folder not found")

    config_path = exp_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
        eval_cmd = config.get("commands", {}).get("eval_explicit")
    
    if not eval_cmd:
        return (model, dataset, fold, -1, "Eval command missing in config.json")

    # Hardcode evaluation mode for Scientific Alignment (KC-level, no fusion)
    # We modify the command to override fusion if it was set to defaults
    if "wandb_predict" in eval_cmd:
        # Patch config.json with strictly sanitized model_config to avoid TypeError in wandb_predict.py
        # This fixes issues where wandb_predict.py (legacy) naively unpacks params into __init__
        cfg_model_config = config.get("model_config", {})
        
        # Reconstruct if empty (legacy behavior emulation but strict)
        if not cfg_model_config:
             cfg_in = config.get("input", {})
             cfg_def = config.get("defaults", {})
             # Merge input over defaults
             raw_params = {**cfg_def, **cfg_in}
        else:
             raw_params = cfg_model_config

        # Allowlist for model parameters (Strict Mapping)
        # Default includes common transformer params
        default_keys = [
            'd_model', 'n_blocks', 'dropout', 'n_heads', 'd_ff', 'num_attn_heads', 
            'n_layers', 'hidden_size', 'n_hidden', 'emb_size', 'dim_s', 'size_list',
            'l2', 'lambda_student', 'lambda_gap', 'final_fc_dim', 
            'num_encoder_blocks', 'n_know', 'd_k', 'd_v', 'd_m', 'n_question', 'n_pid',
            'gamma', 'alpha', 'beta', 'epsilon', 'lambda_lib', 'lambda_trans',
            'window_size', 'v_size', 's_size', 'a_size', 'seq_len'
        ]
        
        specific_keys = {
           'dkt': ['emb_size', 'dropout'],
           'dkt+': ['emb_size', 'dropout', 'lambda_r', 'lambda_w1', 'lambda_w2'], 
           'dkvmn': ['dim_s', 'size_list', 'dropout'],
           'kqn': ['n_hidden', 'n_rnn_hidden', 'n_mlp_hidden', 'dropout'],
           'sakt': ['n_blocks', 'dropout', 'num_attn_heads', 'seq_len', 'img_size'],
           'saint': ['d_model', 'n_blocks', 'dropout', 'n_heads', 'seq_len'],
           'akt': ['d_model', 'n_blocks', 'dropout', 'num_attn_heads', 'd_ff', 'kq_same', 'final_fc_dim', 'separate_qa', 'l2', 'd_k', 'd_v', 'd_m'],
           'idkt': ['d_model', 'n_blocks', 'dropout', 'n_heads', 'd_ff', 'seq_len', 'final_fc_dim', 'l2', 'lambda_student', 'lambda_gap'],
           'atkt': ['emb_size', 'dropout', 'beta', 'epsilon']
        }
        
        allowed_keys = specific_keys.get(model, default_keys)
        
        # Handle aliases
        if model in ['dkt', 'dkt+', 'atkt'] and 'd_model' in raw_params and 'emb_size' not in raw_params:
            raw_params['emb_size'] = raw_params['d_model']

        sanitized_config = {k: v for k, v in raw_params.items() if k in allowed_keys}
        
        # Write sanitized config back to config.json for wandb_predict.py to consume
        config["model_config"] = sanitized_config
        
        # Legacy compatibility: wandb_predict.py might look for "params" or "train_config"
        # We populate "params" with the same valid info to be safe
        if "fold" not in raw_params:
             raw_params["fold"] = fold
        if "model_name" not in raw_params:
             raw_params["model_name"] = model
        if "dataset_name" not in raw_params:
             raw_params["dataset_name"] = dataset
        if "emb_type" not in raw_params:
             raw_params["emb_type"] = "qid" # Default to qid
        config["params"] = raw_params
        # And ensure train_config is present and has necessary fields if missing or minimal
        if "train_config" not in config:
             config["train_config"] = raw_params
        else:
             # Merge sanitized params into train_config just in case
             config["train_config"].update(sanitized_config)

        # Update train_config seq_len as wandb_predict explicitely looks for it for SAINA/SAKT
        if "train_config" in config and "seq_len" in raw_params:
             config["train_config"]["seq_len"] = raw_params["seq_len"]

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
        print(f"[Sanitizer] Updated {config_path} with sanitized model_config for {model}")

        if "cd examples &&" in eval_cmd:
             eval_cmd = eval_cmd.replace("cd examples &&", "")
        
        # Checkpoint Discovery Logic (External Fix for nested paths)
        # Search for .ckpt files in exp_dir
        found_ckpt_dir = None
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    found_ckpt_dir = Path(root)
                    print(f"[Sanitizer] Found checkpoint at {found_ckpt_dir}")
                    break
            if found_ckpt_dir:
                break
        
        # If found, update commands to point to the nested dir so wandb_predict finds it directly
        effective_save_dir = found_ckpt_dir if found_ckpt_dir else exp_dir
        
        # Ensure correct path is used (points to where config.json is)
        import re
        eval_cmd = re.sub(r'--save_dir\s+\S+', f'--save_dir {effective_save_dir}', eval_cmd)
        
        # Also copy the sanitized config.json to the nested dir if it's different so wandb_predict finds it there
        if found_ckpt_dir and found_ckpt_dir != exp_dir:
             import shutil
             nested_config_path = found_ckpt_dir / "config.json"
             # Copy the sanitized config we just wrote
             shutil.copy2(config_path, nested_config_path)
             print(f"[Sanitizer] Copied sanitized config to {nested_config_path}")

        # Add required flags
        eval_cmd = eval_cmd.replace("--fusion_type 'early_fusion,late_fusion'", "--fusion_type none")
        if "--fusion_type" not in eval_cmd:
            eval_cmd += " --fusion_type none"
        # eval_cmd += " --use_all_in_one False" # wandb_predict.py does not support this

    log_path = exp_dir / "eval_benchmark.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}"
    env["EXPERIMENT_DIR"] = str(exp_dir.resolve())

    print(f"[EVAL] {model} on {dataset} fold {fold}")
    # Run from examples/ directory for relative path compatibility
    cwd = Path(PROJECT_ROOT) / "examples"
    
    # Use shell=True for complex commands with env vars
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        process = subprocess.Popen(eval_cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, shell=True, start_new_session=True)
        exit_code = process.wait()
        
    return (model, dataset, fold, exit_code)

def main():
    parser = argparse.ArgumentParser(description="Reproducible multi-model benchmark scheduler.")
    parser.add_argument("--mode", choices=["training", "evaluation", "results"], required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5", help="6 GPUs to use")
    parser.add_argument("--dry_run", action="store_true", help="If set, only print commands without executing them.")
    args = parser.parse_args()

    gpus = args.gpus.split(",")
    max_workers = len(gpus)

    if args.mode == "training":
        # Global timestamp for this benchmark session
        benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"--- BENCHMARK TRAINING QUEUE (Concurrency={max_workers}) ---")
        print(f"Engine: run_repro_experiment.py (Scientific Alignment Mode, Grouped Folds)")
        
        
        # Create campaign folder
        campaign_folder = f"{PROJECT_ROOT}/experiments/{benchmark_timestamp}_benchpaper"
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            idx = 0
            for model in BENCHMARK_MODELS:
                for dataset in BENCHMARK_DATASETS:
                    # Nested directory: [campaign]/[model]/[dataset]
                    # This ensures all experiments are in one timestamped root
                    group_folder = f"{campaign_folder}/{model}/{dataset}"
                    
                    for fold in range(5):
                        gpu_id = gpus[idx % max_workers]
                        delay = min((idx % max_workers) * 20, 120) # Stagger starts (audit can be heavy)
                        print(f"[QUEUE] {model} on {dataset} fold {fold} (GPU {gpu_id})")
                        futures.append(executor.submit(train_worker, model, dataset, fold, gpu_id, delay, group_folder, args.dry_run))
                        idx += 1
            
            for future in as_completed(futures):
                print(f"[COMPLETE] {future.result()}")

    elif args.mode == "evaluation":
        print(f"--- BENCHMARK EVALUATION QUEUE (Concurrency={max_workers}) ---")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            idx = 0
            for dataset in BENCHMARK_DATASETS:
                for model in BENCHMARK_MODELS:
                    for fold in range(5):
                        gpu_id = gpus[idx % max_workers]
                        futures.append(executor.submit(evaluate_worker, model, dataset, fold, gpu_id))
                        idx += 1
            for future in as_completed(futures):
                print(f"[COMPLETE EVAL] {future.result()}")

    elif args.mode == "results":
        import numpy as np
        print(f"{'Dataset':<22} | {'Model':<12} | {'AUC':<15} | {'ACC':<15} | {'Fold Details'}")
        print("-" * 100)
        for dataset in BENCHMARK_DATASETS:
            for model in BENCHMARK_MODELS:
                metrics = {"auc": [], "acc": []}
                fold_details = []
                for fold in range(5):
                    exp_dir = find_experiment_folder(model, dataset, fold)
                    auc, acc = None, None
                    status = "MISSING"

                    if exp_dir:
                        # Try to find the log file or result file. wandb_predict outputs to stdout which we redirected to eval_benchmark.log
                        # But it also produces keys in stdout like "{'testauc': 0.8...}"
                        log_file = exp_dir / "eval_benchmark.log"
                        
                        if log_file.exists():
                            with open(log_file) as f:
                                content = f.read()
                                # Look for dictionary-like output first: {'testauc': 0.817...}
                                import re
                                match = re.search(r"\{'testauc':\s*([\d\.]+),\s*'testacc':\s*([\d\.]+)", content)
                                if match:
                                    auc = float(match.group(1))
                                    acc = float(match.group(2))
                                    status = "OK"
                                else:
                                    # Fallback to plain text "testauc: 0.8..., testacc: 0.7..."
                                    match_auc = re.search(r"testauc:\s*([\d\.]+)", content)
                                    match_acc = re.search(r"testacc:\s*([\d\.]+)", content)
                                    if match_auc and match_acc:
                                        auc = float(match_auc.group(1))
                                        acc = float(match_acc.group(1)) 
                                        status = "OK"
                        
                        # Also check if it wrote to a file like *_test_predictions.txt (unlikely to have metrics summary)
                        # Or if we have eval_results.json (standard)
                        if auc is None:
                             res_files = list(exp_dir.glob("**/eval_results.json")) # Recursive
                             if res_files:
                                with open(res_files[0]) as f:
                                    d = json.load(f)
                                    auc = d.get("test_auc")
                                    acc = d.get("test_acc")
                                    status = "OK"

                        if auc is not None:
                            metrics["auc"].append(auc)
                            metrics["acc"].append(acc)
                            fold_details.append(f"F{fold}:{auc:.4f}")
                        else:
                            fold_details.append(f"F{fold}:Err")
                    else:
                        fold_details.append(f"F{fold}:N/A")
                
                if metrics["auc"]:
                    m_auc, s_auc = np.mean(metrics["auc"]), np.std(metrics["auc"])
                    m_acc, s_acc = np.mean(metrics["acc"]), np.std(metrics["acc"])
                    auc_str = f"{m_auc:.4f}±{s_auc:.4f}"
                    acc_str = f"{m_acc:.4f}±{s_acc:.4f}"
                    details_str = " ".join(fold_details)
                    print(f"{dataset:<22} | {model:<12} | {auc_str:<15} | {acc_str:<15} | {details_str}")

if __name__ == "__main__":
    main()
