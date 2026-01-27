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
assist2009, assist2015, bridge2algebra2006, nips_task34

Usage:
    # ALWAYS run inside the Docker container (pinn-dev)
    
    # 1. Start training in the background
    nohup python3 examples/run_benchmarks_paper.py --mode training > experiments/benchmark_paper_queue.log 2>&1 &

    # 2. Monitor overall status
    tail -f experiments/benchmark_paper_queue.log

    # 3. Launch evaluation (after training finishes)
    python3 examples/run_benchmarks_paper.py --mode evaluation --dataset assist2015

    # 4. Process results & generate cv_results.json
    python3 examples/run_benchmarks_paper.py --mode results --dataset assist2015

Expected Outputs & Metrics Location:
- Reproduced Experiment Folder (Grouped):
    Location: experiments/YYYYMMDD_HHMMSS_benchpaper_[UNIQUE_ID]/
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
import uuid
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BENCHMARK_MODELS = [
    "gtransformer", "akt", "dkt", "sakt", "saint", "dkvmn", "atkt" 
]


# 4 'S' datasets (truncated to sequence length 200)
BENCHMARK_DATASETS = [
    "assist2009", "assist2015", "bridge2algebra2006", "nips_task34", "algebra2005"
]

#BENCHMARK_DATASETS = [
#    "assist2009"
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
    "gtransformer": "wandb_gtransformer_train.py"
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

def train_worker(model, dataset, fold, gpu_id, start_delay=0, parent_folder=None, dry_run=False, epochs=None, param_overrides=None):
    """
    Worker that uses run_repro_experiment.py to launch a training task.
    Logs are written directly into the experiment folder.
    
    Args:
        param_overrides: Dict of parameter overrides to pass to run_repro_experiment.py
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
    if epochs:
        cmd.extend(["--epochs", str(epochs)])
    if dry_run:
        cmd.append("--dry_run")

    if parent_folder:
        cmd.extend(["--parent_folder", parent_folder])
    
    # Add parameter overrides (zero hardcoded defaults philosophy)
    if param_overrides:
        for param_name, param_value in param_overrides.items():
            if param_value is not None:
                cmd.extend([f"--{param_name}", str(param_value)])
    
    # run_repro_experiment MUST be run from PROJECT_ROOT for path consistency
    cwd = Path(PROJECT_ROOT)
    exit_code = run_cmd(cmd, str(scheduler_log), env=env, cwd=cwd)

    return (model, dataset, fold, exit_code)

def find_experiment_folder(model, dataset, fold, campaign_pattern=None):
    """
    Locate the experiment folder created by run_repro_experiment.py.
    Supports both nested and flat campaign structures:
    - Nested: experiments/*_benchpaper/[model]/[dataset]/fold_[fold]_*
    - Flat: experiments/*_benchpaper/fold_[fold]_* (validates model/dataset from config.json)
    
    Args:
        campaign_pattern: Optional glob pattern to filter campaigns (e.g., "*probing_benchpaper")
    """
    base_path = Path(PROJECT_ROOT) / "experiments"
    
    # Strategy 1: Search in timestamped campaign folders (modern structure)
    # We assume folders are named like "YYYYMMDD_HHMMSS_benchpaper"
    if campaign_pattern:
        if "*" not in campaign_pattern:
            campaign_pattern = f"*{campaign_pattern}*"
        campaigns = sorted(list(base_path.glob(campaign_pattern)))
    else:
        # User wants most recent timestamped folder only
        # Filter for timestamped folders (YYYYMMDD_HHMMSS format) and use the newest
        all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        timestamped_dirs = []
        for d in all_dirs:
            name = d.name
            # Check if starts with timestamp pattern (YYYYMMDD_HHMMSS)
            if len(name) >= 15 and name[:8].isdigit() and name[8] == '_' and name[9:15].isdigit():
                timestamped_dirs.append(d)
        
        if timestamped_dirs:
            # Sort by timestamp (folder name) and use only the most recent one
            campaigns = [sorted(timestamped_dirs, key=lambda p: p.name)[-1]]
        else:
            # Fallback: no timestamped folders, use all directories
            campaigns = sorted(all_dirs, key=lambda p: p.name)
    
    potential_folders = []
    
    # Search campaigns in reverse chronological order (newest first)
    for campaign in reversed(campaigns):
        # Explicit check: ensure it's a directory
        if not campaign.is_dir(): continue
        
        # Try nested structure first: campaign/model/dataset/fold_X
        target_path = campaign / model / dataset
        if target_path.exists():
            # Look for the specific fold folder
            fold_matches = list(target_path.glob(f"fold_{fold}_*"))
            potential_folders.extend(fold_matches)
        
        # Try flat structure: campaign/fold_X (common for single-model campaigns)
        else:
            flat_fold_matches = list(campaign.glob(f"fold_{fold}_*"))
            # Also check for exact fold_X pattern without ID suffix
            if not flat_fold_matches:
                flat_fold_matches = list(campaign.glob(f"fold_{fold}"))
            potential_folders.extend(flat_fold_matches)

    # Strategy 2: Fallback / Legacy (Legacy flat structure or manually created)
    # We limit this to direct children of experiments/ to avoid full rglob
    if not potential_folders and not campaign_pattern:
         # Try finding direct folders (old behavior) that match the precise pattern
         potential_folders = list(base_path.glob(f"auto_{model}_{dataset}_fold_{fold}_*"))

    if not potential_folders:
        return None

    # Filter by valid config.json (Validation)
    valid_matches = []
    for m in potential_folders:
        # Skip backup directories
        if "_backup" in str(m):
            continue
            
        config_path = Path(m) / "config.json"
        if not config_path.exists():
            continue
            
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except json.JSONDecodeError:
            continue
        
        # Resolve actual dataset/fold/model from config to be 100% sure
        cfg_in = cfg.get("input", {})
        cfg_def = cfg.get("defaults", {})
        cfg_train = cfg.get("train_config", {})
        
        c_data = cfg_in.get("dataset", cfg_train.get("dataset", cfg_def.get("dataset")))
        c_fold = cfg_in.get("fold", cfg_train.get("fold", cfg_def.get("fold")))
        c_model = cfg_in.get("model", cfg_train.get("model", cfg_def.get("model")))

        # Loose string comparison to avoid type mismatches
        if str(c_data) == str(dataset) and str(c_fold) == str(fold) and str(c_model) == str(model):
            valid_matches.append((os.path.getmtime(m), m))
    
    if not valid_matches:
        return None
    
    # Sort by modification time (newest first)
    valid_matches.sort(key=lambda x: x[0], reverse=True)
    return valid_matches[0][1]

def evaluate_worker(model, dataset, fold, gpu_id, campaign_pattern=None, dual_eval=False, exp_dir_override=None):
    """
    Worker that finds the experiment folder and runs its evaluate_explicit command.
    
    All evaluation parameters (including dual_eval) are read from the eval_explicit 
    command stored in config.json - single source of truth from training time.
    
    Args:
        campaign_pattern: Optional pattern to filter which campaign to evaluate (e.g., "*probing_benchpaper")
        dual_eval: Deprecated - parameter is read from eval_explicit in config.json
        exp_dir_override: If provided, use this directory instead of searching
    """
    if exp_dir_override:
        exp_dir = Path(exp_dir_override)
        if not exp_dir.exists():
            return (model, dataset, fold, -1, "Experiment folder not found")
    else:
        exp_dir = find_experiment_folder(model, dataset, fold, campaign_pattern)
        if not exp_dir:
            return (model, dataset, fold, -1, "Experiment folder not found")

    config_path = exp_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
        # Try evaluate_explicit first (new standard), fall back to eval_explicit for backward compatibility
        eval_cmd = config.get("commands", {}).get("evaluate_explicit") or config.get("commands", {}).get("eval_explicit")
    
    if not eval_cmd:
        return (model, dataset, fold, -1, "Eval command missing in config.json")
    
    # Verify config.json has proper structure for reproducibility
    if "_documentation" not in config:
        print(f"  ⚠️  Warning: {exp_dir.name} has legacy config format (missing _documentation section)")
    
    # Log the explicit commands for audit trail
    train_cmd = config.get("commands", {}).get("train_explicit", "N/A")
    if exp_dir.name not in getattr(evaluate_worker, '_logged_configs', set()):
        # Try to identify campaign root (Strategy: exp_dir / .. / .. if it exists)
        campaign_root = "Individual Folder"
        # Check if we are in a nested structure: campaign/model/dataset/fold_X
        if exp_dir.parent.parent.parent.parent == Path(PROJECT_ROOT) / "experiments":
             campaign_root = exp_dir.parent.parent.parent.name
        # Check if we are in a flat structure: campaign/fold_X
        elif exp_dir.parent.parent == Path(PROJECT_ROOT) / "experiments":
             campaign_root = exp_dir.parent.name
             
        print(f"\n{'='*80}")
        print(f"CAMPAIGN: {campaign_root}")
        print(f"EXPERIMENT: {exp_dir.name}")
        print(f"{'='*80}")
        print(f"Training command (used to train this model):")
        print(f"  {train_cmd}")
        print(f"\nEvaluation command (will be executed now):")
        print(f"  {eval_cmd}")
        print(f"\nConfiguration loaded from: {config_path}")
        print(f"{'='*80}\n")
        if not hasattr(evaluate_worker, '_logged_configs'):
            evaluate_worker._logged_configs = set()
        evaluate_worker._logged_configs.add(exp_dir.name)

    # Force use of current sys.executable to ensure correct environment (fix ModuleNotFoundError)
    # Also fix script paths to use relative paths from PROJECT_ROOT
    if "python" in eval_cmd:
        # Re-order to replace longer paths first and avoid double-replacement (e.g., /usr/bin/python3 -> /usr/bin/python33)
        for old_path in ["/home/vscode/.pykt-env/bin/python3", "/usr/bin/python3", "/usr/bin/python"]:
            if old_path in eval_cmd:
                eval_cmd = eval_cmd.replace(old_path, sys.executable)
                break # Only replace the first match to avoid corruption
        
        eval_cmd = eval_cmd.replace("python3 examples/wandb_predict.py", f"{sys.executable} examples/wandb_predict.py")
        eval_cmd = eval_cmd.replace("python3 examples/wandb_gtransformer_predict.py", f"{sys.executable} examples/wandb_gtransformer_predict.py")
        
        # Fix model-specific predict scripts for gtransformer
        if model == "gtransformer" and "wandb_predict.py" in eval_cmd:
            eval_cmd = eval_cmd.replace("wandb_predict.py", "wandb_gtransformer_predict.py")
        
        # Fix absolute script paths to relative (handle different workspace locations)
        import re
        eval_cmd = re.sub(r'/[^ ]+/examples/(wandb_\w+_predict\.py)', r'examples/\1', eval_cmd)
        # Since we run from examples/ directory, remove the examples/ prefix from script path
        eval_cmd = eval_cmd.replace("examples/wandb", "wandb")
    
    # Hardcode evaluation mode for Scientific Alignment (KC-level, no fusion)
    # We modify the command to override fusion if it was set to defaults
    if "predict.py" in eval_cmd:
        # Strict Mapping for model parameters to avoid TypeError in legacy __init__
        # These must match pykt/models/[model].py __init__ signatures
        specific_keys = {
           'dkt': ['emb_size', 'dropout'],
           'dkt+': ['emb_size', 'dropout', 'lambda_r', 'lambda_w1', 'lambda_w2'], 
           'dkvmn': ['dim_s', 'size_m', 'dropout'],
           'kqn': ['n_hidden', 'n_rnn_hidden', 'n_mlp_hidden', 'dropout'],
           'sakt': ['seq_len', 'emb_size', 'num_attn_heads', 'dropout', 'num_en'], 
           'saint': ['seq_len', 'emb_size', 'num_attn_heads', 'dropout', 'n_blocks'],
           'akt': ['num_attn_heads', 'd_model', 'n_blocks', 'dropout', 'd_ff', 'kq_same', 'final_fc_dim', 'separate_qa', 'l2', 'd_k', 'd_v', 'd_m'],
           'idkt': ['n_blocks', 'dropout', 'n_heads', 'd_ff', 'seq_len', 'final_fc_dim', 'l2', 'lambda_student', 'lambda_gap'],
           'atkt': ['skill_dim', 'answer_dim', 'hidden_dim', 'attention_dim', 'epsilon', 'beta', 'dropout'],
           'gkt': ['hidden_dim', 'emb_size', 'graph_type', 'dropout'],
           'dkt_forget': ['emb_size', 'dropout'],
           'gtransformer': ['d_model', 'n_blocks', 'dropout', 'd_ff', 'kq_same', 'final_fc_dim', 'num_attn_heads', 'separate_qa', 'l2', 'l2_rasch', 'pretrain_dim', 'ablation', 'n_uid', 'lambda_probe', 'active_grounding', 'lambda_ref', 'lambda_initmastery', 'lambda_rate']
         }
        
        # Get raw parameters by merging resolved train_config and input
        cfg_in = config.get("input", {})
        # Prioritize train_config as it contains resolved overrides used during training
        cfg_def = config.get("train_config", config.get("defaults", {}))
        raw_params = {**cfg_def, **cfg_in}
        
        allowed_keys = specific_keys.get(model, [])
        
        # CRITICAL FIX: Parse parameters from train_explicit command string if available
        # This acts as the final source of truth for what was actually run.
        train_cmd = config.get("commands", {}).get("train_explicit", "")
        if train_cmd:
            import re
            for key in allowed_keys:
                # Look for --key VALUE
                match = re.search(rf'--{key}\s+([^\s]+)', train_cmd)
                if match:
                    val = match.group(1)
                    # Try to convert to int or float if possible
                    try:
                        if '.' in val or 'e' in val.lower():
                            val = float(val)
                        else:
                            val = int(val)
                    except:
                        pass
                    raw_params[key] = val
                    # print(f"[Sanitizer] Overrode {key}={val} from train_explicit for {model}")

        sanitized_config = {k: v for k, v in raw_params.items() if k in allowed_keys}
        
        # Ensure proper type conversion for all numeric parameters
        for key in sanitized_config:
            val = sanitized_config[key]
            if isinstance(val, str):
                try:
                    # Try to convert string representations to numeric types
                    if '.' in val or 'e' in val.lower():
                        sanitized_config[key] = float(val)
                    else:
                        sanitized_config[key] = int(val)
                except (ValueError, AttributeError):
                    # Keep as string if conversion fails
                    pass
        
        # Perform alias mapping BEFORE sanitization
        if 'd_model' in raw_params:
            if 'emb_size' not in raw_params: raw_params['emb_size'] = raw_params['d_model']
            if 'dim_s' not in raw_params: raw_params['dim_s'] = raw_params['d_model']
            if 'skill_dim' not in raw_params: raw_params['skill_dim'] = raw_params['d_model']
        if 'd_ff' in raw_params:
            if model == 'atkt':
                raw_params['attention_dim'] = raw_params['d_ff']
        if 'n_blocks' in raw_params:
            if 'num_en' not in raw_params: raw_params['num_en'] = raw_params['n_blocks']
            if 'n_layers' not in raw_params: raw_params['n_layers'] = raw_params['n_blocks']
        if 'n_hidden' in raw_params:
            if 'hidden_dim' not in raw_params: raw_params['hidden_dim'] = raw_params['n_hidden']
        
        # Explicit mapping for ATKT if d_model was used for hidden_dim (to match training bug)
        if model == 'atkt' and 'd_model' in raw_params:
            # Training script had a bug where atkt:hidden_dim was in PARAM_MAP for d_model
            # and it might have overridden the default hidden_dim=64 if d_model was processed first.
            # but in sorted(params.items()), d_model (D) comes before hidden_dim (H).
            # So hidden_dim (64) actually won in training. 
            # Checkpoint [512, 64] confirms hidden_dim=64 and attention_dim=512.
            pass
        
        # Determine data_config (reconstruct if missing or incomplete)
        cfg_data_config = config.get("data_config", {})
        if not cfg_data_config or "num_c" not in cfg_data_config:
            # Try to load from central data_config.json
            try:
                with open(os.path.join(PROJECT_ROOT, "configs/data_config.json")) as f:
                    full_data_config = json.load(f)
                    # Handle S-protocol mapping if needed
                    ds_key = dataset
                    if ds_key not in full_data_config and dataset.endswith("_S"):
                        ds_key = dataset[:-2]
                    if ds_key in full_data_config:
                        cfg_data_config = full_data_config[ds_key]
                        print(f"[Sanitizer] Reconstructed data_config for {dataset} from configs/data_config.json")
            except Exception as e:
                print(f"[Sanitizer] Warning: Could not reconstruct data_config: {e}")

        # Update the config object to ensure these top-level keys exist for wandb_predict.py
        config["model_config"] = sanitized_config
        config["data_config"] = cfg_data_config
        config["params"] = raw_params
        config["train_config"] = {**raw_params, **sanitized_config}
        
        # Legacy compatibility updates
        if "fold" not in config["params"]: config["params"]["fold"] = fold
        if "model_name" not in config["params"]: config["params"]["model_name"] = model
        if "dataset_name" not in config["params"]: config["params"]["dataset_name"] = dataset
        if "emb_type" not in config["params"]: config["params"]["emb_type"] = raw_params.get("emb_type", "qid")
        
        config["train_config"].update(sanitized_config)

        # Persistence: Write modified config back to disk for wandb_predict to find
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"[Sanitizer] Updated {config_path} with sanitized configuration")

        # CRITICAL FIX: Ensure model_config is ALWAYS present for legacy wandb_predict scripts
        # (This block seems redundant now but kept for safety if used elsewhere)
        if "model_config" not in config:
            config["model_config"] = sanitized_config



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
        
        # Ensure correct path is used (relative to PROJECT_ROOT since we run from examples/)
        # Convert to relative path from examples/ directory
        import re
        try:
            rel_save_dir = os.path.relpath(effective_save_dir, Path(PROJECT_ROOT) / "examples")
            eval_cmd = re.sub(r'--save_dir\s+\S+', f'--save_dir {rel_save_dir}', eval_cmd)
        except ValueError as e:
            # Fallback to absolute path if relative fails
            eval_cmd = re.sub(r'--save_dir\s+\S+', f'--save_dir {effective_save_dir.resolve()}', eval_cmd)
        
        # Also copy the sanitized config.json to the nested dir if it's different so wandb_predict finds it there
        if found_ckpt_dir and found_ckpt_dir != exp_dir:
             import shutil
             nested_config_path = found_ckpt_dir / "config.json"
             # Copy the sanitized config we just wrote
             shutil.copy2(config_path, nested_config_path)
             print(f"[Sanitizer] Copied sanitized config to {nested_config_path}")

        # Add required flags
        eval_cmd = eval_cmd.replace("--fusion_type 'early_fusion,late_fusion'", "--fusion_type late_fusion")
        if "--fusion_type" not in eval_cmd:
            eval_cmd += " --fusion_type late_fusion"
        
        # Note: dual_eval parameter is already in eval_explicit command from config.json
        # No need to add it here - the stored command has all parameters
        
        # eval_cmd += " --use_all_in_one False" # wandb_predict.py does not support this

    log_path = exp_dir / "eval_benchmark.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}"
    env["EXPERIMENT_DIR"] = str(exp_dir.resolve())

    print(f"[EVAL] {model} on {dataset} fold {fold}")
    # Run from examples/ directory for relative path compatibility
    cwd = Path(PROJECT_ROOT) / "examples"
    
    # Split command into list for shell=False (safer, prevents process explosion)
    import shlex
    eval_cmd_list = shlex.split(eval_cmd)
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        process = subprocess.Popen(eval_cmd_list, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, shell=False, start_new_session=True)
        exit_code = process.wait()
        
    return (model, dataset, fold, exit_code)

def main():
    parser = argparse.ArgumentParser(
        description="Reproducible multi-model benchmark scheduler.",
        epilog="Any additional arguments (e.g., --n_blocks, --n_heads, --lambda_xxx) are passed through to run_repro_experiment.py"
    )
    parser.add_argument("--mode", choices=["training", "evaluation", "results"], required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5", help="6 GPUs to use")
    parser.add_argument("--dry_run", action="store_true", help="If set, only print commands without executing them.")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset")
    parser.add_argument("--model", type=str, default=None, help="Filter by model")
    parser.add_argument("--fold", type=int, default=None, help="Filter by fold")
    parser.add_argument("--short_title", type=str, default="benchpaper", help="Descriptive label for this benchmark session")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--campaign", type=str, default=None, help="Campaign pattern to filter experiments (e.g., '*probing_benchpaper')")
    parser.add_argument("--experiment_folder", type=str, default=None, help="Specific experiment folder to evaluate (for dual evaluation)")
    parser.add_argument("--dual_eval", action="store_true", help="[DEPRECATED] dual_eval is read from eval_explicit in config.json")
    parser.add_argument("--plots", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, 
                        help="Generate validation plots in results mode (default: true)")
    
    # Use parse_known_args to accept any additional parameters
    args, unknown_args = parser.parse_known_args()

    # Parse unknown arguments (--param_name value format) into param_overrides
    param_overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            param_name = arg[2:]  # Remove '--' prefix
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # Has a value
                value_str = unknown_args[i + 1]
                # Try to infer type
                try:
                    # Try int first
                    param_overrides[param_name] = int(value_str)
                except ValueError:
                    try:
                        # Try float
                        param_overrides[param_name] = float(value_str)
                    except ValueError:
                        # Keep as string
                        param_overrides[param_name] = value_str
                i += 2
            else:
                # Boolean flag (no value)
                param_overrides[param_name] = True
                i += 1
        else:
            i += 1
    
    # Override epochs if specified
    if args.epochs is not None:
        param_overrides["epochs"] = args.epochs

    gpus = args.gpus.split(",")
    max_workers = len(gpus)

    # Determine execution scope
    models_to_run = args.model.split(",") if args.model else BENCHMARK_MODELS
    datasets_to_run = args.dataset.split(",") if args.dataset else BENCHMARK_DATASETS
    folds_to_run = [int(f) for f in str(args.fold).split(",")] if args.fold is not None else range(5)

    if args.mode == "training":
        # Global timestamp for this benchmark session
        benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(random.randint(100000, 999999))  # 6-digit numeric ID
        
        print(f"--- BENCHMARK TRAINING QUEUE (Concurrency={max_workers}) ---")
        print(f"Engine: run_repro_experiment.py (Scientific Alignment Mode, Grouped Folds)")
        
        if param_overrides:
            print(f"Parameter overrides: {param_overrides}")
        
        # Create campaign folder with unique ID and short title
        campaign_folder = f"{PROJECT_ROOT}/experiments/{benchmark_timestamp}_{args.short_title}_{unique_id}"
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            idx = 0
            futures = []
            idx = 0
            for model in models_to_run:
                for dataset in datasets_to_run:
                    # Nested directory: [campaign]/[model]/[dataset]
                    # This ensures all experiments are in one timestamped root
                    group_folder = f"{campaign_folder}/{model}/{dataset}"
                    
                    group_folder = f"{campaign_folder}/{model}/{dataset}"
                    
                    for fold in folds_to_run:
                        gpu_id = gpus[idx % max_workers]
                        delay = min((idx % max_workers) * 20, 120) # Stagger starts (audit can be heavy)
                        print(f"[QUEUE] {model} on {dataset} fold {fold} (GPU {gpu_id})")
                        futures.append(executor.submit(train_worker, model, dataset, fold, gpu_id, delay, group_folder, args.dry_run, args.epochs, param_overrides))
                        idx += 1
            
            for future in as_completed(futures):
                print(f"[COMPLETE] {future.result()}")

    elif args.mode == "evaluation":
        print(f"--- BENCHMARK EVALUATION QUEUE (Concurrency={max_workers}) ---")
        if args.campaign:
            print(f"Campaign filter: {args.campaign}")
        if args.dual_eval:
            print(f"Dual Evaluation Mode: ENABLED (measuring p_sup + p_ref)")
        
        # Handle experiment_folder parameter for targeted evaluation
        if args.experiment_folder:
            print(f"Targeted evaluation mode: {args.experiment_folder}")
            exp_path = Path(args.experiment_folder)
            if not exp_path.exists():
                print(f"ERROR: Experiment folder not found: {exp_path}")
                return
            
            # Parse model/dataset/fold from folder structure or config.json
            config_path = exp_path / "config.json"
            if not config_path.exists():
                print(f"ERROR: config.json not found in {exp_path}")
                return
            
            with open(config_path) as f:
                config = json.load(f)
            
            # Extract metadata from config
            params = config.get("params", {})
            model = params.get("model_name", params.get("model", "gtransformer"))
            dataset = params.get("dataset_name", params.get("dataset", "assist2009"))
            fold = params.get("fold", 0)
            
            print(f"Evaluating: {model} on {dataset} fold {fold}")
            gpu_id = gpus[0]  # Use first GPU for single evaluation
            
            # Run evaluation directly on this folder
            result = evaluate_worker(model, dataset, fold, gpu_id, 
                                    campaign_pattern=None, dual_eval=args.dual_eval,
                                    exp_dir_override=str(exp_path))
            print(f"[COMPLETE] {result}")
            return
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            idx = 0
            for dataset in datasets_to_run:
                for model in models_to_run:
                    for fold in folds_to_run:
                        gpu_id = gpus[idx % max_workers]
                        futures.append(executor.submit(evaluate_worker, model, dataset, fold, gpu_id, 
                                                      args.campaign, args.dual_eval))
                        idx += 1
            for future in as_completed(futures):
                print(f"[COMPLETE EVAL] {future.result()}")

    elif args.mode == "results":
        import numpy as np
        running_dirs = []
        active_schedulers = []
        import re
        try:
            # Use ps -ef for broad compatibility
            ps_output = subprocess.check_output(["ps", "-ef"]).decode()
            for line in ps_output.splitlines():
                if "python" in line and "experiments" in line:
                    # Capture anything that looks like an experiment path
                    parts = line.split()
                    for p in parts:
                        if "experiments/" in p:
                            running_dirs.append(p.rstrip("/"))
                
                # Detect active benchmark schedulers
                if "run_benchmarks_paper.py" in line and "--mode training" in line:
                    m_match = re.search(r'--model ([\w,]+)', line)
                    d_match = re.search(r'--dataset ([\w,]+)', line)
                    s_models = m_match.group(1).split(",") if m_match else BENCHMARK_MODELS
                    s_datasets = d_match.group(1).split(",") if d_match else BENCHMARK_DATASETS
                    active_schedulers.append({"models": s_models, "datasets": s_datasets})
        except Exception: pass

        # 2. Define Sort Order
        DATASET_ORDER = ["assist2009", "assist2015", "algebra2005", "bridge2algebra2006", "nips_task34"]
        
        # 3. Sort datasets and models
        datasets_to_run = sorted(datasets_to_run, key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 999)
        models_to_run = sorted(models_to_run)
        
        # 4. Header with Status column
        print(f"\n--- BENCHMARK RESULTS SUMMARY ---")
        if args.campaign:
            print(f"Filter: Campaign pattern matches '{args.campaign}'")
        else:
            print(f"Mode: Auto-discovery (searching newest directories in experiments/)")
        print(f"{'Model':<12} | {'Dataset':<22} | {'Status':<10} | {'AUC':<15} | {'ACC':<15} | {'Fold Details'}")
        print("-" * 140)
        
        all_results_data = {}
        for model in models_to_run:
            for dataset in datasets_to_run:
                metrics = {"auc": [], "acc": [], "auc_ref": [], "acc_ref": []}
                fold_results = {}
                fold_details = []
                has_ref_metrics = False
                
                missing_count = 0
                err_count = 0
                ok_count = 0
                run_count = 0
                queue_count = 0
                
                # Check if this model/dataset pair is in the scope of any active scheduler
                is_in_scheduler_scope = False
                for sched in active_schedulers:
                    if model in sched["models"] and dataset in sched["datasets"]:
                        is_in_scheduler_scope = True
                        break

                campaign_roots = set()

                for fold in range(5):
                    exp_dir = find_experiment_folder(model, dataset, fold, args.campaign)
                    auc, acc = None, None
                    
                    is_running = False
                    if exp_dir:
                        abs_exp_dir = str(exp_dir.resolve())
                        is_running = any(abs_exp_dir in d for d in running_dirs)
                        
                        # Identify campaign root for reporting
                        if exp_dir.parent.parent.parent.parent == Path(PROJECT_ROOT) / "experiments":
                            campaign_roots.add(exp_dir.parent.parent.parent.name)
                        elif exp_dir.parent.parent == Path(PROJECT_ROOT) / "experiments":
                            campaign_roots.add(exp_dir.parent.name)
                        else:
                            campaign_roots.add("Direct/Legacy")

                    if exp_dir:
                        # 1. Try Structured JSON first (Priority: Question-level Late Fusion)
                        res_files = list(exp_dir.glob("**/eval_results.json"))
                        if res_files:
                            try:
                                with open(res_files[0]) as f:
                                    d = json.load(f)
                                    # Prioritize Question-level Late Fusion metrics as per benchmark protocol
                                    auc = d.get("oriauclate_mean", d.get("testauc", d.get("test_auc")))
                                    acc = d.get("oriacclate_mean", d.get("testacc", d.get("test_acc")))
                                    
                                    # Extract p_ref metrics if available (dual_eval with grounded model)
                                    auc_ref = d.get("oriauclate_mean_ref")
                                    acc_ref = d.get("oriacclate_mean_ref")
                                    
                                    if auc_ref is not None and acc_ref is not None:
                                        has_ref_metrics = True
                                        metrics["auc_ref"].append(auc_ref)
                                        metrics["acc_ref"].append(acc_ref)
                                    
                                    # If metrics found, this fold is complete regardless of ps detection
                                    if auc is not None:
                                        is_running = False
                            except: pass

                        # 2. Fallback to Log File (Regex parsing)
                        if auc is None:
                            log_file = exp_dir / "eval_benchmark.log"
                            if log_file.exists():
                                with open(log_file, "r") as f:
                                    lines = f.readlines()
                                    for line in reversed(lines):
                                        # Generic fallback for older logs
                                        ma = re.search(r"testauc: (0\.\d+)", line)
                                        mac = re.search(r"testacc: (0\.\d+)", line)
                                        if ma and mac:
                                            auc, acc = float(ma.group(1)), float(mac.group(1))
                                            # If metrics found in log, mark as complete
                                            if auc is not None:
                                                is_running = False

                        if auc is not None:
                            metrics["auc"].append(auc)
                            metrics["acc"].append(acc)
                            fold_results[str(fold)] = {"auc": auc, "acc": acc}
                            fold_details.append(f"F{fold}:{auc:.4f}")
                            ok_count += 1
                        elif is_running:
                            fold_details.append(f"F{fold}:Run")
                            run_count += 1
                        elif is_in_scheduler_scope:
                            fold_details.append(f"F{fold}:Wait")
                            queue_count += 1
                        else:
                            fold_details.append(f"F{fold}:Err")
                            err_count += 1
                    else:
                        if is_in_scheduler_scope:
                            fold_details.append(f"F{fold}:Wait")
                            queue_count += 1
                        else:
                            fold_details.append(f"F{fold}:Gap")
                            missing_count += 1

                # Determine Status
                target_count = 5
                if ok_count == target_count:
                    status_val = "PASS"
                elif run_count == target_count or (run_count > 0 and ok_count == 0 and err_count == 0 and queue_count == 0):
                    status_val = "RUNNING"
                elif queue_count == target_count or (queue_count > 0 and ok_count == 0 and err_count == 0 and run_count == 0):
                    status_val = "QUEUED"
                elif missing_count == target_count:
                    status_val = "PENDING"
                elif ok_count > 0:
                    status_val = "PARTIAL"
                elif run_count > 0:
                    status_val = "RE-RUN" # Active retraining
                elif queue_count > 0:
                    status_val = "QUEUED" # Scheduled for retraining
                else:
                    status_val = "FAIL"
                
                if metrics["auc"]:
                    m_auc, s_auc = np.mean(metrics["auc"]), np.std(metrics["auc"])
                    m_acc, s_acc = np.mean(metrics["acc"]), np.std(metrics["acc"])
                    auc_str = f"{m_auc:.4f}±{s_auc:.4f}"
                    acc_str = f"{m_acc:.4f}±{s_acc:.4f}"
                else:
                    auc_str, acc_str = "N/A", "N/A"
                
                # Print result line
                print(f"{model:<12} | {dataset:<22} | {status_val:<10} | {auc_str:<15} | {acc_str:<15} | {' '.join(fold_details)}")
                
                # Show p_ref metrics if available (grounded models with dual_eval)
                if has_ref_metrics and metrics["auc_ref"]:
                    m_auc_ref, s_auc_ref = np.mean(metrics["auc_ref"]), np.std(metrics["auc_ref"])
                    m_acc_ref, s_acc_ref = np.mean(metrics["acc_ref"]), np.std(metrics["acc_ref"])
                    auc_ref_str = f"{m_auc_ref:.4f}±{s_auc_ref:.4f}"
                    acc_ref_str = f"{m_acc_ref:.4f}±{s_acc_ref:.4f}"
                    print(f"  └ p_ref:    | {'':<22} | {'':<10} | {auc_ref_str:<15} | {acc_ref_str:<15} | (BKT logic predictions)")
                
                # Report which campaign folders were used
                if campaign_roots:
                    roots_str = ", ".join(sorted(list(campaign_roots)))
                    print(f"  └ Source: {roots_str}")
                
                if model not in all_results_data:
                    all_results_data[model] = {}
                
                result_dict = {
                    "status": status_val,
                    "auc_mean": float(np.mean(metrics["auc"])) if metrics["auc"] else None,
                    "auc_std": float(np.std(metrics["auc"])) if metrics["auc"] else None,
                    "acc_mean": float(np.mean(metrics["acc"])) if metrics["acc"] else None,
                    "acc_std": float(np.std(metrics["acc"])) if metrics["acc"] else None,
                    "folds_ok": ok_count,
                    "folds_running": run_count,
                    "folds_total": target_count,
                    "fold_data": fold_results
                }
                
                # Add p_ref metrics if available (grounded models)
                if has_ref_metrics and metrics["auc_ref"]:
                    result_dict["auc_ref_mean"] = float(np.mean(metrics["auc_ref"]))
                    result_dict["auc_ref_std"] = float(np.std(metrics["auc_ref"]))
                    result_dict["acc_ref_mean"] = float(np.mean(metrics["acc_ref"]))
                    result_dict["acc_ref_std"] = float(np.std(metrics["acc_ref"]))
                
                all_results_data[model][dataset] = result_dict

        # Determine save location for results (always inside experiment folder)
        if args.campaign:
            # Save to campaign folder if filtering by campaign
            campaign_folders = list(Path(PROJECT_ROOT).glob(f"experiments/*{args.campaign}*"))
            if campaign_folders:
                campaign_folders.sort(key=lambda p: p.name, reverse=True)  # Most recent first
                results_dir = campaign_folders[0]
            else:
                # Fallback if no campaign folder found
                results_dir = Path(PROJECT_ROOT) / "experiments"
        else:
            # When no campaign specified, use the most recent timestamped experiment folder
            base_path = Path(PROJECT_ROOT) / "experiments"
            all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
            timestamped_dirs = []
            for d in all_dirs:
                name = d.name
                # Check if starts with timestamp pattern (YYYYMMDD_HHMMSS)
                if len(name) >= 15 and name[:8].isdigit() and name[8] == '_' and name[9:15].isdigit():
                    timestamped_dirs.append(d)
            
            if timestamped_dirs:
                # Use most recent timestamped experiment folder
                results_dir = sorted(timestamped_dirs, key=lambda p: p.name)[-1]
            else:
                # Fallback to experiments root if no timestamped folders
                results_dir = Path(PROJECT_ROOT) / "experiments"
        
        # Save results to JSON in experiment folder
        json_path = results_dir / "cv_results.json"
        results_metadata = {
            "generated_at": datetime.now().isoformat(),
            "campaign": args.campaign if args.campaign else "all_experiments",
            "protocol": "question-level late fusion (mean)",
            "metrics_note": "AUC/ACC values are question-level averages using late fusion protocol (mean aggregation across skills per question)",
            "p_ref_note": "p_ref metrics available only for grounded models (gtransformer) when dual_eval was used during evaluation",
            "results": all_results_data
        }
        with open(json_path, 'w') as f:
            json.dump(results_metadata, f, indent=4)
        print(f"\n[INFO] Complete results snapshot saved to: {json_path}")
        print(f"[INFO] Protocol: Question-level Late Fusion (mean aggregation)")
        print(f"[INFO] Metrics: oriauclate_mean (supervised), oriauclate_mean_ref (BKT reference)")

        # Post-process: Validation & Diagnostic Plots
        print(f"\n{'='*70}")
        print(f"VALIDATION & VISUALIZATION")
        print(f"{'='*70}")
        
        # Run validation scripts for completed experiments
        for model, datasets in all_results_data.items():
            if model == "gtransformer":
                for dataset, data in datasets.items():
                    if data.get("folds_ok", 0) > 0:
                        # Use Fold 0 for representative plots, fallback to any complete fold
                        fold_dir = find_experiment_folder(model, dataset, 0, args.campaign)
                        if not fold_dir:
                            for f_idx in range(5):
                                fold_dir = find_experiment_folder(model, dataset, f_idx, args.campaign)
                                if fold_dir: break
                        
                        if fold_dir:
                            # Create plots and validation directories at CAMPAIGN level (not fold level)
                            # Navigate up from fold_dir to campaign root
                            # Structure: experiments/<campaign>/gtransformer/<dataset>/fold_X_<id>/
                            campaign_dir = Path(fold_dir).parent.parent.parent
                            plot_dir = campaign_dir / "plots"
                            validation_dir = campaign_dir / "validation"
                            os.makedirs(plot_dir, exist_ok=True)
                            os.makedirs(validation_dir, exist_ok=True)
                            
                            print(f"\n[VALIDATION] Generating results for experiment:")
                            print(f"  Campaign: {campaign_dir.name}")
                            print(f"  Model: {model}")
                            print(f"  Dataset: {dataset}")
                            print(f"  Representative Fold: {Path(fold_dir).name}")
                            print(f"  Output:")
                            print(f"    - Plots: {plot_dir}")
                            print(f"    - Validation: {validation_dir}")
                            
                            # Conditional plot generation based on --plots flag
                            if args.plots:
                                # 1. Generate interpretability validation plots
                                validation_scripts = [
                                    {
                                        "name": "Skill Alignment Heatmap",
                                        "script": "examples/results/generate_skill_alignment_heatmap.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir),
                                            "--min_interactions": "8",
                                            "--top_skills": "50",
                                            "--top_students": "30"
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                        },
                                    {
                                        "name": "H1.3 Functional Alignment Confidence Heatmap",
                                        "script": "examples/validation/generate_skill_alignment_heatmap_h13.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(validation_dir),
                                            "--min_interactions": "5",
                                            "--top_skills": "40",
                                            "--top_students": "25"
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                    },
                                    {
                                        "name": "Structural Encoding Validation (H1.1)",
                                        "script": "examples/results/structural_encoding_validation.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir)
                                        },
                                        "required_files": []  # Auto-discovers .ckpt files
                                    },
                                    {
                                        "name": "Prediction Envelope Gallery",
                                        "script": "examples/results/generate_prediction_envelope_gallery.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir)
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                    },
                                    {
                                        "name": "Cognitive Quadrants Mosaic",
                                        "script": "examples/results/generate_quadrant_analysis.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir)
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                    },
                                    {
                                        "name": "Skill Quadrant Comparison",
                                        "script": "examples/results/generate_skill_quadrant_comparison.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir)
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                    },
                                    {
                                        "name": "Personalization Mosaic",
                                        "script": "examples/results/generate_personalization_mosaic.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir)
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                    },
                                    {
                                        "name": "Initial Mastery Mosaic",
                                        "script": "examples/results/generate_initial_mastery_mosaic.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir)
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                    },
                                    {
                                        "name": "Envelope Distribution",
                                        "script": "examples/results/generate_envelope_distribution.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir)
                                        },
                                        "required_files": ["qid_test_question_predictions_supervised.txt", "qid_test_question_predictions_reference.txt"]
                                    },
                                    {
                                        "name": "Parameter Distribution",
                                        "script": "examples/plot_param_distribution.py",
                                        "args": {"--run_dir": str(fold_dir)},
                                        "required_files": ["final_params.csv"]
                                    },
                                    {
                                        "name": "Mastery Trajectories",
                                        "script": "examples/plot_mastery_mosaic_real.py",
                                        "args": {"--run_dir": str(fold_dir)},
                                        "required_files": ["traj_mastery.csv"]
                                    },
                                    {
                                        "name": "Learning Rate Correlation",
                                        "script": "examples/plot_rate_correlation.py",
                                        "args": {"--run_dir": str(fold_dir)},
                                        "required_files": ["traj_rate.csv"]
                                    },
                                    {
                                        "name": "Student Clustering Visualization",
                                        "script": "examples/results/plot_student_clusters_gtransformer.py",
                                        "args": {
                                            "--exp_dir": str(fold_dir),
                                            "--output_dir": str(plot_dir)
                                        },
                                        "required_files": []  # Auto-discovers checkpoint and config
                                    }
                                ]
                                
                                # Track plot generation stats
                                plots_generated = 0
                                plots_skipped = 0
                                plots_failed = 0
                                missing_ref_predictions = False  # Track if reference predictions were missing
                                
                                for script_info in validation_scripts:
                                    # Check if required files exist
                                    missing_files = []
                                    for req_file in script_info.get("required_files", []):
                                        file_path = Path(fold_dir) / req_file
                                        if not file_path.exists():
                                            missing_files.append(req_file)
                                    
                                    if missing_files:
                                        # Check if missing file is due to dual_eval not being run
                                        if "qid_test_question_predictions_reference.txt" in missing_files:
                                            missing_ref_predictions = True
                                            # Check if model was grounded to provide accurate guidance
                                            eval_results_path = None
                                            for root, dirs, files in os.walk(fold_dir):
                                                if "eval_results.json" in files:
                                                    eval_results_path = Path(root) / "eval_results.json"
                                                    break
                                            
                                            is_grounded = False
                                            if eval_results_path and eval_results_path.exists():
                                                with open(eval_results_path, 'r') as f:
                                                    eval_results = json.load(f)
                                                    is_grounded = eval_results.get("grounded", False)
                                            
                                            print(f"  [WARNING] {script_info['name']} - Missing dual_eval output: {', '.join(missing_files)}")
                                            if not is_grounded:
                                                print(f"            This experiment was trained without grounding (active_grounding=0)")
                                                print(f"            To generate this plot, retrain with ablation=none (default) which sets active_grounding=1")
                                            else:
                                                print(f"            This experiment was trained without dual_eval enabled")
                                                print(f"            To generate this plot, retrain with dual_eval=true in configs/parameter_default.json")
                                        else:
                                            print(f"  [SKIP] {script_info['name']} (missing: {', '.join(missing_files)})")
                                        plots_skipped += 1
                                        continue
                                    
                                    script_path = Path(PROJECT_ROOT) / script_info["script"]
                                    if script_path.exists():
                                        print(f"  [RUN] {script_info['name']}...")
                                        cmd = [sys.executable, str(script_path)]
                                        for arg_name, arg_val in script_info["args"].items():
                                            cmd.extend([arg_name, arg_val])
                                        try:
                                            result = subprocess.run(cmd, check=False, cwd=PROJECT_ROOT,
                                                                  capture_output=True, text=True, timeout=300)
                                            if result.returncode != 0:
                                                print(f"    [WARN] Script exited with code {result.returncode}")
                                                if result.stderr:
                                                    print(f"    Error: {result.stderr[:200]}")
                                                plots_failed += 1
                                            else:
                                                print(f"    [OK] Completed successfully")
                                                plots_generated += 1
                                        except subprocess.TimeoutExpired:
                                            print(f"    [WARN] Timeout (300s)")
                                            plots_failed += 1
                                        except Exception as e:
                                            print(f"    [WARN] Failed: {e}")
                                            plots_failed += 1
                                    else:
                                        print(f"  [SKIP] {script_info['name']} (script not found)")
                                        plots_skipped += 1
                                # Print summary
                                print(f"\n  Plot Generation Summary:")
                                print(f"    ✓ Generated: {plots_generated}")
                                print(f"    ✗ Skipped: {plots_skipped}")
                                print(f"    ⚠ Failed: {plots_failed}")
                                if missing_ref_predictions:
                                    print(f"\n  Note: Some plots were skipped due to missing 'qid_test_question_predictions_reference.txt'")
                                    print(f"        This file requires BOTH:")
                                    print(f"          1. Model trained with grounding (ablation=none, not ablation=all)")
                                    print(f"          2. Evaluation with dual_eval=true in configs/parameter_default.json")
                            else:
                                print(f"\n  [INFO] Plot generation skipped (--plots=false)")
                            
                            # 2. Generate diagnostic probing results for validation
                            print(f"\n  Diagnostic Probing Analysis:")
                            validation_dir = campaign_dir / "validation"
                            validation_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Check if this is a grounded model (gtransformer with active grounding)
                            eval_results_path = None
                            for root, dirs, files in os.walk(fold_dir):
                                if "eval_results.json" in files:
                                    eval_results_path = Path(root) / "eval_results.json"
                                    break
                            
                            is_grounded = False
                            if eval_results_path and eval_results_path.exists():
                                with open(eval_results_path, 'r') as f:
                                    eval_results = json.load(f)
                                    is_grounded = eval_results.get("grounded", False)
                            
                            if not is_grounded:
                                print(f"  [SKIP] Probing analysis (model not grounded)")
                                print(f"         This experiment was trained without active grounding")
                                print(f"         Probing requires ablation=none (default) which sets active_grounding=1")
                            else:
                                # Find checkpoint file
                                checkpoint_path = None
                                for root, dirs, files in os.walk(fold_dir):
                                    for f in files:
                                        if f.endswith(".ckpt") or f == "model_best.pth":
                                            checkpoint_path = Path(root) / f
                                            break
                                    if checkpoint_path:
                                        break
                                
                                if not checkpoint_path or not checkpoint_path.exists():
                                    print(f"  [SKIP] Parameter recovery validation (checkpoint not found)")
                                else:
                                    print(f"  [RUN] Validating parameter recovery...")
                                    print(f"        Checkpoint: {checkpoint_path.name}")
                                    print(f"        Output: {validation_dir}")
                                    
                                    validation_script = Path(PROJECT_ROOT) / "examples" / "validation" / "validate_parameter_recovery.py"
                                    if validation_script.exists():
                                        cmd = [
                                            sys.executable, 
                                            str(validation_script),
                                            "--exp_dir", str(fold_dir),
                                            "--output_dir", str(validation_dir)
                                        ]
                                        
                                        try:
                                            result = subprocess.run(cmd, check=False, cwd=PROJECT_ROOT,
                                                                  capture_output=True, text=True, timeout=600)
                                            if result.returncode != 0:
                                                print(f"    [WARN] Validation exited with code {result.returncode}")
                                                if result.stderr:
                                                    print(f"    Error: {result.stderr[:300]}")
                                            else:
                                                print(f"    [OK] Validation completed successfully")
                                                # Check if results were generated
                                                recovery_file = validation_dir / "recovery_summary.json"
                                                if recovery_file.exists():
                                                    with open(recovery_file, 'r') as f:
                                                        recovery_results = json.load(f)
                                                    print(f"    Results:")
                                                    print(f"      L0 Probe r:   {recovery_results.get('l0_probe', {}).get('pearson_r', 'N/A'):.4f}")
                                                    print(f"      T Probe r:    {recovery_results.get('t_probe', {}).get('pearson_r', 'N/A'):.4f}")
                                                    print(f"      Samples:      {recovery_results.get('n_samples', 'N/A')}")
                                        except subprocess.TimeoutExpired:
                                            print(f"    [WARN] Validation timeout (600s)")
                                        except Exception as e:
                                            print(f"    [WARN] Validation failed: {e}")
                                    else:
                                        print(f"  [SKIP] Validation script not found: {validation_script}")
        
        print(f"\n{'='*70}")
        print(f"VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"Results summary saved to: {json_path}")
        print(f"Individual experiment plots and validation results saved to respective experiment folders")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
