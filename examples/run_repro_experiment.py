#!/usr/bin/env python3
"""
Simplified reproducible experiment launcher with integrated reproduction mode.

TRAINING MODE (default):
    Creates new experiment with 6-digit ID
    python examples/run_repro_experiment.py \
        --dataset assist2015 \
        --short_title baseline

REPRODUCTION MODE:
    Reproduces existing experiment by ID (all parameters read from original config)
    python examples/run_repro_experiment.py \
        --repro_experiment_id 423891
"""
import argparse
import json
import os
import sys
import subprocess
import hashlib
import random
from pathlib import Path
from datetime import datetime

# Model-specific script-to-parameter mapping
MODEL_SCRIPTS = {
    "dkt": "examples/wandb_dkt_train.py",
    "dkt+": "examples/wandb_dkt_plus_train.py",
    "dkt_forget": "examples/wandb_dkt_forget_train.py",
    "kqn": "examples/wandb_kqn_train.py",
    "dkvmn": "examples/wandb_dkvmn_train.py",
    "atkt": "examples/wandb_atkt_train.py",
    "atktfix": "examples/wandb_atkt_train.py",
    "gkt": "examples/wandb_gkt_train.py",
    "sakt": "examples/wandb_sakt_train.py",
    "saint": "examples/wandb_saint_train.py",
    "saint++": "examples/wandb_saint_plus_plus_train.py",
    "akt": "examples/wandb_akt_train.py",
    "gtransformer": "examples/wandb_gtransformer_train.py",
    "idkt": "examples/train_idkt.py",
    "lpkt": "examples/wandb_lpkt_train.py",
    "skvmn": "examples/wandb_skvmn_train.py",
    "deep_irt": "examples/wandb_deep_irt_train.py"
}

# Parameter Translation Map (Canonical -> Script Specific)
PARAM_MAP = {
    "d_model": {
        "dkt": "emb_size", "dkt+": "emb_size", "dkt_forget": "emb_size", "sakt": "emb_size",
        "dkvmn": "dim_s", "atkt": "skill_dim", "saint": "emb_size", "saint++": "emb_size", "gkt": "hidden_dim",
        "kqn": "n_hidden", # KQN uses n_hidden for its_main embedding size, which maps from d_model
        "skvmn": "dim_s", "deep_irt": "dim_s", "akt": "d_model", "gtransformer": "d_model"
    },
    "n_heads": {
        "akt": "num_attn_heads", "gtransformer": "num_attn_heads", "sakt": "num_attn_heads", 
        "saint": "num_attn_heads", "saint++": "num_attn_heads"
    },
    "n_blocks": {
        "sakt": "num_en", "saint": "n_blocks", "saint++": "n_blocks", "akt": "n_blocks", "gtransformer": "n_blocks"
    },
    "d_ff": {
        "atkt": "attention_dim"
    },
    "dataset": {
        "standard_pykt": "dataset_name"
    },
    "model": {
        "standard_pykt": "model_name"
    },
    "epochs": {
        "standard_pykt": "num_epochs"
    }
}

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def select_gpus(num_gpus=None):
    """
    Select GPUs to use for training. Automatically uses ~80% of available GPUs
    unless explicitly specified.
    
    Priority:
    1. If CUDA_VISIBLE_DEVICES is set: use as-is (user has explicit control)
    2. If num_gpus is specified: use first N GPUs
    3. Default: use ~80% of available GPUs (e.g., 6 out of 8)
    
    Args:
        num_gpus: Optional explicit number of GPUs to use
        
    Returns:
        str: GPU selection for CUDA_VISIBLE_DEVICES (e.g., "0,1,2,3,4,5") or empty string
    """
    # Check if user already set CUDA_VISIBLE_DEVICES
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        return ''  # User has explicit control, don't override
    
    try:
        import torch
        if not torch.cuda.is_available():
            return ''  # No CUDA available
        
        total_gpus = torch.cuda.device_count()
        if total_gpus == 0:
            return ''
        
        # Determine how many GPUs to use
        if num_gpus is not None:
            # User explicitly specified
            gpus_to_use = min(num_gpus, total_gpus)
        else:
            # Default: use ~80% of available GPUs (rounded down)
            # For 8 GPUs: 8 * 0.8 = 6.4 → 6 GPUs
            # For 4 GPUs: 4 * 0.8 = 3.2 → 3 GPUs
            gpus_to_use = max(1, int(total_gpus * 0.8))
        
        # Generate GPU ID list: "0,1,2,3,4,5"
        gpu_ids = ','.join(str(i) for i in range(gpus_to_use))
        
        print(f"\n{'='*80}")
        print("GPU SELECTION")
        print(f"{'='*80}")
        print(f"Total GPUs available: {total_gpus}")
        print(f"GPUs selected for training: {gpus_to_use} ({gpus_to_use/total_gpus*100:.0f}%)")
        print(f"CUDA_VISIBLE_DEVICES will be set to: {gpu_ids}")
        print(f"{'='*80}\n")
        
        return gpu_ids
    except ImportError:
        # PyTorch not available
        return ''

def get_required_param(config, section, param_name):
    """
    Get a required parameter from config with strict validation.
    Priority: input section -> defaults section -> ERROR
    No hardcoded fallbacks allowed.
    """
    # Check input section first
    if 'input' in config and param_name in config['input']:
        return config['input'][param_name]
    
    # Check defaults section
    if section in config and param_name in config[section]:
        return config[section][param_name]
    
    # For backward compatibility, also check top-level defaults
    if 'defaults' in config and param_name in config['defaults']:
        return config['defaults'][param_name]
    
    # Parameter not found - this is an error
    raise ValueError(
        f"Required parameter '{param_name}' not found in config.\n"
        f"Expected in: input section (user override) or {section} section (default value).\n"
        f"Please ensure parameter_default.json contains this parameter."
    )

def generate_experiment_id():
    """Generate a unique 6-digit experiment ID."""
    return str(random.randint(100000, 999999))

def find_experiment_folder(experiment_id, base_dir="experiments"):
    """Find experiment folder containing the given experiment ID (recursive search)."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # Search for folder containing the experiment_id (excluding _repro folders for original search)
    candidates = []
    # Search recursively up to 2 levels (e.g. experiments/campaign/fold_ID)
    for folder in base_path.rglob(f"*{experiment_id}*"):
        if folder.is_dir():
            # Prioritize non-repro folders
            if "_repro" not in folder.name:
                return folder
            candidates.append(folder)
    
    # If only repro folders found, return the first one
    return candidates[0] if candidates else None

def create_experiment_folder(model_name, short_title, experiment_id, is_repro=False, parent_folder=None, fold=None):
    """
    Create experiment folder. 
    If parent_folder is provided, uses nested structure: parent_folder/fold_X_ID/
    Otherwise uses flat structure: experiments/YYYYMMDD_HHMMSS_model_title_ID/
    """
    if parent_folder:
        folder_parent = Path(parent_folder)
        folder_parent.mkdir(parents=True, exist_ok=True)
        
        subfolder_name = []
        if fold is not None:
            subfolder_name.append(f"fold_{fold}")
        subfolder_name.append(experiment_id)
        
        folder_name = "_".join(subfolder_name)
        if is_repro:
            folder_name += "_repro"
        
        folder_path = folder_parent / folder_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{model_name}_{short_title}_{experiment_id}"
        if is_repro:
            folder_name += "_repro"
        folder_path = PROJECT_ROOT / "experiments" / folder_name
    
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path

def load_config_from_experiment(experiment_id):
    """Load config.json from existing experiment folder."""
    folder = find_experiment_folder(experiment_id)
    if folder is None:
        raise FileNotFoundError(f"No experiment folder found containing ID: {experiment_id}")
    
    config_path = folder / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {folder}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Verify MD5 integrity on load
    verify_config_md5(config)
    
    return config, folder

def verify_config_md5(config):
    """
    Verify that the MD5 of current defaults section matches the stored md5.
    Prints confirmation or warning message.
    
    Note: config['md5'] should contain the MD5 of the resolved config (defaults + overrides),
    not the original parameter_default.json MD5.
    """
    if 'defaults' not in config or 'md5' not in config:
        print("⚠️  WARNING: Config missing 'defaults' or 'md5' section - cannot verify integrity")
        return False
    
    # Calculate current MD5 from defaults section
    defaults_str = json.dumps(config['defaults'], sort_keys=True)
    current_md5 = hashlib.md5(defaults_str.encode()).hexdigest()
    stored_md5 = config['md5']
    
    if current_md5 == stored_md5:
        print(f"✓ Config integrity verified (MD5: {current_md5})")
        return True
    else:
        print("⚠️  WARNING: Config MD5 mismatch!")
        print(f"   Expected (stored): {stored_md5}")
        print(f"   Got (computed):    {current_md5}")
        print("   This suggests the defaults section has been modified after initial save.")
        return False

def save_config(config, experiment_folder):
    """Save config.json to experiment folder."""
    config_path = experiment_folder / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path

def build_train_command_from_config(train_script, config_path, experiment_dir):
    """Build the training command using --config flag and set EXPERIMENT_DIR."""
    python_path = sys.executable
    return f"EXPERIMENT_DIR={experiment_dir} {python_path} {train_script} --config {config_path}"

def build_explicit_train_command(train_script, params, experiment_dir=None):
    python_path = sys.executable
    # Use absolute path for train script to allow changing working directory
    abs_train_script = os.path.abspath(train_script)
    cmd_parts = [python_path, abs_train_script]
    
    # Launcher-only parameters (not passed to training script)
    launcher_only_params = {'train_script', 'eval_script', 'max_correlation_students', 'short_title'}
    
    # Canonical parameter groups
    architecture_params = {'seq_len', 'd_model', 'n_heads', 'n_blocks', 'd_ff', 'dropout', 'emb_type'}
    runtime_params = {'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'optimizer', 'gradient_clip', 'patience', 'seed'}
    
    model = params.get('model', 'idkt')
    is_standard_pykt = "wandb_" in train_script

    # Determine which parameters to pass based on training script
    if 'train_idkt.py' in train_script:
        allowed_params = {
            'model', 'dataset', 'fold', 'seed', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 
            'optimizer', 'gradient_clip', 'patience', 'seq_len', 'd_model', 'n_heads', 'n_blocks', 
            'd_ff', 'dropout', 'emb_type', 'final_fc_dim', 'l2', 'lambda_student', 'lambda_gap', 
            'lambda_ref', 'lambda_initmastery', 'lambda_rate', 'theory_guided', 'calibrate',
            'bkt_filter', 'bkt_guess_threshold', 'bkt_slip_threshold', 'grounded_init', 'use_wandb',
            'save_dir', '_doc_grounding', '_doc_regularization',
            'answer_dim', 'beta', 'epsilon', 'graph_type', 'lambda_r', 'lambda_w1', 'lambda_w2', 
            'size_m', 'n_hidden', 'n_rnn_hidden', 'n_mlp_hidden', 'hidden_dim', 'num_attn_heads', 
            'num_en', 'skill_dim', 'attention_dim', 'dim_s', 'emb_size'
        }
    elif is_standard_pykt:
        # base set for all standard pykt scripts
        allowed_params = {
            'model', 'dataset', 'fold', 'seed', 'learning_rate', 'dropout', 'use_wandb', 'add_uuid', 
            'save_dir', 'epochs', 'batch_size', 'weight_decay', 'gradient_clip', 'patience',
            'seq_len', 'emb_type', 'emb_path', 'optimizer'
        }
        
        # Architecture if supported by that specific script
        if model == 'akt' or model == 'gtransformer':
            allowed_params.update({
                'd_model', 'n_heads', 'n_blocks', 'd_ff', 'final_fc_dim', 'l2',
                'kq_same', 'separate_qa', 'pretrain_dim', 'ablation', 'n_uid', 'l2_rasch'
            })
        elif model == 'sakt':
            allowed_params.update({'d_model', 'n_heads', 'n_blocks'})
        elif model == 'saint' or model == 'saint++':
            allowed_params.update({'d_model', 'n_heads', 'n_blocks'})
        elif model == 'atkt':
            allowed_params.update({'d_model', 'd_ff', 'answer_dim', 'epsilon', 'beta', 'hidden_dim', 'skill_dim', 'attention_dim'})
        elif model == 'dkvmn':
            allowed_params.update({'d_model', 'size_m'})
        elif model == 'gkt':
            allowed_params.update({'d_model', 'graph_type', 'hidden_dim'})
        elif model in ['dkt', 'skvmn', 'deep_irt']:
            allowed_params.update({'d_model'})
        elif model == 'kqn':
            allowed_params.update({'d_model', 'n_hidden', 'n_rnn_hidden', 'n_mlp_hidden'})
        elif model == 'dkt+':
            allowed_params.update({'d_model', 'lambda_r', 'lambda_w1', 'lambda_w2'})
        elif model == 'dkt_forget':
            allowed_params.update({'d_model', 'num_rgap', 'num_sgap', 'num_pcount'})
    else:
        # Default safety: only pass runtime basics
        allowed_params = {'dataset', 'fold', 'seed', 'epochs', 'batch_size', 'learning_rate', 'save_dir', 'fusion_type'}
    
    # Build command parts, ensuring mapped canonical keys (like d_model) take precedence
    final_params = {}
    is_canonical = {} # translated_key -> bool
    
    for key, value in sorted(params.items()):
        if key in launcher_only_params:
            continue
        if allowed_params is not None and key not in allowed_params:
            continue
        
        translated_key = key
        mapped = False
        if key in PARAM_MAP:
            if model in PARAM_MAP[key]:
                translated_key = PARAM_MAP[key][model]
                mapped = True
            elif is_standard_pykt and "standard_pykt" in PARAM_MAP[key]:
                translated_key = PARAM_MAP[key]["standard_pykt"]
                mapped = True
        
        # Logic: 
        # - If key is new, store it.
        # - If key exists, only overwrite if current 'key' is mapped (canonical) and previous wasn't
        if translated_key not in final_params:
            final_params[translated_key] = value
            is_canonical[translated_key] = mapped
        elif mapped and not is_canonical[translated_key]:
            final_params[translated_key] = value
            is_canonical[translated_key] = True

    for translated_key, value in sorted(final_params.items()):
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{translated_key}")
        elif value is None or value == "null":
            # For reproducibility: pass "null" as string literal (required=True compliance)
            # Special case: construct rasch_path dynamically
            if translated_key == "rasch_path" and "dataset" in params:
                rasch_path = f"data/{params['dataset']}/rasch_targets.pkl"
                cmd_parts.append(f"--{translated_key} {rasch_path}")
            else:
                # Pass "null" as string to satisfy required=True
                cmd_parts.append(f"--{translated_key} null")
        elif value == "None":
            # String "None" should be passed as "null" for consistency
            cmd_parts.append(f"--{translated_key} null")
        else:
            # Don't add quotes - shlex.split will handle proper parsing
            val_str = str(value)
            cmd_parts.append(f"--{translated_key} {val_str}")
    
    # Add save_dir for all models to ensure they save results into the experiment folder
    if experiment_dir:
        cmd_parts.append(f"--save_dir {experiment_dir}")
    
    # Standard PyKT scripts (wandb_*.py) must be run from examples/ 
    # to find '../configs/'. We wrap with surrogate if needed.
    if "wandb_" in train_script:
        if model != "idkt":
            surrogate = os.path.join(PROJECT_ROOT, "tmp/pykt_train_surrogate.py")
            # Build command args list (not string) to avoid shell expansion issues
            surrogate_args = [python_path, surrogate] + cmd_parts[2:]
            # Don't add quotes here - shlex.split will handle them
            command = " ".join(str(arg) for arg in surrogate_args)
            # Set env var for target script
            command = f"PYKT_TARGET_SCRIPT={abs_train_script} {command}"
        else:
            command = " ".join(cmd_parts)
    else:
        command = " ".join(cmd_parts)
        
    return command

def build_trajectory_command(experiment_folder, num_students=10, min_steps=10):
    """
    Build command to extract and display learning trajectories.
    """
    python_path = sys.executable
    trajectory_script = "examples/learning_trajectories.py"
    cmd = f"{python_path} {trajectory_script} --run_dir {experiment_folder} --num_students {num_students} --min_steps {min_steps}"
    return cmd

def build_mastery_states_command(experiment_folder, num_students=15, split='test'):
    """
    Build command to extract mastery states.
    """
    python_path = sys.executable
    mastery_script = "examples/mastery_states.py"
    cmd = f"{python_path} {mastery_script} --run_dir {experiment_folder} --num_students {num_students} --split {split}"
    return cmd


def build_bkt_validation_command(experiment_folder, params):
    """
    Build command to compute BKT validation metric.
    
    Args:
        experiment_folder: Path to experiment directory
        params: Config parameters (needs dataset)
    """
    python_path = sys.executable
    bkt_script = "examples/compute_bkt_correlation.py"
    
    cmd = (
        f"{python_path} {bkt_script} "
        f"--experiment_dir {experiment_folder} "
        f"--dataset {params['dataset']} "
        f"--output_file bkt_validation.json"
    )
    return cmd


def build_analysis_plots_command(experiment_folder, params=None):
    """
    Build command to generate comprehensive analysis plots.
    """
    python_path = sys.executable
    plots_script = "examples/generate_analysis_plots.py"
    cmd = f"{python_path} {plots_script} --run_dir {experiment_folder}"
    
    if params:
        if params.get('bkt_filter', False):
            cmd += " --filter_bkt"
            # Automatically find matching params pkl in the data folder
            dataset = params.get('dataset', 'assist2015')
            cmd += f" --bkt_params_path data/{dataset}/bkt_skill_params.pkl"
            
            # Pass thresholds if available in params
            if 'bkt_guess_threshold' in params:
                cmd += f" --guess_threshold {params['bkt_guess_threshold']}"
            if 'bkt_slip_threshold' in params:
                cmd += f" --slip_threshold {params['bkt_slip_threshold']}"
                
    return cmd


def build_validation_plots_command(experiment_folder):
    """
    Build command to generate specialized validation plots (consensus, residual, uncertainty).
    """
    python_path = sys.executable
    plots_script = "examples/generate_validation_plots.py"
    cmd = f"{python_path} {plots_script} --run_dir {experiment_folder}"
    return cmd


def build_explicit_eval_command(eval_script, experiment_folder, params):
    """
    Build explicit evaluation command.
    - iDKT: uses examples/eval_idkt.py with full parameters
    - Baselines: uses examples/wandb_predict.py with simplified interface
    """
    python_path = sys.executable
    model = params['model']
    
    # 1. iDKT (and related custom models) Evaluation
    if model in ['idkt', 'ikt2', 'ikt3']:
        cmd_parts = [python_path, eval_script]
        
        # Checkpoint and Output
        checkpoint_path = f"{experiment_folder}/best_model.pt"
        cmd_parts.append(f"--checkpoint {checkpoint_path}")
        cmd_parts.append(f"--output_dir {experiment_folder}")
        
        # Base Parameters
        core_eval_params = ['dataset', 'fold', 'batch_size']
        
        # Model-Specific Parameters
        model_params = []
        if model == 'idkt':
            model_params = ['d_model', 'n_heads', 'n_blocks', 'd_ff', 'dropout', 'final_fc_dim', 'l2', 
                           'emb_type', 'seq_len', 'lambda_student', 'lambda_gap', 'lambda_ref', 
                           'lambda_initmastery', 'lambda_rate', 'grounded_init', 'theory_guided']
        elif model == 'ikt3':
            # Add reference_targets_path if present
            if params.get('reference_targets_path'):
                cmd_parts.append(f"--reference_targets_path {params['reference_targets_path']}")
                
        # Append parameters
        all_params = core_eval_params + model_params
        for key in all_params:
            if key in params:
                value = params[key]
                if isinstance(value, bool):
                    # Handle boolean flags like --theory_guided 1
                    cmd_parts.append(f"--{key} {int(value)}")
                else:
                    cmd_parts.append(f"--{key} {value}")
                    
        return " ".join(cmd_parts)

    # 2. Baseline Models Evaluation (PyKT standard)
    else:
        # Baselines use wandb_predict.py which needs to run from examples/ directory
        # It reads configuration directly from the experiment folder's config.json
        predict_script = os.path.join(os.path.dirname(__file__), "wandb_predict.py")
        abs_predict_script = os.path.abspath(predict_script)
        
        cmd_parts = [python_path, abs_predict_script]
        cmd_parts.append(f"--save_dir {experiment_folder}")
        cmd_parts.append(f"--bz {params['batch_size']}")
        cmd_parts.append(f"--use_wandb 0")
        
        # Use late_fusion if requested, otherwise default to none for consistency
        fusion_type = params.get('fusion_type', 'none')
        cmd_parts.append(f"--fusion_type {fusion_type}")
        
        return " ".join(cmd_parts)

def build_eval_command(eval_script, model_path):
    """Build evaluation command (legacy - for simple eval script)."""
    python_path = sys.executable
    return f"{python_path} {eval_script} --model_path {model_path}"

def build_repro_command(repro_script, experiment_id):
    """Build reproduction command."""
    python_path = sys.executable
    return f"{python_path} {repro_script} --repro_experiment_id {experiment_id}"

def compute_training_md5(config):
    """
    Compute MD5 hash based on the 'defaults' section only.
    
    The "defaults" section contains ALL parameters that directly affect
    training results, including model name.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        str: MD5 hash of defaults section only
    """
    if "defaults" not in config:
        raise KeyError("Config missing 'defaults' section - cannot compute training MD5")
    
    # Compute MD5 on ONLY the defaults section
    defaults_str = json.dumps(config["defaults"], sort_keys=True)
    defaults_md5 = hashlib.md5(defaults_str.encode()).hexdigest()
    
    return defaults_md5


def run_parameter_audit():
    """
    Run reproducibility infrastructure audit before launching training/evaluation.
    
    Returns:
        bool: True if audit passed, False otherwise
    """
    print("=" * 80)
    print("REPRODUCIBILITY PRE-FLIGHT CHECK")
    print("=" * 80)
    print("Running parameter audit to verify infrastructure compliance...")
    print()
    
    # Run audit script
    audit_script = Path(__file__).parent / "parameters_audit.py"
    if not audit_script.exists():
        print(f"⚠️  WARNING: Audit script not found: {audit_script}")
        print("Proceeding without audit (not recommended)")
        return True
    
    result = subprocess.run([sys.executable, str(audit_script)], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Pre-flight check PASSED - Safe to proceed")
        return True
    else:
        print("\n❌ Pre-flight check FAILED")
        print("\nReproducibility infrastructure has issues that must be fixed before launching.")
        print("See error messages above for details.")
        print("\nTo bypass this check (NOT RECOMMENDED), set environment variable:")
        print("  export SKIP_PARAMETER_AUDIT=1")
        return False

def reproduce_experiment(repro_experiment_id, num_gpus=None, dry_run=False):
    """
    Reproduce an existing experiment.
    """
    # Load defaults to check for non-default values (needed for get_required_param context if used)
    # But mainly we need to load original config.
    
    print("=" * 80)
    print("REPRODUCIBILITY MODE")
    print("=" * 80)
    print(f"Searching for experiment ID: {repro_experiment_id}")
    
    # Load original config unchanged
    config, original_folder = load_config_from_experiment(repro_experiment_id)
    print(f"✓ Found original experiment: {original_folder.name}")
    print("✓ Loaded config.json")
    
    # Extract model_name and short_title from original config
    try:
        if 'input' in config:
            # New structure: check input first, then defaults
            model_name = get_required_param(config, 'defaults', 'model')
            if 'short_title' not in config['input']:
                raise ValueError("'short_title' missing from input section in config")
            original_short_title = config['input']['short_title']
        else:
            # Backward compatibility
            if 'defaults' in config and 'model' in config['defaults']:
                model_name = config['defaults']['model']
            elif 'training' in config and 'model' in config['training']:
                model_name = config['training']['model']
            elif 'experiment' in config and 'model' in config['experiment']:
                model_name = config['experiment']['model']
            else:
                raise ValueError("'model' parameter not found in config")
            
            if 'experiment' in config and 'short_title' in config['experiment']:
                original_short_title = config['experiment']['short_title']
            elif 'input' in config and 'short_title' in config['input']:
                original_short_title = config['input']['short_title']
            else:
                raise ValueError("'short_title' not found in config")
    except ValueError as e:
        print(f"❌ ERROR: Invalid config structure in experiment {repro_experiment_id}")
        print(f"   {str(e)}")
        sys.exit(1)
    
    repro_short_title = original_short_title
    
    # Create reproduction experiment folder
    repro_folder = create_experiment_folder(
        model_name=model_name,
        short_title=repro_short_title,
        experiment_id=repro_experiment_id,
        is_repro=True
    )
    print(f"✓ Created reproduction folder: {repro_folder.name}")
    
    # Copy config.json unchanged
    save_config(config, repro_folder)
    print("✓ Copied config.json (unchanged)")
    
    # Select GPUs
    gpu_selection = select_gpus(num_gpus)
    
    # Use train_explicit command
    repro_dir_abs = str(repro_folder.absolute())
    if gpu_selection:
        train_command = f"CUDA_VISIBLE_DEVICES={gpu_selection} EXPERIMENT_DIR={repro_dir_abs} {config['commands']['train_explicit']}"
    else:
        train_command = f"EXPERIMENT_DIR={repro_dir_abs} {config['commands']['train_explicit']}"
    
    print(f"\n{'Original experiment:':<25} {original_folder.name}")
    print(f"{'Reproduction folder:':<25} {repro_folder.name}")
    print(f"\n{'Training command:':<25}")
    print(f"  {train_command}")
    
    if dry_run:
        print("\n[DRY RUN] Skipping training execution")
        print("\nTo execute training, run:")
        print(f"  {train_command}")
        return
    
    # Launch training
    print("\n" + "=" * 80)
    print("LAUNCHING REPRODUCTION TRAINING")
    print("=" * 80 + "\n")
    result = subprocess.run(train_command, shell=True)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✓ REPRODUCTION TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nResults saved to: {repro_folder}")
        
        # Launch evaluation
        print("\n" + "=" * 80)
        print("LAUNCHING EVALUATION")
        print("=" * 80 + "\n")
        
        eval_command_explicit = config['commands']['eval_explicit']
        # Update experiment dir for reproduction folder
        original_dir_abs = str(original_folder.absolute())
        eval_command_explicit = eval_command_explicit.replace(original_dir_abs, repro_dir_abs)
        eval_command_full = f"EXPERIMENT_DIR={repro_dir_abs} {eval_command_explicit}"
        
        eval_result = subprocess.run(eval_command_full, shell=True)
        
        if eval_result.returncode == 0:
            print("\n✓ Evaluation completed successfully")
            print(f"  Results saved to: {repro_folder}/eval_results.json")
        else:
            print("\n⚠️  Evaluation failed (non-critical)")
    else:
        print("\n" + "=" * 80)
        print("❌ TRAINING FAILED")
        print("=" * 80)
        sys.exit(result.returncode)

def run_train_fold(args, defaults_config, fold, model_name, dataset, short_title, experiment_id, parent_folder=None):
    """
    Execute a single fold training and evaluation run.
    Returns: (success, metrics_dict)
    """
    # 1. Setup specific params for this fold
    default_train_script = get_required_param(defaults_config, "defaults", "train_script")
    default_eval_script = get_required_param(defaults_config, "defaults", "eval_script")
    
    train_script = args.train_script if args.train_script is not None else default_train_script
    
    # Automatically select training script based on model if using default
    if train_script == default_train_script and model_name in MODEL_SCRIPTS:
        train_script = MODEL_SCRIPTS[model_name]
        
    eval_script = default_eval_script

    # Create experiment folder
    experiment_folder = create_experiment_folder(
        model_name=model_name,
        short_title=short_title,
        experiment_id=experiment_id,
        is_repro=False,
        parent_folder=parent_folder,
        fold=fold
    )
    print(f"✓ Created fold folder: {experiment_folder.name}")

    # 2. Build configuration
    # Start with global defaults
    training_params = defaults_config["defaults"].copy()
    
    # Priority 2: kt_config.json (Model-specific defaults)
    kt_config_path = PROJECT_ROOT / "configs" / "kt_config.json"
    if kt_config_path.exists():
        with open(kt_config_path, 'r') as f:
            kt_config = json.load(f)
            # Apply global train_config if present
            if "train_config" in kt_config:
                for k, v in kt_config["train_config"].items():
                    if k in training_params:
                        training_params[k] = v
            # Apply model-specific parameters
            if model_name in kt_config:
                for k, v in kt_config[model_name].items():
                    if k in training_params:
                        training_params[k] = v
    
    # Priority 3: kt_config_[dataset].json (Dataset-fold-model-specific optimal values)
    # Handle dataset variants (strip _S, _bkt, _quelevel suffixes for lookup)
    canonical_dataset = dataset
    for suffix in ["_S", "_bkt", "_quelevel", "_BKT"]:
        if canonical_dataset.endswith(suffix):
            canonical_dataset = canonical_dataset[:-len(suffix)]
            
    dataset_paths = [
        PROJECT_ROOT / "configs" / f"kt_config_{dataset}.json",
        PROJECT_ROOT / "configs" / f"kt_config_{canonical_dataset}.json"
    ]
    
    config_found = False
    for dataset_config_path in dataset_paths:
        if dataset_config_path.exists():
            with open(dataset_config_path, 'r') as f:
                dataset_config = json.load(f)
                if model_name in dataset_config:
                    fold_str = str(fold)
                    if fold_str in dataset_config[model_name]:
                        opt_params = dataset_config[model_name][fold_str]
                        print(f"  ✓ Applied optimal hyperparameters for {model_name} from {dataset_config_path.name} fold {fold}")
                        
                        # Create reverse mapping to translate tuning keys back to canonical keys
                        REVERSE_PARAM_MAP = {}
                        for canonical_key, model_map in PARAM_MAP.items():
                            if model_name in model_map:
                                script_key = model_map[model_name]
                                REVERSE_PARAM_MAP[script_key] = canonical_key
                            elif "standard_pykt" in model_map:
                                script_key = model_map["standard_pykt"]
                                REVERSE_PARAM_MAP[script_key] = canonical_key

                        for k, v in opt_params.items():
                            # Translate if mapping exists
                            target_key = REVERSE_PARAM_MAP.get(k, k)
                            if target_key in training_params:
                                training_params[target_key] = v
                            # Also check if the raw key is in training_params (for keys not in PARAM_MAP)
                            elif k in training_params:
                                training_params[k] = v
                        config_found = True
                        break
                if config_found:
                    break

    training_params["model"] = model_name
    training_params["dataset"] = dataset
    training_params["fold"] = fold

    # Apply overrides
    overrides = {}
    for param_name in defaults_config["defaults"].keys():
        if param_name in ['train_script', 'eval_script', 'model', 'dataset', 'fold']:
            continue
        arg_value = getattr(args, param_name, None)
        default_value = defaults_config["defaults"][param_name]
        if isinstance(default_value, bool):
            if arg_value != default_value:
                overrides[param_name] = arg_value
                training_params[param_name] = arg_value
        elif arg_value is not None:
            overrides[param_name] = arg_value
            training_params[param_name] = arg_value

    # Build input params for config
    input_params = {
        "short_title": short_title,
        "dataset": dataset,
        "fold": fold,
        "model": model_name
    }
    if args.train_script is not None:
        input_params["train_script"] = args.train_script
    input_params.update(overrides)

    # 3. Build Commands
    experiment_dir_abs = str(experiment_folder.absolute())
    gpu_selection = select_gpus(args.num_gpus)
    
    train_command_explicit = build_explicit_train_command(train_script, training_params, experiment_dir_abs)
    eval_command_explicit = build_explicit_eval_command(eval_script, experiment_dir_abs, training_params)
    mastery_states_command = build_mastery_states_command(experiment_dir_abs, num_students=15, split='test')

    python_path = sys.executable
    
    # 4. Save Config
    config_dict = {
        "experiment": {
            "id": experiment_id,
            "created": datetime.now().isoformat(),
            "repro_command": " ".join(sys.argv), 
            "parent_folder": str(parent_folder) if parent_folder else None
        },
        "input": input_params,
        "defaults": defaults_config["defaults"],
        "train_config": training_params,
        "commands": {
            "train_explicit": train_command_explicit,
            "eval_explicit": eval_command_explicit,
            "mastery_states": mastery_states_command,
            "plot_param_distribution": f"{python_path} examples/plot_param_distribution.py --experiment_dir {experiment_dir_abs}",
            "plot_mastery_mosaic": f"{python_path} examples/plot_mastery_mosaic.py --experiment_dir {experiment_dir_abs}",
            "train_probe": f"{python_path} examples/train_probe.py --experiment_dir {experiment_dir_abs} --dimension difficult"
        }
    }
    
    # Calculate checksum for integrity
    defaults_md5 = compute_training_md5(config_dict)
    config_dict["md5"] = defaults_md5
    
    with open(experiment_folder / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # 5. Execute Training
    if gpu_selection:
        run_cmd = f"CUDA_VISIBLE_DEVICES={gpu_selection} EXPERIMENT_DIR={experiment_dir_abs} {train_command_explicit}"
    else:
        run_cmd = f"EXPERIMENT_DIR={experiment_dir_abs} {train_command_explicit}"

    print(f"  Training Fold {fold}...")
    if args.dry_run:
        print(f"  [DRY RUN] Would run: {run_cmd}")
        return True, {}

    print(f"  Execution CWD: {PROJECT_ROOT / 'examples'}")
    # Ensure PROJECT_ROOT is in PYTHONPATH so pykt can be found
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    # Parse CUDA_VISIBLE_DEVICES from run_cmd if present
    if gpu_selection:
        env["CUDA_VISIBLE_DEVICES"] = gpu_selection
    env["EXPERIMENT_DIR"] = experiment_dir_abs
    
    # Execute without shell=True to prevent process duplication
    # Split the command properly, handling the surrogate wrapper
    import shlex
    cmd_to_run = train_command_explicit
    if "PYKT_TARGET_SCRIPT=" in train_command_explicit:
        # Extract env var and command
        parts = train_command_explicit.split(" ", 1)
        target_script_part = parts[0]  # PYKT_TARGET_SCRIPT=...
        env["PYKT_TARGET_SCRIPT"] = target_script_part.split("=", 1)[1]
        cmd_to_run = parts[1] if len(parts) > 1 else ""
    
    result = subprocess.run(shlex.split(cmd_to_run), cwd=PROJECT_ROOT / "examples", env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Training failed for fold {fold}")
        if result.stdout:
            print(f"Training output:\n{result.stdout[-2000:]}")
        if result.stderr:
            print(f"Training errors:\n{result.stderr[-2000:]}")
        return False, {}

    # 6. Execute Evaluation
    print(f"  Evaluating Fold {fold}...")
    
    # Parse eval command similarly
    eval_result = subprocess.run(shlex.split(eval_command_explicit), cwd=PROJECT_ROOT / "examples", env=env, check=False, capture_output=True, text=True)
    
    # 7. Collect Results
    metrics = {}
    eval_file = experiment_folder / "eval_results.json"
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"  ⚠️ Could not read metrics: {e}")
    
    # Interpretability & Plots (Only for iDKT or if requested)
    if model_name == 'idkt' and not args.dry_run:
        print(f"  Running comprehensive iDKT analysis...")
        analysis_cmd = f"{python_path} examples/generate_idkt_full_analysis.py --experiment_dir {experiment_dir_abs}"
        analysis_result = subprocess.run(analysis_cmd, shell=True)
        if analysis_result.returncode != 0:
            print(f"  ⚠️ Analysis pipeline encountered errors (non-fatal)")

    return True, metrics


def main():
    # PRE-FLIGHT: Run parameter audit before processing any arguments
    skip_audit = os.environ.get('SKIP_PARAMETER_AUDIT', '0') == '1'
    
    if not skip_audit:
        if not run_parameter_audit():
            print("\n" + "=" * 80)
            print("LAUNCH ABORTED - Fix reproducibility issues first")
            print("=" * 80)
            sys.exit(1)
        print() 
    else:
        print("⚠️  WARNING: Parameter audit SKIPPED")
        print()
    
    # First, load defaults to know what parameters are available
    defaults_path = Path(__file__).resolve().parent.parent / "configs" / "parameter_default.json"
    if not defaults_path.exists():
        print(f"❌ ERROR: Defaults file not found: {defaults_path}")
        sys.exit(1)
    
    with open(defaults_path, 'r') as f:
        defaults_config = json.load(f)
    
    available_params = defaults_config["defaults"]
    
    # Create parser with known arguments
    parser = argparse.ArgumentParser(description='Launch training or reproduce experiment (simplified)')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--repro_experiment_id", type=str, help="ID of experiment to reproduce")
    
    # New flags
    parser.add_argument("--cv", action="store_true", help="Run 5-fold Cross Validation")
    
    # Core identity
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--short_title", type=str, help="Short descriptive title for folder name")
    parser.add_argument("--fold", type=int, help="Specific fold (0-4) for single run")
    
    # Execution control
    parser.add_argument("--train_script", type=str, help="Override training script")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force_id", type=str, help="Force a specific experiment ID")
    parser.add_argument("--parent_folder", type=str, help="Parent folder for nesting")
    parser.add_argument('--skip_roster', action='store_true', help='Skip expensive roster CSV export')

    # Dynamically add arguments for ALL parameters in defaults
    for param_name, default_value in available_params.items():
        if param_name in ['train_script', 'model', 'dataset', 'fold', 'eval_script', 'short_title']:
            continue
        if isinstance(default_value, bool):
            if default_value:
                parser.add_argument(f'--{param_name}', action='store_false', 
                                   dest=param_name, help=f'Disable {param_name}')
                parser.add_argument(f'--no_{param_name}', action='store_false',
                                   dest=param_name, help=argparse.SUPPRESS)
            else:
                parser.add_argument(f'--{param_name}', action='store_true',
                                   dest=param_name, help=f'Enable {param_name}')
        elif isinstance(default_value, int):
            parser.add_argument(f'--{param_name}', type=int, default=None)
        elif isinstance(default_value, float):
            parser.add_argument(f'--{param_name}', type=float, default=None)
        else:
            parser.add_argument(f'--{param_name}', type=str, default=None)
    
    args = parser.parse_args()

    # REPRODUCTION MODE
    if args.repro_experiment_id:
        reproduce_experiment(args.repro_experiment_id, args.num_gpus, args.dry_run)
        return

    # TRAINING MODE
    try:
        default_dataset = get_required_param(defaults_config, "defaults", "dataset")
        default_model = get_required_param(defaults_config, "defaults", "model")
        default_short_title = get_required_param(defaults_config, "defaults", "short_title")
        default_fold = get_required_param(defaults_config, "defaults", "fold")
    except ValueError as e:
        print(f"❌ Config Error: {e}")
        sys.exit(1)

    dataset = args.dataset if args.dataset is not None else default_dataset
    model_name = args.model_name if args.model_name is not None else default_model
    short_title = args.short_title if args.short_title is not None else default_short_title
    
    experiment_id = args.force_id if args.force_id else generate_experiment_id()
    
    if args.cv:
        # Cross Validation Mode
        print(f"\n{('='*80)}")
        print(f"🚀 LAUNCHING 5-FOLD CROSS VALIDATION")
        print(f"Model: {model_name}, Dataset: {dataset}")
        print(f"ID: {experiment_id}")
        print(f"{('='*80)}\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_folder_name = f"{timestamp}_{model_name}_{short_title}_CV_{experiment_id}"
        cv_root = Path("experiments") / parent_folder_name
        cv_root.mkdir(parents=True, exist_ok=True)
        print(f"📂 CV Root: {cv_root}")
        
        results = []
        for f in range(5):
            print(f"\n--- Fold {f}/4 ---")
            success, fold_metrics = run_train_fold(
                args, defaults_config, f, model_name, dataset, short_title, experiment_id, parent_folder=cv_root
            )
            if success:
                results.append({
                    "fold": f,
                    "metrics": fold_metrics
                })
        
        print(f"\n{('='*80)}")
        print("📊 CROSS VALIDATION RESULTS")
        print(f"{('='*80)}")
        
        aucs = [r["metrics"].get("test_auc", 0) for r in results if r.get("metrics")]
        accs = [r["metrics"].get("test_acc", 0) for r in results if r.get("metrics")]
        
        mean_auc = sum(aucs)/len(aucs) if aucs else 0
        std_auc = (sum([(x - mean_auc)**2 for x in aucs]) / len(aucs))**0.5 if aucs else 0
        mean_acc = sum(accs)/len(accs) if accs else 0
        std_acc = (sum([(x - mean_acc)**2 for x in accs]) / len(accs))**0.5 if accs else 0
        
        agg_results = {
            "model": model_name,
            "dataset": dataset,
            "experiment_id": experiment_id,
            "folds_completed": len(results),
            "fold_results": results,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_acc": mean_acc,
            "std_acc": std_acc
        }
        
        print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Mean ACC: {mean_acc:.4f} ± {std_acc:.4f}")
        
        with open(cv_root / "cv_results.json", "w") as f:
            json.dump(agg_results, f, indent=4)
        print(f"\nSaved full CV results to: {cv_root}/cv_results.json")

    else:
        # Single Fold Mode
        fold = args.fold if args.fold is not None else default_fold
        print(f"\n🚀 LAUNCHING SINGLE RUN (Fold {fold})")
        run_train_fold(
            args, defaults_config, fold, model_name, dataset, short_title, experiment_id, parent_folder=args.parent_folder
        )

if __name__ == "__main__":
    main()
