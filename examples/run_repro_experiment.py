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
        folder_path = Path("experiments") / folder_name
    
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
    launcher_only_params = {'model', 'train_script', 'eval_script', 'max_correlation_students', 'short_title'}
    
    # Canonical parameter groups
    architecture_params = {'seq_len', 'd_model', 'n_heads', 'n_blocks', 'd_ff', 'dropout', 'emb_type'}
    runtime_params = {'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'optimizer', 'gradient_clip', 'patience', 'seed'}
    
    # Model-specific script-to-parameter mapping
    MODEL_SCRIPTS = {
        "dkt": "examples/wandb_dkt_train.py",
        "dkt+": "examples/wandb_dkt_plus_train.py",
        "dkt_forget": "examples/wandb_dkt_forget_train.py",
        "kqn": "examples/wandb_kqn_train.py",
        "dkvmn": "examples/wandb_dkvmn_train.py",
        "atkt": "examples/wandb_atkt_train.py",
        "gkt": "examples/wandb_gkt_train.py",
        "sakt": "examples/wandb_sakt_train.py",
        "saint": "examples/wandb_saint_train.py",
        "saint++": "examples/wandb_saint_plus_plus_train.py",
        "akt": "examples/wandb_akt_train.py",
        "idkt": "examples/train_idkt.py",
        "lpkt": "examples/wandb_lpkt_train.py",
        "skvmn": "examples/wandb_skvmn_train.py",
        "deep_irt": "examples/wandb_deep_irt_train.py"
    }

    # Parameter Translation Map (Canonical -> Script Specific)
    PARAM_MAP = {
        "d_model": {
            "dkt": "emb_size", "dkt+": "emb_size", "dkt_forget": "emb_size",
            "kqn": "emb_size", "dkvmn": "dim_s", "gkt": "hidden_dim",
            "sakt": "emb_size", "skvmn": "dim_s", "deep_irt": "dim_s"
        },
        "n_heads": {
            "akt": "num_attn_heads", "sakt": "num_attn_heads"
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

    model = params.get('model', 'idkt')
    is_standard_pykt = "wandb_" in train_script

    # Determine which parameters to pass based on training script
    if 'train_idkt.py' in train_script:
        allowed_params = {
            'dataset', 'fold', 'seed', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 
            'optimizer', 'gradient_clip', 'patience', 'seq_len', 'd_model', 'n_heads', 'n_blocks', 
            'd_ff', 'dropout', 'emb_type', 'final_fc_dim', 'l2', 'lambda_student', 'lambda_gap', 
            'lambda_ref', 'lambda_initmastery', 'lambda_rate', 'theory_guided', 'calibrate',
            'bkt_filter', 'bkt_guess_threshold', 'bkt_slip_threshold', 'grounded_init', 'use_wandb',
            'save_dir', '_doc_grounding', '_doc_regularization'
        }
    elif is_standard_pykt:
        # Minimal set for standard pykt scripts (dataset, fold, seed, lr, dropout, wandb, uuid, save_dir)
        allowed_params = {'dataset', 'fold', 'seed', 'learning_rate', 'dropout', 'use_wandb', 'add_uuid', 'save_dir'}
        # Plus architecture if supported by that specific script
        if model in ['akt', 'sakt', 'saint', 'saint++']:
            allowed_params.update({'d_model', 'n_heads', 'n_blocks', 'd_ff'})
        elif model in ['dkt', 'dkt+', 'dkvmn', 'kqn', 'gkt', 'skvmn', 'deep_irt']:
            allowed_params.update({'d_model'})
    else:
        # Default safety: only pass runtime basics
        allowed_params = {'dataset', 'fold', 'seed', 'epochs', 'batch_size', 'learning_rate', 'save_dir'}
    
    # Add all parameters explicitly (exclude launcher-only params and model-specific filtering)
    for key, value in sorted(params.items()):
        if key in launcher_only_params:
            continue
        if allowed_params is not None and key not in allowed_params:
            continue  # Skip parameters not accepted by this training script
        
        # Translate key if necessary
        translated_key = key
        if key in PARAM_MAP:
            if model in PARAM_MAP[key]:
                translated_key = PARAM_MAP[key][model]
            elif is_standard_pykt and "standard_pykt" in PARAM_MAP[key]:
                translated_key = PARAM_MAP[key]["standard_pykt"]

        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{translated_key}")
        elif value is None or value == "null":
            # For reproducibility: pass "null" as string literal (required=True compliance)
            # Special case: construct rasch_path dynamically
            if key == "rasch_path" and "dataset" in params:
                rasch_path = f"data/{params['dataset']}/rasch_targets.pkl"
                cmd_parts.append(f"--{translated_key} {rasch_path}")
            else:
                # Pass "null" as string to satisfy required=True
                cmd_parts.append(f"--{translated_key} null")
        elif value == "None":
            # String "None" should be passed as "null" for consistency
            cmd_parts.append(f"--{translated_key} null")
        else:
            # Quote string values if they contain spaces or special characters
            val_str = str(value)
            if isinstance(value, str):
                # Ensure the value is properly quoted for shell execution
                cmd_parts.append(f"--{translated_key} '{val_str}'")
            else:
                cmd_parts.append(f"--{translated_key} {val_str}")
    
    # Add save_dir for all models to ensure they save results into the experiment folder
    if experiment_dir:
        cmd_parts.append(f"--save_dir {experiment_dir}")
    
    command = " ".join(cmd_parts)
    
    # Standard PyKT scripts (wandb_*.py) must be run from examples/ 
    # to find '../configs/'. We prepend 'cd examples &&' to ensure this.
    if "wandb_" in train_script:
        command = f"cd examples && {command}"
        
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
        predict_script = "wandb_predict.py"
        
        cmd_parts = ["cd examples &&", python_path, predict_script]
        cmd_parts.append(f"--save_dir {experiment_folder}")
        cmd_parts.append(f"--bz {params['batch_size']}")
        cmd_parts.append(f"--use_wandb 0")
        
        # Hardcode fusion_type to match scientific alignment (no fusion)
        cmd_parts.append("--fusion_type none")
        
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


def main():
    # PRE-FLIGHT: Run parameter audit before processing any arguments
    # This ensures reproducibility infrastructure is sound before launching experiments
    skip_audit = os.environ.get('SKIP_PARAMETER_AUDIT', '0') == '1'
    
    if not skip_audit:
        if not run_parameter_audit():
            print("\n" + "=" * 80)
            print("LAUNCH ABORTED - Fix reproducibility issues first")
            print("=" * 80)
            sys.exit(1)
        print()  # Extra newline for readability
    else:
        print("⚠️  WARNING: Parameter audit SKIPPED (SKIP_PARAMETER_AUDIT=1)")
        print("Reproducibility guarantees may be compromised!")
        print()
    
    # First, load defaults to know what parameters are available
    defaults_path = Path(__file__).parent.parent / "configs" / "parameter_default.json"
    if not defaults_path.exists():
        print(f"❌ ERROR: Defaults file not found: {defaults_path}")
        sys.exit(1)
    
    with open(defaults_path, 'r') as f:
        defaults_config = json.load(f)
    
    available_params = defaults_config["defaults"]
    
    # Create parser with known arguments
    parser = argparse.ArgumentParser(description='Launch training or reproduce experiment (simplified)')
    
    # Common parameters
    parser.add_argument('--train_script', type=str, default=None,
                       help='Path to training script (default from config)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name for folder naming (default from config, ignored in reproduction mode)')
    parser.add_argument('--short_title', type=str, default=None,
                       help='Short title for experiment folder (default: from parameter_default.json, ignored in reproduction mode)')
    parser.add_argument('--parent_folder', type=str, default=None,
                       help='Optional parent directory to group experiments (relative to project root)')
    parser.add_argument('--force_id', type=str, default=None,
                       help='Force a specific 6-digit ID for a new experiment')
    
    # Reproduction mode - single parameter
    parser.add_argument('--repro_experiment_id', type=str,
                       help='6-digit experiment ID to reproduce. If provided, activates reproduction mode and ALL other parameters are ignored.')
    
    # Training parameters (only used in training mode, ignored in reproduction)
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (default from config, ignored if --repro_experiment_id is set)')
    parser.add_argument('--fold', type=int, default=None,
                       help='Dataset fold (default from config, ignored if --repro_experiment_id is set)')
    
    # Runtime
    parser.add_argument('--dry_run', action='store_true',
                       help='Create config but do not train')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use (default: ~80%% of available, e.g., 6 out of 8)')
    parser.add_argument('--skip_roster', action='store_true',
                       help='Skip expensive roster CSV export during interpretability evaluation')
    
    # Dynamically add arguments for ALL parameters in defaults
    for param_name, default_value in available_params.items():
        # Skip parameters already defined above
        if param_name in ['train_script', 'model', 'dataset', 'fold', 'eval_script', 'short_title']:
            continue
        
        # Determine argument type based on default value
        if isinstance(default_value, bool):
            # For boolean parameters, use store_true/store_false
            if default_value:
                parser.add_argument(f'--{param_name}', action='store_false', 
                                   dest=param_name,
                                   help=f'Disable {param_name} (default: True)')
                parser.add_argument(f'--no_{param_name}', action='store_false',
                                   dest=param_name,
                                   help=argparse.SUPPRESS)
            else:
                parser.add_argument(f'--{param_name}', action='store_true',
                                   dest=param_name,
                                   help=f'Enable {param_name} (default: False)')
        elif isinstance(default_value, int):
            parser.add_argument(f'--{param_name}', type=int, default=None,
                               help=f'{param_name} (default: {default_value})')
        elif isinstance(default_value, float):
            parser.add_argument(f'--{param_name}', type=float, default=None,
                               help=f'{param_name} (default: {default_value})')
        else:
            parser.add_argument(f'--{param_name}', type=str, default=None,
                               help=f'{param_name} (default: {default_value})')
    
    args = parser.parse_args()
    
    # ========================================
    # REPRODUCTION MODE
    # ========================================
    if args.repro_experiment_id:
        # Load defaults to check for non-default values
        defaults_path = Path(__file__).parent.parent / "configs" / "parameter_default.json"
        if not defaults_path.exists():
            print(f"❌ ERROR: Defaults file not found: {defaults_path}")
            sys.exit(1)
        
        with open(defaults_path, 'r') as f:
            defaults = json.load(f)
        
        # Get default values (no hardcoded fallbacks)
        try:
            default_dataset = get_required_param(defaults, "defaults", "dataset")
            default_fold = get_required_param(defaults, "defaults", "fold")
            default_model = get_required_param(defaults, "defaults", "model")
            default_train_script = get_required_param(defaults, "defaults", "train_script")
        except ValueError as e:
            print("❌ ERROR: Missing required parameter in parameter_default.json")
            print(f"   {str(e)}")
            sys.exit(1)
        
        # Warn about ALL ignored parameters (everything except repro_experiment_id)
        ignored_params = []
        if args.short_title:
            ignored_params.append(f"--short_title {args.short_title}")
        if args.dataset is not None and args.dataset != default_dataset:
            ignored_params.append(f"--dataset {args.dataset}")
        if args.fold is not None and args.fold != default_fold:
            ignored_params.append(f"--fold {args.fold}")
        if args.model_name is not None and args.model_name != default_model:
            ignored_params.append(f"--model_name {args.model_name}")
        if args.train_script is not None and args.train_script != default_train_script:
            ignored_params.append(f"--train_script {args.train_script}")
        
        if ignored_params:
            print("⚠️  WARNING: Reproduction mode activated. The following parameters will be IGNORED:")
            for param in ignored_params:
                print(f"    - {param}")
            print("    ALL parameters will be read from the original experiment's config.json")
            print()
        
        print("=" * 80)
        print("REPRODUCTION MODE")
        print("=" * 80)
        print(f"Searching for experiment ID: {args.repro_experiment_id}")
        
        # Load original config unchanged
        config, original_folder = load_config_from_experiment(args.repro_experiment_id)
        print(f"✓ Found original experiment: {original_folder.name}")
        print("✓ Loaded config.json")
        
        # Extract model_name and short_title from original config (no hardcoded fallbacks)
        try:
            if 'input' in config:
                # New structure: check input first, then defaults
                model_name = get_required_param(config, 'defaults', 'model')
                # short_title is always in input (it's required)
                if 'short_title' not in config['input']:
                    raise ValueError("'short_title' missing from input section in config")
                original_short_title = config['input']['short_title']
            else:
                # Backward compatibility with old config structure
                if 'defaults' in config and 'model' in config['defaults']:
                    model_name = config['defaults']['model']
                elif 'training' in config and 'model' in config['training']:
                    model_name = config['training']['model']
                elif 'experiment' in config and 'model' in config['experiment']:
                    model_name = config['experiment']['model']
                else:
                    raise ValueError("'model' parameter not found in config (checked: defaults, training, experiment)")
                
                if 'experiment' in config and 'short_title' in config['experiment']:
                    original_short_title = config['experiment']['short_title']
                elif 'input' in config and 'short_title' in config['input']:
                    original_short_title = config['input']['short_title']
                else:
                    raise ValueError("'short_title' not found in config (checked: experiment, input)")
        except ValueError as e:
            print(f"❌ ERROR: Invalid config structure in experiment {args.repro_experiment_id}")
            print(f"   {str(e)}")
            sys.exit(1)
        
        # Use original short_title (is_repro=True will add '_repro' suffix automatically)
        repro_short_title = original_short_title
        
        # Create reproduction experiment folder with same ID + _repro suffix
        repro_folder = create_experiment_folder(
            model_name=model_name,
            short_title=repro_short_title,
            experiment_id=args.repro_experiment_id,
            is_repro=True
        )
        print(f"✓ Created reproduction folder: {repro_folder.name}")
        
        # Copy config.json unchanged to reproduction folder
        save_config(config, repro_folder)
        print("✓ Copied config.json (unchanged)")
        
        # Select GPUs for reproduction
        gpu_selection = select_gpus(args.num_gpus)
        
        # Use train_explicit command for complete reproducibility (all params explicit)
        repro_dir_abs = str(repro_folder.absolute())
        if gpu_selection:
            train_command = f"CUDA_VISIBLE_DEVICES={gpu_selection} EXPERIMENT_DIR={repro_dir_abs} {config['commands']['train_explicit']}"
        else:
            train_command = f"EXPERIMENT_DIR={repro_dir_abs} {config['commands']['train_explicit']}"
        
        print(f"\n{'Original experiment:':<25} {original_folder.name}")
        print(f"{'Reproduction folder:':<25} {repro_folder.name}")
        print(f"\n{'Training command:':<25}")
        print(f"  {train_command}")
        
        if args.dry_run:
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
            # Update experiment dir for reproduction folder (use absolute paths for robust replacement)
            original_dir_abs = str(original_folder.absolute())
            eval_command_explicit = eval_command_explicit.replace(original_dir_abs, repro_dir_abs)
            eval_command_full = f"EXPERIMENT_DIR={repro_dir_abs} {eval_command_explicit}"
            
            eval_result = subprocess.run(eval_command_full, shell=True)
            
            if eval_result.returncode == 0:
                print("\n✓ Evaluation completed successfully")
                print(f"  Results saved to: {repro_folder}/eval_results.json")
            elif eval_result.returncode == 1:
                # Exit code 1 typically means evaluation ran but success criteria not met
                print("\n✓ Evaluation completed (success criteria not met)")
                print(f"  Results saved to: {repro_folder}/eval_results.json")
                print("  Check eval_results.json for detailed metrics")
            else:
                print("\n⚠️  Evaluation failed (non-critical)")
                print("  You can run it manually:")
                print(f"    {eval_command_full}")
            
            # Suggest comparison with external script
            print("\n" + "=" * 80)
            print("VERIFY REPRODUCIBILITY")
            print("=" * 80)
            print("\nTo compare results with the original experiment, run:")
            print(f"\n  python examples/compare_reproduction.py {args.repro_experiment_id}")
            print("\nOr specify both folders explicitly:")
            print(f"\n  python examples/compare_reproduction.py \\")
            print(f"    {original_folder} \\")
            print(f"    {repro_folder}")
            print("\n" + "=" * 80)
            
        else:
            print("\n" + "=" * 80)
            print("❌ REPRODUCTION TRAINING FAILED")
            print("=" * 80)
            sys.exit(result.returncode)
    
    # ========================================
    # TRAINING MODE
    # ========================================
    else:
        print("=" * 80)
        print("TRAINING MODE")
        print("=" * 80)
        
        # Load default parameters
        defaults_path = Path("configs/parameter_default.json")
        if not defaults_path.exists():
            raise FileNotFoundError(f"Defaults file not found: {defaults_path}")
        
        with open(defaults_path, 'r') as f:
            defaults = json.load(f)
        
        # Get required parameters from defaults (no hardcoded fallbacks)
        try:
            default_dataset = get_required_param(defaults, "defaults", "dataset")
            default_fold = get_required_param(defaults, "defaults", "fold")
            default_model = get_required_param(defaults, "defaults", "model")
            default_train_script = get_required_param(defaults, "defaults", "train_script")
            default_eval_script = get_required_param(defaults, "defaults", "eval_script")
            default_short_title = get_required_param(defaults, "defaults", "short_title")
        except ValueError as e:
            print("❌ ERROR: Missing required parameter in parameter_default.json")
            print(f"   {str(e)}")
            sys.exit(1)
        
        # Use CLI arguments if provided, otherwise use defaults from config
        dataset = args.dataset if args.dataset is not None else default_dataset
        fold = args.fold if args.fold is not None else default_fold
        model_name = args.model_name if args.model_name is not None else default_model
        train_script = args.train_script if args.train_script is not None else default_train_script
        eval_script = default_eval_script  # eval_script has no CLI argument
        short_title = args.short_title if args.short_title is not None else default_short_title
        
        # Generate new experiment ID
        if args.force_id:
            experiment_id = args.force_id
            print(f"Using forced experiment ID: {experiment_id}")
        else:
            experiment_id = generate_experiment_id()
            print(f"Generated experiment ID: {experiment_id}")
        
        # Create experiment folder
        experiment_folder = create_experiment_folder(
            model_name=model_name,
            short_title=short_title,
            experiment_id=experiment_id,
            is_repro=False,
            parent_folder=args.parent_folder,
            fold=fold
        )
        print(f"✓ Created experiment folder: {experiment_folder.name}")
        
        # Build config from defaults
        # Get training defaults and add model
        training_params = defaults["defaults"].copy()
        training_params["model"] = model_name
        
        # Override dataset and fold (use resolved values, not args directly)
        training_params["dataset"] = dataset
        training_params["fold"] = fold
        
        # Process parameter overrides from command-line arguments
        # Check all parameters in defaults to see if they were overridden via CLI
        overrides = {}
        for param_name in defaults["defaults"].keys():
            # Skip special parameters already handled
            if param_name in ['train_script', 'eval_script', 'model', 'dataset', 'fold']:
                continue
            
            # Check if this parameter was provided via CLI
            arg_value = getattr(args, param_name, None)
            default_value = defaults["defaults"][param_name]
            
            # For boolean parameters, argparse sets them to True/False directly
            # For others, they're None if not provided
            if isinstance(default_value, bool):
                # Boolean was explicitly set if it differs from default
                if arg_value != default_value:
                    overrides[param_name] = arg_value
                    training_params[param_name] = arg_value
                    print(f"  Override: {param_name} = {arg_value}")
            elif arg_value is not None:
                # Non-boolean parameter was explicitly provided
                overrides[param_name] = arg_value
                training_params[param_name] = arg_value
                print(f"  Override: {param_name} = {arg_value} (type: {type(arg_value).__name__})")
        
        # Build input section with ONLY explicitly provided parameters
        input_params = {
            "short_title": args.short_title
        }
        
        # Add to input only if different from default
        if args.dataset is not None and args.dataset != default_dataset:
            input_params["dataset"] = args.dataset
        if args.fold is not None and args.fold != default_fold:
            input_params["fold"] = args.fold
        if args.model_name is not None and args.model_name != default_model:
            input_params["model"] = args.model_name
        if args.train_script is not None and args.train_script != default_train_script:
            input_params["train_script"] = args.train_script
        
        # Add all overrides to input section
        for param_name, param_value in overrides.items():
            input_params[param_name] = param_value
        
        # Build the original run_repro command
        original_command_parts = [sys.executable, sys.argv[0]]
        if args.short_title:
            original_command_parts.extend(["--short_title", args.short_title])
        if args.dataset is not None and args.dataset != default_dataset:
            original_command_parts.extend(["--dataset", args.dataset])
        if args.fold is not None and args.fold != default_fold:
            original_command_parts.extend(["--fold", str(args.fold)])
        if args.model_name is not None and args.model_name != default_model:
            original_command_parts.extend(["--model_name", args.model_name])
        if args.train_script is not None and args.train_script != default_train_script:
            original_command_parts.extend(["--train_script", args.train_script])
        # Add all override parameters to the command (using direct --param value syntax)
        for param_name, param_value in overrides.items():
            original_command_parts.extend([f"--{param_name}", str(param_value)])
        original_command = " ".join(original_command_parts)
        
        # Build commands
        experiment_dir_abs = str(experiment_folder.absolute())
        
        # Select GPUs (unless user already set CUDA_VISIBLE_DEVICES)
        gpu_selection = select_gpus(args.num_gpus)
        
        train_command_explicit = build_explicit_train_command(train_script, training_params, experiment_dir_abs)
        eval_command_explicit = build_explicit_eval_command(eval_script, experiment_dir_abs, training_params)
        mastery_states_command = build_mastery_states_command(experiment_dir_abs, num_students=15, split='test')
        
        # Interpretability correlation commands
        python_path = sys.executable
        head_agreement_command = (
            f"{python_path} examples/compute_head_agreement.py "
            f"--experiment_dir {experiment_dir_abs} "
            f"--update_csv"
        )
        difficulty_fidelity_command = (
            f"{python_path} examples/compute_irt_correlation.py "
            f"--experiment_dir {experiment_dir_abs} "
            f"--dataset {training_params['dataset']} "
            f"--update_csv"
        )
        bkt_correlation_command = (
            f"{python_path} examples/compute_bkt_correlation.py "
            f"--experiment_dir {experiment_dir_abs} "
            f"--dataset {training_params['dataset']} "
            f"--max_students {training_params['max_correlation_students']} "
            f"--output_file bkt_validation_final.json "
            f"--update_csv"
        )
        idkt_interpretability_command = (
            f"{python_path} examples/eval_idkt_interpretability.py "
            f"--checkpoint {experiment_dir_abs}/best_model.pt "
            f"--output_dir {experiment_dir_abs} "
            f"--dataset {training_params['dataset']} "
            f"--fold {training_params['fold']} "
            f"--batch_size 8 "
            f"--d_model {training_params['d_model']} "
            f"--n_heads {training_params['n_heads']} "
            f"--n_blocks {training_params['n_blocks']} "
            f"--d_ff {training_params['d_ff']} "
            f"--dropout {training_params['dropout']} "
            f"--final_fc_dim {training_params['final_fc_dim']} "
            f"--l2 {training_params['l2']} "
            f"--emb_type {training_params['emb_type']} "
            f"--seq_len {training_params['seq_len']} "
            f"--roster_sampling_rate 10 "
            f"--max_correlation_students {training_params['max_correlation_students']}"
        )
        if args.skip_roster:
            idkt_interpretability_command += " --skip_roster"
        
        bkt_validation_command = build_bkt_validation_command(experiment_dir_abs, training_params)
        repro_command = build_repro_command(sys.argv[0], experiment_id)
        
        # Build new config structure - NO redundant typed sections
        # Build config with logical section order: input, commands, experiment, seeds, then reference data
        config = {
            "input": input_params,
            "commands": {
                "run_repro_original": original_command,
                "train_explicit": train_command_explicit,
                "eval_explicit": eval_command_explicit,
                "mastery_states": mastery_states_command,
                "head_agreement": head_agreement_command,
                "difficulty_fidelity": difficulty_fidelity_command,
                "bkt_correlation": bkt_correlation_command,
                "bkt_validation": bkt_validation_command,
                "idkt_interpretability": idkt_interpretability_command,
                "analysis_plots": build_analysis_plots_command(experiment_dir_abs, training_params),
                "validation_plots": build_validation_plots_command(experiment_dir_abs),
                "plot_param_distribution": f"{python_path} examples/plot_param_distribution.py --run_dir {experiment_dir_abs}",
                "plot_mastery_mosaic": f"{python_path} examples/plot_mastery_mosaic_real.py --run_dir {experiment_dir_abs}",
                "train_probe": f"{python_path} examples/train_probe.py --checkpoint {experiment_dir_abs}/best_model.pt --bkt_preds {experiment_dir_abs}/traj_predictions.csv --dataset {training_params['dataset']} --output_dir {experiment_dir_abs} --debug",
                "reproduce": repro_command
            },
            "experiment": {
                "id": experiment_folder.name,
                "short_title": args.short_title,
                "experiment_id": experiment_id,
                "created": experiment_folder.name.split('_')[0] + "_" + experiment_folder.name.split('_')[1]
            },
            "seeds": {
                "primary": training_params["seed"],
                "all": [training_params["seed"]]
            },
            "defaults": defaults["defaults"],  # Pristine copy of training_defaults from parameter_default.json
            "overrides": {k: v for k, v in training_params.items() if k in defaults["defaults"] and training_params[k] != defaults["defaults"][k]},
            "types": defaults["types"],
            "md5": defaults["md5"],  # MD5 of original defaults from parameter_default.json
            "reference": {
                "parameter_default_json": "configs/parameter_default.json"
            }
        }
        
        # Save config (MD5 verification will happen on load)
        save_config(config, experiment_folder)
        print("✓ Saved config.json")
        
        # Verify config integrity immediately after saving
        verify_config_md5(config)
        
        # Select GPUs (unless user already set CUDA_VISIBLE_DEVICES)
        gpu_selection = select_gpus(args.num_gpus)
        
        # Use train_explicit command for complete reproducibility (all params explicit)
        experiment_dir_abs = str(experiment_folder.absolute())
        if gpu_selection:
            train_command = f"CUDA_VISIBLE_DEVICES={gpu_selection} EXPERIMENT_DIR={experiment_dir_abs} {config['commands']['train_explicit']}"
        else:
            train_command = f"EXPERIMENT_DIR={experiment_dir_abs} {config['commands']['train_explicit']}"
        
        print(f"\n{'Experiment ID:':<25} {experiment_id}")
        print(f"{'Dataset:':<25} {dataset}")
        print(f"{'Fold:':<25} {fold}")
        print(f"{'Epochs:':<25} {training_params['epochs']}")
        print(f"{'Batch size:':<25} {training_params['batch_size']}")
        print(f"{'Learning rate:':<25} {training_params['learning_rate']}")
        print(f"\n{'Training command:':<25}")
        print(f"  {train_command}")
        
        if args.dry_run:
            print("\n[DRY RUN] Skipping training execution")
            print("\nTo execute training, run:")
            print(f"  {train_command}")
            return
        
        # Launch training (automatic two-phase for iKT/iKT2 if phase=None, warm-up for iKT3, single-phase for idkt)
        if model_name in ['ikt', 'ikt2'] and training_params.get('phase') is None:
            print("\n" + "=" * 80)
            print(f"AUTOMATIC TWO-PHASE TRAINING ({model_name.upper()})")
            print("=" * 80)
            print("Phase 1 will run until convergence, then automatically switch to Phase 2")
            print("=" * 80 + "\n")
        elif model_name == 'ikt3':
            warmup_epochs = training_params.get('warmup_epochs', 50)
            lambda_target = training_params.get('lambda_target', 0.5)
            print("\n" + "=" * 80)
            print(f"SINGLE-PHASE TRAINING WITH WARM-UP (iKT3)")
            print("=" * 80)
            print(f"λ will gradually increase from 0 to {lambda_target} over {warmup_epochs} epochs")
            print("=" * 80 + "\n")
        elif model_name == 'idkt':
            print("\n" + "=" * 80)
            print(f"SINGLE-PHASE TRAINING (iDKT)")
            print("=" * 80)
            print("Standard training with Rasch-based difficulty regularization")
            print("=" * 80 + "\n")
        
        # Launch training
        print("\n" + "=" * 80)
        print("LAUNCHING TRAINING")
        print("=" * 80 + "\n")
        result = subprocess.run(train_command, shell=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("✓ TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"\nResults saved to: {experiment_folder}")
            
            # Launch evaluation
            print("\n" + "=" * 80)
            print("LAUNCHING EVALUATION")
            print("=" * 80 + "\n")
            
            eval_command_full = f"EXPERIMENT_DIR={experiment_dir_abs} {config['commands']['eval_explicit']}"
            eval_result = subprocess.run(eval_command_full, shell=True)
            
            if eval_result.returncode == 0:
                print("\n✓ Evaluation completed successfully")
                print(f"  Results saved to: {experiment_folder}/eval_results.json")
            elif eval_result.returncode == 1:
                # Exit code 1 typically means evaluation ran but success criteria not met
                print("\n✓ Evaluation completed (success criteria not met)")
                print(f"  Results saved to: {experiment_folder}/eval_results.json")
                print("  Check eval_results.json for detailed metrics")
            else:
                print("\n⚠️  Evaluation failed (non-critical)")
                print("  You can run it manually:")
                print(f"    {eval_command_full}")
            
            # Auto-launch mastery states extraction for legacy iKT models
            if model_name in ['ikt', 'ikt2', 'ikt3']:
                    print("\n" + "=" * 80)
                    print(f"LAUNCHING MASTERY STATES EXTRACTION ({model_name.upper()})")
                    print("=" * 80)
                    if model_name == 'ikt':
                        print("\nExtracting {Mi} skill trajectories for Rasch alignment analysis...")
                    elif model_name == 'ikt2':
                        print("\nExtracting IRT mastery states (θ, β, M_IRT) for interpretability analysis...")
                    elif model_name == 'ikt3':
                        ref_model = training_params.get('reference_model', 'irt')
                        print(f"\nExtracting {ref_model.upper()} alignment factors for interpretability analysis...")
                    
                    mastery_result = subprocess.run(mastery_states_command, shell=True)
                    
                    if mastery_result.returncode == 0:
                        print("\n✓ Mastery states extraction completed")
                        print(f"  Output files in: {experiment_folder}")
                        print(f"    - mastery_test.csv")
                        print(f"    - mastery_test.json")
                        
                        # Compute interpretability metrics for iKT2
                        if model_name == 'ikt2':
                            # Head Agreement (M_IRT vs p_correct) - PRIMARY interpretability metric
                            print("\n" + "=" * 80)
                            print("COMPUTING HEAD AGREEMENT (IRT Consistency)")
                            print("=" * 80)
                            print("\nComparing M_IRT mastery vs p_correct predictions...")
                            
                            head_result = subprocess.run(head_agreement_command, shell=True)
                            
                            if head_result.returncode == 0:
                                print("\n✓ Head agreement computed successfully")
                                print(f"  Output file: {experiment_folder}/head_agreement_test.json")
                                print(f"  Updated: {experiment_folder}/metrics_test.csv")
                            else:
                                print("\n⚠️  Head agreement failed (non-critical)")
                                print("  Note: Requires mastery_test.csv")
                            
                            # Difficulty Fidelity (β_learned vs β_IRT) - SECONDARY validation metric
                            print("\n" + "=" * 80)
                            print("COMPUTING DIFFICULTY FIDELITY")
                            print("=" * 80)
                            print("\nComparing learned difficulty (β) vs IRT-calibrated difficulty...")
                            
                            irt_result = subprocess.run(irt_command, shell=True)
                            
                            if irt_result.returncode == 0:
                                print("\n✓ Difficulty fidelity computed successfully")
                                print(f"  Output file: {experiment_folder}/irt_correlation_test.json")
                                print(f"  Updated: {experiment_folder}/metrics_test.csv")
                            else:
                                print("\n⚠️  Difficulty fidelity failed (non-critical)")
                                print("  Note: Requires rasch_targets.pkl and mastery_test.csv")
                            
                            # BKT correlation (M_IRT vs P(L_t))
                            print("\n" + "=" * 80)
                            print("COMPUTING BKT CORRELATION")
                            print("=" * 80)
                            print("\nComparing model mastery (M_IRT) vs BKT P(L_t)...")
                            
                            bkt_result = subprocess.run(bkt_command, shell=True)
                            
                            if bkt_result.returncode == 0:
                                print("\n✓ BKT correlation computed successfully")
                                print(f"  Output file: {experiment_folder}/bkt_validation_final.json")
                                print(f"  Updated: {experiment_folder}/metrics_test.csv")
                            else:
                                print("\n⚠️  BKT correlation failed (non-critical)")
                                print("  Note: Requires bkt_mastery_states.pkl and mastery_test.csv")
                                print(f"  Check: data/{training_params['dataset']}/bkt_mastery_states.pkl exists")
                                print(f"  Check: {experiment_folder}/mastery_test.csv exists")
                        
                        pass # Add common ikt metrics here if needed
                        
                        # Generate comprehensive analysis plots
                        print("\n" + "=" * 80)
                        print("Generating analysis plots...")
                        print("=" * 80)
                        
                        plots_command = build_analysis_plots_command(experiment_folder, training_params)
                        plots_result = subprocess.run(plots_command, shell=True)
                        
                        if plots_result.returncode == 0:
                            print("\n✓ Analysis plots generated successfully")
                            print(f"  Plot files in: {experiment_folder}/plots/")
                            print(f"    - loss_evolution.png")
                            print(f"    - auc_vs_violations.png")
                            print(f"    - deviation_histogram.png")
                            print(f"    - per_skill_alignment.png")
                        else:
                            print("\n⚠️  Plot generation failed (non-critical)")
                    else:
                        print("\n⚠️  Mastery states extraction failed (non-critical)")
            
            # Auto-launch interpretability alignment for iDKT
            if model_name == 'idkt':
                print("\n" + "=" * 80)
                print("COMPUTING iDKT INTERPRETABILITY ALIGNMENT")
                print("=" * 80)
                print("\nComparing model predictions and projections vs BKT ground truth...")
                
                idkt_inter_result = subprocess.run(idkt_interpretability_command, shell=True)
                
                if idkt_inter_result.returncode == 0:
                    print("\n✓ iDKT interpretability evaluation completed successfully")
                    print(f"  Output file: {experiment_folder}/interpretability_alignment.json")
                    
                    # Generate analysis plots for iDKT
                    print("\n" + "=" * 80)
                    print("Generating analysis plots...")
                    print("=" * 80)
                    
                    plots_command = build_analysis_plots_command(experiment_folder, training_params)
                    plots_result = subprocess.run(plots_command, shell=True)
                    
                    if plots_result.returncode == 0:
                        print("\n✓ Analysis plots generated successfully")
                    else:
                        print("\n⚠️  Plot generation failed (non-critical)")

                    # Generate validation plots for iDKT
                    print("\n" + "=" * 80)
                    print("Generating validation plots (Consensus, Residual, Uncertainty)...")
                    print("=" * 80)
                    
                    val_plots_command = build_validation_plots_command(experiment_folder)
                    val_plots_result = subprocess.run(val_plots_command, shell=True)
                    
                    if val_plots_result.returncode == 0:
                        print("\n✓ Validation plots generated successfully")
                    else:
                        print("\n⚠️  Validation plot generation failed (non-critical)")

                    # Generate Section 10+ interpretability plots for iDKT
                    print("\n" + "=" * 80)
                    print("Generating Section 10+ Interpretability Plots...")
                    print("=" * 80)
                    
                    sect10_plots = [
                        ("Param Distribution", config['commands']['plot_param_distribution']),
                        ("Mastery Mosaic", config['commands']['plot_mastery_mosaic']),
                        ("Diagnostic Probing", config['commands']['train_probe'])
                    ]
                    
                    for plot_name, plot_cmd in sect10_plots:
                        print(f"\n- Generating {plot_name}...")
                        p_res = subprocess.run(plot_cmd, shell=True)
                        if p_res.returncode == 0:
                            print(f"  ✓ {plot_name} generated successfully")
                        else:
                            print(f"  ⚠️  {plot_name} generation failed (non-critical)")
                else:
                    print("\n⚠️  iDKT interpretability evaluation failed (non-critical)")
            
            print("\nTo reproduce this experiment:")
            print("  python examples/run_repro_experiment.py \\")
            print(f"    --repro_experiment_id {experiment_id}")
        else:
            print("\n" + "=" * 80)
            print("❌ TRAINING FAILED")
            print("=" * 80)
            sys.exit(result.returncode)

if __name__ == "__main__":
    main()
