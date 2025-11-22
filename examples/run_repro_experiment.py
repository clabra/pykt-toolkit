#!/usr/bin/env python3
"""
Simplified reproducible experiment launcher with integrated reproduction mode.

TRAINING MODE (default):
    Creates new experiment with 6-digit ID
    python examples/run_repro_experiment.py \
        --train_script examples/train_gainakt2exp.py \
        --model_name gainakt2exp \
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

def find_experiment_folder(experiment_id, base_dir="examples/experiments"):
    """Find experiment folder containing the given experiment ID."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # Search for folder containing the experiment_id (excluding _repro folders for original search)
    candidates = []
    for folder in base_path.iterdir():
        if folder.is_dir() and experiment_id in folder.name:
            # Prioritize non-repro folders
            if "_repro" not in folder.name:
                return folder
            candidates.append(folder)
    
    # If only repro folders found, return the first one
    return candidates[0] if candidates else None

def create_experiment_folder(model_name, short_title, experiment_id, is_repro=False):
    """Create experiment folder with naming convention: YYYYMMDD_HHMMSS_modelname_title_XXXXXX[_repro]"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    folder_name = f"{timestamp}_{model_name}_{short_title}_{experiment_id}"
    if is_repro:
        folder_name += "_repro"
    
    folder_path = Path("examples/experiments") / folder_name
    folder_path.mkdir(parents=True, exist_ok=False)
    
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

def build_explicit_train_command(train_script, params):
    """Build training command with all parameters explicit."""
    python_path = sys.executable
    cmd_parts = [python_path, train_script]
    
    # Launcher-only parameters (not passed to training script)
    launcher_only_params = {'model', 'train_script', 'eval_script', 'max_correlation_students'}
    
    # Add all parameters explicitly (exclude launcher-only params)
    for key, value in sorted(params.items()):
        if key in launcher_only_params:
            continue
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {value}")
    
    return " ".join(cmd_parts)

def build_trajectory_command(experiment_folder, num_students=10, min_steps=10):
    """
    Build command to extract and display learning trajectories.
    """
    python_path = sys.executable
    trajectory_script = "examples/learning_trajectories.py"
    cmd = f"{python_path} {trajectory_script} --run_dir {experiment_folder} --num_students {num_students} --min_steps {min_steps}"
    return cmd


def build_explicit_eval_command(eval_script, experiment_folder, params):
    """
    Build explicit evaluation command with ALL parameters.
    Similar to build_explicit_train_command but for evaluation.
    """
    python_path = sys.executable
    cmd_parts = [python_path, eval_script]
    
    # Add run_dir
    cmd_parts.append(f"--run_dir {experiment_folder}")
    
    # Evaluation-specific parameters (subset of training params)
    # Note: num_students is auto-detected from checkpoint, not passed as parameter
    eval_params = [
        'dataset', 'fold', 'batch_size',  # data
        'seq_len', 'd_model', 'n_heads', 'num_encoder_blocks', 'd_ff', 'dropout', 'emb_type',  # architecture
        'non_negative_loss_weight', 'monotonicity_loss_weight',  # constraints
        'mastery_performance_loss_weight', 'gain_performance_loss_weight',
        'sparsity_loss_weight', 'consistency_loss_weight',
        'bce_loss_weight',  # dual-encoder weights
        'mastery_threshold_init', 'threshold_temperature',  # interpretability parameters
        'monitor_freq'  # monitoring frequency needed by model
    ]
    
    # Boolean flags - ARCHITECTURAL AND INTERPRETABILITY MODES
    # IMPORTANT: These affect model architecture and MUST match between training and evaluation
    # - use_mastery_head: Enables mastery projection head
    # - use_gain_head: Enables gain projection head
    # - intrinsic_gain_attention: Uses attention-derived gains (changes architecture)
    bool_flags = ['use_mastery_head', 'use_gain_head', 'intrinsic_gain_attention']
    
    # Add max_correlation_students (default 300 if not in params)
    max_corr = params.get('max_correlation_students', 300)
    cmd_parts.append(f"--max_correlation_students {max_corr}")
    
    for key in eval_params:
        if key in params:
            value = params[key]
            cmd_parts.append(f"--{key} {value}")
    
    for key in bool_flags:
        if params.get(key, False):
            cmd_parts.append(f"--{key}")
    
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
    
    available_params = defaults_config.get("defaults", {})
    
    # Create parser with known arguments
    parser = argparse.ArgumentParser(description='Launch training or reproduce experiment (simplified)')
    
    # Common parameters
    parser.add_argument('--train_script', type=str, default=None,
                       help='Path to training script (default from config)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name for folder naming (default from config, ignored in reproduction mode)')
    parser.add_argument('--short_title', type=str,
                       help='Short title for experiment folder (required in training mode, ignored in reproduction mode)')
    
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
    
    # Dynamically add arguments for ALL parameters in defaults
    for param_name, default_value in available_params.items():
        # Skip parameters already defined above
        if param_name in ['train_script', 'model', 'dataset', 'fold', 'eval_script']:
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
        
        # Use train_explicit command for complete reproducibility (all params explicit)
        repro_dir_abs = str(repro_folder.absolute())
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
            print("\nNote: Evaluation is automatically launched by the training script")
            
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
        # Validate required parameter in training mode
        if not args.short_title:
            parser.error("--short_title is required in training mode")
        
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
        
        # Validate required parameter (short_title must be provided)
        if not args.short_title:
            print("❌ ERROR: --short_title is required in training mode")
            sys.exit(1)
        
        # Generate new experiment ID
        experiment_id = generate_experiment_id()
        print(f"Generated experiment ID: {experiment_id}")
        
        # Create experiment folder
        experiment_folder = create_experiment_folder(
            model_name=model_name,
            short_title=args.short_title,
            experiment_id=experiment_id,
            is_repro=False
        )
        print(f"✓ Created experiment folder: {experiment_folder.name}")
        
        # Build config from defaults
        # Get training defaults and add model
        training_params = defaults.get("defaults", {}).copy()
        training_params["model"] = model_name
        
        # Override dataset and fold (use resolved values, not args directly)
        training_params["dataset"] = dataset
        training_params["fold"] = fold
        
        # Process parameter overrides from command-line arguments
        # Check all parameters in defaults to see if they were overridden via CLI
        overrides = {}
        for param_name in defaults.get("defaults", {}).keys():
            # Skip special parameters already handled
            if param_name in ['train_script', 'eval_script', 'model', 'dataset', 'fold']:
                continue
            
            # Check if this parameter was provided via CLI
            arg_value = getattr(args, param_name, None)
            default_value = defaults.get("defaults", {}).get(param_name)
            
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
        
        # ARCHITECTURAL CONSTRAINT: Enforce mutual exclusivity before building commands
        # Intrinsic gain attention and projection heads are mutually exclusive
        if training_params.get('intrinsic_gain_attention', False):
            if training_params.get('use_mastery_head', False) or training_params.get('use_gain_head', False):
                print("\n" + "=" * 100)
                print("⚠️  LAUNCHER: ARCHITECTURAL PARAMETER CONFLICT DETECTED")
                print("=" * 100)
                print("intrinsic_gain_attention=True is INCOMPATIBLE with projection heads")
                print("")
                print("  Intrinsic mode uses attention-derived gains directly from the model.")
                print("  Projection heads (use_mastery_head, use_gain_head) are NOT used in this mode.")
                print("  Enabling them wastes ~2M parameters without any benefit.")
                print("")
                print("AUTOMATIC CORRECTION APPLIED IN CONFIG:")
                if training_params.get('use_mastery_head', False):
                    print("  • use_mastery_head: True → False")
                    training_params['use_mastery_head'] = False
                    # Update overrides if it was there
                    if 'use_mastery_head' in overrides:
                        overrides['use_mastery_head'] = False
                if training_params.get('use_gain_head', False):
                    print("  • use_gain_head: True → False")
                    training_params['use_gain_head'] = False
                    # Update overrides if it was there
                    if 'use_gain_head' in overrides:
                        overrides['use_gain_head'] = False
                print("")
                print("All generated commands will reflect corrected parameters (heads disabled).")
                print("Expected model parameters: ~12.7M (vs ~14.7M with unused projection heads)")
                print("=" * 100 + "\n")
        
        # Build commands
        experiment_dir_abs = str(experiment_folder.absolute())
        
        train_command_explicit = build_explicit_train_command(train_script, training_params)
        eval_command_explicit = build_explicit_eval_command(eval_script, experiment_dir_abs, training_params)
        trajectory_command = build_trajectory_command(experiment_dir_abs, num_students=10, min_steps=10)
        repro_command = build_repro_command(sys.argv[0], experiment_id)
        
        # Build new config structure - NO redundant typed sections
        # Build config with logical section order: input, commands, experiment, seeds, then reference data
        config = {
            "input": input_params,
            "commands": {
                "run_repro_original": original_command,
                "train_explicit": train_command_explicit,
                "eval_explicit": eval_command_explicit,
                "learning_trajectories": trajectory_command,
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
            "defaults": defaults.get("defaults", {}),  # Pristine copy of training_defaults from parameter_default.json
            "overrides": {k: v for k, v in training_params.items() if k in defaults.get("defaults", {}) and training_params[k] != defaults["defaults"][k]},
            "types": defaults.get("types", {}),
            "md5": defaults.get("md5", ""),  # MD5 of original defaults from parameter_default.json
            "reference": {
                "parameter_default_json": "configs/parameter_default.json"
            }
        }
        
        # Save config (MD5 verification will happen on load)
        save_config(config, experiment_folder)
        print("✓ Saved config.json")
        
        # Verify config integrity immediately after saving
        verify_config_md5(config)
        
        # Use train_explicit command for complete reproducibility (all params explicit)
        experiment_dir_abs = str(experiment_folder.absolute())
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
            print("\nNote: Evaluation is automatically launched by the training script")
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
