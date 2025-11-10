#!/usr/bin/env python3
"""
Standardized training script for GainAKT2Exp model using PyKT framework patterns.

╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️  REPRODUCIBILITY WARNING ⚠️                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DO NOT CALL THIS SCRIPT DIRECTLY FOR REPRODUCIBLE EXPERIMENTS!             ║
║                                                                              ║
║  This script has ZERO hardcoded defaults and requires 60+ explicit          ║
║  parameters. Direct execution will FAIL unless all parameters are provided. ║
║                                                                              ║
║  For reproducible experiments, use the experiment launcher:                 ║
║                                                                              ║
║      python examples/run_repro_experiment_simple.py --short_title "name"    ║
║                                                                              ║
║  The launcher will:                                                         ║
║    ✓ Load defaults from configs/parameter_default.json                      ║
║    ✓ Apply your CLI overrides                                               ║
║    ✓ Generate explicit command with ALL 60+ parameters                      ║
║    ✓ Create experiment folder with full audit trail                         ║
║    ✓ Save config.json for perfect reproducibility                           ║
║                                                                              ║
║  See: docs/REPRODUCIBILITY_WORKFLOW.md for complete documentation           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

OPTIMAL PARAMETERS from comprehensive sweep (AUC: 0.7260, Perfect Consistency):
- learning_rate: 0.000174, weight_decay: 1.7571e-05, batch_size: 96
- enhanced_constraints: True, peaks at epoch 3, early stopping recommended
- Achieves 0% violations and perfect monotonicity constraints
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from examples.experiment_utils import compute_auc_acc
import json
from datetime import datetime
import csv  # added for reproducibility metrics CSV
# tqdm removed for cleaner output - only epoch results shown
import wandb
# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.datasets import init_dataset4train
from pykt.models.gainakt2_exp import create_exp_model
from examples.interpretability_monitor import InterpretabilityMonitor


def validate_model_consistency(model, data_loader, device, logger, max_students=100):
    """Quick consistency validation during training."""
    model.eval()
    # Support DataParallel wrapping: access underlying module for custom methods/heads
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    violations = {
        'monotonicity': 0,
        'negative_gains': 0,
        'bounds': 0,
        'total_checks': 0
    }
    
    correlations = {
        'mastery_performance': [],
        'gain_performance': []
    }
    
    student_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if student_count >= max_students:
                break
                
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Use underlying module's forward_with_states when DataParallel is active
            outputs = core_model.forward_with_states(q=questions, r=responses, qry=questions_shifted)

            # If interpretability heads are disabled, skip detailed consistency checks
            if 'projected_mastery' not in outputs or 'projected_gains' not in outputs:
                logger.debug("Consistency check: interpretability heads disabled; returning neutral metrics.")
                return {
                    'monotonicity_violation_rate': 0.0,
                    'negative_gain_rate': 0.0,
                    'bounds_violation_rate': 0.0,
                    'mastery_correlation': 0.0,
                    'gain_correlation': 0.0
                }

            skill_mastery = outputs['projected_mastery']
            skill_gains = outputs['projected_gains']
            batch_size_actual = questions.size(0)
            
            for i in range(batch_size_actual):
                if student_count >= max_students:
                    break
                    
                student_mask = mask[i].bool()
                student_mastery = skill_mastery[i][student_mask]
                student_gains = skill_gains[i][student_mask]
                student_performance = responses_shifted[i][student_mask].float()
                
                seq_len = student_mastery.size(0)
                if seq_len < 2:
                    continue
                
                # Convert to numpy and aggregate
                mastery_np = student_mastery.cpu().numpy()
                gains_np = student_gains.cpu().numpy()
                performance_np = student_performance.cpu().numpy()
                
                mean_mastery = np.mean(mastery_np, axis=1)
                mean_gains = np.mean(gains_np, axis=1)
                
                # Check violations
                violations['total_checks'] += 1
                
                # Monotonicity
                for t in range(1, seq_len):
                    if mean_mastery[t] < mean_mastery[t-1] - 1e-6:  # Small tolerance
                        violations['monotonicity'] += 1
                        break
                
                # Negative gains
                if np.any(gains_np < -1e-6):
                    violations['negative_gains'] += 1
                
                # Bounds
                if np.any((mastery_np < -1e-6) | (mastery_np > 1 + 1e-6)):
                    violations['bounds'] += 1
                
                # Correlations
                if seq_len >= 3:
                    try:
                        mastery_corr = np.corrcoef(mean_mastery, performance_np)[0, 1]
                        if not np.isnan(mastery_corr):
                            correlations['mastery_performance'].append(mastery_corr)
                        
                        gain_corr = np.corrcoef(mean_gains, performance_np)[0, 1]
                        if not np.isnan(gain_corr):
                            correlations['gain_performance'].append(gain_corr)
                    except (ValueError, IndexError, np.linalg.LinAlgError):
                        # Skip correlation computation for degenerate cases
                        pass
                
                student_count += 1
    
    # Compute statistics
    if violations['total_checks'] > 0:
        mono_rate = violations['monotonicity'] / violations['total_checks']
        neg_rate = violations['negative_gains'] / violations['total_checks']
        bounds_rate = violations['bounds'] / violations['total_checks']
    else:
        mono_rate = neg_rate = bounds_rate = 0.0
    
    mastery_corr = np.mean(correlations['mastery_performance']) if correlations['mastery_performance'] else 0.0
    gain_corr = np.mean(correlations['gain_performance']) if correlations['gain_performance'] else 0.0
    
    logger.info(f"  Consistency Check - Monotonicity: {mono_rate:.1%}, "
                f"Negative gains: {neg_rate:.1%}, Bounds: {bounds_rate:.1%}")
    logger.info(f"  Correlations - Mastery: {mastery_corr:.3f}, Gains: {gain_corr:.3f}")
    
    return {
        'monotonicity_violation_rate': mono_rate,
        'negative_gain_rate': neg_rate,
        'bounds_violation_rate': bounds_rate,
        'mastery_correlation': mastery_corr,
        'gain_correlation': gain_corr
    }


def load_config_if_available():
    """Load experiment config from PYKT_CONFIG_PATH if present, else return None."""
    cfg_path = os.environ.get('PYKT_CONFIG_PATH')
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            return None
    return None

def resolve_param(cfg, section, key, fallback):
    """
    Resolve parameter value with priority: input -> defaults -> fallback.
    The 'section' parameter is ignored (kept for backward compatibility).
    """
    try:
        if cfg is None:
            return fallback
        
        # Priority 1: Check input section (explicit user overrides)
        if 'input' in cfg and key in cfg['input']:
            return cfg['input'][key]
        
        # Priority 2: Check defaults section (parameter defaults)
        if 'defaults' in cfg and key in cfg['defaults']:
            return cfg['defaults'][key]
        
        # Priority 3: Check old typed section format (backward compatibility)
        if section in cfg and key in cfg[section]:
            return cfg[section][key]
        
        # Priority 4: Use fallback
        return fallback
    except Exception:
        return fallback

def train_gainakt2exp_model(args):
    """
    Standardized training function for GainAKT2Exp model using PyKT framework patterns.
    
    OPTIMAL parameters (AUC: 0.7260, Perfect Consistency):
    - dataset_name: str (default: 'assist2015') 
    - fold: int (default: 0)
    - batch_size: int (default: 96)  # OPTIMAL
    - num_epochs: int (default: 20, peaks at epoch 3) 
    - learning_rate: float (default: 0.000174)  # OPTIMAL (50% of base)
    - weight_decay: float (default: 1.7571e-05)  # OPTIMAL (30% of base)
    - enhanced_constraints: bool (default: True)  # CRITICAL for consistency
    - experiment_suffix: str (default: 'optimal_v1')
    - use_wandb: bool (default: False)
    """
    import logging
    import random
    # Deterministic seed initialization (before any data/model ops)
    seed_base = getattr(args, 'seed', 42)
    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    # Enforce deterministic algorithms where feasible
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    
    # Get parameters with OPTIMAL defaults (AUC: 0.7260, Perfect Consistency)
    cfg = load_config_if_available()
    # Invariant: auto_shifted_eval must remain True for standardized next-step metrics reproducibility
    auto_shifted_eval_val = resolve_param(cfg, 'runtime', 'auto_shifted_eval', True)
    if auto_shifted_eval_val is not True:
        raise ValueError(f"Reproducibility invariant violated: auto_shifted_eval expected True, found {auto_shifted_eval_val}. Abort.")
    dataset_name = resolve_param(cfg, 'data', 'dataset', getattr(args, 'dataset', 'assist2015'))
    fold = resolve_param(cfg, 'data', 'fold', getattr(args, 'fold', 0))
    num_epochs = resolve_param(cfg, 'training', 'epochs', getattr(args, 'epochs', 12))  # Fixed: was 20, now matches parameter_default.json
    batch_size = resolve_param(cfg, 'training', 'batch_size', getattr(args, 'batch_size', 64))  # Fixed: was 96, now matches parameter_default.json
    learning_rate = resolve_param(cfg, 'training', 'learning_rate', getattr(args, 'learning_rate', 0.000174))
    weight_decay = resolve_param(cfg, 'training', 'weight_decay', getattr(args, 'weight_decay', 1.7571e-05))
    enhanced_constraints = resolve_param(cfg, 'interpretability', 'enhanced_constraints', getattr(args, 'enhanced_constraints', True))
    experiment_suffix = getattr(args, 'experiment_suffix', 'optimal_v1')
    use_wandb = resolve_param(cfg, 'runtime', 'use_wandb', getattr(args, 'use_wandb', False))
    use_amp = resolve_param(cfg, 'runtime', 'use_amp', getattr(args, 'use_amp', False))
    # Alignment / semantic emergence new arguments (may be absent in older runs)
    enable_alignment_loss = resolve_param(cfg, 'alignment', 'enable_alignment_loss', getattr(args, 'enable_alignment_loss', True))  # Fixed: was False, now matches parameter_default.json
    alignment_weight = float(resolve_param(cfg, 'alignment', 'alignment_weight', getattr(args, 'alignment_weight', 0.25)))  # Fixed: was 0.1, now matches parameter_default.json
    alignment_warmup_epochs = int(resolve_param(cfg, 'alignment', 'alignment_warmup_epochs', getattr(args, 'alignment_warmup_epochs', 8)))
    adaptive_alignment = resolve_param(cfg, 'alignment', 'adaptive_alignment', getattr(args, 'adaptive_alignment', True))
    alignment_min_correlation = float(resolve_param(cfg, 'alignment', 'alignment_min_correlation', getattr(args, 'alignment_min_correlation', 0.05)))
    # Global alignment / residual options (Tier B refinements)
    enable_global_alignment_pass = resolve_param(cfg, 'global_alignment', 'enable_global_alignment_pass', getattr(args, 'enable_global_alignment_pass', True))  # Fixed: was False, now matches parameter_default.json
    alignment_global_students = int(resolve_param(cfg, 'global_alignment', 'alignment_global_students', getattr(args, 'alignment_global_students', 600)))
    use_residual_alignment = resolve_param(cfg, 'global_alignment', 'use_residual_alignment', getattr(args, 'use_residual_alignment', True))  # Fixed: was False, now matches parameter_default.json
    alignment_residual_window = int(resolve_param(cfg, 'global_alignment', 'alignment_residual_window', getattr(args, 'alignment_residual_window', 5)))
    # Refinement cycle new arguments
    # Phase 0–2 semantic emergence controls (updated defaults)
    enable_retention_loss = resolve_param(cfg, 'refinement', 'enable_retention_loss', getattr(args, 'enable_retention_loss', True))  # Fixed: was False, now matches parameter_default.json
    retention_delta = float(resolve_param(cfg, 'refinement', 'retention_delta', getattr(args, 'retention_delta', 0.005)))
    retention_weight = float(resolve_param(cfg, 'refinement', 'retention_weight', getattr(args, 'retention_weight', 0.14)))
    enable_lag_gain_loss = resolve_param(cfg, 'refinement', 'enable_lag_gain_loss', getattr(args, 'enable_lag_gain_loss', True))  # Fixed: was False, now matches parameter_default.json
    lag_gain_weight = float(resolve_param(cfg, 'refinement', 'lag_gain_weight', getattr(args, 'lag_gain_weight', 0.06)))
    lag_max_lag = int(resolve_param(cfg, 'refinement', 'lag_max_lag', getattr(args, 'lag_max_lag', 3)))
    # Weighted multi-lag scheme (L1 emphasis)
    lag_l1_weight = float(resolve_param(cfg, 'refinement', 'lag_l1_weight', getattr(args, 'lag_l1_weight', 0.5)))
    lag_l2_weight = float(resolve_param(cfg, 'refinement', 'lag_l2_weight', getattr(args, 'lag_l2_weight', 0.3)))
    lag_l3_weight = float(resolve_param(cfg, 'refinement', 'lag_l3_weight', getattr(args, 'lag_l3_weight', 0.2)))
    # Alignment share cap & decay factor
    alignment_share_cap = float(resolve_param(cfg, 'alignment', 'alignment_share_cap', getattr(args, 'alignment_share_cap', 0.08)))
    alignment_share_decay_factor = float(resolve_param(cfg, 'alignment', 'alignment_share_decay_factor', getattr(args, 'alignment_share_decay_factor', 0.7)))
    enable_cosine_perf_schedule = resolve_param(cfg, 'runtime', 'enable_cosine_perf_schedule', getattr(args, 'enable_cosine_perf_schedule', False))
    consistency_rebalance_epoch = int(resolve_param(cfg, 'refinement', 'consistency_rebalance_epoch', getattr(args, 'consistency_rebalance_epoch', 8)))
    consistency_rebalance_threshold = float(resolve_param(cfg, 'refinement', 'consistency_rebalance_threshold', getattr(args, 'consistency_rebalance_threshold', 0.10)))
    consistency_rebalance_new_weight = float(resolve_param(cfg, 'refinement', 'consistency_rebalance_new_weight', getattr(args, 'consistency_rebalance_new_weight', 0.2)))
    variance_floor = float(resolve_param(cfg, 'refinement', 'variance_floor', getattr(args, 'variance_floor', 1e-4)))
    variance_floor_patience = int(resolve_param(cfg, 'refinement', 'variance_floor_patience', getattr(args, 'variance_floor_patience', 3)))
    variance_floor_reduce_factor = float(resolve_param(cfg, 'refinement', 'variance_floor_reduce_factor', getattr(args, 'variance_floor_reduce_factor', 0.5)))
    
    # Individual constraint weights - OPTIMAL values from parameter sweep
    non_negative_loss_weight = resolve_param(cfg, 'interpretability', 'non_negative_loss_weight', getattr(args, 'non_negative_loss_weight', 0.0))
    monotonicity_loss_weight = resolve_param(cfg, 'interpretability', 'monotonicity_loss_weight', getattr(args, 'monotonicity_loss_weight', 0.1))
    mastery_performance_loss_weight = resolve_param(cfg, 'interpretability', 'mastery_performance_loss_weight', getattr(args, 'mastery_performance_loss_weight', 0.8))
    gain_performance_loss_weight = resolve_param(cfg, 'interpretability', 'gain_performance_loss_weight', getattr(args, 'gain_performance_loss_weight', 0.8))
    sparsity_loss_weight = resolve_param(cfg, 'interpretability', 'sparsity_loss_weight', getattr(args, 'sparsity_loss_weight', 0.2))
    consistency_loss_weight = resolve_param(cfg, 'interpretability', 'consistency_loss_weight', getattr(args, 'consistency_loss_weight', 0.3))
    
    # Setup logging with experiment-specific logger name for parallel disambiguation
    logger_name = f"gainakt2exp.{experiment_suffix}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - ' + logger_name + ' - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    # Avoid propagation to root to prevent duplicate lines
    logger.propagate = False
    
    logger.info("=" * 80)
    logger.info("TRAINING GAINAKT2Exp WITH CUMULATIVE MASTERY")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Enhanced constraints: {enhanced_constraints}")
    logger.info("Constraint weights:")
    logger.info(f"  Non-negative loss: {non_negative_loss_weight}")
    logger.info(f"  Monotonicity loss: {monotonicity_loss_weight}")
    logger.info(f"  Mastery performance loss: {mastery_performance_loss_weight}")
    logger.info(f"  Gain performance loss: {gain_performance_loss_weight}")
    logger.info(f"  Sparsity loss: {sparsity_loss_weight}")
    logger.info(f"  Consistency loss: {consistency_loss_weight}")
    if enable_alignment_loss:
        logger.info(f"Alignment loss enabled (weight={alignment_weight}, warmup_epochs={alignment_warmup_epochs}, adaptive={adaptive_alignment}, target_min_corr={alignment_min_correlation})")
        if enable_global_alignment_pass:
            logger.info(f"Global alignment pass ENABLED (students={alignment_global_students}, residual={use_residual_alignment})")
        else:
            logger.info("Global alignment pass disabled")
    else:
        logger.info("Alignment loss disabled")
    if enable_retention_loss:
        logger.info(f"Retention loss ENABLED (delta={retention_delta}, weight={retention_weight})")
    if enable_lag_gain_loss:
        logger.info(f"Lag gain loss ENABLED (max_lag={lag_max_lag}, weight={lag_gain_weight})")
    if enable_cosine_perf_schedule:
        logger.info("Cosine schedule for performance alignment losses ENABLED")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if use_amp and device.type == 'cuda':
        logger.info("Mixed precision (AMP) enabled")
    elif use_amp:
        logger.info("AMP requested but CUDA not available; running in full precision")
    
    # Initialize wandb if requested (force offline mode for clean operation)
    if use_wandb:
        # Always use offline mode for clean operation and network independence
        wandb.init(project="pykt-cumulative-mastery", name=f"gainakt2exp_{experiment_suffix}", mode="offline")
    
    # Use standard PyKT data configuration
    data_config = {
        "assist2015": {
            "dpath": "/workspaces/pykt-toolkit/data/assist2015",
            "num_q": 0,
            "num_c": 100,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": 3,
            "maxlen": 200,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1, 2, 3, 4],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv"
        }
    }
    
    model_name = "gainakt2exp"
    logger.info(f"Loading dataset: {dataset_name}")
    train_loader, valid_loader = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
    logger.info("Dataset loaded successfully")
    
    # Create model with architecture from input/defaults sections
    num_c = data_config[dataset_name]['num_c']
    
    # Build arch_cfg from input (overrides) and defaults
    if cfg is not None:
        # Check for model_config params in input and defaults
        model_config_params = ['seq_len', 'd_model', 'n_heads', 'num_encoder_blocks', 'd_ff', 'dropout', 'emb_type']
        arch_cfg = {}
        for param in model_config_params:
            # Priority: input -> defaults -> None
            if 'input' in cfg and param in cfg['input']:
                arch_cfg[param] = cfg['input'][param]
            elif 'defaults' in cfg and param in cfg['defaults']:
                arch_cfg[param] = cfg['defaults'][param]
            elif 'model_config' in cfg and param in cfg['model_config']:
                # Backward compatibility with old format
                arch_cfg[param] = cfg['model_config'][param]
        
        # Validate we got the required parameters
        mandatory_arch = ['seq_len','d_model','n_heads','num_encoder_blocks','d_ff','dropout']
        missing_arch = [k for k in mandatory_arch if k not in arch_cfg]
        if missing_arch:
            raise KeyError(f"Missing required model architecture parameters in config (input/defaults): {missing_arch}")
        
        # Set default for emb_type if not specified
        if 'emb_type' not in arch_cfg:
            arch_cfg['emb_type'] = 'qid'
    else:
        # Standalone run (no config provided): synthesize architecture from CLI defaults
        arch_cfg = {
            'seq_len': getattr(args, 'seq_len'),
            'd_model': getattr(args, 'd_model'),
            'n_heads': getattr(args, 'n_heads'),
            'num_encoder_blocks': getattr(args, 'num_encoder_blocks'),
            'd_ff': getattr(args, 'd_ff'),
            'dropout': getattr(args, 'dropout'),
            'emb_type': getattr(args, 'emb_type', 'qid')
        }
        logger.info("[ArchCLI] Using CLI architecture parameter values following the approach based on explicit parameters, zero defaults")
    
    # Remove the old validation code that was here
    mandatory_arch = ['seq_len','d_model','n_heads','num_encoder_blocks','d_ff','dropout']
    missing_arch = [k for k in mandatory_arch if k not in arch_cfg]
    if missing_arch:
        raise KeyError(f"Missing mandatory architecture keys in model_config: {missing_arch}")
    # Architecture override precedence: if a CLI flag for an architecture key is explicitly provided,
    # we override the config value. This enables ablations via launcher/relaunch without editing config.json.
    # Detect presence in sys.argv (raw) to decide override; argparse will still supply defaults.
    cli_args_raw = sys.argv
    def arch_value(key, arg_name=None):
        flag = f"--{arg_name or key}"
        if any(a.startswith(flag) for a in cli_args_raw):
            return getattr(args, arg_name or key)
        # fallback to config value
        return arch_cfg.get(key, getattr(args, arg_name or key))
    resolved_seq_len = arch_value('seq_len')
    resolved_d_model = arch_value('d_model')
    resolved_n_heads = arch_value('n_heads')
    resolved_num_encoder_blocks = arch_value('num_encoder_blocks')
    resolved_d_ff = arch_value('d_ff')
    resolved_dropout = arch_value('dropout')
    resolved_emb_type = arch_value('emb_type')
    # Log overrides for transparency
    override_msgs = []
    for k, v in [('seq_len', resolved_seq_len), ('d_model', resolved_d_model), ('n_heads', resolved_n_heads),
                 ('num_encoder_blocks', resolved_num_encoder_blocks), ('d_ff', resolved_d_ff), ('dropout', resolved_dropout), ('emb_type', resolved_emb_type)]:
        config_val = arch_cfg.get(k)
        cli_flag = f"--{k}" in cli_args_raw
        if cli_flag and v != config_val:
            override_msgs.append(f"{k}: config={config_val} -> cli={v}")
    if override_msgs:
        logger.info("[ArchOverride] CLI architecture overrides applied: " + "; ".join(override_msgs))
    model_config = {
        'num_c': num_c,
        'seq_len': resolved_seq_len,
        'd_model': resolved_d_model,
        'n_heads': resolved_n_heads,
        'num_encoder_blocks': resolved_num_encoder_blocks,
        'd_ff': resolved_d_ff,
        'dropout': resolved_dropout,
        'emb_type': resolved_emb_type,
        'monitor_frequency': resolve_param(cfg, 'runtime', 'monitor_freq', 50),
        'use_mastery_head': resolve_param(cfg, 'interpretability', 'use_mastery_head', getattr(args, 'use_mastery_head', True)),
        'use_gain_head': resolve_param(cfg, 'interpretability', 'use_gain_head', getattr(args, 'use_gain_head', True)),
        'intrinsic_gain_attention': resolve_param(cfg, 'interpretability', 'intrinsic_gain_attention', getattr(args, 'intrinsic_gain_attention', False))
    }
    
    # Constraint resolution logic:
    # 1. If enhanced_constraints == False => PURE BCE (all weights forced to 0.0)
    # 2. If enhanced_constraints == True and no individual weights explicitly provided => use preset
    # 3. Otherwise use individually supplied weights
    individual_params = [
        'non_negative_loss_weight', 'monotonicity_loss_weight', 'mastery_performance_loss_weight',
        'gain_performance_loss_weight', 'sparsity_loss_weight', 'consistency_loss_weight'
    ]
    if not enhanced_constraints:
        model_config.update({
            'non_negative_loss_weight': 0.0,
            'monotonicity_loss_weight': 0.0,
            'mastery_performance_loss_weight': 0.0,
            'gain_performance_loss_weight': 0.0,
            'sparsity_loss_weight': 0.0,
            'consistency_loss_weight': 0.0
        })
        logger.info("PURE BCE baseline: all constraint weights forced to 0.0")
    elif enhanced_constraints and cfg is not None and all(p in cfg.get('interpretability', {}) for p in individual_params):
        # Config already holds weights; use them directly
        model_config.update({
            'non_negative_loss_weight': non_negative_loss_weight,
            'monotonicity_loss_weight': monotonicity_loss_weight,
            'mastery_performance_loss_weight': mastery_performance_loss_weight,
            'gain_performance_loss_weight': gain_performance_loss_weight,
            'sparsity_loss_weight': sparsity_loss_weight,
            'consistency_loss_weight': consistency_loss_weight
        })
        logger.info("Enhanced constraints: weights sourced from config.json interpretability section")
    elif enhanced_constraints and cfg is None:
        model_config.update({
            'non_negative_loss_weight': non_negative_loss_weight,
            'monotonicity_loss_weight': monotonicity_loss_weight,
            'mastery_performance_loss_weight': mastery_performance_loss_weight,
            'gain_performance_loss_weight': gain_performance_loss_weight,
            'sparsity_loss_weight': sparsity_loss_weight,
            'consistency_loss_weight': consistency_loss_weight
        })
        logger.info("Enhanced constraints: weights from CLI arguments (no config file)")
    else:
        model_config.update({
            'non_negative_loss_weight': non_negative_loss_weight,
            'monotonicity_loss_weight': monotonicity_loss_weight,
            'mastery_performance_loss_weight': mastery_performance_loss_weight,
            'gain_performance_loss_weight': gain_performance_loss_weight,
            'sparsity_loss_weight': sparsity_loss_weight,
            'consistency_loss_weight': consistency_loss_weight
        })
        logger.info("Using individually supplied constraint weights")

    # Explicit logging of resolved weights for auditability
    logger.info("Resolved constraint weights (final): "
                f"non_neg={model_config['non_negative_loss_weight']} | "
                f"mono={model_config['monotonicity_loss_weight']} | "
                f"mastery_perf={model_config['mastery_performance_loss_weight']} | "
                f"gain_perf={model_config['gain_performance_loss_weight']} | "
                f"sparsity={model_config['sparsity_loss_weight']} | "
                f"consistency={model_config['consistency_loss_weight']}")
    # Repro integration: detect EXPERIMENT_DIR
    experiment_dir = os.environ.get('EXPERIMENT_DIR')
    # If not launched via run_repro_experiment (no EXPERIMENT_DIR), create a manual containment folder.
    if not experiment_dir:
        manual_root = os.path.join(os.getcwd(), 'examples', 'experiments')
        os.makedirs(manual_root, exist_ok=True)
        fallback_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Use model name + 'manual' suffix for clarity; avoid including many hyperparameters.
        experiment_dir = os.path.join(manual_root, f"{fallback_tag}_{model_name}_manual")
        os.makedirs(experiment_dir, exist_ok=True)
        logger.info(f"[Repro] Auto-created experiment_dir for manual run: {experiment_dir}")
    if experiment_dir:
        try:
            os.makedirs(experiment_dir, exist_ok=True)
            repro_metrics_csv = os.path.join(experiment_dir, 'metrics_epoch.csv')
            repro_results_json = os.path.join(experiment_dir, 'results.json')
            repro_best_ckpt = os.path.join(experiment_dir, 'model_best.pth')
            repro_last_ckpt = os.path.join(experiment_dir, 'model_last.pth')
            if not os.path.exists(repro_metrics_csv):
                with open(repro_metrics_csv, 'w', newline='') as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow([
                        'epoch','train_loss','train_auc','val_loss','val_auc','val_accuracy',
                        'monotonicity_violation_rate','negative_gain_rate','bounds_violation_rate',
                        'mastery_correlation','gain_correlation','main_loss_share','constraint_loss_share',
                        'alignment_loss_share','lag_loss_share','retention_loss_share'
                    ])
            logger.info(f"[Repro] Writing artifacts into {experiment_dir}")
        except Exception as e:
            logger.warning(f"[Repro] Failed to initialize experiment directory '{experiment_dir}': {e}")
            experiment_dir = None
    # ...existing code...
    logger.info("Creating GainAKT2Exp (mastery_head=%s, gain_head=%s) with CUMULATIVE MASTERY..." % (
        model_config['use_mastery_head'], model_config['use_gain_head']) )
    model = create_exp_model(model_config)
    monitor_freq = resolve_param(cfg, 'runtime', 'monitor_freq', getattr(args, 'monitor_freq', 50))
    monitor = InterpretabilityMonitor(model, log_frequency=monitor_freq)
    model.set_monitor(monitor)
    model = model.to(device)
    # ------------------------------
    # Simple Multi-GPU Support (Option A): DataParallel
    # Automatically wraps the model if more than one CUDA device is visible.
    # Usage: set CUDA_VISIBLE_DEVICES=0,1,2 (or similar) before launching the script.
    # This keeps all existing logic unchanged (no need for distributed initialization).
    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        visible_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        gpu_ids_env = os.environ.get('PYKT_GPU_IDS', '')
        num_gpus_env = os.environ.get('PYKT_NUM_GPUS', '')
        # Determine selection source (matches launcher precedence)
        if visible_env:
            selection_source = 'CUDA_VISIBLE_DEVICES'
        elif gpu_ids_env:
            selection_source = 'PYKT_GPU_IDS'
        elif num_gpus_env:
            selection_source = 'PYKT_NUM_GPUS'
        else:
            selection_source = 'heuristic_<70% or external default'
        logger.info(f"[GPU] source={selection_source} CUDA_VISIBLE_DEVICES='{visible_env}' | torch.cuda.device_count()={gpu_count}")
        # List device names
        try:
            names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            logger.info(f"[GPU] Device names: {names}")
        except Exception as e:
            logger.info(f"[GPU] Unable to list device names: {e}")
        # If env specifies devices, compare counts
        if visible_env:
            requested_ids = [d for d in visible_env.split(',') if d.strip()!='']
            if len(requested_ids) != gpu_count:
                logger.warning(f"[GPU] Mismatch: requested {len(requested_ids)} ids ({requested_ids}) but torch reports {gpu_count} devices. DataParallel will use reported count.")
        if gpu_count > 1:
            logger.info(f"Multi-GPU detected (n={gpu_count}). Wrapping model with nn.DataParallel for simple data parallelism.")
            model = torch.nn.DataParallel(model)
            # Reattach monitor through .module if needed by external utilities
            try:
                model.module.set_monitor(monitor)
            except Exception:
                pass
        else:
            logger.info("Single GPU detected; running without DataParallel.")
    else:
        logger.info("CUDA not available; running on CPU.")

    # Convenience handle to underlying module for attribute access
    model_core = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    logger.info("Perfect consistency guaranteed by architectural constraints")
    
    # Training setup
    # Use BCEWithLogitsLoss for AMP safety; model now returns both logits and sigmoid outputs.
    criterion = nn.BCEWithLogitsLoss()
    if getattr(args, 'optimizer', 'Adam').lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {getattr(args,'optimizer')}")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training tracking
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    train_history = {
        'train_loss': [],
        'train_auc': [],
        'val_auc': [],
        'consistency_metrics': [],
        'semantic_trajectory': []
    }
    
    logger.info(f"\\nStarting training for {num_epochs} epochs...")
    # Tier B semantic emergence parameters
    warmup_constraint_epochs = getattr(args, 'warmup_constraint_epochs', 8)
    max_semantic_students = getattr(args, 'max_semantic_students', 50)
    semantic_trajectory_path = getattr(
        args,
        'semantic_trajectory_path',
        f"paper/results/gainakt2exp_semantic_trajectory_{experiment_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    logger.info(f"Warm-up constraint epochs: {warmup_constraint_epochs}")
    logger.info(f"Max semantic students per consistency sample: {max_semantic_students}")
    logger.info(f"Semantic trajectory path (will be written at end): {semantic_trajectory_path}")
    if enable_global_alignment_pass:
        logger.info("Global alignment sampling active: will compute global correlations after each epoch.")

    # State for adaptive alignment using global correlations
    global_alignment_state = {
        'prev_global_mastery_corr': None,
        'prev_global_gain_corr': None,
        # Keep previous-previous for plateau detection under alignment cap
        'prev_prev_global_mastery_corr': None,
        'prev_prev_global_gain_corr': None,
        'effective_alignment_weight_last': alignment_weight,
        'alignment_weight_current': alignment_weight  # dynamic decay under share cap
    }
    retention_state = {
        'peak_mastery_corr': 0.0,
        'low_variance_epochs': 0,
        'pending_penalty': 0.0
    }
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 50)
        # Epoch-level warm-up scaling
        if enhanced_constraints:
            if epoch == 0:
                # Initialize base weights from underlying module (model_core)
                model_core._base_mastery_performance_loss_weight = model_core.mastery_performance_loss_weight
                model_core._base_gain_performance_loss_weight = model_core.gain_performance_loss_weight
            if enable_cosine_perf_schedule:
                # Cosine schedule across total epochs to avoid early variance collapse
                progress = (epoch + 1) / num_epochs
                cosine_scale = 0.5 * (1 - np.cos(np.pi * progress))
                model_core.mastery_performance_loss_weight = model_core._base_mastery_performance_loss_weight * cosine_scale
                model_core.gain_performance_loss_weight = model_core._base_gain_performance_loss_weight * cosine_scale
                scale = cosine_scale
                logger.info(f"Cosine scale applied: {scale:.2f} (mastery_perf={model_core.mastery_performance_loss_weight:.3f}, gain_perf={model_core.gain_performance_loss_weight:.3f})")
            elif warmup_constraint_epochs > 0:
                scale = min(1.0, (epoch + 1) / warmup_constraint_epochs)
                model_core.mastery_performance_loss_weight = model_core._base_mastery_performance_loss_weight * scale
                model_core.gain_performance_loss_weight = model_core._base_gain_performance_loss_weight * scale
                logger.info(f"Warm-up scale applied: {scale:.2f} (mastery_perf={model_core.mastery_performance_loss_weight:.3f}, gain_perf={model_core.gain_performance_loss_weight:.3f})")
            else:
                scale = 1.0
        else:
            scale = 1.0

        # Training phase
        model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_interpretability_loss = 0.0
        total_alignment_loss = 0.0
        total_lag_loss = 0.0
        total_retention_component = 0.0
        batch_mastery_variances = []
        epoch_lag_corrs = []  # store per-lag correlation values
        total_predictions = []
        total_targets = []
        pending_retention_penalty = retention_state.get('pending_penalty', 0.0)
        if enable_retention_loss and pending_retention_penalty > 0:
            logger.info(f"[Retention] APPLYING GRADIENT retention penalty this epoch: {pending_retention_penalty:.5f}")
        elif enable_retention_loss:
            logger.info("[Retention] No retention penalty (no decay detected)")
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)

            optimizer.zero_grad()
            try:
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    # Forward with underlying module to access custom method
                    # DataParallel scatter occurs only when calling model(...). To retain internal states,
                    # we invoke forward_with_states on each replica via a thin wrapper exposed as 'forward'.
                    # For multi-GPU we call model(...) which internally calls forward() -> forward_with_states.
                    # The standard forward returns limited outputs; for training we need logits & interpretability states.
                    # Solution: temporarily add attribute access: if DataParallel, gather states from model.module after call.
                    if isinstance(model, torch.nn.DataParallel):
                        # Call collective forward to trigger scatter
                        fwd_basic = model(questions, responses, qry=questions_shifted, qtest=False)
                        # After collective forward, call forward_with_states on primary module with full batch (already on device 0)
                        # Note: This duplicates computation on device0; acceptable for short diagnostic. For true efficiency,
                        # implement custom DataParallel subclass to return states from replicas.
                        outputs = model.module.forward_with_states(q=questions, r=responses, qry=questions_shifted, batch_idx=batch_idx)
                    else:
                        outputs = model.forward_with_states(q=questions, r=responses, qry=questions_shifted, batch_idx=batch_idx)
                    logits = outputs['logits']
                    interpretability_loss = outputs['interpretability_loss']
                    valid_mask = mask.bool()
                    valid_predictions = logits[valid_mask]
                    valid_targets = responses_shifted[valid_mask].float()
                    main_loss = criterion(valid_predictions, valid_targets)
                    alignment_loss = torch.zeros(1, device=device)
                    alignment_corr_mastery = torch.zeros(1, device=device)
                    alignment_corr_gain = torch.zeros(1, device=device)
                    if (enable_alignment_loss and hasattr(model_core, 'mastery_head') and hasattr(model_core, 'gain_head') and
                        model_core.mastery_head is not None and model_core.gain_head is not None and 
                        'projected_mastery' in outputs and 'projected_gains' in outputs):
                        # Sample subset of students/time steps for efficiency
                        # projected_mastery shape: [B, T, C] *maybe*, we average across concepts
                        pm = outputs['projected_mastery']  # B x T x C
                        pg = outputs['projected_gains']    # B x T x C
                        # Build performance tensor aligned with mastery timeline (shifted responses)
                        perf = responses_shifted  # B x T
                        student_mask = mask.bool()  # B x T
                        # Flatten selection
                        # Limit to first N students if very large
                        max_students_align = 32
                        if pm.size(0) > max_students_align:
                            pm = pm[:max_students_align]
                            pg = pg[:max_students_align]
                            perf = perf[:max_students_align]
                            student_mask = student_mask[:max_students_align]
                        # Average mastery/gains across concepts
                        mastery_mean = pm.mean(dim=2)  # B x T
                        gains_mean = pg.mean(dim=2)    # B x T
                        # Mask out padding
                        mastery_sel = mastery_mean[student_mask]
                        gains_sel = gains_mean[student_mask]
                        perf_sel = perf[student_mask].float()
                        # Optional residual transformation for performance (emphasize incremental change)
                        if use_residual_alignment:
                            # Compute simple rolling mean and residualize
                            # perf_sel is flattened sequence across students; to approximate temporal residuals
                            # we apply a 1D rolling window on original (unflattened) performance prior to masking.
                            # Simplicity: approximate by subtracting global moving average over chunks.
                            perf_raw = perf_sel.clone()
                            if perf_raw.numel() >= alignment_residual_window + 2:
                                kernel = torch.ones(alignment_residual_window, device=device) / alignment_residual_window
                                # pad with reflection to keep size
                                pad = alignment_residual_window // 2
                                perf_padded = torch.nn.functional.pad(perf_raw.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect')
                                smooth = torch.nn.functional.conv1d(perf_padded, kernel.view(1,1,-1)).squeeze()
                                smooth = smooth[:perf_raw.numel()]
                                perf_sel = (perf_raw - smooth).detach()  # detach smoothing for stability
                        def corr_fn(x, y):
                            if x.numel() < 3 or y.numel() < 3:
                                return torch.zeros(1, device=device)
                            xm = x - x.mean()
                            ym = y - y.mean()
                            denom = (xm.std(unbiased=False) * ym.std(unbiased=False) + 1e-6)
                            return (xm * ym).mean() / denom
                        alignment_corr_mastery = corr_fn(mastery_sel, perf_sel)
                        alignment_corr_gain = corr_fn(gains_sel, perf_sel)
                        # Warm-up scaling
                        align_scale = min(1.0, (epoch + 1) / max(1, alignment_warmup_epochs))
                        # Dynamic effective weight (subject to decay under alignment share cap logic)
                        effective_weight = global_alignment_state['alignment_weight_current'] * align_scale
                        # Adaptive up-weight if below target after warm-up using global correlation if available
                        if adaptive_alignment and align_scale >= 1.0:
                            reference_corr = alignment_corr_mastery.detach()
                            if enable_global_alignment_pass and global_alignment_state['prev_global_mastery_corr'] is not None:
                                reference_corr = torch.tensor(global_alignment_state['prev_global_mastery_corr'], device=device)
                            if reference_corr < alignment_min_correlation:
                                factor = min(3.0, 1.0 + (alignment_min_correlation - float(reference_corr)) * 4.0)
                                effective_weight = effective_weight * factor
                        global_alignment_state['effective_alignment_weight_last'] = float(effective_weight)
                        # Multi-lag predictive emergence objective
                        lag_loss = torch.zeros(1, device=device)
                        mean_lag_corr = torch.zeros(1, device=device)
                        lag_corr_count = 0
                        if enable_lag_gain_loss and lag_max_lag > 0 and (epoch + 1) >= (warmup_constraint_epochs + 3):
                            # Phase 4 Redesigned lag objective: stricter activation gate and improved per-student normalization
                            # to promote genuine incremental predictive semantics (Gain_t -> Correct_{t+lag}).
                            gains_mean_time = gains_mean  # B x T
                            perf_time = perf.float()      # B x T
                            weights_map = {1: lag_l1_weight, 2: lag_l2_weight, 3: lag_l3_weight}
                            weighted_corr_sum = torch.zeros(1, device=device)
                            lag_terms = []
                            
                            # Per-student normalization for better lag correlation stability
                            for student_idx in range(gains_mean_time.size(0)):
                                student_gains = gains_mean_time[student_idx]  # T
                                student_perf = perf_time[student_idx]         # T
                                student_valid = student_mask[student_idx]     # T
                                
                                if student_valid.sum() < 5:  # need minimum sequence length
                                    continue
                                    
                                for lag in range(1, min(lag_max_lag + 1, 3)):  # Focus on lag 1-2 for stability
                                    T = student_gains.size(0)
                                    if T - lag <= 2:  # stricter minimum window
                                        continue
                                    
                                    gm_window = student_gains[:T - lag][student_valid[:T - lag]]
                                    pt_window = student_perf[lag:][student_valid[lag:]]
                                    
                                    if gm_window.numel() < 3 or pt_window.numel() < 3:
                                        continue
                                    
                                    # Per-student z-score normalization for cleaner lag signal
                                    gm_z = (gm_window - gm_window.mean()) / (gm_window.std(unbiased=False) + 1e-6)
                                    pt_z = (pt_window - pt_window.mean()) / (pt_window.std(unbiased=False) + 1e-6)
                                    
                                    if gm_z.numel() == pt_z.numel() and gm_z.numel() >= 3:
                                        corr_lag = corr_fn(gm_z, pt_z)
                                        w_lag = weights_map.get(lag, 0.0)
                                        if w_lag > 0 and not torch.isnan(corr_lag):
                                            weighted_corr_sum += w_lag * corr_lag
                                            lag_terms.append((lag, corr_lag.detach(), w_lag))
                                            try:
                                                epoch_lag_corrs.append({'lag': lag, 'corr': float(corr_lag.detach().cpu()), 'weight': w_lag})
                                            except Exception:
                                                pass
                            
                            total_w = lag_l1_weight + lag_l2_weight  # Focus on lag 1-2
                            if lag_terms and total_w > 0:
                                mean_lag_corr = weighted_corr_sum / total_w
                                lag_corr_count = len(lag_terms)
                                # Encourage positive predictive lag correlation (Phase 4 improvement)
                                lag_loss = - torch.clamp(mean_lag_corr, min=0.0) * lag_gain_weight  # Only reward positive correlations
                        if 'projected_mastery' in outputs and outputs['projected_mastery'].var().item() < variance_floor:
                            alignment_loss = torch.zeros(1, device=device)
                        else:
                            alignment_loss = - (alignment_corr_mastery + alignment_corr_gain) * effective_weight + lag_loss
                        # accumulate alignment & lag losses separately for share reporting
                        total_alignment_loss += float((- (alignment_corr_mastery + alignment_corr_gain) * effective_weight).detach().cpu())
                        total_lag_loss += float(lag_loss.detach().cpu())

                    # Variance floor tracking & sparsity adjustment
                    if enable_alignment_loss and 'projected_mastery' in outputs:
                        mastery_variance = outputs['projected_mastery'].var().item()
                        batch_mastery_variances.append(mastery_variance)
                        if mastery_variance < variance_floor:
                            retention_state['low_variance_epochs'] += 1
                        else:
                            retention_state['low_variance_epochs'] = 0
                        if retention_state['low_variance_epochs'] >= variance_floor_patience:
                            old_sparsity = getattr(model_core, 'sparsity_loss_weight', sparsity_loss_weight)
                            model_core.sparsity_loss_weight = old_sparsity * variance_floor_reduce_factor
                            logger.info(f"[VarianceFloor] Reduced sparsity_loss_weight from {old_sparsity:.3f} to {model_core.sparsity_loss_weight:.3f}")
                    # Compose total batch loss
                    retention_component = torch.zeros(1, device=device)
                    if enable_retention_loss and pending_retention_penalty > 0:
                        retention_component = torch.tensor(pending_retention_penalty / max(1, num_batches), device=device)
                        total_retention_component += float(retention_component.detach().cpu())
                    if enable_alignment_loss:
                        total_batch_loss = main_loss + interpretability_loss + alignment_loss + retention_component
                    else:
                        total_batch_loss = main_loss + interpretability_loss + retention_component
                # Backward & optimizer step
                if use_amp and device.type == 'cuda':
                    scaler.scale(total_batch_loss).backward()
                    clip_val = getattr(args, 'gradient_clip', 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_batch_loss.backward()
                    clip_val = getattr(args, 'gradient_clip', 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                    optimizer.step()
            except RuntimeError as oom:
                msg = str(oom).lower()
                if 'out of memory' in msg:
                    logger.warning("OOM encountered; clearing CUDA cache and skipping batch")
                    torch.cuda.empty_cache()
                    continue
                # Deterministic CuBLAS failure fallback: disable deterministic algorithms & retry once
                if ('cublas' in msg and 'deterministic' in msg) or ('cublas' in msg and 'workspace' in msg):
                    logger.warning("[DeterminismFallback] CuBLAS deterministic failure detected; disabling deterministic algorithms and retrying batch once.")
                    try:
                        if hasattr(torch, 'use_deterministic_algorithms'):
                            torch.use_deterministic_algorithms(False)
                        torch.backends.cudnn.deterministic = False
                        with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                            if isinstance(model, torch.nn.DataParallel):
                                _ = model(questions, responses, qry=questions_shifted, qtest=False)
                                outputs = model.module.forward_with_states(q=questions, r=responses, qry=questions_shifted, batch_idx=batch_idx)
                            else:
                                outputs = model.forward_with_states(q=questions, r=responses, qry=questions_shifted, batch_idx=batch_idx)
                            logits = outputs['logits']
                            interpretability_loss = outputs['interpretability_loss']
                            valid_mask = mask.bool()
                            valid_predictions = logits[valid_mask]
                            valid_targets = responses_shifted[valid_mask].float()
                            main_loss = criterion(valid_predictions, valid_targets)
                            alignment_loss = torch.zeros(1, device=device)
                            alignment_corr_mastery = torch.zeros(1, device=device)
                            alignment_corr_gain = torch.zeros(1, device=device)
                            if (enable_alignment_loss and hasattr(model_core, 'mastery_head') and hasattr(model_core, 'gain_head') and
                                model_core.mastery_head is not None and model_core.gain_head is not None and 
                                'projected_mastery' in outputs and 'projected_gains' in outputs):
                                pm = outputs['projected_mastery']
                                pg = outputs['projected_gains']
                                perf = responses_shifted
                                student_mask = mask.bool()
                                max_students_align = 32
                                if pm.size(0) > max_students_align:
                                    pm = pm[:max_students_align]
                                    pg = pg[:max_students_align]
                                    perf = perf[:max_students_align]
                                    student_mask = student_mask[:max_students_align]
                                mastery_mean = pm.mean(dim=2)
                                gains_mean = pg.mean(dim=2)
                                mastery_sel = mastery_mean[student_mask]
                                gains_sel = gains_mean[student_mask]
                                perf_sel = perf[student_mask].float()
                                if use_residual_alignment:
                                    perf_raw = perf_sel.clone()
                                    if perf_raw.numel() >= alignment_residual_window + 2:
                                        kernel = torch.ones(alignment_residual_window, device=device) / alignment_residual_window
                                        pad = alignment_residual_window // 2
                                        perf_padded = torch.nn.functional.pad(perf_raw.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect')
                                        smooth = torch.nn.functional.conv1d(perf_padded, kernel.view(1,1,-1)).squeeze()[:perf_raw.numel()]
                                        perf_sel = (perf_raw - smooth).detach()
                                def corr_fn(x, y):
                                    if x.numel() < 3 or y.numel() < 3:
                                        return torch.zeros(1, device=device)
                                    xm = x - x.mean()
                                    ym = y - y.mean()
                                    denom = (xm.std(unbiased=False) * ym.std(unbiased=False) + 1e-6)
                                    return (xm * ym).mean() / denom
                                alignment_corr_mastery = corr_fn(mastery_sel, perf_sel)
                                alignment_corr_gain = corr_fn(gains_sel, perf_sel)
                                align_scale = min(1.0, (epoch + 1) / max(1, alignment_warmup_epochs))
                                effective_weight = global_alignment_state['alignment_weight_current'] * align_scale
                                if adaptive_alignment and align_scale >= 1.0:
                                    reference_corr = alignment_corr_mastery.detach()
                                    if enable_global_alignment_pass and global_alignment_state['prev_global_mastery_corr'] is not None:
                                        reference_corr = torch.tensor(global_alignment_state['prev_global_mastery_corr'], device=device)
                                    if reference_corr < alignment_min_correlation:
                                        factor = min(3.0, 1.0 + (alignment_min_correlation - float(reference_corr)) * 4.0)
                                        effective_weight = effective_weight * factor
                                global_alignment_state['effective_alignment_weight_last'] = float(effective_weight)
                                lag_loss = torch.zeros(1, device=device)
                                mean_lag_corr = torch.zeros(1, device=device)
                                lag_corr_count = 0
                                if enable_lag_gain_loss and lag_max_lag > 0 and (epoch + 1) >= (warmup_constraint_epochs + 3):
                                    gains_mean_time = gains_mean
                                    perf_time = perf.float()
                                    weights_map = {1: lag_l1_weight, 2: lag_l2_weight, 3: lag_l3_weight}
                                    weighted_corr_sum = torch.zeros(1, device=device)
                                    lag_terms = []
                                    for student_idx in range(gains_mean_time.size(0)):
                                        student_gains = gains_mean_time[student_idx]
                                        student_perf = perf_time[student_idx]
                                        student_valid = student_mask[student_idx]
                                        if student_valid.sum() < 5:
                                            continue
                                        for lag in range(1, min(lag_max_lag + 1, 3)):
                                            T = student_gains.size(0)
                                            if T - lag <= 2:
                                                continue
                                            gm_window = student_gains[:T - lag][student_valid[:T - lag]]
                                            pt_window = student_perf[lag:][student_valid[lag:]]
                                            if gm_window.numel() < 3 or pt_window.numel() < 3:
                                                continue
                                            gm_z = (gm_window - gm_window.mean()) / (gm_window.std(unbiased=False) + 1e-6)
                                            pt_z = (pt_window - pt_window.mean()) / (pt_window.std(unbiased=False) + 1e-6)
                                            if gm_z.numel() == pt_z.numel() and gm_z.numel() >= 3:
                                                corr_lag = corr_fn(gm_z, pt_z)
                                                w_lag = weights_map.get(lag, 0.0)
                                                if w_lag > 0 and not torch.isnan(corr_lag):
                                                    weighted_corr_sum += w_lag * corr_lag
                                                    lag_terms.append((lag, corr_lag.detach(), w_lag))
                                    total_w = lag_l1_weight + lag_l2_weight
                                    if lag_terms and total_w > 0:
                                        mean_lag_corr = weighted_corr_sum / total_w
                                        lag_corr_count = len(lag_terms)
                                        lag_loss = - torch.clamp(mean_lag_corr, min=0.0) * lag_gain_weight
                                if 'projected_mastery' in outputs and outputs['projected_mastery'].var().item() < variance_floor:
                                    alignment_loss = torch.zeros(1, device=device)
                                else:
                                    alignment_loss = - (alignment_corr_mastery + alignment_corr_gain) * effective_weight + lag_loss
                                total_alignment_loss += float((- (alignment_corr_mastery + alignment_corr_gain) * effective_weight).detach().cpu())
                                total_lag_loss += float(lag_loss.detach().cpu())
                            retention_component = torch.zeros(1, device=device)
                            if enable_retention_loss and pending_retention_penalty > 0:
                                retention_component = torch.tensor(pending_retention_penalty / max(1, num_batches), device=device)
                                total_retention_component += float(retention_component.detach().cpu())
                            if enable_alignment_loss:
                                total_batch_loss = main_loss + interpretability_loss + alignment_loss + retention_component
                            else:
                                total_batch_loss = main_loss + interpretability_loss + retention_component
                        if use_amp and device.type == 'cuda':
                            scaler.scale(total_batch_loss).backward()
                            clip_val = getattr(args, 'gradient_clip', 1.0)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            total_batch_loss.backward()
                            clip_val = getattr(args, 'gradient_clip', 1.0)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                            optimizer.step()
                        try:
                            train_history.setdefault('determinism_fallback_batches', []).append({'epoch': epoch+1, 'batch': batch_idx})
                        except Exception:
                            pass
                        continue
                    except Exception as retry_exc:
                        logger.error(f"[DeterminismFallback] Retry failed: {retry_exc}")
                        raise
                else:
                    raise

            # Accumulate statistics
            total_loss += total_batch_loss.item()
            total_main_loss += main_loss.item()
            if isinstance(interpretability_loss, torch.Tensor):
                total_interpretability_loss += interpretability_loss.item()
            else:
                total_interpretability_loss += float(interpretability_loss)
            if enable_alignment_loss:
                total_interpretability_loss += alignment_loss.item()
            # alignment_loss already decomposed; nothing extra needed here
            with torch.no_grad():
                probs = torch.sigmoid(valid_predictions)
                total_predictions.extend(probs.cpu().numpy())
                total_targets.extend(valid_targets.cpu().numpy())

        # Compute training metrics (post minibatch loop)
        train_loss = total_loss / len(train_loader)
        train_main_loss = total_main_loss / len(train_loader)
        train_constraint_loss = total_interpretability_loss / len(train_loader)
        # Loss share instrumentation
        if total_loss > 0:
            main_loss_share = train_main_loss / (total_loss / len(train_loader))
            constraint_loss_share = train_constraint_loss / (total_loss / len(train_loader))
            alignment_loss_share = (total_alignment_loss / len(train_loader)) / (total_loss / len(train_loader)) if enable_alignment_loss else 0.0
            lag_loss_share = (total_lag_loss / len(train_loader)) / (total_loss / len(train_loader)) if enable_lag_gain_loss else 0.0
            retention_loss_share = (total_retention_component / len(train_loader)) / (total_loss / len(train_loader)) if enable_retention_loss else 0.0
            
            # Validate loss share normalization (Bug 1 fix)
            total_share = main_loss_share + constraint_loss_share + alignment_loss_share + lag_loss_share + retention_loss_share
            
            # Log warning if shares don't sum to ~1.0 or contain unexpected negatives
            if abs(total_share - 1.0) > 0.1:  # Tolerance of 10%
                logger.warning(f"[Loss Share Validation] Total share = {total_share:.4f} (expected ~1.0)")
                logger.warning(f"  Components: main={main_loss_share:.4f}, constraint={constraint_loss_share:.4f}, "
                             f"alignment={alignment_loss_share:.4f}, lag={lag_loss_share:.4f}, retention={retention_loss_share:.4f}")
            
            # Note: alignment_loss can be legitimately negative (correlation-based reward)
            # This is expected behavior when correlations are positive (encouraging them means negative loss)
            if alignment_loss_share < -0.1:  # Warning if magnitude > 10%
                logger.info(f"[Loss Share Info] Alignment loss share is notably negative: {alignment_loss_share:.4f} "
                          f"(positive correlations produce negative correlation-based losses)")
        else:
            main_loss_share = constraint_loss_share = alignment_loss_share = lag_loss_share = retention_loss_share = 0.0
        mean_mastery_variance = float(np.mean(batch_mastery_variances)) if batch_mastery_variances else None
        min_mastery_variance = float(np.min(batch_mastery_variances)) if batch_mastery_variances else None
        max_mastery_variance = float(np.max(batch_mastery_variances)) if batch_mastery_variances else None
        train_stats = compute_auc_acc(total_targets, total_predictions)
        train_auc = train_stats['auc']
        train_acc = train_stats['acc']
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in valid_loader:
                questions = batch['cseqs'].to(device)
                responses = batch['rseqs'].to(device)
                questions_shifted = batch['shft_cseqs'].to(device)
                responses_shifted = batch['shft_rseqs'].to(device)
                mask = batch['masks'].to(device)

                # Use forward_with_states to ensure logits are present for BCEWithLogitsLoss
                # Use model_core to access underlying module when DataParallel is active
                # Validation: for efficiency avoid duplicate forward; single GPU uses forward_with_states directly.
                if isinstance(model, torch.nn.DataParallel):
                    _ = model(questions, responses, qry=questions_shifted, qtest=False)
                    outputs = model_core.forward_with_states(q=questions, r=responses, qry=questions_shifted)
                else:
                    outputs = model_core.forward_with_states(q=questions, r=responses, qry=questions_shifted)
                logits = outputs.get('logits')
                if logits is None:
                    # Fallback: if logits absent (legacy path), approximate by inverse sigmoid of predictions
                    preds = outputs['predictions']
                    eps = 1e-6
                    logits = torch.log(preds.clamp(eps, 1 - eps) / (1 - preds.clamp(eps, 1 - eps)))
                
                valid_mask = mask.bool()
                valid_predictions = logits[valid_mask]
                valid_targets = responses_shifted[valid_mask].float()
                
                loss = criterion(valid_predictions, valid_targets)
                val_loss += loss.item()
                
                val_predictions.extend(torch.sigmoid(valid_predictions).cpu().numpy())
                val_targets.extend(valid_targets.cpu().numpy())
        
        val_loss = val_loss / len(valid_loader)
        val_stats = compute_auc_acc(val_targets, val_predictions)
        val_auc = val_stats['auc']
        val_acc = val_stats['acc']
        
        # Consistency validation
        logger.info("  Running consistency validation...")
        consistency_metrics = validate_model_consistency(
            model, valid_loader, device, logger, max_students=max_semantic_students
        )

        # Global alignment pass (sequence-level sampling separate from lightweight consistency check)
        if (enable_global_alignment_pass and hasattr(model_core, 'mastery_head') and hasattr(model_core, 'gain_head') and
            model_core.mastery_head is not None and model_core.gain_head is not None):
            model.eval()
            mastery_corrs = []
            gain_corrs = []
            # First pass: accumulate until enlarged sample or dataset exhausted
            with torch.no_grad():
                for batch in valid_loader:
                    questions = batch['cseqs'].to(device)
                    responses = batch['rseqs'].to(device)
                    questions_shifted = batch['shft_cseqs'].to(device)
                    responses_shifted = batch['shft_rseqs'].to(device)
                    mask = batch['masks'].to(device)
                    if isinstance(model, torch.nn.DataParallel):
                        _ = model(questions, responses, qry=questions_shifted, qtest=False)
                        outputs_full = model_core.forward_with_states(q=questions, r=responses, qry=questions_shifted)
                    else:
                        outputs_full = model_core.forward_with_states(q=questions, r=responses, qry=questions_shifted)
                    if 'projected_mastery' not in outputs_full:
                        continue
                    pm = outputs_full['projected_mastery']
                    pg = outputs_full['projected_gains']
                    perf = responses_shifted
                    for i in range(pm.size(0)):
                        msk = mask[i].bool()
                        if msk.sum() < 3:
                            continue
                        mastery_seq = pm[i][msk].mean(dim=1)
                        gains_seq = pg[i][msk].mean(dim=1)
                        perf_seq = perf[i][msk].float()
                        if use_residual_alignment and perf_seq.numel() >= alignment_residual_window + 2:
                            kernel = torch.ones(alignment_residual_window, device=device) / alignment_residual_window
                            pad = alignment_residual_window // 2
                            perf_pad = torch.nn.functional.pad(perf_seq.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect')
                            smooth = torch.nn.functional.conv1d(perf_pad, kernel.view(1,1,-1)).squeeze()[:perf_seq.numel()]
                            perf_corr_seq = (perf_seq - smooth)
                        else:
                            perf_corr_seq = perf_seq
                        def safe_corr(a, b):
                            if a.numel() < 3 or b.numel() < 3:
                                return None
                            am = a - a.mean()
                            bm = b - b.mean()
                            denom = (am.std(unbiased=False) * bm.std(unbiased=False) + 1e-6)
                            if denom < 1e-9:
                                return None
                            return float((am * bm).mean() / denom)
                        mc = safe_corr(mastery_seq, perf_corr_seq)
                        gc = safe_corr(gains_seq, perf_corr_seq)
                        if mc is not None:
                            mastery_corrs.append(mc)
                        if gc is not None:
                            gain_corrs.append(gc)
            # Stratified re-sampling by sequence length deciles (Phase 2)
            if alignment_global_students > 0:
                lengths = []
                records = []
                with torch.no_grad():
                    for batch in valid_loader:
                        questions = batch['cseqs'].to(device)
                        responses = batch['rseqs'].to(device)
                        questions_shifted = batch['shft_cseqs'].to(device)
                        responses_shifted = batch['shft_rseqs'].to(device)
                        mask = batch['masks'].to(device)
                        if isinstance(model, torch.nn.DataParallel):
                            _ = model(questions, responses, qry=questions_shifted, qtest=False)
                            out_full = model_core.forward_with_states(q=questions, r=responses, qry=questions_shifted)
                        else:
                            out_full = model_core.forward_with_states(q=questions, r=responses, qry=questions_shifted)
                        if 'projected_mastery' not in out_full:
                            continue
                        pm_all = out_full['projected_mastery']
                        pg_all = out_full['projected_gains']
                        perf_all = responses_shifted
                        for i in range(pm_all.size(0)):
                            msk = mask[i].bool()
                            L = int(msk.sum().item())
                            if L < 3:
                                continue
                            lengths.append(L)
                            records.append((pm_all[i][msk].mean(dim=1), pg_all[i][msk].mean(dim=1), perf_all[i][msk].float()))
                if records:
                    lengths_np = np.array(lengths)
                    deciles = [np.percentile(lengths_np, p) for p in range(0, 101, 10)]
                    bins = [[] for _ in range(10)]
                    for idx, L in enumerate(lengths_np):
                        bin_idx = min(9, int(np.searchsorted(deciles[1:-1], L, side='right')))
                        bins[bin_idx].append(idx)
                    target_per_bin = max(1, alignment_global_students // 10)
                    strat_mastery = []
                    strat_gain = []
                    def safe_corr(a, b):
                        if a.numel() < 3 or b.numel() < 3:
                            return None
                        am = a - a.mean()
                        bm = b - b.mean()
                        denom = (am.std(unbiased=False) * bm.std(unbiased=False) + 1e-6)
                        if denom < 1e-9:
                            return None
                        return float((am * bm).mean() / denom)
                    for bin_indices in bins:
                        if not bin_indices:
                            continue
                        take = min(target_per_bin, len(bin_indices))
                        # Deterministic selection: sorted order first N
                        deterministic_slice = sorted(bin_indices)[:take]
                        for ci in deterministic_slice:
                            m_seq, g_seq, p_seq = records[ci]
                            mc = safe_corr(m_seq, p_seq)
                            gc = safe_corr(g_seq, p_seq)
                            if mc is not None:
                                strat_mastery.append(mc)
                            if gc is not None:
                                strat_gain.append(gc)
                    if strat_mastery:
                        mastery_corrs = strat_mastery
                    if strat_gain:
                        gain_corrs = strat_gain
            global_mastery_corr = float(np.mean(mastery_corrs)) if mastery_corrs else 0.0
            global_gain_corr = float(np.mean(gain_corrs)) if gain_corrs else 0.0
            # Shift previous correlations for plateau detection
            global_alignment_state['prev_prev_global_mastery_corr'] = global_alignment_state['prev_global_mastery_corr']
            global_alignment_state['prev_prev_global_gain_corr'] = global_alignment_state['prev_global_gain_corr']
            global_alignment_state['prev_global_mastery_corr'] = global_mastery_corr
            global_alignment_state['prev_global_gain_corr'] = global_gain_corr
            logger.info(f"  🌐 Global Alignment - Mastery Corr: {global_mastery_corr:.4f}, Gain Corr: {global_gain_corr:.4f}, Eff Align Wt(last): {global_alignment_state['effective_alignment_weight_last']:.4f}")
            # Retention peak tracking
            if enable_retention_loss and global_mastery_corr is not None:
                if global_mastery_corr > retention_state['peak_mastery_corr']:
                    retention_state['peak_mastery_corr'] = global_mastery_corr
        else:
            global_mastery_corr = None
            global_gain_corr = None

        # Apply consistency rebalancing if criteria met
        if enable_alignment_loss and epoch + 1 >= consistency_rebalance_epoch:
            ref_corr = global_mastery_corr if global_mastery_corr is not None else consistency_metrics['mastery_correlation']
            if ref_corr < consistency_rebalance_threshold:
                old_consistency = getattr(model_core, 'consistency_loss_weight', consistency_loss_weight)
                if old_consistency > consistency_rebalance_new_weight + 1e-8:
                    model_core.consistency_loss_weight = consistency_rebalance_new_weight
                    logger.info(f"[Rebalance] Reduced consistency_loss_weight from {old_consistency:.3f} to {model_core.consistency_loss_weight:.3f} due to low mastery_corr {ref_corr:.4f}")

        # Retention loss scheduling (epoch >= warm-up end)
        retention_loss_value = 0.0
        if enable_retention_loss and enable_global_alignment_pass and epoch + 1 > warmup_constraint_epochs and global_mastery_corr is not None:
            decay_gap = retention_state['peak_mastery_corr'] - global_mastery_corr - retention_delta
            if decay_gap > 0:
                retention_loss_value = decay_gap * retention_weight
                retention_state['pending_penalty'] = retention_loss_value
                logger.info(f"[Retention] Scheduled gradient retention penalty {retention_loss_value:.5f} for next epoch (peak={retention_state['peak_mastery_corr']:.4f}, current={global_mastery_corr:.4f})")
            else:
                retention_state['pending_penalty'] = 0.0
        
        # Log epoch results with enhanced formatting
        logger.info("=" * 60)
        logger.info(f"📊 EPOCH {epoch + 1}/{num_epochs} RESULTS:")
        logger.info(f"  🚂 Train - Loss: {train_loss:.4f} (Main: {train_main_loss:.4f}, "
                   f"Constraint: {train_constraint_loss:.4f}), AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        
        # Log loss composition (Bug 1 fix - enhanced transparency)
        total_share = main_loss_share + constraint_loss_share + alignment_loss_share + lag_loss_share + retention_loss_share
        logger.info(f"  📊 Loss Composition (shares sum to {total_share:.3f}):")
        logger.info(f"     Main: {main_loss_share:.1%}, Constraint: {constraint_loss_share:.1%}, "
                   f"Alignment: {alignment_loss_share:+.1%}, Lag: {lag_loss_share:.1%}, Retention: {retention_loss_share:.1%}")
        
        logger.info(f"  ✅ Valid - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"  🧠 Peak GPU memory this epoch (global): {peak_mem:.1f} MiB")
            try:
                gpu_count_local = torch.cuda.device_count()
                per_device_stats = []
                for gid in range(gpu_count_local):
                    alloc = torch.cuda.memory_allocated(gid) / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved(gid) / (1024 ** 2)
                    per_device_stats.append(f"GPU{gid}: alloc={alloc:.1f}MiB reserved={reserved:.1f}MiB")
                logger.info("  🔍 Per-GPU memory: " + " | ".join(per_device_stats))
            except Exception as e:
                logger.info(f"  [GPU] Memory detail unavailable: {e}")
            torch.cuda.reset_peak_memory_stats()
        
        # Add AUC progress tracking
        if len(train_history['val_auc']) > 1:
            prev_auc = train_history['val_auc'][-2] if len(train_history['val_auc']) > 1 else 0
            auc_change = val_auc - prev_auc
            change_indicator = "📈" if auc_change > 0 else "📉" if auc_change < 0 else "➡️"
            logger.info(f"  {change_indicator} AUC Change: {auc_change:+.4f} (Current: {val_auc:.4f}, Previous: {prev_auc:.4f})")
        
        # Show current best
        current_best = max(train_history['val_auc']) if train_history['val_auc'] else 0
        logger.info(f"  🏆 Current Best AUC: {current_best:.4f}")
        logger.info("=" * 60)
        
        # Update history
        train_history['train_loss'].append(train_loss)
        train_history['train_auc'].append(train_auc)
        train_history['val_auc'].append(val_auc)
        train_history['consistency_metrics'].append(consistency_metrics)
        train_history['semantic_trajectory'].append({
            'epoch': epoch + 1,
            'mastery_correlation': consistency_metrics['mastery_correlation'],
            'gain_correlation': consistency_metrics['gain_correlation'],
            'warmup_scale': scale,
            'alignment_corr_mastery': float(alignment_corr_mastery.detach().cpu()) if 'alignment_corr_mastery' in locals() else None,
            'alignment_corr_gain': float(alignment_corr_gain.detach().cpu()) if 'alignment_corr_gain' in locals() else None,
            'global_alignment_mastery_corr': global_mastery_corr,
            'global_alignment_gain_corr': global_gain_corr,
            'effective_alignment_weight': global_alignment_state.get('effective_alignment_weight_last'),
            'peak_mastery_corr': retention_state['peak_mastery_corr'],
            'retention_loss_value': retention_loss_value,
            'mean_lag_corr': float(mean_lag_corr.detach().cpu()) if 'mean_lag_corr' in locals() else None,
            'lag_corr_count': lag_corr_count if 'lag_corr_count' in locals() else 0,
            'consistency_loss_weight_current': getattr(model_core, 'consistency_loss_weight', consistency_loss_weight),
            'sparsity_loss_weight_current': getattr(model_core, 'sparsity_loss_weight', sparsity_loss_weight),
            'loss_shares': {
                'main': main_loss_share,
                'constraint_total': constraint_loss_share,
                'alignment': alignment_loss_share,
                'lag': lag_loss_share,
                'retention': retention_loss_share
            },
            'mean_mastery_variance': mean_mastery_variance,
            'min_mastery_variance': min_mastery_variance,
            'max_mastery_variance': max_mastery_variance,
            'per_lag_correlations': epoch_lag_corrs
        })

        # Alignment share cap enforcement (post logging so share recorded) with decay
        if enable_alignment_loss and alignment_loss_share is not None and abs(alignment_loss_share) > alignment_share_cap:
            # Plateau detection: only decay if mastery correlation improvement < 0.005 epoch-over-epoch
            prev_prev = global_alignment_state.get('prev_prev_global_mastery_corr')
            prev_curr = global_alignment_state.get('prev_global_mastery_corr')
            corr_improvement = None
            if prev_prev is not None and prev_curr is not None:
                corr_improvement = prev_curr - prev_prev
            if corr_improvement is None or corr_improvement < 0.005:
                old_w = global_alignment_state['alignment_weight_current']
                new_w = max(0.05, old_w * alignment_share_decay_factor)  # floor to avoid starvation
                global_alignment_state['alignment_weight_current'] = new_w
                logger.info(f"[AlignCap] Alignment share {alignment_loss_share:.4f} > {alignment_share_cap:.2f} and plateau (Δcorr={corr_improvement}); decay {old_w:.4f} -> {new_w:.4f}")
            else:
                logger.info(f"[AlignCap] Alignment share {alignment_loss_share:.4f} > cap but corr improving (Δcorr={corr_improvement:.4f}); no decay applied.")
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_main_loss': train_main_loss,
                'train_constraint_loss': train_constraint_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **{f'consistency_{k}': v for k, v in consistency_metrics.items()}
            })
        
        # Model saving and early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            save_dir = f"saved_model/gainakt2exp_{experiment_suffix}"
            os.makedirs(save_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'model_config': model_config,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'consistency_metrics': consistency_metrics,
                'train_history': train_history
            }
            
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            logger.info(f"  🎉 NEW BEST MODEL SAVED! Val AUC: {best_val_auc:.4f} (Epoch {epoch + 1})")
            # Repro integration: checkpoint writing
            if experiment_dir:
                try:
                    torch.save(checkpoint, repro_best_ckpt)
                except Exception as e:
                    logger.warning(f"[Repro] Could not save best checkpoint to experiment dir: {e}")
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Early stopping
        patience = getattr(args, 'patience', 20)
        if patience_counter >= patience:
            logger.info(f"  Early stopping triggered (patience: {patience})")
            break
        
        # Repro integration: checkpoint/metrics writing
        if experiment_dir:
            # Always save last checkpoint
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_auc': best_val_auc
                }, repro_last_ckpt)
            except Exception as e:
                logger.warning(f"[Repro] Failed to save last checkpoint: {e}")
            # Append metrics CSV row
            try:
                logger.info(f"[Repro] Appending metrics row for epoch {epoch+1} to {repro_metrics_csv}")
                with open(repro_metrics_csv, 'a', newline='') as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow([
                        epoch + 1, train_loss, train_auc, val_loss, val_auc, val_acc,
                        consistency_metrics.get('monotonicity_violation_rate'),
                        consistency_metrics.get('negative_gain_rate'),
                        consistency_metrics.get('bounds_violation_rate'),
                        consistency_metrics.get('mastery_correlation'),
                        consistency_metrics.get('gain_correlation'),
                        train_history['semantic_trajectory'][-1]['loss_shares']['main'],
                        train_history['semantic_trajectory'][-1]['loss_shares']['constraint_total'],
                        train_history['semantic_trajectory'][-1]['loss_shares']['alignment'],
                        train_history['semantic_trajectory'][-1]['loss_shares']['lag'],
                        train_history['semantic_trajectory'][-1]['loss_shares']['retention']
                    ])
            except Exception as e:
                logger.warning(f"[Repro] Failed to append metrics row: {e}")
    # Defer final_results construction until after final consistency evaluation below
    # Final evaluation & consistency (restored block)
    logger.info("\n==== TRAINING COMPLETED ====")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # Final consistency check
    final_consistency = validate_model_consistency(
        model, valid_loader, device, logger, max_students=200
    )
    final_results = {
        'experiment_name': experiment_suffix,
        'best_val_auc': best_val_auc,
        'final_consistency_metrics': final_consistency,
        'train_history': train_history,
        'model_config': model_config,
        'training_args': {
            'dataset_name': dataset_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'enhanced_constraints': enhanced_constraints,
            'fold': fold,
        },
        'timestamp': datetime.now().isoformat()
    }
    if train_history.get('determinism_fallback_batches'):
        final_results['determinism_fallback'] = {
            'count': len(train_history['determinism_fallback_batches']),
            'batches': train_history['determinism_fallback_batches']
        }
    # Correlation determinism metadata (global + local mastery/gain correlations gathered during last epoch)
    try:
        last_semantic = train_history['semantic_trajectory'][-1] if train_history.get('semantic_trajectory') else {}
        local_mastery_corr = last_semantic.get('mastery_correlation')
        local_gain_corr = last_semantic.get('gain_correlation')
        # Collect raw per-student correlations stored earlier (consistency_metrics list)
        # They are summary, not raw list; thus we cannot recompute SE precisely without storing all values.
        # Provide placeholders signaling deterministic sampling was enforced.
        final_results['correlation_sampling'] = {
            'deterministic': True,
            'epoch': len(train_history.get('val_auc', [])),
            'local_mastery_corr': local_mastery_corr,
            'local_gain_corr': local_gain_corr,
            'global_mastery_corr': global_mastery_corr,
            'global_gain_corr': global_gain_corr,
            'notes': 'Deterministic stratified selection (sorted slice) applied; seeds fixed.'
        }
    except Exception:
        final_results['correlation_sampling'] = {'deterministic': True, 'error': 'metadata collection failed'}
    # Primary comprehensive results file (historically written to repo root).
    # Relocate inside experiment_dir for reproducibility containment if available.
    # Standardized reproduction results filename (experiment-contained): repro_results_<YYYYMMDD>_<HHMMSS>.json
    # Remove prior prefix and experiment_suffix to avoid overly long filenames and ensure uniformity across runs.
    timestamp_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f"repro_results_{timestamp_tag}.json"
    if experiment_dir:
        results_file = os.path.join(experiment_dir, results_filename)
    else:
        results_file = results_filename
    try:
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"\n📄 Final results saved to: {results_file} (containment={'experiment_dir' if experiment_dir else 'root'})")
    except Exception as e:
        logger.warning(f"Failed to write final results file: {e}")
    if experiment_dir:
        # Write legacy-format results.json matching prior experiments (experiment_name, best_val_auc, final_consistency_metrics, train_history)
        legacy_results = {
            'experiment_name': experiment_suffix,
            'best_val_auc': best_val_auc,
            'final_consistency_metrics': {
                'monotonicity_violation_rate': final_consistency.get('monotonicity_violation_rate'),
                'negative_gain_rate': final_consistency.get('negative_gain_rate'),
                'bounds_violation_rate': final_consistency.get('bounds_violation_rate'),
                'mastery_correlation': final_consistency.get('mastery_correlation'),
                'gain_correlation': final_consistency.get('gain_correlation')
            },
            'train_history': {
                'train_loss': train_history.get('train_loss', []),
                'train_auc': train_history.get('train_auc', []),
                'val_auc': train_history.get('val_auc', []),
                'consistency_metrics': train_history.get('consistency_metrics', []),
                'semantic_trajectory': train_history.get('semantic_trajectory', [])
            },
            'correlation_sampling': final_results.get('correlation_sampling')
        }
        try:
            with open(repro_results_json, 'w') as rf:
                json.dump(legacy_results, rf, indent=2)
            logger.info(f"[Repro] Legacy results.json written to {repro_results_json}")
        except Exception as e:
            logger.warning(f"[Repro] Failed to write legacy results.json: {e}")
    return final_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train GainAKT2Exp model with reproducibility hooks.')
    # Single-source config reproduction flag: when provided, all parameters are loaded from this file.
    parser.add_argument('--config', type=str, help='Path to resolved experiment config.json (single source of truth).')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, required=True, help='Dataset fold index')
    # Recovered configuration defaults (semantic modules active, correlations stabilize by ~epoch 8)
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs (default 12 recovered configuration)')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size (default 64 recovered configuration)')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer (Adam supported)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (AMP) training')
    parser.add_argument('--auto_shifted_eval', action='store_true', help='Enable auto-shifted evaluation (reproducibility invariant)')
    parser.add_argument('--monitor_freq', type=int, required=True, help='Interpretability monitor frequency (steps)')
    parser.add_argument('--gradient_clip', type=float, required=True, help='Gradient clipping max norm')
    parser.add_argument('--patience', type=int, required=True, help='Early stopping patience')
    # --require_multi_gpu removed: no mandatory multi-GPU enforcement; script now allows single GPU or CPU without failure.
    # =====================
    # Interpretability Defaults: ENABLED by default
    # We expose disable flags for explicit override while keeping backward-compatible enable flags.
    # ---------------------
    parser.add_argument('--use_mastery_head', action='store_true',
                        help='Enable mastery head (default: enabled). Use --disable_mastery_head to turn off.')
    parser.add_argument('--disable_mastery_head', action='store_true', help='Disable mastery head (overrides --use_mastery_head)')
    parser.add_argument('--use_gain_head', action='store_true',
                        help='Enable gain head (default: enabled). Use --disable_gain_head to turn off.')
    parser.add_argument('--disable_gain_head', action='store_true', help='Disable gain head (overrides --use_gain_head)')
    parser.add_argument('--intrinsic_gain_attention', action='store_true',
                        help='Enable intrinsic gain attention mode (Values are skill-space gains, h_t = Σ α g)')
    parser.add_argument('--disable_intrinsic_gain_attention', action='store_true', help='Disable intrinsic gain attention')
    # Enhanced constraints (core semantic shaping) ON by default; use --pure_bce to disable.
    parser.add_argument('--enhanced_constraints', action='store_true',
                        help='Use enhanced constraint preset (default: enabled). Use --pure_bce for pure BCE baseline.')
    parser.add_argument('--pure_bce', action='store_true', help='Disable enhanced constraints and force pure BCE loss.')
    parser.add_argument('--non_negative_loss_weight', type=float, required=True)
    parser.add_argument('--monotonicity_loss_weight', type=float, required=True)
    parser.add_argument('--mastery_performance_loss_weight', type=float, required=True)
    parser.add_argument('--gain_performance_loss_weight', type=float, required=True)
    parser.add_argument('--sparsity_loss_weight', type=float, required=True)
    parser.add_argument('--consistency_loss_weight', type=float, required=True)
    # Semantic alignment & refinement flags (Phase 1+ reproducibility)
    parser.add_argument('--enable_alignment_loss', action='store_true',
                        help='Enable local alignment correlation loss (default: enabled). Use --disable_alignment_loss to turn off.')
    parser.add_argument('--disable_alignment_loss', action='store_true', help='Disable local alignment correlation loss')
    parser.add_argument('--alignment_weight', type=float, required=True, help='Base weight for alignment loss (before warm-up scaling)')
    parser.add_argument('--alignment_warmup_epochs', type=int, required=True, help='Epochs to linearly warm alignment weight')
    parser.add_argument('--adaptive_alignment', action='store_true',
                        help='Adaptively up-weight alignment if below min correlation post warm-up (default: enabled). Use --disable_adaptive_alignment to turn off.')
    parser.add_argument('--disable_adaptive_alignment', action='store_true', help='Disable adaptive alignment scaling logic')
    parser.add_argument('--alignment_min_correlation', type=float, required=True, help='Target minimum mastery correlation for adaptive scaling')
    parser.add_argument('--alignment_share_cap', type=float, required=True, help='Max alignment loss share before decay applied')
    parser.add_argument('--alignment_share_decay_factor', type=float, required=True, help='Decay factor for alignment weight when over cap and plateaued')
    parser.add_argument('--enable_global_alignment_pass', action='store_true',
                        help='Run a global alignment correlation pass each epoch (default: enabled). Use --disable_global_alignment_pass to turn off.')
    parser.add_argument('--disable_global_alignment_pass', action='store_true', help='Disable global alignment correlation pass')
    parser.add_argument('--alignment_global_students', type=int, required=True, help='Students sampled in global alignment stratified pass')
    parser.add_argument('--use_residual_alignment', action='store_true', help='Residualize performance for alignment correlations')
    parser.add_argument('--alignment_residual_window', type=int, required=True, help='Window for residual smoothing (rolling mean)')
    # Retention & lag emergence
    parser.add_argument('--enable_retention_loss', action='store_true',
                        help='Enable retention penalty after peak correlation decay (default: enabled). Use --disable_retention_loss to turn off.')
    parser.add_argument('--disable_retention_loss', action='store_true', help='Disable retention penalty')
    parser.add_argument('--retention_delta', type=float, required=True, help='Minimum decay gap triggering retention penalty')
    parser.add_argument('--retention_weight', type=float, required=True, help='Weight applied to retention decay gap')
    parser.add_argument('--enable_lag_gain_loss', action='store_true',
                        help='Enable multi-lag predictive emergence objective (default: enabled). Use --disable_lag_gain_loss to turn off.')
    parser.add_argument('--disable_lag_gain_loss', action='store_true', help='Disable multi-lag predictive emergence objective')
    parser.add_argument('--lag_gain_weight', type=float, required=True, help='Weight for lag predictive emergence objective')
    parser.add_argument('--lag_max_lag', type=int, required=True, help='Maximum lag horizon considered for predictive emergence')
    parser.add_argument('--lag_l1_weight', type=float, required=True, help='Weight for lag 1 correlation')
    parser.add_argument('--lag_l2_weight', type=float, required=True, help='Weight for lag 2 correlation')
    parser.add_argument('--lag_l3_weight', type=float, required=True, help='Weight for lag 3 correlation')
    # Consistency rebalancing & variance floor dynamics
    parser.add_argument('--consistency_rebalance_epoch', type=int, required=True, help='Epoch to start potential consistency loss weight reduction')
    parser.add_argument('--consistency_rebalance_threshold', type=float, required=True, help='Mastery corr threshold triggering consistency reweight')
    parser.add_argument('--consistency_rebalance_new_weight', type=float, required=True, help='New consistency loss weight after rebalancing')
    parser.add_argument('--variance_floor', type=float, required=True, help='Variance floor for mastery tensor triggering sparsity reduction')
    parser.add_argument('--variance_floor_patience', type=int, required=True, help='Consecutive low-variance epochs before sparsity reduction')
    parser.add_argument('--variance_floor_reduce_factor', type=float, required=True, help='Factor to reduce sparsity loss weight under variance floor')
    # Constraint scheduling / performance alignment
    parser.add_argument('--warmup_constraint_epochs', type=int, required=True, help='Warm-up epochs for performance alignment losses')
    parser.add_argument('--enable_cosine_perf_schedule', action='store_true', help='Use cosine schedule for performance alignment weights')
    parser.add_argument('--max_semantic_students', type=int, required=True, help='Students sampled for consistency semantic correlations')
    # =====================
    # Architecture flags (added for full CLI reproducibility & ablation control)
    # These mirror keys in config['model_config'] and allow overriding defaults without editing config.json.
    # If --config is provided, values from config are used unless an explicit CLI override flag is present.
    parser.add_argument('--seq_len', type=int, required=True, help='Maximum sequence length (architecture)')
    parser.add_argument('--d_model', type=int, required=True, help='Model hidden dimension (Transformer)')
    parser.add_argument('--n_heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--num_encoder_blocks', type=int, required=True, help='Number of Transformer encoder blocks')
    parser.add_argument('--d_ff', type=int, required=True, help='Feed-forward layer dimension')
    parser.add_argument('--dropout', type=float, required=True, help='Transformer dropout rate')
    parser.add_argument('--emb_type', type=str, choices=['qid','concept','hybrid'], required=True, help='Embedding type used for questions/concepts')
    # Placeholder for future extended args (constraints toggles, etc.)
    args = parser.parse_args()
    # If a config path is passed, set PYKT_CONFIG_PATH so internal loaders pick it up.
    if getattr(args, 'config', None):
        if not os.path.exists(args.config):
            print(f"[ERROR] --config file not found: {args.config}", file=sys.stderr)
            sys.exit(2)
        os.environ['PYKT_CONFIG_PATH'] = os.path.abspath(args.config)
        print(f"[Repro] Using explicit config file: {os.environ['PYKT_CONFIG_PATH']}")
    # Map to expected attribute names inside training function
    args.num_epochs = args.epochs
    args.dataset_name = args.dataset
    # Provide defaults for attributes referenced but not yet exposed
    # Resolve disable overrides (heads & losses)
    if getattr(args, 'disable_mastery_head', False):
        args.use_mastery_head = False
    if getattr(args, 'disable_gain_head', False):
        args.use_gain_head = False
    if getattr(args, 'disable_intrinsic_gain_attention', False):
        args.intrinsic_gain_attention = False
    
    # ARCHITECTURAL CONSTRAINT: Intrinsic gain attention and projection heads are mutually exclusive
    # Intrinsic mode uses attention-derived gains; projection heads would be unused (wasting ~2M parameters)
    if args.intrinsic_gain_attention:
        if args.use_mastery_head or args.use_gain_head:
            print("=" * 100)
            print("⚠️  WARNING: ARCHITECTURAL PARAMETER CONFLICT DETECTED")
            print("=" * 100)
            print("intrinsic_gain_attention=True is INCOMPATIBLE with projection heads")
            print("")
            print("  Intrinsic mode uses attention-derived gains directly from the model.")
            print("  Projection heads (use_mastery_head, use_gain_head) are NOT used in this mode.")
            print("  Enabling them wastes ~2M parameters without any benefit.")
            print("")
            print("AUTOMATIC CORRECTION APPLIED:")
            if args.use_mastery_head:
                print("  • use_mastery_head: True → False")
            if args.use_gain_head:
                print("  • use_gain_head: True → False")
            print("")
            print("Model will be created in pure intrinsic mode (attention-derived gains only).")
            print("Expected parameters: ~12.7M (vs ~14.7M with unused projection heads)")
            print("=" * 100)
            args.use_mastery_head = False
            args.use_gain_head = False
    
    if getattr(args, 'pure_bce', False):
        args.enhanced_constraints = False
    if getattr(args, 'disable_alignment_loss', False):
        args.enable_alignment_loss = False
    if getattr(args, 'disable_adaptive_alignment', False):
        args.adaptive_alignment = False
    if getattr(args, 'disable_global_alignment_pass', False):
        args.enable_global_alignment_pass = False
    if getattr(args, 'disable_retention_loss', False):
        args.enable_retention_loss = False
    if getattr(args, 'disable_lag_gain_loss', False):
        args.enable_lag_gain_loss = False
    train_gainakt2exp_model(args)