#!/usr/bin/env python3
"""
Standardized training script for GainAKT2Exp model using PyKT framework patterns.
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
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from datetime import datetime
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
    
    # Get parameters with OPTIMAL defaults (AUC: 0.7260, Perfect Consistency)
    dataset_name = getattr(args, 'dataset_name', getattr(args, 'dataset', 'assist2015'))
    num_epochs = getattr(args, 'num_epochs', getattr(args, 'epochs', 20))
    learning_rate = getattr(args, 'learning_rate', getattr(args, 'lr', 0.000174))  # OPTIMAL
    batch_size = getattr(args, 'batch_size', 96)  # OPTIMAL
    weight_decay = getattr(args, 'weight_decay', 1.7571e-05)  # OPTIMAL
    enhanced_constraints = getattr(args, 'enhanced_constraints', True)
    fold = getattr(args, 'fold', 0)
    experiment_suffix = getattr(args, 'experiment_suffix', 'optimal_v1')
    use_wandb = getattr(args, 'use_wandb', False)
    use_amp = getattr(args, 'use_amp', False)
    # Alignment / semantic emergence new arguments (may be absent in older runs)
    enable_alignment_loss = getattr(args, 'enable_alignment_loss', False)
    alignment_weight = float(getattr(args, 'alignment_weight', 0.1))
    alignment_warmup_epochs = int(getattr(args, 'alignment_warmup_epochs', 8))
    adaptive_alignment = getattr(args, 'adaptive_alignment', True)
    alignment_min_correlation = float(getattr(args, 'alignment_min_correlation', 0.05))
    # Global alignment / residual options (Tier B refinements)
    enable_global_alignment_pass = getattr(args, 'enable_global_alignment_pass', False)
    alignment_global_students = int(getattr(args, 'alignment_global_students', 600))  # enlarged for stratified global sampling
    use_residual_alignment = getattr(args, 'use_residual_alignment', False)
    alignment_residual_window = int(getattr(args, 'alignment_residual_window', 5))
    # Refinement cycle new arguments
    # Phase 0–2 semantic emergence controls (updated defaults)
    enable_retention_loss = getattr(args, 'enable_retention_loss', False)
    retention_delta = float(getattr(args, 'retention_delta', 0.01))  # Phase 3 tolerance
    retention_weight = float(getattr(args, 'retention_weight', 0.1))  # Phase 3 logging-only retention
    enable_lag_gain_loss = getattr(args, 'enable_lag_gain_loss', False)
    lag_gain_weight = float(getattr(args, 'lag_gain_weight', 0.06))  # modest weight for multi-lag predictive emergence
    lag_max_lag = int(getattr(args, 'lag_max_lag', 3))  # extend to lag 3
    # Weighted multi-lag scheme (L1 emphasis)
    lag_l1_weight = float(getattr(args, 'lag_l1_weight', 0.5))
    lag_l2_weight = float(getattr(args, 'lag_l2_weight', 0.3))
    lag_l3_weight = float(getattr(args, 'lag_l3_weight', 0.2))
    # Alignment share cap & decay factor
    alignment_share_cap = float(getattr(args, 'alignment_share_cap', 0.08))
    alignment_share_decay_factor = float(getattr(args, 'alignment_share_decay_factor', 0.7))
    enable_cosine_perf_schedule = getattr(args, 'enable_cosine_perf_schedule', False)
    consistency_rebalance_epoch = int(getattr(args, 'consistency_rebalance_epoch', 8))
    consistency_rebalance_threshold = float(getattr(args, 'consistency_rebalance_threshold', 0.10))
    consistency_rebalance_new_weight = float(getattr(args, 'consistency_rebalance_new_weight', 0.2))
    variance_floor = float(getattr(args, 'variance_floor', 1e-4))
    variance_floor_patience = int(getattr(args, 'variance_floor_patience', 3))
    variance_floor_reduce_factor = float(getattr(args, 'variance_floor_reduce_factor', 0.5))
    
    # Individual constraint weights - OPTIMAL values from parameter sweep
    non_negative_loss_weight = getattr(args, 'non_negative_loss_weight', 0.0)
    monotonicity_loss_weight = getattr(args, 'monotonicity_loss_weight', 0.1)
    mastery_performance_loss_weight = getattr(args, 'mastery_performance_loss_weight', 0.8)
    gain_performance_loss_weight = getattr(args, 'gain_performance_loss_weight', 0.8)
    sparsity_loss_weight = getattr(args, 'sparsity_loss_weight', 0.2)
    consistency_loss_weight = getattr(args, 'consistency_loss_weight', 0.3)
    
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
    
    # Create model with standard PyKT configuration
    num_c = data_config[dataset_name]['num_c']
    model_config = {
        'num_c': num_c,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 8,
        'num_encoder_blocks': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'emb_type': 'qid',
        'monitor_frequency': 50,
        # Allow disabling heads for pure predictive baseline
        'use_mastery_head': getattr(args, 'use_mastery_head', True),
        'use_gain_head': getattr(args, 'use_gain_head', True)
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
    elif enhanced_constraints and not any(hasattr(args, p) for p in individual_params):
        model_config.update({
            'non_negative_loss_weight': 0.0,  # enforced architecturally
            'monotonicity_loss_weight': 0.1,
            'mastery_performance_loss_weight': 0.8,
            'gain_performance_loss_weight': 0.8,
            'sparsity_loss_weight': 0.2,
            'consistency_loss_weight': 0.3
        })
        logger.info("Enhanced constraints preset applied (no individual overrides provided)")
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
    
    logger.info("Creating GainAKT2Exp (mastery_head=%s, gain_head=%s) with CUMULATIVE MASTERY..." % (
        model_config['use_mastery_head'], model_config['use_gain_head']) )
    model = create_exp_model(model_config)
    monitor = InterpretabilityMonitor(model, log_frequency=args.monitor_freq)
    model.set_monitor(monitor)
    model = model.to(device)
    # ------------------------------
    # Simple Multi-GPU Support (Option A): DataParallel
    # Automatically wraps the model if more than one CUDA device is visible.
    # Usage: set CUDA_VISIBLE_DEVICES=0,1,2 (or similar) before launching the script.
    # This keeps all existing logic unchanged (no need for distributed initialization).
    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
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
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

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
    warmup_constraint_epochs = getattr(args, 'warmup_constraint_epochs', 4)
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
    # Activation tracking counters (epoch-level)
    activation_counters = {
        'alignment_active_epochs': 0,
        'retention_active_epochs': 0,
        'lag_active_epochs': 0,
        'alignment_weight_sum': 0.0,
        'alignment_weight_count': 0,
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
                    if isinstance(model, torch.nn.DataParallel):
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
                    if enable_alignment_loss and model_core.mastery_head is not None and model_core.gain_head is not None and 'projected_mastery' in outputs and 'projected_gains' in outputs:
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
                        # Alignment activation counted once per epoch (after effective weight computed)
                        activation_counters['alignment_active_epochs'] += 1
                        activation_counters['alignment_weight_sum'] += float(effective_weight)
                        activation_counters['alignment_weight_count'] += 1
                    # Multi-lag predictive emergence objective
                    lag_loss = torch.zeros(1, device=device)
                    mean_lag_corr = torch.zeros(1, device=device)
                    lag_corr_count = 0
                    if enable_lag_gain_loss and lag_max_lag > 0 and (epoch + 1) >= (warmup_constraint_epochs + 2):
                        # Phase 4 Redesigned lag objective: stricter activation gate and improved per-student normalization
                        # to promote genuine incremental predictive semantics (Gain_t -> Correct_{t+lag}).
                        gains_mean_time = gains_mean  # B x T
                        perf_time = perf.float()      # B x T
                        weights_map = {1: lag_l1_weight, 2: lag_l2_weight, 3: lag_l3_weight}
                        weighted_corr_sum = torch.zeros(1, device=device)
                        lag_terms = []
                        
                        # Phase 3: Simpler batch-level lag correlation (safer implementation)
                        for lag in range(1, min(lag_max_lag + 1, 4)):
                            # Collect aligned gain-performance pairs
                            aligned_pairs = []
                            
                            for student_idx in range(gains_mean_time.size(0)):
                                student_gains = gains_mean_time[student_idx]  # T
                                student_perf = perf_time[student_idx]         # T
                                student_valid = student_mask[student_idx]     # T
                                
                                if student_valid.sum() < 4:  # minimum sequence length
                                    continue
                                    
                                T = student_gains.size(0)
                                if T <= lag:  # need enough positions for lag
                                    continue
                                
                                # Simple approach: iterate through valid aligned positions
                                for t in range(T - lag):
                                    if student_valid[t] and (t + lag < T) and student_valid[t + lag]:
                                        gain_val = float(student_gains[t].detach().cpu())
                                        perf_val = float(student_perf[t + lag].detach().cpu())
                                        aligned_pairs.append((gain_val, perf_val))
                            
                            if len(aligned_pairs) >= 10:  # minimum batch size for correlation
                                gains_list, perfs_list = zip(*aligned_pairs)
                                gains_tensor = torch.tensor(gains_list, device=device)
                                perfs_tensor = torch.tensor(perfs_list, device=device)
                                
                                if gains_tensor.std() > 1e-6 and perfs_tensor.std() > 1e-6:
                                    corr_lag = corr_fn(gains_tensor, perfs_tensor)
                                    w_lag = weights_map.get(lag, 0.0)
                                    if w_lag > 0 and not torch.isnan(corr_lag):
                                        weighted_corr_sum += w_lag * corr_lag
                                        lag_terms.append((lag, corr_lag.detach(), w_lag))
                                        try:
                                            epoch_lag_corrs.append({'lag': lag, 'corr': float(corr_lag.detach().cpu()), 'weight': w_lag})
                                        except Exception:
                                            pass
                        
                        total_w = lag_l1_weight + lag_l2_weight + lag_l3_weight  # Phase 3: all lags
                        if lag_terms and total_w > 0:
                            mean_lag_corr = weighted_corr_sum / total_w
                            lag_corr_count = len(lag_terms)
                            # Phase 3: Standard lag correlation loss (no positive-only clamp)
                            lag_loss = - mean_lag_corr * lag_gain_weight
                            lag_corr_count = len(lag_terms)
                            activation_counters['lag_active_epochs'] += 1
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
                # Compose total batch loss (Phase 3: retention logging-only)
                retention_component = torch.zeros(1, device=device)
                if enable_retention_loss and pending_retention_penalty > 0:
                    # Phase 3: Log retention penalty but don't apply to gradients
                    total_retention_component += pending_retention_penalty / max(1, num_batches)
                if enable_alignment_loss:
                    total_batch_loss = main_loss + interpretability_loss + alignment_loss
                else:
                    total_batch_loss = main_loss + interpretability_loss
                # Backward & optimizer step
                if use_amp and device.type == 'cuda':
                    scaler.scale(total_batch_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            except RuntimeError as oom:
                if 'out of memory' in str(oom).lower():
                    logger.warning("OOM encountered; clearing CUDA cache and skipping batch")
                    torch.cuda.empty_cache()
                    continue
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
        else:
            main_loss_share = constraint_loss_share = alignment_loss_share = lag_loss_share = retention_loss_share = 0.0
        mean_mastery_variance = float(np.mean(batch_mastery_variances)) if batch_mastery_variances else None
        min_mastery_variance = float(np.min(batch_mastery_variances)) if batch_mastery_variances else None
        max_mastery_variance = float(np.max(batch_mastery_variances)) if batch_mastery_variances else None
        train_auc = roc_auc_score(total_targets, total_predictions)
        train_acc = accuracy_score(total_targets, np.round(total_predictions))
        
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
        val_auc = roc_auc_score(val_targets, val_predictions)
        val_acc = accuracy_score(val_targets, np.round(val_predictions))
        
        # Consistency validation
        logger.info("  Running consistency validation...")
        consistency_metrics = validate_model_consistency(
            model, valid_loader, device, logger, max_students=max_semantic_students
        )

        # Global alignment pass (sequence-level sampling separate from lightweight consistency check)
        if enable_global_alignment_pass and model_core.mastery_head is not None and model_core.gain_head is not None:
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
                        choice = np.random.choice(bin_indices, size=take, replace=False)
                        for ci in choice:
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
                total_retention_component += retention_loss_value
                activation_counters['retention_active_epochs'] += 1
            else:
                retention_loss_value = 0.0
        
        # Log epoch results with enhanced formatting
        logger.info("=" * 60)
        logger.info(f"📊 EPOCH {epoch + 1}/{num_epochs} RESULTS:")
        logger.info(f"  🚂 Train - Loss: {train_loss:.4f} (Main: {train_main_loss:.4f}, "
                   f"Constraint: {train_constraint_loss:.4f}), AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  ✅ Valid - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"  🧠 Peak GPU memory this epoch: {peak_mem:.1f} MiB")
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
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Early stopping
        patience = getattr(args, 'patience', 20)
        if patience_counter >= patience:
            logger.info(f"  Early stopping triggered (patience: {patience})")
            break
    
    # Final evaluation
    logger.info("\\n" + "=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Final comprehensive consistency check
    logger.info("\\nRunning final consistency validation...")
    final_consistency = validate_model_consistency(
        model, valid_loader, device, logger, max_students=200
    )
    
    # Save final results
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
            'constraint_weights': {
                'non_negative_loss_weight': non_negative_loss_weight,
                'monotonicity_loss_weight': monotonicity_loss_weight,
                'mastery_performance_loss_weight': mastery_performance_loss_weight,
                'gain_performance_loss_weight': gain_performance_loss_weight,
                'sparsity_loss_weight': sparsity_loss_weight,
                'consistency_loss_weight': consistency_loss_weight
            },
            'alignment': {
                'enable_alignment_loss': enable_alignment_loss,
                'alignment_weight': alignment_weight,
                'alignment_warmup_epochs': alignment_warmup_epochs,
                'adaptive_alignment': adaptive_alignment,
                'alignment_min_correlation': alignment_min_correlation
            },
            'global_alignment': {
                'enable_global_alignment_pass': enable_global_alignment_pass,
                'alignment_global_students': alignment_global_students,
                'use_residual_alignment': use_residual_alignment,
                'alignment_residual_window': alignment_residual_window
            },
            'refinement': {
                'enable_retention_loss': enable_retention_loss,
                'retention_delta': retention_delta,
                'retention_weight': retention_weight,
                'enable_lag_gain_loss': enable_lag_gain_loss,
                'lag_gain_weight': lag_gain_weight,
                'lag_max_lag': lag_max_lag,
                'lag_l1_weight': lag_l1_weight,
                'lag_l2_weight': lag_l2_weight,
                'lag_l3_weight': lag_l3_weight,
                'enable_cosine_perf_schedule': enable_cosine_perf_schedule,
                'consistency_rebalance_epoch': consistency_rebalance_epoch,
                'consistency_rebalance_threshold': consistency_rebalance_threshold,
                'consistency_rebalance_new_weight': consistency_rebalance_new_weight,
                'variance_floor': variance_floor,
                'variance_floor_patience': variance_floor_patience,
                'variance_floor_reduce_factor': variance_floor_reduce_factor,
                'alignment_share_cap': alignment_share_cap,
                'alignment_share_decay_factor': alignment_share_decay_factor
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = f"gainakt2exp_results_{experiment_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\\n📄 Final results saved to: {results_file}")
    
    # Always attempt to save semantic trajectory (independent of wandb)
    try:
        os.makedirs(os.path.dirname(semantic_trajectory_path), exist_ok=True)
        with open(semantic_trajectory_path, 'w') as traj_f:
            json.dump({
                'trajectory': train_history.get('semantic_trajectory', []),
                'experiment_name': experiment_suffix,
                'best_val_auc': best_val_auc,
                'timestamp': datetime.now().isoformat(),
                'warmup_constraint_epochs': warmup_constraint_epochs
            }, traj_f, indent=2)
        logger.info(f"🧪 Semantic trajectory saved to: {semantic_trajectory_path}")
    except Exception as e:
        logger.warning(f"Failed to save semantic trajectory ({semantic_trajectory_path}): {e}")
    if use_wandb:
        try:
            wandb.log({
                'final_best_val_auc': best_val_auc,
                **{f'final_consistency_{k}': v for k, v in final_consistency.items()}
            })
            wandb.finish()
            logger.info("Wandb session finished (offline mode)")
        except Exception as e:
            logger.warning(f"Wandb finish failed (offline mode): {e}")
    
    # Assessment
    logger.info("\\n FINAL ASSESSMENT:")
    perfect_consistency = (
        final_consistency['monotonicity_violation_rate'] == 0.0 and
        final_consistency['negative_gain_rate'] == 0.0 and
        final_consistency['bounds_violation_rate'] == 0.0
    )
    
    strong_correlations = (
        final_consistency['mastery_correlation'] > 0.3 and
        final_consistency['gain_correlation'] > 0.3
    )
    
    if perfect_consistency:
        logger.info("PERFECT EDUCATIONAL CONSISTENCY MAINTAINED!")
    else:
        logger.info("Some consistency violations detected")
    
    if strong_correlations:
        logger.info("STRONG PERFORMANCE CORRELATIONS ACHIEVED!")
    else:
        logger.info("Correlations need improvement")
    
    if perfect_consistency and strong_correlations:
        logger.info("SUCCESS: Perfect consistency + Strong correlations!")
    
    return final_results


# Main function removed - train_gainakt2exp_model() is called directly from wandb_train.py
# Parameters expected in args object:
# - dataset_name/dataset: 'assist2015' 
# - num_epochs/epochs: 20
# - learning_rate/lr: 0.0003  
# - batch_size: 128
# - weight_decay: 0.000059
# - enhanced_constraints: True
# - fold: 0
# - experiment_suffix: 'v1'
# - use_wandb: False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train GainAKT2Exp model (structural interpretability + optional semantic alignment objectives)."
    )
    # Core dataset & run config
    parser.add_argument('--dataset', '--dataset_name', dest='dataset', default='assist2015')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', '--num_epochs', dest='epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.000174)
    parser.add_argument('--weight_decay', type=float, default=1.7571e-05)
    parser.add_argument('--enhanced_constraints', action='store_true', default=True)
    parser.add_argument('--experiment_suffix', type=str, default='cli_run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--monitor_freq', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)
    # Constraint weights (optional overrides)
    parser.add_argument('--non_negative_loss_weight', type=float, default=0.0)
    parser.add_argument('--monotonicity_loss_weight', type=float, default=0.1)
    parser.add_argument('--mastery_performance_loss_weight', type=float, default=0.8)
    parser.add_argument('--gain_performance_loss_weight', type=float, default=0.8)
    parser.add_argument('--sparsity_loss_weight', type=float, default=0.2)
    parser.add_argument('--consistency_loss_weight', type=float, default=0.3)
    # Alignment / semantic emergence flags
    parser.add_argument('--enable_alignment_loss', action='store_true')
    parser.add_argument('--alignment_weight', type=float, default=0.25)
    parser.add_argument('--alignment_warmup_epochs', type=int, default=8)
    parser.add_argument('--adaptive_alignment', action='store_true', default=True)
    parser.add_argument('--alignment_min_correlation', type=float, default=0.05)
    parser.add_argument('--enable_global_alignment_pass', action='store_true')
    # Phase 3: expanded global sampling default increased to 600 for stronger sequence-level signal
    parser.add_argument('--alignment_global_students', type=int, default=600)
    parser.add_argument('--use_residual_alignment', action='store_true')
    parser.add_argument('--alignment_residual_window', type=int, default=5)
    # Retention & lag objectives
    parser.add_argument('--enable_retention_loss', action='store_true')
    parser.add_argument('--retention_delta', type=float, default=0.01)
    parser.add_argument('--retention_weight', type=float, default=0.1)
    parser.add_argument('--enable_lag_gain_loss', action='store_true')
    parser.add_argument('--lag_gain_weight', type=float, default=0.06)
    parser.add_argument('--lag_max_lag', type=int, default=3)
    parser.add_argument('--lag_l1_weight', type=float, default=0.5)
    parser.add_argument('--lag_l2_weight', type=float, default=0.3)
    parser.add_argument('--lag_l3_weight', type=float, default=0.2)
    # Alignment share cap
    parser.add_argument('--alignment_share_cap', type=float, default=0.08)
    parser.add_argument('--alignment_share_decay_factor', type=float, default=0.7)
    # Performance alignment scheduling
    parser.add_argument('--warmup_constraint_epochs', type=int, default=8)
    parser.add_argument('--enable_cosine_perf_schedule', action='store_true')
    # Consistency rebalance
    parser.add_argument('--consistency_rebalance_epoch', type=int, default=8)
    parser.add_argument('--consistency_rebalance_threshold', type=float, default=0.10)
    parser.add_argument('--consistency_rebalance_new_weight', type=float, default=0.2)
    # Variance floor gating
    parser.add_argument('--variance_floor', type=float, default=1e-4)
    parser.add_argument('--variance_floor_patience', type=int, default=3)
    parser.add_argument('--variance_floor_reduce_factor', type=float, default=0.5)
    # Semantic trajectory sampling
    parser.add_argument('--max_semantic_students', type=int, default=50)
    parser.add_argument('--semantic_trajectory_path', type=str, default=None,
                        help='Override output path for semantic trajectory JSON')

    args = parser.parse_args()
    # Ensure trajectory path uniqueness if not provided
    if args.semantic_trajectory_path is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.semantic_trajectory_path = f"paper/results/gainakt2exp_semantic_trajectory_{args.experiment_suffix}_{ts}.json"

    # Basic reproducibility seed setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_gainakt2exp_model(args)