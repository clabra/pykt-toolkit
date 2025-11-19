#!/usr/bin/env python3
"""
Simplified GainAKT3Exp Dual-Encoder Training Script (2025-11-16)

ARCHITECTURE:
- BCE Loss + Incremental Mastery Loss ONLY
- All constraint losses REMOVED
- All semantic losses REMOVED  
- Clean implementation without deprecated parameters

For full-featured version with deprecated code, see: train_gainakt3exp.py.old
"""

import os, sys, torch, torch.nn as nn, numpy as np, json, logging, wandb, csv
from examples.experiment_utils import compute_auc_acc

sys.path.insert(0, '/workspaces/pykt-toolkit')
from pykt.datasets import init_dataset4train
from pykt.models.gainakt3_exp import create_exp_model

def resolve_param(cfg, section, key, fallback):
    if cfg and section in cfg and key in cfg[section]:
        return cfg[section][key]
    return fallback

def evaluate_dual_encoders(model, data_loader, device):
    """Evaluate both encoder1 (base) and encoder2 (incremental mastery) predictions."""
    model.eval()
    preds_enc1, preds_enc2, targets = [], [], []
    
    with torch.no_grad():
        for batch in data_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            student_ids = batch.get('uids', None)
            if student_ids is not None:
                student_ids = student_ids.to(device)
            
            # CRITICAL FIX (2025-11-18): Pass qry=None to enable mastery head computation
            outputs = model(q=questions, r=responses, qry=None, qtest=True, student_ids=student_ids)
            predictions = outputs['predictions']
            valid_mask = mask.bool()
            
            # Encoder 1: Base predictions
            y_pred_enc1 = predictions[valid_mask]
            preds_enc1.extend(torch.sigmoid(y_pred_enc1).cpu().numpy())
            
            # Encoder 2: Incremental mastery predictions (if available)
            if 'incremental_mastery_predictions' in outputs:
                im_preds = outputs['incremental_mastery_predictions']
                y_pred_enc2 = im_preds[valid_mask]
                preds_enc2.extend(torch.sigmoid(y_pred_enc2).cpu().numpy())
            else:
                # Fallback to zeros if not available
                preds_enc2.extend(np.zeros(len(y_pred_enc1)))
            
            y_true = responses_shifted[valid_mask]
            targets.extend(y_true.cpu().numpy())
    
    # Compute metrics for both encoders
    metrics_enc1 = compute_auc_acc(np.array(targets), np.array(preds_enc1))
    metrics_enc2 = compute_auc_acc(np.array(targets), np.array(preds_enc2))
    
    return {
        'encoder1_auc': metrics_enc1['auc'],
        'encoder1_acc': metrics_enc1['acc'],
        'encoder2_auc': metrics_enc2['auc'],
        'encoder2_acc': metrics_enc2['acc']
    }

def train_gainakt3exp_dual_encoder(
    dataset_name, model_name, fold, emb_type, save_dir, learning_rate, batch_size, num_epochs, optimizer_name, seed,
    d_model, n_heads, dropout, num_encoder_blocks, d_ff, seq_len, weight_decay, patience, gradient_clip, monitor_freq,
    use_skill_difficulty, use_student_speed, num_students,
    bce_loss_weight, variance_loss_weight, skill_contrastive_loss_weight, beta_spread_regularization_weight,
    gains_projection_bias_std, gains_projection_orthogonal,
    mastery_threshold_init, threshold_temperature,
    beta_skill_init, m_sat_init,
    gamma_student_init, sigmoid_offset, use_wandb, use_amp, auto_shifted_eval, max_correlation_students,
    skill_difficulty_path=None, use_student_velocity=False,
    cfg=None, experiment_suffix="", log_level=logging.INFO
):
    incremental_mastery_loss_weight = 1.0 - bce_loss_weight
    
    logger = logging.getLogger(f"gainakt3exp.{experiment_suffix}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(log_level)
    
    logger.info("="*80)
    logger.info("GainAKT3Exp Dual-Encoder Training")
    logger.info(f"Dataset: {dataset_name}, Epochs: {num_epochs}, LR: {learning_rate}, BS: {batch_size}")
    logger.info(f"BCE weight (λ₁): {bce_loss_weight}, IM weight (λ₂): {incremental_mastery_loss_weight}")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Loading dataset: {dataset_name}, fold: {fold}")
    data_config = {
        dataset_name: {
            "dpath": f"/workspaces/pykt-toolkit/data/{dataset_name}",
            "num_q": 0,
            "num_c": 100,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": 3,
            "maxlen": seq_len,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1, 2, 3, 4],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv"
        }
    }
    train_loader, valid_loader = init_dataset4train(dataset_name, 'gainakt3exp', data_config, fold, batch_size)
    logger.info("Dataset loaded successfully")
    
    num_skills = data_config[dataset_name]['num_c']
    num_questions = data_config[dataset_name]['num_q']
    
    # Extract num_students from the training dataset (instead of using hardcoded parameter)
    # This makes the code generalizable to any dataset
    num_students_from_data = train_loader.dataset.dori['num_students']
    if num_students != num_students_from_data:
        logger.warning(f"Parameter num_students={num_students} differs from actual training data ({num_students_from_data} students)")
        logger.warning(f"Using actual value from dataset: {num_students_from_data}")
        num_students = num_students_from_data
    else:
        logger.info(f"Confirmed: num_students parameter matches dataset ({num_students} students)")
    
    logger.info(f"Skills: {num_skills}, Questions: {num_questions}")
    logger.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    
    # Get num_c from dataset (use num_skills extracted earlier)
    num_c = num_skills
    
    # Model configuration
    model_config = {
        'num_c': num_c, 'emb_type': emb_type, 'seq_len': seq_len, 'd_model': d_model, 'n_heads': n_heads,
        'dropout': dropout, 'num_encoder_blocks': num_encoder_blocks, 'd_ff': d_ff, 'use_skill_difficulty': use_skill_difficulty,
        'use_student_speed': use_student_speed, 'num_students': num_students,
        'mastery_threshold_init': mastery_threshold_init, 'threshold_temperature': threshold_temperature,
        'beta_skill_init': beta_skill_init, 'm_sat_init': m_sat_init,
        'gamma_student_init': gamma_student_init, 'sigmoid_offset': sigmoid_offset,
        'emb_type': emb_type,
        # Deprecated parameters (set to 0 for dual-encoder mode)
        'intrinsic_gain_attention': False,
        'non_negative_loss_weight': 0.0,
        'monotonicity_loss_weight': 0.0,
        'mastery_performance_loss_weight': 0.0,
        'gain_performance_loss_weight': 0.0,
        'sparsity_loss_weight': 0.0,
        'consistency_loss_weight': 0.0,
        'incremental_mastery_loss_weight': incremental_mastery_loss_weight,
        'variance_loss_weight': variance_loss_weight,
        'skill_contrastive_loss_weight': skill_contrastive_loss_weight,  # V3 (2025-11-18)
        'beta_spread_regularization_weight': beta_spread_regularization_weight,  # V3 (2025-11-18)
        'gains_projection_bias_std': gains_projection_bias_std,  # V3+ (2025-11-18)
        'gains_projection_orthogonal': gains_projection_orthogonal,  # V3+ (2025-11-18)
        'skill_difficulty_path': skill_difficulty_path,  # V4 (2025-11-18)
        'use_student_velocity': use_student_velocity,  # V4 (2025-11-18)
        'monitor_frequency': monitor_freq
    }
    
    model = create_exp_model(model_config).to(device)
    
    # Enable multi-GPU training with DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    else:
        logger.info(f"Using single GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # V2 (2025-11-17): Layer-wise learning rates - higher LR for gains_projection
    if optimizer_name.lower() == 'adam':
        # Separate gains_projection parameters for higher learning rate
        gains_projection_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'gains_projection' in name:
                gains_projection_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        gains_projection_lr = learning_rate * 3.0  # 3x boost for gains_projection
        param_groups = [
            {'params': other_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': gains_projection_params, 'lr': gains_projection_lr, 'weight_decay': weight_decay}
        ]
        
        optimizer = torch.optim.Adam(param_groups)
        
        logger.info(f"Layer-wise LR: base={learning_rate:.6f}, gains_projection={gains_projection_lr:.6f} (3x boost)")
        logger.info(f"Gains projection params: {sum(p.numel() for p in gains_projection_params):,}")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_wandb:
        wandb.init(project="gainakt3exp-dual-encoder", config={
            'dataset': dataset_name, 'fold': fold, 'learning_rate': learning_rate, 'batch_size': batch_size,
            'epochs': num_epochs, 'bce_loss_weight': bce_loss_weight, 'd_model': d_model, 'seed': seed
        })
    
    best_valid_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    # Check if launched via run_repro_experiment (EXPERIMENT_DIR set)
    experiment_dir = os.environ.get('EXPERIMENT_DIR')
    if experiment_dir:
        # Use experiment directory from launcher
        exp_dir = experiment_dir
        logger.info(f"Using experiment directory from launcher: {exp_dir}")
    else:
        # Fallback for manual runs
        exp_dir = os.path.join(save_dir, f"exp_{experiment_suffix}")
        logger.info(f"Manual run - using fallback directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save training-specific config (use different name to avoid overwriting launcher's config.json)
    training_config_path = os.path.join(exp_dir, 'training_config.json')
    full_config = {
        'dataset': dataset_name, 'fold': fold, 'model': model_name, 'seed': seed, 'epochs': num_epochs,
        'batch_size': batch_size, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'optimizer': optimizer_name, 'bce_loss_weight': bce_loss_weight,
        'incremental_mastery_loss_weight': incremental_mastery_loss_weight, 'model_config': model_config,
        'runtime': {'use_amp': use_amp, 'use_wandb': use_wandb, 'auto_shifted_eval': auto_shifted_eval,
                    'monitor_freq': monitor_freq, 'gradient_clip': gradient_clip, 'patience': patience}
    }
    with open(training_config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    logger.info(f"Training config saved: {training_config_path}")
    
    # Initialize metrics CSV file
    metrics_csv_path = os.path.join(exp_dir, 'metrics_epoch.csv')
    with open(metrics_csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            'epoch',
            'train_loss', 'train_auc', 'train_acc',
            'val_loss', 'val_auc', 'val_acc',
            'train_bce_loss', 'train_im_loss',
            'val_bce_loss', 'val_im_loss',
            'train_encoder1_auc', 'train_encoder1_acc',
            'val_encoder1_auc', 'val_encoder1_acc',
            'train_encoder2_auc', 'train_encoder2_acc',
            'val_encoder2_auc', 'val_encoder2_acc'
        ])
    logger.info(f"Metrics CSV initialized: {metrics_csv_path}")
    
    # Initialize learned parameters CSV file
    params_csv_path = os.path.join(exp_dir, 'learned_parameters_epoch.csv')
    with open(params_csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            'epoch',
            'theta_global',
            'beta_skill_mean', 'beta_skill_std', 'beta_skill_min', 'beta_skill_max',
            'M_sat_mean', 'M_sat_std', 'M_sat_min', 'M_sat_max',
            'gamma_student_mean', 'gamma_student_std', 'gamma_student_min', 'gamma_student_max',
            'offset'
        ])
    logger.info(f"Learned parameters CSV initialized: {params_csv_path}")
    
    # V4 (2025-11-18): Track global statistics for velocity computation
    # We don't need per-student IDs, just running average across all interactions
    global_correct = 0
    global_total = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_bce, total_im, num_batches = 0.0, 0.0, 0.0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract tensors from batch
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            responses_shifted = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            student_ids = batch.get('uids', None)
            if student_ids is not None:
                student_ids = student_ids.to(device)
            
            # V4 (2025-11-18): Compute student velocities for semantic grounding
            # Per-sequence velocity: success rate of this sequence vs global average
            # No student IDs needed - each sequence gets its own velocity
            student_velocities = None
            if use_student_velocity:
                batch_size = responses.shape[0]
                student_velocities = torch.ones(batch_size, device=device)
                
                # Global success rate from all previous interactions
                global_rate = global_correct / global_total if global_total > 0 else 0.5
                
                # Compute velocity for each sequence based on its own success rate
                # Use responses_shifted which matches mask dimensions
                mask_bool = mask.bool()
                for b in range(batch_size):
                    # Success rate for this sequence (responses_shifted already matches mask length)
                    valid_responses = responses_shifted[b][mask_bool[b]]
                    if len(valid_responses) > 0:
                        sequence_rate = valid_responses.float().mean().item()
                        velocity = sequence_rate / global_rate if global_rate > 0 else 1.0
                        # Clamp to reasonable range [0.5, 2.0]
                        student_velocities[b] = max(0.5, min(2.0, velocity))
                    # else: keep default 1.0 (neutral)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    # CRITICAL FIX (2025-11-18): Pass qry=None to enable mastery head computation
                    # Bug: Mastery computation is inside 'if qry is None' block in model forward()
                    # When qry=questions_shifted (old behavior), mastery head is disabled
                    outputs = model(q=questions, r=responses, qry=None, qtest=False, student_ids=student_ids, student_velocities=student_velocities)
                    predictions = outputs['predictions']
                    valid_mask = mask.bool()
                    y_pred = predictions[valid_mask]
                    y_true = responses_shifted[valid_mask].float()
                    bce_loss = bce_criterion(y_pred, y_true)
                    
                    im_loss = 0.0
                    if 'incremental_mastery_predictions' in outputs:
                        im_preds = outputs['incremental_mastery_predictions']
                        valid_im_preds = im_preds[valid_mask]
                        # CRITICAL: Use current responses (r) NOT shifted responses for IM loss
                        # This matches experiment 714616 behavior (AUC=0.722)
                        # Using responses_shifted would be "correct" but degrades Encoder1 performance
                        im_targets = responses[valid_mask].float()
                        im_loss = bce_criterion(valid_im_preds, im_targets)
                    
                    # V2 (2025-11-17): Add variance loss to encourage skill differentiation
                    var_loss = 0.0
                    if 'variance_loss' in outputs and variance_loss_weight > 0:
                        var_loss = outputs['variance_loss']
                    
                    # V3 (2025-11-18): Add skill-contrastive loss and beta spread regularization
                    contrastive_loss = 0.0
                    if 'skill_contrastive_loss' in outputs and skill_contrastive_loss_weight > 0:
                        contrastive_loss = outputs['skill_contrastive_loss']
                    
                    beta_reg_loss = 0.0
                    if 'beta_spread_regularization' in outputs and beta_spread_regularization_weight > 0:
                        beta_reg_loss = outputs['beta_spread_regularization']
                    
                    loss = (bce_loss_weight * bce_loss + 
                           incremental_mastery_loss_weight * im_loss + 
                           variance_loss_weight * var_loss + 
                           skill_contrastive_loss_weight * contrastive_loss + 
                           beta_spread_regularization_weight * beta_reg_loss)
                
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # CRITICAL FIX (2025-11-18): Pass qry=None to enable mastery head computation
                # Bug: Mastery computation is inside 'if qry is None' block in model forward()
                # When qry=questions_shifted (old behavior), mastery head is disabled
                outputs = model(q=questions, r=responses, qry=None, qtest=False, student_ids=student_ids, student_velocities=student_velocities)
                predictions = outputs['predictions']
                valid_mask = mask.bool()
                y_pred = predictions[valid_mask]
                y_true = responses_shifted[valid_mask].float()
                bce_loss = bce_criterion(y_pred, y_true)
                
                im_loss = 0.0
                if 'incremental_mastery_predictions' in outputs:
                    im_preds = outputs['incremental_mastery_predictions']
                    valid_im_preds = im_preds[valid_mask]
                    # CRITICAL: Use current responses (r) NOT shifted responses for IM loss
                    # This matches experiment 714616 behavior (AUC=0.722)
                    # Using responses_shifted would be "correct" but degrades Encoder1 performance
                    im_targets = responses[valid_mask].float()
                    im_loss = bce_criterion(valid_im_preds, im_targets)
                
                # V2 (2025-11-17): Add variance loss to encourage skill differentiation
                var_loss = 0.0
                if 'variance_loss' in outputs and variance_loss_weight > 0:
                    var_loss = outputs['variance_loss']
                
                # V3 (2025-11-18): Add skill-contrastive loss and beta spread regularization
                contrastive_loss = 0.0
                if 'skill_contrastive_loss' in outputs and skill_contrastive_loss_weight > 0:
                    contrastive_loss = outputs['skill_contrastive_loss']
                
                beta_reg_loss = 0.0
                if 'beta_spread_regularization' in outputs and beta_spread_regularization_weight > 0:
                    beta_reg_loss = outputs['beta_spread_regularization']
                
                loss = (bce_loss_weight * bce_loss + 
                       incremental_mastery_loss_weight * im_loss + 
                       variance_loss_weight * var_loss + 
                       skill_contrastive_loss_weight * contrastive_loss + 
                       beta_spread_regularization_weight * beta_reg_loss)
                
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce_loss.item()
            if isinstance(im_loss, torch.Tensor):
                total_im += im_loss.item()
            num_batches += 1
            
            # V4 (2025-11-18): Update global statistics for velocity tracking
            # Update AFTER forward pass to maintain running average
            if use_student_velocity:
                mask_cpu = mask.cpu().numpy()
                responses_shifted_cpu = responses_shifted.cpu().numpy()
                
                # Update global statistics (respecting mask for valid interactions)
                for resp_seq, mask_seq in zip(responses_shifted_cpu, mask_cpu):
                    valid_mask = mask_seq.astype(bool)
                    # responses_shifted already matches mask length
                    valid_responses = resp_seq[valid_mask]
                    if len(valid_responses) > 0:
                        global_total += len(valid_responses)
                        global_correct += int(valid_responses.sum())
        
        avg_loss = total_loss / num_batches
        avg_bce = total_bce / num_batches
        avg_im = total_im / num_batches
        
        # Evaluate both encoders on train and validation sets
        train_encoder_metrics = evaluate_dual_encoders(model, train_loader, device)
        valid_encoder_metrics = evaluate_dual_encoders(model, valid_loader, device)
        
        # Combined metrics (use encoder1 as main)
        train_auc = train_encoder_metrics['encoder1_auc']
        train_acc = train_encoder_metrics['encoder1_acc']
        valid_auc = valid_encoder_metrics['encoder1_auc']
        valid_acc = valid_encoder_metrics['encoder1_acc']
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, IM: {avg_im:.4f})")
        logger.info(f"  Train - AUC: {train_auc:.4f}, Acc: {train_acc:.4f} | Enc2 AUC: {train_encoder_metrics['encoder2_auc']:.4f}")
        logger.info(f"  Valid - AUC: {valid_auc:.4f}, Acc: {valid_acc:.4f} | Enc2 AUC: {valid_encoder_metrics['encoder2_auc']:.4f}")
        
        # Write metrics to CSV
        with open(metrics_csv_path, 'a', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                epoch + 1,
                f'{avg_loss:.6f}', f'{train_auc:.6f}', f'{train_acc:.6f}',
                f'{avg_loss:.6f}', f'{valid_auc:.6f}', f'{valid_acc:.6f}',  # val_loss same as train for now
                f'{avg_bce:.6f}', f'{avg_im:.6f}',
                f'{avg_bce:.6f}', f'{avg_im:.6f}',  # val BCE/IM same as train for now
                f'{train_encoder_metrics["encoder1_auc"]:.6f}', f'{train_encoder_metrics["encoder1_acc"]:.6f}',
                f'{valid_encoder_metrics["encoder1_auc"]:.6f}', f'{valid_encoder_metrics["encoder1_acc"]:.6f}',
                f'{train_encoder_metrics["encoder2_auc"]:.6f}', f'{train_encoder_metrics["encoder2_acc"]:.6f}',
                f'{valid_encoder_metrics["encoder2_auc"]:.6f}', f'{valid_encoder_metrics["encoder2_acc"]:.6f}'
            ])
        
        # Extract and save learned parameters
        with torch.no_grad():
            # Access the underlying model (handle DataParallel wrapper)
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            theta_global = base_model.theta_global.item()
            beta_skill = base_model.beta_skill
            M_sat = base_model.M_sat
            gamma_student = base_model.gamma_student
            offset = base_model.offset.item()
            
            # Write learned parameters to CSV
            with open(params_csv_path, 'a', newline='') as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([
                    epoch + 1,
                    f'{theta_global:.6f}',
                    f'{beta_skill.mean().item():.6f}', f'{beta_skill.std().item():.6f}',
                    f'{beta_skill.min().item():.6f}', f'{beta_skill.max().item():.6f}',
                    f'{M_sat.mean().item():.6f}', f'{M_sat.std().item():.6f}',
                    f'{M_sat.min().item():.6f}', f'{M_sat.max().item():.6f}',
                    f'{gamma_student.mean().item():.6f}', f'{gamma_student.std().item():.6f}',
                    f'{gamma_student.min().item():.6f}', f'{gamma_student.max().item():.6f}',
                    f'{offset:.6f}'
                ])
        
        if use_wandb:
            wandb.log({
                'epoch': epoch+1, 'train_loss': avg_loss, 'train_bce': avg_bce, 'train_im': avg_im,
                'train_auc': train_auc, 'train_acc': train_acc, 'valid_auc': valid_auc, 'valid_acc': valid_acc,
                'train_encoder2_auc': train_encoder_metrics['encoder2_auc'],
                'valid_encoder2_auc': valid_encoder_metrics['encoder2_auc']
            })
        
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': valid_auc,
                'model_config': full_config.get('model_config', {}),
            }
            model_path = os.path.join(exp_dir, 'model_best.pth')
            torch.save(checkpoint, model_path)
            logger.info(f"✓ New best model saved (AUC: {valid_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save last checkpoint after each epoch
        last_checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_auc': best_valid_auc
        }
        last_model_path = os.path.join(exp_dir, 'model_last.pth')
        torch.save(last_checkpoint, last_model_path)
    
    logger.info("="*80)
    logger.info(f"Training completed. Best valid AUC: {best_valid_auc:.4f}")
    logger.info("="*80)
    
    # Save results.json (legacy format for compatibility)
    results = {
        'best_val_auc': best_valid_auc,
        'final_epoch': epoch+1,
        'dataset': dataset_name,
        'fold': fold,
        'model': model_name
    }
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    if use_wandb:
        wandb.finish()
    
    # Auto-evaluation after successful training
    if exp_dir and auto_shifted_eval:
        logger.info("\n" + "=" * 80)
        logger.info("LAUNCHING AUTO-EVALUATION ON TEST SET")
        logger.info("=" * 80)
        
        import subprocess
        
        # Build evaluation command from training arguments and model config
        eval_cmd = [
            sys.executable,
            'examples/eval_gainakt3exp.py',
            '--run_dir', exp_dir,
            '--max_correlation_students', str(max_correlation_students),
            '--dataset', dataset_name,
            '--fold', str(fold),
            '--batch_size', str(batch_size),
            '--seq_len', str(model_config['seq_len']),
            '--d_model', str(model_config['d_model']),
            '--n_heads', str(model_config['n_heads']),
            '--num_encoder_blocks', str(model_config['num_encoder_blocks']),
            '--d_ff', str(model_config['d_ff']),
            '--dropout', str(model_config['dropout']),
            '--emb_type', model_config['emb_type'],
            '--num_students', str(model_config['num_students']),
            '--bce_loss_weight', str(bce_loss_weight),
            '--mastery_threshold_init', str(model_config['mastery_threshold_init']),
            '--threshold_temperature', str(model_config['threshold_temperature']),
            '--beta_skill_init', str(model_config['beta_skill_init']),
            '--m_sat_init', str(model_config['m_sat_init']),
            '--gamma_student_init', str(model_config['gamma_student_init']),
            '--sigmoid_offset', str(model_config['sigmoid_offset']),
            '--monitor_freq', str(model_config['monitor_frequency'])
        ]
        
        # Add optional flags
        if model_config['use_skill_difficulty']:
            eval_cmd.append('--use_skill_difficulty')
        if model_config['use_student_speed']:
            eval_cmd.append('--use_student_speed')
        
        logger.info(f"Evaluation command: {' '.join(eval_cmd)}")
        
        try:
            result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True, cwd='/workspaces/pykt-toolkit')
            logger.info("✅ Evaluation completed successfully")
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(f"Evaluation stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Evaluation failed with exit code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
        except Exception as e:
            logger.error(f"❌ Evaluation failed with exception: {e}")
        
        # Print learning trajectories command
        logger.info("\n" + "="*80)
        logger.info("INDIVIDUAL STUDENT LEARNING TRAJECTORIES")
        logger.info("="*80)
        logger.info("To analyze detailed learning trajectories for individual students, run:")
        logger.info("")
        trajectory_cmd = [
            sys.executable,
            'examples/learning_trajectories.py',
            '--run_dir', exp_dir,
            '--num_students', '10',
            '--min_steps', '10'
        ]
        logger.info(f"  {' '.join(trajectory_cmd)}")
        logger.info("")
    
    return {'best_valid_auc': best_valid_auc, 'final_epoch': epoch+1, 'exp_dir': exp_dir}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train GainAKT3Exp dual-encoder.')
    parser.add_argument('--config', type=str, help='Config JSON path')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gradient_clip', type=float, required=True)
    parser.add_argument('--patience', type=int, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--auto_shifted_eval', action='store_true')
    parser.add_argument('--monitor_freq', type=int, required=True)
    parser.add_argument('--max_correlation_students', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--num_encoder_blocks', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--emb_type', type=str, required=True, choices=['qid','concept','hybrid'])
    parser.add_argument('--use_skill_difficulty', action='store_true')
    parser.add_argument('--use_student_speed', action='store_true')
    parser.add_argument('--num_students', type=int, required=True)
    parser.add_argument('--mastery_threshold_init', type=float, required=True)
    parser.add_argument('--threshold_temperature', type=float, required=True)
    parser.add_argument('--bce_loss_weight', type=float, required=True)
    parser.add_argument('--variance_loss_weight', type=float, required=True)
    parser.add_argument('--skill_contrastive_loss_weight', type=float, required=True)  # V3 (2025-11-18)
    parser.add_argument('--beta_spread_regularization_weight', type=float, required=True)  # V3 (2025-11-18)
    parser.add_argument('--gains_projection_bias_std', type=float, required=True)  # V3+ (2025-11-18)
    parser.add_argument('--gains_projection_orthogonal', action='store_true')  # V3+ (2025-11-18)
    parser.add_argument('--skill_difficulty_path', type=str, required=True)  # V4 (2025-11-18)
    parser.add_argument('--use_student_velocity', action='store_true')  # V4 (2025-11-18)
    parser.add_argument('--beta_skill_init', type=float, required=True)
    parser.add_argument('--m_sat_init', type=float, required=True)
    parser.add_argument('--gamma_student_init', type=float, required=True)
    parser.add_argument('--sigmoid_offset', type=float, required=True)
    args = parser.parse_args()
    
    cfg = None
    if getattr(args, 'config', None):
        if not os.path.exists(args.config):
            print(f"[ERROR] Config not found: {args.config}", file=sys.stderr)
            sys.exit(2)
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        os.environ['PYKT_CONFIG_PATH'] = os.path.abspath(args.config)
    
    args.num_epochs = args.epochs
    args.dataset_name = args.dataset
    
    results = train_gainakt3exp_dual_encoder(
        dataset_name=args.dataset_name, model_name='gainakt3exp', fold=args.fold, emb_type=args.emb_type,
        save_dir='saved_model', learning_rate=args.learning_rate, batch_size=args.batch_size,
        num_epochs=args.num_epochs, optimizer_name=args.optimizer, seed=args.seed, d_model=args.d_model,
        n_heads=args.n_heads, dropout=args.dropout, num_encoder_blocks=args.num_encoder_blocks, d_ff=args.d_ff,
        seq_len=args.seq_len, use_skill_difficulty=args.use_skill_difficulty, use_student_speed=args.use_student_speed,
        num_students=args.num_students, bce_loss_weight=args.bce_loss_weight, variance_loss_weight=args.variance_loss_weight,
        skill_contrastive_loss_weight=args.skill_contrastive_loss_weight, beta_spread_regularization_weight=args.beta_spread_regularization_weight,
        gains_projection_bias_std=args.gains_projection_bias_std, gains_projection_orthogonal=args.gains_projection_orthogonal,
        skill_difficulty_path=args.skill_difficulty_path, use_student_velocity=args.use_student_velocity,
        mastery_threshold_init=args.mastery_threshold_init, threshold_temperature=args.threshold_temperature,
        beta_skill_init=args.beta_skill_init, m_sat_init=args.m_sat_init,
        gamma_student_init=args.gamma_student_init, sigmoid_offset=args.sigmoid_offset,
        use_wandb=args.use_wandb, use_amp=args.use_amp, auto_shifted_eval=args.auto_shifted_eval,
        monitor_freq=args.monitor_freq, gradient_clip=args.gradient_clip, patience=args.patience,
        weight_decay=args.weight_decay, max_correlation_students=args.max_correlation_students,
        cfg=cfg, experiment_suffix=f"{args.dataset_name}_fold{args.fold}"
    )
    
    print(f"\n✅ Training completed! Best AUC: {results['best_valid_auc']:.4f}")
    print(f"Experiment dir: {results['exp_dir']}")
