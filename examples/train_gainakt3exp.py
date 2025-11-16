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

import os, sys, torch, torch.nn as nn, numpy as np, json, logging, wandb
from examples.experiment_utils import compute_auc_acc

sys.path.insert(0, '/workspaces/pykt-toolkit')
from pykt.datasets import init_dataset4train
from pykt.models.gainakt3_exp import create_exp_model

def resolve_param(cfg, section, key, fallback):
    if cfg and section in cfg and key in cfg[section]:
        return cfg[section][key]
    return fallback

def train_gainakt3exp_dual_encoder(
    dataset_name, model_name, fold, emb_type, save_dir, learning_rate, batch_size, num_epochs,
    optimizer_name, seed, d_model, n_heads, dropout, num_encoder_blocks, d_ff, seq_len,
    use_mastery_head, use_gain_head, use_skill_difficulty, use_student_speed, num_students,
    bce_loss_weight, mastery_threshold_init, threshold_temperature, use_wandb, use_amp,
    auto_shifted_eval, monitor_freq, gradient_clip, patience, weight_decay, max_correlation_students,
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
    
    logger.info(f"Skills: {num_skills}, Questions: {num_questions}")
    logger.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    
    model_config = {
        'num_c': num_skills, 'seq_len': seq_len, 'd_model': d_model, 'n_heads': n_heads,
        'dropout': dropout, 'num_encoder_blocks': num_encoder_blocks, 'd_ff': d_ff, 'use_mastery_head': use_mastery_head,
        'use_gain_head': use_gain_head, 'use_skill_difficulty': use_skill_difficulty,
        'use_student_speed': use_student_speed, 'num_students': num_students,
        'mastery_threshold_init': mastery_threshold_init, 'threshold_temperature': threshold_temperature,
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
        'monitor_frequency': monitor_freq
    }
    
    model = create_exp_model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
    patience_counter = 0
    exp_dir = os.path.join(save_dir, f"exp_{experiment_suffix}")
    os.makedirs(exp_dir, exist_ok=True)
    
    config_path = os.path.join(exp_dir, 'config.json')
    full_config = {
        'dataset': dataset_name, 'fold': fold, 'model': model_name, 'seed': seed, 'epochs': num_epochs,
        'batch_size': batch_size, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'optimizer': optimizer_name, 'bce_loss_weight': bce_loss_weight,
        'incremental_mastery_loss_weight': incremental_mastery_loss_weight, 'model_config': model_config,
        'runtime': {'use_amp': use_amp, 'use_wandb': use_wandb, 'auto_shifted_eval': auto_shifted_eval,
                    'monitor_freq': monitor_freq, 'gradient_clip': gradient_clip, 'patience': patience}
    }
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    logger.info(f"Config saved: {config_path}")
    
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
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(q=questions, r=responses, qry=questions_shifted, qtest=False, student_ids=student_ids)
                    predictions = outputs['predictions']
                    valid_mask = mask.bool()
                    y_pred = predictions[valid_mask]
                    y_true = responses_shifted[valid_mask].float()
                    bce_loss = bce_criterion(y_pred, y_true)
                    
                    im_loss = 0.0
                    if use_mastery_head and 'projected_mastery' in outputs:
                        mastery = outputs['projected_mastery']
                        if mastery.size(1) > 1:
                            mastery_diff = mastery[:, 1:] - mastery[:, :-1]
                            im_loss = torch.clamp(-mastery_diff, min=0.0).mean()
                    
                    loss = bce_loss_weight * bce_loss + incremental_mastery_loss_weight * im_loss
                
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(q=questions, r=responses, qry=questions_shifted, qtest=False, student_ids=student_ids)
                predictions = outputs['predictions']
                valid_mask = mask.bool()
                y_pred = predictions[valid_mask]
                y_true = responses_shifted[valid_mask].float()
                bce_loss = bce_criterion(y_pred, y_true)
                
                im_loss = 0.0
                if use_mastery_head and 'projected_mastery' in outputs:
                    mastery = outputs['projected_mastery']
                    if mastery.size(1) > 1:
                        mastery_diff = mastery[:, 1:] - mastery[:, :-1]
                        im_loss = torch.clamp(-mastery_diff, min=0.0).mean()
                
                loss = bce_loss_weight * bce_loss + incremental_mastery_loss_weight * im_loss
                
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce_loss.item()
            if isinstance(im_loss, torch.Tensor):
                total_im += im_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_bce = total_bce / num_batches
        avg_im = total_im / num_batches
        
        model.eval()
        valid_preds, valid_targets = [], []
        with torch.no_grad():
            for batch in valid_loader:
                questions = batch['cseqs'].to(device)
                responses = batch['rseqs'].to(device)
                questions_shifted = batch['shft_cseqs'].to(device)
                responses_shifted = batch['shft_rseqs'].to(device)
                mask = batch['masks'].to(device)
                student_ids = batch.get('uids', None)
                if student_ids is not None:
                    student_ids = student_ids.to(device)
                
                outputs = model(q=questions, r=responses, qry=questions_shifted, qtest=True, student_ids=student_ids)
                predictions = outputs['predictions']
                valid_mask = mask.bool()
                y_pred = predictions[valid_mask]
                y_true = responses_shifted[valid_mask]
                valid_preds.extend(torch.sigmoid(y_pred).cpu().numpy())
                valid_targets.extend(y_true.cpu().numpy())
        
        metrics = compute_auc_acc(np.array(valid_targets), np.array(valid_preds))
        valid_auc, valid_acc = metrics['auc'], metrics['acc']
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, IM: {avg_im:.4f}) - Valid AUC: {valid_auc:.4f}, Acc: {valid_acc:.4f}")
        
        if use_wandb:
            wandb.log({'epoch': epoch+1, 'train_loss': avg_loss, 'train_bce': avg_bce, 'train_im': avg_im, 'valid_auc': valid_auc, 'valid_acc': valid_acc})
        
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            patience_counter = 0
            model_path = os.path.join(exp_dir, 'best_model.pt')
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                       'valid_auc': valid_auc, 'valid_acc': valid_acc, 'config': full_config}, model_path)
            logger.info(f"✓ New best model saved (AUC: {valid_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    logger.info("="*80)
    logger.info(f"Training completed. Best valid AUC: {best_valid_auc:.4f}")
    logger.info("="*80)
    
    if use_wandb:
        wandb.finish()
    
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
    parser.add_argument('--use_mastery_head', action='store_true')
    parser.add_argument('--disable_mastery_head', action='store_true')
    parser.add_argument('--use_gain_head', action='store_true')
    parser.add_argument('--disable_gain_head', action='store_true')
    parser.add_argument('--use_skill_difficulty', action='store_true')
    parser.add_argument('--use_student_speed', action='store_true')
    parser.add_argument('--num_students', type=int, required=True)
    parser.add_argument('--mastery_threshold_init', type=float, required=True)
    parser.add_argument('--threshold_temperature', type=float, required=True)
    parser.add_argument('--bce_loss_weight', type=float, required=True)
    args = parser.parse_args()
    
    cfg = None
    if getattr(args, 'config', None):
        if not os.path.exists(args.config):
            print(f"[ERROR] Config not found: {args.config}", file=sys.stderr)
            sys.exit(2)
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        os.environ['PYKT_CONFIG_PATH'] = os.path.abspath(args.config)
    
    if getattr(args, 'disable_mastery_head', False):
        args.use_mastery_head = False
    if getattr(args, 'disable_gain_head', False):
        args.use_gain_head = False
    
    args.num_epochs = args.epochs
    args.dataset_name = args.dataset
    
    results = train_gainakt3exp_dual_encoder(
        dataset_name=args.dataset_name, model_name='gainakt3exp', fold=args.fold, emb_type=args.emb_type,
        save_dir='saved_model', learning_rate=args.learning_rate, batch_size=args.batch_size,
        num_epochs=args.num_epochs, optimizer_name=args.optimizer, seed=args.seed, d_model=args.d_model,
        n_heads=args.n_heads, dropout=args.dropout, num_encoder_blocks=args.num_encoder_blocks, d_ff=args.d_ff,
        seq_len=args.seq_len, use_mastery_head=args.use_mastery_head, use_gain_head=args.use_gain_head,
        use_skill_difficulty=args.use_skill_difficulty, use_student_speed=args.use_student_speed,
        num_students=args.num_students, bce_loss_weight=args.bce_loss_weight,
        mastery_threshold_init=args.mastery_threshold_init, threshold_temperature=args.threshold_temperature,
        use_wandb=args.use_wandb, use_amp=args.use_amp, auto_shifted_eval=args.auto_shifted_eval,
        monitor_freq=args.monitor_freq, gradient_clip=args.gradient_clip, patience=args.patience,
        weight_decay=args.weight_decay, max_correlation_students=args.max_correlation_students,
        cfg=cfg, experiment_suffix=f"{args.dataset_name}_fold{args.fold}"
    )
    
    print(f"\n✅ Training completed! Best AUC: {results['best_valid_auc']:.4f}")
    print(f"Experiment dir: {results['exp_dir']}")
