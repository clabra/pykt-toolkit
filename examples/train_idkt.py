#!/usr/bin/env python3
"""
Training script for iDKT (Interpretable Deep Knowledge Tracing) model.

This script follows pykt framework patterns for standard KT model training.
Initial version: iDKT is identical to AKT baseline.

╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️  REPRODUCIBILITY WARNING ⚠️                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DO NOT CALL THIS SCRIPT DIRECTLY FOR REPRODUCIBLE EXPERIMENTS!             ║
║                                                                              ║
║  This script requires explicit parameters. For reproducible experiments:    ║
║                                                                              ║
║      python examples/run_repro_experiment.py --model idkt --short_title ... ║
║                                                                              ║
║  The launcher will generate explicit commands with ALL parameters.          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import pandas as pd
import argparse
import json
import torch
torch.set_num_threads(32)
from torch.optim import SGD, Adam
from torch.nn.functional import binary_cross_entropy
import numpy as np
import csv
import pickle
from datetime import datetime
from sklearn import metrics

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.models import train_model, evaluate, init_model
from pykt.datasets import init_dataset4train
from pykt.utils import set_seed

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'


def parse_args():
    parser = argparse.ArgumentParser(description="Train iDKT model")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset name (assist2009, assist2015, assist2017, statics2011)")
    parser.add_argument("--fold", type=int, required=True,
                      help="Cross-validation fold number")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, required=True,
                      help="Dimension of attention block")
    parser.add_argument("--d_ff", type=int, required=True,
                      help="Dimension of feedforward network")
    parser.add_argument("--n_heads", type=int, required=True,
                      help="Number of attention heads")
    parser.add_argument("--n_blocks", type=int, required=True,
                      help="Number of transformer blocks")
    parser.add_argument("--dropout", type=float, required=True,
                      help="Dropout rate")
    parser.add_argument("--emb_type", type=str, required=True,
                      help="Embedding type (qid)")
    parser.add_argument("--final_fc_dim", type=int, required=True,
                      help="Final fully connected layer dimension")
    parser.add_argument("--seq_len", type=int, required=True,
                      help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--seed", type=int, required=True,
                      help="Random seed")
    parser.add_argument("--epochs", type=int, required=True,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True,
                      help="Batch size")
    parser.add_argument("--learning_rate", type=float, required=True,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, required=True,
                      help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, required=True,
                      help="Optimizer (Adam, SGD)")
    parser.add_argument("--gradient_clip", type=float, required=True,
                      help="Gradient clipping value")
    parser.add_argument("--patience", type=int, required=True,
                      help="Early stopping patience")
    parser.add_argument("--l2", type=float, required=True,
                      help="L2 regularization weight for Rasch difficulty parameters")
    parser.add_argument("--lambda_student", type=float, required=True,
                      help="L2 regularization weight for student capability parameters (v_s)")
    parser.add_argument("--lambda_gap", type=float, required=True,
                      help="L2 regularization weight for student knowledge gap parameters (k_c)")
    
    # Interpretability and Theory-Guided parameters
    parser.add_argument("--lambda_ref", type=float, required=True,
                      help="Weight for prediction alignment loss (L_ref)")
    parser.add_argument("--lambda_initmastery", type=float, required=True,
                      help="Weight for initial mastery alignment loss")
    parser.add_argument("--lambda_rate", type=float, required=True,
                      help="Weight for learning rate alignment loss")
    parser.add_argument("--theory_guided", type=int, required=True,
                      help="Enable theory-guided loss components (0 or 1)")
    parser.add_argument("--calibrate", type=int, required=True, help="Run initial forward pass to recalibrate lambda weights")
    parser.add_argument("--grounded_init", type=int, required=True, help="Initialize theory-guided embeddings from BKT parameters (0 or 1)")
    
    # BKT Filtering parameters (tracked in parameter_default.json)
    parser.add_argument("--bkt_filter", action='store_true',
                      help="Filter out skills with extreme BKT parameters (analysis only)")
    parser.add_argument("--bkt_guess_threshold", type=float, required=True,
                      help="Max guess rate allowed for a skill to be included")
    parser.add_argument("--bkt_slip_threshold", type=float, required=True,
                      help="Max slip rate allowed for a skill to be included")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="saved_model/idkt",
                      help="Directory to save model checkpoints")
    parser.add_argument("--use_wandb", type=int, required=True,
                      help="Use Weights & Biases logging (0 or 1)")
    
    # Doc strings from config (ignored but needed for parser)
    parser.add_argument("--_doc_grounding", type=str, required=True, help="Documentation for grounding logic")
    parser.add_argument("--_doc_regularization", type=str, required=True, help="Documentation for regularization logic")
    
    return parser.parse_args()


def evaluate_idkt_individualized(model, loader, device, args=None, bkt_skill_params=None):
    """
    Specialized evaluation for iDKT with student-level individualization.
    Calculates both performance metrics (AUC, Acc) and grounding losses.
    """
    model.eval()
    y_trues, y_scores = [], []
    
    total_l_sup = 0.0
    total_l_ref = 0.0
    total_l_init = 0.0
    total_l_rate = 0.0
    total_l_reg = 0.0
    steps = 0

    with torch.no_grad():
        for data in loader:
            q, c, r = data["qseqs"].to(device), data["cseqs"].to(device), data["rseqs"].to(device)
            qshft, cshft, rshft = data["shft_qseqs"].to(device), data["shft_cseqs"].to(device), data["shft_rseqs"].to(device)
            sm = data["smasks"].to(device)
            uids = data.get("uids", None)
            if uids is not None:
                uids = uids.to(device)

            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)

            # Forward with uid_data
            y, initmastery, rate, reg_loss, reg_losses_dict = model(cc.long(), cr.long(), cq.long(), uid_data=uids)
            
            y_pred = torch.masked_select(y[:, 1:], sm).detach()
            y_true = torch.masked_select(rshft, sm).detach()

            # 1. Supervised Loss
            l_sup = binary_cross_entropy(y_pred.double(), y_true.double())
            total_l_sup += l_sup.item()
            total_l_reg += reg_loss.item()
            
            # 2. Theory components (if possible)
            if args and args.theory_guided:
                if "bkt_p_correct" in data:
                    bkt_p_correct = data["bkt_p_correct"].to(device)
                    bkt_p_shft = torch.masked_select(bkt_p_correct, sm)
                    l_ref = torch.mean((y_pred - bkt_p_shft)**2)
                    total_l_ref += l_ref.item()
                
                if bkt_skill_params is not None:
                    skills_shft = torch.masked_select(cc.long()[:, 1:], sm)
                    proj_initmastery = torch.masked_select(initmastery[:, 1:], sm)
                    proj_rate = torch.masked_select(rate[:, 1:], sm)
                    
                    ref_initmastery = torch.tensor([bkt_skill_params['params'].get(s.item(), bkt_skill_params['global'])['prior'] 
                                               for s in skills_shft]).to(device)
                    ref_rate = torch.tensor([bkt_skill_params['params'].get(s.item(), bkt_skill_params['global'])['learns'] 
                                           for s in skills_shft]).to(device)
                    
                    l_initmastery = torch.mean((proj_initmastery - ref_initmastery)**2)
                    l_rate = torch.mean((proj_rate - ref_rate)**2)
                    total_l_init += l_initmastery.item()
                    total_l_rate += l_rate.item()

            y_trues.append(y_true.cpu().numpy())
            y_scores.append(y_pred.cpu().numpy())
            steps += 1

    ts = np.concatenate(y_trues, axis=0)
    ps = np.concatenate(y_scores, axis=0)
    auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
    prelabels = [1 if p >= 0.5 else 0 for p in ps]
    acc = metrics.accuracy_score(ts, prelabels)
    
    avg_metrics = {
        'l_sup': total_l_sup / steps,
        'l_reg': total_l_reg / steps,
        'l_ref': total_l_ref / steps,
        'l_init': total_l_init / steps,
        'l_rate': total_l_rate / steps
    }
    
    return auc, acc, avg_metrics


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load data configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    # Convert relative paths to absolute paths
    for dataset_name in data_config:
        if 'dpath' in data_config[dataset_name]:
            dpath = data_config[dataset_name]['dpath']
            if dpath.startswith('../'):
                # Strip '../' and join with project_root
                data_config[dataset_name]['dpath'] = os.path.abspath(os.path.join(project_root, dpath.replace('../', '')))
            elif not os.path.isabs(dpath):
                data_config[dataset_name]['dpath'] = os.path.abspath(os.path.join(project_root, dpath))
    
    # Prepare parameters dict (pykt convention)
    params = {
        'model_name': 'idkt',
        'dataset_name': args.dataset,
        'fold': args.fold,
        'seed': args.seed,
        'emb_type': args.emb_type,
        'save_dir': args.save_dir,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_attn_heads': args.n_heads,
        'n_blocks': args.n_blocks,
        'dropout': args.dropout,
        'final_fc_dim': args.final_fc_dim,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'optimizer': args.optimizer,
        'gradient_clip': args.gradient_clip,
        'patience': args.patience,
        'seq_len': args.seq_len,
        'l2': args.l2,
        'lambda_student': args.lambda_student,
        'lambda_gap': args.lambda_gap,
        'lambda_ref': args.lambda_ref,
        'lambda_initmastery': args.lambda_initmastery,
        'lambda_rate': args.lambda_rate,
        'theory_guided': args.theory_guided,
        'calibrate': args.calibrate,
        'use_wandb': args.use_wandb,
        'add_uuid': 0  # Don't add UUID to save_dir
    }
    
    # Initialize dataset - Use augmented file if theory_guided is on
    print(f"Loading dataset: {args.dataset}, fold: {args.fold}")
    
    if args.theory_guided:
        # Override filenames to use augmented versions
        # NOTE: This assumes augment_with_bkt.py has been run
        if 'train_valid_file' in data_config[args.dataset]:
            orig = data_config[args.dataset]['train_valid_file']
            data_config[args.dataset]['train_valid_file'] = orig.replace('.csv', '_bkt.csv')
            print(f"  Using augmented training file: {data_config[args.dataset]['train_valid_file']}")

    train_loader, valid_loader = init_dataset4train(
        args.dataset, 'idkt', data_config, args.fold, args.batch_size)
    
    # Extract num_students from dataset (uses dense UID-to-index mapping)
    num_students = train_loader.dataset.dori.get("num_students", 0)
    print(f"  Detected {num_students} unique students in training set.")
    
    # Load BKT Skill Parameters for L_param
    bkt_skill_params = None
    if args.theory_guided:
        bkt_params_path = os.path.join(data_config[args.dataset]['dpath'], 'bkt_skill_params.pkl')
        if os.path.exists(bkt_params_path):
            with open(bkt_params_path, 'rb') as f:
                bkt_skill_params = pickle.load(f)
            print(f"  Loaded BKT skill parameters from: {bkt_params_path}")
        else:
            raise FileNotFoundError(f"CRITICAL: BKT skill parameters not found at {bkt_params_path}. "
                                   f"Theory-guided training requires these parameters. No fallbacks allowed.")
    
    # Initialize model
    print(f"Initializing iDKT model...")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, n_blocks={args.n_blocks}")
    print(f"  dropout={args.dropout}, final_fc_dim={args.final_fc_dim}")
    
    # Prepare model config (use parameter names expected by model __init__)
    model_config = {
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_attn_heads': args.n_heads,  # iDKT model expects num_attn_heads
        'n_blocks': args.n_blocks,
        'dropout': args.dropout,
        'final_fc_dim': args.final_fc_dim,
        'l2': args.l2,
        'lambda_student': args.lambda_student,
        'lambda_gap': args.lambda_gap,
        'n_uid': num_students
    }
    
    model = init_model('idkt', model_config, data_config[args.dataset], args.emb_type)
    
    # Initialize theory bases from BKT parameters (Grounded Init)
    if args.theory_guided and bkt_skill_params is not None and args.grounded_init == 1:
        print("  ✓ Performing Grounded Initialization (Textured Grounding)...")
        model.load_theory_params(bkt_skill_params)
    elif args.theory_guided and args.grounded_init == 0:
        print("  ✓ Skipping Grounded Initialization (Autonomous Discovery Mode)")
    
    # Training
    print(f"Starting training for {args.epochs} epochs...")
    print(f"  learning_rate={args.learning_rate}, batch_size={args.batch_size}")
    print(f"  optimizer={args.optimizer}, l2={args.l2}")
    
    # Create optimizer (matching AKT's wandb_train.py behavior exactly)
    if args.optimizer.lower() == "sgd":
        optimizer = SGD(model.parameters(), args.learning_rate, momentum=0.9)
    elif args.optimizer.lower() == "adam":
        optimizer = Adam(model.parameters(), args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Loss Calibration Pass (Warm-up)
    if args.calibrate and args.theory_guided:
        print("\n" + "="*80)
        print("LOSS CALIBRATION PASS (WARM-UP)")
        print("="*80)
        print("Executing initial forward pass to normalize lambda weights...")
        
        model.eval()
        cal_data = next(iter(train_loader))
        with torch.no_grad():
            q, c, r = cal_data["qseqs"].to(device), cal_data["cseqs"].to(device), cal_data["rseqs"].to(device)
            qshft, rshft = cal_data["shft_qseqs"].to(device), cal_data["shft_rseqs"].to(device)
            sm = cal_data["smasks"].to(device)
            uids = cal_data["uids"].to(device)

            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cal_data["shft_cseqs"].to(device)), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)

            y, initmastery, rate, _, _ = model(cc.long(), cr.long(), cq.long(), uid_data=uids)
            y_pred = torch.masked_select(y[:, 1:], sm)
            y_true = torch.masked_select(rshft, sm)
            m_sup = binary_cross_entropy(y_pred.double(), y_true.double()).item()
            
            # Calibration for L_ref
            if "bkt_p_correct" in cal_data:
                bkt_p_shft = torch.masked_select(cal_data["bkt_p_correct"].to(device), sm)
                m_ref = torch.mean((y_pred - bkt_p_shft)**2).item()
                if m_ref > 1e-12: # Check if MSE is not effectively zero
                    target_ratio = args.lambda_ref
                    args.lambda_ref = min(target_ratio * (m_sup / (m_ref + 1e-8)), 100.0)
                    print(f"  ✓ L_ref: Target Ratio {target_ratio*100:.1f}%, Calibrated Lambda: {args.lambda_ref:.6f} (Init MSE: {m_ref:.6e})")
            
            # Calibration for L_param
            if bkt_skill_params is not None:
                skills_shft = torch.masked_select(cc.long()[:, 1:], sm)
                p_initmastery = torch.masked_select(initmastery[:, 1:], sm)
                p_rate = torch.masked_select(rate[:, 1:], sm)
                
                r_init = torch.tensor([bkt_skill_params['params'].get(s.item(), bkt_skill_params['global'])['prior'] 
                                         for s in skills_shft]).to(device)
                r_rate = torch.tensor([bkt_skill_params['params'].get(s.item(), bkt_skill_params['global'])['learns'] 
                                        for s in skills_shft]).to(device)
                
                m_init = torch.mean((p_initmastery - r_init)**2).item()
                m_rate = torch.mean((p_rate - r_rate)**2).item()
                
                if m_init > 1e-12:
                    target_ratio = args.lambda_initmastery
                    args.lambda_initmastery = min(target_ratio * (m_sup / (m_init + 1e-8)), 100.0)
                    print(f"  ✓ L_initmastery: Target Ratio {target_ratio*100:.1f}%, Calibrated Lambda: {args.lambda_initmastery:.6f} (Init MSE: {m_init:.6e})")
                if m_rate > 1e-12:
                    target_ratio = args.lambda_rate
                    args.lambda_rate = min(target_ratio * (m_sup / (m_rate + 1e-8)), 100.0)
                    print(f"  ✓ L_rate: Target Ratio {target_ratio*100:.1f}%, Calibrated Lambda: {args.lambda_rate:.6f} (Init MSE: {m_rate:.6e})")
                
                # Log calibrated weights for reproducibility
                cal_weights = {
                    "lambda_ref": args.lambda_ref,
                    "lambda_initmastery": args.lambda_initmastery,
                    "lambda_rate": args.lambda_rate
                }
                with open(os.path.join(args.save_dir, 'calibration.json'), 'w') as f:
                    json.dump(cal_weights, f, indent=2)
        
        print("Calibrated weights will be used to maintain theory/supervised signal balance.")
        print("="*80 + "\n")

    # Training Loop
    best_valid_auc = 0.0
    best_valid_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    test_auc, test_acc = -1, -1
    window_testauc, window_testacc = -1, -1
    
    # Initialize metrics_epoch.csv
    csv_path = os.path.join(args.save_dir, 'metrics_epoch.csv')
    csv_headers = ['epoch', 'train_loss', 'valid_auc', 'valid_acc', 'train_l_sup', 'train_l_ref', 'train_l_init', 'train_l_rate', 'train_l_rasch', 'train_l_gap', 'train_l_student']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_l_sup = 0.0
        train_l_ref = 0.0
        train_l_initmastery = 0.0
        train_l_rate = 0.0
        train_l_rasch = 0.0
        train_l_gap = 0.0
        train_l_student = 0.0
        train_steps = 0
        
        for data in train_loader:
             # Data preparation
             q, c, r, t = data["qseqs"].to(device), data["cseqs"].to(device), data["rseqs"].to(device), data["tseqs"].to(device)
             qshft, cshft, rshft, tshft = data["shft_qseqs"].to(device), data["shft_cseqs"].to(device), data["shft_rseqs"].to(device), data["shft_tseqs"].to(device)
             m, sm = data["masks"].to(device), data["smasks"].to(device)
             uids = data["uids"].to(device)

             cq = torch.cat((q[:,0:1], qshft), dim=1)
             cc = torch.cat((c[:,0:1], cshft), dim=1)
             cr = torch.cat((r[:,0:1], rshft), dim=1)

             # Forward
             y, initmastery, rate, reg_loss, reg_losses_dict = model(cc.long(), cr.long(), cq.long(), uid_data=uids)
             
             # Standard Supervised Loss (L_SUP)
             y_pred = torch.masked_select(y[:, 1:], sm)
             y_true = torch.masked_select(rshft, sm)
             l_sup = binary_cross_entropy(y_pred.double(), y_true.double())
             
             loss = l_sup + reg_loss
             l_ref_val, l_init_val, l_rate_val = 0.0, 0.0, 0.0
             
             # Theory-Guided Alignment Losses
             if args.theory_guided:
                 # 1. Prediction Alignment Loss (L_ref)
                 if "bkt_p_correct" in data:
                     bkt_p_correct = data["bkt_p_correct"].to(device)
                     bkt_p_shft = torch.masked_select(bkt_p_correct, sm)
                     l_ref = torch.mean((y_pred - bkt_p_shft)**2)
                     loss += args.lambda_ref * l_ref
                     l_ref_val = l_ref.item()
                 
                 # 2. Parameter Consistency Loss (L_param)
                 # We align model's projected 'initmastery' and 'rate' with BKT skill params
                 if bkt_skill_params is not None:
                     skills_shft = torch.masked_select(cc.long()[:, 1:], sm)
                     proj_initmastery = torch.masked_select(initmastery[:, 1:], sm)
                     proj_rate = torch.masked_select(rate[:, 1:], sm)
                     
                     # Get reference values for these skills
                     ref_initmastery = torch.tensor([bkt_skill_params['params'].get(s.item(), bkt_skill_params['global'])['prior'] 
                                               for s in skills_shft]).to(device)
                     ref_rate = torch.tensor([bkt_skill_params['params'].get(s.item(), bkt_skill_params['global'])['learns'] 
                                            for s in skills_shft]).to(device)
                     
                     l_initmastery = torch.mean((proj_initmastery - ref_initmastery)**2)
                     l_rate = torch.mean((proj_rate - ref_rate)**2)
                     
                     l_param = args.lambda_initmastery * l_initmastery + args.lambda_rate * l_rate
                     loss += l_param
                     l_init_val = l_initmastery.item()
                     l_rate_val = l_rate.item()

             optimizer.zero_grad()
             loss.backward()
             if args.gradient_clip > 0:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
             optimizer.step()
             
             train_loss += loss.item()
             train_l_sup += l_sup.item()
             train_l_ref += l_ref_val
             train_l_initmastery += l_init_val
             train_l_rate += l_rate_val
             train_l_rasch += reg_losses_dict['reg_rasch'].item()
             train_l_gap += reg_losses_dict['reg_gap'].item()
             train_l_student += reg_losses_dict['reg_student'].item()
             train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        avg_l_sup = train_l_sup / train_steps
        avg_l_ref = train_l_ref / train_steps
        avg_l_init = train_l_initmastery / train_steps
        avg_l_rate = train_l_rate / train_steps
        avg_l_rasch = train_l_rasch / train_steps
        avg_l_gap = train_l_gap / train_steps
        avg_l_student = train_l_student / train_steps
        
        # Validation
        valid_auc, valid_acc, valid_metrics = evaluate_idkt_individualized(model, valid_loader, device, args, bkt_skill_params)
        print(f"Epoch {epoch}/{args.epochs}: Loss={avg_train_loss:.4f} (Raw SUP={avg_l_sup:.4f}, REF={avg_l_ref:.4f}, IM={avg_l_init:.4f}, RT={avg_l_rate:.4f}), Gap_L2={avg_l_gap:.4f}, Stu_L2={avg_l_student:.4f}, Valid AUC={valid_auc:.4f}")
        
        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'valid_auc': valid_auc,
                'valid_acc': valid_acc,
                'train_l_sup': avg_l_sup,
                'train_l_ref': avg_l_ref,
                'train_l_init': avg_l_init,
                'train_l_rate': avg_l_rate,
                'train_l_rasch': avg_l_rasch,
                'train_l_gap': avg_l_gap,
                'train_l_student': avg_l_student
            })
            
        # Checkpoint and Early Stopping
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_acc = valid_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model directly as best_model.pt
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (AUC: {valid_auc:.4f})")
            
            # Save validation metrics
            metrics_valid_path = os.path.join(args.save_dir, 'metrics_valid.csv')
            with open(metrics_valid_path, 'w') as f:
                f.write('split,auc,acc,l_sup,l_ref,l_init,l_rate,l_reg\n')
                f.write(f"validation,{valid_auc:.6f},{valid_acc:.6f},{valid_metrics['l_sup']:.6f},{valid_metrics['l_ref']:.6f},{valid_metrics['l_init']:.6f},{valid_metrics['l_rate']:.6f},{valid_metrics['l_reg']:.6f}\n")
            print(f"  ✓ Saved validation metrics: {metrics_valid_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Final Evaluation (matching pykt pattern)
    print("\nRunning final evaluation on sets...")
    model_config['n_uid'] = num_students # Ensure test eval uses correct n_uid
    test_auc, test_acc, test_metrics = evaluate_idkt_individualized(model, valid_loader, device, args, bkt_skill_params) # Using validation as proxy for now
    
    # Save results
    results = {
        'valid_auc': float(best_valid_auc),
        'valid_acc': float(best_valid_acc),
        'valid_metrics': valid_metrics,
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'test_metrics': test_metrics,
        'best_epoch': int(best_epoch),
        'params': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                  for k, v in params.items()}
    }
    
    results_path = os.path.join(args.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved training results: {results_path}")


if __name__ == "__main__":
    from sklearn import metrics # Needed for evaluate_idkt_individualized
    main()
