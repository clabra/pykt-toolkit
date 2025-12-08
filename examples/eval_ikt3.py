#!/usr/bin/env python3
"""
Evaluation script for iKT3 model with reference model alignment metrics.

Computes:
- Performance metrics: AUC, Accuracy
- Alignment losses: L_21, L_22, L_23 (IRT), or reference model-specific
- Success criteria validation
- Correlation analysis with reference model
- Interpretable factor statistics

CRITICAL ARCHITECTURAL FLAGS:
- seq_len, d_model, n_heads, num_encoder_blocks, d_ff, dropout, emb_type
  → Must match training configuration exactly
- reference_model: Type of reference model (irt, bkt, etc.)
- reference_targets_path: Path to reference model targets file
  → Must use same reference data used during training

These parameters define the model architecture and reference alignment.
Using wrong values will cause model loading failures or incorrect evaluation.
"""

import sys
import os
import torch
import pickle
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
import argparse

sys.path.insert(0, '/workspaces/pykt-toolkit')

from torch.utils.data import DataLoader
from pykt.datasets.data_loader import KTDataset
from pykt.models.ikt3 import iKT3
from examples.experiment_utils import compute_auc_acc


def prepare_batch_ref_targets(batch, ref_targets, device):
    """
    Extract batch-specific reference targets from full dataset targets.
    
    For IRT reference model:
    - beta_irt: skill difficulties (already tensor, just move to device)
    - theta_irt: extract student abilities for batch (dict -> tensor)
    - m_ref: extract reference predictions for batch (dict -> tensor, or zeros if missing)
    
    Args:
        batch: Dictionary with 'uids' key containing student IDs
        ref_targets: Full dataset targets from reference_model.load_targets()
        device: torch device
    
    Returns:
        Dictionary with batch-specific tensors
    """
    if ref_targets is None:
        return None
    
    batch_targets = {}
    
    # Skill difficulties (already a tensor, same for all students)
    if 'beta_irt' in ref_targets:
        batch_targets['beta_irt'] = ref_targets['beta_irt'].to(device)
    
    # Student abilities - extract for current batch
    # Supports both static (scalar per student) and dynamic (trajectory per student)
    if 'theta_irt' in ref_targets:
        uids = batch.get('uids', None)
        is_dynamic = ref_targets.get('is_dynamic', False)
        
        if uids is not None:
            batch_size, seq_len = batch['cseqs'].shape
            
            if is_dynamic:
                # Dynamic theta: {uid: [L]} trajectories
                theta_batch = torch.zeros(batch_size, seq_len, dtype=torch.float32)
                for i, uid in enumerate(uids):
                    uid_key = torch.tensor(uid).item() if isinstance(uid, torch.Tensor) else uid
                    theta_traj = ref_targets['theta_irt'].get(uid_key, None)
                    if theta_traj is not None:
                        # Convert to tensor if it's a list
                        if isinstance(theta_traj, list):
                            theta_traj = torch.tensor(theta_traj, dtype=torch.float32)
                        actual_len = min(len(theta_traj), seq_len)
                        theta_batch[i, :actual_len] = theta_traj[:actual_len]
                batch_targets['theta_irt'] = theta_batch.to(device)  # [B, L]
            else:
                # Static theta: {uid: scalar}
                theta_values = []
                for uid in uids:
                    uid_key = torch.tensor(uid).item() if isinstance(uid, torch.Tensor) else uid
                    theta_val = ref_targets['theta_irt'].get(uid_key, 0.0)
                    theta_values.append(theta_val)
                batch_targets['theta_irt'] = torch.tensor(theta_values, dtype=torch.float32, device=device)  # [B]
        else:
            # No uids available, use zeros
            batch_size = batch['cseqs'].size(0)
            if is_dynamic:
                seq_len = batch['cseqs'].size(1)
                batch_targets['theta_irt'] = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
            else:
                batch_targets['theta_irt'] = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    # Reference predictions - extract for current batch (or zeros if not available)
    if 'm_ref' in ref_targets:
        uids = batch.get('uids', None)
        if uids is not None and len(ref_targets['m_ref']) > 0:
            # Extract m_ref sequences for students in this batch
            batch_size, seq_len = batch['cseqs'].shape
            m_ref_batch = torch.zeros(batch_size, seq_len, dtype=torch.float32)
            
            for i, uid in enumerate(uids):
                uid_tensor = torch.tensor(uid) if not isinstance(uid, torch.Tensor) else uid
                m_ref_seq = ref_targets['m_ref'].get(uid_tensor.item(), None)
                if m_ref_seq is not None:
                    # Convert to tensor if it's a list
                    if isinstance(m_ref_seq, list):
                        m_ref_seq = torch.tensor(m_ref_seq, dtype=torch.float32)
                    # Pad or truncate to match seq_len
                    actual_len = min(len(m_ref_seq), seq_len)
                    m_ref_batch[i, :actual_len] = m_ref_seq[:actual_len]
                # else: leave as zeros (student not in reference predictions)
            
            batch_targets['m_ref'] = m_ref_batch.to(device)
        else:
            # No reference predictions available, use zeros
            batch_size, seq_len = batch['cseqs'].shape
            batch_targets['m_ref'] = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
    
    return batch_targets


def load_model(checkpoint_path, device):
    """
    Load iKT3 model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
    
    Returns:
        tuple: (model, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    
    model = iKT3(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def evaluate_model(model, test_loader, ref_targets, device):
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: iKT3 model instance
        test_loader: Test data loader
        ref_targets: Reference model targets
        device: torch device
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    model.eval()
    
    # Accumulators
    all_preds = []
    all_labels = []
    
    # Reference model-specific accumulators
    reference_model = model.reference_model
    loss_names = reference_model.get_loss_names()
    loss_accumulators = {name: [] for name in loss_names}
    
    # Interpretable factors
    factor_keys = reference_model.get_interpretable_factors({}).keys()
    all_factors = {key: [] for key in factor_keys}
    all_ref_factors = {key: [] for key in factor_keys}  # Reference values
    
    # Mastery predictions for correlation diagnostic
    all_mastery_irt = []  # Model's IRT predictions
    all_mastery_ref = []  # Reference IRT predictions
    
    with torch.no_grad():
        for batch in test_loader:
            questions = batch['cseqs'].to(device)
            responses = batch['rseqs'].to(device)
            questions_shifted = batch['shft_cseqs'].to(device)
            targets = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device)
            
            # Forward pass
            outputs = model(q=questions, r=responses, qry=questions_shifted)
            
            # Prepare batch-specific reference targets
            batch_ref_targets = prepare_batch_ref_targets(batch, ref_targets, device)
            
            # Compute alignment losses (using λ=1.0 for full evaluation)
            alignment_losses = reference_model.compute_alignment_losses(
                model_outputs=outputs,
                targets=batch_ref_targets,
                lambda_weights={'lambda_interp': 1.0}
            )
            
            # Collect predictions
            preds = outputs['bce_predictions'].detach().cpu().numpy()
            labels_np = targets.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for i in range(len(preds)):
                valid_indices = mask_np[i] == 1
                all_preds.extend(preds[i][valid_indices])
                all_labels.extend(labels_np[i][valid_indices])
            
            # Collect mastery predictions for correlation diagnostic
            if 'mastery_irt' in outputs and 'm_ref' in batch_ref_targets:
                mastery_irt_np = outputs['mastery_irt'].detach().cpu().numpy()
                mastery_ref_np = batch_ref_targets['m_ref'].cpu().numpy()
                for i in range(len(mastery_irt_np)):
                    valid_indices = mask_np[i] == 1
                    all_mastery_irt.extend(mastery_irt_np[i][valid_indices])
                    all_mastery_ref.extend(mastery_ref_np[i][valid_indices])
            
            # Collect alignment losses (per-sample)
            for loss_name in loss_names:
                if loss_name in alignment_losses:
                    loss_accumulators[loss_name].append(alignment_losses[loss_name].item())
            
            # Collect interpretable factors
            factors = reference_model.get_interpretable_factors(outputs)
            for key, values in factors.items():
                if values is not None:
                    values_np = values.detach().cpu().numpy()
                    for i in range(len(values_np)):
                        valid_indices = mask_np[i] == 1
                        all_factors[key].extend(values_np[i][valid_indices])
                    
                    # Collect reference values if available
                    ref_key = f"{key}_irt" if key != 'mastery' else 'm_ref'
                    if ref_key in ref_targets:
                        ref_vals = ref_targets[ref_key]
                        if isinstance(ref_vals, torch.Tensor):
                            ref_vals_np = ref_vals.cpu().numpy() if ref_vals.is_cuda else ref_vals.numpy()
                            # Match with learned values based on batch structure
                            # For simplicity, collect all valid reference values
                            for i in range(len(values_np)):
                                valid_indices = mask_np[i] == 1
                                # This is simplified - proper matching would use student IDs
                                all_ref_factors[key].extend(ref_vals_np[i][valid_indices] if len(ref_vals_np.shape) > 1 else [ref_vals_np])
    
    # Compute performance metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    perf_metrics = compute_auc_acc(all_labels, all_preds)
    
    # Compute average alignment losses
    avg_alignment_losses = {name: np.mean(loss_accumulators[name]) for name in loss_names}
    
    # Compute factor statistics and correlations
    factor_stats = {}
    correlations = {}
    
    # CRITICAL: Compute mastery prediction correlation (l_21 diagnostic)
    if len(all_mastery_irt) > 0 and len(all_mastery_ref) > 0:
        mastery_irt_arr = np.array(all_mastery_irt)
        mastery_ref_arr = np.array(all_mastery_ref)
        
        # Remove any NaN/inf values
        valid_mask = np.isfinite(mastery_irt_arr) & np.isfinite(mastery_ref_arr)
        mastery_irt_clean = mastery_irt_arr[valid_mask]
        mastery_ref_clean = mastery_ref_arr[valid_mask]
        
        if len(mastery_irt_clean) > 0 and len(np.unique(mastery_irt_clean)) > 1 and len(np.unique(mastery_ref_clean)) > 1:
            try:
                pearson_corr, _ = pearsonr(mastery_irt_clean, mastery_ref_clean)
                spearman_corr, _ = spearmanr(mastery_irt_clean, mastery_ref_clean)
                correlations['mastery_prediction_pearson'] = float(pearson_corr)
                correlations['mastery_prediction_spearman'] = float(spearman_corr)
                
                print(f"\n{'='*60}")
                print(f"DIAGNOSTIC: Mastery Prediction Correlation")
                print(f"{'='*60}")
                print(f"Pearson correlation:  {pearson_corr:.4f}")
                print(f"Spearman correlation: {spearman_corr:.4f}")
                print(f"Valid predictions:    {len(mastery_irt_clean)}")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Warning: Could not compute mastery correlation: {e}")
    
    for key in all_factors:
        if len(all_factors[key]) > 0:
            values_arr = np.array(all_factors[key])
            factor_stats[f'{key}_mean'] = float(np.mean(values_arr))
            factor_stats[f'{key}_std'] = float(np.std(values_arr))
            factor_stats[f'{key}_min'] = float(np.min(values_arr))
            factor_stats[f'{key}_max'] = float(np.max(values_arr))
            
            # Compute correlation with reference if available
            if key in all_ref_factors and len(all_ref_factors[key]) > 0:
                ref_arr = np.array(all_ref_factors[key])
                if len(values_arr) == len(ref_arr) and len(np.unique(values_arr)) > 1 and len(np.unique(ref_arr)) > 1:
                    try:
                        pearson_corr, _ = pearsonr(values_arr, ref_arr)
                        spearman_corr, _ = spearmanr(values_arr, ref_arr)
                        correlations[f'{key}_pearson'] = float(pearson_corr)
                        correlations[f'{key}_spearman'] = float(spearman_corr)
                    except:
                        pass
    
    return {
        **perf_metrics,
        **avg_alignment_losses,
        **factor_stats,
        **correlations
    }


def check_success_criteria(metrics, reference_model_type):
    """
    Check if model meets success criteria for reference model alignment.
    
    Args:
        metrics: dict from evaluate_model()
        reference_model_type: 'irt', 'bkt', etc.
    
    Returns:
        dict: Success criteria results
    """
    criteria = {}
    
    if reference_model_type == 'irt':
        # IRT success criteria
        criteria['l_21_performance'] = {
            'value': metrics.get('l_21_performance', float('inf')),
            'threshold': 0.15,
            'passed': metrics.get('l_21_performance', float('inf')) < 0.15
        }
        criteria['l_22_difficulty'] = {
            'value': metrics.get('l_22_difficulty', float('inf')),
            'threshold': 0.10,
            'passed': metrics.get('l_22_difficulty', float('inf')) < 0.10
        }
        criteria['l_23_ability'] = {
            'value': metrics.get('l_23_ability', float('inf')),
            'threshold': 0.15,
            'passed': metrics.get('l_23_ability', float('inf')) < 0.15
        }
        
        # Correlation criteria
        theta_corr = metrics.get('theta_pearson', 0.0)
        beta_corr = metrics.get('beta_pearson', 0.0)
        
        criteria['theta_correlation'] = {
            'value': theta_corr,
            'threshold': 0.85,
            'passed': theta_corr > 0.85
        }
        criteria['beta_correlation'] = {
            'value': beta_corr,
            'threshold': 0.85,
            'passed': beta_corr > 0.85
        }
        
        # Overall success
        criteria['overall'] = all(c['passed'] for c in criteria.values() if isinstance(c, dict))
    
    elif reference_model_type == 'bkt':
        # BKT success criteria (future implementation)
        criteria['l_state'] = {
            'value': metrics.get('l_state', float('inf')),
            'threshold': 0.15,
            'passed': metrics.get('l_state', float('inf')) < 0.15
        }
        criteria['l_params'] = {
            'value': metrics.get('l_params', float('inf')),
            'threshold': 0.10,
            'passed': metrics.get('l_params', float('inf')) < 0.10
        }
        
        criteria['overall'] = all(c['passed'] for c in criteria.values() if isinstance(c, dict))
    
    return criteria


def main():
    parser = argparse.ArgumentParser(description='Evaluate iKT3 model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--reference_targets_path', type=str, required=True, help='Path to reference targets')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("iKT3 EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Reference Targets: {args.reference_targets_path}")
    print()
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, device)
    reference_model_type = config['reference_model_type']
    print(f"✓ Loaded iKT3 model")
    print(f"  Reference Model: {reference_model_type}")
    print(f"  Model Dimension: {config['d_model']}")
    print(f"  Encoder Blocks: {config['num_encoder_blocks']}")
    print()
    
    # Load reference targets
    print("Loading reference targets...")
    ref_targets = model.load_reference_targets(args.reference_targets_path)
    print(f"✓ Loaded reference targets")
    print()
    
    # Load test dataset
    print("Loading test dataset...")
    project_root = '/workspaces/pykt-toolkit'
    with open(os.path.join(project_root, 'configs/data_config.json')) as f:
        data_config = json.load(f)
    
    dataset_config = data_config[args.dataset]
    num_c = dataset_config['num_c']
    
    # Handle relative paths
    dpath = dataset_config['dpath']
    if dpath.startswith('../'):
        dpath = os.path.join(project_root, dpath[3:])
    
    test_file = os.path.join(dpath, dataset_config['test_file'])
    test_dataset = KTDataset(test_file, dataset_config['input_type'], {-1})
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"✓ Loaded test dataset")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Skills: {num_c}")
    print()
    
    # Evaluate
    print("Evaluating...")
    metrics = evaluate_model(model, test_loader, ref_targets, device)
    print("✓ Evaluation complete")
    print()
    
    # Check success criteria
    print("=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['acc']:.4f}")
    print()
    
    print("=" * 80)
    print("ALIGNMENT LOSSES")
    print("=" * 80)
    loss_names = model.reference_model.get_loss_names()
    for loss_name in loss_names:
        if loss_name in metrics:
            print(f"{loss_name}: {metrics[loss_name]:.4f}")
    print()
    
    print("=" * 80)
    print("INTERPRETABLE FACTORS")
    print("=" * 80)
    factor_keys = [k.replace('_mean', '') for k in metrics.keys() if k.endswith('_mean')]
    for key in factor_keys:
        mean_val = metrics.get(f'{key}_mean', 0.0)
        std_val = metrics.get(f'{key}_std', 0.0)
        min_val = metrics.get(f'{key}_min', 0.0)
        max_val = metrics.get(f'{key}_max', 0.0)
        print(f"{key}:")
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        
        # Correlations
        pearson = metrics.get(f'{key}_pearson', None)
        spearman = metrics.get(f'{key}_spearman', None)
        if pearson is not None:
            print(f"  Correlation (Pearson): {pearson:.4f}")
        if spearman is not None:
            print(f"  Correlation (Spearman): {spearman:.4f}")
        print()
    
    print("=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)
    criteria = check_success_criteria(metrics, reference_model_type)
    
    for criterion_name, criterion_data in criteria.items():
        if criterion_name == 'overall':
            continue
        
        if isinstance(criterion_data, dict):
            value = criterion_data['value']
            threshold = criterion_data['threshold']
            passed = criterion_data['passed']
            status = "✓ PASS" if passed else "✗ FAIL"
            
            # Determine comparison operator
            if 'correlation' in criterion_name:
                comp = f"> {threshold}"
            else:
                comp = f"< {threshold}"
            
            print(f"{criterion_name}: {value:.4f} {comp} - {status}")
    
    overall_success = criteria.get('overall', False)
    print()
    print(f"Overall: {'✓ SUCCESS' if overall_success else '✗ NOT MET'}")
    print()
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Convert NumPy types to Python types recursively
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        results_path = os.path.join(args.output_dir, 'eval_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': convert_numpy_types(metrics),
                'criteria': convert_numpy_types(criteria),
                'config': config
            }, f, indent=2)
        
        print(f"✓ Saved results to {results_path}")
        
        # Save CSV for reproducibility standard (metrics_test.csv)
        csv_path = os.path.join(args.output_dir, 'metrics_test.csv')
        try:
            import csv as csv_module
            fieldnames = ['split', 'auc', 'acc', 'l_21_performance', 'l_22_difficulty', 'l_23_ability']
            row_data = {
                'split': 'test',
                'auc': f"{metrics['auc']:.6f}",
                'acc': f"{metrics['acc']:.6f}",
                'l_21_performance': f"{metrics['l_21_performance']:.6f}",
                'l_22_difficulty': f"{metrics['l_22_difficulty']:.6f}",
                'l_23_ability': f"{metrics['l_23_ability']:.6f}"
            }
            # Add correlations if available
            if 'theta_correlation' in metrics:
                fieldnames.extend(['theta_correlation', 'beta_correlation'])
                row_data['theta_correlation'] = f"{metrics['theta_correlation']:.6f}"
                row_data['beta_correlation'] = f"{metrics['beta_correlation']:.6f}"
            
            with open(csv_path, 'w', newline='') as cf:
                writer = csv_module.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)
            print(f"✓ Saved CSV metrics to {csv_path}")
        except Exception as e:
            print(f"⚠️  Warning: could not write metrics_test.csv ({e})")
    
    # Return exit code based on success
    return 0 if overall_success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
