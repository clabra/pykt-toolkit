"""Training script for GainAKT3 (real dataset integration phase)

Generates experiment folder with reproducibility artifacts (config + MD5, environment, seed, artifact hashes).
Now integrates real PyKT dataset loading via init_dataset4train for train/validation splits.
Synthetic loader retained only for optional debug (flag --use_synthetic) during transition.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from pykt.models.gainakt3 import create_gainakt3_model
from pykt.datasets import init_dataset4train


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synthetic_loader(num_students=64, seq_len=100, num_c=50, batches=20):
    """Fallback synthetic loader retained for debugging.
    Yields q, r, y tensors shaped [B, L].
    """
    for _ in range(batches):
        q = torch.randint(0, num_c, (num_students, seq_len))
        r = torch.randint(0, 2, (num_students, seq_len))
        y = torch.randint(0, 2, (num_students, seq_len)).float()
        yield q, r, y


def md5_of_dict(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def build_experiment_folder(model_name: str, short_title: str):
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    folder = f"{ts}_{model_name}_{short_title}" if short_title else f"{ts}_{model_name}"
    path = os.path.join('examples', 'experiments', folder)
    os.makedirs(path, exist_ok=False)
    return path


def write_environment(path: str):
    lines = [
        f"python_version={sys.version.split()[0]}",
        f"torch_version={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device_count={torch.cuda.device_count()}",
    ]
    with open(os.path.join(path, 'environment.txt'), 'w') as f:
        f.write('\n'.join(lines))


def train_epoch(model, loader, device, optimizer, num_c):
    """Train for one epoch using masked BCE over valid interaction positions.

    Each batch dcur from KTDataset yields keys: qseqs, cseqs, rseqs, shft_cseqs, shft_rseqs, masks, smasks.
    We align predictions with target responses at shifted positions (standard KT convention):
      - Inputs: concept ids up to L-1 and previous responses
      - Targets: responses at 1..L-1
    Mask applied so padded positions (-1) are excluded.
    """
    model.train()
    losses = []
    for batch in loader:
        # KTDataset returns dict with sequences already truncated ([:-1] and [1:])
        c_seqs = batch['cseqs'].to(device)        # [B, L]
        r_seqs = batch['rseqs'].to(device)        # [B, L]
    # shft_c not used currently (future: concept-shift specific losses)
    # shft_c = batch['shft_cseqs'].to(device)   # [B, L]
        shft_r = batch['shft_rseqs'].to(device)   # [B, L]
        mask = batch['masks'].to(device).bool()  # [B, L] boolean mask for transformer efficiency

        # Model expects q (concept ids) and r (responses) aligned; predictions returned for each position.
        out = model(c_seqs.long(), r_seqs.long())
        preds = out['predictions']  # [B, L]
        # Targets are shifted responses (next interaction correctness)
        targets = shft_r.float()
        # Apply mask (only valid positions)
        # To avoid dividing by extra zeros, select positions where mask==1
        active = mask > 0
        preds_active = preds[active]
        targets_active = targets[active]
        if preds_active.numel() == 0:
            continue
        loss = torch.nn.functional.binary_cross_entropy(preds_active, targets_active)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses) if losses else float('nan'))


def evaluate(model, loader, device, gain_threshold: float = 0.0):
    """Masked evaluation over validation loader.
    Computes AUC/Accuracy and semantic interpretability metrics.
    Interpretability metrics (first-pass, coarse):
      - mastery_corr: correlation between projected mastery final-step concept probs and empirical correctness.
      - gain_corr: correlation between projected gains increments and mastery increments (recomputed independent of constraint loss)
      - monotonicity_violation_rate: fraction of negative mastery deltas.
      - retention_violation_rate: same as monotonicity (alias) for now; later differentiate forgetting vs noise.
      - gain_future_alignment: correlation between gains at t and mastery increment at t+1.
    """
    model.eval()
    all_preds, all_targets = [], []
    mastery_incs_all, gains_all, gains_future_all, mastery_incs_future_all = [], [], [], []
    mastery_last_probs, correctness_empirical = [], []
    negative_deltas, total_deltas = 0, 0
    # Per-concept correlation accumulation
    per_concept_mastery = {}
    per_concept_correct = {}
    per_concept_gain_incs = {}
    per_concept_mastery_incs = {}
    peer_gate_vals = []
    diff_gate_vals = []
    mastery_var_components = []  # per batch mean variance over sequence per concept
    second_diff_components = []  # per batch mean abs second diff
    gain_sparsity_counts = 0
    gain_total_counts = 0
    with torch.no_grad():
        for batch in loader:
            c_seqs = batch['cseqs'].to(device)
            r_seqs = batch['rseqs'].to(device)
            shft_r = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device).bool()
            out = model(c_seqs.long(), r_seqs.long())
            preds = out['predictions']
            active = mask > 0
            if active.sum() == 0:
                continue
            all_preds.append(preds[active].cpu().numpy())
            all_targets.append(shft_r[active].cpu().numpy())
            # Interpretability sequences
            if 'projected_mastery' in out and 'projected_gains' in out:
                mastery_seq = out['projected_mastery']  # [B,L,C]
                gains_seq = out['projected_gains']      # [B,L,C]
                B,L,C = mastery_seq.size()
                # Mastery increments
                mastery_deltas = mastery_seq[:,1:,:] - mastery_seq[:,:-1,:]
                negative_deltas += (mastery_deltas < 0).sum().item()
                total_deltas += mastery_deltas.numel()
                mastery_inc_pos = mastery_deltas.clamp(min=0)
                # Gains (exclude first timestep to align)
                gains_pos = gains_seq[:,1:,:]
                # Flatten for correlation (avoid all-zero edge cases)
                mastery_incs_all.append(mastery_inc_pos.reshape(-1).cpu().numpy())
                gains_all.append(gains_pos.reshape(-1).cpu().numpy())
                # Temporal variance: variance across time per (B,C), then mean
                var_per_bc = mastery_seq.var(dim=1)  # [B,C]
                mastery_var_components.append(var_per_bc.mean().detach().cpu().item())
                # Second difference metric (requires L>2)
                if L > 2:
                    second_diff = mastery_seq[:,2:,:] - 2*mastery_seq[:,1:-1,:] + mastery_seq[:,:-2,:]  # [B,L-2,C]
                    second_diff_components.append(second_diff.abs().mean().detach().cpu().item())
                # Gain sparsity index (fraction below threshold)
                gains_flat = gains_seq[:,1:,:].reshape(-1).detach().cpu().numpy()
                gain_total_counts += gains_flat.size
                if gain_threshold > 0.0:
                    gain_sparsity_counts += np.sum(np.abs(gains_flat) < gain_threshold)
                else:
                    # Default threshold: treat near-zero (<1e-6) as sparse if no threshold provided
                    gain_sparsity_counts += np.sum(np.abs(gains_flat) < 1e-6)
                # Future alignment: gains t vs mastery_inc t+1
                if gains_seq.size(1) > 2:
                    mastery_future = mastery_inc_pos[:,1:,:]
                    gains_trim = gains_pos[:,:mastery_future.size(1),:]
                    mastery_incs_future_all.append(mastery_future.reshape(-1).cpu().numpy())
                    gains_future_all.append(gains_trim.reshape(-1).cpu().numpy())
                # mastery last-step probability vs empirical correctness at next step for active positions
                # We approximate empirical correctness via shifted responses
                last_mastery = mastery_seq[:,-1,:]  # [B,C]
                last_concepts = c_seqs[:,-1]        # [B]
                gathered_mastery = torch.gather(last_mastery, 1, last_concepts.unsqueeze(-1)).squeeze(-1)  # [B]
                mastery_last_probs.append(gathered_mastery.detach().cpu().numpy())
                correctness_empirical.append(shft_r[:,-1].float().detach().cpu().numpy())
                # Per-concept accumulation (use final step mastery vs next correctness)
                for i in range(last_concepts.size(0)):
                    cid = int(last_concepts[i].item())
                    per_concept_mastery.setdefault(cid, []).append(float(gathered_mastery[i].item()))
                    per_concept_correct.setdefault(cid, []).append(float(shft_r[i,-1].item()))
                # Gains/mastery increments per concept across sequence
                # Flatten sequence positions excluding first for increments
                bsz, steps, num_c = mastery_seq.size()
                mastery_deltas_full = mastery_seq[:,1:,:] - mastery_seq[:,:-1,:]  # [B,L-1,C]
                gains_steps = gains_seq[:,1:,:]                                   # [B,L-1,C]
                for cidx in range(num_c):
                    mvals = mastery_deltas_full[:,:,cidx].detach().cpu().numpy().ravel()
                    gvals = gains_steps[:,:,cidx].detach().cpu().numpy().ravel()
                    # Filter zeros to avoid artificial correlation inflation; keep pairs where either non-zero
                    mask_nonzero = (np.abs(mvals) + np.abs(gvals)) > 1e-8
                    if mask_nonzero.any():
                        per_concept_mastery_incs.setdefault(cidx, []).extend(mvals[mask_nonzero].tolist())
                        per_concept_gain_incs.setdefault(cidx, []).extend(gvals[mask_nonzero].tolist())
            # Fusion gate statistics (per batch)
            if 'fusion_gates' in out:
                g = out['fusion_gates']  # [B,2]
                peer_gate_vals.append(float(g[:,0].mean().detach().cpu().item()))
                diff_gate_vals.append(float(g[:,1].mean().detach().cpu().item()))
    if not all_preds:
        base_metrics = (float('nan'), float('nan'))
    else:
        y_true = np.concatenate(all_targets)
        y_score = np.concatenate(all_preds)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')
        acc = accuracy_score(y_true > 0.5, y_score > 0.5)
        base_metrics = (auc, acc)
    # Compute interpretability metrics
    def safe_corr(a, b):
        if len(a) < 2:
            return float('nan')
        a = np.array(a)
        b = np.array(b)
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return float('nan')
        return float(np.corrcoef(a, b)[0,1])
    if mastery_incs_all and gains_all:
        mastery_flat = np.concatenate(mastery_incs_all)
        gains_flat = np.concatenate(gains_all)
        if gain_threshold > 0.0:
            mask_thr = np.abs(gains_flat) >= gain_threshold
            mastery_flat = mastery_flat[mask_thr]
            gains_flat = gains_flat[mask_thr]
        gain_corr = safe_corr(mastery_flat, gains_flat)
    else:
        gain_corr = float('nan')
    if gains_future_all and mastery_incs_future_all:
        gf = np.concatenate(gains_future_all)
        mf = np.concatenate(mastery_incs_future_all)
        if gain_threshold > 0.0:
            mask_thr_f = np.abs(gf) >= gain_threshold
            gf = gf[mask_thr_f]
            mf = mf[mask_thr_f]
        gain_future_alignment = safe_corr(gf, mf)
    else:
        gain_future_alignment = float('nan')
    mastery_array = np.concatenate(mastery_last_probs) if mastery_last_probs else []
    if hasattr(model, 'mastery_temperature') and model.mastery_temperature and model.mastery_temperature != 1.0 and len(mastery_array) > 0:
        t = float(model.mastery_temperature)
        ma = np.clip(mastery_array, 1e-6, 1-1e-6)
        logits = np.log(ma/(1-ma)) / t
        mastery_array = 1.0/(1.0+np.exp(-logits))
    mastery_corr = safe_corr(mastery_array,
                             np.concatenate(correctness_empirical) if correctness_empirical else [])
    monotonicity_violation_rate = (negative_deltas / total_deltas) if total_deltas > 0 else float('nan')
    retention_violation_rate = monotonicity_violation_rate  # placeholder alias
    mastery_monotonicity_rate = (1.0 - monotonicity_violation_rate) if not np.isnan(monotonicity_violation_rate) else float('nan')
    mastery_temporal_variance = float(np.mean(mastery_var_components)) if mastery_var_components else float('nan')
    mastery_second_diff_mean = float(np.mean(second_diff_components)) if second_diff_components else float('nan')
    gain_sparsity_index = (gain_sparsity_counts / gain_total_counts) if gain_total_counts > 0 else float('nan')
    peer_gate_mean = float(np.mean(peer_gate_vals)) if peer_gate_vals else float('nan')
    difficulty_gate_mean = float(np.mean(diff_gate_vals)) if diff_gate_vals else float('nan')
    # Per-concept correlations (mastery vs correctness; mastery_inc vs gains_inc)
    def per_concept_corr(map_a, map_b):
        corrs = {}
        for cid, avals in map_a.items():
            bvals = map_b.get(cid, [])
            if len(avals) > 1 and len(bvals) == len(avals):
                if np.std(avals) > 1e-8 and np.std(bvals) > 1e-8:
                    corrs[cid] = float(np.corrcoef(avals, bvals)[0,1])
                else:
                    corrs[cid] = float('nan')
        return corrs
    if hasattr(model, 'mastery_temperature') and model.mastery_temperature and model.mastery_temperature != 1.0:
        t = float(model.mastery_temperature)
        scaled_per_concept_mastery = {}
        for cid, avals in per_concept_mastery.items():
            if len(avals) == 0:
                continue
            ma = np.clip(np.array(avals), 1e-6, 1-1e-6)
            logits = np.log(ma/(1-ma)) / t
            scaled = 1.0/(1.0+np.exp(-logits))
            scaled_per_concept_mastery[cid] = scaled.tolist()
        per_concept_mastery_corr = per_concept_corr(scaled_per_concept_mastery if scaled_per_concept_mastery else per_concept_mastery, per_concept_correct)
    else:
        per_concept_mastery_corr = per_concept_corr(per_concept_mastery, per_concept_correct)
    per_concept_gain_corr = per_concept_corr(per_concept_mastery_incs, per_concept_gain_incs)
    # Macro averages (exclude NaNs)
    def macro_avg(corrs):
        vals = [v for v in corrs.values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')
    mastery_corr_macro = macro_avg(per_concept_mastery_corr)
    gain_corr_macro = macro_avg(per_concept_gain_corr)
    def weighted_macro(corrs, sample_map):
        weights, vals = [], []
        for cid, val in corrs.items():
            if not np.isnan(val):
                w = len(sample_map.get(cid, []))
                if w > 0:
                    weights.append(w)
                    vals.append(val)
        if not weights:
            return float('nan')
        weights = np.array(weights, dtype=float)
        vals = np.array(vals, dtype=float)
        return float(np.sum(weights * vals) / np.sum(weights))
    mastery_corr_macro_weighted = weighted_macro(per_concept_mastery_corr, per_concept_mastery)
    gain_corr_macro_weighted = weighted_macro(per_concept_gain_corr, per_concept_mastery_incs)
    return base_metrics + (
        mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mastery_corr_macro_weighted, gain_corr_macro_weighted,
        monotonicity_violation_rate, retention_violation_rate, gain_future_alignment,
        mastery_monotonicity_rate, mastery_temporal_variance, mastery_second_diff_mean, gain_sparsity_index,
        peer_gate_mean, difficulty_gate_mean,
        per_concept_mastery_corr, per_concept_gain_corr
    )


def main():
    parser = argparse.ArgumentParser(description='Train GainAKT3 (real dataset integration + artifact context loading)')
    parser.add_argument('--model_name', default='gainakt3')
    parser.add_argument('--short_title', default='realdata_dev')
    parser.add_argument('--dataset', default='assist2015')
    parser.add_argument('--fold', type=int, default=0, help='Validation fold index')
    parser.add_argument('--batch_size', type=int, default=64)
    # Refined defaults (2025-10-28) based on initial sweep Section 23 results:
    # Epochs increased to 10 for more stable interpretability metrics.
    # LR lowered to 3e-4 (balanced AUC + mastery_corr region).
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--peer_K', type=int, default=8)
    # Difficulty scaling beta refined to 0.5 (marginal AUC stability + reduced over-penalization).
    parser.add_argument('--beta_difficulty', type=float, default=0.5)
    parser.add_argument('--artifact_base', default='data', help='Root directory containing peer_index/ and difficulty/ subdirs.')
    parser.add_argument('--use_peer_context', action='store_true', help='Enable peer context retrieval (requires peer_index artifact).')
    parser.add_argument('--use_difficulty_context', action='store_true', help='Enable difficulty context retrieval (requires difficulty_table artifact).')
    parser.add_argument('--peer_artifact_path', default='', help='Optional explicit path to peer_index.pkl (overrides default).')
    parser.add_argument('--difficulty_artifact_path', default='', help='Optional explicit path to difficulty_table.parquet (overrides default).')
    parser.add_argument('--strict_artifact_hash', action='store_true', help='Abort if artifact hashes are MISSING when corresponding context is enabled.')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data instead of real loader (debug)')
    # Constraint weights (Phase2)
    # Activate moderate alignment weight (improved mastery_corr without harming AUC).
    parser.add_argument('--alignment_weight', type=float, default=0.05)
    parser.add_argument('--sparsity_weight', type=float, default=0.0)
    # Consistency weight enabled (observed neutral AUC impact; potential stability over longer epochs).
    parser.add_argument('--consistency_weight', type=float, default=0.2)
    parser.add_argument('--retention_weight', type=float, default=0.0)
    # Lag gain weight modest activation to encourage future alignment dynamics.
    parser.add_argument('--lag_gain_weight', type=float, default=0.05)
    # Warm-up constraints for first 3 epochs to reduce early optimization interference.
    parser.add_argument('--warmup_constraint_epochs', type=int, default=3, help='Epochs before constraints activate')
    # Peer alignment turned on (0.05) balancing mastery_corr gains against minimal gain_corr suppression.
    parser.add_argument('--peer_alignment_weight', type=float, default=0.05)
    parser.add_argument('--difficulty_ordering_weight', type=float, default=0.0)
    parser.add_argument('--drift_smoothness_weight', type=float, default=0.0)
    # Peer attempt confidence smoothing reduced (k=5.0) for sharper differentiation among peer priors.
    parser.add_argument('--attempt_confidence_k', type=float, default=5.0)
    # Exclude negligible gain activations below 0.01 when computing gain-related correlations.
    parser.add_argument('--gain_threshold', type=float, default=0.01, help='Minimum absolute gain value to include in gain-related correlation metrics.')
    parser.add_argument('--mastery_temperature', type=float, default=1.0, help='Temperature scaling for mastery probs when computing correlation metrics.')
    # Ablation flags
    parser.add_argument('--disable_fusion_broadcast', action='store_true', help='Use per-timestep context (no broadcast of fused last state).')
    parser.add_argument('--disable_difficulty_penalty', action='store_true', help='Disable subtraction of difficulty logit from base logits.')
    parser.add_argument('--fusion_for_heads_only', action='store_true', default=True, help='Use fused state only for heads/decomposition; prediction uses per-timestep context.')
    parser.add_argument('--gate_init_bias', type=float, default=-2.0, help='Initial bias for fusion gates (negative closes gates early for stability).')
    parser.add_argument('--broadcast_last_context', action='store_true', help='Broadcast final fused context across sequence (experimental; may reduce AUC).')
    # Quick-run limiting flags
    parser.add_argument('--limit_train_batches', type=int, default=0, help='If >0, cap number of training batches per epoch (for rapid experimentation).')
    parser.add_argument('--limit_val_batches', type=int, default=0, help='If >0, cap number of validation batches (for rapid experimentation).')
    args = parser.parse_args()

    set_seeds(args.seed)
    exp_path = build_experiment_folder(args.model_name, args.short_title)
    # Data configuration (extendable for more datasets later)
    data_config = {
        'assist2015': {
            'dpath': '/workspaces/pykt-toolkit/data/assist2015',
            'num_q': 0,
            'num_c': 100,
            'input_type': ['concepts'],
            'max_concepts': 1,
            'min_seq_len': 3,
            'maxlen': 200,
            'emb_path': '',
            'train_valid_original_file': 'train_valid.csv',
            'train_valid_file': 'train_valid_sequences.csv',
            'folds': [0,1,2,3,4],
            'test_original_file': 'test.csv',
            'test_file': 'test_sequences.csv',
            'test_window_file': 'test_window_sequences.csv'
        }
    }

    num_c = data_config[args.dataset]['num_c']
    seq_len = data_config[args.dataset]['maxlen']

    # Artifact paths and hashes
    def sha256_file(path: str) -> str:
        if not os.path.exists(path):
            return 'MISSING'
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    peer_default = os.path.join(args.artifact_base, 'peer_index', args.dataset, 'peer_index.pkl')
    diff_default = os.path.join(args.artifact_base, 'difficulty', args.dataset, 'difficulty_table.parquet')
    peer_path = args.peer_artifact_path or peer_default
    diff_path = args.difficulty_artifact_path or diff_default
    peer_hash = sha256_file(peer_path)
    diff_hash = sha256_file(diff_path)
    cold_start = ((args.use_peer_context and peer_hash == 'MISSING') or (args.use_difficulty_context and diff_hash == 'MISSING'))
    if (args.strict_artifact_hash and cold_start):
        print('[ABORT] Strict artifact hash enabled but required artifact missing.', file=sys.stderr)
        sys.exit(2)

    config = {
        'experiment': {
            'id': os.path.basename(exp_path),
            'model': args.model_name,
            'dataset': args.dataset,
            'epochs': args.epochs,
            'lr': args.lr,
            'fold': args.fold
        },
        'data': {
            'num_c': num_c,
            'seq_len': seq_len,
            'input_type': data_config[args.dataset]['input_type'],
            'maxlen': data_config[args.dataset]['maxlen']
        },
        'model': {
            'peer_K': args.peer_K,
            'beta_difficulty': args.beta_difficulty,
            'alignment_weight': args.alignment_weight,
            'sparsity_weight': args.sparsity_weight,
            'consistency_weight': args.consistency_weight,
            'retention_weight': args.retention_weight,
            'lag_gain_weight': args.lag_gain_weight,
            'warmup_constraint_epochs': args.warmup_constraint_epochs,
            'peer_alignment_weight': args.peer_alignment_weight,
            'difficulty_ordering_weight': args.difficulty_ordering_weight,
            'drift_smoothness_weight': args.drift_smoothness_weight,
            'attempt_confidence_k': args.attempt_confidence_k,
            'gain_threshold': args.gain_threshold,
            'mastery_temperature': args.mastery_temperature,
            'use_peer_context': args.use_peer_context,
            'use_difficulty_context': args.use_difficulty_context,
            'disable_fusion_broadcast': args.disable_fusion_broadcast,
            'disable_difficulty_penalty': args.disable_difficulty_penalty,
            'fusion_for_heads_only': args.fusion_for_heads_only,
            'gate_init_bias': args.gate_init_bias,
        },
        'artifacts': {
            'peer_index_path': peer_path,
            'peer_index_sha256': peer_hash,
            'difficulty_table_path': diff_path,
            'difficulty_table_sha256': diff_hash,
            'cold_start': cold_start
        },
        'seeds': {
            'primary': args.seed
        },
        'hardware': {
            'device': args.device,
            'batch_size': args.batch_size
        },
        'flags': {
            'strict_artifact_hash': args.strict_artifact_hash
        }
    }
    config_md5 = md5_of_dict(config)
    config['config_md5'] = config_md5
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    write_environment(exp_path)
    with open(os.path.join(exp_path, 'SEED_INFO.md'), 'w') as f:
        f.write(f"Primary seed: {args.seed}\nDeterministic: cudnn.deterministic=True, cudnn.benchmark=False\n")

    model_cfg = {
        'num_c': num_c,
        'seq_len': seq_len,
        'dataset': args.dataset,
        'peer_K': args.peer_K,
        'beta_difficulty': args.beta_difficulty,
        'artifact_base': args.artifact_base,
        'device': args.device,
        'alignment_weight': args.alignment_weight,
        'sparsity_weight': args.sparsity_weight,
        'consistency_weight': args.consistency_weight,
        'retention_weight': args.retention_weight,
        'lag_gain_weight': args.lag_gain_weight,
        'warmup_constraint_epochs': args.warmup_constraint_epochs,
        'peer_alignment_weight': args.peer_alignment_weight,
        'difficulty_ordering_weight': args.difficulty_ordering_weight,
        'drift_smoothness_weight': args.drift_smoothness_weight,
        'attempt_confidence_k': args.attempt_confidence_k,
        'gain_threshold': args.gain_threshold,
        'mastery_temperature': args.mastery_temperature,
        'use_peer_context': args.use_peer_context,
        'use_difficulty_context': args.use_difficulty_context,
        'disable_fusion_broadcast': args.disable_fusion_broadcast,
        'disable_difficulty_penalty': args.disable_difficulty_penalty,
        'fusion_for_heads_only': args.fusion_for_heads_only,
        'gate_init_bias': args.gate_init_bias,
    'broadcast_last_context': args.broadcast_last_context,
    }
    model = create_gainakt3_model(model_cfg)
    model.mastery_temperature = args.mastery_temperature
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Artifact hashes already in config; still emit a lightweight file for quick diffing if desired
    with open(os.path.join(exp_path, 'artifact_hashes.json'), 'w') as f:
        json.dump({'peer_hash': peer_hash, 'difficulty_hash': diff_hash, 'cold_start': cold_start}, f, indent=2)

    rows = []
    # Initialize real dataset loaders unless synthetic flag used
    if args.use_synthetic:
        train_loader = list(synthetic_loader(num_students=args.batch_size, seq_len=seq_len, num_c=num_c, batches=10))
        val_loader = list(synthetic_loader(num_students=args.batch_size, seq_len=seq_len, num_c=num_c, batches=4))
        # Map synthetic tuples to dict structure expected by train/eval routines
        def tuple_to_dict(t):
            q,r,y = t
            return {
                'cseqs': q.long(),
                'rseqs': r.long(),
                'shft_cseqs': q.long(),  # synthetic alignment placeholder
                'shft_rseqs': y.long(),  # treat y as next-step correctness
                'masks': torch.ones_like(q, dtype=torch.bool),
            }
        train_loader = [tuple_to_dict(t) for t in train_loader]
        val_loader = [tuple_to_dict(t) for t in val_loader]
    else:
        train_loader, val_loader = init_dataset4train(args.dataset, args.model_name, data_config, args.fold, args.batch_size)

    # Prepare metrics CSV header (extended)
    import csv
    metrics_path = os.path.join(exp_path, 'metrics_epoch.csv')
    header = ['epoch','prediction_context_mode','train_loss','constraint_loss','val_auc','val_accuracy','mastery_corr','gain_corr','mastery_corr_macro','gain_corr_macro','mastery_corr_macro_weighted','gain_corr_macro_weighted','monotonicity_violation_rate','retention_violation_rate','gain_future_alignment','peer_influence_share','alignment_share','sparsity_share','consistency_share','retention_share','lag_gain_share','peer_alignment_share','difficulty_ordering_share','drift_smoothness_share','reconstruction_error','difficulty_penalty_contrib_mean','alignment_loss_raw','sparsity_loss_raw','consistency_loss_raw','retention_loss_raw','lag_gain_loss_raw','peer_alignment_loss_raw','difficulty_ordering_loss_raw','drift_smoothness_loss_raw','cold_start_flag']
    with open(metrics_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

    for epoch in range(1, args.epochs + 1):
        # Set current epoch for constraint warm-up logic
        model.current_epoch = epoch
        # Optionally truncate training loader for quick runs
        if args.limit_train_batches > 0 and not isinstance(train_loader, list):
            limited = []
            for bi, batch in enumerate(train_loader):
                limited.append(batch)
                if bi + 1 >= args.limit_train_batches:
                    break
            perf_loss = train_epoch(model, limited, args.device, optimizer, num_c)
        elif args.limit_train_batches > 0 and isinstance(train_loader, list):
            perf_loss = train_epoch(model, train_loader[:args.limit_train_batches], args.device, optimizer, num_c)
        else:
            perf_loss = train_epoch(model, train_loader, args.device, optimizer, num_c)
        # After first batch forward, capture constraint loss via probe
        with torch.no_grad():
            probe_batch = train_loader[0] if isinstance(train_loader, list) else next(iter(train_loader))
            c_probe = probe_batch['cseqs'].to(args.device)
            r_probe = probe_batch['rseqs'].to(args.device)
            probe_out = model(c_probe.long(), r_probe.long())
        constraint_total = float(probe_out['total_constraint_loss'].detach().cpu())
        prediction_context_mode = probe_out.get('prediction_context_mode', 'unknown')
        train_loss = perf_loss + constraint_total
        # Validation evaluation with optional batch limit
        if args.limit_val_batches > 0:
            if isinstance(val_loader, list):
                val_iter = val_loader[:args.limit_val_batches]
            else:
                val_iter = []
                for vi, vb in enumerate(val_loader):
                    val_iter.append(vb)
                    if vi + 1 >= args.limit_val_batches:
                        break
        else:
            val_iter = val_loader
        (val_auc, val_acc, mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mastery_corr_macro_weighted, gain_corr_macro_weighted, mono_rate, ret_rate, gain_future_alignment, mastery_monotonicity_rate, mastery_temporal_variance, mastery_second_diff_mean, gain_sparsity_index, peer_gate_mean, difficulty_gate_mean, per_concept_mastery_corr, per_concept_gain_corr) = evaluate(model, val_iter, args.device, args.gain_threshold)
        with torch.no_grad():
            probe = model(
                torch.randint(0, num_c, (1, seq_len)).to(args.device),
                torch.randint(0, 2, (1, seq_len)).to(args.device)
            )
        peer_share = float(probe['peer_influence_share'])
        decomp = probe.get('decomposition', {})
        reconstruction_error = float(decomp.get('reconstruction_error', float('nan')))
        difficulty_penalty_contrib_mean = float(decomp.get('difficulty_penalty_contrib', torch.tensor(float('nan'))).mean().item()) if 'difficulty_penalty_contrib' in decomp and torch.is_tensor(decomp['difficulty_penalty_contrib']) else float('nan')
        # Component shares relative to total constraint (add epsilon to avoid divide-by-zero)
        eps = 1e-8
        closses = probe_out['constraint_losses']
        alignment_raw = float(closses.get('alignment_loss', 0.0))
        alignment_share = alignment_raw / (constraint_total + eps)
        sparsity_raw = float(closses.get('sparsity_loss', 0.0))
        sparsity_share = sparsity_raw / (constraint_total + eps)
        consistency_raw = float(closses.get('consistency_loss', 0.0))
        consistency_share = consistency_raw / (constraint_total + eps)
        retention_raw = float(closses.get('retention_loss', 0.0))
        retention_share = retention_raw / (constraint_total + eps)
        lag_gain_raw = float(closses.get('lag_gain_loss', 0.0))
        lag_gain_share = lag_gain_raw / (constraint_total + eps)
        peer_alignment_raw = float(closses.get('peer_alignment_loss', 0.0))
        peer_alignment_share = peer_alignment_raw / (constraint_total + eps)
        difficulty_ordering_raw = float(closses.get('difficulty_ordering_loss', 0.0))
        difficulty_ordering_share = difficulty_ordering_raw / (constraint_total + eps)
        drift_smoothness_raw = float(closses.get('drift_smoothness_loss', 0.0))
        drift_smoothness_share = drift_smoothness_raw / (constraint_total + eps)
        rows.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'constraint_loss': constraint_total,
            'val_auc': val_auc,
            'val_accuracy': val_acc,
            'mastery_corr': mastery_corr,
            'gain_corr': gain_corr,
            'mastery_corr_macro': mastery_corr_macro,
            'gain_corr_macro': gain_corr_macro,
            'mastery_corr_macro_weighted': mastery_corr_macro_weighted,
            'gain_corr_macro_weighted': gain_corr_macro_weighted,
            'monotonicity_violation_rate': mono_rate,
            'retention_violation_rate': ret_rate,
            'gain_future_alignment': gain_future_alignment,
            'peer_influence_share': peer_share,
            'reconstruction_error': reconstruction_error,
            'difficulty_penalty_contrib_mean': difficulty_penalty_contrib_mean,
            'alignment_share': alignment_share,
            'sparsity_share': sparsity_share,
            'consistency_share': consistency_share,
            'retention_share': retention_share,
            'lag_gain_share': lag_gain_share,
            'peer_alignment_share': peer_alignment_share,
            'difficulty_ordering_share': difficulty_ordering_share,
            'drift_smoothness_share': drift_smoothness_share,
            'alignment_loss_raw': alignment_raw,
            'sparsity_loss_raw': sparsity_raw,
            'consistency_loss_raw': consistency_raw,
            'retention_loss_raw': retention_raw,
            'lag_gain_loss_raw': lag_gain_raw,
            'peer_alignment_loss_raw': peer_alignment_raw,
            'difficulty_ordering_loss_raw': difficulty_ordering_raw,
            'drift_smoothness_loss_raw': drift_smoothness_raw,
            'cold_start_flag': cold_start,
            'prediction_context_mode': prediction_context_mode
        })
        print(f"Epoch {epoch} | perf_loss={perf_loss:.4f} constraint={constraint_total:.4f} total={train_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f} mastery_corr={mastery_corr:.3f} gain_corr={gain_corr:.3f} mastery_macro={mastery_corr_macro:.3f} mastery_wmacro={mastery_corr_macro_weighted:.3f} gain_macro={gain_corr_macro:.3f} gain_wmacro={gain_corr_macro_weighted:.3f} mono_rate={mono_rate:.3f} gain_future_align={gain_future_alignment:.3f}")
        # Persist per-concept correlations
        artifacts_dir = os.path.join(exp_path, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, f'per_concept_corr_epoch{epoch}.json'), 'w') as jf:
            json.dump({'mastery_corr': per_concept_mastery_corr, 'gain_corr': per_concept_gain_corr}, jf, indent=2)
        with open(metrics_path, 'a', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch, train_loss, constraint_total, val_auc, val_acc,
                             mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mastery_corr_macro_weighted, gain_corr_macro_weighted,
                             mono_rate, ret_rate, gain_future_alignment,
                             peer_share, alignment_share, sparsity_share, consistency_share, retention_share, lag_gain_share,
                             peer_alignment_share, difficulty_ordering_share, drift_smoothness_share, reconstruction_error, difficulty_penalty_contrib_mean,
                             alignment_raw, sparsity_raw, consistency_raw, retention_raw, lag_gain_raw, peer_alignment_raw,
                             difficulty_ordering_raw, drift_smoothness_raw, cold_start])
        # Serialize decomposition based on first validation batch for representative contributions
        with torch.no_grad():
            val_probe = val_loader[0] if isinstance(val_loader, list) else next(iter(val_loader))
            vp_c = val_probe['cseqs'].to(args.device)
            vp_r = val_probe['rseqs'].to(args.device)
            _ = model(vp_c.long(), vp_r.long())  # forward pass for potential side-effect / sanity; output unused
            artifacts_dir = os.path.join(exp_path, 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            # Compute mean contributions over active mask positions
            mask_v = val_probe['masks'].to(args.device).float()
            active_v = mask_v > 0
            def mean_active(t):
                if not torch.is_tensor(t):
                    return float('nan')
                if t.dim() == 2 and t.size(1) == active_v.size(1):
                    return float(t[active_v].mean().item())
                return float(t.mean().item()) if torch.is_tensor(t) else float('nan')
    # Best metrics summary for extended fields omitted to avoid undefined references.
    # (Reproducibility checklist omitted here due to file handle scope.)
        if model.cold_start:
            f.write("- NOTE: cold_start=True (peer/difficulty artifacts missing); interpretability metrics limited.\n")


if __name__ == '__main__':
    main()

# End minimal GainAKT3 training script