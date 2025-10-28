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
        mask = batch['masks'].to(device).float()  # [B, L]

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


def evaluate(model, loader, device):
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
    with torch.no_grad():
        for batch in loader:
            c_seqs = batch['cseqs'].to(device)
            r_seqs = batch['rseqs'].to(device)
            shft_r = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device).float()
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
        gain_corr = safe_corr(mastery_flat, gains_flat)
    else:
        gain_corr = float('nan')
    if gains_future_all and mastery_incs_future_all:
        gain_future_alignment = safe_corr(np.concatenate(gains_future_all), np.concatenate(mastery_incs_future_all))
    else:
        gain_future_alignment = float('nan')
    mastery_corr = safe_corr(np.concatenate(mastery_last_probs) if mastery_last_probs else [],
                             np.concatenate(correctness_empirical) if correctness_empirical else [])
    monotonicity_violation_rate = (negative_deltas / total_deltas) if total_deltas > 0 else float('nan')
    retention_violation_rate = monotonicity_violation_rate  # placeholder alias
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
    per_concept_mastery_corr = per_concept_corr(per_concept_mastery, per_concept_correct)
    per_concept_gain_corr = per_concept_corr(per_concept_mastery_incs, per_concept_gain_incs)
    # Macro averages (exclude NaNs)
    def macro_avg(corrs):
        vals = [v for v in corrs.values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')
    mastery_corr_macro = macro_avg(per_concept_mastery_corr)
    gain_corr_macro = macro_avg(per_concept_gain_corr)
    return base_metrics + (mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, monotonicity_violation_rate, retention_violation_rate, gain_future_alignment, per_concept_mastery_corr, per_concept_gain_corr)


def main():
    parser = argparse.ArgumentParser(description='Train GainAKT3 (real dataset integration + artifact context loading)')
    parser.add_argument('--model_name', default='gainakt3')
    parser.add_argument('--short_title', default='realdata_dev')
    parser.add_argument('--dataset', default='assist2015')
    parser.add_argument('--fold', type=int, default=0, help='Validation fold index')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--peer_K', type=int, default=8)
    parser.add_argument('--beta_difficulty', type=float, default=1.0)
    parser.add_argument('--artifact_base', default='data', help='Root directory containing peer_index/ and difficulty/ subdirs.')
    parser.add_argument('--use_peer_context', action='store_true', help='Enable peer context retrieval (requires peer_index artifact).')
    parser.add_argument('--use_difficulty_context', action='store_true', help='Enable difficulty context retrieval (requires difficulty_table artifact).')
    parser.add_argument('--peer_artifact_path', default='', help='Optional explicit path to peer_index.pkl (overrides default).')
    parser.add_argument('--difficulty_artifact_path', default='', help='Optional explicit path to difficulty_table.parquet (overrides default).')
    parser.add_argument('--strict_artifact_hash', action='store_true', help='Abort if artifact hashes are MISSING when corresponding context is enabled.')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data instead of real loader (debug)')
    # Constraint weights (Phase2)
    parser.add_argument('--alignment_weight', type=float, default=0.0)
    parser.add_argument('--sparsity_weight', type=float, default=0.0)
    parser.add_argument('--consistency_weight', type=float, default=0.0)
    parser.add_argument('--retention_weight', type=float, default=0.0)
    parser.add_argument('--lag_gain_weight', type=float, default=0.0)
    parser.add_argument('--warmup_constraint_epochs', type=int, default=0, help='Epochs before constraints activate')
    parser.add_argument('--peer_alignment_weight', type=float, default=0.0)
    parser.add_argument('--difficulty_ordering_weight', type=float, default=0.0)
    parser.add_argument('--drift_smoothness_weight', type=float, default=0.0)
    parser.add_argument('--attempt_confidence_k', type=float, default=10.0)
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
            'use_peer_context': args.use_peer_context,
            'use_difficulty_context': args.use_difficulty_context
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
        'use_peer_context': args.use_peer_context,
        'use_difficulty_context': args.use_difficulty_context
    }
    model = create_gainakt3_model(model_cfg)
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
                'masks': torch.ones_like(q).long(),
            }
        train_loader = [tuple_to_dict(t) for t in train_loader]
        val_loader = [tuple_to_dict(t) for t in val_loader]
    else:
        train_loader, val_loader = init_dataset4train(args.dataset, args.model_name, data_config, args.fold, args.batch_size)

    # Prepare metrics CSV header (extended)
    import csv
    metrics_path = os.path.join(exp_path, 'metrics_epoch.csv')
    header = ['epoch','train_loss','constraint_loss','val_auc','val_accuracy','mastery_corr','gain_corr','mastery_corr_macro','gain_corr_macro','monotonicity_violation_rate','retention_violation_rate','gain_future_alignment','peer_influence_share','alignment_share','sparsity_share','consistency_share','retention_share','lag_gain_share','peer_alignment_share','difficulty_ordering_share','drift_smoothness_share','reconstruction_error']
    with open(metrics_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

    for epoch in range(1, args.epochs + 1):
        # Set current epoch for constraint warm-up logic
        model.current_epoch = epoch
        perf_loss = train_epoch(model, train_loader, args.device, optimizer, num_c)
        # After one batch forward, constraints were applied internally; we need a probe to capture aggregated constraint loss.
        with torch.no_grad():
            probe_batch = train_loader[0] if isinstance(train_loader, list) else next(iter(train_loader))
            c_probe = probe_batch['cseqs'].to(args.device)
            r_probe = probe_batch['rseqs'].to(args.device)
            probe_out = model(c_probe.long(), r_probe.long())
        constraint_total = float(probe_out['total_constraint_loss'].detach().cpu())
        train_loss = perf_loss + constraint_total
        (val_auc, val_acc, mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mono_rate, ret_rate, gain_future_alignment, per_concept_mastery_corr, per_concept_gain_corr) = evaluate(model, val_loader, args.device)
        with torch.no_grad():
            probe = model(
                torch.randint(0, num_c, (1, seq_len)).to(args.device),
                torch.randint(0, 2, (1, seq_len)).to(args.device)
            )
        peer_share = float(probe['peer_influence_share'])
        # Decomposition aggregation (mean over probe sequence for logging; full epoch dump below uses validation batch)
        decomp = probe.get('decomposition', {})
        reconstruction_error = float(decomp.get('reconstruction_error', float('nan')))
        # Component shares relative to total constraint (add epsilon to avoid divide-by-zero)
        eps = 1e-8
        closses = probe_out['constraint_losses']
        alignment_share = float(closses.get('alignment_loss', 0.0)) / (constraint_total + eps)
        sparsity_share = float(closses.get('sparsity_loss', 0.0)) / (constraint_total + eps)
        consistency_share = float(closses.get('consistency_loss', 0.0)) / (constraint_total + eps)
        retention_share = float(closses.get('retention_loss', 0.0)) / (constraint_total + eps)
        lag_gain_share = float(closses.get('lag_gain_loss', 0.0)) / (constraint_total + eps)
        peer_alignment_share = float(closses.get('peer_alignment_loss', 0.0)) / (constraint_total + eps)
        difficulty_ordering_share = float(closses.get('difficulty_ordering_loss', 0.0)) / (constraint_total + eps)
        drift_smoothness_share = float(closses.get('drift_smoothness_loss', 0.0)) / (constraint_total + eps)
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
            'monotonicity_violation_rate': mono_rate,
            'retention_violation_rate': ret_rate,
            'gain_future_alignment': gain_future_alignment,
            'peer_influence_share': peer_share,
            'reconstruction_error': reconstruction_error,
            'alignment_share': alignment_share,
            'sparsity_share': sparsity_share,
            'consistency_share': consistency_share,
            'retention_share': retention_share,
            'lag_gain_share': lag_gain_share,
            'peer_alignment_share': peer_alignment_share,
            'difficulty_ordering_share': difficulty_ordering_share,
            'drift_smoothness_share': drift_smoothness_share
        })
        print(f"Epoch {epoch} | perf_loss={perf_loss:.4f} constraint={constraint_total:.4f} total={train_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f} mastery_corr={mastery_corr:.3f} gain_corr={gain_corr:.3f} mastery_macro={mastery_corr_macro:.3f} gain_macro={gain_corr_macro:.3f} mono_rate={mono_rate:.3f} gain_future_align={gain_future_alignment:.3f}")
        # Persist per-concept correlations
        artifacts_dir = os.path.join(exp_path, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, f'per_concept_corr_epoch{epoch}.json'), 'w') as jf:
            json.dump({'mastery_corr': per_concept_mastery_corr, 'gain_corr': per_concept_gain_corr}, jf, indent=2)
        with open(metrics_path, 'a', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch, train_loss, constraint_total, val_auc, val_acc,
                             mastery_corr, gain_corr, mastery_corr_macro, gain_corr_macro, mono_rate, ret_rate, gain_future_alignment,
                             peer_share, alignment_share, sparsity_share, consistency_share, retention_share, lag_gain_share,
                             peer_alignment_share, difficulty_ordering_share, drift_smoothness_share, reconstruction_error])
        # Serialize decomposition based on first validation batch for representative contributions
        with torch.no_grad():
            val_probe = val_loader[0] if isinstance(val_loader, list) else next(iter(val_loader))
            vp_c = val_probe['cseqs'].to(args.device)
            vp_r = val_probe['rseqs'].to(args.device)
            vp_out = model(vp_c.long(), vp_r.long())
            dcmp = vp_out.get('decomposition', {})
            artifacts_dir = os.path.join(exp_path, 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            # Compute mean contributions over active mask positions
            mask_v = val_probe['masks'].to(args.device).float()
            active_v = mask_v > 0
            def mean_active(t):
                if not torch.is_tensor(t):
                    return float('nan')
                if t.dim() == 2:
                    # If second dimension matches sequence length, apply mask; else aggregate over batch directly
                    if t.size(1) == active_v.size(1):
                        vals = t[active_v]
                    else:
                        vals = t.mean(dim=1)  # collapse sequence-like singleton
                else:
                    vals = t
                return float(vals.mean().detach().cpu()) if vals.numel() > 0 else float('nan')
            decomp_summary = {
                'epoch': epoch,
                'mastery_contrib_mean': mean_active(dcmp.get('mastery_contrib')),
                'peer_prior_contrib_mean': mean_active(dcmp.get('peer_prior_contrib')),
                'difficulty_fused_contrib_mean': mean_active(dcmp.get('difficulty_fused_contrib')),
                'value_stream_contrib_mean': mean_active(dcmp.get('value_stream_contrib')),
                'concept_contrib_mean': mean_active(dcmp.get('concept_contrib')),
                'bias_contrib_mean': mean_active(dcmp.get('bias_contrib')),
                'difficulty_penalty_contrib_mean': mean_active(dcmp.get('difficulty_penalty_contrib')),
                'reconstruction_error': float(dcmp.get('reconstruction_error', float('nan')))
            }
            with open(os.path.join(artifacts_dir, f'decomposition_epoch{epoch}.json'), 'w') as df:
                json.dump(decomp_summary, df, indent=2)

    with open(os.path.join(exp_path, 'results.json'), 'w') as f:
        json.dump({'config_md5': config_md5, 'epochs': rows}, f, indent=2)
    # Write JSON summary of epochs (includes constraint breakdown per row)

    torch.save(model.state_dict(), os.path.join(exp_path, 'model_last.pth'))
    best = max(rows, key=lambda x: x['val_auc'])
    torch.save({'state_dict': model.state_dict(), 'best_epoch': best['epoch'], 'val_auc': best['val_auc']}, os.path.join(exp_path, 'model_best.pth'))

    with open(os.path.join(exp_path, 'README.md'), 'w') as f:
        f.write(f"# Experiment {os.path.basename(exp_path)}\n\n")
        f.write("GainAKT3 training on real dataset split (assist2015) with masked losses.\n\n")
        f.write("## Summary\n")
        f.write(f"Best epoch: {best['epoch']} val_auc={best['val_auc']:.4f} val_acc={best['val_accuracy']:.4f} mastery_corr={best.get('mastery_corr', float('nan')):.4f} gain_corr={best.get('gain_corr', float('nan')):.4f} mastery_corr_macro={best.get('mastery_corr_macro', float('nan')):.4f} gain_corr_macro={best.get('gain_corr_macro', float('nan')):.4f}\n")
        f.write("\n## Interpretability Metrics (best epoch)\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Mastery Correlation (global) | {best.get('mastery_corr', float('nan')):.4f} |\n")
        f.write(f"| Gain Correlation (global) | {best.get('gain_corr', float('nan')):.4f} |\n")
        f.write(f"| Mastery Correlation (macro) | {best.get('mastery_corr_macro', float('nan')):.4f} |\n")
        f.write(f"| Gain Correlation (macro) | {best.get('gain_corr_macro', float('nan')):.4f} |\n")
        f.write(f"| Monotonicity Violation Rate | {best.get('monotonicity_violation_rate', float('nan')):.4f} |\n")
        f.write(f"| Retention Violation Rate | {best.get('retention_violation_rate', float('nan')):.4f} |\n")
        f.write(f"| Gain Future Alignment | {best.get('gain_future_alignment', float('nan')):.4f} |\n")
        f.write(f"| Peer Influence Share | {best.get('peer_influence_share', float('nan')):.4f} |\n")
        f.write("\nReproducibility Checklist (partial)\n")
        f.write("- Config saved with MD5\n- Environment captured\n- Seeds recorded\n- Train/validation loaded via init_dataset4train\n")
        f.write(f"- Artifact hashes logged (peer={model.peer_hash}, difficulty={model.diff_hash}, cold_start={model.cold_start})\n")
        if model.cold_start:
            f.write("- NOTE: cold_start=True (peer/difficulty artifacts missing); interpretability metrics limited.\n")


if __name__ == '__main__':
    main()

# End minimal GainAKT3 training script