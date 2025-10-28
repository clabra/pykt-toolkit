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
    Computes AUC/Accuracy on active positions only.
    """
    model.eval()
    all_preds, all_targets = [], []
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
    if not all_preds:
        return float('nan'), float('nan')
    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_preds)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')
    acc = accuracy_score(y_true > 0.5, y_score > 0.5)
    return auc, acc


def main():
    parser = argparse.ArgumentParser(description='Train GainAKT3 (real dataset integration)')
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
    parser.add_argument('--artifact_base', default='data')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data instead of real loader (debug)')
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
            'beta_difficulty': args.beta_difficulty
        },
        'seeds': {
            'primary': args.seed
        },
        'hardware': {
            'device': args.device,
            'batch_size': args.batch_size
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
        'device': args.device
    }
    model = create_gainakt3_model(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    with open(os.path.join(exp_path, 'artifact_hashes.json'), 'w') as f:
        json.dump({'peer_hash': model.peer_hash, 'difficulty_hash': model.diff_hash, 'cold_start': model.cold_start}, f, indent=2)

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

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, args.device, optimizer, num_c)
        val_auc, val_acc = evaluate(model, val_loader, args.device)
        with torch.no_grad():
            probe = model(
                torch.randint(0, num_c, (1, seq_len)).to(args.device),
                torch.randint(0, 2, (1, seq_len)).to(args.device)
            )
        peer_share = float(probe['peer_influence_share'])
        rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_auc': val_auc, 'val_accuracy': val_acc, 'peer_influence_share': peer_share})
        print(f"Epoch {epoch} | loss={train_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")

    with open(os.path.join(exp_path, 'results.json'), 'w') as f:
        json.dump({'config_md5': config_md5, 'epochs': rows}, f, indent=2)
    import csv
    with open(os.path.join(exp_path, 'metrics_epoch.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(model.state_dict(), os.path.join(exp_path, 'model_last.pth'))
    best = max(rows, key=lambda x: x['val_auc'])
    torch.save({'state_dict': model.state_dict(), 'best_epoch': best['epoch'], 'val_auc': best['val_auc']}, os.path.join(exp_path, 'model_best.pth'))

    with open(os.path.join(exp_path, 'README.md'), 'w') as f:
        f.write(f"# Experiment {os.path.basename(exp_path)}\n\n")
        f.write("GainAKT3 training on real dataset split (assist2015) with masked losses.\n\n")
        f.write("## Summary\n")
        f.write(f"Best epoch: {best['epoch']} val_auc={best['val_auc']:.4f} val_acc={best['val_accuracy']:.4f}\n")
        f.write("\nReproducibility Checklist (partial)\n")
        f.write("- Config saved with MD5\n- Environment captured\n- Seeds recorded\n- Train/validation loaded via init_dataset4train\n")
        f.write(f"- Artifact hashes logged (peer={model.peer_hash}, difficulty={model.diff_hash}, cold_start={model.cold_start})\n")
        if model.cold_start:
            f.write("- NOTE: cold_start=True (peer/difficulty artifacts missing); interpretability metrics limited.\n")


if __name__ == '__main__':
    main()

# End minimal GainAKT3 training script