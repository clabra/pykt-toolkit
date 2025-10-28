"""Evaluation script for GainAKT3 (real dataset integration)

Loads best checkpoint and evaluates on real validation (fold) and test splits using PyKT loaders.
Falls back to synthetic loader if --use_synthetic specified.
Masked metrics computed over valid interaction positions only.
"""
import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

from pykt.models.gainakt3 import create_gainakt3_model
from pykt.datasets import init_dataset4train, init_test_datasets


def synthetic_loader(num_students=64, seq_len=100, num_c=50, batches=6):
    for _ in range(batches):
        q = torch.randint(0, num_c, (num_students, seq_len))
        r = torch.randint(0, 2, (num_students, seq_len))
        y = torch.randint(0, 2, (num_students, seq_len)).float()
        yield {'cseqs': q.long(), 'rseqs': r.long(), 'shft_rseqs': y.long(), 'masks': torch.ones_like(q).long()}


def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            cseqs = batch['cseqs'].to(device)
            rseqs = batch['rseqs'].to(device)
            shft_r = batch['shft_rseqs'].to(device)
            mask = batch['masks'].to(device).float()
            out = model(cseqs.long(), rseqs.long())
            preds = out['predictions']
            active = mask > 0
            if active.sum() == 0:
                continue
            preds_all.append(preds[active].cpu().numpy())
            labels_all.append(shft_r[active].cpu().numpy())
    if not preds_all:
        return float('nan'), float('nan')
    y_true = np.concatenate(labels_all)
    y_score = np.concatenate(preds_all)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')
    acc = accuracy_score(y_true > 0.5, y_score > 0.5)
    return auc, acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate GainAKT3 model (real data)")
    parser.add_argument('--experiment_path', required=True, help='Path to experiment folder containing model_best.pth and config.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic evaluation instead of real loaders')
    parser.add_argument('--fold', type=int, default=0, help='Validation fold index used during training')
    args = parser.parse_args()

    config_path = os.path.join(args.experiment_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json at {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_cfg = config.get('data', {})
    model_cfg = {
        'num_c': data_cfg.get('num_c', 50),
        'seq_len': data_cfg.get('seq_len', 100),
        'dataset': config.get('experiment', {}).get('dataset', 'assist2015'),
        'peer_K': config.get('model', {}).get('peer_K', 8),
        'beta_difficulty': config.get('model', {}).get('beta_difficulty', 1.0),
        'artifact_base': 'data',
        'device': args.device
    }
    model = create_gainakt3_model(model_cfg)

    ckpt_best = os.path.join(args.experiment_path, 'model_best.pth')
    if os.path.exists(ckpt_best):
        state = torch.load(ckpt_best, map_location=args.device)
        sd = state.get('state_dict', state)
        model.load_state_dict(sd)
    else:
        print(f"Warning: best checkpoint not found at {ckpt_best}; evaluating random-initialized model.")

    if args.use_synthetic:
        val_loader = synthetic_loader(num_students=64, seq_len=model_cfg['seq_len'], num_c=model_cfg['num_c'], batches=6)
        test_loader = synthetic_loader(num_students=64, seq_len=model_cfg['seq_len'], num_c=model_cfg['num_c'], batches=6)
    else:
        # Reconstruct data_config minimal for loader
        data_config = {
            model_cfg['dataset']: {
                'dpath': f"/workspaces/pykt-toolkit/data/{model_cfg['dataset']}",
                'num_q': 0,
                'num_c': model_cfg['num_c'],
                'input_type': ['concepts'],
                'max_concepts': 1,
                'min_seq_len': 3,
                'maxlen': model_cfg['seq_len'],
                'emb_path': '',
                'train_valid_original_file': 'train_valid.csv',
                'train_valid_file': 'train_valid_sequences.csv',
                'folds': [0,1,2,3,4],
                'test_original_file': 'test.csv',
                'test_file': 'test_sequences.csv',
                'test_window_file': 'test_window_sequences.csv'
            }
        }
        # Validation loader reconstruction (same fold index) + test loader
        train_loader, val_loader = init_dataset4train(model_cfg['dataset'], 'gainakt3', data_config, args.fold, 64)
        test_loader, _, _, _ = init_test_datasets(data_config[model_cfg['dataset']], 'gainakt3', 64)
    val_auc, val_acc = evaluate(model, val_loader, args.device)
    test_auc, test_acc = evaluate(model, test_loader, args.device)

    # Interpretability metrics snapshot
    with torch.no_grad():
        sample_out = model(torch.randint(0, model_cfg['num_c'], (1, model_cfg['seq_len'])).to(args.device),
                           torch.randint(0, 2, (1, model_cfg['seq_len'])).to(args.device))
    interpretability = {
        'peer_influence_share': float(sample_out['peer_influence_share']),
        'difficulty_adjustment_magnitude': float(sample_out['difficulty_adjustment_magnitude']),
        'peer_hash': sample_out['peer_hash'],
        'difficulty_hash': sample_out['difficulty_hash'],
        'cold_start': bool(sample_out['cold_start'])
    }

    results = {
        'val_auc': val_auc,
        'val_accuracy': val_acc,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'interpretability': interpretability
    }
    out_path = os.path.join(args.experiment_path, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
