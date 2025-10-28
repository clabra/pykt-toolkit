#!/usr/bin/env python3
"""Build peer similarity and difficulty artifacts for GainAKT3.

Generates:
  data/peer_index/<dataset>/peer_index.pkl
  data/difficulty/<dataset>/difficulty_table.parquet
  Each with METADATA.json including SHA256 and generation parameters.

This script avoids modifying existing data_original; it derives statistics from already processed data (placeholder demo).

Phase: Initial deterministic scaffold. Replace placeholder computations with real aggregation later.
"""
import os
import json
import argparse
import hashlib
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def write_metadata(path: str, meta: dict):
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)

def build_peer_index(dataset: str, out_dir: str, K: int, seed: int):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    # Placeholder centroids in d_model space = 128
    centroids = rng.normal(size=(K,128)).astype('float32')
    peer_index = {
        'peer_state_cluster_centroids': centroids.tolist(),
        'peer_correct_rate': 0.5,
        'peer_attempt_count': 1000,
        'skill_local_transfer_vector': rng.normal(size=(128,)).astype('float32').tolist(),
        'version': 1
    }
    pkl_path = os.path.join(out_dir, 'peer_index.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(peer_index, f)
    meta = {
        'source': dataset,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'generator_commit': '<pending>',
        'parameters': {'K': K, 'seed': seed},
        'sha256': sha256_bytes(pickle.dumps(peer_index)),
        'artifact': 'peer_index'
    }
    write_metadata(os.path.join(out_dir, 'METADATA.json'), meta)
    return pkl_path, meta

def build_difficulty_table(dataset: str, out_dir: str, seed: int):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    # Placeholder difficulty rows: 50 items x 4 time buckets
    items = np.arange(50)
    buckets = np.arange(4)
    rows = []
    for b in buckets:
        for it in items:
            difficulty_logit = rng.normal() - 0.5
            attempt_count = rng.integers(50, 500)
            correct_rate = rng.uniform(0.2, 0.9)
            moving_variance = rng.uniform(0.01, 0.2)
            stability_score = 1.0 - moving_variance
            rows.append({
                'time_bucket': int(b),
                'item_id': int(it),
                'difficulty_logit': float(difficulty_logit),
                'attempt_count': int(attempt_count),
                'correct_rate': float(correct_rate),
                'moving_variance': float(moving_variance),
                'stability_score': float(stability_score)
            })
    df = pd.DataFrame(rows)
    parquet_path = os.path.join(out_dir, 'difficulty_table.parquet')
    df.to_parquet(parquet_path, index=False)
    meta = {
        'source': dataset,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'generator_commit': '<pending>',
        'parameters': {'seed': seed, 'time_buckets': 4, 'items': 50},
        'sha256': sha256_bytes(df.to_parquet(index=False)),
        'artifact': 'difficulty_table'
    }
    write_metadata(os.path.join(out_dir, 'METADATA.json'), meta)
    return parquet_path, meta

def parse_args():
    p = argparse.ArgumentParser(description='Build peer & difficulty artifacts for GainAKT3.')
    p.add_argument('--dataset', type=str, default='assist2015')
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_dir', type=str, default='data')
    return p.parse_args()

def main():
    args = parse_args()
    peer_dir = os.path.join(args.data_dir, 'peer_index', args.dataset)
    diff_dir = os.path.join(args.data_dir, 'difficulty', args.dataset)
    peer_path, peer_meta = build_peer_index(args.dataset, peer_dir, args.K, args.seed)
    diff_path, diff_meta = build_difficulty_table(args.dataset, diff_dir, args.seed)
    summary = {
        'dataset': args.dataset,
        'peer_index': peer_meta,
        'difficulty_table': diff_meta
    }
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
