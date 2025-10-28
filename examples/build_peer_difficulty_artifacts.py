#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deterministic preprocessing script to build peer similarity and historical difficulty artifacts
for GainAKT3 external context modules.

Outputs (per dataset):
  data/peer_index/<dataset>/peer_index.pkl
  data/peer_index/<dataset>/METADATA.json
  data/difficulty/<dataset>/difficulty_table.parquet
  data/difficulty/<dataset>/METADATA.json

The script derives:
  - peer_correct_rate[item_id]
  - peer_attempt_count[item_id]
  - peer_state_cluster_centroids (K x d_mastery) using k-means on student mastery proxy vectors
  - skill_local_transfer_vector[item_id] (avg gain proxy across peers after item)
  - difficulty_logit (item-level) using a simple Rasch-like logistic calibration: logit(p_correct)
  - moving_variance & stability_score across time buckets

Determinism:
  - Set global seeds (numpy, random, torch if available) from --seed.
  - KMeans uses fixed n_init=1 and a deterministic initialization (k-means++ with seeded RNG).
  - All parameter and environment metadata written to METADATA.json with sha256 checksums of main artifact file.

Assumptions:
  - Processed interactions are available under data/<dataset>/interactions.csv with columns:
        student_id,item_id,skill_id,timestamp,correct
    (If not present, the script aborts.)
  - skill_id may be multi-valued separated by ';'. We treat first skill_id for simplicity initially.

We keep implementation lean; advanced artifact refinements (drift windows, confidence intervals) can be added later.
"""
import argparse
import json
import os
import sys
import hashlib
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description='Build peer & difficulty artifacts for GainAKT3.')
    p.add_argument('--dataset', required=True, help='Dataset name (folder under data/)')
    p.add_argument('--peer_K', type=int, default=8, help='Number of peer mastery centroids.')
    p.add_argument('--gamma', type=float, default=4.0, help='Scaling factor for centroid softmax weights (future use).')
    p.add_argument('--time_bucket', choices=['week','month'], default='week', help='Temporal bucketing granularity for difficulty drift.')
    p.add_argument('--min_attempts', type=int, default=5, help='Minimum attempts to trust peer/difficulty stats; below uses global averages.')
    p.add_argument('--seed', type=int, default=42, help='Global seed for determinism.')
    p.add_argument('--output_root', default='data', help='Base data directory.')
    p.add_argument('--mastery_proxy', choices=['item_correct_sequence','student_item_accuracy'], default='item_correct_sequence', help='Strategy for building mastery proxy vectors.')
    p.add_argument('--max_rows', type=int, default=0, help='Optional cap for interactions rows for faster prototyping (0 = no cap).')
    return p.parse_args()


def load_interactions(dataset: str, output_root: str, max_rows: int):
    path = os.path.join(output_root, dataset, 'interactions.csv')
    if not os.path.exists(path):
        print(f'ERROR: interactions file not found at {path}', file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    required_cols = {'student_id','item_id','skill_id','timestamp','correct'}
    if not required_cols.issubset(df.columns):
        print(f'ERROR: interactions.csv missing required columns: {required_cols - set(df.columns)}', file=sys.stderr)
        sys.exit(1)
    if max_rows > 0:
        df = df.head(max_rows)
    return df


def bucket_timestamp(series: pd.Series, granularity: str):
    dt = pd.to_datetime(series, unit='s', errors='coerce')
    if granularity == 'week':
        return dt.dt.isocalendar().week.astype(int)
    else:  # month
        return dt.dt.month.astype(int)


def build_peer_index(df: pd.DataFrame, args) -> dict:
    # Basic peer stats
    item_groups = df.groupby('item_id')
    peer_correct_rate = item_groups['correct'].mean().to_dict()
    peer_attempt_count = item_groups['correct'].count().to_dict()

    # Mastery proxy per student (mean correctness per primary skill)
    df['primary_skill'] = df['skill_id'].astype(str).str.split(';').str[0]
    skill_list = sorted(df['primary_skill'].unique())
    skill_index = {s: i for i, s in enumerate(skill_list)}
    num_skills = len(skill_list)
    student_skill_correct = defaultdict(lambda: np.zeros(num_skills, dtype=np.float32))
    student_skill_counts = defaultdict(lambda: np.zeros(num_skills, dtype=np.int32))
    for row in df.itertuples(index=False):
        sid = row.student_id
        sk = getattr(row, 'primary_skill')
        si = skill_index[sk]
        student_skill_correct[sid][si] += row.correct
        student_skill_counts[sid][si] += 1

    student_vectors = []
    for sid in student_skill_correct:
        counts = student_skill_counts[sid]
        vec = np.divide(student_skill_correct[sid], np.maximum(counts, 1), dtype=np.float32)
        student_vectors.append(vec)

    if len(student_vectors) == 0:
        print('ERROR: No student vectors constructed; aborting.')
        sys.exit(1)

    if not SKLEARN_AVAILABLE:
        print('WARNING: sklearn not available; generating deterministic random centroids.')
        rng = np.random.default_rng(args.seed)
        arr = np.stack(student_vectors, axis=0)
        if arr.shape[0] >= args.peer_K:
            sel = rng.choice(arr.shape[0], size=args.peer_K, replace=False)
            centroids = arr[sel]
        else:
            # Pad by sampling with replacement
            sel = rng.choice(arr.shape[0], size=args.peer_K, replace=True)
            centroids = arr[sel]
    else:
        kmeans = KMeans(n_clusters=args.peer_K, random_state=args.seed, n_init=1)
        centroids = kmeans.fit(np.stack(student_vectors, axis=0)).cluster_centers_

    # Simplified transfer vectors: centroid mean
    centroid_mean = centroids.mean(axis=0)
    skill_local_transfer_vector = {item_id: centroid_mean.astype(np.float32)
                                   for item_id in peer_correct_rate.keys()}

    peer_index = {
        'peer_correct_rate': peer_correct_rate,
        'peer_attempt_count': peer_attempt_count,
        'peer_state_cluster_centroids': centroids.astype(np.float32),
        'skill_local_transfer_vector': skill_local_transfer_vector,
        'skills': skill_list,
        'num_skills': num_skills,
        'mastery_proxy_type': args.mastery_proxy,
    }
    return peer_index


def build_difficulty_table(df: pd.DataFrame, args) -> pd.DataFrame:
    df['primary_skill'] = df['skill_id'].astype(str).str.split(';').str[0]
    df['time_bucket'] = bucket_timestamp(df['timestamp'], args.time_bucket)

    item_stats = df.groupby(['time_bucket', 'item_id']).agg(
        attempts=('correct', 'count'),
        correct_rate=('correct', 'mean')
    ).reset_index()
    eps = 1e-6
    # Rasch-like inverted difficulty (higher correct_rate -> lower difficulty)
    item_stats['difficulty_logit'] = np.log(np.clip(1 - item_stats['correct_rate'], eps, 1 - eps) / np.clip(item_stats['correct_rate'], eps, 1 - eps))

    var_map = {}
    stability_map = {}
    for item_id, group in item_stats.groupby('item_id'):
        cr = group['correct_rate'].values
        window = 3
        mv = []
        for i in range(len(cr)):
            start = max(0, i - window + 1)
            mv.append(float(np.var(cr[start:i+1])))
        mv = np.array(mv)
        norm = mv / (mv.max() + 1e-8) if mv.max() > 0 else mv
        stability = 1.0 - norm
        for idx, tb in enumerate(group['time_bucket'].values):
            var_map[(tb, item_id)] = mv[idx]
            stability_map[(tb, item_id)] = stability[idx]

    item_stats['moving_variance'] = [var_map[(tb, i)] for tb, i in zip(item_stats['time_bucket'], item_stats['item_id'])]
    item_stats['stability_score'] = [stability_map[(tb, i)] for tb, i in zip(item_stats['time_bucket'], item_stats['item_id'])]
    return item_stats


def write_peer_index(peer_index: dict, dataset: str, output_root: str, args):
    out_dir = os.path.join(output_root, 'peer_index', dataset)
    os.makedirs(out_dir, exist_ok=True)
    import pickle
    artifact_path = os.path.join(out_dir, 'peer_index.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump(peer_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    sha = sha256_file(artifact_path)
    metadata = {
        'source': dataset,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'generator_commit': get_git_commit(),
        'parameters': {
            'K': args.peer_K,
            'time_bucket': args.time_bucket,
            'mastery_proxy': args.mastery_proxy,
        },
        'sha256': sha,
        'artifact': 'peer_index.pkl'
    }
    with open(os.path.join(out_dir, 'METADATA.json'), 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return artifact_path, sha


def write_difficulty_table(difficulty_df: pd.DataFrame, dataset: str, output_root: str, args):
    out_dir = os.path.join(output_root, 'difficulty', dataset)
    os.makedirs(out_dir, exist_ok=True)
    artifact_path = os.path.join(out_dir, 'difficulty_table.parquet')
    difficulty_df.to_parquet(artifact_path, index=False)
    sha = sha256_file(artifact_path)
    metadata = {
        'source': dataset,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'generator_commit': get_git_commit(),
        'parameters': {
            'time_bucket': args.time_bucket,
            'min_attempts': args.min_attempts,
        },
        'sha256': sha,
        'artifact': 'difficulty_table.parquet'
    }
    with open(os.path.join(out_dir, 'METADATA.json'), 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return artifact_path, sha


def get_git_commit():
    try:
        import subprocess
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        return out
    except Exception:
        return 'unknown'


def main():
    args = parse_args()
    deterministic_seed(args.seed)
    df = load_interactions(args.dataset, args.output_root, args.max_rows)

    print('[INFO] Building peer index...')
    peer_index = build_peer_index(df, args)
    peer_path, peer_sha = write_peer_index(peer_index, args.dataset, args.output_root, args)
    print(f'[INFO] Peer index written: {peer_path} (sha256={peer_sha})')

    print('[INFO] Building difficulty table...')
    difficulty_df = build_difficulty_table(df, args)
    diff_path, diff_sha = write_difficulty_table(difficulty_df, args.dataset, args.output_root, args)
    print(f'[INFO] Difficulty table written: {diff_path} (sha256={diff_sha})')

    summary = {
        'dataset': args.dataset,
        'peer_index_path': peer_path,
        'peer_index_sha256': peer_sha,
        'difficulty_table_path': diff_path,
        'difficulty_table_sha256': diff_sha,
        'generated_at': datetime.utcnow().isoformat() + 'Z'
    }
    print('[SUMMARY]', json.dumps(summary))

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
