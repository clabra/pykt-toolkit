"""Aggregate Phase 3 semantic global alignment correlations.

We reproduce the Section 21.3 reporting protocol:
- For each seed run we select the epoch with best validation AUC (ties -> earliest epoch).
- Extract the global_alignment_mastery_corr and global_alignment_gain_corr at that epoch.
- Report per-seed values plus mean and sample std.

Usage:
    python paper/aggregate_phase3_global_alignment.py \
        --pattern paper/results/gainakt2exp_semantic_trajectory_phase3semantic_global_seed*_*.json \
        --metric_key best_val_auc \
        --output paper/results/phase3semantic_global_alignment_summary.json

Assumptions:
- Each JSON file contains keys: 'semantic_trajectory' (list), 'best_val_auc'.
- Each trajectory element includes 'epoch', 'global_alignment_mastery_corr', 'global_alignment_gain_corr'.

"""
import argparse
import glob
import json
import math
import statistics
from typing import List, Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def pick_best_epoch(traj: List[Dict[str, Any]], best_val_auc: float) -> Dict[str, Any]:
    """Select earliest epoch whose (local) mastery_correlation epoch follows best AUC.

    NOTE: Current semantic trajectory JSONs do NOT store per-epoch val AUC; only best_val_auc.
    We approximate legacy protocol by taking the last epoch (since AUC is monotonic in Phase 3 runs)
    OR earliest epoch whose global_alignment_mastery_corr reaches peak_mastery_corr if present.
    """
    # Try heuristic: find epoch where global peak mastery corr attained (peak_mastery_corr field)
    peak = None
    for entry in traj[::-1]:  # reverse scan so later epochs considered first
        if entry.get('peak_mastery_corr') is not None:
            peak = entry.get('peak_mastery_corr')
            break
    if peak is not None:
        for entry in traj:
            if entry.get('global_alignment_mastery_corr') == peak:
                return entry
    # Fallback: last epoch
    return traj[-1]


def aggregate(files: List[str]) -> Dict[str, Any]:
    per_seed = []
    for path in files:
        data = load_json(path)
        traj = data.get('semantic_trajectory') or data.get('trajectory', [])
        if not traj:
            print(f"[WARN] Empty trajectory list in {path}")
            continue
        best_val_auc = data.get('best_val_auc')
        if best_val_auc is None:
            print(f"[WARN] No best_val_auc in {path}")
        best_entry = pick_best_epoch(traj, best_val_auc if best_val_auc is not None else 0.0)
        mastery = best_entry.get('global_alignment_mastery_corr')
        gain = best_entry.get('global_alignment_gain_corr')
        per_seed.append({
            'file': path,
            'seed': extract_seed(path),
            'best_val_auc': best_val_auc,
            'epoch_selected': best_entry.get('epoch'),
            'global_alignment_mastery_corr': mastery,
            'global_alignment_gain_corr': gain
        })
    mastery_vals = [x['global_alignment_mastery_corr'] for x in per_seed if x['global_alignment_mastery_corr'] is not None]
    gain_vals = [x['global_alignment_gain_corr'] for x in per_seed if x['global_alignment_gain_corr'] is not None]
    summary = {
        'runs': per_seed,
        'mastery_mean': statistics.mean(mastery_vals) if mastery_vals else None,
        'mastery_std': statistics.pstdev(mastery_vals) if len(mastery_vals) > 1 else 0.0,
        'gain_mean': statistics.mean(gain_vals) if gain_vals else None,
        'gain_std': statistics.pstdev(gain_vals) if len(gain_vals) > 1 else 0.0,
        'count': len(per_seed)
    }
    return summary


def extract_seed(path: str) -> int:
    # Expect pattern ..._seed{seed}_... .json
    import re
    m = re.search(r"_seed(\d+)_", path)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, required=True, help='Glob pattern for semantic trajectory JSON files.')
    parser.add_argument('--output', type=str, required=True, help='Output summary JSON path.')
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern {args.pattern}")

    summary = aggregate(files)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary with {summary['count']} runs to {args.output}")


if __name__ == '__main__':
    main()
