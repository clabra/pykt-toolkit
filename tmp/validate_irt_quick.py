"""
Quick IRT Target Validation - Focus on M_ref Correlation

Investigates root cause of IRT alignment failure.
"""

import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import sys
import os

# Load IRT targets
targets_path = 'data/assist2015/irt_dynamic_targets_fold0.pkl'
print("Loading IRT targets...")
with open(targets_path, 'rb') as f:
    data = pickle.load(f)

beta_irt = data['skill_difficulties']
theta_data = data['theta_trajectories']
m_ref_data = data['m_ref_trajectories']

print(f"✓ Loaded Dynamic IRT targets")
print(f"  Skills: {len(beta_irt)}")
print(f"  Students: {len(theta_data)}")
print(f"  M_ref sequences: {len(m_ref_data)}")
print()

# Load dataset to get actual responses
sys.path.append(os.getcwd())
from pykt.datasets import init_dataset4train
import json

with open('configs/data_config.json') as f:
    data_config = json.load(f)

# Fix paths
for dataset_key in data_config:
    if 'dpath' in data_config[dataset_key]:
        dpath = data_config[dataset_key]['dpath']
        if dpath.startswith('../data'):
            data_config[dataset_key]['dpath'] = dpath.replace('../data', 'data')

print("Loading dataset...")
train_loader, valid_loader = init_dataset4train('assist2015', 'dkt', data_config, 0, 64)
print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
print()

# Collect M_ref predictions and actual responses
print("Analyzing M_ref correlation...")
all_m_ref = []
all_responses = []

for split_name, loader in [('train', train_loader), ('valid', valid_loader)]:
    for batch in loader:
        uids = batch['uids'].numpy()  # [B]
        responses = batch['shft_rseqs'].numpy()  # [B, L]
        masks = batch['smasks'].numpy()  # [B, L]
        
        for i, uid in enumerate(uids):
            uid_key = int(uid)
            
            if uid_key in m_ref_data:
                m_ref_seq = m_ref_data[uid_key]
                response_seq = responses[i]
                mask_seq = masks[i]
                
                # Only use valid (non-masked) positions
                for t in range(len(response_seq)):
                    if mask_seq[t] and t < len(m_ref_seq):
                        all_m_ref.append(m_ref_seq[t])
                        all_responses.append(response_seq[t])

all_m_ref = np.array(all_m_ref)
all_responses = np.array(all_responses)

print(f"Total interactions analyzed: {len(all_responses):,}")
print()

# Compute correlations
pearson_corr, p_pearson = pearsonr(all_m_ref, all_responses)

# Compute AUC
auc = roc_auc_score(all_responses, all_m_ref)

# Compute calibration
mae = np.mean(np.abs(all_m_ref - all_responses))
rmse = np.sqrt(np.mean((all_m_ref - all_responses) ** 2))

print("=" * 80)
print("RESULTS: M_ref Quality Assessment")
print("=" * 80)
print()
print(f"Pearson Correlation:  {pearson_corr:.4f} (p={p_pearson:.4e})")
print(f"AUC (Predictive):     {auc:.4f}")
print(f"MAE:                  {mae:.4f}")
print(f"RMSE:                 {rmse:.4f}")
print()

print(f"M_ref statistics:")
print(f"  Mean:  {np.mean(all_m_ref):.4f}")
print(f"  Std:   {np.std(all_m_ref):.4f}")
print(f"  Range: [{np.min(all_m_ref):.4f}, {np.max(all_m_ref):.4f}]")
print()

print(f"Response statistics:")
print(f"  Mean:  {np.mean(all_responses):.4f} (success rate)")
print(f"  Count 1s: {np.sum(all_responses == 1):,}")
print(f"  Count 0s: {np.sum(all_responses == 0):,}")
print()

# DIAGNOSIS
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()

if pearson_corr > 0.7:
    print("✅ IRT targets are HIGH QUALITY")
    print("   M_ref correlates strongly with actual responses")
    print("   Problem is likely in model architecture or training")
    print()
    print("   NEXT STEPS:")
    print("   1. Increase λ_target to 0.9 (prioritize alignment over performance)")
    print("   2. Train for 100+ epochs to allow full warm-up")
    print("   3. Try learnable scaling factor: M = σ(α × (θ - β))")
    print("   4. Check if model capacity is sufficient")
elif pearson_corr > 0.5:
    print("⚠️  IRT targets are MODERATE QUALITY")
    print("   M_ref partially correlates with responses")
    print("   IRT captures some patterns but not all")
    print()
    print("   NEXT STEPS:")
    print("   1. Recalibrate IRT with more iterations")
    print("   2. Try 2PL or 3PL IRT models (more flexible)")
    print("   3. Consider alternative reference model (BKT)")
    print("   4. Or switch to Path A (performance-first, λ=0.0)")
else:
    print("❌ IRT targets are POOR QUALITY")
    print("   M_ref does NOT correlate with actual responses")
    print("   IRT formula fundamentally incompatible with dataset")
    print("   → THIS IS THE ROOT CAUSE of alignment failure")
    print()
    print("   EXPLANATION:")
    print(f"   - l_21 = BCE(M_IRT, M_ref) measures if learned predictions match M_ref")
    print(f"   - But M_ref itself doesn't match reality (correlation={pearson_corr:.2f})")
    print(f"   - Model can't align to bad targets no matter how high λ is")
    print()
    print("   NEXT STEPS:")
    print("   1. Recalibrate IRT from scratch (check convergence)")
    print("   2. Verify Rasch assumptions (constant ability, unidimensionality)")
    print("   3. Try different IRT implementation or model")
    print("   4. Switch to different reference model (BKT, DAS3H)")
    print("   5. Or abandon alignment (Path A: set λ=0.0, optimize for AUC)")

print()
print("=" * 80)
