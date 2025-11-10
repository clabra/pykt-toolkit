#!/usr/bin/env python3
"""
Quick diagnostic: Check if threshold mechanism is computing skill_readiness.
"""
import torch
import numpy as np
from pathlib import Path

exp_dir = Path('examples/experiments/20251110_041636_gainakt2exp_threshold_v3_seed42_370193')
ckpt_path = exp_dir / 'model_best.pth'

print("=" * 80)
print("THRESHOLD DIAGNOSTIC")
print("=" * 80)

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')
model_config = ckpt['model_config']
state_dict = ckpt['model_state_dict']

print("\n1. CHECK THRESHOLD PARAMETER")
print("-" * 80)
print(f"use_learnable_threshold in config: {model_config.get('use_learnable_threshold')}")

has_threshold = any('threshold' in k for k in state_dict.keys())
print(f"Has threshold parameter in state_dict: {has_threshold}")

if 'module.threshold_raw' in state_dict:
    threshold_raw = state_dict['module.threshold_raw'].item()
    threshold = torch.sigmoid(torch.tensor(threshold_raw)).item()
    print(f"  ✓ threshold_raw: {threshold_raw:.6f}")
    print(f"  ✓ sigmoid(threshold_raw): {threshold:.6f}")
    
    if abs(threshold - 0.5) < 0.05:
        print(f"  ⚠️  Barely moved from initialization!")
else:
    print(f"  ❌ NO threshold_raw in checkpoint")

print("\n2. CHECK PREDICTION HEAD DIMENSIONS")
print("-" * 80)

# Check prediction head to see if it has the extra dimension
pred_head_keys = [k for k in state_dict.keys() if 'prediction_head' in k]
print(f"Prediction head layers: {len(pred_head_keys)}")

for key in sorted(pred_head_keys):
    val = state_dict[key]
    print(f"  {key.replace('module.', '')}: shape={val.shape}")
    if 'weight' in key and len(val.shape) == 2:
        in_dim, out_dim = val.shape[1], val.shape[0]
        print(f"    → Input dim: {in_dim}, Output dim: {out_dim}")
        
        # For d_model=512: without threshold [512*3=1536], with threshold [512*3+1=1537]
        if in_dim == 1537:
            print(f"    ✓ Has +1 dimension for skill_readiness!")
        elif in_dim == 1536:
            print(f"    ❌ NO extra dimension - threshold not integrated!")

print("\n3. ANALYZE TRAINING METRICS")
print("-" * 80)

import csv
metrics_csv = exp_dir / 'metrics_epoch.csv'
if metrics_csv.exists():
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Epochs trained: {len(rows)}")
    print("\nCorrelation evolution:")
    print("Epoch | Mastery Corr | Gain Corr")
    print("------|--------------|----------")
    for row in rows:
        epoch = row['epoch']
        mastery_corr = float(row['mastery_correlation'])
        gain_corr = float(row['gain_correlation'])
        print(f"  {epoch:2s}  |    {mastery_corr:.4f}    |  {gain_corr:.4f}")
    
    last_mastery = float(rows[-1]['mastery_correlation'])
    if last_mastery < 0.1:
        print(f"\n❌ CRITICAL: Mastery correlation stuck at {last_mastery:.4f}")
        print(f"   This is the ROOT PROBLEM!")

print("\n4. DIAGNOSIS")
print("=" * 80)

# Load model to test forward pass
from pykt.models.gainakt2_exp import create_exp_model
model = create_exp_model(model_config)

# Check if model has methods
print(f"Model has apply_soft_threshold: {hasattr(model, 'apply_soft_threshold')}")
print(f"Model has compute_skill_readiness: {hasattr(model, 'compute_skill_readiness')}")
print(f"Model has threshold_raw: {hasattr(model, 'threshold_raw')}")

if hasattr(model, 'threshold_raw'):
    print(f"\n✓ Model initialized with threshold!")
    print(f"  threshold_raw value: {model.threshold_raw.item():.6f}")
else:
    print(f"\n❌ Model NOT initialized with threshold!")
    print(f"   This means create_exp_model didn't pass the parameter properly")

# Test with dummy data
print("\n5. TEST FORWARD PASS")
print("-" * 80)

# Create dummy batch
B, L, C = 2, 10, 100
q = torch.randint(0, C, (B, L))
r = torch.randint(0, 2, (B, L))

try:
    with torch.no_grad():
        # Don't load weights, just test structure
        output = model.forward_with_states(q=q, r=r, qry=q, batch_idx=0)
    
    print(f"✓ Forward pass successful")
    print(f"  Output keys: {list(output.keys())}")
    
    if 'skill_readiness' in output:
        print(f"  ✓ skill_readiness present!")
        readiness = output['skill_readiness']
        print(f"    Shape: {readiness.shape}")
    else:
        print(f"  ❌ skill_readiness NOT present!")
        print(f"     The threshold mechanism is NOT being used during forward pass")
        
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

conclusion_points = []

# Check if threshold was properly initialized
if not has_threshold:
    conclusion_points.append("❌ Threshold parameter missing from checkpoint")
elif abs(threshold - 0.5) < 0.05:
    conclusion_points.append(f"⚠️  Threshold barely learned (still ~0.5)")

# Check correlations
if last_mastery < 0.1:
    conclusion_points.append(f"❌ Mastery correlation critically low ({last_mastery:.4f})")
    conclusion_points.append("   ROOT CAUSE: Mastery values don't reflect actual skill levels")

# Check integration
if hasattr(model, 'threshold_raw'):
    conclusion_points.append("✓ Model has threshold parameter")
else:
    conclusion_points.append("❌ Model doesn't have threshold parameter")

if conclusion_points:
    for point in conclusion_points:
        print(point)

print("\n💡 RECOMMENDATION:")
print("The learnable threshold approach has fundamental issues:")
print("1. Mastery values from projection head don't correlate with performance")
print("2. Adding threshold on top of bad mastery doesn't help")
print("3. The mastery-performance loss is not strong enough to force correlation")
print("\nSuggested next steps:")
print("- Abandon threshold approach (doesn't address root cause)")
print("- Focus on fixing mastery projection to actually reflect skill levels")
print("- Consider: stronger supervision, different architecture, or auxiliary tasks")

print("\n" + "=" * 80)
