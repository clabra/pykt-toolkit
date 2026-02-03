# BKT Repeat Bug Fix - February 2, 2026

## Bug Summary

**Critical bug discovered in `examples/train_bkt.py`**: The BKT parameter estimation was including **repeat/review problems** in the training data, leading to severely biased parameter estimates.

⚠️ **IMPORTANT**: This bug only affects **ablation=none** experiments (which use BKT grounding/probing losses). Models trained with **ablation=all** are completely unaffected as they skip all BKT-related losses.

## Impact

### Affected Datasets
- **algebra2005**: 31.4% of observations were repeats
- **bridge2algebra2006**: 0.4% of observations were repeats  
- **assist2009**: 16.4% of observations were repeats

### Symptoms
1. **Artificially low learning rates (T≈0)**: 
   - algebra2005: 73.5% of observations had T < 0.01
   - bridge2algebra2006: 39.1% of observations had T < 0.01
   - Most frequent skills had T = 0.001-0.003 (essentially zero)

2. **Inflated prior knowledge (L0)**:
   - bridge2algebra2006 skill 28: 97.6% first-attempt correct rate
   - bridge2algebra2006: 50% of all skills had >80% correct rate

3. **Poor structural encoding selectivity**:
   - algebra2005 L₀ selectivity: 0.200 (should be higher)
   - Bridge/algebra T selectivity: 0.358 (suppressed by T≈0)

## Root Cause

**BKT models initial learning, not review/practice.** When repeat problems are included:

1. **Students already know the skill** → High first-attempt success rate
2. **No room for learning to occur** → BKT estimates T≈0 (no transition to mastery)
3. **High guess rates compensate** → BKT finds best fit is "students guess correctly" not "students learn"

Example (algebra2005 skill 29 - 8.7% of all data):
```
Including repeats:
  L₀ = 0.625, T = 0.0009, Slip = 0.159, Guess = 0.720
  → 72% guess rate! Students aren't learning, they're guessing.
```

## The Fix

### Code Changes

**File**: `examples/train_bkt.py`  
**Function**: `prepare_bkt_data(df)`

**Before** (BUGGY):
```python
for order_id, (skill, response, mask) in enumerate(zip(concepts, responses, selectmasks)):
    if mask == 1:
        records.append({
            'user_id': uid,
            'skill_name': skill,
            'correct': response,
            'order_id': order_id
        })
```

**After** (FIXED):
```python
# Parse is_repeat column
if has_repeat_col:
    is_repeats = [int(r) for r in row['is_repeat'].split(',') if r != '-1']
else:
    is_repeats = [0] * len(concepts)

# EXCLUDE REPEATS when training BKT
for order_id, (skill, response, mask, is_repeat) in enumerate(zip(concepts, responses, selectmasks, is_repeats)):
    if mask == 1 and is_repeat == 0:  # CRITICAL FIX
        records.append({...})
```

### What Changed

1. **Check for `is_repeat` column** in dataset
2. **Filter out observations where `is_repeat == 1`** before BKT training
3. **Log the number of filtered observations** for transparency

## Next Steps

### Datasets Need Retraining

All BKT parameters must be regenerated with the fix:

```bash
# Retrain BKT for each affected dataset
python examples/train_bkt.py --dataset algebra2005
python examples/train_bkt.py --dataset bridge2algebra2006
python examples/train_bkt.py --dataset assist2009
python examples/train_bkt.py --dataset assist2015
python examples/train_bkt.py --dataset nips_task34
```

### Expected Improvements

After retraining with the fix:

1. **Higher learning rates (T)**: Skills should show meaningful learning (T > 0.01)
2. **More realistic L₀**: Prior knowledge won't be inflated by repeated practice
3. **Lower guess rates**: BKT will model actual learning, not guessing
4. **Better selectivity**: Structural encoding probes will have more variance to recover

### Regenerate Targets

After BKT retraining, regenerate targets:

```bash
python examples/generate_bkt_soft_labels.py --dataset algebra2005
python examples/generate_bkt_soft_labels.py --dataset bridge2algebra2006
# ... etc
```

### Rerun Validation

After target regeneration, rerun structural encoding validation:

```bash
python examples/results/structural_encoding_validation.py --exp_dir <experiment_path>
```

## Historical Context

This bug explains why:
- algebra2005 showed weak L₀ selectivity (Δ=0.200) despite good variance
- Top skills in algebra/bridge had T≈0.001-0.003 (seemed pedagogically implausible)
- Selectivity varied dramatically across datasets (not just dataset characteristics)

The bug was hidden because **the biased BKT parameters appeared statistically valid** - high guess rates and low learning rates CAN be legitimate in poorly designed curricula. However, the true cause was data contamination from including review problems.

## Validation Checklist

After applying the fix:

- [ ] Retrain BKT for all 5 datasets
- [ ] Verify T values are no longer clustered at ~0.001
- [ ] Regenerate BKT targets (bkt_targets_train_valid.npz)
- [ ] Rerun structural encoding validation
- [ ] Update H1.1 table in paper.md with new selectivity scores
- [ ] Compare old vs new BKT parameters to quantify impact

---

**Status**: Bug identified and fixed in code. Datasets awaiting retraining.
