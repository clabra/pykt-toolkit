# Implementation Plan: BKT Augmentation for Raw Datasets

**Objective**: Augment raw interaction datasets (e.g., `assist2009`) with population-level BKT parameters and predictions.

## 1. Directory Structure
Ensure the following directory exists for the target dataset:
`data/[dataset_name]/bkt/`

## 2. Default Mapping (Step 1)
Create a `default_dictionary.json` mapping the raw CSV columns to pyBKT's expected fields.
Example for `assist2009`:
```json
{
  "order_id": "order_id",
  "skill_name": "skill_name",
  "correct": "correct",
  "user_id": "user_id"
}
```
Save to: `data/assist2009/bkt/default_dictionary.json`

## 3. Fit Model and Save Parameters (Steps 2 & 3)
- Use `pyBKT.models.Model` to fit the entire dataset.
- Extract trained parameters per skill.
- Save population-level estimates to `data/assist2009/bkt/parameters.json`.
- Structure of `parameters.json`:
```json
{
  "skill_name_1": {
    "prior": 0.1,
    "learns": 0.2,
    "slips": 0.1,
    "guesses": 0.3
  },
  ...
}
```

## 4. Dataset Augmentation (Step 4)
- Augment the raw CSV with:
  1. `bkt_p_correct`: Probability of correctness predicted by BKT.
  2. `bkt_p_l0`: $P(L_0)$ (prior) for that interaction's skill.
  3. `bkt_p_t`: $P(T)$ (learn rate) for that interaction's skill.
- Save as: `data/assist2009/bkt/skill_builder_data_corrected_collapsed_bkt.csv`.

## 5. Script Implementation (Step 5)
Create `examples/augment_raw_with_bkt.py` with the following CLI:
`python examples/augment_raw_with_bkt.py --dataset assist2009`

The script will:
1. Lookup raw path from `dname2paths`.
2. Determine `defaults` dictionary (hardcoded or loaded from JSON).
3. Perform training.
4. Export parameters.
5. Generate the augmented CSV.

## 6. Verification
- Compare `bkt_p_correct` with actual `correct` values (simple sanity check).
- Verify `bkt_p_l0` and `bkt_p_t` match the values in `parameters.json` for each skill.
