# H1.1 Structural Encoding Validation - Campaign Processing

This directory contains scripts for running diagnostic probing validation across entire campaigns.

## Scripts

### `run_structural_validation_campaign.py`

Automates H1.1 structural encoding validation for all datasets in a campaign directory.

**What it does:**
1. Auto-detects all datasets in campaign (or processes specified datasets)
2. Runs `structural_encoding_validation.py` for each fold of each dataset
3. Aggregates results across folds using `aggregate_structural_validation.py`
4. Generates campaign-level summary table for Paper Table 4

**Usage:**

```bash
# Process all datasets in campaign (auto-detect)
python examples/validation/run_structural_validation_campaign.py \
  --campaign_dir experiments/20260202_222106_benchpaper_698838

# Process specific datasets only
python examples/validation/run_structural_validation_campaign.py \
  --campaign_dir experiments/20260202_222106_benchpaper_698838 \
  --datasets algebra2005,assist2015,bridge2algebra2006,nips_task34

# Skip folds that already have results (resume interrupted run)
python examples/validation/run_structural_validation_campaign.py \
  --campaign_dir experiments/20260202_222106_benchpaper_698838 \
  --skip_existing
```

**Output Files:**

1. **Per-fold results** (each dataset):
   - `experiments/<campaign>/gtransformer/<dataset>/validation/structural_encoding_fold_0.json`
   - `experiments/<campaign>/gtransformer/<dataset>/validation/structural_encoding_fold_1.json`
   - ... through fold_4.json
   - Individual plots: `h11_fidelity_l0_fold_0.png`, `h11_fidelity_t_fold_0.png`, etc.

2. **Aggregated results** (each dataset):
   - `experiments/<campaign>/gtransformer/<dataset>/validation/structural_encoding_aggregated.json`
   - Contains mean ± std for all metrics across 5 folds

3. **Campaign-level summary**:
   - `experiments/<campaign>/structural_validation_summary.csv`
   - CSV with all datasets and constructs, ready for analysis

4. **Paper table**:
   - `experiments/<campaign>/h11_structural_encoding_table.md`
   - Formatted markdown table ready to paste into `paper/paper.md` (Paper Table 4)

**Example Output:**

```
======================================================================
H1.1 Structural Encoding Validation Campaign
======================================================================
Campaign: 20260202_222106_benchpaper_698838
Datasets: algebra2005, assist2015, bridge2algebra2006, nips_task34
======================================================================

──────────────────────────────────────────────────────────────────────
Processing dataset: algebra2005
──────────────────────────────────────────────────────────────────────
  Found 5 folds

  [fold_0]
  Running: python examples/results/structural_encoding_validation.py ...
  ✓ Completed validation for fold_0_<id>

  ... (folds 1-4)

  Completed 5/5 folds

  Aggregating results for algebra2005...

  --- L0 ---
  fidelity_r2         : 0.5495 ± 0.0636
  fidelity_pearson    : 0.7434 ± 0.0407
  selectivity_std     : 0.6160 ± 0.0652
  selectivity_strict  : 0.5700 ± 0.0802

  --- T ---
  fidelity_r2         : 0.5146 ± 0.0797
  fidelity_pearson    : 0.7209 ± 0.0496
  selectivity_std     : 0.5696 ± 0.0802
  selectivity_strict  : 0.5200 ± 0.0900

... (other datasets)

======================================================================
Generating campaign-level summary...
======================================================================

✓ Campaign summary saved to: experiments/<campaign>/structural_validation_summary.csv
✓ Paper table saved to: experiments/<campaign>/h11_structural_encoding_table.md

======================================================================
Campaign validation complete!
======================================================================
```

## Individual Scripts

### `structural_encoding_validation.py` (in `examples/results/`)

Runs validation for a single fold. Called automatically by campaign script.

```bash
python examples/results/structural_encoding_validation.py \
  --exp_dir experiments/<campaign>/gtransformer/<dataset>/fold_0_<id> \
  --output_dir experiments/<campaign>/gtransformer/<dataset>/validation
```

### `aggregate_structural_validation.py` (in `examples/results/`)

Aggregates results across folds for one dataset. Called automatically by campaign script.

```bash
python examples/results/aggregate_structural_validation.py \
  --campaign_dir experiments/<campaign>/gtransformer/<dataset>
```

## Metrics Explained

**Fidelity (R²)**: How well linear probes can recover BKT parameters from transformer hidden states
- Range: [0, 1] (higher = better encoding)
- R² = 1 means perfect recovery, R² = 0 means no predictive power

**Pearson (r)**: Linear correlation between predicted and true BKT parameters
- Range: [-1, 1] (values near ±1 = strong correlation)
- Measures strength of linear relationship

**Selectivity (Δ R²)**: Difference between fidelity and control task R²
- **Standard**: Observation-level shuffling (baseline control)
- **Strict**: Skill-consistent shuffling (stronger control, prevents artifacts)
- **Threshold > 0.5**: Strong structural encoding (BKT is dominant organizing principle)

## Time Estimates

For a campaign with 4 datasets × 5 folds = 20 validation runs:

- **Per fold**: ~3-5 minutes (includes t-SNE computation)
- **Total time**: ~60-100 minutes for full campaign
- **Parallelization**: Currently sequential; can be parallelized if needed

## Troubleshooting

**Issue**: `No datasets found in campaign`
- **Solution**: Check that campaign directory contains `gtransformer/` subdirectory

**Issue**: `Validation failed for fold_X`
- **Solution**: Check that fold directory contains:
  - Model checkpoint: `best_valid_model.pth`
  - Test data: `test_question_window_sequences.csv`
  - BKT parameters: `data/<dataset>/bkt_skill_params.pkl`

**Issue**: Script interrupted mid-run
- **Solution**: Resume with `--skip_existing` flag to skip completed folds

## Integration with Paper Workflow

After running campaign validation:

1. Review campaign summary: `experiments/<campaign>/structural_validation_summary.csv`
2. Copy table from: `experiments/<campaign>/h11_structural_encoding_table.md`
3. Paste into `paper/paper.md` under "Paper Table 4 - Probing Selectivity"
4. Add notes about dataset-specific findings and interpretation

## Example: Bug-Fixed BKT Campaign

```bash
# Process the 4-dataset campaign from Feb 2-3, 2026
python examples/validation/run_structural_validation_campaign.py \
  --campaign_dir experiments/20260202_222106_benchpaper_698838 \
  --datasets algebra2005,assist2015,bridge2algebra2006,nips_task34

# Process assist2009 (separate campaign with 8 attention heads)
python examples/validation/run_structural_validation_campaign.py \
  --campaign_dir experiments/20260202_222258_benchpaper_893468 \
  --datasets assist2009
```

Expected results: Complete Paper Table 4 with all 5 datasets showing selectivity metrics for L₀ and T constructs, validating H1.1 Structural Encoding hypothesis.
