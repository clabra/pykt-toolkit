## Datasets
### assist2015

**Student Counts:**
- Total unique students: 19,093
- Train+Valid: 15,275 students (split into 5 folds for cross-validation)
  - Fold 0: 3,055 students
  - Fold 1: 3,055 students
  - Fold 2: 3,055 students
  - Fold 3: 3,055 students
  - Fold 4: 3,055 students
- Test: 3,818 students (fold=-1, held-out evaluation set)

**Skills:** 100 unique skills

**Cross-Validation:** When training with `--fold 0`, the model uses folds 1-4 for training (~12,220 students) and fold 0 for validation (3,055 students). The test set remains completely separate.

**Data Files:**
- **Training & Validation:**
  - `train_valid_sequences.csv` (26M) - CSV with all train+valid sequences (folds 0-4)
  - `train_valid_sequences.csv_0.pkl` (8.4M) - Pickle for fold 0 validation data only
  - `train_valid_sequences.csv_1_2_3_4.pkl` (34M) - Pickle for folds 1-4 training data when using fold 0
  
- **Test & Evaluation:**
  - `test_sequences.csv` (9.1M) - CSV with all test sequences (fold=-1)
  - `test_sequences.csv_-1.pkl` (11M) - Pickle for test data
  - `test_window_sequences.csv` (15M) - Windowed test sequences for temporal evaluation
  
- **Pre-computed Baselines:**
  - `bkt_targets.pkl` (201M) - BKT parameters trained on all 15,275 train+valid students
  - `rasch_test_iter300.pkl` (201M) - Rasch IRT difficulty estimates (300 iterations)
  - `keyid2idx.json` - ID mapping (original IDs ↔ sequential indices)

**Usage:** Training scripts automatically load fold-specific pickle files (e.g., `train_valid_sequences.csv_0.pkl` for validation when `--fold 0`). The pykt framework handles data loading through `pykt/datasets/init_dataset.py`.

### assist2009

**Student Counts:**
- Total unique students: 3,852
- Train+Valid: 3,082 students (split into 5 folds for cross-validation)
  - Fold 0: 617 students
  - Fold 1: 617 students
  - Fold 2: 616 students
  - Fold 3: 616 students
  - Fold 4: 616 students
- Test: 770 students (fold=-1, held-out evaluation set)

**Skills:** 123 unique skills

**Cross-Validation:** When training with `--fold 0`, the model uses folds 1-4 for training (~2,465 students) and fold 0 for validation (617 students). The test set remains completely separate.

**Data Files:**
- **Training & Validation:**
  - `train_valid_sequences.csv` (11M) - CSV with all train+valid sequences (folds 0-4)
  - `train_valid_sequences.csv_0.pkl` (3.2M) - Pickle for fold 0 validation data only
  - `train_valid_sequences.csv_1_2_3_4.pkl` (13M) - Pickle for folds 1-4 training data when using fold 0
  - `train_valid_sequences_quelevel.csv` (8.4M) - Question-level sequences for question-based models
  
- **Test & Evaluation:**
  - `test_sequences.csv` (3.3M) - CSV with all test sequences (fold=-1)
  - `test_sequences.csv_-1.pkl` (3.9M) - Pickle for test data
  - `test_window_sequences.csv` (81M) - Windowed test sequences for temporal evaluation
  - `test_sequences_quelevel.csv` (2.1M) - Question-level test sequences
  
- **Pre-computed Baselines:**
  - `keyid2idx.json` (327K) - ID mapping (original IDs ↔ sequential indices)

**Usage:** Training scripts automatically load fold-specific pickle files (e.g., `train_valid_sequences.csv_0.pkl` for validation when `--fold 0`). The pykt framework handles data loading through `pykt/datasets/init_dataset.py`.

## Mapping of student IDs and zero-based sequential indices

The `keyid2idx.json` file (e.g., `data/assist2015/keyid2idx.json`) is a bidirectional mapping dictionary that converts between original dataset IDs and zero-based sequential indices used internally by the pykt framework.

**Structure**:
```json
{
  "concepts": {"original_skill_id": index, ...},
  "uid": {"original_student_id": index, ...}
}
```

**Purpose**:

1. **Skill/Concept Mapping (`concepts`)**: Maps original skill IDs from the dataset to sequential indices (0, 1, 2, ...). Example: `"7014": 0` means skill ID 7014 in the original dataset becomes index 0 in the model.

2. **Student ID Mapping (`uid`)**: Maps original student IDs to sequential indices. Example: `"223214": 1038` means student ID 223214 becomes index 1038.

**Usage in Code**:
- **Forward mapping**: `keyid2idx['concepts']['7014']` → `0` (convert dataset ID to model index)
- **Reverse mapping**: Create inverse dict to convert back: `idx2key = {v: k for k, v in keyid2idx['concepts'].items()}`

This file is generated during data preprocessing and must be consistent across train/validation/test splits to ensure the same skill ID always maps to the same index.

## Windowed Test Sequences

**Purpose**: `test_window_sequences.csv` files are used for evaluating models on long student sequences using a sliding window approach.

**How it works**:

When a student's interaction sequence exceeds the maximum sequence length M (typically 200 interactions), the pykt framework applies a **sliding window** to create multiple overlapping sub-sequences. This allows:

1. **Handling long sequences**: Students with hundreds of interactions can be evaluated without truncation
2. **Temporal robustness testing**: Models are tested on different temporal segments of a student's learning journey
3. **More comprehensive evaluation**: Provides additional test samples from students with rich interaction histories

**Example**:

For a student with 201 interactions and M=200:
- **test_sequences.csv**: Contains 1 row with all 201 interactions (possibly truncated to 200)
- **test_window_sequences.csv**: Contains multiple rows with sliding windows:
  - Window 1: interactions 1-200
  - Window 2: interactions 2-201
  - (additional windows depending on implementation)

This explains why `test_window_sequences.csv` files are larger than standard test files:
- ASSIST2015: 6,103 rows (windowed) vs 3,867 rows (standard)
- ASSIST2009: Similar expansion for students with long sequences

**Evaluation Metrics**:

During model evaluation, both metrics are typically reported:
- **testauc/testacc**: Performance on standard test set (`test_sequences.csv`)
- **window_testauc/window_testacc**: Performance on windowed test set (`test_window_sequences.csv`)

The windowed metrics test temporal generalization - ensuring models perform consistently across different segments of a student's learning trajectory, not just on the most recent interactions.

**Implementation**: See `docs/source/contribute.md` for details on the window format preprocessing.