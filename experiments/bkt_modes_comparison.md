# BKT Evaluation Modes Comparison - assist2009

## Mode 1: Skill-Level Evaluation

**Training**: Skill-level BKT on 4 folds
**Validation**: Skill-level (fold 5)
**Test**: Skill-level (held-out test set with fold=-1)

### Results (5-fold CV):

**Validation (skill-level):**
- AUC:  0.7100 ± 0.0069
- ACC:  0.7093 ± 0.0048
- RMSE: 0.4414 ± 0.0027

**Test (skill-level):**
- AUC:  **0.7144 ± 0.0005**
- ACC:  0.7026 ± 0.0007
- RMSE: 0.4458 ± 0.0002

---

## Mode 2: Question-Level Evaluation (Late Fusion)

**Training**: Skill-level BKT on 4 folds
**Validation**: Skill-level (fold 5)
**Test**: Question-level with late fusion (mean), no model updates

**Prediction Method**: Uses trained P(L₀), P(S), P(G) parameters
- P(correct) = P(L₀) × (1 - P(S)) + (1 - P(L₀)) × P(G)
- No belief updates during test evaluation (prevents data leakage)
- Averages skill-level predictions for multi-skill questions

### Results (5-fold CV):

**Validation (skill-level):**
- AUC:  0.7100 ± 0.0069
- ACC:  0.7093 ± 0.0048
- RMSE: 0.4414 ± 0.0027

**Test (question-level with late fusion):**
- AUC:  **0.6097 ± 0.0008**
- ACC:  0.6556 ± 0.0050
- RMSE: 0.4678 ± 0.0003

---

## Key Findings:

1. **No Data Leakage**: Question-level AUC (0.6097) is now reasonable, not 1.0
2. **Performance Drop**: Question-level AUC is lower than skill-level (0.6097 vs 0.7144)
   - This is expected: using only P(L₀) without sequential updates
   - BKT's strength is updating beliefs based on observed responses
3. **Fair Comparison**: This matches neural model evaluation protocol
   - Neural models also use late fusion for multi-skill questions
   - Both use pre-trained models without test-time updates

## Comparison with Neural Model (gtransformer with Active Grounding):

**Neural Model (fold 4):**
- Question-level AUC (late fusion): **0.5133**
- Question-level ACC (late fusion): 0.6395

**BKT Question-level (fold 4):**
- Question-level AUC (late fusion): **0.6095**
- Question-level ACC (late fusion): 0.6548

**Winner**: BKT outperforms the neural model on question-level evaluation!
- BKT: 0.6097 ± 0.0008 AUC
- Neural: 0.5133 AUC (fold 4)
