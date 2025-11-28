# Option 1b Implementation Plan

**Date**: 2025-11-28  
**Issue**: Phase 1 overfitting due to student-specific Rasch target memorization  
**Solution**: Skill-centric regularization with learnable difficulty embeddings

---

## Quick Summary

**Problem**: Model memorizes training students' Rasch targets $M_{rasch}[s,k] = \sigma(\theta_s - \beta_k)$ (includes student abilities $\theta_s$), failing to generalize to validation students.

**Solution**: Replace per-student targets with learnable skill difficulty embeddings $\beta_k$ regularized toward IRT-calibrated values.

**Impact**: Prevents overfitting, improves generalization, maintains interpretability.

---

## Implementation Checklist

### ✅ Completed
- [x] Diagnosed overfitting issue (experiment 20251128_162143_ikt_test_285650)
- [x] Analyzed root cause (student-specific target memorization)
- [x] Designed Option 1b solution
- [x] Documented issue and plan in `paper/ikt_architecture_approach.md`
- [x] Created implementation plan (this file)

### 🔲 Phase 1: Model Architecture (`pykt/models/ikt.py`)

- [ ] **1.1**: Add skill difficulty embedding to `__init__`:
  ```python
  self.skill_difficulty_emb = nn.Embedding(num_c, 1)
  ```

- [ ] **1.2**: Remove `rasch_targets` parameter from `forward()` signature

- [ ] **1.3**: Update `forward()` to compute skill-only targets:
  ```python
  beta_skills = self.skill_difficulty_emb.weight.squeeze()  # [K]
  mastery_targets = torch.sigmoid(-beta_skills)  # [K]
  beta_batch = mastery_targets.unsqueeze(0).unsqueeze(0).expand(B, L, K)
  ```

- [ ] **1.4**: Add `beta_targets` to return dict:
  ```python
  return {'y': y_pred, 'skill_vector': skill_vector, 'beta_targets': beta_batch}
  ```

- [ ] **1.5**: Update `compute_loss()` signature to accept `beta_irt` and `lambda_reg`

- [ ] **1.6**: Implement regularization loss:
  ```python
  beta_learned = self.skill_difficulty_emb.weight.squeeze()
  L_reg = ((beta_learned - beta_irt) ** 2).mean()
  ```

- [ ] **1.7**: Update Phase 1 loss: `L_total = L1 + lambda_reg * L_reg`

- [ ] **1.8**: Update Phase 2 loss: `L_total = L1 + lambda_penalty * L2_penalty + lambda_reg * L_reg`

- [ ] **1.9**: Update docstring to explain new approach

### 🔲 Phase 2: Training Script (`examples/train_ikt.py`)

- [ ] **2.1**: Load skill difficulties from `rasch_targets.pkl`:
  ```python
  skill_difficulties = torch.tensor(
      [rasch_data['skill_difficulties'][k] for k in range(num_skills)],
      dtype=torch.float32
  )
  ```

- [ ] **2.2**: Initialize model embedding with IRT values:
  ```python
  with torch.no_grad():
      model.skill_difficulty_emb.weight.copy_(
          skill_difficulties.unsqueeze(1)
      )
  ```

- [ ] **2.3**: Remove per-student `rasch_batch` construction loop

- [ ] **2.4**: Remove `rasch_targets` from `model.forward()` call

- [ ] **2.5**: Update `compute_loss()` call to pass `beta_irt` and `lambda_reg`

- [ ] **2.6**: Update metrics collection to include `L_reg`

- [ ] **2.7**: Add `L_reg` column to `metrics_validation.csv`

### 🔲 Phase 3: Evaluation Script (`examples/eval_ikt.py`)

- [ ] **3.1**: Remove `load_rasch_targets()` call

- [ ] **3.2**: Remove `rasch_batch` construction

- [ ] **3.3**: Remove `rasch_targets` from `model.forward()` call

- [ ] **3.4**: Add difficulty correlation metric:
  ```python
  beta_learned = model.skill_difficulty_emb.weight.squeeze().cpu().numpy()
  difficulty_correlation = np.corrcoef(beta_learned, beta_irt)[0, 1]
  ```

- [ ] **3.5**: Add correlation to `metrics_test.json`

### 🔲 Phase 4: Configuration Updates

- [ ] **4.1**: Add `lambda_reg` to `configs/parameter_default.json`:
  ```json
  {
    "lambda_reg": 0.1,
    "lambda_reg_help": "Regularization strength for skill difficulty embeddings"
  }
  ```

- [ ] **4.2**: Update `examples/parameters_audit.py` to include `lambda_reg` in iKT parameters

- [ ] **4.3**: Update `examples/check_defaults_consistency.py` if needed

### 🔲 Phase 5: Documentation

- [ ] **5.1**: ✅ Update `paper/ikt_architecture_approach.md` (DONE)

- [ ] **5.2**: Update `paper/STATUS_iKT.md` with new approach

- [ ] **5.3**: Update model docstring in `pykt/models/ikt.py`

- [ ] **5.4**: Update reproducibility documentation if needed

### 🔲 Phase 6: Testing and Validation

- [ ] **6.1**: Smoke test (2 epochs):
  ```bash
  ./run.sh --short_title option1b_smoke --model ikt --epochs 2 --dataset assist2015 --fold 0
  ```

- [ ] **6.2**: Check training/validation convergence (no overfitting)

- [ ] **6.3**: Verify all metrics computed correctly

- [ ] **6.4**: Ablation study with different `lambda_reg` values:
  - [ ] `lambda_reg=0.01`
  - [ ] `lambda_reg=0.1` (default)
  - [ ] `lambda_reg=1.0`

- [ ] **6.5**: Full training run (200 epochs)

- [ ] **6.6**: Compare with baseline (old approach)

- [ ] **6.7**: Validate success criteria

---

## Success Criteria

After implementation and training, verify:

1. **No Overfitting**: Validation MSE decreases or stays stable (not increasing like before)
2. **Difficulty Alignment**: Correlation between learned and IRT difficulties > 0.8
3. **Performance**: AUC on test set ≥ baseline (old approach)
4. **Interpretability**: Violation rate < 10% in Phase 2

---

## Files to Modify

### Core Model
- `pykt/models/ikt.py` (major changes)

### Training/Evaluation
- `examples/train_ikt.py` (moderate changes)
- `examples/eval_ikt.py` (moderate changes)

### Configuration
- `configs/parameter_default.json` (add lambda_reg)
- `examples/parameters_audit.py` (add lambda_reg to expected params)

### Documentation
- `paper/ikt_architecture_approach.md` ✅ (done)
- `paper/STATUS_iKT.md` (update approach description)

---

## Rollback Plan

If Option 1b doesn't improve generalization:

1. Git revert to commit before Option 1b implementation
2. Consider alternative approaches:
   - **Option 2**: Use validation set for early stopping in Phase 1
   - **Option 3**: Add dropout to mastery head
   - **Option 4**: Reduce Phase 1 epochs significantly

Current commit before Option 1b: `2ccb6ef`

---

## Notes

- Keep old approach code commented for comparison
- Document all parameter changes in commit messages
- Run full evaluation pipeline after each major change
- Monitor GPU memory usage (embeddings add minimal overhead)
- Test with smaller datasets first (algebra05) if needed

---

## Timeline Estimate

- Phase 1 (Model): 1-2 hours
- Phase 2 (Training): 1 hour
- Phase 3 (Eval): 30 minutes
- Phase 4 (Config): 15 minutes
- Phase 5 (Docs): 30 minutes
- Phase 6 (Testing): Variable (depends on training time)

**Total implementation**: ~4 hours  
**Testing/validation**: 1-2 days (including training runs)
