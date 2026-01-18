# Dual Evaluation Quick Reference

## Quick Start

### 1. Launch Dual Evaluation (Recommended)
```bash
cd /home/conchalabra/projects/dl/pykt-toolkit/examples
./launch_dual_eval.sh "0,1,2" 3
```

**Parameters:**
- First arg: GPU IDs (comma-separated, quoted)
- Second arg: Max parallel workers

### 2. Monitor Progress
```bash
tail -f examples/dual_eval_*.log
```

### 3. Check Individual Experiment Logs
```bash
# Supervised predictions (p_sup)
tail saved_model/gtransformer_*/eval_dual_supervised.log

# Reference predictions (p_ref - BKT logic)
tail saved_model/gtransformer_*/eval_dual_reference.log
```

---

## Alternative: Direct Python Invocation

```bash
cd /home/conchalabra/projects/dl/pykt-toolkit/examples
python3 run_dual_evaluation.py --gpu_ids "0,1,2" --max_workers 3
```

---

## Expected Output

### During Execution
```
[HH:MM:SS] ================================================================================
[HH:MM:SS] Evaluating Exp 090230: Grounded only (no probing)
[HH:MM:SS] Config: Grounded=✅, Probing=❌, Personalization=❌
[HH:MM:SS] Prediction Type: supervised
[HH:MM:SS] GPU: 0
[HH:MM:SS] ================================================================================

[HH:MM:SS] ✅ Evaluation completed successfully
[HH:MM:SS]    AUC: 0.7800
[HH:MM:SS]    ACC: 0.7423
[HH:MM:SS]    Log: saved_model/gtransformer_assist2009_qid_20260115_090230/eval_dual_supervised.log
```

### Final Summary
```
================================================================================
SUMMARY: Dual Evaluation Results
================================================================================

Exp ID  Description                       Config                              AUC (p_sup)  AUC (p_ref)  Δ_AUC
133835  Ablated baseline (2/8)            Grounded=❌, Probing=❌, Person=❌    0.7803       N/A         N/A
090230  Grounded only (no probing)        Grounded=✅, Probing=❌, Person=❌    0.7800       0.665       0.115
334772  Aligned Grounding (with probing)  Grounded=✅, Probing=✅, Person=❌    0.7786       0.722       0.057
948799  Full (with personalization)       Grounded=✅, Probing=✅, Person=✅    0.7785       0.728       0.051

================================================================================
KEY FINDINGS
================================================================================
1. Probing improves p_ref: +5.7 pp (0.665 → 0.722)
2. Personalization improves p_ref: +0.6 pp (0.722 → 0.728)
3. Interpretability cost: -0.53 pp (0.7838 → 0.7785 via 123509→948799)
4. BKT logic beats classical BKT: +11.8 pp (0.728 vs 0.610)
```

### Output Files
```
examples/
├── dual_evaluation_results.csv          # Summary table (CSV format)
└── dual_eval_YYYYMMDD_HHMMSS.log       # Full execution log

saved_model/gtransformer_assist2009_qid_*/
├── eval_dual_supervised.log             # p_sup evaluation log
├── eval_dual_reference.log              # p_ref evaluation log (if grounded)
└── eval_results.json                    # Latest results (overwritten per run)
```

---

## Troubleshooting

### Check if experiments exist
```bash
ls -ld saved_model/gtransformer_assist2009_qid_*090230*
ls -ld saved_model/gtransformer_assist2009_qid_*334772*
ls -ld saved_model/gtransformer_assist2009_qid_*948799*
ls -ld saved_model/gtransformer_assist2009_qid_*133835*
```

### If experiment not found
- Update experiment paths in `run_dual_evaluation.py` EXPERIMENTS dict
- Use `find` to locate: `find saved_model -name "*090230*" -type d`

### Check GPU availability
```bash
nvidia-smi
```

### Kill background process
```bash
# Find process
ps aux | grep run_dual_evaluation

# Kill by PID
kill <PID>
```

---

## Timeline Estimates

| GPUs | Workers | Approx. Time |
|------|---------|--------------|
| 3    | 3       | ~30-45 min   |
| 2    | 2       | ~45-60 min   |
| 1    | 1       | ~90-120 min  |

**Note**: Time varies based on:
- Dataset size (assist2009 ≈ 17.7K questions)
- GPU model (V100/A100/etc)
- I/O speed

---

## Protocol Compliance

✅ **Question-level evaluation**: `--fusion_type late_fusion`  
✅ **Mean average fusion**: Default late fusion mode  
✅ **Same as benchmarks**: Matches `run_benchmarks_paper.py --mode evaluation`  
✅ **GPU distribution**: Round-robin across available GPUs  
✅ **CPU throttling**: 5 threads per worker  

See `examples/DUAL_EVAL_ALIGNMENT.md` for detailed protocol verification.

---

## Next Steps

After dual evaluation completes:

1. **Analyze Results**
   ```bash
   cat examples/dual_evaluation_results.csv
   ```

2. **Update Validation Tables** (Step 3)
   - Edit `paper/paper_validation.md`
   - Replace single-AUC tables with dual-evaluation tables
   - Add p_ref column to Section 5.2 (Ablation Studies)
   - Update Section 5.6 (Baseline Comparisons)

3. **Revise Narrative** (Step 4)
   - Emphasize triple contribution:
     1. New SOTA architecture (0.7838 > 0.7825 AKT)
     2. Nearly-free interpretability (-0.53 pp cost)
     3. Practical BKT predictions (+11.8 pp vs classical)

---

## Questions?

See documentation:
- Protocol details: `assistant/quickstart.txt` (lines 66-80)
- Alignment verification: `examples/DUAL_EVAL_ALIGNMENT.md`
- Original concern: `paper/interpretability_concern_analysis.md`
- Refined narrative: `paper/interpretability_validation_refined.md`
