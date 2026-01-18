# Dual Evaluation Protocol Alignment

## Overview
This document confirms that `run_dual_evaluation.py` follows the **exact same evaluation protocol** as `run_benchmarks_paper.py` to ensure fair, apples-to-apples comparison.

## Protocol Verification

### ✅ 1. Question-Level, Late Fusion Evaluation

**run_benchmarks_paper.py** (lines 468-470):
```python
# Add required flags
eval_cmd = eval_cmd.replace("--fusion_type 'early_fusion,late_fusion'", "--fusion_type late_fusion")
if "--fusion_type" not in eval_cmd:
    eval_cmd += " --fusion_type late_fusion"
```

**run_dual_evaluation.py** (lines 92-97):
```python
cmd = [
    sys.executable,
    "wandb_gtransformer_predict.py",
    "--save_dir", save_dir,
    "--bz", "64",
    "--use_wandb", "0",
    "--fusion_type", "late_fusion",  # Question-level late fusion (mean average)
    "--prediction_type", prediction_type
]
```

**Alignment**: ✅ Both use `--fusion_type late_fusion`

---

### ✅ 2. Execution from examples/ Directory

**run_benchmarks_paper.py** (line 481):
```python
# Run from examples/ directory for relative path compatibility
cwd = Path(PROJECT_ROOT) / "examples"
```

**run_dual_evaluation.py** (line 100):
```python
# Execute evaluation from examples/ directory (matches run_benchmarks_paper.py)
cwd = PROJECT_ROOT / "examples"
```

**Alignment**: ✅ Both execute from `PROJECT_ROOT/examples/`

---

### ✅ 3. GPU Distribution via CUDA_VISIBLE_DEVICES

**run_benchmarks_paper.py** (line 477):
```python
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

**run_dual_evaluation.py** (lines 83-84):
```python
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

**Alignment**: ✅ Both set `CUDA_VISIBLE_DEVICES` per worker

---

### ✅ 4. CPU Thread Limiting

**run_benchmarks_paper.py** (lines 130-131):
```python
# CPU Throttling: 80% of 40 cores / 6 workers = ~5 threads each
env["OMP_NUM_THREADS"] = "5"
env["MKL_NUM_THREADS"] = "5"
```

**run_dual_evaluation.py** (lines 85-86):
```python
env["OMP_NUM_THREADS"] = "5"  # Match run_benchmarks_paper.py CPU throttling
env["MKL_NUM_THREADS"] = "5"
```

**Alignment**: ✅ Both set 5 threads per process

---

### ✅ 5. PYTHONPATH Configuration

**run_benchmarks_paper.py** (line 478):
```python
env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}"
```

**run_dual_evaluation.py** (line 87):
```python
env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}"
```

**Alignment**: ✅ Both ensure PROJECT_ROOT is in PYTHONPATH

---

### ✅ 6. Evaluation Command Pattern

**run_benchmarks_paper.py** (stored in config.json):
```bash
/usr/bin/python3 examples/wandb_gtransformer_predict.py \
    --save_dir <path> \
    --bz 64 \
    --use_wandb 0 \
    --fusion_type late_fusion
```

**run_dual_evaluation.py** (lines 90-97):
```python
cmd = [
    sys.executable,  # Equivalent to python3
    "wandb_gtransformer_predict.py",
    "--save_dir", save_dir,
    "--bz", "64",
    "--use_wandb", "0",
    "--fusion_type", "late_fusion",
    "--prediction_type", prediction_type  # ONLY DIFFERENCE
]
```

**Alignment**: ✅ Identical except for `--prediction_type` parameter

---

### ✅ 7. Log File Location

**run_benchmarks_paper.py** (line 474):
```python
log_path = exp_dir / "eval_benchmark.log"
```

**run_dual_evaluation.py** (line 101):
```python
log_path = Path(save_dir) / f"eval_dual_{prediction_type}.log"
```

**Alignment**: ✅ Both write logs to experiment directory

---

### ✅ 8. Parallel Execution with ProcessPoolExecutor

**run_benchmarks_paper.py** uses ProcessPoolExecutor for training (lines 650-675)
**run_dual_evaluation.py** uses ProcessPoolExecutor for evaluation (lines 155-185)

**Alignment**: ✅ Both leverage parallel execution for efficiency

---

## Summary

| Protocol Aspect | run_benchmarks_paper.py | run_dual_evaluation.py | Status |
|---|---|---|---|
| Fusion Type | `late_fusion` | `late_fusion` | ✅ |
| Working Directory | `PROJECT_ROOT/examples/` | `PROJECT_ROOT/examples/` | ✅ |
| GPU Assignment | `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | ✅ |
| CPU Threads | 5 threads/process | 5 threads/process | ✅ |
| PYTHONPATH | Includes PROJECT_ROOT | Includes PROJECT_ROOT | ✅ |
| Batch Size | 64 | 64 | ✅ |
| Evaluation Script | `wandb_gtransformer_predict.py` | `wandb_gtransformer_predict.py` | ✅ |
| Log Location | Experiment directory | Experiment directory | ✅ |
| Parallel Execution | ProcessPoolExecutor | ProcessPoolExecutor | ✅ |

## Key Difference

The **only** difference between benchmark evaluation and dual evaluation is the addition of:
```bash
--prediction_type {supervised|reference}
```

This parameter controls whether we evaluate:
- **supervised**: Neural head predictions (`p_sup`) — default mode used in all prior benchmarks
- **reference**: BKT logic wrapper predictions (`p_ref`) — interpretable predictions for validation

## Conclusion

✅ **run_dual_evaluation.py is fully aligned with run_benchmarks_paper.py evaluation protocol**

This ensures that:
1. Dual evaluation results are directly comparable to benchmark results
2. Question-level late fusion is consistently applied
3. Resource management (GPU/CPU) follows production standards
4. The only variable is the prediction source (neural vs. BKT logic)

## References

- Evaluation Protocol: `assistant/quickstart.txt` (lines 66-80)
- Benchmark Implementation: `examples/run_benchmarks_paper.py` (lines 258-490)
- Dual Evaluation: `examples/run_dual_evaluation.py`
- Background Launcher: `examples/launch_dual_eval.sh`
