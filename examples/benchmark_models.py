#!/usr/bin/env python3
"""
Benchmark multiple KT models with consistent parameters.

This script trains multiple knowledge tracing models sequentially and collects
their results for comparison. Designed to work without WandB dependencies.

After training completes, an evaluation shell script is automatically generated
that can be run manually to evaluate all successfully trained models on test data.

Usage:

    # Run benchmark in background (survives terminal closure, uses 6 GPUs in parallel)
    nohup python benchmark_models.py > /dev/null 2>&1 &

    # Monitor training progress (logs in experiments/<timestamp>_benchmark_<dataset>/out.log)
    tail -f experiments/*_benchmark_*/out.log
    
    # Check running benchmark processes
    ps aux | grep benchmark_models.py | grep -v grep
    
    # Custom output file
    python benchmark_models.py --models akt,dkt,sakt,ikt3 --dataset assist2009 \
                                --output benchmark_results.csv
    
    # Multiple folds
    python benchmark_models.py --models akt,sakt --dataset assist2009 --folds 0,1,2,3,4
    
    # Override model parameters
    python benchmark_models.py --models akt,dkt --dataset assist2009 \
                                --learning_rate 0.001 --dropout 0.3
    
    # Run training and evaluation automatically (recommended for complete results)
    python benchmark_models.py --models akt,dkt,sakt --dataset assist2009 --auto_eval
    
    # Parallel execution (DEFAULT - uses 6 GPUs simultaneously)
    python benchmark_models.py --models akt,dkt,sakt,saint,lpkt,kqn --dataset assist2009
    
    # Sequential execution (one model at a time, uses all 6 GPUs per model)
    python benchmark_models.py --models akt,dkt,sakt --dataset assist2009 --sequential
    
    # Parallel with auto evaluation (RECOMMENDED for complete benchmark)
    python benchmark_models.py --models akt,dkt,sakt --dataset assist2009 --auto_eval

Output Files (all in experiments/<timestamp>_benchmark_<dataset>/):
    - results.csv: Training results (validation metrics); updated with test metrics if --auto_eval used
    - config.json: Full benchmark configuration
    - evaluate.sh: Evaluation script for test data
    - error.log: Error log for failed trainings/evaluations
    - out.log: Complete output log of all operations

Evaluation:
    Option 1 (Automatic - Recommended):
        Use --auto_eval flag to run training and evaluation in sequence.
        Results with test metrics saved to results.csv
        
    Option 2 (Manual):
        After training completes, run the generated evaluation script:
            bash experiments/<timestamp>_benchmark_<dataset>/evaluate.sh
        
        Then check results.csv for updated metrics.
    
    Evaluation scripts used:
    - wandb_predict.py: Standard pykt evaluation for most models (full test set, concept-level)
      * Must be run from examples/ directory
      * Requires --save_dir pointing to trained model directory with config.json
      * Uses --use_wandb=0 for local evaluation without logging
      * Automatically skips question-level evaluation for single-skill datasets (e.g., ASSIST2015)
    - eval_ikt3.py: For iKT3 model
    
    Note: For datasets without test_question_file in config (ASSIST2015, statics2011, poj),
    only concept-level evaluation is performed. Multi-skill datasets (ASSIST2009) include
    additional question-level evaluation with early/late fusion.

Default Parameters:
    --dataset: "assist2015"
    --folds: "0" (single fold)
    --seed: 42
    --output: saved_model/benchmark_results_<dataset>_<timestamp>.csv
    --learning_rate: None (uses model script defaults, e.g., AKT: 1e-4)
    --d_model: None (uses model script defaults, e.g., AKT: 256)
    --dropout: None (uses model script defaults, e.g., AKT: 0.2)
    
    Fixed parameters (always set):
    --use_wandb: "0" (disabled for benchmarking)
    --add_uuid: "0" (clean directory names)
    
    Internal configuration:
    - timeout: 7200 seconds (2 hours per training)
    - parallel mode (DEFAULT): Uses 6 GPUs (1 model per GPU) for faster execution
    - sequential mode (--sequential): All models use GPUs 0-5 but run one at a time
    
    Note: All other parameters (num_blocks, d_ff, num_attn_heads, etc.) 
    come from individual model training scripts.
"""

import argparse
import subprocess
import sys
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading


# Model training script mapping (all models in pykt/models with training scripts)
MODEL_SCRIPTS = {
    "ikt3": "train_ikt3.py",
    "gkt": "wandb_gkt_train.py",
    "gainakt2exp": "wandb_gainakt2exp_train.py",
    "akt": "wandb_akt_train.py",
    "atdkt": "wandb_atdkt_train.py",
    "atkt": "wandb_atkt_train.py",
    "cskt": "wandb_cskt_train.py",
    "datakt": "wandb_datakt_train.py",
    "deep_irt": "wandb_deep_irt_train.py",
    "dimkt": "wandb_dimkt_train.py",
    "dkt": "wandb_dkt_train.py",
    "dkt+": "wandb_dkt_plus_train.py",
    "dkt_forget": "wandb_dkt_forget_train.py",
    "dkvmn": "wandb_dkvmn_train.py",
    "dtransformer": "wandb_dtransformer_train.py",
    "extrakt": "wandb_extrakt_train.py",
    "folibikt": "wandb_folibikt_train.py",
    "hawkes": "wandb_hawkes_train.py",
    "hcgkt": "wandb_hcgkt_train.py",
    "iekt": "wandb_iekt_train.py",
    "kqn": "wandb_kqn_train.py",
    "lefokt_akt": "wandb_lefokt_akt_train.py",
    "lpkt": "wandb_lpkt_train.py",
    "promptkt": "wandb_promptkt_train.py",
    "qdkt": "wandb_qdkt_train.py",
    "qikt": "wandb_qikt_train.py",
    "rekt": "wandb_rekt_train.py",
    "rkt": "wandb_rkt_train.py",
    "robustkt": "wandb_robustkt_train.py",
    "saint": "wandb_saint_train.py",
    "saint++": "wandb_saint_plus_plus_train.py",
    "sakt": "wandb_sakt_train.py",
    "simplekt": "wandb_simplekt_train.py",
    "skvmn": "wandb_skvmn_train.py",
    "sparsekt": "wandb_sparsekt_train.py",
    "stablekt": "wandb_stablekt_train.py",
    "ukt": "wandb_ukt_train.py",
}

# Model evaluation script mapping
# Standard pykt evaluation uses wandb_predict.py for full test set evaluation
MODEL_EVAL_SCRIPTS = {
    "akt": "wandb_predict.py",
    "atdkt": "wandb_predict.py",
    "atkt": "wandb_predict.py",
    "cskt": "wandb_predict.py",
    "datakt": "wandb_predict.py",
    "deep_irt": "wandb_predict.py",
    "dimkt": "wandb_predict.py",
    "dkt": "wandb_predict.py",
    "dkt+": "wandb_predict.py",
    "dkt_forget": "wandb_predict.py",
    "dkvmn": "wandb_predict.py",
    "dtransformer": "wandb_predict.py",
    "extrakt": "wandb_predict.py",
    "folibikt": "wandb_predict.py",
    "gainakt2exp": "wandb_predict.py",
    "gkt": "wandb_predict.py",
    "hawkes": "wandb_predict.py",
    "hcgkt": "wandb_predict.py",
    "iekt": "wandb_predict.py",
    "ikt3": "eval_ikt3.py",
    "kqn": "wandb_predict.py",
    "lefokt_akt": "wandb_predict.py",
    "lpkt": "wandb_predict.py",
    "promptkt": "wandb_predict.py",
    "qdkt": "wandb_predict.py",
    "qikt": "wandb_predict.py",
    "rekt": "wandb_predict.py",
    "rkt": "wandb_predict.py",
    "robustkt": "wandb_predict.py",
    "saint": "wandb_predict.py",
    "saint++": "wandb_predict.py",
    "sakt": "wandb_predict.py",
    "simplekt": "wandb_predict.py",
    "skvmn": "wandb_predict.py",
    "sparsekt": "wandb_predict.py",
    "stablekt": "wandb_predict.py",
    "ukt": "wandb_predict.py",
}


def parse_training_output(output):
    """
    Parse training output to extract final metrics.
    
    Args:
        output: String output from training script
        
    Returns:
        dict with metrics (testauc, testacc, validauc, validacc, best_epoch, save_dir)
    """
    metrics = {
        'testauc': None,
        'testacc': None,
        'window_testauc': None,
        'window_testacc': None,
        'validauc': None,
        'validacc': None,
        'best_epoch': None,
        'train_loss': None,
        'save_dir': None
    }
    
    # Look for the final summary line
    # Format: "fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch"
    lines = output.strip().split('\n')
    for line in lines:
        # Match the metrics line (contains tab-separated values)
        if '\t' in line and not line.startswith('fold'):
            parts = line.split('\t')
            if len(parts) >= 10:
                try:
                    metrics['testauc'] = float(parts[3]) if parts[3] != '-1' else None
                    metrics['testacc'] = float(parts[4]) if parts[4] != '-1' else None
                    metrics['window_testauc'] = float(parts[5]) if parts[5] != '-1' else None
                    metrics['window_testacc'] = float(parts[6]) if parts[6] != '-1' else None
                    metrics['validauc'] = float(parts[7])
                    metrics['validacc'] = float(parts[8])
                    metrics['best_epoch'] = int(parts[9])
                except (ValueError, IndexError):
                    continue
        
        # Also try to extract from epoch logs for iKT3
        if 'Best Epoch' in line or 'best epoch' in line.lower():
            match = re.search(r'AUC[:\s]+([0-9.]+)', line)
            if match:
                metrics['validauc'] = float(match.group(1))
            match = re.search(r'ACC[:\s]+([0-9.]+)', line)
            if match:
                metrics['validacc'] = float(match.group(1))
        
        # Extract save_dir path
        if 'save_dir' in line.lower() or 'saved to' in line.lower():
            # Look for directory paths like "saved_model/..." or absolute paths
            match = re.search(r'(saved_model/[^\s]+)', line)
            if match:
                metrics['save_dir'] = match.group(1)
            # Also check for full paths
            match = re.search(r'(/[^\s]+/saved_model/[^\s]+)', line)
            if match:
                metrics['save_dir'] = match.group(1)
    
    return metrics


def train_model(model_name, dataset, fold, seed, extra_args=None, gpu_id=None, benchmark_dir=None, error_log_path=None):
    """
    Train a single model and return results.
    
    Args:
        model_name: Name of the model to train
        dataset: Dataset name
        fold: Fold number
        seed: Random seed
        extra_args: Additional command-line arguments (dict)
        gpu_id: GPU ID to use for training (None = use all available)
        benchmark_dir: Directory for benchmark outputs (model checkpoints will be saved here)
        error_log_path: Path to error log file
        
    Returns:
        dict with training results
    """
    if model_name not in MODEL_SCRIPTS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_SCRIPTS.keys())}")
    
    script = MODEL_SCRIPTS[model_name]
    script_path = (Path(__file__).parent / script).resolve()
    
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    # Set environment variables for GPU/CPU resource management
    env = os.environ.copy()
    
    if gpu_id is not None:
        # Assign specific GPU for parallel execution
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Limit CPU threads per process (30 total / 6 parallel = 5 per process)
        env['OMP_NUM_THREADS'] = '5'
        env['MKL_NUM_THREADS'] = '5'
    else:
        # Use 6 out of 8 GPUs (75%) for sequential execution
        env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
        env['OMP_NUM_THREADS'] = '30'
        env['MKL_NUM_THREADS'] = '30'
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset_name", dataset,
        "--fold", str(fold),
        "--seed", str(seed),
        "--use_wandb", "0",  # Disable wandb for benchmark
        "--add_uuid", "0",  # Disable UUID for cleaner directory names
    ]
    
    # Add extra arguments
    if extra_args:
        for key, value in extra_args.items():
            cmd.extend([f"--{key}", str(value)])
    
    gpu_info = f" [GPU {gpu_id}]" if gpu_id is not None else ""
    print(f"\n{'='*80}")
    print(f"Training {model_name} on {dataset} (fold {fold}, seed {seed}){gpu_info}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run training with environment variables for resource management
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            env=env,
        )
        
        elapsed_time = time.time() - start_time
        
        # Parse output
        output = result.stdout + result.stderr
        metrics = parse_training_output(output)
        
        # Add metadata
        metrics['model'] = model_name
        metrics['dataset'] = dataset
        metrics['fold'] = fold
        metrics['seed'] = seed
        metrics['training_time_sec'] = elapsed_time
        metrics['exit_code'] = result.returncode
        metrics['success'] = result.returncode == 0 and metrics['validauc'] is not None
        
        # Add extra args to results for full reproducibility
        if extra_args:
            for key, value in extra_args.items():
                metrics[f'param_{key}'] = value
        
        if result.returncode != 0:
            error_msg = f"Training failed with exit code {result.returncode}\n"
            error_msg += f"Model: {model_name}, Dataset: {dataset}, Fold: {fold}, Seed: {seed}\n"
            error_msg += f"STDOUT:\n{result.stdout[-1000:]}\n"
            error_msg += f"STDERR:\n{result.stderr[-1000:]}\n"
            error_msg += f"{'='*80}\n"
            print(f"WARNING: {error_msg}")
            
            # Log error to file
            if error_log_path:
                with open(error_log_path, 'a') as f:
                    f.write(f"[{datetime.now().isoformat()}] {error_msg}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        error_msg = f"Training timed out after {elapsed_time:.1f}s\n"
        error_msg += f"Model: {model_name}, Dataset: {dataset}, Fold: {fold}, Seed: {seed}\n"
        error_msg += f"{'='*80}\n"
        print(f"ERROR: {error_msg}")
        
        # Log error to file
        if error_log_path:
            with open(error_log_path, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] TIMEOUT: {error_msg}")
        
        error_result = {
            'model': model_name,
            'dataset': dataset,
            'fold': fold,
            'seed': seed,
            'training_time_sec': elapsed_time,
            'exit_code': -1,
            'success': False,
            'error': 'timeout'
        }
        if extra_args:
            for key, value in extra_args.items():
                error_result[f'param_{key}'] = value
        return error_result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"{str(e)}\n"
        error_msg += f"Model: {model_name}, Dataset: {dataset}, Fold: {fold}, Seed: {seed}\n"
        error_msg += f"{'='*80}\n"
        print(f"ERROR: {error_msg}")
        
        # Log error to file
        if error_log_path:
            with open(error_log_path, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] EXCEPTION: {error_msg}")
                import traceback
                f.write(traceback.format_exc() + "\n")
        
        error_result = {
            'model': model_name,
            'dataset': dataset,
            'fold': fold,
            'seed': seed,
            'training_time_sec': elapsed_time,
            'exit_code': -1,
            'success': False,
            'error': str(e)
        }
        if extra_args:
            for key, value in extra_args.items():
                error_result[f'param_{key}'] = value
        return error_result
        
    except Exception as e:
        # Catch any unexpected errors
        elapsed_time = time.time() - start_time
        error_msg = f"Unexpected error: {str(e)}\n"
        error_msg += f"Model: {model_name}, Dataset: {dataset}, Fold: {fold}, Seed: {seed}\n"
        print(f"ERROR: {error_msg}")
        
        if error_log_path:
            with open(error_log_path, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {error_msg}")
                import traceback
                f.write(traceback.format_exc() + "\n")
        
        error_result = {
            'model': model_name,
            'dataset': dataset,
            'fold': fold,
            'seed': seed,
            'training_time_sec': elapsed_time,
            'exit_code': -1,
            'success': False,
            'error': str(e)
        }
        if extra_args:
            for key, value in extra_args.items():
                error_result[f'param_{key}'] = value
        return error_result


def run_benchmark(models, dataset, folds, seed, benchmark_dir, extra_args=None, parallel=False, max_parallel=6):
    """
    Run benchmark across multiple models and folds.
    
    Args:
        models: List of model names
        dataset: Dataset name
        folds: List of fold numbers
        seed: Random seed
        benchmark_dir: Directory to save all benchmark outputs
        extra_args: Additional arguments for training
        parallel: If True, run multiple models in parallel on different GPUs
        max_parallel: Maximum number of parallel training jobs (default: 6 GPUs)
        
    Returns:
        DataFrame with results
    """
    results = []
    
    total_experiments = len(models) * len(folds)
    current = 0
    
    # Verify benchmark directory exists
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark directory does not exist: {benchmark_dir}. "
                                "Directory must be created before running benchmark.")
    
    # Define output files in benchmark directory
    output_file = str(benchmark_path / "results.csv")
    config_file = str(benchmark_path / "config.json")
    error_log_path = str(benchmark_path / "error.log")
    out_log_path = str(benchmark_path / "out.log")
    
    # Redirect stdout and stderr to out.log
    log_file = open(out_log_path, 'w', buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    
    # Initialize error log
    with open(error_log_path, 'w') as f:
        f.write(f"Benchmark Error Log\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {dataset}, Models: {models}, Folds: {folds}, Seed: {seed}\n")
        f.write(f"{'='*80}\n\n")
    
    # Prepare benchmark configuration for saving
    benchmark_config = {
        'dataset': dataset,
        'folds': folds,
        'seed': seed,
        'models': models,
        'use_wandb': '0',
        'add_uuid': '0',
        'timeout_sec': 7200,
        'benchmark_dir': benchmark_dir,
        'output_file': output_file,
        'config_file': config_file,
        'error_log': error_log_path,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Build reproducibility command
    cmd_parts = [
        'python benchmark_models.py',
        f'--models {",".join(models)}',
        f'--dataset {dataset}',
        f'--folds {",".join(map(str, folds))}',
        f'--seed {seed}',
    ]
    
    # Add extra arguments to config and command
    if extra_args:
        benchmark_config.update({f'param_{k}': v for k, v in extra_args.items()})
        for k, v in extra_args.items():
            cmd_parts.append(f'--{k} {v}')
    
    # Complete reproducibility command
    reproducibility_command = ' '.join(cmd_parts)
    benchmark_config['reproducibility_command'] = reproducibility_command
    
    # Save benchmark config as JSON
    with open(config_file, 'w') as f:
        json.dump(benchmark_config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models)}")
    print(f"Dataset: {dataset}")
    print(f"Folds: {folds}")
    print(f"Seed: {seed}")
    print(f"Execution mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    if parallel:
        print(f"Max parallel jobs: {max_parallel} (1 model per GPU)")
    print(f"Fixed params: use_wandb=0, add_uuid=0, timeout=7200s")
    if extra_args:
        print(f"Override params: {', '.join(f'{k}={v}' for k, v in extra_args.items())}")
    print(f"Total experiments: {total_experiments}")
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"  - Results: {output_file}")
    print(f"  - Config: {config_file}")
    print(f"  - Error log: {error_log_path}")
    print(f"  - Output log: {out_log_path}")
    print(f"  - Model checkpoints: saved_model/ (pykt default location)")
    print(f"\nTo monitor progress in real-time:")
    print(f"  tail -f {out_log_path}")
    print(f"\nReproducibility command:")
    print(f"  {reproducibility_command}")
    print(f"{'='*80}\n")
    
    if parallel:
        # Parallel execution using ProcessPoolExecutor
        # Create list of all training jobs
        jobs = []
        gpu_id = 0
        for model in models:
            for fold in folds:
                jobs.append((model, dataset, fold, seed, extra_args, gpu_id % max_parallel, benchmark_dir, error_log_path))
                gpu_id += 1
        
        # Lock for thread-safe result saving
        results_lock = threading.Lock()
        completed = 0
        
        print(f"Starting parallel execution with {max_parallel} workers...\n")
        
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(train_model, *job): job 
                for job in jobs
            }
            
            # Process completed jobs as they finish
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                model_name = job[0]
                fold = job[2]
                gpu = job[5]
                
                try:
                    result = future.result()
                    with results_lock:
                        results.append(result)
                        completed += 1
                        
                        # Save intermediate results
                        df = pd.DataFrame(results)
                        df.to_csv(output_file, index=False)
                        
                        # Print summary
                        print(f"\n[{completed}/{total_experiments}] {model_name} (fold {fold}, GPU {gpu}) completed")
                        if result['success']:
                            print(f"✓ Success: validauc={result.get('validauc', 'N/A'):.4f}, "
                                  f"validacc={result.get('validacc', 'N/A'):.4f}, "
                                  f"time={result['training_time_sec']:.1f}s")
                        else:
                            print(f"✗ Failed: {result.get('error', 'unknown error')}")
                        print(f"Results saved to {output_file}")
                        
                except Exception as e:
                    print(f"ERROR processing {model_name} (fold {fold}): {str(e)}")
                    with results_lock:
                        results.append({
                            'model': model_name,
                            'dataset': dataset,
                            'fold': fold,
                            'seed': seed,
                            'success': False,
                            'error': str(e)
                        })
    else:
        # Sequential execution (original behavior)
        for model in models:
            for fold in folds:
                current += 1
                print(f"\n[{current}/{total_experiments}] Training {model} (fold {fold})")
                
                result = train_model(model, dataset, fold, seed, extra_args, gpu_id=None, 
                                   benchmark_dir=benchmark_dir, error_log_path=error_log_path)
                results.append(result)
                
                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
                
                # Print summary
                if result['success']:
                    print(f"✓ Success: validauc={result.get('validauc', 'N/A'):.4f}, "
                          f"validacc={result.get('validacc', 'N/A'):.4f}, "
                          f"time={result['training_time_sec']:.1f}s")
                else:
                    print(f"✗ Failed: {result.get('error', 'unknown error')}")
    
    # Final results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"  - Results: {output_file}")
    print(f"  - Config: {config_file}")
    print(f"  - Error log: {error_log_path}")
    print(f"\nTraining Commands Executed:")
    for model in models:
        for fold in folds:
            print(f"  - {model} (fold {fold}): python {MODEL_SCRIPTS[model]} --dataset_name {dataset} --fold {fold} --seed {seed} --use_wandb 0 --add_uuid 0")
    print(f"\nTo reproduce this benchmark:")
    print(f"  {reproducibility_command}")
    print(f"\nSummary:")
    print(df[['model', 'fold', 'validauc', 'validacc', 'success', 'training_time_sec']].to_string())
    
    # Aggregate by model
    if len(folds) > 1:
        print(f"\n{'='*80}")
        print(f"AGGREGATE RESULTS (mean ± std across folds)")
        print(f"{'='*80}")
        agg = df[df['success']].groupby('model').agg({
            'validauc': ['mean', 'std'],
            'validacc': ['mean', 'std'],
            'training_time_sec': ['mean']
        }).round(4)
        print(agg.to_string())
    
    # Restore original stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    
    print(f"Complete log saved to: {out_log_path}")
    
    return df


def find_model_save_dir(model, dataset, fold, seed, benchmark_dir=None):
    """
    Find the saved model directory based on naming patterns.
    
    Args:
        model: Model name
        dataset: Dataset name
        fold: Fold number
        seed: Random seed
        benchmark_dir: Benchmark directory to search in (optional)
        
    Returns:
        Relative path to save_dir or None if not found
    """
    # First search in benchmark directory if provided
    if benchmark_dir:
        benchmark_path = Path(benchmark_dir)
        pattern = f"{dataset}_{model}_*_{seed}_{fold}_*"
        matches = list(benchmark_path.glob(pattern))
        if matches:
            return str(matches[0].relative_to(Path(__file__).parent))
    
    # Fallback to default saved_model directory
    saved_model_dir = Path(__file__).parent / "saved_model"
    
    # Pattern: dataset_model_*_seed_fold_*
    pattern = f"{dataset}_{model}_*_{seed}_{fold}_*"
    
    # First try direct children
    matches = list(saved_model_dir.glob(pattern))
    if matches:
        # Return relative path from examples directory
        return str(matches[0].relative_to(saved_model_dir.parent))
    
    # Also search in subdirectories (e.g., saved_model/old)
    matches = list(saved_model_dir.glob(f"**/{pattern}"))
    if matches:
        # Return relative path from examples directory
        return str(matches[0].relative_to(saved_model_dir.parent))
    
    return None


def run_evaluations(df, dataset, seed, benchmark_dir):
    """
    Run evaluations for all successfully trained models and update results.
    
    Args:
        df: DataFrame with benchmark training results
        dataset: Dataset name
        seed: Random seed used in training
        benchmark_dir: Benchmark directory containing results
        
    Returns:
        Updated DataFrame with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING EVALUATIONS")
    print(f"{'='*80}")
    
    output_file = str(Path(benchmark_dir) / "results.csv")
    error_log_path = str(Path(benchmark_dir) / "error.log")
    
    eval_results = []
    successful_models = df[df['success'] == True]
    total_evals = len(successful_models)
    
    for idx, (_, row) in enumerate(successful_models.iterrows(), 1):
        model = row['model']
        fold = row['fold']
        model_seed = row['seed']
        
        print(f"\n[{idx}/{total_evals}] Evaluating {model} (fold {fold})")
        
        # Find save_dir
        save_dir = find_model_save_dir(model, dataset, fold, model_seed, benchmark_dir)
        
        if not save_dir:
            print(f"WARNING: Could not find saved model directory for {model} fold {fold}")
            eval_results.append({
                'model': model,
                'fold': fold,
                'testauc': None,
                'testacc': None,
                'eval_success': False,
                'eval_error': 'save_dir not found'
            })
            continue
        
        # Get evaluation script
        eval_script = MODEL_EVAL_SCRIPTS.get(model, "wandb_eval.py")
        
        # Build evaluation command
        # Note: wandb_predict.py must run from examples/ directory due to relative paths
        cmd = [
            sys.executable,
            eval_script,
            "--save_dir", save_dir,
            "--use_wandb", "0",
            "--bz", "256"  # Batch size for evaluation
        ]
        
        # wandb_predict.py performs standard full test set evaluation:
        # - Concept-level: test_sequences.csv (all datasets)
        # - Question-level: test_question_sequences.csv (only multi-skill datasets)
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for evaluation
            )
            
            # Parse evaluation output
            output = result.stdout + result.stderr
            testauc = None
            testacc = None
            
            # Look for test metrics in output
            for line in output.split('\n'):
                if 'testauc' in line.lower() or 'test_auc' in line.lower():
                    match = re.search(r'([0-9.]+)', line)
                    if match and testauc is None:
                        testauc = float(match.group(1))
                if 'testacc' in line.lower() or 'test_acc' in line.lower():
                    match = re.search(r'([0-9.]+)', line)
                    if match and testacc is None:
                        testacc = float(match.group(1))
            
            eval_results.append({
                'model': model,
                'fold': fold,
                'testauc': testauc,
                'testacc': testacc,
                'eval_success': result.returncode == 0,
                'eval_error': None if result.returncode == 0 else f'exit_code_{result.returncode}'
            })
            
            if result.returncode == 0:
                print(f"✓ Evaluation completed: testauc={testauc}, testacc={testacc}")
            else:
                error_msg = f"Evaluation failed with exit code {result.returncode}\n"
                error_msg += f"Model: {model}, Fold: {fold}\n"
                error_msg += f"{'='*80}\n"
                print(f"✗ {error_msg}")
                
                # Log evaluation error
                with open(error_log_path, 'a') as f:
                    f.write(f"[{datetime.now().isoformat()}] EVAL_FAILED: {error_msg}")
                
        except subprocess.TimeoutExpired:
            error_msg = f"Evaluation timed out\nModel: {model}, Fold: {fold}\n{'='*80}\n"
            print(f"ERROR: {error_msg}")
            
            # Log evaluation timeout
            with open(error_log_path, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] EVAL_TIMEOUT: {error_msg}")
            
            eval_results.append({
                'model': model,
                'fold': fold,
                'testauc': None,
                'testacc': None,
                'eval_success': False,
                'eval_error': 'timeout'
            })
        except Exception as e:
            error_msg = f"{str(e)}\nModel: {model}, Fold: {fold}\n{'='*80}\n"
            print(f"ERROR: {error_msg}")
            
            # Log evaluation exception
            with open(error_log_path, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] EVAL_EXCEPTION: {error_msg}")
                import traceback
                f.write(traceback.format_exc() + "\n")
            
            eval_results.append({
                'model': model,
                'fold': fold,
                'testauc': None,
                'testacc': None,
                'eval_success': False,
                'eval_error': str(e)
            })
    
    # Merge evaluation results with training results
    eval_df = pd.DataFrame(eval_results)
    df_merged = df.merge(
        eval_df[['model', 'fold', 'testauc', 'testacc', 'eval_success', 'eval_error']], 
        on=['model', 'fold'], 
        how='left',
        suffixes=('_train', '_eval')
    )
    
    # Update testauc and testacc columns with evaluation results
    df_merged['testauc'] = df_merged['testauc_eval'].combine_first(df_merged['testauc'])
    df_merged['testacc'] = df_merged['testacc_eval'].combine_first(df_merged['testacc'])
    df_merged = df_merged.drop(columns=['testauc_eval', 'testacc_eval'], errors='ignore')
    
    # Save consolidated results to results.csv (overwrite training-only results)
    results_file = str(Path(benchmark_dir) / "results.csv")
    df_merged.to_csv(results_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"EVALUATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"Results with evaluation metrics saved to: {results_file}")
    print(f"\nEvaluation Summary:")
    if 'eval_success' in df_merged.columns:
        print(df_merged[['model', 'fold', 'validauc', 'validacc', 'testauc', 'testacc', 'eval_success']].to_string())
    else:
        print(df_merged[['model', 'fold', 'validauc', 'validacc', 'testauc', 'testacc']].to_string())
    print(f"{'='*80}\n")
    
    return df_merged


def generate_eval_script(df, dataset, seed, benchmark_dir, extra_args=None):
    """
    Generate shell script to evaluate all successfully trained models.
    
    Args:
        df: DataFrame with benchmark results
        dataset: Dataset name
        seed: Random seed used in training
        benchmark_dir: Benchmark directory containing results
        extra_args: Additional arguments used in training
        
    Returns:
        Path to generated shell script
    """
    # Create script path in benchmark directory
    output_file = str(Path(benchmark_dir) / "results.csv")
    script_path = str(Path(benchmark_dir) / "evaluate.sh")
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#\n")
        f.write("# Evaluation script for benchmark models\n")
        f.write("# Generated automatically by benchmark_models.py\n")
        f.write("#\n")
        f.write(f"# Dataset: {dataset}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"# Benchmark results: {output_file}\n")
        f.write("#\n")
        f.write("# This script evaluates all successfully trained models on test data.\n")
        f.write("# Run manually after all training completes:\n")
        f.write(f"#   bash {script_path}\n")
        f.write("#\n\n")
        
        f.write("# Activate virtual environment\n")
        f.write("source /home/vscode/.pykt-env/bin/activate\n\n")
        
        f.write("# Change to examples directory (required by wandb_predict.py)\n")
        f.write("cd /workspaces/pykt-toolkit/examples\n\n")
        
        f.write("# Navigate to examples directory\n")
        f.write("cd /workspaces/pykt-toolkit/examples\n\n")
        
        f.write("echo '========================================'\n")
        f.write("echo 'Starting Model Evaluations'\n")
        f.write("echo '========================================'\n\n")
        
        # Track evaluation count
        eval_count = 0
        
        # Generate evaluation commands for successful training runs
        for idx, row in df[df['success'] == True].iterrows():
            model = row['model']
            fold = row['fold']
            model_seed = row['seed']
            eval_count += 1
            
            # Try to find the save_dir
            save_dir = find_model_save_dir(model, dataset, fold, model_seed, benchmark_dir)
            
            # Get evaluation script
            eval_script = MODEL_EVAL_SCRIPTS.get(model, "wandb_eval.py")
            
            f.write(f"# Evaluation {eval_count}: {model} (fold {fold})\n")
            f.write(f"echo ''\n")
            f.write(f"echo '[{eval_count}] Evaluating {model} (fold {fold})...'\n")
            
            if not save_dir:
                f.write(f"echo 'WARNING: Could not find saved model directory for {model} fold {fold}'\n")
                f.write(f"echo 'Skipping evaluation...'\n\n")
                continue
            
            # For most models using wandb_predict.py (standard pykt evaluation)
            if eval_script == "wandb_predict.py":
                f.write(f"python {eval_script} \\\n")
                f.write(f"    --save_dir {save_dir} \\\n")
                f.write(f"    --use_wandb 0 \\\n")
                f.write(f"    --bz 256\n")
            else:
                # For model-specific evaluation scripts (e.g., eval_ikt3.py)
                f.write(f"python {eval_script} \\\n")
                f.write(f"    --save_dir {save_dir} \\\n")
                f.write(f"    --use_wandb 0")
                
                # Add extra args if they were used in training
                if extra_args:
                    for key, value in extra_args.items():
                        f.write(f" \\\n    --{key} {value}")
                f.write("\n")
            
            f.write(f"\nif [ $? -eq 0 ]; then\n")
            f.write(f"    echo '✓ {model} (fold {fold}) evaluation completed'\n")
            f.write(f"else\n")
            f.write(f"    echo '✗ {model} (fold {fold}) evaluation failed'\n")
            f.write(f"fi\n\n")
        
        f.write("echo ''\n")
        f.write("echo '========================================'\n")
        f.write("echo 'All Evaluations Complete'\n")
        f.write("echo '========================================'\n")
        f.write("echo ''\n")
        f.write(f"echo 'Total evaluations run: {eval_count}'\n")
        f.write("echo 'Check individual model directories for test results.'\n")
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple KT models with consistent parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark 3 models on assist2009
  python benchmark_models.py --models akt,dkt,sakt --dataset assist2009
  
  # Include iKT3 in comparison
  python benchmark_models.py --models akt,sakt,ikt3 --dataset assist2009
  
  # Run across all folds
  python benchmark_models.py --models akt,sakt --dataset assist2009 --folds 0,1,2,3,4
  
  # Custom learning rate
  python benchmark_models.py --models akt,dkt --dataset assist2009 --learning_rate 0.001

Available models:
  """ + ", ".join(sorted(MODEL_SCRIPTS.keys()))
        )
    
    # Each model's training script (e.g., wandb_akt_train.py) has its own defaults
    parser.add_argument("--models", type=str, default="akt",
                        help="Comma-separated list of models to benchmark (default: akt)")
    #parser.add_argument("--models", type=str, default="akt,dkt,dkt+,sakt,saint,dkvmn", help="Comma-separated list of models to benchmark (default: akt,dkt,dkt+,sakt,saint,dkvmn)")
    parser.add_argument("--dataset", type=str, default="assist2015",
                        help="Dataset name (default: assist2015)")
    parser.add_argument("--folds", type=str, default="0",
                        help="Comma-separated list of folds (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file (default: <timestamp>_benchmark_results_<dataset>.csv)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate for all models")
    parser.add_argument("--d_model", type=int, default=None,
                        help="Override d_model for transformer models")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout rate")
    parser.add_argument("--auto_eval", action="store_true",
                        help="Automatically run evaluation after training completes")
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="Run multiple models in parallel on different GPUs (default: enabled)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially instead of parallel (slower, uses all GPUs per model)")
    parser.add_argument("--max_parallel", type=int, default=6,
                        help="Maximum number of parallel jobs (default: 6, one per GPU)")
    
    args = parser.parse_args()
    
    # Parse models and folds
    models = [m.strip() for m in args.models.split(",")]
    folds = [int(f.strip()) for f in args.folds.split(",")]
    
    # Validate models
    invalid_models = [m for m in models if m not in MODEL_SCRIPTS]
    if invalid_models:
        print(f"Error: Unknown models: {', '.join(invalid_models)}")
        print(f"Available models: {', '.join(sorted(MODEL_SCRIPTS.keys()))}")
        sys.exit(1)
    
    # Create benchmark directory with pattern: YYYYMMDD_HHMMSS_benchmark_[dataset]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir_name = f"{timestamp}_benchmark_{args.dataset}"
    
    # Use relative path from examples directory
    experiments_dir = Path(__file__).resolve().parent / ".." / "experiments"
    experiments_dir = experiments_dir.resolve()  # Resolve to absolute path
    
    # Ensure experiments directory exists
    if not experiments_dir.exists():
        raise FileNotFoundError(f"experiments directory does not exist: {experiments_dir}. "
                                "Directory must exist before running benchmark.")
    
    benchmark_dir = experiments_dir / benchmark_dir_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Build extra arguments
    extra_args = {}
    if args.learning_rate is not None:
        extra_args['learning_rate'] = args.learning_rate
    if args.d_model is not None:
        extra_args['d_model'] = args.d_model
    if args.dropout is not None:
        extra_args['dropout'] = args.dropout
    
    # Run benchmark
    try:
        # Use parallel by default unless --sequential is specified
        use_parallel = args.parallel and not args.sequential
        df = run_benchmark(models, args.dataset, folds, args.seed, str(benchmark_dir), extra_args, 
                          parallel=use_parallel, max_parallel=args.max_parallel)
        
        # Run evaluations if requested
        if args.auto_eval:
            df = run_evaluations(df, args.dataset, args.seed, str(benchmark_dir))
        
        # Generate evaluation script (for manual execution if needed)
        eval_script_path = generate_eval_script(df, args.dataset, args.seed, str(benchmark_dir), extra_args)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION SCRIPT GENERATED")
        print(f"{'='*80}")
        print(f"Script: {eval_script_path}")
        print(f"Models to evaluate: {df['success'].sum()}/{len(df)}")
        
        if args.auto_eval:
            print(f"\n✓ Evaluations completed automatically")
            print(f"   Results (with test metrics) saved to: {benchmark_dir / 'results.csv'}")
            print(f"\nTo re-run evaluations manually:")
        else:
            print(f"\nTo evaluate all trained models on test data, run:")
        
        print(f"  bash {eval_script_path}")
        print(f"\nOr make it executable and run directly:")
        print(f"  chmod +x {eval_script_path}")
        print(f"  ./{eval_script_path}")
        print(f"\nTo run evaluation in background with logging:")
        eval_log = eval_script_path.replace('.sh', '_log_$(date +%Y%m%d_%H%M%S).log')
        print(f"  nohup bash {eval_script_path} > {eval_log} 2>&1 &")
        print(f"\nTo monitor evaluation progress:")
        print(f"  tail -f {eval_log.replace('_$(date +%Y%m%d_%H%M%S)', '_*')}")
        print(f"{'='*80}\n")
        
        # Exit with error if any experiment failed
        if not df['success'].all():
            sys.exit(1)
        if args.auto_eval and 'eval_success' in df.columns and not df['eval_success'].all():
            print("WARNING: Some evaluations failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
