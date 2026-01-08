import subprocess
import os
import concurrent.futures
import time
import sys

def run_experiment(dataset, fold, gpu_id):
    # Use the same python executable
    python_exe = sys.executable
    cmd = [
        python_exe, "examples/run_repro_experiment.py",
        "--dataset", dataset,
        "--fold", str(fold),
        "--short_title", "unconstrained",
        "--lambda_ref", "0.0",
        "--lambda_initmastery", "0.0",
        "--lambda_rate", "0.0",
        "--theory_guided", "0",
        "--calibrate", "0"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Starting {dataset} fold {fold} on GPU {gpu_id}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {dataset} fold {fold}: {result.stderr}")
    return result.stdout

def main():
    datasets = ["assist2009_S", "assist2015_S", "bridge2algebra2006_S", "nips_task34_S"]
    folds = range(5)
    # Use 5 GPUs as per operational standards
    gpus = ["0", "1", "2", "3", "4"]
    
    tasks = []
    for dataset in datasets:
        for fold in folds:
            tasks.append((dataset, fold))
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for i, (dataset, fold) in enumerate(tasks):
            gpu_id = gpus[i % len(gpus)]
            futures.append(executor.submit(run_experiment, dataset, fold, gpu_id))
            time.sleep(20) # Stagger to avoid audit race conditions
            
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()
