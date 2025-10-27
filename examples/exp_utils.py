#!/usr/bin/env python3
"""Experiment utility helpers for reproducible GainAKT2Exp launches.

We isolate generic reusable functions here to avoid modifying existing model scripts.
Functions:
- make_experiment_dir(model_name, short_title, base_dir)
- capture_environment(out_path)
- atomic_write_json(obj, path)
- append_epoch_csv(row_dict, csv_path, header)
- timestamped_logger(name, log_path)
"""
from __future__ import annotations
import os
import json
import csv
import hashlib
import platform
import sys
import subprocess
import logging
import datetime
from typing import Dict, Any, List

TS_FMT = "%Y%m%d_%H%M%S"

def now_ts():
    return datetime.datetime.utcnow().strftime(TS_FMT)

def make_experiment_dir(model_name: str, short_title: str, base_dir: str = "examples/experiments") -> str:
    ts = now_ts()
    exp_id = f"{ts}_{model_name}_{short_title}".lower()
    path = os.path.join(base_dir, exp_id)
    os.makedirs(path, exist_ok=False)
    os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
    return path

def capture_environment(out_path: str):
    info = {
        "python": sys.version.replace('\n',' '),
        "platform": platform.platform(),
        "cuda_available": False,
        "torch_version": None,
        "cuda_version": None,
        "git_commit": None,
        "git_branch": None,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    # Git metadata
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        info["git_commit"], info["git_branch"] = commit, branch
    except Exception:
        pass
    with open(out_path, 'w') as f:
        for k,v in info.items():
            f.write(f"{k}: {v}\n")

def atomic_write_json(obj: Dict[str,Any], path: str):
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def compute_config_hash(obj: Dict[str,Any]) -> str:
    s = json.dumps(obj, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

def append_epoch_csv(row: Dict[str,Any], csv_path: str, header: List[str]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in header})

def timestamped_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger

__all__ = [
    'make_experiment_dir','capture_environment','atomic_write_json','append_epoch_csv','timestamped_logger','compute_config_hash','now_ts'
]
