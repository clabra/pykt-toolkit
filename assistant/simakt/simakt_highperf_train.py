#!/usr/bin/env python
# coding=utf-8
"""
SimAKT High-Performance Training Script - Optimized for GPU + Multi-Core Systems

PERFORMANCE OPTIMIZATIONS:
- GPU Acceleration: Automatic CUDA detection and utilization
- Mixed Precision Training: 16-bit training for 2x speedup + memory savings
- Optimized Data Loading: Efficient multi-worker data pipeline
- Memory Optimization: Gradient accumulation, pin memory, prefetch
- Large Batch Processing: Automatic batch size scaling based on GPU memory

HARDWARE UTILIZATION:
- GPU: Full CUDA acceleration with mixed precision
- CPU: Multi-worker data loading + computation threading
- Memory: Optimized batch sizes and memory management
- I/O: Prefetching and non-blocking data transfers

USAGE:
    # Maximum performance (will auto-detect and use all GPUs)
    python assistant/simakt_highperf_train.py --dataset_name=assist2015 --use_wandb=0
    
    # Conservative GPU usage
    python assistant/simakt_highperf_train.py --dataset_name=assist2015 --gpu_mode=conservative --use_wandb=0
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import random
import multiprocessing as mp
import time
from datetime import datetime
import gc

# ...existing imports...