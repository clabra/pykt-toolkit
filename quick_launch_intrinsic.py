#!/usr/bin/env python
import subprocess
import os

# Use the optimal config
config_path = 'configs/gainakt2exp_optimal.json'

# Launch intrinsic mode
cmd = [
    'python', 'examples/train_gainakt2exp.py',
    '--config', config_path,
    '--intrinsic_gain_attention'  # Add intrinsic mode flag
]

print(f"Launching intrinsic experiment with config: {config_path}")
print(f"Command: {' '.join(cmd)}")

# Run in background
subprocess.Popen(cmd, stdout=open('nohup_intrinsic.out', 'w'), stderr=subprocess.STDOUT)
print("Check nohup_intrinsic.out for progress")
