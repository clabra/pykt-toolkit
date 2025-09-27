#!/bin/bash
# Quick GPU Test Command
/home/vscode/.pykt-env/bin/python -c "
import torch
print(\"=== Quick GPU Test ===\")
print(f\"PyTorch version: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"🎉 SUCCESS! GPUs: {torch.cuda.device_count()}\")
    print(f\"GPU 0: {torch.cuda.get_device_name(0)}\")
    # Quick computation test
    x = torch.randn(100, 100, device=\"cuda:0\")
    print(f\"✅ GPU computation works!\")
else:
    print(\"❌ GPU access not working yet\")
"
