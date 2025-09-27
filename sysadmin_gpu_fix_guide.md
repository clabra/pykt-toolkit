# System Administrator GPU Fix Guide
## Host-Level Docker/NVIDIA Configuration Issues

If the container rebuild doesn't resolve GPU access, the system administrator needs to address these host-level configuration issues:

## 🔍 **1. Verify NVIDIA Container Runtime Installation**

The host system needs the NVIDIA Container Runtime to pass GPUs to containers.

### Check if installed:
```bash
# On the HOST system (not container):
which nvidia-container-runtime
docker info | grep -i nvidia
```

### Install if missing:
```bash
# Ubuntu/Debian:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
sudo systemctl restart docker
```

## 🔍 **2. Configure Docker Daemon for GPU Support**

### Check current Docker daemon configuration:
```bash
sudo cat /etc/docker/daemon.json
```

### Required configuration (create/update `/etc/docker/daemon.json`):
```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

### Restart Docker after changes:
```bash
sudo systemctl restart docker
```

## 🔍 **3. Verify NVIDIA Driver on Host**

### Check host NVIDIA driver:
```bash
nvidia-smi
cat /proc/driver/nvidia/version
```

### Required: Driver version >= 450.80.02 for CUDA 11.8

### If driver missing/outdated:
```bash
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

## 🔍 **4. Test Host GPU Access**

### Test basic Docker GPU access:
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Expected output: Should show all GPUs
### If this fails, the issue is definitely host-level

## 🔍 **5. Check Container Runtime Permissions**

### Verify Docker can access NVIDIA devices:
```bash
ls -la /dev/nvidia*
# Should show devices owned by root with proper permissions

# Check if docker group has access:
getent group docker
```

### Fix permissions if needed:
```bash
sudo usermod -aG docker $USER
# Logout and login again
```

## 🔍 **6. Corporate/Security Policy Issues**

### Common enterprise restrictions:
- **Container security policies** blocking GPU access
- **SELinux/AppArmor** restrictions on device access
- **Corporate Docker registry** restrictions
- **Network policies** blocking NVIDIA package downloads

### Check SELinux (RHEL/CentOS):
```bash
getenforce
# If "Enforcing", may need SELinux policies for containers
```

### Check AppArmor (Ubuntu):
```bash
aa-status | grep docker
# May need AppArmor profile updates
```

## 🔍 **7. VS Code Dev Container Specific Issues**

### Check VS Code Docker integration:
```bash
# Verify VS Code can access Docker socket:
ls -la /var/run/docker.sock
```

### Required VS Code settings (in workspace .vscode/settings.json):
```json
{
    "dev.containers.dockerComposePath": "docker-compose",
    "dev.containers.dockerPath": "docker"
}
```

## 🔍 **8. Kubernetes/Container Platform Issues**

If running on Kubernetes/OpenShift:

### Required: NVIDIA Device Plugin
```bash
kubectl get pods -n kube-system | grep nvidia
# Should show nvidia-device-plugin pods running
```

### Required: GPU resource allocation
```yaml
# In pod spec:
resources:
  limits:
    nvidia.com/gpu: 8
```

## 🔍 **9. Complete Diagnostic Script**

Run this on the HOST system to diagnose all issues:

```bash
#!/bin/bash
echo "=== HOST GPU DIAGNOSTIC ==="

echo "1. NVIDIA Driver:"
nvidia-smi --version || echo "❌ NVIDIA driver not found"

echo "2. Docker version:"
docker --version

echo "3. NVIDIA Container Runtime:"
which nvidia-container-runtime || echo "❌ nvidia-container-runtime not found"

echo "4. Docker daemon config:"
cat /etc/docker/daemon.json 2>/dev/null || echo "❌ No Docker daemon config"

echo "5. Test GPU container:"
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi || echo "❌ GPU container test failed"

echo "6. Docker info:"
docker info | grep -i nvidia || echo "❌ No NVIDIA info in Docker"
```

## 📞 **What to Tell Your System Administrator**

> "Our container configuration is correct - we've verified all GPU devices are visible and CUDA libraries are properly configured. However, PyTorch still cannot access the GPUs. This indicates a host-level Docker/NVIDIA Container Runtime configuration issue. Please review the GPU diagnostic guide and ensure:
> 
> 1. NVIDIA Container Runtime is installed and configured
> 2. Docker daemon has GPU support enabled  
> 3. NVIDIA drivers are compatible (>=450.80.02)
> 4. No corporate security policies are blocking GPU access
> 
> The specific test that should work is:
> `docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi`
> 
> This should show all 8 Tesla V100 GPUs. If it doesn't, the issue is definitely host-level."

## 🔧 **Emergency Workaround**

If GPU access cannot be fixed immediately, PyKT can run on CPU:

```python
# Force CPU-only mode in your training scripts:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or in PyTorch:
import torch
torch.cuda.is_available = lambda: False
```

Note: This will be significantly slower but allows development to continue.
