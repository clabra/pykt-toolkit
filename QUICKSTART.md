# Quick Start Guide: Running pyKT in a Container

This guide explains how to set up and run the pyKT toolkit in a Docker container with GPU support.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)
- NVIDIA GPU with drivers installed (optional, but recommended for training)
- NVIDIA Container Toolkit installed (for GPU support)

### Installing NVIDIA Container Toolkit (Ubuntu/Debian)

docker-compose.yml

```bash
vservices:
  pinn:
    build:
      context: .
      dockerfile: .devcontainer/Dockerfile
      args:
        # These allow the container user to match your host user ID, preventing permission issues
        # You can override these in a .env file or shell environment
        USERNAME: vscode
        USER_UID: "${USER_UID:-1008}"
        USER_GID: "${USER_GID:-1008}"
    image: pinn:dev
    container_name: pinn-dev

    # Mount the current directory to the container workspace
    volumes:
      - .:/workspaces/pykt-toolkit

    # Keep the container running so you can attach to it
    command: /bin/bash -c "source /home/vscode/.pykt-env/bin/activate && sleep infinity"

    # Configuration for NVIDIA GPUs
    deploy:
      resources:
        limits:
          # Soft limit - can use up to 300GB RAM when available (shared machine friendly)
          memory: 300G
          # Soft limit - can use up to 20 CPU cores when available
          cpus: '20'
        reservations:
          # No hard reservations - use resources opportunistically
          devices:
            - driver: nvidia
              # Use specific GPUs via CUDA_VISIBLE_DEVICES env var instead of reserving all
              count: all
              capabilities: [ gpu ]

    # Runtime settings from devcontainer.json
    shm_size: '16gb'
    privileged: true
    init: true
    working_dir: /workspaces/pykt-toolkit

    # Increase system limits for deep learning workloads
    ulimits:
      # Maximum number of open file descriptors (important for DataLoader with many workers)
      nofile:
        soft: 65536
        hard: 65536
      # Maximum number of processes (important for multi-GPU training)
      nproc:
        soft: 65536
        hard: 65536
      # Memory lock limit (allows pinning memory for faster GPU transfers)
      memlock:
        soft: -1
        hard: -1
      # Stack size
      stack:
        soft: 67108864
        hard: 67108864

    # Use tmpfs for faster temporary file I/O (useful for caching)
    tmpfs:
      - /tmp:size=32G,mode=1777

    # Environment variables
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      # PyTorch optimizations (conservative for shared machine)
      - OMP_NUM_THREADS=20
      - MKL_NUM_THREADS=20
      - OPENBLAS_NUM_THREADS=20
      - NUMEXPR_NUM_THREADS=20
      # Enable CUDA optimizations
      - CUDA_LAUNCH_BLOCKING=0
      - CUDA_CACHE_MAXSIZE=2147483648
      # Memory allocator optimizations for PyTorch
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Setup Steps

### 1. Build and Start the Container

From the project root directory, run:

```bash
UID=$(id -u) GID=$(id -g) docker compose up -d --build
```

This command:

- Builds the Docker image with all dependencies
- Starts the container in detached mode
- Passes your user ID to avoid permission issues with mounted files

**Note:** The first build may take 5-10 minutes as it installs CUDA, PyTorch, and all dependencies.

### 2. Enter the Container

Once the container is running, open a shell inside it:

```bash
docker compose exec pinn bash
```

You should see a prompt like:

```
(.pykt-env) vscode@<container-id>:/workspaces/pykt-toolkit$
```

### 3. Install the Project in Editable Mode

Inside the container, run:

```bash
pip install -e .
```

This installs the `pykt` package in editable mode, allowing you to modify code and see changes immediately.

### 4. Verify the Installation

Check that PyTorch can access your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output (if GPU is available):

```
CUDA available: True
GPU count: 1
```

## Working with the Container

### Running Training Scripts

Navigate to the examples directory and run a training script:

```bash
cd examples
python train_model.py --model_name=akt --dataset_name=assist2009
```

### Editing Code

All changes you make to files in the project directory (on your host machine or inside the container) are immediately reflected in both places due to the volume mount. The editable install ensures Python imports use your latest code.

### Managing the Container

**Stop the container:**

```bash
docker compose down
```

**Restart the container:**

```bash
UID=$(id -u) GID=$(id -g) docker compose up -d
```

**View container logs:**

```bash
docker compose logs -f pykt-toolkit
```

**Check container status:**

```bash
docker compose ps
```

## Troubleshooting

### Permission Errors

If you encounter permission errors when creating files inside the container, run:

```bash
docker compose exec pykt-toolkit sudo chown -R vscode:vscode /workspaces/pykt-toolkit
```

### GPU Not Detected

1. Verify NVIDIA drivers are installed on the host:

   ```bash
   nvidia-smi
   ```

2. Check that the NVIDIA Container Toolkit is installed:

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
   ```

3. If you don't have a GPU, you can still run the container by commenting out the `deploy` section in `docker-compose.yml`.

### Container Won't Start

Check the logs for errors:

```bash
docker compose logs pykt-toolkit
```

### BASF Network Issues

The Dockerfile includes BASF-specific certificate configuration. If you're not on the BASF network, you may need to modify `.devcontainer/devcontainer-basf/Dockerfile` to remove or update the certificate download section (lines 34-43).

## Alternative: VS Code DevContainers

If you use VS Code, you can also open this project using the DevContainer configuration:

1. Install the "Dev Containers" extension
2. Press `F1` and select "Dev Containers: Reopen in Container"
3. VS Code will automatically build and connect to the container

## Latex Installation and Setup

To work with Latex it's recommended to include a LaTeX environment based on the LaTeX Workshop extension and TeX Live running inside the project's Docker container. 

### Building Latex Documents

There are three ways to compile the LaTeX documents:

1. **Integrated Dev Container (Recommended)**:
   - Ensure the following packages are installed in the `Dockerfile`:
     ```bash
     RUN apt-get update && apt-get install -y --no-install-recommends \
       texlive-latex-extra \
       texlive-fonts-recommended \
       texlive-bibtex-extra \
       texlive-science \
       texlive-publishers \
       latexmk \
       ghostscript
     ```
   - If working in VS Code, install the **LaTeX Workshop** extension. It provides "Build on Save" support, and you can also use the "Build LaTeX Project" button or `Ctrl+Alt+B` to build the document directly within the containerized environment.

2. **Convenience Script (Host-to-Container)**:
   - You can trigger the containerized `latexmk` from your host terminal using a script (e.g., `paper/compile.sh`):

     ```bash
     #!/bin/bash
     docker exec -w /workspaces/pykt-toolkit/paper/latex [CONTAINER_NAME] latexmk -pdf paper.tex
     ```
   - This allows you to generate the PDF without opening the container, provided the container is running.

3. **Host-Bridge Shims (IDE Integration from Host)**:
   - We have installed shims in `~/.local/bin/` on the host machine (`latexmk`, `pdflatex`, `bibtex`, etc.).
   - These shims automatically redirect host-based IDE commands (like the "Build" button) to the container.
   - **Note**: Ensure `~/.local/bin` is in your host `$PATH` and reload the IDE window if you encounter `ENOENT` errors when trying to build your latex document.

### Technical Details

- **Base Image**: The environment is built into the `.devcontainer/Dockerfile`.
- **Path Translation**: Host shims automatically translate absolute paths (e.g., `/home/username/...` to `/workspaces/...`) to maintain compatibility between host and container filesystems.
## Additional Preprocessing for iDKT Model

The **iDKT (Interpretable Deep Knowledge Tracing)** model requires additional preprocessing steps to augment datasets with BKT (Bayesian Knowledge Tracing) parameters. This enables the model to ground its deep learning representations in classical psychometric theory.

### Prerequisites

Ensure `pyBKT` is installed:

```bash
pip install pyBKT
```

### Step 1: Train BKT Model

For each dataset you want to use with iDKT, first train a BKT model to learn skill-level parameters (prior knowledge, learning rate, slip, guess):

```bash
python examples/train_bkt.py --dataset <dataset_name>
```

**Example:**
```bash
python examples/train_bkt.py --dataset algebra2005
python examples/train_bkt.py --dataset bridge2algebra2006
python examples/train_bkt.py --dataset nips_task34
```

This creates:
- `data/<dataset>/bkt_mastery_states.pkl` - Trained BKT model

### Step 2: Augment Sequences with BKT Data

After training the BKT model, augment the sequence files with BKT-derived mastery trajectories and correctness predictions:

```bash
python examples/augment_with_bkt.py --dataset <dataset_name>
```

**Example:**
```bash
python examples/augment_with_bkt.py --dataset algebra2005
python examples/augment_with_bkt.py --dataset bridge2algebra2006
python examples/augment_with_bkt.py --dataset nips_task34
```

This creates:
- `data/<dataset>/train_valid_sequences_bkt.csv` - Augmented training sequences
- `data/<dataset>/test_sequences_bkt.csv` - Augmented test sequences
- `data/<dataset>/bkt_skill_params.pkl` - Skill-level BKT parameters

### What Gets Added

The augmentation process adds two columns to each sequence:
- **`bkt_mastery`**: P(Learned) trajectory - the probability that the student has mastered each skill at each timestep
- **`bkt_p_correct`**: Predicted probability of correctness based on BKT parameters

These values serve as grounding targets for iDKT's interpretability-by-design approach, allowing the model to align its internal representations with established psychometric theory.

### Verification

After preprocessing, verify the files exist:

```bash
ls -lh data/<dataset>/*bkt*
```

You should see:
```
bkt_mastery_states.pkl
bkt_skill_params.pkl
train_valid_sequences_bkt.csv
test_sequences_bkt.csv
```

### Training iDKT

Once preprocessing is complete, you can train iDKT models using the standard training workflow:

```bash
python examples/run_repro_experiment.py --dataset <dataset_name>
```

The training script will automatically use the `*_bkt.csv` files when available.

## Next Steps

- Review the documentation in `docs/` for detailed model information
- Check `examples/` for training and evaluation scripts
- Read `contribute.pdf` for guidelines on adding new models
- See `examples/reproducibility.md` for experiment best practices
