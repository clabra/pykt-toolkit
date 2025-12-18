# Use an official NVIDIA CUDA image to ensure compatibility and simplify setup.
# The version 11.8 matches the toolkit version previously installed manually.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Switch to root for system-level installations
USER root

# Avoid questions when installing packages
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Accept proxy settings as build arguments
ARG http_proxy
ARG https_proxy

# Set environment variables for proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

# Update package lists and install base dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    software-properties-common \
    gnupg \
    build-essential \
    wget \
    meld \
    git \
    lsof \
    sudo \
    poppler-utils

# Download and install the corporate CA certificate
RUN curl -O https://nexus.roqs.basf.net/repository/cdn/ca/BASF_internal_and_public_ca_bundle.crt --insecure && \
    cp BASF_internal_and_public_ca_bundle.crt /usr/local/share/ca-certificates/ && \
    rm BASF_internal_and_public_ca_bundle.crt && \
    update-ca-certificates

# Set environment variables for CA certificates
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
ENV CURL_CA_BUNDLE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt

# Install Node.js in a single layer
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Add Python PPA and install Python 3.8
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-venv python3.8-distutils python3-pip python3-tk python3.8-tk && \
    python3.8 -m pip install --upgrade pip setuptools wheel virtualenv && \
    rm -rf /var/lib/apt/lists/*

# Create vscode user FIRST with consistent UID/GID before any environment setup
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Set Python 3.8 as the default python (fixed typo)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Install Node.js global packages as root
RUN npm install -g @anthropic-ai/claude-code --force --no-os-check

# Now switch to vscode user and create everything in their home directory to avoid permission issues
USER vscode

# Create virtual environment in user's home directory (completely owned by vscode)
RUN python3.8 -m venv /home/vscode/.pykt-env && \
    #/home/vscode/.pykt-env/bin/pip install --upgrade pip setuptools wheel build
    # Add uv to be able to install MCP Servers that require it
    /home/vscode/.pykt-env/bin/pip install --upgrade pip setuptools wheel build uv

# Install PyTorch and GPU packages compatible with CUDA 11.8 from the base image.
# Fixed: Use explicit CUDA 11.8 wheel installation for proper GPU support
RUN /home/vscode/.pykt-env/bin/pip install \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install NVIDIA CUDA runtime libraries explicitly for PyTorch compatibility
# This comprehensive list ensures all necessary components are available for PyTorch.
RUN /home/vscode/.pykt-env/bin/pip install \
    nvidia-cuda-runtime-cu11 \
    nvidia-cuda-cupti-cu11 \
    nvidia-cudnn-cu11 \
    nvidia-cublas-cu11 \
    nvidia-cusparse-cu11 \
    nvidia-cusolver-cu11 \
    nvidia-cufft-cu11 \
    nvidia-curand-cu11

# Install core pyKT dependencies
RUN /home/vscode/.pykt-env/bin/pip install \
    "numpy>=1.17.2" \
    "pandas>=1.1.5" \
    "scikit-learn" \
    "wandb>=0.12.9" \
    "entmax" \
    "tqdm" \
    "einops" \
    "matplotlib" \
    "jupyter" \
    "seaborn" \
    "pyBKT"

# Install additional ML packages
RUN /home/vscode/.pykt-env/bin/pip install \
    "transformers<4.21.0" \
    "datasets<2.5.0" \
    "accelerate<0.13.0" \
    "safetensors<0.3.0" \
    "tokenizers<0.13.0" && \
    /home/vscode/.pykt-env/bin/pip install torch-geometric && \
    # Install DGL compatible with CUDA 11.8
    /home/vscode/.pykt-env/bin/pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Install tensorboard
RUN /home/vscode/.pykt-env/bin/pip install tensorboard

# Set up shell environment to automatically activate the virtual environment for new terminals.
# We intentionally do NOT set LD_LIBRARY_PATH here to create a "clean" environment
# that allows scripts to bootstrap their own GPU settings.
RUN echo "export PATH=\"/home/vscode/.pykt-env/bin:$PATH\"" >> /home/vscode/.bashrc && \
    echo "export VIRTUAL_ENV=\"/home/vscode/.pykt-env\"" >> /home/vscode/.bashrc && \
    echo "source /home/vscode/.pykt-env/bin/activate" >> /home/vscode/.bashrc && \
    echo "export PATH=/home/vscode/.local/bin:\$PATH" >> /home/vscode/.bashrc

# Set working directory
WORKDIR /workspaces/pykt-toolkit

# Update PATH for vscode user to use the home directory virtual environment
ENV PATH="/home/vscode/.pykt-env/bin:/home/vscode/.local/bin:${PATH}"
ENV VIRTUAL_ENV=/home/vscode/.pykt-env

# Start shell with virtual environment activated
CMD ["/bin/bash", "-c", "source /home/vscode/.pykt-env/bin/activate && exec /bin/bash"]