# Use Ubuntu base image and install CUDA manually for better compatibility
FROM ubuntu:20.04

# Switch to root for system-level installations
USER root

# Avoid questions when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Install NVIDIA CUDA toolkit and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    gnupg \
    build-essential \
    curl \
    wget \
    meld \
    git \
    lsof \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA package repositories
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-8 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.0-1_all.deb

# Download the corporate CA certificate
RUN curl -O https://nexus.roqs.basf.net/repository/cdn/ca/BASF_internal_and_public_ca_bundle.crt --insecure

# Copy the certificate to /usr/local/share/ca-certificates/
RUN cp BASF_internal_and_public_ca_bundle.crt /usr/local/share/ca-certificates/

# Update the CA certificates
RUN update-ca-certificates

# Set environment variables for CA certificates
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
ENV CURL_CA_BUNDLE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt

# Set NVIDIA/CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Accept proxy settings as build arguments
ARG http_proxy
ARG https_proxy

# Set environment variables for proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

# Install Python 3.7 and tools including tkinter support
RUN echo "Installing Python 3.7..." && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.7 python3.7-dev python3.7-venv python3.7-distutils python3-pip python3-tk python3.7-tk && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    python3.7 -m pip install --upgrade pip setuptools wheel virtualenv && \
    rm -rf /var/lib/apt/lists/*

# Create vscode user FIRST with consistent UID/GID before any environment setup
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Install Node.js global packages as root
RUN npm install -g @anthropic-ai/claude-code --force --no-os-check

# Now switch to vscode user and create everything in their home directory to avoid permission issues
USER vscode

# Create virtual environment in user's home directory (completely owned by vscode)
RUN python3.7 -m venv /home/vscode/.pykt-env && \
    /home/vscode/.pykt-env/bin/pip install --upgrade pip setuptools wheel build

# Install PyTorch and GPU packages
RUN /home/vscode/.pykt-env/bin/pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 || \
    /home/vscode/.pykt-env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core pyKT dependencies
RUN /home/vscode/.pykt-env/bin/pip install \
    'numpy>=1.17.2' \
    'pandas>=1.1.5' \
    'scikit-learn' \
    'wandb>=0.12.9' \
    'entmax' \
    'tqdm' \
    'einops' \
    'matplotlib' \
    'jupyter' \
    'seaborn'

# Install additional ML packages
RUN /home/vscode/.pykt-env/bin/pip install \
    'transformers<4.21.0' \
    'datasets<2.5.0' \
    'accelerate<0.13.0' \
    'safetensors<0.3.0' \
    'tokenizers<0.13.0' && \
    /home/vscode/.pykt-env/bin/pip install torch-geometric && \
    (/home/vscode/.pykt-env/bin/pip install dgl-cu113 -f https://data.dgl.ai/wheels/cu113/repo.html || \
     /home/vscode/.pykt-env/bin/pip install dgl)

# Install tensorboard
RUN /home/vscode/.pykt-env/bin/pip install tensorboard

# Set up shell environment to use the virtual environment
RUN echo 'export PATH="/home/vscode/.pykt-env/bin:$PATH"' >> /home/vscode/.bashrc && \
    echo 'export VIRTUAL_ENV="/home/vscode/.pykt-env"' >> /home/vscode/.bashrc && \
    echo 'source /home/vscode/.pykt-env/bin/activate' >> /home/vscode/.bashrc && \
    echo "export PATH=/home/vscode/.local/bin:\$PATH" >> /home/vscode/.bashrc

# Set working directory
WORKDIR /workspaces/pykt-toolkit

# Update PATH for vscode user to use the home directory virtual environment
ENV PATH="/home/vscode/.pykt-env/bin:/home/vscode/.local/bin:${PATH}"
ENV VIRTUAL_ENV=/home/vscode/.pykt-env

# Start shell with virtual environment activated
CMD ["/bin/bash", "-c", "source /home/vscode/.pykt-env/bin/activate && exec /bin/bash"]