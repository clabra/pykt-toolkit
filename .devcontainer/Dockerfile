# Use the base image from Ubuntu 22.04
FROM ubuntu:22.04

# Avoid questions when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates \
    build-essential \
    curl \
    wget \
    meld \
    git \
    lsof \
    sudo 

# Download the corporate CA certificate
#RUN curl -O https://nexus.roqs.basf.net/repository/cdn/ca/BASF_internal_and_public_ca_bundle.crt --insecure

# Copy the certificate to /usr/local/share/ca-certificates/
#RUN cp BASF_internal_and_public_ca_bundle.crt /usr/local/share/ca-certificates/

# Update the CA certificates
#RUN update-ca-certificates

# Verify certificate installation
#RUN curl -O https://nexus.roqs.basf.net/repository/cdn/ca/BASF_internal_and_public_ca_bundle.crt

# Set environment variables for CA certificates
#ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
#ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt
#ENV CURL_CA_BUNDLE=/usr/local/share/ca-certificates/BASF_internal_and_public_ca_bundle.crt

# Install Google Cloud CLI (required for Gemini)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-cli

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/miniconda/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Accept conda terms of service for specific channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda

# Create a conda environment with Python 3.7.5
RUN conda create --name=pykt python=3.7.5 -y

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

RUN npm install -g @anthropic-ai/claude-code --force --no-os-check

# Add 'vscode' user and configure permissions
RUN useradd -m vscode && \
    echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    usermod -aG sudo vscode

# Switch to vscode user
USER vscode

# Set working directory (this will be mounted by VS Code)
WORKDIR /workspaces/pykt-toolkit

# Initialize conda for vscode user
RUN /opt/miniconda/bin/conda init bash && \
    echo "source activate pykt" >> /home/vscode/.bashrc && \
    echo "export PATH=/home/vscode/.local/bin:\$PATH" >> /home/vscode/.bashrc

# Update PATH for vscode user
ENV PATH="/opt/miniconda/bin:/home/vscode/.local/bin:${PATH}"

# Set the conda environment to activate by default
ENV CONDA_DEFAULT_ENV=pykt

# Start a shell with conda environment activated
CMD ["/bin/bash", "-c", "source activate pykt && exec /bin/bash"]
