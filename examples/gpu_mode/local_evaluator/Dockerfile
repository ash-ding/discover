# Dockerfile for GPU Kernel Evaluation Worker
# Provides isolated environment for running potentially crashing kernel code
# Compatible with both Podman and Docker

FROM docker.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch 2.5.1 with CUDA 12.1 (latest available)
RUN pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install Triton (latest compatible version)
RUN pip install triton

# Install evaluation dependencies
RUN pip install \
    numpy \
    pyyaml

# Create workspace
WORKDIR /workspace

# Copy evaluation worker scripts
COPY eval_worker.py /workspace/
COPY pool_worker.py /workspace/

# The lib directory will be mounted at runtime
# No need to copy it into the image

# Entry point
ENTRYPOINT ["python", "/workspace/eval_worker.py"]
