FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
    wandb pyyaml opencv-python-headless timm datasets pillow transformers