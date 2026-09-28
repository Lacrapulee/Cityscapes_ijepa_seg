FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Seul requirements.txt entre dans l'image : le code, les données et les checkpoints
# sont montés en volume par docker-compose (cf. .dockerignore).
# --break-system-packages : sans effet sur le Python conda de l'image, mais évite un
# échec si une future image de base marque son Python comme "externally managed".
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt
