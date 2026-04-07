#!/usr/bin/env bash
set -euo pipefail

export CONDA_DIR="${CONDA_DIR:-/opt/conda}"
export PATH="$CONDA_DIR/bin:$PATH"
ENV_NAME="genatator_pipeline"

install_miniconda() {
  echo "[startup] Installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniconda.sh
}

install_env() {
  echo "[startup] Creating conda env ${ENV_NAME}..."
  conda config --set always_yes yes --set changeps1 no
  conda config --set channel_priority strict
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

  conda create -n "$ENV_NAME" python=3.11

  conda run --no-capture-output -n "$ENV_NAME" pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
  conda run --no-capture-output -n "$ENV_NAME" pip install causal-conv1d==1.4.0 --no-build-isolation
  conda run --no-capture-output -n "$ENV_NAME" pip install mamba-ssm==2.2.2 --no-build-isolation
  conda run --no-capture-output -n "$ENV_NAME" pip install flash-attn==2.6.3 --no-build-isolation
  conda run --no-capture-output -n "$ENV_NAME" pip install -r /app/docker/requirements.txt
  conda run --no-capture-output -n "$ENV_NAME" pip install -e /app

  conda clean -afy
}

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  install_miniconda
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[startup] Conda env ${ENV_NAME} already exists."
else
  install_env
fi

exec "$CONDA_DIR/bin/conda" run --no-capture-output -n "$ENV_NAME" python /app/docker/app.py
