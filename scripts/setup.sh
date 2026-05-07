#!/usr/bin/env bash
set -euo pipefail

# Minimal environment setup script for VM / Colab / k8s container
# - creates a venv at /workspace/venv
# - installs dependencies from requirements.txt (attempts to install torch wheel matching CUDA if requested)
# - prepares datasets via datamodules
# - optionally runs driver script

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-/workspace/venv}
CUDA_VERSION=${CUDA_VERSION:-11.8}  # set to 11.8 or 12.1 or 'auto'
RUN_DRIVER=${RUN_DRIVER:-1}

echo "Using python: $PYTHON"

if [ ! -x "$(command -v $PYTHON)" ]; then
  echo "ERROR: $PYTHON not found in PATH" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv at $VENV_DIR"
  $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

echo "Installing dependencies from requirements.txt"

# Install torch wheel first if CUDA version is specified and index is available
if [ "$CUDA_VERSION" = "auto" ] || [ -z "$CUDA_VERSION" ]; then
  echo "Installing requirements (no specific CUDA wheel)"
  pip install -r /workspace/requirements.txt
else
  case "$CUDA_VERSION" in
    11.8)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
      ;;
    12.1)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
      ;;
    *)
      TORCH_INDEX_URL=""
      ;;
  esac

  if [ -n "$TORCH_INDEX_URL" ]; then
    echo "Installing torch==2.2.0 and torchvision==0.17.0 from $TORCH_INDEX_URL"
    pip install --upgrade pip
    pip install --index-url "$TORCH_INDEX_URL" torch==2.2.0 torchvision==0.17.0
    # Install remaining requirements but skip torch/torchvision if present
    pip install -r /workspace/requirements.txt --ignore-installed --no-deps
  else
    pip install -r /workspace/requirements.txt
  fi
fi

echo "Preparing datasets (CUB example)"
python - <<'PY'
try:
    from data.cub2011parts_datamodule import CUB2011Parts
    dm = CUB2011Parts(data_dir='/workspace/data', batch_size=32, num_workers=4)
    dm.prepare_data()
    dm.setup()
    print('CUB dataset prepared under /workspace/data')
except Exception as e:
    print('Warning: CUB prepare_data() failed:', e)
    pass
PY

if [ "$RUN_DRIVER" -eq 1 ]; then
  echo "Running driver script"
  bash /workspace/scripts/driver.sh || echo "Driver script failed"
else
  echo "Skipping driver run (RUN_DRIVER=$RUN_DRIVER)"
fi

echo "Setup complete"
