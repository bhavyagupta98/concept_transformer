#!/usr/bin/env bash

# Force bash execution (if called with sh)
if [ -z "$BASH_VERSION" ]; then
  exec bash "$0" "$@"
fi

# Minimal environment setup script for VM / Colab / k8s container
# - creates a venv at /workspace/concept_transformer/venv
# - installs dependencies from requirements.txt (attempts to install torch wheel matching CUDA if requested)
# - prepares datasets via datamodules
# - optionally runs driver script

set -eu

# Determine base directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-$BASE_DIR/venv}
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

. "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

echo "Installing dependencies from requirements.txt"

# Install torch wheel first if CUDA version is specified and index is available
if [ "$CUDA_VERSION" = "auto" ] || [ -z "$CUDA_VERSION" ]; then
  echo "Installing requirements (no specific CUDA wheel)"
  pip install -r "$BASE_DIR/requirements.txt"
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
    pip install -r "$BASE_DIR/requirements.txt" --ignore-installed --no-deps
  else
    pip install -r "$BASE_DIR/requirements.txt"
  fi
fi

echo "Preparing datasets (CUB example)"
python - <<'PY'
import sys
sys.path.insert(0, '$BASE_DIR')
try:
    from data.cub2011parts_datamodule import CUB2011Parts
    dm = CUB2011Parts(data_dir='$BASE_DIR/data', batch_size=32, num_workers=4)
    dm.prepare_data()
    dm.setup()
    print('CUB dataset prepared under $BASE_DIR/data')
except Exception as e:
    print('Warning: CUB prepare_data() failed:', e)
    pass
PY

if [ "$RUN_DRIVER" -eq 1 ]; then
  echo "Running driver script"
  bash "$SCRIPT_DIR/driver.sh" || echo "Driver script failed"
else
  echo "Skipping driver run (RUN_DRIVER=$RUN_DRIVER)"
fi

echo "Setup complete"
