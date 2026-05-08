# ConceptTransformer — VM / Local Setup & Run

Guide for first-time VM/instance setup (GPU), downloading datasets separately, and running experiments.

**Prerequisites**:
- Linux / macOS or any VM with GPU drivers installed
- CUDA runtime matching PyTorch (recommended: CUDA 11.8 for this repo's wheels)
- `git`, `python3` (3.10+), `python3-venv`, `pip`
- `curl` and `unzip` for downloading/extracting datasets (install via `apt-get install curl unzip` on Linux)

## Step 1: Clone the Repo

```bash
git clone https://github.com/bhavyagupta98/concept_transformer.git
cd concept_transformer
```

## Step 2: Create & Activate Virtual Environment

```bash
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

Verify the venv is active (you should see `(venv)` in your shell prompt).

## Step 3: Prepare Datasets

Create a workspace data directory (persistent location for all datasets):

```bash
mkdir -p /workspace/data
cd /workspace/data
```

**Option A: Automatic download (MNIST only)**

If you only need MNIST (quick smoke test):
```bash
. venv/bin/activate  # Ensure venv is active
cd concept_transformer
sh scripts/setup.sh
```

This automatically downloads MNIST to `/root/data/mnist/`.

**Option B: Download & extract CUB dataset using curl**

For CUB dataset, download and extract manually:

1. Install dependencies (if not already installed):
```bash
# Linux (Ubuntu/Debian)
apt-get update
apt-get install -y curl unzip

# macOS
brew install curl unzip  # Usually pre-installed
```

2. Download the CUB-200-2011 dataset and extract:
```bash
cd /workspace/data
# If download is already a .tgz file:
curl -L -o CUB_200_2011.tgz \
  "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -xzf CUB_200_2011.tgz

curl -L -o segmentations.tgz \
  "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"

tar -xzf segmentations.tgz

# You should now have: /workspace/data/CUB_200_2011/
ls /workspace/data/CUB_200_2011/  # Should see: images/, attributes/, parts/, *.txt files
```


Verify extraction:
```bash
ls -la /workspace/data/CUB_200_2011/  # Should see: images/, attributes/, parts/, *.txt files
```

## Step 4: Run Experiments

Always activate the venv first:

```bash
cd /path/to/concept_transformer
. venv/bin/activate
```

**MNIST Training (quick smoke test)**:

```bash
. venv/bin/activate
python3 ctc_mnist.py
```

**CUB Training**:

```bash
. venv/bin/activate
python3 cvit_cub.py --data_dir /workspace/data --max_epochs 1 --batch_size 8
```

**General experiment arguments**:
- `--max_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--num_workers`: Number of DataLoader workers (default: 8; reduce if you hit memory issues)
- `--data_dir`: Path to the dataset root (parent of `CUB_200_2011/` or `mnist/` folder) — use `/workspace/data`
- For more options: `python3 ctc_mnist.py --help` or `python3 cvit_cub.py --help`
```

**Troubleshooting / Notes**:
- Always run experiments from the activated `venv` on VM: `. venv/bin/activate` then `python3 ...`.
- If you see missing packages (ModuleNotFoundError), re-run `pip install -r requirements.txt` inside the activated `venv`.
- If PyTorch CUDA mismatch errors occur, ensure the system CUDA and installed `torch` wheel are compatible (this repo expects `torch==2.2.0` built for cu118 by default).
- Check checkpoints and outputs in the repo subfolders (e.g., `mnist_ctc/`, `cub_cvit/`).
- If DataLoader worker crashes with "Bus error" or shared memory issues, reduce `--num_workers` (try 0 or 2).
- If `curl` fails to download, verify the dataset URL is correct and you have internet access.
