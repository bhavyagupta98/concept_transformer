# ConceptTransformer — Colab Quickstart

Quick instructions for running the repo in Google Colab (GPU runtime). This guide covers venv-free setup, downloading data separately, and running experiments.

**Notebook / Runtime setup**:
1. Open a new Colab notebook and select `Runtime > Change runtime type > GPU`.
2. In a cell, clone the repo:

```python
!git clone https://github.com/bhavyagupta98/concept_transformer.git
%cd concept_transformer
```

## Step 1: Install Python Dependencies

```python
!pip install --no-cache-dir -r requirements.txt
```

Notes:
- Colab's preinstalled PyTorch + CUDA may differ from the wheel pinned in `requirements.txt`. If `pip install -r requirements.txt` fails on `torch`, prefer Colab's default `torch` or follow PyTorch's official Colab install instructions.

## Step 2: Mount Google Drive (for persistent storage)

```python
from google.colab import drive
drive.mount('/content/drive')
```

This allows you to persist datasets and checkpoints.

## Step 3: Prepare Datasets

Create a workspace data directory in your Google Drive:

```python
import os
data_root = '/content/drive/MyDrive/workspace/data'
os.makedirs(data_root, exist_ok=True)
```

**Option A: Download MNIST automatically**

```python
%cd /content/concept_transformer
!bash scripts/setup.sh
```


**Option B: Download & extract CUB dataset using curl**

For CUB dataset, download and extract manually in Colab:

1. Download the CUB-200-2011 dataset using curl and extract:

```python
import os
import subprocess

data_root = '/content/drive/MyDrive/workspace/data'
os.chdir(data_root)

# Download CUB dataset (handling both .zip and .tgz)
print("Downloading CUB-200-2011...")
!curl -L -o CUB_200_2011.tgz \
  "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

!tar -xzf CUB_200_2011.tgz

!curl -L -o segmentations.tgz \
  "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"

!tar -xzf segmentations.tgz
# Verify extraction
print("Verifying extraction...")
result = subprocess.run(['ls', '-la', f'{data_root}/CUB_200_2011/'], 
                        capture_output=True, text=True)
print(result.stdout)
```

Alternatively, if you have the file locally, upload it to Drive and then extract:

**MNIST Training (quick smoke test)**:

```python
%cd /content/concept_transformer
!python3 ctc_mnist.py
```

**CUB Training**:

```python
%cd /content/concept_transformer
!python3 cvit_cub.py --data_dir /content/drive/MyDrive/workspace/data --max_epochs 1 --batch_size 8
```

**General experiment arguments**:
- `--max_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--num_workers`: Number of DataLoader workers (default: 8; reduce if you hit memory issues)
- `--data_dir`: Path to the dataset root (parent of `CUB_200_2011/` or `mnist/` folder) — use `/content/drive/MyDrive/workspace/data`
- For more options: `!python3 ctc_mnist.py --help` or `!python3 cvit_cub.py --help`

## Notes

- **Colab sessions are ephemeral** — to persist datasets and checkpoints across sessions, store everything in Google Drive at `/content/drive/MyDrive/workspace/`.
- If you prefer local training (VM), see `README_VM.md` for detailed VM setup and dataset extraction instructions.
- If DataLoader worker crashes, reduce `--num_workers` (try 0 or 2).
- Large file uploads to Drive may be slow; consider using `curl` directly in Colab cells instead (see Option B above).

