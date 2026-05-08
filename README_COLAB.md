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
!bash scripts/setup.sh --data-root /content/drive/MyDrive/workspace/data
```

This downloads MNIST to `/content/drive/MyDrive/workspace/data/mnist/`.

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
subprocess.run(['curl', '-L', '-o', 'CUB_200_2011.zip', 
                'http://www.vision.caltech.edu/datasets/cub_200_2011/CUB_200_2011.tgz'], check=True)

# Check if downloaded file is .tgz or .zip
import tarfile
import zipfile

cub_file = 'CUB_200_2011.zip'  # Renamed from .tgz for consistency
if zipfile.is_zipfile(cub_file):
    print("Extracting .zip file...")
    with zipfile.ZipFile(cub_file, 'r') as zip_ref:
        zip_ref.extractall(data_root)
    
    # If .zip contained a .tgz, extract that too
    tgz_files = subprocess.run(['find', data_root, '-name', '*.tgz'], 
                                capture_output=True, text=True)
    for tgz in tgz_files.stdout.strip().split('\n'):
        if tgz:
            print(f"Extracting {tgz}...")
            with tarfile.open(tgz, 'r:gz') as tar:
                tar.extractall(data_root)
            os.remove(tgz)
elif tarfile.is_tarfile(cub_file):
    print("Extracting .tar.gz file...")
    with tarfile.open(cub_file, 'r:gz') as tar:
        tar.extractall(data_root)

# Clean up archive
os.remove(cub_file)

# Verify extraction
print("Verifying extraction...")
result = subprocess.run(['ls', '-la', f'{data_root}/CUB_200_2011/'], 
                        capture_output=True, text=True)
print(result.stdout)
```

Alternatively, if you have the file locally, upload it to Drive and then extract:

```python
import os
import zipfile
import tarfile

data_root = '/content/drive/MyDrive/workspace/data'

# If you uploaded CUB_200_2011.zip to Drive, extract it
cub_zip = f'{data_root}/CUB_200_2011.zip'
if os.path.exists(cub_zip):
    print("Extracting uploaded .zip...")
    with zipfile.ZipFile(cub_zip, 'r') as zip_ref:
        zip_ref.extractall(data_root)
    
    # Handle nested .tgz if present
    tgz_files = [f for f in os.listdir(data_root) if f.endswith('.tgz')]
    for tgz in tgz_files:
        print(f"Extracting {tgz}...")
        with tarfile.open(os.path.join(data_root, tgz), 'r:gz') as tar:
            tar.extractall(data_root)
        os.remove(os.path.join(data_root, tgz))
    
    os.remove(cub_zip)

# Verify extraction
os.system(f'ls -la {data_root}/CUB_200_2011/')
```

You should see: `images/`, `attributes/`, `parts/`, and `.txt` metadata files.

## Step 4: Run Experiments

**MNIST Training (quick smoke test)**:

```python
%cd /content/concept_transformer
!python3 ctc_mnist.py --data_dir /content/drive/MyDrive/workspace/data --max_epochs 1 --batch_size 32
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
