# ConceptTransformer — Colab Quickstart

Quick instructions for running the repo in Google Colab (GPU runtime). Colab does not require a local `venv`; we install packages into the notebook environment.

**Notebook / Runtime setup**:
1. Open a new Colab notebook and select `Runtime > Change runtime type > GPU`.
2. In a cell, clone the repo or mount your Google Drive and copy the repo there:

```bash
# clone into ephemeral Colab filesystem
!git clone https://github.com/bhavyagupta98/concept_transformer.git
%cd concept_transformer
```

**Install Python dependencies**:
```bash
# install required packages (may take several minutes)
!pip install --no-deps -r requirements.txt
```

Notes:
- Colab's preinstalled PyTorch + CUDA may differ from the wheel pinned in `requirements.txt`. If `pip install -r requirements.txt` fails on `torch`, prefer Colab's default `torch` or follow PyTorch's official Colab install instructions to get a compatible `torch`+`cuda` build.

**Run experiments (Colab cells)**:
```bash
# from a Colab cell
!python3 ctc_mnist.py
# or run individual scripts as needed
```

**Data and checkpoints**:
- Colab sessions are ephemeral — to persist datasets or checkpoints, mount Google Drive and use a workspace folder there.

```python
from google.colab import drive
drive.mount('/content/drive')
# copy or symlink checkpoint/data directories to /content/drive/MyDrive/...
```

**When to use VM instead**:
- For long runs, multi-GB datasets, or reproducible GPU environments prefer a VM (see `README_VM.md`). On VM, create and use a `venv` as documented.
