# ConceptTransformer — VM / Local Setup & Run

Short guide for first-time VM/instance setup (GPU) and quick steps for code updates and running experiments. Run experiments from the project's virtual environment (`venv`) unless noted.

**Prerequisites**:
- Linux / macOS or any VM with GPU drivers installed
- CUDA runtime matching PyTorch (recommended: CUDA 11.8 for this repo's wheels)
- `git`, `python3` (3.10+), `python3-venv`, `pip`

**First-time setup (VM)**:
1. Clone the repo:

```bash
git clone https://github.com/bhavyagupta98/concept_transformer.git
cd concept_transformer
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
```

3. Install required Python packages:

```bash
# inside the activated venv
pip install --no-cache-dir -r requirements.txt
```

4. (Optional) Run the project setup helper (downloads/prepares datasets):

```bash
bash scripts/setup.sh
```

**Run experiments (examples)**:

Always activate the venv first:

```bash
. venv/bin/activate
python3 ctc_mnist.py          # MNIST smoke test / training
# or
bash scripts/driver.sh        # environment-controlled driver
```

**Updating code & syncing to a running k8s pod**:
- If the process runs on the same VM, use `git pull` / `git fetch` + `git reset --hard origin/main`.
- If you need to update a k8s pod that does not have `git`, copy the repo from your VM:

```bash
# from VM (repo root)
# create a tar excluding large venvs
tar --exclude='./venv' --exclude='./venv_upgraded' -czf /tmp/ctc_changes.tar.gz .
# copy into pod (example pod name and namespace)
kubectl cp /tmp/ctc_changes.tar.gz seelab/concept-transformer-gpu-runner:/workspace/ -c ct-runner
kubectl exec -it concept-transformer-gpu-runner -n seelab -- bash -lc "cd /workspace && tar -xzf ctc_changes.tar.gz && rm ctc_changes.tar.gz"
```

**Troubleshooting / Notes**:
- Always run experiments from the activated `venv` on VM: `. venv/bin/activate` then `python3 ...`.
- If you see missing packages (ModuleNotFoundError), re-run `pip install -r requirements.txt` inside the activated `venv`.
- If PyTorch CUDA mismatch errors occur, ensure the system CUDA and installed `torch` wheel are compatible (this repo expects `torch==2.2.0` built for cu118 by default).
- Check checkpoints and outputs in the repo subfolders (e.g., `mnist_ctc/`).
