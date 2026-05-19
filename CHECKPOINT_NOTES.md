# Checkpoint Notes (2026-05-17)

This note captures the latest setup state, commands, and outputs used to generate Figure 2 scaling plots for MNIST even/odd.

## What we are doing

- Rebuild scaling JSON from existing checkpoints (no retraining).
- Plot Figure 2 from the paper as a single two-panel figure:
  - Left: test accuracy vs N training samples.
  - Right: val explanation loss vs N training samples (or test proxy if val is missing).
- Include two curves for $\lambda_{expl}=0.0$ and $\lambda_{expl}=2.0$.
- Save PNG outputs into the PVC-mounted `/workspace/data/figs` directory.

## Key files

- `k8s/runtime-job-gpu-minimal.yaml`
  - Job name: `kg-coop-ct`
  - Rebuilds scaling JSON from checkpoints + plotting
  - Writes output to PVC at `/workspace/data/figs`
- `scripts/generate_figures.py`
  - Adds Figure 2 plotting from `experiments/binary_mnist_scaling_ES.json`
  - Optional example plots

## Commands (inside the pod from `/workspace/concept_transformer`)

Activate venv:

```bash
. venv/bin/activate
```

Rebuild scaling JSON from checkpoints:

```bash
PYTHONPATH=/workspace/concept_transformer python scripts/rebuild_scaling_json.py \
  --runs_root /workspace/concept_transformer/binary_mnist_scaling_ES \
  --output_json /workspace/concept_transformer/experiments/binary_mnist_scaling_ES.json
```

Plot Figure 2 from JSON (two panels, two lambdas):

```bash
PYTHONPATH=/workspace/concept_transformer python scripts/generate_figures.py \
  --skip_examples \
  --scaling_json /workspace/concept_transformer/experiments/binary_mnist_scaling_ES.json \
  --output_dir /workspace/data/figs \
  --scaling_lambdas 0.0,2.0
```

Expected PNG outputs:

- `/workspace/data/figs/figure2.png`

## Kubernetes Job

Apply the job:

```bash
kubectl apply -f k8s/runtime-job-gpu-minimal.yaml -n seelab
```

Stream logs:

```bash
kubectl logs -f job/kg-coop-ct -n seelab
```

## Copy outputs back to local

```bash
kubectl cp seelab/<pod-name>:/workspace/data/figs ./figs
```

Note: Replace `<pod-name>` with the actual pod created by the job (e.g., `kg-coop-ct-xxxxx`).
