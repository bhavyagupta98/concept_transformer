# Checkpoint Notes (2026-05-17)

This note captures the latest setup state, commands, and outputs used to generate Figure 2 scaling plots for MNIST even/odd.

## What we are doing

- Run the MNIST scaling experiments (varying N training samples) to generate an `epyc` JSON.
- Plot Figure 2 from the paper:
  - Test accuracy vs N training samples.
  - Val explanation loss vs N training samples.
- Save PNG outputs into the PVC-mounted `/workspace/data/figs` directory.

## Key files

- `k8s/runtime-job-gpu-minimal.yaml`
  - Job name: `kg-coop-ct`
  - Runs scaling experiments + plotting
  - Writes output to PVC at `/workspace/data/figs`
- `scripts/generate_figures.py`
  - Adds Figure 2 plotting from `experiments/binary_mnist_scaling_ES.json`
  - Optional example plots

## Commands (inside the pod from `/workspace/concept_transformer`)

Activate venv:

```bash
. venv/bin/activate
```

Run scaling experiments (generates JSON):

```bash
PYTHONPATH=/workspace/concept_transformer python run_mnist_scaling_experiments.py
```

Plot Figure 2 from JSON:

```bash
PYTHONPATH=/workspace/concept_transformer python scripts/generate_figures.py \
  --skip_examples \
  --scaling_json experiments/binary_mnist_scaling_ES.json \
  --output_dir /workspace/data/figs
```

Expected PNG outputs:

- `/workspace/data/figs/figure2_test_acc_vs_n.png`
- `/workspace/data/figs/figure2_val_expl_loss_vs_n.png`

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
