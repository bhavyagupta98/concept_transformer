# Top-$k$ Concept Attention Experiment Plan

## Goal

Study whether inference-time top-$k$ sparsity over concept attention can improve interpretability with minimal loss in accuracy, while keeping the current Concept Transformer code path unchanged and reproducible.

## Core Constraint

The existing training and evaluation scripts should remain untouched as much as possible. Any new behavior should live in separate wrapper classes, helper modules, or standalone evaluation scripts so that current checkpoints, training runs, and results remain comparable.

## Guiding Principle

We will treat this as an additive experiment:

- Preserve the current model implementation and training logic.
- Add a separate inference-time path for top-$k$ masking.
- Reuse existing checkpoints and datasets.
- Keep the original dense baseline as the reference point.

## Phase 1: Minimal Ablation

Start with a non-spatial (global) top-$k$ ablation only. We evaluate on both MNIST (quickest) and CUB ViT (more realistic).

### MNIST Variant

### Why this first

- It is the smallest possible change.
- It directly answers the accuracy-versus-interpretability question.
- It avoids altering the training objective.
- It is easy to compare against the current dense baseline.

### CUB ViT Variant

After validating on MNIST, run the same top-$k$ ablation on CUB-200-2011 using the ViT backbone. This is more realistic and interpretability gains may be more pronounced on real-world fine-grained classification.

Key differences from MNIST:
- Model: `cub_cvit` (Vision Transformer backbone with 13 non-spatial + 95 spatial concepts).
- Only non-spatial concepts are pruned; spatial attention is kept dense.
- Larger model and dataset; expect longer evaluation time.
- Data requirement: CUB-200-2011 must be downloaded and placed at the path specified in `--data_dir`.

### Proposed variants

- Dense baseline.
- Top-1 attention.
- Top-3 attention.
- Top-5 attention.

### Phase 1 Commands

#### MNIST (Quick Validation)

```bash
# Local (using venv_upgraded)
./venv_upgraded/bin/python run_topk_phase1_mnist.py --run_path mnist_ctc/ExplanationMNIST_expl5.0/binary_mnist_best_ckpt.ckpt --k_values 1,3,5

# On Kubernetes pod
kubectl exec -it concept-transformer-gpu-runner -- bash -c "cd /workspace/concept_transformer && . venv/bin/activate && python run_topk_phase1_mnist.py --run_path mnist_ctc/ExplanationMNIST_expl5.0/binary_mnist_best_ckpt.ckpt --k_values 1,3,5"
```

#### CUB ViT (Full Evaluation)

```bash
# Local (using venv_upgraded)
./venv_upgraded/bin/python run_topk_phase1_cub.py --run_path cub_cvit/CUB2011Parts_expl1.0/cub_best_ckpt.ckpt --k_values 1,3,5 --data_dir ~/data/cub2011/ --batch_size 16

# On Kubernetes pod
kubectl exec -it concept-transformer-gpu-runner -- bash -c "cd /workspace/concept_transformer && . venv/bin/activate && python run_topk_phase1_cub.py --run_path cub_cvit/CUB2011Parts_expl1.0/cub_best_ckpt.ckpt --k_values 1,3,5 --data_dir /workspace/data/cub2011/ --batch_size 16"
```

Notes:
- Adjust checkpoint paths (`--run_path`) based on your actual trained run directory and filename.
- For CUB, ensure the dataset is available at `--data_dir` (default: `~/data/cub2011/`).
- CUB evaluation is slower than MNIST; expect 5-15 minutes depending on GPU and batch size.

### Expected output

- Test accuracy for each variant.
- Number of active concepts per prediction.
- A short qualitative comparison on a few examples.

## Phase 2: Spatial Extension, If Time Permits

If the non-spatial experiment looks promising, extend the same idea to spatial concept attention.

### Why this is optional

- Spatial attention is harder to summarize.
- It is less immediately interpretable than global concept attention.
- It may require extra care in defining what counts as an active concept.

### Suggested scope

- Apply the same top-$k$ logic to spatial concept attention.
- Report token-level or patch-level active concept statistics separately.
- Keep the non-spatial results as the main comparison.

## Phase 3: Runtime / Compute Analysis, If Time Permits

Only attempt runtime analysis if the top-$k$ path can actually reduce executed work.

### Important caveat

If the implementation only masks attention after it is computed, the runtime may not meaningfully improve. In that case, accuracy and interpretability results are still valid, but compute claims should be framed as theoretical rather than measured speedup.

### Compute metrics to consider

- Median inference latency per batch.
- Throughput on the same hardware.
- Theoretical reduction in active attention entries.

## Metrics

### Primary

- Test accuracy.

### Secondary

- Average number of active concepts per prediction.
- Sparsity level induced by top-$k$.

### Optional

- Inference latency.
- Throughput.
- Memory usage, if measurable with low effort.

## Reporting Tables

### Table 1: Accuracy vs $k$

Columns:

- Model variant.
- $k$ value.
- Test accuracy.
- Accuracy delta vs dense baseline.

### Table 2: Interpretability Summary

Columns:

- Model variant.
- $k$ value.
- Average active concepts per prediction.
- Notes on qualitative conciseness.

### Optional Table 3: Runtime Summary

Columns:

- Model variant.
- $k$ value.
- Median latency.
- Throughput.
- Relative change vs baseline.

## Why Top-$k$ Matters

Top-$k$ evaluation is mainly about the accuracy-versus-sparsity trade-off at inference time, not about shrinking checkpoint size. The model weights stay the same; what changes is how many concepts are allowed to contribute to each prediction. That makes the result useful for interpretability because it shows whether the model can stay accurate while relying on fewer active concepts. If the masking is applied only after attention scores are computed, then runtime gains are limited, so the primary claim should stay focused on sparsity and explainability rather than deployment size.

## Logged Phase 1 Results

### MNIST Top-$k$ Ablation

Recorded from the phase 1 evaluation command on the MNIST checkpoint.

| variant | k | accuracy | delta vs dense | avg_active_concepts |
|---|---:|---:|---:|---:|
| dense | all | 0.9899 | 0.0000 | 10.00 |
| top-k | 1 | 0.9890 | -0.0009 | 1.00 |
| top-k | 3 | 0.9901 | +0.0002 | 3.00 |
| top-k | 5 | 0.9899 | +0.0000 | 5.00 |

Interpretation:

- The sparse variants preserve dense-model accuracy to within about 0.1 percentage points.
- The active concept count drops exactly to $k$, so the interpretability/sparsity gain is direct and easy to report.
- On MNIST, top-3 and top-5 are effectively indistinguishable from the dense baseline on accuracy, which makes this a strong Phase 1 sanity check for the sparsity idea.

### CUB ViT Top-$k$ Ablation

Recorded from the phase 1 evaluation command on the CUB ViT checkpoint.

| variant | k | accuracy | delta vs dense | avg_active_concepts |
|---|---:|---:|---:|---:|
| dense | all | 0.8357 | 0.0000 | 13.00 |
| top-k | 1 | 0.1515 | -0.6842 | 1.00 |
| top-k | 3 | 0.6512 | -0.1845 | 3.00 |
| top-k | 5 | 0.7741 | -0.0616 | 5.00 |
| top-k | 8 | 0.8108 | -0.0249 | 8.00 |
| top-k | 10 | 0.8150 | -0.0207 | 10.00 |

Interpretation:

- This is a meaningful result because the sparse variants clearly expose the accuracy sensitivity of the CUB model to concept budget.
- Top-1 is too aggressive for this checkpoint and causes a severe accuracy collapse, so it is not a viable setting for CUB.
- Top-3 and top-5 are much better, and top-8/top-10 get very close to the dense baseline while still enforcing a measurable sparsity budget.
- The trend is monotonic: as k increases, accuracy recovers toward the dense model, which is what we would expect if concept budget is the main bottleneck.
- Compared with MNIST, CUB is substantially less tolerant of extreme sparsity, which is exactly the kind of dataset-dependent behavior the experiment is meant to reveal.

## Implementation Strategy

We should prefer a separate module or wrapper class rather than modifying the existing model directly.

Recommended structure:

- Keep the current dense Concept Transformer unchanged.
- Add a small inference wrapper that applies top-$k$ masking.
- Add a separate evaluation script that loads a checkpoint and runs the dense and sparse variants.
- Keep logging and result aggregation in a standalone analysis script or notebook.

This keeps the original training path reproducible and makes it easy to disable the experiment entirely.

## Success Criteria

The experiment is useful if:

- Accuracy drops only modestly for small $k$.
- Interpretability improves in a clear and reportable way.
- The comparison is simple enough to reproduce from existing checkpoints.

## Minimum Deliverable

If time is limited, the minimal useful result is:

- Dense baseline vs top-1/top-3/top-5.
- Test accuracy comparison.
- Active concept count comparison.
- Short qualitative discussion.

## Nice-to-Have Deliverable

If there is extra time:

- Spatial concept variant.
- Runtime comparison.
- A combined table or figure summarizing the trade-off between sparsity and performance.

## Kubernetes: Copy & Run Commands

Below are concise, copy-pasteable commands to copy only the experiment files into the GPU pod defined by `k8s/runtime-pod-gpu-minimal.yaml`, prepare the environment, run Phase 1, and retrieve results.

Assumptions:
- Pod name is `concept-transformer-gpu-runner` (see `k8s/runtime-pod-gpu-minimal.yaml`).
- Target repo path inside pod: `/workspace/concept_transformer`.
- Checkpoint path (adjust if needed): `mnist_ctc/ExplanationMNIST_expl5.0/binary_mnist_best_ckpt.ckpt`.

Step A — create/apply the pod and wait for ready:

```bash
kubectl apply -f k8s/runtime-pod-gpu-minimal.yaml
kubectl wait --for=condition=Ready pod/concept-transformer-gpu-runner --timeout=180s
kubectl get pods -l app=concept-transformer -o wide
```

Step B — copy only changed experiment files into the pod:

```bash
POD=concept-transformer-gpu-runner
kubectl exec -it $POD -- mkdir -p /workspace/concept_transformer
kubectl cp TOPK_EXPERIMENT_PLAN.md $POD:/workspace/concept_transformer/TOPK_EXPERIMENT_PLAN.md
kubectl cp topk_attention.py   $POD:/workspace/concept_transformer/topk_attention.py
kubectl cp run_topk_phase1_mnist.py $POD:/workspace/concept_transformer/run_topk_phase1_mnist.py
kubectl cp run_topk_phase1_cub.py $POD:/workspace/concept_transformer/run_topk_phase1_cub.py
```

Step C — prepare runtime in the pod (interactive shell):

```bash
kubectl exec -it $POD -- /bin/bash
cd /workspace/concept_transformer
python3 -m venv venv
. venv/bin/activate
pip install --no-cache-dir -r requirements.txt
# ensure checkpoint is present at the expected path
```

Step D — run the Phase‑1 top‑k evaluation (inside pod with venv activated):

**MNIST (quick validation):**

```bash
python run_topk_phase1_mnist.py --run_path mnist_ctc/ExplanationMNIST_expl5.0/binary_mnist_best_ckpt.ckpt --k_values 1,3,5
```

**CUB ViT (full evaluation):**

```bash
python run_topk_phase1_cub.py --run_path cub_cvit/CUB2011Parts_expl1.0/cub_best_ckpt.ckpt --k_values 1,3,5 --data_dir /workspace/data/cub2011/ --batch_size 16
```

Or run non-interactively from host:

**MNIST:**

```bash
kubectl exec -it $POD -- bash -c "cd /workspace/concept_transformer && . venv/bin/activate && python run_topk_phase1_mnist.py --run_path mnist_ctc/ExplanationMNIST_expl5.0/binary_mnist_best_ckpt.ckpt --k_values 1,3,5"
```

**CUB ViT:**

```bash
kubectl exec -it $POD -- bash -c "cd /workspace/concept_transformer && . venv/bin/activate && python run_topk_phase1_cub.py --run_path cub_cvit/CUB2011Parts_expl1.0/cub_best_ckpt.ckpt --k_values 1,3,5 --data_dir /workspace/data/cub2011/ --batch_size 16"
```

Step E — retrieve results/logs:

```bash
kubectl logs $POD > pod_run_topk_output.log
# If the script writes files under /workspace/concept_transformer/results:
kubectl cp $POD:/workspace/concept_transformer/results ./results
```

Notes:
- If you prefer the pod to clone the repo itself, the YAML already sets `GIT_REPO` and the container command shows how to run `scripts/setup.sh` inside the pod.
- Copying the `venv_upgraded/` directory is possible but large; installing requirements inside the pod is preferred.
- If your cluster uses a namespace, add `-n <namespace>` to all `kubectl` commands.
