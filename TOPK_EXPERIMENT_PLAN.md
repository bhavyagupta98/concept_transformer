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

Start with a non-spatial top-$k$ ablation only.

### Why this first

- It is the smallest possible change.
- It directly answers the accuracy-versus-interpretability question.
- It avoids altering the training objective.
- It is easy to compare against the current dense baseline.

### Proposed variants

- Dense baseline.
- Top-1 attention.
- Top-3 attention.
- Top-5 attention.

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
