#!/usr/bin/env bash
set -euo pipefail

# Driver script to run key experiments in the repository.
# Controlled by environment variables:
#  RUN_MNIST (default 1) - run `ctc_mnist.py` smoke test
#  RUN_CUB (default 0) - run `cvit_cub.py` (heavy)
#  RUN_SCALING (default 0) - run `run_mnist_scaling_experiments.py` (heavy)

RUN_MNIST=${RUN_MNIST:-1}
RUN_CUB=${RUN_CUB:-0}
RUN_SCALING=${RUN_SCALING:-0}

echo "Driver starting: RUN_MNIST=$RUN_MNIST, RUN_CUB=$RUN_CUB, RUN_SCALING=$RUN_SCALING"

source /workspace/venv/bin/activate
cd /workspace

if [ "$RUN_MNIST" -eq 1 ]; then
  echo "Running MNIST smoke test"
  python3 ctc_mnist.py --max_epochs 1 --n_train_samples 200 --batch_size 32 --warmup 0 || echo "ctc_mnist failed"
fi

if [ "$RUN_CUB" -eq 1 ]; then
  echo "Running CUB (cvit_cub.py)"
  python3 cvit_cub.py --data_dir /workspace/data --max_epochs 1 --batch_size 8 || echo "cvit_cub failed"
fi

if [ "$RUN_SCALING" -eq 1 ]; then
  echo "Running mnist scaling experiments"
  python3 run_mnist_scaling_experiments.py || echo "run_mnist_scaling_experiments failed"
fi

echo "Driver finished"
