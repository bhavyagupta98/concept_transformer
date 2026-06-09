#!/usr/bin/env python3
"""Run the phase 1 MNIST top-k ablation on an existing checkpoint.

This script keeps the training path unchanged and evaluates the dense model
versus sparse inference variants on the same test set.
"""

from argparse import ArgumentParser

import torch

from ctc import load_exp
from topk_attention import patch_ctc_model_for_topk


def build_parser():
    parser = ArgumentParser(description="Phase 1 MNIST top-k evaluation")
    parser.add_argument(
        "--run_path",
        required=True,
        help="Checkpoint path or experiment directory to load with load_exp",
    )
    parser.add_argument(
        "--k_values",
        default="1,3,5",
        help="Comma-separated top-k values to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for the test batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional override for DataLoader workers",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on, defaults to cuda if available else cpu",
    )
    return parser


def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    total_correct = 0
    total_examples = 0
    total_active_concepts = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x, _, _, y = batch
            x = x.to(device)
            y = y.to(device)

            logits, _, concept_attn, _ = model(x)
            preds = torch.argmax(logits, dim=-1)

            total_correct += (preds == y).sum().item()
            total_examples += y.numel()

            if concept_attn is not None:
                total_active_concepts += (concept_attn > 0).sum(dim=-1).float().sum().item()

    accuracy = total_correct / total_examples
    avg_active_concepts = total_active_concepts / total_examples if total_examples else 0.0
    return {
        "accuracy": accuracy,
        "avg_active_concepts": avg_active_concepts,
    }


def format_row(name, k_value, metrics):
    return (
        f"| {name:<12} | {str(k_value):<6} | "
        f"{metrics['accuracy']:.4f} | {metrics['avg_active_concepts']:.2f} |"
    )


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, data_module = load_exp(args.run_path)
    if args.batch_size is not None:
        data_module.batch_size = args.batch_size
    if args.num_workers is not None:
        data_module.num_workers = args.num_workers

    test_loader = data_module.test_dataloader()

    dense_metrics = evaluate_model(model, test_loader, device)

    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    sparse_results = []
    for k_value in k_values:
        sparse_model = patch_ctc_model_for_topk(model, k_value)
        sparse_metrics = evaluate_model(sparse_model, test_loader, device)
        sparse_results.append((k_value, sparse_metrics))

    print("| variant      | k      | accuracy | avg_active_concepts |")
    print("|--------------|--------|----------|---------------------|")
    print(format_row("dense", "all", dense_metrics))
    for k_value, metrics in sparse_results:
        print(format_row("top-k", k_value, metrics))


if __name__ == "__main__":
    main()
