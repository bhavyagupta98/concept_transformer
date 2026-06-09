#!/usr/bin/env python3
"""Run the phase 1 CUB ViT top-k ablation on an existing checkpoint.

This script evaluates non-spatial concept attention sparsity on CUB-200-2011
using the same dense checkpoint, comparing accuracy and interpretability across
top-k values (1, 3, 5).

Note: This focuses on non-spatial concepts only (global concept attention).
Spatial concept attention (patch-level) is unchanged for Phase 1.
"""

from argparse import ArgumentParser

import torch

from ctc import load_exp
from topk_attention import patch_ctc_model_for_topk


def build_parser():
    parser = ArgumentParser(description="Phase 1 CUB ViT top-k evaluation (non-spatial)")
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
        default=16,
        help="Test batch size (default: 16 for CUB to manage GPU memory)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/data/cub2011/",
        help="Path to CUB dataset root directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on, defaults to cuda if available else cpu",
    )
    return parser


def evaluate_model(model, dataloader, device, dataset_size=None):
    """Evaluate model and return accuracy and average active concepts.
    
    Args:
        model: Lightning model to evaluate
        dataloader: Test dataloader
        device: Device to run on
        dataset_size: Optional total size of dataset (for reporting)
    
    Returns:
        dict with 'accuracy' and 'avg_active_concepts' keys
    """
    model.eval()
    model.to(device)

    total_correct = 0
    total_examples = 0
    total_active_concepts = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x, _, _, y = batch
            x = x.to(device)
            y = y.to(device)

            logits, _, concept_attn, _ = model(x)
            preds = torch.argmax(logits, dim=-1)

            total_correct += (preds == y).sum().item()
            total_examples += y.numel()

            if concept_attn is not None:
                total_active_concepts += (concept_attn > 0).sum(dim=-1).float().sum().item()

            num_batches += 1
            if batch_idx % 10 == 0:
                print(f"  Evaluated {total_examples} examples...")

    accuracy = total_correct / total_examples if total_examples else 0.0
    avg_active_concepts = total_active_concepts / total_examples if total_examples else 0.0
    return {
        "accuracy": accuracy,
        "avg_active_concepts": avg_active_concepts,
        "num_batches": num_batches,
    }


def format_row(name, k_value, metrics):
    return (
        f"| {name:<12} | {str(k_value):<6} | "
        f"{metrics['accuracy']:.4f} | {metrics['avg_active_concepts']:.2f} |"
    )


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("[Phase 1 CUB ViT] Loading checkpoint and data module...")
    model, data_module = load_exp(args.run_path)
    
    # Override data module settings if provided
    data_module.batch_size = args.batch_size
    data_module.num_workers = args.num_workers
    data_module.data_dir = args.data_dir
    
    # Setup data
    data_module.prepare_data()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    dataset_size = len(data_module.cub_test) if hasattr(data_module, "cub_test") else None

    print(f"\n[Phase 1 CUB ViT] Evaluating dense baseline...")
    dense_metrics = evaluate_model(model, test_loader, device, dataset_size)

    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    sparse_results = []
    
    for k_value in k_values:
        print(f"\n[Phase 1 CUB ViT] Evaluating top-{k_value}...")
        sparse_model = patch_ctc_model_for_topk(model, k_value)
        sparse_metrics = evaluate_model(sparse_model, test_loader, device, dataset_size)
        sparse_results.append((k_value, sparse_metrics))

    print("\n" + "=" * 65)
    print("| variant      | k      | accuracy | avg_active_concepts |")
    print("|--------------|--------|----------|---------------------|")
    print(format_row("dense", "all", dense_metrics))
    for k_value, metrics in sparse_results:
        print(format_row("top-k", k_value, metrics))
    print("=" * 65)


if __name__ == "__main__":
    main()
