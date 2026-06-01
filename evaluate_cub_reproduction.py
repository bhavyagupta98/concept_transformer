#!/usr/bin/env python3
"""Evaluate CUB reproduction checkpoints and export report-ready artifacts.

This script is additive and does not modify training/model code. It:
- evaluates with-concepts and baseline checkpoints on the same CUB test split
- writes accuracy tables and paper-vs-ours comparison
- optionally saves one correct and one incorrect qualitative explanation image
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ctc import load_exp


DEFAULT_PAPER_WITH_CONCEPTS = 88.0
DEFAULT_PAPER_WITHOUT_CONCEPTS = 76.9


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CUB reproduction checkpoints")
    parser.add_argument("--with_concepts_ckpt", required=True, help="Path to with-concepts checkpoint")
    parser.add_argument("--baseline_ckpt", required=True, help="Path to baseline checkpoint")
    parser.add_argument("--data_dir", default="/workspace/data", help="CUB data root")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--save_examples", action="store_true")
    parser.add_argument("--with_concepts_label", default="CT with concepts")
    parser.add_argument("--baseline_label", default="CT without concepts")
    parser.add_argument("--paper_with_concepts", type=float, default=DEFAULT_PAPER_WITH_CONCEPTS)
    parser.add_argument("--paper_without_concepts", type=float, default=DEFAULT_PAPER_WITHOUT_CONCEPTS)
    parser.add_argument("--top_global_k", type=int, default=8)
    parser.add_argument("--top_spatial_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def unnorm_cub(img_tensor):
    sd = np.array([0.229, 0.224, 0.225])
    mu = np.array([0.485, 0.456, 0.406])
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * sd + mu, 0.0, 1.0)


def load_raw_cub_image(dataset, idx):
    """Load the original RGB image used by the sample index (cropped if available)."""
    sample_metadata = dataset.data.iloc[idx]
    rel_path = sample_metadata.filepath
    path = os.path.join(dataset.dataset_root, dataset.sample_folder, rel_path)
    if not os.path.isfile(path):
        fallback = os.path.join(dataset.dataset_root, dataset.base_folder, rel_path)
        if os.path.isfile(fallback):
            path = fallback
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def load_attribute_names(data_dir):
    attr_path = os.path.join(os.path.expanduser(data_dir), "CUB_200_2011", "attributes.txt")
    if not os.path.isfile(attr_path):
        parent_alt = os.path.join(os.path.expanduser(data_dir), "attributes.txt")
        if os.path.isfile(parent_alt):
            attr_path = parent_alt
    if not os.path.isfile(attr_path):
        return None

    names = []
    with open(attr_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                names.append(parts[1])
    return np.array(names)


def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            x, _, _, y = batch
            x = x.to(device)
            y = y.to(device)

            logits, _, _, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_examples += y.numel()

    accuracy = total_correct / total_examples if total_examples else 0.0
    return accuracy, total_examples


def collect_predictions(model, dataloader, device):
    model.eval()
    model.to(device)

    records = []
    running_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            x, _, _, y = batch
            x = x.to(device)
            y = y.to(device)

            logits, _, concept_attn, spatial_concept_attn = model(x)
            preds = torch.argmax(logits, dim=-1)

            bsz = x.shape[0]
            for i in range(bsz):
                records.append(
                    {
                        "idx": running_idx + i,
                        "pred": int(preds[i].item()),
                        "gt": int(y[i].item()),
                        "correct": bool(preds[i].item() == y[i].item()),
                        "concept_attn": concept_attn[i].detach().cpu() if concept_attn is not None else None,
                        "spatial_concept_attn": (
                            spatial_concept_attn[i].detach().cpu() if spatial_concept_attn is not None else None
                        ),
                    }
                )
            running_idx += bsz

    return records


def topk_indices_and_scores(vec, k):
    if vec is None:
        return [], []
    # Some attention tensors come as shape [1, n_concepts]; flatten to 1D so
    # returned indices are plain ints (not nested lists).
    vec = vec.reshape(-1)
    k = max(1, min(k, vec.shape[0]))
    vals, idxs = torch.topk(vec, k=k, dim=0)
    return idxs.tolist(), vals.tolist()


def topk_spatial_concepts(spatial_attn, k):
    if spatial_attn is None:
        return [], []
    # Aggregate spatial concept saliency over patches by max to highlight any strong local evidence.
    concept_scores = spatial_attn.max(dim=0).values
    return topk_indices_and_scores(concept_scores, k)


def maybe_class_name(dm, class_idx):
    try:
        raw = dm.cub_test.class_names[class_idx]
        return raw.split("/")[0][4:]
    except Exception:
        return str(class_idx)


def render_example(example_record, data_module, output_path, attr_names, top_global_k=8, top_spatial_k=8):
    ds = data_module.cub_test
    # Use original RGB sample for faithful visualization (not normalized tensor values).
    img = load_raw_cub_image(ds, example_record["idx"])

    pred_name = maybe_class_name(data_module, example_record["pred"])
    gt_name = maybe_class_name(data_module, example_record["gt"])

    global_idxs, global_scores = topk_indices_and_scores(example_record["concept_attn"], top_global_k)
    spatial_idxs, spatial_scores = topk_spatial_concepts(example_record["spatial_concept_attn"], top_spatial_k)

    global_lines = []
    for i, s in zip(global_idxs, global_scores):
        if attr_names is not None:
            attr_id = ds.non_spatial_attributes_pos[i]
            name = attr_names[attr_id - 1]
        else:
            name = f"global_concept_{i}"
        global_lines.append(f"{name} ({s:.3f})")

    spatial_lines = []
    for i, s in zip(spatial_idxs, spatial_scores):
        if attr_names is not None:
            attr_id = ds.spatial_attributes_pos[i]
            name = attr_names[attr_id - 1]
        else:
            name = f"spatial_concept_{i}"
        spatial_lines.append(f"{name} ({s:.3f})")

    fig = plt.figure(figsize=(12, 6))
    ax_img = plt.subplot2grid((1, 2), (0, 0))
    ax_txt = plt.subplot2grid((1, 2), (0, 1))

    # Overlay spatial saliency heatmap derived from max concept attention per patch.
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("CUB Example")

    attn = example_record["spatial_concept_attn"]
    if attn is not None:
        n_patch = int(np.sqrt(attn.shape[0]))
        patch_scores = attn.max(dim=1).values.reshape(n_patch, n_patch).numpy()
        patch_scores = np.nan_to_num(patch_scores, nan=0.0, posinf=0.0, neginf=0.0)
        vmax = float(patch_scores.max()) if patch_scores.size else 1.0
        vmin = float(patch_scores.min()) if patch_scores.size else 0.0
        if vmax > vmin:
            patch_scores = (patch_scores - vmin) / (vmax - vmin)
        else:
            patch_scores = np.zeros_like(patch_scores)

        heat = Image.fromarray(np.uint8(patch_scores * 255.0)).resize(
            (img.shape[1], img.shape[0]), Image.BILINEAR
        )
        heat = np.asarray(heat).astype(np.float32) / 255.0
        ax_img.imshow(heat, cmap="jet", alpha=0.35 * heat)

    ax_txt.axis("off")
    header = "correct" if example_record["correct"] else "wrong"
    lines = [
        f"Prediction: {pred_name} ({header})",
        f"Ground Truth: {gt_name}",
        "",
        "Global explanations:",
    ]
    lines.extend([f"- {x}" for x in global_lines[:top_global_k]])
    lines.append("")
    lines.append("Spatial explanations:")
    lines.extend([f"- {x}" for x in spatial_lines[:top_spatial_k]])

    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "idx": example_record["idx"],
        "prediction_idx": example_record["pred"],
        "ground_truth_idx": example_record["gt"],
        "prediction_name": pred_name,
        "ground_truth_name": gt_name,
        "correct": example_record["correct"],
        "top_global": global_lines,
        "top_spatial": spatial_lines,
        "output_path": output_path,
    }


def checkpoint_meta(model):
    hp = getattr(model, "hparams", {})
    as_dict = dict(hp) if hasattr(hp, "items") else {}
    return {
        "expl_lambda": as_dict.get("expl_lambda", None),
        "baseline": as_dict.get("baseline", None),
        "max_epochs": as_dict.get("max_epochs", None),
        "batch_size": as_dict.get("batch_size", None),
        "data_dir": as_dict.get("data_dir", None),
        "data_name": as_dict.get("data_name", None),
        "ctc_model": as_dict.get("ctc_model", None),
    }


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.output_dir)
    examples_dir = os.path.join(args.output_dir, "cub_explanation_examples")
    if args.save_examples:
        ensure_dir(examples_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading with-concepts checkpoint...")
    with_model, with_dm = load_exp(args.with_concepts_ckpt)
    with_dm.batch_size = args.batch_size
    with_dm.num_workers = args.num_workers
    with_dm.data_dir = args.data_dir
    with_dm.prepare_data()
    with_dm.setup(stage="test")
    with_loader = with_dm.test_dataloader()

    print("Loading baseline checkpoint...")
    base_model, base_dm = load_exp(args.baseline_ckpt)
    base_dm.batch_size = args.batch_size
    base_dm.num_workers = args.num_workers
    base_dm.data_dir = args.data_dir
    base_dm.prepare_data()
    base_dm.setup(stage="test")
    base_loader = base_dm.test_dataloader()

    print("Evaluating with-concepts...")
    with_acc, with_n = evaluate_model(with_model, with_loader, device)
    print("Evaluating baseline...")
    base_acc, base_n = evaluate_model(base_model, base_loader, device)

    ts = datetime.now(timezone.utc).isoformat()
    with_meta = checkpoint_meta(with_model)
    base_meta = checkpoint_meta(base_model)

    acc_rows = [
        {
            "model_variant": "with_concepts",
            "label": args.with_concepts_label,
            "checkpoint_path": args.with_concepts_ckpt,
            "expl_lambda": with_meta["expl_lambda"],
            "baseline_flag": with_meta["baseline"],
            "test_accuracy": with_acc,
            "num_test_samples": with_n,
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "eval_timestamp": ts,
        },
        {
            "model_variant": "baseline",
            "label": args.baseline_label,
            "checkpoint_path": args.baseline_ckpt,
            "expl_lambda": base_meta["expl_lambda"],
            "baseline_flag": base_meta["baseline"],
            "test_accuracy": base_acc,
            "num_test_samples": base_n,
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "eval_timestamp": ts,
        },
    ]

    acc_csv = os.path.join(args.output_dir, "cub_reproduction_accuracy.csv")
    write_csv(
        acc_csv,
        [
            "model_variant",
            "label",
            "checkpoint_path",
            "expl_lambda",
            "baseline_flag",
            "test_accuracy",
            "num_test_samples",
            "data_dir",
            "batch_size",
            "num_workers",
            "eval_timestamp",
        ],
        acc_rows,
    )

    acc_json = os.path.join(args.output_dir, "cub_reproduction_accuracy.json")
    with open(acc_json, "w", encoding="utf-8") as f:
        json.dump(acc_rows, f, indent=2)

    paper_without = args.paper_without_concepts / 100.0
    paper_with = args.paper_with_concepts / 100.0
    paper_gain = paper_with - paper_without
    our_gain = with_acc - base_acc

    vs_rows = [
        {
            "metric": "ct_without_concepts_accuracy",
            "paper_value": paper_without,
            "our_value": base_acc,
            "delta_our_minus_paper": base_acc - paper_without,
        },
        {
            "metric": "ct_with_concepts_accuracy",
            "paper_value": paper_with,
            "our_value": with_acc,
            "delta_our_minus_paper": with_acc - paper_with,
        },
        {
            "metric": "concept_gain",
            "paper_value": paper_gain,
            "our_value": our_gain,
            "delta_our_minus_paper": our_gain - paper_gain,
        },
    ]

    vs_csv = os.path.join(args.output_dir, "cub_reproduction_vs_paper.csv")
    write_csv(vs_csv, ["metric", "paper_value", "our_value", "delta_our_minus_paper"], vs_rows)

    summary_md = os.path.join(args.output_dir, "cub_reproduction_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# CUB Reproduction Summary\n\n")
        f.write(f"- With concepts ({args.with_concepts_label}): {with_acc:.4f}\n")
        f.write(f"- Baseline ({args.baseline_label}): {base_acc:.4f}\n")
        f.write(f"- Our gain (with - baseline): {our_gain:.4f}\n")
        f.write(f"- Paper gain (88.0 - 76.9): {paper_gain:.4f}\n")

    metadata = {
        "generated_at": ts,
        "device": str(device),
        "with_concepts_checkpoint": args.with_concepts_ckpt,
        "baseline_checkpoint": args.baseline_ckpt,
        "with_concepts_accuracy": with_acc,
        "baseline_accuracy": base_acc,
        "paper_with_concepts": paper_with,
        "paper_without_concepts": paper_without,
        "paper_gain": paper_gain,
        "our_gain": our_gain,
    }

    if args.save_examples:
        print("Collecting examples from with-concepts model...")
        records = collect_predictions(with_model, with_loader, device)
        correct = next((r for r in records if r["correct"]), None)
        wrong = next((r for r in records if not r["correct"]), None)

        attr_names = load_attribute_names(args.data_dir)

        examples_meta = {}
        if correct is not None:
            path = os.path.join(examples_dir, "correct_example.png")
            examples_meta["correct_example"] = render_example(
                correct,
                with_dm,
                path,
                attr_names,
                top_global_k=args.top_global_k,
                top_spatial_k=args.top_spatial_k,
            )

        if wrong is not None:
            path = os.path.join(examples_dir, "error_example.png")
            examples_meta["error_example"] = render_example(
                wrong,
                with_dm,
                path,
                attr_names,
                top_global_k=args.top_global_k,
                top_spatial_k=args.top_spatial_k,
            )

        meta_path = os.path.join(examples_dir, "examples_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(examples_meta, f, indent=2)

        metadata["examples_metadata"] = meta_path

    run_meta_path = os.path.join(args.output_dir, "cub_reproduction_metadata.json")
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"- {acc_csv}")
    print(f"- {acc_json}")
    print(f"- {vs_csv}")
    print(f"- {summary_md}")
    if args.save_examples:
        print(f"- {examples_dir}")


if __name__ == "__main__":
    main()
