#!/usr/bin/env python3
"""Rebuild scaling JSON from existing checkpoints without retraining."""
import argparse
import json
import os
import re

import torch
import pytorch_lightning as pl

import data
from ctc.ctc_model import CTCModel


def _parse_run_dir(name):
    match = re.match(r"^expl(?P<expl>[-\d\.]+)_N(?P<n>\d+)$", name)
    if not match:
        return None
    return float(match.group("expl")), int(match.group("n"))


def _find_checkpoint(run_dir):
    preferred = os.path.join(run_dir, "binary_mnist_best_ckpt.ckpt")
    if os.path.isfile(preferred):
        return preferred

    ckpts = [
        os.path.join(run_dir, f)
        for f in os.listdir(run_dir)
        if f.endswith(".ckpt")
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]


def _get_trainer():
    if torch.cuda.is_available():
        return pl.Trainer(
            accelerator="gpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )


def _run_test(ckpt_path):
    model = CTCModel.load_from_checkpoint(ckpt_path)
    data_module = getattr(data, model.hparams.data_name)(**model.hparams)
    data_module.prepare_data()
    data_module.setup()

    trainer = _get_trainer()
    results = trainer.test(model, datamodule=data_module, ckpt_path=None)
    return results[0] if results else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_root",
        default="./binary_mnist_scaling_ES",
        help="Directory containing expl*_N* run folders",
    )
    parser.add_argument(
        "--output_json",
        default="./experiments/binary_mnist_scaling_ES.json",
        help="Path to write the reconstructed JSON",
    )
    args = parser.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    output_json = os.path.abspath(args.output_json)

    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"Run root not found: {runs_root}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    records = []
    for name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue

        parsed = _parse_run_dir(name)
        if not parsed:
            continue

        expl_lambda, n_train_samples = parsed
        ckpt = _find_checkpoint(run_dir)
        if not ckpt:
            continue

        metrics = _run_test(ckpt)
        records.append(
            {
                "expl_lambda": expl_lambda,
                "n_train_samples": n_train_samples,
                "test_acc": metrics.get("test_acc"),
                "test_expl_loss": metrics.get("test_expl_loss"),
                "test_ce_loss": metrics.get("test_ce_loss"),
                "test_l1_loss": metrics.get("test_l1_loss"),
            }
        )

    payload = {
        "project_name": "binary_mnist_scaling_ES",
        "records": records,
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(records)} records to {output_json}")


if __name__ == "__main__":
    main()
