#!/usr/bin/env python3
"""Generate figures from a trained checkpoint (MNIST odd/even) and save PNGs.

Usage:
  python scripts/generate_figures.py --run_path ./mnist_ctc/ExplanationMNIST_expl5.0/ \
      --output_dir /workspace/data/figs --logdir ./logs

  python scripts/generate_figures.py --skip_examples \
      --scaling_json experiments/binary_mnist_scaling_ES.json \
      --output_dir /workspace/data/figs
"""
import os
import argparse
import logging
import json

import torch
import matplotlib.pyplot as plt

from ctc import load_exp
from viz_utils import batch_predict_results, plot_explanation


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def plot_prediction(results, data_module, idx, outpath):
    img = data_module.mnist_test[idx][0].squeeze()

    predict_labs = {0: 'even', 1: 'odd'}
    correct_labs = {0: 'wrong', 1: 'correct'}

    predict = predict_labs[int(results['preds'][idx].item())]
    correct = correct_labs[int(results['correct'][idx].item())]

    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot(121)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(1,-1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(1,-1), ax3)
    ax3.set_title('concept attention scores')

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_tensorboard_scalars(logdir, output_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as e:
        logging.warning('Could not import tensorboard event accumulator: %s', e)
        return

    if not os.path.isdir(logdir):
        logging.warning('Logdir %s not found, skipping scalar plots', logdir)
        return

    logging.info('Loading TensorBoard logs from %s', logdir)
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    logging.info('Found %d scalar tags', len(tags))
    for tag in ('val_acc','val_expl_loss'):
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            plt.figure()
            plt.plot(steps, vals, '-o')
            plt.xlabel('step')
            plt.ylabel(tag)
            plt.title(tag)
            outpath = os.path.join(output_dir, f'{tag}.png')
            plt.savefig(outpath)
            plt.close()
            logging.info('Wrote scalar plot: %s', outpath)
        else:
            logging.info('Tag %s not found in logs', tag)


def _collect_records(obj, out):
    if isinstance(obj, dict):
        if 'n_train_samples' in obj:
            out.append(obj)
        for v in obj.values():
            _collect_records(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_records(v, out)


def plot_scaling_from_json(json_path, output_dir, prefix='figure2', expl_lambdas=None):
    if not os.path.isfile(json_path):
        logging.warning('Scaling JSON not found: %s', json_path)
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    records = []
    _collect_records(data, records)

    points = []
    used_test_expl = False
    for rec in records:
        n = rec.get('n_train_samples')
        if n is None:
            continue
        expl_lambda = rec.get('expl_lambda')
        test_acc = rec.get('test_acc') or rec.get('test_accuracy')
        val_expl = rec.get('val_expl_loss')
        test_expl = rec.get('test_expl_loss')
        if val_expl is None and test_expl is not None:
            val_expl = test_expl
            used_test_expl = True
        if test_acc is None and val_expl is None:
            continue
        points.append(
            {
                'n': int(n),
                'expl_lambda': expl_lambda,
                'test_acc': test_acc,
                'val_expl_loss': val_expl,
            }
        )

    if not points:
        logging.warning('No scaling records with n_train_samples found in %s', json_path)
        return

    if used_test_expl:
        logging.warning('val_expl_loss not found; using test_expl_loss as a proxy')

    # Aggregate by expl_lambda then n_train_samples
    by_lambda = {}
    for p in points:
        lam = p.get('expl_lambda')
        by_lambda.setdefault(lam, {})
        by_lambda[lam].setdefault(p['n'], {'test_acc': [], 'val_expl_loss': []})
        if p['test_acc'] is not None:
            by_lambda[lam][p['n']]['test_acc'].append(p['test_acc'])
        if p['val_expl_loss'] is not None:
            by_lambda[lam][p['n']]['val_expl_loss'].append(p['val_expl_loss'])

    if expl_lambdas is None:
        expl_lambdas = sorted(by_lambda.keys(), key=lambda v: (v is None, v))

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    def _std(vals, mean):
        if not vals or mean is None:
            return None
        return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

    # Plot combined figure with two panels and one curve per expl_lambda
    fig, (ax_acc, ax_expl) = plt.subplots(1, 2, figsize=(10, 4))
    plotted_acc = False
    plotted_expl = False

    for lam in expl_lambdas:
        if lam not in by_lambda:
            logging.warning('expl_lambda %s not found in scaling records', lam)
            continue

        ns = sorted(by_lambda[lam].keys())

        acc_ns = []
        acc_means = []
        acc_stds = []
        expl_ns = []
        expl_means = []
        expl_stds = []

        for n in ns:
            acc_vals = by_lambda[lam][n]['test_acc']
            if acc_vals:
                mean = _mean(acc_vals)
                acc_ns.append(n)
                acc_means.append(mean)
                acc_stds.append(_std(acc_vals, mean))

            expl_vals = by_lambda[lam][n]['val_expl_loss']
            if expl_vals:
                mean = _mean(expl_vals)
                expl_ns.append(n)
                expl_means.append(mean)
                expl_stds.append(_std(expl_vals, mean))

        label = f'$\\lambda_{{expl}}$={lam}' if lam is not None else 'lambda_expl=unknown'

        if acc_means:
            ax_acc.errorbar(acc_ns, acc_means, yerr=acc_stds, fmt='-o', capsize=3, label=label)
            plotted_acc = True

        if expl_means:
            ax_expl.errorbar(expl_ns, expl_means, yerr=expl_stds, fmt='-o', capsize=3, label=label)
            plotted_expl = True

    if plotted_acc:
        ax_acc.set_xlabel('N training samples')
        ax_acc.set_ylabel('test accuracy')
        ax_acc.set_title('Test accuracy vs N training samples')
        ax_acc.legend()
    else:
        logging.warning('No test_acc values found for scaling plot')

    if plotted_expl:
        ax_expl.set_xlabel('N training samples')
        ax_expl.set_ylabel('val explanation loss')
        ax_expl.set_title('Val explanation loss vs N training samples')
        ax_expl.legend()
    else:
        logging.warning('No val_expl_loss values found for scaling plot')

    if plotted_acc or plotted_expl:
        fig.tight_layout()
        outpath = os.path.join(output_dir, f'{prefix}.png')
        fig.savefig(outpath)
        plt.close(fig)
        logging.info('Wrote scaling plot: %s', outpath)


def _parse_scaling_lambdas(raw):
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    return [float(p) for p in parts] if parts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', default=None, help='Checkpoint file or directory containing ckpt')
    parser.add_argument('--output_dir', default='./output_figs', help='Directory to write PNGs')
    parser.add_argument('--logdir', default='./logs', help='TensorBoard logdir to extract scalars from')
    parser.add_argument('--num_correct', type=int, default=3, help='Number of correct examples to save')
    parser.add_argument('--num_wrong', type=int, default=3, help='Number of wrong examples to save')
    parser.add_argument('--skip_examples', action='store_true', help='Skip example image plots')
    parser.add_argument('--scaling_json', default=None, help='Path to epyc scaling JSON output')
    parser.add_argument('--scaling_prefix', default='figure2', help='Prefix for scaling plot filenames')
    parser.add_argument(
        '--scaling_lambdas',
        default=None,
        help='Comma-separated expl_lambda values to plot (default: all found in JSON)',
    )
    args = parser.parse_args()

    setup_logger()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info('Output directory: %s', args.output_dir)

    if not args.skip_examples:
        if not args.run_path:
            parser.error('--run_path is required unless --skip_examples is set')
        logging.info('Loading experiment from %s', args.run_path)
        model, data_module = load_exp(args.run_path)

        try:
            test_len = len(data_module.mnist_test)
            logging.info('MNIST test set size: %d', test_len)
        except Exception:
            logging.info('MNIST test set size: unknown')

        logging.info('Running model prediction on test set')
        try:
            from pytorch_lightning import Trainer
            results = batch_predict_results(Trainer().predict(model, data_module))
        except Exception as e:
            logging.error('Prediction failed: %s', e)
            raise

        try:
            n_correct = int((results['correct'] == 1).sum().item())
            n_wrong = int((results['correct'] == 0).sum().item())
            logging.info('Prediction results: %d correct, %d wrong', n_correct, n_wrong)
        except Exception:
            logging.info('Prediction results: summary unavailable')

        # Save some correct predictions
        correct_inds = (results['correct'] == 1).nonzero(as_tuple=False).squeeze()
        if isinstance(correct_inds, torch.Tensor) and correct_inds.numel() == 0:
            logging.warning('No correct predictions found')
        else:
            if isinstance(correct_inds, torch.Tensor) and correct_inds.dim() == 0:
                correct_inds = correct_inds.unsqueeze(0)
            n = min(args.num_correct, correct_inds.numel())
            logging.info('Saving %d correct examples', n)
            for i in range(n):
                idx = int(correct_inds[i].item())
                outpath = os.path.join(args.output_dir, f'correct_{i}_idx{idx}.png')
                plot_prediction(results, data_module, idx, outpath)
                logging.info('Saved correct example: %s', outpath)

        # Save some wrong predictions
        wrong_inds = (results['correct'] == 0).nonzero(as_tuple=False).squeeze()
        if isinstance(wrong_inds, torch.Tensor) and wrong_inds.numel() == 0:
            logging.warning('No wrong predictions found')
        else:
            if isinstance(wrong_inds, torch.Tensor) and wrong_inds.dim() == 0:
                wrong_inds = wrong_inds.unsqueeze(0)
            n = min(args.num_wrong, wrong_inds.numel())
            logging.info('Saving %d wrong examples', n)
            for i in range(n):
                idx = int(wrong_inds[i].item())
                outpath = os.path.join(args.output_dir, f'wrong_{i}_idx{idx}.png')
                plot_prediction(results, data_module, idx, outpath)
                logging.info('Saved wrong example: %s', outpath)

        # Plot TensorBoard scalars if available
        plot_tensorboard_scalars(args.logdir, args.output_dir)

    if args.scaling_json:
        logging.info('Plotting scaling results from %s', args.scaling_json)
        scaling_lambdas = _parse_scaling_lambdas(args.scaling_lambdas)
        plot_scaling_from_json(
            args.scaling_json,
            args.output_dir,
            prefix=args.scaling_prefix,
            expl_lambdas=scaling_lambdas,
        )
    else:
        logging.info('No scaling JSON provided; skipping Figure 2 plots')

    logging.info('Done')


if __name__ == '__main__':
    main()
