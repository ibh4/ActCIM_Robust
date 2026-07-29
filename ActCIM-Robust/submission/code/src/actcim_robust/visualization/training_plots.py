from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .common import COLORS, LINE_STYLES, MARKERS, setup_plot_style, save_figure


def plot_training_curves(metrics, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(metrics, dict):
        train_losses = metrics.get("train_loss", [])
        val_losses = metrics.get("val_loss", [])
        val_accs = metrics.get("val_accuracy", [])
    else:
        train_losses = [m.get("train_loss", 0) for m in metrics]
        val_losses = [m.get("val_loss", 0) for m in metrics]
        val_accs = [m.get("val_accuracy", 0) for m in metrics]

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, color=COLORS["blue"], linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_losses, color=COLORS["red"], linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    if val_accs:
        ax2.plot(epochs, val_accs, color=COLORS["green"], linewidth=2, label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy")
        ax2.legend()

    save_figure(fig, output_dir / "training_curves")


def plot_method_robustness_curve(all_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (method_name, sweep_data) in enumerate(all_results.items()):
        color = list(COLORS.values())[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        if isinstance(sweep_data, dict):
            results_list = sweep_data.get("results", [])
        else:
            results_list = sweep_data

        alphas = [r["alpha"] for r in results_list]
        accs = [r["accuracy"] for r in results_list]

        ax1.plot(alphas, accs, marker=marker, color=color, linewidth=2, markersize=6, label=method_name)

        if results_list:
            clean_alphas = [a for a in alphas if a == 0]
            clean_acc = results_list[0]["accuracy"] if clean_alphas else 0
            drops = [max(0, clean_acc - a) for a in accs] if clean_acc else [1 - a for a in accs]
        else:
            clean_acc = 0
            drops = []
        ax2.plot(alphas, drops, marker=marker, color=color, linewidth=2, markersize=6, label=method_name)

    ax1.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Robustness Curves")
    ax1.legend()

    ax2.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Accuracy Drop")
    ax2.set_title("Robustness Degradation")
    ax2.legend()

    save_figure(fig, output_dir / "method_robustness_curve")
