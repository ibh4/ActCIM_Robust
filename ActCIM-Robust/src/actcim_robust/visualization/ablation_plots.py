from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .common import COLORS, LINE_STYLES, MARKERS, setup_plot_style, save_figure


def plot_ablation_results(ablation_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(ablation_df, "iterrows"):
        variants = ablation_df.get("variant", ablation_df.get("method", [])).tolist()
        accuracies = ablation_df.get("accuracy", ablation_df.get("clean_accuracy", [])).tolist()
        nonlinear_accs = ablation_df.get("nonlinear_accuracy", ablation_df.get("worst_accuracy", [])).tolist()
    else:
        variants = [r.get("variant", r.get("method", f"V{i}")) for i, r in enumerate(ablation_df)]
        accuracies = [r.get("accuracy", r.get("clean_accuracy", 0)) for r in ablation_df]
        nonlinear_accs = [r.get("nonlinear_accuracy", r.get("worst_accuracy", 0)) for r in ablation_df]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax.bar(x - width / 2, accuracies, width, label="Clean Accuracy",
                   color=COLORS["blue"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, nonlinear_accs, width, label="Under Nonlinearity",
                   color=COLORS["red"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study Results")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    save_figure(fig, output_dir / "ablation_results")


def plot_ablation_robustness_curve(ablation_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(ablation_df, "iterrows"):
        for i, (_, row) in enumerate(ablation_df.iterrows()):
            variant = row.get("variant", row.get("method", f"V{i}"))
            if "alphas" in row and "accuracies" in row:
                alphas = row["alphas"]
                accs = row["accuracies"]
                color = list(COLORS.values())[i % len(COLORS)]
                marker = MARKERS[i % len(MARKERS)]
                ax.plot(alphas, accs, marker=marker, color=color, linewidth=2,
                        markersize=6, label=variant)
    else:
        for i, row in enumerate(ablation_df):
            variant = row.get("variant", row.get("method", f"V{i}"))
            color = list(COLORS.values())[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]

            if "alphas" in row and "accuracies" in row:
                ax.plot(row["alphas"], row["accuracies"], marker=marker, color=color,
                        linewidth=2, markersize=6, label=variant)
            elif "results" in row:
                results = row["results"]
                alphas = [r["alpha"] for r in results]
                accs = [r["accuracy"] for r in results]
                ax.plot(alphas, accs, marker=marker, color=color, linewidth=2,
                        markersize=6, label=variant)

    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study: Robustness Curves")
    ax.legend()
    save_figure(fig, output_dir / "ablation_robustness_curve")
