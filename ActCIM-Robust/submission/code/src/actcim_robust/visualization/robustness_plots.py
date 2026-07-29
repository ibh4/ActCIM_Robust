from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .common import (
    COLORS,
    LINE_STYLES,
    MARKERS,
    setup_plot_style,
    save_figure,
)


def plot_accuracy_vs_alpha(sweep_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sweep_results, dict):
        results_list = sweep_results.get("results", [])
    else:
        results_list = sweep_results

    alphas = [r["alpha"] for r in results_list]
    accs = [r["accuracy"] for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, accs, marker="o", color=COLORS["blue"], linewidth=2, markersize=7)
    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7, label="Clean (alpha=0)")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Nonlinearity Strength (Alpha)")
    ax.legend()
    save_figure(fig, output_dir / "accuracy_vs_alpha")


def plot_accuracy_drop_vs_alpha(sweep_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sweep_results, dict):
        results_list = sweep_results.get("results", [])
        clean_acc = sweep_results.get("clean_accuracy", results_list[0]["accuracy"] if results_list else 0)
    else:
        results_list = sweep_results
        clean_idx = next((i for i, r in enumerate(results_list) if r["alpha"] == 0), 0)
        clean_acc = results_list[clean_idx]["accuracy"] if results_list else 0

    alphas = [r["alpha"] for r in results_list]
    drops = [clean_acc - r["accuracy"] for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, drops, marker="s", color=COLORS["red"], linewidth=2, markersize=7)
    ax.axhline(y=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Accuracy Drop vs Alpha")
    save_figure(fig, output_dir / "accuracy_drop_vs_alpha")


def plot_loss_vs_alpha(sweep_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sweep_results, dict):
        results_list = sweep_results.get("results", [])
    else:
        results_list = sweep_results

    alphas = [r["alpha"] for r in results_list]
    losses = [r["loss"] for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, losses, marker="D", color=COLORS["orange"], linewidth=2, markersize=7)
    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss vs Alpha")
    save_figure(fig, output_dir / "loss_vs_alpha")


def plot_ece_vs_alpha(sweep_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sweep_results, dict):
        results_list = sweep_results.get("results", [])
    else:
        results_list = sweep_results

    alphas = [r["alpha"] for r in results_list]
    eces = [r.get("ece", 0) for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, eces, marker="v", color=COLORS["purple"], linewidth=2, markersize=7)
    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("ECE vs Alpha")
    save_figure(fig, output_dir / "ece_vs_alpha")


def plot_confidence_vs_alpha(sweep_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sweep_results, dict):
        results_list = sweep_results.get("results", [])
    else:
        results_list = sweep_results

    alphas = [r["alpha"] for r in results_list]
    confs = [r.get("mean_confidence", 0) for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, confs, marker="p", color=COLORS["green"], linewidth=2, markersize=7)
    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Mean Confidence vs Alpha")
    save_figure(fig, output_dir / "confidence_vs_alpha")


def plot_method_comparison(all_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method_name, sweep_data) in enumerate(all_results.items()):
        color = list(COLORS.values())[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]

        if isinstance(sweep_data, dict):
            results_list = sweep_data.get("results", [])
        else:
            results_list = sweep_data

        alphas = [r["alpha"] for r in results_list]
        accs = [r["accuracy"] for r in results_list]
        ax.plot(alphas, accs, marker=marker, color=color, linestyle=linestyle,
                linewidth=2, markersize=7, label=method_name)

    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Method Comparison: Accuracy vs Alpha")
    ax.legend()
    save_figure(fig, output_dir / "method_comparison")


def plot_worst_case_comparison(all_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = []
    worst_accs = []
    for method_name, sweep_data in all_results.items():
        if isinstance(sweep_data, dict):
            results_list = sweep_data.get("results", [])
        else:
            results_list = sweep_data
        if results_list:
            methods.append(method_name)
            worst_accs.append(min(r["accuracy"] for r in results_list))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, worst_accs, color=[COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]][:len(methods)])
    for bar, val in zip(bars, worst_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Worst-Case Accuracy")
    ax.set_title("Worst-Case Accuracy Comparison")
    save_figure(fig, output_dir / "worst_case_comparison")


def plot_aurc_comparison(all_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ..evaluation.robustness_metrics import compute_aurc

    methods = []
    aurc_values = []
    for method_name, sweep_data in all_results.items():
        if isinstance(sweep_data, dict):
            results_list = sweep_data.get("results", [])
        else:
            results_list = sweep_data
        accs = [r["accuracy"] for r in results_list]
        alphas = [r["alpha"] for r in results_list]
        methods.append(method_name)
        aurc_values.append(compute_aurc(accs, alphas))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, aurc_values, color=[COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]][:len(methods)])
    for bar, val in zip(bars, aurc_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Area Under Robustness Curve")
    ax.set_title("AURC Comparison")
    save_figure(fig, output_dir / "aurc_comparison")


def plot_positive_negative_asymmetry(sweep_results, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sweep_results, dict):
        results_list = sweep_results.get("results", [])
    else:
        results_list = sweep_results

    pos_data = [(r["alpha"], r["accuracy"]) for r in results_list if r["alpha"] > 0]
    neg_data = [(r["alpha"], r["accuracy"]) for r in results_list if r["alpha"] < 0]

    fig, ax = plt.subplots(figsize=(8, 5))

    if neg_data:
        neg_alphas = [a for a, _ in neg_data]
        neg_accs = [v for _, v in neg_data]
        ax.plot([abs(a) for a in neg_alphas], neg_accs, marker="o", color=COLORS["red"],
                linewidth=2, markersize=7, label="Negative Alpha")

    if pos_data:
        pos_alphas = [a for a, _ in pos_data]
        pos_accs = [v for _, v in pos_data]
        ax.plot(pos_alphas, pos_accs, marker="s", color=COLORS["blue"],
                linewidth=2, markersize=7, label="Positive Alpha")

    ax.set_xlabel("|Alpha|")
    ax.set_ylabel("Accuracy")
    ax.set_title("Positive vs Negative Alpha Asymmetry")
    ax.legend()
    save_figure(fig, output_dir / "positive_negative_asymmetry")
