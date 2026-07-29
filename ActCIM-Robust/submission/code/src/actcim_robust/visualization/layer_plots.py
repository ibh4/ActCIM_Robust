from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .common import COLORS, LINE_STYLES, MARKERS, setup_plot_style, save_figure


def plot_layer_sensitivity_bar(sensitivity_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(sensitivity_df, "sort_values"):
        df = sensitivity_df.sort_values("sensitivity_score", ascending=False).head(20)
        names = df["layer_name"].tolist()
        scores = df["sensitivity_score"].tolist()
    else:
        rows = sorted(sensitivity_df, key=lambda r: r["sensitivity_score"], reverse=True)[:20]
        names = [r["layer_name"] for r in rows]
        scores = [r["sensitivity_score"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, scores, color=COLORS["blue"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Sensitivity Score")
    ax.set_title("Layer Sensitivity Ranking (Top 20)")
    save_figure(fig, output_dir / "layer_sensitivity_bar")


def plot_layer_sensitivity_heatmap(sensitivity_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(sensitivity_df, "pivot"):
        conv_layers = [c for c in sensitivity_df.columns if "accuracy_drop" in c and "conv" in c.lower()
                       or "features" in c.lower() or "layer" in c.lower()]
    else:
        conv_layers = None

    fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(sensitivity_df, "sort_values"):
        df = sensitivity_df.copy()
        df = df.sort_values("sensitivity_score", ascending=True)
        cols = ["neg_04_accuracy_drop", "pos_04_accuracy_drop", "sensitivity_score"]
        cols = [c for c in cols if c in df.columns]
        heatmap_data = df[cols].head(30)
    else:
        rows = sorted(sensitivity_df, key=lambda r: r["sensitivity_score"])[:30]
        raw_data = {}
        for col in ["neg_04_accuracy_drop", "pos_04_accuracy_drop", "sensitivity_score"]:
            raw_data[col] = [r.get(col, 0) for r in rows]
        labels = [r["layer_name"] for r in rows]
        heatmap_data = np.column_stack([raw_data[col] for col in raw_data.keys()])
        cols = list(raw_data.keys())

    im = ax.imshow(heatmap_data if isinstance(heatmap_data, np.ndarray) else heatmap_data.values,
                   aspect="auto", cmap="RdYlBu_r")
    if not isinstance(heatmap_data, np.ndarray):
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    else:
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    plt.colorbar(im, ax=ax)
    ax.set_title("Layer Sensitivity Heatmap")
    save_figure(fig, output_dir / "layer_sensitivity_heatmap")


def plot_sensitivity_vs_depth(sensitivity_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(sensitivity_df, "sort_values"):
        df = sensitivity_df.sort_values("layer_index")
        depths = df["layer_index"].tolist()
        scores = df["sensitivity_score"].tolist()
    else:
        rows = sorted(sensitivity_df, key=lambda r: r["layer_index"])
        depths = [r["layer_index"] for r in rows]
        scores = [r["sensitivity_score"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, scores, marker="o", color=COLORS["blue"], linewidth=2, markersize=6)
    ax.set_xlabel("Layer Index (Depth)")
    ax.set_ylabel("Sensitivity Score")
    ax.set_title("Sensitivity vs Depth")
    save_figure(fig, output_dir / "sensitivity_vs_depth")


def plot_sensitivity_vs_param_count(sensitivity_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(sensitivity_df, "sort_values"):
        param_counts = sensitivity_df["param_count"].tolist()
        scores = sensitivity_df["sensitivity_score"].tolist()
    else:
        param_counts = [r["param_count"] for r in sensitivity_df]
        scores = [r["sensitivity_score"] for r in sensitivity_df]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(param_counts, scores, c=COLORS["blue"], s=60, alpha=0.7)
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Sensitivity Score")
    ax.set_title("Sensitivity vs Parameter Count")
    ax.set_xscale("log")
    save_figure(fig, output_dir / "sensitivity_vs_param_count")


def plot_positive_negative_layer_gap(sensitivity_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(sensitivity_df, "sort_values"):
        df = sensitivity_df.copy()
        if "pos_04_accuracy_drop" in df.columns and "neg_04_accuracy_drop" in df.columns:
            df["gap"] = abs(df["pos_04_accuracy_drop"] - df["neg_04_accuracy_drop"])
            df = df.sort_values("gap", ascending=False).head(20)
            names = df["layer_name"].tolist()
            pos_drops = df["pos_04_accuracy_drop"].tolist()
            neg_drops = df["neg_04_accuracy_drop"].tolist()
        else:
            names = []
            pos_drops = []
            neg_drops = []
    else:
        rows = sorted(sensitivity_df, key=lambda r: abs(
            r.get("pos_04_accuracy_drop", 0) - r.get("neg_04_accuracy_drop", 0)), reverse=True)[:20]
        names = [r["layer_name"] for r in rows]
        pos_drops = [r.get("pos_04_accuracy_drop", 0) for r in rows]
        neg_drops = [r.get("neg_04_accuracy_drop", 0) for r in rows]

    if not names:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.barh(x + width / 2, pos_drops, width, label="Alpha +0.4", color=COLORS["red"], alpha=0.8)
    ax.barh(x - width / 2, neg_drops, width, label="Alpha -0.4", color=COLORS["blue"], alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy Drop")
    ax.set_title("Positive vs Negative Alpha: Per-Layer Accuracy Drop")
    ax.legend()
    save_figure(fig, output_dir / "positive_negative_layer_gap")


def plot_layer_error_accumulation(error_df, alpha_sign, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(error_df, "query"):
        filtered = error_df[error_df["alpha_sign"] == alpha_sign]
        layers = filtered["layer_name"].tolist()
        errors = filtered["relative_l2"].tolist()
    else:
        filtered = [r for r in error_df if r["alpha_sign"] == alpha_sign]
        layers = [r["layer_name"] for r in filtered]
        errors = [r["relative_l2"] for r in filtered]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(layers)), errors, marker="o", color=COLORS["red"] if "neg" in alpha_sign else COLORS["blue"],
            linewidth=2, markersize=6)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title(f"Error Accumulation Across Layers ({alpha_sign})")
    save_figure(fig, output_dir / f"layer_error_accumulation_{alpha_sign}")


def plot_layer_cosine_similarity(error_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(error_df, "query"):
        neg = error_df[error_df["alpha_sign"] == "neg_04"]
        pos = error_df[error_df["alpha_sign"] == "pos_04"]
        layers_neg = neg["layer_name"].tolist()
        cos_neg = neg["cosine_similarity"].tolist()
        layers_pos = pos["layer_name"].tolist()
        cos_pos = pos["cosine_similarity"].tolist()
    else:
        neg = [r for r in error_df if r["alpha_sign"] == "neg_04"]
        pos = [r for r in error_df if r["alpha_sign"] == "pos_04"]
        layers_neg = [r["layer_name"] for r in neg]
        cos_neg = [r["cosine_similarity"] for r in neg]
        layers_pos = [r["layer_name"] for r in pos]
        cos_pos = [r["cosine_similarity"] for r in pos]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(layers_neg)), cos_neg, marker="o", color=COLORS["red"],
            linewidth=2, markersize=6, label="Alpha -0.4")
    ax.plot(range(len(layers_pos)), cos_pos, marker="s", color=COLORS["blue"],
            linewidth=2, markersize=6, label="Alpha +0.4")
    ax.set_xticks(range(len(layers_neg)))
    ax.set_xticklabels(layers_neg, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity Across Layers")
    ax.legend()
    save_figure(fig, output_dir / "layer_cosine_similarity")


def plot_layer_mean_std_shift(error_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if hasattr(error_df, "query"):
        neg = error_df[error_df["alpha_sign"] == "neg_04"]
        pos = error_df[error_df["alpha_sign"] == "pos_04"]
        layers = neg["layer_name"].tolist()
        mean_shift_neg = neg["mean_shift"].tolist()
        mean_shift_pos = pos["mean_shift"].tolist()
        std_ratio_neg = neg["std_ratio"].tolist()
        std_ratio_pos = pos["std_ratio"].tolist()
    else:
        neg = [r for r in error_df if r["alpha_sign"] == "neg_04"]
        pos = [r for r in error_df if r["alpha_sign"] == "pos_04"]
        layers = [r["layer_name"] for r in neg]
        mean_shift_neg = [r["mean_shift"] for r in neg]
        mean_shift_pos = [r["mean_shift"] for r in pos]
        std_ratio_neg = [r["std_ratio"] for r in neg]
        std_ratio_pos = [r["std_ratio"] for r in pos]

    ax1.plot(range(len(layers)), mean_shift_neg, marker="o", color=COLORS["red"],
             linewidth=2, markersize=6, label="Alpha -0.4")
    ax1.plot(range(len(layers)), mean_shift_pos, marker="s", color=COLORS["blue"],
             linewidth=2, markersize=6, label="Alpha +0.4")
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Shift")
    ax1.set_title("Mean Shift Across Layers")
    ax1.legend()

    ax2.plot(range(len(layers)), std_ratio_neg, marker="o", color=COLORS["red"],
             linewidth=2, markersize=6, label="Alpha -0.4")
    ax2.plot(range(len(layers)), std_ratio_pos, marker="s", color=COLORS["blue"],
             linewidth=2, markersize=6, label="Alpha +0.4")
    ax2.axhline(y=1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Std Ratio")
    ax2.set_title("Std Ratio Across Layers")
    ax2.legend()

    save_figure(fig, output_dir / "layer_mean_std_shift")


def plot_activation_distribution_shift(error_df, output_dir):
    setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(error_df, "query"):
        neg = error_df[error_df["alpha_sign"] == "neg_04"]
        pos = error_df[error_df["alpha_sign"] == "pos_04"]
        layers = neg["layer_name"].tolist()
        sign_flip_neg = neg["sign_flip_ratio"].tolist()
        sign_flip_pos = pos["sign_flip_ratio"].tolist()
        sat_neg = neg["saturation_ratio"].tolist()
        sat_pos = pos["saturation_ratio"].tolist()
    else:
        neg = [r for r in error_df if r["alpha_sign"] == "neg_04"]
        pos = [r for r in error_df if r["alpha_sign"] == "pos_04"]
        layers = [r["layer_name"] for r in neg]
        sign_flip_neg = [r["sign_flip_ratio"] for r in neg]
        sign_flip_pos = [r["sign_flip_ratio"] for r in pos]
        sat_neg = [r["saturation_ratio"] for r in neg]
        sat_pos = [r["saturation_ratio"] for r in pos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(range(len(layers)), sign_flip_neg, marker="o", color=COLORS["red"],
             linewidth=2, markersize=6, label="Alpha -0.4")
    ax1.plot(range(len(layers)), sign_flip_pos, marker="s", color=COLORS["blue"],
             linewidth=2, markersize=6, label="Alpha +0.4")
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Sign Flip Ratio")
    ax1.set_title("Sign Flip Ratio Across Layers")
    ax1.legend()

    ax2.plot(range(len(layers)), sat_neg, marker="o", color=COLORS["red"],
             linewidth=2, markersize=6, label="Alpha -0.4")
    ax2.plot(range(len(layers)), sat_pos, marker="s", color=COLORS["blue"],
             linewidth=2, markersize=6, label="Alpha +0.4")
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Saturation Ratio")
    ax2.set_title("Saturation Ratio Across Layers")
    ax2.legend()

    save_figure(fig, output_dir / "activation_distribution_shift")
