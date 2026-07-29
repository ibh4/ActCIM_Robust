from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOR_MAP = {
    "clean": "#2196F3",
    "random_nat": "#FF9800",
    "sgr_nat": "#4CAF50",
}

LABEL_MAP = {
    "clean": "Clean",
    "random_nat": "Random-NAT",
    "sgr_nat": "SGR-NAT",
}

METHOD_ORDER = ["clean", "random_nat", "sgr_nat"]


def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, path_stem):
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_stem}.pdf", dpi=300, bbox_inches="tight")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    grouped = {}
    for method in METHOD_ORDER:
        mdf = df[df["method"] == method].sort_values("alpha").reset_index(drop=True)
        if len(mdf) > 0:
            grouped[method] = mdf
    return grouped


def compute_aurc(alphas, accs):
    a = np.array(alphas)
    b = np.array(accs)
    idx = np.argsort(a)
    a_sorted = a[idx]
    b_sorted = b[idx]
    if len(a_sorted) < 2:
        return float(np.mean(b_sorted))
    area = np.trapezoid(b_sorted, a_sorted)
    width = a_sorted[-1] - a_sorted[0]
    if width == 0:
        return float(np.mean(b_sorted))
    return float(area / width)


def compute_aurc_positive(alphas, accs):
    pos = [(a, acc) for a, acc in zip(alphas, accs) if a >= 0]
    if len(pos) < 2:
        return float(np.mean([v for _, v in pos])) if pos else float("nan")
    a_pos = np.array([p[0] for p in pos])
    b_pos = np.array([p[1] for p in pos])
    idx = np.argsort(a_pos)
    a_sorted = a_pos[idx]
    b_sorted = b_pos[idx]
    area = np.trapezoid(b_sorted, a_sorted)
    width = a_sorted[-1] - a_sorted[0]
    return float(area / width) if width > 0 else float("nan")


# ─── Figure 1: Accuracy vs Alpha ─────────────────────────────────────────────
def figure_01_accuracy_vs_alpha(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        ax.plot(df["alpha"], df["test_accuracy"],
                marker="o", color=COLOR_MAP[method_name],
                label=LABEL_MAP[method_name])
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Alpha (Nonlinearity Strength)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Nonlinearity Strength")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, output_dir / "01_accuracy_vs_alpha_all_methods")
    plt.close(fig)


# ─── Figure 2: Accuracy Drop vs Alpha ────────────────────────────────────────
def figure_02_accuracy_drop_vs_alpha(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        ax.plot(df["alpha"], df["accuracy_drop_from_clean_reference"],
                marker="o", color=COLOR_MAP[method_name],
                label=LABEL_MAP[method_name])
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Alpha (Nonlinearity Strength)")
    ax.set_ylabel("Accuracy Drop from Clean Reference")
    ax.set_title("Accuracy Drop vs Nonlinearity Strength")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, output_dir / "02_accuracy_drop_vs_alpha_all_methods")
    plt.close(fig)


# ─── Figure 3: ECE vs Alpha ──────────────────────────────────────────────────
def figure_03_ece_vs_alpha(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        ax.plot(df["alpha"], df["ece_15_bins"],
                marker="o", color=COLOR_MAP[method_name],
                label=LABEL_MAP[method_name])
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Alpha (Nonlinearity Strength)")
    ax.set_ylabel("ECE (15 Bins)")
    ax.set_title("ECE vs Nonlinearity Strength")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, output_dir / "03_ece_vs_alpha_all_methods")
    plt.close(fig)


# ─── Figure 4: Confidence vs Alpha ───────────────────────────────────────────
def figure_04_confidence_vs_alpha(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        ax.plot(df["alpha"], df["mean_confidence"],
                marker="o", color=COLOR_MAP[method_name],
                label=LABEL_MAP[method_name])
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Alpha (Nonlinearity Strength)")
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Mean Confidence vs Nonlinearity Strength")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, output_dir / "04_confidence_vs_alpha_all_methods")
    plt.close(fig)


# ─── Figure 5: Worst-Case Accuracy Bar ───────────────────────────────────────
def figure_05_worst_case_accuracy(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = []
    worst_accs = []
    colors = []
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        row = df[df["alpha"] == 0.8]
        if len(row) == 0:
            continue
        methods.append(LABEL_MAP[method_name])
        worst_accs.append(float(row["test_accuracy"].iloc[0]))
        colors.append(COLOR_MAP[method_name])
    bars = ax.bar(methods, worst_accs, color=colors)
    for bar, val in zip(bars, worst_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Accuracy at Alpha = +0.8")
    ax.set_title("Worst-Case Accuracy Comparison")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    save_figure(fig, output_dir / "05_worst_case_accuracy_comparison")
    plt.close(fig)


# ─── Figure 6: AURC All Comparison ───────────────────────────────────────────
def figure_06_aurc_all(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = []
    aurc_vals = []
    colors = []
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        aurc = compute_aurc(df["alpha"].tolist(), df["test_accuracy"].tolist())
        methods.append(LABEL_MAP[method_name])
        aurc_vals.append(aurc)
        colors.append(COLOR_MAP[method_name])
    bars = ax.bar(methods, aurc_vals, color=colors)
    for bar, val in zip(bars, aurc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("AURC")
    ax.set_title("AURC Comparison (All Alphas)")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    save_figure(fig, output_dir / "06_aurc_all_comparison")
    plt.close(fig)


# ─── Figure 7: AURC Positive Comparison ──────────────────────────────────────
def figure_07_aurc_positive(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = []
    aurc_vals = []
    colors = []
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        aurc = compute_aurc_positive(df["alpha"].tolist(), df["test_accuracy"].tolist())
        methods.append(LABEL_MAP[method_name])
        aurc_vals.append(aurc)
        colors.append(COLOR_MAP[method_name])
    bars = ax.bar(methods, aurc_vals, color=colors)
    for bar, val in zip(bars, aurc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("AURC (Positive Alpha Only)")
    ax.set_title("AURC Comparison (Positive Alphas)")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    save_figure(fig, output_dir / "07_aurc_positive_comparison")
    plt.close(fig)


# ─── Figure 8: Clean Acc vs Worst Acc ────────────────────────────────────────
def figure_08_clean_vs_worst(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        clean_row = df[df["alpha"] == 0.0]
        worst_row = df[df["alpha"] == 0.8]
        if len(clean_row) == 0 or len(worst_row) == 0:
            continue
        clean_acc = float(clean_row["test_accuracy"].iloc[0])
        worst_acc = float(worst_row["test_accuracy"].iloc[0])
        ax.scatter(clean_acc, worst_acc, color=COLOR_MAP[method_name],
                   s=120, label=LABEL_MAP[method_name], zorder=5)
        ax.annotate(LABEL_MAP[method_name], (clean_acc, worst_acc),
                    textcoords="offset points", xytext=(10, 6), fontsize=11)
    min_val = ax.get_xlim()[0] if ax.get_xlim() else 0.80
    max_val = ax.get_xlim()[1] if ax.get_xlim() else 0.96
    ax.plot([min_val, max_val], [min_val, max_val],
            "k--", alpha=0.3, linewidth=1, label="Ideal (x=y)")
    ax.set_xlabel("Clean Accuracy (Alpha=0)")
    ax.set_ylabel("Worst-Case Accuracy (Alpha=+0.8)")
    ax.set_title("Clean Accuracy vs Worst-Case Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, output_dir / "08_clean_accuracy_vs_worst_accuracy")
    plt.close(fig)


# ─── Figure 9: Clean Acc vs AURC ─────────────────────────────────────────────
def figure_09_clean_vs_aurc(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in METHOD_ORDER:
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        clean_row = df[df["alpha"] == 0.0]
        if len(clean_row) == 0:
            continue
        clean_acc = float(clean_row["test_accuracy"].iloc[0])
        aurc = compute_aurc(df["alpha"].tolist(), df["test_accuracy"].tolist())
        ax.scatter(clean_acc, aurc, color=COLOR_MAP[method_name],
                   s=120, label=LABEL_MAP[method_name], zorder=5)
        ax.annotate(LABEL_MAP[method_name], (clean_acc, aurc),
                    textcoords="offset points", xytext=(10, 6), fontsize=11)
    ax.set_xlabel("Clean Accuracy (Alpha=0)")
    ax.set_ylabel("AURC")
    ax.set_title("Clean Accuracy vs AURC")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, output_dir / "09_clean_accuracy_vs_aurc")
    plt.close(fig)


# ─── Figure 10: Asymmetry Gap ────────────────────────────────────────────────
def figure_10_asymmetry_gap(methods_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    abs_alphas = [0.2, 0.4, 0.6, 0.8]
    n_methods = len(METHOD_ORDER)
    bar_width = 0.18
    x = np.arange(len(abs_alphas))

    for i, method_name in enumerate(METHOD_ORDER):
        if method_name not in methods_data:
            continue
        df = methods_data[method_name]
        gaps = []
        for alpha_abs in abs_alphas:
            row_pos = df[df["alpha"] == alpha_abs]
            row_neg = df[df["alpha"] == -alpha_abs]
            if len(row_pos) == 0 or len(row_neg) == 0:
                gaps.append(0)
                continue
            acc_pos = float(row_pos["test_accuracy"].iloc[0])
            acc_neg = float(row_neg["test_accuracy"].iloc[0])
            gaps.append(acc_neg - acc_pos)
        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(x + offset, gaps, bar_width,
                      color=COLOR_MAP[method_name], label=LABEL_MAP[method_name],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, gaps):
            label_y = bar.get_height() + 0.003 if bar.get_height() >= 0 else bar.get_height() - 0.008
            va = "bottom" if bar.get_height() >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f"{val:.3f}", ha="center", va=va, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"|alpha|={a}" for a in abs_alphas])
    ax.set_ylabel("Asymmetry Gap (Acc_{-a} - Acc_{+a})")
    ax.set_title("Positive-Negative Asymmetry Gap")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    save_figure(fig, output_dir / "10_asymmetry_gap_comparison")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    csv_path = Path(
        r"I:\比赛项目\存算一体高校挑战赛\ActCIM-Robust"
        r"\results\post_training\all_methods_alpha_sweep.csv"
    )
    output_dir = Path(
        r"I:\比赛项目\存算一体高校挑战赛\ActCIM-Robust"
        r"\results\figures\post_training"
    )

    setup_style()
    methods_data = load_data(csv_path)
    print(f"Loaded methods: {list(methods_data.keys())}")

    figure_01_accuracy_vs_alpha(methods_data, output_dir)
    print("  [1/10] Accuracy vs Alpha")
    figure_02_accuracy_drop_vs_alpha(methods_data, output_dir)
    print("  [2/10] Accuracy Drop vs Alpha")
    figure_03_ece_vs_alpha(methods_data, output_dir)
    print("  [3/10] ECE vs Alpha")
    figure_04_confidence_vs_alpha(methods_data, output_dir)
    print("  [4/10] Confidence vs Alpha")
    figure_05_worst_case_accuracy(methods_data, output_dir)
    print("  [5/10] Worst-Case Accuracy Bar")
    figure_06_aurc_all(methods_data, output_dir)
    print("  [6/10] AURC All Comparison")
    figure_07_aurc_positive(methods_data, output_dir)
    print("  [7/10] AURC Positive Comparison")
    figure_08_clean_vs_worst(methods_data, output_dir)
    print("  [8/10] Clean Acc vs Worst Acc")
    figure_09_clean_vs_aurc(methods_data, output_dir)
    print("  [9/10] Clean Acc vs AURC")
    figure_10_asymmetry_gap(methods_data, output_dir)
    print("  [10/10] Asymmetry Gap")

    output_files = sorted(output_dir.glob("*"))
    print(f"\nGenerated {len(output_files)} files:")
    for f in output_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
