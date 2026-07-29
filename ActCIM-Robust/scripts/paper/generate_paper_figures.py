"""Generate all paper figures (300 DPI PNG + PDF) from the project's real
result CSV/JSON files. Every figure embeds its number, bilingual title,
axis units, legend and a full caption (what it shows / how it is computed /
what conclusion it supports). A manifest (JSON + Markdown) is written too.

Data sources (all verified against checkpoints in the audit):
  results/post_training/all_methods_alpha_sweep.csv        (clean/random_nat/sgr_nat, seed 42)
  results/post_training/fixed_nat_alpha_sweep.csv          (fixed_nat, seed 42 - verified)
  results/post_training/multi_seed/fixed_nat_seed_*.csv    (seeds 2026/3407, descriptive)
  results/post_training/fixed_nat_comparison.json          (aggregate metrics)
  results/analysis/layer_sensitivity_ranked.csv
  results/analysis/layer_error_accumulation.csv
  results/post_training/calibration/*_bins.csv             (15-bin reliability data)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
POST = ROOT / "results" / "post_training"
ANALYSIS = ROOT / "results" / "analysis"
CALIB = POST / "calibration"
OUT = ROOT / "results" / "figures" / "paper_final"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- fonts
for fp in [
    "/System/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]:
    if Path(fp).exists():
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Songti SC", "STHeiti", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

METHOD_COLOR = {
    "clean": "#555555",
    "random_nat": "#1f77b4",
    "sgr_nat": "#2ca02c",
    "fixed_nat": "#d62728",
}
METHOD_LABEL = {
    "clean": "Clean 基线",
    "random_nat": "Random-NAT",
    "sgr_nat": "SGR-NAT",
    "fixed_nat": "Fixed-NAT (α=+0.4)",
}

MANIFEST: list[dict] = []


def save_figure(fig, name: str, number: int, title: str, caption: str,
                sources: list[str]) -> None:
    """Embed the numbered caption below the figure, save 300-DPI PNG + PDF."""
    wrapped = "\n".join(textwrap.wrap(f"图{number}  {caption}", width=62))
    n_lines = wrapped.count("\n") + 1
    w, h = fig.get_size_inches()
    cap_h = 0.155 * n_lines + 0.30
    fig.set_size_inches(w, h + cap_h)
    frac = cap_h / (h + cap_h)
    try:
        fig.tight_layout(rect=(0, frac, 1, 1))
    except Exception:
        pass
    fig.text(0.035, 0.012, wrapped, fontsize=7.6, ha="left", va="bottom",
             color="#222222")
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    MANIFEST.append({
        "number": number, "name": name, "title": title,
        "png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT)),
        "sources": sources, "caption": caption,
    })
    print(f"saved fig{number:02d}: {name}")


# ---------------------------------------------------------------- data
sweep = pd.read_csv(POST / "all_methods_alpha_sweep.csv")
fx = pd.read_csv(POST / "fixed_nat_alpha_sweep.csv")
comp = json.load(open(POST / "fixed_nat_comparison.json"))["methods"]
ranked = pd.read_csv(ANALYSIS / "layer_sensitivity_ranked.csv")
accum = pd.read_csv(ANALYSIS / "layer_error_accumulation.csv")
ms2026 = pd.read_csv(POST / "multi_seed" / "fixed_nat_seed_2026_alpha_sweep.csv")
ms3407 = pd.read_csv(POST / "multi_seed" / "fixed_nat_seed_3407_alpha_sweep.csv")


def method_curve(m: str) -> pd.DataFrame:
    if m == "fixed_nat":
        d = fx.sort_values("alpha")
    else:
        d = sweep[sweep["method"] == m].sort_values("alpha")
    return d


# ================================================================ fig 1
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)

    def box(x, y, w, h, text, fc, fs=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec="#333333", lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#111111")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.2, color="#333333"))

    box(0.15, 3.6, 1.9, 1.35, "CIFAR-10\n45k 训练 / 5k 验证\n10k 测试", "#dce9f7")
    box(2.55, 3.6, 2.2, 1.35, "ResNet-18-CIFAR\n基线训练 50 epoch\nval acc 94.84%", "#dce9f7")
    box(5.35, 3.6, 4.4, 1.35,
        "非线性注入框架  y = W·f_α(x)\nf_α(x)=m·[α(x/m)^3+(1-α)(x/m)],  m=max|x|\n"
        "实际生效 4 层: layer4.1.conv1/conv2,\nlayer4.0.downsample.0, fc（控制器键碰撞）",
        "#fde9d9", fs=7.8)
    box(0.15, 1.55, 3.1, 1.35,
        "Fixed-NAT:  α=+0.4 固定\nRandom-NAT: α~U(-0.8,0.8)\nSGR-NAT: 灵敏度引导+KL一致性",
        "#e2efda", fs=8.0)
    box(3.75, 1.55, 3.0, 1.35,
        "统一 α-Sweep 评估\nα∈{-0.8,…,+0.8} 11 点\n10 000 张全测试集", "#e2efda")
    box(7.25, 1.55, 2.55, 1.35,
        "指标\nWorst-Case Acc / AURC\nECE / 方向不对称 Gap", "#f2dcdb")
    box(2.0, 0.05, 6.0, 0.95,
        "核心结果: Fixed-NAT(+0.4) Worst 81.25%→91.79%, AURC 0.9283→0.9374, "
        "Clean 保持 94.02%", "#fff2cc", fs=8.4)

    arrow(2.05, 4.28, 2.55, 4.28)
    arrow(4.75, 4.28, 5.35, 4.28)
    arrow(7.55, 3.6, 1.7, 2.95)
    arrow(3.25, 2.22, 3.75, 2.22)
    arrow(6.75, 2.22, 7.25, 2.22)
    arrow(5.0, 1.55, 5.0, 1.0)

    ax.set_title("图1  ActCIM-Robust 方法总体流程 | Fig. 1  Overall pipeline of ActCIM-Robust",
                 fontsize=11)
    save_figure(
        fig, "fig01_pipeline", 1,
        "ActCIM-Robust 方法总体流程图",
        "ActCIM-Robust 总体流程：先在 CIFAR-10 上训练 ResNet-18 基线，再通过输入端非线性"
        "注入框架 y=W·f_α(x) 模拟存算一体阵列的激活传输非线性（f_α 采用逐张量动态最大值归一化；"
        "受控制器命名碰撞影响，实际生效层为深层 4 层）；随后以 Fixed/Random/SGR 三种 NAT 策略"
        "微调，并在全测试集上执行 11 点 α-Sweep，统一计算 Worst-Case Accuracy、AURC、ECE 与"
        "方向不对称指标。该图支持的结论：Fixed-NAT(+0.4) 在保持 Clean 94.02% 的同时将最差"
        "准确率提升至 91.79%。",
        ["src/actcim_robust/nonlinearity/*", "results/post_training/fixed_nat_comparison.json",
         "scripts/paper/verify_injection_scope.py"])


# ================================================================ fig 2
def fig02_nonlinearity():
    x = np.linspace(-1, 1, 401)
    alphas = [-0.8, -0.4, 0.0, 0.4, 0.8]
    cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for a in alphas:
        y = a * x ** 3 + (1 - a) * x
        c = cmap((a + 0.8) / 1.6)
        axes[0].plot(x, y, color=c, lw=1.8, label=f"α={a:+.1f}" if a else "α=0 (恒等)")
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(np.abs(x) > 1e-9, y / x, 1 - a)
        axes[1].plot(x, g, color=c, lw=1.8)
    axes[0].set_xlabel("归一化输入 x/m（无量纲）")
    axes[0].set_ylabel("归一化输出 f_α(x)/m（无量纲）")
    axes[0].set_title("(a) 非线性传输函数 f_α")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[1].axhline(1.0, color="#999999", lw=0.8, ls="--")
    axes[1].set_xlabel("归一化输入 x/m（无量纲）")
    axes[1].set_ylabel("增益 f_α(x)/x（无量纲）")
    axes[1].set_title("(b) 正负 α 的增益响应：α>0 压缩小信号，α<0 扩张")
    axes[1].annotate("α>0：小信号增益<1\n（激活压缩）", xy=(0.05, 0.28),
                     xycoords="axes fraction", fontsize=8, color="#d62728")
    axes[1].annotate("α<0：小信号增益>1\n（激活扩张）", xy=(0.62, 0.80),
                     xycoords="axes fraction", fontsize=8, color="#1f77b4")
    fig.suptitle("图2  非线性函数及正负 α 响应 | Fig. 2  Nonlinear transfer function and ±α response",
                 fontsize=11)
    save_figure(
        fig, "fig02_nonlinearity", 2,
        "非线性函数及正负 Alpha 响应",
        "非线性传输函数 f_α(x)=m·[α(x/m)^3+(1-α)(x/m)]（m=max|x| 逐张量动态归一化，按源码 "
        "function.py 绘制）。(a) 不同 α 下的输入-输出曲线；(b) 增益 f_α(x)/x：α>0 时小幅值"
        "激活的增益低于 1（压缩、趋向三次饱和），α<0 时增益高于 1（扩张）。该图支持的结论："
        "正 α 压缩会系统性削弱中小幅值激活的信息，是模型对正 α 更敏感的机理来源。",
        ["src/actcim_robust/nonlinearity/function.py"])


# ================================================================ fig 3
def fig03_alpha_sweep():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for m in ["clean", "random_nat", "sgr_nat", "fixed_nat"]:
        d = method_curve(m)
        ax.plot(d["alpha"], d["test_accuracy"] * 100, marker="o", ms=4,
                lw=1.8, color=METHOD_COLOR[m], label=METHOD_LABEL[m])
    ax.axvline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.annotate("clean 基线在 α=+0.8\n跌至 81.25%", xy=(0.8, 81.25),
                xytext=(0.28, 84.2), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("Fixed-NAT 最差点 91.79%", xy=(0.8, 91.79), xytext=(0.22, 89.0),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#d62728"))
    ax.set_xlabel("非线性强度 α（无量纲）")
    ax.set_ylabel("Top-1 测试准确率（%）")
    ax.set_ylim(80, 96)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("图3  四种方法的 Accuracy-α 曲线 | Fig. 3  Accuracy-α curves of four methods",
                 fontsize=11)
    save_figure(
        fig, "fig03_alpha_sweep", 3,
        "四种方法 Accuracy-Alpha 曲线",
        "四种方法在 α∈[-0.8,+0.8] 共 11 个点上的 Top-1 测试准确率（10 000 张全测试集，"
        "seed 42 checkpoint，均已复评核验）。Clean/Random-NAT/SGR-NAT 三条曲线几乎重合，"
        "在 α=+0.8 处分别跌至 81.25%/81.30%/82.06%；Fixed-NAT(+0.4) 全程平坦，最差点 "
        "91.79%（α=+0.8），α=0 处 94.02%。该图支持的结论：Fixed-NAT 显著抬升最差准确率"
        "（+10.54 pp），且 SGR-NAT 与 Random-NAT 表现接近。",
        ["results/post_training/all_methods_alpha_sweep.csv",
         "results/post_training/fixed_nat_alpha_sweep.csv"])


# ================================================================ fig 4
def fig04_layer_sensitivity():
    d = ranked.sort_values("layer_index")
    idx = np.arange(len(d))
    labels = [f"{int(r.layer_index):02d}:{r.layer_name}" for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    w = 0.4
    ax.bar(idx - w / 2, d["neg_04_accuracy_drop"] * 100, w,
           color="#1f77b4", label="α=-0.4 准确率下降")
    ax.bar(idx + w / 2, d["pos_04_accuracy_drop"] * 100, w,
           color="#d62728", label="α=+0.4 准确率下降")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=75, fontsize=6.6)
    ax.set_xlabel("层索引:控制器登记名（按前向深度排列）")
    ax.set_ylabel("单层注入时准确率下降（百分点, pp）")
    ax.set_ylim(-1.0, 1.4)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.legend(fontsize=8.5)
    ax.set_title("图4  单层注入敏感性（128 样本批，分辨率 0.78 pp）| Fig. 4  Per-layer sensitivity",
                 fontsize=10.5)
    save_figure(
        fig, "fig04_layer_sensitivity", 4,
        "层敏感性排序图",
        "单层注入 α=±0.4 时相对无注入的准确率下降（基线模型，固定 128 样本批，精度分辨率 "
        "1/128~0.78 pp）。由于控制器以局部名称索引 wrapper，21 条登记名实际映射到 4 个唯一"
        "生效层，因此各名称组内数值重复，最大差异仅 ±0.78 pp。该图如实呈现原始 CSV，支持的"
        "结论是：在该退化口径与样本量下，单层注入的敏感性差异微弱，不足以给出可靠的逐层排序"
        "（详见审计说明），层间差异需由图5的逐层误差累积佐证。",
        ["results/analysis/layer_sensitivity_ranked.csv",
         "scripts/paper/verify_injection_scope.py"])


# ================================================================ fig 5
def fig05_error_accumulation():
    neg = accum[accum["alpha_sign"] == "neg_04"].reset_index(drop=True)
    pos = accum[accum["alpha_sign"] == "pos_04"].reset_index(drop=True)

    def depth_key(name: str) -> tuple:
        n = name.replace(".module", "")
        if n == "conv1":
            return (0, 0, 0, 0)
        if n == "fc":
            return (9, 0, 0, 0)
        # layerL.B.xxx
        parts = n.split(".")
        stage = int(parts[0].replace("layer", ""))
        block = int(parts[1])
        sub = parts[2]
        order = {"conv1": 0, "conv2": 1, "downsample": 2}.get(sub, 3)
        return (stage, block, order, 0)

    order_idx = sorted(range(len(neg)), key=lambda i: depth_key(neg["layer_name"][i]))
    neg = neg.iloc[order_idx].reset_index(drop=True)
    pos = pos.iloc[order_idx].reset_index(drop=True)
    idx = np.arange(len(neg))
    labels = [n.replace(".module", "") for n in neg["layer_name"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(idx, neg["relative_l2"], marker="s", ms=4, lw=1.6,
            color="#1f77b4", label="α=-0.4（扩张）")
    ax.plot(idx, pos["relative_l2"], marker="o", ms=4, lw=1.6,
            color="#d62728", label="α=+0.4（压缩）")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=75, fontsize=6.6)
    ax.set_xlabel("层（按前向深度排列）")
    ax.set_ylabel("激活相对 L2 误差 $\\|\\Delta a\\|_2/\\|a\\|_2$（无量纲）")
    ax.legend(fontsize=8.5, loc="upper left")
    first_nz = int(np.argmax(neg["relative_l2"].values > 0))
    ax.axvspan(first_nz - 0.5, len(neg) - 0.5, color="#fff2cc", alpha=0.5)
    ax.annotate("仅深层 4 个生效层出现非零误差\n（前 17 层未被启用，误差为 0）",
                xy=(first_nz, 0.05), xytext=(2.0, 0.55), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_title("图5  误差沿深度逐层累积 | Fig. 5  Layer-wise error accumulation", fontsize=11)
    save_figure(
        fig, "fig05_error_accumulation", 5,
        "误差逐层累积图",
        "全局注入 α=±0.4 时各层输出激活相对干净前向的相对 L2 误差（基线模型，128 样本批，"
        "activation hook 逐层对比）。误差自 layer4.0.downsample.0 起才非零并沿深度快速放大，"
        "至 fc 输入处 α=-0.4 达 0.978、α=+0.4 达 0.605——与实证核验的“实际仅 4 个深层生效”"
        "完全一致。该图支持的结论：非线性误差在深层被逐级放大，且负 α 在激活幅值上造成的相对"
        "扰动大于正 α，但正 α 因压缩语义信息而导致更大的准确率损失（对照图3、图6）。",
        ["results/analysis/layer_error_accumulation.csv",
         "scripts/paper/verify_injection_scope.py"])


# ================================================================ fig 6
def fig06_worst_accuracy():
    methods = ["clean", "random_nat", "sgr_nat", "fixed_nat"]
    worst = [comp[m]["worst_case_accuracy"] * 100 for m in methods]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar([METHOD_LABEL[m] for m in methods], worst,
                  color=[METHOD_COLOR[m] for m in methods], width=0.55)
    for b, v in zip(bars, worst):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}%",
                ha="center", fontsize=9)
    ax.axhline(worst[0], color="#555555", lw=0.9, ls="--")
    ax.annotate("+10.54 pp", xy=(3, worst[3]), xytext=(2.35, 87.5),
                fontsize=10, color="#d62728", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_ylabel("最差准确率 Worst-Case Accuracy（%）")
    ax.set_ylim(75, 95)
    ax.set_title("图6  最差准确率对比（α∈[-0.8,+0.8] 最小值）| Fig. 6  Worst-case accuracy",
                 fontsize=11)
    save_figure(
        fig, "fig06_worst_accuracy", 6,
        "Worst Accuracy 对比图",
        "四种方法在 11 点 α-Sweep 上的最差准确率（均出现在 α=+0.8）。Clean 81.25%、"
        "Random-NAT 81.30%、SGR-NAT 82.06%、Fixed-NAT(+0.4) 91.79%。数值取自 "
        "fixed_nat_comparison.json 并经 checkpoint 复评核验（单一 seed 42 训练协议，"
        "非多种子统计量）。该图支持的结论：Fixed-NAT 将最差准确率提高 10.54 pp，"
        "是四种方法中唯一实质改善最坏情形的方案。",
        ["results/post_training/fixed_nat_comparison.json"])


# ================================================================ fig 7
def fig07_tradeoff():
    methods = ["clean", "random_nat", "sgr_nat", "fixed_nat"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    offsets = {"clean": (-0.002, 0.0006), "random_nat": (-0.002, -0.0009),
               "sgr_nat": (-0.045, 0.0006), "fixed_nat": (0.012, -0.0004)}
    for m in methods:
        x = comp[m]["alpha0_accuracy"] * 100
        y = comp[m]["aurc_all"]
        ax.scatter(x, y, s=140, color=METHOD_COLOR[m], zorder=3,
                   edgecolor="white", linewidth=1.2, label=METHOD_LABEL[m])
        dx, dy = offsets[m]
        ax.annotate(f"({x:.2f}%, {y:.4f})", (x, y), xytext=(x + dx, y + dy),
                    fontsize=7.5)
    ax.margins(x=0.14, y=0.10)
    ax.set_xlabel("Clean Accuracy（α=0 测试准确率, %）")
    ax.set_ylabel("AURC（Accuracy-α 曲线归一化面积，无量纲）")
    ax.set_title("图7  Clean Accuracy-AURC 权衡 | Fig. 7  Clean accuracy vs AURC trade-off",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="center left")
    save_figure(
        fig, "fig07_tradeoff", 7,
        "Clean Accuracy-AURC 权衡图",
        "各方法 α=0 准确率（横轴）与 AURC（纵轴，Accuracy-α 曲线在 [-0.8,+0.8] 上的梯形积分"
        "除以区间宽度 1.6）。Fixed-NAT 以 0.21 pp 的 Clean 代价（94.23%→94.02%）换取 AURC "
        "从 0.9283 提升至 0.9374；Random-NAT/SGR-NAT 几乎停留在基线位置。该图支持的结论："
        "Fixed-NAT 的鲁棒性收益远大于其微小的干净精度损失，位于权衡前沿。",
        ["results/post_training/fixed_nat_comparison.json"])


# ================================================================ fig 8
def fig08_ece_alpha():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for m in ["clean", "random_nat", "sgr_nat", "fixed_nat"]:
        d = method_curve(m)
        ax.plot(d["alpha"], d["ece_15_bins"], marker="o", ms=4, lw=1.8,
                color=METHOD_COLOR[m], label=METHOD_LABEL[m])
    ax.set_xlabel("非线性强度 α（无量纲）")
    ax.set_ylabel("ECE（15 等宽区间期望校准误差，无量纲）")
    ax.axvline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.annotate("α=+0.8：置信度崩溃至~0.25\nECE~0.56（严重欠自信）", xy=(0.8, 0.5576),
                xytext=(-0.05, 0.47), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("Fixed-NAT ECE~0.44\n置信度回升至~0.48", xy=(0.8, 0.4359),
                xytext=(0.30, 0.30), fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#d62728"))
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("图8  ECE-α 曲线 | Fig. 8  Expected calibration error vs α", fontsize=11)
    save_figure(
        fig, "fig08_ece_alpha", 8,
        "ECE-Alpha 曲线",
        "四种方法在 α-Sweep 各点的 15-bin 期望校准误差（ECE = 各置信度区间 |准确率-平均置信度|"
        " 的样本加权和）。负 α 侧模型保持约 0.99 的平均置信度（轻度过自信，ECE<0.07）；"
        "α=+0.8 时 clean/random/sgr 的平均置信度崩溃至约 0.25，而准确率仍有 81-82%，形成"
        "严重欠自信（ECE 0.55-0.56）；Fixed-NAT 平均置信度 0.48、准确率 91.8%，ECE 降至 "
        "0.436（对照图9）。该图支持的结论：强正非线性引起的失准是置信度崩溃型欠自信，"
        "Fixed-NAT 显著减轻但并未完全消除该失准。",
        ["results/post_training/all_methods_alpha_sweep.csv",
         "results/post_training/fixed_nat_alpha_sweep.csv"])


# ================================================================ fig 9
def fig09_reliability():
    panels = [
        ("clean_alpha_0_bins.csv", "(a) Clean 基线, α=0", 0.0326),
        ("clean_alpha_pos_08_bins.csv", "(b) Clean 基线, α=+0.8", 0.5602),
        ("fixed_nat_alpha_pos_08_bins.csv", "(c) Fixed-NAT, α=+0.8", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, (fname, title, ece_known) in zip(axes, panels):
        path = CALIB / fname
        d = pd.read_csv(path)
        centers = (d["bin_lower"] + d["bin_upper"]) / 2
        width = (d["bin_upper"] - d["bin_lower"]).iloc[0] * 0.92
        n = d["sample_count"].sum()
        ece = (d["sample_count"] / n * (d["bin_accuracy"] - d["mean_confidence"]).abs()
               ).where(d["sample_count"] > 0, 0.0).sum()
        mask = d["sample_count"] > 0
        ax.bar(centers[mask], d.loc[mask, "bin_accuracy"], width=width,
               color="#4c72b0", alpha=0.85, edgecolor="white", label="区间准确率")
        ax.plot([0, 1], [0, 1], ls="--", color="#d62728", lw=1.2, label="理想校准 y=x")
        ax.set_title(f"{title}\nECE={ece:.4f}", fontsize=9)
        ax.set_xlabel("预测置信度（无量纲）")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("区间内经验准确率（无量纲）")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("图9  可靠性图（15 等宽置信度区间）| Fig. 9  Reliability diagrams", fontsize=11)
    save_figure(
        fig, "fig09_reliability", 9,
        "可靠性图（ECE 校准分析）",
        "15 等宽置信度区间的可靠性图：柱高为区间内经验准确率，红虚线为理想校准。(a) Clean "
        "基线在 α=0 时接近对角线（ECE=0.033，略过自信）；(b) α=+0.8 时置信度整体坍缩——"
        "38.07% 的样本落入置信度<0.20 的两个区间，各区间经验准确率普遍高于置信度（柱体位于"
        "对角线上方，严重欠自信，ECE=0.560，平均置信度仅 0.252）；(c) Fixed-NAT 在 α=+0.8 时"
        "仍偏欠自信但明显更贴近对角线（ECE=0.436，平均置信度回升至 0.482）。(c) 面板由 "
        "seed 42 checkpoint 按同一 15-bin 协议补算（仅评估，脚本见来源）。该图支持的结论："
        "正 α 压缩引发的是置信度崩溃型欠自信而非过自信，Fixed-NAT 显著缓解该失准并将准确率"
        "提升至 91.8%。",
        ["results/post_training/calibration/clean_alpha_0_bins.csv",
         "results/post_training/calibration/clean_alpha_pos_08_bins.csv",
         "results/post_training/calibration/fixed_nat_alpha_pos_08_bins.csv",
         "scripts/paper/compute_fixed_nat_calibration.py"])


# ================================================================ fig 10
def fig10_multiseed():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    curves = [
        (fx.sort_values("alpha"), "Fixed-NAT seed 42（主结果）", "#d62728", "-"),
        (ms2026.sort_values("alpha"), "Fixed-NAT seed 2026", "#ff7f0e", "--"),
        (ms3407.sort_values("alpha"), "Fixed-NAT seed 3407", "#9467bd", "-."),
    ]
    for d, lab, c, ls in curves:
        ax.plot(d["alpha"], d["test_accuracy"] * 100, marker="o", ms=3.5,
                lw=1.6, ls=ls, color=c, label=lab)
    dclean = method_curve("clean")
    ax.plot(dclean["alpha"], dclean["test_accuracy"] * 100, color="#555555",
            lw=1.2, ls=":", label="Clean 基线（参考）")
    ax.set_xlabel("非线性强度 α（无量纲）")
    ax.set_ylabel("Top-1 测试准确率（%）")
    ax.set_ylim(80, 96)
    ax.legend(fontsize=8.2, loc="lower left")
    ax.set_title("图10  Fixed-NAT 三个训练种子的 α-Sweep 一致性（描述性）| Fig. 10  Seed consistency",
                 fontsize=10.5)
    save_figure(
        fig, "fig10_multiseed", 10,
        "Fixed-NAT 多训练种子一致性（描述性检查）",
        "Fixed-NAT 三个训练种子（42/2026/3407）各自 checkpoint 的完整 11 点 α-Sweep（评估协议"
        "与种子一致，均为全测试集）。α=+0.8 处最差点分别为 91.79%/91.36%/91.60%（极差 "
        "0.43 pp），α=0 处为 94.02%/94.29%/94.23%。仅 n=3 且对比方法（Random/SGR-NAT）只有"
        "单种子，因此本图仅作描述性一致性检查，不构成统计显著性证据。该图支持的结论："
        "Fixed-NAT 的鲁棒性增益在三个训练种子下方向与量级一致。",
        ["results/post_training/fixed_nat_alpha_sweep.csv",
         "results/post_training/multi_seed/fixed_nat_seed_2026_alpha_sweep.csv",
         "results/post_training/multi_seed/fixed_nat_seed_3407_alpha_sweep.csv"])


# ================================================================ main
if __name__ == "__main__":
    fig01_pipeline()
    fig02_nonlinearity()
    fig03_alpha_sweep()
    fig04_layer_sensitivity()
    fig05_error_accumulation()
    fig06_worst_accuracy()
    fig07_tradeoff()
    fig08_ece_alpha()
    fig09_reliability()
    fig10_multiseed()

    (OUT / "figures_manifest.json").write_text(
        json.dumps(MANIFEST, ensure_ascii=False, indent=2))
    lines = ["# 论文图片清单（figures manifest）", "",
             "所有图片均由 `scripts/paper/generate_paper_figures.py` 从真实结果文件生成，"
             "300 DPI PNG 与矢量 PDF 双格式。", ""]
    for m in MANIFEST:
        lines += [f"## 图{m['number']}  {m['title']}", "",
                  f"- 文件: `{m['png']}` / `{m['pdf']}`",
                  f"- 数据来源: " + "; ".join(f"`{s}`" for s in m["sources"]),
                  f"- 图注: {m['caption']}", ""]
    (OUT / "figures_manifest.md").write_text("\n".join(lines))
    print(f"\nAll {len(MANIFEST)} figures written to {OUT}")
