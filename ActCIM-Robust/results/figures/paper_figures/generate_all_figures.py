#!/usr/bin/env python3
"""Generate all 8 paper-ready figures for ActCIM-Robust project."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np
import pandas as pd
import csv
import os
from scipy.integrate import trapezoid

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = r"I:\比赛项目\存算一体高校挑战赛\ActCIM-Robust\results"
OUT_DIR = r"I:\比赛项目\存算一体高校挑战赛\ActCIM-Robust\results\figures\paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Data paths
ALL_METHODS_CSV = os.path.join(DATA_DIR, "post_training", "all_methods_alpha_sweep.csv")
FIXED_NAT_CSV = os.path.join(DATA_DIR, "post_training", "fixed_nat_alpha_sweep.csv")
FIXED_NAT_SEED_2026_CSV = os.path.join(DATA_DIR, "post_training", "multi_seed", "fixed_nat_seed_2026_alpha_sweep.csv")
FIXED_NAT_SEED_3407_CSV = os.path.join(DATA_DIR, "post_training", "multi_seed", "fixed_nat_seed_3407_alpha_sweep.csv")
LAYER_SENS_CSV = os.path.join(DATA_DIR, "analysis", "layer_sensitivity_ranked.csv")
ERROR_ACCUM_CSV = os.path.join(DATA_DIR, "analysis", "layer_error_accumulation.csv")

# Color scheme
COLORS = {
    'clean': '#757575',
    'random_nat': '#FF9800',
    'sgr_nat': '#4CAF50',
    'fixed_nat': '#1E88E5',
}

LABELS = {
    'clean': 'Clean',
    'random_nat': 'Random-NAT',
    'sgr_nat': 'SGR-NAT',
    'fixed_nat': 'Fixed-NAT',
}

# Matplotlib rcParams
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
})

FIG_SIZE = (6, 4.5)

# ─── Data loading ────────────────────────────────────────────────────────────
df_all = pd.read_csv(ALL_METHODS_CSV)
df_fixed = pd.read_csv(FIXED_NAT_CSV)
df_fixed_s2026 = pd.read_csv(FIXED_NAT_SEED_2026_CSV)
df_fixed_s3407 = pd.read_csv(FIXED_NAT_SEED_3407_CSV)
df_layer_sens = pd.read_csv(LAYER_SENS_CSV)
df_error = pd.read_csv(ERROR_ACCUM_CSV)

# Verify data loaded
for name, df in [("all_methods", df_all), ("fixed_nat", df_fixed),
                 ("fixed_s2026", df_fixed_s2026), ("fixed_s3407", df_fixed_s3407),
                 ("layer_sens", df_layer_sens), ("error_accum", df_error)]:
    print(f"Loaded {name}: {len(df)} rows")


def save_figure(fig, basename):
    """Save figure as both PNG and PDF."""
    png_path = os.path.join(OUT_DIR, f"{basename}.png")
    pdf_path = os.path.join(OUT_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    png_size = os.path.getsize(png_path)
    pdf_size = os.path.getsize(pdf_path)
    print(f"  Saved {basename}.png ({png_size:,} bytes)")
    print(f"  Saved {basename}.pdf ({pdf_size:,} bytes)")
    plt.close(fig)


def compute_aurc(alphas, accuracies):
    """Compute AURC normalized by alpha range."""
    area = trapezoid(accuracies, alphas)
    alpha_range = alphas[-1] - alphas[0]
    return area / alpha_range


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Nonlinear function f_alpha(x)
# ═══════════════════════════════════════════════════════════════════════════════
def fig01_nonlinear_function():
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.linspace(-1, 1, 500)
    alphas = [-0.8, -0.4, 0, 0.4, 0.8]
    cmap_copper = plt.cm.RdYlGn
    colors_list = ['#d73027', '#fc8d59', '#757575', '#91bfdb', '#1a9850']

    for alpha, c in zip(alphas, colors_list):
        y = alpha * x**3 + (1 - alpha) * x
        linestyle = '--' if alpha == 0 else '-'
        linewidth = 2.5 if alpha == 0 else 1.8
        ax.plot(x, y, color=c, linestyle=linestyle, linewidth=linewidth,
                label=rf'$\alpha = {alpha:+.1f}$')

    # Diagonal reference
    ax.plot(x, x, color='#aaaaaa', linestyle=':', linewidth=1, alpha=0.5, label=r'$y=x$ (identity)')

    # Annotate regions
    ax.annotate('Compression\n(positive α)', xy=(0.65, 0.30), fontsize=9,
                ha='center', color='#1a9850',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', alpha=0.8))
    ax.annotate('Expansion\n(negative α)', xy=(-0.65, -0.30), fontsize=9,
                ha='center', color='#d73027',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', alpha=0.8))

    # Fixed points
    ax.scatter([-1, 0, 1], [-1, 0, 1], c='black', s=30, zorder=5)
    for px, py in [(-1, -1), (0, 0), (1, 1)]:
        offset_x = 0.12 if px >= 0 else -0.12
        offset_y = 0.12 if py >= 0 else -0.12
        ax.annotate(rf'$x={px}$', (px, py), textcoords="offset points",
                    xytext=(12 if px >= 0 else -20, -12 if py >= 0 else -20),
                    fontsize=7, ha='center')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Normalized $x$')
    ax.set_ylabel(r'$f_\alpha(x)$')
    ax.set_title(r'Cubic Nonlinearity Function $f_\alpha(x)$', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    fig.tight_layout()
    save_figure(fig, 'fig01_nonlinear_function')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Method flowchart
# ═══════════════════════════════════════════════════════════════════════════════
def fig02_method_flowchart():
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edgecolor='#1E88E5'):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor=edgecolor,
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                fontweight='bold', zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=2.0, color='#333333'))

    # Boxes
    colors_boxes = ['#E3F2FD', '#C8E6C9', '#FFE0B2', '#F3E5F5']
    edges = ['#1E88E5', '#4CAF50', '#FF9800', '#9C27B0']

    draw_box(ax, 1.5, 5.5, 2.2, 1.0, 'CIFAR-10\nInput', colors_boxes[0], edges[0])
    draw_box(ax, 4.8, 5.5, 3.0, 1.0, 'ResNet-18\nBackbone', colors_boxes[1], edges[1])
    draw_box(ax, 8.5, 5.5, 3.4, 1.0, 'NonlinearInput\nWrapper $f_\\alpha$', colors_boxes[2], edges[2])
    draw_box(ax, 5.0, 2.5, 3.0, 1.0, 'Classification\n(10 classes)', colors_boxes[3], edges[3])

    # Arrows horizontal
    draw_arrow(ax, 2.6, 5.5, 3.3, 5.5)
    draw_arrow(ax, 6.3, 5.5, 6.8, 5.5)

    # Arrow down from Wrapper to Classification
    draw_arrow(ax, 8.5, 5.0, 6.5, 3.0)

    # Annotations
    ax.text(3.0, 6.8, 'Forward\nPass', ha='center', fontsize=8, color='#555555', style='italic')

    # Training label
    ax.text(1.5, 2.0, 'Training: Inject nonlinearity\nat every Conv2d &\nLinear layer input', 
            ha='center', fontsize=8, color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.7))

    fig.tight_layout()
    save_figure(fig, 'fig02_method_flowchart')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Accuracy vs Alpha (all 4 methods)
# ═══════════════════════════════════════════════════════════════════════════════
def fig03_accuracy_vs_alpha():
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Clean
    clean = df_all[df_all['method'] == 'clean']
    ax.plot(clean['alpha'], clean['test_accuracy'], color=COLORS['clean'],
            linewidth=1.8, marker='o', markersize=5, markerfacecolor='white',
            markeredgewidth=1.2, label=LABELS['clean'])

    # Random-NAT
    random_nat = df_all[df_all['method'] == 'random_nat']
    ax.plot(random_nat['alpha'], random_nat['test_accuracy'], color=COLORS['random_nat'],
            linewidth=1.8, linestyle='--', marker='s', markersize=5,
            markerfacecolor='white', markeredgewidth=1.2, label=LABELS['random_nat'])

    # SGR-NAT
    sgr_nat = df_all[df_all['method'] == 'sgr_nat']
    ax.plot(sgr_nat['alpha'], sgr_nat['test_accuracy'], color=COLORS['sgr_nat'],
            linewidth=1.8, linestyle=':', marker='D', markersize=5,
            markerfacecolor='white', markeredgewidth=1.2, label=LABELS['sgr_nat'])

    # Fixed-NAT (seed 42)
    ax.plot(df_fixed['alpha'], df_fixed['test_accuracy'], color=COLORS['fixed_nat'],
            linewidth=2.2, linestyle='-', marker='^', markersize=6,
            markerfacecolor='white', markeredgewidth=1.5, label=LABELS['fixed_nat'])

    # Shade alpha=+0.4 region
    ax.axvspan(0.38, 0.42, alpha=0.15, color=COLORS['fixed_nat'], zorder=0)
    ax.annotate(r'Fixed-NAT training $\alpha=+0.4$', xy=(0.4, 0.82),
                ha='center', fontsize=8, color=COLORS['fixed_nat'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                xytext=(0.4, 0.80), rotation=0)

    # Annotate asymmetry
    ax.annotate('Positive α degrades\nmore severely', xy=(0.6, 0.89),
                fontsize=7.5, color='#c62828', ha='center', style='italic')

    ax.set_xlabel(r'Nonlinearity Strength $\alpha$')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy vs Nonlinearity Strength $\\alpha$', fontsize=12, fontweight='bold')
    ax.set_ylim(0.80, 0.96)
    ax.legend(loc='lower left', framealpha=0.9, ncol=2)
    fig.tight_layout()
    save_figure(fig, 'fig03_accuracy_vs_alpha')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Layer Sensitivity Ranked
# ═══════════════════════════════════════════════════════════════════════════════
def fig04_layer_sensitivity_ranked():
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Sort by sensitivity_score ascending (so highest is at top when plotting from bottom)
    df_sorted = df_layer_sens.sort_values('sensitivity_score', ascending=True)

    # Shorten layer names
    def shorten_layer(name):
        mapping = {
            'conv1': 'conv1',
            'conv2': 'conv2',
            '0': 'DS0',
            '1': 'DS1',
            'fc': 'FC',
        }
        # Handle layerX.Y.convZ patterns
        parts = name.split('.')
        if len(parts) > 1:
            shortened = ''
            for p in parts:
                if p.startswith('layer'):
                    shortened += f'L{p[5:]}'
                elif p == 'conv1':
                    shortened += 'C1'
                elif p == 'conv2':
                    shortened += 'C2'
                elif p == 'downsample':
                    pass  # handled by next '0' or '1'
                elif p in ('0', '1'):
                    shortened += p
                elif p == 'module':
                    pass
                else:
                    shortened += p
            return shortened
        return name

    layer_labels = [shorten_layer(n) for n in df_sorted['layer_name']]

    # Colors based on layer_type
    bar_colors = ['#1E88E5' if lt == 'Conv2d' else '#E53935' for lt in df_sorted['layer_type']]

    bars = ax.barh(range(len(df_sorted)), df_sorted['sensitivity_score'], color=bar_colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(layer_labels, fontsize=6)

    mean_sens = df_sorted['sensitivity_score'].mean()
    ax.axvline(x=mean_sens, color='#333333', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(mean_sens + 0.0001, len(df_sorted) - 1,
            f'Mean: {mean_sens:.4f}', fontsize=7, va='top', color='#333333')

    ax.set_xlabel('Sensitivity Score')
    ax.set_ylabel('Layer (shortened name)')
    ax.set_title('Per-Layer Sensitivity Ranking', fontsize=12, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1E88E5', label='Conv2d'),
        mpatches.Patch(facecolor='#E53935', label='Linear'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, 'fig04_layer_sensitivity_ranked')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Error Accumulation
# ═══════════════════════════════════════════════════════════════════════════════
def fig05_error_accumulation():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)

    # Build ordered layer list with depth index
    unique_layers_ordered = [
        'conv1.module',
        'layer1.0.conv1.module', 'layer1.0.conv2.module',
        'layer1.1.conv1.module', 'layer1.1.conv2.module',
        'layer2.0.conv1.module', 'layer2.0.conv2.module',
        'layer2.0.downsample.0.module',
        'layer2.1.conv1.module', 'layer2.1.conv2.module',
        'layer3.0.conv1.module', 'layer3.0.conv2.module',
        'layer3.0.downsample.0.module',
        'layer3.1.conv1.module', 'layer3.1.conv2.module',
        'layer4.0.conv1.module', 'layer4.0.conv2.module',
        'layer4.0.downsample.0.module',
        'layer4.1.conv1.module', 'layer4.1.conv2.module',
        'fc.module',
    ]

    # Filter to only those in the data
    layer_names_in_data = df_error['layer_name'].unique()
    ordered = [l for l in unique_layers_ordered if l in layer_names_in_data]

    # Create depth index
    depth_map = {name: i for i, name in enumerate(ordered)}

    # Extract data for each alpha sign
    neg_data = df_error[df_error['alpha_sign'] == 'neg_04'].copy()
    pos_data = df_error[df_error['alpha_sign'] == 'pos_04'].copy()

    neg_data['depth'] = neg_data['layer_name'].map(depth_map)
    pos_data['depth'] = pos_data['layer_name'].map(depth_map)

    neg_data = neg_data.sort_values('depth')
    pos_data = pos_data.sort_values('depth')

    # Subplot 1: relative_l2
    ax1.plot(neg_data['depth'], neg_data['relative_l2'], color='#d73027',
             linewidth=1.8, marker='o', markersize=4, label=r'$\alpha = -0.4$')
    ax1.plot(pos_data['depth'], pos_data['relative_l2'], color='#1E88E5',
             linewidth=1.8, marker='s', markersize=4, label=r'$\alpha = +0.4$')
    ax1.set_ylabel('Relative L2 Distance')
    ax1.set_title('Layer-wise Error Accumulation', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(bottom=-0.05)

    # Subplot 2: cosine_similarity
    ax2.plot(neg_data['depth'], neg_data['cosine_similarity'], color='#d73027',
             linewidth=1.8, marker='o', markersize=4, label=r'$\alpha = -0.4$')
    ax2.plot(pos_data['depth'], pos_data['cosine_similarity'], color='#1E88E5',
             linewidth=1.8, marker='s', markersize=4, label=r'$\alpha = +0.4$')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_xlabel('Layer Depth Index')
    ax2.legend(loc='lower left', framealpha=0.9)
    ax2.set_ylim(bottom=min(neg_data['cosine_similarity'].min(),
                             pos_data['cosine_similarity'].min()) - 0.02)

    # X-axis labels
    tick_indices = list(range(0, len(ordered), 2))
    tick_labels = [shorten_layer_name(ordered[i]) for i in tick_indices]
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=6)

    fig.tight_layout()
    save_figure(fig, 'fig05_error_accumulation')


def shorten_layer_name(name):
    # Remove .module suffix
    name = name.replace('.module', '')
    parts = name.split('.')
    if len(parts) > 1:
        shortened = ''
        for p in parts:
            if p.startswith('layer'):
                shortened += f'L{p[5:]}'
            elif p == 'conv1':
                shortened += '.c1'
            elif p == 'conv2':
                shortened += '.c2'
            elif p == 'downsample':
                shortened += '.ds'
            else:
                shortened += f'.{p}'
        return shortened.lstrip('.')
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Worst-Case Accuracy at alpha=+0.8
# ═══════════════════════════════════════════════════════════════════════════════
def fig06_worst_case_comparison():
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Single values
    clean_acc = df_all[(df_all['method'] == 'clean') & (df_all['alpha'] == 0.8)]['test_accuracy'].values[0]
    random_acc = df_all[(df_all['method'] == 'random_nat') & (df_all['alpha'] == 0.8)]['test_accuracy'].values[0]
    sgr_acc = df_all[(df_all['method'] == 'sgr_nat') & (df_all['alpha'] == 0.8)]['test_accuracy'].values[0]

    # Fixed-NAT multi-seed
    fixed_accs = [
        df_fixed[df_fixed['alpha'] == 0.8]['test_accuracy'].values[0],
        df_fixed_s2026[df_fixed_s2026['alpha'] == 0.8]['test_accuracy'].values[0],
        df_fixed_s3407[df_fixed_s3407['alpha'] == 0.8]['test_accuracy'].values[0],
    ]
    fixed_mean = np.mean(fixed_accs)
    fixed_std = np.std(fixed_accs, ddof=1)

    methods = ['Clean', 'Random-\nNAT', 'SGR-\nNAT', 'Fixed-NAT\n(3 seeds)']
    values = [clean_acc, random_acc, sgr_acc, fixed_mean]
    bar_colors = [COLORS['clean'], COLORS['random_nat'], COLORS['sgr_nat'], COLORS['fixed_nat']]

    x_pos = [0, 1, 2, 3]
    bars = ax.bar(x_pos, values, color=bar_colors, edgecolor='white', linewidth=0.8, width=0.5)

    # Error bar for Fixed-NAT
    ax.errorbar(3, fixed_mean, yerr=fixed_std, fmt='none', ecolor='#333333',
                capsize=6, linewidth=1.5, capthick=1.5)

    # Value labels
    for i, (x, v) in enumerate(zip(x_pos, values)):
        ax.text(x, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Fixed-NAT seeds annotation
    ax.text(3, fixed_mean - 0.025,
            f'Seeds: {", ".join(f"{a:.4f}" for a in fixed_accs)}\nMean={fixed_mean:.4f}, Std={fixed_std:.4f}',
            ha='center', fontsize=6.5, color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Test Accuracy')
    ax.set_title(r'Worst-Case Accuracy ($\alpha=+0.8$)', fontsize=12, fontweight='bold')
    ax.set_ylim(0.75, 0.95)

    fig.tight_layout()
    save_figure(fig, 'fig06_worst_case_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Clean Accuracy vs AURC
# ═══════════════════════════════════════════════════════════════════════════════
def fig07_clean_vs_aurc():
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Compute AURC for each method
    def get_aurc_and_clean(method_df, alphas_col='alpha', acc_col='test_accuracy'):
        alphas = method_df[alphas_col].values
        accs = method_df[acc_col].values
        aurc = compute_aurc(alphas, accs)
        clean_acc = method_df[method_df[alphas_col] == 0][acc_col].values[0]
        return clean_acc, aurc

    # Clean
    clean_df = df_all[df_all['method'] == 'clean']
    clean_acc, clean_aurc = get_aurc_and_clean(clean_df)

    # Random-NAT
    rand_df = df_all[df_all['method'] == 'random_nat']
    rand_acc, rand_aurc = get_aurc_and_clean(rand_df)

    # SGR-NAT
    sgr_df = df_all[df_all['method'] == 'sgr_nat']
    sgr_acc, sgr_aurc = get_aurc_and_clean(sgr_df)

    # Fixed-NAT 3 seeds
    fixed_data = []
    for df_seed, label in [(df_fixed, 's42'), (df_fixed_s2026, 's2026'), (df_fixed_s3407, 's3407')]:
        acc, aurc = get_aurc_and_clean(df_seed)
        fixed_data.append((acc, aurc, label))

    # Plot single points
    method_points = [
        (clean_acc, clean_aurc, 'Clean', COLORS['clean'], 'o'),
        (rand_acc, rand_aurc, 'Random-NAT', COLORS['random_nat'], 's'),
        (sgr_acc, sgr_aurc, 'SGR-NAT', COLORS['sgr_nat'], 'D'),
    ]

    for x, y, label, color, marker in method_points:
        ax.scatter(x, y, c=color, s=100, marker=marker, edgecolors='white',
                   linewidths=1, zorder=5)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    # Fixed-NAT with jitter
    np.random.seed(42)
    for acc, aurc, label in fixed_data:
        jx = np.random.uniform(-0.0001, 0.0001)
        jy = np.random.uniform(-0.0001, 0.0001)
        ax.scatter(acc + jx, aurc + jy, c=COLORS['fixed_nat'], s=120, marker='^',
                   edgecolors='white', linewidths=1, zorder=6)

    # Fixed-NAT centroid and label
    fixed_accs = [d[0] for d in fixed_data]
    fixed_aurcs = [d[1] for d in fixed_data]
    ax.scatter(np.mean(fixed_accs), np.mean(fixed_aurcs), c='white', s=30,
               marker='^', zorder=7)
    ax.annotate('Fixed-NAT\n(3 seeds)', (np.mean(fixed_accs), np.mean(fixed_aurcs)),
                textcoords="offset points", xytext=(10, -15), fontsize=8,
                fontweight='bold', color=COLORS['fixed_nat'],
                arrowprops=dict(arrowstyle='->', color=COLORS['fixed_nat'], lw=0.8))

    # Ideal corner annotation
    ax.annotate('Ideal\n(high clean,\nhigh robust)', xy=(0.943, 0.938),
                fontsize=7, color='#2E7D32', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.7))

    ax.set_xlabel('Clean Accuracy ($\\alpha=0$)')
    ax.set_ylabel('AURC (All $\\alpha$)')
    ax.set_title('Clean Accuracy vs Robustness (AURC)', fontsize=12, fontweight='bold')

    fig.tight_layout()
    save_figure(fig, 'fig07_clean_vs_aurc')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: ECE Calibration
# ═══════════════════════════════════════════════════════════════════════════════
def fig08_ece_calibration():
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Clean ECE
    clean_ece = df_all[df_all['method'] == 'clean']
    ax.plot(clean_ece['alpha'], clean_ece['ece_15_bins'], color=COLORS['clean'],
            linewidth=1.8, marker='o', markersize=5, markerfacecolor='white',
            markeredgewidth=1.2, label='Clean (ECE)')

    # Fixed-NAT ECE
    ax.plot(df_fixed['alpha'], df_fixed['ece_15_bins'], color=COLORS['fixed_nat'],
            linewidth=2.2, marker='^', markersize=6, markerfacecolor='white',
            markeredgewidth=1.5, label='Fixed-NAT (ECE)')

    # Shade between curves
    alphas_common = sorted(set(clean_ece['alpha'].values) & set(df_fixed['alpha'].values))
    clean_ece_dict = dict(zip(clean_ece['alpha'], clean_ece['ece_15_bins']))
    fixed_ece_dict = dict(zip(df_fixed['alpha'], df_fixed['ece_15_bins']))
    common_alphas = np.array(sorted(alphas_common))
    clean_vals = np.array([clean_ece_dict[a] for a in common_alphas])
    fixed_vals = np.array([fixed_ece_dict[a] for a in common_alphas])
    ax.fill_between(common_alphas, clean_vals, fixed_vals, alpha=0.1,
                    color=COLORS['fixed_nat'], label='Improvement gap')

    # Annotate at alpha=+0.8
    clean_ece_08 = clean_ece[clean_ece['alpha'] == 0.8]['ece_15_bins'].values[0]
    fixed_ece_08 = df_fixed[df_fixed['alpha'] == 0.8]['ece_15_bins'].values[0]
    ax.annotate(f'Clean:\nECE={clean_ece_08:.3f}', xy=(0.8, clean_ece_08),
                fontsize=7, color=COLORS['clean'], ha='right',
                xytext=(-10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FAFAFA', alpha=0.8))
    ax.annotate(f'Fixed-NAT:\nECE={fixed_ece_08:.3f}', xy=(0.8, fixed_ece_08),
                fontsize=7, color=COLORS['fixed_nat'], ha='left',
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', alpha=0.8))

    ax.set_xlabel(r'Nonlinearity Strength $\alpha$')
    ax.set_ylabel('ECE (15 bins)')
    ax.set_title(r'Calibration Error (ECE) vs $\alpha$', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, 'fig08_ece_calibration')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Generating ActCIM-Robust Paper Figures")
    print("=" * 60)

    print("\n[1/8] Fig 01: Nonlinear function f_alpha(x)")
    fig01_nonlinear_function()

    print("\n[2/8] Fig 02: Method flowchart")
    fig02_method_flowchart()

    print("\n[3/8] Fig 03: Accuracy vs Alpha")
    fig03_accuracy_vs_alpha()

    print("\n[4/8] Fig 04: Layer Sensitivity Ranked")
    fig04_layer_sensitivity_ranked()

    print("\n[5/8] Fig 05: Error Accumulation")
    fig05_error_accumulation()

    print("\n[6/8] Fig 06: Worst-Case Comparison")
    fig06_worst_case_comparison()

    print("\n[7/8] Fig 07: Clean Accuracy vs AURC")
    fig07_clean_vs_aurc()

    print("\n[8/8] Fig 08: ECE Calibration")
    fig08_ece_calibration()

    print("\n" + "=" * 60)
    print("All figures generated. Verifying output...")

    # Verify all 16 files
    expected = [
        'fig01_nonlinear_function', 'fig02_method_flowchart',
        'fig03_accuracy_vs_alpha', 'fig04_layer_sensitivity_ranked',
        'fig05_error_accumulation', 'fig06_worst_case_comparison',
        'fig07_clean_vs_aurc', 'fig08_ece_calibration',
    ]
    all_ok = True
    for fig_name in expected:
        for ext in ['png', 'pdf']:
            fpath = os.path.join(OUT_DIR, f"{fig_name}.{ext}")
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                status = "OK" if size > 0 else "EMPTY"
                if size == 0:
                    all_ok = False
                print(f"  {fig_name}.{ext}: {size:>10,} bytes [{status}]")
            else:
                print(f"  {fig_name}.{ext}: MISSING!")
                all_ok = False

    print(f"\nTotal files: {len(expected) * 2} expected")
    print(f"All valid: {all_ok}")
