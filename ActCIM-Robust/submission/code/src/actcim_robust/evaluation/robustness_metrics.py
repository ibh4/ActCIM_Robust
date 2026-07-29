from __future__ import annotations

import numpy as np


def compute_accuracy_drop(clean_acc, nonlinear_acc):
    return clean_acc - nonlinear_acc


def compute_worst_case_accuracy(accuracies):
    if len(accuracies) == 0:
        return 0.0
    return float(min(accuracies))


def compute_aurc(accuracies, alphas):
    if len(accuracies) == 0 or len(alphas) == 0:
        return 0.0
    sorted_pairs = sorted(zip(alphas, accuracies), key=lambda x: x[0])
    sorted_alphas = np.array([p[0] for p in sorted_pairs])
    sorted_accs = np.array([p[1] for p in sorted_pairs])
    if len(sorted_alphas) < 2:
        return float(np.mean(sorted_accs))
    a = sorted_alphas
    b = sorted_accs
    area = np.trapz(b, a)
    total_width = sorted_alphas[-1] - sorted_alphas[0]
    if total_width == 0:
        return float(np.mean(sorted_accs))
    return float(area / total_width)


def compute_positive_negative_gap(acc_pos_alpha, acc_neg_alpha):
    if isinstance(acc_pos_alpha, list):
        acc_pos_alpha = np.array(acc_pos_alpha)
    if isinstance(acc_neg_alpha, list):
        acc_neg_alpha = np.array(acc_neg_alpha)
    return float(np.mean(acc_pos_alpha) - np.mean(acc_neg_alpha))


def compute_relative_improvement(baseline_acc, improved_acc):
    if baseline_acc == 0:
        return 0.0
    return float((improved_acc - baseline_acc) / abs(baseline_acc))


def compute_mean_accuracy_across_alpha(accuracies):
    if len(accuracies) == 0:
        return 0.0
    return float(np.mean(accuracies))
