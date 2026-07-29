from __future__ import annotations

import numpy as np
from scipy import stats


def compute_mean_std(values):
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def compute_confidence_interval(values, confidence=0.95):
    if len(values) < 2:
        return float(values[0]) if values else 0.0, float(values[0]) if values else 0.0
    arr = np.array(values, dtype=np.float64)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    sem = std_val / np.sqrt(len(arr))
    df = len(arr) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_crit * sem
    return float(mean_val - margin), float(mean_val + margin)


def compute_cohens_d(group1, group2):
    g1 = np.array(group1, dtype=np.float64)
    g2 = np.array(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def compute_improvement_summary(baseline_metrics, method_metrics, method_name):
    summary = {"method": method_name}
    for key in baseline_metrics:
        if key in method_metrics:
            base_val = baseline_metrics[key]
            method_val = method_metrics[key]
            if isinstance(base_val, (int, float)) and isinstance(method_val, (int, float)):
                delta = method_val - base_val
                rel = (delta / abs(base_val)) if abs(base_val) > 1e-12 else 0.0
                summary[key] = {
                    "baseline": base_val,
                    "method": method_val,
                    "absolute_delta": float(delta),
                    "relative_delta": float(rel),
                }
    return summary
